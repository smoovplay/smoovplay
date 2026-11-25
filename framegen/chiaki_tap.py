"""Frames from chiaki-ng (PS5 Remote Play) through its frame tap: see chiaki_tap/frametap.h for the layout.

chiaki-ng, built with the tap and started with CHIAKI_FRAMETAP=1, copies every compressed video frame the PS5
sends into a shared-memory ring. FrameTap reads that ring; ChiakiSource decodes it into Ticks for the pipeline.

Compared with a capture card, each frame also carries what the stream knows about it:
  - size: bytes of the compressed frame. A repeat costs the encoder little, so size says roughly which frames are
    new (SizeRepeatDetector, in extra["size_new"]): kept for the frame log, the stream check and test mode. The
    pipeline tells new frames from repeats by the picture instead (cadence.DuplicateDetector(learn=True), as for
    an app's window): a decoded repeat is never an exact copy (the encoder spends its few bytes sharpening it),
    but it changes 10 times or more less than a new frame. In real sessions that split them 93-99% right, where
    size managed 3-35% (a repeat can be nearly half a new frame's size, and the size check stuck after still
    scenes).
  - damaged: frames were lost before this one, or it's the PS5's recovery frame after a loss. The pipeline
    holds instead of interpolating, so a broken frame isn't smeared into the generated one.
Windows and Linux: the operating system parts are in tapsys.py.
"""

from __future__ import annotations

import csv
import ctypes
import mmap
import struct
import sys
import tempfile
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .io import LatestFrameMailbox, Tick
from .streamcheck import StreamCheck
from .sync import CAPTURE_LOCK
from .tapsys import Signal, open_map, qpc, qpc_freq

NAME = "Local\\chiaki-frametap"
EVENT_NAME = "Local\\chiaki-frametap-frame"
MAGIC = 0x50544643  # 'CFTP'
VERSION = 1
HEADER = 4096
SLOT_HEADER = 64
SLOTS = 32
SLOT_SIZE = 2 * 1024 * 1024
TOTAL = HEADER + SLOTS * SLOT_SIZE
CAPACITY = SLOTS * SLOT_SIZE  # the slots' share of the mapping, however a writer divides it (docs/frame-tap.md)
CODECS = {0: "h264", 1: "hevc", 2: "hevc"}  # ChiakiCodec: H264, H265, H265_HDR
CODEC_NAMES = {0: "H.264", 1: "H.265", 2: "H.265 HDR", 16: "NV12 (decoded)", 17: "BGRA (decoded)"}
# Decoded frames from any app (docs/frame-tap.md, framegen/tap.py): a writer of these sets version 2 and divides
# CAPACITY into slots of its own size (raw_layout), since a decoded 1080p frame doesn't fit a 2 MiB slot.
RAW_CODECS = {16: "nv12", 17: "bgra"}
VERSION_RAW = 2
RAW_BYTES = {"nv12": 1.5, "bgra": 4}
# Header `colour` (offset 92) for NV12: 0 auto (BT.709 at 720 lines and up, else BT.601; limited range), then
# 1 BT.709 limited, 2 BT.709 full, 3 BT.601 limited, 4 BT.601 full, as (bt709, full_range). BGRA is always full RGB.
COLOURS = {1: (True, False), 2: (True, True), 3: (False, False), 4: (False, True)}
DAMAGE_HOLD = 6  # frames held (not interpolated) after a gap in the stream, while the picture repairs itself
U64_MAX = 2**64 - 1
FEATURE_PRESS, FEATURE_AUDIO_DELAY, FEATURE_AUDIO_TAP = 1, 2, 4  # header `features` bits (frametap.h)
FEATURE_VARIABLE = 8  # frames come when the picture changes, not once a refresh: the reader puts them on a beat
FLAG_TRUNCATED, FLAG_REPEAT, FLAG_NEW = 1, 2, 4  # slot `flags`: REPEAT/NEW are the writer's own word on the frame


def raw_layout(fmt: str, width: int, height: int) -> tuple[int, int]:
    """(slot_count, slot_size) for decoded frames of this format and size: slots a whole number of MiB that hold
    one frame and its slot header, as many as fit in CAPACITY (at most 32). ValueError if fewer than 4 fit."""
    need = SLOT_HEADER + int(width * height * RAW_BYTES[fmt])
    size = -(-need // (1 << 20)) << 20
    count = min(SLOTS, CAPACITY // size)
    if count < 4:
        raise ValueError(f"a {width}x{height} {fmt} frame is too big for the frame tap ({need / 1e6:.1f} MB; "
                         f"4 have to fit in {CAPACITY >> 20} MiB)")
    return count, size
# The game's sound, in a mapping of its own (frametap.h, "Sound ring")
AUDIO_SUFFIX = "-audio"
AUDIO_MAGIC = 0x41544643  # 'CFTA'
AUDIO_HEADER, AUDIO_SLOT_HEADER, AUDIO_SLOTS, AUDIO_SLOT_SIZE = 64, 32, 512, 8192
AUDIO_TOTAL = AUDIO_HEADER + AUDIO_SLOTS * AUDIO_SLOT_SIZE
BUTTON_OPTIONS, BUTTON_PS = 1 << 12, 1 << 15  # ChiakiControllerButton



@dataclass
class Header:
    magic: int
    version: int
    codec: int
    width: int
    height: int
    max_fps: int
    write_seq: int
    session: int
    qpc_freq: int
    slots: int = SLOTS  # the writer's slot_count and slot_size (version 2; a version 1 writer always uses these)
    slot_size: int = SLOT_SIZE
    colour: int = 0  # NV12's colours (COLOURS), from a version 2 writer
    features: int = 0

    @property
    def live(self) -> bool:
        return self.magic == MAGIC


@dataclass
class Packet:
    seq: int
    t_arrival: float  # time.perf_counter() timebase
    data: bytes
    frames_lost: int
    recovered: bool
    truncated: bool
    flags: int = 0  # the slot's flags (FLAG_*)
    size: int = 0  # bytes in the slot (read(into=...) leaves data empty)


class FrameTap:
    """The shared ring. Either side may create it first; both see the same memory."""

    def __init__(self, name: str = NAME, event_name: str = EVENT_NAME) -> None:
        self.mm = open_map(name, TOTAL)
        self.event = Signal(self.mm, event_name)

    def header(self) -> Header:
        magic, version, n, size, codec, w, h, fps = struct.unpack_from("<8I", self.mm, 0)
        (write_seq,) = struct.unpack_from("<Q", self.mm, 32)
        (session,) = struct.unpack_from("<I", self.mm, 44)
        (freq,) = struct.unpack_from("<Q", self.mm, 48)
        (features,) = struct.unpack_from("<I", self.mm, 64)
        (colour,) = struct.unpack_from("<I", self.mm, 92)
        if version < VERSION_RAW or not (4 <= n <= SLOTS and SLOT_HEADER < size and n * size <= CAPACITY):
            n, size, colour = SLOTS, SLOT_SIZE, 0  # version 1 (chiaki-ng): the fixed layout
        return Header(magic, version, codec, w, h, fps, write_seq, session, freq, n, size, colour, features)

    def request_idr(self) -> None:
        struct.pack_into("<I", self.mm, 40, 1)

    def heartbeat(self) -> None:
        struct.pack_into("<Q", self.mm, 56, qpc())

    # The delay test's button presses and the sound delay (frametap.h, offsets 64-91). A chiaki-ng built from an
    # older patch leaves `features` at 0 and ignores both.
    def features(self) -> int:
        return struct.unpack_from("<I", self.mm, 64)[0] if self.header().live else 0

    def press(self, buttons: int, hold_ms: int = 100) -> int:
        """Ask chiaki-ng to press `buttons` (a ChiakiControllerButton mask) for hold_ms. Returns a token for
        press_sent()."""
        (count,) = struct.unpack_from("<I", self.mm, 76)
        struct.pack_into("<I", self.mm, 72, hold_ms)
        struct.pack_into("<I", self.mm, 68, buttons)
        return count

    def press_sent(self, token: int, qpc_freq: int) -> float | None:
        """When chiaki-ng sent the press to the PS5 (time.perf_counter() timebase), or None if it hasn't yet."""
        (count,) = struct.unpack_from("<I", self.mm, 76)
        if count == token or not qpc_freq:
            return None
        (q,) = struct.unpack_from("<Q", self.mm, 80)
        return time.perf_counter() - (qpc() - q) / qpc_freq

    def set_audio_delay(self, ms: int) -> None:
        struct.pack_into("<I", self.mm, 88, max(0, min(500, int(ms))))

    def wait(self, timeout_s: float) -> None:
        self.event.wait(timeout_s)

    def read(self, seq: int, qpc_freq: int, slots: int = SLOTS, slot_size: int = SLOT_SIZE,
             into=None) -> Packet | None:
        """Copy frame `seq` out of its slot, or None if the writer has already reused the slot. slots, slot_size:
        the writer's layout (Header.slots, .slot_size). into: a uint8 numpy array to copy the frame into instead of
        a new bytes object (a decoded frame, megabytes); the Packet's data is then b"" and its size is in `size`."""
        off = HEADER + (seq % slots) * slot_size
        s0, q, size, lost, rec, flags = struct.unpack_from("<QQIiII", self.mm, off)
        if s0 != seq or size > slot_size - SLOT_HEADER:
            return None
        if into is None:
            data = self.mm[off + SLOT_HEADER: off + SLOT_HEADER + size]
        else:
            if size > into.size:
                return None
            into[:size] = np.frombuffer(self.mm, np.uint8, size, off + SLOT_HEADER)
            data = b""
        if struct.unpack_from("<Q", self.mm, off)[0] != seq:  # rewritten while we copied
            return None
        age = (qpc() - q) / qpc_freq if qpc_freq else 0.0
        return Packet(seq, time.perf_counter() - age, data, lost, bool(rec), bool(flags & FLAG_TRUNCATED), flags,
                      size)

    # Writer side, for tests, the fake PS5 (scripts/frametap_fake.py) and the decoded-frame sender
    # (scripts/frametap_send.py); chiaki-ng uses frametap.c.
    _slots, _slot_size = SLOTS, SLOT_SIZE

    def begin(self, codec: int, width: int, height: int, max_fps: int = 60, colour: int = 0,
              features: int | None = None) -> None:
        """Start a stream. A decoded format (RAW_CODECS) gets version 2 and slots of its own (raw_layout).
        features, if given, replaces the header's features field."""
        struct.pack_into("<I", self.mm, 0, 0)
        freq = qpc_freq()
        if codec in RAW_CODECS:
            version, (self._slots, self._slot_size) = VERSION_RAW, raw_layout(RAW_CODECS[codec], width, height)
        else:
            version, self._slots, self._slot_size = VERSION, SLOTS, SLOT_SIZE
        struct.pack_into("<7I", self.mm, 4, version, self._slots, self._slot_size, codec, width, height, max_fps)
        struct.pack_into("<QI", self.mm, 32, 0, 0)
        struct.pack_into("<Q", self.mm, 48, freq)
        struct.pack_into("<I", self.mm, 92, colour if codec in RAW_CODECS else 0)
        if features is not None:
            struct.pack_into("<I", self.mm, 64, features)
        (session,) = struct.unpack_from("<I", self.mm, 44)
        struct.pack_into("<I", self.mm, 44, session + 1)
        struct.pack_into("<I", self.mm, 0, MAGIC)

    def push(self, data, frames_lost: int = 0, recovered: bool = False, new: bool | None = None) -> None:
        """One frame: compressed bytes, or a decoded frame (bytes or a uint8 numpy array, rows packed tightly).
        new: the writer's own word on it (True: new content, False: a repeat of the frame before; None: unknown)."""
        (seq,) = struct.unpack_from("<Q", self.mm, 32)
        off = HEADER + (seq % self._slots) * self._slot_size
        n = data.nbytes if isinstance(data, np.ndarray) else len(data)
        size = min(n, self._slot_size - SLOT_HEADER)
        flags = (FLAG_TRUNCATED if size < n else 0) | {True: FLAG_NEW, False: FLAG_REPEAT, None: 0}[new]
        struct.pack_into("<Q", self.mm, off, U64_MAX)
        struct.pack_into("<QIiII", self.mm, off + 8, qpc(), size, frames_lost, int(recovered), flags)
        if isinstance(data, np.ndarray):
            np.frombuffer(self.mm, np.uint8, size, off + SLOT_HEADER)[:] = data.reshape(-1)[:size]
        else:
            self.mm[off + SLOT_HEADER: off + SLOT_HEADER + size] = bytes(data[:size])
        struct.pack_into("<Q", self.mm, off, seq)
        struct.pack_into("<Q", self.mm, 32, seq + 1)
        self.event.set()

    def take_idr_request(self) -> bool:
        (r,) = struct.unpack_from("<I", self.mm, 40)
        if r:
            struct.pack_into("<I", self.mm, 40, 0)
        return bool(r)

    def set_features(self, features: int) -> None:
        struct.pack_into("<I", self.mm, 64, features)

    def take_press_request(self) -> int:
        (buttons,) = struct.unpack_from("<I", self.mm, 68)
        if buttons:
            struct.pack_into("<I", self.mm, 68, 0)
        return buttons

    def press_done(self) -> None:
        """The press went to the (fake) PS5 now."""
        struct.pack_into("<Q", self.mm, 80, qpc())
        (count,) = struct.unpack_from("<I", self.mm, 76)
        struct.pack_into("<I", self.mm, 76, count + 1)

    def audio_delay_ms(self) -> int:
        return struct.unpack_from("<I", self.mm, 88)[0]

    def end(self) -> None:
        struct.pack_into("<I", self.mm, 0, 0)
        self.event.set()

    def close(self) -> None:
        self.event.close()
        self.mm.close()


class AudioTap:
    """The game's sound from chiaki-ng (frametap.h, "Sound ring"): 16-bit PCM chunks timed by their arrival, for
    instant replay. A chiaki-ng from before this ring never writes it (its `features` lack FEATURE_AUDIO_TAP)."""

    def __init__(self, name: str = NAME + AUDIO_SUFFIX) -> None:
        self.mm = open_map(name, AUDIO_TOTAL)
        self.next: int | None = None  # the next chunk to read; None: from wherever the writer is now

    def live(self) -> bool:
        return struct.unpack_from("<I", self.mm, 0)[0] == AUDIO_MAGIC

    def reset(self, from_start: bool = False) -> None:
        """Forget the read position (a new stream): the next read starts at the writer's position, or at its first
        chunk."""
        self.next = 0 if from_start else None

    def read_new(self) -> list:
        """Every chunk written since the last call (at most the ring's worth), as framegen.io.Sound."""
        from .io import Sound
        magic = struct.unpack_from("<I", self.mm, 0)[0]
        write_seq, freq = struct.unpack_from("<QQ", self.mm, 24)
        if magic != AUDIO_MAGIC or not freq:
            self.next = None
            return []
        if self.next is None or self.next > write_seq:  # first read, or the writer started over
            self.next = write_seq
        self.next = max(self.next, write_seq - AUDIO_SLOTS + 1)  # fell a whole ring behind: what's left
        out = []
        now_q, now_t = qpc(), time.perf_counter()
        while self.next < write_seq:
            seq = self.next
            self.next += 1
            off = AUDIO_HEADER + (seq % AUDIO_SLOTS) * AUDIO_SLOT_SIZE
            s0, q, size, rate, channels = struct.unpack_from("<QQIII", self.mm, off)
            if s0 != seq or size > AUDIO_SLOT_SIZE - AUDIO_SLOT_HEADER:
                continue
            data = self.mm[off + AUDIO_SLOT_HEADER: off + AUDIO_SLOT_HEADER + size]
            if struct.unpack_from("<Q", self.mm, off)[0] != seq:  # rewritten while we copied
                continue
            out.append(Sound(now_t - (now_q - q) / freq, data, rate, channels))
        return out

    # Writer side, for tests and the fake PS5 (scripts/frametap_fake.py); chiaki-ng uses frametap.c.
    def begin(self, rate: int = 48000, channels: int = 2) -> None:
        struct.pack_into("<I", self.mm, 0, 0)
        struct.pack_into("<5I", self.mm, 4, 1, AUDIO_SLOTS, AUDIO_SLOT_SIZE, rate, channels)
        struct.pack_into("<QQ", self.mm, 24, 0, qpc_freq())
        struct.pack_into("<I", self.mm, 0, AUDIO_MAGIC)

    def push(self, pcm: bytes, t_qpc: int | None = None) -> None:
        rate, channels = struct.unpack_from("<II", self.mm, 16)
        cap = (AUDIO_SLOT_SIZE - AUDIO_SLOT_HEADER) // (2 * channels) * (2 * channels)
        q0 = qpc() if t_qpc is None else t_qpc
        (freq,) = struct.unpack_from("<Q", self.mm, 32)
        for i in range(0, len(pcm), cap):
            part = pcm[i:i + cap]
            (seq,) = struct.unpack_from("<Q", self.mm, 24)
            off = AUDIO_HEADER + (seq % AUDIO_SLOTS) * AUDIO_SLOT_SIZE
            struct.pack_into("<Q", self.mm, off, U64_MAX)
            struct.pack_into("<QIII", self.mm, off + 8, q0 + round(i // (2 * channels) * freq / rate), len(part),
                             rate, channels)
            self.mm[off + AUDIO_SLOT_HEADER: off + AUDIO_SLOT_HEADER + len(part)] = part
            struct.pack_into("<Q", self.mm, off, seq)
            struct.pack_into("<Q", self.mm, 24, seq + 1)

    def end(self) -> None:
        struct.pack_into("<I", self.mm, 0, 0)

    def close(self) -> None:
        self.mm.close()


def _nal_types(data: bytes, codec: str):
    """NAL unit types in an Annex-B access unit."""
    i, n = 0, len(data)
    while True:
        i = data.find(b"\x00\x00\x01", i)
        if i < 0 or i + 3 >= n:
            return
        b = data[i + 3]
        yield (b >> 1) & 0x3F if codec == "hevc" else b & 0x1F
        i += 3


def is_keyframe(data: bytes, codec: str) -> bool:
    # H.264: IDR slice (5). HEVC: IRAP pictures, BLA/IDR/CRA (16-21).
    return any((t == 5) if codec == "h264" else (16 <= t <= 21) for t in _nal_types(data, codec))


PARAM_NALS = {"hevc": {32, 33, 34}, "h264": {7, 8}}  # VPS/SPS/PPS, SPS/PPS


def nal_units(data: bytes, codec: str):
    """(type, unit with a 3-byte start code) for each NAL unit in an Annex-B access unit."""
    starts, i = [], 0
    while (i := data.find(b"\x00\x00\x01", i)) >= 0:
        starts.append(i)
        i += 3
    for k, s in enumerate(starts):
        if s + 3 >= len(data):
            return
        b = data[s + 3]
        end = starts[k + 1] if k + 1 < len(starts) else len(data)
        yield ((b >> 1) & 0x3F if codec == "hevc" else b & 0x1F), b"\x00\x00\x01" + data[s + 3:end]


def param_sets(data: bytes, codec: str) -> bytes:
    """The codec header NAL units in an access unit, or b"" if none.

    chiaki-ng sends these once, as a packet of their own at the start of a stream (the PS5 delivers them out of
    band), so a decoder that missed them can't decode anything until the next stream.
    """
    return b"".join(u for t, u in nal_units(data, codec) if t in PARAM_NALS[codec])


class SizeRepeatDetector:
    """New frame or repeat, from compressed size alone. (Not the pipeline's call, which is the picture's: see the
    module docstring. Kept for the stream check, test mode and the frame log.)

    A frame is new if it's at least `ratio` of a typical recent new frame (the 75th percentile of the last 16
    non-keyframe sizes: in a 30 fps game at 60 Hz half the frames are new, so that lands on a new one).
    Unsure cases lean towards "repeat": that only shows a real frame without a generated one before it,
    while calling a repeat new makes the scheduler stall a tick. Keyframes are big whatever they contain: if
    the recent frames alternate new/repeat (a 30 fps game), a keyframe gets the opposite of the frame before
    it; otherwise it returns None and the pipeline falls back to the image check.
    """

    BIG_RUN = 6  # this many "big" frames among the last 2 * BIG_RUN aren't refresh frames but a busier scene

    def __init__(self, ratio: float = 0.4, window: int = 16) -> None:
        from collections import deque
        self.ratio = ratio
        self._recent: deque[tuple[int, bool]] = deque(maxlen=2 * self.BIG_RUN)  # (size, big) of the last frames
        self.sizes: deque[int] = deque(maxlen=window)
        self.decisions: deque[bool] = deque(maxlen=6)
        self.ref: float | None = None  # the last update's typical new frame size (None: keyframe or warming up)
        self.big = False  # the last update was treated as a keyframe for its size

    def update(self, size: int, keyframe: bool) -> bool | None:
        # The PS5's refresh frames are ordinary slices by NAL type, but far bigger than anything recent.
        self.big = not keyframe and len(self.sizes) >= 8 and size > 4 * float(np.percentile(self.sizes, 75))
        # After a still scene (a menu, a pause: ~1 KB frames), the new frames of the motion that follows (and often
        # the repeats) look big, and big frames never joined the sizes, so the check stayed silent for the rest of
        # the session (55-91% of ticks in real logs). Many of them recently is the new normal: start from those.
        if not keyframe:
            self._recent.append((size, self.big))
        if sum(b for _, b in self._recent) >= self.BIG_RUN:
            self.sizes.clear()
            self.sizes.extend(s for s, _ in list(self._recent)[:-1])  # (this frame joins below, as an ordinary one)
            self._recent.clear()
            self.big = False
        self.ref = None
        if keyframe or self.big:
            d = list(self.decisions)
            alternating = len(d) >= 4 and all(a != b for a, b in zip(d, d[1:]))
            decision = (not d[-1]) if alternating else None
        else:
            self.sizes.append(size)
            if len(self.sizes) >= 4:
                self.ref = float(np.percentile(self.sizes, 75))
            decision = (size >= self.ratio * self.ref) if self.ref is not None else None
        if decision is not None:
            self.decisions.append(decision)
        return decision


_NV12_TO_BGR_CUDA_SRC = r"""
extern "C" __global__ void nv12_to_bgr_u8(
    const unsigned char* __restrict__ y_plane,
    const unsigned char* __restrict__ uv_plane,
    unsigned char* __restrict__ out_bgr,
    int H, int W,
    int uv_h, int uv_w,
    int y_row_stride, int y_col_stride,
    int uv_row_stride, int uv_col_stride, int uv_ch_stride,
    float yo, float ys, float cs,
    float c_bu, float c_gu, float c_gv, float c_rv
) {
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= H || col >= W) return;

    float lum = ((float)y_plane[row * y_row_stride + col * y_col_stride] - yo) * ys;

    float src_y = ((float)row + 0.5f) * ((float)uv_h / (float)H) - 0.5f;
    float src_x = (float)col * ((float)uv_w / (float)W);
    if (src_y < 0.0f) src_y = 0.0f;
    if (src_x < 0.0f) src_x = 0.0f;
    float max_y = (float)(uv_h - 1);
    float max_x = (float)(uv_w - 1);
    if (src_y > max_y) src_y = max_y;
    if (src_x > max_x) src_x = max_x;

    int y0 = (int)src_y;
    int x0 = (int)src_x;
    int y1 = (y0 + 1 < uv_h) ? (y0 + 1) : y0;
    int x1 = (x0 + 1 < uv_w) ? (x0 + 1) : x0;
    float wy = src_y - (float)y0;
    float wx = src_x - (float)x0;
    float w00 = (1.0f - wy) * (1.0f - wx);
    float w01 = (1.0f - wy) * wx;
    float w10 = wy * (1.0f - wx);
    float w11 = wy * wx;

    int off00 = y0 * uv_row_stride + x0 * uv_col_stride;
    int off01 = y0 * uv_row_stride + x1 * uv_col_stride;
    int off10 = y1 * uv_row_stride + x0 * uv_col_stride;
    int off11 = y1 * uv_row_stride + x1 * uv_col_stride;

    float u = (w00 * (float)uv_plane[off00]
             + w01 * (float)uv_plane[off01]
             + w10 * (float)uv_plane[off10]
             + w11 * (float)uv_plane[off11] - 128.0f) * cs;
    float v = (w00 * (float)uv_plane[off00 + uv_ch_stride]
             + w01 * (float)uv_plane[off01 + uv_ch_stride]
             + w10 * (float)uv_plane[off10 + uv_ch_stride]
             + w11 * (float)uv_plane[off11 + uv_ch_stride] - 128.0f) * cs;

    float b = lum + c_bu * u;
    float g = lum + c_gu * u + c_gv * v;
    float r = lum + c_rv * v;

    b = b < 0.0f ? 0.0f : (b > 255.0f ? 255.0f : b);
    g = g < 0.0f ? 0.0f : (g > 255.0f ? 255.0f : g);
    r = r < 0.0f ? 0.0f : (r > 255.0f ? 255.0f : r);

    int out_off = (row * W + col) * 3;
    out_bgr[out_off + 0] = (unsigned char)rintf(b);
    out_bgr[out_off + 1] = (unsigned char)rintf(g);
    out_bgr[out_off + 2] = (unsigned char)rintf(r);
}
"""


def _nv12_to_bgr_params(colorspace: int, color_range: int, height: int = 0) -> tuple[float, float, float, float, float, float, float]:
    """(yo, ys, cs, c_bu, c_gu, c_gv, c_rv) matching FFmpeg's NV12 -> BGR24 conversion."""
    is_bt709 = colorspace == 1 or (colorspace in (0, 2) and height >= 720)
    kr, kb = (0.2126, 0.0722) if is_bt709 else (0.299, 0.114)
    kg = 1.0 - kr - kb
    ys, yo, cs = (1.0, 0.0, 1.0) if color_range == 2 else (255.0 / 219.0, 16.0, 255.0 / 224.0)
    return (yo, ys, cs, 2.0 * (1.0 - kb), -2.0 * kb * (1.0 - kb) / kg, -2.0 * kr * (1.0 - kr) / kg, 2.0 * (1.0 - kr))


class _FusedNV12Kernel:
    """Lazily compiles _NV12_TO_BGR_CUDA_SRC via NVRTC and launches it on PyTorch's current CUDA stream."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._failed = False
        self._drv = None
        self._ptx: bytes | None = None
        self._funcs: dict[int, tuple[ctypes.c_void_p, ctypes.c_void_p]] = {}  # cuCtx -> (cuMod, cuFunc)
        self._out_bufs: dict[tuple[int, int, str], tuple[object, object]] = {}
        self._out_idx: dict[tuple[int, int, str], int] = {}
        self._inflight: dict[tuple[int, int, str], list[tuple[object, object] | None]] = {}

    @staticmethod
    def _load_nvrtc():
        import ctypes.util
        import torch
        torch_lib = Path(torch.__file__).resolve().parent / "lib"
        candidates: list[str] = []
        if sys.platform == "win32":
            candidates.extend(str(p) for p in sorted(torch_lib.glob("nvrtc64_*.dll"), reverse=True))
            candidates.extend(("nvrtc64_120_0.dll", "nvrtc64_124_0.dll", "nvrtc64_126_0.dll", "nvrtc64_112_0.dll"))
        else:
            candidates.extend(str(p) for p in sorted(torch_lib.glob("libnvrtc.so*"), reverse=True))
            found = ctypes.util.find_library("nvrtc")
            if found:
                candidates.append(found)
            candidates.append("libnvrtc.so")
        for path in candidates:
            try:
                return ctypes.CDLL(path)
            except OSError:
                continue
        return None

    def _compile_ptx(self) -> bytes:
        nvrtc = self._load_nvrtc()
        if nvrtc is None:
            raise RuntimeError("NVRTC library not found")
        nvrtc.nvrtcCreateProgram.argtypes = [
            ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p, ctypes.c_char_p,
            ctypes.c_int, ctypes.c_void_p, ctypes.c_void_p,
        ]
        nvrtc.nvrtcCompileProgram.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_char_p)]
        nvrtc.nvrtcGetProgramLogSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetProgramLog.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcGetPTXSize.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_size_t)]
        nvrtc.nvrtcGetPTX.argtypes = [ctypes.c_void_p, ctypes.c_char_p]
        nvrtc.nvrtcDestroyProgram.argtypes = [ctypes.POINTER(ctypes.c_void_p)]

        prog = ctypes.c_void_p()
        src = _NV12_TO_BGR_CUDA_SRC.encode("utf-8")
        if nvrtc.nvrtcCreateProgram(ctypes.byref(prog), src, b"nv12_to_bgr.cu", 0, None, None) != 0:
            raise RuntimeError("nvrtcCreateProgram failed")
        try:
            # Compile to PTX (no arch-specific SASS) so the CUDA driver JITs it for any GPU including Blackwell.
            if nvrtc.nvrtcCompileProgram(prog, 0, None) != 0:
                log_sz = ctypes.c_size_t()
                log_txt = ""
                if nvrtc.nvrtcGetProgramLogSize(prog, ctypes.byref(log_sz)) == 0 and log_sz.value > 1:
                    log_buf = ctypes.create_string_buffer(log_sz.value)
                    if nvrtc.nvrtcGetProgramLog(prog, log_buf) == 0:
                        log_txt = log_buf.value.decode("utf-8", errors="replace").strip()
                if log_txt:
                    print(f"[chiaki] nvrtcCompileProgram log:\n{log_txt}", file=sys.stderr)
                raise RuntimeError(f"nvrtcCompileProgram failed: {log_txt}" if log_txt else "nvrtcCompileProgram failed")
            size = ctypes.c_size_t()
            if nvrtc.nvrtcGetPTXSize(prog, ctypes.byref(size)) != 0 or size.value == 0:
                raise RuntimeError("nvrtcGetPTXSize failed")
            buf = ctypes.create_string_buffer(size.value)
            if nvrtc.nvrtcGetPTX(prog, buf) != 0:
                raise RuntimeError("nvrtcGetPTX failed")
            return buf.raw
        finally:
            nvrtc.nvrtcDestroyProgram(ctypes.byref(prog))

    def _get_func(self):
        if self._failed:
            return None, None
        with self._lock:
            if self._failed:
                return None, None
            try:
                if self._drv is None:
                    drv = ctypes.CDLL("nvcuda.dll" if sys.platform == "win32" else "libcuda.so.1")
                    drv.cuCtxGetCurrent.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
                    drv.cuModuleLoadData.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_char_p]
                    drv.cuModuleGetFunction.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p, ctypes.c_char_p]
                    drv.cuLaunchKernel.argtypes = [
                        ctypes.c_void_p,
                        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                        ctypes.c_uint, ctypes.c_uint, ctypes.c_uint,
                        ctypes.c_uint, ctypes.c_void_p,
                        ctypes.POINTER(ctypes.c_void_p), ctypes.c_void_p,
                    ]
                    self._drv = drv
                if self._ptx is None:
                    self._ptx = self._compile_ptx()
                ctx = ctypes.c_void_p()
                if self._drv.cuCtxGetCurrent(ctypes.byref(ctx)) != 0 or not ctx.value:
                    return None, None
                entry = self._funcs.get(int(ctx.value))
                if entry is None:
                    mod = ctypes.c_void_p()
                    fn = ctypes.c_void_p()
                    if self._drv.cuModuleLoadData(ctypes.byref(mod), self._ptx) != 0:
                        raise RuntimeError("cuModuleLoadData failed")
                    if self._drv.cuModuleGetFunction(ctypes.byref(fn), mod, b"nv12_to_bgr_u8") != 0:
                        raise RuntimeError("cuModuleGetFunction failed")
                    entry = (mod, fn)
                    self._funcs[int(ctx.value)] = entry
                return self._drv, entry[1]
            except Exception as e:  # noqa: BLE001
                self._failed = True
                print(f"[chiaki] fused NV12->BGR CUDA kernel unavailable ({e}); using PyTorch fallback")
                return None, None

    def flush(self) -> None:
        """Synchronize any in-flight kernel events and release held NVDEC frame references."""
        with self._lock:
            pending: list[tuple[object, object]] = []
            for slots in self._inflight.values():
                for i, entry in enumerate(slots):
                    if entry is not None:
                        pending.append(entry)
                        slots[i] = None
        for ev, _ in pending:
            ev.synchronize()

    def launch(self, y, uv, colorspace: int, color_range: int, hold=None):
        import torch
        if (y.dtype != torch.uint8 or uv.dtype != torch.uint8 or not y.is_cuda or not uv.is_cuda
                or y.ndim != 2 or uv.ndim != 3 or uv.shape[2] != 2):
            return None
        drv, fn = self._get_func()
        if fn is None:
            return None
        h, w = int(y.shape[0]), int(y.shape[1])
        uv_h, uv_w = int(uv.shape[0]), int(uv.shape[1])
        key = (h, w, str(y.device))
        with self._lock:
            pair = self._out_bufs.get(key)
            if pair is None:
                pair = (
                    torch.empty((h, w, 3), dtype=torch.uint8, device=y.device),
                    torch.empty((h, w, 3), dtype=torch.uint8, device=y.device),
                )
                self._out_bufs[key] = pair
                self._out_idx[key] = 0
                self._inflight[key] = [None, None]
            idx = self._out_idx[key]
            self._out_idx[key] = idx ^ 1
            out = pair[idx]
            prev = self._inflight[key][idx]
            self._inflight[key][idx] = None
        if prev is not None:
            prev[0].synchronize()
            del prev
        yo, ys, cs, c_bu, c_gu, c_gv, c_rv = _nv12_to_bgr_params(colorspace, color_range, h)
        args = (
            ctypes.c_void_p(y.data_ptr()),
            ctypes.c_void_p(uv.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(h), ctypes.c_int(w),
            ctypes.c_int(uv_h), ctypes.c_int(uv_w),
            ctypes.c_int(y.stride(0)), ctypes.c_int(y.stride(1)),
            ctypes.c_int(uv.stride(0)), ctypes.c_int(uv.stride(1)), ctypes.c_int(uv.stride(2)),
            ctypes.c_float(yo), ctypes.c_float(ys), ctypes.c_float(cs),
            ctypes.c_float(c_bu), ctypes.c_float(c_gu), ctypes.c_float(c_gv), ctypes.c_float(c_rv),
        )
        ptrs = (ctypes.c_void_p * len(args))(*(ctypes.addressof(a) for a in args))
        cur_stream = torch.cuda.current_stream(y.device)
        stream = ctypes.c_void_p(int(cur_stream.cuda_stream))
        rc = drv.cuLaunchKernel(
            fn,
            (w + 15) // 16, (h + 15) // 16, 1,
            16, 16, 1,
            0, stream, ptrs, None,
        )
        if rc != 0:
            self._failed = True
            return None
        ev = torch.cuda.Event()
        ev.record(cur_stream)
        with self._lock:
            self._inflight[key][idx] = (ev, (hold, y, uv))
        return out


_FUSED_NV12_KERNEL = _FusedNV12Kernel()


def _cuda_frame_to_bgr_torch(y, uv, colorspace: int, color_range: int):
    """Reference PyTorch NV12 -> HxWx3 uint8 BGR conversion (fallback when NVRTC isn't available)."""
    import torch
    import torch.nn.functional as F
    h, w = int(y.shape[0]), int(y.shape[1])
    uv_w = int(uv.shape[1])
    yo, ys, cs, c_bu, c_gu, c_gv, c_rv = _nv12_to_bgr_params(colorspace, color_range, h)
    lum = (y.float() - yo) * ys
    c_v = F.interpolate(uv.permute(2, 0, 1).unsqueeze(0).float() - 128.0, size=(h, uv_w),
                        mode="bilinear", align_corners=False)[0]
    src_x = (torch.arange(w, device=y.device, dtype=torch.float32) * (uv_w / float(w))).clamp_(0.0, float(uv_w - 1))
    x0 = src_x.long()
    x1 = (x0 + 1).clamp_(max=uv_w - 1)
    wx = (src_x - x0.float()).view(1, 1, w)
    c = (c_v[:, :, x0] * (1.0 - wx) + c_v[:, :, x1] * wx) * cs
    u, v = c[0], c[1]
    b = lum + c_bu * u
    g = lum + c_gu * u + c_gv * v
    r = lum + c_rv * v
    return torch.stack([b, g, r], -1).clamp_(0, 255).round_().to(torch.uint8)


def cuda_frame_to_bgr(fr):
    """A decoded NVDEC frame (NV12, still on the GPU) -> HxWx3 uint8 BGR CUDA tensor, without leaving the GPU.

    Matches FFmpeg's CPU conversion: BT.709 when tagged or HD (>=720p) with unspecified colorspace, else BT.601;
    limited range unless tagged full; H.264 Type-0 left-cosited horizontal chroma siting.
    Uses a single fused CUDA kernel (bilinear UV + color matrix + clamp/round/uint8) into a preallocated
    2-buffer ping-pong tensor on PyTorch's current stream, holding the source NVDEC frame alive via an async
    cuda.Event until that buffer slot is reused so FFmpeg never recycles the surface while the kernel reads it.
    """
    import torch
    if fr.format.name != "cuda":  # the decoder fell back to software for this stream
        return torch.from_numpy(fr.to_ndarray(format="bgr24")).cuda()
    y = torch.from_dlpack(fr.planes[0].__dlpack__())   # H x W
    uv = torch.from_dlpack(fr.planes[1].__dlpack__())  # H/2 x W/2 x 2
    out = _FUSED_NV12_KERNEL.launch(y, uv, fr.colorspace, fr.color_range, hold=fr)
    if out is not None:
        return out
    out = _cuda_frame_to_bgr_torch(y, uv, fr.colorspace, fr.color_range)
    if y.is_cuda:
        torch.cuda.current_stream(y.device).synchronize()
    return out


class ChiakiSource:
    """Decoded frames from the tap, as Ticks. Waits for chiaki-ng to start streaming.

    Joins the live stream (never replays old frames) and asks the PS5 for a keyframe whenever it can't decode:
    at start, after falling more than the ring's 32 frames behind, or after a decode error.
    """
    app, tag = False, "[chiaki]"  # chiaki-ng's frames (__init__: or another app's, through framegen/tap.py)

    def __init__(self, decoder: str = "sw", wait_s: float = 120.0, log_path: str | None = None,
                 tap_name: str = NAME, test=None, app: bool = False) -> None:
        """test: a framegen.sources.TestMode ("Simulate a 30 fps game"), applied as frames are decoded, before
        the queue, so a frame dropped from a full queue never shifts which frames it keeps. app: another app writes
        chiaki-ng's layout (framegen/tap.py open_tap): the log and notes say "[tap]" and "the app", not chiaki-ng."""
        from .sources import source_info
        self.app, self.tag = app, "[tap]" if app else "[chiaki]"
        self.tap = FrameTap(tap_name, tap_name + "-frame")
        self.sound = AudioTap(tap_name + AUDIO_SUFFIX)  # the game's sound, for instant replay
        self._sound_session: int | None = None
        self.test = test
        self._last_idr = 0.0
        self._header_file = Path(tempfile.gettempdir()) / (tap_name.split("\\")[-1] + "-header.bin")
        self.decoder_kind = decoder
        t_end = time.perf_counter() + wait_s
        printed = False
        while not (h := self.tap.header()).live:
            if not printed:
                print("[tap] waiting for an app to start sending frames..." if app else
                      "[chiaki] waiting for chiaki-ng to start a stream (run it with CHIAKI_FRAMETAP=1)...")
                printed = True
            if time.perf_counter() > t_end:
                raise TimeoutError("no app is sending frames to the frame tap" if app else
                                   "no chiaki-ng stream on the frame tap")
            self.tap.wait(0.25)
        self.w, self.h, self.codec_id = h.width, h.height, h.codec
        self.info = source_info("chiaki", hz=float(h.max_fps))
        what = f"{CODEC_NAMES.get(h.codec, h.codec)} {h.width}x{h.height}, up to {h.max_fps} fps"
        print(f"[tap] frames: {what}" if app else f"[chiaki] stream: {what}")  # (the launcher reads both)
        if h.max_fps < 60:
            print(f"[tap] WARNING: the stream is set to {h.max_fps} fps. Send 60 so the pipeline sees the console's "
                  "60 Hz output (it expects repeats, not every other frame missing)." if app else
                  "[chiaki] WARNING: the stream is set to 30 fps. Set chiaki-ng's frame rate to 60 so the "
                  "pipeline sees the PS5's 60 Hz output (it expects repeats, not every other frame missing).")
        self.hdr_stream = h.codec == 2  # chiaki-ng's H.265 HDR codec: 10-bit HDR10 from the PS5
        self.convert = None  # set_hdr(): a framegen.hdr.StreamConverter, when SmoovPlay's HDR setting is on
        self._hdr_told = False
        self.q: LatestFrameMailbox = LatestFrameMailbox()
        self.dropped = 0  # decoded frames dropped because the pipeline fell behind
        self.skipped = 0  # stream frames never decoded (ring overrun or waiting for a keyframe)
        self.idr_requests = 0
        self.check = StreamCheck()  # how well the size check read this game: reported when the session ends
        self._stop = False
        self._log = None
        if log_path:
            f = open(log_path, "w", newline="")
            self._log = (f, csv.writer(f))
            self._log[1].writerow(["tick", "seq", "t_arrival_ms", "size", "frames_lost", "recovered", "keyframe",
                                   "decode_ms", "queue_ms", "size_says_new", "image_says_new", "max_diff",
                                   "mean_diff", "action", "proc_ms"])
        self._t0 = time.perf_counter()
        self._beat = self._t0  # when the reader's loop last went round (why() says if it stopped)
        self._no_header = False  # joined a running stream whose codec header went past, and none was saved
        self._ring = (-1, self._t0)  # why(): the ring's write count when it last moved, and when
        self.decoder_dpb_delay: int | None = None  # packets held in DPB before first frame emitted (0 = verified)
        # The reader starts right away, not when the pipeline is ready: a new stream's first packet is its codec
        # header, and the ring only holds half a second. But until the pipeline asks for frames (self._go: its loop
        # has started) it only watches for that header, with no GPU work, so the decoder never starts up on the GPU
        # alongside the pipeline's own set-up (TensorRT, VFG, the window's GPU sharing). Frames decoded then would
        # be dropped anyway. hold() stops it decoding again while the pipeline restarts.
        self._go = threading.Event()
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _new_decoder(self, codec: str):
        import av
        from av.codec.hwaccel import HWAccel
        if self.decoder_kind == "sw":
            ctx = av.CodecContext.create(codec, "r")
            ctx.thread_type = "SLICE"  # frame threading would delay output by a frame per thread
        elif self.decoder_kind == "cuda":
            # NVDEC, and the frames stay on the GPU (handed to PyTorch through DLPack). current_ctx: decode in
            # PyTorch's context, made current on this thread in _reader. FFmpeg's default would be a second context,
            # which takes turns on the GPU with RIFE; primary_ctx refuses because PyTorch set it up differently.
            ctx = av.CodecContext.create(codec, "r", hwaccel=HWAccel(
                "cuda", options={"primary_ctx": "0", "current_ctx": "1"}, is_hw_owned=True))
            ctx.thread_type = "SLICE"
            ctx.thread_count = 1  # GPU ASIC decodes on its own; 1 CPU thread guarantees zero frame-thread queue
        else:
            ctx = av.CodecContext.create(codec, "r", hwaccel=HWAccel(self.decoder_kind))
            ctx.thread_type = "SLICE"
            ctx.thread_count = 1
        ctx.flags |= av.codec.context.Flags.low_delay
        return ctx

    def _check_dpb(self, dec, dpb_lag: int, *, initial: bool = True, pre_idr: int = 0) -> bool:
        """Check decoder DPB lag / B-frame / delay flags and log status. Returns True if a warning was emitted."""
        self.decoder_dpb_delay = dpb_lag
        has_b = bool(getattr(dec, "has_b_frames", False))
        ctx_delay = int(getattr(dec, "delay", 0) or 0)
        if dpb_lag > 0 or has_b or ctx_delay > 0:
            print(f"{self.tag} WARNING: {self.decoder_kind} decoder buffered {dpb_lag} frame(s) in DPB "
                  f"(has_b_frames={int(has_b)}, delay={ctx_delay})")
            return True
        if initial:
            extra = f", skipped {pre_idr} pre-IDR packet(s)" if pre_idr > 0 else ""
            print(f"{self.tag} {self.decoder_kind} decoder: 0-frame DPB verified "
                  f"(low_delay=1, has_b_frames=0, delay=0{extra})")
        return False

    def _reader(self) -> None:
        import av

        from .winutil import boost_thread
        boost_thread()  # the decoder thread feeds every frame, so it gets the same treatment as the pipeline's
        on_gpu = False      # PyTorch's CUDA context made current on this thread, for NVDEC (once going)
        session, seq, dec, codec = None, 0, None, None
        params = b""        # this stream's codec header, once seen
        pending = b""       # the header, still to go into the decoder (with the next picture: not every decoder
        #                     takes it on its own)
        decoded = False     # a picture has come out of this stream
        dpb_held = 0        # packets fed to the current decoder before its first emitted picture
        pkt_idx = 0         # monotonic packet index fed to dec (tagged via pkt.pts to measure true DPB lag)
        dpb_recheck = False # re-check has_b_frames / delay after mid-stream SPS/PPS change
        damaged_left = 0    # frames still treated as damaged after a gap
        errors = 0          # decode errors in a row
        passed = False      # frames went by undecoded (joined late, the pipeline not ready yet, or held)
        held = -1           # the frame the reader waits at for the pipeline to be ready (see below)
        late = False
        t_join, told = 0.0, False
        sizes = SizeRepeatDetector()
        while not self._stop:
            self._beat = time.perf_counter()
            h = self.tap.header()
            if not h.live:
                if decoded:  # the stream we were showing ended
                    print("[tap] the app stopped sending frames" if self.app else "[chiaki] stream ended")
                    self.q.close()
                    return
                self.tap.wait(0.25)  # not started yet, or being reconnected
                continue
            if h.session != session:  # a new stream
                session, codec = h.session, CODECS.get(h.codec, "h264")
                if (h.width, h.height) != (self.w, self.h):
                    print(f"{self.tag} stream size changed to {h.width}x{h.height}; restart run.py", file=sys.stderr)
                    self.q.close()
                    return
                dec, params, pending, decoded, dpb_held, pkt_idx, dpb_recheck = None, b"", b"", False, 0, 0, False
                damaged_left, errors, held = 0, 0, -1
                sizes = SizeRepeatDetector()
                # Read it from its first packet (the codec header) while that's still in the ring.
                seq = 0 if h.write_seq <= SLOTS - 4 else h.write_seq
                late = passed = seq > 0
                if late:  # joining a running stream: an earlier run.py may have saved its header
                    params = self._load_header(session, h.codec)
                    if params:
                        print(f"{self.tag} joined a running stream, using its saved codec header")
                    else:
                        self._ask_idr()
                self._no_header = late and not params
                t_join, told = time.perf_counter(), False
            going = self._go.is_set()
            if going and dec is None:
                if self.decoder_kind == "cuda" and not on_gpu:
                    import torch
                    torch.zeros(1, device="cuda")  # PyTorch's CUDA context, current on this thread for the decoder
                    on_gpu = True
                dec, pending, dpb_held, pkt_idx, dpb_recheck = self._new_decoder(codec), params, 0, 0, False
            if going and passed and params:
                # Frames went by undecoded: ask the PS5 for a refresh frame and hold what comes out until the
                # picture is whole again.
                passed, damaged_left = False, 60
                self._ask_idr()
            if late and not params and not told and time.perf_counter() - t_join > 3:
                print(f"{self.tag} this stream was already running, so its codec header went past before the "
                      "pipeline started. " + ("Restart the stream in the app" if self.app else
                                              "In chiaki-ng press Ctrl+O, click the x to end the stream, then connect "
                                              "again") + ": the pipeline picks up the new stream from its start.")
                told = True
            if h.write_seq == seq:
                self.tap.heartbeat()
                self.tap.wait(0.1)
                continue
            if h.write_seq - seq > SLOTS - 4:  # fell behind: the oldest slots are being overwritten
                if going:
                    self.skipped += h.write_seq - seq
                    damaged_left = DAMAGE_HOLD
                    self._ask_idr()
                else:
                    passed = True
                seq = h.write_seq
                continue
            if not going and seq == held:  # at the stream's first picture, waiting for the pipeline (see below)
                self.tap.heartbeat()
                self.tap.wait(0.1)
                continue
            p = self.tap.read(seq, h.qpc_freq)
            seq += 1
            if p is None:
                if going:
                    self.skipped += 1
                    damaged_left = DAMAGE_HOLD
                    self._ask_idr()
                else:
                    passed = True
                continue
            ps = param_sets(p.data, codec)
            if ps:
                if ps != params:
                    self._save_header(session, h.codec, ps)
                    if decoded:
                        dpb_recheck = True
                params, self._no_header = ps, False
                if all(t in PARAM_NALS[codec] for t, _ in nal_units(p.data, codec)):  # the header on its own
                    pending = ps
                    continue
            elif not params:  # nothing is decodable without the header
                if going:
                    self.skipped += 1
                continue
            if not going:  # the pipeline isn't ready: only the header was wanted
                if not passed:
                    # A picture: leave it in the ring, which holds half a second, and start decoding there once the
                    # pipeline is ready, as if it had been all along. If it takes longer, the ring moves on (passed)
                    # and the PS5 is asked for a refresh frame instead.
                    seq = held = p.seq
                continue
            key = is_keyframe(p.data, codec)
            t = time.perf_counter()
            try:
                in_idx = pkt_idx
                pkt_idx += 1
                pkt = av.Packet(pending + p.data if pending else p.data)
                pkt.pts = in_idx
                with CAPTURE_LOCK:  # no GPU decoding while the pipeline captures its CUDA graphs (startup only)
                    frames = dec.decode(pkt)
                    images = [self._image(fr) for fr in frames]
                pending, errors = b"", 0
            except av.error.FFmpegError as e:
                # Keep the decoder (it holds the header and reference frames) and hold the next few frames; the
                # PS5 answers a keyframe request with a refresh frame.
                print(f"{self.tag} decode error at frame {p.seq}: {e}", file=sys.stderr)
                damaged_left, errors = DAMAGE_HOLD, errors + 1
                if errors >= 30:  # stuck: start a decoder afresh from the saved header
                    _FUSED_NV12_KERNEL.flush()
                    dec, errors, pending, dpb_held, pkt_idx, dpb_recheck = self._new_decoder(codec), 0, params, 0, 0, False
                self._ask_idr()
                continue
            if not frames:  # (a decoder that holds a frame back in DPB, or drops pre-IDR P-frames)
                if not decoded:
                    dpb_held += 1
                continue
            last_pts = frames[-1].pts
            dpb_lag = max(0, in_idx - int(last_pts)) if last_pts is not None else dpb_held
            if not decoded:
                self._check_dpb(dec, dpb_lag, initial=True, pre_idr=dpb_held if dpb_lag == 0 else 0)
            elif dpb_recheck:
                dpb_recheck = False
                self._check_dpb(dec, dpb_lag, initial=False)
            elif dpb_lag > (self.decoder_dpb_delay or 0):
                self._check_dpb(dec, dpb_lag, initial=False)
            decoded = True
            size_new = sizes.update(len(p.data), key)
            damaged = bool(p.frames_lost > 0 or p.recovered or p.truncated or damaged_left > 0)
            for img in images:
                decode_ms = (time.perf_counter() - t) * 1000
                # Ready = arrival + decode time: what the pipeline's "behind" check has always measured from.
                tick = Tick(img, p.t_arrival, 0, damaged=damaged,
                            t_ready=p.t_arrival + decode_ms / 1000,
                            extra={"seq": p.seq, "size": len(p.data), "frames_lost": p.frames_lost,
                                   "recovered": p.recovered, "keyframe": key, "decode_ms": decode_ms,
                                   "dpb_lag": dpb_lag,
                                   # the size check's call: for the stream check, test mode and the frame log
                                   "size_new": size_new, "size_ref": sizes.ref, "big": sizes.big})
                if self.test is not None and (tick := self.test(tick)) is None:
                    continue
                if self.q.put(tick):
                    self.dropped += 1
            damaged_left = max(0, damaged_left - 1)
        _FUSED_NV12_KERNEL.flush()

    def _image(self, fr):
        if self.convert is not None:
            img = self.convert(fr)
            if not self.convert.said:
                self.convert.said = True
                print(f"{self.tag} HDR: the stream is {self.convert.describe()}")
            return img
        return cuda_frame_to_bgr(fr) if self.decoder_kind == "cuda" else fr.to_ndarray(format="bgr24")

    def set_hdr(self, out: str | None, display_peak: float | None = None, why: str = "") -> None:
        """SmoovPlay's HDR setting (framegen/hdr.py), before the frames start: out "sdr8" (tone-mapped to SDR),
        "pq16" (kept HDR10, for HDR output) or None (off: as before; why: the reason, if it's on but can't be used).
        Only an HDR stream is converted."""
        self._hdr_told = True
        if out is None:
            if self.hdr_stream and why:
                print(f"{self.tag} WARNING: this stream is HDR, but {why}, so its colours look washed out. "
                      + ("Have the app send it without HDR." if self.app else
                         "Pick a codec without HDR in chiaki-ng's Video settings."))
            elif self.hdr_stream:
                print(f"{self.tag} WARNING: this stream is HDR, and HDR is off in SmoovPlay's settings, so its colours "
                      "look washed out. Turn on HDR in SmoovPlay's Settings (Picture), or "
                      + ("have the app send it without HDR." if self.app else "pick a codec without HDR in chiaki-ng."))
            return
        if not self.hdr_stream:
            print(f"{self.tag} HDR is on in SmoovPlay's settings, but this stream isn't HDR: shown as usual."
                  + ("" if self.app else " (For HDR, pick chiaki-ng's H.265 HDR codec.)"))
            return
        from .hdr import StreamConverter
        import torch
        self.convert = StreamConverter(out, "cuda" if torch.cuda.is_available() else "cpu", display_peak=display_peak)
        print(f"{self.tag} HDR stream: " + ("kept HDR for the HDR display" if out == "pq16" else "shown in SDR")
              + " (HDR is Experimental)")

    def _ask_idr(self) -> None:
        now = time.perf_counter()
        if now - self._last_idr < 0.5:  # chiaki-ng re-asks by itself; don't flood the PS5
            return
        self._last_idr = now
        self.tap.request_idr()
        self.idr_requests += 1

    def _save_header(self, session: int, codec: int, params: bytes) -> None:
        try:
            self._header_file.write_bytes(struct.pack("<II", session, codec) + params)
        except OSError:
            pass

    def _load_header(self, session: int, codec: int) -> bytes:
        """The codec header an earlier run.py saved for this same stream, if any."""
        try:
            b = self._header_file.read_bytes()
        except OSError:
            return b""
        return b[8:] if len(b) > 8 and struct.unpack_from("<II", b) == (session, codec) else b""

    # The delay test and the sound delay go through chiaki-ng (frametap.h). Both need a chiaki-ng from this version
    # of the patch; can_press()/can_delay_audio() say whether the one streaming is.
    def can_press(self) -> bool:
        return bool(self.tap.features() & FEATURE_PRESS)

    def can_delay_audio(self) -> bool:
        return bool(self.tap.features() & FEATURE_AUDIO_DELAY)

    def press(self, buttons: int, hold_ms: int = 100) -> int:
        return self.tap.press(buttons, hold_ms)

    def press_sent(self, token: int) -> float | None:
        return self.tap.press_sent(token, self.tap.header().qpc_freq)

    def set_audio_delay(self, ms: float) -> None:
        self.tap.set_audio_delay(round(ms))

    # Instant replay's sound: chiaki-ng copies the game's sound into a ring of its own (a chiaki-ng from before that
    # leaves it empty, and replays are silent).
    def has_sound(self) -> bool:
        return bool(self.tap.features() & FEATURE_AUDIO_TAP) and self.sound.live()

    def read_sound(self) -> list:
        """The game's sound that arrived since the last call (framegen.io.Sound chunks); [] if there's none."""
        h = self.tap.header()
        if not h.live:
            return []
        if h.session != self._sound_session:  # a new stream: start from its current sound
            self._sound_session = h.session
            self.sound.reset()
        return self.sound.read_new()

    def hold(self, on: bool) -> None:
        """Stop decoding while the pipeline restarts (on), and start again from a fresh frame (off)."""
        if on:
            self._go.clear()
            return
        self.q.clear()  # frames from before the restart are stale by now
        self._go.set()

    def why(self) -> str:
        """What the ring says about a wait for frames: a hint under the window's "No picture" note ("" if none)."""
        now = time.perf_counter()
        h = self.tap.header()
        if h.write_seq != self._ring[0]:
            self._ring = (h.write_seq, now)
        if not h.live:
            return ("The app isn't streaming. Start its stream (or wait while it reconnects)" if self.app else
                    "chiaki-ng isn't streaming. Connect to the PS5 in chiaki-ng (or wait while it reconnects)")
        if now - self._beat > 3:
            return "SmoovPlay's video decoder has stopped. Press Q, then Play again"
        if now - self._ring[1] > 1:
            return ("The app" if self.app else "chiaki-ng") + " isn't sending any frames. Is its stream still running?"
        if self._no_header:
            return ("This stream started before SmoovPlay, so its start is missing.\n"
                    + ("Restart the stream in the app" if self.app else
                       "In chiaki-ng press Ctrl+O, click the x to end the stream, then connect again"))
        return ""

    def __iter__(self):
        if not self._hdr_told:  # nobody said what to do with HDR: as with the setting off
            self.set_hdr(None)
        self.dropped = 0  # frames dropped while RIFE was loading don't count
        self.q.dropped = 0
        self._go.set()  # the pipeline is ready: start decoding
        from .io import queue_ticks
        yield from queue_ticks(self.q, "the PS5", self.why)

    def log_tick(self, tick: Tick, diff, action, proc_ms: float, t_start: float) -> None:
        """Per-tick CSV row: stream facts next to the pipeline's decisions, for tuning after a real session."""
        self.check.add_tick(tick, diff)
        if self._log is None:
            return
        e = tick.extra
        hint = e.get("size_new")
        self._log[1].writerow([tick.index, e.get("seq"), f"{(tick.t_arrival - self._t0) * 1000:.2f}", e.get("size"),
                               e.get("frames_lost"), int(bool(e.get("recovered"))), int(bool(e.get("keyframe"))),
                               f"{e.get('decode_ms', 0):.2f}", f"{(t_start - tick.t_arrival) * 1000:.2f}",
                               "" if hint is None else int(hint), int(diff.is_new), f"{diff.max_diff:.2f}",
                               f"{diff.mean_diff:.3f}", action.value, f"{proc_ms:.2f}"])

    def close(self) -> None:
        self._stop = True
        self.q.close()
        if self._log is not None:
            self._log[0].close()
        try:
            self.sound.close()
        except (BufferError, ValueError):  # still being read: it goes with the process
            pass
        print(f"{self.tag} decoded frames dropped: {self.dropped}, stream frames skipped: {self.skipped}, "
              f"keyframe requests: {self.idr_requests}")
