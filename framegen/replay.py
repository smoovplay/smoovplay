"""Instant replay: the last 30 s to 5 min of SmoovPlay's picture, kept in memory and saved when asked.

Three tracks are kept while it's on (settings.replay), each as MPEG-TS in memory, a piece per encoded packet, so the
oldest can be dropped and any stretch cut out at a keyframe (one every second):
- after: every frame SmoovPlay shows, real and AI, timed when it was shown (so 90 fps really is 90), except at
  120 fps, where it's every other frame (60 fps: encoding all 120 a second fell behind, and took time from play);
- before: each new frame from the game, once, timed when it arrived plus SmoovPlay's delay, so it lines up with
  after: the game's own frames as they'd look without SmoovPlay;
- sound: the game's sound (PS5 Remote Play through chiaki-ng's frame tap, chiaki_tap/frametap.h, or a capture
  card's), timed the same way, so it meets both pictures.

Saving (R in the video window, or Ctrl+Alt+R from anywhere) copies a stretch into an MP4 in Videos\\SmoovPlay without
encoding it again: a second or two. A before & after video (B, Ctrl+Alt+B, or the "Saved clip" setting) is encoded
afresh by `python -m framegen.replay compose`, in a process of its own at low priority, so play isn't disturbed.

The encoding is done by the graphics card's video encoder (NVENC, else AMD's or Intel's, else x264 on the CPU), which
is separate from the parts that run the AI and draw the picture. Frames are turned into YUV on the graphics card, on
the stream the rest of the pipeline uses, copied to memory on a side stream, and encoded on a thread of their own;
the encoder lets go of Python's lock while it works, so the frame pacer keeps its timing.
"""
from __future__ import annotations

import argparse
import ctypes
import heapq
import os
import queue
import shutil
import subprocess
import sys
import tempfile
import threading
import time
from collections import deque
from fractions import Fraction
from pathlib import Path

import av
import numpy as np
from av.video.frame import PictureType

from .paths import data_dir

ROOT = Path(__file__).resolve().parents[1]
# A before & after video being made keeps a file here named after its process id, holding the video's path, so the
# launcher knows. It carries on after the video window and the launcher close; updating SmoovPlay would stop it.
SAVING_DIR = data_dir() / "saving"
TB = 90000  # MPEG-TS clock: every timestamp here is in 1/90000 s
OFFSET = 10 * TB  # timestamps start 10 s in, so none is ever negative
LENGTHS = (0, 30, 60, 120, 300)  # settings.replay's choices, in seconds (0: off)
KEEP_EXTRA_S = 2.0  # kept beyond the length, so a cut can start at the keyframe before it
KEY_EVERY_S = 1.0  # a keyframe at least this often: where a saved clip can start
# Bits per pixel per frame. An encoder's rate control spreads its bit rate over its nominal frame rate, so the bit
# rate is given for that and each frame gets the same share whatever the real rate: 0.14 is ~26 Mbit/s for 1080p
# at 90 fps (every frame shown); 0.2 is ~12 Mbit/s for a 30 fps game's own frames.
AFTER_BPP, BEFORE_BPP = 0.14, 0.2
AFTER_NOMINAL, BEFORE_NOMINAL = 120, 60
MAX_BITRATE = 80_000_000
SOUND_RATE, SOUND_BITRATE = 48000, 160_000
POOL = {"after": 10, "before": 6}  # frames on their way to the encoder, per track; more and they're dropped
TS_OVERHEAD = 1.04  # MPEG-TS packet headers
DIVIDER_BGR = (0, 176, 255)  # the line between before and after: amber

# H.264 encoders, best first, with low-delay options (no B-frames, so a packet comes out for each frame at once).
ENCODERS = {
    "h264_nvenc": {"preset": "p4", "rc": "vbr", "bf": "0", "forced-idr": "1", "zerolatency": "1", "delay": "0",
                   "spatial-aq": "1"},
    "h264_amf": {"usage": "lowlatency", "quality": "balanced", "rc": "vbr_peak", "bf": "0"},
    "h264_qsv": {"preset": "veryfast", "forced_idr": "1", "bf": "0", "async_depth": "1"},
    "libx264": {"preset": "superfast", "tune": "zerolatency", "forced-idr": "1", "crf": "21"},
}
if sys.platform == "darwin":
    # A Mac's media engine, first since none of the others are there. It hands back its first packets a few frames
    # late, which the ring doesn't mind, since each packet keeps its own time.
    ENCODERS = {"h264_videotoolbox": {"realtime": "1", "prio_speed": "1", "allow_sw": "0", "bf": "0"}, **ENCODERS}
_works: dict[str, bool] = {}


def memory_mb(length_s: float, w: int = 1920, h: int = 1080, hz: int = 90, game_fps: float = 30) -> float:
    """Roughly how much memory a replay of this length holds (the settings' hints quote it for 1080p, 90 fps)."""
    bits = (AFTER_BPP * hz + BEFORE_BPP * game_fps) * w * h + SOUND_BITRATE
    return bits * TS_OVERHEAD * (length_s + KEEP_EXTRA_S) / 8 / 1e6


def pick_encoder(prefer: str | None = None) -> str:
    """The first H.264 encoder that works on this PC (`prefer` only, if given)."""
    for name in ([prefer] if prefer else list(ENCODERS)):
        if name not in _works:
            _works[name] = _try_encoder(name)
        if _works[name]:
            return name
    raise RuntimeError(f"no H.264 encoder works here (tried {prefer or ', '.join(ENCODERS)})")


def _try_encoder(name: str) -> bool:
    try:
        ctx = av.CodecContext.create(name, "w")
        ctx.width, ctx.height, ctx.pix_fmt = 256, 144, "yuv420p"
        ctx.time_base, ctx.framerate = Fraction(1, TB), Fraction(60)
        ctx.options = dict(ENCODERS.get(name, {}))
        ctx.open()
        f = av.VideoFrame.from_ndarray(np.full((216, 256), 128, np.uint8), format="yuv420p")
        f.pts = 0
        got = list(ctx.encode(f)) + list(ctx.encode(None))
        return bool(got)
    except Exception:  # noqa: BLE001 - not on this PC (no such card or driver)
        return False


def _configure_video(stream, name: str, w: int, h: int, nominal: int, bits_per_frame: float, bt709: bool) -> None:
    cc = stream.codec_context
    cc.width, cc.height, cc.pix_fmt = w, h, "yuv420p"
    cc.time_base = stream.time_base = Fraction(1, TB)
    cc.framerate = Fraction(nominal)
    cc.gop_size = 2 * nominal  # keyframes are asked for every KEY_EVERY_S anyway
    opts = dict(ENCODERS.get(name, {}))
    if name != "libx264":  # x264 goes by quality (crf); the others by bit rate
        rate = int(min(MAX_BITRATE, bits_per_frame * nominal))
        cc.bit_rate = rate
        opts.update(maxrate=str(rate * 3 // 2), bufsize=str(rate))
    cc.options = opts
    # BT.709 from the graphics card; OpenCV's conversion (no CUDA) is BT.601's. Limited range either way.
    cc.colorspace, cc.color_primaries, cc.color_trc, cc.color_range = (1 if bt709 else 6), 1, 1, 1


# ---- colour ----------------------------------------------------------------------------------------------------
# BT.709, limited range, for BGR input: Y = 16 + 219 (kr R + kg G + kb B), Cb/Cr = 128 + 224 (...), with R..B in 0..1.
_KR, _KB = 0.2126, 0.0722
_KG = 1 - _KR - _KB
_M709 = np.array([[_KB, _KG, _KR],  # Y, from B G R
                  [0.5, -_KG / (1 - _KB) / 2, -_KR / (1 - _KB) / 2],  # Cb
                  [-_KB / (1 - _KR) / 2, -_KG / (1 - _KR) / 2, 0.5]],  # Cr
                 np.float32) * np.array([[219], [224], [224]], np.float32) / 255


def bgr_to_i420(bgr: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 BGR (even size) -> I420 (BT.709, limited range) as one (H*3/2, W) array: Y, then U, then V. The
    same sums as the graphics card's (Recorder._gpu_i420); used for the before & after labels and in tests."""
    h, w = bgr.shape[:2]
    x = bgr.astype(np.float32)
    y = x @ _M709[0] + 16
    s = x.reshape(h // 2, 2, w // 2, 2, 3).mean(axis=(1, 3))
    uv = s @ _M709[1:].T + 128
    planes = [y, uv[..., 0], uv[..., 1]]
    return np.concatenate([np.clip(np.rint(p), 0, 255).astype(np.uint8).ravel() for p in planes]).reshape(h * 3 // 2, w)


def _planes(flat: np.ndarray, h: int, w: int):
    """Views of the Y, U and V planes of an (H*3/2, W) I420 array."""
    r, n = flat.reshape(-1), h * w
    q = n // 4
    return r[:n].reshape(h, w), r[n:n + q].reshape(h // 2, w // 2), r[n + q:].reshape(h // 2, w // 2)


# ---- MPEG-TS in memory ----------------------------------------------------------------------------------------
class _Collect:
    """What PyAV writes MPEG-TS into: each packet's bytes, taken after it's written (flush_packets)."""

    def __init__(self) -> None:
        self.parts: list[bytes] = []

    def write(self, b) -> int:
        self.parts.append(bytes(b))
        return len(b)

    def take(self) -> bytes:
        data = b"".join(self.parts)
        self.parts.clear()
        return data


class _Chunks:
    """Reads a stretch (a TS head and pieces) as one file, without joining it into one big copy."""

    def __init__(self, head: bytes, pieces) -> None:
        self._parts = [head] + [p[2] for p in pieces]
        self._i = self._off = 0

    def read(self, n: int = -1) -> bytes:
        out, need = [], n if n is not None and n >= 0 else 1 << 62
        while need > 0 and self._i < len(self._parts):
            p = self._parts[self._i]
            take = p[self._off:self._off + need]
            out.append(take)
            need -= len(take)
            self._off += len(take)
            if self._off >= len(p):
                self._i, self._off = self._i + 1, 0
        return b"".join(out)


def _si_packets(data: bytes) -> bytes:
    """The PAT, PMT and SDT packets in a TS piece: put before any stretch so it can be read on its own."""
    keep = []
    for i in range(0, len(data) - 187, 188):
        pid = ((data[i + 1] & 0x1F) << 8) | data[i + 2]
        if pid in (0x0000, 0x0011, 0x1000):  # PAT, SDT, and the PMT where FFmpeg puts it
            keep.append(data[i:i + 188])
    return b"".join(keep)


class _Track:
    """One stream of the replay: encoded packets as MPEG-TS pieces (pts, keyframe, bytes), oldest first."""

    def __init__(self, name: str) -> None:
        self.name = name
        self.ring: deque = deque()
        self.keys: deque = deque()  # pts of the keyframes in ring
        self.head = b""
        self.nbytes = 0
        self.container = self.stream = None
        self._io: _Collect | None = None
        self.fmt = None  # video: (w, h); sound: (rate, channels)
        self.last_pts: int | None = None
        self.next_key: int | None = None

    @property
    def open(self) -> bool:
        return self.container is not None

    def start(self, make_stream, fmt) -> None:
        self.close()
        self.clear()
        self._io = _Collect()
        self.container = av.open(self._io, "w", format="mpegts",
                                 container_options={"flush_packets": "1", "pes_payload_size": "0"})
        self.stream = make_stream(self.container)
        self.fmt = fmt

    def clear(self) -> None:
        self.ring.clear()
        self.keys.clear()
        self.head, self.nbytes, self.last_pts, self.next_key = b"", 0, None, None

    def add(self, packets) -> None:
        for p in packets:
            if p.pts is None:
                continue
            pts, key = round(p.pts * p.time_base * TB), bool(p.is_keyframe)  # every track on the 90 kHz clock
            self.container.mux(p)
            data = self._io.take()
            if not self.head:
                self.head = _si_packets(data)
            self.ring.append((pts, key, data))
            self.nbytes += len(data)
            if key or self.name == "sound":
                self.keys.append(pts)

    def newest(self) -> int | None:
        return self.ring[-1][0] if self.ring else None

    def trim(self, cut: int) -> None:
        """Drop what's older than `cut`, keeping the keyframe at or before it."""
        while len(self.keys) >= 2 and self.keys[1] <= cut:
            k = self.keys.popleft()
            k = self.keys[0]
            while self.ring and self.ring[0][0] < k:
                self.nbytes -= len(self.ring.popleft()[2])

    def close(self) -> None:
        if self.container is not None:
            try:
                self.container.close()
            except Exception:  # noqa: BLE001 - the trailer of a stream nobody reads
                pass
        self.container = self.stream = None


def _cut(ring: list, start: int, end: int, at_key: bool = True) -> list:
    """The pieces of a stretch: from the last keyframe at or before `start` (or the first piece at or after it), up
    to `end`."""
    i = 0
    for j, (pts, key, _) in enumerate(ring):
        if pts > start:
            break
        if key or not at_key:
            i = j
    if not at_key:
        while i < len(ring) and ring[i][0] < start:
            i += 1
    return [p for p in ring[i:] if p[0] <= end]


# ---- saving ---------------------------------------------------------------------------------------------------
def videos_folder() -> Path:
    """Where clips go: SMOOVPLAY_REPLAY_DIR if set, else Videos\\SmoovPlay (wherever Windows keeps Videos), or on a
    Mac Movies/SmoovPlay."""
    if os.environ.get("SMOOVPLAY_REPLAY_DIR"):
        return Path(os.environ["SMOOVPLAY_REPLAY_DIR"])
    base = Path.home() / ("Movies" if sys.platform == "darwin" else "Videos")
    if sys.platform == "win32":
        try:
            import uuid
            guid = (ctypes.c_ubyte * 16).from_buffer_copy(uuid.UUID("18989B1D-99B5-455B-841C-AB7C74E4DDFC").bytes_le)
            ptr = ctypes.c_wchar_p()
            if ctypes.windll.shell32.SHGetKnownFolderPath(ctypes.byref(guid), 0, None, ctypes.byref(ptr)) == 0:
                base = Path(ptr.value)
                ctypes.windll.ole32.CoTaskMemFree(ptr)
        except Exception:  # noqa: BLE001 - fall back to ~/Videos
            pass
    return base / "SmoovPlay"


def _unique(path: Path) -> Path:
    """`path`, or "name (2).mp4", "name (3).mp4" and so on if it's taken. Its ".part" file is made here, exclusively,
    so two saves started in the same second (R pressed twice) can't write the same file; the writer fills that .part
    and moves it over `path`. Checking that `path` exists alone wasn't enough, because the writing goes to the
    .part, and both saves saw a free name."""
    n = 2
    out = path
    while True:
        if not out.exists():
            part = out.with_name(out.name + ".part")
            try:
                with open(part, "x"):
                    pass
                if not out.exists():
                    return out
                part.unlink(missing_ok=True)  # finished by another save between the two checks
            except FileExistsError:
                pass  # another save is writing it
        out = path.with_name(f"{path.stem} ({n}){path.suffix}")
        n += 1


def _dur(secs: float) -> str:
    if secs < 90:
        return f"{round(secs)} s"
    m = f"{secs / 60:.1f}".rstrip("0").rstrip(".")
    return f"{m} min"


def _short(path: Path) -> str:
    return f"{path.parent.parent.name}\\{path.parent.name}\\{path.name}"


def _snap_fps(fps: float) -> int:
    for f in (24, 30, 40, 45, 50, 60, 90, 120):
        if abs(fps - f) <= 0.05 * f:
            return f
    return round(fps)


def to_mp4(path: Path, video: tuple, sound: tuple | None, base: int, end: int) -> None:
    """Remux a video stretch (head, pieces) and its sound into an MP4, timestamps from `base`, without re-encoding.
    If it fails, its .part goes too, even the empty one _unique made before anything was written."""
    tmp = path.with_name(path.name + ".part")
    try:
        _remux(tmp, video, sound, base, end)
    except BaseException:
        tmp.unlink(missing_ok=True)
        raise
    os.replace(tmp, path)


def _remux(tmp: Path, video: tuple, sound: tuple | None, base: int, end: int) -> None:
    vin = av.open(_Chunks(*video), "r", format="mpegts")
    sin = av.open(_Chunks(*sound), "r", format="mpegts") if sound is not None and sound[1] else None
    out = av.open(str(tmp), "w", format="mp4", container_options={"movflags": "+faststart"})
    try:
        pairs = [(vin, vin.streams.video[0])]
        if sin is not None and sin.streams.audio:
            pairs.append((sin, sin.streams.audio[0]))
        outs = [out.add_stream_from_template(s) for _, s in pairs]

        def packets(k: int):
            inp, s_in = pairs[k]
            n = 0
            for p in inp.demux(s_in):
                if p.pts is None or p.dts is None or not base <= p.pts <= end:
                    continue
                p.pts -= base
                p.dts -= base
                p.stream = outs[k]
                n += 1
                yield float(p.dts * p.time_base), k, n, p

        for *_, p in heapq.merge(*(packets(k) for k in range(len(pairs))), key=lambda x: x[:3]):
            out.mux(p)
    finally:
        out.close()
        vin.close()
        if sin is not None:
            sin.close()


def _write_ts(path: Path, head: bytes, pieces) -> None:
    with open(path, "wb") as f:
        f.write(head)
        for p in pieces:
            f.write(p[2])


# ---- the recorder ---------------------------------------------------------------------------------------------
_STOP = object()


class Recorder:
    """Instant replay. The pipeline calls frame() for each frame it shows, real_frame() for each new frame from the
    game, and set_delay() with its delay; attach_sound() gives it the game's sound. save() writes a clip.

    length_s: how much to keep (0: off; set_length changes it). clip: what save() writes by default ("smoovplay",
    "compare" or "both"); layout: a before & after video's ("split" or "side"). device: "cuda" (YUV made on the
    graphics card), "cpu" (OpenCV), or "auto". notify(text, seconds): messages for the user (the window's toasts).
    hotkeys: Ctrl+Alt+R and Ctrl+Alt+B from anywhere while it's on. size: the stream's (w, h), so warm_up() can
    open the encoders before play. drop_when_behind=False (tests): wait for the encoder instead of dropping a frame.

    Opening a video encoder holds Python's lock for 0.1-0.2 s (NVENC). warm_up() does it before play starts; turned
    on during play, the encoder's thread does it, and play pauses for that long once.
    """

    def __init__(self, length_s: int = 0, clip: str = "smoovplay", layout: str = "split", out_dir: Path | None = None,
                 encoder: str | None = None, device: str = "auto", notify=None, hotkeys: bool = False,
                 drop_when_behind: bool = True, size: tuple | None = None, mark_dir: Path | None = None) -> None:
        self.clip, self.layout = clip, layout
        self.mark_dir = mark_dir  # SAVING_DIR when SmoovPlay runs it: before & after videos say they're being made
        self.size = (size[0] & ~1, size[1] & ~1) if size else None
        self._prep_lock = threading.Lock()
        self.out_dir = Path(out_dir) if out_dir is not None else None
        self._prefer = encoder
        self.encoder = ""
        self._notify_fn = notify
        self._hotkeys_wanted = hotkeys
        self._hotkeys: Hotkeys | None = None
        if device == "auto":
            torch = sys.modules.get("torch")
            device = "cuda" if torch is not None and torch.cuda.is_available() else "cpu"
        self.device = device
        self._torch = None
        self._ready = False  # the graphics card's side and the encoder are set up (on the encoder's thread: play
        self._building: set = set()  # never waits for them, a few frames are skipped instead)
        self._block = not drop_when_behind
        self._t0: float | None = None  # the clock's zero: set by the first frame or sound, a little before it
        self._delay_s = 0.04  # until the pipeline says
        self.length = 0
        self.dropped = 0
        self.after_gap_s = 0.0  # the after track keeps a frame only this long after the last one (set_output_hz)
        self._after_last: float | None = None
        self._lock = threading.Lock()
        self._q: queue.Queue = queue.Queue(maxsize=sum(POOL.values()) + 64)
        self.tracks = {n: _Track(n) for n in ("after", "before", "sound")}
        self._pools: dict = {}
        self._sound_fn = None
        # Settings "Looks in clips": fn(frame) -> the frame with the look drawn on (the video window's baked()), for
        # the SmoovPlay video only; the before frames stay as the game sent them. None: clips as the game looked.
        self.look = None
        # HDR output (framegen/hdr.py): frames come as float16 HDR10, and this turns them into the 8-bit SDR frames
        # clips are made of (hdr.pq16_to_sdr8), the same way "shown in SDR" would show them. None: frames are SDR.
        self.hdr_to_sdr = None
        self._resampler = None
        self._a_next: float | None = None  # the time (perf_counter) the next sound sample written is heard
        self._a_pts = 0
        self._skip_s = 0.0
        self._drifts: deque = deque()
        self._warned: set = set()
        self._cv = threading.Condition()
        self._quick = 0  # saves still writing their clip (a before & after video then goes to a process of its own)
        self._making: list[subprocess.Popen] = []  # before & after videos' processes
        self._worker = threading.Thread(target=self._work, name="replay", daemon=True)
        self._worker.start()
        self.set_length(length_s)

    # ---- settings
    @property
    def on(self) -> bool:
        return self.length > 0

    def set_length(self, seconds: int) -> None:
        was = self.length
        self.length = int(seconds)
        if was and not self.length:
            self._q.put(("reset",))
            print("[replay] off")
        elif self.length and not was:
            self._q.put(("prepare",))
            print(f"[replay] on: keeping the last {_dur(self.length)}")
            self._notify(f"Instant replay on: R saves the last {_dur(self.length)}, B a before & after "
                         "(Ctrl+Alt+R and Ctrl+Alt+B from anywhere)", 6.0)
        elif self.length != was:
            print(f"[replay] keeping the last {_dur(self.length)}")
        if self._hotkeys_wanted:
            if self.length and self._hotkeys is None:
                self._hotkeys = Hotkeys({"ctrl+alt+r": lambda: self.save(), "ctrl+alt+b": lambda: self.save("compare")})
                self._hotkeys.start()
            elif not self.length and self._hotkeys is not None:
                self._hotkeys.stop()
                self._hotkeys = None

    def set_delay(self, ms: float) -> None:
        """SmoovPlay's delay (a real frame's arrival to the screen): the before frames and the sound move by it."""
        self._delay_s = max(0.0, ms / 1000)

    def attach_sound(self, fn) -> None:
        """fn() -> new framegen.io.Sound chunks since the last call (ChiakiSource/CardSource.read_sound); polled."""
        self._sound_fn = fn

    def add_sound(self, chunks) -> None:
        """Sound chunks (framegen.io.Sound), for a source that pushes them."""
        if self.length and chunks:
            self._zero(chunks[0].t)
            self._q.put(("s", list(chunks)))

    # ---- frames
    def set_output_hz(self, hz: int) -> None:
        """The rate frames are shown at: from 120 a second, every other one is kept (60 fps clips)."""
        self.after_gap_s = 0.75 / 60 if hz >= 120 else 0.0  # 1.5 frames at 120: alternate ones, whatever the jitter

    def frame(self, frame, t: float | None = None) -> None:
        """A frame just shown (HxWx3 uint8 BGR, numpy or CUDA), at t (perf_counter; default now)."""
        if self.length:
            t = time.perf_counter() if t is None else t
            if self.after_gap_s and self._after_last is not None and t - self._after_last < self.after_gap_s:
                return
            self._after_last = t
            if self.look is not None:
                try:
                    frame = self.look(frame)
                except Exception as e:  # noqa: BLE001 - then without the look
                    self._warn("look", f"[replay] can't put the look in clips ({str(e)[:200]})")
            self._put("after", frame, t)

    def real_frame(self, frame, t_arrival: float) -> None:
        """A new frame from the game, that arrived at t_arrival (perf_counter)."""
        if self.length:
            self._put("before", frame, t_arrival + self._delay_s)

    def _zero(self, t: float) -> None:
        if self._t0 is None:
            self._t0 = t - 5.0  # (OFFSET leaves 10 s more before any timestamp would go below zero)

    def _put(self, name: str, frame, t: float) -> None:
        self._zero(t)
        try:
            if self.hdr_to_sdr is not None and getattr(frame, "is_floating_point", lambda: False)():
                frame = self.hdr_to_sdr(frame)
            item = self._convert(name, frame)
        except Exception as e:  # noqa: BLE001 - a replay must never take the stream down
            self._warn("convert", f"[replay] can't take frames ({str(e)[:200]})")
            return
        if item is None:
            self.dropped += 1
            return
        try:
            self._q.put(("v", name, t) + item, block=self._block, timeout=10 if self._block else None)
        except queue.Full:
            if item[3] is not None:
                item[3]()
            self.dropped += 1

    def _convert(self, name: str, frame):
        """-> (w, h, payload, release): payload an (H*3/2, W) I420 array, or (array, CUDA event, GPU tensor) still
        being copied; release() hands its buffer back. None: no buffer free (the encoder is behind)."""
        is_tensor = hasattr(frame, "is_cuda")
        if not self._ready and not (self._block and self._wait(lambda: self._ready)):
            return None
        if self.device == "cuda":
            torch = self._torch
            x = frame if is_tensor else torch.from_numpy(np.ascontiguousarray(frame))
            if not x.is_cuda:
                x = x.to("cuda", non_blocking=True)
            return self._gpu_i420(name, x)
        import cv2
        a = frame.cpu().numpy() if is_tensor else frame
        h, w = a.shape[0] & ~1, a.shape[1] & ~1
        return w, h, cv2.cvtColor(np.ascontiguousarray(a[:h, :w, :3]), cv2.COLOR_BGR2YUV_I420), None

    @staticmethod
    def _wait(done, timeout: float = 10.0) -> bool:
        t_end = time.perf_counter() + timeout
        while not done() and time.perf_counter() < t_end:
            time.sleep(0.002)
        return done()

    def warm_up(self) -> None:
        """Open the encoders and make the buffers now (before play), if it's on."""
        if self.length:
            t0 = time.perf_counter()
            self._prepare()
            print(f"[replay] ready in {time.perf_counter() - t0:.2f} s")

    def _prepare(self) -> None:
        with self._prep_lock:
            if self.device == "cuda" and self._torch is None:
                import torch
                self._side = torch.cuda.Stream()
                self._torch = torch
                self._yuv(torch.zeros((4, 4, 3), dtype=torch.uint8, device="cuda"))  # load its kernels now
            if self.size is not None:
                w, h = self.size
                for name in ("after", "before"):
                    if not self.tracks[name].open or self.tracks[name].fmt != (w, h):
                        self._open_track(name, w, h)
                    pool = self._pools.get(name)
                    if self.device == "cuda" and (pool is None or pool["size"] != (w, h)):
                        self._make_pool(name, w, h)
            self._ready = True

    def _open_track(self, name: str, w: int, h: int) -> None:
        """Start a video track with the encoder that works (the first time: the best of ENCODERS that opens)."""
        tr = self.tracks[name]
        tried = []
        for enc in [self.encoder] if self.encoder else [self._prefer] if self._prefer else list(ENCODERS):
            try:
                with self._lock:
                    tr.start(lambda c: self._video_stream(c, enc, w, h, name == "before"), (w, h))
                tr.stream.codec_context.open()  # now, rather than at the first frame
            except Exception as e:  # noqa: BLE001 - not on this PC: the next one
                tried.append(f"{enc}: {str(e)[:80]}")
                tr.close()
                continue
            if not self.encoder:
                self.encoder = enc
                print(f"[replay] encoding with {enc}")
            return
        raise RuntimeError("no H.264 encoder works here (" + "; ".join(tried) + ")")

    def _yuv(self, x):
        """(H, W, 3) uint8 BGR on the graphics card -> flat I420, BT.709 limited range (bgr_to_i420's sums, as plain
        per-pixel arithmetic: no matrix library to start up)."""
        torch = self._torch
        h, w = x.shape[:2]
        f = x.float()
        s = f.view(h // 2, 2, w // 2, 2, 3).mean(dim=(1, 3))
        out = torch.empty(h * w * 3 // 2, dtype=torch.uint8, device=x.device)
        hw, q = h * w, h * w // 4
        for row, add, src, a, b in ((_M709[0], 16.0, f, 0, hw), (_M709[1], 128.0, s, hw, hw + q),
                                    (_M709[2], 128.0, s, hw + q, None)):
            c = (src[..., 0] * float(row[0])).add_(src[..., 1], alpha=float(row[1])).add_(src[..., 2], alpha=float(row[2]))
            out[a:b].copy_(c.add_(add).round_().clamp_(0, 255).reshape(-1))
        return out

    def _make_pool(self, name: str, w: int, h: int) -> None:
        """Pinned buffers for frames on their way from the graphics card to the encoder."""
        torch = self._torch
        bufs = [torch.empty(h * w * 3 // 2, dtype=torch.uint8, pin_memory=True) for _ in range(POOL[name])]
        self._pools[name] = {"size": (w, h), "bufs": bufs, "free": list(range(len(bufs)))}
        self._building.discard(name)

    def _gpu_i420(self, name: str, x):
        torch = self._torch
        h, w = x.shape[0] & ~1, x.shape[1] & ~1
        pool = self._pools.get(name)
        if pool is None or pool["size"] != (w, h):  # made on the encoder's thread; skip frames till then
            if name not in self._building:
                self._building.add(name)
                self._q.put(("pool", name, w, h))
            if not (self._block and self._wait(lambda: name not in self._building)):
                return None
            pool = self._pools[name]
        if not pool["free"]:
            return None
        k = pool["free"].pop()
        out = self._yuv(x[:h, :w, :3])
        ready = torch.cuda.Event()
        ready.record()
        with torch.cuda.stream(self._side):  # the copy to memory doesn't hold up the pipeline's next frame
            self._side.wait_event(ready)
            pool["bufs"][k].copy_(out, non_blocking=True)
            done = torch.cuda.Event()
            done.record(self._side)
        out.record_stream(self._side)

        def release(pool=pool, k=k):
            pool["free"].append(k)
        return w, h, (pool["bufs"][k].numpy(), done, out), release

    # ---- the encoder's thread
    def _work(self) -> None:
        next_poll = 0.0
        while True:
            try:
                item = self._q.get(timeout=0.02)
            except queue.Empty:
                item = None
            try:
                if item is _STOP:
                    break
                if item is not None:
                    kind = item[0]
                    if kind == "v":
                        self._encode_video(*item[1:])
                    elif kind == "s":
                        for c in item[1]:
                            self._encode_sound(c)
                    elif kind == "reset":
                        self._reset()
                    elif kind == "prepare":
                        self._prepare()
                    elif kind == "pool":
                        self._make_pool(*item[1:])
                now = time.perf_counter()
                if self._sound_fn is not None and self.length and now >= next_poll:
                    next_poll = now + 0.02
                    try:
                        chunks = self._sound_fn()
                    except Exception:  # noqa: BLE001 - the source closing
                        chunks = []
                    if chunks:
                        self._zero(chunks[0].t)
                    for c in chunks:
                        self._encode_sound(c)
            except Exception as e:  # noqa: BLE001 - one bad frame or chunk mustn't stop the replay
                self._warn(f"work{type(e).__name__}", f"[replay] {type(e).__name__}: {str(e)[:200]}")
            finally:
                if item is not None:
                    self._q.task_done()
        self._reset()

    def _reset(self) -> None:
        with self._lock:
            for tr in self.tracks.values():
                tr.close()
                tr.clear()
            self._a_next, self._resampler = None, None
            self._pools.clear()
            self._building.clear()

    def _video_stream(self, container, enc: str, w: int, h: int, before: bool):
        nominal, bpp = (BEFORE_NOMINAL, BEFORE_BPP) if before else (AFTER_NOMINAL, AFTER_BPP)
        s = container.add_stream(enc, rate=nominal)
        _configure_video(s, enc, w, h, nominal, bpp * w * h, self.device == "cuda")
        return s

    def _encode_video(self, name: str, t: float, w: int, h: int, payload, release) -> None:
        if isinstance(payload, tuple):
            arr, done, _gpu = payload
            done.synchronize()
        else:
            arr = payload
        try:
            if not self.length:
                return
            frame = av.VideoFrame.from_ndarray(arr.reshape(h * 3 // 2, w), format="yuv420p")
        finally:
            if release is not None:
                release()
        tr = self.tracks[name]
        if not tr.open or tr.fmt != (w, h):
            self._open_track(name, w, h)
        pts = max(0, OFFSET + round((t - self._t0) * TB))
        if tr.last_pts is not None and pts <= tr.last_pts:
            pts = tr.last_pts + 1
        tr.last_pts = pts
        frame.pts, frame.time_base = pts, Fraction(1, TB)
        if tr.next_key is None or pts >= tr.next_key:
            frame.pict_type = PictureType.I
            tr.next_key = pts + round(KEY_EVERY_S * TB)
        packets = tr.stream.encode(frame)
        with self._lock:
            tr.add(packets)
            self._trim()

    def _sound_stream(self, container):
        s = container.add_stream("aac", rate=SOUND_RATE)
        s.codec_context.layout = "stereo"
        s.codec_context.bit_rate = SOUND_BITRATE
        return s

    def _encode_sound(self, s) -> None:
        if not self.length or not s.pcm or s.channels < 1:
            return
        tr = self.tracks["sound"]
        if not tr.open or tr.fmt != (s.rate, s.channels):
            with self._lock:
                tr.start(self._sound_stream, (s.rate, s.channels))
            self._resampler = av.AudioResampler(format="fltp", layout="stereo", rate=SOUND_RATE, frame_size=1024)
            self._a_next = None
        bpf = 2 * s.channels
        pcm = s.pcm[:len(s.pcm) // bpf * bpf]
        if s.channels > 2:  # surround: its first two channels (front left and right)
            pcm = np.frombuffer(pcm, np.int16).reshape(-1, s.channels)[:, :2].tobytes()
            bpf = 4
        n = len(pcm) // bpf
        t_due = s.t + self._delay_s  # when the picture it goes with is shown
        if self._a_next is None:
            self._a_next, self._skip_s = t_due, 0.0
            self._zero(s.t)
            self._a_pts = max(0, OFFSET * SOUND_RATE // TB + round((t_due - self._t0) * SOUND_RATE))
            self._drifts.clear()
        # The sound plays on, sample after sample; arrival times jitter. Only a gap (silence put in) or getting ahead
        # (sound skipped) that's large, or that's lasted a second, moves it, so jitter never makes a click.
        drift = t_due - self._a_next
        self._drifts.append((s.t, drift))
        while self._drifts and s.t - self._drifts[0][0] > 1.0:
            self._drifts.popleft()
        steady = len(self._drifts) > 1 and s.t - self._drifts[0][0] >= 0.9
        if drift > 0.25 or (steady and min(d for _, d in self._drifts) > 0.06):
            if drift > 30:  # a long pause: start the sound's timeline again (the clip will have a gap)
                self._a_next = t_due
                self._a_pts += round(drift * SOUND_RATE)
            else:
                self._write_sound(bytes(round(drift * s.rate) * bpf), s.rate, min(2, s.channels))
                self._a_next += drift
            self._drifts.clear()
            self._skip_s = 0.0
        elif drift < -0.25 or (steady and max(d for _, d in self._drifts) < -0.06):
            self._skip_s = -drift
            self._drifts.clear()
        if self._skip_s > 0:
            drop = min(n, round(self._skip_s * s.rate))
            self._skip_s = max(0.0, self._skip_s - drop / s.rate)
            pcm, n = pcm[drop * bpf:], n - drop
        if n:
            self._write_sound(pcm, s.rate, min(2, s.channels))
            self._a_next += n / s.rate

    def _write_sound(self, pcm: bytes, rate: int, channels: int) -> None:
        if not pcm:
            return
        layout = "mono" if channels == 1 else "stereo"
        f = av.AudioFrame.from_ndarray(np.frombuffer(pcm, np.int16).reshape(1, -1), format="s16", layout=layout)
        f.sample_rate = rate
        tr = self.tracks["sound"]
        packets = []
        for o in self._resampler.resample(f):
            o.pts, o.time_base = self._a_pts, Fraction(1, SOUND_RATE)
            self._a_pts += o.samples
            packets += tr.stream.encode(o)
        with self._lock:
            tr.add(packets)

    def _trim(self) -> None:
        newest = [tr.newest() for tr in self.tracks.values() if tr.ring]
        if newest:
            cut = max(newest) - round((self.length + KEEP_EXTRA_S) * TB)
            for tr in self.tracks.values():
                tr.trim(cut)

    def flush(self, timeout: float = 5.0) -> bool:
        """Wait until every frame and chunk given so far is encoded."""
        t_end = time.perf_counter() + timeout
        while self._q.unfinished_tasks and time.perf_counter() < t_end:
            time.sleep(0.005)
        return not self._q.unfinished_tasks

    def held(self) -> dict:
        """What's held: seconds and megabytes per track."""
        with self._lock:
            return {n: ((tr.ring[-1][0] - tr.ring[0][0]) / TB if len(tr.ring) > 1 else 0.0, tr.nbytes / 1e6)
                    for n, tr in self.tracks.items()}

    # ---- saving
    def save(self, kind: str | None = None, layout: str | None = None, wait: bool = False) -> list:
        """Save the replay: "smoovplay" (what was shown), "compare" (before & after) or "both"; default self.clip.
        In the background unless wait=True, which returns the files written."""
        kind, layout = kind or self.clip, layout or self.layout
        if not self.length:
            self._notify("Instant replay is off: turn it on in Settings > Instant replay", 5.0)
            return []
        if wait:
            return self._save(kind, layout)
        # A daemon: close() waits for the clip itself, and a before & after video carries on in its own process.
        t = threading.Thread(target=self._save, args=(kind, layout, True), name="replay-save", daemon=True)
        with self._cv:
            self._quick += 1
        t.start()
        return []

    def _save(self, kind: str, layout: str, counted: bool = False) -> list:
        handed = threading.Event()

        def handoff() -> None:  # the clip is written: close() needn't wait for this save any more
            if counted and not handed.is_set():
                handed.set()
                with self._cv:
                    self._quick -= 1
                    self._cv.notify_all()
        try:
            return self._save_inner(kind, layout, handoff)
        except Exception as e:  # noqa: BLE001 - e.g. a full disk
            print(f"[replay] save failed: {type(e).__name__}: {e}")
            self._notify(f"Couldn't save the replay: {str(e)[:120]}", 8.0)
            return []
        finally:
            handoff()

    def _save_inner(self, kind: str, layout: str, handoff=lambda: None) -> list:
        self.flush(1.0)
        with self._lock:
            snap = {n: (tr.head, list(tr.ring)) for n, tr in self.tracks.items()}
            length = self.length
        head, ring = snap["after"]
        if len(ring) < 2:
            self._notify("Instant replay has nothing to save yet", 4.0)
            return []
        end = ring[-1][0]
        after = _cut(ring, end - length * TB, end)
        base = after[0][0]
        secs = (end - base) / TB
        shead, sring = snap["sound"]
        sound = (shead, _cut(sring, base, end, at_key=False)) if sring else None
        folder = self.out_dir or videos_folder()
        folder.mkdir(parents=True, exist_ok=True)
        stamp = time.strftime("%Y-%m-%d %H-%M-%S")
        saved = []
        if kind in ("smoovplay", "both"):
            path = _unique(folder / f"SmoovPlay {stamp}.mp4")
            t0 = time.perf_counter()
            to_mp4(path, (head, after), sound, base, end)
            print(f"[replay] saved {secs:.1f} s -> {path} ({path.stat().st_size / 1e6:.0f} MB, "
                  f"{time.perf_counter() - t0:.1f} s)")
            self._notify(f"Saved the last {_dur(secs)}: {_short(path)}", 6.0)
            saved.append(path)
        if kind in ("compare", "both"):
            bhead, bring = snap["before"]
            before = _cut(bring, base, end)
            if len(before) < 2:
                self._notify("No game frames to compare yet", 4.0)
                return saved
            span_b = max(1e-6, (before[-1][0] - max(base, before[0][0])) / TB)
            n_b = sum(1 for p in before if p[0] >= base)
            labels = (f"Original {_snap_fps(max(1, n_b - 1) / span_b)} fps",
                      f"SmoovPlay {_snap_fps((len(after) - 1) / max(1e-6, secs))} fps")
            path = _unique(folder / f"SmoovPlay {stamp} before and after.mp4")
            tmp = Path(tempfile.mkdtemp(prefix="smoovplay-replay-"))
            _write_ts(tmp / "after.ts", head, after)
            _write_ts(tmp / "before.ts", bhead, before)
            cmd = [sys.executable, "-m", "framegen.replay", "compose", "--after", str(tmp / "after.ts"),
                   "--before", str(tmp / "before.ts"), "--out", str(path), "--layout", layout,
                   "--labels", *labels, "--encoder", self.encoder, "--cleanup", str(tmp)]
            if sound is not None and sound[1]:
                _write_ts(tmp / "sound.ts", *sound)
                cmd += ["--sound", str(tmp / "sound.ts")]
            if self.mark_dir is not None:
                cmd += ["--mark", str(self.mark_dir)]
            self._notify(f"Making a before & after video of the last {_dur(secs)} in the background. "
                         "Keep playing.", 6.0)
            flags = 0
            if sys.platform == "win32":
                flags = subprocess.BELOW_NORMAL_PRIORITY_CLASS | subprocess.CREATE_NO_WINDOW
            t0 = time.perf_counter()
            # A process of its own, which carries on if play stops or SmoovPlay closes: only its result waits here.
            proc = subprocess.Popen(cmd, cwd=ROOT, creationflags=flags, stdin=subprocess.DEVNULL,
                                    stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, encoding="utf-8",
                                    errors="replace")
            if sys.platform != "win32":  # the same below-normal priority, as nice 10
                try:
                    os.setpriority(os.PRIO_PROCESS, proc.pid, 10)
                except OSError:
                    pass
            with self._lock:
                self._making = [p for p in self._making if p.poll() is None] + [proc]
            print(f"[replay] before & after of {secs:.1f} s: being made in the background (process {proc.pid})")
            handoff()
            out, _ = proc.communicate()
            if self.mark_dir is not None:  # reported here, so the launcher needn't
                (Path(self.mark_dir) / f"{proc.pid}.txt").unlink(missing_ok=True)
            if proc.returncode == 0 and path.exists():
                print(f"[replay] saved before & after -> {path} ({time.perf_counter() - t0:.0f} s)")
                self._notify(f"Saved before & after: {_short(path)}", 8.0)
                saved.append(path)
            else:
                print(f"[replay] before & after failed ({proc.returncode}): {(out or '')[-800:]}")
                self._notify("Couldn't make the before & after video (see the log)", 8.0)
                shutil.rmtree(tmp, ignore_errors=True)
        return saved

    # ---- the rest
    def _notify(self, text: str, seconds: float = 4.0) -> None:
        if self._notify_fn is not None:
            try:
                self._notify_fn(text, seconds)
            except Exception:  # noqa: BLE001 - a closed window
                pass

    def _warn(self, key: str, text: str) -> None:
        if key not in self._warned:
            self._warned.add(key)
            print(text)

    def close(self, wait_saves: float = 0.0) -> None:
        """Stops recording. A clip being saved is finished first (waiting up to wait_saves seconds); a before & after
        video being made carries on in its own process, and isn't waited for."""
        if self._hotkeys is not None:
            self._hotkeys.stop()
            self._hotkeys = None
        self.length = 0
        self._q.put(_STOP)
        self._worker.join(timeout=5)
        end = time.monotonic() + wait_saves
        with self._cv:
            while self._quick > 0 and time.monotonic() < end:
                self._cv.wait(max(0.0, end - time.monotonic()))
            unfinished = self._quick
        if unfinished:
            print(f"[replay] {unfinished} clip(s) weren't finished saving after {wait_saves:.0f} s")
        with self._lock:
            making = sum(1 for p in self._making if p.poll() is None)
        if making:
            print(f"[replay] {making} before & after video(s) still being made: they carry on in the background")
        if self.dropped:
            print(f"[replay] frames dropped (the encoder fell behind): {self.dropped}")


# ---- system-wide keys -----------------------------------------------------------------------------------------
class Hotkeys:
    """Keys that work whichever window is in front (RegisterHotKey), on a thread of their own: {"ctrl+alt+r": fn}."""

    MODS = {"alt": 0x1, "ctrl": 0x2, "shift": 0x4, "win": 0x8}
    MOD_NOREPEAT, WM_HOTKEY, WM_QUIT = 0x4000, 0x0312, 0x0012

    def __init__(self, keys: dict) -> None:
        self.keys = keys
        self._tid = None
        self._thread = threading.Thread(target=self._run, name="hotkeys", daemon=True)

    def start(self) -> None:
        if sys.platform == "win32":
            self._thread.start()

    def _run(self) -> None:
        from ctypes import wintypes
        user32 = ctypes.windll.user32
        self._tid = threading.get_native_id()
        msg = wintypes.MSG()
        user32.PeekMessageW(ctypes.byref(msg), None, 0, 0, 0)  # this thread's message queue, for stop()
        fns = {}
        for i, (combo, fn) in enumerate(self.keys.items(), 1):
            *mods, key = combo.lower().split("+")
            m = sum(self.MODS[x] for x in mods) | self.MOD_NOREPEAT
            if user32.RegisterHotKey(None, i, m, ord(key.upper())):
                fns[i] = fn
            else:
                print(f"[replay] {combo.title()} is taken by another program; use the video window's key")
        try:
            while user32.GetMessageW(ctypes.byref(msg), None, 0, 0) > 0:
                if msg.message == self.WM_HOTKEY and msg.wParam in fns:
                    try:
                        fns[msg.wParam]()
                    except Exception as e:  # noqa: BLE001
                        print(f"[replay] {e}")
        finally:
            for i in fns:
                user32.UnregisterHotKey(None, i)

    def stop(self) -> None:
        if self._tid is not None:
            ctypes.windll.user32.PostThreadMessageW(self._tid, self.WM_QUIT, 0, 0)
        if self._thread.is_alive():
            self._thread.join(timeout=2)


# ---- before & after -------------------------------------------------------------------------------------------
def _label(text: str, h_frame: int):
    """A label: (Y, U, V, alpha for Y, alpha for U and V), white text on a translucent dark box."""
    import cv2
    s = max(0.5, h_frame / 1080)
    font, size, thick = cv2.FONT_HERSHEY_SIMPLEX, 1.0 * s, max(1, round(2 * s))
    (tw, th), base = cv2.getTextSize(text, font, size, thick)
    pad = round(14 * s)
    W, H = (tw + 2 * pad + 1) & ~1, (th + base + 2 * pad + 1) & ~1
    img = np.zeros((H, W, 3), np.uint8)
    cv2.putText(img, text, (pad, pad + th), font, size, (255, 255, 255), thick, cv2.LINE_AA)
    a = np.maximum(0.55, img[..., 1].astype(np.float32) / 255)
    y, u, v = _planes(bgr_to_i420(img), H, W)
    return y.astype(np.float32), u.astype(np.float32), v.astype(np.float32), a, \
        a.reshape(H // 2, 2, W // 2, 2).mean(axis=(1, 3))


def _blend(planes, label, x: int, y: int) -> None:
    ly, lu, lv, ay, auv = label
    h, w = ly.shape
    x, y = x & ~1, y & ~1
    for p, l_, a, (px, py) in ((planes[0], ly, ay, (x, y)), (planes[1], lu, auv, (x // 2, y // 2)),
                              (planes[2], lv, auv, (x // 2, y // 2))):
        r = p[py:py + l_.shape[0], px:px + l_.shape[1]]
        r[:] = np.rint(a[:r.shape[0], :r.shape[1]] * l_[:r.shape[0], :r.shape[1]]
                       + (1 - a[:r.shape[0], :r.shape[1]]) * r).astype(np.uint8)


def _crop(planes, h: int, w: int) -> list:
    """Y, U, V planes cut to an even h x w picture."""
    return [planes[0][:h, :w], planes[1][:h // 2, :w // 2], planes[2][:h // 2, :w // 2]]


def _yuv(frame):
    if frame.format.name != "yuv420p":
        frame = frame.reformat(format="yuv420p")
    return frame.to_ndarray(), frame.height, frame.width


def compose(after: str, before: str, out_path: str, sound: str | None = None, layout: str = "split",
            labels: tuple = ("Original", "SmoovPlay"), encoder: str | None = None) -> float:
    """A before & after video: `before`'s frames (the game's own, held until the next) and `after`'s (every frame
    shown), at after's times, split down the middle or side by side, with labels; `sound` copied in. Returns its
    length in seconds."""
    import cv2
    out_path = Path(out_path)
    tmp = out_path.with_name(out_path.name + ".part")
    ain, bin_ = av.open(after), av.open(before)
    sin = av.open(sound) if sound else None
    out = None
    try:
        a_s, b_s = ain.streams.video[0], bin_.streams.video[0]
        a_s.thread_type = b_s.thread_type = "AUTO"
        bframes = (f for f in bin_.decode(b_s) if f.pts is not None)
        cur_b, next_b = None, next(bframes, None)
        enc = pick_encoder(encoder) if encoder else pick_encoder()
        vs = as_ = None
        audio = iter(())
        pending_a = None
        base = None
        n = 0
        last = 0
        for fa in ain.decode(a_s):
            if fa.pts is None:
                continue
            if base is None:
                base = fa.pts
                h, w = fa.height & ~1, fa.width & ~1
                if layout == "side":
                    scale = min(1.0, 4096 / (2 * w))
                    hw_, oh = int(w * scale) & ~1, int(h * scale) & ~1
                    ow = 2 * hw_
                else:
                    hw_, ow, oh = w, w, h
                out = av.open(str(tmp), "w", format="mp4", container_options={"movflags": "+faststart"})
                vs = out.add_stream(enc, rate=AFTER_NOMINAL)
                _configure_video(vs, enc, ow, oh, AFTER_NOMINAL, AFTER_BPP * ow * oh, True)
                if sin is not None and sin.streams.audio:
                    a_in = sin.streams.audio[0]
                    as_ = out.add_stream_from_template(a_in)
                    audio = (p for p in sin.demux(a_in) if p.pts is not None and p.dts is not None and p.pts >= base)
                    pending_a = next(audio, None)
                lab_b, lab_a = _label(labels[0], oh), _label(labels[1], oh)
                amber = _planes(bgr_to_i420(np.full((2, 2, 3), DIVIDER_BGR, np.uint8)), 2, 2)
                amber = [int(p[0, 0]) for p in amber]
            while next_b is not None and next_b.pts <= fa.pts:
                cur_b, next_b = next_b, next(bframes, None)
            fb = cur_b if cur_b is not None else next_b
            A, ha, wa = _yuv(fa)
            B, hb, wb = _yuv(fb) if fb is not None else (A, ha, wa)
            pa, pb = _planes(A, ha, wa), _planes(B, hb, wb)
            if (hb, wb) != (ha, wa):
                pb = [cv2.resize(p, (q.shape[1], q.shape[0]), interpolation=cv2.INTER_AREA) for p, q in zip(pb, pa)]
            pa, pb = _crop(pa, h, w), _crop(pb, h, w)
            if layout == "side":
                if hw_ != w:
                    pa = [cv2.resize(p, (hw_ >> (i > 0), oh >> (i > 0)), interpolation=cv2.INTER_AREA)
                          for i, p in enumerate(pa)]
                    pb = [cv2.resize(p, (hw_ >> (i > 0), oh >> (i > 0)), interpolation=cv2.INTER_AREA)
                          for i, p in enumerate(pb)]
                planes = [np.ascontiguousarray(np.hstack([b, a])) for a, b in zip(pa, pb)]
            else:
                planes = [np.ascontiguousarray(a) for a in pa]
                for i, (p, b) in enumerate(zip(planes, pb)):
                    half = (w // 2) >> (i > 0)
                    p[:, :half] = b[:, :half]
            mid = (ow // 2) & ~1
            planes[0][:, mid - 2:mid + 2] = amber[0]
            planes[1][:, mid // 2 - 1:mid // 2 + 1] = amber[1]
            planes[2][:, mid // 2 - 1:mid // 2 + 1] = amber[2]
            m = round(24 * oh / 1080)
            _blend(planes, lab_b, m, m)
            _blend(planes, lab_a, mid + m, m)
            flat = np.concatenate([p.ravel() for p in planes]).reshape(oh * 3 // 2, ow)
            vf = av.VideoFrame.from_ndarray(flat, format="yuv420p")
            vf.pts, vf.time_base = fa.pts - base, Fraction(1, TB)
            last = fa.pts - base
            for p in vs.encode(vf):
                out.mux(p)
            while pending_a is not None and pending_a.pts - base <= last:
                pending_a.pts -= base
                pending_a.dts -= base
                pending_a.stream = as_
                out.mux(pending_a)
                pending_a = next(audio, None)
            n += 1
        if out is None:
            raise RuntimeError("no frames in the clip")
        for p in vs.encode(None):
            out.mux(p)
    finally:
        if out is not None:
            out.close()
        for c in (ain, bin_, sin):
            if c is not None:
                c.close()
    os.replace(tmp, out_path)
    return last / TB


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Instant replay tools (SmoovPlay runs this itself).")
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("compose", help="make a before & after video from instant replay's tracks")
    c.add_argument("--after", required=True)
    c.add_argument("--before", required=True)
    c.add_argument("--sound")
    c.add_argument("--out", required=True)
    c.add_argument("--layout", choices=["split", "side"], default="split")
    c.add_argument("--labels", nargs=2, default=["Original", "SmoovPlay"])
    c.add_argument("--encoder", default=None)
    c.add_argument("--cleanup", metavar="FOLDER", help="delete this folder (the tracks) when done")
    c.add_argument("--mark", metavar="FOLDER", help="keep FOLDER/<process id>.txt (the video's path) while making it, "
                   "for the launcher; if it fails, the file stays, with why")
    a = ap.parse_args(argv)
    mark = None
    if a.mark:
        try:
            Path(a.mark).mkdir(parents=True, exist_ok=True)
            mark = Path(a.mark) / f"{os.getpid()}.txt"
            mark.write_text(a.out + "\n", encoding="utf-8")
        except OSError:
            mark = None
    try:
        secs = compose(a.after, a.before, a.out, a.sound, a.layout, tuple(a.labels), a.encoder or None)
    except BaseException as e:
        Path(a.out + ".part").unlink(missing_ok=True)
        if mark is not None:
            try:
                mark.write_text(f"{a.out}\nfailed: {type(e).__name__}: {str(e)[:300]}\n", encoding="utf-8")
            except OSError:
                pass
        raise
    finally:
        if a.cleanup:
            shutil.rmtree(a.cleanup, ignore_errors=True)
    if mark is not None:
        mark.unlink(missing_ok=True)
    try:
        print(f"[replay] before & after: {secs:.1f} s -> {a.out}")
    except OSError:  # SmoovPlay has closed meanwhile: nobody is reading
        pass
    return 0


if __name__ == "__main__":
    sys.exit(main())
