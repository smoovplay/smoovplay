"""Frames another app hands SmoovPlay through the frame tap (docs/frame-tap.md), the same shared memory chiaki-ng
writes PS5 Remote Play's frames into.

A writer sends either kind of frame:
- compressed (H.264 or H.265, version 1 of the layout): read by framegen.chiaki_tap.ChiakiSource exactly as
  chiaki-ng's are, with repeats told from each frame's size;
- decoded (NV12 or BGRA, version 2): read here by TapSource. Nothing to decode: each frame is copied out of its
  slot, uploaded and converted to BGR on the graphics card. Repeats are the writer's word when it gives one
  (FLAG_NEW / FLAG_REPEAT), else the pictures are compared (framegen.sources.make_detector, as for an app's window).

A writer that sends a frame only when the picture changes sets FEATURE_VARIABLE, and run.py puts it on a 60 Hz beat
(framegen.sources.Clock60), as for window capture.

    python run.py --source tap [--tap NAME] --display
    python scripts/frametap_send.py VIDEO [--format nv12|bgra]   # a stand-in writer, in another terminal

Experimental: tested with scripts/frametap_send.py, not yet with another app.
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import replace

import numpy as np

from .chiaki_tap import (AUDIO_SUFFIX, CODEC_NAMES, CODECS, COLOURS, FEATURE_VARIABLE, FLAG_NEW, FLAG_REPEAT,
                         RAW_BYTES, RAW_CODECS, AudioTap, FrameTap, Header)
from .io import Tick
from .sync import CAPTURE_LOCK

DEFAULT_NAME = "Local\\smoovplay-tap"  # chiaki-ng's is Local\chiaki-frametap
# Compressed frames' decoder: NVDEC on Windows, Apple's VideoToolbox on a Mac (ChiakiSource._new_decoder)
DEFAULT_DECODER = "videotoolbox" if sys.platform == "darwin" else "cuda"


def wait_for_writer(tap: FrameTap, wait_s: float) -> Header:
    """The header once a writer has started a stream. TimeoutError after wait_s."""
    t_end, said = time.perf_counter() + wait_s, False
    while not (h := tap.header()).live:
        if not said:
            print("[tap] waiting for an app to start sending frames...")
            said = True
        if time.perf_counter() > t_end:
            raise TimeoutError("no app is sending frames to the frame tap")
        tap.wait(0.25)
    return h


def open_tap(name: str = DEFAULT_NAME, decoder: str = DEFAULT_DECODER, wait_s: float = 600.0, test=None,
             log_path: str | None = None):
    """(kind, source) for whatever the writer on `name` sends: ("chiaki", a ChiakiSource) for compressed frames,
    ("tap", a TapSource) for decoded ones, put on a 60 Hz beat if the writer sets FEATURE_VARIABLE. test: a
    framegen.sources.TestMode, for a writer that sends every refresh (left out for a variable one, and ValueError if
    it's on). decoder: for compressed frames, "cuda" (NVDEC), "videotoolbox" (a Mac), "d3d11va" or "sw"."""
    tap = FrameTap(name, name + "-frame")
    h = wait_for_writer(tap, wait_s)
    if h.codec in RAW_CODECS:
        variable = bool(h.features & FEATURE_VARIABLE)
        if variable and test is not None and test.on:
            tap.close()
            raise ValueError("test mode needs an app that sends every refresh; this one sends only new pictures")
        src = TapSource(tap, name, h, test=None if variable else test)
        if variable:
            from .sources import Clock60
            return "tap", Clock60(src)
        return "tap", src
    tap.close()
    if h.codec not in CODECS:
        raise ValueError(f"the app sends frames in a format SmoovPlay doesn't know (codec {h.codec})")
    from .chiaki_tap import ChiakiSource
    src = ChiakiSource(decoder, wait_s=wait_s, log_path=log_path, tap_name=name, test=test, app=True)
    src.info = replace(src.info, name="Frame tap", audio="The app plays its own sound",
                       controls="Your controller connects to the app as usual")
    return "chiaki", src


class TapSource:
    """Decoded frames (NV12 or BGRA) from the frame tap, as Ticks. Joins the live stream (never replays old
    frames); reads nothing until the pipeline asks for frames, and nothing while it restarts (hold).

    device: "cuda" (default when there is CUDA), "cpu" (torch on the CPU, for tests) or None (numpy and OpenCV).
    """

    def __init__(self, tap: FrameTap, name: str, header: Header, test=None, device: str | None = "auto") -> None:
        from .capture import _cuda_device
        from .sources import source_info
        self.tap, self.test = tap, test
        self.sound = AudioTap(name + AUDIO_SUFFIX)  # the app's sound, if it sends any, for instant replay
        self.codec, self.session = header.codec, header.session
        self.fmt = RAW_CODECS[header.codec]
        self.w, self.h = header.width, header.height
        self.n = int(self.w * self.h * RAW_BYTES[self.fmt])
        self.bt709, self.full_range = COLOURS.get(header.colour, (self.h >= 720, False))
        variable = bool(header.features & FEATURE_VARIABLE)
        self.info = source_info("tap", hz=float(header.max_fps or 60), rate="variable" if variable else "fixed",
                                test_mode=not variable)
        self.device = _cuda_device() if device == "auto" else device
        print(f"[tap] frames: {CODEC_NAMES[header.codec]} {self.w}x{self.h}, "
              + ("frames when the picture changes" if variable else f"{header.max_fps} a second"))
        self.q: queue.Queue = queue.Queue(maxsize=3)
        self.dropped = 0  # frames dropped because the pipeline fell behind
        self.skipped = 0  # frames the writer overwrote before they were read
        self._host = None  # (pinned torch buffer, its numpy view): each frame is copied into it, then uploaded
        self._buf = np.empty(self.n, np.uint8) if self.device is None else None  # the same, without torch
        self._stop = False
        self._go = threading.Event()
        self._beat = self._ring_t = time.perf_counter()
        self._ring_seq = -1
        self._thread = threading.Thread(target=self._reader, daemon=True)
        self._thread.start()

    def _frame_buffer(self) -> np.ndarray:
        if self.device is None:
            return self._buf
        if self._host is None:
            import torch
            t = torch.empty(self.n, dtype=torch.uint8, pin_memory=self.device == "cuda")
            self._host = (t, t.numpy())
        return self._host[1]

    def _convert(self):
        """The frame in the buffer -> HxWx3 uint8 BGR (a tensor on self.device, or numpy)."""
        if self.device is None:
            a = self._buf
            if self.fmt == "bgra":
                return a.reshape(self.h, self.w, 4)[..., :3].copy()
            from .capture import raw_to_bgr_cpu
            return raw_to_bgr_cpu(a, "nv12", self.w, self.h)
        t = self._host[0].to(self.device)  # from pinned memory: done before the buffer is reused
        if self.fmt == "bgra":
            return t.view(self.h, self.w, 4)[..., :3].contiguous()
        from .capture import raw_to_bgr
        return raw_to_bgr(t, "nv12", self.w, self.h, self.bt709, self.full_range)

    def _reader(self) -> None:
        from .winutil import boost_thread
        boost_thread()
        seq = None  # the next frame to read; None: from the writer's newest
        try:
            while not self._stop:
                self._beat = time.perf_counter()
                h = self.tap.header()
                if not h.live:
                    print("[tap] the app stopped sending frames")
                    break
                if h.session != self.session:  # the writer started over
                    if (h.codec, h.width, h.height) != (self.codec, self.w, self.h):
                        print(f"[tap] the frames changed to {CODEC_NAMES.get(h.codec, h.codec)} {h.width}x{h.height}; "
                              "restart SmoovPlay", file=sys.stderr)
                        break
                    self.session, seq = h.session, None
                if not self._go.is_set() or seq is None:  # the pipeline isn't ready: nothing read, no GPU work
                    seq = h.write_seq if self._go.is_set() else None
                    self.tap.heartbeat()
                    self.tap.wait(0.05)
                    continue
                if h.write_seq == seq:
                    self.tap.heartbeat()
                    self.tap.wait(0.1)
                    continue
                if h.write_seq - seq >= h.slots - 1:  # fell behind: the oldest slots are being rewritten
                    self.skipped += h.write_seq - 1 - seq
                    seq = h.write_seq - 1
                p = self.tap.read(seq, h.qpc_freq, h.slots, h.slot_size, into=self._frame_buffer())
                seq += 1
                if p is None or p.size < self.n:  # overwritten while we read it, or a short frame
                    self.skipped += 1
                    continue
                with CAPTURE_LOCK:  # no GPU work while the pipeline captures its CUDA graphs (startup only)
                    img = self._convert()
                new_hint = True if p.flags & FLAG_NEW else False if p.flags & FLAG_REPEAT else None
                tick = Tick(img, p.t_arrival, 0, new_hint=new_hint,
                            damaged=bool(p.frames_lost > 0 or p.recovered or p.truncated),
                            t_ready=time.perf_counter(), extra={"seq": p.seq})
                if self.test is not None and (tick := self.test(tick)) is None:
                    continue
                try:
                    self.q.put_nowait(tick)
                except queue.Full:
                    self.q.get_nowait()
                    self.dropped += 1
                    self.q.put_nowait(tick)
        except Exception as e:  # noqa: BLE001 - end the stream rather than leave the pipeline waiting
            print(f"[tap] stopped: {str(e)[:200]}", file=sys.stderr)
        self.q.put(None)

    def __iter__(self):
        from .io import queue_ticks
        self._go.set()
        yield from queue_ticks(self.q, "the app", self.why)

    def hold(self, on: bool) -> None:
        """Stop reading while the pipeline restarts (on), and start again from the newest frame (off)."""
        if on:
            self._go.clear()
            return
        while True:  # frames from before the restart are stale by now
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self._go.set()

    def why(self) -> str:
        """A hint under the window's "No picture" note while no frames come ("" if none)."""
        now = time.perf_counter()
        h = self.tap.header()
        if h.write_seq != self._ring_seq:
            self._ring_seq, self._ring_t = h.write_seq, now
        if not h.live:
            return "The app stopped sending frames"
        if now - self._beat > 3:
            return "SmoovPlay stopped reading frames. Press Q, then Play again"
        if now - self._ring_t > 1:
            return "The app isn't sending any frames. Is it still running?"
        return ""

    # Instant replay's sound, if the writer fills the sound ring (docs/frame-tap.md); the app plays its own sound.
    def has_sound(self) -> bool:
        return self.sound.live()

    def read_sound(self) -> list:
        return self.sound.read_new()

    def can_press(self) -> bool:
        return False

    def can_delay_audio(self) -> bool:
        return False

    def set_audio_delay(self, ms: float) -> None:
        pass

    def close(self) -> None:
        self._stop = True
        self._go.set()
        self._thread.join(timeout=2)
        if not self._thread.is_alive():  # (a reader stuck mid-copy keeps the memory mapped until it exits)
            self.tap.close()
            self.sound.close()
        print(f"[tap] dropped frames: {self.dropped}, overwritten before they were read: {self.skipped}")
