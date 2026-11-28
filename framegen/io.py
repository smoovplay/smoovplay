"""Frame sources (PS5 simulator, plain file, capture card) and sinks (window, file)."""

from __future__ import annotations

import os
import queue
import random
import subprocess
import sys
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import NamedTuple

import cv2
import numpy as np

from .fill import Panorama, layout, panorama_t, visible, wider
from .glblit import Blitter

TICK = 1.0 / 60.0


@dataclass
class Tick:
    frame: np.ndarray  # HxWx3 uint8 BGR (or a CUDA tensor from a GPU decoder), never mutated after yield
    t_arrival: float  # perf_counter() when the frame reached this PC (Remote Play: when chiaki-ng got it, before
    # decoding); latency is measured from here
    index: int  # tick number
    # Ground truth, only known for simulated input (and test mode):
    content: int | None = None  # source-frame index shown on this tick
    truth_new: bool | None = None
    # What the source knows about the frame (framegen.sources.SourceInfo says which a source fills in):
    new_hint: bool | None = None  # new content (True) or a repeat (False), by the source's own account; None: the
    # pipeline compares pictures
    damaged: bool = False  # known to be corrupt (frames lost before it); held like a scene cut, not blended
    t_ready: float | None = None  # when the pipeline could first have it (after decoding); None: t_arrival
    t_source: float | None = None  # the source's own time for the frame, in seconds (a file's position; later, a
    # capture card's timestamp), for timing frames by the source's clock rather than by arrival
    extra: dict = field(default_factory=dict)  # anything else, for logs (e.g. Remote Play's per-frame facts)


class Waiting(NamedTuple):
    """What a live source yields instead of blocking when no frame has come for a moment (the stream stalled,
    chiaki-ng is reconnecting, the card lost its signal). The pipeline then keeps the window answering: its keys, the
    launcher's commands and Windows' "not responding" check, which closed SmoovPlay mid-session when a stream
    stalled with the loop stuck waiting for a frame."""
    since: float  # seconds since the last frame
    what: str  # where frames come from, for the message on screen ("the PS5")
    why: str = ""  # what the source knows about it, as a hint for the player ("" if nothing)


WAIT_POLL_S = 0.1  # how long a source waits for a frame before saying so


def queue_ticks(q: queue.Queue, what: str, why=None):
    """A live source's ticks, as its reader thread puts them in `q` (None ends them), numbered from 0, with a
    Waiting in between whenever none has come for WAIT_POLL_S. why: () -> the source's hint for the Waiting."""
    i, last = 0, time.perf_counter()
    while True:
        try:
            tick = q.get(timeout=WAIT_POLL_S)
        except queue.Empty:
            yield Waiting(time.perf_counter() - last, what, why() if why is not None else "")
            continue
        if tick is None:
            return
        tick.index = i
        yield tick
        i += 1
        last = time.perf_counter()


class Sound(NamedTuple):
    """A chunk of the game's sound, for instant replay (framegen/replay.py): what a source's read_sound() returns."""
    t: float  # perf_counter() when its first sample reached this PC
    pcm: bytes  # interleaved signed 16-bit samples
    rate: int
    channels: int


# ----------------------------------------------------------------------------- sources
class _Decoder(threading.Thread):
    """Decodes a video file on a background thread (cv2 releases the GIL while decoding)."""

    def __init__(self, path: str, maxsize: int = 16) -> None:
        super().__init__(daemon=True)
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            raise FileNotFoundError(path)
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.q: queue.Queue = queue.Queue(maxsize)
        self.start()

    def run(self) -> None:
        while True:
            ok, f = self.cap.read()
            self.q.put(f if ok else None)
            if not ok:
                return

    def __iter__(self):
        while (f := self.q.get()) is not None:
            yield f


class Ps5SimSource:
    """Turns a native 60 fps recording into what a PS5 sends for a 30 fps game.

    Source frames 0,2,4,... become the game's real frames; each is scanned out for two
    60 Hz ticks (three with probability `hitch`, to mimic a dropped game frame). The
    odd source frames are never shown, so they serve as ground truth for the
    interpolated midpoints (`gt(i)`).

    A ~30 fps file (e.g. real PS5 footage) is used frame-for-frame instead: every frame
    is a game frame, there is no ground truth, and scoring is skipped.
    """

    def __init__(self, path: str, realtime: bool = False, hitch: float = 0.0, noise: float = 0.0,
                 max_ticks: int | None = None, seed: int = 0) -> None:
        from .sources import source_info
        self.dec = _Decoder(path)
        self.w, self.h = self.dec.w, self.dec.h
        self.info = source_info("sim", live=realtime)
        self.realtime, self.hitch, self.max_ticks = realtime, hitch, max_ticks
        self.step = 2 if self.dec.fps > 45 else 1  # 60 fps source: drop odd frames; 30 fps: use all
        if self.step == 1:
            print(f"[sim] {self.dec.fps:.2f} fps source: using every frame as a game frame (no ground truth)")
        self.rng = random.Random(seed)
        self.history: OrderedDict[int, np.ndarray] = OrderedDict()
        self._noise = None
        if noise > 0:  # precomputed noise planes so adding noise stays cheap in realtime
            r = np.random.default_rng(seed)
            n = [np.clip(r.normal(0, noise, (self.h, self.w, 3)), -255, 255) for _ in range(6)]
            self._noise = [(np.clip(x, 0, 255).astype(np.uint8), np.clip(-x, 0, 255).astype(np.uint8)) for x in n]

    def gt(self, i: int) -> np.ndarray | None:
        return self.history.get(i)

    def __iter__(self):
        t0 = time.perf_counter()
        tick = 0
        for i, f in enumerate(self.dec):
            self.history[i] = f
            while len(self.history) > 8:
                self.history.popitem(last=False)
            if i % self.step:
                continue  # odd frames of a 60 fps source: the game "never rendered" these
            repeats = 3 if self.rng.random() < self.hitch else 2
            for r in range(repeats):
                if self.max_ticks is not None and tick >= self.max_ticks:
                    return
                out = f
                if self._noise is not None:
                    pos, neg = self._noise[self.rng.randrange(len(self._noise))]
                    out = cv2.subtract(cv2.add(f, pos), neg)
                if self.realtime:
                    # A capture card delivers on the HDMI clock whether or not we keep up, so
                    # latency is measured from the scheduled arrival (includes any backlog).
                    t_arr = t0 + tick * TICK
                    if (wait := t_arr - time.perf_counter()) > 0:
                        time.sleep(wait)
                else:
                    t_arr = time.perf_counter()
                yield Tick(out, t_arr, tick, content=i if self.step == 2 else None,
                           truth_new=(r == 0) if self.step == 2 else None, t_source=i / self.dec.fps)
                tick += 1

    def close(self) -> None:
        pass


class FileSource:
    """Plays a file's frames as-is, e.g. a raw recording from the capture card. test: a framegen.sources.TestMode,
    to play a 60 fps recording as a 30 fps game."""

    def __init__(self, path: str, realtime: bool = False, max_ticks: int | None = None, test=None) -> None:
        from .sources import source_info
        self.dec = _Decoder(path)
        self.w, self.h = self.dec.w, self.dec.h
        self.realtime, self.max_ticks, self.test = realtime, max_ticks, test
        self.info = source_info("file", live=realtime, hz=self.dec.fps)

    def __iter__(self):
        t0 = time.perf_counter()
        for i, f in enumerate(self.dec):
            if self.max_ticks is not None and i >= self.max_ticks:
                return
            if self.realtime and (wait := t0 + i * TICK - time.perf_counter()) > 0:
                time.sleep(wait)
            tick = Tick(f, time.perf_counter(), i, t_source=i / self.dec.fps)
            if self.test is None or (tick := self.test(tick)) is not None:
                yield tick

    def close(self) -> None:
        pass


class CaptureSource:
    """Live capture card via OpenCV (DirectShow). Experimental: not tested with a real card yet, and not offered in the
    launcher until it is.

    Frames are read on a thread and queued (not "latest only"), because dropping a
    frame would break duplicate detection. If processing can't keep up the queue
    overflows and the oldest frame is dropped, which is reported as `dropped`.

    device: the card's DirectShow index, or a video file that stands in for one (played at its own frame rate on
    the same thread-and-queue path), to test the capture path and test mode without a card. A file waits for the
    pipeline to start reading, so all of it is played; a card is live, and what comes before is dropped.
    test: a framegen.sources.TestMode, applied before the queue.
    """

    def __init__(self, device: int | str, width: int = 1920, height: int = 1080, fps: int = 60,
                 fourcc: str = "MJPG", backend: int = cv2.CAP_DSHOW, test=None) -> None:
        from .sources import source_info
        self.stand_in = isinstance(device, str)
        if self.stand_in:
            self.cap = cv2.VideoCapture(device)
            if not self.cap.isOpened():
                raise FileNotFoundError(device)
        else:
            self.cap = cv2.VideoCapture(device, backend)
            if not self.cap.isOpened():
                raise RuntimeError(f"could not open capture device {device}")
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            self.cap.set(cv2.CAP_PROP_FPS, fps)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.w = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.h = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS) or 60.0
        print(f"[capture] {self.w}x{self.h} @ {self.fps:.2f} fps"
              + (f", a file standing in for a capture card: {device}" if self.stand_in else f" requested {fourcc}"))
        self.info = source_info("capture", hz=self.fps,
                                name="Capture card" + (" (a file standing in)" if self.stand_in else ""))
        self.test = test
        self.q: queue.Queue = queue.Queue(maxsize=3)
        self.dropped = 0
        self._stop = False
        self._go = threading.Event()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self) -> None:
        if self.stand_in:
            self._go.wait()
        t0, n = time.perf_counter(), 0
        while not self._stop:
            ok, f = self.cap.read()
            if not ok:
                self.q.put(None)
                return
            if self.stand_in and (wait := t0 + n / self.fps - time.perf_counter()) > 0:
                time.sleep(wait)  # a card delivers on the HDMI clock
            tick = Tick(f, time.perf_counter(), 0, t_source=n / self.fps if self.stand_in else None)
            n += 1
            if self.test is not None and (tick := self.test(tick)) is None:
                continue
            try:
                self.q.put_nowait(tick)
            except queue.Full:
                self.q.get_nowait()
                self.dropped += 1
                self.q.put_nowait(tick)

    def __iter__(self):
        self.dropped = 0  # frames dropped while RIFE was loading don't count
        self._go.set()
        yield from queue_ticks(self.q, "the capture card")

    def close(self) -> None:
        self._stop = True
        self._go.set()
        self.cap.release()
        print(f"[capture] dropped frames: {self.dropped}")




# ----------------------------------------------------------------------------- sinks
def _is_cuda(frame) -> bool:
    return getattr(frame, "is_cuda", False)


def _host(frame) -> np.ndarray:
    """A frame as a numpy array (CUDA tensors are copied to the CPU)."""
    return frame.cpu().numpy() if _is_cuda(frame) else frame


# Low delay (DXGI) is for windowed play. Fullscreen already skips the compositor's copy with the standard presentation,
# and Low delay in fullscreen could freeze (Present blocked for seconds, or for good), so by default ("none") a session
# that starts fullscreen uses the standard presentation, and a Low delay window stays a window until the next Play.
# Under test (scripts/dxgi_smoke.py): "monitor", GLFW's fullscreen (an always-on-top window exactly over the monitor);
# "borderless", a plain borderless window over it, as games' borderless mode. See framegen/dxgi_present.py.
# The standard presentation's fullscreen is that borderless window too, unless this is "monitor": GLFW's stays on top
# of everything, so Alt+Tab to another program left the video over it.
FULLSCREEN_STYLE = os.environ.get("FRAMEGEN_FULLSCREEN_STYLE", "none")

# With Windows' HDR on, OpenGL's own fullscreen (and an OpenGL window over the whole monitor too) switches the display
# to SDR while the window is in front, and back to HDR when another window is: the picture's colours change with each
# Alt+Tab or click, and the display's SDR mode can look different again (warmer, say). A DXGI swap chain keeps
# Windows' HDR on in fullscreen, and Windows shows the SDR picture in it as it shows any SDR window. So when the main
# display has Windows' HDR on, the window presents through DXGI (8-bit) and fullscreen is a borderless window over the
# monitor, as with HDR output. FRAMEGEN_KEEP_HDR=0 turns this off (OpenGL's own presentation, as before).
KEEP_HDR = os.environ.get("FRAMEGEN_KEEP_HDR", "1") != "0"
KEEP_HDR_REASON = ("With Windows' HDR on, the window always uses the low-delay presentation: the standard one switches "
                   "the display to SDR in fullscreen, which changes the colours.")


def windows_hdr_on(device: str | None = None) -> bool:
    """Whether the chosen display (or main display, if None) has Windows' HDR on (False if it can't be told)."""
    if sys.platform != "win32":
        return False
    try:
        from .dxgi_present import display_hdr
        d = display_hdr(device=device) if device else display_hdr()
        return bool(d and d["hdr"])
    except Exception:  # noqa: BLE001 - unknown: OpenGL's own presentation, as before
        return False


def _first_size(glfw, w: int, h: int) -> tuple[int, int]:
    """The window's size when it opens. It's the frame's, at most 1600x900, and on a Mac also at most 90% of the
    main display's usable area (a 13-inch MacBook is about 1470 points wide), keeping the frame's shape."""
    ww, wh = min(w, 1600), min(h, 900)
    if sys.platform == "darwin":
        try:
            _, _, aw, ah = glfw.get_monitor_workarea(glfw.get_primary_monitor())
            k = min(1.0, 0.9 * aw / ww, 0.9 * ah / wh)
            ww, wh = int(ww * k), int(wh * k)
        except Exception:  # noqa: BLE001 - then the size other systems get
            pass
    return ww, wh


def window_monitor(win: tuple[int, int, int, int], monitors: list[tuple]):
    """The monitor a window (x, y, w, h) is mostly on, of monitors given as (handle, x, y, w, h): the one with the
    most of the window on it, or the first (GLFW lists the primary monitor first) when it's on none of them."""
    x, y, w, h = win
    best, most = monitors[0][0], 0
    for mon, mx, my, mw, mh in monitors:
        area = max(0, min(x + w, mx + mw) - max(x, mx)) * max(0, min(y + h, my + mh) - max(y, my))
        if area > most:
            best, most = mon, area
    return best


def _mac_hide_bars(hide: bool) -> None:
    """On a Mac, hides the menu bar and the Dock while SmoovPlay is the app in front, so a borderless window can cover
    the whole screen like fullscreen. They slide in when the pointer reaches them, and come back by themselves when
    another app is in front. hide=False puts them back. This goes through the Objective-C runtime, as GLFW has no
    call for it. A failure only leaves them showing."""
    import ctypes
    import ctypes.util
    try:
        objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        objc.objc_getClass.restype, objc.objc_getClass.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
        objc.sel_registerName.restype, objc.sel_registerName.argtypes = ctypes.c_void_p, [ctypes.c_char_p]
        send = ctypes.cast(objc.objc_msgSend, ctypes.c_void_p).value
        get = ctypes.CFUNCTYPE(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)(send)
        put = ctypes.CFUNCTYPE(None, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_ulong)(send)
        app = get(objc.objc_getClass(b"NSApplication"), objc.sel_registerName(b"sharedApplication"))
        # NSApplicationPresentationAutoHideDock | AutoHideMenuBar, or Default
        put(app, objc.sel_registerName(b"setPresentationOptions:"), (1 | 4) if hide else 0)
    except Exception as e:  # noqa: BLE001 - it only affects the bars, and the picture goes on
        print(f"[display] couldn't {'hide' if hide else 'show'} the menu bar and the Dock ({str(e)[:160]})")


def _say_if_slow(what: str, t0: float, limit_ms: float = 250.0) -> None:
    """A line in the log when a window call took far longer than a frame (a stall worth reporting)."""
    ms = (time.perf_counter() - t0) * 1000
    if ms > limit_ms:
        print(f"[display] {what} took {ms:.0f} ms")


def _draw_text(frame: np.ndarray, text: str) -> np.ndarray:
    frame = frame.copy()
    for i, line in enumerate(text.split("\n")):
        y = 40 + 40 * i
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, line, (20, y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def _gl_image(GL, img: np.ndarray, hdr: bool):
    """An 8-bit BGR image as (pixels, internal format, pixel type) for a GL upload: as it is, or for an HDR10 picture
    (framegen/hdr.py) as 16-bit PQ with white at SDR graphics white, so it looks as on an SDR one."""
    if not hdr:
        return np.ascontiguousarray(img), GL.GL_RGBA8, GL.GL_UNSIGNED_BYTE
    from .hdr import sdr_to_pq16_image
    return np.ascontiguousarray(sdr_to_pq16_image(img)), GL.GL_RGBA16, GL.GL_UNSIGNED_SHORT


class _Panel:
    """A BGR image in a GL texture of its own, blitted 1:1 over the frame (menu, messages). hdr: over an HDR10
    picture (see _gl_image). blitter: the window's (framegen/glblit.py)."""

    def __init__(self, GL, hdr: bool = False, blitter: Blitter | None = None) -> None:
        self.GL, self.w, self.h, self.hdr = GL, 0, 0, hdr
        self.blitter = blitter if blitter is not None else Blitter(GL)
        self.tex, self.fbo = GL.glGenTextures(1), GL.glGenFramebuffers(1)

    def update(self, img: np.ndarray) -> None:
        GL = self.GL
        h, w = img.shape[:2]
        img, fmt, kind = _gl_image(GL, img, self.hdr)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        if (w, h) != (self.w, self.h):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, fmt, w, h, 0, GL.GL_BGR, kind, img)
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex, 0)
            self.w, self.h = w, h
        else:
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, w, h, GL.GL_BGR, kind, img)

    def blit(self, x: int, top: int) -> None:
        """top: GL y of the image's top edge (GL y grows upwards; image rows go down)."""
        self.blitter.copy(self.tex, self.w, self.h, self.fbo, (0, 0, self.w, self.h),
                          (x, top, x + self.w, top - self.h), False)

    def close(self) -> None:
        self.GL.glDeleteFramebuffers(1, [self.fbo])
        self.GL.glDeleteTextures([self.tex])


class MiniText(str):
    """The minimal stats overlay (Stats overlay style: Minimal), from the pipeline: a str (the plain line, and a
    second one while a test runs) for sinks that just print text, plus what GLWindowSink draws from it:
    parts, [(text, kind)] with kind "main", "dim" or "warn"; health, how well the PC keeps up ("good", "tight",
    "behind", "off" with frame generation off, "" not known yet); second, the test line or ""."""

    parts: tuple = ()
    health: str = ""
    second: str = ""

    @classmethod
    def make(cls, parts, health: str = "", second: str = "") -> "MiniText":
        t = cls("  ".join(x for x, _ in parts) + (f"\n{second}" if second else ""))
        t.parts, t.health, t.second = tuple(parts), health, second
        return t


# BGR: the minimal overlay's dot (how well the PC keeps up) and its text
HEALTH_COLORS = {"good": (110, 200, 80), "tight": (40, 180, 250), "behind": (70, 70, 235), "off": (150, 150, 150),
                 "": (110, 110, 110)}
MINI_TEXT = {"main": (245, 245, 245), "dim": (185, 185, 185), "warn": (40, 180, 250)}


# Stats overlay size: times the screen's own scale (its height / 1080). Small sits in the edge games keep clear of
# their HUD (title safe area, about 3-5% of the screen): at 1440p its box is about 24 px tall, 4 px from the edges.
OVERLAY_SIZES = {"small": 0.8, "medium": 1.0, "large": 1.5}


def _mini(t: MiniText, ui: float) -> np.ndarray:
    """The minimal overlay: a small dark box with a dot (green keeping up, amber tight, red falling behind) and a few
    numbers, plus the test line under them while a test runs."""
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.52 * ui, 1 if ui < 1.6 else 2
    small = 0.42 * ui
    pad, gap, r = max(3, int(6 * ui)), int(10 * ui), max(3, int(4.5 * ui))
    th_main = cv2.getTextSize("90", font, fs, th)[0][1]
    widths = [cv2.getTextSize(x, font, fs, th)[0][0] for x, _ in t.parts]
    line_w = 2 * r + gap + sum(widths) + gap * max(0, len(widths) - 1)
    second_w, second_h = (cv2.getTextSize(t.second, font, small, 1)[0] if t.second else (0, 0))
    w = 2 * pad + max(line_w, second_w)
    h = 2 * pad + th_main + (second_h + int(8 * ui) if t.second else 0)
    img = np.full((h, w, 3), (30, 27, 25), np.uint8)
    base = pad + th_main
    cv2.circle(img, (pad + r, base - th_main // 2), r, HEALTH_COLORS.get(t.health, HEALTH_COLORS[""]), -1, cv2.LINE_AA)
    x = pad + 2 * r + gap
    for (text, kind), tw in zip(t.parts, widths):
        cv2.putText(img, text, (x, base), font, fs, MINI_TEXT.get(kind, MINI_TEXT["dim"]), th, cv2.LINE_AA)
        x += tw + gap
    if t.second:
        cv2.putText(img, t.second, (pad, base + int(8 * ui) + second_h), font, small, MINI_TEXT["dim"], 1, cv2.LINE_AA)
    return img


def _detail(text: str, ui: float, min_w: int = 0) -> np.ndarray:
    """The detailed overlay: its line (and a second, the advice or a test line) on a small dark box like the minimal
    one's, at the same text size. min_w: the box only grows, so it doesn't jitter as the numbers change width."""
    font, fs, th = cv2.FONT_HERSHEY_SIMPLEX, 0.52 * ui, 1 if ui < 1.6 else 2
    pad, lead = max(3, int(6 * ui)), int(8 * ui)
    lines = text.split("\n")[:2]
    line_h = cv2.getTextSize("90", font, fs, th)[0][1]
    w = max(min_w, 2 * pad + max(cv2.getTextSize(x, font, fs, th)[0][0] for x in lines))
    img = np.full((2 * pad + len(lines) * line_h + (len(lines) - 1) * lead, w, 3), (30, 27, 25), np.uint8)
    for i, line in enumerate(lines):
        cv2.putText(img, line, (pad, pad + line_h + i * (line_h + lead)), font, fs,
                    MINI_TEXT["main" if i == 0 else "dim"], th, cv2.LINE_AA)
    return img


def _pill(text: str, ui: float, color=(238, 238, 238)) -> np.ndarray:
    """A short message (one line, or a few split by newlines) on a dark rounded-off box."""
    fs, th = 0.7 * ui, 1 if ui < 1.4 else 2
    lines = text.split("\n")
    sizes = [cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, fs, th)[0] for t in lines]
    tw, tht = max(w for w, _ in sizes), max(h for _, h in sizes)
    padx, pady, gap = int(18 * ui), int(14 * ui), int(12 * ui)
    img = np.full((len(lines) * (tht + gap) - gap + 2 * pady + int(6 * ui), tw + 2 * padx, 3), (34, 30, 28),
                  np.uint8)
    img[:, : max(2, int(5 * ui))] = (40, 170, 255)
    for k, t in enumerate(lines):
        cv2.putText(img, t, (padx, pady + tht + k * (tht + gap)), cv2.FONT_HERSHEY_SIMPLEX, fs, color, th,
                    cv2.LINE_AA)
    return img


class GLWindowSink:
    """OpenGL window (GLFW): texture upload + draw, ~1-2 ms per 1080p frame, with an in-video settings menu.

    Display settings come from `settings` (framegen.settings) and apply live; see SPECS there for what each
    does. vsync off presents immediately (lowest latency, may tear); on a G-Sync/FreeSync display vsync on lets
    VRR follow our frame timing instead of tearing. When the window is bigger than the frame it upscales: "fsr"
    with AMD FSR 1 (~0.1 ms GPU at 1440p), "vsr" with NVIDIA RTX Video Super Resolution (needs NVIDIA's package
    or the RTX Video SDK, see framegen/rtxvsr.py, and frames on the GPU), "bilinear" with a plain filtered blit.
    Sharpening (AMD's RCAS, framegen/fsr.py) goes on after any of them, or on the picture as it is, and a look
    (framegen/looks.py, drawn by framegen/looks_gl.py) after that, at the window's size.
    Keys: m/Tab = settings menu (arrows + Enter in it, Esc back a page), i = interpolation on/off for this session, s = stats
    overlay (off, minimal, detailed), f = fullscreen, u = next upscaler, k = look on/off, j = next look (Shift+J:
    the one before), l = delay test (PS5 Remote Play), r = save the instant replay, b = save a before & after video
    of it (framegen/replay.py), q/Esc = quit.

    restart_reasons(): labels of pending restart-only changes (the menu then offers "Restart now", which makes
    write() return "restart"). choices: {key: allowed values} for options the program can't use right now;
    unavailable: {key: reason} greys out a whole row. remote: a framegen.remote.Remote (the launcher's Settings
    page), polled with the window's own events so its changes apply like the menu's.

    hdr (settings "hdr" = HDR output, framegen/hdr.py): frames are HDR10 (float16 PQ BT.2020) and the window presents
    them through a 10-bit DXGI swap chain in HDR10's colour space, whatever "present" says; fullscreen is then a
    borderless window over the monitor. Looks and RTX VSR are off. If that swap chain can't be made (no HDR display,
    no NVIDIA interop), self.hdr10 is False and the window is the usual SDR one: the caller then sends SDR frames.

    With Windows' HDR on for the main display (KEEP_HDR), an SDR picture also goes through DXGI, an 8-bit swap chain,
    whatever "present" says, and fullscreen is a borderless window over the monitor: OpenGL's own fullscreen would
    switch the display to SDR while it's in front (self.keep_hdr; the "present" row says why it can't be changed).

    overlay_of (window capture, framegen/window.py): () -> (the app's picture area, its monitor) in screen pixels, or
    None while it's minimised. The window then sits over the app's picture (fullscreen: over its monitor), clicks
    and keys go through to the app, it never takes the focus, and it's left out of the capture. Settings change on
    the launcher's Settings page, since the keys go to the app.
    """

    NAMES = {"fsr": "FSR 1", "vsr": "RTX VSR", "bilinear": "bilinear"}

    # Shown when the window opens, and again when the mouse moves over it.
    KEYS_HINT = "M: menu    F: fullscreen    Esc: quit"
    fill, panorama = "fit", None  # Fill the screen (set from the settings in __init__)

    def __init__(self, w: int, h: int, settings=None, restart_reasons=None, choices: dict | None = None,
                 unavailable: dict | None = None, overlay_of=None, hdr: bool = False, hdr_peak: float = 1000.0,
                 window_keys: dict | None = None, clean_source: bool = False, monitor=None) -> None:
        import glfw
        from OpenGL import GL

        from .glsl import context_hints
        from .menu import Menu
        from .settings import Settings

        self.settings = st = settings if settings is not None else Settings()
        scaler, vsr_quality = st.scaler, st.vsr_quality
        self.glfw, self.GL, self.w, self.h = glfw, GL, w, h
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        if monitor is not None and overlay_of is None:
            self.target_monitor = monitor
        elif overlay_of is None and getattr(st, "monitor", "primary") != "primary":
            from .displays import resolve_monitor
            self.target_monitor = resolve_monitor(st.monitor)
        else:
            self.target_monitor = None
        self.dx = None  # a DxgiPresenter when the window presents through DXGI (_choose_presentation)
        self.overlay_of = overlay_of
        self.hdr10 = False  # presenting HDR10 (hdr asked for, and the swap chain could be made)
        self.keep_hdr = False  # presenting SDR through DXGI because Windows' HDR is on (KEEP_HDR)
        self.fs_style = FULLSCREEN_STYLE
        self.windowed_pos = None
        self._choose_presentation(w, h, hdr, hdr_peak)
        if self.dx is None:
            if overlay_of is not None:  # borderless, on top, click-through, and not focused when it opens
                for hint, value in ((glfw.DECORATED, glfw.FALSE), (glfw.FLOATING, glfw.TRUE),
                                    (glfw.FOCUSED, glfw.FALSE), (glfw.FOCUS_ON_SHOW, glfw.FALSE),
                                    (glfw.MOUSE_PASSTHROUGH, glfw.TRUE)):
                    glfw.window_hint(hint, value)
            else:
                glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            context_hints(glfw)  # OpenGL 4.1 core on a Mac, for the shaders
            ww, wh = _first_size(glfw, w, h)
            self.win = self.glwin = glfw.create_window(ww, wh, "SmoovPlay", None, None)
            glfw.default_window_hints()
            if overlay_of is None:
                self.windowed_pos = self._place_on_monitor(self.win, ww, wh)
                glfw.show_window(self.win)
            glfw.make_context_current(self.win)
        glfw.swap_interval(1 if st.vsync else 0)
        # Fullscreen stays up when another window takes the focus (a click on a second display, a notification), as
        # borderless fullscreen does in games. GLFW's default minimises it, which with DXGI could also stall Present.
        glfw.set_window_attrib(self.win, glfw.AUTO_ICONIFY, glfw.FALSE)
        self._set_icon(glfw)
        self.overlay, self.cmd, self.remote = st.overlay, None, None
        self.fill, self.panorama = st.fill, None  # Fill the screen (framegen/fill.py); Panorama's pass once used
        self._borderless = False  # fullscreen as a borderless window (_set_borderless)
        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_cursor_pos_callback(self.win, self._on_mouse)
        self._hint_next = 0.0  # moving the mouse shows the keys again, at most this often
        # settle(): in the first seconds chiaki-ng can show its window again (settle_hook, run.py, minimises it; later
        # too, each time the video window gets the keys back, as Alt+Tab to chiaki-ng's window restores it), and all
        # along a program can take the keys without the player doing anything (they're taken back).
        self.settle_hook = None
        self._was_front = True
        self._had_front = False  # the video window has had the keys since it opened (settle)
        self._take_tries = 0  # settle's tries at taking the keys the window opened without
        t = time.perf_counter()
        self._settle_until, self._settle_next, self._any_until = t + 10.0, 0.0, t + 15.0
        self._front_at: float | bool = t  # when the video window last had the keys (from when it opened); False: off
        self._keyed = False
        self._took_from: set = set()

        self.blitter = Blitter(GL)  # plain copies to the screen: glBlitFramebuffer, or a shader where it's refused
        self.tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE)  # FSR samples past edges
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)
        if self.hdr10:  # float16 PQ frames, as they are
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_BGR, GL.GL_HALF_FLOAT, None)
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE, None)
        GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
        self.fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex, 0)

        # Frames that are already on the GPU (CUDA tensors) are copied into the texture directly.
        self.interop = None
        renderer = GL.glGetString(GL.GL_RENDERER).decode(errors="replace")
        try:
            from .glcuda import CudaGLTexture
            self.interop = CudaGLTexture(int(self.tex), w, h, hdr=self.hdr10)
            print(f"[display] {renderer}: frames go to the screen straight from the GPU")
        except Exception as e:  # noqa: BLE001 - any failure just means the CPU upload path
            print(f"[display] {renderer}: GPU->screen copy unavailable ({e}); uploading frames from the CPU")
        self.gpu = self.interop is not None

        from .fsr import FsrScaler, GpuTimer
        self.fsr = self.vsr = self.vsr_out = None
        try:
            self.fsr = FsrScaler(GL, None, hdr=self.hdr10)  # (its sharpness is set for each frame: _sharpness)
        except Exception as e:  # noqa: BLE001 - shader trouble just means another scaler
            print(f"[display] FSR 1 unavailable ({str(e)[:200]})")
        self.vsr_why = ("it isn't used with HDR output yet" if self.hdr10 else
                        "it needs frames decoded on the GPU (the default for PS5 Remote Play)")
        self._vsr_args = None  # (w, h, clean) while RTX VSR can run but isn't made yet (_vsr_ready)
        if self.gpu and not self.hdr10:  # RTX VSR works on frames that are already CUDA tensors (8-bit ones)
            from .rtxvsr import unavailable_reason
            why = unavailable_reason()
            if why is None:
                self._vsr_args = (w, h, clean_source)
            self.vsr_why = why
            if why is not None and scaler == "vsr":
                print(f"[display] RTX VSR unavailable ({why})")
        self.modes = [m for m, ok in (("fsr", self.fsr), ("vsr", self._vsr_args), ("bilinear", True)) if ok]
        self.mode = scaler if scaler in self.modes else self.modes[0]
        if self.mode == "vsr" and not self._vsr_ready():
            self.mode = self.modes[0]
        details = {"fsr": "", "vsr": f" (quality {vsr_quality})", "bilinear": ""}
        print(f"[display] upscaling with {self._name(self.mode)}{details[self.mode]}, sharpening "
              f"{self._sharp_text()}; u cycles "
              + " / ".join(self._name(m) for m in self.modes) + "; m opens the settings menu")
        self.timer = GpuTimer(GL)  # GPU time of the scale/draw step, reported on close

        # Looks: which one is on (the settings, or the launcher's Looks page while it edits one) and its drawing.
        from .looks import LookState
        from .looks_gl import LookRenderer
        self.looks = LookState(st)
        self.look_gl = None
        if self.hdr10:  # looks are made for SDR pictures
            print("[display] Looks are off with HDR output")
        else:
            try:
                self.look_gl = LookRenderer(GL, self.looks.library.lut)
            except Exception as e:  # noqa: BLE001 - looks are optional
                print(f"[display] Looks unavailable ({str(e)[:200]})")
        self._frame_no = 0
        self._bake = None  # instant replay with "Looks in clips": (texture, framebuffer, CudaGLReader)
        self._bake_sharp = None  # and the sharpened frame the look is drawn from: (texture, framebuffer)
        self._bake_said = False

        self.choices, self.unavailable = dict(choices or {}), dict(unavailable or {})
        self.note_presentation()
        self.menu = Menu(st, allowed=self.allowed, disabled=self.disabled, restart_reasons=restart_reasons)
        self.menu_panel = _Panel(GL, self.hdr10, self.blitter)
        self.toast_panel = _Panel(GL, self.hdr10, self.blitter)
        # the stats overlay, Minimal or Detailed
        self.mini_panel, self._mini_state = _Panel(GL, self.hdr10, self.blitter), None
        self._detail_w, self._detail_ui = 0, None  # the detailed box's width so far, at this size (it only grows)
        self._menu_top = None  # (GL y of the menu's top, the picture's (y0, height) it was placed in) while it's open
        self._menu_state = self._toast_state = None
        self._toast_text, self._toast_until = "M: menu    F: fullscreen    Esc: quit", time.perf_counter() + 8.0
        self._waited = -1  # wait(): the whole second the "no picture" note last said (-1: no wait going on)
        self._unwatch = st.watch(self._apply)
        self._overlay_at, self._overlay_check = None, 0.0
        self.hotkeys, self._cursor_at = None, None
        if overlay_of is not None:
            self._make_overlay()
            self._start_hotkeys(window_keys)
        else:
            if st.fullscreen:
                self._set_fullscreen(True)
            self.take_focus()

    def take_focus(self) -> None:
        """The keys (M, R, K...) go to this window from the start, without a click on it first: the program in front
        when it opens is usually chiaki-ng, where the player just connected (framegen/winutil.bring_to_front). Not
        for window capture's overlay, which never takes the focus."""
        if self.overlay_of is not None or sys.platform != "win32":
            return
        from .winutil import bring_to_front
        try:
            if not bring_to_front(self.glfw.get_win32_window(self.win)):
                print("[display] couldn't bring the video window to the front; click it once so the keys go to it")
        except Exception as e:  # noqa: BLE001 - only the focus: the video goes on
            print(f"[display] couldn't bring the video window to the front ({str(e)[:160]})")

    # Programs that take the keys by themselves during play: chiaki-ng (its stream window), the launcher (minimising
    # itself) and the desktop (where the keys land when the window in front goes away). Any program in the first
    # seconds, when the window has just opened.
    TAKERS = ("chiaki.exe", "smoovplay.exe", "explorer.exe", "")

    def settle(self) -> None:
        """Four times a second: in the first seconds, settle_hook (run.py: minimise chiaki-ng again if it showed its
        window), and after them each time the video window is back in front (Alt+Tab to chiaki-ng's window restores
        it, and its unsmoothed picture would stay behind, or over, this one); and all along, the keys taken back for
        the video window when another program took them with no keyboard or mouse input from the player since the
        video window had them (the player switching on purpose, Alt+Tab or a click, is input, and left alone; a
        controller isn't, and switches nothing)."""
        now = time.perf_counter()
        if now < self._settle_next or self.overlay_of is not None:
            return
        self._settle_next = now + 0.25
        early = now < self._settle_until
        if early:
            self._run_settle_hook()
        if sys.platform != "win32" or self._front_at is False:
            return
        from .winutil import bring_to_front, foreground, ms_since_input
        try:
            ours = self.glfw.get_win32_window(self.win)
            hwnd, exe = foreground()
            back = hwnd == ours and not getattr(self, "_was_front", True)
            self._was_front = hwnd == ours
            if back and not early:
                self._run_settle_hook()
            if hwnd == ours:
                self._front_at, self._had_front = now, True
                return
            if not getattr(self, "_had_front", True) and now < self._any_until and exe in self.TAKERS:
                # Opened without the keys (Windows let the program in front keep them): take them, even if the mouse
                # moved meanwhile (moving it isn't choosing another window). A few tries, then the usual rule.
                self._take_tries = getattr(self, "_take_tries", 0) + 1
                if bring_to_front(ours, nudge=True):
                    self._front_at, self._had_front = now, True
                    print(f"[display] the video window opened behind {exe or 'no window'}: took the keys")
                elif self._take_tries >= 4:
                    self._had_front = True
                    print(f"[display] the video window opened behind {exe or 'no window'} and couldn't take the keys "
                          "(click it once)")
                return
            idle = ms_since_input()
            if idle is None or idle / 1000.0 < now - self._front_at:
                return  # the player did something since: theirs to choose
            if now >= self._any_until and exe not in self.TAKERS:
                return
            if bring_to_front(ours):
                self._front_at = now
                if exe not in self._took_from:
                    self._took_from.add(exe)
                    who = exe if exe in self.TAKERS[:3] else ("no window" if not exe else "another program")
                    print(f"[display] {who} took the keys: gave them back to the video window")
        except Exception as e:  # noqa: BLE001
            print(f"[display] couldn't check which window is in front ({str(e)[:160]})")
            self._front_at = False  # (stops checking)

    def take_keys(self) -> None:
        """Right after the window opens, while the program the player was in (chiaki-ng) is still in front: take the
        keys from it before it's minimised (run.py), which would leave no window in front to take them from."""
        if sys.platform != "win32" or self.overlay_of is not None:
            return
        from .winutil import bring_to_front
        try:
            if bring_to_front(self.glfw.get_win32_window(self.win), nudge=True):
                self._front_at, self._had_front = time.perf_counter(), True
        except Exception as e:  # noqa: BLE001 - settle() tries again
            print(f"[display] couldn't take the keys ({str(e)[:160]})")

    def _run_settle_hook(self) -> None:
        if self.settle_hook is None:
            return
        try:
            self.settle_hook()
        except Exception as e:  # noqa: BLE001 - only tidying up: the video goes on
            print(f"[display] {str(e)[:160]}")
            self.settle_hook = None

    def _on_mouse(self, win, x, y) -> None:
        """Moving the mouse over the video shows the keys for a moment (not over the menu or another message)."""
        now = time.perf_counter()
        if now >= self._hint_next and not self.menu.open and now >= self._toast_until:
            self._hint_next = now + 20.0
            self.toast(self.KEYS_HINT, 3.0 if self.hotkeys is None else 6.0)

    def _name(self, mode: str) -> str:
        if mode == "vsr":
            return f"RTX VSR q{self.vsr.quality if self.vsr is not None else self.settings.vsr_quality}"
        return self.NAMES[mode]

    def _vsr_ready(self) -> bool:
        """Make RTX VSR the first time it's picked, so a session that never uses it loads nothing for it (NVIDIA's
        model and its graphics memory). If it can't start, it comes off the list (False)."""
        if self.vsr is not None:
            return True
        if self._vsr_args is None:
            return False
        from .rtxvsr import make_vsr
        w, h, clean = self._vsr_args
        try:
            self.vsr = make_vsr(w, h, self.settings.vsr_quality, clean=clean)
        except Exception as e:  # noqa: BLE001 - VSR is optional
            self.vsr_why, self._vsr_args = str(e)[:200], None
            if "vsr" in self.modes:
                self.modes.remove("vsr")
            print(f"[display] RTX VSR unavailable ({self.vsr_why})")
            return False
        print(f"[display] RTX VSR {self.vsr.how}"
              + (": NVIDIA's models for clean video (a capture card's raw frames)" if clean else ""))
        return True

    # ---- presentation ----------------------------------------------------------
    def _set_icon(self, glfw) -> None:
        """SmoovPlay's logo in the title bar and on the taskbar instead of Python's: the plain SP at 16-32 px, the
        full logo at 48 and 64 (framegen/assets, from docs/brand). A missing icon never stops a session. Not on a Mac,
        where windows have no icons of their own and GLFW only warns. The Dock shows the program's icon there."""
        if sys.platform == "darwin":
            return
        try:
            from pathlib import Path

            from .imagefile import read_image
            images = []
            for n in (16, 24, 32, 48, 64):
                bgra = read_image(Path(__file__).with_name("assets") / f"icon_{n}.png", cv2.IMREAD_UNCHANGED)
                if bgra is not None and bgra.shape == (n, n, 4):
                    images.append((n, n, bgra[..., [2, 1, 0, 3]]))
            if images:
                glfw.set_window_icon(self.win, len(images), images)
        except Exception as e:
            print(f"[display] window icon not set: {e}")

    def _choose_presentation(self, w: int, h: int, hdr: bool, hdr_peak: float) -> None:
        """How frames reach the screen: self.dx, a DXGI swap chain, or None for OpenGL's own; and how fullscreen is
        done (self.fs_style). DXGI for HDR output (hdr), when the chosen display has Windows' HDR on (KEEP_HDR), and
        for settings "present" = Low delay in a window. Window capture's overlay always draws with OpenGL."""
        st = self.settings
        if self.overlay_of is None:
            mon = getattr(self, "target_monitor", None)
            device = mon.device if mon is not None else None
            hdr_on = KEEP_HDR and (windows_hdr_on(device) if device else windows_hdr_on())
            if hdr:
                self._open_dxgi(w, h, hdr=True, peak=hdr_peak)
                self.hdr10 = self.dx is not None
            if self.dx is None and hdr_on:  # (also when HDR output's 10-bit swap chain couldn't be made: SDR, then)
                self._open_dxgi(w, h, keep_hdr=True)
                self.keep_hdr = self.dx is not None
            if self.dx is not None:
                # Fullscreen with this swap chain is a borderless window: GLFW's fullscreen could stall it.
                self.fs_style = "borderless" if FULLSCREEN_STYLE == "none" else FULLSCREEN_STYLE
                return
            if hdr_on:
                return  # (it said why; Low delay would fail the same way)
        if st.present != "dxgi":
            return  # the standard presentation
        if self.overlay_of is not None:
            print("[display] window capture: standard presentation (the overlay draws with OpenGL)")
        elif not st.fullscreen or FULLSCREEN_STYLE != "none":
            self._open_dxgi(w, h)
        else:
            print("[display] fullscreen: standard presentation (Low delay is for windowed play; fullscreen already "
                  "skips the compositor)")

    def note_presentation(self) -> None:
        """The "present" row can't be changed while Windows' HDR keeps the window on DXGI (it overrides HDR output's
        reason: keep_hdr is HDR output's fallback too). run.py puts it back if it drops HDR output's rows."""
        if self.keep_hdr:
            self.unavailable["present"] = KEEP_HDR_REASON

    def _open_dxgi(self, w: int, h: int, hdr: bool = False, peak: float = 1000.0, keep_hdr: bool = False) -> None:
        """Settings "present" = "dxgi" (Experimental): the video window gets a DXGI flip-model swap chain and no OpenGL
        context of its own (GLFW NO_API); OpenGL draws on a hidden window (framegen/dxgi_present.py). In a window that
        skips the compositor's copy: less delay, and G-Sync/FreeSync work. Leaves self.dx None (and no windows) if
        any of it fails, so the caller opens the usual OpenGL window. hdr: a 10-bit HDR10 swap chain (HDR output).
        keep_hdr: an 8-bit one because Windows' HDR is on (KEEP_HDR); only what's logged differs. DXGI is Windows'
        own, so on other systems this fails before any window opens, and says so."""
        glfw = self.glfw
        win = glwin = None
        try:
            if sys.platform != "win32":
                raise RuntimeError("DXGI is Windows-only")
            from .dxgi_present import DxgiPresenter
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            glwin = glfw.create_window(64, 64, "framegen (OpenGL)", None, None)
            glfw.default_window_hints()
            if not glwin:
                raise RuntimeError("couldn't open an OpenGL context")
            glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
            glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
            win = glfw.create_window(min(w, 1600), min(h, 900), "SmoovPlay", None, None)
            glfw.default_window_hints()
            if not win:
                raise RuntimeError("couldn't open the window")
            self.windowed_pos = self._place_on_monitor(win, min(w, 1600), min(h, 900))
            glfw.show_window(win)
            glfw.make_context_current(glwin)
            fw, fh = glfw.get_framebuffer_size(win)
            self.dx = DxgiPresenter(self.GL, glfw.get_win32_window(win), fw, fh, hdr=hdr, peak_nits=peak)
            self.win, self.glwin = win, glwin
            from .dxgi_present import BUFFERS
            test = ("copy", 2) != (self.dx.mode, BUFFERS) or FULLSCREEN_STYLE != "none"
            print("[display] window presentation: DXGI flip model"
                  + (", HDR10 (10-bit, PQ, BT.2020)" if hdr else "")
                  + ((", tearing allowed with VSync off" + (" in fullscreen" if keep_hdr else "")) if self.dx.tearing
                     else "")
                  + (", as Windows' HDR is on: fullscreen keeps it on (FRAMEGEN_KEEP_HDR=0: OpenGL's own)" if keep_hdr
                     else "") + " (Experimental)"
                  + (f" [testing: {self.dx.mode}, {BUFFERS} buffers, fullscreen {FULLSCREEN_STYLE}]" if test else ""))
        except Exception as e:  # noqa: BLE001 - any failure: OpenGL's own presentation
            glfw.default_window_hints()
            glfw.make_context_current(None)
            for wnd in (win, glwin):
                if wnd:
                    glfw.destroy_window(wnd)
            self.dx = None
            self.windowed_pos = None
            if hdr:
                print(f"[display] HDR output unavailable ({str(e)[:200]}); showing HDR in SDR")
            elif keep_hdr:
                print(f"[display] Windows' HDR is on, but DXGI presentation is unavailable ({str(e)[:200]}); using "
                      "OpenGL's own. Fullscreen then switches the display to SDR while it's in front, which can change "
                      "the colours; a window keeps HDR on")
            else:
                print(f"[display] low-delay window presentation unavailable ({str(e)[:200]}); using OpenGL's own")

    def _target(self) -> int:
        """The framebuffer the picture is drawn into: the window's own, or DXGI's offscreen one."""
        return self.dx.fbo if self.dx is not None else 0

    def _may_tear(self) -> bool:
        """Whether VSync off may tear. Fullscreen, and a window with Low delay picked, as before. Not a window that's
        on DXGI only because Windows' HDR is on (KEEP_HDR): with the standard presentation it never tore."""
        return not self.keep_hdr or self._borderless or self.settings.present == "dxgi"

    def _present(self) -> None:
        if self.dx is not None:
            if self.glfw.get_window_attrib(self.win, self.glfw.ICONIFIED):
                return  # minimised: nothing to show, and a flip-model Present can wait for a window nobody sees
            t0 = time.perf_counter()
            self.dx.present(self.settings.vsync, tear=self._may_tear())
            _say_if_slow("DXGI Present", t0)
        else:
            self.glfw.swap_buffers(self.win)

    # ---- settings ------------------------------------------------------------
    def allowed(self, key: str, value) -> bool:
        if key == "scaler":
            return value in self.modes
        return value in self.choices.get(key, (value,))

    def disabled(self, key: str) -> str | None:
        if key == "vsr_quality" and self.vsr is None and self._vsr_args is None:
            return f"RTX VSR isn't available: {self.vsr_why}."
        if key == "sharpening" and self.fsr is None:
            return "Sharpening uses FSR 1's shaders, which couldn't start on this graphics card."
        return self.unavailable.get(key)

    def _apply(self, key: str) -> None:
        """A setting changed (menu, hotkey or elsewhere): apply the ones the window owns."""
        st, glfw = self.settings, self.glfw
        if key == "scaler" and st.scaler in self.modes:
            if st.scaler != "vsr" or self._vsr_ready():
                self.mode = st.scaler
            else:
                self.toast(f"RTX VSR couldn't start: {self.vsr_why}", 5.0)
        elif key == "vsr_quality" and self.vsr is not None:
            self.vsr.quality = st.vsr_quality
        elif key == "overlay":
            self.overlay = st.overlay
        elif key == "fill":
            self.fill = st.fill
            if not self.menu.open:  # a hotkey: say what it did, and that it's the same picture
                from .settings import BY_KEY
                self.toast(f"Fill the screen: {BY_KEY['fill'].name(st.fill)}"
                           + ("" if st.fill == "fit" else "\nThe same 16:9 picture: consoles send 16:9 only"), 3.0)
            return
        elif key == "fullscreen":
            if self.dx is not None and st.fullscreen and self.fs_style == "none":
                self.toast("Fullscreen starts next time you press Play: Low delay works in a window only", 5.0)
                return
            self._set_fullscreen(st.fullscreen)
            if st.fullscreen and self.dx is None and self.overlay_of is None and KEEP_HDR:
                device = None
                if sys.platform == "win32" and hasattr(glfw, "get_win32_adapter"):
                    try:
                        device = glfw.get_win32_adapter(self._monitor_under_window()) or None
                    except Exception:  # noqa: BLE001
                        device = None
                if not device:
                    mon = getattr(self, "target_monitor", None)
                    device = mon.device if mon is not None else None
                if windows_hdr_on(device) if device else windows_hdr_on():
                    # Windows' HDR was turned on after the window opened (with OpenGL's own presentation)
                    self.toast("Windows' HDR is on now: fullscreen switches the display to SDR until you press Play again",
                               6.0)
                    return
        elif key == "monitor":
            if not self.menu.open:
                from .settings import BY_KEY
                self.toast(f"Monitor: {BY_KEY['monitor'].name(st.monitor)} (next Play)", 4.0)
            return
        elif key == "vsync":
            glfw.swap_interval(1 if st.vsync else 0)
        elif key in ("look", "look_on"):
            if not self.menu.open:
                self.toast(f"Look: {self.looks.name() if st.look_on else 'Off'}")
            return
        if not self.menu.open:  # a hotkey: say what it did
            from .settings import BY_KEY
            spec = BY_KEY[key]
            self.toast(f"{spec.label}: {spec.name(getattr(st, key))}")

    def _make_overlay(self) -> None:
        """Window capture: no focus ever (WS_EX_NOACTIVATE: the app keeps it, and some apps stop drawing without it),
        and out of the capture, which then sees the app underneath (framegen/window.py)."""
        import ctypes

        from .window import exclude_from_capture
        hwnd = self.glfw.get_win32_window(self.win)
        if not exclude_from_capture(hwnd):
            print("[display] couldn't leave this window out of the capture (that needs Windows 10 version 2004 or "
                  "newer); it would capture itself. Put the app's window on another monitor.")
        u32 = ctypes.windll.user32
        u32.GetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int]
        u32.SetWindowLongW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_long]
        u32.SetWindowLongW(hwnd, -20, u32.GetWindowLongW(hwnd, -20) | 0x08000000)  # GWL_EXSTYLE |= WS_EX_NOACTIVATE
        self._follow()

    def _start_hotkeys(self, picked: dict | None) -> None:
        """Window capture: the overlay never has the keys, so its own come through system-wide shortcuts
        (framegen/hotkeys.py), registered for this thread, the one that draws (_events takes their messages). Says
        which it got, over the picture and to the launcher ("[window-keys] ...")."""
        from .hotkeys import Hotkeys, describe
        try:
            self.hotkeys = Hotkeys(picked)
        except Exception as e:  # noqa: BLE001 - no shortcuts: the launcher's Stop and Settings page still work
            print(f"[display] no shortcuts ({str(e)[:120]}): Stop and Settings on SmoovPlay's Play page")
        keys = describe(self.hotkeys.got) if self.hotkeys is not None else ""
        if self.hotkeys is not None:
            print(self.hotkeys.report(), flush=True)
            for name in self.hotkeys.taken:
                print(f"[display] {name} is taken by another program")
        self.KEYS_HINT = ("SmoovPlay's keys here (other keys go to the app):\n" + keys if keys
                          else "Keys go to the app: use Stop and Settings on SmoovPlay's Play page")
        self.toast(self.KEYS_HINT, 12.0)

    # the video window's key each menu key stands for
    _HOTKEY_GLFW = {"Up": "UP", "Down": "DOWN", "Left": "LEFT", "Right": "RIGHT", "Enter": "ENTER", "Esc": "ESCAPE",
                    "Backspace": "BACKSPACE"}

    def _events(self) -> None:
        """glfw.poll_events, after the shortcuts' messages (window capture): they come to the thread, not a window,
        and GLFW would drop them. Each works as its key does in the video window's own (M, F, Q...); the menu's keys
        are taken only while it's open."""
        if self.hotkeys is not None:
            from .hotkeys import ACTIONS
            self.hotkeys.set_nav(self.menu.open)
            for what in self.hotkeys.poll():
                key = self._HOTKEY_GLFW[what[4:]] if what.startswith("nav:") else ACTIONS[what][1]
                self._on_key(self.win, getattr(self.glfw, "KEY_" + key), 0, self.glfw.PRESS, 0)
                self.hotkeys.set_nav(self.menu.open)
        self.glfw.poll_events()

    def _hint_on_hover(self, r) -> None:
        """Window capture: the overlay gets no mouse events (clicks go to the app), so the pointer is looked at here
        (from _follow): moving it over the picture shows the keys again, at most every 20 s."""
        import ctypes
        from ctypes import wintypes as wt
        pt = wt.POINT()
        if not ctypes.windll.user32.GetCursorPos(ctypes.byref(pt)):
            return
        at, moved = (pt.x, pt.y), self._cursor_at is not None and (pt.x, pt.y) != self._cursor_at
        self._cursor_at = at
        if moved and r.x <= pt.x < r.x + r.w and r.y <= pt.y < r.y + r.h:
            self._on_mouse(self.win, 0, 0)

    def _follow(self) -> None:
        """Window capture: keeps the overlay over the app's picture (fullscreen: its monitor), four times a second."""
        now = time.perf_counter()
        if now < self._overlay_check:
            return
        self._overlay_check = now + 0.25
        where = self.overlay_of()
        if where is None:  # minimised: stay put until it's back
            return
        client, monitor = where
        r = monitor if self.settings.fullscreen else client
        if self.hotkeys is not None:
            self._hint_on_hover(r)
        at = (r.x, r.y, r.w, r.h)
        if at != self._overlay_at:
            self._overlay_at = at
            self.glfw.set_window_pos(self.win, r.x, r.y)
            self.glfw.set_window_size(self.win, r.w, r.h)

    def _glfw_monitor(self):
        """The GLFW monitor matching `target_monitor`, else GLFW's primary monitor."""
        glfw = self.glfw
        mon = getattr(self, "target_monitor", None)
        if mon is None:
            return glfw.get_primary_monitor()
        monitors = glfw.get_monitors() or []
        if sys.platform == "win32" and hasattr(glfw, "get_win32_adapter"):
            want = mon.device.lower()
            for m in monitors:
                try:
                    if (glfw.get_win32_adapter(m) or "").lower() == want:
                        return m
                except Exception:  # noqa: BLE001
                    pass
        for m in monitors:
            try:
                mx, my = glfw.get_monitor_pos(m)
                mode = glfw.get_video_mode(m)
                if (mx, my, mode.size.width, mode.size.height) == (mon.x, mon.y, mon.width, mon.height):
                    return m
            except Exception:  # noqa: BLE001
                pass
        for m in monitors:
            try:
                if glfw.get_monitor_pos(m) == (mon.x, mon.y):
                    return m
            except Exception:  # noqa: BLE001
                pass
        return glfw.get_primary_monitor()

    def _place_on_monitor(self, win, ww: int, wh: int) -> tuple[int, int, int, int]:
        """Places a windowed GL/DXGI window centred on the chosen monitor (moving before sizing)."""
        glfw = self.glfw
        mon = self._glfw_monitor()
        ax = ay = aw = ah = 0
        if mon is not None:
            if hasattr(glfw, "get_monitor_workarea"):
                ax, ay, aw, ah = glfw.get_monitor_workarea(mon)
            if aw <= 0 or ah <= 0:
                ax, ay = glfw.get_monitor_pos(mon)
                mode = glfw.get_video_mode(mon)
                aw, ah = mode.size.width, mode.size.height
        if aw > 0 and ah > 0:
            ww = min(ww, max(320, aw - 80))
            wh = min(wh, max(180, ah - 80))
            wx = ax + max(0, (aw - ww) // 2)
            wy = ay + max(40 if ah > wh + 40 else 0, (ah - wh) // 2)
        else:
            wx, wy = 80, 80
        glfw.set_window_pos(win, wx, wy)
        glfw.set_window_size(win, ww, wh)
        return (wx, wy, ww, wh)

    def _set_fullscreen(self, on: bool) -> None:
        glfw, win = self.glfw, self.win
        if self.overlay_of is not None:  # over the app's monitor, as a borderless window: no fullscreen mode change
            self._overlay_at, self._overlay_check = None, 0.0
            self._follow()
            return
        if self.fs_style == "borderless" or (self.dx is None and self.fs_style != "monitor"):
            self._set_borderless(on)
            return
        if bool(glfw.get_window_monitor(win)) == on:
            return
        t0 = time.perf_counter()
        if not on:
            x, y, ww, wh = self.windowed_pos or (80, 80, min(self.w, 1600), min(self.h, 900))
            glfw.set_window_monitor(win, None, x, y, ww, wh, 0)
        else:
            self.windowed_pos = (*glfw.get_window_pos(win), *glfw.get_window_size(win))
            mon = self._monitor_under_window()
            mode = glfw.get_video_mode(mon)
            glfw.set_window_monitor(win, mon, 0, 0, mode.size.width, mode.size.height, mode.refresh_rate)
        _say_if_slow("fullscreen " + ("on" if on else "off"), t0)
        glfw.swap_interval(1 if self.settings.vsync else 0)

    def _monitor_under_window(self):
        """Where fullscreen goes: the monitor the window is mostly on, so F fills the screen it was moved to rather
        than the primary one."""
        glfw = self.glfw
        monitors = []
        for mon in glfw.get_monitors() or []:
            mode = glfw.get_video_mode(mon)
            if mode is not None:
                monitors.append((mon, *glfw.get_monitor_pos(mon), mode.size.width, mode.size.height))
        if not monitors:
            return glfw.get_primary_monitor()
        return window_monitor((*glfw.get_window_pos(self.win), *glfw.get_window_size(self.win)), monitors)

    def _set_borderless(self, on: bool) -> None:
        """Fullscreen as a borderless window over the monitor, as games' borderless mode (the standard presentation's
        fullscreen, and HDR output's): no GLFW fullscreen monitor, so no always-on-top window (Alt+Tab to another
        program puts that one in front) and no video mode request. Windows tucks the taskbar under it while it's in
        front. On a Mac the menu bar and the Dock are hidden while SmoovPlay is in front (_mac_hide_bars)."""
        glfw, win = self.glfw, self.win
        if self._borderless == on:
            return
        t0 = time.perf_counter()
        if on:
            self.windowed_pos = (*glfw.get_window_pos(win), *glfw.get_window_size(win))
            mon = self._monitor_under_window()
            mx, my = glfw.get_monitor_pos(mon)
            mode = glfw.get_video_mode(mon)
            if sys.platform == "darwin":  # first, or macOS keeps the window under the menu bar
                _mac_hide_bars(True)
            glfw.set_window_attrib(win, glfw.DECORATED, glfw.FALSE)
            glfw.set_window_pos(win, mx, my)
            glfw.set_window_size(win, mode.size.width, mode.size.height)
        else:
            x, y, ww, wh = self.windowed_pos or (80, 80, min(self.w, 1600), min(self.h, 900))
            if sys.platform == "darwin":
                _mac_hide_bars(False)
            glfw.set_window_attrib(win, glfw.DECORATED, glfw.TRUE)
            glfw.set_window_pos(win, x, y)
            glfw.set_window_size(win, ww, wh)
        self._borderless = on
        _say_if_slow("borderless fullscreen " + ("on" if on else "off"), t0)

    def toast(self, text: str, seconds: float = 2.0) -> None:
        """A short message in the top-right corner."""
        self._toast_text, self._toast_until = text, time.perf_counter() + seconds

    _NAV = {"UP": "up", "DOWN": "down", "LEFT": "left", "RIGHT": "right", "ENTER": "enter", "KP_ENTER": "enter",
            "BACKSPACE": "back", "ESCAPE": "back"}  # Esc: back a page, and on the main page it closes the menu

    def _on_key(self, win, key, scancode, action, mods) -> None:
        glfw, st = self.glfw, self.settings
        nav = {getattr(glfw, "KEY_" + k): v for k, v in self._NAV.items()}
        if action == glfw.REPEAT and not (self.menu.open and key in nav):
            return
        if action not in (glfw.PRESS, glfw.REPEAT):
            return
        self._keyed = True  # the player has pressed a key here
        if self.menu.open and key in nav:
            r = self.menu.key(nav[key])
            if r == "close":
                self.menu.open = False
            elif r == "restart":
                self.menu.open = False
                self.cmd = "restart"
            return
        if key in (glfw.KEY_M, glfw.KEY_TAB):
            self.menu.open = not self.menu.open
            self._toast_until = 0.0
        elif key in (glfw.KEY_Q, glfw.KEY_ESCAPE):
            self.cmd = "quit"
        elif key == glfw.KEY_I:
            st.override("interp", not st.interp)  # a quick A/B, not saved
        elif key == glfw.KEY_S:  # Off, Minimal, Detailed, Off...
            if not st.overlay:
                st.set("overlay_style", "minimal")
                st.set("overlay", True)
            elif st.overlay_style == "minimal":
                st.set("overlay_style", "detailed")
            else:
                st.set("overlay", False)
        elif key == glfw.KEY_U:
            st.step("scaler", 1, self.allowed)
        elif key == glfw.KEY_Z:
            st.step("fill", 1, self.allowed)
        elif key == glfw.KEY_K:
            self.looks.toggle()
        elif key == glfw.KEY_J:
            self.looks.next(-1 if mods & glfw.MOD_SHIFT else 1)
        elif key == glfw.KEY_F:
            st.set("fullscreen", not st.fullscreen)
        elif key == glfw.KEY_L:
            self.cmd = "delay_test"  # the pipeline runs it (PS5 Remote Play only)
        elif key == glfw.KEY_R:
            self.cmd = "save_replay"  # instant replay (framegen/replay.py), through the pipeline
        elif key == glfw.KEY_B:
            self.cmd = "save_compare"

    # ---- drawing -------------------------------------------------------------
    def _layout(self) -> tuple[int, int, float, int, int, int, int]:
        fw, fh = self.glfw.get_framebuffer_size(self.win)
        if self.dx is not None:
            self.dx.resize(fw, fh)  # cheap when the size hasn't changed
        s = min(fw / self.w, fh / self.h)  # letterbox, keep aspect
        if self.overlay_of is not None:  # window capture: the frame is the app's window squeezed to the stream's
            return fw, fh, s, fw, fh, 0, 0  # size, so it's stretched back over it to fill the overlay exactly
        dw, dh, x0, y0 = layout(self.fill, fw, fh, self.w, self.h)  # (Fill the screen: always the same 16:9 picture)
        return fw, fh, s, dw, dh, x0, y0

    def _panorama_on(self, fw: int, fh: int) -> bool:
        return self.fill == "panorama" and self.overlay_of is None and wider(fw, fh, self.w, self.h)

    def _ui_rect(self, fw: int, fh: int, dw: int, dh: int, x0: int, y0: int) -> tuple[int, int, int, int]:
        """Where the menu, messages and stats go: the picture as seen on the screen."""
        return (0, 0, fw, fh) if self._panorama_on(fw, fh) else visible(dw, dh, x0, y0, fw, fh)

    def _draw_ui(self, x0: int, y0: int, dw: int, dh: int) -> None:
        """Menu and messages, on top of everything else. Images are redrawn only when they change."""
        ui = max(0.5, dh / 1080)
        if not self.menu.open:
            self._menu_top = None
        if self.menu.open:
            state = (self.menu.state(), round(ui, 3))
            if state != self._menu_state:
                self.menu_panel.update(self.menu.render(ui))
                self._menu_state = state
            p = self.menu_panel
            # Centred when it opens, then its top stays put (a row appearing, such as Restart now, grows it
            # downwards), so the rows never move under the highlight. Inside the picture whatever happens.
            where = (y0, dh)
            if self._menu_top is None or self._menu_top[1] != where:
                self._menu_top = (y0 + (dh + p.h) // 2, where)
            top = min(y0 + dh, max(self._menu_top[0], y0 + p.h))
            self.menu_panel.blit(x0 + int(32 * ui), top)
        elif time.perf_counter() < self._toast_until:
            state = (self._toast_text, round(ui, 3))
            if state != self._toast_state:
                self.toast_panel.update(_pill(self._toast_text, ui))
                self._toast_state = state
            p = self.toast_panel
            m = int(16 * ui)
            below = 0  # the stats overlay in the top right: messages go under it
            if self.overlay and getattr(self.settings, "overlay_corner", "") == "top_right" and self._mini_state:
                below = self.mini_panel.h + max(2, int(4 * self._stats_ui(dh)))
            p.blit(x0 + dw - p.w - m, y0 + dh - m - below)

    def busy(self, fn, text: str):
        """Run fn() on a worker thread while the window stays responsive, showing the last frame and `text`.
        Returns fn()'s result (or raises its exception). Keys pressed meanwhile (e.g. q) take effect after."""
        GL, glfw = self.GL, self.glfw
        out: list = []

        def work() -> None:
            try:
                out.append((True, fn()))
            except BaseException as e:  # noqa: BLE001 - handed back to the caller
                out.append((False, e))

        self.menu.open = False
        worker = threading.Thread(target=work, daemon=True)
        worker.start()
        t0 = time.perf_counter()
        while worker.is_alive():
            fw, fh, s, dw, dh, x0, y0 = self._layout()
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._target())
            GL.glViewport(0, 0, fw, fh)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self._plain(x0, y0, dw, dh)
            dots = "." * (1 + int(time.perf_counter() - t0) % 3)
            self.toast(f"{text}{dots:<3}", 1.0)
            self._draw_ui(*visible(dw, dh, x0, y0, fw, fh))
            self._present()
            self._events()
            if self.remote is not None and self.remote.poll() == "quit":  # live settings still apply; a
                self.cmd = "quit"  # restart is already under way, but a Stop is kept for when it's done
            worker.join(0.05)
        self._toast_until = 0.0
        ok, value = out[0]
        if not ok:
            raise value
        return value

    def wait(self, w: Waiting) -> str | None:
        """No new frame for a moment (framegen/io.Waiting): the window keeps answering Windows, its keys and the
        launcher, and from 2 s on says so over the last picture. Returns "quit" or "restart" when asked for; any other
        key waits for the next frame (write() hands it on)."""
        glfw = self.glfw
        n = int(w.since)
        if n >= 2 and (n != self._waited or self.menu.open) and self.overlay_of is None:
            self._waited = n
            self.toast(f"No picture from {w.what} for {n} s. Esc quits" + (f"\n{w.why}" if w.why else ""), 1.5)
            fw, fh, s, dw, dh, x0, y0 = self._layout()
            GL = self.GL
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self._target())
            GL.glViewport(0, 0, fw, fh)
            GL.glClear(GL.GL_COLOR_BUFFER_BIT)
            self._plain(x0, y0, dw, dh)  # the last frame, plain (no upscaler or look)
            self._draw_ui(*visible(dw, dh, x0, y0, fw, fh))
            self._present()
        self.settle()
        self._events()
        if self.remote is not None:
            r = self.remote.poll()  # the launcher's Stop sends quit
            if r == "quit" or (r == "restart" and self.cmd is None):
                self.cmd = r
        if glfw.window_should_close(self.win):
            return "quit"
        if self.cmd in ("quit", "restart"):
            cmd, self.cmd = self.cmd, None
            return cmd
        return None

    def write(self, frame, text: str | None = None) -> str | None:
        """frame: HxWx3 uint8 BGR, a numpy array or a CUDA tensor."""
        GL, glfw = self.GL, self.glfw
        if self._waited >= 0:  # a picture again after a wait: its note goes
            self._waited, self._toast_until = -1, 0.0
        if self.overlay_of is not None:
            self._follow()
        on_gpu = _is_cuda(frame) and self.interop is not None
        if on_gpu:
            self.interop.upload(frame)
        else:
            GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
            host = np.ascontiguousarray(_host(frame))
            GL.glTexSubImage2D(GL.GL_TEXTURE_2D, 0, 0, 0, self.w, self.h, GL.GL_BGR,
                               GL.GL_HALF_FLOAT if host.dtype == np.float16 else GL.GL_UNSIGNED_BYTE, host)
        fw, fh, s, dw, dh, x0, y0 = self._layout()
        target = screen = self._target()
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)
        ux, uy, uw, uh = self._ui_rect(fw, fh, dw, dh, x0, y0)
        pano = self._panorama_on(fw, fh)
        if pano:  # Panorama: the picture is drawn as for Fit (to the screen's height) into its texture, then bent
            if self.panorama is None:  # across the whole screen below
                self.panorama = Panorama(GL)
            target, x0, y0 = self.panorama.target(dw, dh), 0, 0
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, target)
        upscale = dw > self.w or dh > self.h
        mode = self.mode
        if mode == "vsr" and not on_gpu:  # a CPU frame: VSR needs it on the GPU
            mode = "fsr" if self.fsr is not None else "bilinear"
        fsr = upscale and mode == "fsr"
        sharp = self._sharpness(fsr)  # RCAS for this frame: FSR 1's own pass, or on its own after the others
        label = self._name(mode) if upscale else "no upscale"
        if sharp is not None and not fsr:
            label += " + sharpening"
        elif fsr and sharp is None:
            label += " (no sharpening)"
        self._frame_no += 1
        plan = self.look_gl.plan(self.looks.current(), self.looks.rev) if self.look_gl is not None else None
        q = self.timer.begin()
        if fsr:
            self.fsr.sharpness = sharp
        if plan is not None:  # the upscaler draws into the look's input, and the look onto the screen
            if fsr:
                src, fbo = self.look_gl.input_target(dw, dh)
                self.fsr.draw(self.tex, self.w, self.h, fbo, 0, 0, dw, dh)
                flip_in = True  # (FSR draws upright)
            elif sharp is not None:  # RCAS draws into the look's input, upright like FSR
                sized = self._vsr_render(dw, dh)[0] if upscale and mode == "vsr" else self._sized(dw, dh)
                src, fbo = self.look_gl.input_target(dw, dh)
                self.fsr.sharpen(sized, fbo, 0, 0, dw, dh, sharp)
                flip_in = True
            elif upscale and mode == "vsr":
                src, flip_in = self._vsr_render(dw, dh)[0], False
            else:
                src, flip_in = self.tex, False  # the look's first pass resizes it
            self.look_gl.draw(plan, src, flip_in, target, x0, y0, dw, dh, True, self._frame_no)
            GL.glViewport(0, 0, fw, fh)
            label += " + look"
        elif fsr:
            self.fsr.draw(self.tex, self.w, self.h, target, x0, y0, dw, dh)
            GL.glViewport(0, 0, fw, fh)
        elif sharp is not None:
            sized = self._vsr_render(dw, dh)[0] if upscale and mode == "vsr" else self._sized(dw, dh)
            self.fsr.sharpen(sized, target, x0, y0, dw, dh, sharp)
            GL.glViewport(0, 0, fw, fh)
        elif upscale and mode == "vsr":
            self._vsr_draw(x0, y0, dw, dh)
        else:
            self._plain(x0, y0, dw, dh)
        if pano:
            self.panorama.draw(screen, fw, fh, panorama_t(fw, fh, self.w, self.h))
            label += " + panorama"
        self.timer.end(q, f"{label} {dw}x{dh}")
        if self.overlay and isinstance(text, MiniText):
            self._draw_stats(text, ux, uy, uw, uh)
        elif self.overlay and text:
            first, nl, rest = text.partition("\n")  # the upscaler goes on the first line; a second is the test line
            self._draw_stats(f"{first}  {label}{nl}{rest}", ux, uy, uw, uh)
        self._draw_ui(ux, uy, uw, uh)
        self._present()
        self.settle()
        self._events()
        if self.remote is not None:
            r = self.remote.poll()  # the launcher's Stop sends quit
            if r == "quit" or (r == "restart" and self.cmd is None):
                self.cmd = r
        if glfw.window_should_close(self.win):
            return "quit"
        cmd, self.cmd = self.cmd, None
        return cmd

    def _sharpness(self, fsr: bool) -> float | None:
        """RCAS's sharpness for a frame (Settings "Sharpening", in stops below its maximum), or None for none. fsr:
        FSR 1 is enlarging this one. Carried over from FSR sharpening (0.3.14 and before), it applies only then, as
        it did, until the player changes Sharpening (sharpen_fsr_only)."""
        from .settings import SHARPEN_STOPS
        st = self.settings
        stops = SHARPEN_STOPS.get(st.sharpening)
        if stops is None or self.fsr is None or (st.sharpen_fsr_only and not fsr):
            return None
        return stops

    def _sharp_text(self) -> str:
        """Sharpening for the log: "Medium", "Medium, only while FSR 1 enlarges (as before 0.3.15)", "Off"."""
        from .settings import BY_KEY
        st = self.settings
        name = BY_KEY["sharpening"].name(st.sharpening)
        if self.fsr is None:
            return "unavailable (FSR 1's shaders couldn't start)"
        if st.sharpen_fsr_only and st.sharpening != "off":
            return name + ", only while FSR 1 enlarges (as before 0.3.15)"
        return name

    def _sized(self, dw: int, dh: int) -> int:
        """The frame at dw x dh (row 0 = image top) for sharpening on its own: its texture as it is at the stream's
        size, otherwise a bilinear copy into FSR's stage texture (as bilinear, or no upscale, draws it)."""
        if (dw, dh) == (self.w, self.h):
            return self.tex
        tex, fbo = self.fsr.stage(dw, dh)
        self.GL.glBindFramebuffer(self.GL.GL_DRAW_FRAMEBUFFER, fbo)
        self.blitter.copy(self.tex, self.w, self.h, self.fbo, (0, 0, self.w, self.h), (0, 0, dw, dh), True)
        return tex

    def _plain(self, x0: int, y0: int, dw: int, dh: int) -> None:
        """The frame's texture into the picture's rectangle with no upscaler: a bilinear copy (blit) to the bound
        draw framebuffer. Image row 0 is the top and GL's row 0 the bottom, so the destination's Y is flipped."""
        self.blitter.copy(self.tex, self.w, self.h, self.fbo, (0, 0, self.w, self.h), (x0, y0 + dh, x0 + dw, y0),
                          True)

    def _vsr_draw(self, x0: int, y0: int, dw: int, dh: int) -> None:
        """RTX VSR from the staged RGBA frame straight into a window-sized GL texture, then a 1:1 blit."""
        tex, fbo = self._vsr_render(dw, dh)
        self.blitter.copy(tex, dw, dh, fbo, (0, 0, dw, dh), (x0, y0 + dh, x0 + dw, y0), False)

    def _vsr_render(self, dw: int, dh: int) -> tuple[int, int]:
        """RTX VSR into the window-sized texture (row 0 = the image's top); returns (texture, framebuffer)."""
        if self.vsr_out is None or self.vsr_out[3:5] != (dw, dh):
            self._vsr_target(dw, dh)
        tex, fbo, img = self.vsr_out[:3]
        stream = self.vsr.stream.cuda_stream
        arr = img.map(stream)
        try:
            self.vsr(self.interop.rgba, arr, dw, dh)
        finally:
            img.unmap(stream)  # orders GL's read after VSR's write, no CPU wait
        return tex, fbo

    def _vsr_target(self, w: int, h: int) -> None:
        """(Re)make the VSR output texture at the window's image size. RGBA8, or, for NVIDIA's package's packed
        10-bit output, an R32UI texture that CUDA copies the words into, read by GL through an RGB10_A2 view of the
        same memory (both 32 bits a pixel, R in the low bits as NVIDIA packs it), so nothing unpacks them."""
        from .glcuda import CudaGLImage

        GL = self.GL
        self._vsr_target_free()
        store = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, store)
        if self.vsr.out_format == "rgb10a2":
            GL.glTexStorage2D(GL.GL_TEXTURE_2D, 1, GL.GL_R32UI, w, h)
            tex = GL.glGenTextures(1)
            GL.glTextureView(tex, GL.GL_TEXTURE_2D, store, GL.GL_RGB10_A2, 0, 1, 0, 1)
            GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
            GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_SWIZZLE_A, GL.GL_ONE)  # NVIDIA leaves alpha 0 at times
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
            tex = store
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST)
        GL.glTexParameteri(GL.GL_TEXTURE_2D, GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_READ_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)
        self.vsr_out = (tex, fbo, CudaGLImage(int(store)), w, h, store)

    def _vsr_target_free(self) -> None:
        if self.vsr_out is not None:
            tex, fbo, img = self.vsr_out[:3]
            if img.array.value:
                self.vsr.forget(img.array)
            img.close()
            self.GL.glDeleteFramebuffers(1, [fbo])
            self.GL.glDeleteTextures(sorted({int(tex), int(self.vsr_out[5])}))  # a view and its storage, or one
            self.vsr_out = None

    def _stats_ui(self, dh: int) -> float:
        """The stats overlay's scale: the screen's own (its height / 1080) times Stats overlay size."""
        k = OVERLAY_SIZES.get(getattr(self.settings, "overlay_size", "small"), OVERLAY_SIZES["small"])
        return max(0.6, max(0.5, dh / 1080) * k)

    def _draw_stats(self, text: str, x0: int, y0: int, dw: int, dh: int) -> None:
        """The stats overlay (Minimal, a MiniText, or Detailed, a str) in the corner picked (Stats overlay position),
        tucked into its edges, out of the way of the game's HUD. Drawn again only when it changes."""
        ui = self._stats_ui(dh)
        if isinstance(text, MiniText):
            state = ("mini", str(text), text.parts, text.health, round(ui, 3))
            if state != self._mini_state:
                self.mini_panel.update(_mini(text, ui))
        else:
            state = ("detail", text, round(ui, 3))
            if state != self._mini_state:
                if self._detail_ui != round(ui, 3):
                    self._detail_w, self._detail_ui = 0, round(ui, 3)
                img = _detail(text, ui, self._detail_w)
                self._detail_w = img.shape[1]
                self.mini_panel.update(img)
        self._mini_state = state
        p, m = self.mini_panel, max(2, int(4 * ui))
        corner = getattr(self.settings, "overlay_corner", "top_left")
        x = x0 + dw - p.w - m if corner.endswith("right") else x0 + m
        p.blit(x, y0 + p.h + m if corner.startswith("bottom") else y0 + dh - m)

    # ---- looks
    def remote_commands(self) -> dict:
        """The launcher's Looks page, through framegen/remote.py: `look-show <look JSON>` shows a look it's editing,
        `look-hide` goes back to the settings' look, `looks-reload` reads the user's looks again after a change."""
        import json

        def show(text: str) -> None:
            try:
                self.looks.show(json.loads(text))
            except ValueError as e:
                print(f"[looks] can't show that look: {e}")

        def reload(_text: str) -> None:
            self.looks.reload()
            if self.remote is not None:
                self.remote.publish()  # the new list of looks

        return {"look-show": show, "look-hide": lambda _text: self.looks.show(None), "looks-reload": reload}

    def baked(self, frame):
        """The frame just shown, sharpened and with the look drawn on, at the stream's size: for instant replay with
        "Looks in clips" on (framegen/replay.py). Call right after write(), on its thread. The frame itself when
        neither is on, or when the frames aren't on the graphics card (then it says so, once). Sharpening carried
        over from FSR sharpening isn't used here: it was only while FSR 1 enlarged the picture, and a clip isn't."""
        plan = self.look_gl.plan(self.looks.current(), self.looks.rev) if self.look_gl is not None else None
        sharp = self._sharpness(False)
        if plan is None and sharp is None:
            return frame
        if not (_is_cuda(frame) and self.interop is not None):
            if not self._bake_said:
                self._bake_said = True
                what = "sharpening" if plan is None else "the look" if sharp is None else "the look or sharpening"
                print(f"[looks] Looks in clips needs the frames on the graphics card; clips are saved without {what}")
                self.toast(f"Clips are saved without {what} (it needs the frames on the graphics card)", 5.0)
            return frame
        import torch
        GL = self.GL
        if self._bake is None:
            from .glcuda import CudaGLReader
            tex, fbo = self._bake_target()
            self._bake = (tex, fbo, CudaGLReader(int(tex), self.w, self.h))
        tex, fbo, reader = self._bake
        src = self.tex
        if sharp is not None and plan is None:
            self.fsr.sharpen(self.tex, fbo, 0, 0, self.w, self.h, sharp, flip=False)
        elif sharp is not None:  # sharpened first, as on the screen, then the look
            if self._bake_sharp is None:
                self._bake_sharp = self._bake_target()
            src = self._bake_sharp[0]
            self.fsr.sharpen(self.tex, self._bake_sharp[1], 0, 0, self.w, self.h, sharp, flip=False)
        if plan is not None:
            self.look_gl.draw(plan, src, False, fbo, 0, 0, self.w, self.h, False, self._frame_no, "bake")
        GL.glViewport(0, 0, *self.glfw.get_framebuffer_size(self.win))
        rgba = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device=frame.device)
        reader.read(rgba)
        return rgba[..., [2, 1, 0]]  # BGR, like the frame

    def _bake_target(self) -> tuple[int, int]:
        """An RGBA8 texture at the stream's size and its framebuffer, for baked(). Filtered without mipmaps, so the
        look can read it (the sharpened frame it's drawn from)."""
        GL = self.GL
        tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, self.w, self.h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)
        return tex, fbo

    def hide(self) -> None:
        """Take the window off the screen now, before what's left of stopping (clips finishing saving): close()
        follows."""
        try:
            self.glfw.hide_window(self.win)
            self._events()
        except Exception:  # noqa: BLE001 - closing anyway
            pass

    def close(self) -> None:
        if self.hotkeys is not None:
            self.hotkeys.close()
            self.hotkeys = None
        self.GL.glFinish()
        self.timer._collect()
        private = self.vsr is not None and self.vsr.private  # NVIDIA's package: its timings stay off the log
        timed = {k: v for k, v in self.timer.ms.items() if not (private and "RTX VSR" in k)}
        if timed:
            parts = [f"{k} p50 {np.median(v):.2f} / p95 {np.percentile(v, 95):.2f} ms (n={len(v)})"
                     for k, v in sorted(timed.items(), key=lambda kv: -len(kv[1]))]
            print("[display] GPU time to draw each frame: " + "; ".join(parts))
        self.timer.close()
        self._unwatch()
        self.menu_panel.close()
        self.toast_panel.close()
        self.mini_panel.close()
        self.blitter.close()
        if self.vsr is not None:
            self._vsr_target_free()
            self.vsr.close()
            v = self.vsr.gpu_ms
            if v and not self.vsr.private:
                print(f"[display] RTX VSR q{self.vsr.quality} GPU time per frame: p50 {np.median(v):.2f} / "
                      f"p95 {np.percentile(v, 95):.2f} ms (n={len(v)})")
        if self.fsr is not None:
            self.fsr.close()
        if self._bake is not None:
            self._bake[2].close()
            self.GL.glDeleteFramebuffers(1, [self._bake[1]])
            self.GL.glDeleteTextures([self._bake[0]])
        if self._bake_sharp is not None:
            self.GL.glDeleteFramebuffers(1, [self._bake_sharp[1]])
            self.GL.glDeleteTextures([self._bake_sharp[0]])
        if self.look_gl is not None:
            self.look_gl.close()
        self.looks.close()
        if self.interop is not None:
            self.interop.close()
        if self.dx is not None:
            self.dx.close()
            self.glfw.destroy_window(self.glwin)
        if self._borderless and sys.platform == "darwin":
            _mac_hide_bars(False)  # for the next window, as the program goes on
        self.glfw.destroy_window(self.win)
        self.glfw.terminate()


class WindowSink:
    """OpenCV window (fallback; its GDI paint costs 5-12 ms/frame, too slow for 60 Hz on Windows).
    Keys: i = toggle interpolation, s = stats overlay, f = fullscreen, r/b = save the instant replay (plain or
    before & after), q/Esc = quit."""

    NAME = "SmoovPlay"

    def __init__(self, w: int, h: int, monitor=None) -> None:
        cv2.namedWindow(self.NAME, cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)
        cv2.resizeWindow(self.NAME, min(w, 1600), min(h, 900))
        if monitor is not None:
            cv2.moveWindow(self.NAME, monitor.x + 80, monitor.y + 80)
        self.full = False
        self.overlay = True

    gpu = False

    def write(self, frame, text: str | None = None) -> str | None:
        frame = _host(frame)
        if self.overlay and text:
            frame = _draw_text(frame, text)
        cv2.imshow(self.NAME, frame)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            return "quit"
        if k == ord("f"):
            self.full = not self.full
            cv2.setWindowProperty(self.NAME, cv2.WND_PROP_FULLSCREEN,
                                  cv2.WINDOW_FULLSCREEN if self.full else cv2.WINDOW_NORMAL)
        if k == ord("s"):
            self.overlay = not self.overlay
        if k == ord("i"):
            return "toggle_interp"
        if k in (ord("r"), ord("b")):
            return "save_replay" if k == ord("r") else "save_compare"
        return None

    def close(self) -> None:
        cv2.destroyWindow(self.NAME)


class FileSink:
    """Encodes output frames to a 60 fps file with NVENC (falls back to x264)."""

    def __init__(self, path: str, w: int, h: int, fps: float = 60.0) -> None:
        enc = ["-c:v", "h264_nvenc", "-preset", "p5", "-cq", "18"] if _has_nvenc() else \
              ["-c:v", "libx264", "-preset", "veryfast", "-crf", "16"]
        self.p = subprocess.Popen(
            ["ffmpeg", "-v", "error", "-y", "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}",
             "-r", str(fps), "-i", "-", *enc, "-pix_fmt", "yuv420p", path],
            stdin=subprocess.PIPE)

    gpu = False

    def write(self, frame, text: str | None = None) -> None:
        self.p.stdin.write(memoryview(np.ascontiguousarray(_host(frame))))

    def close(self) -> None:
        self.p.stdin.close()
        self.p.wait()


class MultiSink:
    def __init__(self, *sinks) -> None:
        self.sinks = [s for s in sinks if s is not None]
        # Hand out CUDA tensors if any sink can use them directly; the others copy to the CPU themselves.
        self.gpu = any(getattr(s, "gpu", False) for s in self.sinks)

    def write(self, frame, text: str | None = None):
        cmd = None
        for s in self.sinks:
            cmd = s.write(frame, text) or cmd
        return cmd

    def wait(self, w: Waiting) -> str | None:
        cmd = None
        for s in self.sinks:
            if hasattr(s, "wait"):
                cmd = s.wait(w) or cmd
        return cmd

    def toast(self, text: str, seconds: float = 2.0) -> None:
        for s in self.sinks:
            if hasattr(s, "toast"):
                s.toast(text, seconds)

    def close(self) -> None:
        for s in self.sinks:
            s.close()


_nvenc: bool | None = None


def _has_nvenc() -> bool:
    global _nvenc
    if _nvenc is None:
        r = subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "color=s=256x256:d=0.1",
                            "-c:v", "h264_nvenc", "-f", "null", "-"], capture_output=True)
        _nvenc = r.returncode == 0
    return _nvenc
