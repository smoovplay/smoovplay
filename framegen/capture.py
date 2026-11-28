"""Capture cards (Experimental): a console's HDMI output through a capture card, read with FFmpeg's DirectShow input
(PyAV, already installed), so nothing extra to download.

How (research_notes/Expansion planning/sources.md, section 2):
- Raw frames. The card's own NV12 or YUY2, straight off USB with no decoding, timed by the card's clock. MJPEG
  (compressed; most cheap cards offer it) only when the card has nothing else at the size and rate asked for: it's
  decoded on the GPU (NVDEC) where it can be, else on the CPU, which can stutter.
- Converted to BGR on the GPU: one upload of the raw frame (3-4 MB at 1080p), then a few tensor operations, so the
  pipeline gets CUDA frames as it does from Remote Play's GPU decoder, and RTX VSR works too. Without CUDA, OpenCV
  converts on the CPU (BT.601, a slight colour shift on HD video).
- Colours: HDMI video is limited range (16-235) unless the card says it's full range; BT.709 at 720 lines and up,
  BT.601 below, as video players assume.
- Sound: the card's own audio device, opened with the video (one DirectShow graph, so both are timed by the same
  clock) and played on this PC through framegen/audio.py, held back to meet the picture. pick_audio() finds it by
  name; --capture-audio names another, or "off".

Untested with real cards. Written from FFmpeg's documentation and source and other viewers' notes (NitLink, MIT;
CapturePlay and Veyra, GPL, ideas only), and tested with a raw video file standing in for a card
(tests/test_capture.py). The launcher marks capture cards Experimental.

    python -m framegen.capture --list [--json]   # the video and audio capture devices Windows has
    python -m framegen.capture --formats NAME    # what a device offers, and the format SmoovPlay would pick
"""
from __future__ import annotations

import argparse
import json
import os
import queue
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass

import numpy as np

from .sync import CAPTURE_LOCK

# Raw formats, best first: NV12 is the smallest uncompressed format (4:2:0), YUY2/UYVY are 4:2:2 (more USB bandwidth,
# same picture after 4:2:0 HDMI sources). Anything compressed comes after all of them, MJPEG last but one.
RAW_RANK = {"nv12": 0, "yuyv422": 1, "uyvy422": 2, "yuv420p": 3, "bgr24": 4, "bgr0": 4, "rgb24": 4}
COMPRESSED_RANK = {"mjpeg": 10}
BYTES_PER_PIXEL = {"nv12": 1.5, "yuv420p": 1.5, "yuyv422": 2, "uyvy422": 2, "bgr24": 3, "rgb24": 3, "bgr0": 4}
# A compressed frame's size, for sizing DirectShow's queue (open_options): a generous MJPEG frame, 2.5 MB at 4K. Read
# as raw (3 bytes a pixel), the queue held 50 to 200 frames, one to three seconds of delay once the PC fell behind.
COMPRESSED_BYTES_PER_PIXEL = 0.3
CONVERTIBLE = ("nv12", "yuyv422", "uyvy422")  # converted here; anything else raw goes through PyAV's own conversion
SOUND_CHECK_S = 10.0  # how long after starting a card's sound is checked for, once (CardSource._say_sound)
HEARD_LEVEL = 64  # a 16-bit sample above this (of 32767, about -54 dBFS) is sound, not a silent line
# A frame that comes off DirectShow more than this many frames later than the card's frames usually do: the PC has
# fallen behind the card, and frames are skipped until it catches up (CatchUp).
BEHIND_FRAMES = 2.5
DAMAGED = object()  # a compressed frame the decoder couldn't read: skipped (CardSource._damaged)
# No frame readable for this long: the card's frames aren't readable at all, and capture stops. Generous, as a stick
# can send broken frames for a moment while it settles on the signal.
DAMAGED_GIVE_UP_S = 5.0


@dataclass(frozen=True)
class Device:
    name: str  # what Windows calls it, and what to open it by
    kind: str  # "video", "audio" or "unknown" (a device whose type FFmpeg couldn't tell, e.g. a virtual camera that
    # isn't running)
    alt: str = ""  # DirectShow's unique name, for when two devices share a name


@dataclass(frozen=True)
class Mode:
    """One way a device can send video, as DirectShow lists it."""
    fmt: str  # pixel format for raw video ("nv12", "yuyv422", ...), or the codec ("mjpeg", "h264")
    raw: bool
    w: int
    h: int
    fps_min: float
    fps_max: float
    range: str = ""  # "tv" (limited) or "pc" (full) when the card says; "" when it doesn't
    space: str = ""  # colour matrix, when the card says ("bt709", "bt470bg", ...)

    def label(self) -> str:
        fps = f"{self.fps_max:g}" if abs(self.fps_max - self.fps_min) < 0.01 else f"{self.fps_min:g}-{self.fps_max:g}"
        return f"{self.w}x{self.h} {fps} fps {self.fmt.upper() if not self.raw else self.fmt}"


# ----------------------------------------------------------------------------- what Windows has
def _dshow_log(url: str, options: dict) -> str:
    """FFmpeg's DirectShow listings come out in its log, not as data: open with a listing option and read the log."""
    import av
    import av.logging
    old = av.logging.get_level()
    av.logging.set_level(av.logging.INFO)
    try:
        with av.logging.Capture() as logs:
            try:
                av.open(url, format="dshow", options=options).close()
            except Exception:  # noqa: BLE001 - a listing always "fails": FFmpeg exits after printing it
                pass
    finally:
        av.logging.set_level(old)
    return "".join(msg for _, name, msg in logs if name == "dshow")


def parse_devices(text: str) -> list[Device]:
    """Devices from `-list_devices true` output. FFmpeg 5+ tags each device with the kinds of output it has:
    '"Name" (video)', '(audio)', '(none)' when it couldn't look, or several, '(video, audio)' or '(video, unknown)',
    as cards that also carry their sound (or a stream FFmpeg has no name for) show. Older versions put devices
    under "DirectShow video devices" / "DirectShow audio devices" headings. Both are read, and any device with
    video counts as video."""
    out: list[Device] = []
    heading = "unknown"
    for line in text.splitlines():
        line = re.sub(r"^\[dshow @ [0-9A-Fa-fx]+\]\s?", "", line)
        if "DirectShow video devices" in line:
            heading = "video"
        elif "DirectShow audio devices" in line:
            heading = "audio"
        elif m := re.match(r'\s*Alternative name "(.*)"\s*$', line):
            if out:
                out[-1] = Device(out[-1].name, out[-1].kind, m.group(1))
        elif m := re.match(r'\s*"(.*)"\s*(?:\(([a-z, ]*)\))?\s*$', line):
            kinds = {k.strip() for k in m.group(2).split(",")} if m.group(2) is not None else {heading}
            kind = "video" if "video" in kinds else "audio" if "audio" in kinds else "unknown"
            out.append(Device(m.group(1), kind))
    return out


# DirectShow's device list (ole32's System Device Enumerator), read through its COM interfaces with ctypes
_CLSID_SYSTEM_DEVICE_ENUM = "{62BE5D10-60EB-11D0-BD3B-00A0C911CE86}"
_IID_ICREATE_DEV_ENUM = "{29840822-5B84-11D0-BD3B-00A0C911CE86}"
_IID_IPROPERTY_BAG = "{55272A00-42CB-11CE-8135-00AA004BB851}"
_CATEGORIES = (("video", "{860BB310-5D01-11D0-BD3B-00A0C911CE86}"),  # CLSID_VideoInputDeviceCategory
               ("audio", "{33D9A762-90C8-11D0-BD43-00A0C911CE86}"))  # CLSID_AudioInputDeviceCategory


def dshow_names() -> list[Device]:
    """The video and audio input devices DirectShow lists, by the names FFmpeg opens them by, read from Windows'
    device list (each device's property bag) without loading any device's driver. OSError if DirectShow can't be
    asked.

    FFmpeg's own listing (-list_devices) also opens every device to see what it carries, so one faulty camera or
    virtual-camera driver can crash or hang it, and SmoovPlay with it, before any card is listed. OBS lists names
    this way first for the same reason. Opening a card by name (CardSource.open) loads only that card's driver."""
    import ctypes
    import uuid
    from ctypes import wintypes as wt

    ole32, oleaut32 = ctypes.WinDLL("ole32"), ctypes.WinDLL("oleaut32")
    ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]
    oleaut32.VariantClear.argtypes = [ctypes.c_void_p]

    class GUID(ctypes.Structure):
        _fields_ = [("bytes", ctypes.c_ubyte * 16)]

    class VARIANT(ctypes.Structure):  # 24 bytes on 64-bit Windows: the type, 6 reserved bytes, then the value
        _fields_ = [("vt", ctypes.c_ushort), ("r1", ctypes.c_ushort), ("r2", ctypes.c_ushort), ("r3", ctypes.c_ushort),
                    ("value", ctypes.c_void_p), ("value2", ctypes.c_void_p)]

    def guid(s: str) -> GUID:
        g = GUID()
        ctypes.memmove(g.bytes, uuid.UUID(s).bytes_le, 16)
        return g

    def call(obj, index: int, *args, argtypes=()):
        """Method `index` of a COM object's table (0-2: QueryInterface, AddRef, Release), returning its HRESULT."""
        table = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
        return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(table[index])(obj, *args)

    def release(obj) -> None:
        if obj:
            call(obj, 2)

    def check(hr: int, what: str) -> None:
        if hr < 0:
            raise OSError(f"DirectShow: {what} failed (0x{hr & 0xFFFFFFFF:08X})")

    init = ole32.CoInitializeEx(None, 2)  # COINIT_APARTMENTTHREADED, as FFmpeg's DirectShow input uses
    if init < 0 and init & 0xFFFFFFFF != 0x80010106:  # RPC_E_CHANGED_MODE: COM is already set up another way: works
        check(init, "CoInitializeEx")
    out: list[Device] = []
    enum = ctypes.c_void_p()
    try:
        check(ole32.CoCreateInstance(ctypes.byref(guid(_CLSID_SYSTEM_DEVICE_ENUM)), None, 1,  # CLSCTX_INPROC_SERVER
                                     ctypes.byref(guid(_IID_ICREATE_DEV_ENUM)), ctypes.byref(enum)), "CoCreateInstance")
        for kind, category in _CATEGORIES:
            monikers = ctypes.c_void_p()
            hr = call(enum, 3, ctypes.byref(guid(category)), ctypes.byref(monikers), 0,  # CreateClassEnumerator
                      argtypes=(ctypes.c_void_p, ctypes.c_void_p, wt.DWORD))
            check(hr, "CreateClassEnumerator")
            if not monikers:  # S_FALSE: nothing in this category
                continue
            try:
                for _ in range(256):
                    m, got = ctypes.c_void_p(), wt.ULONG()
                    if call(monikers, 3, 1, ctypes.byref(m), ctypes.byref(got),  # IEnumMoniker::Next
                            argtypes=(wt.ULONG, ctypes.c_void_p, ctypes.c_void_p)) != 0 or not m:
                        break
                    try:
                        bag = ctypes.c_void_p()
                        iid = guid(_IID_IPROPERTY_BAG)
                        if call(m, 9, None, None, ctypes.byref(iid), ctypes.byref(bag),  # IMoniker::BindToStorage
                                argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)) < 0:
                            continue  # OBS and FFmpeg skip a device without a property bag too
                        try:
                            v = VARIANT()
                            if call(bag, 3, "FriendlyName", ctypes.byref(v), None,  # IPropertyBag::Read
                                    argtypes=(wt.LPCWSTR, ctypes.c_void_p, ctypes.c_void_p)) < 0:
                                continue
                            name = ctypes.wstring_at(v.value) if v.vt == 8 and v.value else ""  # VT_BSTR
                            oleaut32.VariantClear(ctypes.byref(v))
                        finally:
                            release(bag)
                        alt, ctx, shown = "", ctypes.c_void_p(), ctypes.c_void_p()
                        if ole32.CreateBindCtx(0, ctypes.byref(ctx)) >= 0:
                            if call(m, 20, ctx, None, ctypes.byref(shown),  # IMoniker::GetDisplayName
                                    argtypes=(ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p)) >= 0 and shown:
                                alt = ctypes.wstring_at(shown).replace(":", "_")  # FFmpeg's "Alternative name"
                                ole32.CoTaskMemFree(shown)
                            release(ctx)
                        if name:
                            out.append(Device(name, kind, alt))
                    finally:
                        release(m)
            finally:
                release(monikers)
    finally:
        release(enum)
        if init >= 0:
            ole32.CoUninitialize()
    return out


def list_devices() -> list[Device]:
    """The capture devices Windows has, by name (dshow_names), or from FFmpeg's own listing if DirectShow's list
    can't be read that way. An empty video list is looked at once more after half a second: a card can be missing
    for a moment while its driver starts, or while another app lets it go. On a Mac, AVFoundation's devices
    (avfoundation_devices)."""
    if sys.platform == "darwin":
        return avfoundation_devices()
    devices: list[Device] = []
    for attempt in range(2):
        if attempt:
            time.sleep(0.5)
        try:
            devices = dshow_names()
        except OSError:
            devices = parse_devices(_dshow_log("dummy", {"list_devices": "true"}))
        if any(d.kind == "video" for d in devices):
            break
    return devices


def avfoundation_devices() -> list[Device]:
    """A Mac's cameras, capture cards and microphones, from FFmpeg's AVFoundation listing. Each one's alt is its index
    there, which is what a Mac opens a card by. Screens ("Capture screen N") are left out."""
    import av
    import av.logging
    old = av.logging.get_level()
    av.logging.set_level(av.logging.INFO)
    try:
        with av.logging.Capture() as logs:
            try:
                av.open("", format="avfoundation", options={"list_devices": "true"}).close()
            except Exception:  # noqa: BLE001 - a listing always raises once FFmpeg has printed it
                pass
    finally:
        av.logging.set_level(old)
    return parse_avfoundation("".join(msg for _, name, msg in logs if name.startswith("AVFoundation")))


def parse_avfoundation(text: str) -> list[Device]:
    devices, kind = [], "unknown"
    for line in text.splitlines():
        if "video devices" in line:
            kind = "video"
        elif "audio devices" in line:
            kind = "audio"
        elif (m := re.match(r"\s*\[(\d+)\]\s+(.+?)\s*$", line)) and not m.group(2).startswith("Capture screen"):
            devices.append(Device(m.group(2), kind, m.group(1)))
    return devices


# Words in device names that say nothing about which card it is.
_GENERIC = {"usb", "usb2", "usb3", "audio", "video", "capture", "device", "digital", "interface", "input", "hdmi",
            "line", "microphone", "mic", "camera", "webcam", "stream", "streaming", "game", "high", "definition",
            "the", "and", "for", "with", "pro", "plus"}


def _words(name: str) -> set[str]:
    return {w for w in re.findall(r"[a-z0-9]+", name.lower()) if len(w) >= 3 and w not in _GENERIC}


# AVerMedia's cards add a sound device named "Not To Use (<card>-Audio)" next to the real one
_NOT_TO_USE = re.compile(r"\bnot to use\b", re.I)


def _card_name(name: str) -> str:
    """A device's name without a "-Video"/"-Audio" ending, lower case ("Live Gamer Ultra 2.1-Video" -> "live gamer
    ultra 2.1"), unless only generic words would be left ("USB Video" stays as it is)."""
    low = name.lower().strip()
    core = re.sub(r"[\s_-]*(?:video|audio)$", "", low)
    return core if core != low and _words(core) else low


def pick_audio(video: str, devices: list[Device]) -> str | None:
    """The capture card's own sound device: one named like the video device (e.g. "USB3.0 Capture" and "Digital
    Audio Interface (USB3.0 Capture)", "Cam Link 4K" for both, or "Live Gamer Ultra 2.1-Video" and "HDMI/Line In
    (Live Gamer Ultra 2.1-Audio)"), else None. A device whose name says nothing (only generic words) matches only on
    the whole name. AVerMedia's "Not To Use" sound device is never picked."""
    audio = [d.name for d in devices if d.kind == "audio" and not _NOT_TO_USE.search(d.name)]
    v = _card_name(video)
    whole = [a for a in audio if v and (v in a.lower() or a.lower() in v)]
    if whole:
        return min(whole, key=len)
    vw = _words(video)
    scored = [(len(vw & _words(a)), a) for a in audio]
    best = max(scored, default=(0, None))
    return best[1] if best[0] > 0 and sum(n == best[0] for n, _ in scored) == 1 else None


# FourCCs FFmpeg's DirectShow input lists as "unknown compression type" but that are worth knowing about: a card's
# 10-bit HDR format, which FFmpeg can't capture (its DirectShow input has no P010), so such modes are noted, not used
_UNREADABLE = {0x30313050: "p010", 0x36313050: "p016"}
_MODE = re.compile(r"(?:pixel_format=(?P<pix>\S+)|vcodec=(?P<codec>\S+)"
                   r"|unknown compression type (?P<fourcc>0x[0-9A-Fa-f]+))\s+"
                   r"min s=(?P<w0>\d+)x(?P<h0>\d+) fps=(?P<f0>[\d.e+-]+)\s+"
                   r"max s=(?P<w1>\d+)x(?P<h1>\d+) fps=(?P<f1>[\d.e+-]+)"
                   r"(?:\s*\((?P<color>[^)]*)\))?")


def parse_modes(text: str) -> list[Mode]:
    """Video modes from `-list_options true` output, e.g.
        pixel_format=nv12  min s=1920x1080 fps=60 max s=1920x1080 fps=60 (tv, bt709/bt709/bt709, topleft)
        vcodec=mjpeg  min s=1920x1080 fps=30 max s=1920x1080 fps=60
    Each line becomes one Mode at its largest size (cards list one line per size). Duplicates are dropped."""
    modes: list[Mode] = []
    for m in _MODE.finditer(text):
        if m.group("fourcc") is not None and int(m.group("fourcc"), 16) not in _UNREADABLE:
            continue
        raw = m.group("pix") is not None
        rng = space = ""
        if m.group("color"):
            parts = [p.strip() for p in m.group("color").split(",")]
            rng = parts[0] if parts and parts[0] in ("tv", "pc") else ""
            if len(parts) > 1:
                space = parts[1].split("/")[0]
                space = "" if space == "unknown" else space
        fmt = m.group("pix") or m.group("codec") or _UNREADABLE[int(m.group("fourcc"), 16)]
        mode = Mode(fmt, raw, int(m.group("w1")), int(m.group("h1")),
                    float(m.group("f0")), float(m.group("f1")), rng, space)
        if mode not in modes:
            modes.append(mode)
    return modes


def open_failure(device: str, log: str, error: str) -> str:
    """Why a card wouldn't open, in plain words, from FFmpeg's DirectShow messages; "" when they don't say."""
    text = f"{log}\n{error}"
    if "Could not run graph" in text or "Could not RenderStream" in text or "0X800700AA" in text.upper():  # ERROR_BUSY
        return (f"{device} is in use by another app. Close OBS, Elgato's apps (also in the tray by the clock), "
                "Discord, a browser tab using the camera, or anything else that shows the card, then press Play "
                "again: many cards, Elgato's USB ones among them, work with one app at a time")
    if "E_ACCESSDENIED" in text or "0X80070005" in text.upper() or "access is denied" in text.lower():
        return (f"Windows wouldn't let SmoovPlay open {device}. Turn on Settings > Privacy & security > Camera > "
                "Let desktop apps access your camera (and the same for Microphone), then press Play again")
    if "Could not find video device" in text:
        return f"Windows no longer lists {device}: reconnect the card, then press Look again on the Play page"
    if "Could not set video options" in text:
        return f"{device} refused the video format SmoovPlay asked for"
    return ""


class FormatRefused(RuntimeError):
    """The card wouldn't take the format asked for ("Could not set video options"): another may do (CardSource.open)."""


def _open_card(device: str, opts: dict, audio_device: str | None = None):
    """av.open for a card's picture (and its sound device, if given), with a plain-words reason when it fails
    (open_failure); FormatRefused when it's the format the card turned down."""
    import av
    import av.logging
    old = av.logging.get_level()
    av.logging.set_level(av.logging.WARNING)
    url = f"video={device}" + (f":audio={audio_device}" if audio_device else "")
    try:
        with av.logging.Capture() as logs:
            try:
                return av.open(url, format="dshow", options=opts)
            except Exception as e:  # noqa: BLE001 - reworded below when FFmpeg's messages say why
                log = "".join(msg for _, name, msg in logs if name == "dshow")
                why = open_failure(device, log, str(e))
                if "Could not set video options" in f"{log}\n{e}":
                    raise FormatRefused(why) from e
                if why:
                    raise RuntimeError(why) from e
                raise
    finally:
        av.logging.set_level(old)


def _open_with_sound(device: str, opts: dict, audio_device: str | None):
    """The card's picture, with its sound device when there is one and it opens; FormatRefused goes to the caller."""
    if audio_device:
        try:  # a small sound buffer: DirectShow's default is half a second or more
            return _open_card(device, {**opts, "audio_buffer_size": "20"}, audio_device)
        except FormatRefused:
            raise
        except Exception as e:  # noqa: BLE001 - busy, gone, or no sound: the picture on its own
            if audio_device == device:
                print(f"[capture] no sound from the card itself ({str(e)[:120]}): playing without sound; pick "
                      "a sound device in the launcher, or with --capture-audio (see --list-devices)")
            else:
                print(f"[capture] couldn't open the sound device ({str(e)[:120]}); playing without sound")
    return _open_card(device, opts)


def open_tries(mode: Mode, fps: float,
               modes: Callable[[], list[Mode]]) -> Iterator[tuple[str, Mode | None, dict]]:
    """(what, mode, av.open options) to open a card with, in order, for when it refuses one (#72): the mode picked;
    one interval slower (the card's listing is rounded); at the format's own rate; the card's other formats at that
    size, in pick_mode's order (`modes()` lists them, only if it comes to that); and whatever the card sends."""
    yield mode.label(), mode, open_options(mode, fps)
    yield f"{mode.label()} one step slower", mode, open_options(mode, fps, slower=1)
    own = open_options(mode, fps)
    del own["framerate"]
    yield f"{mode.label()} at its own rate", mode, own
    others = [m for m in modes() if (m.w, m.h) == (mode.w, mode.h) and m != mode
              and (m.fmt in RAW_RANK or m.fmt in COMPRESSED_RANK)]
    for m in sorted(others, key=lambda m: (RAW_RANK.get(m.fmt, COMPRESSED_RANK.get(m.fmt, 99)), -m.fps_max)):
        yield m.label(), m, open_options(m, fps)
    yield "the card's own format", None, {"rtbufsize": own["rtbufsize"]}


def list_modes(device: str) -> list[Mode]:
    return parse_modes(_dshow_log(f"video={device}", {"list_options": "true"}))


def pick_mode(modes: list[Mode], w: int = 1920, h: int = 1080, fps: float = 60.0) -> tuple[Mode | None, str]:
    """The mode to ask for, and a note for the log when it isn't what we'd want ("" when it is).

    The size asked for if the card has it (else the largest at or under it, else the smallest), at a rate that
    reaches `fps` if possible, in the best format: raw before compressed, NV12 first."""
    if not modes:
        return None, "the card listed no video formats"
    usable = [m for m in modes if m.fmt in RAW_RANK or m.fmt in COMPRESSED_RANK]
    if not usable:
        return None, "the card offers no format SmoovPlay can read (" + ", ".join(sorted({m.fmt for m in modes})) + ")"
    sizes = {(m.w, m.h) for m in usable}
    if (w, h) in sizes:
        size = (w, h)
    else:
        under = [s for s in sizes if s[0] <= w and s[1] <= h]
        size = max(under, key=lambda s: s[0] * s[1]) if under else min(sizes, key=lambda s: s[0] * s[1])
    at = [m for m in usable if (m.w, m.h) == size]
    fast = [m for m in at if m.fps_max >= fps - 0.5]
    pool = fast or at
    best = min(pool, key=lambda m: (RAW_RANK.get(m.fmt, COMPRESSED_RANK.get(m.fmt, 99)), -m.fps_max))
    notes = []
    if size != (w, h):
        notes.append(f"no {w}x{h} mode, using {size[0]}x{size[1]}")
    if not fast:
        notes.append(f"no mode reaches {fps:g} fps at that size (best {best.fps_max:g})")
    if not best.raw:
        notes.append(f"only {best.fmt.upper()} (compressed) at that size and rate: decoded on the GPU where it can be")
    if all(m.w * m.h <= 1280 * 720 and m.fps_max <= 30.5 for m in usable) and (w * h > 1280 * 720 or fps > 30.5):
        notes.append("the card offers 1280x720 at 30 fps at most, as cards do on a USB 2 port: connect it to a USB 3 "
                     "port (blue, ideally on the back of the PC)")
    if any(m.fmt in _UNREADABLE.values() for m in modes):
        notes.append("the card also offers 10-bit HDR, which SmoovPlay can't read yet, so it uses 8-bit "
                     f"{best.fmt.upper()}: if the picture looks washed out, turn HDR off on the console")
    return best, "; ".join(notes)


def frame_interval(mode: Mode, rate: float) -> int:
    """The time between frames, in DirectShow's 100 ns units, for `rate` in a mode, kept within the mode's own range.
    A card lists its range as intervals, which FFmpeg prints as rates to 6 digits ("59.9999" for 166667), so the
    interval is worked back from those (#72: asking for 59.9999 meant 166666, faster than the card goes, and it
    refused)."""
    fastest = round(1e7 / mode.fps_max) if mode.fps_max > 0 else 1
    slowest = round(1e7 / mode.fps_min) if mode.fps_min > 0 else fastest
    return min(max(round(1e7 / rate), fastest), max(slowest, fastest))


def exact_rate(interval: int) -> str:
    """A frame rate for FFmpeg's DirectShow input that it turns back into exactly `interval`. It keeps the rate's
    numerator and denominator at 1001000 at most, as a "num:den" pair (a decimal goes through a double first), then
    works out den * 10^7 // num, rounding down, which must land on the interval: so the largest den that keeps num in
    bounds, and the num that gives the interval."""
    den = max(1, 1_001_000 * interval // 10_000_000)
    num = den * 10_000_000 // interval
    return f"{num}:{den}"


MAX_SOUND_LATE_S = 0.080  # the most sound_late_s() takes off (or adds to) the sound's delay
CALIBRATION_N = 100  # packets of each (about 2 s) that sound_late_s() is measured on, once, at the start


def open_options(mode: Mode, fps: float = 60.0, queue_frames: int = 4, slower: int = 0) -> dict:
    """av.open options for a mode. rtbufsize caps DirectShow's own queue at a few frames: it only fills when we fall
    behind, and frames dropped there are counted by FFmpeg ("real-time buffer too full"). A compressed format's frames
    are counted at COMPRESSED_BYTES_PER_PIXEL. slower: that many 100 ns steps more between frames (open_tries)."""
    rate = min(fps, mode.fps_max) if mode.fps_max >= fps - 0.5 else mode.fps_max
    per_pixel = BYTES_PER_PIXEL.get(mode.fmt, 3) if mode.raw else COMPRESSED_BYTES_PER_PIXEL
    opts = {"video_size": f"{mode.w}x{mode.h}", "framerate": exact_rate(frame_interval(mode, rate) + slower),
            "rtbufsize": str(int(mode.w * mode.h * per_pixel * queue_frames))}
    opts["pixel_format" if mode.raw else "vcodec"] = mode.fmt
    return opts


# ----------------------------------------------------------------------------- colour conversion
def ycbcr_to_bgr(y, u, v, bt709: bool, full_range: bool):
    """Full-resolution Y, Cb, Cr (float tensors, 0-255 scale) -> HxWx3 uint8 BGR, on whatever device they're on.
    The same maths as framegen.chiaki_tap.cuda_frame_to_bgr (FFmpeg's)."""
    import torch
    kr, kb = (0.2126, 0.0722) if bt709 else (0.299, 0.114)
    kg = 1.0 - kr - kb
    ys, yo, cs = (1.0, 0.0, 1.0) if full_range else (255 / 219, 16.0, 255 / 224)
    lum = (y - yo) * ys
    u, v = (u - 128.0) * cs, (v - 128.0) * cs
    b = lum + 2 * (1 - kb) * u
    g = lum - (2 * kb * (1 - kb) / kg) * u - (2 * kr * (1 - kr) / kg) * v
    r = lum + 2 * (1 - kr) * v
    return torch.stack([b, g, r], -1).clamp_(0, 255).round_().to(torch.uint8)


def raw_to_bgr(raw, fmt: str, w: int, h: int, bt709: bool, full_range: bool):
    """One raw frame (a flat uint8 tensor: NV12, YUY2 or UYVY) -> HxWx3 uint8 BGR tensor on the same device."""
    import torch.nn.functional as F

    def up(c):  # chroma to full size (bilinear, as FFmpeg's default scaler does near enough)
        return F.interpolate(c[None, None].float(), size=(h, w), mode="bilinear", align_corners=False)[0, 0]

    if fmt == "nv12":
        y = raw[: w * h].view(h, w).float()
        uv = raw[w * h: w * h * 3 // 2].view(h // 2, w // 2, 2)
        return ycbcr_to_bgr(y, up(uv[..., 0]), up(uv[..., 1]), bt709, full_range)
    if fmt in ("yuyv422", "uyvy422"):
        p = raw[: w * h * 2].view(h, w // 2, 4)
        iy0, iu, iy1, iv = (0, 1, 2, 3) if fmt == "yuyv422" else (1, 0, 3, 2)
        y = p[..., [iy0, iy1]].reshape(h, w).float()
        return ycbcr_to_bgr(y, up(p[..., iu]), up(p[..., iv]), bt709, full_range)
    raise ValueError(f"can't convert {fmt}")


def raw_to_bgr_cpu(raw: np.ndarray, fmt: str, w: int, h: int) -> np.ndarray:
    """The CPU fallback (no CUDA): OpenCV's converters, which are BT.601 limited range."""
    import cv2
    if fmt == "nv12":
        return cv2.cvtColor(raw[: w * h * 3 // 2].reshape(h * 3 // 2, w), cv2.COLOR_YUV2BGR_NV12)
    if fmt == "yuyv422":
        return cv2.cvtColor(raw[: w * h * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_YUY2)
    if fmt == "uyvy422":
        return cv2.cvtColor(raw[: w * h * 2].reshape(h, w, 2), cv2.COLOR_YUV2BGR_UYVY)
    raise ValueError(f"can't convert {fmt}")


# ----------------------------------------------------------------------------- the source
class CatchUp:
    """Whether the PC has fallen behind a card. A card's frames don't wait: when reading one takes longer than a frame
    (4K at full motion detail), DirectShow queues the rest, and the delay grows by a frame every few until its queue
    is full. So each frame's lateness (when it came off DirectShow, against its own timestamp) is compared with the
    least of the last WINDOW_S seconds, and one more than BEHIND_FRAMES later than that is behind: the reader skips
    reading it, which empties DirectShow's queue in a moment. The window follows a card's clock as it drifts (about
    40 ms a minute on some, see sound_late_raw_s)."""

    WINDOW_S = 10.0

    def __init__(self, fps: float) -> None:
        self.limit = max(BEHIND_FRAMES / max(fps, 1.0), 0.030)
        self._lows: deque = deque()  # (time, lateness), the lateness rising: the window's least is first

    def behind(self, now: float, late: float) -> bool:
        while self._lows and self._lows[-1][1] >= late:
            self._lows.pop()
        self._lows.append((now, late))
        while self._lows[0][0] < now - self.WINDOW_S:
            self._lows.popleft()
        return late - self._lows[0][1] > self.limit


class CardSource:
    """Frames from a capture card as Ticks: read on a thread and queued (not "latest only", which would break repeat
    detection); when the pipeline falls behind, the oldest frame goes and is counted in `dropped`. When the reading
    itself falls behind the card (CatchUp), frames are skipped before they're read, counted in `caught_up`. A card's
    frames are read on the GPU only once the pipeline is ready (iterating), and not while it restarts (hold): its CUDA
    graphs are captured then, and GPU work on another thread breaks the capture (sync.CAPTURE_LOCK).

    Open a card with CardSource.open(name). Two kinds of file stand in for one, played at their own rate from when
    the pipeline starts reading: CardSource.raw_file(path, w, h, fmt, fps), raw frames read exactly as a card's are
    (every line of the card path but the av.open call), and CardSource.video_file(path), any video, decoded and
    turned into NV12 first. test: a framegen.sources.TestMode, applied before the queue.
    device: "cuda" (default when there is CUDA), "cpu" (torch on the CPU, for tests) or None (numpy and OpenCV).
    colour: "auto", or (bt709, full_range) to override.
    audio: what plays the sound, if the container has any: audio(rate, channels) -> an object with push(pcm bytes),
    set_delay(seconds) and close() (framegen.audio.Player; a fake in tests), or None for no sound.
    """

    def __init__(self, container, name: str, live: bool, test=None, device: str | None = "auto",
                 colour="auto", range_hint: str = "", queue_size: int = 3, decode_to: str = "", audio=None,
                 sound_name: str = "") -> None:
        from .sources import source_info
        self.container, self.live, self.test = container, live, test
        self.stream = container.streams.video[0]
        self.astream = container.streams.audio[0] if audio is not None and container.streams.audio else None
        self._audio_out, self.player, self._resampler = audio, None, None
        self._audio_wait: list = []  # a file's sound, held until the picture gets to it: (time, pcm)
        self._replay_sound: deque = deque(maxlen=2000)  # the sound as it's played, for instant replay (read_sound)
        # How late each stream arrives after its own timestamps (the same clock in one DirectShow graph), so the
        # sound can be held back by the picture's delay less however much later it already comes.
        self._late_v: list[float] = []
        self._late_a: list[float] = []
        cc = self.stream.codec_context
        self.w, self.h = cc.width, cc.height
        self.raw = cc.name == "rawvideo"
        self.fmt = (cc.format.name if cc.format is not None else "") if self.raw else cc.name
        self.decode_to = decode_to  # a video file standing in: decoded, then sent on as raw frames of this format
        if decode_to:
            self.raw, self.fmt = True, decode_to
        rate = self.stream.average_rate or self.stream.base_rate or self.stream.guessed_rate
        self.fps = float(rate) if rate else 60.0
        if device == "auto":
            device = _cuda_device()
        self.device = device
        self.bt709, self.full_range = (self.h >= 720, range_hint == "pc") if colour == "auto" else colour
        self.info = source_info("capture", hz=self.fps, name=name)
        self.dropped = 0
        self.damaged = 0  # frames the decoder couldn't read, skipped (_damaged)
        self._damaged_since: float | None = None  # when the current run of them began
        # A live card's frames skipped to catch up with it (CatchUp). Only where each frame stands alone (raw, MJPEG):
        # a skipped H.264 frame would break the ones after it.
        self.caught_up = 0
        self._catch = CatchUp(self.fps) if live and (self.raw or self.fmt == "mjpeg") else None
        self._stop = False
        self._go = threading.Event()
        self.q: queue.Queue = queue.Queue(maxsize=queue_size)
        self._host = None  # a reusable (pinned, with CUDA) buffer for raw frames on their way to the GPU
        # MJPEG on the GPU: NVDEC decodes it and the frame stays there (as Remote Play's do). Made on the reader's
        # thread, in PyTorch's CUDA context; if it fails, the CPU decodes instead.
        self._gpu_jpeg = not self.raw and self.fmt == "mjpeg" and device == "cuda"
        self._hw = None
        how = (("GPU" if device == "cuda" else "CPU (torch)" if device else "CPU (OpenCV)")
               if self.raw and self.fmt in CONVERTIBLE else "GPU (NVDEC)" if self._gpu_jpeg else "PyAV, CPU")
        print(f"[capture] {name}: {self.w}x{self.h} {self.fmt} at {self.fps:g} fps; decoded and converted on {how}"
              f" ({'BT.709' if self.bt709 else 'BT.601'}, {'full' if self.full_range else 'limited'} range)")
        if not self.raw and not self._gpu_jpeg:
            print(f"[capture] {self.fmt.upper()} is compressed: each frame is decoded on the CPU, which can stutter. "
                  "If the card offers NV12 or YUY2 at this size, that's better.")
        # The sound, watched so the log says when none came, or only silence (#41): count, heard, said yet
        self.sound_name = sound_name or "the card"
        self._sound_n, self._heard, self._sound_checked, self._t_go = 0, False, False, None
        if self.astream is not None:
            cc = self.astream.codec_context
            print(f"[capture] sound from {self.sound_name}: {cc.sample_rate} Hz, {cc.channels} channel(s), played on "
                  "this PC, held back to meet the picture")
        threading.Thread(target=self._reader, daemon=True, name="capture").start()

    # ---- opening
    @classmethod
    def open(cls, device: str, w: int = 1920, h: int = 1080, fps: float = 60.0, mode: Mode | None = None,
             audio_device: str | None = None, audio=None, **kw) -> "CardSource":
        """audio_device: the card's sound device (pick_audio), opened with the video; audio: what plays it. If the card
        refuses the format, the next of open_tries is tried, each said in the log."""
        modes: list[Mode] | None = None
        note = ""
        if mode is None:
            modes = list_modes(device)
            mode, note = pick_mode(modes, w, h, fps)
            if mode is None:
                raise RuntimeError(f"capture card {device!r}: {note}")
        if note:
            print(f"[capture] {note}")

        def card_modes() -> list[Mode]:
            nonlocal modes
            if modes is None:
                try:
                    modes = list_modes(device)
                except Exception:  # noqa: BLE001 - only for trying more formats
                    modes = []
            return modes

        refused = []
        for what, m, opts in open_tries(mode, fps, card_modes):
            try:
                container = _open_with_sound(device, opts, audio_device if audio is not None else None)
            except FormatRefused:
                refused.append(what)
                print(f"[capture] {device} refused {what}")
                continue
            if refused:
                print(f"[capture] {device} took {what}")
            has_sound = bool(getattr(getattr(container, "streams", None), "audio", None))
            return cls(container, f"Capture card ({device})", live=True, range_hint=m.range if m else "",
                       audio=audio, sound_name=audio_device if audio_device and has_sound else "", **kw)
        offers = ", ".join(m.label() for m in card_modes()) or "nothing it lists"
        raise RuntimeError(f"{device} refused every video format SmoovPlay tried. Close Elgato's apps (4K Capture "
                           "Utility, Studio, Camera Hub: also in the tray by the clock) and OBS, then press Play "
                           f"again; if it still refuses, try with HDR off on the console. The card offers: {offers}")

    @classmethod
    def raw_file(cls, path: str, w: int, h: int, fmt: str, fps: float = 60.0, **kw) -> "CardSource":
        import av
        container = av.open(path, format="rawvideo",
                            options={"video_size": f"{w}x{h}", "pixel_format": fmt, "framerate": f"{fps:g}"})
        return cls(container, "Capture card (a raw file standing in)", live=False, **kw)

    @classmethod
    def video_file(cls, path: str, **kw) -> "CardSource":
        import av
        container = av.open(path)
        container.streams.video[0].thread_type = "AUTO"
        return cls(container, "Capture card (a recording standing in)", live=False, decode_to="nv12", **kw)

    # ---- reading
    def _gpu_decoder(self):
        import av
        import torch
        from av.codec.hwaccel import HWAccel
        torch.zeros(1, device="cuda")  # PyTorch's CUDA context, current on this thread (see ChiakiSource._new_decoder)
        ctx = av.CodecContext.create("mjpeg", "r", hwaccel=HWAccel(
            "cuda", options={"primary_ctx": "0", "current_ctx": "1"}, is_hw_owned=True))
        ctx.flags |= av.codec.context.Flags.low_delay
        return ctx

    def _convert_gpu(self, packet):
        """MJPEG through NVDEC. Raises when it can't, and _convert asks the CPU whether it's the frame or the GPU."""
        from .chiaki_tap import cuda_frame_to_bgr
        if self._hw is None:
            self._hw = self._gpu_decoder()
        frames = self._hw.decode(packet)
        return cuda_frame_to_bgr(frames[-1]) if frames else None

    def _convert_cpu(self, packet):
        frames = self.stream.codec_context.decode(packet)
        if not frames:
            return None
        bgr = frames[-1].to_ndarray(format="bgr24")
        if self.device:
            import torch
            return torch.from_numpy(bgr).to(self.device)
        return bgr

    def _convert(self, packet):
        """The frame in a packet; None when there's none yet, DAMAGED when the decoder can't read it."""
        import av
        if self._gpu_jpeg:
            try:
                return self._convert_gpu(packet)
            except Exception as e:  # noqa: BLE001 - e.g. 4:2:2 JPEGs the GPU path can't hand over, or a broken frame
                gpu_error = e
            try:
                frame = self._convert_cpu(packet)
            except av.error.FFmpegError as e:
                return self._damaged(e)  # the CPU can't read it either: the frame itself, and NVDEC stays on
            print(f"[capture] MJPEG on the GPU didn't work ({str(gpu_error)[:120]}); decoding on the CPU instead")
            self._gpu_jpeg, self._hw = False, None
            return frame
        if not self.raw or self.fmt not in CONVERTIBLE:
            try:
                return self._convert_cpu(packet)
            except av.error.FFmpegError as e:
                return self._damaged(e)
        return self._convert_raw(np.frombuffer(packet, dtype=np.uint8))

    def _caught_up(self) -> None:
        """A frame skipped because the reading fell behind the card (CatchUp): counted, and said once."""
        self.caught_up += 1
        if self.caught_up == 1:
            fix = ("Capture card resolution 1440p, or Quarter motion detail, keeps up" if self.w * self.h > 2560 * 1440
                   else "a lower Capture card resolution or motion detail keeps up")
            print(f"[capture] this PC is behind the card at {self.w}x{self.h}: frames are skipped to keep the delay "
                  f"low ({fix})")

    def _damaged(self, e: Exception):
        """A frame the decoder can't read, skipped and counted: cheap cards send the odd broken JPEG, often the first
        one. Capture stops only when none can be read for DAMAGED_GIVE_UP_S, and says so."""
        now = time.perf_counter()
        self.damaged += 1
        if self.damaged == 1:
            print(f"[capture] the card sent a damaged {self.fmt.upper()} frame: skipped ({str(e)[:120]})")
        if self._damaged_since is None:
            self._damaged_since = now
        elif now - self._damaged_since >= DAMAGED_GIVE_UP_S:
            raise RuntimeError(f"none of the card's {self.fmt.upper()} frames could be read for "
                               f"{DAMAGED_GIVE_UP_S:g} s ({self.damaged} damaged)")
        return DAMAGED

    def _convert_raw(self, data: np.ndarray):
        n = int(self.w * self.h * BYTES_PER_PIXEL[self.fmt])
        if data.size < n:
            return None  # a short packet (a format change mid-stream): skip it
        if self.device is None:
            return raw_to_bgr_cpu(data, self.fmt, self.w, self.h)
        import torch
        if self._host is None:
            t = torch.empty(n, dtype=torch.uint8, pin_memory=self.device == "cuda")
            self._host = (t, t.numpy())
        t, a = self._host
        a[:] = data[:n]  # the one copy on the CPU; the upload below finishes before the next frame reuses it
        return raw_to_bgr(t.to(self.device), self.fmt, self.w, self.h, self.bt709, self.full_range)

    def _reader(self) -> None:
        from .io import Tick
        if not self.live:
            self._go.wait()
        t0, n = time.perf_counter(), 0
        try:
            for frame, t_arrival, t_src in self._frames():
                if self._stop:
                    break
                if not self.live and (wait := t0 + n / self.fps - time.perf_counter()) > 0:
                    time.sleep(wait)  # a card delivers on the HDMI clock
                    t_arrival = time.perf_counter()
                if self._audio_wait:  # a file: its sound goes out as the picture gets to it
                    now_s = (t_src if t_src is not None else n / self.fps) + 1 / self.fps
                    while self._audio_wait and self._audio_wait[0][0] <= now_s:
                        self._play(self._audio_wait.pop(0)[1])
                if frame is None:  # a damaged frame, or one skipped to catch up: numbered all the same, as a frame
                    n += 1         # the card sent
                    continue
                tick = Tick(frame, t_arrival, 0, t_source=t_src, t_ready=time.perf_counter())
                # Numbered as they come off the card, so a frame dropped from the queue below (the PC behind for a
                # moment) doesn't shift the count: Game frame rate 30 fps takes every other one as new
                # (cadence.RhythmLock), and a shift by one made it move its beat later, a visible hitch.
                tick.extra["seq"] = n
                n += 1
                if self.test is not None and (tick := self.test(tick)) is None:
                    continue
                try:
                    self.q.put_nowait(tick)
                except queue.Full:
                    self.q.get_nowait()
                    self.dropped += 1
                    self.q.put_nowait(tick)
        except Exception as e:  # noqa: BLE001 - the card went away, or the file ended badly: end the stream
            print(f"[capture] stopped: {str(e)[:200]}")
        self.q.put(None)

    def _frames(self):
        """(frame, arrival time, the source's own time in seconds) for each frame, as it comes; the frame is None for
        a damaged one, which _reader numbers but doesn't send on. Sound packets on the way go to the player (a file's
        wait for the picture: _reader)."""
        streams = [self.stream] + ([self.astream] if self.astream is not None else [])
        if self.decode_to:
            from av.video.reformatter import ColorRange, Colorspace
            # into the colours the raw path will read them as (an MJPEG file is full range, a card's NV12 isn't)
            space = Colorspace.ITU709 if self.bt709 else Colorspace.ITU601
            rng = ColorRange.JPEG if self.full_range else ColorRange.MPEG
            start = None
            for packet in self.container.demux(*streams):
                if packet.stream is not self.stream:
                    self._sound(packet, time.perf_counter())
                    continue
                for fr in packet.decode():
                    t_arrival = time.perf_counter()
                    start = fr.time if start is None else start
                    raw = fr.reformat(format=self.decode_to, dst_colorspace=space, dst_color_range=rng)
                    with CAPTURE_LOCK:
                        frame = self._convert_raw(raw.to_ndarray().ravel())
                    if frame is not None:
                        yield frame, t_arrival, fr.time
            return
        for packet in self.container.demux(*streams):
            if packet.size == 0:
                continue
            t_arrival = time.perf_counter()
            if packet.stream is not self.stream:
                self._sound(packet, t_arrival)
                continue
            t_src = float(packet.pts * packet.time_base) if packet.pts is not None else None
            behind = False
            if t_src is not None:
                self._late_v = (self._late_v + [t_arrival - t_src])[-120:]
                behind = self._catch is not None and self._catch.behind(t_arrival, t_arrival - t_src)
            if self.live and not self._go.is_set():
                continue  # the pipeline is loading or restarting: not read, and nothing on the GPU (class docs)
            if not self._sound_checked and self._t_go is not None and t_arrival - self._t_go >= SOUND_CHECK_S:
                self._sound_checked = True
                self._say_sound(f"in {SOUND_CHECK_S:g} s")
            if behind:
                self._caught_up()
                yield None, t_arrival, None  # numbered as a frame the card sent, as a damaged one is
                continue
            with CAPTURE_LOCK:
                frame = self._convert(packet)
            if frame is DAMAGED:
                yield None, t_arrival, None
                continue
            if frame is not None:
                self._damaged_since = None
                yield frame, t_arrival, t_src

    # ---- sound
    def _sound(self, packet, t_arrival: float) -> None:
        """A sound packet: decoded to 16-bit PCM, then played (a card) or held for the picture (a file)."""
        import av
        if packet.size == 0:
            return
        try:
            frames = packet.decode()
        except Exception:  # noqa: BLE001 - a damaged packet: skip it
            return
        for fr in frames:
            if self._resampler is None:
                channels = min(2, fr.layout.nb_channels if hasattr(fr.layout, "nb_channels") else len(fr.layout.channels))
                self._resampler = av.AudioResampler(format="s16", layout="stereo" if channels == 2 else "mono",
                                                    rate=fr.sample_rate)
                self._rate, self._channels = fr.sample_rate, channels
            pcm = b"".join(o.to_ndarray().tobytes() for o in self._resampler.resample(fr))
            if not pcm:
                continue
            if self.live:
                if fr.time is not None:
                    self._late_a = (self._late_a + [t_arrival - (fr.time + fr.samples / fr.sample_rate)])[-120:]
                self._sound_n += 1
                if not self._heard and np.abs(np.frombuffer(pcm, dtype=np.int16)).max() > HEARD_LEVEL:
                    self._heard = True
                self._play(pcm)
            else:
                self._audio_wait.append((fr.time if fr.time is not None else 0.0, pcm))

    def _say_sound(self, when: str) -> None:
        """Into the log, if the card's sound was opened but none came, or only silence (#41: a stick that worked,
        with no sound and nothing in the log to say why)."""
        if self.astream is None or self._heard:
            return
        if self._sound_n == 0:
            print(f"[capture] no sound came from {self.sound_name} {when}: pick the card's sound device on the Play "
                  "page, and check Windows' sound settings (Input) show its level moving")
        else:
            print(f"[capture] the sound from {self.sound_name} was silent {when}: on the console, send the sound out "
                  "over HDMI (on a PS5: Settings > Sound > Audio Output), with nothing plugged into the controller")

    def _play(self, pcm: bytes) -> None:
        from .io import Sound
        # Timed like the picture it goes with: a card's sound comes this much later than its picture.
        self._replay_sound.append(Sound(time.perf_counter() - self.sound_late_s(), pcm, self._rate, self._channels))
        if self._audio_out is None:
            return
        if self.player is None:
            try:
                self.player = self._audio_out(self._rate, self._channels)
            except Exception as e:  # noqa: BLE001 - no output device: the picture on its own
                print(f"[capture] can't play the sound ({str(e)[:120]}); playing without it")
                self._audio_out, self.astream = None, None
                return
            if self._delay_s is not None:
                self.player.set_delay(self._delay_s)
        self.player.push(pcm)
        if (now := time.perf_counter()) >= self._sound_report_t:
            self._sound_report_t = now + 60.0
            self._report_sound(final=False)
            self._report_clocks()

    def _report_sound(self, final: bool) -> None:
        """Into the log: how the sound played (framegen.audio.sound_report), once a minute when something went
        wrong, and for the whole session at the end, so diagnostics say what any clicks were."""
        stats = getattr(self.player, "stats", None)
        if stats is None:  # (a stand-in player)
            return
        from .audio import sound_report
        try:
            now = stats()
        except Exception:  # noqa: BLE001 - only a log line
            return
        line = sound_report(now, None if final else (self._sound_seen or {}))
        if not final:
            self._sound_seen = now
        if line:
            print(f"[capture] sound{' this session' if final else ', last minute'}: {line}")

    def _report_clocks(self) -> None:
        """Into the log, once a minute at most and only when it's news: how late the card's sound and picture arrive
        against their own timestamps now, against what sound_late_s() measured at the start. The picture's drifts
        steadily (see sound_late_raw_s: about 43 ms a minute on an Elgato HD60 X, 150 lines in one long session), so
        it's said the first time it's over 20 ms, and again only if it jumps (over 200 ms in a minute)."""
        lates = self.sound_lates()
        if lates is None:
            return
        drift = lates[0] - lates[1] - self.sound_late_raw_s()
        last = getattr(self, "_clock_last", None)  # (None until it's been said once)
        if last is not None or abs(drift) > 0.020:
            self._clock_last = drift
        if abs(drift) > 0.020 and (last is None or abs(drift - last) > 0.200):
            print(f"[capture] sound clock: sound arrives {lates[0] * 1000:.0f} ms and picture {lates[1] * 1000:.0f} ms "
                  f"after their timestamps, {drift * 1000:+.0f} ms off from the start" +
                  (" (it drifts on, about 40 ms a minute on some cards; the sound's delay doesn't follow it: it uses "
                   f"{self.sound_late_s() * 1000:+.0f} ms, measured at the start)" if last is None else ""))

    _delay_s: float | None = None
    _late_cal: float | None = None
    _sound_report_t = 0.0
    _sound_seen: dict | None = None
    _rate, _channels = 48000, 2

    def read_sound(self) -> list:
        """The sound played since the last call (framegen.io.Sound chunks), for instant replay."""
        out = []
        while self._replay_sound:
            out.append(self._replay_sound.popleft())
        return out

    def sound_lates(self) -> tuple[float, float] | None:
        """(median lateness of the sound, of the picture), each against its own timestamps; None until known."""
        if len(self._late_v) < 10 or len(self._late_a) < 10:
            return None
        return float(np.median(self._late_a)), float(np.median(self._late_v))

    def sound_late_raw_s(self) -> float:
        """How much later the sound arrives than the picture, after their own timestamps, the lateness of the
        first CALIBRATION_N packets of each, then kept for the session (0 until known). The two clocks drift
        apart afterwards (the card's picture by about 40 ms a minute: a console sends 59.94 Hz, the card numbers
        its frames at 60), which is no delay of the picture's, so it's not followed."""
        if self._late_cal is None:
            if len(self._late_v) < CALIBRATION_N or len(self._late_a) < CALIBRATION_N:
                return 0.0
            self._late_cal = float(np.median(self._late_a) - np.median(self._late_v))
        return self._late_cal

    def sound_late_s(self) -> float:
        """sound_late_raw_s(), kept within +-MAX_SOUND_LATE_S (a start-up hold-up could give a wild one)."""
        return max(-MAX_SOUND_LATE_S, min(MAX_SOUND_LATE_S, self.sound_late_raw_s()))

    # The pipeline's side (as for chiaki-ng, framegen.chiaki_tap.ChiakiSource): the sound is held back to meet the
    # picture (Sound in sync); the delay test needs Remote Play.
    def can_press(self) -> bool:
        return False

    def can_delay_audio(self) -> bool:
        return self.astream is not None and self._audio_out is not None

    def set_audio_delay(self, ms: float) -> None:
        self._delay_s = max(0.0, ms / 1000 - self.sound_late_s())
        if self.player is not None:
            self.player.set_delay(self._delay_s)

    def __iter__(self):
        self.dropped = 0  # frames dropped while the pipeline was loading don't count
        if self._t_go is None:
            self._t_go = time.perf_counter()
        self._go.set()
        from .io import queue_ticks
        yield from queue_ticks(self.q, "the capture card")

    def hold(self, on: bool) -> None:
        """Stop reading while the pipeline restarts (on): a card's frames still come and go, but nothing is read on
        the GPU. Off: start again from the next frame (the queued ones are stale by now)."""
        if on:
            self._go.clear()
            return
        while True:
            try:
                self.q.get_nowait()
            except queue.Empty:
                break
        self._go.set()

    def close(self) -> None:
        self._stop = True
        self._go.set()
        try:
            self.container.close()
        except Exception:  # noqa: BLE001 - closing
            pass
        if self.player is not None:
            self.player.close()
            self._report_sound(final=True)
        if self._t_go is not None:
            self._say_sound("this session")
        print(f"[capture] dropped frames: {self.dropped}"
              + (f", skipped to catch up with the card: {self.caught_up}" if self.caught_up else "")
              + (f", damaged frames from the card skipped: {self.damaged}" if self.damaged else ""))


def _cuda_device() -> str | None:
    try:
        import torch
        return "cuda" if torch.cuda.is_available() else None
    except Exception:  # noqa: BLE001 - no torch: OpenCV on the CPU
        return None


# ----------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the capture devices Windows has")
    ap.add_argument("--raw", action="store_true",
                    help="with --list: FFmpeg's own listing, as it prints it (it opens every device, so a faulty "
                         "driver shows up here)")
    ap.add_argument("--formats", metavar="NAME", help="what a video device offers, and what SmoovPlay would pick")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if a.formats:
        modes = list_modes(a.formats)
        best, note = pick_mode(modes)
        if a.json:
            print(json.dumps({"modes": [asdict(m) for m in modes], "pick": asdict(best) if best else None,
                              "note": note}))
        else:
            for m in modes:
                print(("* " if m == best else "  ") + m.label())
            print(note or ("would use " + best.label() if best else "nothing usable"))
        return 0
    if a.list and a.raw:
        print(_dshow_log("dummy", {"list_devices": "true"}) or "FFmpeg listed nothing")
        return 0
    error = None
    try:
        devices = list_devices()
    except Exception as e:  # noqa: BLE001 - said in the list (the launcher's diagnostics show it), not a crash
        devices, error = [], f"{type(e).__name__}: {e}"
    if a.json:
        print(json.dumps({"video": [asdict(d) for d in devices if d.kind in ("video", "unknown")],
                          "audio": [asdict(d) for d in devices if d.kind == "audio"]}
                         | ({"error": error} if error else {})))
    else:
        for d in devices:
            print(f"{d.kind:7}  {d.name}")
        if not devices:
            print("no capture devices found" + (f" ({error})" if error else ""))
    return 0


if __name__ == "__main__":
    code = main()
    # Leave without Python's clean-up: a camera driver that crashes as it's unloaded at exit would turn a list that's
    # already out into a failure for the launcher
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)
