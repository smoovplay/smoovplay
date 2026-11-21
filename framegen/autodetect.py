"""Auto-detect: suggests settings for this PC from a short test of the graphics card, the display and what's installed.
The launcher's Settings page runs it ("Auto-detect"), and setup applies it to a new install.

    python -m framegen.autodetect                    # what it found and suggests: "[detect] ..." lines, then one JSON line
    python -m framegen.autodetect --apply            # also saves the suggestions
    python -m framegen.autodetect --apply --if-new   # only when there's no settings file yet (what setup runs)

The test times RIFE on a 1080p frame (the Remote Play stream) for a few seconds: one new real frame plus one in-between
frame, at full and at half motion detail, each with the engine "Auto" runs there (DirectML without an NVIDIA card).
If NVIDIA VFG is set up, it times its three quality modes the same way. recommend() turns those times and the display into settings; it's a plain
function, so the tests can try it on any PC.

VFG's times are shown on the launcher's Settings page and written to logs\autodetect.txt only (not the session log
that Copy diagnostics reads): NVIDIA's licence forbids publishing VFG benchmark results.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from .settings import BY_KEY, DEFAULT_PATH, Settings

PREFIX = "[detect-result] "
STREAM = (1920, 1080)

# An in-between frame (a new real frame's preparation, RIFE, the upscaler) has to be ready within one display tick,
# 1000/hz ms. Calibrated on an RTX 4070 Ti SUPER over PS5 Remote Play, where this test measures 9.3 ms at full motion
# detail (TensorRT) and 6.8 at half (PyTorch): 90 fps at full (85% of the tick) showed 99% of the extra frames and
# 120 fps at half (83%) 98%, but 90 fps with RTX VSR on top (98%) only 77%. 90% sits between them.
SHARE = 0.90
LAPTOP_SHARE = 0.80  # laptop graphics slow down as they warm up
# Upscaler times per frame at 1440p on that card (scaled here by the card's speed and the display's size).
FSR_MS = 0.12
VSR_MS = {1: 1.3, 2: 1.6, 3: 3.6, 4: 4.9}
REF_HALF_MS = 6.8  # this test's half-motion time on that card: the yardstick for "the card's speed"
REF_PIXELS = 2560 * 1440
TRT_SPEEDUP = 0.88  # TensorRT's time at full motion detail vs PyTorch's, when its engine isn't built yet to time
# Frame rate and motion detail, best first. 90 fps at full detail beats 120 at half: in testing it looked better.
OPTIONS = ((120, "full"), (90, "full"), (120, "half"), (90, "half"), (60, "full"), (60, "half"))
VFG_QUALITIES = ("high", "medium", "low")  # NVIDIA VFG's modes, most careful first


@dataclass(frozen=True)
class Display:
    width: int
    height: int
    hz: int  # the refresh rate Windows runs it at
    max_hz: int  # the fastest it can run at this resolution


def primary_display() -> Display | None:
    """The main display (where the video window opens), or None if it can't be read (not Windows)."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes as wt

    class DisplayDevice(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("DeviceName", wt.WCHAR * 32), ("DeviceString", wt.WCHAR * 128),
                    ("StateFlags", wt.DWORD), ("DeviceID", wt.WCHAR * 128), ("DeviceKey", wt.WCHAR * 128)]

    class DevMode(ctypes.Structure):
        _fields_ = [("dmDeviceName", wt.WCHAR * 32), ("dmSpecVersion", wt.WORD), ("dmDriverVersion", wt.WORD),
                    ("dmSize", wt.WORD), ("dmDriverExtra", wt.WORD), ("dmFields", wt.DWORD),
                    ("dmPositionX", wt.LONG), ("dmPositionY", wt.LONG), ("dmDisplayOrientation", wt.DWORD),
                    ("dmDisplayFixedOutput", wt.DWORD), ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short),
                    ("dmYResolution", ctypes.c_short), ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short),
                    ("dmFormName", wt.WCHAR * 32), ("dmLogPixels", wt.WORD), ("dmBitsPerPel", wt.DWORD),
                    ("dmPelsWidth", wt.DWORD), ("dmPelsHeight", wt.DWORD), ("dmDisplayFlags", wt.DWORD),
                    ("dmDisplayFrequency", wt.DWORD), ("dmICMMethod", wt.DWORD), ("dmICMIntent", wt.DWORD),
                    ("dmMediaType", wt.DWORD), ("dmDitherType", wt.DWORD), ("dmReserved1", wt.DWORD),
                    ("dmReserved2", wt.DWORD), ("dmPanningWidth", wt.DWORD), ("dmPanningHeight", wt.DWORD)]

    u32 = ctypes.windll.user32
    dd = DisplayDevice(cb=ctypes.sizeof(DisplayDevice))
    i, name = 0, None
    while u32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
        if dd.StateFlags & 4:  # DISPLAY_DEVICE_PRIMARY_DEVICE
            name = dd.DeviceName
            break
        i += 1
        dd = DisplayDevice(cb=ctypes.sizeof(DisplayDevice))
    now = DevMode(dmSize=ctypes.sizeof(DevMode))
    if not u32.EnumDisplaySettingsW(name, -1, ctypes.byref(now)):  # ENUM_CURRENT_SETTINGS
        return None
    best, m = now.dmDisplayFrequency, 0
    while True:
        mode = DevMode(dmSize=ctypes.sizeof(DevMode))
        if not u32.EnumDisplaySettingsW(name, m, ctypes.byref(mode)):
            break
        if (mode.dmPelsWidth, mode.dmPelsHeight) == (now.dmPelsWidth, now.dmPelsHeight):
            best = max(best, mode.dmDisplayFrequency)
        m += 1
    return Display(now.dmPelsWidth, now.dmPelsHeight, now.dmDisplayFrequency, best)


def nvidia_smi(samples: int = 3) -> dict:
    """Driver version and how busy the card is (%), from nvidia-smi; {} if it isn't there. Busy is the lowest of a few
    readings half a second apart: a game keeps the card busy, a moment's work (like setup's own, just before) doesn't."""
    driver, busy = None, []
    for i in range(samples):
        if i:
            time.sleep(0.5)
        try:
            out = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,utilization.gpu", "--format=csv,noheader,nounits"],
                                 capture_output=True, text=True, timeout=10,
                                 creationflags=0x08000000 if sys.platform == "win32" else 0).stdout  # CREATE_NO_WINDOW
            d, b = [x.strip() for x in out.splitlines()[0].split(",")[:2]]
            driver = d
            busy.append(int(b))
        except Exception:  # noqa: BLE001 - only used for the summary and the "busy" note
            break
    return {"driver": driver, "busy": min(busy)} if busy else {}


def time_rife(model: str, log: Callable[[str], None] = print) -> dict:
    """ms for one new real frame plus one in-between frame, at 1080p: {"full"/"half": {"ms", "engine", "estimated"}}.
    Each motion detail uses the engine Auto runs there (trt.auto_engine). A TensorRT engine that isn't built yet
    (setup builds it; otherwise the first start does, in a minute or two) isn't built here: PyTorch is timed instead
    and TensorRT's time estimated from it."""
    import torch

    from .interpolator import RifeInterpolator
    from .trt import auto_engine

    w, h = STREAM
    frame = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, device="cuda")
    out = {}
    for motion, scale in (("full", 1.0), ("half", 0.5)):
        engine, estimated = auto_engine(scale), False
        log(f"timing the AI model at {motion} motion detail ({'TensorRT' if engine == 'trt' else 'PyTorch'})")
        try:
            it = RifeInterpolator(w, h, model, scale, backend=engine, build_engine=False)
        except FileNotFoundError:
            it = RifeInterpolator(w, h, model, scale, backend="torch")
            estimated = True
        it.load(frame)
        it.prepare(0)
        ms = _time_ticks(it) * (TRT_SPEEDUP if estimated else 1.0)
        out[motion] = {"ms": round(ms, 2), "engine": engine, "estimated": estimated}
        del it
        torch.cuda.empty_cache()
    return out


def time_dml(model: str, log: Callable[[str], None] = print) -> dict:
    """time_rife() for the DirectML engine (an AMD or Intel card; framegen/dml.py). Each tick waits for its frame
    (it comes back to the CPU), so a plain clock times it, CPU<->GPU copies included, as a session has them."""
    import torch

    from .dml import DmlInterpolator

    w, h = STREAM
    frame = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8)
    out = {}
    for motion, scale in (("full", 1.0), ("half", 0.5)):
        log(f"timing the AI model at {motion} motion detail (DirectML)")
        it = DmlInterpolator(w, h, model, scale)
        it.load(frame)
        it.prepare(0)
        rounds = []
        for r in range(7):  # the first two let the card's clocks come up from idle
            t0 = time.perf_counter()
            for _ in range(12):
                it.prepare(1)
                it.interpolate(1, 0.5)
            if r >= 2:
                rounds.append((time.perf_counter() - t0) * 1000 / 12)
        out[motion] = {"ms": round(sorted(rounds)[len(rounds) // 2], 2), "engine": "dml", "estimated": False}
        del it
    return out


def _time_ticks(it) -> float:
    """Median ms of one tick with a new real frame: prepare it (from `in_u8`) and draw the in-between frame."""
    import torch

    def tick() -> None:
        it.prepare(1)
        it.interpolate(1, 0.5)

    start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    rounds = []
    for r in range(7):  # the first two let the card's clocks come up from idle
        torch.cuda.synchronize()
        start.record()
        for _ in range(12):
            tick()
        end.record()
        torch.cuda.synchronize()
        if r >= 2:
            rounds.append(start.elapsed_time(end) / 12)
    return sorted(rounds)[len(rounds) // 2]


def time_vfg(log: Callable[[str], None] = print) -> dict:
    """NVIDIA VFG's time for the same tick at each quality, at 1080p: {"low"/"medium"/"high": {"ms"}}. The two
    frames differ (a smooth pattern, moved), so it can't take a shortcut for a still picture. Stays on this PC."""
    import torch
    import torch.nn.functional as F

    from .vfg import VfgInterpolator

    w, h = STREAM
    g = torch.Generator(device="cuda").manual_seed(1)
    small = torch.randint(0, 256, (1, 3, h // 16, w // 16), dtype=torch.uint8, device="cuda", generator=g)
    a = F.interpolate(small.float(), size=(h, w), mode="bilinear", align_corners=False).round().to(torch.uint8)
    a = a[0].permute(1, 2, 0).contiguous()
    b = torch.roll(a, shifts=(6, 14), dims=(0, 1))
    out = {}
    for q in ("low", "medium", "high"):
        log(f"timing NVIDIA VFG at {q} quality")
        it = VfgInterpolator(w, h, q)
        it.load(a)
        it.prepare(0)
        it.load(b)  # each tick then prepares b into slot 1: a -> b, every time
        out[q] = {"ms": round(_time_ticks(it), 2)}
        it.close()
        del it
        torch.cuda.empty_cache()
    return out


def _fps(hz: int) -> str:
    return BY_KEY["hz"].name(hz)


def recommend(current: dict, timing: dict, display: Display | None, *, gpu: str = "", trt: bool = False,
              vsr: str | None = None, busy: int | None = None, vfg: dict | None = None, vfg_why: str = "",
              hdr_display: bool = False) -> dict:
    """Settings for this PC. current: the settings now (Settings.values()); timing: time_rife()'s result; vsr: why
    RTX VSR can't be used, or None if it can; busy: how busy the card was before the test (%); vfg: time_vfg()'s
    result if NVIDIA VFG is set up, else None, with vfg_why saying why; hdr_display: the main display has Windows'
    HDR on.
    Returns {"settings": {key: value}, "reasons": {key: text}, "notes": [text]}; only frame generation and picture
    settings are suggested, the rest are left as they are. NVIDIA VFG is suggested when it's set up and keeps up at
    least as high a frame rate as RIFE: setting it up is the user's choice to use it."""
    share = LAPTOP_SHARE if "laptop" in gpu.lower() else SHARE
    speed = timing["half"]["ms"] / REF_HALF_MS  # above 1 = slower than the card the numbers above come from
    pixels = display.width * display.height if display else REF_PIXELS
    enlarge = pixels > STREAM[0] * STREAM[1] * 1.05  # a 1080p display shows the stream as it is
    fsr_ms = FSR_MS * speed * pixels / REF_PIXELS if enlarge else 0.0
    rate = display.hz if display else 60
    top = 120 if rate >= 119 else 90 if rate >= 89 else 60  # 119 = 119.88 Hz

    def cost(motion: str) -> float:
        return timing[motion]["ms"] + fsr_ms

    def fits(hz: int, motion: str) -> bool:
        return cost(motion) <= share * 1000 / hz

    hz, motion = next(((h, m) for h, m in OPTIONS if h <= top and fits(h, m)), (60, "half"))
    settings, reasons, notes = {}, {}, []

    # ---- model: NVIDIA VFG at the highest frame rate it keeps up with, then its most careful quality that fits
    def vfg_fits(h: int, q: str) -> bool:
        return vfg[q]["ms"] + fsr_ms <= share * 1000 / h

    rife = BY_KEY["model"].default
    vfg_pick = next(((h, q) for h in (120, 90, 60) if h <= top for q in VFG_QUALITIES if vfg_fits(h, q)),
                    None) if vfg else None
    dml = timing["full"].get("engine") == "dml"
    if vfg_pick is not None and vfg_pick[0] >= hz:
        _hdr(settings, reasons, current=current, hdr_display=hdr_display, dml=dml)
        return _recommend_vfg(vfg_pick, vfg, hz, settings, reasons, notes, current=current, rate=rate, top=top,
                              share=share, fsr_ms=fsr_ms, speed=speed, pixels=pixels, enlarge=enlarge, vsr=vsr,
                              display=display, busy=busy)
    _hdr(settings, reasons, current=current, hdr_display=hdr_display, dml=dml)
    if vfg:
        settings["model"] = rife
        reasons["model"] = (f"NVIDIA VFG is set up, but it only keeps up at {_fps(vfg_pick[0])} on this graphics "
                            f"card, and RIFE at {_fps(hz)}." if vfg_pick else
                            f"NVIDIA VFG is set up, but it can't keep up at {_fps(60)} on this graphics card.")
    elif current.get("model") == "vfg":
        settings["model"] = rife
        reasons["model"] = vfg_why or "NVIDIA VFG isn't set up on this PC."
    tick = 1000 / hz

    # ---- frame rate
    settings["hz"] = hz
    t = cost(motion)
    if not fits(hz, motion):
        reasons["hz"] = (f"Even 60 fps is tight on this graphics card: an in-between frame takes {t:.1f} ms, and "
                         f"60 fps leaves {tick:.1f} ms. Expect some frames to be shown twice.")
    elif top == 60:
        reasons["hz"] = f"Your display runs at {rate} Hz, so 60 fps is the most it can show."
    elif hz == top:
        reasons["hz"] = (f"Your display runs at {rate} Hz, and your graphics card draws an in-between frame in "
                         f"{t:.1f} ms, inside the {tick:.1f} ms each frame has at {_fps(hz)}.")
    elif hz == 90 and motion == "full" and fits(120, "half"):
        reasons["hz"] = ("120 fps would only fit with half motion detail on this graphics card. 90 fps keeps full "
                         "detail, and looked better in testing.")
    else:
        need = min(cost(m) for m in ("full", "half"))
        reasons["hz"] = (f"At {_fps(top)} each frame has {1000 / top:.1f} ms, but your graphics card needs {need:.1f} "
                         f"ms for an in-between frame. {_fps(hz)} leaves it enough time.")

    # ---- motion detail (run.py: Auto is half at 120 fps for a stream up to 1440p, full otherwise)
    auto_motion = "half" if hz == 120 else "full"
    settings["motion"] = "auto" if motion == auto_motion else motion
    if motion == "half" and hz < 120:
        reasons["motion"] = (f"Half detail takes {timing['half']['ms']:.1f} ms instead of {timing['full']['ms']:.1f}, "
                             f"which is what lets {_fps(hz)} keep up on this graphics card.")
    elif motion == "full" and hz == 120:
        reasons["motion"] = (f"This graphics card is fast enough for full detail even at 120 fps "
                             f"({timing['full']['ms']:.1f} ms of {tick:.1f}).")
    elif hz == 120:
        reasons["motion"] = "Auto uses half detail at 120 fps, which this graphics card needs."
    else:
        reasons["motion"] = "Auto uses full detail here: the sharpest on fast motion."

    # ---- engine
    settings["engine"] = "auto"
    if timing["full"].get("engine") == "dml":
        reasons["engine"] = ("Without an NVIDIA card, RIFE runs through DirectML (Experimental). It's slower than "
                             "on an NVIDIA card of the same speed, and hasn't been tried with a game on AMD or Intel "
                             "graphics yet: Send feedback on the Help page says how it went.")
    elif trt:
        reasons["engine"] = ("TensorRT is installed. Auto uses it at full motion detail and PyTorch at half: the "
                             "faster of the two for each.")
    else:
        reasons["engine"] = ("TensorRT isn't installed, so Auto uses PyTorch, about 12% slower at full motion "
                             "detail. Repair SmoovPlay, on the Help page, puts it back.")
    if timing["full"].get("estimated"):
        notes.append("TensorRT hasn't been set up for this graphics card yet, so its speed is estimated. It does "
                     "that the first time you play, in a minute or two.")

    _upscaler(settings, reasons, current=current, hz=hz, left=share * tick - timing[motion]["ms"], fsr_ms=fsr_ms,
              speed=speed, pixels=pixels, enlarge=enlarge, vsr=vsr, display=display)
    _notes(notes, settings, display=display, hz=hz, rate=rate, share=share, busy=busy)
    return {"settings": settings, "reasons": reasons, "notes": notes}


def _recommend_vfg(pick: tuple, vfg: dict, rife_hz: int, settings: dict, reasons: dict, notes: list, *, current: dict,
                   rate: int, top: int, share: float, fsr_ms: float, speed: float, pixels: int, enlarge: bool,
                   vsr: str | None, display: Display | None, busy: int | None) -> dict:
    """recommend(), when NVIDIA VFG is the pick. RIFE's engine and motion detail aren't used, so not suggested."""
    hz, q = pick
    tick, name, t = 1000 / hz, BY_KEY["vfg_quality"].name(q), vfg[q]["ms"] + fsr_ms
    settings["model"], settings["vfg_quality"], settings["hz"] = "vfg", q, hz
    reasons["model"] = ("NVIDIA VFG is set up on this PC and keeps up at " + _fps(hz)
                        + (", like RIFE." if rife_hz == hz else f", where RIFE keeps up at {_fps(rife_hz)}."))
    if top == 60:
        reasons["hz"] = f"Your display runs at {rate} Hz, so 60 fps is the most it can show."
    elif hz == top:
        reasons["hz"] = (f"Your display runs at {rate} Hz, and NVIDIA VFG draws an in-between frame in {t:.1f} ms at "
                         f"{name} quality, inside the {tick:.1f} ms each frame has at {_fps(hz)}.")
    else:
        need = vfg["low"]["ms"] + fsr_ms
        reasons["hz"] = (f"At {_fps(top)} each frame has {1000 / top:.1f} ms, but NVIDIA VFG needs {need:.1f} ms for "
                         f"an in-between frame even at Low quality. {_fps(hz)} leaves it enough time.")
    better = [x for x in VFG_QUALITIES[:VFG_QUALITIES.index(q)]]
    reasons["vfg_quality"] = (f"{name} is the most careful quality with time to spare at {_fps(hz)}" +
                              (f": {BY_KEY['vfg_quality'].name(better[-1])} takes {vfg[better[-1]]['ms']:.1f} ms."
                               if better else "."))
    _upscaler(settings, reasons, current=current, hz=hz, left=share * tick - vfg[q]["ms"], fsr_ms=fsr_ms,
              speed=speed, pixels=pixels, enlarge=enlarge, vsr=vsr, display=display)
    _notes(notes, settings, display=display, hz=hz, rate=rate, share=share, busy=busy)
    return {"settings": settings, "reasons": reasons, "notes": notes}


def _hdr(settings: dict, reasons: dict, *, current: dict, hdr_display: bool, dml: bool) -> None:
    """HDR: Auto, for a player still on Off (the default before Auto). Not without an NVIDIA card, which it needs."""
    if dml or current.get("hdr") != "off":
        return
    settings["hdr"] = "auto"
    reasons["hdr"] = ("Auto turns HDR on by itself for a PS5 streaming in HDR, and picks chiaki-ng's codec to match: "
                      + ("this display has Use HDR on in Windows, so the picture stays HDR all the way." if hdr_display
                         else "this display doesn't have HDR on in Windows, so an HDR stream is fitted to it."))


def _upscaler(settings: dict, reasons: dict, *, current: dict, hz: int, left: float, fsr_ms: float, speed: float,
              pixels: int, enlarge: bool, vsr: str | None, display: Display | None) -> None:
    """AMD FSR 1, unless RTX VSR is picked and still fits in what frame generation leaves (`left` ms)."""
    def vsr_ms(q: int) -> float:
        return VSR_MS[q] * speed * pixels / REF_PIXELS

    size = f"{display.width}×{display.height}" if display else "your display"
    if current.get("scaler") == "vsr" and vsr is None and enlarge:
        q = next((q for q in (4, 3, 2, 1) if vsr_ms(q) <= left), None)
        if q is not None:
            settings["scaler"], settings["vsr_quality"] = "vsr", q
            reasons["scaler"] = (f"RTX VSR, as you picked: it fits alongside {_fps(hz)} "
                                 f"(about {vsr_ms(q):.1f} ms a frame at {size}).")
            reasons["vsr_quality"] = (f"{BY_KEY['vsr_quality'].name(q)} is the highest quality with time to spare at "
                                      f"{_fps(hz)}.")
        else:
            settings["scaler"] = "fsr"
            room = f"more than {_fps(hz)} leaves ({left:.1f} ms)" if left >= 0.1 else f"and {_fps(hz)} leaves no time for it"
            reasons["scaler"] = (f"RTX VSR needs about {vsr_ms(1):.1f} ms a frame at {size}, {room}. AMD FSR 1 looked "
                                 "about the same in testing and takes about 0.1 ms.")
    else:
        settings["scaler"] = "fsr"
        if current.get("scaler") == "vsr":
            reasons["scaler"] = (f"RTX VSR isn't set up on this PC ({vsr})." if vsr else
                                 f"Your display is {size}, the stream's own size, so there's nothing for RTX VSR to "
                                 "enlarge. AMD FSR 1 is the lightest.")
        elif not enlarge:
            reasons["scaler"] = (f"Your display is {size}, the stream's own size, so the picture is hardly "
                                 "enlarged. AMD FSR 1 is the lightest and plenty.")
        else:
            reasons["scaler"] = (f"Sharp and almost free: about {max(fsr_ms, 0.05):.1f} ms a frame at {size}. RTX VSR "
                                 "looked about the same in testing and takes far longer.")


def _notes(notes: list, settings: dict, *, display: Display | None, hz: int, rate: int, share: float,
           busy: int | None) -> None:
    if display is None:
        notes.append("Couldn't read the display's refresh rate, so this assumes 60 Hz.")
    elif display.max_hz > display.hz + 5:
        notes.append(f"Your display can run at {display.max_hz} Hz, but Windows has it at {display.hz} Hz. Raise it "
                     "(Windows Settings > System > Display > Advanced display), then run Auto-detect again for a "
                     "smoother frame rate.")
    if display is not None and hz > 60 and rate % hz > 2 and hz - rate % hz > 2:
        notes.append(f"{_fps(hz)} doesn't divide evenly into your display's {rate} Hz, so keep G-Sync or FreeSync "
                     "on (most fast displays have one) for even motion.")
    if share == LAPTOP_SHARE:
        notes.append("Laptop graphics slow down as they warm up, so this leaves extra room.")
    if busy is not None and busy >= 30:
        notes.append(f"Your graphics card was already {busy}% busy, so the test may have run slow. Close games and "
                     "video apps, then run it again.")
    for k, v in settings.items():
        assert v in BY_KEY[k].choices, (k, v)


VFG_MIN_DRIVER = (570, 65)  # NVIDIA VFG's oldest driver (framegen/vfg.py)


def vfg_offer(not_set_up: bool, gpu: str, driver: str | None) -> str:
    """What Auto-detect says about setting up NVIDIA VFG, or "" for nothing. Said when the graphics card can run it
    (an RTX 40 or 50 series card: not_set_up is vfg_unavailable() == NOT_SET_UP) and it isn't set up. It's NVIDIA's
    own download, under NVIDIA's licence, so setting it up is the player's choice, but they should know their card can
    run it; the launcher puts a button to set it up next to this. Once it's set up, Auto-detect times it and suggests
    it if it keeps up (recommend()). No speed or quality claims: NVIDIA's licence forbids publishing VFG's results."""
    if not not_set_up:
        return ""
    name = gpu.replace("NVIDIA ", "").replace("GeForce ", "").strip() or "graphics card"
    text = (f"Your {name} can also run NVIDIA VFG, NVIDIA's own frame generation, in place of RIFE. It's a separate "
            "download from NVIDIA, under NVIDIA's licence. Set it up, and Auto-detect times it and suggests it if it "
            "keeps up on this PC.")
    if vfg_driver_too_old(driver):
        text += (f" It needs NVIDIA driver {VFG_MIN_DRIVER[0]}.{VFG_MIN_DRIVER[1]} or newer, and this PC has {driver}: "
                 "update it first (see the PC check).")
    return text


def vfg_driver_too_old(driver: str | None) -> bool:
    """Whether nvidia-smi's driver version (e.g. "566.36", always two digits after the point on Windows) is older than
    NVIDIA VFG's oldest. Unknown or in another form: False, and VFG itself says if it can't start."""
    try:
        return bool(driver) and tuple(int(x) for x in driver.split(".")[:2]) < VFG_MIN_DRIVER
    except ValueError:
        return False


def detect(current: dict, log: Callable[[str], None] = print, timing: dict | None = None,
           vfg: dict | None = None) -> dict:
    """The whole test: what was found ("found": label/value/detail rows for the launcher) plus recommend()'s result.
    timing, vfg: use these times instead of timing RIFE and NVIDIA VFG (for testing the launcher without loading the
    graphics card). With timing but no vfg, VFG isn't timed or suggested."""
    log("reading the display")
    display = primary_display()
    smi = nvidia_smi()
    log("starting the graphics card")
    import torch
    dml_mode = not torch.cuda.is_available()
    if dml_mode:  # an AMD or Intel card: RIFE through DirectML (framegen/dml.py, Experimental)
        from . import dml
        why = dml.unavailable_reason()
        if why:
            raise RuntimeError(f"no NVIDIA card PyTorch can use, and DirectML can't run here either ({why})")
        adapter = dml.pick_adapter()
        gpu, vram = adapter["name"], adapter["vram_gb"]
    else:
        gpu = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 2**30

    from .rtxvsr import unavailable_reason
    from .trt import trt_available
    from .vfg import vfg_unavailable
    trt, vsr = trt_available(), unavailable_reason()
    if dml_mode:
        trt, vsr = False, "it needs an NVIDIA RTX card"
    vfg_why = "" if vfg is not None else vfg_unavailable()
    model = current.get("model", BY_KEY["model"].default)
    model = BY_KEY["model"].default if model == "vfg" else model
    testing = timing is not None
    if timing is None:
        timing = time_dml(model, log) if dml_mode else time_rife(model, log)
    if vfg is None and not vfg_why and not testing:
        try:
            vfg = time_vfg(log)
        except Exception as e:  # noqa: BLE001 - a broken VFG install mustn't stop the rest of the test
            vfg_why = f"NVIDIA VFG couldn't start: {str(e)[:160]}"
            log(vfg_why)
    hdr_display = False  # the main display has Windows' HDR on (HDR output and Auto use it)
    if sys.platform == "win32":
        try:
            from .dxgi_present import display_hdr
            d = display_hdr()
            hdr_display = bool(d and d["hdr"])
        except Exception:  # noqa: BLE001 - can't tell: as without HDR
            pass
    log("choosing settings")
    result = recommend(current, timing, display, gpu=gpu, trt=trt, vsr=vsr, busy=smi.get("busy"), vfg=vfg,
                       vfg_why=vfg_why, hdr_display=hdr_display)

    name = gpu.replace("NVIDIA ", "").replace("GeForce ", "")
    engines = {"trt": "TensorRT", "torch": "PyTorch", "dml": "DirectML (Experimental)"}
    full, half = timing["full"], timing["half"]
    result["found"] = [
        {"label": "Graphics card", "value": f"{name}, {vram:.0f} GB",
         "detail": f"Driver {smi['driver']}" if smi.get("driver") else ""},
        {"label": "AI speed", "value": f"{full['ms']:.1f} ms per in-between frame",
         "detail": f"Full motion detail, {engines[full['engine']]}{' (estimated)' if full['estimated'] else ''}. "
                   f"Half: {half['ms']:.1f} ms, {engines[half['engine']]}."},
        {"label": "Display", "value": f"{display.width}×{display.height} at {display.hz} Hz" if display else "Unknown",
         "detail": "; ".join(x for x in (f"Can run at {display.max_hz} Hz" if display and display.max_hz > display.hz + 5
                                         else "", "HDR on in Windows" if hdr_display else "") if x)},
        {"label": "TensorRT", "value": "NVIDIA cards only" if dml_mode else "Installed" if trt else "Not installed",
         "detail": ""},
        {"label": "RTX VSR", "value": "NVIDIA cards only" if dml_mode else "Ready" if vsr is None else "Not set up",
         "detail": ""},
    ]
    if vfg:  # on this PC's Settings page only (NVIDIA's licence: VFG's results aren't for publishing)
        result["found"].append({"label": "NVIDIA VFG", "value": f"{vfg['medium']['ms']:.1f} ms per in-between frame",
                                "detail": f"Medium quality. Low: {vfg['low']['ms']:.1f} ms, "
                                          f"High: {vfg['high']['ms']:.1f} ms."})
    else:
        from .vfg import NOT_SET_UP
        result["found"].append({"label": "NVIDIA VFG",
                                "value": "Set up" if not vfg_why else "Not set up" if vfg_why == NOT_SET_UP
                                else "Not available",
                                "detail": "" if not vfg_why else "Optional: see below." if vfg_why == NOT_SET_UP
                                else vfg_why})
        offer = vfg_offer(vfg_why == NOT_SET_UP, gpu, smi.get("driver"))
        if offer:
            result["vfg_offer"] = offer  # the launcher shows it with a button that sets VFG up...
            result["vfg_offer_ready"] = not vfg_driver_too_old(smi.get("driver"))  # ...greyed out until the driver is
    result["timing"] = timing
    result["vfg_timing"] = vfg
    result["busy"] = smi.get("busy")
    return result


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--apply", action="store_true", help="save the suggested settings")
    ap.add_argument("--if-new", action="store_true", help="only if there's no settings file yet")
    ap.add_argument("--settings", type=Path, default=DEFAULT_PATH, help=f"settings file (default {DEFAULT_PATH})")
    ap.add_argument("--timing", metavar="FULL,HALF", help="for testing: use these ms instead of timing the AI model")
    ap.add_argument("--vfg-timing", metavar="LOW,MEDIUM,HIGH",
                    help="for testing: NVIDIA VFG's ms (as if it's set up) instead of timing it")
    a = ap.parse_args(argv)
    if a.if_new and a.settings.exists():
        print("[detect] settings already saved, so they're left as they are", flush=True)
        return 0
    settings = Settings.load(a.settings)
    try:
        timing = None
        if a.timing:
            from .trt import auto_engine
            full, half = (float(x) for x in a.timing.split(","))
            timing = {"full": {"ms": full, "engine": auto_engine(1.0), "estimated": False},
                      "half": {"ms": half, "engine": "torch", "estimated": False}}
        vfg = None
        if a.vfg_timing:
            vfg = {q: {"ms": float(x)} for q, x in zip(("low", "medium", "high"), a.vfg_timing.split(","))}
        result = detect(settings.values(), log=lambda m: print(f"[detect] {m}", flush=True), timing=timing, vfg=vfg)
    except RuntimeError as e:
        print(f"[detect] ERROR: {e}", flush=True)
        return 2
    print(PREFIX + json.dumps(result), flush=True)  # ASCII (\u escapes), whatever the console's code page
    if a.apply and a.if_new and (result.get("busy") or 0) >= 30:
        # Setup: suggestions from a busy card would be too low, and nobody is looking at them. The defaults stay.
        print(f"[detect] the graphics card was {result['busy']}% busy, so nothing was saved", flush=True)
    elif a.apply:
        for k, v in result["settings"].items():
            settings.set(k, v)
        print(f"[detect] saved to {a.settings}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
