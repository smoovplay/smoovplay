"""Multi-monitor selection: discovery, stable IDs, HDR/auto-detect per monitor, and window placement."""
from __future__ import annotations

import ctypes
import sys
from types import SimpleNamespace

import pytest

from framegen import autodetect, displays, dxgi_present, io
from framegen.menu import Menu
from framegen.settings import BY_KEY, Settings, specs_json

MON_PRIMARY = displays.Monitor(
    id="ACI3433-UID24835",
    device=r"\\.\DISPLAY2",
    name="ROG PG348Q",
    x=0,
    y=0,
    width=3440,
    height=1440,
    hz=100,
    max_hz=100,
    primary=True,
    hdr=False,
    max_nits=270.0,
)
MON_SECOND = displays.Monitor(
    id="DEL4273-UID24833",
    device=r"\\.\DISPLAY1",
    name="DELL U3223QE",
    x=3440,
    y=0,
    width=3840,
    height=2160,
    hz=60,
    max_hz=60,
    primary=False,
    hdr=True,
    max_nits=400.0,
)


@pytest.fixture(autouse=True)
def _restore_displays():
    choices = list(displays.CHOICES)
    names = dict(displays.NAMES)
    hints = dict(displays.HINTS)
    try:
        yield
    finally:
        displays.CHOICES[:] = choices
        displays.NAMES.clear()
        displays.NAMES.update(names)
        displays.HINTS.clear()
        displays.HINTS.update(hints)


def test_stable_id_and_disambiguation():
    path = r"\\?\DISPLAY#DEL4273#5&26b2da4e&0&UID24833#{e6f07b5f-ee97-4a90-b076-33f57bf4eaa7}"
    assert displays.stable_id(r"\\.\DISPLAY1", path) == "DEL4273-UID24833"
    assert displays.stable_id(r"\\.\DISPLAY1", "") == "DISPLAY1"
    assert displays.stable_id(r"\\.\DISPLAY2", "bad#path") == "DISPLAY2"

    a = displays.Monitor("DEL4273-UID1", r"\\.\DISPLAY1", "Dell U3223QE", 0, 0, 3840, 2160, 60, 60, primary=True)
    b = displays.Monitor("DEL4273-UID1", r"\\.\DISPLAY2", "Dell U3223QE", 3840, 0, 3840, 2160, 60, 60)
    out = displays._disambiguate([a, b])
    assert [m.id for m in out] == ["DEL4273-UID1-DISPLAY1", "DEL4273-UID1-DISPLAY2"]
    assert [m.name for m in out] == ["Dell U3223QE (1)", "Dell U3223QE (2)"]


def test_clean_name_strips_non_ascii_and_generic_names():
    assert displays._clean_name("DELL™ U3223QE", r"\\.\DISPLAY1", 0) == "DELL U3223QE"
    assert displays._clean_name("Generic PnP Monitor", r"\\.\DISPLAY2", 0) == "Display 2"
    assert displays._clean_name("", r"\\.\DISPLAY3", 1) == "Display 3"


def test_refresh_hides_monitor_setting_for_zero_or_one_monitor():
    displays.refresh([])
    assert displays.CHOICES == ["primary"]
    assert displays.NAMES == {"primary": "Main display"}
    assert not BY_KEY["monitor"].applies_to({})

    displays.refresh([MON_PRIMARY])
    assert displays.CHOICES == ["primary"]
    assert not BY_KEY["monitor"].applies_to({})
    menu = Menu(Settings())
    assert "monitor" not in [r[1].key for r in menu.page_rows("Display") if r[0] == "setting"]


def test_refresh_exposes_choices_hints_and_menu_row_for_multiple_monitors():
    displays.refresh([MON_PRIMARY, MON_SECOND])
    assert displays.CHOICES == ["primary", MON_PRIMARY.id, MON_SECOND.id]
    assert BY_KEY["monitor"].applies_to({})
    assert all(len(t) <= 140 for t in displays.HINTS.values())
    assert all(name.isascii() for name in displays.NAMES.values())

    menu = Menu(Settings())
    display_keys = [r[1].key for r in menu.page_rows("Display") if r[0] == "setting"]
    assert display_keys[0] == "monitor"

    spec_data = next(s for s in specs_json()["specs"] if s["key"] == "monitor")
    assert [c["value"] for c in spec_data["choices"]] == ["primary", MON_PRIMARY.id, MON_SECOND.id]
    assert [c["name"] for c in spec_data["choices"]] == ["Main display", "ROG PG348Q", "DELL U3223QE"]
    assert "HDR on" in spec_data["choices"][2]["hint"]


def test_resolve_monitor_matches_id_device_and_name():
    mons = [MON_PRIMARY, MON_SECOND]
    assert displays.resolve_monitor(None, mons) == MON_PRIMARY
    assert displays.resolve_monitor("primary", mons) == MON_PRIMARY
    assert displays.resolve_monitor(MON_SECOND.id, mons) == MON_SECOND
    assert displays.resolve_monitor(MON_SECOND.id.lower(), mons) == MON_SECOND
    assert displays.resolve_monitor("DISPLAY1", mons) == MON_SECOND
    assert displays.resolve_monitor(r"\\.\display1", mons) == MON_SECOND
    assert displays.resolve_monitor("DELL U3223QE", mons) == MON_SECOND
    assert displays.resolve_monitor("u3223", mons) == MON_SECOND
    assert displays.resolve_monitor("MISSING-ID", mons, fallback=True) == MON_PRIMARY
    assert displays.resolve_monitor("MISSING-ID", mons, fallback=False) is None


@pytest.mark.skipif(sys.platform != "win32", reason="Windows DisplayConfig structs")
def test_display_config_struct_sizes_match_windows_sdk():
    assert ctypes.sizeof(displays._LUID) == 8
    assert ctypes.sizeof(displays._PATH_INFO) == 72
    assert ctypes.sizeof(displays._MODE_INFO) == 64
    assert ctypes.sizeof(displays._SOURCE_DEVICE_NAME) == 84
    assert ctypes.sizeof(displays._TARGET_DEVICE_NAME) == 420


def test_display_hdr_selects_by_device(monkeypatch):
    outs = [
        {"name": r"\\.\DISPLAY2", "monitor": 1, "hdr": False, "max_nits": 270.0},
        {"name": r"\\.\DISPLAY1", "monitor": 2, "hdr": True, "max_nits": 400.0},
    ]
    monkeypatch.setattr(dxgi_present, "outputs", lambda: outs)
    assert dxgi_present.display_hdr(device=r"\\.\display1")["hdr"] is True
    assert dxgi_present.display_hdr(device=r"\\.\DISPLAY2")["hdr"] is False
    assert dxgi_present.display_hdr(device=r"\\.\DISPLAY9") is None


def test_choose_presentation_checks_hdr_on_target_monitor(monkeypatch):
    outs = {
        r"\\.\display2": {"hdr": False},
        r"\\.\display1": {"hdr": True},
    }
    monkeypatch.setattr(io.sys, "platform", "win32")
    monkeypatch.setattr(dxgi_present, "display_hdr", lambda hwnd=None, device=None: outs.get((device or r"\\.\display2").lower()))
    monkeypatch.setattr(io, "KEEP_HDR", True)
    monkeypatch.setattr(io, "FULLSCREEN_STYLE", "none")

    def run_choice(target_mon):
        w = io.GLWindowSink.__new__(io.GLWindowSink)
        w.settings = SimpleNamespace(present="gl", fullscreen=True)
        w.overlay_of, w.dx, w.hdr10, w.keep_hdr, w.fs_style = None, None, False, False, "none"
        w._borderless, w.target_monitor = False, target_mon
        opened = []

        def open_dxgi(width, height, hdr=False, peak=1000.0, keep_hdr=False):
            opened.append((hdr, keep_hdr))
            w.dx = SimpleNamespace(hdr=hdr)

        monkeypatch.setattr(w, "_open_dxgi", open_dxgi)
        w._choose_presentation(1920, 1080, hdr=False, hdr_peak=1000.0)
        return w, opened

    w_sdr, opened_sdr = run_choice(MON_PRIMARY)
    assert opened_sdr == [] and not w_sdr.keep_hdr and w_sdr.fs_style == "none"

    w_hdr, opened_hdr = run_choice(MON_SECOND)
    assert opened_hdr == [(False, True)] and w_hdr.keep_hdr and w_hdr.fs_style == "borderless"

    # Also test primary HDR on, secondary HDR off: choosing secondary stays on OpenGL
    outs[r"\\.\display2"] = {"hdr": True}
    outs[r"\\.\display1"] = {"hdr": False}
    w_hdr2, _ = run_choice(MON_PRIMARY)
    w_sdr2, _ = run_choice(MON_SECOND)
    assert w_hdr2.keep_hdr and not w_sdr2.keep_hdr


def _fake_glfw(monitors, *, has_adapter=True):
    calls = []

    class FakeGlfw:
        DECORATED, FALSE, TRUE = 1, 0, 1

        def get_monitors(self):
            return monitors

        def get_primary_monitor(self):
            return monitors[0]

        def get_monitor_pos(self, m):
            return m["pos"]

        def get_monitor_workarea(self, m):
            return m["workarea"]

        def get_video_mode(self, m):
            return SimpleNamespace(size=SimpleNamespace(width=m["w"], height=m["h"]), refresh_rate=m["hz"])

        def get_window_pos(self, win):
            return win["pos"]

        def get_window_size(self, win):
            return win["size"]

        def set_window_pos(self, win, x, y):
            calls.append(("pos", x, y))
            win["pos"] = (x, y)

        def set_window_size(self, win, w, h):
            calls.append(("size", w, h))
            win["size"] = (w, h)

        def set_window_attrib(self, win, attr, val):
            calls.append(("attrib", attr, val))

        def get_window_monitor(self, win):
            return win.get("monitor")

        def set_window_monitor(self, win, mon, x, y, w, h, hz):
            calls.append(("monitor", mon, x, y, w, h, hz))
            win["monitor"] = mon

        def swap_interval(self, n):
            calls.append(("swap", n))

    if has_adapter:
        FakeGlfw.get_win32_adapter = lambda self, m: m["adapter"]
    return FakeGlfw(), calls


def test_glfw_monitor_matching_and_fallbacks(monkeypatch):
    monkeypatch.setattr(io.sys, "platform", "win32")
    m_pri = {"adapter": r"\\.\DISPLAY2", "pos": (0, 0), "workarea": (0, 0, 3440, 1400), "w": 3440, "h": 1440, "hz": 100}
    m_sec = {"adapter": r"\\.\DISPLAY1", "pos": (3440, 0), "workarea": (3440, 0, 3840, 2120), "w": 3840, "h": 2160, "hz": 60}

    w = io.GLWindowSink.__new__(io.GLWindowSink)
    w.target_monitor = MON_SECOND

    w.glfw, _ = _fake_glfw([m_pri, m_sec], has_adapter=True)
    assert w._glfw_monitor() is m_sec

    # Fallback to (x, y, w, h) when get_win32_adapter is missing
    w.glfw, _ = _fake_glfw([m_pri, m_sec], has_adapter=False)
    assert w._glfw_monitor() is m_sec

    # Fallback to (x, y) when mode size differs (e.g. DPI/scaling mismatch)
    m_sec_scaled = {**m_sec, "w": 3072, "h": 1728}
    w.glfw, _ = _fake_glfw([m_pri, m_sec_scaled], has_adapter=False)
    assert w._glfw_monitor() is m_sec_scaled

    # Fallback to primary when target monitor does not match
    w.target_monitor = displays.Monitor("OTHER", r"\\.\DISPLAY9", "Other", 9999, 0, 1920, 1080, 60, 60)
    assert w._glfw_monitor() is m_pri


def test_window_placement_borderless_and_fullscreen_use_target_monitor(monkeypatch):
    monkeypatch.setattr(io.sys, "platform", "win32")
    m_pri = {"adapter": r"\\.\DISPLAY2", "pos": (0, 0), "workarea": (0, 0, 3440, 1400), "w": 3440, "h": 1440, "hz": 100}
    m_sec = {"adapter": r"\\.\DISPLAY1", "pos": (3440, 0), "workarea": (3440, 0, 3840, 2120), "w": 3840, "h": 2160, "hz": 60}
    glfw, calls = _fake_glfw([m_pri, m_sec])
    win = {"pos": (0, 0), "size": (1600, 900), "monitor": None}

    w = io.GLWindowSink.__new__(io.GLWindowSink)
    w.glfw, w.win, w.overlay_of, w.dx, w.fs_style, w._borderless = glfw, win, None, None, "none", False
    w.w, w.h, w.settings, w.target_monitor = 1920, 1080, SimpleNamespace(vsync=True), MON_SECOND

    placed = w._place_on_monitor(win, 1600, 900)
    # Centered inside secondary monitor workarea (3440, 0, 3840, 2120), pos set before size
    assert placed == (3440 + (3840 - 1600) // 2, (2120 - 900) // 2, 1600, 900)
    assert calls[:2] == [("pos", placed[0], placed[1]), ("size", 1600, 900)]
    w.windowed_pos = placed

    calls.clear()
    w._set_fullscreen(True)
    assert w._borderless is True
    assert ("pos", 3440, 0) in calls and ("size", 3840, 2160) in calls

    calls.clear()
    w._set_fullscreen(False)
    assert w._borderless is False
    assert ("pos", placed[0], placed[1]) in calls and ("size", 1600, 900) in calls

    # Exclusive monitor fullscreen mode also targets m_sec
    calls.clear()
    w.fs_style = "monitor"
    w._set_fullscreen(True)
    assert ("monitor", m_sec, 0, 0, 3840, 2160, 60) in calls

    # Dragging the window onto m_pri later still fullscreens on the monitor it is on (0.3.12 behavior)
    w._set_fullscreen(False)
    win["pos"] = (100, 100)
    calls.clear()
    w.fs_style = "none"
    w._set_fullscreen(True)
    assert w._borderless is True
    assert ("pos", 0, 0) in calls and ("size", 3440, 1440) in calls


def test_autodetect_uses_chosen_monitor(monkeypatch):
    import torch

    from framegen import rtxvsr, trt, vfg
    displays.refresh([MON_PRIMARY, MON_SECOND])
    monkeypatch.setattr(displays, "list_monitors", lambda: [MON_PRIMARY, MON_SECOND])
    monkeypatch.setattr(autodetect.sys, "platform", "win32")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA GeForce RTX 4070")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: SimpleNamespace(total_memory=12 * 2**30))
    monkeypatch.setattr(autodetect, "nvidia_smi", lambda samples=3: {"driver": "621.24", "busy": 0})
    monkeypatch.setattr(trt, "trt_available", lambda: True)
    monkeypatch.setattr(rtxvsr, "unavailable_reason", lambda: None)
    monkeypatch.setattr(vfg, "vfg_unavailable", lambda: "")

    fast_timing = {"full": {"ms": 6.0, "engine": "trt", "estimated": False},
                   "half": {"ms": 4.0, "engine": "torch", "estimated": False}}

    res_pri = autodetect.detect({"monitor": "primary"}, log=lambda m: None, timing=fast_timing)
    assert res_pri["settings"]["hz"] == 90
    row_pri = next(f for f in res_pri["found"] if f["label"] == "Display")
    assert row_pri["value"] == "3440×1440 at 100 Hz"
    assert "ROG PG348Q" in row_pri["detail"] and "HDR on" not in row_pri["detail"]

    res_sec = autodetect.detect({"monitor": MON_SECOND.id}, log=lambda m: None, timing=fast_timing)
    assert res_sec["settings"]["hz"] == 60
    row_sec = next(f for f in res_sec["found"] if f["label"] == "Display")
    assert row_sec["value"] == "3840×2160 at 60 Hz"
    assert "DELL U3223QE" in row_sec["detail"] and "HDR on" in row_sec["detail"]


def test_monitor_setting_unused_with_window_capture():
    spec = BY_KEY["monitor"]
    assert spec.unused_with("window") == "App window capture stays on the monitor the app's window is on."
    for kind in ("chiaki", "capture", "tap", "file", "sim"):
        assert spec.unused_with(kind) is None
