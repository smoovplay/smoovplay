"""Window capture (framegen/window.py, Experimental) with stand-ins for Windows and Desktop Duplication: no real
window or screen is read. Also the learned repeat rule it uses (cadence.DuplicateDetector(learn=True))."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

from framegen import window
from framegen.cadence import DuplicateDetector
from framegen.sources import EXPERIMENTAL, KINDS, list_sources, make_detector
from framegen.window import AppWindow, Grab, Rect, WindowSource, graph, pick, plan_grab

MON = Rect(0, 0, 2560, 1440)
OUTPUTS = ["\\\\.\\DISPLAY1", "\\\\.\\DISPLAY2"]


# ----------------------------------------------------------------------------- where to grab
def test_grab_is_the_picture_area_relative_to_its_monitor():
    second = Rect(2560, 0, 1920, 1080)
    g = plan_grab(Rect(2600, 100, 1280, 720), second, OUTPUTS[1], OUTPUTS, 144)
    assert g == Grab(1, Rect(40, 100, 1280, 720), 144)


def test_grab_keeps_the_part_on_screen_with_even_sizes():
    g = plan_grab(Rect(-101, 50, 1281, 721), MON, OUTPUTS[0], OUTPUTS, 60)
    assert g.region == Rect(0, 50, 1180, 720)


def test_grab_explains_what_it_cant_do():
    assert "main graphics adapter" in plan_grab(Rect(0, 0, 1280, 720), MON, "\\\\.\\DISPLAY9", OUTPUTS, 60)
    assert "too little" in plan_grab(Rect(2500, 0, 1280, 720), MON, OUTPUTS[0], OUTPUTS, 60)


def test_grab_looks_at_least_60_times_a_second():
    assert plan_grab(Rect(0, 0, 1280, 720), MON, OUTPUTS[0], OUTPUTS, 30).hz == 60


def test_graph_asks_for_changes_only_at_a_fixed_size():
    spec = graph(Grab(1, Rect(40, 100, 1280, 720), 144), 1920, 1080)
    assert spec.startswith("ddagrab=output_idx=1:framerate=144:video_size=1280x720:offset_x=40:offset_y=100")
    assert "dup_frames=0" in spec and "draw_mouse=0" in spec
    assert spec.endswith("scale=1920:1080:flags=bilinear,format=bgr24")


# ----------------------------------------------------------------------------- picking the window
WINDOWS = [AppWindow(11, "PS Remote Play", "remoteplay.exe", 1280, 720),
           AppWindow(12, "Xbox", "xboxpcapp.exe", 1920, 1080),
           AppWindow(13, "Cloud gaming - Browser", "msedge.exe", 1600, 900),
           AppWindow(14, "Other tab - Browser", "msedge.exe", 800, 600)]


@pytest.mark.parametrize("query, hwnd", [("ps-remote-play", 11), ("xbox", 12), ("exe:XboxPcApp.exe", 12),
                                         ("exe:remoteplay", 11), ("hwnd:13", 13), ("cloud gaming", 13),
                                         ("exe:msedge.exe", 13), ("moonlight", None), ("hwnd:99", None),
                                         ("hwnd:14|exe:msedge.exe", 14), ("hwnd:99|exe:msedge.exe", 13)])
def test_pick(query, hwnd):
    w = pick(query, WINDOWS)
    assert (w.hwnd if w else None) == hwnd


def test_known_apps_have_names():
    assert WINDOWS[0].app == "PS Remote Play" and WINDOWS[2].app == "Edge"
    assert AppWindow(1, "x", "someplayer.exe", 640, 360).app == "someplayer"


def test_the_xbox_app_is_found_by_either_of_its_programs():
    # Its Start menu entry runs XboxPcAppCE.exe; the older Store version in the same package is XboxPcApp.exe
    desktop = AppWindow(21, "Xbox", "xboxpcappce.exe", 1920, 1080)
    assert desktop.app == "Xbox app"
    assert pick("xbox", [desktop]) is desktop
    assert pick("exe:XboxPcApp.exe", [desktop]) is desktop  # what an older pick was saved as
    assert pick("exe:xboxpcappce", WINDOWS).hwnd == 12


class _Shell:
    """Stand-ins for the calls find_windows makes: a desktop of top-level windows, some of them Store app frames
    (ApplicationFrameHost.exe) with the app's own window inside."""
    TOP = {  # hwnd: (program, title, children as (hwnd, class, program))
        100: ("applicationframehost.exe", "Xbox", [(101, "Windows.UI.Core.CoreWindow", "xboxpcapp.exe")]),
        200: ("applicationframehost.exe", "Settings", [(201, "Windows.UI.Core.CoreWindow", "systemsettings.exe")]),
        300: ("applicationframehost.exe", "Suspended app", []),  # Windows took the app's window out of the frame
        400: ("xboxpcappce.exe", "Xbox", []),
        500: ("explorer.exe", "Documents", []),
        600: ("applicationframehost.exe", "Store game",
              [(601, "ApplicationFrameTitleBarWindow", "applicationframehost.exe"), (602, "SomeOtherClass", "storegame.exe")]),
    }
    IMMERSIVE: set = set()  # frames in full screen: in Windows' own layer, which EnumWindows doesn't list
    AUMID: dict = {}  # frames' AppUserModelIDs, as the shell has them

    def __init__(self, monkeypatch):
        monkeypatch.setattr(window, "_aumid", lambda h: self.AUMID.get(int(h), ""))
        exe = {h: e for h, (e, _, _) in self.TOP.items()}
        exe |= {c: e for _, _, kids in self.TOP.values() for c, _, e in kids}
        title = {h: t for h, (_, t, _) in self.TOP.items()}

        def find_child(parent, after, cls, name):
            if parent is None:  # top-level windows of a class, wherever they are: here only Store app frames
                frames = [h for h, (e, _, _) in self.TOP.items() if e == "applicationframehost.exe"]
                assert cls == "ApplicationFrameWindow"
                i = frames.index(after) + 1 if after in frames else 0
                return frames[i] if i < len(frames) else None
            kids = self.TOP.get(parent, (None, None, []))[2]
            if cls is not None:
                return next((c for c, k, _ in kids if k == cls), None)
            hs = [c for c, _, _ in kids]
            i = hs.index(after) + 1 if after in hs else 0
            return hs[i] if i < len(hs) else None

        def set_pid(h, ref):
            ref._obj.value = h  # a process per window is enough here
        u32 = type("U32", (), {
            "EnumWindows": staticmethod(lambda each, _: [each(h, 0) for h in self.TOP if h not in self.IMMERSIVE]),
            "IsWindowVisible": staticmethod(lambda h: True), "GetWindow": staticmethod(lambda h, rel: None),
            "GetWindowLongW": staticmethod(lambda h, i: 0), "IsIconic": staticmethod(lambda h: False),
            "GetWindowThreadProcessId": staticmethod(set_pid),
            "GetWindowTextLengthW": staticmethod(lambda h: len(title[h])),
            "GetWindowTextW": staticmethod(lambda h, buf, n: setattr(buf, "value", title[h])),
            "FindWindowExW": staticmethod(find_child)})()
        dwm = type("Dwm", (), {"DwmGetWindowAttribute": staticmethod(lambda *a: 0)})()  # nothing cloaked
        monkeypatch.setattr(window, "_api", lambda: (u32, None, dwm))
        monkeypatch.setattr(window, "_exe_of", lambda h, cache: exe.get(int(h), ""))
        monkeypatch.setattr(window, "client_rect", lambda h: Rect(0, 0, 1280, 720))


@pytest.mark.skipif(sys.platform != "win32", reason="listing windows (EnumWindows) is Windows-only")
def test_a_store_apps_frame_is_listed_as_the_app_inside_it(monkeypatch):
    _Shell(monkeypatch)
    found = {w.hwnd: w for w in window.find_windows()}
    assert set(found) == {100, 400, 600}  # not Settings, a frame with no app in it, or Explorer
    assert found[100].exe == "xboxpcapp.exe" and found[100].app == "Xbox app"  # the frame's handle, the app's name
    assert found[400].app == "Xbox app" and found[600].exe == "storegame.exe"
    assert [w.hwnd for w in window.find_windows()][-1] == 600  # known apps first
    assert pick("xbox").hwnd in (100, 400)
    left: dict = {}
    window.find_windows(left)
    assert left == {}  # every frame was listed by EnumWindows: nothing in full screen


class _FullScreenShell(_Shell):
    """The Xbox app in full screen (Shift+Win+Enter, or its own button): its frame is only found by FindWindowEx."""
    TOP = {
        100: ("applicationframehost.exe", "Xbox", [(101, "Windows.UI.Core.CoreWindow", "xboxpcapp.exe")]),
        300: ("applicationframehost.exe", "Suspended app", []),
        500: ("explorer.exe", "Documents", []),
        600: ("applicationframehost.exe", "Store game", [(602, "Windows.UI.Core.CoreWindow", "storegame.exe")]),
    }
    IMMERSIVE = {100, 300, 600}
    AUMID = {100: "Microsoft.GamingApp_8wekyb3d8bbwe!Microsoft.Xbox.AppL", 300: "Microsoft.WindowsCalculator_8wekyb3d8bbwe!App"}


class _SwitchedAwayShell(_FullScreenShell):
    """...and then switched away from (to the launcher): the frame stays in full screen, but the shell suspends the
    app and takes its window out of the frame, so only the frame's AppUserModelID says which app it is."""
    TOP = {**_FullScreenShell.TOP, 100: ("applicationframehost.exe", "Xbox", [])}


@pytest.mark.skipif(sys.platform != "win32", reason="listing windows (EnumWindows) is Windows-only")
@pytest.mark.parametrize("shell", [_FullScreenShell, _SwitchedAwayShell])
def test_a_store_app_in_full_screen_is_noted_not_listed(monkeypatch, shell):
    shell(monkeypatch)
    left: dict = {}
    assert window.find_windows(left) == []
    # Only streaming apps are noted (never another app, here the Store game), and a suspended app that isn't one
    # (the Calculator) isn't either
    assert left == {"xboxpcapp.exe": {window.FULL_SCREEN}}


@pytest.mark.skipif(sys.platform != "win32", reason="listing windows (EnumWindows) is Windows-only")
def test_play_says_to_leave_full_screen(monkeypatch):
    _FullScreenShell(monkeypatch)
    with pytest.raises(RuntimeError, match="the Xbox app is in full screen.*Shift\\+Win\\+Enter"):
        window.open_window("hwnd:100|exe:xboxpcapp.exe")
    with pytest.raises(RuntimeError, match="the Xbox app is in full screen"):
        window.open_window("exe:XboxPcAppCE")  # the Xbox app's other program: the same app
    with pytest.raises(RuntimeError, match="no app window matches"):  # another app was asked for
        window.open_window("exe:moonlight.exe")


# ----------------------------------------------------------------------------- the source
class _Frame:
    def __init__(self, i, w, h):
        self.i, self.w, self.h, self.time = i, w, h, i / 60

    def to_ndarray(self):
        return np.full((self.h, self.w, 3), self.i % 256, np.uint8)


class _Container:
    def __init__(self, spec, frames, w, h):
        self.spec, self.frames, self.w, self.h, self.closed = spec, frames, w, h, False

    def decode(self, video=0):
        for i in self.frames:
            yield _Frame(i, self.w, self.h)

    def close(self):
        self.closed = True


class _Desk:
    """Stand-ins for the Windows calls: the window's place, whether it's still there, the monitors."""

    def __init__(self, monkeypatch):
        self.rects = [Rect(100, 100, 1280, 720)]
        self.alive = True
        self.opened: list[_Container] = []
        u32 = type("U32", (), {"IsWindow": staticmethod(lambda h: self.alive)})()
        monkeypatch.setattr(window, "_api", lambda: (u32, None, None))
        monkeypatch.setattr(window, "client_rect", lambda h: self.rects[0])
        monkeypatch.setattr(window, "monitor_of", lambda h: (MON, OUTPUTS[0], 60))
        monkeypatch.setattr(window, "dda_outputs", lambda: OUTPUTS)
        monkeypatch.setattr(window, "CHECK_S", 0.0)  # look at the window after every frame

    def open(self, spec):
        n = len(self.opened)
        c = _Container(spec, range(n * 10, n * 10 + 3), 640, 360)
        self.opened.append(c)
        if n == 0:
            self.rects.append(Rect(300, 100, 1280, 720))  # moved after the first frame
        if n == 1:
            self.alive = False  # closed during the second capture
        return c


def test_source_follows_a_moving_window_and_ends_when_it_closes(monkeypatch):
    desk = _Desk(monkeypatch)
    rects = iter([Rect(100, 100, 1280, 720), Rect(100, 100, 1280, 720), Rect(300, 100, 1280, 720)])
    monkeypatch.setattr(window, "client_rect", lambda h: next(rects, Rect(300, 100, 1280, 720)))
    src = WindowSource(WINDOWS[0], size=(640, 360), open_fn=desk.open)
    assert src.info.kind == "window" and src.info.rate == "variable"
    assert src.where() == (Rect(100, 100, 1280, 720), MON)

    ticks = []
    for tick in src:
        ticks.append(tick)
        if len(ticks) > 20:
            break
    assert len(desk.opened) == 2  # restarted once, after the move
    assert "offset_x=100" in desk.opened[0].spec and "offset_x=300" in desk.opened[1].spec
    assert all(c.closed for c in desk.opened)
    assert src.restarts == 1
    assert [t.frame.shape for t in ticks[:1]] == [(360, 640, 3)]
    assert [t.index for t in ticks] == list(range(len(ticks)))
    assert src.where()[0] == Rect(300, 100, 1280, 720)


def test_overlay_covers_what_is_captured_when_the_window_hangs_off_the_screen(monkeypatch):
    desk = _Desk(monkeypatch)
    monkeypatch.setattr(window, "client_rect", lambda h: Rect(-101, 50, 1281, 721))
    src = WindowSource(WINDOWS[0], size=(640, 360), open_fn=desk.open)
    assert src.where() == (Rect(0, 50, 1180, 720), MON)  # the on-screen part, even sizes: the frame is exactly that


def test_overlay_draws_the_frame_over_the_whole_window():
    # The frame is the app's window squeezed to the stream's size: it's stretched back to fill the overlay, not
    # letterboxed to 16:9 (which squeezed a browser window's picture and covered part of it with bars)
    from framegen.io import GLWindowSink
    sink = GLWindowSink.__new__(GLWindowSink)
    sink.w, sink.h, sink.dx = 1920, 1080, None
    sink.glfw = type("G", (), {"get_framebuffer_size": staticmethod(lambda win: (2560, 1305))})()
    sink.win, sink.overlay_of = None, lambda: None
    assert sink._layout()[3:] == (2560, 1305, 0, 0)
    sink.overlay_of = None  # its own window: letterboxed, aspect kept
    assert sink._layout()[3:] == (2320, 1305, 120, 0)


def test_source_refuses_a_minimised_window(monkeypatch):
    _Desk(monkeypatch)
    monkeypatch.setattr(window, "client_rect", lambda h: None)
    with pytest.raises(RuntimeError, match="minimised"):
        WindowSource(WINDOWS[0], open_fn=lambda spec: None)


def test_source_keeps_trying_for_a_while_then_gives_up(monkeypatch, capsys):
    """A capture that fails is tried again for RETRY_S (a screen mode change can take seconds), more slowly as it
    goes, saying each reason once (issue 21: it gave up after five tries in two seconds)."""
    desk = _Desk(monkeypatch)
    monkeypatch.setattr(window, "CHECK_S", 0.5)
    clock = [0.0]
    monkeypatch.setattr(window.time, "perf_counter", lambda: clock[0])
    monkeypatch.setattr(window.time, "sleep", lambda s: clock.__setitem__(0, clock[0] + s))
    tries = []

    def broken(spec):
        tries.append(clock[0])
        raise OSError("[Errno 542398533] Generic error in an external library")

    src = WindowSource(WINDOWS[0], open_fn=broken)
    assert list(src) == []
    assert desk.alive
    assert tries[-1] - tries[0] >= window.RETRY_S and len(tries) <= 10
    out = capsys.readouterr().out
    assert out.count("capture stopped") == 1 and f"trying again for up to {window.RETRY_S:.0f} s" in out
    assert out.count("keeps failing") == 1


def test_capture_errors_say_windows_reason_in_words():
    from types import SimpleNamespace

    from framegen.dda import CaptureLost
    from framegen.window import capture_error
    generic = OSError("[Errno 542398533] Generic error in an external library")
    said = capture_error(generic, "AcquireNextFrame failed: 887a0026")  # FFmpeg's line has Windows' code
    assert said.startswith("AcquireNextFrame failed: 887a0026: ") and "changed mode" in said
    assert "too many programs" in capture_error(RuntimeError("DuplicateOutput failed (HRESULT 0x887A0022)"))
    assert "changed mode" in capture_error(CaptureLost("Desktop Duplication lost access"))
    assert capture_error(generic) == str(generic)  # nothing more known: as it came
    # FFmpeg's error lines from the capture just stopped: ddagrab's, over other errors and warnings
    logs = [(16, "Parsed_scale_2", "something else"), (16, "Parsed_ddagrab_0", "AcquireNextFrame failed: 887a0026\n"),
            (24, "Parsed_ddagrab_0", "a warning")]
    me = SimpleNamespace(_ffmpeg_logs=logs)
    assert WindowSource._ffmpeg_error(me) == "AcquireNextFrame failed: 887a0026"
    assert me._ffmpeg_logs == [] and WindowSource._ffmpeg_error(me) == ""


# ----------------------------------------------------------------------------- the source list and the detector
def test_window_is_an_experimental_source():
    assert "window" in EXPERIMENTAL and KINDS["window"].rate == "variable"
    assert not KINDS["window"].test_mode
    assert any(s["kind"] == "window" and not s["hidden"] and s["experimental"] for s in list_sources())
    assert make_detector(KINDS["window"]).learn
    assert not make_detector(KINDS["capture"]).learn


def _decoded_30_in_60(n: int, seed: int = 0):
    """A 30 fps game in a 60 Hz app window: new frames (moving) on even ticks, and repeats that are the frame before
    with a decoding artefact in one 16x16 block. As measured on a decoded stream, a repeat's average difference is
    tiny while its worst block is over the fixed threshold."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (128, 128, 3)).astype(np.float32)
    frames, truth = [], []
    for i in range(n):
        if i % 2 == 0:
            base = np.roll(base, 8, axis=1)  # motion
            frame, new = base, True
        else:
            frame = base.copy()
            by, bx = rng.integers(0, 8, 2) * 16
            frame[by:by + 16, bx:bx + 16] += rng.choice([-5.0, 5.0])  # one blocky artefact
            new = False
        frames.append(torch.from_numpy(np.clip(frame, 0, 255).astype(np.uint8)))
        truth.append(new)
    return frames, truth


def test_learned_rule_tells_noisy_repeats_from_new_frames():
    frames, truth = _decoded_30_in_60(80)
    fixed = DuplicateDetector()
    learned = DuplicateDetector(learn=True)
    got_fixed = [fixed.update(f).is_new for f in frames]
    got_learned = [learned.update(f).is_new for f in frames]
    warm = DuplicateDetector.LEARN_MIN + 1
    agree = lambda got: sum(g == t for g, t in zip(got[warm:], truth[warm:])) / len(truth[warm:])  # noqa: E731
    assert agree(got_fixed) < 0.7  # the artefacts trip the fixed line
    assert agree(got_learned) == 1.0


def test_learned_rule_falls_back_without_two_bands():
    still = torch.zeros((64, 64, 3), dtype=torch.uint8)
    d = DuplicateDetector(learn=True)
    assert [d.update(still).is_new for _ in range(20)][1:] == [False] * 19  # nothing moves: repeats by the fixed rule


# ----------------------------------------------------------------------------- the game's rhythm
def _decoded(n: int, period: int = 2, start: int = 0, unclear=(), hitch=(), fast_from=None, seed: int = 1):
    """Like _decoded_30_in_60, for a game at 60/period fps whose first new frame is `start`. At `unclear` ticks the
    pictures alone mislead: a repeat with artefacts in 9 blocks (just over the line) or a new frame that barely
    changed (just under it). `hitch`: a new frame that comes a frame late (an exact repeat first), moving the beat.
    From `fast_from` on every frame is new (a 60 fps menu)."""
    rng = np.random.default_rng(seed)
    base = rng.integers(0, 256, (128, 128, 3)).astype(np.float32)
    frames, truth = [], []
    for i in range(n):
        new = (i - start) % period == 0 or (fast_from is not None and i >= fast_from)
        if i in hitch:
            new, start = False, i + 1
            frame = base
        elif new and i in unclear:
            base = base.copy()
            for b in rng.choice(64, 22, replace=False):
                by, bx = divmod(int(b), 8)
                base[by * 16:by * 16 + 16, bx * 16:bx * 16 + 16] += 1.0
            frame = base
        elif new:
            base = np.roll(base, 8, axis=1)
            frame = base
        else:
            frame = base.copy()
            for b in rng.choice(64, 9 if i in unclear else 1, replace=False):
                by, bx = divmod(int(b), 8)
                frame[by * 16:by * 16 + 16, bx * 16:bx * 16 + 16] += rng.choice([-5.0, 5.0])
        frames.append(torch.from_numpy(np.clip(frame, 0, 255).astype(np.uint8)))
        truth.append(new)
    return frames, truth


class _NoRhythm(DuplicateDetector):
    RHYTHM_PERIODS = ()


@pytest.mark.parametrize("period, start", [(2, 0), (2, 1), (3, 0), (3, 1), (3, 2)])
def test_rhythm_calls_unclear_frames_as_the_game_goes(period, start):
    n = 120
    unclear = {i for i in range(60, n) if i % 7 == 0}
    frames, truth = _decoded(n, period, start, unclear=unclear)
    alone, kept = _NoRhythm(learn=True), DuplicateDetector(learn=True)
    got_alone = [alone.update(f).is_new for f in frames]
    got_kept = [kept.update(f).is_new for f in frames]
    assert kept.rhythm is not None and kept.rhythm[0] == period
    assert sum(got_alone[i] != truth[i] for i in unclear) >= len(unclear) // 2  # the pictures alone get these wrong
    assert [got_kept[i] for i in range(40, n)] == truth[40:]  # the rhythm gets them right
    assert kept.held >= len(unclear) // 2


def test_rhythm_lets_go_of_a_hitch_and_finds_the_new_beat():
    frames, truth = _decoded(150, hitch={60}, unclear={i for i in range(100, 150) if i % 7 == 0})  # a frame a tick late
    assert truth[58:64] == [True, False, False, True, False, True]
    d = DuplicateDetector(learn=True)
    got = [d.update(f).is_new for f in frames]
    assert got[60] is False  # an exact repeat is a repeat, whatever the rhythm says
    assert d.rhythm is not None  # found again
    assert got[40:] == truth[40:]


def test_rhythm_lets_go_when_the_game_speeds_up():
    frames, truth = _decoded(140, fast_from=80)
    d = DuplicateDetector(learn=True)
    got = [d.update(f).is_new for f in frames]
    assert all(got[82:])  # every frame of the 60 fps menu is new
    assert d.rhythm is None or d.rhythm[0] == 1


# ----------------------------------------------------------------------------- the GPU path
class _Cap:
    def __init__(self, frames):
        self.frames, self.closed = list(frames), False

    def grab(self, timeout_ms):
        if not self.frames:
            return None
        i = self.frames.pop(0)
        return (np.full((360, 640, 3), i, np.uint8), float(i), float(i) - 0.01) if i is not None else None

    def close(self):
        self.closed = True


def test_gpu_path_hands_on_its_frames_and_skips_quiet_moments(monkeypatch):
    desk = _Desk(monkeypatch)
    monkeypatch.setattr(window, "CHECK_S", 10.0)
    caps = []

    def open_gpu(g, w, h):
        assert (g.region, w, h) == (Rect(100, 100, 1280, 720), 640, 360)
        caps.append(_Cap([1, None, 2, None, None, 3]))
        return caps[-1]

    src = WindowSource(WINDOWS[0], size=(640, 360), gpu=open_gpu)
    assert src.path == "GPU"
    ticks = []
    for t in src:
        ticks.append(t)
        if len(ticks) == 3:
            src.close()
    assert [int(t.frame[0, 0, 0]) for t in ticks] == [1, 2, 3]
    assert [t.t_source for t in ticks] == [0.99, 1.99, 2.99]
    assert [t.ready_event for t in ticks] == [None, None, None]
    assert caps[0].closed and not desk.opened

    # 4-tuple grab() with a CUDA ready event propagates ready_event onto Tick
    ev = object()

    class _EventCap(_Cap):
        def grab(self, timeout_ms):
            got = super().grab(timeout_ms)
            return (*got, ev) if got is not None else None

    src2 = WindowSource(WINDOWS[0], size=(640, 360), gpu=lambda g, w, h: _EventCap([5]))
    t = next(iter(src2))
    src2.close()
    assert int(t.frame[0, 0, 0]) == 5 and t.ready_event is ev


def test_gpu_path_falls_back_to_the_cpu(monkeypatch):
    from framegen.dda import GpuCaptureUnavailable
    desk = _Desk(monkeypatch)

    def no_gpu(g, w, h):
        raise GpuCaptureUnavailable("the monitor is rotated")

    src = WindowSource(WINDOWS[0], size=(640, 360), open_fn=desk.open, gpu=no_gpu)
    ticks = list(src)
    assert src.path == "CPU" and len(desk.opened) >= 1 and ticks


def test_to_bgr_scales_on_the_gpu_or_cpu():
    from framegen.dda import to_bgr
    bgra = torch.zeros((720, 1280, 4), dtype=torch.uint8)
    bgra[..., 0], bgra[..., 1], bgra[..., 2], bgra[..., 3] = 10, 20, 30, 255
    out = to_bgr(bgra, 640, 360)
    assert out.shape == (360, 640, 3) and out.dtype == torch.uint8
    assert out[0, 0].tolist() == [10, 20, 30]
    assert to_bgr(bgra, 1280, 720).shape == (720, 1280, 3)


def test_open_window_scales_to_the_streams_size(monkeypatch):
    _Desk(monkeypatch)
    monkeypatch.setattr(window, "find_windows", lambda left_out=None: list(WINDOWS))
    monkeypatch.setattr(window, "_cuda", lambda: False)
    src = window.open_window("ps-remote-play")
    assert (src.w, src.h) == window.SIZE == (1920, 1080)



@pytest.mark.skipif(sys.platform != "win32", reason="listing windows (EnumWindows) is Windows-only")
def test_a_running_streaming_app_left_off_the_list_says_why(monkeypatch):
    shell = _Shell(monkeypatch)
    u32 = window._api()[0]
    hidden, cloaked = {400}, set()
    monkeypatch.setattr(u32, "IsWindowVisible", lambda h: h not in hidden, raising=False)
    monkeypatch.setattr(window._api()[2], "DwmGetWindowAttribute",
                        lambda h, attr, ref, size: setattr(ref._obj, "value", int(h in cloaked)) or 0, raising=False)
    left_out: dict = {}
    window.find_windows(left_out)
    assert left_out == {}  # the Xbox app's Store version is listed, so nothing is missing

    monkeypatch.setitem(shell.TOP, 100, ("applicationframehost.exe", "Xbox", []))  # ...and now it's suspended
    window.find_windows(left_out)
    assert left_out == {"xboxpcappce.exe": {"no window on screen"}}  # in the tray

    hidden.clear()
    cloaked.add(400)
    left_out = {}
    window.find_windows(left_out)
    assert left_out == {"xboxpcappce.exe": {"on another desktop, or not shown yet"}}
    assert "explorer.exe" not in str(left_out) and "Documents" not in str(left_out)  # other apps are never noted
