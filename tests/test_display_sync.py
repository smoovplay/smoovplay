"""Presenting in time with the display: the sync plan, measured screen times, settings, and the pacer handing over."""
from __future__ import annotations

import json
import time
from types import SimpleNamespace

import pytest
import torch

from framegen import display_sync, io
from framegen.display_sync import DisplayTimes, plan, refreshes_per_frame
from framegen.io import TICK, Tick
from framegen.pipeline import DuplicateDetector, Pipeline
from framegen.settings import Settings


@pytest.mark.parametrize("refresh,hz,n", [
    (60, 60, 1), (59, 60, 1), (120, 60, 2), (119, 60, 2), (240, 60, 4), (180, 90, 2), (120, 120, 1),
    (240, 120, 2), (144, 60, 0), (144, 120, 0), (165, 60, 0), (144, 90, 0), (60, 120, 0), (300, 60, 0),
])
def test_refreshes_per_frame(refresh, hz, n):
    assert refreshes_per_frame(refresh, hz) == n


def test_plan_modes():
    p = plan(120, 60, vsync=True, vrr=False)
    assert (p.mode, p.interval, p.tear, p.display_paces, p.note) == ("locked", 2, False, True, "")
    p = plan(144, 60, vsync=True, vrr=False)
    assert (p.mode, p.interval, p.tear, p.display_paces) == ("vsync", 1, False, False)
    assert "144 Hz" in p.note and "120 Hz" in p.note and "G-Sync" in p.note
    assert "slower" in plan(60, 120, vsync=True, vrr=False).note
    for refresh in (60, 144):  # G-Sync wins over VSync, on any refresh rate: at once, pacer kept
        p = plan(refresh, 60, vsync=True, vrr=True)
        assert (p.mode, p.interval, p.tear, p.display_paces) == ("vrr", 0, True, False)
    p = plan(144, 60, vsync=False, vrr=False)
    assert (p.mode, p.interval, p.tear, p.display_paces) == ("off", 0, True, False)
    assert "2 refreshes" in plan(120, 60, True, False).describe(120, 60)


def test_display_times_counts_refreshes_and_delay():
    d = DisplayTimes()
    period = 1 / 120
    # Presents 1..4 sent 10 ms before the refresh they went up on; frames held 2, 2, then 3 refreshes.
    up = {1: 100, 2: 102, 3: 104, 4: 107}
    for k, r in up.items():
        d.sent(k, r * period - 0.010)
    for k, r in up.items():  # each sample a refresh after the frame went up
        d.observe(k, r, r + 1, (r + 1) * period, period)
    d.observe(4, 107, 109, 109 * period, period)  # nothing new: ignored
    assert d.held == [2, 2, 3]
    assert all(abs(ms - 10.0) < 0.01 for ms in d.delay_ms), d.delay_ms
    assert abs(1 / d.period_s - 120) < 0.5
    d.sent(6, 0.0)
    d.observe(6, 111, 111, 111 * period, period)  # 5 came and went between samples
    assert d.unseen == 1 and d.held == [2, 2, 3]
    line = d.summary()
    assert "2 refreshes 67%" in line and "3 refreshes 33%" in line and "Present to screen" in line
    assert DisplayTimes().summary() == ""


def test_vsync_and_vrr_default_off_and_saved_values_are_kept(tmp_path):
    assert Settings().vsync is False and Settings().vrr is False
    old = tmp_path / "old.json"
    old.write_text(json.dumps({"vsync": True, "hz": 60}), encoding="utf-8")
    s = Settings.load(old)
    assert s.vsync is True and s.vrr is False  # previously chosen VSync on is preserved
    s.set("vsync", False)
    assert Settings.load(old).vsync is False
    assert json.loads(old.read_text(encoding="utf-8"))["vrr"] is False


def _sink_with(dx, refresh_hz, **st):
    w = io.GLWindowSink.__new__(io.GLWindowSink)
    w.settings = SimpleNamespace(**{"hz": 60, "vsync": True, "vrr": False, "fullscreen": False, **st})
    w.dx, w._borderless = dx, False
    w.glfw = SimpleNamespace(swap_interval=lambda n: intervals.append(n))
    w._sync_key, w._refresh_hz, w._refresh_next, w._sync_noted, w._note_pending = None, 60.0, 0.0, set(), ""
    w._read_refresh = lambda: refresh_hz
    intervals = []
    return w, intervals


def test_window_follows_the_plan(capsys):
    w, intervals = _sink_with(None, 120.0)
    w._update_sync(force=True)
    assert intervals == [2] and not w.paced_by_display  # OpenGL: swap interval only, the pacer keeps its timer
    w.settings.vsync = False
    w._update_sync(force=True)
    assert intervals[-1] == 0
    w, intervals = _sink_with(None, 144.0)
    w._update_sync(force=True)
    assert intervals == [1] and "120 Hz" in w._note_pending
    dx = SimpleNamespace(waitable=1234, refresh_hz=0.0)
    w, _ = _sink_with(dx, 120.0)
    w._update_sync(force=True)
    assert w.paced_by_display and dx.refresh_hz == 120.0
    w.settings.vrr = True
    w._update_sync(force=True)
    assert not w.paced_by_display and w.sync.mode == "vrr"
    assert "G-Sync" in capsys.readouterr().out


def test_pipeline_leaves_pacing_to_the_display():
    class _DummyInterp:
        h = w = 8
        device = "cpu"
        private = False

        def __init__(self):
            self.in_u8 = self.out_u8 = torch.zeros((8, 8, 3), dtype=torch.uint8)

        def load(self, frame):
            self.in_u8 = frame

        def prepare(self, slot):
            pass

        def interpolate(self, slot, t=0.5):
            return self.out_u8

    p = Pipeline.__new__(Pipeline)
    p.interp = _DummyInterp()
    p.detector = DuplicateDetector()
    p.cut_threshold = 25.0
    p.stats = None
    p._out_pinned = torch.empty((8, 8, 3), dtype=torch.uint8)
    p._out_np = p._out_pinned.numpy()

    gaps = []

    class _Sink:
        gpu = False
        paced_by_display = True
        last_wait_s = 0.0

        def write(self, frame, text=None):
            gaps.append(time.perf_counter())

    def burst():  # frames all at once: a timer pacer would space them 16 ms apart
        for i in range(12):
            now = time.perf_counter()
            yield Tick(torch.full((8, 8, 3), i % 200, dtype=torch.uint8), now, i, new_hint=True, t_ready=now)

    t0 = time.perf_counter()
    p.run(burst(), _Sink(), label="RIFE",
          settings=Settings({"hz": 120, "even_out": False, "overlay": False}), live=True)
    assert len(gaps) >= 12 and time.perf_counter() - t0 < 12 * TICK * 0.5


def test_display_sync_module_is_documented():
    assert display_sync.__doc__ and "locked" in display_sync.__doc__
