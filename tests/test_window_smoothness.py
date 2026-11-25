"""Window-capture smoothness, pacing, GPU unchanged-frame skip, and DXGI overlay presentation tests."""
from __future__ import annotations

import threading
import time
from types import SimpleNamespace

import numpy as np
import torch

from framegen import dda, io, winutil
from framegen.io import TICK, Tick
from framegen.pipeline import DuplicateDetector, Pipeline
from framegen.settings import Settings
from framegen.sources import Clock60, source_info


def test_sleep_until_respects_mock_sleep_and_hits_real_deadline():
    slept = []
    t = [10.0]

    def fake_clock():
        return t[0]

    def fake_sleep(dt):
        slept.append(dt)
        t[0] += dt

    end = winutil.sleep_until(10.015, clock=fake_clock, sleep=fake_sleep)
    assert len(slept) == 1 and abs(slept[0] - 0.015) < 1e-9 and abs(end - 10.015) < 1e-9
    assert winutil.sleep_until(10.010, clock=fake_clock, sleep=fake_sleep) == 10.015
    assert len(slept) == 1  # already past deadline

    assert winutil.high_res_timer(True) is True
    try:
        target = time.perf_counter() + 0.008
        woke = winutil.sleep_until(target)
        assert woke >= target and (woke - target) < 0.005
    finally:
        assert winutil.high_res_timer(False) is True


def test_sleep_until_spin_lets_other_threads_run():
    # Clock60's reader must be able to hand in a frame while the beat spins toward its deadline.
    winutil.high_res_timer(True)
    try:
        early = 0
        for _ in range(10):
            woke = []

            def other():
                time.sleep(0.0005)
                woke.append(time.perf_counter())

            t = threading.Thread(target=other)
            deadline = time.perf_counter() + 0.004
            t.start()
            end = winutil.sleep_until(deadline, spin_s=0.010)  # the whole wait is the spin
            t.join()
            assert end >= deadline
            early += woke[0] < deadline
    finally:
        winutil.high_res_timer(False)
    assert early >= 8  # (a bare busy-wait holds the GIL: 0 of 10)


def test_dda_is_unchanged_detects_bit_exact_and_single_pixel_diffs():
    a = torch.zeros((64, 64, 4), dtype=torch.uint8)
    b = a.clone()
    assert not dda.is_unchanged(a, None)
    assert dda.is_unchanged(a, b)
    b[31, 42, 1] = 1
    assert not dda.is_unchanged(a, b)


def test_clock60_phase_locks_misaligned_arrivals_and_reports_lag():
    class _Misaligned:
        info = source_info("window")

        def __init__(self):
            self.closed = False

        def __iter__(self):
            t0 = time.perf_counter()
            for i in range(18):
                # Arrive right before the next beat (~0.85 * period after previous beat)
                target = t0 + (i + 0.85) * (1.0 / 60.0)
                if (w := target - time.perf_counter()) > 0:
                    time.sleep(w)
                yield Tick(np.full((4, 4, 3), i, np.uint8), time.perf_counter(), i)

        def close(self):
            self.closed = True

    winutil.high_res_timer(True)
    try:
        clock = Clock60(_Misaligned(), hz=60.0)
        ages = [t.t_ready - t.t_arrival for t in clock if t.new_hint is None]
        clock.close()
    finally:
        winutil.high_res_timer(False)
    assert len(ages) >= 12
    # Early frames start misaligned (> 0.5 tick), then PLL pulls arrival-to-ready age toward ~0.35 * TICK
    assert ages[-1] < ages[1] or np.median(ages[-4:]) < 0.011

    # Overrun past a beat tags the next tick with clock_lag_ms
    clock_t = [0.0]

    def fake_clock():
        return clock_t[0]

    def fake_sleep(dt):
        clock_t[0] += dt

    class _SlowConsumerSource:
        info = source_info("window")

        def __init__(self):
            self.gate = threading.Event()

        def __iter__(self):
            for i in range(4):
                yield Tick(np.full((2, 2, 3), i, np.uint8), clock_t[0], i)
                if i == 0:
                    self.gate.wait(timeout=2.0)

        def close(self):
            self.gate.set()

    src = _SlowConsumerSource()
    c = Clock60(src, hz=60.0, clock=fake_clock, sleep=fake_sleep)
    it = iter(c)
    first = next(it)
    assert "clock_lag_ms" not in first.extra
    clock_t[0] += 0.025  # consumer blocked > 1 beat before pulling next tick
    src.gate.set()
    second = next(it)
    c.close()
    assert second.extra.get("clock_lag_ms", 0.0) > 5.0


class _DummyInterp:
    h = w = 8
    device = "cpu"
    private = False

    def __init__(self, interp_sleep=0.0):
        self.in_u8 = self.out_u8 = torch.zeros((8, 8, 3), dtype=torch.uint8)
        self.interp_sleep = interp_sleep

    def load(self, frame):
        self.in_u8 = frame

    def prepare(self, slot):
        pass

    def interpolate(self, slot, t=0.5):
        if self.interp_sleep:
            time.sleep(self.interp_sleep)
        return self.out_u8


def _cpu_pipeline(interp):
    p = Pipeline.__new__(Pipeline)
    p.interp = interp
    p.detector = DuplicateDetector()
    p.cut_threshold = 25.0
    p.stats = None
    p._out_pinned = torch.empty((8, 8, 3), dtype=torch.uint8)
    p._out_np = p._out_pinned.numpy()
    return p


def test_live_60hz_pacer_evens_out_present_intervals_without_inflating_busy_ms():
    winutil.high_res_timer(True)
    try:
        present_times = []

        class _Sink:
            gpu = False

            def write(self, frame, text=None):
                present_times.append(time.perf_counter())

        def live_stream():
            t0 = time.perf_counter()
            for i in range(16):
                target = t0 + i * TICK
                if (w := target - time.perf_counter()) > 0:
                    time.sleep(w)
                yield Tick(torch.full((8, 8, 3), (i // 2) % 200, dtype=torch.uint8),
                           target, i, new_hint=(i % 2 == 0), t_ready=time.perf_counter())

        # Each interp tick takes ~10 ms; repeat ticks take ~0 ms. Without 60 Hz pacing, presents alternate ~26 ms and ~6 ms.
        st = _cpu_pipeline(_DummyInterp(interp_sleep=0.010)).run(
            live_stream(), _Sink(), label="RIFE",
            settings=Settings({"hz": 60, "even_out": False, "overlay": False}), live=True,
        )
    finally:
        winutil.high_res_timer(False)

    intervals_ms = np.diff(present_times[4:]) * 1000.0
    assert intervals_ms.min() > 11.0 and intervals_ms.max() < 23.0, intervals_ms
    # Intentional pacer wait (~10 ms on repeat ticks) is excluded from busy_ms and sink_ms
    assert np.median(st.sink_ms) < 1.0 and np.percentile(st.sink_ms, 95) < 3.0
    assert np.median(st.busy_ms) < 15.0


def test_clock_lag_triggers_behind_and_skips_extra_frames_at_90hz():
    class _Sink:
        gpu = False

        def write(self, frame, text=None):
            pass

    def lagged_stream():
        for i in range(10):
            now = time.perf_counter()
            yield Tick(torch.full((8, 8, 3), (i // 2) % 200, dtype=torch.uint8),
                       now, i, new_hint=(i % 2 == 0), t_ready=now,
                       extra={"clock_lag_ms": 14.0} if i >= 2 else {})

    st = _cpu_pipeline(_DummyInterp()).run(
        lagged_stream(), _Sink(), label="RIFE",
        settings=Settings({"hz": 90, "even_out": False, "overlay": False}), live=False,
    )
    assert st.behind >= 8 and st.quarter_skipped >= 3


def test_window_overlay_uses_dxgi_when_present_is_dxgi(monkeypatch):
    for fullscreen in (False, True):
        w = io.GLWindowSink.__new__(io.GLWindowSink)
        w.settings = SimpleNamespace(present="dxgi", fullscreen=fullscreen)
        w.overlay_of = lambda: None
        w.dx, w.hdr10, w.keep_hdr, w.fs_style = None, False, False, "none"
        w._borderless = False
        opened = []
        monkeypatch.setattr(w, "_open_dxgi", lambda *a, **kw: (opened.append((a, kw)), setattr(w, "dx", object())))
        w._choose_presentation(1920, 1080, hdr=False, hdr_peak=1000.0)
        assert len(opened) == 1 and w.dx is not None


def test_async_cuda_event_handoff_propagates_through_clock60_and_pipeline(monkeypatch):
    # 1. Clock60 preserves ready_event on both fresh and repeated beats
    ev0, ev1 = object(), object()
    clock_t = [0.0]

    def fake_clock():
        return clock_t[0]

    def fake_sleep(dt):
        clock_t[0] += dt

    class _EventSource:
        info = source_info("window")

        def __init__(self):
            self.gate = threading.Event()

        def __iter__(self):
            yield Tick(torch.zeros((8, 8, 3), dtype=torch.uint8), clock_t[0], 0, ready_event=ev0)
            self.gate.wait(timeout=2.0)
            yield Tick(torch.ones((8, 8, 3), dtype=torch.uint8), clock_t[0], 1, ready_event=ev1)

        def close(self):
            self.gate.set()

    src = _EventSource()
    c = Clock60(src, hz=60.0, clock=fake_clock, sleep=fake_sleep)
    it = iter(c)
    t0 = next(it)
    t1 = next(it)  # repeat beat of first frame
    src.gate.set()
    # Wait briefly for reader thread to publish second frame
    for _ in range(50):
        t2 = next(it)
        if t2.ready_event is ev1:
            break
        time.sleep(0.002)
    c.close()
    assert t0.ready_event is ev0 and t1.ready_event is ev0 and t2.ready_event is ev1

    # 2. Pipeline calls current_stream().wait_event(tick.ready_event) before interp.load(tick.frame)
    order = []

    class _FakeStream:
        def wait_event(self, ev):
            order.append(("wait", ev))

    class _RecordingInterp(_DummyInterp):
        def load(self, frame):
            order.append(("load", int(frame[0, 0, 0])))
            super().load(frame)

    monkeypatch.setattr(torch.cuda, "current_stream", lambda *a, **kw: _FakeStream())

    def event_ticks():
        now = time.perf_counter()
        yield Tick(torch.full((8, 8, 3), 7, dtype=torch.uint8), now, 0, new_hint=True, ready_event=ev0)
        yield Tick(torch.full((8, 8, 3), 9, dtype=torch.uint8), now + TICK, 1, new_hint=False, ready_event=ev1)

    _cpu_pipeline(_RecordingInterp()).run(event_ticks(), enabled=False, live=False)
    assert order == [("wait", ev0), ("load", 7), ("wait", ev1), ("load", 9)]


def test_async_cuda_event_handoff_real_gpu_stream():
    if not torch.cuda.is_available():
        return
    producer = torch.cuda.Stream()
    bgra = torch.empty((16, 16, 4), dtype=torch.uint8, device="cuda")
    with torch.cuda.stream(producer):
        bgra[..., 0].fill_(12)
        bgra[..., 1].fill_(34)
        bgra[..., 2].fill_(56)
        bgra[..., 3].fill_(255)
        frame = dda.to_bgr(bgra, 8, 8)
        ready = torch.cuda.Event()
        ready.record(producer)

    loaded = []

    class _GpuInterp(_DummyInterp):
        device = "cuda"

        def load(self, f):
            loaded.append(f.clone())
            self.in_u8 = f

    def one_tick():
        yield Tick(frame, time.perf_counter(), 0, new_hint=True, ready_event=ready)

    _cpu_pipeline(_GpuInterp()).run(one_tick(), enabled=False, live=False)
    torch.cuda.current_stream().synchronize()
    assert len(loaded) == 1
    assert loaded[0][0, 0].cpu().tolist() == [12, 34, 56]


def test_dda_close_synchronizes_stream():
    synced = []

    class _FakeStream:
        def synchronize(self):
            synced.append(True)

    cap = dda.DdaCapture.__new__(dda.DdaCapture)
    cap._stream = _FakeStream()
    cap._prev_bgra = None
    cap.dup = cap.res = cap.texture = cap.context = cap.device = cap.output = cap.adapter = cap.factory = None
    cap._held = False
    cap.close()
    assert synced == [True] and cap._stream is None
