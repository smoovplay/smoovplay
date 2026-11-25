"""Tests for SmoovPlay-side latency wins:
1. No pre-real-frame sleep_until pacer holdback at 60 Hz (passthrough / framegen-off / post-jitter).
2. 0-frame DPB reorder buffering verification in ChiakiSource._new_decoder.
3. Exact math equivalence and fallback safety of the fused single-kernel NV12 -> BGR CUDA converter.
"""
from __future__ import annotations

import time

import numpy as np
import pytest
import torch

from framegen import chiaki_tap, pipeline
from framegen.io import TICK, Tick
from framegen.pipeline import Pipeline
from framegen.settings import Settings


class _DummyInterp:
    device = "cpu"
    model_name = "dummy"
    backend = "cpu"
    steady_hud = False
    h = 8
    w = 8

    def __init__(self, interp_sleep: float = 0.0) -> None:
        self.in_u8 = None
        self._interp_sleep = interp_sleep
        self.slots = [None, None]

    def load(self, frame) -> None:
        self.in_u8 = frame if isinstance(frame, torch.Tensor) else torch.from_numpy(frame)

    def prepare(self, slot: int) -> None:
        self.slots[slot] = self.in_u8

    def interpolate(self, slot: int, t: float = 0.5):
        if self._interp_sleep:
            time.sleep(self._interp_sleep)
        return self.in_u8.float() / 255.0

    def same_as(self, prev) -> bool:
        return False


def _cpu_pipeline(interp=None) -> Pipeline:
    from framegen.cadence import DuplicateDetector
    p = Pipeline.__new__(Pipeline)
    p.interp = interp or _DummyInterp()
    p.detector = DuplicateDetector()
    p.cut_threshold = 25.0
    p.stats = None
    p.waiting = None
    p._out_pinned = torch.empty((8, 8, 3), dtype=torch.uint8)
    p._out_np = p._out_pinned.numpy()
    return p


def test_live_60hz_passthrough_has_no_pacer_holdback_after_jitter(monkeypatch):
    """In 60 fps passthrough (all new real frames), a jitter spike must NOT latch sleep_until holdback."""
    sleeps: list[float] = []
    real_sleep = pipeline.sleep_until

    def spy_sleep(deadline: float, *args, **kwargs):
        sleeps.append(deadline - time.perf_counter())
        return real_sleep(deadline, *args, **kwargs)

    monkeypatch.setattr(pipeline, "sleep_until", spy_sleep)

    holdbacks_ms: list[float] = []

    class _Sink:
        gpu = False

        def write(self, frame, text=None):
            holdbacks_ms.append((time.perf_counter() - last_ready[0]) * 1000.0)

    last_ready = [0.0]

    def stream_with_jitter():
        t0 = time.perf_counter()
        for i in range(12):
            # Frame 2 arrives +11 ms late (network jitter spike); subsequent frames arrive on time.
            jitter = 0.011 if i == 2 else 0.0
            target = t0 + i * TICK + jitter
            if (w := target - time.perf_counter()) > 0:
                time.sleep(w)
            now = time.perf_counter()
            last_ready[0] = now
            yield Tick(torch.full((8, 8, 3), (i * 19) % 250, dtype=torch.uint8),
                       now, i, new_hint=True, t_ready=now)

    _cpu_pipeline().run(
        stream_with_jitter(), _Sink(), label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False, "interp": True}),
        live=True,
    )
    assert sleeps == [], f"unexpected 60 Hz pacer sleeps on 60 fps real frames: {sleeps}"
    assert max(holdbacks_ms[3:]) < 6.0, holdbacks_ms


def test_live_60hz_framegen_off_has_no_pacer_holdback(monkeypatch):
    """With frame generation off, real frames at 60 Hz must never wait on sleep_until."""
    sleeps: list[float] = []
    monkeypatch.setattr(pipeline, "sleep_until", lambda d, *a, **kw: (sleeps.append(d), time.perf_counter())[1])

    class _Sink:
        gpu = False

        def write(self, frame, text=None):
            pass

    def stream():
        t0 = time.perf_counter()
        for i in range(10):
            target = t0 + i * TICK + (0.010 if i == 2 else 0.0)
            if (w := target - time.perf_counter()) > 0:
                time.sleep(w)
            now = time.perf_counter()
            yield Tick(torch.full((8, 8, 3), (i // 2) * 30, dtype=torch.uint8),
                       now, i, new_hint=(i % 2 == 0), t_ready=now)

    _cpu_pipeline().run(
        stream(), _Sink(), label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False, "interp": False}),
        live=True,
    )
    assert sleeps == []


def test_live_60hz_30fps_interp_still_evens_out_cadence():
    """30->60 Hz frame generation still paces the fast repeat tick after a slow INTERP tick."""
    present_times: list[float] = []

    class _Sink:
        gpu = False

        def write(self, frame, text=None):
            present_times.append(time.perf_counter())

    def live_30fps_stream():
        t0 = time.perf_counter()
        for i in range(16):
            target = t0 + i * TICK
            if (w := target - time.perf_counter()) > 0:
                time.sleep(w)
            yield Tick(torch.full((8, 8, 3), (i // 2) % 200, dtype=torch.uint8),
                       target, i, new_hint=(i % 2 == 0), t_ready=time.perf_counter())

    st = _cpu_pipeline(_DummyInterp(interp_sleep=0.010)).run(
        live_30fps_stream(), _Sink(), label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False}),
        live=True,
    )
    intervals_ms = np.diff(present_times[4:]) * 1000.0
    assert intervals_ms.min() > 11.0 and np.percentile(intervals_ms, 90) < 23.0, intervals_ms
    assert 15.0 < float(np.median(intervals_ms)) < 18.5, intervals_ms
    assert np.median(st.sink_ms) < 2.0
    assert np.median(st.busy_ms) < 15.0


def _eval_fused_nv12_formula_cpu(y_np: np.ndarray, uv_np: np.ndarray, colorspace: int, color_range: int) -> np.ndarray:
    """Exact NumPy float32 evaluation of _NV12_TO_BGR_CUDA_SRC for bit-level verification without a GPU."""
    h, w = y_np.shape
    uv_h, uv_w, _ = uv_np.shape
    yo, ys, cs, c_bu, c_gu, c_gv, c_rv = [np.float32(x) for x in chiaki_tap._nv12_to_bgr_params(colorspace, color_range, h)]

    rows = np.arange(h, dtype=np.float32)[:, None]
    cols = np.arange(w, dtype=np.float32)[None, :]
    src_y = np.clip((rows + np.float32(0.5)) * np.float32(uv_h / h) - np.float32(0.5), np.float32(0.0), np.float32(uv_h - 1))
    src_x = np.clip(cols * np.float32(uv_w / w), np.float32(0.0), np.float32(uv_w - 1))
    y0 = src_y.astype(np.int32)
    x0 = src_x.astype(np.int32)
    y1 = np.minimum(y0 + 1, uv_h - 1)
    x1 = np.minimum(x0 + 1, uv_w - 1)
    wy = src_y - y0.astype(np.float32)
    wx = src_x - x0.astype(np.float32)
    w00 = (np.float32(1.0) - wy) * (np.float32(1.0) - wx)
    w01 = (np.float32(1.0) - wy) * wx
    w10 = wy * (np.float32(1.0) - wx)
    w11 = wy * wx

    uv_f = uv_np.astype(np.float32)
    u = (w00 * uv_f[y0, x0, 0] + w01 * uv_f[y0, x1, 0] + w10 * uv_f[y1, x0, 0] + w11 * uv_f[y1, x1, 0] - np.float32(128.0)) * cs
    v = (w00 * uv_f[y0, x0, 1] + w01 * uv_f[y0, x1, 1] + w10 * uv_f[y1, x0, 1] + w11 * uv_f[y1, x1, 1] - np.float32(128.0)) * cs
    lum = (y_np.astype(np.float32) - yo) * ys

    b = np.rint(np.clip(lum + c_bu * u, np.float32(0.0), np.float32(255.0))).astype(np.uint8)
    g = np.rint(np.clip(lum + c_gu * u + c_gv * v, np.float32(0.0), np.float32(255.0))).astype(np.uint8)
    r = np.rint(np.clip(lum + c_rv * v, np.float32(0.0), np.float32(255.0))).astype(np.uint8)
    return np.stack([b, g, r], axis=-1)


@pytest.mark.parametrize("colorspace", [1, 5])  # BT.709, BT.601
@pytest.mark.parametrize("color_range", [1, 2])  # limited, full
def test_fused_nv12_to_bgr_math_matches_torch_reference(colorspace: int, color_range: int):
    rng = np.random.default_rng(42 + colorspace * 10 + color_range)
    # Include non-contiguous pitched row stride (like NVDEC DLPack surfaces: width 64 inside pitch 128)
    y_pitched = rng.integers(0, 256, size=(36, 128), dtype=np.uint8)
    uv_pitched = rng.integers(0, 256, size=(18, 64, 2), dtype=np.uint8)
    y_np = y_pitched[:, :64]
    uv_np = uv_pitched[:, :32, :]

    ref = chiaki_tap._cuda_frame_to_bgr_torch(
        torch.from_numpy(y_np), torch.from_numpy(uv_np), colorspace, color_range
    ).numpy()
    fused = _eval_fused_nv12_formula_cpu(y_np, uv_np, colorspace, color_range)

    assert fused.shape == ref.shape == (36, 64, 3)
    assert fused.dtype == np.uint8
    # Float32 order-of-operations can differ by at most 1 LSB on exact half-integers
    diff = np.abs(fused.astype(np.int16) - ref.astype(np.int16))
    assert diff.max() <= 1, f"max diff {diff.max()} > 1"
    assert (diff == 0).mean() > 0.99


def test_fused_nv12_kernel_returns_none_on_cpu_tensors():
    y = torch.zeros((16, 16), dtype=torch.uint8)
    uv = torch.zeros((8, 8, 2), dtype=torch.uint8)
    assert chiaki_tap._FUSED_NV12_KERNEL.launch(y, uv, 1, 1) is None


def test_nv12_to_bgr_params_unspecified_colorspace_by_height():
    """Unspecified colorspace (0 or 2) defaults to BT.709 at >=720p and BT.601 below 720p."""
    bt709 = chiaki_tap._nv12_to_bgr_params(1, 1, 1080)
    bt601 = chiaki_tap._nv12_to_bgr_params(5, 1, 480)
    for cs in (0, 2):
        assert chiaki_tap._nv12_to_bgr_params(cs, 1, 1080) == bt709
        assert chiaki_tap._nv12_to_bgr_params(cs, 1, 720) == bt709
        assert chiaki_tap._nv12_to_bgr_params(cs, 1, 480) == bt601
    # Explicit tags must be respected regardless of height.
    assert chiaki_tap._nv12_to_bgr_params(5, 1, 1080) == bt601
    assert chiaki_tap._nv12_to_bgr_params(1, 1, 480) == bt709


def test_dpb_warning_on_ctx_delay_and_pre_idr_no_false_alarm(capsys):
    """ctx_delay > 0 warns even when dpb_lag == 0, while pre-IDR P-frame skips do not false-alarm."""
    class _FakeDec:
        def __init__(self, has_b_frames: bool, delay: int) -> None:
            self.has_b_frames = has_b_frames
            self.delay = delay

    src = chiaki_tap.ChiakiSource.__new__(chiaki_tap.ChiakiSource)
    src.tag = "[tap]"
    src.decoder_kind = "cuda"
    src.decoder_dpb_delay = None

    assert src._check_dpb(_FakeDec(False, 2), 0, initial=True) is True
    out = capsys.readouterr().out
    assert "WARNING:" in out and "delay=2" in out

    assert src._check_dpb(_FakeDec(False, 0), 0, initial=True, pre_idr=3) is False
    out = capsys.readouterr().out
    assert src.decoder_dpb_delay == 0
    assert "0-frame DPB verified" in out and "WARNING:" not in out
    assert "skipped 3 pre-IDR packet(s)" in out

    assert src._check_dpb(_FakeDec(False, 0), 2, initial=True) is True
    out = capsys.readouterr().out
    assert "WARNING:" in out and "buffered 2 frame(s)" in out

    assert src._check_dpb(_FakeDec(False, 1), 0, initial=False) is True
    out = capsys.readouterr().out
    assert "WARNING:" in out and "delay=1" in out


def test_fused_nv12_ping_pong_buffer_safety():
    """Successive CUDA launches alternate between 2 preallocated buffers without clobbering the previous frame."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    rng = np.random.default_rng(123)
    frames = []
    for _ in range(3):
        y_p = torch.from_numpy(rng.integers(0, 256, size=(36, 128), dtype=np.uint8)).cuda()
        uv_p = torch.from_numpy(rng.integers(0, 256, size=(18, 64, 2), dtype=np.uint8)).cuda()
        frames.append((y_p[:, :64], uv_p[:, :32, :]))

    out0 = chiaki_tap._FUSED_NV12_KERNEL.launch(*frames[0], 2, 1)
    assert out0 is not None
    ref0 = chiaki_tap._cuda_frame_to_bgr_torch(*frames[0], 2, 1)

    out1 = chiaki_tap._FUSED_NV12_KERNEL.launch(*frames[1], 2, 1)
    assert out1 is not None
    assert out0.data_ptr() != out1.data_ptr()
    # Frame 0 must remain intact after frame 1 was written to the second ping-pong buffer.
    diff0 = (out0.to(torch.int16) - ref0.to(torch.int16)).abs()
    assert int(diff0.max().item()) <= 1

    out2 = chiaki_tap._FUSED_NV12_KERNEL.launch(*frames[2], 2, 1)
    assert out2 is not None
    assert out2.data_ptr() == out0.data_ptr()
    ref2 = chiaki_tap._cuda_frame_to_bgr_torch(*frames[2], 2, 1)
    diff2 = (out2.to(torch.int16) - ref2.to(torch.int16)).abs()
    assert int(diff2.max().item()) <= 1


def test_decoder_low_delay_zero_frame_dpb():
    """Verify ChiakiSource._new_decoder configures low_delay + SLICE and decodes IPPP H.264 with 0-frame DPB."""
    av = pytest.importorskip("av")
    enc = av.CodecContext.create("libx264", "w")
    enc.width, enc.height, enc.pix_fmt = 160, 120, "yuv420p"
    enc.options = {"preset": "ultrafast", "tune": "zerolatency", "profile": "high"}
    pkts = []
    for i in range(6):
        fr = av.VideoFrame.from_ndarray(np.full((120, 160, 3), i * 35, dtype=np.uint8), format="bgr24")
        pkts.extend(bytes(p) for p in enc.encode(fr))
    pkts.extend(bytes(p) for p in enc.encode(None))
    assert len(pkts) == 6

    dummy_src = object.__new__(chiaki_tap.ChiakiSource)
    dummy_src.decoder_kind = "sw"
    dec = dummy_src._new_decoder("h264")
    assert dec.flags & av.codec.context.Flags.low_delay
    assert str(dec.thread_type).endswith("SLICE")

    outs = [len(dec.decode(av.Packet(b))) for b in pkts]
    assert outs == [1, 1, 1, 1, 1, 1], f"expected 0-frame DPB (1 frame per packet), got {outs}"
    assert not dec.has_b_frames


def _mailbox_tick(seq: int) -> Tick:
    return Tick(np.zeros((2, 2, 3), np.uint8), time.perf_counter(), 0, extra={"seq": seq})


def test_mailbox_overwrites_and_returns_newest_immediately():
    from framegen.io import LatestFrameMailbox, WAIT_POLL_S

    mailbox = LatestFrameMailbox()
    results = [mailbox.put(_mailbox_tick(i)) for i in range(5)]

    assert results == [False, True, True, True, True]
    assert mailbox.dropped == 4

    t0 = time.perf_counter()
    got = mailbox.get(timeout=WAIT_POLL_S)
    elapsed = time.perf_counter() - t0

    assert got is not None
    assert got.extra["seq"] == 4
    assert elapsed < WAIT_POLL_S


def test_mailbox_queue_ticks_delivers_newest_and_waiting():
    import threading
    from framegen.io import LatestFrameMailbox, WAIT_POLL_S, Waiting, queue_ticks

    mailbox = LatestFrameMailbox()
    for i in range(5):
        mailbox.put(_mailbox_tick(i))
    mailbox.close()

    got = list(queue_ticks(mailbox, "the PS5"))
    assert len(got) == 1
    assert isinstance(got[0], Tick)
    assert got[0].index == 0
    assert got[0].extra["seq"] == 4
    assert mailbox.dropped == 4

    mailbox2 = LatestFrameMailbox()

    def feed():
        time.sleep(WAIT_POLL_S * 2.2)
        mailbox2.put(_mailbox_tick(42))
        mailbox2.close()

    threading.Thread(target=feed, daemon=True).start()
    items = list(queue_ticks(mailbox2, "the PS5"))
    waits = [x for x in items if isinstance(x, Waiting)]
    ticks = [x for x in items if isinstance(x, Tick)]
    assert len(waits) >= 1
    assert len(ticks) == 1
    assert ticks[0].index == 0
    assert ticks[0].extra["seq"] == 42


def test_mailbox_clear_and_chiaki_hold_false():
    import queue
    import threading
    from framegen.io import LatestFrameMailbox, WAIT_POLL_S

    mailbox = LatestFrameMailbox()
    mailbox.put(_mailbox_tick(0))
    mailbox.put(_mailbox_tick(1))
    assert mailbox.dropped == 1

    mailbox.clear()
    with pytest.raises(queue.Empty):
        mailbox.get(timeout=0.01)

    mailbox.put(_mailbox_tick(2))
    got = mailbox.get(timeout=WAIT_POLL_S)
    assert got.extra["seq"] == 2
    assert mailbox.dropped == 1

    src = chiaki_tap.ChiakiSource.__new__(chiaki_tap.ChiakiSource)
    src._go = threading.Event()
    src._go.set()
    src.q = LatestFrameMailbox()

    src.hold(True)
    assert not src._go.is_set()
    src.q.put(_mailbox_tick(10))
    src.hold(False)
    assert src._go.is_set()
    with pytest.raises(queue.Empty):
        src.q.get(timeout=0.01)


def test_mailbox_concurrent_producer_consumer():
    import queue
    import threading
    from framegen.io import LatestFrameMailbox, WAIT_POLL_S

    mailbox = LatestFrameMailbox()
    total = 50

    def producer():
        for i in range(total):
            mailbox.put(_mailbox_tick(i))
            time.sleep(0.001)
        mailbox.close()

    t = threading.Thread(target=producer, daemon=True)
    t.start()

    received: list[int] = []
    while True:
        try:
            item = mailbox.get(timeout=WAIT_POLL_S)
        except queue.Empty:
            continue
        if item is None:
            break
        received.append(item.extra["seq"])

    t.join(timeout=2.0)
    assert received == sorted(received)
    assert len(set(received)) == len(received)
    assert len(received) + mailbox.dropped == total


def test_passthrough_bypasses_detector_and_interp_load():
    """Pure passthrough (interp=False, no delay test, no scoring) skips it.load and detector.update."""
    import math

    p = _cpu_pipeline()
    detector_calls = 0
    load_calls = 0
    orig_update = p.detector.update
    orig_load = p.interp.load

    def spy_update(frame_u8):
        nonlocal detector_calls
        detector_calls += 1
        return orig_update(frame_u8)

    def spy_load(frame):
        nonlocal load_calls
        load_calls += 1
        return orig_load(frame)

    p.detector.update = spy_update
    p.interp.load = spy_load

    frames = [torch.full((8, 8, 3), i * 25, dtype=torch.uint8) for i in range(6)]
    written = []
    diffs = []

    class _Sink:
        gpu = False

        def write(self, frame, text=None):
            written.append(frame)

    def on_tick(tick, diff, action, proc_ms, t_start):
        diffs.append(diff)

    p.run(
        (Tick(f, time.perf_counter(), i, new_hint=None) for i, f in enumerate(frames)),
        _Sink(),
        label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False, "interp": False}),
        on_tick=on_tick,
    )
    assert detector_calls == 0
    assert load_calls == 0
    assert written == frames
    assert len(diffs) == len(frames)
    assert all(math.isnan(d.mean_diff) and math.isnan(d.max_diff) for d in diffs)

    # With interp=True, both detector.update and it.load are called normally.
    p.run(
        (Tick(f, time.perf_counter(), i, new_hint=None) for i, f in enumerate(frames)),
        _Sink(),
        label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False, "interp": True}),
    )
    assert detector_calls == len(frames)
    assert load_calls == len(frames)


def test_live_120hz_framegen_off_and_60fps_passthrough_no_sleep(monkeypatch):
    """--hz 120 with interp=False (and steady 60 fps passthrough) never sleeps in the hz > 60 pacer."""
    sleeps: list[float] = []
    monkeypatch.setattr(pipeline, "sleep_until", lambda d, *a, **kw: (sleeps.append(d), time.perf_counter())[1])

    class _Sink:
        gpu = False

        def write(self, frame, text=None):
            pass

    def bunched_stream(n: int = 10, new_hint_fn=lambda i: i % 2 == 0):
        t0 = time.perf_counter()
        for i in range(n):
            now = t0 + i * 0.002  # 2 ms apart (< 1/120 s = 8.33 ms): would trigger sleep_until if gated wrong
            yield Tick(torch.full((8, 8, 3), (i * 31) % 250, dtype=torch.uint8),
                       now, i, new_hint=new_hint_fn(i), t_ready=now)

    # 1) --hz 120 with interp=False -> zero sleep_until calls
    _cpu_pipeline().run(
        bunched_stream(),
        _Sink(),
        label="RIFE",
        settings=Settings({"hz": 120, "even_out": False, "overlay": False, "interp": False}),
        live=True,
    )
    assert sleeps == []

    # 2) Steady 60 fps passthrough (all new real frames) at --hz 120 with interp=True -> zero sleep_until calls
    #    after the 2-tick cadence startup (tick 1 interpolates once before tick 2 enters PENDING passthrough).
    steady_sleeps: list[float] = []
    monkeypatch.setattr(pipeline, "sleep_until", lambda d, *a, **kw: (steady_sleeps.append(d), time.perf_counter())[1])
    _cpu_pipeline().run(
        bunched_stream(n=10, new_hint_fn=lambda i: True),
        _Sink(),
        label="RIFE",
        settings=Settings({"hz": 120, "even_out": False, "overlay": False, "interp": True}),
        live=True,
        on_tick=lambda tick, *_: steady_sleeps.clear() if tick.index == 3 else None,
    )
    assert steady_sleeps == []


def test_async_cuda_event_timing_without_per_tick_sync(monkeypatch):
    """gpu_out=True uses async cuda.Event pairs for proc_interp_ms with zero per-tick Stream.synchronize()."""
    if not torch.cuda.is_available():
        pytest.skip("CUDA not available")

    import math

    class _CudaDummyInterp:
        device = torch.device("cuda")
        model_name = "dummy_cuda"
        backend = "cuda"
        steady_hud = False
        h = 64
        w = 64

        def __init__(self) -> None:
            self.in_u8 = None
            self.slots = [None, None]

        def load(self, frame) -> None:
            self.in_u8 = frame

        def prepare(self, slot: int) -> None:
            self.slots[slot] = self.in_u8

        def interpolate(self, slot: int, t: float = 0.5):
            a = self.slots[0] if self.slots[0] is not None else self.in_u8
            b = self.slots[1] if self.slots[1] is not None else self.in_u8
            return ((a.float() + b.float()) * 0.5).to(torch.uint8)

    p = Pipeline.__new__(Pipeline)
    from framegen.cadence import DuplicateDetector
    p.interp = _CudaDummyInterp()
    p.detector = DuplicateDetector()
    p.cut_threshold = 25.0
    p.stats = None
    p.waiting = None
    p._out_pinned = torch.empty((64, 64, 3), dtype=torch.uint8)
    p._out_np = p._out_pinned.numpy()

    class _GpuSink:
        gpu = True

        def write(self, frame, text=None):
            assert torch.is_tensor(frame) and frame.is_cuda

    sync_calls = 0
    orig_stream_sync = torch.cuda.Stream.synchronize

    def spy_stream_sync(self, *args, **kwargs):
        nonlocal sync_calls
        sync_calls += 1
        return orig_stream_sync(self, *args, **kwargs)

    monkeypatch.setattr(torch.cuda.Stream, "synchronize", spy_stream_sync)

    frames = [torch.full((64, 64, 3), (i // 2) * 30, dtype=torch.uint8, device="cuda") for i in range(12)]
    st = p.run(
        (Tick(f, time.perf_counter(), i, new_hint=(i % 2 == 0)) for i, f in enumerate(frames)),
        _GpuSink(),
        label="RIFE",
        settings=Settings({"hz": 60, "even_out": False, "overlay": False, "interp": True}),
    )
    assert sync_calls == 0
    assert st.actions["interp"] > 0
    assert len(st.proc_interp_ms) == st.actions["interp"]
    assert all(math.isfinite(ms) and ms > 0.0 for ms in st.proc_interp_ms), st.proc_interp_ms
