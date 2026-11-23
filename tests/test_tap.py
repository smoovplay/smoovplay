"""The frame tap's decoded frames (framegen/tap.py, docs/frame-tap.md): layout, flags, and TapSource end to end on
the CPU, with the Python writer standing in for another app."""
import sys
import threading
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32" and not sys.platform.startswith("linux"),
                                reason="the frame tap is Windows and Linux only")

from framegen.chiaki_tap import (CAPACITY, FEATURE_VARIABLE, FLAG_NEW, FLAG_REPEAT, SLOT_SIZE, SLOTS,  # noqa: E402
                                 FrameTap, raw_layout)
from framegen.io import Tick, Waiting  # noqa: E402

NAME = "Local\\smoovplay-tap-test"


@pytest.fixture
def tap():
    t = FrameTap(NAME, NAME + "-frame")
    t.end()
    yield t
    t.end()
    t.close()


def test_raw_layout_fits_the_same_memory():
    assert raw_layout("bgra", 1920, 1080) == (8, 8 << 20)
    assert raw_layout("nv12", 1920, 1080) == (21, 3 << 20)
    assert raw_layout("bgra", 2560, 1440) == (4, 15 << 20)
    for fmt, w, h in (("bgra", 1920, 1080), ("nv12", 2560, 1440), ("nv12", 64, 36)):
        n, size = raw_layout(fmt, w, h)
        assert 4 <= n <= SLOTS and n * size <= CAPACITY
    with pytest.raises(ValueError):
        raw_layout("bgra", 3840, 2160)  # one 4K BGRA frame is half the memory


def test_compressed_writer_keeps_the_fixed_layout(tap):
    tap.begin(1, 1920, 1080, 60)
    h = tap.header()
    assert (h.version, h.slots, h.slot_size) == (1, SLOTS, SLOT_SIZE)


def test_raw_roundtrip_with_hints(tap):
    tap.begin(17, 64, 36, 60, colour=2)
    h = tap.header()
    assert (h.version, h.codec, h.colour) == (2, 17, 2) and (h.slots, h.slot_size) == raw_layout("bgra", 64, 36)
    a = np.arange(64 * 36 * 4, dtype=np.uint32).astype(np.uint8).reshape(36, 64, 4)
    tap.push(a, new=True)
    tap.push(a, new=False)
    tap.push(a, frames_lost=1)
    h = tap.header()
    out = np.zeros(a.size, np.uint8)
    p0 = tap.read(0, h.qpc_freq, h.slots, h.slot_size, into=out)
    assert p0.size == a.size and p0.flags & FLAG_NEW and np.array_equal(out, a.reshape(-1))
    p1 = tap.read(1, h.qpc_freq, h.slots, h.slot_size, into=out)
    assert p1.flags & FLAG_REPEAT and not p1.flags & FLAG_NEW
    p2 = tap.read(2, h.qpc_freq, h.slots, h.slot_size, into=out)
    assert p2.frames_lost == 1 and not p2.flags & (FLAG_NEW | FLAG_REPEAT)


def test_a_bad_layout_falls_back_to_the_fixed_one(tap):
    tap.begin(17, 64, 36, 60)
    import struct
    struct.pack_into("<II", tap.mm, 8, 64, 2 << 20)  # 64 slots of 2 MiB: more than the memory holds
    h = tap.header()
    assert (h.slots, h.slot_size) == (SLOTS, SLOT_SIZE)


def ticks(src, n, timeout=5.0):
    """The source's first n ticks (Waitings skipped)."""
    out, t_end = [], time.perf_counter() + timeout
    for t in src:
        if isinstance(t, Waiting):
            assert time.perf_counter() < t_end, "no frames came"
            continue
        out.append(t)
        if len(out) == n:
            break
    return out


def send(tap, frames, period=0.004):
    """Push (array, kwargs) pairs from a thread, a little apart, once the reader is listening."""
    def run():
        time.sleep(0.3)
        for a, kw in frames:
            tap.push(a, **kw)
            time.sleep(period)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def test_tap_source_bgra_end_to_end(tap):
    from framegen.tap import TapSource
    w, h = 64, 36
    tap.begin(17, w, h, 60)
    src = TapSource(FrameTap(NAME, NAME + "-frame"), NAME, tap.header(), device=None)
    try:
        frames = []
        for i in range(6):
            a = np.zeros((h, w, 4), np.uint8)
            a[..., 0], a[..., 1], a[..., 2], a[..., 3] = 10 * i, 100, 200, 255  # B, G, R, A
            frames.append((a, {"new": [True, False, None][i % 3], "frames_lost": int(i == 4)}))
        send(tap, frames)
        got = ticks(src, 6)
        assert [t.extra["seq"] for t in got] == list(range(6))
        assert all(isinstance(t, Tick) and t.frame.shape == (h, w, 3) for t in got)
        assert [int(t.frame[0, 0, 0]) for t in got] == [0, 10, 20, 30, 40, 50]
        assert tuple(got[0].frame[0, 0]) == (0, 100, 200)  # BGR, alpha dropped
        assert [t.new_hint for t in got] == [True, False, None, True, False, None]
        assert [t.damaged for t in got] == [False, False, False, False, True, False]
    finally:
        tap.end()
        src.close()


def test_tap_source_nv12_grey(tap):
    from framegen.tap import TapSource
    w, h = 64, 36
    tap.begin(16, w, h, 60)
    src = TapSource(FrameTap(NAME, NAME + "-frame"), NAME, tap.header(), device=None)
    try:
        nv12 = np.full(w * h * 3 // 2, 128, np.uint8)  # mid grey, neutral chroma
        send(tap, [(nv12, {})] * 2)
        got = ticks(src, 2)
        assert got[0].frame.shape == (h, w, 3)
        assert np.abs(got[0].frame.astype(int) - 130).max() <= 3  # limited range 128 -> about 130
    finally:
        tap.end()
        src.close()


def test_tap_source_ends_with_the_stream(tap):
    from framegen.tap import TapSource
    tap.begin(17, 16, 16, 60)
    src = TapSource(FrameTap(NAME, NAME + "-frame"), NAME, tap.header(), device=None)
    try:
        send(tap, [(np.zeros((16, 16, 4), np.uint8), {})])
        assert len(ticks(src, 1)) == 1
        tap.end()
        rest = [t for t in src if not isinstance(t, Waiting)]
        assert rest == []  # the iterator finished
    finally:
        src.close()


def test_open_tap_variable_writer_gets_a_beat(tap):
    from framegen.sources import Clock60, TestMode
    from framegen.tap import open_tap
    tap.begin(17, 16, 16, 60, features=FEATURE_VARIABLE)
    kind, src = open_tap(NAME, wait_s=1, test=TestMode(False))
    try:
        assert kind == "tap" and isinstance(src, Clock60) and not src.info.test_mode
        assert src.info.rate == "variable"
    finally:
        src.close()
    with pytest.raises(ValueError):
        open_tap(NAME, wait_s=1, test=TestMode(True))


def test_another_apps_compressed_frames_are_not_called_chiaki_ng(tap, capsys):
    """GreenFrame-style: an app writes chiaki-ng's compressed layout. Same reader, its own wording: the launcher reads
    "[tap] frames:" as the stream found, and nothing tells the player to use chiaki-ng."""
    from framegen.tap import open_tap
    tap.begin(0, 2560, 1440, 60)  # H.264, as an Xbox Remote Play stream
    kind, src = open_tap(NAME, "sw", wait_s=1)
    try:
        assert kind == "chiaki" and src.app and src.info.name == "Frame tap"
        tap.end()
        why = src.why()
    finally:
        src.close()
    out = capsys.readouterr().out
    assert "[tap] frames: H.264 2560x1440, up to 60 fps" in out
    assert "[chiaki]" not in out and "chiaki-ng" not in out and "chiaki-ng" not in why and "app" in why


def test_make_detector_learns_for_the_tap():
    from framegen.sources import make_detector, source_info
    assert make_detector(source_info("tap")).learn
