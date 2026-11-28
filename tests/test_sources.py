"""The source layer: test mode, the 60 Hz clock, the per-source repeat detector, the source list, and that every
source path (a file, a capture card stood in for by a file, and Remote Play through a fake PS5) gives the same
frames as before. CPU only."""
import json
import os
import subprocess
import sys
import threading
import time
from fractions import Fraction
from pathlib import Path

import cv2
import numpy as np
import pytest
import torch

from framegen.io import CaptureSource, FileSource, Ps5SimSource, Tick, Waiting
from framegen.sources import Clock60, TestMode, list_sources, make_detector, source_info

ROOT = Path(__file__).resolve().parent.parent
W, H = 128, 96


@pytest.fixture(scope="module")
def clip60(tmp_path_factory) -> Path:
    """A 60 fps clip whose every frame is different (a bar moving across, and the frame number in the colour)."""
    path = tmp_path_factory.mktemp("clips") / "moving_60.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (W, H))
    for i in range(24):
        f = np.full((H, W, 3), (40 + 8 * i) % 256, np.uint8)
        f[:, (5 * i) % (W - 16):(5 * i) % (W - 16) + 16] = 255
        out.write(f)
    out.release()
    return path


def _tick(i, frame=None):
    return Tick(np.full((2, 2, 3), i, np.uint8) if frame is None else frame, float(i), i)


# ----------------------------------------------------------------------------- test mode
def test_test_mode_plays_60fps_as_30fps_and_keeps_the_left_out_frames():
    tm = TestMode(True)
    ticks = [tm(_tick(i)) for i in range(12)]
    assert [t.content for t in ticks] == [0, 0, 2, 2, 4, 4, 6, 6, 8, 8, 10, 10]
    assert [t.truth_new for t in ticks] == [True, False] * 6
    assert [t.new_hint for t in ticks] == [True, False] * 6
    for i in range(1, 12, 2):
        assert ticks[i].frame is ticks[i - 1].frame  # the kept frame again
    assert int(tm.gt(11)[0, 0, 0]) == 11 and int(tm.gt(5)[0, 0, 0]) == 5
    assert tm.gt(1) is None  # only the last few are kept
    assert tm.gt(4) is None  # kept frames are shown, not truth


def test_test_mode_off_passes_ticks_through_and_switching_on_starts_clean():
    tm = TestMode(False)
    t = _tick(0)
    assert tm(t) is t and t.content is None and t.new_hint is None
    tm.set_on(True)
    assert tm(_tick(1)).content == 0
    assert tm(_tick(2)).content == 0  # repeat of the first
    tm(_tick(3))
    tm.set_on(False)
    tm.set_on(True)
    assert tm(_tick(4)) is None  # count is odd and nothing from before may be repeated
    assert tm(_tick(5)).content == 4


# ----------------------------------------------------------------------------- the three file-fed paths agree
def _record(ticks, gt=None):
    out = []
    for t in ticks:
        truth = gt(t.content + 1) if gt is not None and t.content is not None and t.truth_new else None
        out.append((t.frame.copy(), t.content, t.truth_new, t.new_hint, truth))
    return out


def _same(a, b):
    assert len(a) == len(b) and len(a) > 0
    for (fa, ca, ta, ha, _), (fb, cb, tb, hb, _) in zip(a, b):
        assert np.array_equal(fa, fb) and (ca, ta) == (cb, tb)


def test_a_raw_file_in_test_mode_matches_the_ps5_simulation(clip60):
    sim = Ps5SimSource(str(clip60))
    tm = TestMode(True)
    raw = FileSource(str(clip60), test=tm)
    a = _record(sim)
    b = _record(raw, None)
    _same(a, b)
    assert all(h is None for *_, h, _ in a)  # the simulation leaves repeats to the picture check
    assert [h for *_, h, _ in b] == [True, False] * (len(b) // 2)
    assert raw.info.kind == "file" and sim.info.kind == "sim" and not sim.info.test_mode


def test_a_file_standing_in_for_a_capture_card_matches_the_file_path(clip60):
    file_ticks = _record(FileSource(str(clip60), test=TestMode(True)))
    tm = TestMode(True)
    cap = CaptureSource(str(clip60), test=tm)
    assert (cap.w, cap.h) == (W, H) and cap.info.kind == "capture" and cap.stand_in
    t0 = time.perf_counter()
    got, truths = [], []
    for t in cap:
        if isinstance(t, Waiting):  # the reader fell behind for a moment (GitHub's shared Macs, 28 Nov): no frame
            continue
        got.append((t.frame.copy(), t.content, t.truth_new, t.new_hint, None))
        if not t.truth_new:
            truths.append((t.content + 1, tm.gt(t.content + 1)))
    elapsed = time.perf_counter() - t0
    cap.close()
    _same(file_ticks, got)
    assert cap.dropped == 0
    assert elapsed > 0.3  # paced like a card: 24 frames at 60 fps take 0.4 s
    raw = [f for f in FileSource(str(clip60))]
    for k, truth in truths:  # the left-out frame is the real one
        assert truth is not None and np.array_equal(truth, raw[k].frame)


# ----------------------------------------------------------------------------- Remote Play through a fake PS5
@pytest.mark.skipif(sys.platform not in ("win32", "darwin") and not sys.platform.startswith("linux"),
                    reason="the frame tap is Windows, Linux and macOS only")
@pytest.mark.parametrize("native", [True, False])
def test_remote_play_through_a_fake_ps5(clip60, native):
    """chiaki path end to end on the CPU (libx264 + software decoding): a 60 fps game in test mode, or a 30 fps
    game as it is, where the compressed sizes say which frames are repeats."""
    from framegen.chiaki_tap import ChiakiSource
    tap = f"Local\\sp-test-sources-{os.getpid()}-{int(native)}"
    fake = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "frametap_fake.py"), str(clip60), "--sw",
                             "--seconds", "5", "--delay", "0.3", "--tap", tap, "--mbps", "2"]
                            + (["--native"] if native else []), stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    try:
        tm = TestMode(native)
        src = ChiakiSource("sw", wait_s=20, tap_name=tap, test=tm)
        assert (src.w, src.h) == (W, H) and src.info.kind == "chiaki" and src.info.repeats == "pixels"
        ticks, kept = [], {}
        for t in src:
            if isinstance(t, Waiting):  # no frame for a moment (the fake PS5 starting, or ending): not a tick
                continue
            ticks.append(t)
            if native and t.truth_new:
                kept[t.content] = t.frame
            if native and t.truth_new is False and t.content in kept:
                assert t.frame is kept[t.content]  # the kept frame again
        src.close()
    finally:
        out = fake.communicate(timeout=30)[0].decode(errors="replace")
        from framegen.chiaki_tap import AUDIO_SUFFIX
        from framegen.tapsys import unlink
        unlink(tap, tap + AUDIO_SUFFIX)  # shared memory on Linux and macOS outlives both processes
    assert len(ticks) >= 20, out
    assert [t.index for t in ticks] == list(range(len(ticks)))
    for t in ticks:
        assert t.t_ready is not None and t.t_ready >= t.t_arrival
        assert "decode_ms" in t.extra and "new_hint" not in t.extra and "damaged" not in t.extra
        assert t.damaged is False  # nothing was lost
    if native:
        assert [t.truth_new for t in ticks] == [True, False] * (len(ticks) // 2) + [True] * (len(ticks) % 2)
        assert all(t.new_hint == t.truth_new for t in ticks)
        assert all(t.content == 2 * (i // 2) for i, t in enumerate(ticks))
    else:
        assert all(t.new_hint is None for t in ticks)  # the pipeline tells repeats by the picture
        hints = [t.extra["size_new"] for t in ticks[8:]]  # the size check, once it has seen a few frames
        alternating = sum(h == (i % 2 == 0) for i, h in enumerate(hints))
        assert max(alternating, len(hints) - alternating) >= 0.9 * len(hints), hints


def _videotoolbox_decodes() -> bool:
    """Whether VideoToolbox decodes H.264 here. A virtual Mac, like a CI runner, may have no media engine."""
    import av
    from av.codec.hwaccel import HWAccel
    try:
        enc = av.CodecContext.create("libx264", "w")
        enc.width, enc.height, enc.pix_fmt, enc.time_base = W, H, "yuv420p", Fraction(1, 60)
        enc.open()
        frame = av.VideoFrame.from_ndarray(np.zeros((H, W, 3), np.uint8), format="bgr24").reformat(format="yuv420p")
        packets = list(enc.encode(frame)) + list(enc.encode(None))
        dec = av.CodecContext.create("h264", "r", hwaccel=HWAccel("videotoolbox"))
        return any(f.format.name == "nv12" for p in packets + [None] for f in dec.decode(p))
    except Exception:  # noqa: BLE001
        return False


@pytest.mark.skipif(sys.platform != "darwin", reason="VideoToolbox is a Mac's hardware decoder")
def test_remote_play_on_a_mac_decodes_with_videotoolbox(clip60, capsys):
    """run.py's default decoder, cuda (NVDEC), means VideoToolbox on a Mac, and gives the pictures software
    decoding does."""
    from framegen.chiaki_tap import AUDIO_SUFFIX, ChiakiSource
    from framegen.tapsys import unlink
    if not _videotoolbox_decodes():
        pytest.skip("VideoToolbox doesn't decode on this Mac")

    def play(decoder: str) -> list:
        tap = f"Local\\sp-test-vt-{os.getpid()}"
        fake = subprocess.Popen([sys.executable, str(ROOT / "scripts" / "frametap_fake.py"), str(clip60), "--sw",
                                 "--seconds", "5", "--delay", "0.3", "--tap", tap, "--mbps", "2", "--native"],
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        try:
            src = ChiakiSource(decoder, wait_s=20, tap_name=tap, test=TestMode(True))
            ticks = [t for t in src if not isinstance(t, Waiting)]
            src.close()
        finally:
            out = fake.communicate(timeout=30)[0].decode(errors="replace")
            unlink(tap, tap + AUDIO_SUFFIX)
        assert len(ticks) >= 20, out
        return [t.frame for t in ticks if t.truth_new], src.decoder_kind

    sw, _ = play("sw")
    capsys.readouterr()
    vt, kind = play("cuda")
    assert kind == "videotoolbox" and "[chiaki] decoding with VideoToolbox" in capsys.readouterr().out
    assert len(vt) == len(sw)
    for a, b in zip(vt, sw):
        assert a.shape == (H, W, 3) and np.abs(a.astype(int) - b).max() <= 2


# ----------------------------------------------------------------------------- 60 Hz clock
class _Uneven:
    """A variable-rate source: frames after the given gaps (seconds)."""

    def __init__(self, gaps):
        self.gaps = gaps
        self.info = source_info("capture")
        self.closed = False

    def __iter__(self):
        for i, g in enumerate(self.gaps):
            time.sleep(g)
            yield Tick(np.full((2, 2, 3), i, np.uint8), time.perf_counter(), i, t_source=i * 0.01)

    def close(self):
        self.closed = True


@pytest.mark.skipif(sys.platform == "darwin", reason="a real-time beat: on GitHub's shared macOS machines it came out "
                    "7.5 ms off where the test allows 3 (24 Nov), and SmoovPlay doesn't run on macOS yet")
def test_clock60_puts_uneven_frames_on_a_steady_beat():
    gaps = [0.0, 0.035, 0.004, 0.06, 0.012, 0.03]
    clock = Clock60(_Uneven(gaps), hz=100)
    out = list(clock)
    fresh = [t for t in out if t.new_hint is None]
    repeats = [t for t in out if t.new_hint is False]
    assert len(fresh) + clock.overtaken == len(gaps)  # every frame is handed on, unless a newer one overtook it
    assert len(repeats) >= 5  # the gaps are filled with repeats
    ids = [int(t.frame[0, 0, 0]) for t in out]
    assert ids == sorted(ids) and ids[0] == 0 and ids[-1] == len(gaps) - 1
    for prev, t in zip(out, out[1:]):
        if t.new_hint is False:
            assert t.frame is prev.frame and not t.damaged
    beats = np.diff([t.t_ready for t in out])
    assert abs(float(np.median(beats)) - 0.01) < 0.003
    assert all(t.t_ready >= t.t_arrival for t in out)  # handed on at its beat, never before it arrived
    assert [t.index for t in out] == list(range(len(out)))
    clock.close()
    assert clock.source.closed  # passed through to the source


# ----------------------------------------------------------------------------- repeat detector per source
def test_detector_per_source():
    assert make_detector(source_info("capture")).reference == "new"
    for kind in ("chiaki", "file", "sim"):
        assert make_detector(source_info(kind)).reference == "previous"
    assert {k for k in ("chiaki", "capture", "window", "tap", "file", "sim") if make_detector(source_info(k)).learn}         == {"chiaki", "window", "tap"}


def test_remote_play_repeats_are_told_by_the_picture():
    """A decoded Remote Play repeat isn't an exact copy: the encoder sharpens a few blocks (here one block moves by
    3, over the fixed line of 2), while a new frame moves everywhere. As in real 30 fps sessions (new frames' mean
    difference 1-9, repeats' 0.02-0.2): the learned check calls them right, the fixed line calls repeats new."""
    rng = np.random.default_rng(5)
    base = rng.integers(40, 200, (64, 64, 3)).astype(np.int16)
    frames, truth = [], []
    for i in range(120):
        if i % 2 == 0:  # new: the whole picture changes
            base = base + rng.choice([-5, 5])
            f = base.copy()
        else:  # repeat: one 16x16 block sharpened a little
            f = base.copy()
            f[:16, :16] += 3
        frames.append(torch.from_numpy(np.clip(f, 0, 255).astype(np.uint8)))
        truth.append(i % 2 == 0)
    learned = make_detector(source_info("chiaki"))
    fixed = make_detector(source_info("file"))
    by_learned = [learned.update(f).is_new for f in frames]
    by_fixed = [fixed.update(f).is_new for f in frames]
    assert by_learned[20:] == truth[20:]
    assert sum(by_fixed[20:]) == 100  # every repeat called new


def test_comparing_with_the_last_new_frame_catches_a_slow_change():
    frames = [torch.full((64, 64, 3), 100 + int(1.5 * i), dtype=torch.uint8) for i in range(8)]  # +1.5 a frame
    prev = make_detector(source_info("file"), 2.0)
    new = make_detector(source_info("capture"), 2.0)
    by_prev = [prev.update(f).is_new for f in frames][1:]
    by_new = [new.update(f).is_new for f in frames][1:]
    assert not any(by_prev)  # each step is under the threshold
    assert sum(by_new) >= 3  # but it adds up against the last new frame


# ----------------------------------------------------------------------------- the list for the launcher
def test_source_list_offers_remote_play_capture_cards_app_windows_and_the_frame_tap():
    shown = [s["kind"] for s in list_sources() if not s["hidden"]]
    assert shown == ["chiaki", "capture", "window", "tap"]  # video files only for testing
    assert all(not s["hidden"] for s in list_sources(show_all=True))


def test_capture_cards_app_windows_and_the_frame_tap_are_marked_experimental():
    # The launcher shows a badge and a Send feedback button for these: built without the hardware to test them.
    flags = {s["kind"]: s["experimental"] for s in list_sources(show_all=True)}
    assert flags == {"chiaki": False, "capture": True, "window": True, "tap": True, "file": False}


def test_run_py_lists_sources_quickly_without_loading_the_pipeline():
    t0 = time.perf_counter()
    r = subprocess.run([sys.executable, str(ROOT / "run.py"), "--list-sources", "--json"], capture_output=True,
                       text=True, cwd=ROOT, env={**os.environ, "SMOOVPLAY_SOURCES": ""})
    assert r.returncode == 0, r.stderr
    data = json.loads(r.stdout)
    assert [s["kind"] for s in data["sources"]] == ["chiaki", "capture", "window", "tap", "file"]
    assert data["sources"][0]["name"] == "PS5 Remote Play" and data["sources"][0]["repeats"] == "pixels"
    assert time.perf_counter() - t0 < 5  # no torch import
