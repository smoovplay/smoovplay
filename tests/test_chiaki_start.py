"""Remote Play's reader while the pipeline gets ready: it does no decoding (no GPU work) until the pipeline's loop
asks for frames, starts at the stream's first picture if that comes quickly, and otherwise asks the PS5 for a
refresh frame. The same after a restart (hold). And the hints under the window's "No picture" note. CPU only."""
import os
import subprocess
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32" and not sys.platform.startswith("linux"),
                                reason="the frame tap is Windows and Linux only")

from framegen.chiaki_tap import ChiakiSource, FrameTap  # noqa: E402
from framegen.io import Waiting  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
W, H = 128, 96


def _clip(path: Path, frames: int) -> Path:
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60.0, (W, H))
    for i in range(frames):
        f = np.full((H, W, 3), (40 + 3 * i) % 256, np.uint8)
        f[:, (5 * i) % (W - 16):(5 * i) % (W - 16) + 16] = 255
        out.write(f)
    out.release()
    return path


@pytest.fixture(scope="module")
def clip(tmp_path_factory) -> Path:
    return _clip(tmp_path_factory.mktemp("clips") / "moving_6s.avi", 360)


def _fake(clip: Path, tap: str):
    """A 60 fps game streamed the way chiaki-ng does it: the codec header once, as a packet of its own, and
    keyframes only when asked for."""
    return subprocess.Popen([sys.executable, str(ROOT / "scripts" / "frametap_fake.py"), str(clip), "--sw",
                             "--native", "--chiaki-like", "--seconds", "6", "--delay", "0.3", "--tap", tap,
                             "--mbps", "2", "--no-sound"], stdout=subprocess.PIPE, stderr=subprocess.STDOUT)


@pytest.fixture
def made(monkeypatch):
    """When each decoder was made (a decoder is the reader's first GPU work, with decoder "cuda")."""
    times: list = []
    real = ChiakiSource._new_decoder

    def spy(self, codec):
        times.append(time.perf_counter())
        return real(self, codec)

    monkeypatch.setattr(ChiakiSource, "_new_decoder", spy)
    return times


def _take(it, n: int, out: list) -> None:
    """The next n frames from the source's ticks (Waiting skipped), or as many as come before it ends."""
    for t in it:
        if isinstance(t, Waiting):
            continue
        out.append(t)
        if len(out) >= n:
            return


def test_a_pipeline_ready_quickly_starts_at_the_streams_first_picture(clip, made):
    tap = f"Local\\sp-test-start-quick-{os.getpid()}"
    fake = _fake(clip, tap)
    try:
        src = ChiakiSource("sw", wait_s=20, tap_name=tap)
        time.sleep(0.4)  # the stream's first frames go by while the pipeline loads
        assert made == []  # no decoder yet
        t_go = time.perf_counter()
        ticks: list = []
        _take(iter(src), 10_000, ticks)
        src.close()
    finally:
        out = fake.communicate(timeout=30)[0].decode(errors="replace")
    assert len(made) == 1 and made[0] >= t_go, out
    assert len(ticks) >= 340, out  # every frame (360), from the first
    assert src.idr_requests == 0 and src.skipped == 0, out
    assert not any(t.damaged for t in ticks)


def test_a_pipeline_ready_later_gets_a_refresh_frame_and_so_does_a_restart(clip, made):
    tap = f"Local\\sp-test-start-late-{os.getpid()}"
    fake = _fake(clip, tap)
    try:
        src = ChiakiSource("sw", wait_s=20, tap_name=tap)
        time.sleep(1.5)  # over a second of frames: the ring has moved past the stream's start
        assert made == [] and src.idr_requests == 0  # nothing decoded, nothing asked for yet
        it = iter(src)
        first: list = []
        _take(it, 40, first)
        asked = src.idr_requests
        src.hold(True)  # a restart: the reader stops decoding
        time.sleep(1.0)
        src.hold(False)
        after: list = []
        _take(it, 10_000, after)
        src.close()
    finally:
        out = fake.communicate(timeout=30)[0].decode(errors="replace")
    assert len(made) == 1, out  # one decoder, made once the pipeline asked for frames; kept through the restart
    assert len(first) == 40 and asked >= 1, out
    assert first[0].damaged  # held until the refresh frame has made the picture whole
    assert src.idr_requests > asked, out  # and again after the restart
    assert len(after) >= 60, out
    assert "keyframe requests answered" in out and "answered 0" not in out, out


def test_the_no_picture_hints(monkeypatch):
    name = f"Local\\sp-test-why-{os.getpid()}"
    src = ChiakiSource.__new__(ChiakiSource)  # just the parts why() reads: no reader thread
    src.tap = FrameTap(name, name + "-frame")
    src._beat = time.perf_counter()
    src._ring = (-1, src._beat)
    src._no_header = False
    try:
        assert "isn't streaming" in src.why()
        src.tap.begin(1, 64, 64, 60)
        assert src.why() == ""
        clock = [time.perf_counter() + 1.5]
        monkeypatch.setattr(time, "perf_counter", lambda: clock[0])
        assert "isn't sending any frames" in src.why()  # the ring hasn't moved for over a second
        src.tap.push(b"\x00\x00\x01frame")
        src._beat = clock[0]
        assert src.why() == ""
        src._no_header = True
        assert "started before SmoovPlay" in src.why() and "Ctrl+O" in src.why()
        clock[0] += 5
        src._ring = (src.tap.header().write_seq, clock[0])
        assert "decoder has stopped" in src.why()  # the reader's loop hasn't gone round for 5 s
    finally:
        src.tap.close()
