"""Debug snapshots (framegen/debugsnap.py): what a snapshot writes, and that it's a session-only setting."""
from types import SimpleNamespace

import cv2
import numpy as np

from framegen import debugsnap
from framegen.settings import BY_KEY


def test_snapshot_writes_the_frame_and_the_mask(tmp_path):
    frame = np.full((36, 64, 3), 128, np.uint8)
    mask = np.zeros((36, 64), np.float32)
    mask[4:8, 10:30] = 1.0  # 80 of 2304 pixels
    line = debugsnap.write(tmp_path / "debug-x", 3, frame, mask)
    assert (tmp_path / "debug-x" / "frame_003.jpg").exists()
    saved = cv2.imread(str(tmp_path / "debug-x" / "mask_003.png"), cv2.IMREAD_GRAYSCALE)
    assert saved[5, 20] == 255 and saved[20, 50] == 0
    assert "3.47%" in line


def test_without_steady_hud_there_is_no_mask(tmp_path):
    line = debugsnap.write(tmp_path, 1, None, None)
    assert "off" in line and not list(tmp_path.glob("mask_*"))


def test_mask_of_an_engine_without_steady_hud():
    assert debugsnap.mask_of(SimpleNamespace()) is None
    assert debugsnap.mask_of(SimpleNamespace(steady_mask_now=lambda: None)) is None


def test_debug_is_session_only():
    spec = BY_KEY["debug"]
    assert spec.saved is False and spec.default is False
