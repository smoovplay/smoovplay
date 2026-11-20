"""Debug snapshots (settings "debug", this session only): every few seconds, the real frame and where Steady HUD acts
on it, as files beside the session's log, plus a line in the log. For checking what the pipeline does in a real game
without a debugger; nothing is written while it's off."""
from __future__ import annotations

from pathlib import Path

import numpy as np

EVERY = 150  # real frames between snapshots: 5 s of a 30 fps game


def mask_of(interp) -> np.ndarray | None:
    """The interpolator's current Steady HUD mask as an HxW float array (1 = the real pixels), or None if it has none
    or Steady HUD is off."""
    get = getattr(interp, "steady_mask_now", None)
    return get() if get is not None else None


def write(folder: Path, n: int, frame: np.ndarray | None, mask: np.ndarray | None) -> str:
    """Saves snapshot `n` into `folder` (frame_NNN.jpg, mask_NNN.png) and returns the log line about it."""
    import cv2
    folder.mkdir(parents=True, exist_ok=True)
    if frame is not None and frame.dtype == np.uint8:
        cv2.imwrite(str(folder / f"frame_{n:03d}.jpg"), frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
    if mask is None:
        return f"[debug] snapshot {n}: Steady HUD is off, so no mask"
    cv2.imwrite(str(folder / f"mask_{n:03d}.png"), (mask.clip(0.0, 1.0) * 255).astype(np.uint8))
    return f"[debug] snapshot {n}: Steady HUD keeps the real pixels on {(mask > 0.5).mean() * 100:.2f}% of the frame"
