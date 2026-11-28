"""Where SmoovPlay keeps its own files (settings, looks, card timing, clips being saved).

Windows: %APPDATA%\\framegen, as the launcher expects. macOS: ~/Library/Application Support/framegen. Elsewhere:
$APPDATA/framegen if it's set, else ~/framegen (as before).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path


def data_dir() -> Path:
    if sys.platform == "darwin" and not os.environ.get("APPDATA"):
        return Path.home() / "Library" / "Application Support" / "framegen"
    return Path(os.environ.get("APPDATA", Path.home())) / "framegen"
