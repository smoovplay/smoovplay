"""chiaki-ng's PS5 video codec, kept in step with SmoovPlay's HDR setting (settings "hdr", framegen/hdr.py).

chiaki-ng (Qt's QSettings) keeps its settings in the registry, under HKEY_CURRENT_USER\\Software\\Chiaki\\Chiaki\\
settings, as text: codec_local_ps5 and codec_remote_ps5 (a PS5 at home, and over the internet) are "h264", "h265" or
"h265_hdr"; without them, chiaki-ng uses H.265. It reads them when it connects, so a change here applies to the next
stream, whether chiaki-ng is open or not. run.py calls match_codec() before it waits for chiaki-ng, which the
launcher starts only then.

  Auto      H.265 HDR when the main display has Windows' HDR on and the card is NVIDIA's (SmoovPlay's HDR needs CUDA):
            an HDR stream then gets HDR output.
  On        (shown in SDR, or HDR output) H.265 HDR, on an NVIDIA card.
  Off       chiaki-ng's codec is left as the player set it.
It only ever switches to H.265 HDR, never back: SmoovPlay fits an HDR stream to a display without HDR anyway, and
chiaki-ng's settings are shared with any separate chiaki-ng install, so switching back could undo a choice made there
on purpose. H.264 is changed only to turn HDR on, which needs H.265. FRAMEGEN_CHIAKI_CODEC=0 leaves chiaki-ng's
settings alone.
"""
from __future__ import annotations

import os
import sys

KEY = r"Software\Chiaki\Chiaki\settings"
NAMES = ("codec_local_ps5", "codec_remote_ps5")
HDR, PLAIN = "h265_hdr", "h265"
LABELS = {"h264": "H.264", "h265": "H.265", "h265_hdr": "H.265 HDR"}


def wanted(hdr_setting: str, current: str | None, hdr_display: bool, nvidia: bool) -> str | None:
    """The PS5 codec chiaki-ng should switch to, or None to leave `current` as it is (None: not set, so H.265)."""
    if hdr_setting != "off" and nvidia and (hdr_setting in ("sdr", "hdr10") or hdr_display):
        return HDR if (current or PLAIN) != HDR else None
    return None


def _hdr_display() -> bool:
    try:
        from .dxgi_present import display_hdr
        d = display_hdr()
        return bool(d and d["hdr"])
    except Exception:  # noqa: BLE001 - unknown: as without HDR
        return False


def match_codec(hdr_setting: str, nvidia: bool, key: str = KEY) -> str | None:
    """Sets chiaki-ng's PS5 codec for SmoovPlay's HDR setting (see the module docstring), and says so. Returns the
    codec it set, or None if nothing changed. key: the registry key (tests use their own); it's only made (when
    chiaki-ng hasn't run yet) if there's something to write."""
    if sys.platform != "win32" or os.environ.get("FRAMEGEN_CHIAKI_CODEC", "") == "0" or hdr_setting == "off":
        return None
    hdr_display = nvidia and hdr_setting == "auto" and _hdr_display()
    import winreg
    current: dict[str, str | None] = {name: None for name in NAMES}
    try:
        with winreg.OpenKey(winreg.HKEY_CURRENT_USER, key) as k:
            for name in NAMES:
                try:
                    value = winreg.QueryValueEx(k, name)[0]
                    current[name] = value if isinstance(value, str) else None
                except FileNotFoundError:
                    pass
    except FileNotFoundError:
        pass  # chiaki-ng hasn't run yet: H.265, its default
    except OSError as e:
        print(f"[chiaki] couldn't read chiaki-ng's codec setting ({e}); pick it in chiaki-ng's Video settings")
        return None
    changes = {name: new for name in NAMES if (new := wanted(hdr_setting, current[name], hdr_display, nvidia))}
    if not changes:
        return None
    try:
        with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_WRITE) as k:
            for name, new in changes.items():
                winreg.SetValueEx(k, name, 0, winreg.REG_SZ, new)
    except OSError as e:
        print(f"[chiaki] couldn't set chiaki-ng's codec ({e}); pick it in chiaki-ng's Video settings")
        return None
    was = current[next(iter(changes))] or PLAIN
    why = "this display has Windows' HDR on" if hdr_display else "SmoovPlay's HDR is on"
    print(f"[chiaki] chiaki-ng's PS5 codec set to {LABELS[HDR]} (was {LABELS.get(was, was)}), as {why}; "
          "it takes effect when chiaki-ng connects")
    return HDR
