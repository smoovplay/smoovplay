"""Display discovery and multi-monitor selection (Windows).

Lists active monitors with stable IDs, friendly names, desktop bounds, refresh rates and Windows HDR status, and
keeps the "Monitor" setting's choices (`CHOICES`, `NAMES`, `HINTS`) up to date. Uses only the standard library and
Windows APIs via `ctypes`, so `python -m framegen.settings` and `python -m framegen.displays` stay fast.
"""
from __future__ import annotations

import argparse
import ctypes
import json
import re
import sys
from dataclasses import asdict, dataclass

DEFAULT_MONITOR = "primary"

# Mutated in place by refresh() so framegen.settings.SPECS, Menu and specs_json() always see the current monitors.
CHOICES: list[str] = [DEFAULT_MONITOR]
NAMES: dict[str, str] = {DEFAULT_MONITOR: "Main display"}
HINTS: dict[str, str] = {DEFAULT_MONITOR: "Always uses Windows' main display."}

_GENERIC_NAMES = {"generic pnp monitor", "generic non-pnp monitor"}


@dataclass(frozen=True)
class Monitor:
    id: str
    device: str  # Win32 GDI device name, e.g. \\.\DISPLAY1
    name: str  # ASCII-safe friendly name (also shown in the OpenCV in-video menu)
    x: int
    y: int
    width: int
    height: int
    hz: int  # current Windows refresh rate
    max_hz: int  # highest refresh rate supported at (width, height)
    primary: bool = False
    hdr: bool = False
    max_nits: float = 0.0
    hmonitor: int = 0


def _short_device(device: str) -> str:
    short = device[4:] if device.startswith("\\\\.\\") else device
    return re.sub(r"[^A-Za-z0-9_-]", "", short) or "DISPLAY"


def stable_id(device: str, device_path: str = "") -> str:
    """A stable, filename- and JSON-safe ID for a monitor (no backslashes, spaces or PII).

    Uses the hardware PnP ID and instance tail from Windows' monitor device path (e.g.
    `\\\\?\\DISPLAY#DEL4273#5&26b2da4e&0&UID24833#{...}` -> `DEL4273-UID24833`), falling back to `DISPLAY1`.
    """
    parts = [p for p in (device_path or "").split("#") if p]
    if len(parts) >= 3:
        pnp = re.sub(r"[^A-Za-z0-9_-]", "", parts[1])
        tokens = [t for t in parts[2].split("&") if t]
        tail = re.sub(r"[^A-Za-z0-9_-]", "", tokens[-1]) if tokens else ""
        if pnp and tail:
            return f"{pnp}-{tail}"
    return _short_device(device)


def _clean_name(raw: str, device: str, index: int) -> str:
    ascii_name = "".join(ch for ch in (raw or "") if 32 <= ord(ch) < 127).strip()
    if ascii_name and ascii_name.lower() not in _GENERIC_NAMES:
        return ascii_name
    digits = "".join(ch for ch in device if ch.isdigit())
    return f"Display {digits}" if digits else f"Display {index + 1}"


def _trim_hint(text: str, limit: int = 140) -> str:
    return text if len(text) <= limit else text[: limit - 3] + "..."


# ---------------------------------------------------------------------------------------------- Windows display APIs
if sys.platform == "win32":
    from ctypes import wintypes as wt

    class _LUID(ctypes.Structure):
        _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]

    class _PATH_SOURCE_INFO(ctypes.Structure):
        _fields_ = [("adapterId", _LUID), ("id", wt.UINT), ("modeInfoIdx", wt.UINT), ("statusFlags", wt.UINT)]

    class _RATIONAL(ctypes.Structure):
        _fields_ = [("Numerator", wt.UINT), ("Denominator", wt.UINT)]

    class _PATH_TARGET_INFO(ctypes.Structure):
        _fields_ = [("adapterId", _LUID), ("id", wt.UINT), ("modeInfoIdx", wt.UINT),
                    ("outputTechnology", wt.UINT), ("rotation", wt.UINT), ("scaling", wt.UINT),
                    ("refreshRate", _RATIONAL), ("scanLineOrdering", wt.UINT), ("targetAvailable", wt.BOOL),
                    ("statusFlags", wt.UINT)]

    class _PATH_INFO(ctypes.Structure):
        _fields_ = [("sourceInfo", _PATH_SOURCE_INFO), ("targetInfo", _PATH_TARGET_INFO), ("flags", wt.UINT)]

    class _MODE_INFO(ctypes.Structure):
        _fields_ = [("infoType", wt.UINT), ("id", wt.UINT), ("adapterId", _LUID), ("_pad", ctypes.c_ubyte * 48)]

    class _DEVICE_INFO_HEADER(ctypes.Structure):
        _fields_ = [("type", wt.UINT), ("size", wt.UINT), ("adapterId", _LUID), ("id", wt.UINT)]

    class _SOURCE_DEVICE_NAME(ctypes.Structure):
        _fields_ = [("header", _DEVICE_INFO_HEADER), ("viewGdiDeviceName", wt.WCHAR * 32)]

    class _TARGET_DEVICE_NAME(ctypes.Structure):
        _fields_ = [("header", _DEVICE_INFO_HEADER), ("flags", wt.UINT), ("outputTechnology", wt.UINT),
                    ("edidManufactureId", wt.WORD), ("edidProductCodeId", wt.WORD),
                    ("connectorInstance", wt.UINT), ("monitorFriendlyDeviceName", wt.WCHAR * 64),
                    ("monitorDevicePath", wt.WCHAR * 128)]

    class _DISPLAY_DEVICE(ctypes.Structure):
        _fields_ = [("cb", wt.DWORD), ("DeviceName", wt.WCHAR * 32), ("DeviceString", wt.WCHAR * 128),
                    ("StateFlags", wt.DWORD), ("DeviceID", wt.WCHAR * 128), ("DeviceKey", wt.WCHAR * 128)]

    class _DEVMODE(ctypes.Structure):
        _fields_ = [("dmDeviceName", wt.WCHAR * 32), ("dmSpecVersion", wt.WORD), ("dmDriverVersion", wt.WORD),
                    ("dmSize", wt.WORD), ("dmDriverExtra", wt.WORD), ("dmFields", wt.DWORD),
                    ("dmPositionX", wt.LONG), ("dmPositionY", wt.LONG), ("dmDisplayOrientation", wt.DWORD),
                    ("dmDisplayFixedOutput", wt.DWORD), ("dmColor", ctypes.c_short), ("dmDuplex", ctypes.c_short),
                    ("dmYResolution", ctypes.c_short), ("dmTTOption", ctypes.c_short), ("dmCollate", ctypes.c_short),
                    ("dmFormName", wt.WCHAR * 32), ("dmLogPixels", wt.WORD), ("dmBitsPerPel", wt.DWORD),
                    ("dmPelsWidth", wt.DWORD), ("dmPelsHeight", wt.DWORD), ("dmDisplayFlags", wt.DWORD),
                    ("dmDisplayFrequency", wt.DWORD), ("dmICMMethod", wt.DWORD), ("dmICMIntent", wt.DWORD),
                    ("dmMediaType", wt.DWORD), ("dmDitherType", wt.DWORD), ("dmReserved1", wt.DWORD),
                    ("dmReserved2", wt.DWORD), ("dmPanningWidth", wt.DWORD), ("dmPanningHeight", wt.DWORD)]


def _query_display_config() -> dict[str, tuple[str, str]]:
    """Maps GDI device name (upper case, e.g. `\\\\.\\DISPLAY1`) -> (friendly_name, device_path)."""
    if sys.platform != "win32":
        return {}
    try:
        u32 = ctypes.windll.user32
        qdc_only_active_paths = 2
        get_source_name, get_target_name = 1, 2
        n_paths, n_modes = wt.UINT(), wt.UINT()
        if u32.GetDisplayConfigBufferSizes(qdc_only_active_paths, ctypes.byref(n_paths), ctypes.byref(n_modes)) != 0:
            return {}
        if not n_paths.value:
            return {}
        paths = (_PATH_INFO * n_paths.value)()
        modes = (_MODE_INFO * max(1, n_modes.value))()
        if u32.QueryDisplayConfig(qdc_only_active_paths, ctypes.byref(n_paths), paths,
                                  ctypes.byref(n_modes), modes, None) != 0:
            return {}
        out: dict[str, tuple[str, str]] = {}
        for i in range(n_paths.value):
            p = paths[i]
            sn = _SOURCE_DEVICE_NAME(
                header=_DEVICE_INFO_HEADER(get_source_name, ctypes.sizeof(_SOURCE_DEVICE_NAME),
                                           p.sourceInfo.adapterId, p.sourceInfo.id))
            if u32.DisplayConfigGetDeviceInfo(ctypes.byref(sn)) != 0 or not sn.viewGdiDeviceName:
                continue
            key = sn.viewGdiDeviceName.upper()
            if key in out:
                continue  # keep the first target for mirrored/duplicated sources
            tn = _TARGET_DEVICE_NAME(
                header=_DEVICE_INFO_HEADER(get_target_name, ctypes.sizeof(_TARGET_DEVICE_NAME),
                                           p.targetInfo.adapterId, p.targetInfo.id))
            if u32.DisplayConfigGetDeviceInfo(ctypes.byref(tn)) == 0:
                out[key] = (tn.monitorFriendlyDeviceName, tn.monitorDevicePath)
        return out
    except Exception:  # noqa: BLE001 - fallback to EnumDisplayDevicesW
        return {}


def _dxgi_by_device() -> dict[str, dict]:
    try:
        from .dxgi_present import outputs
        return {o["name"].upper(): o for o in outputs() if o.get("name")}
    except Exception:  # noqa: BLE001 - DXGI optional
        return {}


def _disambiguate(monitors: list[Monitor]) -> list[Monitor]:
    """Ensure every monitor has a unique `id` and case-insensitively unique `name`."""
    if len(monitors) <= 1:
        return monitors
    id_counts: dict[str, int] = {}
    name_counts: dict[str, int] = {}
    for m in monitors:
        id_counts[m.id] = id_counts.get(m.id, 0) + 1
        name_counts[m.name.lower()] = name_counts.get(m.name.lower(), 0) + 1
    name_seen: dict[str, int] = {}
    out: list[Monitor] = []
    for m in monitors:
        mid = f"{m.id}-{_short_device(m.device)}" if id_counts[m.id] > 1 else m.id
        key = m.name.lower()
        if name_counts[key] > 1:
            n = name_seen.get(key, 0) + 1
            name_seen[key] = n
            mname = f"{m.name} ({n})"
        else:
            mname = m.name
        out.append(Monitor(mid, m.device, mname, m.x, m.y, m.width, m.height, m.hz, m.max_hz,
                           primary=m.primary, hdr=m.hdr, max_nits=m.max_nits, hmonitor=m.hmonitor))
    return out


def list_monitors() -> list[Monitor]:
    """Every active desktop monitor, primary first then left-to-right."""
    if sys.platform != "win32":
        return []
    u32 = ctypes.windll.user32
    cfg = _query_display_config()
    dxgi = _dxgi_by_device()
    found: list[Monitor] = []
    i = 0
    while True:
        dd = _DISPLAY_DEVICE(cb=ctypes.sizeof(_DISPLAY_DEVICE))
        if not u32.EnumDisplayDevicesW(None, i, ctypes.byref(dd), 0):
            break
        i += 1
        if not (dd.StateFlags & 1):  # DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
            continue
        device = dd.DeviceName
        now = _DEVMODE(dmSize=ctypes.sizeof(_DEVMODE))
        if not u32.EnumDisplaySettingsW(device, -1, ctypes.byref(now)):  # ENUM_CURRENT_SETTINGS
            continue
        w, h = int(now.dmPelsWidth), int(now.dmPelsHeight)
        if w <= 0 or h <= 0:
            continue
        hz = int(now.dmDisplayFrequency)
        if hz <= 1:
            hz = 60
        best_hz = hz
        m_idx = 0
        while True:
            mode = _DEVMODE(dmSize=ctypes.sizeof(_DEVMODE))
            if not u32.EnumDisplaySettingsW(device, m_idx, ctypes.byref(mode)):
                break
            if mode.dmPelsWidth == w and mode.dmPelsHeight == h:
                best_hz = max(best_hz, int(mode.dmDisplayFrequency))
            m_idx += 1

        friendly, dev_path = cfg.get(device.upper(), ("", ""))
        if not dev_path:
            mon_id = _DISPLAY_DEVICE(cb=ctypes.sizeof(_DISPLAY_DEVICE))
            if u32.EnumDisplayDevicesW(device, 0, ctypes.byref(mon_id), 1):  # EDD_GET_DEVICE_INTERFACE_NAME
                dev_path = mon_id.DeviceID
        if not friendly:
            mon_str = _DISPLAY_DEVICE(cb=ctypes.sizeof(_DISPLAY_DEVICE))
            if u32.EnumDisplayDevicesW(device, 0, ctypes.byref(mon_str), 0):
                friendly = mon_str.DeviceString

        dx = dxgi.get(device.upper(), {})
        found.append(Monitor(
            id=stable_id(device, dev_path),
            device=device,
            name=_clean_name(friendly, device, len(found)),
            x=int(now.dmPositionX),
            y=int(now.dmPositionY),
            width=w,
            height=h,
            hz=hz,
            max_hz=best_hz,
            primary=bool(dd.StateFlags & 4),  # DISPLAY_DEVICE_PRIMARY_DEVICE
            hdr=bool(dx.get("hdr", False)),
            max_nits=float(dx.get("max_nits", 0.0)),
            hmonitor=int(dx.get("monitor", 0)),
        ))

    found.sort(key=lambda m: (not m.primary, m.x, m.y, m.device))
    return _disambiguate(found)


def refresh(monitors: list[Monitor] | None = None) -> list[Monitor]:
    """Updates `CHOICES`, `NAMES` and `HINTS` in place from connected monitors (or a given list, for tests)."""
    monitors = list_monitors() if monitors is None else list(monitors)
    NAMES.clear()
    HINTS.clear()
    NAMES[DEFAULT_MONITOR] = "Main display"
    if len(monitors) <= 1:
        CHOICES[:] = [DEFAULT_MONITOR]
        if monitors:
            m = monitors[0]
            HINTS[DEFAULT_MONITOR] = _trim_hint(
                f"Windows' main display ({m.name}, {m.width}×{m.height} at {m.hz}\u00a0Hz)."
            )
        else:
            HINTS[DEFAULT_MONITOR] = "Always uses Windows' main display."
        return monitors

    primary = next((m for m in monitors if m.primary), monitors[0])
    CHOICES[:] = [DEFAULT_MONITOR] + [m.id for m in monitors]
    HINTS[DEFAULT_MONITOR] = _trim_hint(
        f"Always uses Windows' main display (currently {primary.name}, {primary.width}×{primary.height} "
        f"at {primary.hz}\u00a0Hz)."
    )
    for m in monitors:
        NAMES[m.id] = m.name
        parts = [f"{m.width}×{m.height} at {m.hz}\u00a0Hz"]
        if m.hdr:
            parts.append("HDR on")
        if m.primary:
            parts.append("main display")
        HINTS[m.id] = _trim_hint(" · ".join(parts) + ".")
    return monitors


def resolve_monitor(choice: str | None, monitors: list[Monitor] | None = None, *,
                    fallback: bool = True) -> Monitor | None:
    """Turns a setting value, device name (DISPLAY1) or monitor name into a connected Monitor."""
    monitors = list_monitors() if monitors is None else list(monitors)
    if not monitors:
        return None
    primary = next((m for m in monitors if m.primary), monitors[0])
    if not choice or choice.lower() == DEFAULT_MONITOR:
        return primary
    low = choice.lower()
    stripped = low[4:] if low.startswith("\\\\.\\") else low
    for m in monitors:
        if m.id == choice:
            return m
    for m in monitors:
        if m.id.lower() == low:
            return m
    for m in monitors:
        dev = m.device.lower()
        dev_short = dev[4:] if dev.startswith("\\\\.\\") else dev
        if dev == low or dev_short == stripped:
            return m
    for m in monitors:
        if m.name.lower() == low:
            return m
    for m in monitors:
        if low in m.name.lower():
            return m
    return primary if fallback else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m framegen.displays", description=__doc__)
    ap.add_argument("--json", action="store_true", help="print monitors as JSON")
    a = ap.parse_args(argv)
    monitors = refresh()
    if a.json:
        print(json.dumps({"monitors": [asdict(m) for m in monitors]}))
        return 0
    if not monitors:
        print("No monitors found.")
        return 0
    for m in monitors:
        tags = []
        if m.primary:
            tags.append("primary")
        if m.hdr:
            tags.append(f"HDR {m.max_nits:.0f} nits" if m.max_nits > 0 else "HDR")
        extra = f"  [{', '.join(tags)}]" if tags else ""
        print(f"{m.id:18s}  {m.name:20s}  {m.device:12s}  {m.width}x{m.height} @ {m.hz} Hz "
              f"(max {m.max_hz} Hz) at ({m.x}, {m.y}){extra}")
    return 0


refresh()

if __name__ == "__main__":
    sys.exit(main())
