"""Window capture (Experimental): frames from another app's window, such as Sony's PS Remote Play app, the Xbox app,
Moonlight's window, or a browser playing a cloud game.

    python -m framegen.window --list [--json]     # the app windows it could capture (for the launcher's picker)
    python run.py --source window --window "exe:RemotePlay.exe" --display

How it works:
- The window is picked by its program ("exe:RemotePlay.exe"), a preset ("ps-remote-play", "xbox", "moonlight"), its
  handle ("hwnd:123", as the launcher's picker passes it) or part of its title. Its picture area (the client area,
  without the title bar) is found on screen.
- Desktop Duplication copies that part of the monitor when it changes. With CUDA it's copied on the graphics card
  and stays there (framegen/dda.py), as Remote Play's frames do; otherwise, or if that can't start (the monitor on
  another graphics card, Windows HDR mode), FFmpeg's ddagrab brings it to the CPU as BGR (through PyAV: no extra
  package).
- SmoovPlay's own window sits on top as a click-through overlay that never takes the focus (the app keeps it:
  some apps stop updating without it), and is left out of the capture (io.py, exclude_from_capture: Windows'
  WDA_EXCLUDEFROMCAPTURE, Windows 10 2004 or newer), so the capture sees the app underneath.
- Frames come when the app draws, not on a fixed beat: run.py puts them on a 60 Hz beat (sources.Clock60). An app
  decodes its own stream, so its repeats are never quite identical: they're told from new frames by a rule learned
  from the last few frames (cadence.DuplicateDetector(learn=True)).
- Frames are scaled to 1920x1080 (SIZE), whatever the window's size: the size Remote Play's stream has, so the
  TensorRT engines setup builds are used (a new size would build one, about a minute) and the settings fit. The
  overlay shows the result at the window's own size.
- The window is looked at twice a second. If it moved or changed size, capture restarts there. If it's minimised,
  the last frame stays up until it's back.

Limits (Experimental, not tried with a real app yet):
- The app's window must stay on screen, not minimised, and not covered by other windows (SmoovPlay's aside).
- The monitor it's on has to be one the main graphics adapter drives (DXGI adapter 0, where ddagrab looks).
- The app plays its own sound, which comes a little ahead of the picture.
- Without the GPU path, frames go through the CPU (about 2-4 ms more at 1080p).
"""
from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from ctypes import wintypes as wt
from dataclasses import asdict, dataclass

# Apps people stream games through, by program name: preset -> (its programs, what it's called). Browsers come with
# no preset (a cloud game is one tab among many): pick their window from the list.
PRESETS = {
    "ps-remote-play": (("remoteplay.exe",), "PS Remote Play"),
    # The Xbox app's Start menu entry runs XboxPcAppCE.exe (a desktop app); its older Store version, XboxPcApp.exe, is
    # still in the same package and draws inside a Store app frame (STORE_FRAME, below)
    "xbox": (("xboxpcappce.exe", "xboxpcapp.exe"), "Xbox app"),
    "moonlight": (("moonlight.exe",), "Moonlight"),
}
PRESET_EXES = {exe for exes, _ in PRESETS.values() for exe in exes}
KNOWN = {exe: name for exes, name in PRESETS.values() for exe in exes} | {
    "chrome.exe": "Chrome", "msedge.exe": "Edge", "firefox.exe": "Firefox", "geforcenow.exe": "GeForce NOW"}
MIN_SIZE = (320, 180)  # smaller windows aren't game streams
# Store (UWP) apps don't own their top-level window: Windows' ApplicationFrameHost.exe draws the frame, and the app's
# own window (a Windows.UI.Core.CoreWindow) sits inside it. Such a frame is listed under the app inside it.
STORE_FRAME = "applicationframehost.exe"
STORE_FRAME_CLASS = "ApplicationFrameWindow"
FULL_SCREEN = "in full screen"  # find_windows' reason for a Store app in full screen, which isn't listed
# Streaming apps' Store packages (an AppUserModelID's part before "!"), for a frame whose app is suspended: in full
# screen, switching to another window (the launcher, to press Look again) suspends it and takes its window out.
STORE_PACKAGES = {"microsoft.gamingapp_8wekyb3d8bbwe": "xboxpcapp.exe"}
SKIP_EXE = {"explorer.exe", "textinputhost.exe", STORE_FRAME, "shellexperiencehost.exe",
            "searchhost.exe", "startmenuexperiencehost.exe", "systemsettings.exe", "smoovplay.exe"}
CHECK_S = 0.5  # how often the window's place is looked at
RETRY_S = 10.0  # capture that keeps failing is tried again for this long (a screen mode change can take seconds)

# Why Windows took Desktop Duplication away or refused it: DXGI's codes, as FFmpeg's ddagrab and framegen/dda.py
# give them (lower-case hex, without 0x).
DXGI_REASONS = {
    "887a0026": "Windows took the capture away: the screen changed mode (resolution, refresh rate, HDR, a program "
                "going fullscreen) or a secure screen showed",
    "80070005": "Windows' secure screen is up (a UAC prompt or the lock screen)",
    "887a0022": "too many programs are capturing the screen at once (a recorder or an overlay)",
    "887a0004": "this screen can't be captured from the main graphics card",
    "887a002b": "the Windows session is remote or disconnected",
    "887a0005": "the graphics driver stopped or restarted",
}


def capture_error(e: BaseException, ffmpeg: str = "") -> str:
    """What stopped the capture, for the log: FFmpeg's own line when there is one (it has Windows' code, which the
    exception's "Generic error in an external library" hasn't), else the error, with a known code in words."""
    text = (ffmpeg or str(e)).strip()[:160]
    low = f"{ffmpeg} {e}".lower()
    if "lost access" in low:  # framegen/dda.py's CaptureLost
        low += " 887a0026"
    for code, words in DXGI_REASONS.items():
        if code in low:
            return f"{text}: {words}"
    return text
SIZE = (1920, 1080)  # what frames are scaled to, for the pipeline


@dataclass(frozen=True)
class Rect:
    x: int
    y: int
    w: int
    h: int

    def clamp(self, other: "Rect") -> "Rect":
        """The part of this rectangle inside `other` (zero size if none)."""
        x0, y0 = max(self.x, other.x), max(self.y, other.y)
        x1, y1 = min(self.x + self.w, other.x + other.w), min(self.y + self.h, other.y + other.h)
        return Rect(x0, y0, max(0, x1 - x0), max(0, y1 - y0))


@dataclass(frozen=True)
class AppWindow:
    hwnd: int
    title: str
    exe: str  # the program's file name, lower case
    w: int  # client area size, in pixels
    h: int

    @property
    def app(self) -> str:
        """What the app is called, for people: a known app's name, else its program name."""
        return KNOWN.get(self.exe, self.exe.removesuffix(".exe"))


# ----------------------------------------------------------------------------- Windows calls
class _RECT(ctypes.Structure):
    _fields_ = [("left", wt.LONG), ("top", wt.LONG), ("right", wt.LONG), ("bottom", wt.LONG)]


class _MONITORINFOEX(ctypes.Structure):
    _fields_ = [("cbSize", wt.DWORD), ("rcMonitor", _RECT), ("rcWork", _RECT), ("dwFlags", wt.DWORD),
                ("szDevice", wt.WCHAR * 32)]


class _DEVMODE(ctypes.Structure):  # DEVMODEW, display fields
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


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wt.DWORD), ("Data2", wt.WORD), ("Data3", wt.WORD), ("Data4", ctypes.c_ubyte * 8)]


class _OUTPUT_DESC(ctypes.Structure):  # DXGI_OUTPUT_DESC
    _fields_ = [("DeviceName", wt.WCHAR * 32), ("DesktopCoordinates", _RECT), ("AttachedToDesktop", wt.BOOL),
                ("Rotation", ctypes.c_uint), ("Monitor", wt.HMONITOR)]


_IID_IDXGIFactory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))
_ready = False


def _api():
    """user32 / kernel32 / dwmapi with their argument types set, once; the process made DPI aware (per monitor) first,
    so window positions are real pixels, as Desktop Duplication's are."""
    global _ready
    u32, k32, dwm = ctypes.windll.user32, ctypes.windll.kernel32, ctypes.windll.dwmapi
    if not _ready:
        try:
            u32.SetProcessDpiAwarenessContext(ctypes.c_void_p(-4))  # PER_MONITOR_AWARE_V2 (fails harmlessly if set)
        except Exception:  # noqa: BLE001 - before Windows 10 1703
            pass
        u32.EnumWindows.argtypes = [ctypes.c_void_p, wt.LPARAM]
        for f in ("IsWindowVisible", "IsIconic", "IsWindow", "GetWindowTextLengthW"):
            getattr(u32, f).argtypes = [wt.HWND]
        u32.GetWindowTextW.argtypes = [wt.HWND, wt.LPWSTR, ctypes.c_int]
        u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
        u32.GetWindow.argtypes, u32.GetWindow.restype = [wt.HWND, wt.UINT], wt.HWND
        u32.FindWindowExW.argtypes, u32.FindWindowExW.restype = [wt.HWND, wt.HWND, wt.LPCWSTR, wt.LPCWSTR], wt.HWND
        u32.GetWindowLongW.argtypes = [wt.HWND, ctypes.c_int]
        u32.GetClientRect.argtypes = [wt.HWND, ctypes.POINTER(_RECT)]
        u32.ClientToScreen.argtypes = [wt.HWND, ctypes.POINTER(wt.POINT)]
        u32.MonitorFromWindow.argtypes, u32.MonitorFromWindow.restype = [wt.HWND, wt.DWORD], wt.HMONITOR
        u32.GetMonitorInfoW.argtypes = [wt.HMONITOR, ctypes.POINTER(_MONITORINFOEX)]
        u32.EnumDisplaySettingsW.argtypes = [wt.LPCWSTR, wt.DWORD, ctypes.POINTER(_DEVMODE)]
        u32.SetWindowDisplayAffinity.argtypes = [wt.HWND, wt.DWORD]
        k32.OpenProcess.restype = wt.HANDLE
        k32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
        k32.CloseHandle.argtypes = [wt.HANDLE]
        dwm.DwmGetWindowAttribute.argtypes = [wt.HWND, wt.DWORD, ctypes.c_void_p, wt.DWORD]
        _ready = True
    return u32, k32, dwm


def _exe_of(hwnd: int, cache: dict) -> str:
    u32, k32, _ = _api()
    pid = wt.DWORD()
    u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value not in cache:
        cache[pid.value] = ""
        h = k32.OpenProcess(0x1000, False, pid.value)  # PROCESS_QUERY_LIMITED_INFORMATION
        if h:
            buf, n = ctypes.create_unicode_buffer(1024), wt.DWORD(1024)
            if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)):
                cache[pid.value] = os.path.basename(buf.value).lower()
            k32.CloseHandle(h)
    return cache[pid.value]


def _hosted_exe(frame: int, cache: dict) -> str:
    """The app inside a Store app's frame (STORE_FRAME): the program of its CoreWindow, or of the first child window
    another process owns. "" when there's none: the app is suspended, minimised or closing (Windows takes its window
    out of the frame then)."""
    u32, _, _ = _api()
    child = u32.FindWindowExW(frame, None, "Windows.UI.Core.CoreWindow", None)
    if child:
        return _exe_of(child, cache)
    host = _exe_of(frame, cache)
    child = u32.FindWindowExW(frame, None, None, None)
    for _ in range(64):  # a frame has a handful of children; the cap only guards against a loop
        if not child:
            break
        exe = _exe_of(child, cache)
        if exe and exe != host:
            return exe
        child = u32.FindWindowExW(frame, child, None, None)
    return ""


class _GUID(ctypes.Structure):
    _fields_ = [("a", wt.DWORD), ("b", wt.WORD), ("c", wt.WORD), ("d", ctypes.c_ubyte * 8)]


class _PROPERTYKEY(ctypes.Structure):
    _fields_ = [("fmtid", _GUID), ("pid", wt.DWORD)]


class _PROPVARIANT(ctypes.Structure):
    _fields_ = [("vt", wt.USHORT), ("r1", wt.USHORT), ("r2", wt.USHORT), ("r3", wt.USHORT),
                ("ptr", ctypes.c_void_p), ("pad", ctypes.c_void_p)]


def _aumid(hwnd: int) -> str:
    """A window's AppUserModelID (the shell's PKEY_AppUserModel_ID), "" if it has none or it can't be read. A Store
    app's frame keeps its app's even while the app inside is suspended. COM is set up for the call and the thread
    left as it was."""
    try:
        ole32, shell32 = ctypes.windll.ole32, ctypes.windll.shell32
        iid = _GUID(0x886D8EEB, 0x8CF2, 0x4446, (ctypes.c_ubyte * 8)(0x8D, 0x02, 0xCD, 0xBA, 0x1D, 0xBD, 0xCF, 0x99))
        key = _PROPERTYKEY(_GUID(0x9F4C2855, 0x9F79, 0x4B39,
                                 (ctypes.c_ubyte * 8)(0xA8, 0xD0, 0xE1, 0xD4, 0x2D, 0xE1, 0xD5, 0xF3)), 5)
        ole32.CoInitializeEx.argtypes = [ctypes.c_void_p, wt.DWORD]
        ours = ole32.CoInitializeEx(None, 2) in (0, 1)  # S_OK / S_FALSE: balance it; RPC_E_CHANGED_MODE: leave it
        try:
            shell32.SHGetPropertyStoreForWindow.argtypes = [wt.HWND, ctypes.POINTER(_GUID), ctypes.POINTER(ctypes.c_void_p)]
            store = ctypes.c_void_p()
            if shell32.SHGetPropertyStoreForWindow(hwnd, ctypes.byref(iid), ctypes.byref(store)) != 0 or not store:
                return ""
            vtbl = ctypes.cast(ctypes.cast(store, ctypes.POINTER(ctypes.c_void_p))[0], ctypes.POINTER(ctypes.c_void_p))
            get_value = ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, ctypes.POINTER(_PROPERTYKEY),
                                           ctypes.POINTER(_PROPVARIANT))(vtbl[5])  # IPropertyStore::GetValue
            release = ctypes.WINFUNCTYPE(wt.ULONG, ctypes.c_void_p)(vtbl[2])
            pv = _PROPVARIANT()
            try:
                ok = get_value(store, ctypes.byref(key), ctypes.byref(pv)) == 0
                return ctypes.wstring_at(pv.ptr) if ok and pv.vt == 31 and pv.ptr else ""  # VT_LPWSTR
            finally:
                ole32.PropVariantClear(ctypes.byref(pv))
                release(store)
        finally:
            if ours:
                ole32.CoUninitialize()
    except (OSError, AttributeError, ValueError):
        return ""


def _store_app_of(frame: int, cache: dict) -> str:
    """The program of the app in a Store app's frame: the one inside it, or for a suspended streaming app, its
    program by its AppUserModelID. "" if neither."""
    return _hosted_exe(frame, cache) or STORE_PACKAGES.get(_aumid(frame).split("!")[0].lower(), "")


def client_rect(hwnd: int) -> Rect | None:
    """The window's picture area on screen, in pixels; None if it's gone or minimised."""
    u32, _, _ = _api()
    if not u32.IsWindow(hwnd) or u32.IsIconic(hwnd):
        return None
    r, p = _RECT(), wt.POINT(0, 0)
    if not u32.GetClientRect(hwnd, ctypes.byref(r)) or not u32.ClientToScreen(hwnd, ctypes.byref(p)):
        return None
    return Rect(p.x, p.y, r.right - r.left, r.bottom - r.top)


def monitor_of(hwnd: int) -> tuple[Rect, str, int]:
    """The monitor the window is (mostly) on: its rectangle, device name (\\\\.\\DISPLAY1) and refresh rate."""
    u32, _, _ = _api()
    mon = u32.MonitorFromWindow(hwnd, 2)  # MONITOR_DEFAULTTONEAREST
    mi = _MONITORINFOEX()
    mi.cbSize = ctypes.sizeof(mi)
    u32.GetMonitorInfoW(mon, ctypes.byref(mi))
    r = mi.rcMonitor
    dm = _DEVMODE()
    dm.dmSize = ctypes.sizeof(dm)
    hz = dm.dmDisplayFrequency if u32.EnumDisplaySettingsW(mi.szDevice, 0xFFFFFFFF, ctypes.byref(dm)) else 60  # current
    return Rect(r.left, r.top, r.right - r.left, r.bottom - r.top), mi.szDevice, int(hz) if hz > 1 else 60


def _com(obj, index: int, *argtypes):
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])


def dda_outputs() -> list[str]:
    """The device names of the outputs (monitors) on DXGI adapter 0, in order: ddagrab's output_idx counts these."""
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory)) < 0:
        return []
    names: list[str] = []
    try:
        adapter = ctypes.c_void_p()
        if _com(factory, 12, wt.UINT, ctypes.POINTER(ctypes.c_void_p))(factory, 0, ctypes.byref(adapter)) < 0:
            return []  # IDXGIFactory1::EnumAdapters1
        try:
            for i in range(16):
                out = ctypes.c_void_p()
                if _com(adapter, 7, wt.UINT, ctypes.POINTER(ctypes.c_void_p))(adapter, i, ctypes.byref(out)) < 0:
                    break  # IDXGIAdapter::EnumOutputs: DXGI_ERROR_NOT_FOUND, no more
                desc = _OUTPUT_DESC()
                _com(out, 7, ctypes.POINTER(_OUTPUT_DESC))(out, ctypes.byref(desc))  # IDXGIOutput::GetDesc
                _com(out, 2)(out)  # Release
                names.append(desc.DeviceName)
        finally:
            _com(adapter, 2)(adapter)
    finally:
        _com(factory, 2)(factory)
    return names


def exclude_from_capture(hwnd: int) -> bool:
    """Leaves a window of ours out of screen captures (and so out of Desktop Duplication): WDA_EXCLUDEFROMCAPTURE,
    Windows 10 2004 or newer. False if Windows refused (older Windows, or no desktop composition)."""
    u32, _, _ = _api()
    return bool(u32.SetWindowDisplayAffinity(hwnd, 0x11))


# ----------------------------------------------------------------------------- finding the window
def find_windows(left_out: dict | None = None) -> list[AppWindow]:
    """Top-level windows that could be a game stream: visible, not minimised away from view, big enough, not a
    tool window or part of Windows' shell, and not this process's own. A Store app's frame counts as the app inside
    it. Known apps first.

    left_out, if given, gets why a streaming app (PRESETS) that's running wasn't listed: {program: {reasons}}, for
    the launcher's hint and diagnostics. Only those apps' programs are noted, never other windows or titles."""
    u32, _, dwm = _api()
    own, exes, found, seen = os.getpid(), {}, [], set()

    def note(exe: str, why: str) -> None:
        if left_out is not None and exe in PRESET_EXES:
            left_out.setdefault(exe, set()).add(why)

    def is_cloaked(hwnd) -> bool:
        cloaked = wt.DWORD()
        dwm.DwmGetWindowAttribute(hwnd, 14, ctypes.byref(cloaked), ctypes.sizeof(cloaked))  # DWMWA_CLOAKED
        return bool(cloaked.value)

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def each(hwnd, _):
        seen.add(hwnd)
        pid = wt.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if pid.value == own:
            return True
        # Hidden, owned by another window (dialogs, popups) or a tool window (GWL_EXSTYLE & WS_EX_TOOLWINDOW)
        if not u32.IsWindowVisible(hwnd) or u32.GetWindow(hwnd, 4) or u32.GetWindowLongW(hwnd, -20) & 0x80:
            if left_out is not None:
                note(_exe_of(hwnd, exes), "no window on screen")
            return True
        exe = _exe_of(hwnd, exes)
        if exe == STORE_FRAME:
            exe = _hosted_exe(hwnd, exes)
        if not exe or exe in SKIP_EXE:
            return True
        if is_cloaked(hwnd):
            note(exe, "on another desktop, or not shown yet")
            return True
        n = u32.GetWindowTextLengthW(hwnd)
        buf = ctypes.create_unicode_buffer(n + 1)
        u32.GetWindowTextW(hwnd, buf, n + 1)
        r = client_rect(hwnd)
        iconic = bool(u32.IsIconic(hwnd))
        if not buf.value:
            note(exe, "a window with no title")
            return True
        if r is None and not (iconic and exe in KNOWN):  # minimised: only apps it knows
            note(exe, "minimised" if iconic else "closing")
            return True
        if r is not None and (r.w < MIN_SIZE[0] or r.h < MIN_SIZE[1]):
            note(exe, "its window is too small")
            return True
        found.append(AppWindow(int(hwnd), buf.value, exe, r.w if r else 0, r.h if r else 0))
        return True

    u32.EnumWindows(each, 0)
    if left_out is not None:
        # A Store app in full screen (the Xbox app's own full screen, or Shift+Win+Enter): Windows moves its frame out
        # of the desktop's layer of windows into one of its own (an "immersive" z-order band), and since Windows 8
        # EnumWindows lists only the desktop's. FindWindowEx still finds the frame. It's noted rather than listed: an
        # ordinary window, as SmoovPlay's is, may not be able to show over it. Switched away from, it's also cloaked
        # and its app suspended, which is when someone presses Look again: still noted, the app known by its ID.
        frame = u32.FindWindowExW(None, None, STORE_FRAME_CLASS, None)
        for _ in range(256):  # a handful of frames; the cap only guards against a loop
            if not frame:
                break
            if frame not in seen and u32.IsWindowVisible(frame):
                note(_store_app_of(frame, exes), FULL_SCREEN)
            frame = u32.FindWindowExW(None, frame, STORE_FRAME_CLASS, None)
    if left_out is not None:  # an app with a window listed isn't missing, whichever of its programs drew it
        listed = {w.app for w in found}
        for exe in [e for e in left_out if KNOWN[e] in listed]:
            del left_out[exe]
    return sorted(found, key=lambda w: (w.exe not in KNOWN, w.exe not in PRESET_EXES, w.app.lower()))


def pick(query: str, windows: list[AppWindow] | None = None) -> AppWindow | None:
    """The window `query` names: "hwnd:N", "exe:NAME", a preset ("ps-remote-play"), or part of a title. Of several
    matches, the biggest. Several queries split by "|" are tried in turn: the launcher passes "hwnd:N|exe:NAME", so
    an app reopened since its list was read is still found."""
    windows = find_windows() if windows is None else windows
    if "|" in query:
        return next((w for q in query.split("|") if q.strip() and (w := pick(q, windows))), None)
    q = query.strip()
    if q.lower().startswith("hwnd:"):
        hwnd = int(q[5:], 0)
        return next((w for w in windows if w.hwnd == hwnd), None)
    if q.lower() in PRESETS:
        exes = set(PRESETS[q.lower()][0])
        matches = [w for w in windows if w.exe in exes]
    elif q.lower().startswith("exe:"):
        exe = q[4:].strip().lower()
        exe = exe if exe.endswith(".exe") else exe + ".exe"
        same = next((set(exes) for exes, _ in PRESETS.values() if exe in exes), {exe})  # either Xbox app program
        matches = [w for w in windows if w.exe in same]
    else:
        matches = [w for w in windows if q.lower() in w.title.lower()]
    return max(matches, key=lambda w: w.w * w.h, default=None)


# ----------------------------------------------------------------------------- capture geometry
@dataclass(frozen=True)
class Grab:
    """What ddagrab is asked for: output_idx, the region (relative to that monitor) and how often to look."""
    output: int
    region: Rect  # relative to the monitor's top-left
    hz: int


def plan_grab(client: Rect, monitor: Rect, device: str, outputs: list[str], hz: int) -> Grab | str:
    """Where to grab a window's picture from, or why it can't be. The region is the part of the picture area on the
    window's monitor, with even sizes (for the colour conversion)."""
    if device not in outputs:
        return ("its monitor isn't on the main graphics adapter, which window capture reads from: move the window to "
                "the monitor plugged into the graphics card")
    part = client.clamp(monitor)
    w, h = part.w & ~1, part.h & ~1
    if w < MIN_SIZE[0] or h < MIN_SIZE[1]:
        return "too little of the window is on screen: move it fully onto one monitor"
    return Grab(outputs.index(device), Rect(part.x - monitor.x, part.y - monitor.y, w, h), max(int(hz), 60))


def graph(g: Grab, out_w: int, out_h: int) -> str:
    """The FFmpeg filter graph: Desktop Duplication of the region, only when the screen changes, to the CPU as BGR at
    the source's fixed size."""
    r = g.region
    return (f"ddagrab=output_idx={g.output}:framerate={g.hz}:video_size={r.w}x{r.h}:offset_x={r.x}:offset_y={r.y}"
            f":draw_mouse=0:dup_frames=0,hwdownload,format=bgra,scale={out_w}:{out_h}:flags=bilinear,format=bgr24")


# ----------------------------------------------------------------------------- the source
class WindowSource:
    """Ticks from an app's window, as the app draws them (a variable rate: wrap it in sources.Clock60). Iterate it
    once. `where()` says where the overlay should go: (the picture area, the monitor), both in screen pixels, or
    None while the window is minimised or gone."""

    def __init__(self, target: AppWindow, size: tuple[int, int] | None = None, open_fn=None, gpu=None) -> None:
        """open_fn: opens the CPU path (FFmpeg's ddagrab) from a filter graph. gpu: opens the GPU path, (Grab, out_w,
        out_h) -> a dda.DdaCapture-like object; None picks dda.DdaCapture when CUDA is there (and no open_fn is
        given), False uses the CPU path only."""
        from .sources import source_info
        self.target = target
        self._open = open_fn or _open_lavfi
        self._open_gpu = gpu or (_open_dda if gpu is None and open_fn is None and _cuda() else None)
        self._said_gpu = False
        self._client = client_rect(target.hwnd)
        if self._client is None:
            raise RuntimeError(f"the {target.app} window is minimised: restore it, then press Play again")
        mon, device, hz = monitor_of(target.hwnd)
        plan = plan_grab(self._client, mon, device, dda_outputs(), hz)
        if isinstance(plan, str):
            raise RuntimeError(f"can't capture the {target.app} window: {plan}")
        self._grab, self._monitor = plan, mon
        self.w, self.h = size or (plan.region.w, plan.region.h)
        self.info = source_info("window", name=f"{target.app} window", hz=float(hz))
        self.restarts = 0
        self.unchanged = 0
        self._stop = False
        self._container = None
        self.path = "GPU" if self._open_gpu else "CPU"
        print(f"[window] {target.app} ({target.exe}): {plan.region.w}x{plan.region.h} on monitor {plan.output + 1} "
              f"at {hz} Hz, captured as {self.w}x{self.h}")

    def where(self) -> tuple[Rect, Rect] | None:
        """Where the overlay goes: the part of the window's picture area that's captured (on its monitor, even
        sizes), in screen pixels, and the monitor. The frame is that area, so the overlay lines up with the app even
        when the window hangs off the screen."""
        if self._client is None or self._grab is None:
            return None
        r, m = self._grab.region, self._monitor
        return Rect(m.x + r.x, m.y + r.y, r.w, r.h), m

    def _moved(self) -> bool:
        """Re-reads the window's place; True if capture has to restart (moved, resized, or to another monitor)."""
        client = client_rect(self.target.hwnd)
        if client == self._client:
            return False
        self._client = client
        if client is None:
            return True
        mon, device, hz = monitor_of(self.target.hwnd)
        plan = plan_grab(client, mon, device, dda_outputs(), hz)
        if isinstance(plan, str):
            print(f"[window] {plan}")
            self._client = None
            return True
        self._grab, self._monitor = plan, mon
        return True

    def __iter__(self):
        from .io import Tick
        u32, _, _ = _api()
        n, check, fails = 0, time.perf_counter() + CHECK_S, 0
        failing, good, said = None, 0, ""  # when it started failing; frames since the last failure; the reason said
        while not self._stop:
            if not u32.IsWindow(self.target.hwnd):
                print(f"[window] the {self.target.app} window closed")
                return
            if self._client is None:  # minimised, or somewhere it can't be captured: the last frame stays up
                time.sleep(CHECK_S)
                self._moved()
                continue
            try:
                for got in self._frames():
                    now = time.perf_counter()
                    if self._stop:
                        return
                    if got is not None:
                        frame, t_arrival, t_source, *rest = got
                        yield Tick(frame, t_arrival, n, t_source=t_source, t_ready=time.perf_counter(),
                                   ready_event=rest[0] if rest else None)
                        n, good = n + 1, good + 1
                        if good >= 30:  # working again (a capture opened anew hands over a frame before failing)
                            fails, failing, said = 0, None, ""
                    if now >= check:
                        check = now + CHECK_S
                        if self._moved():
                            self.restarts += 1
                            break
                        if not u32.IsWindow(self.target.hwnd):
                            break
            except Exception as e:  # noqa: BLE001 - lost the duplication (a mode change, a UAC prompt): retry
                from .dda import GpuCaptureUnavailable
                if isinstance(e, GpuCaptureUnavailable) and self._open_gpu is not None:
                    print(f"[window] frames go through the CPU: the GPU path can't run here ({e})")
                    self._open_gpu, self.path = None, "CPU"
                    continue
                if self._stop:
                    return
                why, now = capture_error(e, self._ffmpeg_error()), time.perf_counter()
                fails, good = fails + 1, 0
                failing = now if failing is None else failing
                if now - failing >= RETRY_S:
                    print(f"[window] capture keeps failing ({why}); stopping")
                    return
                if why != said:  # each reason once, not every try
                    print(f"[window] capture stopped ({why}); trying again"
                          + (f" for up to {RETRY_S:.0f} s" if fails == 1 else ""))
                    said = why
                time.sleep(min(2.0, CHECK_S * fails))  # 0.5, 1, 1.5, then every 2 s
                self._moved()
            finally:
                self._close_container()

    def _frames(self):
        """(frame, t_arrival, t_source) each time the window's picture changes, from one capture (it ends for a
        restart); None now and then while nothing changes, so the window can be looked at."""
        if self._open_gpu is not None:
            r = self._grab.region
            self._container = cap = self._open_gpu(self._grab, self.w, self.h)
            if not self._said_gpu:
                self._said_gpu = True
                print(f"[window] capturing on the GPU ({r.w}x{r.h} -> {self.w}x{self.h})")
            while not self._stop:
                yield cap.grab(100)
        else:
            # FFmpeg's own error lines, kept for the log: they have Windows' code (_ffmpeg_error)
            import av.logging
            old = av.logging.get_level()
            if old is None or old < av.logging.ERROR:
                av.logging.set_level(av.logging.ERROR)
            try:
                with av.logging.Capture() as self._ffmpeg_logs:
                    self._container = self._open(graph(self._grab, self.w, self.h))
                    for frame in self._container.decode(video=0):
                        yield frame.to_ndarray(), time.perf_counter(), frame.time
            finally:
                av.logging.set_level(old)

    def _ffmpeg_error(self) -> str:
        """FFmpeg's last error line from the capture just stopped (ddagrab's, e.g. "AcquireNextFrame failed:
        887a0026"), or ""."""
        logs, self._ffmpeg_logs = getattr(self, "_ffmpeg_logs", None) or [], []
        errors = [(str(name), str(msg).strip()) for level, name, msg in logs if level is not None and level <= 16]
        mine = [msg for name, msg in errors if "ddagrab" in name]  # (16: av.logging.ERROR and worse)
        return (mine or [msg for _, msg in errors] or [""])[-1]

    def _close_container(self) -> None:
        c, self._container = self._container, None
        if c is not None:
            self.unchanged += int(getattr(c, "unchanged", 0))
            try:
                c.close()
            except Exception:  # noqa: BLE001 - closing
                pass

    def close(self) -> None:
        """Stops at the next frame. The capture is closed by the thread reading it (closing it from here, mid-read,
        isn't safe); Clock60's reader is a daemon thread, so a capture still waiting for the screen to change doesn't
        hold up the exit."""
        self._stop = True
        skipped = f" ({self.unchanged} unchanged desktop captures skipped)" if self.unchanged else ""
        print(f"[window] capture restarted {self.restarts} time(s) for moves; frames came through the {self.path}{skipped}")


def _open_lavfi(spec: str):
    import av
    return av.open(spec, format="lavfi")


def _open_dda(g: Grab, out_w: int, out_h: int):
    from .dda import DdaCapture
    return DdaCapture(g.output, g.region.x, g.region.y, g.region.w, g.region.h, out_w, out_h)


def _cuda() -> bool:
    try:
        import torch
        return torch.cuda.is_available()
    except Exception:  # noqa: BLE001 - no torch (DirectML installs have it, CPU-only): the CPU path
        return False


def open_window(query: str) -> WindowSource:
    """--window QUERY: the window, or a clear error naming what was looked for."""
    left_out: dict = {}
    target = pick(query, find_windows(left_out))
    if target is None:
        # The app asked for ("exe:NAME" in the query; any app if it names none) in full screen: say how to leave it
        asked = {KNOWN.get(q.strip()[4:].strip().lower().removesuffix(".exe") + ".exe")
                 for q in query.split("|") if q.strip().lower().startswith("exe:")} - {None}
        full = sorted(KNOWN[e] for e, why in left_out.items() if FULL_SCREEN in why and (not asked or KNOWN[e] in asked))
        if full:
            raise RuntimeError(f"the {full[0]} is in full screen, where SmoovPlay can't find its window: leave full "
                               "screen (its full screen button, or Shift+Win+Enter), then press Play again")
        raise RuntimeError(f"no app window matches {query!r}: open the app and start streaming, then press Play again "
                           "(python -m framegen.window --list shows what's found)")
    return WindowSource(target, size=SIZE)


# ----------------------------------------------------------------------------- command line
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--list", action="store_true", help="the app windows that could be captured")
    ap.add_argument("--json", action="store_true")
    a = ap.parse_args(argv)
    if sys.platform != "win32":
        print(json.dumps({"windows": []}) if a.json else "window capture needs Windows")
        return 0
    left_out: dict = {}
    windows = find_windows(left_out)
    missed = [{"exe": exe, "app": KNOWN[exe], "why": sorted(why)} for exe, why in sorted(left_out.items())]
    if a.json:
        print(json.dumps({"windows": [{**asdict(w), "app": w.app, "known": w.exe in KNOWN} for w in windows],
                          "left_out": missed}))
    else:
        for w in windows:
            print(f"hwnd:{w.hwnd:<10} {w.app:16} {w.w}x{w.h}  {w.title}")
        if not windows:
            print("no app windows found")
        for m in missed:
            print(f"{m['app']} ({m['exe']}) is running but left out: {', '.join(m['why'])}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
