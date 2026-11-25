"""Minimise and restore another program's windows (Windows; does nothing elsewhere).

Used to minimise chiaki-ng while the pipeline's window is in front: chiaki-ng keeps streaming, decoding and
reading the controller, but stops drawing video nobody sees, which otherwise takes GPU time from RIFE.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time

# SW_SHOWMINNOACTIVE minimises without activating another window (SW_MINIMIZE activates the next one in line, which
# can take the keys away from the video window).
SW_SHOWNOACTIVATE, SW_MINIMIZE, SW_SHOWMINNOACTIVE, SW_RESTORE = 4, 6, 7, 9


def _image_name(pid: int) -> str:
    """The file name of process `pid`'s program, lower case (e.g. "chiaki.exe"); "" if it can't be read."""
    from ctypes import wintypes as wt
    k32 = ctypes.windll.kernel32
    k32.OpenProcess.restype = wt.HANDLE
    k32.QueryFullProcessImageNameW.argtypes = [wt.HANDLE, wt.DWORD, wt.LPWSTR, ctypes.POINTER(wt.DWORD)]
    k32.CloseHandle.argtypes = [wt.HANDLE]
    h = k32.OpenProcess(0x1000, False, pid)  # PROCESS_QUERY_LIMITED_INFORMATION
    if not h:
        return ""
    try:
        buf, n = ctypes.create_unicode_buffer(1024), wt.DWORD(1024)
        return os.path.basename(buf.value).lower() if k32.QueryFullProcessImageNameW(h, 0, buf, ctypes.byref(n)) else ""
    finally:
        k32.CloseHandle(h)


def windows_of(exe: str) -> list[int]:
    """Visible top-level windows whose process image is `exe` (e.g. "chiaki.exe")."""
    if sys.platform != "win32":
        return []
    from ctypes import wintypes as wt
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
    u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    u32.IsWindowVisible.argtypes = [wt.HWND]
    names: dict[int, str] = {}
    found: list[int] = []

    def image(pid: int) -> str:
        if pid not in names:
            names[pid] = _image_name(pid)
        return names[pid]

    @ctypes.WINFUNCTYPE(wt.BOOL, wt.HWND, wt.LPARAM)
    def visit(hwnd, _):
        pid = wt.DWORD()
        u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
        if u32.IsWindowVisible(hwnd) and image(pid.value) == exe.lower():
            found.append(hwnd)
        return True

    u32.EnumWindows(visit, 0)
    return found


def show(hwnds: list[int], cmd: int) -> None:
    if sys.platform == "win32":
        for h in hwnds:
            ctypes.windll.user32.ShowWindow(ctypes.c_void_p(h), cmd)


def minimised(hwnd: int) -> bool:
    return sys.platform == "win32" and bool(ctypes.windll.user32.IsIconic(ctypes.c_void_p(hwnd)))


def foreground() -> tuple[int, str]:
    """The window in front (it gets the keys) and its program's file name, lower case; (0, "") if none."""
    if sys.platform != "win32":
        return 0, ""
    from ctypes import wintypes as wt
    u32 = ctypes.windll.user32
    u32.GetForegroundWindow.restype = wt.HWND
    u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    hwnd = u32.GetForegroundWindow()
    if not hwnd:
        return 0, ""
    pid = wt.DWORD()
    u32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return int(hwnd), _image_name(pid.value)


def ms_since_input() -> int | None:
    """Milliseconds since the last keyboard or mouse input to this PC (a controller doesn't count); None off Windows."""
    if sys.platform != "win32":
        return None

    class _LII(ctypes.Structure):
        _fields_ = [("cbSize", ctypes.c_uint), ("dwTime", ctypes.c_uint)]
    lii = _LII(ctypes.sizeof(_LII), 0)
    if not ctypes.windll.user32.GetLastInputInfo(ctypes.byref(lii)):
        return None
    return (ctypes.windll.kernel32.GetTickCount() - lii.dwTime) & 0xFFFFFFFF


def bring_to_front(hwnd: int, nudge: bool = False) -> bool:
    """Puts our window in front with the keyboard focus, so its keys work without a click first. Windows only lets
    the program that's in front hand the focus on (SetForegroundWindow from anyone else just flashes the taskbar
    button), and when the video window opens, the program in front is chiaki-ng, where the player just pressed
    connect. So this briefly shares that program's input queue (AttachThreadInput), which lets the focus move.
    Call it from the window's own thread. Returns whether our window ended up in front.

    nudge: if that wasn't enough (no window in front to share with, or Windows refused), ask again with Alt held
    down. Windows lets a program that's just had a key press take the front; the press goes to the program in front,
    the release to ours, and Alt alone does nothing in either."""
    if sys.platform != "win32" or not hwnd:
        return False
    from ctypes import wintypes as wt
    u32, k32 = ctypes.windll.user32, ctypes.windll.kernel32
    u32.GetForegroundWindow.restype = wt.HWND
    u32.GetWindowThreadProcessId.argtypes = [wt.HWND, ctypes.POINTER(wt.DWORD)]
    u32.GetWindowThreadProcessId.restype = wt.DWORD
    u32.AttachThreadInput.argtypes = [wt.DWORD, wt.DWORD, wt.BOOL]
    for f in (u32.BringWindowToTop, u32.SetForegroundWindow, u32.SetFocus):
        f.argtypes = [wt.HWND]
    fg = u32.GetForegroundWindow()
    if fg == hwnd:
        return True
    me = k32.GetCurrentThreadId()
    other = u32.GetWindowThreadProcessId(fg, None) if fg else 0
    joined = bool(other and other != me and u32.AttachThreadInput(me, other, True))
    try:
        u32.BringWindowToTop(hwnd)
        u32.SetForegroundWindow(hwnd)
        u32.SetFocus(hwnd)
    finally:
        if joined:
            u32.AttachThreadInput(me, other, False)
    if nudge and u32.GetForegroundWindow() != hwnd:
        VK_MENU, KEYUP = 0x12, 0x0002
        u32.keybd_event(VK_MENU, 0, 0, 0)
        try:
            u32.SetForegroundWindow(hwnd)
            u32.SetFocus(hwnd)
        finally:
            u32.keybd_event(VK_MENU, 0, KEYUP, 0)
    return u32.GetForegroundWindow() == hwnd


# ----------------------------------------------------------------------------- scheduling priority
# Keeps frames on time when other programs are busy (a browser, a download, a game launcher updating). Set
# FRAMEGEN_NO_BOOST=1 to turn both off, e.g. to measure the difference.
_AVRT_PRIORITY_HIGH = 1
_D3DKMT_PRIORITY_HIGH = 4  # D3DKMT_SCHEDULINGPRIORITYCLASS: idle 0, below 1, normal 2, above 3, high 4, realtime 5


def _boost_off() -> bool:
    return sys.platform != "win32" or os.environ.get("FRAMEGEN_NO_BOOST") == "1"


def boost_thread(task: str = "Games") -> bool:
    """Registers the calling thread with Windows' multimedia class scheduler (MMCSS) under `task`, at high
    priority, so it gets the CPU on time. Lasts until the thread ends. Returns whether it worked."""
    if _boost_off():
        return False
    try:
        avrt = ctypes.WinDLL("avrt")
        avrt.AvSetMmThreadCharacteristicsW.restype = ctypes.c_void_p
        avrt.AvSetMmThreadCharacteristicsW.argtypes = [ctypes.c_wchar_p, ctypes.POINTER(ctypes.c_uint32)]
        avrt.AvSetMmThreadPriority.argtypes = [ctypes.c_void_p, ctypes.c_int]
        index = ctypes.c_uint32(0)
        h = avrt.AvSetMmThreadCharacteristicsW(task, ctypes.byref(index))
        if not h:
            return False
        avrt.AvSetMmThreadPriority(h, _AVRT_PRIORITY_HIGH)
        return True
    except OSError:
        return False


_ES_SYSTEM_REQUIRED, _ES_DISPLAY_REQUIRED, _ES_CONTINUOUS = 0x1, 0x2, 0x80000000


def keep_screen_on(on: bool = True) -> bool:
    """While the video plays, Windows doesn't turn the screen off or sleep, as video players and games ask. Without
    it, the power plan's "Turn off display after" (15 minutes, often) blanks the screen mid-game, because a controller
    isn't input to Windows: only a keyboard or a mouse is (chiaki-ng only asks on Linux). Lasts until the calling
    thread ends or calls again with on=False. Returns whether it worked."""
    if sys.platform != "win32":
        return False
    try:
        k32 = ctypes.WinDLL("kernel32")
        k32.SetThreadExecutionState.restype = ctypes.c_uint32
        k32.SetThreadExecutionState.argtypes = [ctypes.c_uint32]
        flags = _ES_CONTINUOUS | (_ES_DISPLAY_REQUIRED | _ES_SYSTEM_REQUIRED if on else 0)
        return k32.SetThreadExecutionState(flags) != 0
    except OSError:
        return False


def boost_gpu() -> bool:
    """Raises this process's GPU scheduling priority to high (what games and video players use), so other
    programs' GPU work doesn't delay ours. Returns whether it worked."""
    if _boost_off():
        return False
    try:
        gdi = ctypes.WinDLL("gdi32")
        gdi.D3DKMTSetProcessSchedulingPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_int]
        k32 = ctypes.WinDLL("kernel32")
        k32.GetCurrentProcess.restype = ctypes.c_void_p
        status = gdi.D3DKMTSetProcessSchedulingPriorityClass(k32.GetCurrentProcess(), _D3DKMT_PRIORITY_HIGH)
        return status == 0
    except (OSError, AttributeError):
        return False


def high_res_timer(on: bool = True) -> bool:
    """Requests a 1 ms Windows system timer resolution (timeBeginPeriod/timeEndPeriod) so short pacing
    sleeps wake on time instead of overshooting by up to 15.6 ms. Returns whether the call succeeded."""
    if sys.platform != "win32":
        return False
    try:
        winmm = ctypes.WinDLL("winmm")
        fn = winmm.timeBeginPeriod if on else winmm.timeEndPeriod
        fn.argtypes = [ctypes.c_uint]
        fn.restype = ctypes.c_uint
        return fn(1) == 0
    except (OSError, AttributeError):
        return False


def sleep_until(deadline: float, clock=time.perf_counter, sleep=time.sleep, spin_s: float = 0.002) -> float:
    """Sleep until `deadline`, using OS sleep for the coarse wait and a short busy-wait for the last ~2 ms
    so frame presentation lands with sub-millisecond jitter. Mocked `sleep` calls bypass spinning.

    The busy-wait yields the GIL on every pass (time.sleep(0)): a bare `pass` loop would hold it for the whole
    spin, so another Python thread (Clock60's reader handing in a frame that just arrived) couldn't run until the
    deadline, and the frame would miss the beat it came in time for."""
    rem = deadline - clock()
    if rem <= 0:
        return clock()
    if sleep is not time.sleep or spin_s <= 0:
        sleep(rem)
        return clock()
    if rem > spin_s + 0.0005:
        sleep(rem - spin_s)
    while (now := clock()) < deadline:
        time.sleep(0)  # let waiting threads have the GIL; returns at once when none is waiting
    return now
