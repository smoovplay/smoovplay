"""What the frame tap needs from the operating system: named shared memory, the clock its timestamps use, and
the reader's wake-up when a frame lands. Windows and Linux, matching chiaki_tap/frametap.c.

  Windows  a named file mapping, QueryPerformanceCounter, and a named auto-reset event ("<name>-frame")
  Linux    /dev/shm/<the name after its last backslash>, CLOCK_MONOTONIC in nanoseconds (what
           time.perf_counter() reads there), and a futex on the low 32 bits of the ring's write_seq
"""
from __future__ import annotations

import ctypes
import mmap
import os
import sys
import time

WINDOWS = sys.platform == "win32"
LINUX = sys.platform.startswith("linux")
SUPPORTED = WINDOWS or LINUX

if WINDOWS:
    _k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    _k32.CreateEventW.restype = ctypes.c_void_p
    _k32.CreateEventW.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int, ctypes.c_wchar_p]
    _k32.WaitForSingleObject.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
    _k32.SetEvent.argtypes = [ctypes.c_void_p]
    _k32.CloseHandle.argtypes = [ctypes.c_void_p]

    def qpc() -> int:
        v = ctypes.c_int64()
        _k32.QueryPerformanceCounter(ctypes.byref(v))
        return v.value

    def qpc_freq() -> int:
        f = ctypes.c_int64()
        _k32.QueryPerformanceFrequency(ctypes.byref(f))
        return f.value

elif LINUX:
    _libc = ctypes.CDLL(None, use_errno=True)
    _libc.syscall.restype = ctypes.c_long
    _SYS_FUTEX = {"x86_64": 202, "aarch64": 98}.get(os.uname().machine)
    _FUTEX_WAIT, _FUTEX_WAKE = 0, 1

    class _Timespec(ctypes.Structure):
        _fields_ = [("tv_sec", ctypes.c_long), ("tv_nsec", ctypes.c_long)]

    def qpc() -> int:
        return time.clock_gettime_ns(time.CLOCK_MONOTONIC)

    def qpc_freq() -> int:
        return 1_000_000_000


def unsupported() -> RuntimeError:
    return RuntimeError("the chiaki frame tap works on Windows and Linux only")


def open_map(name: str, size: int) -> mmap.mmap:
    """The named shared memory, created if neither side has made it yet."""
    if WINDOWS:
        return mmap.mmap(-1, size, tagname=name)
    if not LINUX:
        raise unsupported()
    path = "/dev/shm/" + name.rsplit("\\", 1)[-1]
    fd = os.open(path, os.O_RDWR | os.O_CREAT, 0o600)
    try:
        if os.fstat(fd).st_size < size:
            os.ftruncate(fd, size)
        return mmap.mmap(fd, size)
    finally:
        os.close(fd)  # the mapping keeps the memory


class Signal:
    """A frame landed. One reader waits; the writer sets it after every frame and at the stream's end.

    Behaves like Windows' auto-reset event: a wait() returns at once if the writer set it since the previous
    wait(), and otherwise sleeps until it does or the timeout passes. On Linux it is a futex on write_seq's low
    word (offset 32 of `mm`), so it needs nothing beyond the shared memory itself."""

    def __init__(self, mm: mmap.mmap, event_name: str, seq_offset: int = 32) -> None:
        if WINDOWS:
            self._event = _k32.CreateEventW(None, 0, 0, event_name)
            return
        if not LINUX or _SYS_FUTEX is None:
            raise unsupported()
        self._word = ctypes.c_uint32.from_buffer(mm, seq_offset)
        self._addr = ctypes.addressof(self._word)
        self._seen = self._word.value

    def wait(self, timeout_s: float) -> None:
        if WINDOWS:
            _k32.WaitForSingleObject(self._event, int(timeout_s * 1000))
            return
        cur = self._word.value
        if cur == self._seen:  # nothing new since the last wait: sleep (returns at once if it changes first)
            ts = _Timespec(int(timeout_s), int((timeout_s % 1) * 1e9))
            _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._addr), _FUTEX_WAIT, ctypes.c_uint32(cur),
                          ctypes.byref(ts), None, 0)
        self._seen = self._word.value

    def set(self) -> None:
        if WINDOWS:
            _k32.SetEvent(self._event)
        else:
            _libc.syscall(_SYS_FUTEX, ctypes.c_void_p(self._addr), _FUTEX_WAKE, 1, None, None, 0)

    def close(self) -> None:
        if WINDOWS:
            _k32.CloseHandle(self._event)
        else:
            del self._word  # release the buffer export, so the mmap can close
