"""A capture card's 10-bit P010 video through Windows Media Foundation, for HDR (Experimental).

FFmpeg's DirectShow input (framegen/capture.py) has no P010, so a card's 10-bit format can't be opened that way.
Media Foundation's source reader reads it straight off the card: tried with an Elgato HD60 X at 1920x1080 60 fps, a
steady 60 fps. Written with ctypes against Windows' own DLLs (mfplat, mf, mfreadwrite), nothing to install.

The reader runs on a thread of its own (COM multithreaded, as Media Foundation wants), from opening the card to
closing it, and hands each frame's bytes over in a small queue: the Y plane (16-bit samples, the value in the top 10
bits), then the interleaved U/V plane at half size, `stride` bytes a row.

    python -m framegen.mf_capture "Elgato HD60 X"     # the card's formats as Media Foundation lists them
"""
from __future__ import annotations

import queue
import sys
import threading
import time
from dataclasses import dataclass

import numpy as np

MF_VERSION = 0x00020070
FIRST_VIDEO = 0xFFFFFFFC  # MF_SOURCE_READER_FIRST_VIDEO_STREAM
READER_ERROR, READER_END, READER_TICK = 0x1, 0x2, 0x100
P010 = "30313050"  # the subtype GUID's first part (FourCC)

# vtable slots (IUnknown's three come first)
_GET_UINT32, _GET_UINT64, _GET_GUID, _GET_ALLOCATED_STRING, _SET_GUID = 7, 8, 10, 13, 24
_ACTIVATE_OBJECT, _SHUTDOWN_OBJECT = 33, 34  # IMFActivate
_SOURCE_SHUTDOWN = 12  # IMFMediaSource
_GET_NATIVE, _GET_CURRENT, _SET_CURRENT, _READ_SAMPLE = 5, 6, 7, 9  # IMFSourceReader
_TO_CONTIGUOUS = 41  # IMFSample
_LOCK, _UNLOCK = 3, 4  # IMFMediaBuffer

_SOURCE_TYPE = "c60ac5fe-252a-478f-a0ef-bc8fa5f7cad3"  # MF_DEVSOURCE_ATTRIBUTE_SOURCE_TYPE
_VIDCAP = "8ac3587a-4ae7-42d8-99e0-0a6013eef90f"  # ..._SOURCE_TYPE_VIDCAP_GUID
_FRIENDLY_NAME = "60d0e559-52f8-4fa2-bbce-acdb34a8ec01"  # MF_DEVSOURCE_ATTRIBUTE_FRIENDLY_NAME
_IID_MEDIA_SOURCE = "279a808d-aec7-40c8-9c6b-a6b492c78a66"
_SUBTYPE = "f7e34c9a-42e8-4714-b74b-cb29d72c35e5"  # MF_MT_SUBTYPE
_FRAME_SIZE = "1652c33d-d6b2-4012-b834-72030849a37d"  # MF_MT_FRAME_SIZE
_FRAME_RATE = "c459a2e8-3d2c-4e44-b132-fee5156c7bb0"  # MF_MT_FRAME_RATE
_STRIDE = "644b4e48-1e02-4516-b0eb-c01ca9d49ac6"  # MF_MT_DEFAULT_STRIDE


@dataclass(frozen=True)
class MfMode:
    fmt: str  # "p010", "nv12", "yuy2", ... (lower-case FourCC) or the subtype's first part
    w: int
    h: int
    fps: float


def pick_p010(modes: list[MfMode], w: int, h: int, fps: float) -> MfMode | None:
    """The card's P010 mode at exactly w x h, the slowest that reaches `fps` (else the fastest), or None."""
    at = [m for m in modes if m.fmt == "p010" and (m.w, m.h) == (w, h)]
    fast = [m for m in at if m.fps >= fps - 0.5]
    return min(fast, key=lambda m: m.fps) if fast else max(at, key=lambda m: m.fps, default=None)


class _Com:
    """The few COM and Media Foundation calls used, through ctypes. One per thread that uses them."""

    def __init__(self) -> None:
        import ctypes
        import uuid
        from ctypes import wintypes as wt
        self.ct, self.wt, self.uuid = ctypes, wt, uuid
        self.ole32 = ctypes.WinDLL("ole32")
        self.mfplat, self.mf = ctypes.WinDLL("mfplat"), ctypes.WinDLL("mf")
        self.mfrw = ctypes.WinDLL("mfreadwrite")
        self.ole32.CoTaskMemFree.argtypes = [ctypes.c_void_p]

        class Guid(ctypes.Structure):
            _fields_ = [("b", ctypes.c_ubyte * 16)]
        self.Guid = Guid

    def guid(self, s: str):
        g = self.Guid()
        self.ct.memmove(g.b, self.uuid.UUID(s).bytes_le, 16)
        return g

    def call(self, obj, index: int, *args, argtypes=()):
        ct = self.ct
        table = ct.cast(obj, ct.POINTER(ct.POINTER(ct.c_void_p))).contents
        return ct.WINFUNCTYPE(ct.c_long, ct.c_void_p, *argtypes)(table[index])(obj, *args)

    def release(self, obj) -> None:
        if obj:
            self.call(obj, 2)

    @staticmethod
    def check(hr: int, what: str) -> None:
        if hr < 0:
            raise OSError(f"Media Foundation: {what} failed (0x{hr & 0xFFFFFFFF:08X})")

    def u32(self, attrs, key: str) -> int | None:
        v = self.ct.c_uint32()
        ok = self.call(attrs, _GET_UINT32, self.ct.byref(self.guid(key)), self.ct.byref(v),
                       argtypes=(self.ct.c_void_p, self.ct.c_void_p)) >= 0
        return v.value if ok else None

    def u64(self, attrs, key: str) -> int | None:
        v = self.ct.c_uint64()
        ok = self.call(attrs, _GET_UINT64, self.ct.byref(self.guid(key)), self.ct.byref(v),
                       argtypes=(self.ct.c_void_p, self.ct.c_void_p)) >= 0
        return v.value if ok else None

    def mode(self, mt) -> MfMode:
        g = self.Guid()
        sub = "?"
        if self.call(mt, _GET_GUID, self.ct.byref(self.guid(_SUBTYPE)), self.ct.byref(g),
                     argtypes=(self.ct.c_void_p, self.ct.c_void_p)) >= 0:
            s = str(self.uuid.UUID(bytes_le=bytes(g.b)))[:8]
            sub = bytes.fromhex(s)[::-1].decode("ascii", "replace").lower() if s.isalnum() else s
        size, rate = self.u64(mt, _FRAME_SIZE) or 0, self.u64(mt, _FRAME_RATE) or 0
        return MfMode(sub, size >> 32, size & 0xFFFFFFFF, (rate >> 32) / max(1, rate & 0xFFFFFFFF))


class P010Reader:
    """A card's P010 video, read on a thread of its own. Use open() (None and a reason when the card has no P010 at
    that size, or can't be opened). get(timeout) -> (frame bytes as a uint8 array, arrival time, the card's own time
    in seconds) or None once it has stopped (error: why)."""

    def __init__(self, device: str, w: int, h: int, fps: float, queue_frames: int = 2) -> None:
        self.device, self.w, self.h, self.want_fps = device, w, h, fps
        self.fps, self.stride, self.modes = fps, w * 2, []
        self.error = ""
        self.dropped = 0  # frames this reader dropped because the pipeline hadn't taken the last ones yet
        self._q: queue.Queue = queue.Queue(maxsize=queue_frames)
        self._ready = threading.Event()
        self._stop = False
        self._done = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True, name="mf-capture")

    @classmethod
    def open(cls, device: str, w: int, h: int, fps: float, timeout_s: float = 15.0) -> tuple["P010Reader | None", str]:
        r = cls(device, w, h, fps)
        r._thread.start()
        if not r._ready.wait(timeout_s):
            r._stop = True
            return None, f"Media Foundation took over {timeout_s:g} s to open {device}"
        if r.error:
            return None, r.error
        return r, ""

    @property
    def nbytes(self) -> int:
        return self.stride * self.h * 3 // 2

    def get(self, timeout: float = 0.5):
        """The next frame, None once the reader has stopped, or raises queue.Empty after `timeout`."""
        item = self._q.get(timeout=timeout)
        return item if item is not False else None

    def _put(self, item) -> None:
        try:
            self._q.put_nowait(item)
        except queue.Full:
            try:
                self._q.get_nowait()
                if item is not False:
                    self.dropped += 1
            except queue.Empty:
                pass
            self._q.put_nowait(item)

    def _run(self) -> None:
        try:
            com = _Com()
        except OSError as e:
            self.error = f"Media Foundation isn't available ({e})"
            self._ready.set()
            return
        ct, byref, c_void_p = com.ct, com.ct.byref, com.ct.c_void_p
        init = com.ole32.CoInitializeEx(None, 0)  # COINIT_MULTITHREADED
        started = com.mfplat.MFStartup(MF_VERSION, 0) >= 0
        devices, count, source, reader, act = ct.POINTER(c_void_p)(), ct.c_uint32(), c_void_p(), c_void_p(), None
        try:
            if not started:
                raise OSError("MFStartup failed")
            attrs = c_void_p()
            com.check(com.mfplat.MFCreateAttributes(byref(attrs), 1), "MFCreateAttributes")
            try:
                com.check(com.call(attrs, _SET_GUID, byref(com.guid(_SOURCE_TYPE)), byref(com.guid(_VIDCAP)),
                                   argtypes=(c_void_p, c_void_p)), "SetGUID")
                com.check(com.mf.MFEnumDeviceSources(attrs, byref(devices), byref(count)), "MFEnumDeviceSources")
            finally:
                com.release(attrs)
            names = []
            for i in range(count.value):
                name, n = ct.c_wchar_p(), ct.c_uint32()
                com.call(devices[i], _GET_ALLOCATED_STRING, byref(com.guid(_FRIENDLY_NAME)), byref(name), byref(n),
                         argtypes=(c_void_p, c_void_p, c_void_p))
                names.append(name.value or "")
                if name.value:
                    com.ole32.CoTaskMemFree(name)
            exact = [i for i, n in enumerate(names) if n.lower() == self.device.lower()]
            near = [i for i, n in enumerate(names) if self.device.lower() in n.lower()]
            pick = (exact or near or [None])[0]
            if pick is None:
                raise LookupError(f"Media Foundation doesn't list {self.device}")
            act = devices[pick]
            com.check(com.call(act, _ACTIVATE_OBJECT, byref(com.guid(_IID_MEDIA_SOURCE)), byref(source),
                               argtypes=(c_void_p, c_void_p)), "opening the card")
            com.check(com.mfrw.MFCreateSourceReaderFromMediaSource(source, None, byref(reader)), "the source reader")
            types = []
            for i in range(1000):
                mt = c_void_p()
                if com.call(reader, _GET_NATIVE, FIRST_VIDEO, i, byref(mt), argtypes=(ct.c_uint32, ct.c_uint32,
                                                                                         c_void_p)) < 0:
                    break
                types.append((com.mode(mt), mt))
            self.modes = list(dict.fromkeys(m for m, _ in types))
            want = pick_p010(self.modes, self.w, self.h, self.want_fps)
            if want is None:
                for _, mt in types:
                    com.release(mt)
                sizes = sorted({(m.w, m.h) for m in self.modes if m.fmt == "p010"}, reverse=True)
                raise LookupError(f"{self.device} has no 10-bit (P010) mode at {self.w}x{self.h}" + (
                    " (it has " + ", ".join(f"{w}x{h}" for w, h in sizes) + ")" if sizes else ""))
            chosen = next(mt for m, mt in types if m == want)
            try:
                com.check(com.call(reader, _SET_CURRENT, FIRST_VIDEO, None, chosen,
                                   argtypes=(ct.c_uint32, c_void_p, c_void_p)), "choosing P010")
            finally:
                for _, mt in types:
                    com.release(mt)
            cur = c_void_p()
            com.check(com.call(reader, _GET_CURRENT, FIRST_VIDEO, byref(cur), argtypes=(ct.c_uint32, c_void_p)),
                      "reading the format back")
            stride = com.u32(cur, _STRIDE)
            com.release(cur)
            self.fps = want.fps
            if stride is not None and ct.c_int32(stride).value >= self.w * 2:  # (negative: bottom-up, not for YUV)
                self.stride = ct.c_int32(stride).value
            self._ready.set()
            self._read(com, reader)
        except (OSError, LookupError) as e:
            self.error = str(e)
        finally:
            self._ready.set()
            com.release(reader)
            if source:
                com.call(source, _SOURCE_SHUTDOWN)  # lets the card go now, not when the process ends
                com.release(source)
            if act is not None:
                com.call(act, _SHUTDOWN_OBJECT)
            for i in range(count.value):
                com.release(devices[i])
            if devices:
                com.ole32.CoTaskMemFree(ct.cast(devices, c_void_p))
            if started:
                com.mfplat.MFShutdown()
            if init >= 0:
                com.ole32.CoUninitialize()
            self._put(False)
            self._done.set()

    def _read(self, com, reader) -> None:
        ct, byref, c_void_p = com.ct, com.ct.byref, com.ct.c_void_p
        n = self.nbytes
        while not self._stop:
            idx, flags, ts, sample = ct.c_uint32(), ct.c_uint32(), ct.c_longlong(), c_void_p()
            com.check(com.call(reader, _READ_SAMPLE, FIRST_VIDEO, 0, byref(idx), byref(flags), byref(ts),
                               byref(sample), argtypes=(ct.c_uint32, ct.c_uint32, c_void_p, c_void_p, c_void_p,
                                                        c_void_p)), "reading a frame")
            if flags.value & (READER_ERROR | READER_END):
                raise OSError(f"the card stopped sending video (flags 0x{flags.value:x})")
            if not sample:
                continue
            t_arrival = time.perf_counter()
            buf = c_void_p()
            try:
                com.check(com.call(sample, _TO_CONTIGUOUS, byref(buf), argtypes=(c_void_p,)), "a frame's buffer")
                p, cap, length = c_void_p(), ct.c_uint32(), ct.c_uint32()
                com.check(com.call(buf, _LOCK, byref(p), byref(cap), byref(length),
                                   argtypes=(c_void_p, c_void_p, c_void_p)), "locking a frame")
                try:
                    if length.value < n:
                        continue  # a short frame (the format changing): skip it
                    frame = np.empty(n, np.uint8)
                    ct.memmove(frame.ctypes.data, p.value, n)
                finally:
                    com.call(buf, _UNLOCK)
            finally:
                com.release(buf)
                com.release(sample)
            self._put((frame, t_arrival, ts.value / 1e7))

    def close(self, wait_s: float = 2.0) -> None:
        """Stops reading and lets the card go (waits up to wait_s for the frame being read to come)."""
        self._stop = True
        try:
            while True:
                self._q.get_nowait()  # room for the reader's last put
        except queue.Empty:
            pass
        self._done.wait(wait_s)


def main(argv: list[str]) -> int:
    if not argv:
        print('usage: python -m framegen.mf_capture "<card name>"')
        return 2
    r = P010Reader(argv[0], 0, 0, 60.0)  # there's no 0x0 mode: it lists the formats, then stops
    r._thread.start()
    r._done.wait(15.0)
    for m in r.modes:
        print(f"{m.fmt:5} {m.w}x{m.h} {m.fps:g} fps")
    if not r.modes:
        print(r.error)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
