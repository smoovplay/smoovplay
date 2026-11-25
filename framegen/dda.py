"""Window capture on the GPU: Desktop Duplication of one monitor, the window's part copied on the graphics card and
handed to CUDA, so frames never go through the CPU (framegen/window.py; FFmpeg's ddagrab is the CPU fallback).

Each time the screen changes, Windows hands over the monitor's picture as a Direct3D 11 texture
(IDXGIOutputDuplication::AcquireNextFrame). If the window's part changed (Windows says which parts did), it's copied
into a texture of ours that CUDA can read (cudaGraphicsD3D11RegisterResource), then into a PyTorch tensor: BGR at the
source's size, scaled on the GPU if the window isn't that size. When Windows last updated the screen
(LastPresentTime) comes with it, on the same clock as time.perf_counter.

Needs the monitor on the graphics card CUDA runs on (a desktop PC with the monitor plugged into the NVIDIA card),
not rotated, and not in Windows HDR mode; otherwise GpuCaptureUnavailable says why and window.py uses ddagrab.
"""
from __future__ import annotations

import ctypes
from ctypes import POINTER, byref, c_int, c_long, c_size_t, c_uint, c_void_p

from .dxgi_present import (D3D11_BIND_SHADER_RESOURCE, D3D11_SDK_VERSION, D3D11_USAGE_DEFAULT,
                           D3D_DRIVER_TYPE_UNKNOWN, DEVICE_CREATE_TEXTURE2D, DXGI_FORMAT_B8G8R8A8_UNORM,
                           FACTORY_ENUM_ADAPTERS1, IID_ID3D11Texture2D, IID_IDXGIFactory1, GUID, Texture2DDesc,
                           _check, _com, _query, _release)

IID_IDXGIOutput1 = "00cddea8-939b-4b83-a340-a685226666cc"
ADAPTER_ENUM_OUTPUTS = 7
OUTPUT1_DUPLICATE_OUTPUT = 22
DUP_GET_DESC, DUP_ACQUIRE_NEXT_FRAME, DUP_GET_DIRTY_RECTS, DUP_GET_MOVE_RECTS, DUP_RELEASE_FRAME = 7, 8, 9, 10, 14
CONTEXT_COPY_SUBRESOURCE_REGION = 46
DXGI_ERROR_WAIT_TIMEOUT = ctypes.c_long(0x887A0027).value
DXGI_ERROR_ACCESS_LOST = ctypes.c_long(0x887A0026).value
DXGI_ERROR_MORE_DATA = ctypes.c_long(0x887A0003).value
_CUDA_MAP_READ_ONLY = 1
_MEMCPY_DEVICE_TO_DEVICE = 3


class GpuCaptureUnavailable(RuntimeError):
    """The GPU path can't run on this PC or monitor (the reason is the message): use the CPU path instead."""


class CaptureLost(RuntimeError):
    """Windows took the duplication away (a mode change, a UAC prompt, a fullscreen app): start again."""


class _Rect(ctypes.Structure):
    _fields_ = [("left", c_long), ("top", c_long), ("right", c_long), ("bottom", c_long)]


class _MoveRect(ctypes.Structure):  # DXGI_OUTDUPL_MOVE_RECT
    _fields_ = [("x", c_long), ("y", c_long), ("dest", _Rect)]


class _DuplDesc(ctypes.Structure):  # DXGI_OUTDUPL_DESC
    _fields_ = [("Width", c_uint), ("Height", c_uint), ("RefreshNum", c_uint), ("RefreshDen", c_uint),
                ("Format", c_uint), ("ScanlineOrdering", c_uint), ("Scaling", c_uint), ("Rotation", c_uint),
                ("InSystemMemory", c_int)]


class _FrameInfo(ctypes.Structure):  # DXGI_OUTDUPL_FRAME_INFO
    _fields_ = [("LastPresentTime", ctypes.c_longlong), ("LastMouseUpdateTime", ctypes.c_longlong),
                ("AccumulatedFrames", c_uint), ("RectsCoalesced", c_int), ("ProtectedContentMaskedOut", c_int),
                ("PointerX", c_long), ("PointerY", c_long), ("PointerVisible", c_int),
                ("TotalMetadataBufferSize", c_uint), ("PointerShapeBufferSize", c_uint)]


class _Box(ctypes.Structure):  # D3D11_BOX
    _fields_ = [("left", c_uint), ("top", c_uint), ("front", c_uint), ("right", c_uint), ("bottom", c_uint),
                ("back", c_uint)]


def _overlaps(r: _Rect, x: int, y: int, w: int, h: int) -> bool:
    return r.left < x + w and r.right > x and r.top < y + h and r.bottom > y


def is_unchanged(cur, prev) -> bool:
    """True when `cur` and `prev` tensors are bit-identical (e.g. our own overlay triggered a DWM wake)."""
    import torch
    return prev is not None and bool(torch.equal(cur, prev))


class DdaCapture:
    """The part (x, y, w, h) of output `output` of DXGI adapter 0 (window.plan_grab's Grab), as HxWx3 uint8 BGR CUDA
    tensors of out_w x out_h. Use it from one thread; close() it when done (a monitor allows only a few)."""

    def __init__(self, output: int, x: int, y: int, w: int, h: int, out_w: int, out_h: int) -> None:
        import torch
        from .glcuda import cudart_path
        self.x, self.y, self.w, self.h, self.out_w, self.out_h = x, y, w, h, out_w, out_h
        self.factory = self.adapter = self.output = self.device = self.context = self.dup = None
        self.texture, self.res, self._held = None, c_void_p(), False
        self.copies = self.skipped = self.unchanged = 0  # copied; elsewhere on monitor; bit-identical desktop wakes
        self._prev_bgra = None
        self._stream = None
        freq = ctypes.c_longlong()
        ctypes.windll.kernel32.QueryPerformanceFrequency(byref(freq))
        self._qpf = float(freq.value)
        try:
            if not torch.cuda.is_available():
                raise GpuCaptureUnavailable("no CUDA")
            self.rt = rt = ctypes.CDLL(cudart_path())
            p, u, s = c_void_p, c_uint, c_size_t
            rt.cudaD3D11GetDevice.argtypes = [POINTER(c_int), p]
            rt.cudaGraphicsD3D11RegisterResource.argtypes = [POINTER(p), p, u]
            rt.cudaGraphicsResourceSetMapFlags.argtypes = [p, u]
            rt.cudaGraphicsMapResources.argtypes = [c_int, POINTER(p), p]
            rt.cudaGraphicsUnmapResources.argtypes = [c_int, POINTER(p), p]
            rt.cudaGraphicsSubResourceGetMappedArray.argtypes = [POINTER(p), p, u, u]
            rt.cudaMemcpy2DFromArrayAsync.argtypes = [p, s, p, s, s, s, s, c_int, p]
            rt.cudaGraphicsUnregisterResource.argtypes = [p]
            torch.cuda.current_device()  # the runtime initialised, PyTorch's context current on this thread
            self._stream = torch.cuda.Stream()  # dedicated non-blocking stream: never serializes with RIFE graphs
            self._open(output)
        except Exception:
            self.close()
            raise

    def _open(self, output: int) -> None:
        import torch
        self.factory = c_void_p()
        _check(ctypes.windll.dxgi.CreateDXGIFactory1(byref(GUID.of(IID_IDXGIFactory1)), byref(self.factory)),
               "CreateDXGIFactory1")
        self.adapter = c_void_p()
        _check(_com(self.factory, FACTORY_ENUM_ADAPTERS1, [c_uint, POINTER(c_void_p)], 0, byref(self.adapter)),
               "EnumAdapters1")
        cuda_dev = c_int(-1)
        if self.rt.cudaD3D11GetDevice(byref(cuda_dev), self.adapter) or cuda_dev.value != torch.cuda.current_device():
            raise GpuCaptureUnavailable("the monitor isn't on the graphics card CUDA runs on")
        out = c_void_p()
        _check(_com(self.adapter, ADAPTER_ENUM_OUTPUTS, [c_uint, POINTER(c_void_p)], output, byref(out)),
               "EnumOutputs")
        self.output = _query(out, IID_IDXGIOutput1)
        _release(out)
        if self.output is None:
            raise GpuCaptureUnavailable("Desktop Duplication needs Windows 8 or newer")
        self.device, self.context, level = c_void_p(), c_void_p(), c_uint()
        _check(ctypes.windll.d3d11.D3D11CreateDevice(self.adapter, D3D_DRIVER_TYPE_UNKNOWN, None, 0, None, 0,
                                                     D3D11_SDK_VERSION, byref(self.device), byref(level),
                                                     byref(self.context)), "D3D11CreateDevice")
        self.dup = c_void_p()
        hr = _com(self.output, OUTPUT1_DUPLICATE_OUTPUT, [c_void_p, POINTER(c_void_p)], self.device, byref(self.dup))
        if hr < 0:  # e.g. DXGI_ERROR_UNSUPPORTED in Windows HDR mode, or too many duplications of this monitor
            raise GpuCaptureUnavailable(f"DuplicateOutput failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")
        desc = _DuplDesc()
        _com(self.dup, DUP_GET_DESC, [POINTER(_DuplDesc)], byref(desc), restype=None)
        if desc.Rotation not in (0, 1):
            raise GpuCaptureUnavailable("the monitor is rotated")
        if desc.InSystemMemory or desc.Format != DXGI_FORMAT_B8G8R8A8_UNORM:
            raise GpuCaptureUnavailable(f"the screen's picture isn't 8-bit on the GPU (format {desc.Format})")
        if self.x + self.w > desc.Width or self.y + self.h > desc.Height:
            raise GpuCaptureUnavailable("the region is outside the monitor")
        td = Texture2DDesc(self.w, self.h, 1, 1, DXGI_FORMAT_B8G8R8A8_UNORM, 1, 0, D3D11_USAGE_DEFAULT,
                           D3D11_BIND_SHADER_RESOURCE, 0, 0)
        self.texture = c_void_p()
        _check(_com(self.device, DEVICE_CREATE_TEXTURE2D, [POINTER(Texture2DDesc), c_void_p, POINTER(c_void_p)],
                    byref(td), None, byref(self.texture)), "CreateTexture2D")
        err = self.rt.cudaGraphicsD3D11RegisterResource(byref(self.res), self.texture, 0)
        if err:
            self.res = c_void_p()
            raise GpuCaptureUnavailable(f"CUDA can't read a Direct3D texture here (cudaError {err})")
        self.rt.cudaGraphicsResourceSetMapFlags(self.res, _CUDA_MAP_READ_ONLY)
        self.bgra = torch.empty((self.h, self.w, 4), dtype=torch.uint8, device="cuda")

    def _changed(self, info: _FrameInfo) -> bool:
        """Whether this screen update touched our part of the monitor (Windows lists the parts it redrew)."""
        if info.TotalMetadataBufferSize == 0:
            return True
        buf = (ctypes.c_ubyte * info.TotalMetadataBufferSize)()
        need = c_uint()
        for slot, item in ((DUP_GET_MOVE_RECTS, _MoveRect), (DUP_GET_DIRTY_RECTS, _Rect)):
            hr = _com(self.dup, slot, [c_uint, c_void_p, POINTER(c_uint)], len(buf), buf, byref(need))
            if hr < 0:
                return True  # can't tell: copy it
            rects = (item * (need.value // ctypes.sizeof(item))).from_buffer(buf)
            for r in rects:
                if _overlaps(r.dest if item is _MoveRect else r, self.x, self.y, self.w, self.h):
                    return True
        return False

    def grab(self, timeout_ms: int = 100):
        """(frame, t_acquired, t_present, ready) when our part of the screen has changed, else None (no change within
        the timeout, or a change elsewhere). `ready` is a torch.cuda.Event recorded on `self._stream` after `to_bgr`,
        so the consumer's CUDA stream can `wait_event(ready)` instead of blocking this thread on the CPU. Raises
        CaptureLost when Windows takes the duplication away."""
        import time

        import torch
        self._release_frame()
        info, resource = _FrameInfo(), c_void_p()
        hr = _com(self.dup, DUP_ACQUIRE_NEXT_FRAME, [c_uint, POINTER(_FrameInfo), POINTER(c_void_p)], timeout_ms,
                  byref(info), byref(resource))
        if hr == DXGI_ERROR_WAIT_TIMEOUT:
            return None
        if hr == DXGI_ERROR_ACCESS_LOST:
            raise CaptureLost("Desktop Duplication lost access")
        _check(hr, "AcquireNextFrame")
        self._held = True
        t_acquired = time.perf_counter()
        try:
            if info.LastPresentTime == 0 or not self._changed(info):  # only the mouse moved, or another part changed
                self.skipped += info.LastPresentTime != 0
                return None
            tex = _query(resource, IID_ID3D11Texture2D)
            if tex is None:
                raise GpuCaptureUnavailable("the screen's picture isn't a Direct3D texture")
            try:
                box = _Box(self.x, self.y, 0, self.x + self.w, self.y + self.h, 1)
                _com(self.context, CONTEXT_COPY_SUBRESOURCE_REGION,
                     [c_void_p, c_uint, c_uint, c_uint, c_uint, c_void_p, c_uint, POINTER(_Box)],
                     self.texture, 0, 0, 0, 0, tex, 0, byref(box), restype=None)
            finally:
                _release(tex)
        finally:
            _release(resource)
        stream = self._stream if self._stream is not None else torch.cuda.current_stream()
        s = c_void_p(stream.cuda_stream)
        res = byref(self.res)
        _cuda(self.rt.cudaGraphicsMapResources(1, res, s), "map")  # waits for the copy above
        try:
            arr = c_void_p()
            _cuda(self.rt.cudaGraphicsSubResourceGetMappedArray(byref(arr), self.res, 0, 0), "mapped array")
            _cuda(self.rt.cudaMemcpy2DFromArrayAsync(c_void_p(self.bgra.data_ptr()), self.w * 4, arr, 0, 0,
                                                     self.w * 4, self.h, _MEMCPY_DEVICE_TO_DEVICE, s), "copy")
        finally:
            _cuda(self.rt.cudaGraphicsUnmapResources(1, res, s), "unmap")
        self._release_frame()  # Windows can reuse its picture now: ours is copied
        with torch.cuda.stream(stream):
            if is_unchanged(self.bgra, self._prev_bgra):
                self.unchanged += 1
                return None
            if self._prev_bgra is None:
                self._prev_bgra = self.bgra.clone()
            else:
                self._prev_bgra.copy_(self.bgra)
            frame = to_bgr(self.bgra, self.out_w, self.out_h)
            ready = torch.cuda.Event()
            ready.record(stream)
        self.copies += 1
        t_present = info.LastPresentTime / self._qpf  # QueryPerformanceCounter, as time.perf_counter on Windows
        return frame, t_acquired, (t_present if abs(t_present - t_acquired) < 1.0 else None), ready

    def _release_frame(self) -> None:
        if self._held:
            self._held = False
            _com(self.dup, DUP_RELEASE_FRAME, [])

    def close(self) -> None:
        stream, self._stream = self._stream, None
        if stream is not None:
            try:
                stream.synchronize()
            except Exception:  # noqa: BLE001 - closing
                pass
        self._prev_bgra = None
        if self.dup:
            self._release_frame()
        if self.res:
            self.rt.cudaGraphicsUnregisterResource(self.res)
            self.res = c_void_p()
        for name in ("texture", "dup", "context", "device", "output", "adapter", "factory"):
            _release(getattr(self, name))
            setattr(self, name, None)


def to_bgr(bgra, out_w: int, out_h: int):
    """HxWx4 BGRA -> out_h x out_w x 3 BGR, on the GPU (bilinear, smoothed when shrinking, as FFmpeg's scale is
    near enough)."""
    import torch.nn.functional as F
    h, w = bgra.shape[:2]
    if (w, h) == (out_w, out_h):
        return bgra[..., :3].contiguous()
    x = bgra[..., :3].permute(2, 0, 1).unsqueeze(0).float()
    x = F.interpolate(x, size=(out_h, out_w), mode="bilinear", align_corners=False, antialias=out_w < w)
    return x.squeeze(0).permute(1, 2, 0).round_().clamp_(0, 255).to(dtype=bgra.dtype).contiguous()


def _cuda(err: int, what: str) -> None:
    if err:
        raise CaptureLost(f"CUDA {what} failed (cudaError {err})")
