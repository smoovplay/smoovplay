"""CUDA -> OpenGL texture copy, so frames already on the GPU are shown without a trip through the CPU.

The usual path downloads each frame to the CPU (~0.5-1 ms) and uploads it again with glTexSubImage2D
(~1.5-2 ms for 1080p BGR). Here the GL texture is registered with CUDA once and each frame is a
device-to-device copy into it (~0.05 ms). Uses the CUDA runtime that PyTorch already loaded, so it shares
PyTorch's context and stream ordering.
"""
from __future__ import annotations

import ctypes
import glob
import os
import sys

import torch

_GRAPHICS_REGISTER_WRITE_DISCARD = 2
_MEMCPY_DEVICE_TO_DEVICE = 3
_GL_TEXTURE_2D = 0x0DE1


def cudart_path() -> str:
    """PyTorch's CUDA runtime: cudart64_*.dll beside torch on Windows; on Linux, the libcudart.so PyTorch has
    already loaded (it ships in the nvidia-cuda-runtime wheel, or in torch/lib for some builds)."""
    if sys.platform == "win32":
        libdir = os.path.join(os.path.dirname(torch.__file__), "lib")
        for path in sorted(glob.glob(os.path.join(libdir, "cudart64_*.dll"))):
            return path
        raise OSError("PyTorch's CUDA runtime (cudart64_*.dll) not found")
    torch.cuda.init()
    with open("/proc/self/maps") as maps:
        for line in maps:
            path = line.split()[-1]
            if "/libcudart.so" in path:
                return path
    raise OSError("PyTorch's CUDA runtime (libcudart.so) isn't loaded")


def _cudart() -> ctypes.CDLL:
    return ctypes.CDLL(cudart_path())


class CudaGLTexture:
    """Copies HxWx3 uint8 BGR CUDA tensors into an existing GL_RGBA8 texture of the same size; with hdr, HxWx3
    float16 ones (framegen/hdr.py pq16) into a GL_RGBA16F texture.

    Create it, and call `upload`, on the thread whose GL context owns the texture.
    """

    def __init__(self, tex: int, w: int, h: int, hdr: bool = False) -> None:
        self.w, self.h = w, h
        self.bpp = 8 if hdr else 4  # bytes per staged pixel
        self.rt = rt = _cudart()
        p, i, u, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_size_t
        rt.cudaGraphicsGLRegisterImage.argtypes = [ctypes.POINTER(p), u, u, u]
        rt.cudaGraphicsMapResources.argtypes = [i, ctypes.POINTER(p), p]
        rt.cudaGraphicsUnmapResources.argtypes = [i, ctypes.POINTER(p), p]
        rt.cudaGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(p), p, u, u]
        rt.cudaMemcpy2DToArrayAsync.argtypes = [p, s, s, p, s, s, s, i, p]
        rt.cudaGraphicsUnregisterResource.argtypes = [p]
        torch.cuda.current_device()  # make sure the runtime is initialised on this thread
        self.res = p()
        self._check(rt.cudaGraphicsGLRegisterImage(ctypes.byref(self.res), tex, _GL_TEXTURE_2D,
                                                   _GRAPHICS_REGISTER_WRITE_DISCARD), "register texture")
        # RGBA staging tensor (the texture is RGBA8, or RGBA16F; CUDA can't map 3-channel GL textures). Alpha stays
        # opaque.
        self.rgba = (torch.ones((h, w, 4), dtype=torch.float16, device="cuda") if hdr else
                     torch.full((h, w, 4), 255, dtype=torch.uint8, device="cuda"))

    def _check(self, err: int, what: str) -> None:
        if err:
            raise RuntimeError(f"CUDA-GL interop: {what} failed (cudaError {err})")

    def upload(self, bgr: torch.Tensor) -> None:
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        self.rgba[..., :3].copy_(bgr.flip(-1))
        res = ctypes.byref(self.res)
        self._check(self.rt.cudaGraphicsMapResources(1, res, stream), "map")
        try:
            arr = ctypes.c_void_p()
            self._check(self.rt.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(arr), self.res, 0, 0), "get array")
            self._check(self.rt.cudaMemcpy2DToArrayAsync(arr, 0, 0, ctypes.c_void_p(self.rgba.data_ptr()),
                                                         self.w * self.bpp, self.w * self.bpp, self.h,
                                                         _MEMCPY_DEVICE_TO_DEVICE, stream), "copy")
        finally:
            # Unmapping orders GL's use of the texture after the copy (stream semantics), no CPU wait needed.
            self._check(self.rt.cudaGraphicsUnmapResources(1, res, stream), "unmap")

    def close(self) -> None:
        if self.res:
            self.rt.cudaGraphicsUnregisterResource(self.res)
            self.res = ctypes.c_void_p()


_GRAPHICS_REGISTER_SURFACE_LOAD_STORE = 4


class CudaGLImage:
    """A GL texture that CUDA writes (RTX VSR's output): GL_RGBA8 through a surface object, or GL_R32UI by copies.

    map() gives the texture's CUDA array; unmap() hands it back to GL, ordered on the same stream.
    """

    def __init__(self, tex: int) -> None:
        self.rt = rt = _cudart()
        p, i, u = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint
        rt.cudaGraphicsGLRegisterImage.argtypes = [ctypes.POINTER(p), u, u, u]
        rt.cudaGraphicsMapResources.argtypes = [i, ctypes.POINTER(p), p]
        rt.cudaGraphicsUnmapResources.argtypes = [i, ctypes.POINTER(p), p]
        rt.cudaGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(p), p, u, u]
        rt.cudaGraphicsUnregisterResource.argtypes = [p]
        self.res, self.array = p(), p()
        err = rt.cudaGraphicsGLRegisterImage(ctypes.byref(self.res), tex, _GL_TEXTURE_2D,
                                             _GRAPHICS_REGISTER_SURFACE_LOAD_STORE)
        if err:
            raise RuntimeError(f"CUDA-GL interop: register output texture failed (cudaError {err})")

    def map(self, stream: int) -> ctypes.c_void_p:
        s = ctypes.c_void_p(stream)
        err = self.rt.cudaGraphicsMapResources(1, ctypes.byref(self.res), s)
        if not err:
            err = self.rt.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(self.array), self.res, 0, 0)
            if err:
                self.rt.cudaGraphicsUnmapResources(1, ctypes.byref(self.res), s)
        if err:
            raise RuntimeError(f"CUDA-GL interop: map output texture failed (cudaError {err})")
        return self.array

    def unmap(self, stream: int) -> None:
        self.rt.cudaGraphicsUnmapResources(1, ctypes.byref(self.res), ctypes.c_void_p(stream))

    def close(self) -> None:
        if self.res:
            self.rt.cudaGraphicsUnregisterResource(self.res)
            self.res = ctypes.c_void_p()


_GRAPHICS_REGISTER_READ_ONLY = 1


class CudaGLReader:
    """Copies a GL_RGBA8 texture (w x h) into an (h, w, 4) uint8 CUDA tensor: a look drawn for instant replay
    (framegen/looks_gl.py). Texture row 0 lands in tensor row 0. Create and use it on the GL context's thread."""

    def __init__(self, tex: int, w: int, h: int) -> None:
        self.w, self.h = w, h
        self.rt = rt = _cudart()
        p, i, u, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_size_t
        rt.cudaGraphicsGLRegisterImage.argtypes = [ctypes.POINTER(p), u, u, u]
        rt.cudaGraphicsMapResources.argtypes = [i, ctypes.POINTER(p), p]
        rt.cudaGraphicsUnmapResources.argtypes = [i, ctypes.POINTER(p), p]
        rt.cudaGraphicsSubResourceGetMappedArray.argtypes = [ctypes.POINTER(p), p, u, u]
        rt.cudaMemcpy2DFromArrayAsync.argtypes = [p, s, p, s, s, s, s, i, p]
        rt.cudaGraphicsUnregisterResource.argtypes = [p]
        torch.cuda.current_device()
        self.res = p()
        err = rt.cudaGraphicsGLRegisterImage(ctypes.byref(self.res), tex, _GL_TEXTURE_2D, _GRAPHICS_REGISTER_READ_ONLY)
        if err:
            raise RuntimeError(f"CUDA-GL interop: register texture for reading failed (cudaError {err})")

    def read(self, out: torch.Tensor) -> None:
        """Into `out` (contiguous (h, w, 4) uint8 on the GPU), ordered on the current stream after GL's drawing."""
        stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
        res = ctypes.byref(self.res)
        err = self.rt.cudaGraphicsMapResources(1, res, stream)
        if err:
            raise RuntimeError(f"CUDA-GL interop: map failed (cudaError {err})")
        try:
            arr = ctypes.c_void_p()
            err = (self.rt.cudaGraphicsSubResourceGetMappedArray(ctypes.byref(arr), self.res, 0, 0)
                   or self.rt.cudaMemcpy2DFromArrayAsync(ctypes.c_void_p(out.data_ptr()), self.w * 4, arr, 0, 0,
                                                         self.w * 4, self.h, _MEMCPY_DEVICE_TO_DEVICE, stream))
            if err:
                raise RuntimeError(f"CUDA-GL interop: copy failed (cudaError {err})")
        finally:
            self.rt.cudaGraphicsUnmapResources(1, res, stream)

    def close(self) -> None:
        if self.res:
            self.rt.cudaGraphicsUnregisterResource(self.res)
            self.res = ctypes.c_void_p()
