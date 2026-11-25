"""NVIDIA RTX Video Super Resolution (RTX VSR) on CUDA frames, two ways (make_vsr picks):

- VfxVsr, what the installed app uses (Experimental): the VideoSuperRes effect in NVIDIA's own `nvidia-vfx` package,
  the one NVIDIA VFG comes in (framegen/vfg_setup.py). Each user downloads it from NVIDIA after accepting NVIDIA's
  licence, so SmoovPlay ships nothing of NVIDIA's. That licence forbids publishing benchmark results: its timings
  stay on the user's PC (never in the log, which Copy diagnostics includes). Like VFG, the launcher only offers the
  package on RTX 40 and 50 series cards.
- RtxVsr, for developers: the RTX Video SDK (free developer download, NVIDIA RTX SDKs licence), not part of this
  repo: unpack it, point NV_RTX_VIDEO_SDK at that folder, and build the small ctypes wrapper with
  native\\rtxvsr\\build.bat. The wrapper links a library from the SDK, so it can't ship in the MIT app. The model
  itself (nvngx_vsr.dll) is loaded from the SDK's bin folder; NVIDIA's runtime in the driver may update it.
  FRAMEGEN_VSR=sdk picks it when both are there (=vfx the other).

Input is the window's 8-bit RGBA CUDA tensor; output goes to a CUDA array (a mapped GL texture), on PyTorch's
current stream, so it orders with the interpolation and display work without CPU waits. RtxVsr writes RGBA8.
VfxVsr writes NVIDIA's packed 10-bit RGB (out_format "rgb10a2") into an R32UI texture that the window reads through
an RGB10_A2 view of the same memory, so the output is never unpacked on our side.
"""
from __future__ import annotations

import ctypes
import os
from pathlib import Path

import torch

from .glcuda import _cudart, cudart_path

ROOT = Path(__file__).resolve().parent.parent
SDK_DIR = Path(os.environ["NV_RTX_VIDEO_SDK"]) if os.environ.get("NV_RTX_VIDEO_SDK") else None
WRAPPER = ROOT / "native" / "rtxvsr" / "build" / "rtxvsr.dll"
DATA_DIR = ROOT / "out" / "ngx"  # NGX logs / caches

QUALITY = {"bicubic": 0, "low": 1, "medium": 2, "high": 3, "ultra": 4}
CLEAN = 15  # VfxVsr: quality + CLEAN = NVIDIA's high-bitrate models (16-19), for sources without compression

_MEMCPY_DEVICE_TO_DEVICE = 3
_ARRAY_SURFACE_LOAD_STORE = 0x02
_RESOURCE_TYPE_ARRAY = 0


class _ChannelDesc(ctypes.Structure):
    _fields_ = [("x", ctypes.c_int), ("y", ctypes.c_int), ("z", ctypes.c_int), ("w", ctypes.c_int),
                ("f", ctypes.c_int)]


_RGBA8 = _ChannelDesc(8, 8, 8, 8, 1)  # cudaChannelFormatKindUnsigned


def _array_resource_desc(array: ctypes.c_void_p) -> ctypes.Array:
    """cudaResourceDesc for a CUDA array: resType at offset 0, the array handle at offset 8, rest zero.
    Over-allocated so newer, larger layouts of the struct read zeros."""
    buf = (ctypes.c_ubyte * 256)()
    ctypes.c_int.from_buffer(buf, 0).value = _RESOURCE_TYPE_ARRAY
    ctypes.c_void_p.from_buffer(buf, 8).value = array.value
    return buf


def _linear_texture_desc() -> ctypes.Array:
    """cudaTextureDesc: clamp addressing, linear filtering, normalised float reads, normalised coordinates
    (what the SDK's own CUDA sample uses for its input)."""
    buf = (ctypes.c_ubyte * 256)()
    for i in range(3):
        ctypes.c_int.from_buffer(buf, 4 * i).value = 1  # addressMode[i] = cudaAddressModeClamp
    ctypes.c_int.from_buffer(buf, 12).value = 1  # filterMode = cudaFilterModeLinear
    ctypes.c_int.from_buffer(buf, 16).value = 1  # readMode = cudaReadModeNormalizedFloat
    ctypes.c_int.from_buffer(buf, 40).value = 1  # normalizedCoords
    return buf


def sdk_reason() -> str | None:
    """None if the RTX Video SDK and the wrapper are in place (developers), otherwise what is missing."""
    if SDK_DIR is None:
        return "NV_RTX_VIDEO_SDK isn't set"
    if not (SDK_DIR / "bin" / "Windows" / "x64" / "rel" / "nvngx_vsr.dll").exists():
        return f"RTX Video SDK not found at {SDK_DIR} (set NV_RTX_VIDEO_SDK)"
    if not WRAPPER.exists():
        return f"wrapper not built: run native\\rtxvsr\\build.bat ({WRAPPER} missing)"
    return None


def vfx_reason() -> str | None:
    """None if NVIDIA's package is set up and this card can run it, otherwise why not (what people see)."""
    from .vfg_setup import MIN_COMPUTE, installed
    if not torch.cuda.is_available():
        return "it needs an NVIDIA RTX card"
    if torch.cuda.get_device_capability(0) < MIN_COMPUTE:
        return ("it comes with NVIDIA VFG's download from NVIDIA, which SmoovPlay offers on RTX 40 and 50 series "
                "cards. AMD FSR 1 works on any card")
    if not installed():
        return ("it comes with NVIDIA VFG's download from NVIDIA: set up NVIDIA VFG on SmoovPlay's Settings page "
                "(Interpolation model). AMD FSR 1 works on any card")
    return None


def backend() -> tuple[str | None, str | None]:
    """("vfx" or "sdk", None) for the way RTX VSR can run here, or (None, why it can't). NVIDIA's package first;
    FRAMEGEN_VSR=sdk picks the SDK when both are there."""
    try:
        vfx = vfx_reason()
    except Exception as e:  # noqa: BLE001 - a broken install must never stop the window opening
        vfx = f"NVIDIA's package couldn't be checked: {str(e)[:120]}"
    ways = {"vfx": vfx, "sdk": sdk_reason()}
    for name in (("sdk", "vfx") if os.environ.get("FRAMEGEN_VSR") == "sdk" else ("vfx", "sdk")):
        if ways[name] is None:
            return name, None
    return None, vfx


def unavailable_reason() -> str | None:
    """None if RTX VSR can run here (either way), otherwise why not."""
    return backend()[1]


def make_vsr(src_w: int, src_h: int, quality: int = QUALITY["high"], clean: bool = False):
    """RTX VSR for src_w x src_h frames, through NVIDIA's package or the SDK (backend()). clean: the source has no
    compression (a capture card's raw frames), so NVIDIA's package uses its high-bitrate models. Raises RuntimeError
    if it can't run."""
    name, why = backend()
    if name is None:
        raise RuntimeError(why)
    return VfxVsr(src_w, src_h, quality, clean) if name == "vfx" else RtxVsr(src_w, src_h, quality)


class RtxVsr:
    """Upscales src_w x src_h RGBA frames into a caller-provided RGBA CUDA array with RTX VSR, through the RTX Video
    SDK.

    Create it, and call it, on one thread (NGX isn't thread-safe), after PyTorch has initialised CUDA there.
    The CUDA stream current at creation is the one VSR runs on.
    """
    out_format = "rgba8"  # what it writes into the output array
    in_format = "rgba8"
    private = False  # its timings may go in the log
    how = "through the RTX Video SDK"

    def __init__(self, src_w: int, src_h: int, quality: int = QUALITY["high"]) -> None:
        why = sdk_reason()
        if why:
            raise RuntimeError(why)
        self.w, self.h, self.quality = src_w, src_h, int(quality)
        torch.zeros(1, device="cuda")  # CUDA context current on this thread
        self.rt = rt = _cudart()
        p, i, u, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_uint, ctypes.c_size_t
        rt.cudaMallocArray.argtypes = [ctypes.POINTER(p), ctypes.POINTER(_ChannelDesc), s, s, u]
        rt.cudaFreeArray.argtypes = [p]
        rt.cudaCreateTextureObject.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), p, p, p]
        rt.cudaDestroyTextureObject.argtypes = [ctypes.c_ulonglong]
        rt.cudaCreateSurfaceObject.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), p]
        rt.cudaDestroySurfaceObject.argtypes = [ctypes.c_ulonglong]
        rt.cudaMemcpy2DToArrayAsync.argtypes = [p, s, s, p, s, s, s, i, p]
        rt.cudaMemcpy2DFromArrayAsync.argtypes = [p, s, p, s, s, s, s, i, p]

        self.dll = dll = ctypes.CDLL(str(WRAPPER))
        dll.rtxvsr_set_cudart.argtypes = [ctypes.c_wchar_p]
        dll.rtxvsr_create.argtypes = [p, p, ctypes.c_wchar_p, ctypes.c_wchar_p]
        dll.rtxvsr_evaluate.argtypes = [ctypes.c_ulonglong, ctypes.c_ulonglong] + [u] * 6 + [i]
        self._ngx(dll.rtxvsr_set_cudart(cudart_path()), "load CUDA runtime")
        drv = ctypes.CDLL("nvcuda.dll")
        ctx = p()
        self._cuda(drv.cuCtxGetCurrent(ctypes.byref(ctx)), "get CUDA context")
        self.stream = torch.cuda.current_stream()
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        rc = dll.rtxvsr_create(ctx, p(self.stream.cuda_stream),
                               str(SDK_DIR / "bin" / "Windows" / "x64" / "rel"), str(DATA_DIR))
        if rc == -1:
            raise RuntimeError("RTX VSR is not available on this GPU/driver (needs an RTX 20-series or newer)")
        if rc == -2:
            raise RuntimeError("RTX VSR needs a newer NVIDIA driver")
        self._ngx(rc, "create VSR")

        self.in_array = p()
        self._cuda(rt.cudaMallocArray(ctypes.byref(self.in_array), ctypes.byref(_RGBA8), src_w, src_h, 0),
                   "allocate input array")
        self.tex = ctypes.c_ulonglong()
        self._res_in, self._tex_desc = _array_resource_desc(self.in_array), _linear_texture_desc()
        self._cuda(rt.cudaCreateTextureObject(ctypes.byref(self.tex), self._res_in, self._tex_desc, None),
                   "create input texture")
        self._surfaces: dict[int, ctypes.c_ulonglong] = {}  # output array handle -> surface object
        self._events: list[tuple[torch.cuda.Event, torch.cuda.Event]] = []
        self.gpu_ms: list[float] = []

    def _cuda(self, err: int, what: str) -> None:
        if err:
            raise RuntimeError(f"RTX VSR: {what} failed (CUDA error {err})")

    def _ngx(self, rc: int, what: str) -> None:
        if rc:
            raise RuntimeError(f"RTX VSR: {what} failed (code {rc & 0xFFFFFFFF:#x})")

    def surface(self, array: ctypes.c_void_p) -> ctypes.c_ulonglong:
        """Surface object for an output array (cached; arrays must be 8-bit RGBA with surface load/store)."""
        key = array.value
        if key not in self._surfaces:
            surf = ctypes.c_ulonglong()
            self._cuda(self.rt.cudaCreateSurfaceObject(ctypes.byref(surf), _array_resource_desc(array)),
                       "create output surface")
            self._surfaces[key] = surf
        return self._surfaces[key]

    def forget(self, array: ctypes.c_void_p) -> None:
        """Drop the cached surface of an output array that is about to be freed or re-registered."""
        surf = self._surfaces.pop(array.value, None)
        if surf is not None:
            self.rt.cudaDestroySurfaceObject(surf)

    def __call__(self, rgba: torch.Tensor, out_array: ctypes.c_void_p, out_w: int, out_h: int,
                 quality: int | None = None) -> None:
        """Upscale rgba (h x w x 4 uint8, contiguous CUDA) into out_array's top-left out_w x out_h."""
        stream = ctypes.c_void_p(self.stream.cuda_stream)
        self._cuda(self.rt.cudaMemcpy2DToArrayAsync(self.in_array, 0, 0, ctypes.c_void_p(rgba.data_ptr()),
                                                    self.w * 4, self.w * 4, self.h,
                                                    _MEMCPY_DEVICE_TO_DEVICE, stream), "copy input")
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record(self.stream)
        self._ngx(self.dll.rtxvsr_evaluate(self.tex, self.surface(out_array), self.w, self.h, 0, 0, out_w, out_h,
                                           self.quality if quality is None else int(quality)), "evaluate")
        end.record(self.stream)
        self._events.append((start, end))
        self._collect()

    def _collect(self) -> None:
        """GPU time of finished evaluations (read without waiting)."""
        while self._events and self._events[0][1].query():
            s, e = self._events.pop(0)
            self.gpu_ms.append(s.elapsed_time(e))

    def new_output_array(self, w: int, h: int) -> ctypes.c_void_p:
        """A CUDA-owned RGBA8 array usable as output (for tests and offscreen use)."""
        arr = ctypes.c_void_p()
        self._cuda(self.rt.cudaMallocArray(ctypes.byref(arr), ctypes.byref(_RGBA8), w, h,
                                           _ARRAY_SURFACE_LOAD_STORE), "allocate output array")
        return arr

    def read_array(self, arr: ctypes.c_void_p, w: int, h: int) -> torch.Tensor:
        """Copy an RGBA8 array into a new h x w x 4 CUDA tensor (stream-ordered)."""
        out = torch.empty((h, w, 4), dtype=torch.uint8, device="cuda")
        self._cuda(self.rt.cudaMemcpy2DFromArrayAsync(ctypes.c_void_p(out.data_ptr()), w * 4, arr, 0, 0, w * 4, h,
                                                      _MEMCPY_DEVICE_TO_DEVICE,
                                                      ctypes.c_void_p(self.stream.cuda_stream)), "read output")
        return out

    def free_array(self, arr: ctypes.c_void_p) -> None:
        self.forget(arr)
        self.rt.cudaFreeArray(arr)

    def close(self) -> None:
        torch.cuda.synchronize()
        self._collect()
        for surf in self._surfaces.values():
            self.rt.cudaDestroySurfaceObject(surf)
        self._surfaces.clear()
        if self.tex.value:
            self.rt.cudaDestroyTextureObject(self.tex)
            self.tex = ctypes.c_ulonglong()
        if self.in_array.value:
            self.rt.cudaFreeArray(self.in_array)
            self.in_array = ctypes.c_void_p()
        self.dll.rtxvsr_shutdown()


# ---- through NVIDIA's package ---------------------------------------------------------------------------------------
_OPAQUE = -(1 << 30)  # the alpha bits (30-31) set, as an int32


def pack_rgba10(rgba: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """HxWx4 RGBA (uint8, or float16 0..1) -> `out`, HxW int32 of NVIDIA's packed RGB10A2: R in bits 0-9, G 10-19,
    B 20-29, alpha 30-31 (opaque). uint8 values are widened exactly (v10 = v8 * 1023 / 255); floats rounded."""
    if rgba.is_floating_point():
        v = rgba[..., :3].float().clamp_(0.0, 1.0).mul_(1023.0).round_().to(torch.int32)
    else:
        v = rgba[..., :3].to(torch.int32)
        v = (v << 2) | (v >> 6)
    return torch.bitwise_or(v[..., 0] | (v[..., 1] << 10), (v[..., 2] << 20) | _OPAQUE, out=out)


def pack_bgr10(bgr: torch.Tensor, out: torch.Tensor) -> torch.Tensor:
    """HxWx3 BGR (uint8, or float16 0..1) -> `out`, HxW int32 of NVIDIA's packed RGB10A2: R in bits 0-9, G 10-19,
    B 20-29, alpha 30-31 (opaque). uint8 values are widened exactly (v10 = v8 * 1023 / 255); floats rounded."""
    if bgr.is_floating_point():
        v = bgr.float().clamp_(0.0, 1.0).mul_(1023.0).round_().to(torch.int32)
    else:
        v = bgr.to(torch.int32)
        v = (v << 2) | (v >> 6)
    return torch.bitwise_or(v[..., 2] | (v[..., 1] << 10), (v[..., 0] << 20) | _OPAQUE, out=out)


class VfxVsr:
    """Upscales src_w x src_h RGBA frames with RTX VSR through NVIDIA's own package (nvidia-vfx's VideoSuperRes), into
    a caller-provided CUDA array of 32-bit words (an R32UI texture) as NVIDIA's packed 10-bit RGB.

    The same interface as RtxVsr. Create and call it on one thread, after PyTorch has initialised CUDA there; the CUDA
    stream current at creation is the one it runs on.
    """
    out_format = "rgb10a2"
    in_format = "rgb10a2"
    private = True  # NVIDIA's licence: no benchmark results published, so its timings stay out of the log
    how = "through NVIDIA's package (Experimental)"

    def __init__(self, src_w: int, src_h: int, quality: int = QUALITY["high"], clean: bool = False) -> None:
        why = vfx_reason()
        if why:
            raise RuntimeError(why)
        from .vfg import import_vfx
        nvvfx = import_vfx()
        self.w, self.h, self.clean = src_w, src_h, clean
        torch.zeros(1, device="cuda")  # CUDA context current on this thread
        self.stream = torch.cuda.current_stream()
        self.rt = rt = _cudart()
        p, i, s = ctypes.c_void_p, ctypes.c_int, ctypes.c_size_t
        rt.cudaMemcpy2DToArrayAsync.argtypes = [p, s, s, p, s, s, s, i, p]
        self._quality = int(quality)
        self.fx = nvvfx.VideoSuperRes(quality=self.level(self._quality, clean), device=torch.cuda.current_device(),
                                      image_encoding=nvvfx.ImageEncoding.RGB10A2)
        self.size = (src_w * 2, src_h * 2)  # the output size, set again for each window size (no reload needed)
        self.fx.output_width, self.fx.output_height = self.size
        self.fx.load()
        self.inp = torch.zeros((src_h, src_w), dtype=torch.int32, device="cuda")
        self._warmed_size: tuple[int, int] | None = None
        self.gpu_ms: list[float] = []  # (never filled: private)
        self.warm_up()

    def warm_up(self, out_w: int | None = None, out_h: int | None = None) -> None:
        """Run one dummy VSR pass at startup so TensorRT/VFX kernels are ready before tick 1."""
        target = (out_w, out_h) if (out_w is not None and out_h is not None) else self.size
        if self._warmed_size == target:
            return
        if target != self.size:
            self.size = target
            self.fx.output_width, self.fx.output_height = target
        with torch.cuda.stream(self.stream):
            self.inp.zero_()
            self.fx.run(self.inp.view(torch.uint32), non_blocking=True, stream_ptr=self.stream.cuda_stream)
        self.stream.synchronize()
        self._warmed_size = target

    @staticmethod
    def level(quality: int, clean: bool) -> int:
        """NVIDIA's QualityLevel for our 1-4 (Low to Ultra): the same numbers, or its high-bitrate models (16-19)
        for a clean source."""
        q = min(4, max(1, int(quality)))
        return q + CLEAN if clean else q

    @property
    def quality(self) -> int:
        return self._quality

    @quality.setter
    def quality(self, value: int) -> None:
        self._quality = int(value)
        self.fx.quality = self.level(self._quality, self.clean)

    def forget(self, array: ctypes.c_void_p) -> None:
        """(Nothing is cached per output array.)"""

    def __call__(self, frame: torch.Tensor, out_array: ctypes.c_void_p, out_w: int, out_h: int,
                 quality: int | None = None) -> None:
        """Upscale frame (HxWx4 RGBA, HxWx3 BGR, or HxW packed RGB10A2 int32/uint32, CUDA) into out_array's
        top-left out_w x out_h, as packed RGB10A2."""
        if quality is not None and int(quality) != self._quality:
            self.quality = quality
        if (out_w, out_h) != self.size:
            self.size = (out_w, out_h)
            self.fx.output_width, self.fx.output_height = out_w, out_h
        stream = self.stream.cuda_stream
        if frame.ndim == 2 and frame.dtype in (torch.int32, torch.uint32):
            with torch.cuda.stream(self.stream):
                inp = frame if frame.is_contiguous() else frame.contiguous()
            src = inp.view(torch.uint32)
        else:
            with torch.cuda.stream(self.stream):
                if frame.shape[-1] == 3:
                    pack_bgr10(frame, self.inp)
                else:
                    pack_rgba10(frame, self.inp)
            src = self.inp.view(torch.uint32)
        out = self.fx.run(src, non_blocking=True, stream_ptr=stream)
        img = torch.from_dlpack(out.image)  # (out_h, out_w) words in NVIDIA's memory until the next run: copied now
        if tuple(img.shape) != (out_h, out_w):
            raise RuntimeError(f"RTX VSR: {tuple(img.shape)} out, {(out_h, out_w)} expected")
        err = self.rt.cudaMemcpy2DToArrayAsync(out_array, 0, 0, ctypes.c_void_p(img.data_ptr()),
                                               img.stride(0) * 4, out_w * 4, out_h, _MEMCPY_DEVICE_TO_DEVICE,
                                               ctypes.c_void_p(stream))
        if err:
            raise RuntimeError(f"RTX VSR: copy output failed (CUDA error {err})")

    def close(self) -> None:
        torch.cuda.synchronize()
        self.fx.close()
