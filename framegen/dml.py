"""DirectML engine: RIFE on any DirectX 12 graphics card (AMD, Intel or NVIDIA) through ONNX Runtime.

For PCs without an NVIDIA card; NVIDIA cards keep the CUDA/TensorRT engine (framegen/interpolator.py, trt.py).
`DmlInterpolator` is a drop-in for `RifeInterpolator`: the same load / prepare / interpolate calls on HxWx3 uint8
BGR frames, here CPU tensors.

The model runs as three ONNX graphs, converted from the RIFE weights once per resolution and flow scale (about 30 s
with PyTorch on the CPU; no NVIDIA card needed) and cached in models/dml/:

* encode: uint8 BGR frame -> [image | features] (fp32, padded), kept on the GPU in one of two ping-pong slots.
* mask:   two slots + Steady HUD's threshold -> where the real pixels replace the model's (framegen/steady.py's
          maths), kept on the GPU; run once per real frame while Steady HUD is on.
* flow:   two slots + timestep + that mask -> the in-between frame as uint8 BGR, written straight into `out_u8`.
          With Steady HUD off the mask input is all zeros, so the blend changes nothing.

The uint8 conversions are inside the graphs, so only 6 MB frames cross between the CPU and the GPU. The math is
FastIFNet's (framegen/fastrife.py): fp16 convolutions, fp32 warps. The one change is that the warp grid is built
from a row and a column at run time, since an exported full-size grid is a 17 MB constant copied into every warp.
"""
from __future__ import annotations

import ctypes
import os
import time
from ctypes import wintypes
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from .fastrife import FastIFNet
from .interpolator import DEFAULT_MODEL_DIR
from .steady import STEADY_HUD_EDGE, STEADY_HUD_EDGE_R

EXPORT_VERSION = 2  # bump when the exported graphs change, so cached files are rebuilt
MODEL_DIR = DEFAULT_MODEL_DIR / "dml"
VENDORS = {0x10DE: "NVIDIA", 0x1002: "AMD", 0x8086: "Intel"}


# ------------------------------------------------------------------------------------------------ graphics cards
class _LUID(ctypes.Structure):
    _fields_ = [("LowPart", wintypes.DWORD), ("HighPart", wintypes.LONG)]


class _AdapterDesc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
    _fields_ = [("Description", wintypes.WCHAR * 128), ("VendorId", wintypes.UINT), ("DeviceId", wintypes.UINT),
                ("SubSysId", wintypes.UINT), ("Revision", wintypes.UINT),
                ("DedicatedVideoMemory", ctypes.c_size_t), ("DedicatedSystemMemory", ctypes.c_size_t),
                ("SharedSystemMemory", ctypes.c_size_t), ("AdapterLuid", _LUID), ("Flags", wintypes.UINT)]


class _GUID(ctypes.Structure):
    _fields_ = [("Data1", wintypes.DWORD), ("Data2", wintypes.WORD), ("Data3", wintypes.WORD),
                ("Data4", ctypes.c_ubyte * 8)]


_IID_IDXGIFactory1 = _GUID(0x770AAE78, 0xF26F, 0x4DBA, (ctypes.c_ubyte * 8)(0xA8, 0x29, 0x25, 0x3C, 0x83, 0xD1, 0xB3, 0x87))


def _com(obj: ctypes.c_void_p, index: int, *argtypes):
    """Method `index` of a COM object's vtable, callable with the object as its first argument."""
    vtbl = ctypes.cast(obj, ctypes.POINTER(ctypes.POINTER(ctypes.c_void_p))).contents
    return ctypes.WINFUNCTYPE(ctypes.c_long, ctypes.c_void_p, *argtypes)(vtbl[index])


def adapters() -> list[dict]:
    """The graphics adapters in DXGI order, which is what DirectML's device_id counts (software adapters skipped)."""
    factory = ctypes.c_void_p()
    if ctypes.windll.dxgi.CreateDXGIFactory1(ctypes.byref(_IID_IDXGIFactory1), ctypes.byref(factory)) < 0:
        return []
    found = []
    try:
        for i in range(16):
            adapter = ctypes.c_void_p()
            if _com(factory, 12, wintypes.UINT, ctypes.POINTER(ctypes.c_void_p))(factory, i, ctypes.byref(adapter)) < 0:
                break  # DXGI_ERROR_NOT_FOUND: no more adapters
            desc = _AdapterDesc1()
            _com(adapter, 10, ctypes.POINTER(_AdapterDesc1))(adapter, ctypes.byref(desc))
            _com(adapter, 2)(adapter)  # Release
            if desc.Flags & 2:  # DXGI_ADAPTER_FLAG_SOFTWARE (Microsoft Basic Render Driver)
                continue
            found.append({"index": i, "name": desc.Description, "vendor": VENDORS.get(desc.VendorId, "other"),
                          "vram_gb": desc.DedicatedVideoMemory / 2**30})
    finally:
        _com(factory, 2)(factory)
    return found


def unavailable_reason() -> str | None:
    """Why the DirectML engine can't run here, or None if it can."""
    try:
        import onnxruntime as ort
    except Exception as e:  # noqa: BLE001
        return f"ONNX Runtime isn't installed ({str(e)[:80]})"
    if "DmlExecutionProvider" not in ort.get_available_providers():
        return "this ONNX Runtime has no DirectML (it needs the onnxruntime-directml package)"
    if not adapters():
        return "no DirectX 12 graphics adapter found"
    return None


def pick_adapter() -> dict:
    """The adapter with the most dedicated memory (the graphics card, not integrated graphics).
    SMOOVPLAY_DML_ADAPTER=<index> picks one instead (e.g. to try integrated graphics)."""
    found = adapters()
    if not found:
        raise RuntimeError("no DirectX 12 graphics adapter found")
    forced = os.environ.get("SMOOVPLAY_DML_ADAPTER")
    if forced is not None:
        for a in found:
            if a["index"] == int(forced):
                return a
        raise RuntimeError(f"SMOOVPLAY_DML_ADAPTER={forced}: no such adapter ({[a['index'] for a in found]})")
    return max(found, key=lambda a: a["vram_gb"])


# ------------------------------------------------------------------------------------------------ conversion
class _DmlIFNet(FastIFNet):
    """FastIFNet with the warp grid built from a row and a column, so the export stays small (same math)."""

    def __init__(self, net, ph: int, pw: int, dtype: torch.dtype, device: torch.device) -> None:
        super().__init__(net, ph, pw, dtype, device)
        del self.base_grid
        self.gx = torch.linspace(-1.0, 1.0, pw, device=device).view(1, 1, 1, pw)
        self.gy = torch.linspace(-1.0, 1.0, ph, device=device).view(1, 1, ph, 1)
        self.sx, self.sy = 2.0 / (pw - 1), 2.0 / (ph - 1)

    def _warp(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        f = flow.float()
        grid = torch.cat([f[:, 0:1] * self.sx + self.gx, f[:, 1:2] * self.sy + self.gy], 1).permute(0, 2, 3, 1)
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)


class _EncodeGraph(torch.nn.Module):
    def __init__(self, encode: torch.nn.Module, pad: tuple, dtype: torch.dtype) -> None:
        super().__init__()
        self.encode, self.pad, self.dtype = encode, pad, dtype
        self.register_buffer("rgb", torch.tensor([2, 1, 0]))

    def forward(self, frame: torch.Tensor) -> torch.Tensor:  # (H, W, 3) uint8 BGR
        x = frame.to(self.dtype).permute(2, 0, 1).index_select(0, self.rgb).unsqueeze(0) / 255.0
        if self.pad[1] or self.pad[3]:
            x = F.pad(x, self.pad)
        return FastIFNet.make_src(x, self.encode(x))


class _FlowGraph(torch.nn.Module):
    def __init__(self, fast: _DmlIFNet, h: int, w: int) -> None:
        super().__init__()
        self.fast, self.h, self.w = fast, h, w
        self.blocks = torch.nn.ModuleList(fast.blocks)  # registers the weights
        self.register_buffer("bgr", torch.tensor([2, 1, 0]))

    def forward(self, src0: torch.Tensor, src1: torch.Tensor, t: torch.Tensor, w: torch.Tensor) -> torch.Tensor:
        mid = self.fast(src0, src1, t).float()  # (1, 3, ph, pw) RGB
        # Steady HUD (framegen/steady.py's blend): the real pixels at t where the mask says; w is all zeros when off
        real = src0[:, :3] + (src1[:, :3] - src0[:, :3]) * t
        mid = mid + (real - mid) * w
        mid = mid[0, :, : self.h, : self.w].index_select(0, self.bgr).permute(1, 2, 0)
        return mid.clamp(0.0, 1.0).mul(255.0).round().to(torch.uint8)  # (H, W, 3) uint8 BGR


class _MaskGraph(torch.nn.Module):
    """Where Steady HUD acts between two real frames: framegen/steady.py's SteadyHud.mask() without its state, so it
    exports. Both frames' edges are worked out each time (the CUDA path keeps the older one's), in fp32."""

    def __init__(self) -> None:
        super().__init__()
        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]])
        self.register_buffer("sobel", torch.stack([gx, gx.t()]).unsqueeze(1))  # (2, 1, 3, 3): x and y gradients
        self.register_buffer("luma", torch.tensor([0.299, 0.587, 0.114]).view(1, 3, 1, 1))  # RGB

    def _edge(self, rgb: torch.Tensor) -> torch.Tensor:
        g = F.conv2d(F.conv2d(rgb, self.luma), self.sobel, padding=1).abs().sum(1, keepdim=True)
        return (g > STEADY_HUD_EDGE).float()

    def forward(self, src0: torch.Tensor, src1: torch.Tensor, thr: torch.Tensor) -> torch.Tensor:
        """src0/src1: the two slots; thr: (1, 1, 1, 1), the threshold, above 0. -> (1, 1, ph, pw), 1 = real pixels."""
        a, b = src0[:, :3], src1[:, :3]
        d = (b - a).abs().amax(1, keepdim=True)
        w = ((thr - d) * (2.0 / thr)).clamp(0.0, 1.0)
        r = STEADY_HUD_EDGE_R
        return w * F.max_pool2d(torch.maximum(self._edge(a), self._edge(b)), 2 * r + 1, 1, r)


def model_paths(version: str, scale: float, width: int, height: int) -> tuple[Path, Path, Path]:
    """The encode, flow and mask graphs' files. The encoder and the mask don't depend on the flow scale."""
    tag = f"rife{version}_{width}x{height}"
    return (MODEL_DIR / f"{tag}_encode_v{EXPORT_VERSION}.onnx",
            MODEL_DIR / f"{tag}_s{scale}_flow_v{EXPORT_VERSION}.onnx",
            MODEL_DIR / f"{tag}_mask_v{EXPORT_VERSION}.onnx")


def convert(version: str, scale: float, width: int, height: int) -> tuple[Path, Path, Path]:
    """Converts RIFE to the three ONNX graphs (skipping any that are already cached). Runs on the CPU."""
    from .interpolator import RifeInterpolator
    paths = model_paths(version, scale, width, height)
    if all(p.exists() for p in paths):
        return paths
    enc_path, flow_path, mask_path = paths
    print(f"[dml] preparing RIFE {version} for {width}x{height} (once, about 30 s)...")
    t0 = time.perf_counter()
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    cpu = torch.device("cpu")
    it = RifeInterpolator(width, height, version, scale, fp16=True, cuda_graph=False, device="cpu")
    frame = torch.zeros((height, width, 3), dtype=torch.uint8)
    with torch.no_grad():
        if not enc_path.exists():
            _export(_EncodeGraph(it.encode, it.pad, it.dtype), (frame,), enc_path, ["frame"], ["src"])
        src = torch.rand(tuple(it.src[0].shape))
        if not flow_path.exists():
            fast = _DmlIFNet(it.flownet, it.ph, it.pw, it.dtype, cpu)
            t = torch.full((1, 1, 1, 1), 0.5)
            w = torch.zeros((1, 1, it.ph, it.pw))
            _export(_FlowGraph(fast, height, width), (src, src.clone(), t, w), flow_path,
                    ["src0", "src1", "t", "w"], ["frame"])
        if not mask_path.exists():
            thr = torch.full((1, 1, 1, 1), 0.03)
            _export(_MaskGraph(), (src, src.clone(), thr), mask_path, ["src0", "src1", "thr"], ["w"])
    print(f"[dml] prepared in {time.perf_counter() - t0:.0f} s")
    return paths


def _export(module: torch.nn.Module, args: tuple, path: Path, inputs: list[str], outputs: list[str]) -> None:
    part = path.with_suffix(".part")
    torch.onnx.export(module, args, str(part), dynamo=False, opset_version=17, input_names=inputs,
                      output_names=outputs)
    part.replace(path)


# ------------------------------------------------------------------------------------------------ engine
class DmlInterpolator:
    """RIFE through ONNX Runtime + DirectML. Same calls as RifeInterpolator; frames are CPU tensors."""

    backend = "dml"

    def __init__(self, width: int, height: int, version: str = "4.22.lite", scale: float = 1.0,
                 adapter: dict | None = None) -> None:
        import onnxruntime as ort
        self.ort = ort
        self.w, self.h = width, height
        self.version, self.scale = version, scale
        self.device = torch.device("cpu")  # where in_u8 / out_u8 live
        self.adapter = adapter or pick_adapter()
        dev = self.adapter["index"]
        enc_path, flow_path, mask_path = convert(version, scale, width, height)
        self.encode = self._session(enc_path, dev)
        self.flow = self._session(flow_path, dev)
        self.mask = self._session(mask_path, dev)

        self.in_u8 = torch.zeros((height, width, 3), dtype=torch.uint8)
        self.out_u8 = torch.zeros((height, width, 3), dtype=torch.uint8)
        self._t = np.full((1, 1, 1, 1), 0.5, dtype=np.float32)
        self._thr = np.zeros((1, 1, 1, 1), dtype=np.float32)  # Steady HUD's threshold (set_steady)
        # CPU OrtValues that share memory with these arrays. A bound output is written in place on every run, but a
        # CPU input is copied to the GPU when it's *bound*, so the frame and the timestep are re-bound on each call.
        self._frame_in = ort.OrtValue.ortvalue_from_numpy(self.in_u8.numpy())
        self._t_in = ort.OrtValue.ortvalue_from_numpy(self._t)
        self._thr_in = ort.OrtValue.ortvalue_from_numpy(self._thr)
        frame_out = ort.OrtValue.ortvalue_from_numpy(self.out_u8.numpy())
        shape = self.encode.get_outputs()[0].shape
        self.slots = [ort.OrtValue.ortvalue_from_shape_and_type(shape, np.float32, "dml", dev) for _ in range(2)]
        # Steady HUD (set_steady): the mask graph writes `_w_mask` on the GPU once per real frame; with it off, the
        # flow graph reads `_w_zero` instead (copied up once, when bound), so the blend changes nothing.
        mask_shape = (1, 1, shape[2], shape[3])
        self._w_mask = ort.OrtValue.ortvalue_from_shape_and_type(mask_shape, np.float32, "dml", dev)
        self._w_zero = ort.OrtValue.ortvalue_from_numpy(np.zeros(mask_shape, dtype=np.float32))
        self.steady, self._slot = 0.0, None

        self._prep, self._interp, self._masks = [], [], []
        for slot in (0, 1):
            b = self.encode.io_binding()
            b.bind_ortvalue_output("src", self.slots[slot])
            self._prep.append(b)
            b = self.flow.io_binding()  # newest frame in `slot`, the one before it in the other
            b.bind_ortvalue_input("src0", self.slots[1 - slot])
            b.bind_ortvalue_input("src1", self.slots[slot])
            b.bind_ortvalue_input("w", self._w_zero)
            b.bind_ortvalue_output("frame", frame_out)
            self._interp.append(b)
            b = self.mask.io_binding()
            b.bind_ortvalue_input("src0", self.slots[1 - slot])
            b.bind_ortvalue_input("src1", self.slots[slot])
            b.bind_ortvalue_output("w", self._w_mask)
            self._masks.append(b)

    def _session(self, path: Path, device_id: int):
        ort = self.ort
        so = ort.SessionOptions()
        so.enable_mem_pattern = False  # DirectML requires these two
        so.execution_mode = ort.ExecutionMode.ORT_SEQUENTIAL
        so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        s = ort.InferenceSession(str(path), so, providers=[("DmlExecutionProvider", {"device_id": device_id})])
        if s.get_providers()[0] != "DmlExecutionProvider":
            raise RuntimeError(f"DirectML isn't available (ONNX Runtime providers: {s.get_providers()})")
        return s

    def describe(self) -> str:
        return f"DirectML on {self.adapter['name']}"

    def load(self, frame) -> None:
        """Copy an HxWx3 uint8 BGR frame (CPU tensor or numpy array) into the static input."""
        if torch.is_tensor(frame):
            self.in_u8.copy_(frame)
        else:
            np.copyto(self.in_u8.numpy(), frame)

    def prepare(self, slot: int) -> None:
        """Convert+encode `in_u8` into ping-pong slot `slot` (0/1), on the GPU."""
        b = self._prep[slot]
        b.bind_ortvalue_input("frame", self._frame_in)  # uploads in_u8 as it is now
        self.encode.run_with_iobinding(b)
        self._slot = slot
        if self.steady > 0:
            self._steady_mask(slot)

    def _steady_mask(self, slot: int) -> None:
        """Where Steady HUD acts between the frame before `slot` and `slot`, into `_w_mask` on the GPU."""
        b = self._masks[slot]
        b.bind_ortvalue_input("thr", self._thr_in)
        self.mask.run_with_iobinding(b)

    def steady_mask_now(self) -> np.ndarray | None:
        """The current pair's Steady HUD mask (HxW, 1 = the real pixels), or None while it's off. For debug
        snapshots only: it runs the mask graph again into the CPU, so it isn't for every frame."""
        if self.steady <= 0 or self._slot is None:
            return None
        shape = self._w_zero.shape()
        w = np.zeros(shape, dtype=np.float32)
        b = self.mask.io_binding()
        b.bind_ortvalue_input("src0", self.slots[1 - self._slot])
        b.bind_ortvalue_input("src1", self.slots[self._slot])
        b.bind_ortvalue_input("thr", self._thr_in)
        b.bind_ortvalue_output("w", self.ort.OrtValue.ortvalue_from_numpy(w))
        self.mask.run_with_iobinding(b)
        return w[0, 0, : self.h, : self.w]

    def set_steady(self, threshold: float) -> None:
        """Steady HUD on (threshold: largest real-frame difference, 0..1, that counts as still) or off (0)."""
        if threshold > 0:
            self._thr.fill(threshold)
            if self.steady <= 0:
                for b in self._interp:
                    b.bind_ortvalue_input("w", self._w_mask)
                if self._slot is not None:  # turned on mid-pair: the current pair's mask now
                    self._steady_mask(self._slot)
        elif self.steady > 0:
            for b in self._interp:
                b.bind_ortvalue_input("w", self._w_zero)
        self.steady = threshold

    def interpolate(self, newest_slot: int, t: float = 0.5) -> torch.Tensor:
        """Frame at fraction t from slot (1-newest_slot) to newest_slot -> `out_u8` (overwritten by each call)."""
        self._t.fill(t)
        b = self._interp[newest_slot]
        b.bind_ortvalue_input("t", self._t_in)
        self.flow.run_with_iobinding(b)
        return self.out_u8

    def speed_test(self, iters: int = 60, warmup_s: float = 1.0) -> tuple[float, float]:
        """Milliseconds for prepare (once per real frame) and interpolate (once per generated frame).
        Timed as whole cycles: prepare's output stays on the GPU, so it may finish after the call returns, while
        interpolate waits for its frame. The warm-up lets an idle card raise its clocks first."""
        end = time.perf_counter() + warmup_s
        while time.perf_counter() < end:
            self.prepare(0)
            self.interpolate(0)
        t0 = time.perf_counter()
        for i in range(iters):
            self.interpolate(i % 2, 1 / 3 if i % 2 else 2 / 3)
        interp = (time.perf_counter() - t0) * 1000 / iters
        t0 = time.perf_counter()
        for i in range(iters):
            self.prepare(i % 2)
            self.interpolate(i % 2)
        cycle = (time.perf_counter() - t0) * 1000 / iters
        return max(cycle - interp, 0.0), interp
