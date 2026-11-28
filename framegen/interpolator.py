"""RIFE frame interpolator tuned for a live stream: 1 to 3 in-between frames per real frame (60, 90 or 120 fps).

Design notes
------------
* Every buffer is pre-allocated at a fixed resolution, so the per-frame work can
  be captured once as CUDA graphs and replayed (removes Python/launch overhead,
  which matters more than FLOPs for these small networks).
* RIFE v4.2x splits work into `encode(frame) -> features` and
  `flownet(img0, img1, f0, f1) -> middle frame`. In a stream each real frame is
  the "next" frame of one pair and the "previous" frame of the following pair,
  so we encode it exactly once into one of two ping-pong slots.
* Frames enter and leave as HxWx3 uint8 BGR (OpenCV / capture-card layout), or with hdr=True as HxWx3 float16 BGR
  PQ BT.2020 (framegen/hdr.py "pq16", settings "hdr" = HDR output). The model sees 0..1 either way.
"""

from __future__ import annotations

import importlib
import math
import subprocess
import urllib.error
import urllib.request
from pathlib import Path

import torch
import torch.nn.functional as F

from .fastrife import FastIFNet
# Steady HUD (shared with NVIDIA VFG); its constants are still importable from here, as before
from .steady import STEADY_HUD_EDGE, STEADY_HUD_EDGE_R, STEADY_HUD_THRESHOLD, SteadyHud  # noqa: F401
from .sync import CAPTURE_LOCK

# version -> (module name, padding modulo). Modulo values match vs-rife.
MODELS: dict[str, tuple[str, int]] = {
    "4.22.lite": ("IFNet_HDv3_v4_22_lite", 32),
    "4.25.lite": ("IFNet_HDv3_v4_25_lite", 128),
    "4.25": ("IFNet_HDv3_v4_25", 64),
    "4.26": ("IFNet_HDv3_v4_26", 64),
}
WEIGHTS_URL = "https://github.com/HolyWu/vs-rife/releases/download/model/flownet_v{version}.pkl"
DEFAULT_MODEL_DIR = Path(__file__).resolve().parent.parent / "models"


def ensure_weights(version: str, model_dir: Path = DEFAULT_MODEL_DIR) -> Path:
    model_dir.mkdir(parents=True, exist_ok=True)
    path = model_dir / f"flownet_v{version}.pkl"
    if not path.exists():
        url = WEIGHTS_URL.format(version=version)
        print(f"[rife] downloading {url}")
        tmp = path.with_suffix(".part")
        try:
            urllib.request.urlretrieve(url, tmp)
        except urllib.error.URLError:  # e.g. Python's CA bundle can't verify; Windows curl uses the OS store
            subprocess.run(["curl", "-sfL", "-o", str(tmp), url], check=True)
        tmp.replace(path)
    return path


class RifeInterpolator:
    def __init__(
        self,
        width: int,
        height: int,
        version: str = "4.22.lite",
        scale: float = 1.0,
        fp16: bool = True,
        cuda_graph: bool = True,
        device: str = "cuda",
        reference: bool = False,
        backend: str = "torch",
        build_engine: bool = True,
        hdr: bool = False,
    ) -> None:
        """`reference=True` runs the unmodified vs-rife forward (for validating FastIFNet).
        backend="trt" runs the flow network as a TensorRT engine (built and cached on first use; with
        build_engine=False a missing engine raises FileNotFoundError instead).
        hdr: frames are float16 PQ (framegen/hdr.py pq16) instead of uint8; `in_u8` is then their 8-bit stand-in
        for the pipeline's checks, worked out in load()."""
        if version not in MODELS:
            raise ValueError(f"unknown model {version!r}; choose from {list(MODELS)}")
        if scale not in (0.25, 0.5, 1.0, 2.0):
            raise ValueError("scale must be 0.25, 0.5, 1.0 or 2.0")

        self.w, self.h = width, height
        self.version, self.scale = version, scale
        self.device = torch.device(device)
        self.dtype = torch.half if fp16 else torch.float
        self.hdr = hdr
        self.frame_dtype = torch.float16 if hdr else torch.uint8  # what load() takes and interpolate() gives
        torch.backends.cudnn.benchmark = True

        module_name, modulo = MODELS[version]
        mod = importlib.import_module(f"framegen.rife.{module_name}")
        state = torch.load(ensure_weights(version), map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items() if k.startswith("module.")}

        self.flownet = mod.IFNet(scale=scale)
        # Checkpoints also carry training-only weights (teacher/caltime); ignore those, but every
        # parameter the inference model has must be present.
        missing = self.flownet.load_state_dict(state, strict=False).missing_keys
        if missing:
            raise RuntimeError(f"RIFE {version}: checkpoint is missing {missing[:5]}...")
        self.encode = mod.Head()
        self.encode.load_state_dict({k[len("encode."):]: v for k, v in state.items() if k.startswith("encode.")})
        for m in (self.flownet, self.encode):
            m.eval().to(self.device, self.dtype)

        # Pad to a multiple the pyramid can divide cleanly.
        step = max(modulo, int(modulo / scale))
        self.pw = math.ceil(width / step) * step
        self.ph = math.ceil(height / step) * step
        self.pad = (0, self.pw - width, 0, self.ph - height)

        dev, dt, ph, pw = self.device, self.dtype, self.ph, self.pw
        self.flow_div = torch.tensor([(pw - 1.0) / 2.0, (ph - 1.0) / 2.0], dtype=torch.float, device=dev)
        gx = torch.linspace(-1.0, 1.0, pw, device=dev).view(1, 1, 1, pw).expand(1, 1, ph, pw)
        gy = torch.linspace(-1.0, 1.0, ph, device=dev).view(1, 1, ph, 1).expand(1, 1, ph, pw)
        self.grid = torch.cat([gx, gy], 1).contiguous()
        self.timestep = torch.full((1, 1, ph, pw), 0.5, dtype=dt, device=dev)
        self._t = 0.5
        # Steady HUD (framegen/steady.py): its tensors are read by the captured graphs, so changing them (t_now,
        # still_thr) needs no recapture.
        self.steady = 0.0
        self.hud = SteadyHud(ph, pw, dev)
        self._slot: int | None = None  # the newest prepared slot

        # Static I/O buffers (graph inputs/outputs), in the frames' own format. in_u8: the input as 8 bits, for the
        # pipeline's checks (the input itself, unless hdr). out_u8: the output (float16 if hdr).
        self.inp = torch.zeros((height, width, 3), dtype=self.frame_dtype, device=dev)
        self.out_u8 = torch.zeros((height, width, 3), dtype=self.frame_dtype, device=dev)
        self.in_u8 = torch.zeros((height, width, 3), dtype=torch.uint8, device=dev) if hdr else self.inp
        with torch.no_grad():
            fshape = self.encode(torch.zeros((1, 3, ph, pw), dtype=dt, device=dev)).shape
        self.fast = None if reference else FastIFNet(self.flownet, ph, pw, dt, dev)
        if self.fast is not None:  # per slot: fp32 [img | features]
            self.src = [torch.zeros((1, 3 + fshape[1], ph, pw), device=dev) for _ in range(2)]
        else:
            self.img = [torch.zeros((1, 3, ph, pw), dtype=dt, device=dev) for _ in range(2)]
            self.feat = [torch.zeros(fshape, dtype=dt, device=dev) for _ in range(2)]

        self.trt = None
        if backend == "trt":
            if self.fast is None:
                raise ValueError("backend='trt' needs the fast path (reference=False)")
            from .trt import TrtIFNet
            self.trt = TrtIFNet(self.fast, tuple(self.src[0].shape), f"{version}_s{scale}", dev, build=build_engine)
        elif backend != "torch":
            raise ValueError(f"unknown backend {backend!r}")

        self._graphs: dict[tuple, torch.cuda.CUDAGraph] = {}
        self.cuda_graph = cuda_graph
        if cuda_graph:
            for slot in (0, 1):
                self._graphs[("prep", slot)] = self._capture(lambda s=slot: self._prepare_eager(s))
                self._graphs[("mask", slot)] = self._capture(lambda s=slot: self._steady_mask(1 - s, s))
                for steady in (False, True):  # two variants, so Steady HUD costs nothing when it's off
                    self._graphs[("interp", slot, steady)] = self._capture(
                        lambda s=slot, st=steady: self._interp_eager(1 - s, s, st))

    # ---- eager kernels (also what gets captured into graphs) -----------------
    @torch.no_grad()
    def _prepare_eager(self, slot: int) -> None:
        x = self.inp.permute(2, 0, 1).flip(0).unsqueeze(0).to(self.dtype)  # BGR->RGB
        if not self.hdr:
            x = x.div_(255.0)  # 0..1 (pq16 already is)
        if self.pad[1] or self.pad[3]:
            x = F.pad(x, self.pad)
        if self.fast is not None:
            self.src[slot].copy_(FastIFNet.make_src(x, self.encode(x)))
        else:
            self.img[slot].copy_(x)
            self.feat[slot].copy_(self.encode(x))

    @torch.no_grad()
    def _interp_eager(self, a: int, b: int, steady: bool = False) -> None:
        if self.trt is not None:
            mid = self.trt(self.src[a], self.src[b])
        elif self.fast is not None:
            mid = self.fast(self.src[a], self.src[b])
        else:
            mid = self.flownet(
                self.img[a], self.img[b], self.timestep, self.flow_div, self.grid, self.feat[a], self.feat[b]
            )
        if steady:
            mid = self._steady(mid, a, b)
        mid = mid[0, :, : self.h, : self.w].flip(0).permute(1, 2, 0)  # RGB->BGR, HWC
        if self.hdr:
            self.out_u8.copy_(mid.clamp(0.0, 1.0))
        else:
            self.out_u8.copy_(mid.clamp(0.0, 1.0).mul(255.0).round_().to(torch.uint8))

    def _rgb(self, slot: int) -> torch.Tensor:
        """Slot's real frame, (1, 3, ph, pw) RGB 0..1."""
        return (self.src[slot][:, :3] if self.fast is not None else self.img[slot]).float()

    def _steady(self, mid: torch.Tensor, a: int, b: int) -> torch.Tensor:
        """Steady HUD (framegen/steady.py): the real pixels where the HUD is, between real frames a and b."""
        return self.hud.blend(mid, self._rgb(a), self._rgb(b))

    def _steady_mask(self, a: int, b: int) -> None:
        """Where Steady HUD acts between real frames a -> b (once per real frame, after prepare)."""
        self.hud.mask(a, b, self._rgb(a), self._rgb(b))

    def set_steady(self, threshold: float) -> None:
        """Steady HUD on (threshold: largest real-frame difference, 0..1, that counts as still) or off (0)."""
        if threshold > 0:
            self.hud.still_thr.fill_(threshold)
            if self.steady <= 0 and self._slot is not None:  # turned on mid-pair: the current pair's mask now
                old = 1 - self._slot  # (the older frame's edges weren't worked out while it was off)
                self.hud.edges(old, self._rgb(old))
                self._steady_mask(old, self._slot)
        self.steady = threshold

    def steady_mask_now(self):
        """The current pair's Steady HUD mask (HxW numpy, 1 = the real pixels), or None while it's off. For debug
        snapshots: it copies the mask to the CPU, so it isn't for every frame."""
        if self.steady <= 0:
            return None
        return self.hud.w[0, 0, : self.h, : self.w].float().cpu().numpy()

    def _capture(self, fn) -> torch.cuda.CUDAGraph:
        with CAPTURE_LOCK:
            return self._capture_locked(fn)

    def _capture_locked(self, fn) -> torch.cuda.CUDAGraph:
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):  # warm-up (cudnn autotune, allocator) before capture
                fn()
        torch.cuda.current_stream().wait_stream(side)
        g = torch.cuda.CUDAGraph()
        # thread_local, and CAPTURE_LOCK: the Remote Play reader decodes on the GPU in another thread. With the
        # default ("global") its CUDA calls fail during capture, and FFmpeg's decoder crashes on that.
        with torch.cuda.graph(g, capture_error_mode="thread_local"):
            fn()
        return g

    # ---- public API ----------------------------------------------------------
    def prepare(self, slot: int) -> None:
        """Convert+encode `in_u8` into ping-pong slot `slot` (0/1); with Steady HUD on, also where it acts between
        the previous real frame and this one."""
        if self.cuda_graph:
            self._graphs[("prep", slot)].replay()
            if self.steady > 0:
                self._graphs[("mask", slot)].replay()
        else:
            self._prepare_eager(slot)
            if self.steady > 0:
                self._steady_mask(1 - slot, slot)
        self._slot = slot

    def interpolate(self, newest_slot: int, t: float = 0.5) -> torch.Tensor:
        """Frame at fraction t (0.5 = midpoint) from slot (1-newest_slot) to newest_slot -> `out_u8`.
        Each call overwrites `out_u8`, so for several timesteps copy each result out before the next."""
        if t != self._t:  # the graphs read these static planes, so no recapture is needed
            if self.trt is not None:
                self.trt.set_timestep(t)
            if self.fast is not None:
                self.fast.set_timestep(t)
            else:
                self.timestep.fill_(t)
            self.hud.t_now.fill_(t)
            self._t = t
        steady = self.steady > 0
        if self.cuda_graph:
            self._graphs[("interp", newest_slot, steady)].replay()
        else:
            self._interp_eager(1 - newest_slot, newest_slot, steady)
        return self.out_u8

    def load(self, frame_u8: torch.Tensor) -> None:
        """Copy an HxWx3 frame (GPU or pinned CPU; uint8, or float16 if hdr) into the static input."""
        self.inp.copy_(frame_u8, non_blocking=True)
        if self.hdr:
            from .hdr import to_u8
            to_u8(self.inp, self.in_u8)
