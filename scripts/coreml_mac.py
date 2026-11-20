"""RIFE 4.22 lite as a Core ML model: does it beat PyTorch (MPS) on Apple Silicon? Converts, checks, times.

The converted model takes the two real frames (RGB, 0..1, padded) and the timestep, and draws the AI frame:
both encoders and the flow network in one call. In the app each real frame would be encoded once, so with 2 or 3
AI frames per real frame this overstates the cost slightly (the encoder is ~5% of the work).

  python3 scripts/coreml_mac.py                 # convert (first run, ~1-3 min) and benchmark
  python3 scripts/coreml_mac.py --quick         # 1080p only
  python3 scripts/coreml_mac.py --variant lean3 # one model: base (upstream's forward pass) or lean3 (lighter)
  python3 scripts/coreml_mac.py --no-ane        # GPU only (the Neural Engine runs are slow)
  python3 scripts/coreml_mac.py --convert-only  # just build the models (works on Linux too)
"""

from __future__ import annotations

import argparse
import math
import platform
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
from framegen.interpolator import RifeInterpolator, ensure_weights  # noqa: E402
from framegen.rife import IFNet_HDv3_v4_22_lite as rife_mod  # noqa: E402

RES = {"720p": (1280, 720), "1080p": (1920, 1080)}
MODULO = 32  # 4.22.lite padding
OUT_DIR = ROOT / "models" / "coreml"
FRAME_MS = 1000.0 / 30.0
BUDGET = FRAME_MS * 0.80  # same 20% margin as bench_mac.py
TARGETS = {60: 1, 90: 2, 120: 3}


def _resize(t: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    """Bilinear resize to a fixed size (the same as scale_factor here: sizes divide exactly). Fixed sizes keep the
    traced graph free of shape arithmetic, which Core ML's converter can't take."""
    return t if tuple(t.shape[2:]) == size else F.interpolate(t, size=size, mode="bilinear", align_corners=False)


VARIANTS = ("base", "lean3")  # base: upstream's forward pass as is; lean3: lighter, see RifePair


class RifePair(nn.Module):
    """framegen/fastrife.py's forward as a plain module (no channels_last, timestep as an input).

    variant "base" is upstream's forward pass, only with fixed-size resizes. "lean3" is lighter: mask and features no
    longer go through full resolution (each block's output is resized straight to the next block's size), the
    intermediate warps happen at the next block's size, and only the last warp is at full size. It also samples the
    last warp slightly further down and right (see `bias`). An approximation, checked against the reference (PSNR)."""

    def __init__(self, scale: float, ph: int, pw: int, variant: str = "base") -> None:
        super().__init__()
        assert variant in VARIANTS, variant
        self.variant = variant
        state = torch.load(ensure_weights("4.22.lite"), map_location="cpu", weights_only=True)
        state = {k.replace("module.", ""): v for k, v in state.items() if k.startswith("module.")}
        net = rife_mod.IFNet(scale=scale)
        net.load_state_dict(state, strict=False)
        self.encode = rife_mod.Head()
        self.encode.load_state_dict({k[len("encode."):]: v for k, v in state.items() if k.startswith("encode.")})
        self.blocks = nn.ModuleList([getattr(net, f"block{i}") for i in range(4)])
        self.scales = list(net.scale_list)
        gx = torch.linspace(-1.0, 1.0, pw).view(1, 1, pw).expand(1, ph, pw)
        gy = torch.linspace(-1.0, 1.0, ph).view(1, ph, 1).expand(1, ph, pw)
        self.register_buffer("base_grid", torch.stack([gx, gy], dim=-1).contiguous())
        self.register_buffer("flow_to_grid", torch.tensor([2.0 / (pw - 1), 2.0 / (ph - 1)]))
        # RIFE 4.22 lite's output sits a little off the real frames even on a still picture or a pure translation
        # (measured on 1080p: about 0.17 px up and 0.05-0.1 px left, the same everywhere in the image and whatever the
        # motion). Alternating with the real frames that shows as a tremble. "lean3" samples that much further down and
        # right at the last step (flow_x/flow_y are in full-size pixels).
        self.register_buffer("bias", torch.tensor([0.05, 0.17, 0.05, 0.17]).view(1, 4, 1, 1))
        self.full = (ph, pw)
        self.sizes = [(int(ph / s), int(pw / s)) for s in self.scales]
        for k, (sh, sw) in enumerate(self.sizes):  # the same normalised grid at each block's size (lean3 warps there)
            gx = torch.linspace(-1.0, 1.0, sw).view(1, 1, sw).expand(1, sh, sw)
            gy = torch.linspace(-1.0, 1.0, sh).view(1, sh, 1).expand(1, sh, sw)
            self.register_buffer(f"grid{k}", torch.stack([gx, gy], dim=-1).contiguous())

    def _warp(self, src: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        grid = flow.permute(0, 2, 3, 1) * self.flow_to_grid + self.base_grid
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _warp_k(self, src: torch.Tensor, flow: torch.Tensor, k: int) -> torch.Tensor:
        """Warp at block k's size: the flow is in full-resolution pixels, and a displacement in normalised coordinates
        is the same at any size."""
        grid = flow.permute(0, 2, 3, 1) * self.flow_to_grid + getattr(self, f"grid{k}")
        return F.grid_sample(src, grid, mode="bilinear", padding_mode="border", align_corners=True)

    def _forward_lean3(self, src0: torch.Tensor, src1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """The flow network at each block's own size (see _lean_flow), then a single warp at full size."""
        flow, mask = self._lean_flow(src0, src1, t)
        flow = flow - self.bias
        w0 = self._warp(src0[:, :3], flow[:, :2])
        w1 = self._warp(src1[:, :3], flow[:, 2:4])
        m = torch.sigmoid(mask)
        return (w0 * m + w1 * (1 - m)).clamp(0.0, 1.0)

    def _lean_flow(self, src0: torch.Tensor, src1: torch.Tensor, t: torch.Tensor):
        """lean3's flow network: (flow, mask) at full size, flow[:, :2] to src0 and flow[:, 2:4] to src1. The original
        warps at full size and then shrinks the result; here each block warps at its own size and the flow only goes
        up to full size at the end."""
        s0 = [_resize(src0, sz) for sz in self.sizes]
        s1 = [_resize(src1, sz) for sz in self.sizes]
        flow = mask = feat = None
        for i, (blk, s, sz) in enumerate(zip(self.blocks, self.scales, self.sizes)):
            if flow is None:
                a, b = s0[0], s1[0]
                parts = [a[:, :3], b[:, :3], a[:, 3:], b[:, 3:], t.expand(1, 1, sz[0], sz[1])]
            else:
                fl = _resize(flow, sz)
                a, b = self._warp_k(s0[i], fl[:, :2], i), self._warp_k(s1[i], fl[:, 2:4], i)
                parts = [a[:, :3], b[:, :3], a[:, 3:], b[:, 3:], t.expand(1, 1, sz[0], sz[1]), mask, feat, fl / s]
            tmp = blk.lastconv(blk.convblock(blk.conv0(torch.cat(parts, 1))))
            fd = tmp[:, :4] * s
            flow = fd if flow is None else fl + fd
            mask, feat = tmp[:, 4:5], tmp[:, 5:]
            if i < 3:
                mask, feat = _resize(mask, self.sizes[i + 1]), _resize(feat, self.sizes[i + 1])
        return _resize(flow, self.full), _resize(mask, self.full)

    def forward(self, img0: torch.Tensor, img1: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        src0 = torch.cat([img0, self.encode(img0)], 1)
        src1 = torch.cat([img1, self.encode(img1)], 1)
        if self.variant == "lean3":
            return self._forward_lean3(src0, src1, t)
        flow = mask = feat = None
        w0, w1 = src0, src1
        for i, (blk, s, sz) in enumerate(zip(self.blocks, self.scales, self.sizes)):
            a, b = _resize(w0, sz), _resize(w1, sz)
            ts = t.expand(1, 1, sz[0], sz[1])
            parts = [a[:, :3], b[:, :3], a[:, 3:], b[:, 3:], ts]
            if flow is not None:
                parts += [_resize(mask, sz), _resize(feat, sz), _resize(flow, sz) / s]
            tmp = blk.lastconv(blk.convblock(blk.conv0(torch.cat(parts, 1))))
            tmp = _resize(tmp, self.full)
            fd = tmp[:, :4] * s
            flow = fd if flow is None else flow + fd
            mask, feat = tmp[:, 4:5], tmp[:, 5:]
            last = i == 3
            w0 = self._warp(src0[:, :3] if last else src0, flow[:, :2])
            w1 = self._warp(src1[:, :3] if last else src1, flow[:, 2:4])
        m = torch.sigmoid(mask)
        return (w0 * m + w1 * (1 - m)).clamp(0.0, 1.0)


def padded(w: int, h: int, scale: float) -> tuple[int, int]:
    step = max(MODULO, int(MODULO / scale))
    return math.ceil(w / step) * step, math.ceil(h / step) * step


def convert(res: str, scale: float, variant: str = "base") -> Path:
    """Build (once) and return the .mlpackage. An existing package with the same name is reused as it is: after
    changing RifePair, delete models/coreml/ (or the one package)."""
    import coremltools as ct
    w, h = RES[res]
    pw, ph = padded(w, h, scale)
    tag = "" if variant == "base" else f"_{variant}"
    path = OUT_DIR / f"rife422lite_{res}_{'full' if scale == 1.0 else 'half'}{tag}.mlpackage"
    if path.exists():
        return path
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[coreml] converting {path.name} ({pw}x{ph})...", flush=True)
    model = RifePair(scale, ph, pw, variant).eval()
    ex = (torch.rand(1, 3, ph, pw), torch.rand(1, 3, ph, pw), torch.full((1, 1, 1, 1), 0.5))
    with torch.no_grad():
        traced = torch.jit.trace(model, ex)
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType("img0", shape=ex[0].shape, dtype=np.float16),
                ct.TensorType("img1", shape=ex[1].shape, dtype=np.float16),
                ct.TensorType("t", shape=ex[2].shape, dtype=np.float16)],
        outputs=[ct.TensorType("out", dtype=np.float16)],
        convert_to="mlprogram",
        compute_precision=ct.precision.FLOAT16,
        minimum_deployment_target=ct.target.macOS15,
    )
    ml.save(str(path))
    return path


def to_input(frame_bgr: np.ndarray, pw: int, ph: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    x = np.zeros((1, 3, ph, pw), np.float16)
    x[0, :, :h, :w] = frame_bgr[:, :, ::-1].transpose(2, 0, 1).astype(np.float16) / np.float16(255)
    return x


def from_output(out: np.ndarray, w: int, h: int) -> np.ndarray:
    o = out[0, :, :h, :w].astype(np.float32).transpose(1, 2, 0)[:, :, ::-1]
    return np.clip(np.round(o * 255.0), 0, 255).astype(np.uint8)


def make_pair(w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    import cv2
    img = cv2.imread(str(ROOT / "framegen" / "samples" / "gta6-beach.jpg"))
    img = cv2.resize(img, (w + 16, h + 16), interpolation=cv2.INTER_AREA)
    return img[8:8 + h, 8:8 + w].copy(), img[4:4 + h, 0:w].copy()


def reference(a: np.ndarray, b: np.ndarray, scale: float) -> np.ndarray:
    h, w = a.shape[:2]
    it = RifeInterpolator(w, h, "4.22.lite", scale, fp16=False, cuda_graph=False, device="cpu", reference=True)
    it.load(torch.from_numpy(a)); it.prepare(0)
    it.load(torch.from_numpy(b)); it.prepare(1)
    return it.interpolate(1, 0.5).numpy().copy()


def chip() -> str:
    try:
        import subprocess
        return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or platform.machine()
    except Exception:  # noqa: BLE001
        return platform.machine()


def psnr(x: np.ndarray, y: np.ndarray) -> float:
    mse = float(np.mean((x.astype(np.float32) - y.astype(np.float32)) ** 2))
    return float("inf") if mse == 0 else 10 * math.log10(255.0 ** 2 / mse)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--convert-only", action="store_true")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--variant", choices=VARIANTS + ("all",), default="all", help="model variant(s) to test")
    ap.add_argument("--no-ane", action="store_true", help="skip the Neural Engine and All runs (GPU only, faster)")
    a = ap.parse_args()

    sizes = [("1080p", 1.0), ("1080p", 0.5)] if a.quick else \
            [("720p", 1.0), ("720p", 0.5), ("1080p", 1.0), ("1080p", 0.5)]
    variants = VARIANTS if a.variant == "all" else (a.variant,)
    cases = [(res, scale, v) for v in variants for res, scale in sizes]
    paths = {c: convert(*c) for c in cases}
    if a.convert_only:
        print("converted:", *[p.name for p in paths.values()])
        return

    import coremltools as ct
    units = {"GPU": ct.ComputeUnit.CPU_AND_GPU}
    if not a.no_ane:
        units |= {"Neural Engine": ct.ComputeUnit.CPU_AND_NE, "All": ct.ComputeUnit.ALL}
    print(f"\n{chip()} | {platform.mac_ver()[0] and 'macOS ' + platform.mac_ver()[0]} | "
          f"coremltools {ct.__version__} | torch {torch.__version__}\n")
    hdr = (f"{'model':<6} {'res':<6} {'motion':<6} {'runs on':<14} | {'load s':>6} {'AI frame':>8} | "
           f"{'60fps':>6} {'90fps':>6} {'120fps':>6} | {'PSNR':>6} | verdict")
    print(hdr)
    print("-" * len(hdr))
    best = None
    refs: dict[tuple[str, float], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for (res, scale, variant), path in paths.items():
        w, h = RES[res]
        pw, ph = padded(w, h, scale)
        if (res, scale) not in refs:
            fa, fb = make_pair(w, h)
            refs[res, scale] = fa, fb, reference(fa, fb, scale)
        fa, fb, ref = refs[res, scale]
        feed = {"img0": to_input(fa, pw, ph), "img1": to_input(fb, pw, ph), "t": np.full((1, 1, 1, 1), 0.5, np.float16)}
        for uname, unit in units.items():
            label = f"{variant:<6} {res:<6} {'full' if scale == 1.0 else 'half':<6} {uname:<14}"
            try:
                t0 = time.perf_counter()
                m = ct.models.MLModel(str(path), compute_units=unit)
                out = m.predict(feed)["out"]
                load = time.perf_counter() - t0
                for _ in range(5):
                    m.predict(feed)
                times = []
                for _ in range(a.iters):
                    t = time.perf_counter()
                    m.predict(feed)
                    times.append((time.perf_counter() - t) * 1000.0)
                ms = sorted(times)[len(times) // 2]
            except Exception as e:  # noqa: BLE001
                print(f"{label} | FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}", flush=True)
                continue
            q = psnr(from_output(out, w, h), ref)
            cost = {fps: n * ms for fps, n in TARGETS.items()}
            ok = [fps for fps, c in cost.items() if c <= BUDGET]
            verdict = f"up to {max(ok)} fps" if ok else "below 60 fps"
            warn = "  (!) picture differs" if q < 35 else ""
            print(f"{label} | {load:6.1f} {ms:8.1f} | {cost[60]:6.1f} {cost[90]:6.1f} {cost[120]:6.1f} | "
                  f"{q:6.1f} | {verdict}{warn}", flush=True)
            if q >= 35 and res == "1080p" and (best is None or ms < best[1]):
                best = (label, ms, verdict)

    print(f"\nTimes in ms (median), including the copy in and out. 60/90/120 = 1, 2 or 3 AI frames per real frame; "
          f"each needs <= {BUDGET:.1f} ms\n(a 30 fps stream, 20% kept free). PSNR in dB against upstream's PyTorch "
          "fp32 reference on the CPU. Compare with bench_mac.py (PyTorch MPS).")
    if best:
        print(f"\nFastest correct at 1080p: {' '.join(best[0].split())} -> {best[1]:.1f} ms, {best[2]}")


if __name__ == "__main__":
    main()
