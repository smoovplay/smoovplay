"""Can a Mac run SmoovPlay's frame generation? RIFE 4.22 lite timed on Apple Silicon (PyTorch "mps" backend).

Measures what each real frame of a 30 fps stream costs (prepare + 1, 2 or 3 AI frames) and checks the picture
against a CPU reference, so a fast-but-wrong backend can't pass. No console, no stream: synthetic timing plus one
sample screenshot.

  python3 scripts/bench_mac.py              # the full table (about 2-4 minutes)
  python3 scripts/bench_mac.py --quick      # 1080p only, the app's default settings
  python3 scripts/bench_mac.py --device cpu # same thing on the CPU (to compare)
"""

from __future__ import annotations

import os

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")  # an op MPS lacks runs on the CPU (and we report it)

import argparse
import platform
import sys
import time
import warnings
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from framegen.interpolator import RifeInterpolator  # noqa: E402

RES = {"720p": (1280, 720), "1080p": (1920, 1080), "1440p": (2560, 1440)}
FRAME_MS = 1000.0 / 30.0  # a 30 fps game sends a real frame every 33.3 ms
MARGIN = 0.80  # leave 20% of each frame for decoding, upscaling and showing it
TARGETS = {60: 1, 90: 2, 120: 3}  # output fps -> AI frames per real frame


def sync(dev: str) -> None:
    if dev == "mps":
        torch.mps.synchronize()
    elif dev == "cuda":
        torch.cuda.synchronize()


def timed(fn, dev: str, iters: int) -> float:
    """Median ms of fn() over iters runs, after a warm-up."""
    for _ in range(5):
        fn()
    sync(dev)
    times = []
    for _ in range(iters):
        t = time.perf_counter()
        fn()
        sync(dev)
        times.append((time.perf_counter() - t) * 1000.0)
    times.sort()
    return times[len(times) // 2]


def make_pair(w: int, h: int) -> tuple[torch.Tensor, torch.Tensor]:
    """Two frames 'moving': a sample screenshot and the same one shifted 12 px right and 4 px down."""
    try:
        import cv2
        img = cv2.imread(str(ROOT / "framegen" / "samples" / "gta6-beach.jpg"))
        img = cv2.resize(img, (w + 16, h + 16), interpolation=cv2.INTER_AREA)
        a = torch.from_numpy(img[8:8 + h, 8:8 + w].copy())
        b = torch.from_numpy(img[4:4 + h, 0:w].copy())  # the picture pans 8 px and 4 px
        return a, b
    except Exception:  # noqa: BLE001 - no OpenCV: random frames still time correctly
        g = torch.Generator().manual_seed(0)
        a = torch.randint(0, 256, (h, w, 3), dtype=torch.uint8, generator=g)
        return a, torch.roll(a, shifts=(4, 12), dims=(0, 1))


def run_pair(it: RifeInterpolator, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    it.load(a.to(it.device))
    it.prepare(0)
    it.load(b.to(it.device))
    it.prepare(1)
    return it.interpolate(1, 0.5).cpu().clone()


def psnr(x: torch.Tensor, y: torch.Tensor) -> float:
    mse = (x.float() - y.float()).pow(2).mean().item()
    return float("inf") if mse == 0 else 10 * torch.log10(torch.tensor(255.0 ** 2 / mse)).item()


def bench(dev: str, res: str, scale: float, fp16: bool, fast: bool, iters: int, ref_cache: dict) -> dict:
    w, h = RES[res]
    a, b = make_pair(w, h)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        it = RifeInterpolator(w, h, "4.22.lite", scale, fp16=fp16, cuda_graph=False, device=dev,
                              reference=not fast)
        out = run_pair(it, a, b)
        fb = b.to(dev)
        prep = timed(lambda: (it.load(fb), it.prepare(1)), dev, iters)
        interp = timed(lambda: it.interpolate(1, 0.5), dev, iters)
        copy = timed(lambda: it.out_u8.cpu(), dev, iters)  # the frame back to the CPU (the app keeps it on the GPU)
    fallback = sorted({str(w.message).split("\n")[0][:90] for w in caught if "fall back" in str(w.message).lower()
                       or "fallback" in str(w.message).lower()})

    key = (res, scale)
    if key not in ref_cache:  # CPU fp32 reference forward: what the picture should be
        ref = RifeInterpolator(w, h, "4.22.lite", scale, fp16=False, cuda_graph=False, device="cpu", reference=True)
        ref_cache[key] = run_pair(ref, a, b)
    q = psnr(out, ref_cache[key])
    return {"prep": prep, "interp": interp, "copy": copy, "psnr": q, "fallback": fallback}


def verdict(r: dict) -> str:
    ok = [fps for fps, n in TARGETS.items() if r["prep"] + n * r["interp"] <= FRAME_MS * MARGIN]
    return f"up to {max(ok)} fps" if ok else "below 60 fps"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--device", default="mps", choices=["mps", "cpu", "cuda"])
    ap.add_argument("--quick", action="store_true", help="1080p, the app's default settings only")
    ap.add_argument("--iters", type=int, default=30)
    a = ap.parse_args()

    if a.device == "mps" and not torch.backends.mps.is_available():
        sys.exit("MPS isn't available: this needs a Mac with Apple Silicon (M1 or later), macOS 12.3+ and "
                 "a PyTorch built with MPS (the normal `pip install torch` is).")

    chip = platform.processor() or platform.machine()
    try:
        import subprocess
        chip = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True, text=True,
                              timeout=5).stdout.strip() or chip
    except Exception:  # noqa: BLE001
        pass
    print(f"{chip} | macOS {platform.mac_ver()[0] or '-'} | Python {platform.python_version()} | "
          f"torch {torch.__version__} | device {a.device}\n")

    if a.quick:
        cases = [("1080p", 1.0, True, True), ("1080p", 0.5, True, True)]
    else:
        cases = [(res, scale, fp16, fast)
                 for res in ("720p", "1080p", "1440p")
                 for scale in (1.0, 0.5)
                 for fp16 in (True, False)
                 for fast in (True, False)]

    hdr = (f"{'res':<6} {'motion':<6} {'prec':<5} {'path':<9} | {'prep':>6} {'AI frame':>8} {'copy':>6} | "
           f"{'60fps':>6} {'90fps':>6} {'120fps':>6} | {'PSNR':>6} | verdict")
    print(hdr)
    print("-" * len(hdr))
    ref_cache: dict = {}
    results = []
    for res, scale, fp16, fast in cases:
        label = f"{res:<6} {'full' if scale == 1.0 else 'half':<6} {'fp16' if fp16 else 'fp32':<5} " \
                f"{'fast' if fast else 'reference':<9}"
        try:
            r = bench(a.device, res, scale, fp16, fast, a.iters, ref_cache)
        except Exception as e:  # noqa: BLE001 - one failing combination shouldn't end the run
            print(f"{label} | FAILED: {type(e).__name__}: {str(e).splitlines()[0][:80]}")
            continue
        results.append((label, r))
        cost = {fps: r["prep"] + n * r["interp"] for fps, n in TARGETS.items()}
        warn = "  (!) picture differs" if r["psnr"] < 35 else ""
        print(f"{label} | {r['prep']:6.1f} {r['interp']:8.1f} {r['copy']:6.1f} | "
              f"{cost[60]:6.1f} {cost[90]:6.1f} {cost[120]:6.1f} | {r['psnr']:6.1f} | {verdict(r)}{warn}", flush=True)

    print(f"\nTimes in ms. A 30 fps stream sends a real frame every {FRAME_MS:.1f} ms; the 60/90/120 columns are "
          f"what each real frame costs\n(prepare + 1, 2 or 3 AI frames). The verdict keeps {round((1 - MARGIN) * 100)}% "
          f"of the frame free for decoding and display, so it needs <= {FRAME_MS * MARGIN:.1f} ms.\n"
          "PSNR compares the AI frame with a CPU fp32 reference: above ~40 dB is the same picture; below 35 is a "
          "backend problem.")
    fb = sorted({m for _, r in results for m in r["fallback"]})
    if fb:
        print("\nSome operations ran on the CPU instead of the GPU (slower than it could be):")
        for m in fb:
            print("  ", m)
    good = [(l, r) for l, r in results if r["psnr"] >= 35]
    if good:
        best = min(good, key=lambda x: x[1]["prep"] + x[1]["interp"])
        print(f"\nFastest correct setting: {best[0].split()[0]} {' '.join(best[0].split()[1:])} -> {verdict(best[1])}")


if __name__ == "__main__":
    main()
