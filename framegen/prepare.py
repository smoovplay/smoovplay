"""One-time preparation after install: RIFE weights, the TensorRT engine for a 1080p stream, and (on a new install)
settings picked for this PC by framegen.autodetect.

    python -m framegen.prepare            # what the SmoovPlay installer runs

Building an engine takes a minute or two, so doing it here means the first stream starts at once. Only full motion
detail gets one: that's where the "Auto" engine uses TensorRT (trt.auto_engine). Without an NVIDIA card (AMD or
Intel), RIFE is converted for DirectML instead, at full and half motion detail (framegen/dml.py, Experimental).
Progress goes to stdout as "[prepare] ..." lines (the launcher shows them). Exit code 0 = ready (PyTorch is used when
TensorRT isn't available), 2 = neither an NVIDIA card nor DirectML can run RIFE here.
"""
from __future__ import annotations

import argparse
import gc
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--model", default="4.22.lite")
    ap.add_argument("--scales", default="1.0", help="flow scales to build engines for (1.0 = full motion detail)")
    ap.add_argument("--no-detect", action="store_true", help="don't pick settings for this PC (framegen.autodetect)")
    a = ap.parse_args()

    print("[prepare] checking the graphics card", flush=True)
    import torch
    if sys.platform == "darwin" and not torch.cuda.is_available():
        return prepare_mac(a)
    if not torch.cuda.is_available():
        return prepare_dml(a)
    name = torch.cuda.get_device_name(0)
    major, minor = torch.cuda.get_device_capability(0)
    print(f"[prepare] {name} (compute {major}.{minor}), PyTorch {torch.__version__}", flush=True)

    from .interpolator import ensure_weights
    ensure_weights(a.model)
    print(f"[prepare] RIFE {a.model} weights ready", flush=True)

    from .trt import trt_available
    if trt_available():
        build_engines(a)
    else:
        print("[prepare] TensorRT isn't available, so the PyTorch engine will be used (about 12% slower)", flush=True)
    if not a.no_detect:
        pick_settings()
    print("[prepare] done", flush=True)
    return 0


def prepare_mac(a: argparse.Namespace) -> int:
    """On a Mac, RIFE runs through PyTorch on Apple's GPU (Metal, framegen/gpu.py), so there's nothing to build
    ahead."""
    import torch

    from . import gpu
    if gpu.device() != "mps":
        print("[prepare] ERROR: PyTorch can't use this Mac's GPU (Metal). SmoovPlay needs an Apple silicon Mac",
              flush=True)
        return 2
    print(f"[prepare] {gpu.name()}: RIFE runs through PyTorch {torch.__version__} on Apple's GPU (Metal)", flush=True)
    from .interpolator import ensure_weights
    ensure_weights(a.model)
    print(f"[prepare] RIFE {a.model} weights ready", flush=True)
    if not a.no_detect:
        pick_settings()
    print("[prepare] done", flush=True)
    return 0


def prepare_dml(a: argparse.Namespace) -> int:
    """No NVIDIA CUDA here: RIFE through DirectML (an AMD or Intel card), converted once on the CPU."""
    from . import dml
    why = dml.unavailable_reason()
    if why:
        print(f"[prepare] ERROR: no NVIDIA card PyTorch can use, and DirectML can't run here either ({why}). "
              "Update the graphics driver, then repair SmoovPlay on the Help page", flush=True)
        return 2
    print(f"[prepare] {dml.pick_adapter()['name']}: no NVIDIA card, so RIFE runs through DirectML (Experimental)",
          flush=True)
    from .interpolator import ensure_weights
    ensure_weights(a.model)
    print(f"[prepare] RIFE {a.model} weights ready", flush=True)
    scales = (1.0, 0.5)  # Auto uses half motion detail at 120 fps
    for i, scale in enumerate(scales, 1):
        label = {1.0: "full", 0.5: "half"}[scale]
        print(f"[prepare] preparing RIFE for DirectML, step {i} of {len(scales)} ({label} motion detail); "
              "takes about half a minute", flush=True)
        t0 = time.perf_counter()
        dml.convert(a.model, scale, a.width, a.height)
        print(f"[prepare] step {i} of {len(scales)} ready ({time.perf_counter() - t0:.0f} s)", flush=True)
    if not a.no_detect:
        pick_settings()
    print("[prepare] done", flush=True)
    return 0


def pick_settings() -> None:
    """A new install starts with settings picked for this PC; an existing settings file is left alone."""
    print("[prepare] picking settings for this PC", flush=True)
    from .autodetect import main as autodetect
    try:
        autodetect(["--apply", "--if-new"])
    except Exception as e:  # noqa: BLE001 - the defaults work everywhere; Auto-detect is on the Settings page
        print(f"[prepare] couldn't pick settings ({e}); the defaults are used", flush=True)


def build_engines(a: argparse.Namespace) -> None:
    import torch

    from .interpolator import RifeInterpolator
    scales = [float(s) for s in a.scales.split(",") if s.strip()]
    for i, scale in enumerate(scales, 1):
        label = {1.0: "full", 0.5: "half"}.get(scale, str(scale))
        print(f"[prepare] optimising for this GPU, step {i} of {len(scales)} ({label} motion detail); "
              "takes a minute or two", flush=True)
        t0 = time.perf_counter()
        interp = RifeInterpolator(a.width, a.height, a.model, scale, cuda_graph=False, backend="trt")
        del interp
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[prepare] step {i} of {len(scales)} ready ({time.perf_counter() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    sys.exit(main())
