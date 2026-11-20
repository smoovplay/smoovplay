# SmoovPlay on macOS (Experimental)

Can a Mac with Apple Silicon run SmoovPlay's frame generation (RIFE 4.22 lite) fast enough? These two scripts
answer that on any M-series Mac. They time the model and check its picture against upstream's reference. They
don't stream from a console, and nothing here changes how SmoovPlay runs on Windows.

| Script | What it measures |
|---|---|
| [`scripts/bench_mac.py`](../scripts/bench_mac.py) | `framegen.interpolator` as it is, on PyTorch's `mps` backend (`cuda_graph=False`). |
| [`scripts/coreml_mac.py`](../scripts/coreml_mac.py) | The same network converted to Core ML. Two models: `base` (upstream's forward pass) and `lean3` (lighter, see below). Each runs on the GPU (`CPU_AND_GPU`), the Neural Engine (`CPU_AND_NE`) and all compute units (`ALL`). |

## Running the benchmarks on another Mac

You need an Apple Silicon Mac (M1 or later), Python 3.12 and about 1.5 GB of free disk space (about 0.9 GB for the Python packages, 150 MB for the models). On first use the scripts
download the RIFE weights (20 MB) into `models/`. `coreml_mac.py` then converts the Core ML models into
`models/coreml/`, which takes a few minutes.

```sh
git clone https://github.com/smoovplay/smoovplay && cd smoovplay
uv venv -p 3.12 .venv-mac        # or: python3.12 -m venv .venv-mac
uv pip install --python .venv-mac/bin/python torch==2.15.0 coremltools==9.0 numpy opencv-python
#   (pip: .venv-mac/bin/pip install torch==2.15.0 coremltools==9.0 numpy opencv-python)

.venv-mac/bin/python scripts/bench_mac.py --quick    # PyTorch MPS, 1080p (about 1 minute)
.venv-mac/bin/python scripts/coreml_mac.py --no-ane  # Core ML on the GPU, 720p and 1080p, base and lean3
.venv-mac/bin/python scripts/bench_mac.py            # the full MPS table, 720p to 1440p (a few minutes)
.venv-mac/bin/python scripts/coreml_mac.py           # also the Neural Engine runs (slow, they take a while)
```

The versions are pinned to the ones measured below, so results stay comparable (coremltools 9.0 warns that
it wasn't tested with torch 2.15; the conversion works). Close other heavy apps first. Plug a laptop into power, because on battery the GPU can run slower. Each script
prints the chip, macOS and library versions above its table, so the whole output can be pasted as it is.

**How to read the tables.** A 30 fps stream brings a real frame every 33.3 ms. The 60/90/120 columns show what
each real frame costs with 1, 2 or 3 AI frames. A verdict passes when that fits in 26.7 ms (20% of the frame is
left for decoding and display). PSNR compares the AI frame with upstream's PyTorch fp32 forward pass on the CPU.
Above ~40 dB is the same picture to the eye. Below 35 dB means a backend or conversion problem. Core ML times
include copying the frames in and out as tensors.

## Results: Apple M5 Pro

Mac mini, Apple M5 Pro, macOS 27.2.1, Python 3.12, torch 2.15.0, coremltools 9.0. Measured on 18/11/2026 against
upstream `main` at 0153ab6. Times are in ms, median of 30 runs, for the AI frame at "motion half" (`scale` 0.5)
unless the row says otherwise.

| Backend | 720p | 1080p | 1080p, motion full | PSNR (1080p) | Best verdict at 1080p |
|---|---|---|---|---|---|
| PyTorch MPS, fp16 fast path | 13.2 (+2.7 prepare) | 24.7 (+5.4 prepare) | 35.3 | 66 dB | below 60 fps (30.0 ms per real frame) |
| Core ML `base`, GPU | 11.0 | 22.7 | 29.8 | 52 dB | 60 fps |
| Core ML `lean3`, GPU | 7.5 | 14.9 | 24.7 | 39.6 dB (47.9 at motion full) | 60 fps |
| Core ML `base`, Neural Engine | 230 | 507 | 641 | 53 dB | no |

- **PyTorch MPS** gives the right picture, and no operation falls back to the CPU. At 1080p it misses 60 fps by a
  little: 30.0 ms per real frame against a 26.7 ms budget. 720p reaches 60 fps.
- **Core ML on the GPU** is faster than MPS for the same network: 1080p 60 fps fits, and 720p reaches 90 fps
  (`base`) or 120 fps (`lean3`). 1080p 90 fps doesn't fit (2 × 14.9 ms = 29.8 ms).
- **The Neural Engine is about 20× slower than the GPU** for this network, and "All" runs at the GPU's speed. Use
  `CPU_AND_GPU`.
- **`lean3`** is an approximation made for Core ML. Mask and features stay at each block's size instead of
  going through full resolution, the intermediate warps run at the block's size, and only the last warp is at
  full size. It is about a third faster than `base` at 1080p. Its picture is further from the reference (39.6-47.9 dB
  against 52 dB), so it needs a visual check on HUDs and text before it becomes a default. It also corrects
  RIFE 4.22 lite's constant sub-pixel offset (about 0.17 px up and 0.05-0.1 px left on 1080p). On a still
  picture that offset shows as a tremble between real and AI frames.

Core ML notes:

- The converter can't handle the shape arithmetic that `F.interpolate(scale_factor=...)` traces to. `RifePair` in
  `coreml_mac.py` resizes to fixed sizes instead. This is only in the Core ML path, and
  `framegen/` is unchanged.
- A converted package is reused by file name. After changing `RifePair`, delete `models/coreml/` before running
  again.

## Open questions

- **Base M chips (M1-M4, not Pro or Max).** Their numbers decide whether 1080p 60 fps can be the default on a Mac,
  or whether it becomes a Pro/Max option with 720p as the default. Results from those chips are welcome.
- **A long session** (10+ minutes): temperature, clocks and memory under a steady load.
- **End-to-end latency** of the whole chain, once frame pacing uses `framegen/cadence.py`.
