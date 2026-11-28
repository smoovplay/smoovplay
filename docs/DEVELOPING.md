# Developing SmoovPlay

Everything for working on SmoovPlay itself: every command and tool, settings and hotkeys, how Instant replay and
Looks work inside, the scheduler, upscaling, NVIDIA VFG, measured results and known limits. For what SmoovPlay is
and how to get it, see the [README](../../README.md).

## What's where

| Path | What's in it |
|---|---|
| [`framegen/`](../framegen/) | The pipeline: video sources (file, capture card, app window, chiaki-ng frame tap), repeat and cut detection, RIFE interpolation (PyTorch and TensorRT), FSR 1 and RTX VSR upscaling, the OpenGL window and the settings menu. |
| [`run.py`](../run.py) | Command-line entry point. |
| [`chiaki_tap/`](../chiaki_tap/) | The patch that adds a frame tap to chiaki-ng, and how to build it. |
| [`docs/frame-tap.md`](frame-tap.md) | The frame tap's format, so any app can send its frames to SmoovPlay (Experimental), with a small C writer to start from. |
| [`docs/macos.md`](macos.md), [`docs/linux.md`](linux.md) | macOS benchmarks, and PS5 Remote Play on Linux. Experimental and community-run: we don't test or support them ourselves. |
| [`installer/`](../installer/) | `SmoovPlay-Setup.exe` and the SmoovPlay launcher, in C# and WPF. |
| [`native/rtxvsr/`](../native/rtxvsr/) | Optional wrapper around NVIDIA's RTX Video SDK. |
| [`scripts/`](../scripts/) | Benchmarks, a model check and a fake chiaki-ng stream for testing without a PS5. |
| [`tests/`](../tests/) | `pytest` tests. |

### Setup

Needs 64-bit Windows 10 or 11, an NVIDIA RTX card with a recent driver (or an AMD or Intel card through DirectML,
Experimental), and Python 3.12.

```powershell
py -3.12 -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu128
.venv\Scripts\python -m pip install -r requirements.txt
```

RIFE weights download automatically into `models/` on first run. To build the installer and launcher, see
[installer/README.md](../installer/README.md); to build the patched chiaki-ng, see [chiaki_tap/README.md](../chiaki_tap/README.md).

### Usage

Any 60 fps gameplay recording works for testing; the examples use `testdata\clip_1080p60.mp4`.

```powershell
# Simulate PS5 30 fps output from a native 60 fps recording and watch it live:
.venv\Scripts\python run.py testdata\clip_1080p60.mp4 --realtime --display

# Score interpolated frames against the real frames that were dropped:
.venv\Scripts\python run.py testdata\clip_1080p60.mp4 --score --hitch 0.03 --save-worst out\worst.jpg

# Write the result to a file (NVENC):
.venv\Scripts\python run.py testdata\clip_1080p60.mp4 --out out\clip_interp_60fps.mp4

# Capture card (Experimental; a player's AVerMedia Live Gamer 4K works, other cards are untested):
.venv\Scripts\python run.py --list-devices
.venv\Scripts\python run.py --capture 0 --display

# Another app's window (Experimental, not yet confirmed with a real game stream): a preset, exe:NAME, hwnd:N or part of its title
.venv\Scripts\python -m framegen.window --list
.venv\Scripts\python run.py --source window --window ps-remote-play --display

# PS5 Remote Play through the patched chiaki-ng (the CSV: one row per frame, stream facts and decisions):
.venv\Scripts\python run.py --chiaki --display --chiaki-log out\session.csv

# Any app that writes the frame tap (docs/frame-tap.md, Experimental): compressed or decoded frames. The
# stand-in sends a video's decoded frames (--format bgra, --variable, --hints):
.venv\Scripts\python run.py --source tap --display
.venv\Scripts\python scripts\frametap_send.py testdata\clip_1080p60.mp4 --hints

# Every Remote Play session ends with a stream check: how well the size-based repeat check read the game (frame
# rate, the 30 fps new/repeat rhythm, likely misreads by the picture, sizes, arrival times, a verdict). It goes in
# the log and, with --results, in stream-check-<stamp>.txt beside the results. The same report from a saved CSV:
.venv\Scripts\python scripts\stream_check_replay.py out\session.csv

# Test mode: a 60 fps game played as if it were 30 fps, each AI frame scored against the frame left out, with a
# plain-language report rewritten every 10 s and at the end (the launcher's results card reads it):
.venv\Scripts\python run.py --chiaki --test-mode --display --results out\results.txt --chiaki-log out\session_as30.csv

# The same report for a file (always played as a 30 fps game, see the scheduler notes below):
.venv\Scripts\python run.py testdata\clip_1080p60.mp4 --score --results out\results.txt
```

`scripts/frametap_fake.py testdata\clip_1080p60.mp4 --native` stands in for a PS5 running a 60 fps game, so test
mode can be tried without a console.

**HDR** (Experimental: [`framegen/hdr.py`](../framegen/hdr.py), setting `hdr`, `--hdr off|sdr|hdr10`) only changes anything for a
chiaki-ng stream in HDR (codec H.265 HDR: 10-bit HEVC, PQ, BT.2020) with the setting on. `sdr` tone-maps it to 8-bit
SDR as it's decoded (BT.2390 roll-off, SDR white at 203 nits), so everything after that is unchanged. `hdr10` keeps
the frames HDR10 as half floats through RIFE and FSR 1 into a 10-bit DXGI swap chain in HDR10's colour space; it
needs a display with Windows' HDR on, and falls back to `sdr` without one. To try it without a PS5 or an HDR display:

```powershell
.venv\Scripts\python scripts\frametap_fake.py testdata\clip_1080p60.mp4 --hdr --tap Local\sp-hdr
$env:FRAMEGEN_HDR_FORCE = "1"; $env:FRAMEGEN_PRESENT_SNAP = "out\hdr_snaps"   # the 10-bit chain on an SDR display
.venv\Scripts\python run.py --chiaki --chiaki-tap Local\sp-hdr --display --test-mode --hdr hdr10
```

The snapshots are 16-bit PNGs of the PQ values in the back buffer (what an HDR display would get). Also:
`FRAMEGEN_HDR_PEAK` (the source's brightest highlights, default 1000 nits) and `FRAMEGEN_HDR_SDR_NITS` (SDR white,
default 203).

Other tools: `scripts/benchmark.py` (GPU timing per model and resolution), `scripts/bench_120.py` (30 → 120 speed
and quality), `scripts/check_fast.py` (checks the optimised forward pass against the reference), `scripts/latency_video.py`
(delay from a slow-motion phone video) and `pytest tests`.

<details>
<summary><b>Settings, the in-video menu and hotkeys</b></summary>

The settings are in two places, which stay in step:

- **The launcher's Settings page.** With nothing playing, changes are saved for the next session. While a stream
  plays, the page shows **LIVE** and changes reach the video straight away, as if made in the video window's menu.
- **The video window's menu.** Press **M** (or Tab). Its main page has the quick picks (frame generation, output
  frame rate, upscaler) and a row for each section (Smoothness, Picture, Looks, Display, Instant replay, PS5 Remote
  Play, as on the Settings page): ↑/↓ pick, ←/→ change a setting, Enter or → opens a section, Esc or Backspace goes
  back a page, and M (or Esc on the main page) closes it. It reopens where it was left. The video keeps playing
  behind it, so changes can be judged as they're made.

Both save to `%APPDATA%\framegen\settings.json`. Each setting explains itself on the page, and each choice in a
dropdown has a line on when to pick it.

**Auto-detect**, at the top of the Settings page, suggests the frame rate, interpolation model, motion detail, engine
and upscaler for this PC, with the reason for each; **Apply** sets them. It takes about 10 seconds:

- It times RIFE on a 1080p frame, at full and half motion detail, with the engine Auto would use.
- It reads the main display's refresh rate and resolution, and checks for TensorRT and RTX VSR.
- It picks the fastest frame rate the display can show where an in-between frame fits in 90% of a display tick.
  That margin is calibrated from live tests on an RTX 4070 Ti SUPER. 90 fps at full detail is preferred over 120 at
  half, as in testing.
- It notes anything worth fixing: a display Windows runs below its top rate, a graphics card busy with something
  else, or a frame rate that needs G-Sync or FreeSync for even motion.
- It keeps its times next to the settings file (`card-timing.json`, with the card's name). Auto motion detail goes
  by them: for a stream up to 1080p, it's half wherever full doesn't fit 90% of a tick on that card (an RTX 2080 Ti
  at 60 fps, for one). The times aren't kept if the card was busy, or if they came from `--timing`.

Setup runs it on a new install (`python -m framegen.prepare` → `python -m framegen.autodetect --apply --if-new`),
and doesn't save if the card is busy. The logic is `recommend()` in `framegen/autodetect.py`; the tests try it on
made-up PCs.

| Setting | Choices | Applies |
|---|---|---|
| Frame generation | On / Off | live |
| Output frame rate | 60 / 90 / 120 fps | live (falls back to 60 when the GPU can't keep up) |
| Even out 40-60 fps | On / Off (Experimental: even motion for games between 30 and 60 fps) | live |
| Dip smoothing | Off / On (Experimental: a 30 fps game's dips become a brief slow-down) | live |
| Upscaler | AMD FSR 1 / NVIDIA RTX VSR / Bilinear (others may come later) | live |
| FSR sharpening | Max … Minimal | live |
| RTX VSR quality | Low / Medium / High / Ultra (Experimental: RTX VSR comes with NVIDIA VFG's download) | live |
| Stats overlay, Fullscreen, VSync | On / Off | live |
| Stats overlay style | Minimal (frames on screen, the game's rate, the delay, a keeping-up dot) / Detailed | live |
| Stats overlay position, size | Top left / Top right / Bottom left / Bottom right; Small (out of the HUD's way) / Medium / Large | live |
| Window presentation | Standard / Low delay (Experimental: DXGI flip model, so a window skips the compositor's copy and G-Sync works in a window; fullscreen uses Standard; `--present gl\|dxgi`). With Windows' HDR on for the main display it's always DXGI, fullscreen as a borderless window: OpenGL's own fullscreen switches the display to SDR while it's in front (`FRAMEGEN_KEEP_HDR=0` turns this off) | next Play |
| Minimise chiaki-ng | On / Off | live (PS5 Remote Play only) |
| Sound in sync | On / Off (chiaki-ng holds its sound back to meet the picture) | live (PS5 Remote Play only) |
| Simulate a 30 fps game | Off / On | live, this session only, never saved (any 60 Hz source: Remote Play, a capture card, or a file with `--raw`; `--test-mode`) |
| Debug snapshots | Off / On (this session only, never saved: every 5 s the real frame and Steady HUD's mask go into `logs\debug-<session>\`, with a `[debug]` line in the log; [`framegen/debugsnap.py`](../framegen/debugsnap.py)) | live |
| Steady HUD | Off / On (off by default; results vary by game; RIFE on CUDA and DirectML, and NVIDIA VFG, [`framegen/steady.py`](../framegen/steady.py)) | live |
| Look, Which look | Off / On, and SmoovPlay's looks and yours (made on the launcher's Looks page) | live |
| Looks in clips | Off / On (the look in saved instant replays too; only SmoovPlay's half of a before & after video) | live |
| Interpolation model | RIFE 4.22 lite / NVIDIA VFG (RTX 40 and 50 series, optional download, see below) | **restart** |
| VFG quality | Low / Medium / High (NVIDIA VFG only) | **restart** |
| AI engine | Auto (TensorRT at full motion detail, PyTorch at half: the faster for each) / TensorRT / PyTorch (RIFE only) | **restart** |
| Motion detail | Auto / Full / Half (RIFE only) | **restart** |

Settings marked *restart* are saved at once. The menu and the Settings page then show **Restart now**, which
rebuilds the AI engine in a few seconds without dropping the stream or closing the window; the first TensorRT
build for a new resolution takes about 2 minutes. Adding a setting means adding one `Spec` to
`framegen/settings.py`: the menu, the launcher's Settings page, the saved file and the command line all read from
that list. The launcher talks to a running session through its stdin and stdout (`run.py --control`; see
`framegen/remote.py`).

Hotkeys: `i` interpolation on/off for this session (instant A/B, not saved), `s` stats overlay (off, minimal,
detailed), `f` fullscreen, `u` next upscaler, `l` delay test (PS5 Remote Play), `r` save the instant replay, `b` save a before &
after video of it, `k` look on/off, `j` next look (`Shift+J` back), `q` or Esc quit. A short note in the corner confirms each one. While instant replay is on,
`Ctrl+Alt+R` and `Ctrl+Alt+B` work from any window.

Instant replay (`framegen/replay.py`) keeps three tracks in memory as MPEG-TS pieces, one per encoded packet: every
frame shown (timed when it was shown, so 90 fps is 90), each new game frame (timed when it arrived plus SmoovPlay's
delay, so it lines up), and the game's sound, which chiaki-ng copies into a second shared-memory ring (see
[`chiaki_tap/`](../chiaki_tap/)) or a capture card's player hands over. Frames go to YUV on the graphics card (0.3 ms
at 1080p), to memory on a side stream, and to NVENC (else AMF, QSV or x264) on a thread of their own. Saving cuts
from the keyframe before the start (one a second) and remuxes to MP4 without re-encoding; a before & after video is
composed and encoded by `python -m framegen.replay compose` in a low-priority process of its own.

Fill the screen (`framegen/fill.py`): where the picture goes on a screen that isn't its shape. Consoles send 16:9 only,
so on an ultrawide each mode lays out the same 16:9 picture: Fit and Zoom are a rectangle (Zoom's goes past the
screen's edges), Stretch is the whole screen, and Panorama draws Fit into a half-float texture, then bends it across
the screen with u = atan(t·x) / atan(t), t picked so the middle is 5% (21:9) to 15% (32:9) wider than true. Messages and
the menu go on the part of the picture that's on screen. Clips and the instant replay record the stream, not the screen,
so they're unchanged.

Looks (`framegen/looks.py`, `framegen/looks_gl.py`): a look is JSON, `{"name", "nodes": [{"type", "on", "mix",
...}]}`, cleaned by `normalize()`. Yours are files in `%APPDATA%\framegen\looks`, with LUTs in its `luts` folder. The
video window draws a look in passes: the per-pixel effects share one shader, and each effect that reads neighbouring
pixels (bloom, halation, blur, clarity, aberration, sharpening) starts a new pass on a half-float image; bloom,
halation, blur and clarity use a pyramid of up to 6 halvings. `looks.render()` does the same maths in NumPy, and the tests compare the
shaders with it. With **Looks in clips** on, the look is drawn again at the stream's size into a texture that CUDA
reads, for the SmoovPlay track only. A share code is `SPL1-` and base64url of a compact binary: each value as its
difference from the default, deflated when that's shorter, with a 16-bit checksum; a newer version's extra
settings are skipped by older ones. The launcher's Looks page uses `python -m framegen.looks` (`schema`, `encode`,
`decode`, `import`, `save`, `delete`, `add-lut`) and shows edits live with the `look-show`, `look-hide` and
`looks-reload` commands (`framegen/remote.py`). Its preview is `python -m framegen.looks_preview`, a process that
draws looks on the sample pictures (`framegen/samples`) with the video window's own shaders in a hidden OpenGL context
(about a millisecond a picture; NumPy without OpenGL), one JSON request per line on stdin.
`python -m framegen.looks preview LOOK IN.png OUT.png` puts a look (an id or a code) on a picture.

Command-line flags (`--hz`, `--scaler`, `--fsr-sharpness`, `--vsr-quality`, `--engine`/`--trt`, `--off`,
`--vsync`, `--fullscreen`, `--minimize-chiaki`, `--steady-hud`, `--present gl|dxgi`, `--test-mode`, `--model vfg`
with `--vfg-quality`) override the saved settings for one run without saving them;
`--settings none` starts from defaults and saves nothing. `--model 4.25.lite|4.25|4.26` tries another RIFE
version for testing; those aren't offered in the menus until they're tested for real-time play and setup prepares
them.
</details>

<details>
<summary><b>NVIDIA VFG (optional, RTX 40 and 50 series)</b></summary>

SmoovPlay can draw the in-between frames with NVIDIA's own Video Frame Generation (VFG, part of NVIDIA's Video
Effects SDK) instead of RIFE. It isn't part of SmoovPlay or its downloads. **Set up NVIDIA VFG**, under
Interpolation model on the Settings page, downloads NVIDIA's `nvidia-vfx` package from NVIDIA's own package index
into `runtime\vfx` (about 540 MB, pinned by version and file hash), and only once you accept NVIDIA's licence:
the NVIDIA Software License Agreement, the Product-Specific Terms for NVIDIA AI Products and the NVIDIA Open Model
License. **Remove** deletes it again. NVIDIA supports it on RTX 40 and 50 series cards only; RIFE works on every RTX
card. If VFG can't start, SmoovPlay plays with RIFE instead and says why.

NVIDIA's licence doesn't allow publishing VFG benchmark results, so this page has none, and SmoovPlay keeps VFG's
timings and quality scores out of its logs and out of Copy diagnostics. They're shown on your own PC only: in the
stats overlay, in Auto-detect (which times VFG too once it's set up, and suggests it when it keeps up at least as
well as RIFE) and in test mode's report.

From the command line: `python -m framegen.vfg_setup --install --accept-nvidia-licence --target <folder>` (or
`--status`, `--remove`), then set `FRAMEGEN_VFX_PATH=<folder>` and run `run.py --model vfg --vfg-quality medium`.
</details>

<details>
<summary><b>Upscaling</b></summary>

When the window is bigger than the frame (fullscreen on a 1440p monitor, say), it upscales:

- **AMD FSR 1** (default): edge-adaptive upscaling plus mild sharpening, run as two shader passes while the
  frame is drawn, so real and generated frames get the same treatment. It costs about 0.1 ms of GPU time per
  frame at 2560×1440 (bilinear: about 0.04 ms).
- **NVIDIA RTX Video Super Resolution** (Experimental): also cleans up compression artefacts. It runs an AI model
  on every frame drawn (at 90 Hz, three per real frame), which can cost some of the extra in-between frames. It
  only works when frames are decoded on the GPU (the default for Remote Play), and not with HDR output yet. It runs
  one of two ways (`framegen/rtxvsr.py`):
  - **NVIDIA's package** (what the installed app uses): the VideoSuperRes effect in `nvidia-vfx`, the package
    NVIDIA VFG comes in, so it's there once NVIDIA VFG is set up (RTX 40 and 50 series). SmoovPlay ships nothing of
    NVIDIA's. Frames go in and out as NVIDIA's packed 10-bit RGB; the window copies the output words into an R32UI
    texture and draws it through an RGB10_A2 view of the same memory, so nothing unpacks them. A capture card's
    uncompressed frames get NVIDIA's models for clean video. Its licence forbids publishing benchmark results, so
    its timings stay out of the log.
  - **The [RTX Video SDK](https://developer.nvidia.com/rtx-video-sdk)** (free, NVIDIA developer login), for
    developers: unpack it wherever `NV_RTX_VIDEO_SDK` points and build `native\rtxvsr\build.bat` once. The wrapper
    links a library from the SDK, so it can't ship in the MIT app. It costs 1.3–4.9 ms per frame at 1440p (quality
    Low…Ultra, RTX 4070 Ti SUPER), and in side-by-side use it looked about the same as FSR 1. `FRAMEGEN_VSR=sdk`
    picks it when NVIDIA's package is there too.

The overlay names the scaler in use, and the GPU time per frame is printed on exit.

Where no upscaler runs (the window is the frame's size or smaller, or Bilinear), and for the menu and messages,
the frame is copied with `glBlitFramebuffer`. A driver that refuses that (an AMD one did, on the first frame) gets
the same copy drawn by a shader from then on, said once in the log (`framegen/glblit.py`). `FRAMEGEN_BLIT=shader`
uses the shader from the start, to try that path on any card.
</details>

<details>
<summary><b>The scheduler: repeats, hitches and scene cuts</b></summary>

A 30 fps game in a 60 Hz stream arrives as `A A B B C C`. On the tick a new frame arrives we show
`mid(prev, new)`; on its repeat tick we show the real frame:

```
input : A   A   B         B   C         C
output: A   A   mid(A,B)  B   mid(B,C)  C
```

At 60 Hz the added delay is one tick (about 16.7 ms) plus processing, not a full 30 fps frame. At 90 and 120 Hz
two or three frames are drawn between each pair. The scheduler also handles hitches (`A A A B`), stretches of
real 60 fps content (passed through one tick late) and scene cuts (held instead of ghosting two shots together;
a cut must be a sudden jump in frame difference, not just a large one, so fast camera motion isn't mistaken for a
cut). See `framegen/cadence.py`.

A file of about 30 fps is fed frame for frame as a 60 Hz stream with repeats; a file of about 60 fps has its odd
frames dropped and kept as ground truth for `--score`. Test mode does the same to a live 60 fps Remote Play stream
(`ChiakiSource(test=TestMode(True))`, switchable mid-stream from the menu's Simulate a 30 fps game), and scoring turns on with it.

Duplicate detection compares 16×16 block means: noise averages out, but a small moving object still trips the
maximum. Over PS5 Remote Play, an app's window and the frame tap, a decoded repeat is never an exact copy, so the
detector learns the line between repeats and new frames from the last 16 frames' mean differences (two bands, a
repeat's 10 or more times smaller); in real 30 fps Remote Play sessions it called new frames 99.5-100% and repeats
93-99% right over Ethernet (83% over Wi-Fi). Each frame's compressed size is kept in the frame log and the stream
check.

`framegen/fastrife.py` is a restructured RIFE forward pass (same maths, verified to about 60 dB against the
reference) that cut 1080p interpolation from about 16 ms to about 9 ms: one warp per direction instead of two,
inputs downsampled before concatenation, fp32 copies made once per frame, NHWC convolutions. All per-frame GPU
work is captured as CUDA graphs, and TensorRT engines are built once per resolution.
</details>

<details>
<summary><b>Results (RTX 4070 Ti SUPER, 1080p, RIFE 4.22.lite)</b></summary>

Live over Remote Play at 90 fps, in test mode (a 60 fps game played as 30): 35.6 ms (p50) from a compressed frame
reaching the frame tap to the buffer swap that puts it on screen, and 99% of in-between frames on time. 22.2 ms of
that is the hold any 3× interpolator needs. Not included: the PS5's encoding, the network, chiaki-ng's own receiving
and the display.

A native 30 fps PS5 game, live over Remote Play at 90 fps (17 Nov 2026, Ethernet, two sessions of 32 and 67 s):
35.8 and 35.4 ms (p50) and 45.1 and 41.5 ms (p95) on the same measure, with 94% and 99% of the extra in-between
frames shown (the rest fell back to 60 fps for a moment). The tick that draws a real frame's two in-between frames
took 17.6 to 17.8 ms (p50), upload and the first one's display included. Over Wi-Fi, frames that arrive late and
in clumps lower the share of extra frames shown (78% in one session).

Quality: native 60 fps game recordings → keep the even frames (the "30 fps game") → interpolate → compare
with the real odd frames. 3% random hitches injected. PyTorch fp16.

| Clip | RIFE | 50/50 blend | repeat (what the PS5 shows) |
|---|---|---|---|
| Skate (third person, city) | **36.2 dB** | 31.0 dB | 28.9 dB |
| Deadlock (fast combat, effects) | **25.8 dB** | 23.2 dB | 21.1 dB |

Duplicate detection: 100% correct on both clips (2,444 ticks, with hitches). The worst frames are moments when
the camera passes through geometry and the in-between frame shows content in neither neighbour. The 4.25 and
4.26 models score within ±0.2 dB but cost about 0.8 ms more; half-resolution flow (`--scale 0.5`) saves about
3 ms but costs 1.5–2 dB.

Timing from a recorded clip, not a live stream, paced at 60 Hz with the OpenGL window (20 s, 1,202 ticks). At
60 Hz a real frame is held for one tick (16.7 ms) rather than two, so this is lower than the live 90 fps figure:

| | p50 | p99 | max |
|---|---|---|---|
| interpolation tick (upload → RIFE → download) | 12.2 ms | 12.7 ms | 12.8 ms |
| repeat tick | 1.2 ms | 1.6 ms | 9.7 ms |
| present (texture upload + blit) | 2.6 ms | 3.1 ms | 18.5 ms |
| **real frame arrival → presented** | **20.8 ms** | 21.8 ms | 22.8 ms |
| same, interpolation off (passthrough) | about 7 ms | | |

2 of 1,202 ticks exceeded the 16.7 ms budget; no backlog built up.
</details>

<details>
<summary><b>Known limits</b></summary>

- **Capture cards are Experimental.** A player's AVerMedia Live Gamer 4K (GC573) works since 0.3.1, its sound from
  0.3.2 (not yet confirmed); other cards are tested with recordings standing in for one (`--capture`). On a PS5, turn off HDCP while you use the card (Settings → System → HDMI) and keep it at 60 Hz, not 120 Hz.
- **Window capture is Experimental.** Not yet confirmed with a real game stream; 0.3.1 recognises the current Xbox
  app. The app's sound comes ahead of the smooth
  picture and its window has to stay visible. In Windows HDR mode, on a rotated monitor or without an NVIDIA card,
  frames go through the CPU instead (a few ms more).
- **HDR is Experimental.** A contributor has played a real PS5 HDR stream with HDR output on an HDR monitor
  (Windows 11, RTX 5070 Ti). The tests use a stand-in HDR stream (`frametap_fake.py --hdr`) and read back the 10-bit
  buffer HDR output presents (on an SDR display, with `FRAMEGEN_HDR_FORCE=1`). Capture cards and window capture
  stay SDR.
- **4K** input doesn't fit in real time, even with TensorRT.
- **90/120 Hz to a file** (`--out`) isn't supported; those rates are window-only.
- **Soft scene cuts** between two dark shots can occasionally slip past the cut detector and show one blended frame.
- **Packaging**: the installer is unsigned ([SmartScreen warns](../README.md#smartscreen)) and
  about 140 MB, because chiaki-ng is embedded. The launcher updates itself from GitHub releases.
- In a window, Standard presentation goes through the Windows compositor, which adds about 12 ms. Fullscreen (`f`)
  avoids it, and so does Window presentation: Low delay (Experimental), in a window too. With Windows' HDR on, the
  window always presents through DXGI and fullscreen is a borderless window, so the display stays in HDR (OpenGL's
  own fullscreen switched it to SDR while the window was in front, and the colours changed with every Alt+Tab).
</details>


## Features in detail

- **Smoother, not faster.** The extra frames are generated: motion gets smoother, but the game still responds like
  30 fps, and a little later than without SmoovPlay. Great for driving, exploring and story missions; not for
  competitive online play.
- **About 36 ms at 90 fps** (about 20 ms at 60 fps). That's how long a frame spends in SmoovPlay, from arriving to
  being on screen, on top of Remote Play's own delay.
  About 22 ms of it is a wait every frame generator has: each real frame is held back while the two in-between
  frames before it are shown. When the graphics card falls behind for a moment, SmoovPlay drops to 60 fps rather
  than add delay.
- **1080p.** Remote Play arrives at 1080p, so the picture is a little softer than HDMI straight from the console.
- **HDR** (*Experimental*, Settings → HDR, off by default). For a PS5 streaming in HDR: pick the H.265 HDR codec in
  chiaki-ng, then turn HDR on. **On, shown in SDR** gives the stream its real colours on any monitor. **On, HDR
  output** keeps it HDR all the way to a monitor with Windows' HDR on, with Looks and RTX VSR off. A contributor has
  played a real PS5 HDR stream this way on an HDR monitor, and since 0.3.2 fullscreen keeps Windows' HDR on instead
  of switching the display to SDR. Instant replay clips are saved in SDR either way.
- **Some wobble.** HUD text over fast motion, very fast camera turns and thin objects crossing each other can wobble
  a little. Press **I** in the video window to compare with frame generation off.
- **Sound and picture stay in step.** chiaki-ng plays the sound, and SmoovPlay tells it to hold the sound back
  by as long as the picture spends in SmoovPlay, measured as you play (**Sound in sync**, on by default).
- **Delay test:** press **L** in the video window, standing still where Options opens a menu. chiaki-ng presses
  Options six times and SmoovPlay times each answer, then shows the PS5 and Remote Play's part, its own part and
  the two together.
- **Instant replay** (Settings, **Instant replay**: off until you pick a length, 30 seconds to 5 minutes) keeps
  the last stretch of what you played. Press **R** in the video window (or **Ctrl+Alt+R** from anywhere) to save
  it to your Videos folder, in SmoovPlay, as an MP4 at the frame rate you see, with the game's own sound (PS5
  Remote Play and capture cards; window capture has none). Saving takes a second or two. **B** (**Ctrl+Alt+B**)
  saves a **before & after** video instead: the game's own frames next to SmoovPlay's, split down the middle or
  side by side, made in the background in about as long as the clip. The video encoder on your graphics card does
  the recording, apart from the AI; a 1-minute replay at 1080p and 90 fps keeps about 310 MB in memory. Turning it
  on during play pauses the picture for about 0.2 s once, while the encoders start.
- **Looks** (the launcher's **Looks** page; off until you use one) put a grade on the picture. A look is a stack of
  effects applied top to bottom, like the nodes of a color grading app, each with its own amount: color, color
  wheels, split tone, black & white, LUTs (`.cube` files), bloom, halation, blur, clarity, chromatic aberration,
  sharpening, film grain, vignette, scanlines, letterbox and color-blind assist. SmoovPlay comes with 14 (Cinematic,
  Film, Vivid, Noir, Golden hour, Neon night, Retro VHS, Bleach bypass, Dreamy, Arctic, Sepia and three for color
  blindness). Try any change on one of them, and **Save as mine** keeps it as a look of your own (**Reset** puts
  SmoovPlay's back). The page draws the look on sample screenshots from Grand Theft
  Auto VI and five other games as you change it (hold the picture to see it without the look, and the arrows go to
  the next game). While you play, it shows your changes on the video as you make them, and in the video window **K** turns the look on and off and **J** goes to the next one (**Shift+J**
  back). **Copy code** gives a short code (`SPL1-`, about 35 to 60 characters) that rebuilds the look on
  anyone's SmoovPlay, through **Paste a code**. A look takes about 0.02 to 0.35 ms of graphics card time a frame at
  1440p. Saved clips stay clean unless **Looks in clips** is on (Settings, Instant replay); then only SmoovPlay's
  half of a before & after video gets the look.
- **Measure the whole delay yourself** with a phone's 240 fps slow motion and `scripts/latency_video.py`: film
  the controller and screen (`press`: button to screen), or the PS5's own HDMI picture beside SmoovPlay's
  (`side`: what Remote Play and SmoovPlay add). The script's help says how to film it.
- **60 fps games pass through unchanged.** There are no gaps to fill. To see what SmoovPlay would do with one
  anyway, use [test mode](../README.md#only-have-60-fps-games-use-test-mode).
- **Games between 40 and 60 fps** (unlocked performance modes) send their frames unevenly over a 60 Hz stream, so
  they stutter. **Even out 40-60 fps** (on by default, Experimental) times each frame and draws evenly spaced ones
  from them, for about one frame (17 ms) more delay. Tested with real footage played as 43 and 47 fps games, not yet
  with a real one.
- **30 fps games that dip below 30** hold a frame for 3 ticks (50 ms) or 4 (67 ms, waiting for the next 30 fps
  slot, as Gotham Knights' hitches did on PS5 Remote Play), then jump. **Dip smoothing** (off by
  default, Experimental; framegen/cadence.py DipWatch and FrameTimeline.at_arrival) times each frame by the tick it
  arrived on while the game dips (two dips within a second; off 3 s after the last) and draws evenly spaced frames
  from them, with up to 3 ticks of delay. On simulated games holding 10-20% of their frames, motion unevenness fell
  from 0.28-0.58 to 0.02-0.11 (0 = even), for 17-25 ms more delay with 50 ms holds and 33-49 ms with 67 ms
  holds (90 fps the most), and none for a game holding 30. Every frame shown while it's on is a generated one, so
  the GPU does more: through the fake PS5 at 90 fps (`frametap_fake.py --game-fps 27`, encoder on the same GPU) the
  median delay went from 37 to 66 ms and extra frames shown from 97% to 79-83% (two runs). Not tried with a real game
  that dips yet.
- **One frame generator at a time.** Leave NVIDIA Smooth Motion, AMD Fluid Motion Frames and Lossless Scaling off
  for SmoovPlay and chiaki-ng: on top of SmoovPlay they'd make frames out of its generated ones (more wobble) and
  add delay. Your other games aren't affected.
