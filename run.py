"""SmoovPlay's pipeline: real-time frame generation (a 30 fps game shown at 60, 90 or 120 fps) and upscaling, for
PS5 Remote Play, a capture card (Experimental) or a video file.

Examples
--------
  # Simulate PS5 output from a native-60fps recording, watch it in a window at real speed:
  python run.py testdata/clip_1080p60.mp4 --display --realtime

  # Same, as fast as possible, write the result and score it against the real dropped frames:
  python run.py testdata/clip_1080p60.mp4 --out out/clip_interp.mp4 --score

  # A capture card (Experimental, not tested with a real card yet): by name or number from --list-devices, or a
  # recording standing in for one:
  python run.py --capture "USB3.0 Capture" --display
  python run.py --capture 0 --display
  python run.py --capture testdata/clip_1080p60.mp4 --test-mode --display

  # Another app's window (Experimental): Sony's PS Remote Play app, the Xbox app, Moonlight, a browser... by preset,
  # program, handle or part of its title (python -m framegen.window --list):
  python run.py --source window --window ps-remote-play --display
  python run.py --source window --window "exe:XboxPcApp.exe" --display

  # PS5 Remote Play through chiaki-ng's frame tap (start chiaki-ng with CHIAKI_FRAMETAP=1, see chiaki_tap/):
  python run.py --chiaki --display --chiaki-log out/session.csv

  # Any app that sends its frames through the frame tap (docs/frame-tap.md, Experimental), compressed or decoded;
  # scripts/frametap_send.py stands in for one:
  python run.py --source tap --display

  # Test mode: a 60 fps PS5 game played as if it were 30 fps, each generated frame scored against the real
  # one, with a plain-language report (also in the window's menu: Simulate a 30 fps game):
  python run.py --chiaki --test-mode --display --results out/results.txt

  # NVIDIA VFG in place of RIFE (RTX 40/50; set up with framegen/vfg_setup.py, folder in FRAMEGEN_VFX_PATH):
  python run.py --chiaki --display --model vfg --vfg-quality medium

  # The sources, for the launcher (--source picks one: chiaki, capture, window, tap or file):
  python run.py --list-sources --json

Settings (frame rate, upscaler, ...) are chosen in the window's menu (key M) and saved to
%APPDATA%\\framegen\\settings.json. Command-line flags override them for one run without saving.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if not sys.path or sys.path[0] != _ROOT:
    sys.path.insert(0, _ROOT)

if __name__ == "__main__" and "--list-sources" in sys.argv:  # answered before the heavy imports below
    from framegen.sources import main as list_sources
    sys.exit(list_sources(sys.argv[1:]))

import argparse
import gc
import os
import threading
import time

# torch's OpenMP threads wait for work asleep instead of spinning. Without CUDA (DirectML) the repeat checks run on the
# CPU every frame, and threads spinning between them kept most of the CPU's cores busy for the whole session. Set
# before torch loads, as OpenMP reads it once.
os.environ.setdefault("OMP_WAIT_POLICY", "PASSIVE")

import cv2
import torch

from framegen.interpolator import MODELS, RifeInterpolator
from framegen.io import (TICK, CaptureSource, FileSink, FileSource, GLWindowSink, MultiSink, Ps5SimSource,
                         WindowSink)
from framegen.pipeline import Pipeline
from framegen.settings import BY_KEY, DEFAULT_PATH, SPECS, Settings
from framegen.sources import TestMode, make_detector
from framegen.vfg import vfg_unavailable

# Test mode's report says this when it includes NVIDIA VFG: the launcher shows it, but Copy diagnostics leaves it
# out (installer/launcher Diagnostics looks for this text), because NVIDIA's licence forbids sharing VFG's results.
PRIVATE_RESULTS = ("These results include NVIDIA VFG. NVIDIA's licence doesn't allow sharing them, so they stay on "
                   "this PC: Copy diagnostics leaves them out.")
# The video loop stuck this long while playing: the log gets where each thread is. Under Windows' 5 s, after which it
# calls the window "not responding" and the player may close it.
WATCHDOG_S = 4.0
STARVED_S = 8.0  # no picture from the source this long: the same, once per stall
FROZEN_S = 4.0  # Python itself stuck this long (no thread of ours can run, the watchdog included): faulthandler's dump


def frozen_dump(on: bool, armed: bool, file=None, after: float = FROZEN_S) -> bool:
    """The last resort when Python freezes as a whole (a call that never lets go of Python, so the watchdog can't run
    either): Python's faulthandler writes where every thread is from a timer of its own, `after` seconds from now,
    unless pushed back by another call. on: arm it, or push it back; else cancel it if `armed`. Returns whether it's
    armed. Its lines ("Thread 0x... (most recent call first):" and 'File "...", line N in ...') go straight to stderr,
    so the session log has them even when Windows ends the process; Copy diagnostics masks the user folder in them."""
    import faulthandler
    try:
        if on:
            faulthandler.dump_traceback_later(after, file=sys.stderr if file is None else file)
            return True
        if armed:
            faulthandler.cancel_dump_traceback_later()
    except (AttributeError, OSError, RuntimeError, ValueError):  # no stderr with a file descriptor to write to
        pass
    return False


def _mmss(seconds: float) -> str:
    return f"{int(seconds) // 60}:{int(seconds) % 60:02d}"


def gpu_name() -> str:
    """The graphics card doing the work: CUDA's, or (without it) the one DirectML picks."""
    if torch.cuda.is_available():
        return torch.cuda.get_device_name(0)
    try:
        from framegen.dml import pick_adapter
        return pick_adapter()["name"]
    except Exception:  # noqa: BLE001
        return "unknown"


def write_results(path: str, sessions: list, size: str, settings: Settings) -> bool:
    """Test mode's report, in plain words (the launcher shows it after a session and adds it to Copy diagnostics).
    sessions: (session config, Stats) for each session of this run that was tested; a settings change that needs
    a restart (e.g. 120 fps) starts a new one. Written whole each time, via a temporary file, so it's never seen
    half-written."""
    sessions = [(cfg, st) for cfg, st in sessions if st is not None and (st.simulated or st.psnr_interp)]
    if not sessions:
        return False
    played = sum(len(st.busy_ms) for _, st in sessions) * TICK
    simulated = sum(st.simulated for _, st in sessions) * TICK
    lines = [f"SmoovPlay test results, {time.strftime('%Y-%m-%d %H:%M')}",
             f"Stream: {size}, played {_mmss(played)}"
             + (f", {_mmss(simulated)} of it as a simulated 30 fps game" if simulated else ""),
             f"Graphics card: {gpu_name()}",
             f"Upscaler: {BY_KEY['scaler'].name(settings.scaler)}"]
    if any(cfg["model"] == "vfg" for cfg, _ in sessions):
        lines += [PRIVATE_RESULTS]
    for i, (cfg, st) in enumerate(sessions, 1):
        head = f"{f'Part {i}' if len(sessions) > 1 else 'AI engine'}: "
        if cfg.get("engine") == "dml":
            lines += ["", head + f"DirectML (Experimental), motion detail {cfg['motion']}, RIFE {cfg['model']}"]
            continue
        if cfg["model"] == "vfg":
            lines += ["", head + f"NVIDIA VFG, quality {BY_KEY['vfg_quality'].name(cfg['vfg_quality'])}"]
            lines += st.results(full_motion=False)  # no motion detail setting: the advice suggests a lower rate
            continue
        engine = "TensorRT" if cfg["engine"] == "trt" else "PyTorch"
        motion = {1.0: "Full", 0.5: "Half", 0.25: "Quarter"}.get(cfg["motion"], f"{cfg['motion']:g}x")
        lines += ["", head + f"{engine}, motion detail {motion}, RIFE {cfg['model']}"]
        lines += st.results(full_motion=cfg["motion"] >= 1)
    try:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(out.name + ".tmp")
        tmp.write_text("\n".join(lines) + "\n", encoding="utf-8")
        tmp.replace(out)
        return True
    except OSError as e:
        print(f"[results] couldn't write {path}: {e}")
        return False


def write_stream_check(src, results: str | None) -> None:
    """Remote Play: how well the size check read this game (framegen.streamcheck), into the log (so Copy
    diagnostics has it) and, when the launcher asked for results, beside them as stream-check-<stamp>.txt."""
    check = getattr(src, "check", None)
    lines = check.report() if check is not None else []
    if not lines:
        return
    print("\n".join(lines))
    if not results:
        return
    out = Path(results)
    path = out.with_name(out.name.replace("results-", "stream-check-", 1) if "results-" in out.name
                         else "stream-check.txt")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"SmoovPlay stream check, {time.strftime('%Y-%m-%d %H:%M')}\n" + "\n".join(lines) + "\n",
                        encoding="utf-8")
        print(f"[results] stream check -> {path}")
    except OSError as e:
        print(f"[results] couldn't write {path}: {e}")


def open_capture(a, w: int, h: int, test):
    """--capture: a card by name or number (framegen.capture), or a file standing in for one."""
    if a.capture_backend == "opencv":
        device = int(a.capture) if a.capture.isdigit() else a.capture
        return CaptureSource(device, w, h, fourcc=a.capture_fourcc, test=test)
    from framegen import capture
    colour = {"auto": "auto", "709-limited": (True, False), "709-full": (True, True), "601-limited": (False, False),
              "601-full": (False, True)}[a.capture_colour]
    sound = None if a.capture_audio == "off" else sound_player()
    if Path(a.capture).is_file():  # a recording standing in for a card: decoded, sent through the raw NV12 path
        return capture.CardSource.video_file(a.capture, test=test, colour=colour, audio=sound)
    name = a.capture
    devices = capture.list_devices()
    if name.isdigit():
        video = [d for d in devices if d.kind in ("video", "unknown")]
        if int(name) >= len(video):
            sys.exit(f"[capture] no capture device {name}: Windows lists {len(video)} (see --list-devices)")
        name = video[int(name)].name
    audio_device = None
    if sound is not None:
        audio = [d.name for d in devices if d.kind == "audio"]
        if a.capture_audio == "auto":
            audio_device = capture.pick_audio(name, devices)
            if audio_device is None:
                # Some cards carry their sound on the video device itself (in OBS: Video Capture Device with no custom
                # audio device), not as a sound device of their own, e.g. the AVerMedia GC573: try that. FFmpeg looks
                # for a video device with a sound pin by that name; CardSource.open plays the picture alone if not.
                audio_device = name
                print("[capture] no sound device named like the card: trying the card's own sound")
        elif a.capture_audio.isdigit() and int(a.capture_audio) < len(audio):
            audio_device = audio[int(a.capture_audio)]
        else:
            audio_device = a.capture_audio
    mode = None
    if a.capture_format != "auto":
        modes = [m for m in capture.list_modes(name) if m.fmt == a.capture_format]
        mode, note = capture.pick_mode(modes, w, h, a.capture_fps)
        if mode is None:
            sys.exit(f"[capture] {name} has no {a.capture_format} mode ({note})")
    return capture.CardSource.open(name, w, h, a.capture_fps, mode=mode, test=test, colour=colour,
                                   audio_device=audio_device, audio=sound)


def sound_player():
    """What plays a capture card's sound (framegen.audio.Player), or None when sounddevice isn't installed."""
    try:
        import sounddevice  # noqa: F401
    except Exception as e:  # noqa: BLE001 - not installed, or PortAudio missing
        print(f"[capture] no sound: sounddevice isn't available ({str(e)[:80]})")
        return None
    from framegen.audio import Player
    return Player


FRAME_LOGS_KEPT = 10


def frame_log(results: str | None) -> str | None:
    """Played from the launcher (--results): a per-frame log of the stream next to the session's own (frames-<the
    session's time>.csv: sizes, timing and the pipeline's calls, no pictures; about 1.5 MB per 5 min), for tuning
    the repeat check on real games. The last FRAME_LOGS_KEPT are kept."""
    if not results:
        return None
    r = Path(results)
    stamp = r.stem.split("-", 1)[1] if "-" in r.stem else r.stem
    try:
        for old in sorted(r.parent.glob("frames-*.csv"))[:-(FRAME_LOGS_KEPT - 1)]:
            old.unlink()
    except OSError:
        pass
    path = r.parent / f"frames-{stamp}.csv"
    print(f"[run] frame log -> {path.name}")
    return str(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video", nargs="?", help="input video file")
    ap.add_argument("--source", choices=["chiaki", "capture", "window", "tap", "file"],
                    help="where frames come from (default: from the flags below; a video file if one is given)")
    ap.add_argument("--list-sources", action="store_true",
                    help="list the sources and exit; with --json, as JSON for the launcher")
    ap.add_argument("--json", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--window", metavar="APP",
                    help="--source window: the app's window, as a preset (ps-remote-play, xbox, moonlight), exe:NAME, "
                         "hwnd:N or part of its title (framegen/window.py, Experimental)")
    ap.add_argument("--window-keys", metavar="KEYS", default=None,
                    help="--source window: SmoovPlay's shortcuts while the app has the keys, as "
                         '"menu=Ctrl+Alt+K;quit=Ctrl+Alt+End" (actions: menu, quit, fullscreen, stats, smooth; the '
                         'others keep their defaults, framegen/hotkeys.py)')
    ap.add_argument("--capture", metavar="NAME|N|FILE",
                    help="a capture card by name or number (from --list-devices), or a video file standing in for "
                         "one (played in real time through the capture path, to test it without a card)")
    ap.add_argument("--capture-size", default=None, metavar="WxH",
                    help="the size to ask the card for (default: Capture card resolution in the settings, 1080p)")
    ap.add_argument("--capture-fps", type=float, default=60.0)
    ap.add_argument("--capture-format", default="auto",
                    help="auto (NV12, then YUY2, then MJPEG: framegen/capture.py) or one the card lists")
    ap.add_argument("--capture-colour", default="auto", choices=["auto", "709-limited", "709-full", "601-limited",
                                                                    "601-full"],
                    help="how to read the card's colours (auto: limited range unless the card says full; BT.709 "
                         "from 720p up)")
    ap.add_argument("--capture-audio", default="auto", metavar="auto|off|NAME|N",
                    help="the card's sound, played on this PC in step with the picture: auto (the sound device named "
                         "like the card), off, or a sound device by name or number from --list-devices")
    ap.add_argument("--capture-backend", default="pyav", choices=["pyav", "opencv"],
                    help="pyav: FFmpeg's DirectShow input, raw frames converted on the GPU (default); opencv: the "
                         "older OpenCV path, for comparison")
    ap.add_argument("--capture-fourcc", default="MJPG", help="[opencv backend] MJPG, YUY2 or NV12")
    ap.add_argument("--list-devices", action="store_true", help="list capture devices and exit")
    ap.add_argument("--chiaki", action="store_true", help="PS5 Remote Play frames from chiaki-ng's frame tap")
    ap.add_argument("--chiaki-decoder", default="cuda", choices=["cuda", "sw", "d3d11va"],
                    help="how to decode the Remote Play stream (default cuda: NVDEC, frames stay on the GPU, ~1.5 ms "
                         "a frame; sw takes ~10 ms a frame on gameplay)")
    ap.add_argument("--chiaki-log", metavar="CSV", help="write one row per tick: stream facts and pipeline decisions")
    ap.add_argument("--minimize-chiaki", action=argparse.BooleanOptionalAction, default=None,
                    help="minimise chiaki-ng's window while ours is up (it keeps streaming but stops drawing video "
                         "nobody sees, freeing GPU time); restored on exit. Default: the saved setting (on)")
    ap.add_argument("--test-mode", "--chiaki-as-30", dest="test_mode", action="store_true",
                    help="test mode: play a 60 fps game as if it were 30 fps by keeping every other frame, and rate "
                         "each generated frame against the real one that was left out (also in the window's menu: "
                         "Simulate a 30 fps game). Remote Play, a capture card, or a video file with --raw")
    ap.add_argument("--chiaki-tap", default=None, help=argparse.SUPPRESS)  # another shared-memory name, for tests
    ap.add_argument("--tap", metavar="NAME", default=None,
                    help="--source tap: the frame tap's shared-memory name (default Local\\smoovplay-tap; "
                         "docs/frame-tap.md)")
    ap.add_argument("--raw", action="store_true",
                    help="play the file's frames as-is (a real capture recording) instead of simulating PS5 30fps")
    ap.add_argument("--realtime", action="store_true", help="pace file input at 60 Hz like a capture card")
    ap.add_argument("--max-ticks", type=int)
    ap.add_argument("--hitch", type=float, default=0.0, help="[sim] probability a game frame is held 3 ticks")
    ap.add_argument("--noise", type=float, default=0.0, help="[sim] gaussian noise sigma added per tick")
    ap.add_argument("--model", default=None, choices=list(MODELS) + ["vfg"],
                    help="RIFE version, or vfg for NVIDIA VFG (default: the saved setting, 4.22.lite). The other "
                         "RIFE versions are for testing")
    ap.add_argument("--vfg-quality", default=None, choices=list(BY_KEY["vfg_quality"].choices),
                    help="NVIDIA VFG's quality mode (default: the saved setting, medium)")
    ap.add_argument("--scale", type=float, default=None, help="flow resolution scale (default 1.0, 0.5 above 1440p)")
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--engine", choices=["auto", "trt", "torch"], default=None,
                    help="run the flow network with TensorRT (engine built once per resolution, cached in models/trt) "
                         "or PyTorch; auto = TensorRT at full motion detail if installed, PyTorch at half (faster "
                         "there). Default: the saved setting (auto)")
    ap.add_argument("--trt", action="store_true", help="same as --engine trt")
    ap.add_argument("--no-graph", action="store_true", help="disable CUDA graphs")
    ap.add_argument("--dup-threshold", type=float, default=2.0)
    ap.add_argument("--off", action="store_true", help="start with interpolation disabled (toggle with 'i')")
    ap.add_argument("--display", action="store_true", help="show output in an OpenGL window")
    ap.add_argument("--settings", default=str(DEFAULT_PATH), metavar="JSON",
                    help="settings file the menu loads and saves (default %(default)s); 'none' = defaults, unsaved")
    # These override the saved settings for this run (not saved); the menu can change them live.
    ap.add_argument("--vsync", action=argparse.BooleanOptionalAction, default=None,
                    help="vsync the window (use with G-Sync/FreeSync displays)")
    ap.add_argument("--fullscreen", action=argparse.BooleanOptionalAction, default=None, help="start fullscreen")
    ap.add_argument("--fill", default=None, choices=BY_KEY["fill"].choices,
                    help="on a screen that isn't 16:9: fit (bars), zoom (crops), stretch or panorama; all of them the same 16:9 picture")
    ap.add_argument("--hdr", default=None, choices=BY_KEY["hdr"].choices,
                    help="an HDR stream from PS5 Remote Play: off (as before), sdr (tone-mapped to SDR) or hdr10 (HDR "
                         "output, Experimental: an HDR display with Windows' HDR on). Default: the saved setting")
    ap.add_argument("--present", default=None, choices=BY_KEY["present"].choices,
                    help="how the window presents: gl = OpenGL's own, dxgi = DXGI flip model (Experimental: less "
                         "delay in a window, G-Sync in a window). Default: the saved setting")
    ap.add_argument("--steady-hud", action=argparse.BooleanOptionalAction, default=None,
                    help="keep unchanged parts of the picture (HUD, map) as the game drew them. Default: the saved setting")
    ap.add_argument("--scaler", default=None, choices=["fsr", "vsr", "bilinear"],
                    help="how the window upscales when it is bigger than the frame (cycle live with 'u'); "
                         "vsr = NVIDIA RTX Video Super Resolution, needs the RTX Video SDK (framegen/rtxvsr.py)")
    ap.add_argument("--vsr-quality", type=int, default=None, choices=[1, 2, 3, 4],
                    help="RTX VSR quality: 1 low ... 4 ultra (more GPU time per frame)")
    ap.add_argument("--fsr-sharpness", type=float, default=None, choices=BY_KEY["fsr_sharpness"].choices,
                    help="FSR 1 sharpening, in stops below maximum: 0 = sharpest, 1 = half, 2 = quarter")
    ap.add_argument("--look", default=None, metavar="ID|off",
                    help="turn a look on for this run (e.g. vivid, my:crisp) or 'off' to disable looks")
    ap.add_argument("--hz", type=int, default=None, choices=[60, 90, 120],
                    help="output rate for a 30 fps game: 120 = 3 generated frames per real one (half-res flow by "
                         "default), 90 = 2 (needs a VRR/G-Sync screen or a 90 Hz mode to look even). Needs --display; "
                         "falls back to 60 Hz output whenever it can't keep up")
    ap.add_argument("--game-fps", default=None, choices=BY_KEY["game_fps"].choices,
                    help="game's own frame rate: auto (read from the stream) or 30 (every other tick is new, so "
                         "CBR/dark/still scenes never fool Even out 40-60 fps). Default: 30 for --source tap, else "
                         "the saved setting")
    ap.add_argument("--display-cv", action="store_true", help="use the (slow) OpenCV window instead of OpenGL")
    ap.add_argument("--out", help="write output video")
    ap.add_argument("--score", action="store_true", help="[sim] PSNR of interpolated frames vs real dropped frames")
    ap.add_argument("--save-worst", metavar="JPG", help="[with --score] save prev | RIFE | real | next for the worst frame")
    ap.add_argument("--control", action="store_true",
                    help="take settings commands on stdin and print [settings-state] lines (the launcher's Settings "
                         "page; see framegen/remote.py)")
    ap.add_argument("--results", metavar="TXT",
                    help="when test mode (a simulated 30 fps game) or --score was used, write a plain-language report "
                         "of performance and quality here: every 10 s while playing, and at the end")
    a = ap.parse_args()
    window_keys = None  # --source window: the shortcuts picked (framegen/hotkeys.py)

    if a.list_devices:
        from framegen.capture import main as capture_main
        return capture_main(["--list"] + (["--json"] if a.json else []))
    # Without NVIDIA's CUDA (an AMD or Intel card), RIFE runs through DirectML (framegen/dml.py, Experimental): frames
    # stay on the CPU between steps, and Remote Play is decoded with Direct3D 11 instead of NVDEC.
    dml_mode = not torch.cuda.is_available()
    if dml_mode:
        from framegen import dml
        why = dml.unavailable_reason()
        if why:
            print("SmoovPlay needs an NVIDIA RTX card (CUDA), or DirectML for AMD and Intel cards (Experimental), and "
                  f"neither works here: {why}. Update the graphics driver, then try again.", file=sys.stderr)
            return 1
        print(f"[run] no NVIDIA CUDA here: RIFE runs through DirectML (Experimental) on {dml.pick_adapter()['name']}")
        if a.chiaki_decoder == "cuda":
            a.chiaki_decoder = "d3d11va"

    kind = a.source or ("chiaki" if a.chiaki else "capture" if a.capture is not None else "file" if a.video else None)
    picked = kind  # as asked for: an app's compressed frames on the tap become "chiaki" below, without chiaki-ng
    # Test mode ("Simulate a 30 fps game") works with any 60 Hz source; a video file simulates a PS5 by itself
    # unless --raw plays it as-is.
    test = TestMode(a.test_mode) if kind in ("chiaki", "capture", "tap") or (kind == "file" and a.raw) else None
    if a.test_mode and test is None:
        ap.error("--test-mode needs Remote Play, a capture card, the frame tap, or a video file with --raw")
    webrtc_bridge = None
    if kind == "tap":
        from framegen.tap import DEFAULT_NAME, open_tap
        from framegen.webrtc_tap import DEFAULT_PORT, WebRtcTapBridge
        tap_name = a.tap or DEFAULT_NAME
        if sys.platform == "win32":
            try:
                webrtc_bridge = WebRtcTapBridge(tap_name=tap_name, port=DEFAULT_PORT)
                webrtc_bridge.start()
                print(f"[webrtc-tap] listening on ws://127.0.0.1:{webrtc_bridge.port} -> {tap_name}")
            except OSError:
                webrtc_bridge = None  # another bridge is already listening on the port
        try:  # compressed frames come through the same reader as chiaki-ng's, so they count as "chiaki" from here
            kind, src = open_tap(tap_name, a.chiaki_decoder, wait_s=600, test=test,
                                 log_path=a.chiaki_log)
        except (ValueError, TimeoutError) as e:
            if webrtc_bridge is not None:
                webrtc_bridge.stop()
            print(f"[tap] {e}", file=sys.stderr)
            return 1
        if not src.info.test_mode:  # an app that sends only new pictures: no test mode (as for window capture)
            test = None
    elif kind == "chiaki":
        from framegen.chiaki_tap import NAME, ChiakiSource
        src = ChiakiSource(a.chiaki_decoder, wait_s=600, log_path=a.chiaki_log or frame_log(a.results),
                           tap_name=a.chiaki_tap or NAME, test=test)
    elif kind == "capture":
        if a.capture is None:
            ap.error("--source capture needs --capture N (a device) or --capture FILE")
        if a.capture_size:
            cw, ch = map(int, a.capture_size.split("x"))
        else:  # Capture card resolution (the settings are loaded again below, with the command line's overrides)
            from framegen.settings import CAPTURE_SIZES
            saved = Settings() if a.settings.lower() == "none" else Settings.load(Path(a.settings))
            cw, ch = CAPTURE_SIZES[saved.capture_size]
        print(f"[capture] asking the card for {cw}x{ch}" + ("" if (cw, ch) == (1920, 1080) else
              " (Capture card resolution: every new frame is drawn at this size, so it costs more than 1080p)"))
        src = open_capture(a, cw, ch, test)
    elif kind == "window":
        if not a.window:
            ap.error("--source window needs --window APP (python -m framegen.window --list shows the windows)")
        from framegen.hotkeys import parse_arg
        try:
            window_keys = parse_arg(a.window_keys)
        except ValueError as e:
            ap.error(f"--window-keys: {e}")
        from framegen.sources import Clock60
        from framegen.window import open_window
        try:
            src = Clock60(open_window(a.window))  # the app draws when it likes: put it on a 60 Hz beat
        except RuntimeError as e:
            print(f"[window] {e}", file=sys.stderr)
            return 1
    elif kind == "file" and a.video:
        src = (FileSource(a.video, a.realtime, a.max_ticks, test=test) if a.raw else
               Ps5SimSource(a.video, a.realtime, a.hitch, a.noise, a.max_ticks))
    else:
        ap.error("give a video file, --capture N or --chiaki")
    print(f"[run] source: {src.info.name}")

    settings = Settings() if a.settings.lower() == "none" else Settings.load(Path(a.settings))
    for key, value in (("hz", a.hz), ("scaler", a.scaler), ("vsr_quality", a.vsr_quality),
                       ("fsr_sharpness", a.fsr_sharpness), ("engine", "trt" if a.trt else a.engine),
                       ("interp", False if a.off else None), ("vsync", a.vsync), ("fullscreen", a.fullscreen),
                       ("minimize_chiaki", a.minimize_chiaki), ("simulate_30", True if a.test_mode else None),
                       ("steady_hud", a.steady_hud), ("vfg_quality", a.vfg_quality), ("present", a.present),
                       ("hdr", a.hdr), ("fill", a.fill), ("game_fps", a.game_fps)):
        if value is not None:
            settings.override(key, value)
    if picked == "tap" and a.game_fps is None and settings.game_fps != "30":
        settings.override("game_fps", "30")
        print("[run] tap: locking game_fps=30 (override with --game-fps auto)")
    if a.look is not None:
        if a.look.lower() == "off":
            settings.override("look_on", False)
        else:
            from framegen.looks import Library
            Library()
            if a.look not in BY_KEY["look"].choices:
                ap.error(f"--look: unknown look {a.look!r} (available: {', '.join(BY_KEY['look'].choices)})")
            settings.override("look", a.look)
            settings.override("look_on", True)
    # --model: one of the settings' choices, or another RIFE version for testing (not something the menu offers)
    test_model = a.model if a.model is not None and a.model not in BY_KEY["model"].choices else None
    if a.model is not None and test_model is None:
        settings.override("model", a.model)
    if test is not None:  # the menu's "Simulate a 30 fps game" switches test mode live
        settings.watch(lambda key: test.set_on(settings.simulate_30) if key == "simulate_30" else None)
    if settings.hz > 60 and (a.out or not a.display):
        if a.hz is not None:
            ap.error("--hz 90/120 needs --display, and doesn't write --out yet")
        print(f"[run] output {settings.hz} Hz needs --display without --out; 60 Hz this run")
        settings.override("hz", 60)

    # HDR (framegen/hdr.py, settings "hdr"): only for a PS5 Remote Play stream in HDR, and only with the setting on;
    # otherwise nothing below changes. hdr_out: what the stream's frames become ("sdr8": tone-mapped to SDR; "pq16":
    # kept HDR10, for HDR output), or None. hdr10[0]: the pipeline carries HDR10 frames (half floats) to the window.
    hdr_out, hdr_why, hdr_peak = None, "", None
    if kind == "chiaki" and settings.hdr != "off" and getattr(src, "hdr_stream", False):
        if dml_mode:
            hdr_why = "SmoovPlay's HDR needs an NVIDIA card for now"
        else:
            hdr_out = "sdr8"
            if settings.hdr == "hdr10":
                from framegen.dxgi_present import HDR_FORCE, display_hdr
                shown = display_hdr()
                if not a.display or a.display_cv or a.out:
                    print("[run] HDR output needs the video window; showing HDR in SDR")
                elif HDR_FORCE or (shown is not None and shown["hdr"]):
                    hdr_out = "pq16"
                    if shown is not None and shown["hdr"] and shown["max_nits"] > 0:
                        hdr_peak = shown["max_nits"]  # highlights brighter than the display are rolled off to it
                    print("[run] HDR output: HDR10"
                          + (f" to a display that goes to about {hdr_peak:.0f} nits" if hdr_peak else "")
                          + (" (testing: FRAMEGEN_HDR_FORCE)" if HDR_FORCE else ""))
                else:
                    print("[run] HDR output: this display doesn't have Windows' HDR on (Settings > System > Display > "
                          "Use HDR), so HDR is shown in SDR")
    hdr10 = [hdr_out == "pq16"]

    from framegen.trt import auto_engine, trt_available
    has_trt = trt_available()
    if settings.engine == "trt" and not has_trt:
        print("[run] TensorRT isn't installed (see framegen/trt.py); using PyTorch")
    vfg_why = [vfg_unavailable()]  # why NVIDIA VFG can't be used ("" if it can); a failed start sets it too
    rife = BY_KEY["model"].default
    if settings.model == "vfg" and vfg_why[0]:
        print(f"[run] {vfg_why[0]} Using {BY_KEY['model'].name(rife)}.")
        settings.override("model", rife)  # this run only: the saved choice comes back once VFG is set up

    # Auto motion detail also goes by Auto-detect's last time at full detail on this graphics card: where that doesn't
    # fit a frame (an RTX 20 card at 60 fps), Auto is half
    from framegen.autodetect import LAPTOP_SHARE, SHARE, auto_scale, saved_full_ms
    card = gpu_name()
    card_full_ms = None if a.settings.lower() == "none" else saved_full_ms(Path(a.settings), card)
    card_share = LAPTOP_SHARE if "laptop" in card.lower() else SHARE
    if (a.scale is None and settings.motion == "auto" and card_full_ms is not None
            and auto_scale(src.w, src.h, settings.hz, card_full_ms, card_share) < auto_scale(src.w, src.h, settings.hz)):
        print(f"[run] motion detail Auto: half, as Auto-detect timed full detail at {card_full_ms:.1f} ms on this "
              f"graphics card, more than {settings.hz} fps leaves")

    def session_config() -> dict:
        """What a session is built from; a restart is needed when this changes (keys: settings.SPECS)."""
        if (test_model or settings.model) == "vfg" and not dml_mode:
            return {"model": "vfg", "vfg_quality": settings.vfg_quality}
        if a.scale is not None:
            scale = a.scale
        else:  # Auto (autodetect.auto_scale). On an RTX 4070 Ti SUPER (scripts/benchmark.py), per 30 fps game frame
            # (33 ms): 120 Hz at 1080p needs half (full ~54 ms with display work); 1440p at 90 too (38 ms; half 26); 4K
            # even at 60 Hz needs a quarter (full 52 ms, half 36, quarter 30: its motion worked out at 960x540, as
            # half's is at 1080p).
            scale = {"full": 1.0, "half": 0.5}.get(settings.motion) or auto_scale(
                src.w, src.h, settings.hz, card_full_ms, card_share)
        engine = auto_engine(scale) if settings.engine == "auto" else settings.engine
        if dml_mode:
            engine = "dml"
        elif not has_trt:
            engine = "torch"
        model = test_model or settings.model
        return {"engine": engine, "motion": scale, "model": BY_KEY["model"].default if model == "vfg" else model}

    started: dict = {}

    def restart_reasons() -> list[str]:
        cfg_ = session_config()
        if cfg_["model"] != started.get("model"):  # RIFE <-> VFG: the other's settings don't count as reasons
            return [BY_KEY["model"].label]
        return [BY_KEY[k].label for k, v in cfg_.items() if v != started.get(k)]

    def build(cfg: dict):
        if cfg["model"] == "vfg":
            from framegen.vfg import VfgInterpolator
            return VfgInterpolator(src.w, src.h, cfg["vfg_quality"], hdr=hdr10[0])
        if cfg["engine"] == "dml":
            from framegen.dml import DmlInterpolator
            return DmlInterpolator(src.w, src.h, cfg["model"], cfg["motion"])
        return RifeInterpolator(src.w, src.h, cfg["model"], cfg["motion"], fp16=not a.fp32, cuda_graph=not a.no_graph,
                                backend=cfg["engine"], hdr=hdr10[0])

    def build_or_rife(cfg: dict, busy=lambda fn, text: fn()):
        """build(cfg); if NVIDIA VFG fails to start (e.g. a broken install or driver), RIFE instead, so play goes on.
        Returns (cfg, interpolator)."""
        try:
            return cfg, busy(lambda: build(cfg), "Starting NVIDIA VFG" if cfg["model"] == "vfg" else None)
        except Exception as e:  # noqa: BLE001
            if cfg["model"] != "vfg":
                raise
            vfg_why[0] = f"NVIDIA VFG couldn't start: {str(e)[:200]}"
            print(f"[run] {vfg_why[0]}; using {BY_KEY['model'].name(rife)}")
            settings.override("model", rife)
            model_rows()
            cfg = session_config()
            return cfg, busy(lambda: build(cfg), f"Starting {BY_KEY['model'].name(rife)}")

    def describe(cfg: dict) -> str:
        if cfg["model"] == "vfg":
            return f"NVIDIA VFG quality={cfg['vfg_quality']}"
        if cfg["engine"] == "dml":
            return f"RIFE {cfg['model']} scale={cfg['motion']}  DirectML"
        return (f"RIFE {cfg['model']} scale={cfg['motion']} {'fp32' if a.fp32 else 'fp16'} "
                f"graphs={'off' if a.no_graph else 'on'}  {'TensorRT' if cfg['engine'] == 'trt' else 'PyTorch'}")

    def label_of(cfg: dict) -> str:  # the overlay's first words
        hdr_tag = {"sdr8": "  HDR in SDR", "pq16": "  HDR10"}.get(hdr_out, "")
        if cfg["model"] == "vfg":
            return f"VFG {cfg['vfg_quality']}{hdr_tag}"
        engine = {"trt": " TRT", "dml": " DML"}.get(cfg["engine"], "")
        return f"RIFE {cfg['model']}{engine}{' half' if cfg['motion'] < 1 else ''}{hdr_tag}"

    window = None

    def model_rows(key: str = "model") -> None:
        """Grey out the rows the chosen model doesn't use (the window's menu and the launcher's Settings page)."""
        rows = getattr(window, "unavailable", None)
        if key not in ("model", "vfg_quality") or rows is None:
            return
        rows.clear()
        rows.update(unavailable)
        if (test_model or settings.model) == "vfg":
            for k in ("engine", "motion"):
                rows.setdefault(k, "Not used with NVIDIA VFG.")
        else:
            rows.setdefault("vfg_quality", vfg_why[0] or "Only used with NVIDIA VFG.")
        if hasattr(window, "choices"):
            window.choices["model"] = tuple(m for m in BY_KEY["model"].choices if m != "vfg" or not vfg_why[0])

    cfg = session_config()
    print(f"[run] {src.w}x{src.h}  {describe(cfg)}  output {settings.hz} Hz")
    cfg, interp = build_or_rife(cfg)

    unavailable = {s.key: why for s in SPECS if (why := s.unused_with(picked or ""))}  # e.g. Capture card resolution
    if test is None:
        unavailable["simulate_30"] = ("Not with window capture yet." if kind == "window" else
                                      "Video files already play as a 30 fps game.")
    if kind != "chiaki":
        unavailable["hdr"] = "Only used with PS5 Remote Play."
    elif dml_mode:
        unavailable["hdr"] = "Not without an NVIDIA card yet."
    hdr10_rows = {k: "Not with HDR output yet." for k in ("look_on", "look", "look_clips")}
    hdr10_rows["present"] = "HDR output always uses the low-delay presentation."
    if hdr10[0]:
        unavailable.update(hdr10_rows)
    if a.scale is not None:
        unavailable["motion"] = f"Set to {a.scale} by --scale on the command line."
    if test_model:
        unavailable["model"] = f"RIFE {test_model}, set by --model on the command line, for testing."
    if dml_mode:
        unavailable["engine"] = "Without an NVIDIA card, RIFE runs through DirectML (Experimental)."
        unavailable["steady_hud"] = "Not with DirectML yet."
        unavailable.setdefault("model", "NVIDIA VFG needs an NVIDIA RTX card.")
    choices = {"engine": ("auto", "trt", "torch") if has_trt else ("auto", "torch"),
               "model": tuple(m for m in BY_KEY["model"].choices if m != "vfg" or not vfg_why[0])}
    if a.out:
        choices["hz"] = (60,)
        unavailable["hz"] = "90/120 fps can't be recorded with --out yet."
    tap_overlay = None
    if webrtc_bridge is not None and webrtc_bridge.active:
        from framegen.webrtc_tap import find_webview_overlay
        tap_overlay = find_webview_overlay()
        if tap_overlay is not None:
            print("[webrtc-tap] overlaying SmoovPlay over SmoovPlayXboxWebView (XboxWebView keeps controller focus)")
    overlay_fn = src.where if kind == "window" else tap_overlay
    window = (WindowSink(src.w, src.h) if a.display_cv else
              GLWindowSink(src.w, src.h, settings, restart_reasons, choices, unavailable,
                           overlay_of=overlay_fn, hdr=hdr10[0],
                           window_keys=window_keys if overlay_fn is not None else None,
                           hdr_peak=min(hdr_peak or 1e9, 1000.0),
                           clean_source=kind == "capture" and bool(getattr(src, "raw", False))) if a.display else None)
    if hdr10[0] and not getattr(window, "hdr10", False):  # the window couldn't present HDR10: shown in SDR
        hdr10[0], hdr_out, hdr_peak = False, "sdr8", None
        for k in hdr10_rows:
            unavailable.pop(k, None)
            if window is not None:
                window.unavailable.pop(k, None)
        if getattr(window, "keep_hdr", False):  # on 8-bit DXGI instead, as Windows' HDR is on: "present" stays fixed
            window.note_presentation()
            unavailable["present"] = window.unavailable["present"]
        interp = None
        gc.collect()
        torch.cuda.empty_cache()
        cfg, interp = build_or_rife(session_config())  # with 8-bit frames
    if kind == "chiaki" and hasattr(src, "set_hdr"):
        src.set_hdr(hdr_out, hdr_peak if hdr10[0] else None, why=hdr_why)
    sink = MultiSink(window, FileSink(a.out, src.w, src.h) if a.out else None)
    settings.watch(model_rows)  # before Remote's own watcher, so what it publishes is already up to date
    model_rows()
    remote = None
    if a.control and isinstance(window, GLWindowSink):
        from framegen.remote import Remote
        window.remote = remote = Remote(settings, window.allowed, window.disabled, restart_reasons,
                                        handlers=window.remote_commands())

    hidden: list[int] = []

    def minimize_chiaki(key: str = "minimize_chiaki") -> None:
        if key != "minimize_chiaki" or not (picked == "chiaki" and window is not None):
            return
        from framegen.winutil import SW_SHOWMINNOACTIVE, SW_SHOWNOACTIVATE, show, windows_of
        if settings.minimize_chiaki and not hidden:
            hidden[:] = windows_of("chiaki.exe")
            show(hidden, SW_SHOWMINNOACTIVE)  # (without activating another window: the keys stay on the video)
            print(f"[run] minimised chiaki-ng ({len(hidden)} window{'s' if len(hidden) != 1 else ''}); "
                  "restored on exit")
        elif not settings.minimize_chiaki and hidden:
            show(hidden, SW_SHOWNOACTIVATE)  # back, without taking focus from our window
            hidden.clear()

    def keep_chiaki_down() -> None:
        """For the video window's first seconds, and each time it's back in front after that (GLWindowSink.settle):
        chiaki-ng can show its window again just after the stream starts (its stream view opening), and Alt+Tab to it
        restores it, after it was minimised. Minimise it again."""
        if not (picked == "chiaki" and settings.minimize_chiaki):
            return
        from framegen.winutil import SW_SHOWMINNOACTIVE, minimised, show, windows_of
        back = [h for h in windows_of("chiaki.exe") if not minimised(h)]
        if back:
            show(back, SW_SHOWMINNOACTIVE)
            hidden.extend(h for h in back if h not in hidden)
            print(f"[run] chiaki-ng showed its window again: minimised it ({len(back)})")

    settings.watch(minimize_chiaki)
    if isinstance(window, GLWindowSink):
        window.take_keys()  # before chiaki-ng goes down, while it's in front to take them from
    minimize_chiaki()
    if isinstance(window, GLWindowSink):
        window.settle_hook = keep_chiaki_down

    # Instant replay (framegen/replay.py): only while playing on screen. It lasts across restarts, so a restart for a
    # setting doesn't lose what's held.
    replay = None
    if sink.sinks:
        from framegen.replay import SAVING_DIR, Recorder
        replay = Recorder(settings.replay, clip=settings.replay_save, layout=settings.replay_layout,
                          notify=getattr(sink, "toast", None), hotkeys=True, size=(src.w, src.h),
                          mark_dir=SAVING_DIR)
        replay.warm_up()  # its encoders open now, not during play
        if hdr10[0]:  # clips are SDR: HDR10 frames are tone-mapped as they go in, as "shown in SDR" would show them
            from framegen.hdr import pq16_to_sdr8
            replay.hdr_to_sdr = pq16_to_sdr8
        if hasattr(src, "read_sound"):
            replay.attach_sound(src.read_sound)

        def replay_settings(key: str = "look_clips") -> None:
            if key == "replay":
                replay.set_length(settings.replay)
            elif key == "replay_save":
                replay.clip = settings.replay_save
            elif key == "replay_layout":
                replay.layout = settings.replay_layout
            elif key == "look_clips":  # the look in the SmoovPlay video too, drawn by the window at the stream's size
                replay.look = window.baked if settings.look_clips and isinstance(window, GLWindowSink) else None

        settings.watch(replay_settings)
        replay_settings()

    # --results: rewritten every 10 s while playing (the launcher's Stop ends the process without warning) and at
    # the end. `done` holds finished sessions; `live` the running Pipeline (dropped before a restart frees its GPU
    # memory) and what it runs with.
    done: list = []
    live: list = [None, None]
    size = f"{src.w}x{src.h}"

    def results_now() -> bool:
        try:
            pipe_, cfg_ = live  # one read; the session loop clears it before moving the stats into `done`
            running = [(cfg_, pipe_.stats)] if pipe_ is not None else []
            return write_results(a.results, list(done) + running, size, settings)
        except Exception as e:  # noqa: BLE001 - a report must never take the stream down
            print(f"[results] {e}")
            return False

    stop_results = threading.Event()

    def results_loop() -> None:
        while not stop_results.wait(10):
            results_now()

    results_thread = threading.Thread(target=results_loop, name="results", daemon=True)
    if a.results:
        results_thread.start()

    def watchdog() -> None:
        """If the video loop stops going round while playing (it should several times a second, even with no frames
        coming), or no frame has come for a while, says where every thread is, once per stall, so a freeze or a
        black screen in the log says what it was stuck on. While the loop goes round it keeps pushing back
        frozen_dump's timer, which goes off only if Python freezes as a whole and this thread can't run either."""
        import traceback
        home = str(Path.home())

        def where(headline: str) -> None:
            names = {t.ident: t.name for t in threading.enumerate()}
            lines = [f"{headline} Where each thread is:"]
            for ident, frame in sys._current_frames().items():
                if ident == threading.get_ident():
                    continue
                lines.append(f"  thread {names.get(ident, ident)}:")
                lines += ["    " + s.rstrip().replace("\n", "\n    ") for s in traceback.format_stack(frame)[-12:]]
            print("\n".join(lines).replace(home, "~"), flush=True)  # (no user name in Copy diagnostics)

        stuck = starved = armed = False
        while not stop_results.wait(1.0):
            pipe_ = live[0]
            age = time.perf_counter() - pipe_.beat if pipe_ is not None else 0.0
            w = pipe_.waiting if pipe_ is not None else None
            # Only while playing (a TensorRT build between sessions may hold Python on purpose), and not once this
            # thread has seen the loop stuck: it says where everything is itself, below
            armed = frozen_dump(pipe_ is not None and age < WATCHDOG_S, armed)
            if age >= WATCHDOG_S and not stuck:
                stuck = True
                where(f"[watchdog] the video loop hasn't moved for {age:.0f} s.")
            elif age < WATCHDOG_S and stuck:
                stuck = False
                print("[watchdog] the video loop is going again", flush=True)
            if w is not None and w.since >= STARVED_S and not starved:
                starved = True
                why = w.why.replace("\n", " ")
                where(f"[watchdog] no picture from {w.what} for {w.since:.0f} s" + (f" ({why})." if why else "."))
            elif w is None and starved:
                starved = False
                print("[watchdog] pictures again", flush=True)
        frozen_dump(False, armed)

    if sink.sinks:
        threading.Thread(target=watchdog, name="watchdog", daemon=True).start()

    ticks = iter(src)  # one stream for every session, so a restart doesn't reconnect or reopen anything
    if sink.sinks:  # playing on screen: keep frames on time when other programs are busy
        from framegen.winutil import boost_gpu, boost_thread, high_res_timer, keep_screen_on
        print(f"[run] scheduling: MMCSS {'on' if boost_thread() else 'off'}, "
              f"GPU priority {'high' if boost_gpu() else 'normal'}, "
              f"1ms timer {'on' if high_res_timer(True) else 'off'}, "
              f"screen kept on {'yes' if keep_screen_on() else 'no'}")  # (a controller isn't input to Windows)
    try:
        while True:
            started.clear()
            started.update(cfg)
            print("[run] playing")  # the launcher's cue that the video is up again after a restart
            if remote is not None:
                remote.publish()  # the restart list starts empty again
            label = label_of(cfg)
            pipe = Pipeline(interp, make_detector(src.info, a.dup_threshold))
            live[:] = [pipe, cfg]
            stats = pipe.run(ticks, sink if sink.sinks else None, score=a.score, label=label,
                             on_tick=getattr(src, "log_tick", None), settings=settings,
                             gt=test.gt if test is not None else getattr(src, "gt", None),
                             remote=src if kind in ("chiaki", "capture") and hasattr(src, "can_press") else None,
                             replay=replay, live=bool(src.info.live))
            live[:] = [None, None]  # first, so a report written meanwhile never counts this session twice
            if a.results:
                done.append((cfg, stats))
            if stats.exit != "restart":
                break
            print(stats.summary())
            reasons = ", ".join(restart_reasons()) or "settings"
            if hasattr(src, "hold"):
                src.hold(True)  # no decoding on the GPU while the pipeline is rebuilt
            del pipe
            interp = None
            gc.collect()
            torch.cuda.empty_cache()
            cfg = session_config()
            print(f"[run] restarting for {reasons}: {describe(cfg)}")
            try:
                cfg, interp = build_or_rife(cfg, lambda fn, text: window.busy(fn, f"Restarting ({reasons})"))
            except Exception as e:  # noqa: BLE001 - e.g. a TensorRT build failure: carry on with PyTorch
                if session_config().get("engine") != "trt":
                    raise
                print(f"[run] TensorRT failed ({str(e)[:300]}); using PyTorch")
                settings.override("engine", "torch")
                cfg = session_config()
                interp = window.busy(lambda: build(cfg), "Restarting with PyTorch")
            if hasattr(src, "hold"):
                src.hold(False)
    finally:
        # The video window goes at once: a window left up while clips finish saving looks frozen (and Windows
        # offers to end it, which stops the save).
        for s in getattr(sink, "sinks", [sink]):
            if hasattr(s, "hide"):
                s.hide()
        if sink.sinks:  # the screen can turn off again, while clips finish saving
            from framegen.winutil import high_res_timer, keep_screen_on
            keep_screen_on(False)
            high_res_timer(False)
        if kind == "chiaki" and hasattr(src, "set_audio_delay"):
            src.set_audio_delay(0)  # chiaki-ng's sound back on time (it would also lapse once we stop)
        if hidden:
            from framegen.winutil import SW_RESTORE, show
            show(hidden, SW_RESTORE)
        if replay is not None:
            replay.close(wait_saves=30)  # a clip being saved is finished first; before & after carries on alone
        sink.close()
        src.close()
        if webrtc_bridge is not None:
            webrtc_bridge.stop()
        stop_results.set()
        if results_thread.is_alive():
            results_thread.join(timeout=5)  # so the last write below isn't racing one already under way
    print(stats.summary())
    if a.results and results_now():
        print(f"[results] test results -> {a.results}")
    write_stream_check(src, a.results)
    if a.save_worst and stats.worst is not None:
        p, tick, *imgs = stats.worst
        tiles = [cv2.resize(im, (im.shape[1] // 2, im.shape[0] // 2), interpolation=cv2.INTER_AREA) for im in imgs]
        for t, name in zip(tiles, ("previous real", f"RIFE ({p:.1f} dB)", "actual in-between", "next real")):
            cv2.putText(t, name, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 0, 0), 5, cv2.LINE_AA)
            cv2.putText(t, name, (15, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)
        from framegen.imagefile import write_image
        write_image(a.save_worst, cv2.vconcat([cv2.hconcat(tiles[:2]), cv2.hconcat(tiles[2:])]))
        print(f"[run] worst frame (tick {tick}) -> {a.save_worst}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
