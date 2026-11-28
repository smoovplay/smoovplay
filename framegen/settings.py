"""User settings: one registry (SPECS) that the in-video menu, the launcher's Settings page, the command line and
the saved file all share.

To add a setting: add a Spec here, then read `settings.<key>` where it matters (or `settings.watch(fn)` to hear
about changes). `live=False` settings are saved at once but take effect at the next in-app restart; the menu
offers "Restart now" when any are pending.

`python -m framegen.settings` prints the registry as JSON for the launcher (specs_json()), so it never keeps a copy.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from . import displays, fill, looks

DEFAULT_PATH = Path(os.environ.get("APPDATA", Path.home())) / "framegen" / "settings.json"


@dataclass(frozen=True)
class Spec:
    key: str
    label: str
    choices: tuple
    default: Any
    names: dict = field(default_factory=dict)  # value -> text shown; missing values are shown as str(value)
    live: bool = True  # applies immediately; False = on the next restart
    help: str = ""
    saved: bool = True  # False: this session only, never written to the settings file (e.g. test modes)
    group: str = ""  # section on the launcher's Settings page; SPECS lists each section's rows together
    quick: bool = False  # one of the few big choices at the top of the launcher's Settings page
    applies: tuple = ()  # (key, values): only used, and only shown, while setting `key` is one of `values`
    later: str = ""  # a note at the end of the launcher's dropdown, about choices that aren't offered yet
    hints: dict = field(default_factory=dict)  # value -> a line under that choice in the launcher's dropdown (only
    # there, so no-break spaces keep "90 Hz" together; the in-video menu's font can't draw them, so help has none)
    impact: dict = field(default_factory=dict)  # "picture" and/or "speed": a rough line on what it does to the picture
    # and to speed or delay, shown in the launcher and the in-video menu (plain ASCII). Numbers are measured (IMPACT_PC).
    experimental: bool = False  # built without the hardware (or a real game) to test it: tagged as Experimental in the
    # launcher (with Send feedback) and in the in-video menu
    sources: tuple = ()  # only used with these sources (framegen/sources.py kinds; () = all). With another one, the
    # launcher (for the source picked on its Play page) and the in-video menu (run.py) grey it out, with its note:
    elsewhere: dict = field(default_factory=dict)  # source kind -> why it isn't used there ("": any other source)

    def name(self, value: Any) -> str:
        return self.names.get(value, str(value))

    def applies_to(self, values: dict) -> bool:
        """Whether this setting is used with these values (e.g. VFG quality only while NVIDIA VFG is picked)."""
        if len(self.choices) <= 1:
            return False
        return not self.applies or values.get(self.applies[0]) in self.applies[1]

    def unused_with(self, source: str) -> str | None:
        """Why this setting isn't used with frames from `source` (a framegen/sources.py kind), or None if it is."""
        if not self.sources or source in self.sources:
            return None
        return self.elsewhere.get(source) or self.elsewhere.get("") or "Not used with this source."


ON_OFF = {True: "On", False: "Off"}
CAPTURE_SIZES = {"1080p": (1920, 1080), "1440p": (2560, 1440), "2160p": (3840, 2160)}  # Capture card resolution
IMPACT_PC = "an RTX 4070 Ti SUPER"  # where the numbers in `impact` and the hints were measured
# scripts/looks_timing.py: GPU time of the built-in looks on a 1440p window (0.02 to 0.34 ms) and at 1080p for clips
# (0.01 to 0.19 ms), measured while chiaki-ng streamed.
LOOK_SPEED = "About 0.02 to 0.35 ms a frame of graphics card time for SmoovPlay's looks (1440p)."
LOOK_CLIPS_SPEED = "Up to about 0.2 ms more a frame of graphics card time, to draw it again at 1080p."

SPECS: tuple[Spec, ...] = (
    Spec("interp", "Frame generation", (True, False), True, ON_OFF, group="Smoothness", quick=True,
         hints={False: "No new frames: the stream exactly as the game sends it, with the least delay."},
         impact={"picture": "Smoother motion. Small, fast details can wobble a little.",
                 "speed": "Adds delay: about 20 ms at 60 fps, about 40 ms at 90 or 120 fps."},
         help="Draws new frames between the ones the game sends, so a 30 fps game moves like 60, 90 or 120 fps. "
              "Off shows the stream exactly as it arrives. In the video window, I turns it off and on for this "
              "session only, to compare."),
    Spec("hz", "Output frame rate", (60, 90, 120), 60, {60: "60 fps", 90: "90 fps", 120: "120 fps"},
         group="Smoothness", quick=True,
         hints={60: "One new frame for each real one. Any supported graphics card, any display. Adds about 20\u00a0ms of delay.",
                90: "Two new frames for each real one. Needs a 90\u00a0Hz or faster display. Adds about 40\u00a0ms of "
                    "delay.",
                120: "Three for each real one. Needs a 120\u00a0Hz display and a fast card. Adds about 40\u00a0ms of "
                     "delay."},
         help="How many frames you see each second. Higher is smoother, but needs a faster graphics card and a "
              "display that runs at least that fast (G-Sync or FreeSync helps). Auto-detect, on SmoovPlay's "
              "Settings page, picks the most this PC keeps up with."),
    # framegen/cadence.py FrameTimeline. Tried on simulated 40-55 fps games (tests/test_timeline.py), not a real one yet.
    Spec("even_out", "Even out 40-60 fps", (True, False), True, ON_OFF, group="Smoothness",
         applies=("interp", (True,)),
         help="For games that run between about 40 and 60 fps, like unlocked performance modes. Their frames "
              "arrive unevenly, some staying on screen twice as long as others, so motion stutters. SmoovPlay "
              "times each one and draws evenly spaced frames from them at 60, 90 or 120 fps. 30 fps and 60 fps "
              "games aren't affected. Experimental: not tried with a real game yet.",
         impact={"picture": "Even motion in 40-60 fps games, instead of stutter.",
                 "speed": "About one frame (17 ms) more delay than showing their frames as they come."}),
    # framegen/cadence.py RhythmLock. Auto reads the game's frame rate from the stream (sizes, the picture), which a
    # still or dark scene can fool (Gotham Knights, locked at 30, read as 38-47 fps on PS5 Remote Play).
    Spec("game_fps", "Game frame rate", ("auto", "30"), "auto", {"auto": "Auto", "30": "30 fps"}, group="Smoothness",
         applies=("interp", (True,)),
         hints={"auto": "SmoovPlay works out the game's frame rate as it plays. The default.",
                "30": "For a game locked at 30\u00a0fps. Every other frame is taken as new, so nothing can misread it."},
         help="Auto works out the game's frame rate from the stream as it plays. A still or dark scene can fool it "
              "into reading a 30 fps game as 40 or more, which makes motion uneven. If you know the game is locked "
              "at 30 fps (most console quality modes), pick 30 fps: every other frame is taken as new, so motion "
              "stays even. Menus that run at 60 fps look a little less smooth with it.",
         impact={"picture": "Even motion in a game locked at 30 fps, whatever the scene.",
                 "speed": "None. At most one frame (17 ms) later for a moment after the game stutters."}),
    # framegen/cadence.py DipWatch + FrameTimeline.at_arrival. Tried on simulated 30 fps games that hold frames for 3 or
    # 4 ticks (tests/test_dips.py, scratchpad numbers in notes/backlog.md), not a real one yet. Off by default: it adds
    # delay while the game dips.
    Spec("even_dips", "Dip smoothing", (False, True), False, ON_OFF, group="Smoothness",
         applies=("interp", (True,)), experimental=True,
         help="For 30 fps games that sometimes drop below 30, as some big open-world games do on a base console. "
              "At each dip a frame stays up longer and the next one jumps. With this on, SmoovPlay spaces the "
              "frames it draws by when the game's frames arrived, so a dip looks like a brief slow-down instead. "
              "It's only on while the game dips, and adds delay then.",
         impact={"picture": "A dip below 30 fps looks like a brief slow-down, not a freeze and a jump.",
                 "speed": "While it dips: 17 to 50 ms more delay, and more graphics card work."}),
    # RIFE: only the model SmoovPlay is tuned and tested with (setup prepares it). interpolator.MODELS has more, for
    # testing with run.py --model; one becomes a choice here once it's tested in real time and setup prepares it.
    # NVIDIA VFG (framegen/vfg.py): offered once the user has set it up on the launcher's Settings page. Its licence
    # forbids publishing benchmark results, so nothing here compares its speed or quality with RIFE's.
    Spec("model", "Interpolation model", ("4.22.lite", "vfg"), "4.22.lite",
         {"4.22.lite": "RIFE 4.22 lite", "vfg": "NVIDIA VFG"}, live=False,
         group="Smoothness", quick=True, later="Support for other interpolation models may come in a later version.",
         hints={"4.22.lite": "Fast and tested. The default.",
                "vfg": "NVIDIA's own frame generation, for RTX 40 and 50 series cards. A separate download "
                       "from NVIDIA."},
         help="The AI model that draws the in-between frames. RIFE 4.22 lite is the one SmoovPlay is tuned and "
              "tested with, fast enough for 90 and 120 fps. NVIDIA VFG (Video Frame Generation) is NVIDIA's own, "
              "for RTX 40 and 50 series cards: SmoovPlay's Settings page downloads it from NVIDIA once you accept "
              "NVIDIA's licence."),
    Spec("vfg_quality", "VFG quality", ("low", "medium", "high"), "medium",
         {"low": "Low", "medium": "Medium", "high": "High"}, live=False, group="Smoothness", applies=("model", ("vfg",)),
         hints={"low": "The lightest on your graphics card.", "medium": "Balanced. The default.",
                "high": "The most careful, and the heaviest."},
         help="How much work NVIDIA VFG puts into each in-between frame. Higher takes more of your graphics "
              "card's time, which can cost in-between frames at 90 and 120 fps. Only used with NVIDIA VFG."),
    # v2 (near edges only, interpolator._steady): on eight 60 fps gameplay clips (scripts/steady_hud_eval.py) it fixes
    # 96% of the visibly wobbling HUD pixels for 0.06 dB over the whole frame, at about 0.5 ms per real frame plus
    # 0.15 per AI frame. On three sections of PS5 GTA V it cost 0.1 to 1.2 dB over the whole frame. Off by default:
    # judged by eye (scripts/steady_hud_compare.py), RIFE alone looked as good.
    # framegen/steady.py, after RIFE or NVIDIA VFG alike (with VFG, each in-between frame is also converted to apply it).
    Spec("steady_hud", "Steady HUD", (False, True), False, ON_OFF, group="Smoothness",
         applies=("model", ("4.22.lite", "vfg")),
         help="Keeps the parts of the picture that don't move, like the map, health bar and subtitles, exactly as "
              "the game drew them, so they don't wobble when the scene behind them moves. Results vary by game, and "
              "often there's no visible difference. If a small, fast object ever flickers, turn it off.",
         impact={"picture": "Steadier HUD in some games; in others, slightly worse elsewhere.",
                 "speed": "About 0.5 ms more work per real frame, and a little more delay."}),
    Spec("engine", "AI engine", ("auto", "trt", "torch"), "auto",
         {"auto": "Auto", "trt": "TensorRT", "torch": "PyTorch"}, live=False, group="Smoothness", applies=("model", ("4.22.lite",)),
         hints={"auto": "The faster one for your settings. Recommended.",
                "trt": "Fastest at full motion detail. Its first start on a graphics card takes a minute or two.",
                "torch": "Starts at once. Fastest at half motion detail."},
         help="The software that runs the AI model on your graphics card. Both draw the same frames; only the "
              "speed differs. Auto uses TensorRT at full motion detail, where it's about 12% faster, and PyTorch "
              "at half, where PyTorch is faster. Only used with RIFE.",
         impact={"picture": "None: both draw the same frames.",
                 "speed": "TensorRT is about 12% faster at full motion detail, PyTorch at half."}),
    Spec("motion", "Motion detail", ("auto", "full", "half"), "auto",
         {"auto": "Auto", "full": "Full", "half": "Half"}, live=False, group="Smoothness", applies=("model", ("4.22.lite",)),
         hints={"auto": "Full; Half at 120 fps, a 1440p capture at 90, or where Full is too slow for this card; less for 4K. Recommended.",
                "full": "Sharpest on fast motion. The slowest.",
                "half": "About 40% faster. A little softer on fast detail."},
         help="How finely RIFE tracks movement. Full follows small, fast details best, like railings or distant "
              "cars; Half takes about 40% less time but softens them a little. Auto uses Half at 120 fps, for a "
              "1440p capture at 90, or where Auto-detect timed Full too slow for your graphics card at the frame "
              "rate you picked; a quarter for a 4K capture; and Full otherwise. Only used with RIFE.",
         impact={"picture": "Half is a little softer on small, fast details.",
                 "speed": "Half takes about 40% less time per AI frame."}),
    # A test mode (framegen/sources.py TestMode), for any source that sends every frame (not App window: run.py).
    # Session only, so the launcher's Settings page leaves it out: its Play page has "Test with a 60 fps game".
    Spec("simulate_30", "Simulate a 30 fps game", (False, True), False, ON_OFF, saved=False, group="Smoothness",
         help="For testing with a 60 fps game: every other frame is left out, so it plays like a 30 fps game. At "
              "60 and 120 fps each AI frame is also scored against the real frame it replaced. For a game locked "
              "at 30 fps, use Game frame rate: 30 fps instead. This session only."),
    Spec("scaler", "Upscaler", ("fsr", "vsr", "bilinear"), "fsr",
         {"fsr": "AMD FSR 1", "vsr": "NVIDIA RTX VSR", "bilinear": "Bilinear"}, group="Picture", quick=True,
         later="Support for other upscalers may come in a later version.",
         hints={"fsr": "Sharp and almost free: about 0.1\u00a0ms a frame. Recommended.",
                "vsr": "NVIDIA's AI upscaler, Experimental. It comes with NVIDIA VFG's download (RTX 40 and 50 "
                       "series); without it, FSR\u00a01 is used. Heavier.",
                "bilinear": "Plain resizing, the softest. For comparison."},
         help="Enlarges the picture from the game (usually 1080p) to fill your window or screen, so it only "
              "matters when that's bigger than 1080p, like a 1440p or 4K display. Key: U.",
         impact={"picture": "Only matters on a display bigger than the stream, like 1440p or 4K.",
                 "speed": "About 0.1 ms a frame for FSR 1. RTX VSR runs an AI model on every frame, which can "
                          "cost in-between frames at 90 and 120 fps."}),
    Spec("fsr_sharpness", "FSR sharpening", (0.0, 0.25, 0.5, 1.0, 2.0), 0.5,
         {0.0: "Max", 0.25: "High", 0.5: "Medium", 1.0: "Low", 2.0: "Minimal"}, group="Picture", applies=("scaler", ("fsr",)),
         hints={0.0: "Crispest. Can make compression blocks stand out.", 0.5: "Balanced. The default.",
                2.0: "Softest. Best for a blocky, low-quality stream."},
         help="How strongly FSR 1 sharpens the enlarged picture. More looks crisper; less keeps the blocky patches "
              "of a weak Wi-Fi stream from standing out. Only used with AMD FSR 1.",
         impact={"picture": "More is crisper; less keeps a blocky stream calmer.", "speed": "No difference."}),
    Spec("vsr_quality", "RTX VSR quality", (1, 2, 3, 4), 2, {1: "Low", 2: "Medium", 3: "High", 4: "Ultra"},
         group="Picture", applies=("scaler", ("vsr",)), experimental=True,
         help="How hard RTX VSR works to clean up and sharpen the picture. Higher looks cleaner but asks more of the "
              "GPU, which can cost in-between frames at 90 and 120 fps. Only used with RTX VSR, which comes with "
              "NVIDIA VFG's download from NVIDIA. On a capture card's uncompressed picture it uses NVIDIA's models "
              "for clean video.",
         impact={"picture": "Higher is cleaner and sharper.",
                 "speed": "Each step up asks more of the GPU."}),
    # framegen/hdr.py. Off by default, and then nothing changes: an HDR stream comes out washed out, and SmoovPlay says
    # so. "hdr10" needs the DXGI presentation (NVIDIA) and Windows' HDR; without them run.py uses "sdr" for the session.
    Spec("hdr", "HDR", ("off", "sdr", "hdr10"), "off",
         {"off": "Off", "sdr": "On, shown in SDR", "hdr10": "On, HDR output"}, live=False,
         group="Picture", experimental=True,
         hints={"off": "For a stream without HDR. The default.",
                "sdr": "Plays an HDR stream on any display, with its colours and brightness fitted to it.",
                "hdr10": "Keeps it HDR on an HDR display, with Use HDR on in Windows' display settings. Not yet "
                         "tried on an HDR screen."},
         help="Experimental: not yet tried with a real PS5 HDR stream, or on an HDR screen. For Remote Play in HDR, "
              "pick the H.265 HDR codec in chiaki-ng, then turn this on. Shown in SDR fits HDR to any display. HDR "
              "output keeps it HDR on a display with Use HDR on in Windows (without one, it's shown in SDR). A "
              "stream without HDR isn't changed. Applies the next time you press Play.",
         impact={"picture": "An HDR stream's true colours, instead of a washed-out picture.",
                 "speed": "About 0.7 ms of graphics card time for each frame from the game."}),
    # run.py asks the card for this size (framegen/capture.py pick_mode: else the largest it has under it). Every new
    # frame is drawn at the size captured, so the cost grows with it: notes in the hints, measured by scripts/benchmark.py.
    Spec("capture_size", "Capture card resolution", tuple(CAPTURE_SIZES), "1080p",
         {"1080p": "1080p", "1440p": "1440p", "2160p": "4K"}, live=False, group="Picture", experimental=True,
         hints={"1080p": "The lightest: 60, 90 or 120 fps on a fast card. Recommended. The default.",
                "1440p": "About twice the work of 1080p: fine at 60 fps on most cards, but 90 can run sluggish. "
                         "Sharper on a 1440p or 4K screen.",
                "2160p": "About four times the work of 1080p: likely to run sluggish, even at 60 fps on a fast card. "
                         "Only for a 4K screen."},
         help="Capture cards only (Remote Play is 1080p at most). Higher is sharper on a bigger screen, but every new "
              "frame is drawn at this size, so it can run quite sluggish. A capture bigger than your screen gains "
              "nothing. The console must send it too. Experimental; applies the next time you press Play.",
         impact={"picture": "Sharper on a 1440p or 4K screen; no gain on a 1080p one.",
                 "speed": "1440p: about twice the work of 1080p. 4K: about four times. Both can run sluggish."},
         sources=("capture",),
         elsewhere={"chiaki": "Only used with a capture card. PS5 Remote Play streams 1080p at most.",
                    "window": "Only used with a capture card. App window capture is scaled to 1080p.",
                    "tap": "Only used with a capture card. The app sends its frames at its own size.",
                    "": "Only used with a capture card."}),
    # framegen/looks.py (the looks; "Which look" lists the built-in ones, then the user's, from looks.CHOICES, which
    # the looks library keeps up to date) and framegen/looks_gl.py (drawing them, after the upscaler).
    Spec("look_on", "Look", (False, True), False, ON_OFF, group="Looks",
         help="Effects on top of the picture, like a color grade, film grain or bloom: one of SmoovPlay's looks, or "
              "your own, made on SmoovPlay's Looks page. K turns the look on and off, J goes to the next one. Saved "
              "clips stay as the game looked unless Looks in clips is on.",
         impact={"picture": "Changes the colors and style of the picture, the way you choose.",
                 "speed": LOOK_SPEED}),
    Spec("look", "Which look", looks.CHOICES, looks.DEFAULT_LOOK, looks.NAMES, group="Looks",
         applies=("look_on", (True,)),
         help="The look to use: SmoovPlay's own, then the ones you made or pasted on the Looks page. J goes to the "
              "next one, Shift+J back."),
    Spec("monitor", "Monitor", displays.CHOICES, displays.DEFAULT_MONITOR, displays.NAMES, live=False,
         group="Display", hints=displays.HINTS,
         help="The screen SmoovPlay opens on. Auto-detect and the HDR check use it too.",
         sources=("chiaki", "capture", "tap", "file", "sim"),
         elsewhere={"window": "App window capture stays on the monitor the app's window is on."}),
    Spec("fullscreen", "Fullscreen", (False, True), False, ON_OFF, group="Display",
         help="Fills the whole screen. It also skips the Windows desktop compositor, which removes about 12 ms "
              "of delay, so it's the best way to play. Key: F.",
         impact={"speed": "About 12 ms less delay than a window, and G-Sync or FreeSync works."}),
    # framegen/fill.py. Consoles send 16:9 only, so on an ultrawide these are ways of laying out the same 16:9 picture,
    # never more of the game: the names, hints and help say so.
    Spec("fill", "Fill the screen", fill.FILLS, "fit",
         {"fit": "Fit", "zoom": "Zoom", "stretch": "Stretch", "panorama": "Panorama"}, group="Display",
         hints={"fit": "All of the picture, true to shape, with black bars on an ultrawide. The default.",
                "zoom": "Fills the screen by cutting off the top and bottom of the picture, which can cut the HUD.",
                "stretch": "Fills the screen by making everything wider.",
                "panorama": "The middle stays close to true shape and the sides take the stretch, like a TV's wide "
                            "zoom. Ultrawide only."},
         help="For an ultrawide screen. Consoles send 16:9 only, so none of these is real ultrawide: there's no more "
              "of the game to show. They crop or stretch the same 16:9 picture. Key: Z.",
         impact={"picture": "The same 16:9 picture, cropped or stretched: consoles send 16:9 only.",
                 "speed": "No real difference. Panorama adds one pass over the screen: about 0.02 ms."}),
    Spec("vsync", "VSync", (False, True), False, ON_OFF, group="Display",
         help="Off: the lowest delay, but fast camera turns can show a tear line across the picture. On: no "
              "tearing, with a little more delay. With a G-Sync or FreeSync display, On gives you both: no tearing "
              "and low delay.",
         impact={"picture": "On: no tear lines.", "speed": "On can add up to one refresh of delay: 8 ms at 120 Hz, 17 at 60."}),
    # framegen/dxgi_present.py. Standard stays the default until Low delay has been measured on more PCs.
    Spec("present", "Window presentation", ("gl", "dxgi"), "gl", {"gl": "Standard", "dxgi": "Low delay (Experimental)"},
         live=False, group="Display", applies=("fullscreen", (False,)),
         hints={"gl": "OpenGL's own way. Tested. In a window, Windows adds a copy of each frame.",
                "dxgi": "Hands frames over the way games do, so a window is about as quick as fullscreen and G-Sync "
                        "works. Experimental."},
         help="How the video reaches the screen. In a window, Standard goes through the Windows compositor, which "
              "adds about 12 ms and keeps G-Sync and FreeSync from working. Low delay hands frames over the way games "
              "do, for playing in a window; fullscreen uses Standard, as quick there. With Windows' HDR on, it's always "
              "Low delay, so fullscreen keeps HDR on. Experimental; applies the next time you press Play.",
         impact={"speed": "Low delay should save the 12 ms a window adds. Still being measured."}),
    Spec("overlay", "Stats overlay", (True, False), True, ON_OFF, group="Display",
         help="Numbers in a corner of the video: how many frames a second are on screen, the game's own frame rate, "
              "and whether your PC keeps up. Key: S (Off, Minimal, Detailed)."),
    Spec("overlay_style", "Stats overlay style", ("minimal", "detailed"), "minimal",
         {"minimal": "Minimal", "detailed": "Detailed"}, group="Display", applies=("overlay", (True,)),
         hints={"minimal": "A small line: frames on screen, the game's frame rate, the delay, and a dot for how well "
                           "your PC keeps up.",
                "detailed": "Everything: AI frame time, the upscaler, and at 90 and 120 fps the share of extra frames "
                            "shown."},
         help="Minimal: frames on screen, the game's frame rate (amber when it isn't 30 fps) and the delay, with a "
              "dot: green, your PC keeps up; amber, it's tight; red, it's falling behind, so try a lower frame rate. "
              "Detailed adds the AI frame time, the upscaler and, at 90 and 120 fps, the extra frames shown (below "
              "about 95%, your PC is falling behind)."),
    Spec("overlay_corner", "Stats overlay position", ("top_left", "top_right", "bottom_left", "bottom_right"),
         "top_left", {"top_left": "Top left", "top_right": "Top right", "bottom_left": "Bottom left",
                      "bottom_right": "Bottom right"}, group="Display", applies=("overlay", (True,)),
         help="The corner the stats go in, tucked into its edges. Pick one your game's own HUD leaves free."),
    Spec("overlay_size", "Stats overlay size", ("small", "medium", "large"), "small",
         {"small": "Small", "medium": "Medium", "large": "Large"}, group="Display", applies=("overlay", (True,)),
         hints={"small": "Out of the way of the game's own HUD.", "medium": "Easier to read from further away.",
                "large": "For a TV across the room."},
         help="How big the stats are. Small stays out of the way of the game's own HUD; Large is for a TV across the "
              "room."),
    # framegen/replay.py. Memory in the hints: replay.memory_mb() for 1080p at 90 fps (tests/test_replay.py checks).
    Spec("replay", "Instant replay", (0, 30, 60, 120, 300), 0,
         {0: "Off", 30: "Last 30 seconds", 60: "Last minute", 120: "Last 2 minutes", 300: "Last 5 minutes"},
         group="Instant replay",
         hints={0: "Nothing is kept.", 30: "Keeps about 160 MB in memory (1080p at 90 fps).",
                60: "Keeps about 310 MB in memory (1080p at 90 fps).",
                120: "Keeps about 610 MB in memory (1080p at 90 fps).",
                300: "Keeps about 1.5 GB in memory (1080p at 90 fps)."},
         help="Keeps the last stretch of what you played, so you can save it after something good happens: press R "
              "in the video window, or Ctrl+Alt+R from anywhere. Clips go to your Videos folder, in SmoovPlay, as "
              "MP4 at the frame rate you see, with the game's own sound (PS5 Remote Play and capture cards; window "
              "capture has none). B saves a before & after video instead.",
         impact={"picture": "None.",
                 "speed": "About 0.3 ms of graphics card time a frame; the encoding is on its video encoder, not "
                          "the AI's part."}),
    Spec("replay_save", "Saved clip", ("smoovplay", "compare", "both"), "smoovplay",
         {"smoovplay": "SmoovPlay video", "compare": "Before & after", "both": "Both"},
         group="Instant replay", applies=("replay", (30, 60, 120, 300)),
         hints={"smoovplay": "What you saw, with the game's sound. Saved in a second or two.",
                "compare": "The game's own frames next to SmoovPlay's, in one video. Made in the background, in "
                           "about as long as the clip.",
                "both": "Both videos."},
         help="What R saves. SmoovPlay video is what you saw, at 60, 90 or 120 fps. Before & after puts the game's "
              "own frames next to SmoovPlay's in one video, to show the difference; it's made in the background "
              "and takes about as long as the clip. B always saves a before & after video."),
    Spec("replay_layout", "Before & after layout", ("split", "side"), "split",
         {"split": "Split screen", "side": "Side by side"}, group="Instant replay",
         applies=("replay", (30, 60, 120, 300)),
         hints={"split": "One picture: the left half as the game sent it, the right half from SmoovPlay.",
                "side": "Both whole pictures next to each other, in a video twice as wide."},
         help="How a before & after video is laid out. Split screen is one picture: the left half as the game sent "
              "it, the right half from SmoovPlay, with a line between. Side by side shows both whole pictures, in "
              "a video twice as wide."),
    Spec("look_clips", "Looks in clips", (False, True), False, ON_OFF, group="Instant replay",
         applies=("replay", (30, 60, 120, 300)),
         help="Puts the look you're using into saved clips too. Off keeps clips as the game looked, whatever look "
              "you play with. In a before & after video, only the SmoovPlay side gets the look.",
         impact={"picture": "Clips look like what you see, look included.", "speed": LOOK_CLIPS_SPEED}),
    Spec("minimize_chiaki", "Minimise chiaki-ng", (True, False), True, ON_OFF, group="PS5 Remote Play",
         help="Minimises chiaki-ng's own window while SmoovPlay's video is open, so your graphics card doesn't "
              "spend time drawing a copy nobody sees. chiaki-ng keeps streaming, playing sound and reading your "
              "controller, and its window comes back when you stop.",
         impact={"speed": "Saves the time your graphics card spends drawing chiaki-ng's window."},
         sources=("chiaki",), elsewhere={"": "Only used with PS5 Remote Play."}),
    Spec("audio_sync", "Sound in sync", (True, False), True, ON_OFF, group="PS5 Remote Play",
         help="The sound (chiaki-ng's, or a capture card's, which SmoovPlay plays) would come as it arrives, while "
              "the picture spends a little longer in SmoovPlay. On holds the sound back by that time (about 30 ms "
              "at 90 fps, measured as you play), so they line up. Press L in the video window for a delay test: it "
              "times the whole trip, from a button press to SmoovPlay's screen (PS5 Remote Play).",
         impact={"speed": "None: only the sound waits, by as long as the picture does."}),
)
BY_KEY = {s.key: s for s in SPECS}


class Settings:
    """Current values, saved to JSON on every change. Values given on the command line (`override`) apply to
    this run only and aren't saved unless changed in the menu."""

    def __init__(self, values: dict | None = None, path: Path | None = None) -> None:
        self.path = path
        self._listeners: list[Callable[[str], None]] = []
        self._v = {s.key: s.default for s in SPECS}
        self._saved = dict(self._v)  # what the file holds (or would)
        for k, v in (values or {}).items():
            if k in BY_KEY and BY_KEY[k].saved and v in BY_KEY[k].choices:
                self._v[k] = self._saved[k] = v

    @classmethod
    def load(cls, path: Path = DEFAULT_PATH) -> "Settings":
        try:
            values = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            values = {}
        return cls(values if isinstance(values, dict) else {}, path)

    def __getattr__(self, key: str) -> Any:
        try:
            return self.__dict__["_v"][key]
        except KeyError:
            raise AttributeError(key) from None

    def values(self) -> dict:
        return dict(self._v)

    def watch(self, fn: Callable[[str], None]) -> Callable[[], None]:
        """fn(key) is called after any value changes. Returns a function that stops it."""
        self._listeners.append(fn)
        return lambda: self._listeners.remove(fn) if fn in self._listeners else None

    def override(self, key: str, value: Any) -> None:
        """This run only (command-line flags, the I key)."""
        self._put(key, value, save=False)

    def set(self, key: str, value: Any) -> None:
        self._put(key, value, save=True)

    def _put(self, key: str, value: Any, save: bool) -> None:
        if value not in BY_KEY[key].choices:
            raise ValueError(f"{key}: {value!r} is not one of {BY_KEY[key].choices}")
        changed = self._v[key] != value
        self._v[key] = value
        if save and BY_KEY[key].saved:
            self._saved[key] = value
            self.save()
        if changed:
            for fn in list(self._listeners):
                fn(key)

    def step(self, key: str, delta: int, allowed=None) -> Any:
        """Move to the next/previous choice (wrapping), skipping values `allowed(key, value)` rejects."""
        choices = BY_KEY[key].choices
        i = choices.index(self._v[key])
        for _ in range(len(choices)):
            i = (i + delta) % len(choices)
            if allowed is None or allowed(key, choices[i]):
                self.set(key, choices[i])
                break
        return self._v[key]

    def save(self) -> None:
        if self.path is None:
            return
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(self._saved, indent=2), encoding="utf-8")
        except OSError as e:
            print(f"[settings] couldn't save {self.path}: {e}")


def specs_json(specs: tuple[Spec, ...] = SPECS) -> dict:
    """The registry for the launcher's Settings page: every setting with its choices in display order, and where the
    saved file is. Values are JSON types; the launcher sends a choice back exactly as given here."""
    return {
        "path": str(DEFAULT_PATH),
        "impact_pc": IMPACT_PC,
        "specs": [{"key": s.key, "label": s.label, "group": s.group, "help": s.help, "default": s.default,
                   "live": s.live, "saved": s.saved, "later": s.later, "quick": s.quick, "impact": s.impact,
                   "experimental": s.experimental,
                   "sources": list(s.sources) or None,
                   "elsewhere": s.elsewhere or None,
                   "applies": {"key": s.applies[0], "values": list(s.applies[1])} if s.applies else None,
                   "choices": [{"value": v, "name": s.name(v), "hint": s.hints.get(v, "")} for v in s.choices]}
                  for s in specs],
    }


if __name__ == "__main__":
    print(json.dumps(specs_json()))
