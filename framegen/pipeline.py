"""Source -> duplicate detection -> scheduler -> RIFE or NVIDIA VFG -> sink, with latency/quality stats."""

from __future__ import annotations

import time
from collections import Counter, deque
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch

from . import debugsnap, delaytest
from .cadence import Action, CadenceScheduler, CutDetector, DipWatch, DuplicateDetector, FrameTimeline, RhythmLock
from .hdr import level8, to_u8
from .interpolator import STEADY_HUD_THRESHOLD, RifeInterpolator
from .io import TICK, MiniText, Waiting


# --score's "still areas": pixels within this many levels (of 255) in both real frames, like a HUD or map.
STILL_LEVELS = 3
# --score's "HUD-like areas": still in both real frames *and* the real in-between one, on an edge (Sobel L1 of the
# real frame's brightness above HUD_EDGE, about a 32-level step), so text, icons and map lines rather than flat
# sky. A pixel there is "visibly off" when the AI frame misses the real one by more than HUD_OFF_LEVELS: the
# wobble a viewer sees, which a whole-frame score dilutes.
HUD_EDGE = 128.0
HUD_OFF_LEVELS = 12
# Test mode's quality, for people: the share of a frame's pixels visibly off from the real frame (more than
# HUD_OFF_LEVELS of 255 in any colour, about 5% brighter or darker), lower is better. PSNR (dB) stays in the log.
VISIBLE_LEVELS = HUD_OFF_LEVELS


def pct(share: float) -> str:
    """0.062 -> "6.2%", 0.12 -> "12%": a share as people read it."""
    v = 100 * share
    return f"{v:.1f}%" if v < 10 else f"{v:.0f}%"


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return _db((a.float() - b.float()).pow_(2).mean().item())


_SOBEL: dict = {}


def hud_mask(a: torch.Tensor, b: torch.Tensor, gt: torch.Tensor, still: torch.Tensor | None = None) -> torch.Tensor:
    """HUD-like pixels for scoring (HxWx1 float 0/1): unchanged within STILL_LEVELS across the real frames a, gt
    (the real in-between one) and b, all HxWx3 BGR 0..255, and on an edge of gt's brightness (see HUD_EDGE)."""
    a, b, gt = a.float(), b.float(), gt.float()
    if still is None:
        still = ((b - a).abs().amax(-1, keepdim=True) <= STILL_LEVELS).float()
    static = still * ((gt - a).abs().amax(-1, keepdim=True) <= STILL_LEVELS).float()
    k = _SOBEL.get(gt.device)
    if k is None:
        gx = torch.tensor([[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]], device=gt.device)
        k = _SOBEL[gt.device] = torch.stack([gx, gx.t()]).unsqueeze(1)  # (2, 1, 3, 3)
    luma = (gt * torch.tensor([0.114, 0.587, 0.299], device=gt.device)).sum(-1)[None, None]
    g = torch.nn.functional.conv2d(torch.nn.functional.pad(luma, (1, 1, 1, 1), mode="replicate"), k)
    edge = (g.abs().sum(1)[0] > HUD_EDGE).float().unsqueeze(-1)
    return static * edge


def _np(frame) -> np.ndarray:
    """A private numpy copy of a frame (numpy array or CUDA tensor), as 8 bits (an HDR frame's 8-bit stand-in)."""
    if torch.is_tensor(frame):
        return to_u8(frame).cpu().numpy() if frame.is_floating_point() else frame.cpu().numpy()
    return frame.copy()


def _db(mse: float) -> float:
    return float("inf") if mse == 0 else 10.0 * np.log10(255.0**2 / mse)


class GameRate:
    """The game's own frame rate: new frames among the last `n` ticks of a 60 Hz stream. Counted by ticks rather
    than by the clock, so a file read faster than real time reads the same. None until `n` ticks are in."""

    def __init__(self, n: int = 120) -> None:
        self._new: deque = deque(maxlen=n)

    def add(self, new: bool | None) -> None:
        """One tick: new content (True) or a repeat (False). None (the source can't say) is left out."""
        if new is not None:
            self._new.append(bool(new))

    def clear(self) -> None:
        self._new.clear()

    def fps(self) -> float | None:
        n = len(self._new)
        return sum(self._new) / n / TICK if n and n == self._new.maxlen else None


class ShownRate:
    """Frames put on screen per second over the last `window` seconds: what a GPU overlay calls the frame rate."""

    def __init__(self, window: float = 1.0) -> None:
        self.window, self._t, self._start = window, deque(), None

    def add(self, t: float) -> None:
        self._start = t if self._start is None else self._start
        self._t.append(t)
        while self._t[0] <= t - self.window:
            self._t.popleft()

    def fps(self) -> float | None:
        if self._start is None or self._t[-1] - self._start < self.window:
            return None
        return len(self._t) / self.window


# How long the game has to stay off the rate that's needed before the overlay says so, and before the one-off message
RATE_NOTE_S, RATE_TOAST_S = 3.0, 10.0
# Said when "Simulate a 30 fps game" goes off by itself: the game was at 30 fps already, and Simulate halved it
SIMULATE_OFF = "Simulate turned off: this game is at about {g} fps already, and Simulate halved that"


def rate_advice(game_fps: float | None, test: bool, even: bool = False) -> str:
    """One line for someone whose game isn't at the rate that's needed, or "" when it is (or it isn't known yet).
    test: "Simulate a 30 fps game" is on, which needs a game at a steady 60 fps. Otherwise frame generation, which
    fills a 30 fps game's gaps: a faster game's frames mostly pass through as they are, unless `even` (Even out
    40-60 fps is on), which spaces a 40-60 fps game's frames evenly."""
    if game_fps is None:
        return ""
    g = round(game_fps)
    if test:
        if g >= 50:
            return ""
        if g <= 38:
            return f"Simulate is for 60 fps games. This one is at {g} fps already: turn Simulate off (menu: M)"
        return f"Simulate needs a steady 60 fps game. This one is at about {g}, so quality isn't measured"
    if g <= 38 or (even and g <= 56):
        return ""
    if g >= 55:
        return f"Game at {g} fps: no gaps to fill. Frame generation is for 30 fps modes (Quality/Fidelity)"
    return f"Game at about {g} fps, not 30: few gaps to fill. Its 30 fps mode (Quality/Fidelity) is smoother"


@dataclass
class Stats:
    proc_interp_ms: list[float] = field(default_factory=list)  # tick processing when we interpolate
    proc_other_ms: list[float] = field(default_factory=list)  # tick processing otherwise
    sink_ms: list[float] = field(default_factory=list)  # time spent in display/encode
    queue_ms: list[float] = field(default_factory=list)  # arrival -> processing starts (backlog)
    score_ms: list[float] = field(default_factory=list)  # --score, done after the frame is shown
    busy_ms: list[float] = field(default_factory=list)  # processing + display per tick
    # Arrival of each real (new) frame -> it is first presented. Passthrough would be ~upload+present;
    # interpolation adds one 60 Hz tick on top, because the midpoint must go out first.
    real_delay_ms: list[float] = field(default_factory=list)
    actions: Counter = field(default_factory=Counter)
    dedup: Counter = field(default_factory=Counter)  # tp/tn/fp/fn vs ground truth
    psnr_interp: list[float] = field(default_factory=list)
    psnr_repeat: list[float] = field(default_factory=list)  # what you'd see without interpolation
    psnr_blend: list[float] = field(default_factory=list)  # naive 50/50 crossfade
    # Share of each scored frame's pixels visibly off from the real one (see VISIBLE_LEVELS): the AI frame, and the
    # last real frame repeated instead (what you'd see without frame generation).
    off_interp: list[float] = field(default_factory=list)
    off_repeat: list[float] = field(default_factory=list)
    # The same, only over pixels that are (almost) unchanged between the two real frames: HUD, maps, still
    # backgrounds. Where Steady HUD acts; for tuning it (--score), not shown to users.
    psnr_still: list[float] = field(default_factory=list)
    psnr_still_repeat: list[float] = field(default_factory=list)
    still_frac: list[float] = field(default_factory=list)
    psnr_hud: list[float] = field(default_factory=list)  # HUD-like areas (see HUD_EDGE)
    hud_off: list[float] = field(default_factory=list)  # share of them visibly off (see HUD_OFF_LEVELS)
    hud_frac: list[float] = field(default_factory=list)
    worst: tuple | None = None  # (psnr, tick, prev, interp, gt, next)
    quarter_shown: int = 0  # 90/120 Hz: extra in-between frames shown (1/4 and 3/4, or 1/3)
    quarter_skipped: int = 0  # ones skipped because the pipeline was behind (shown at 60 Hz instead)
    behind: int = 0  # ticks whose frame waited for the pipeline to finish earlier ones (more than 60% of a tick)
    hz_used: set = field(default_factory=set)  # output rates used during the run
    simulated: int = 0  # ticks played with "Simulate a 30 fps game" on
    # Frames played in test mode, by the source's word on whether the game itself drew each one new or repeated one.
    # Test mode needs a game at a steady 60 fps, with next to no repeats; only frames with new ones on both sides
    # are scored (TestMode.gt).
    test_game_new: int = 0
    test_game_repeat: int = 0
    delay_tests: list = field(default_factory=list)  # the delay test's results (L), a line each
    # Even out 40-60 fps (FrameTimeline): ticks played that way, its game's frame rate, and output frames that showed
    # the newest frame again because the next one came late.
    uneven_ticks: int = 0
    uneven_fps: list = field(default_factory=list)
    uneven_waits: int = 0
    dip_ticks: int = 0  # Dip smoothing: ticks played with frames timed as they arrived (waits count above)
    # Time between the game's new frames as they reached SmoovPlay (stamped as they came off the stream, card or
    # window, before any processing): late frames and bunches point at the network or the capture (arrival_line)
    new_gaps_ms: list = field(default_factory=list)
    wall_s: float = 0.0
    exit: str = "end"  # why run() returned: "end" (source finished), "quit" or "restart" (asked by the sink)
    stale: int = 0  # frames skipped because they arrived before run() started (queued during a restart)
    # The engine's numbers mustn't be shared (NVIDIA VFG: its licence forbids publishing benchmark results), and the
    # log goes into Copy diagnostics, so summary() leaves out timings and scores. The overlay and test mode's report
    # (results(); Copy diagnostics leaves it out, see run.write_results) still show them on this PC.
    private: bool = False
    PRIVATE_NOTE = ("NVIDIA VFG: timings and quality scores are left out of the log, because NVIDIA's licence doesn't "
                    "allow sharing them and the log goes into Copy diagnostics. The stats overlay (S) and test mode's "
                    "report show them on this PC.")

    def arrival_line(self) -> str:
        """How evenly the game's new frames reached SmoovPlay, or "" for a short session. A steady stream has its
        frames the same time apart (33 ms for a 30 fps game); Wi-Fi or a busy capture shows as late frames and
        bunches (two close together after a late one), which look like stutter whatever the PC does."""
        gaps = self.new_gaps_ms
        if len(gaps) < 20:
            return ""
        usual = float(np.median(gaps))
        late = sum(g > 1.5 * usual for g in gaps) / len(gaps)
        bunched = sum(g < 0.5 * usual for g in gaps) / len(gaps)
        return (f"game frames as they arrived: {len(gaps) + 1} new, usually {usual:.1f} ms apart; "
                f"{100 * late:.1f}% late (over 1.5x that), {100 * bunched:.1f}% bunched (under half); "
                f"longest gap {max(gaps):.0f} ms")

    def summary(self) -> str:
        def pct(xs, p):
            return float(np.percentile(xs, p)) if xs else float("nan")

        n = len(self.proc_interp_ms) + len(self.proc_other_ms)
        lines = [
            f"ticks={n}  wall={self.wall_s:.1f}s",
            f"actions: {dict(self.actions)}",
        ]
        if arrival := self.arrival_line():  # (the game's frames, not the engine's work: in a private log too)
            lines.append(arrival)
        tg = self.test_game_new + self.test_game_repeat
        if tg:
            lines.append(f"test mode: {self.test_game_repeat} of {tg} frames were repeats in the game itself (a steady "
                         "60 fps game has next to none); only frames with new ones on both sides are scored")
        if self.uneven_ticks:
            lines.append(f"even timing (a game at about {np.median(self.uneven_fps):.0f} fps): {self.uneven_ticks} "
                         f"ticks, {self.uneven_waits} output frames waited for a late frame")
        if self.dip_ticks:
            lines.append(f"dip timing (a 30 fps game dipping below it): {self.dip_ticks} ticks")
        if self.private:
            if self.dedup:
                lines.append(f"dedup vs truth: {dict(self.dedup)}")
            return "\n".join(lines + [self.PRIVATE_NOTE])
        lines += self.delay_tests
        lines[0] += f"  ({n / max(self.wall_s, 1e-9):.1f} ticks/s)"
        rows = (("processing, interp ticks", self.proc_interp_ms), ("processing, other ticks", self.proc_other_ms),
                ("display/encode", self.sink_ms), ("scoring, after display", self.score_ms),
                ("queueing (backlog)", self.queue_ms),
                ("real frame arrival->presented", self.real_delay_ms))
        for name, xs in rows:
            if xs:
                lines.append("{:<30} ms  p50={:6.2f}  p95={:6.2f}  p99={:6.2f}  max={:6.2f}".format(
                    name, pct(xs, 50), pct(xs, 95), pct(xs, 99), max(xs)))
        if self.stale:
            lines.append(f"stale frames skipped at start (queued before this run): {self.stale}")
        lines.append(f"ticks over the {TICK * 1000:.1f} ms budget (processing + display): "
                     f"{sum(b > TICK * 1000 for b in self.busy_ms)}")
        if self.quarter_shown or self.quarter_skipped:
            due = self.quarter_shown + self.quarter_skipped
            lines.append(f"extra in-between frames (90/120 Hz) shown {self.quarter_shown} of {due} "
                         f"({100 * self.quarter_shown / due:.0f}%); the rest fell back to 60 Hz because it was behind")
        if self.dedup:
            lines.append(f"dedup vs truth: {dict(self.dedup)}")
        if self.psnr_interp:
            fin = lambda xs: [x for x in xs if np.isfinite(x)]  # noqa: E731
            lines.append(
                "PSNR vs real in-between frame (dB, higher=better): RIFE {:.2f} | blend {:.2f} | repeat {:.2f}"
                "  (n={}, RIFE worst {:.2f})".format(
                    np.mean(fin(self.psnr_interp)), np.mean(fin(self.psnr_blend)),
                    np.mean(fin(self.psnr_repeat)), len(self.psnr_interp), min(self.psnr_interp)))
        if self.off_interp:
            lines.append("visibly off (more than {} levels): RIFE {:.2f}% | repeat {:.2f}%  (worst {:.2f}%)".format(
                VISIBLE_LEVELS, 100 * np.mean(self.off_interp), 100 * np.mean(self.off_repeat),
                100 * max(self.off_interp)))
        if self.psnr_still:
            fin = lambda xs: [x for x in xs if np.isfinite(x)]  # noqa: E731
            lines.append("  still areas ({:.0f}% of pixels): RIFE {:.2f} | repeat {:.2f}".format(
                100 * np.mean(self.still_frac), np.mean(fin(self.psnr_still)), np.mean(fin(self.psnr_still_repeat))))
        if self.psnr_hud:
            fin = lambda xs: [x for x in xs if np.isfinite(x)]  # noqa: E731
            lines.append("  HUD-like areas ({:.1f}% of pixels): RIFE {:.2f} dB, {:.2f}% visibly off".format(
                100 * np.mean(self.hud_frac), np.mean(fin(self.psnr_hud)) if fin(self.psnr_hud) else float("inf"),
                100 * np.mean(self.hud_off)))
        return "\n".join(lines)

    def results(self, full_motion: bool | None = None) -> list[str]:
        """Test mode's plain-language report: can this PC keep up, and how close are the AI frames to the real
        ones. Safe to call from another thread while run() is still adding to the lists. full_motion: whether
        motion ran at full resolution (None: unknown), so the advice doesn't suggest what's already on."""
        budget = TICK * 1000
        interp, busy, delay = list(self.proc_interp_ms), list(self.busy_ms), list(self.real_delay_ms)
        ai, repeat = list(self.off_interp), list(self.off_repeat)
        rates = sorted(set(self.hz_used)) or [60]
        lines = ["Output: " + " and ".join(f"{h} fps" for h in rates)]
        if interp:
            lines.append(f"AI frame time: {np.mean(interp):.1f} ms on average, {np.percentile(interp, 95):.1f} ms "
                         f"for the slowest 5% (a 60 fps frame lasts {budget:.1f} ms)")
        due = self.quarter_shown + self.quarter_skipped
        # 90/120 fps: behind, not "took over 16.7 ms", as the pacer sleeps between a tick's two frames on purpose.
        # 60 fps: took over 16.7 ms to make (before the screen), not behind. With VSync each frame waits for the
        # screen's refresh, up to a whole frame however fast the PC is, so "behind" called fast PCs slow.
        n, behind = len(busy), self.behind
        made = interp + list(self.proc_other_ms)
        slow = sum(ms > budget for ms in made)
        if due and n:
            lines.append(f"Running behind: {100 * behind / n:.1f}% of frames ({behind} of {n}) arrived while the PC "
                         "was still busy with earlier ones")
        elif made:
            lines.append(f"Slow frames: {100 * slow / len(made):.1f}% ({slow} of {len(made)}) took longer to make "
                         "than a 60 fps frame lasts")
        if due:
            lines.append(f"Extra frames for 90/120 fps: {100 * self.quarter_shown / due:.0f}% shown; the rest fell "
                         "back to 60 fps because the PC was behind")
        if delay:
            lines.append(f"Delay: a real frame reached the screen {np.median(delay):.0f} ms after it arrived (median)")
        # Test mode needs a game at a steady 60 fps: where the game itself repeated a frame, there's no real
        # in-between frame, so those aren't scored (TestMode.gt), and it's said when the game wasn't at 60.
        tg = self.test_game_new + self.test_game_repeat
        not_60 = tg >= 120 and self.test_game_repeat / tg > 0.1
        if not_60:
            fps = 60 * self.test_game_new / tg
            lines.append(f"Game: it ran at about {fps:.0f} fps, not a steady 60. Simulate a 30 fps game is for 60 fps "
                         "games; a 30 fps game doesn't need it, as frame generation works on it as it is")
        # At 90 fps only the odd 60 fps fallback (a busy moment) has a real frame to compare with, and a game that
        # isn't at 60 leaves few fair frames: a handful of those says nothing, so it takes 5 s of them.
        only_90 = bool(self.hz_used) and self.hz_used <= {90}
        if len(ai) >= (300 if not_60 or only_90 else 1) and repeat:
            lines.append(f"Quality: {pct(np.mean(ai))} of each AI frame was visibly off from the real frame it stood "
                         f"in for, against {pct(np.mean(repeat))} when the last frame is repeated instead (no AI). "
                         "Lower is closer to the real thing; visibly off means more than about 5% brighter or darker. "
                         f"Slow scenes score well either way; the gap shows in motion ({len(ai)} frames measured, "
                         f"worst {pct(max(ai))}" + ("; only where the game drew every frame)" if not_60 else ")"))
        elif not_60:
            lines.append("Quality: not measured, as the game wasn't at a steady 60 fps")
        elif interp:
            lines.append("Quality: not measured yet (it's measured at 60 and 120 fps output; 90 fps has no real "
                         "frame to compare with)")
        if self.uneven_ticks:  # (no figure: its median can round to 60, which read "about 60 fps, between 30 and 60")
            lines.append(f"Game: for {self.uneven_ticks / 60:.0f} s it ran between 30 and 60 fps, and its frames were "
                         "evened out (Even out 40-60 fps)")
        if self.dip_ticks:
            lines.append(f"Game: for {self.dip_ticks / 60:.0f} s it dipped below 30 fps, and its frames were spaced by "
                         "when they arrived (Dip smoothing)")
        lines += list(self.delay_tests)
        if not interp:
            lines.append("Verdict: frame generation was off, so there's nothing to judge")
        else:
            late, shown = slow / max(len(made), 1), self.quarter_shown / due if due else 1.0
            top = f"{max(rates)} fps"
            lighter = (["a lower output rate"] if max(rates) > 60 else []) + (
                ["Motion detail: Half"] if full_motion is not False else [])
            either = " or ".join(lighter)
            # At 90 and 120 fps the extra frames say it: a tick that arrives while the PC is still busy only costs
            # something when it has to skip one (a repeat tick waiting a moment loses nothing on screen). At 60 fps
            # there are no extra frames, so frames that took longer to make than a frame lasts are what's left.
            ok, mostly = (shown >= 0.95, shown >= 0.8) if due else (late < 0.01, late < 0.05)
            if ok:
                lines.append(f"Verdict: this PC keeps up at {top}")
            elif mostly:
                lines.append(f"Verdict: this PC mostly keeps up at {top}, but falls behind now and then"
                             + (f". {either[:1].upper()}{either[1:]} would smooth that out" if lighter else ""))
            else:
                lines.append(f"Verdict: this PC is falling behind at {top}"
                             + (f". Try {either}" if lighter else ", even on the lightest settings"))
        return lines


class Pipeline:
    def __init__(self, interp: RifeInterpolator, detector: DuplicateDetector | None = None,
                 cut_threshold: float = 25.0) -> None:
        self.interp = interp
        self.detector = detector or DuplicateDetector()
        self.cut_threshold = cut_threshold  # see CutDetector
        h, w = interp.h, interp.w
        pin = torch.cuda.is_available()  # (pinned memory needs CUDA; DirectML's frames are plain CPU tensors)
        dt = getattr(interp, "frame_dtype", torch.uint8)  # float16 with HDR output (framegen/hdr.py)
        self._in_pinned = torch.empty((h, w, 3), dtype=dt, pin_memory=pin)
        self._out_pinned = torch.empty((h, w, 3), dtype=dt, pin_memory=pin)
        self._in_np = self._in_pinned.numpy()
        self._out_np = self._out_pinned.numpy()
        self.stats: Stats | None = None  # the running session's, for reports written while it plays
        self.beat = time.perf_counter()  # when run()'s loop last went round (run.py's watchdog reads it)
        self.debug_dir: Path | None = None  # where Debug snapshots go (framegen/debugsnap.py); run.py sets it
        self.waiting = None  # the source's Waiting while no frame is coming, else None (the watchdog reads it too)

    def run(self, source, sink=None, enabled: bool = True, score: bool = False, label: str = "",
            on_tick=None, hz: int = 60, settings=None, gt=None, remote=None, replay=None) -> Stats:
        """on_tick(tick, diff, action, proc_ms, t_start), if given, is called after each tick is processed.
        remote: the source, when it can press a button for the delay test (L: PS5 Remote Play's chiaki-ng,
        framegen.chiaki_tap.ChiakiSource) or hold its sound back to meet the picture (settings.audio_sync: chiaki-ng,
        or a capture card's sound, framegen.capture.CardSource); None for other sources.
        gt(content) -> the real frame that was left out, for scoring; default `source.gt` (pass it when `source`
        is an iterator over the real source, which doesn't have it).
        settings (framegen.settings.Settings): if given, its `interp` and `hz` are followed live instead of
        `enabled` and `hz`.
        replay (framegen.replay.Recorder): instant replay. It gets every frame shown, each new frame from the game
        and the pipeline's delay, and the window's R and B keys save it.

        hz=120 or 90 (needs a sink): more generated frames per 30 fps frame, shown at least 1/hz apart.
          120: a new frame B after A shows A->B at 1/4 then 1/2; its repeat tick shows 3/4 then B.
           90: the new frame's tick shows 1/3 then 2/3; its repeat tick shows B.
        When the pipeline is running behind, a tick skips the extra frame(s) and shows what 60 Hz would (the 1/2
        frame, or B), so it never builds a backlog.
        """
        if settings is not None:
            enabled, hz = settings.interp, settings.hz if sink is not None else 60
        it, sched, st = self.interp, CadenceScheduler(), Stats(private=bool(getattr(self.interp, "private", False)))
        steady = None  # Steady HUD as last applied to the interpolator (followed live from settings)
        real_frames = 0  # new frames prepared while Debug snapshots is on (framegen/debugsnap.py)
        self.stats = st
        quarter_due = False  # 120 Hz: the previous tick interpolated, so this repeat tick owes the 3/4 frame
        behind_ms = 1000 * TICK * 0.6  # start-of-tick lag above which the extra frames are skipped
        gap = 0.95 / hz  # time between two presents, 5% short so the schedule has slack to catch up
        next_due = 0.0
        cuts = CutDetector(self.cut_threshold)
        slot = 0
        key_frame: np.ndarray | None = None  # last frame that arrived as new content
        key_content: int | None = None
        # Scoring: --score, or live while "Simulate a 30 fps game" is on (the source keeps the left-out frames).
        scoring = score or bool(settings is not None and settings.simulate_30)
        gt = gt or getattr(source, "gt", None)
        gt_fn = gt if scoring else None
        st.hz_used.add(hz)
        if replay is not None and hasattr(replay, "set_output_hz"):
            replay.set_output_hz(hz)
        key_gpu: torch.Tensor | None = None  # GPU copy of key_frame, for scoring
        # A sink that takes CUDA tensors (the GL window with CUDA-GL interop) gets generated frames without a
        # GPU->CPU->GPU round trip; otherwise they're downloaded into pinned memory as before.
        gpu_out = bool(getattr(sink, "gpu", False))
        last_proc = 0.0
        unshown: float | None = None  # arrival time of the newest real frame not yet presented as itself
        t_start = time.perf_counter()
        # Overlay: one line for every frame shown (real and in-between), refreshed a few times a second with
        # averages since the last refresh, so it reads steadily instead of changing every frame.
        ov_text, ov_next, ov_mark = label, 0.0, (0, 0, 0)
        # What's really happening, next to what's set: the game's frame rate as the pipeline sees it (30 in test
        # mode), the game's own (test mode reads it from the source's hints), and frames shown per second. A game
        # that isn't at the rate frame generation (or test mode) needs gets a line on the overlay and, once, a
        # message (see rate_advice).
        seen_rate, game_rate, shown_rate = GameRate(), GameRate(), ShownRate()
        delay_test, test_mark = None, 0  # the delay test while it runs, and where its stretch of real_delay_ms starts
        # A 40-60 fps game: its frames' own timing, and evenly spaced frames from it (FrameTimeline). in_slot: which
        # frame (TimedFrame.id) is in each of the interpolator's two slots; timed frames go in only when needed.
        # Both start afresh when interpolation is toggled: a new timeline numbers its frames from 0 again, so a
        # slot still holding an old id could pass for a new frame and pair it with a picture from before.
        timeline, timed, in_slot = FrameTimeline(), False, [-1, -1]
        # A 30 fps game dipping below 30 (Dip smoothing): its frames timed as they arrive, on the same
        # timeline. dipping: the setting is on and DipWatch says the game is dipping now.
        dip_watch, dipping = DipWatch(), False
        reached: tuple = ()  # timed: the frames whose time came with the frame about to be shown
        audio_next, audio_ms = 0.0, None  # when the sound delay is next looked at, and what chiaki-ng was last told
        replay_next = 0.0  # when instant replay is next told the delay
        rhythm, rhythm_moves, rhythm_note = None, 0, 0.0  # Game frame rate 30 fps: the lock, its moves, next note
        off_since: float | None = None  # since when the game has been off the rate that's needed
        note, toasted = "", set()
        # Simulate a 30 fps game on a game that's at 30 already: whether the game has been at 60 since Simulate went
        # on (then a 30 fps stretch is a cutscene, not the game), and since when it has been at about 30
        test_at_60, at30_since = False, None
        last_new_at = None  # when the last new frame arrived (Stats.new_gaps_ms)

        def test_line() -> str:
            """Test mode's badge, and quality over the last ~30 scored frames (about a second). 90 fps draws frames
            at 1/3 and 2/3, where a 60 fps game has no real frame to compare against."""
            text = "TEST: 60 fps game played as 30 fps"
            q = list(zip(st.off_interp[-30:], st.off_repeat[-30:]))
            if enabled and hz == 90:
                text += "  quality: measured at 60 and 120 fps"
            elif enabled and q:
                text += (f"  quality: {pct(np.mean([x for x, _ in q]))} visibly off from the real frames"
                         f" (no AI {pct(np.mean([y for _, y in q]))})")
            return text

        def overlay_text() -> str:
            nonlocal ov_mark
            n0, s0, k0 = ov_mark
            ov_mark = (len(st.proc_interp_ms), st.quarter_shown, st.quarter_skipped)
            g, s = seen_rate.fps(), shown_rate.fps()
            proc = st.proc_interp_ms[n0:]
            shown, skipped = st.quarter_shown - s0, st.quarter_skipped - k0
            testing = settings is not None and settings.simulate_30
            if settings is not None and settings.overlay_style == "minimal":
                return mini_text(g, s, proc, shown, skipped, testing)
            rates = (f"game {g:.0f} fps{' evened out' if timed else ''}, showing {s:.0f}"
                     if g is not None and s is not None else "")
            if not enabled:
                text = f"{label}  INTERP OFF" + (f"  {rates}" if rates else "")
            else:
                text = f"{label}  INTERP ON  {rates or f'{hz} Hz'}  interp {np.mean(proc) if proc else 0.0:5.1f} ms"
                if hz > 60 and shown + skipped:
                    text += f"  extra frames {100 * shown / (shown + skipped):3.0f}%"
            if delay_test is not None:  # the delay test, while it runs, is the second line
                text += "\n" + delay_test.status()
            elif note:  # the game isn't at the rate that's needed: that's the second line
                text += "\n" + note
            elif testing:
                text += "\n" + test_line()
            return text

        def mini_text(g, s, proc, shown: int, skipped: int, testing: bool) -> MiniText:
            """Stats overlay style Minimal: frames on screen, the game's rate (amber while it's off the rate that's
            needed, which the message about it explains), SmoovPlay's delay, and how well the PC keeps up: at 90 and
            120 fps by the extra frames shown, at 60 by the AI's share of each frame's time."""
            parts = [(f"{s:.0f} fps" if s is not None else f"{hz} fps", "main")]
            if g is not None:
                parts.append((f"game {g:.0f}", "warn" if note else "dim"))
            if st.real_delay_ms:
                parts.append((f"delay {np.median(st.real_delay_ms[-120:]):.0f} ms", "dim"))
            if not enabled:
                health = "off"
            elif hz > 60 and shown + skipped:
                f = shown / (shown + skipped)
                health = "good" if f >= 0.95 else "tight" if f >= 0.8 else "behind"
            elif proc:
                load = float(np.mean(proc)) / (TICK * 1000)
                health = "good" if load < 0.75 else "tight" if load < 0.95 else "behind"
            else:
                health = ""
            # Under the numbers, only while a test runs: the delay test, or test mode (with the advice, when the game
            # isn't the 60 fps it needs; outside tests that advice is a message and the amber game rate).
            second = (delay_test.status() if delay_test is not None else (note or test_line()) if testing else "")
            return MiniText.make(parts, health, second)

        for tick in source:
            self.beat = time.perf_counter()
            if isinstance(tick, Waiting):  # no frame for a moment: keep the window answering until one comes
                self.waiting = tick
                if sink is not None and hasattr(sink, "wait"):
                    cmd = sink.wait(tick)
                    if cmd in ("quit", "restart"):
                        st.exit = cmd
                        break
                continue
            self.waiting = None
            if tick.t_arrival < t_start:  # queued while the previous session was restarting: show live frames only
                st.stale += 1
                continue
            if settings is not None:  # changed in the menu (or by a hotkey) since the last tick?
                want_hz = settings.hz if sink is not None else 60
                if want_hz != hz:
                    hz, gap, quarter_due, ov_next = want_hz, 0.95 / want_hz, False, 0.0
                    st.hz_used.add(hz)
                    if replay is not None and hasattr(replay, "set_output_hz"):
                        replay.set_output_hz(hz)
                    print(f"[pipeline] output {hz} Hz")
                if (score or settings.simulate_30) != scoring:
                    scoring = not scoring
                    gt_fn = gt if scoring else None
                    key_gpu, ov_next = None, 0.0
                    seen_rate.clear()  # test mode changes what the pipeline sees; the game's own rate stays
                    off_since, note = None, ""
                    print(f"[pipeline] test mode (simulated 30 fps game) {'ON' if settings.simulate_30 else 'OFF'}")
                if settings.simulate_30:
                    st.simulated += 1
                if settings.interp != enabled:
                    enabled, sched, quarter_due, ov_next = settings.interp, CadenceScheduler(), False, 0.0
                    timeline, in_slot = FrameTimeline(), [-1, -1]  # afresh too: its frames are from before
                    print(f"[pipeline] interpolation {'ON' if enabled else 'OFF'}")
                if settings.steady_hud != steady:
                    steady = settings.steady_hud
                    if hasattr(it, "set_steady"):
                        it.set_steady(STEADY_HUD_THRESHOLD if steady else 0.0)
                        print(f"[pipeline] steady HUD {'ON' if steady else 'OFF'}")
                    elif steady:
                        print("[pipeline] steady HUD isn't available with this engine")
            t0 = time.perf_counter()
            st.queue_ms.append((t0 - tick.t_arrival) * 1000)
            if torch.is_tensor(tick.frame):  # already on the GPU (hardware decoder)
                it.load(tick.frame)
            else:
                np.copyto(self._in_np, tick.frame)
                it.load(self._in_pinned)
            diff = self.detector.update(it.in_u8)
            # A source that knows better (the frame tap's flags, test mode) overrides the image check.
            is_new = diff.is_new if tick.new_hint is None else tick.new_hint
            # Dip smoothing, by the picture's own call (the 30 fps lock below can't see a dip); not in
            # test mode, which decides what's new itself, nor without a window
            dip_watch.update(is_new)
            dipping = (dip_watch.dipping and sink is not None and gt_fn is None and "game_new" not in tick.extra
                       and settings is not None and getattr(settings, "even_dips", False))
            seen_new = is_new
            # Game frame rate set to 30 fps: every other tick is new, the checks only choosing which (not in test
            # mode, which decides itself what's new)
            lock30 = (settings is not None and getattr(settings, "game_fps", "auto") == "30"
                      and "game_new" not in tick.extra)
            if lock30 != (rhythm is not None):
                rhythm = RhythmLock() if lock30 else None
                print(f"[pipeline] game frame rate: {'30 fps (every other frame is new)' if lock30 else 'Auto'}")
            if rhythm is not None:
                is_new = rhythm.update(is_new, tick.extra.get("size"), bool(tick.extra.get("keyframe")),
                                       tick.extra.get("seq"))
                if dipping:  # its frames timed as they arrived, which the lock's every-other-tick can't follow
                    is_new = seen_new
                if rhythm.moves != rhythm_moves:
                    rhythm_moves = rhythm.moves
                    if t0 >= rhythm_note:  # the game stuttered: its frames moved by a tick
                        rhythm_note = t0 + 10.0
                        print(f"[pipeline] 30 fps: the game's frames moved by a tick ({rhythm_moves} so far)")
            seen_rate.add(is_new)
            if is_new:
                if last_new_at is not None:
                    st.new_gaps_ms.append((tick.t_arrival - last_new_at) * 1000)
                last_new_at = tick.t_arrival
            # In test mode the source's own word on the game's frame (TestMode puts it in extra["game_new"]).
            game_new = tick.extra["game_new"] if "game_new" in tick.extra else is_new
            game_rate.add(game_new)
            if "game_new" in tick.extra and game_new is not None:
                if game_new:
                    st.test_game_new += 1
                else:
                    st.test_game_repeat += 1
            if delay_test is not None:
                delay_test.feed(delaytest.small(it.in_u8), tick.t_arrival, time.perf_counter())
                if delay_test.done:
                    ours = st.real_delay_ms[test_mark:]
                    result = delay_test.result(float(np.median(ours)) if ours else None)
                    st.delay_tests.append(result)
                    if not st.private:  # with NVIDIA VFG it stays on screen and in test mode's report
                        print(f"[pipeline] {result}")
                    if hasattr(sink, "toast"):
                        sink.toast(result.replace("Delay test: ", "", 1), 15.0)
                    delay_test, ov_next = None, 0.0
            # Sound: chiaki-ng plays it as it arrives, while the picture spends a while in SmoovPlay. Every 2 s,
            # chiaki-ng is told to hold the sound back by that while (the median of the last ~2 s of real frames).
            sync = settings is None or getattr(settings, "audio_sync", True)
            if remote is not None and t0 >= audio_next and (st.real_delay_ms or not sync):
                audio_next = t0 + 2.0
                want = round(float(np.median(st.real_delay_ms[-120:]))) if sync else 0
                if (audio_ms is None or abs(want - audio_ms) >= 4) and remote.can_delay_audio():
                    if want and (audio_ms is None or abs(want - audio_ms) >= 10):
                        print(f"[pipeline] sound held back {want} ms to meet the picture")
                    elif not want and audio_ms:
                        print("[pipeline] sound no longer held back (Sound in sync is off)")
                    remote.set_audio_delay(want)
                    audio_ms = want
            # Instant replay times the game's own frames and its sound by the same delay, to line up with the picture.
            if replay is not None and t0 >= replay_next and st.real_delay_ms:
                replay_next = t0 + 2.0
                replay.set_delay(float(np.median(st.real_delay_ms[-120:])))

            if tick.truth_new is not None:
                st.dedup[("t" if is_new == tick.truth_new else "f") + ("p" if is_new else "n")] += 1

            to_score = None
            early_cmd = None  # 90/120 Hz: the first of this tick's two frames, already shown
            mid_t = 0.5  # the generated frame this tick ends on
            # Behind = waiting frames, not decode time: a Remote Play frame's arrival is before its ~3-11 ms decode.
            t_avail = tick.t_arrival if tick.t_ready is None else tick.t_ready
            behind = (t0 - t_avail) * 1000 >= behind_ms
            st.behind += behind
            extra_ok = hz > 60 and sink is not None and not behind

            def present(frame, text: str):
                # Frame pacer: each present is due one gap after the previous one was *due*, so sleep overshoot and
                # present time don't accumulate; when late it goes straight away and the schedule restarts from now.
                nonlocal next_due
                if hz > 60:
                    now = time.perf_counter()
                    if next_due > now:
                        time.sleep(next_due - now)
                    next_due = max(next_due, now) + gap
                cmd = sink.write(frame, text)
                t_now = time.perf_counter()
                shown_rate.add(t_now)
                if replay is not None:
                    replay.frame(frame, t_now)  # after it's on screen, so it adds no delay
                return cmd

            def show_early(t: float) -> bool:
                nonlocal early_cmd
                if not extra_ok:
                    st.quarter_skipped += 1
                    return False
                early_cmd = present(self._generated(it.interpolate(slot, t), gpu_out), ov_text)
                st.quarter_shown += 1
                return True

            if not enabled:
                action, out = Action.CURRENT, tick.frame
                quarter_due = timed = False
            else:
                # A frame the source flags as damaged (e.g. Remote Play packet loss) is held like a cut, so its
                # corruption isn't blended into the generated frame. `|` so the cut detector still sees every frame.
                is_cut = is_new and (cuts.update(diff.mean_diff, diff.cut) | tick.damaged)
                action = sched.step(is_new, is_cut)
                timeline.set_at_arrival(dipping)
                added = timeline.tick(is_new, is_cut, tick.frame, tick.t_arrival)
                was_timed = timed
                # Not before the timeline has a frame: interpolation turned on at a repeat tick, while the game was
                # already dipping, asked place() for frames it didn't have (and the timeline was never fed while
                # interpolation was off, so it held frames from before; it starts afresh at the toggle now).
                timed = bool(timeline.frames) and (dipping or (timeline.uneven and sink is not None and gt_fn is None
                                                               and (settings is None or getattr(settings, "even_out", True))))
                if timed != was_timed:
                    if timed and dipping:
                        unshown = None
                        print("[pipeline] game dipping below 30 fps: timing its frames as they arrived")
                        if hasattr(sink, "toast") and "dips" not in toasted:
                            toasted.add("dips")
                            sink.toast("Game dipping below 30 fps: SmoovPlay is spacing its frames by when they "
                                       "arrived (Dip smoothing)", 6.0)
                    elif timed:
                        unshown = None
                        print(f"[pipeline] game at about {timeline.fps()} fps: evening out its frames")
                        if hasattr(sink, "toast") and "even" not in toasted:
                            toasted.add("even")
                            sink.toast(f"Game at about {timeline.fps()} fps: SmoovPlay is spacing its frames evenly "
                                       "(Even out 40-60 fps)", 6.0)
                    else:
                        sched = CadenceScheduler()  # starts afresh, from the next new frame
                        action = sched.step(is_new, is_cut)
                        print("[pipeline] even timing off")
                if is_new and not timed:
                    slot = 1 - slot
                    it.prepare(slot)  # (in_u8 holds this tick's frame)
                    in_slot[slot] = added.id
                    if settings is not None and getattr(settings, "debug", False) and self.debug_dir is not None:
                        real_frames += 1
                        if real_frames % debugsnap.EVERY == 0:
                            frame = getattr(it, "in_u8", None)
                            frame = frame.cpu().numpy() if torch.is_tensor(frame) else frame
                            print(debugsnap.write(self.debug_dir, real_frames // debugsnap.EVERY, frame,
                                                  debugsnap.mask_of(it)), flush=True)
                if timed:
                    if dipping:
                        st.dip_ticks += 1
                    else:
                        st.uneven_ticks += 1
                        if is_new:
                            st.uneven_fps.append(timeline.fps())
                    quarter_due = False
                    phases = timeline.slots(hz)
                    if len(phases) > 1 and not extra_ok:
                        st.quarter_skipped += len(phases) - 1
                        phases = phases[-1:]
                    else:
                        st.quarter_shown += len(phases) - 1
                    for k, phi in enumerate(phases):
                        pl = timeline.place(phi)
                        if pl.b is None:
                            frame = pl.a.frame
                        else:
                            slot = self._timed_pair(pl.a, pl.b, slot, in_slot)
                            frame = self._generated(it.interpolate(slot, pl.t), gpu_out)
                        st.uneven_waits += pl.waiting
                        if k < len(phases) - 1:
                            early_cmd = present(frame, ov_text) or early_cmd
                            t_now = time.perf_counter()
                            st.real_delay_ms.extend((t_now - f.t_arrival) * 1000 for f in pl.reached)
                        else:
                            out, reached = frame, pl.reached
                            action = Action.INTERP if pl.b is not None else Action.CURRENT
                elif hz == 120 and action is Action.INTERP:
                    show_early(0.25)
                elif hz == 120 and quarter_due and action is Action.CURRENT and not is_new:
                    show_early(0.75)
                elif hz == 90 and action is Action.INTERP and show_early(1 / 3):
                    mid_t = 2 / 3
                quarter_due = action is Action.INTERP
                if timed:
                    pass  # shown above
                elif action is Action.INTERP:
                    out = self._generated(it.interpolate(slot, mid_t), gpu_out)
                    # --score has ground truth only at the midpoint
                    if gt_fn is not None and key_gpu is not None and mid_t == 0.5:
                        to_score = (key_frame, key_content)  # scored once the frame is on screen
                elif action in (Action.PENDING, Action.HOLD):
                    out = key_frame
                elif rhythm is not None and key_frame is not None:
                    # Game frame rate 30 fps, and the lock called this tick a repeat. After a game holds a frame for
                    # a third tick (it dipped below 30), the lock is a tick out of step until its vote moves, and this
                    # tick carries the game's next frame: shown now, the in-between frame before it would follow on
                    # the next tick, stepping back. The frame the lock last called new keeps the order, a tick late.
                    out = key_frame
                else:
                    out = tick.frame
            prev_key = key_gpu
            if is_new:
                key_frame, key_content = tick.frame, tick.content
                if gt_fn is not None:
                    key_gpu = it.in_u8.clone()

            t_ready = time.perf_counter()
            last_proc = (t_ready - t0) * 1000
            (st.proc_interp_ms if action is Action.INTERP else st.proc_other_ms).append(last_proc)
            st.actions[action.value] += 1
            if on_tick is not None:
                on_tick(tick, diff, action, last_proc, t0)

            cmd = None
            if sink is not None:
                if t_ready >= ov_next:
                    test = settings is not None and settings.simulate_30
                    even = sink is not None and (settings is None or getattr(settings, "even_out", True))
                    game_fps = game_rate.fps()
                    # Simulate on a game at about 30 fps already plays it at 15 (e.g. started with the launcher's
                    # "Test with a 60 fps game"): after RATE_TOAST_S it goes off by itself (TestMode follows the
                    # setting). Not when the game has been at 60 since Simulate went on: a 30 fps cutscene.
                    test_at_60 = test and (test_at_60 or (game_fps is not None and game_fps >= 50))
                    at30 = test and not test_at_60 and game_fps is not None and 24 <= game_fps < 38.5
                    at30_since = None if not at30 else t_ready if at30_since is None else at30_since
                    if at30 and t_ready - at30_since >= RATE_TOAST_S:
                        settings.override("simulate_30", False)
                        test, at30_since = False, None
                        toasted.add(True)  # its own message instead of the advice
                        said = SIMULATE_OFF.format(g=round(game_fps))
                        print(f"[pipeline] {said}")
                        if hasattr(sink, "toast"):
                            sink.toast(said, 8.0)
                    advice = rate_advice(game_fps, test, even) if enabled or test else ""
                    if not advice:
                        off_since, note = None, ""
                    else:
                        off_since = t_ready if off_since is None else off_since
                        note = advice if t_ready - off_since >= RATE_NOTE_S else ""
                        if t_ready - off_since >= RATE_TOAST_S and test not in toasted:  # once a session for each
                            toasted.add(test)
                            print(f"[pipeline] {advice}")
                            if hasattr(sink, "toast"):
                                sink.toast(advice, 8.0)
                    ov_text, ov_next = overlay_text(), t_ready + 0.25
                cmd = present(out, ov_text) or early_cmd
            t_shown = time.perf_counter()
            st.busy_ms.append((t_shown - t0) * 1000)
            if sink is not None:
                st.sink_ms.append((t_shown - t_ready) * 1000)

            # Track when each real frame first reaches the screen (timed: when the game time shown reaches it).
            if timed:
                st.real_delay_ms.extend((t_shown - f.t_arrival) * 1000 for f in reached)
                reached = ()
            elif action is Action.PENDING and unshown is not None:
                st.real_delay_ms.append((t_shown - unshown) * 1000)
                unshown = None
            elif is_new:
                if action is Action.CURRENT:
                    st.real_delay_ms.append((t_shown - tick.t_arrival) * 1000)
                else:
                    unshown = tick.t_arrival
            elif unshown is not None:
                st.real_delay_ms.append((t_shown - unshown) * 1000)
                unshown = None

            if replay is not None and is_new:  # the game's own frame, for a before & after video
                replay.real_frame(tick.frame, tick.t_arrival)

            if to_score is not None:
                self._score(st, tick, prev_key, *to_score, gt_fn, keep_worst=score)
                st.score_ms.append((time.perf_counter() - t_shown) * 1000)

            if sink is not None:
                if cmd in ("quit", "restart"):
                    st.exit = cmd
                    break
                if cmd == "delay_test" and delay_test is None:
                    if remote is None or not remote.can_press():
                        msg = ("The delay test works with PS5 Remote Play, through the chiaki-ng that comes with this "
                               "SmoovPlay" if remote is None or not hasattr(remote, "press") else
                               "The delay test needs the chiaki-ng that comes with this SmoovPlay: open it from there")
                        print(f"[pipeline] {msg}")
                        if hasattr(sink, "toast"):
                            sink.toast(msg, 6.0)
                    else:
                        delay_test, test_mark, ov_next = delaytest.DelayTest(remote.press, remote.press_sent), len(
                            st.real_delay_ms), 0.0
                        print("[pipeline] delay test started")
                if cmd in ("save_replay", "save_compare"):  # R and B
                    if replay is not None:
                        replay.save("compare" if cmd == "save_compare" else None)
                    elif hasattr(sink, "toast"):
                        sink.toast("Instant replay isn't available here", 4.0)
                if cmd == "toggle_interp":  # the OpenCV window's i key (the GL window changes settings itself)
                    if settings is not None:
                        settings.override("interp", not settings.interp)  # picked up at the next tick
                    else:
                        enabled = not enabled
                        sched = CadenceScheduler()
                        timeline, in_slot = FrameTimeline(), [-1, -1]  # afresh too: its frames are from before
                        print(f"[pipeline] interpolation {'ON' if enabled else 'OFF'}")

        st.wall_s = time.perf_counter() - t_start
        return st

    def _load(self, frame) -> None:
        """A frame (GPU tensor or HxWx3 uint8 array) into the interpolator's input."""
        if torch.is_tensor(frame):
            self.interp.load(frame)
        else:
            np.copyto(self._in_np, frame)
            self.interp.load(self._in_pinned)

    def _timed_pair(self, a, b, slot: int, in_slot: list) -> int:
        """Even timing: frames a then b (TimedFrames) in the interpolator's two slots, b the newer, putting in what
        isn't there yet. Returns b's slot. (Overwrites the input: only after this tick's frame has been looked at.)"""
        if in_slot[slot] == b.id and in_slot[1 - slot] == a.id:
            return slot
        if in_slot[slot] != a.id:
            if in_slot[1 - slot] == a.id:  # a is the older of the two already: b goes in after it
                slot = 1 - slot
            else:
                slot = 1 - slot
                self._load(a.frame)
                self.interp.prepare(slot)
                in_slot[slot] = a.id
        slot = 1 - slot
        self._load(b.frame)
        self.interp.prepare(slot)
        in_slot[slot] = b.id
        return slot

    def _generated(self, frame: torch.Tensor, gpu_out: bool):
        """A generated frame for the sink, waited for (a sync point, so timings are real)."""
        if gpu_out:
            torch.cuda.current_stream().synchronize()
            return frame
        self._out_pinned.copy_(frame)  # blocking D2H copy
        return self._out_np

    def _score(self, st: Stats, tick, a: torch.Tensor, prev, prev_content, gt_fn, keep_worst: bool = True) -> None:
        """a: the previous real frame on the GPU (prev is the same frame on the CPU, kept for --save-worst).
        keep_worst: copy the worst frame's images to the CPU (--save-worst); test mode skips that hitch."""
        if prev_content is None or tick.content is None or tick.content - prev_content != 2:
            return  # only score clean 2-tick steps, where the in-between source frame exists
        gt_np = gt_fn(prev_content + 1)
        if gt_np is None:
            return
        # In 8-bit levels: HDR frames (float16) through hdr.level8, so thresholds and dB read as for SDR ones.
        gt = level8(gt_np if torch.is_tensor(gt_np) else torch.from_numpy(gt_np).to(self.interp.device))
        b = self.interp.in_u8
        out = level8(self.interp.out_u8)
        blend = ((a.float() + b.float()) * 0.5).round_()
        still = ((b.float() - a.float()).abs().amax(-1, keepdim=True) <= STILL_LEVELS).float()
        n_still = still.sum().clamp_min(1.0) * 3
        errs = [(x - gt).pow_(2) for x in (out.float(), a.float(), blend)]
        off = (out.float() - gt).abs().amax(-1, keepdim=True) > VISIBLE_LEVELS  # visibly off, AI frame
        off_repeat = (a.float() - gt).abs().amax(-1, keepdim=True) > VISIBLE_LEVELS  # and the last frame repeated
        mse = torch.stack([e.mean() for e in errs] + [(errs[0] * still).sum() / n_still,
                                                       (errs[1] * still).sum() / n_still, still.mean(),
                                                       off.float().mean(), off_repeat.float().mean()]).tolist()
        p, p_repeat, p_blend = map(_db, mse[:3])
        if mse[5] > 0:
            st.psnr_still.append(_db(mse[3]))
            st.psnr_still_repeat.append(_db(mse[4]))
            st.still_frac.append(mse[5])
        st.psnr_interp.append(p)
        st.psnr_repeat.append(p_repeat)
        st.psnr_blend.append(p_blend)
        st.off_interp.append(mse[6])
        st.off_repeat.append(mse[7])
        hud = hud_mask(a, b, gt, still)
        n_hud = hud.sum().item()
        if n_hud:
            st.psnr_hud.append(_db((errs[0] * hud).sum().item() / (3 * n_hud)))
            st.hud_off.append((off.float() * hud).sum().item() / n_hud)
            st.hud_frac.append(n_hud / hud.numel())
        if keep_worst and (st.worst is None or p < st.worst[0]):
            st.worst = (p, tick.index, _np(prev), _np(self.interp.out_u8), _np(gt_np), _np(tick.frame))
