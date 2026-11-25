"""Duplicate-frame detection and output scheduling for a 60 Hz signal carrying 30 fps content.

A console's video (PS5 Remote Play or HDMI) runs at 60 Hz, so a 30 fps game arrives as A A B B C C ...
(each real frame scanned out twice). At 60 Hz output we emit, one tick behind the input:

    input : A   A   B        B   C        C
    output: A   A   mid(A,B) B   mid(B,C) C

i.e. on the tick a new frame arrives we show the midpoint between it and the previous
real frame, and on its duplicate tick we show the real frame itself. Net added delay is
one 60 Hz tick (~16.7 ms) plus processing time. At 90 and 120 Hz, two or three frames are drawn between each pair
(see pipeline.py).

The scheduler also has to handle the cadence breaking:
  * game hitch (A A A B B): the extra duplicate is just shown again.
  * real 60 fps content (menus, some cutscenes: A B C D): two new frames in a row, so
    we pass frames through one tick late instead of interpolating.
  * scene cut: interpolating across a cut produces a ghosted blend, so hold instead.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from enum import Enum

import torch
import torch.nn.functional as F


class Action(Enum):
    INTERP = "interp"  # show midpoint(prev real, current)
    CURRENT = "current"  # show the incoming frame as-is
    PENDING = "pending"  # show the previous new frame (not yet displayed), current becomes pending
    HOLD = "hold"  # show the previous real frame again (scene cut)


@dataclass
class CutFeatures:
    """How differently two frames are *built*, whatever their brightness: for cuts between dark shots, where the
    plain difference stays small."""
    corr_dist: float  # 1 - correlation of the two frames' block brightness, 0 (same layout) .. 2 (inverted)
    hist_dist: float  # half the L1 distance between brightness histograms, 0 (same) .. 1 (no overlap)
    luma: float  # this frame's mean brightness, 0..255
    contrast: float  # this frame's brightness standard deviation over blocks, 0..255


@dataclass
class FrameDiff:
    is_new: bool
    max_diff: float  # max 16x16 block-mean difference, 0..255
    mean_diff: float  # mean absolute difference over blocks, 0..255
    cut: CutFeatures | None = None  # None on the first frame


_LUMA = torch.tensor([0.114, 0.587, 0.299]).view(1, 3, 1, 1)  # BGR -> Y (BT.601)
HIST_BINS = 32


def _cut_stats(prev: torch.Tensor, cur: torch.Tensor) -> torch.Tensor:
    """CutFeatures' four numbers as one tensor (no device sync), from two (1, 3, h, w) BGR block-mean images."""
    w = _LUMA.to(cur.device)
    yp, yc = (prev * w).sum(1).flatten(), (cur * w).sum(1).flatten()
    dp, dc = yp - yp.mean(), yc - yc.mean()
    sp, sc = dp.square().mean().sqrt(), dc.square().mean().sqrt()
    # Correlation is undefined for a flat frame (a black screen); call two flat frames alike, and a flat frame
    # next to a detailed one fully different.
    denom = (sp * sc).clamp_min(1e-6)
    corr = (dp * dc).mean() / denom
    flat = (sp < 1.0) | (sc < 1.0)
    corr = torch.where(flat, ((sp < 1.0) & (sc < 1.0)).float(), corr)
    hp = torch.histc(yp, HIST_BINS, 0, 255) / yp.numel()
    hc = torch.histc(yc, HIST_BINS, 0, 255) / yc.numel()
    return torch.stack([1 - corr, 0.5 * (hp - hc).abs().sum(), yc.mean(), sc])


def cut_features(prev: torch.Tensor, cur: torch.Tensor) -> CutFeatures:
    return CutFeatures(*_cut_stats(prev, cur).tolist())


class DuplicateDetector:
    """Classifies each incoming frame as new content or a repeat of the previous one.

    Compares 16x16 block means rather than raw pixels: block averaging suppresses
    per-pixel capture noise, while the max over blocks still catches small, local
    motion (a character moving on a static background barely moves a global mean).
    Over a digital HDMI capture, repeats are normally bit-identical (max_diff == 0).

    reference: what each frame is compared with. "previous": the frame before it. "new": the last frame found
    to be new content, so a slow change that stays under the threshold from one frame to the next still adds up
    and is caught (for capture cards, where a little noise can hide a small real change).

    learn: for a window captured from an app that decodes its own stream (window capture), whose repeats are never
    quite identical, so a fixed threshold calls most of them new. It watches the last LEARN_N mean differences: when
    they fall in two bands (repeats and new frames: the top fifth over LEARN_RATIO times the bottom fifth), the line
    between them is their geometric mean; otherwise the fixed rule. On a decoded 30 fps stream it agreed with the
    stream's own repeat flags 99% of the time, against 50% for the fixed rule.

    With learn, it also keeps to the game's rhythm: once the last RHYTHM_N calls follow one exactly (new every 2nd
    frame for a 30 fps game, every 3rd for 20 fps), a frame whose difference is close to the line (less than SURE of
    the way, in log terms, from the line to the other band) is called as the rhythm says. A clear exception (a
    hitch, a cut to a 60 fps menu) drops the rhythm, as does losing the two bands; it's found again after
    RHYTHM_N steady frames. `held` counts the calls the rhythm made.
    """

    LEARN_N, LEARN_MIN, LEARN_RATIO = 16, 8, 4.0
    RHYTHM_N, RHYTHM_PERIODS, SURE = 24, (2, 3), 0.5

    def __init__(self, threshold: float = 2.0, block: int = 16, reference: str = "previous",
                 learn: bool = False) -> None:
        if reference not in ("previous", "new"):
            raise ValueError(f"reference: {reference!r}")
        self.threshold = threshold
        self.block = block
        self.reference = reference
        self.learn = learn
        self._means: deque = deque(maxlen=self.LEARN_N)
        self._prev: torch.Tensor | None = None
        self._calls: deque = deque(maxlen=self.RHYTHM_N)  # the last calls, oldest first
        self.rhythm: tuple[int, int] | None = None  # (period, where the next new frame falls in it), once steady
        self.held = 0

    def reset(self) -> None:
        self._prev = None
        self._means.clear()
        self._calls.clear()
        self.rhythm = None

    def _learned(self, mean: float) -> tuple[bool, float] | None:
        """The two-band rule's call on a frame's mean difference and how sure it is (0 on the line, 1 at a band's
        middle), or None when the recent frames don't show two bands."""
        if len(self._means) < self.LEARN_MIN:
            return None
        xs = sorted(self._means)
        lo = max(xs[round(0.2 * (len(xs) - 1))], 1e-3)
        hi = xs[round(0.8 * (len(xs) - 1))]
        if hi <= self.LEARN_RATIO * lo:
            return None
        line = math.sqrt(lo * hi)
        return mean > line, abs(math.log(max(mean, 1e-6) / line)) / math.log(hi / line)

    def _keep_rhythm(self, is_new: bool, sure: float | None) -> bool:
        """The call after the game's rhythm has had its say (learn only), and the rhythm brought up to date."""
        if self.rhythm is not None:
            period, phase = self.rhythm
            expected = phase == 0
            if sure is None or (expected != is_new and sure >= self.SURE):
                self.rhythm = None  # no bands, or a clear exception: listen to the pictures alone for a while
                self._calls.clear()
            else:
                if expected != is_new:
                    self.held += 1
                    is_new = expected
                self.rhythm = (period, (phase + 1) % period)
        self._calls.append(is_new)
        if self.rhythm is None and sure is not None and len(self._calls) == self.RHYTHM_N:
            calls = list(self._calls)
            for period in self.RHYTHM_PERIODS:
                first = calls.index(True) if True in calls else -1
                if 0 <= first < period and all(c == ((i - first) % period == 0) for i, c in enumerate(calls)):
                    self.rhythm = (period, (len(calls) - first) % period)
                    break
        return is_new

    @torch.no_grad()
    def update(self, frame_u8: torch.Tensor) -> FrameDiff:
        x = frame_u8.permute(2, 0, 1).unsqueeze(0).float()
        small = F.avg_pool2d(x, self.block, ceil_mode=True)
        if self._prev is None:
            self._prev = small
            return FrameDiff(True, float("inf"), float("inf"))
        d = (small - self._prev).abs_()
        stats = torch.cat([torch.stack([d.amax(), d.mean()]), _cut_stats(self._prev, small)]).tolist()  # one sync
        is_new = stats[0] > self.threshold
        if self.learn:
            learned = self._learned(stats[1])
            self._means.append(stats[1])
            if learned is not None:
                is_new = learned[0]
            is_new = self._keep_rhythm(is_new, learned[1] if learned is not None else None)
        if is_new or self.reference == "previous":
            self._prev = small
        return FrameDiff(is_new, stats[0], stats[1], CutFeatures(*stats[2:]))


class RhythmLock:
    """Game frame rate set to 30 fps: a 30 fps game at 60 Hz alternates new frame, repeat, so every other tick is new,
    whatever the checks make of a still scene, a dark one or a busy stream. The checks only choose which of the two
    ticks it is (the phase), by a vote over the last WINDOW ticks; it moves when the other phase wins by MARGIN (a
    hitch shifts the game's frames by a tick). A repeat is a copy of the frame before it, so the wrong phase only
    shows each frame a tick (17 ms) later: the vote can take its time and ignore misreads.

    The evidence for a tick: with compressed sizes (PS5 Remote Play), whether it's bigger than the tick before (a new
    frame costs the encoder more than the repeat after it, however small both are, and that needs no line between
    "small" and "big"); else the checks' new-or-repeat call. Keyframes and ticks with no call say nothing.

    seq: the stream's own frame number (Remote Play), or a capture card's count as frames come off the card (before
    SmoovPlay's queue, which drops one when the PC falls behind). Frames are counted by it rather than as they arrive,
    so a frame lost on the way doesn't shift the count (in a real Gotham Knights session, 23 lost frames moved the
    rhythm 11 times when counted as they arrived)."""

    WINDOW, MARGIN, MIN = 32, 10, 8

    def __init__(self) -> None:
        self._votes: deque = deque(maxlen=self.WINDOW)  # (frame number, looks new)
        self._n = 0
        self._last: tuple[int, int] | None = None  # the previous frame's (number, size)
        self.phase: int | None = None  # new frames fall on ticks whose number % 2 == phase
        self.moves = 0  # times the phase moved after it was first found

    def update(self, call: bool | None, size: int | None = None, keyframe: bool = False,
               seq: int | None = None) -> bool:
        n = seq if seq is not None else self._n
        self._n = n + 1
        if size is not None:  # against the frame just before it (not one from before a lost frame)
            prev = self._last
            looks = None if keyframe or prev is None or prev[0] != n - 1 else size > prev[1]
            self._last = None if keyframe else (n, size)
        else:
            looks = call
        if looks is not None:
            self._votes.append((n, looks))
        if len(self._votes) >= self.MIN:
            agree = [sum(v == (i % 2 == p) for i, v in self._votes) for p in (0, 1)]
            best = 0 if agree[0] >= agree[1] else 1
            if self.phase is None:
                self.phase = best
            elif best != self.phase and agree[best] - agree[self.phase] >= self.MARGIN:
                self.phase, self.moves = best, self.moves + 1
        if self.phase is None:  # the first few ticks: as the checks say
            return True if call is None else bool(call)
        return n % 2 == self.phase


class CutDetector:
    """Flags scene cuts among *new* frames, where interpolating would ghost two shots together.

    Like ffmpeg's scdet, a cut must be both large and a sudden jump over the previous new
    frame's change: fast-but-continuous motion has consistently large diffs and doesn't
    trigger. Tuned on a 30 fps trailer that cuts every ~3 s (catches 53/53 cuts ffmpeg finds) against gameplay
    clips (0-3 holds per 600 frames, only on camera-through-geometry moments).
    """

    # A second test, on how the picture is built rather than how much it changed, catches cuts between two dark
    # shots: their plain difference stays small, but the layout of light and dark changes completely. Like the
    # first test it needs a sudden jump, so a fast camera turn (consistently changing layout) isn't a cut.
    STRUCTURE = 0.75  # corr_dist jump that marks a cut (tuned with scripts/cut_eval.py)
    MIN_DIFF = 6.0  # ...when the frame also changed at least this much (mean_diff), so noise alone never cuts

    def __init__(self, threshold: float = 25.0, use_structure: bool = True) -> None:
        self.threshold = threshold
        self.use_structure = use_structure
        self.prev = 0.0
        self.prev_corr = 0.0

    def update(self, mean_diff: float, f: CutFeatures | None = None) -> bool:
        score = min(mean_diff, mean_diff - self.prev)
        self.prev = mean_diff
        cut = score > self.threshold
        if self.use_structure and f is not None:
            jump = min(f.corr_dist, f.corr_dist - self.prev_corr)
            self.prev_corr = f.corr_dist
            cut = cut or (jump > self.STRUCTURE and mean_diff > self.MIN_DIFF)
        return cut


class CadenceScheduler:
    """Pure state machine: (is_new, is_cut) per tick -> Action."""

    def __init__(self) -> None:
        self.started = False
        self.pending = False  # a new frame has been received but not yet displayed as itself

    def step(self, is_new: bool, is_cut: bool = False) -> Action:
        if not is_new:
            self.pending = False
            return Action.CURRENT  # duplicate of the latest real frame
        if not self.started:
            self.started = True
            return Action.CURRENT
        if self.pending:
            return Action.PENDING  # back-to-back new frames: source is >30 fps here
        self.pending = True
        return Action.HOLD if is_cut else Action.INTERP



UNEVEN_ON = (39, 56)  # new frames in the last second at which the even timing below switches on...
UNEVEN_OFF = (35, 58)  # ...and outside which it switches off again
UNEVEN_DWELL = 60  # ticks (a second) the rate has to stay there first, so a 30 fps game's 60 fps menu doesn't
# flip it on and off
UNEVEN_MAX_DELAY = 2.0  # ticks

# Dip smoothing (DipWatch, FrameTimeline.at_arrival): a 30 fps game that misses its slot shows a frame
# for 3 ticks (50 ms) or, waiting for the next 30 fps slot, 4 (67 ms; Gotham Knights' hitches on PS5 Remote Play).
DIP_HOLD = (3, 5)  # ticks a frame stays up to count as a dip; longer is a pause, a load or a still scene
DIP_ON_HOLDS = 2  # dips in the last second (60 ticks) that switch it on...
DIP_FPS = (15, 34)  # ...while the game's rate is in this range (a 40-60 fps game is Even out 40-60's)
DIP_OFF_TICKS = 180  # ticks (3 s) without a dip, or out of that range, that switch it off again
DIP_MAX_DELAY = 3.0  # ticks: enough to wait out a 4-tick hold


@dataclass
class TimedFrame:
    """A new frame on the timeline: its time in the game (ticks, see FrameTimeline), the tick it arrived on, whether
    a scene cut (or a damaged frame) comes just before it, and the pipeline's own: the frame and its arrival time."""
    id: int
    t: float
    arrival: int
    cut: bool = False
    frame: object = None
    t_arrival: float = 0.0
    reached: bool = False  # it has been on screen (drawn as itself, or its time has passed)


@dataclass
class Place:
    """What to show at one display time: frame `a` itself, or (`b` not None) the frame at fraction t from a to b.
    c: the game time shown. reached: frames on screen for the first time with this one, drawn as themselves or
    passed (for the delay stats, which the sound sync follows)."""
    a: TimedFrame
    b: TimedFrame | None = None
    t: float = 0.0
    c: float = 0.0
    reached: tuple = ()
    waiting: bool = False  # the game time shown is past the newest frame: the next one is late


class DipWatch:
    """Whether a game that runs at up to 30 fps is dipping below it right now (Dip smoothing).

    At 30 fps on a 60 Hz stream each frame stays up 2 ticks. A frame the game finished late stays up 3 or 4 ticks
    and the next one jumps, so motion freezes and lurches. While it's dipping, the pipeline times the game's frames
    as they arrived (FrameTimeline.at_arrival) and draws evenly spaced frames from them, so a dip shows as a short
    slow-down. That needs more delay (up to DIP_MAX_DELAY ticks), so it's only on while the game dips: DIP_ON_HOLDS
    dips within a second, with the game at DIP_FPS; off after DIP_OFF_TICKS without one. update() takes the
    picture's own call on each tick (not the 30 fps lock's, which can't see a dip).
    """

    def __init__(self) -> None:
        self.n = -1
        self.dipping = False
        self._last_new: int | None = None
        self._holds: deque = deque()  # ticks at which a held frame's successor arrived, over the last DIP_OFF_TICKS
        self._recent: deque = deque(maxlen=60)  # new (1) or repeat (0), over the last second
        self._out = 0  # ticks in a row with the game's rate outside DIP_FPS

    def update(self, is_new: bool) -> bool:
        self.n += 1
        self._recent.append(1 if is_new else 0)
        if is_new:
            if self._last_new is not None and DIP_HOLD[0] <= self.n - self._last_new <= DIP_HOLD[1]:
                self._holds.append(self.n)
            self._last_new = self.n
        while self._holds and self._holds[0] <= self.n - DIP_OFF_TICKS:
            self._holds.popleft()
        fps = sum(self._recent) if len(self._recent) == self._recent.maxlen else None
        in_range = fps is not None and DIP_FPS[0] <= fps <= DIP_FPS[1]
        self._out = 0 if in_range else self._out + 1
        if not self.dipping:
            self.dipping = in_range and sum(h > self.n - 60 for h in self._holds) >= DIP_ON_HOLDS
        elif not self._holds or self._out >= DIP_OFF_TICKS:
            self.dipping = False
        return self.dipping


class FrameTimeline:
    """Even timing for games whose frame rate is neither 30 nor 60: about 40 to 60 fps, as in unlocked "performance"
    modes with VRR off. CadenceScheduler keeps working for 30 fps games; the pipeline uses this while `uneven`.

    At 60 Hz such a game's frames arrive unevenly. At 45 fps it's new, repeat, new, new, new, repeat, ..., so frames
    stay on screen for one tick or two and motion stutters. The game drew them evenly, though, and each one was
    finished some time in the tick before it arrived. So the timeline gives each new frame a time in ticks: the last
    one's plus the recent average spacing, kept within that tick. The pipeline then shows the game's time `delay`
    ticks behind the display's (place()), drawing the in-between frame at the right fraction, so motion is even at
    60, 90 or 120 fps.

    at_arrival (Dip smoothing, while DipWatch says the game is dipping): each new frame's time is the
    tick it arrived on, not the average spacing, and the delay may grow to DIP_MAX_DELAY, so a frame the game held
    for 3 or 4 ticks is waited for and the frames either side are spread to match.

    The two frames either side of a display time can be older than the newest two (a frame can arrive while there's
    still a frame to draw before the one before it), so the pipeline hands a frame to the interpolator only when
    place() first asks for it. delay is the least that kept the game time shown from passing the newest frame over
    the last dozen frames, so the next frame has nearly always arrived when it's needed.
    """

    def __init__(self, k: int = 12) -> None:
        self.n = -1  # ticks seen
        self.frames: deque[TimedFrame] = deque()  # the frame at or before the last game time shown, and later ones
        self.delay = 1.0  # ticks between the display's time and the game time it shows
        self.uneven = False
        self._ids = 0
        self._arrivals: deque = deque(maxlen=k + 1)  # ticks at which the last few new frames arrived
        self._need: deque = deque(maxlen=k)  # for each of them: the least delay that would have waited for it
        self._last: TimedFrame | None = None
        self._recent: deque = deque(maxlen=60)  # new (1) or repeat (0), over the last second
        self._dwell = 0
        self._hz, self._j = 0, 0  # slots(): the output rate, and the next output frame's number
        self.at_arrival = False  # Dip smoothing: frames timed as they arrived (set_at_arrival)

    def set_at_arrival(self, on: bool) -> None:
        if on == self.at_arrival:
            return
        self.at_arrival = on
        if on:  # the delay the last few frames would have needed, timed as they arrived (the dips that switched it on)
            a = list(self._arrivals)
            self._need.clear()
            self._need.extend(y - x - 1 for x, y in zip(a, a[1:]) if y - x <= DIP_HOLD[1])
            self.delay = max(self.delay, self._want_delay())

    @property
    def period(self) -> float:
        """The recent average spacing of new frames, in ticks (a gap over 3 ticks, a hitch, counts as 3)."""
        a = list(self._arrivals)
        d = [min(3, y - x) for x, y in zip(a, a[1:])]
        return sum(d) / len(d) if d else 1.0

    def fps(self) -> int | None:
        """New frames in the last second (the last 60 ticks), or None until there's a second of them."""
        return sum(self._recent) if len(self._recent) == self._recent.maxlen else None

    def _want_delay(self) -> float:
        return min(DIP_MAX_DELAY if self.at_arrival else UNEVEN_MAX_DELAY, max(self._need, default=1.0) + 0.05)

    def tick(self, is_new: bool, is_cut: bool = False, frame=None, t_arrival: float = 0.0) -> TimedFrame | None:
        """One tick of the stream; a new frame goes on the timeline (and is returned)."""
        self.n += 1
        self._recent.append(1 if is_new else 0)
        added = None
        if is_new:
            self._arrivals.append(self.n)
            last = self._last
            if last is None:
                t = float(self.n)
            elif self.at_arrival and self.n - last.arrival <= DIP_HOLD[1]:
                t = max(float(self.n), last.t + 0.05)  # as it arrived, a held frame's wait included
                self._need.append(self.n - 1 - last.t)
            elif self.n - last.arrival > 3:  # after a hitch: as it comes
                t = max(self.n - 0.95, last.t + 0.05)
            else:
                # Finished in the tick before it arrived: (n - 1, n]. (The 0.05 keeps two frames' times apart.)
                t = min(float(self.n), max(self.n - 0.95, last.t + self.period))
                # Until it came (the tick before), the game time shown had to stay at or before the last frame's.
                self._need.append(self.n - 1 - last.t)
            added = TimedFrame(self._ids, t, self.n, is_cut, frame, t_arrival)
            self._ids += 1
            self.frames.append(added)
            self._last = added
            while len(self.frames) > 4:  # (only when they come much faster than they're shown)
                self.frames.popleft()
            want = self._want_delay()
            self.delay += (0.3 if want > self.delay else 0.05) * (want - self.delay)
        fps = self.fps()
        want_on = self.uneven
        if fps is not None:
            if UNEVEN_ON[0] <= fps <= UNEVEN_ON[1]:
                want_on = True
            elif fps <= UNEVEN_OFF[0] or fps >= UNEVEN_OFF[1]:
                want_on = False
        self._dwell = self._dwell + 1 if want_on != self.uneven else 0
        if self._dwell >= UNEVEN_DWELL:
            self.uneven, self._dwell = want_on, 0
            if want_on:
                self.delay = self._want_delay()
        return added

    def slots(self, hz: int) -> list[float]:
        """The display times (in ticks) of the output frames due this tick: one at 60 fps, one or two at 90, two
        at 120."""
        step = 60 / hz
        if hz != self._hz or self._j * step <= self.n - 1:  # a new rate, or not called for a while: start afresh
            self._hz, self._j = hz, math.floor((self.n - 1) / step + 1e-9) + 1
        out = []
        while self._j * step <= self.n + 1e-9:
            out.append(self._j * step)
            self._j += 1
        return out

    def place(self, phi: float) -> Place | None:
        """What to show at display time phi (ticks); None before the first frame."""
        c = phi - self.delay
        fr = self.frames
        if not fr:
            return None
        while len(fr) >= 2 and fr[1].t <= c:  # frames whose time has passed (the last of them stays)
            fr.popleft()
        reached = tuple(f for f in fr if not f.reached and f.t <= c)
        for f in reached:
            f.reached = True
        a = fr[0]
        if len(fr) == 1 or c <= a.t:
            return Place(a, c=c, reached=reached, waiting=len(fr) == 1 and c > a.t + 1e-6)
        b = fr[1]
        t = (c - a.t) / (b.t - a.t)
        if b.cut or t < 0.04:  # across a cut, the frame before it until the new one's time comes
            return Place(a, c=c, reached=reached)
        if t > 0.96:
            # b itself is on screen from now, so its delay counts from here, not from the next display time, when
            # its time has passed. (Timed as it arrived, on a whole tick, a frame was drawn as itself a tick before
            # its time passed, so Dip smoothing logged every frame a tick late, and the sound, which
            # follows the logged delay, was held back that much too long.)
            if not b.reached:
                b.reached = True
                reached += (b,)
            return Place(b, c=c, reached=reached)
        return Place(a, b, t, c, reached)
