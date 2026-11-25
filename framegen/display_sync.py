"""How each frame is handed to the display: on a screen refresh, or as soon as it's ready.

Settings "vsync" and "vrr" (G-Sync / FreeSync) and the monitor's refresh rate decide it (plan()):

  vrr     G-Sync/FreeSync: each frame is presented as soon as it's ready (sync interval 0, tearing allowed, which a
          variable-refresh display needs to follow the frame rate in a window). The pipeline's pacer is the clock.
  locked  VSync on, and the refresh rate is a whole multiple of the output rate (60 fps on 60/120/240 Hz, 120 on 120/
          240): every frame is held for the same number of refreshes (sync interval N). With DXGI (Low delay) the
          display paces the frames through the swap chain's frame latency waitable object, not a timer.
  vsync   VSync on, but the refresh rate isn't a multiple (60 fps on 144 Hz): frames still start on a refresh, so no
          tearing, but some last 2 refreshes and some 3. note says what to change.
  off     VSync off: presented at once, tearing allowed. The lowest delay.
"""
from __future__ import annotations

from dataclasses import dataclass

MAX_INTERVAL = 4  # DXGI's (and WGL's useful) largest sync interval
TOLERANCE_HZ = 1.5  # Windows reports 59.94 Hz as 59, 119.88 as 119


@dataclass(frozen=True)
class SyncPlan:
    mode: str  # "vrr", "locked", "vsync" or "off"
    interval: int  # sync interval for Present / swap_interval: refreshes each frame is held (0: at once)
    tear: bool  # tearing allowed (only with interval 0)
    display_paces: bool  # the display, not the pipeline's timer, spaces the frames (when the presenter can wait)
    note: str = ""  # why frames can't be even, and what would make them so ("" when they are)

    def describe(self, refresh_hz: float, hz: int) -> str:
        if self.mode == "vrr":
            return f"G-Sync/FreeSync: each frame presented as soon as it's ready ({hz} fps on a {refresh_hz:g} Hz display)"
        if self.mode == "locked":
            return (f"locked to the display: {hz} fps on {refresh_hz:g} Hz, each frame held for {self.interval} "
                    f"refresh{'es' if self.interval > 1 else ''}")
        if self.mode == "vsync":
            return f"VSync on a {refresh_hz:g} Hz display that can't show {hz} fps evenly"
        return "VSync off: presented at once, may tear"


def refreshes_per_frame(refresh_hz: float, hz: int) -> int:
    """N when refresh_hz is (about) N x hz for N in 1..MAX_INTERVAL, else 0."""
    if hz <= 0 or refresh_hz <= 0:
        return 0
    n = round(refresh_hz / hz)
    return n if 1 <= n <= MAX_INTERVAL and abs(refresh_hz - n * hz) <= TOLERANCE_HZ * n else 0


def even_rates(hz: int, upto: int = 360) -> list[int]:
    """Refresh rates that show hz fps evenly, e.g. 60 -> [60, 120, 180, 240]."""
    return [hz * n for n in range(1, MAX_INTERVAL + 1) if hz * n <= upto]


def plan(refresh_hz: float, hz: int, vsync: bool, vrr: bool) -> SyncPlan:
    if vrr:
        return SyncPlan("vrr", 0, True, False)
    if not vsync:
        return SyncPlan("off", 0, True, False)
    n = refreshes_per_frame(refresh_hz, hz)
    if n:
        return SyncPlan("locked", n, False, True)
    if refresh_hz < hz - TOLERANCE_HZ:
        note = (f"This display runs at {refresh_hz:g} Hz, slower than {hz} fps output: pick {int(refresh_hz)} fps "
                "or less, or a faster refresh rate in Windows")
    else:
        fits = [r for r in even_rates(hz) if r <= refresh_hz + TOLERANCE_HZ] or [hz]
        note = (f"This display runs at {refresh_hz:g} Hz, which {hz} fps doesn't divide into, so some frames last "
                f"longer than others. Set Windows to {fits[-1]} Hz, or turn on G-Sync / FreeSync if the display has it")
    return SyncPlan("vsync", 1, False, False, note)


class DisplayTimes:
    """When frames actually reached the screen, from DXGI's frame statistics (IDXGISwapChain::GetFrameStatistics),
    not when they were sent: how many refreshes each frame stayed up, and the time from Present to the screen.

    sent(present_count, t) after each Present (t: time.perf_counter()); observe(...) with each statistics sample. A
    sample names the latest Present that reached the screen (PresentCount) and the refresh it went up on
    (PresentRefreshCount); its time is the sample's own refresh (SyncRefreshCount at SyncQPCTime) less the refreshes
    since. With one sample per Present, a frame can come and go between two samples: it's counted in `unseen`, and
    the refreshes across that gap aren't counted in `held`."""

    KEEP = 600  # frames remembered (10 s at 60 fps)

    def __init__(self) -> None:
        self._sent: dict[int, float] = {}
        self._last: tuple[int, int] | None = None  # (PresentCount, PresentRefreshCount) last seen
        self._sync: tuple[int, float] | None = None  # (SyncRefreshCount, its time) last seen, for the refresh period
        self.period_s = 0.0
        self.held: list[int] = []  # refreshes each frame stayed on screen (consecutive frames only)
        self.delay_ms: list[float] = []  # Present call -> on screen
        self.unseen = 0  # frames that went by between two samples (replaced before a sample saw them, or dropped)

    def sent(self, present_count: int, t: float) -> None:
        self._sent[present_count] = t
        if len(self._sent) > self.KEEP:
            for k in sorted(self._sent)[: len(self._sent) - self.KEEP]:
                del self._sent[k]

    def observe(self, present_count: int, present_refresh: int, sync_refresh: int, sync_t: float,
                fallback_period_s: float) -> None:
        if self._sync is not None and sync_refresh > self._sync[0]:
            p = (sync_t - self._sync[1]) / (sync_refresh - self._sync[0])
            if 0.002 < p < 0.05:  # 20-500 Hz: a refresh, not a gap in the samples
                self.period_s = p if not self.period_s else 0.9 * self.period_s + 0.1 * p
        self._sync = (sync_refresh, sync_t)
        if self._last is not None and present_count <= self._last[0]:
            return  # nothing new on screen
        if self._last is not None:
            gap = present_count - self._last[0]
            if gap == 1:
                self._add(self.held, present_refresh - self._last[1])
            else:
                self.unseen += gap - 1
        self._last = (present_count, present_refresh)
        t = self._sent.pop(present_count, None)
        if t is not None:
            shown = sync_t - (sync_refresh - present_refresh) * (self.period_s or fallback_period_s)
            self._add(self.delay_ms, max(0.0, (shown - t) * 1000))

    def _add(self, xs: list, v) -> None:
        xs.append(v)
        if len(xs) > 4 * self.KEEP:
            del xs[: len(xs) - 2 * self.KEEP]

    def summary(self) -> str:
        """One line for the log ("" before any frame was seen)."""
        if not self.held and not self.delay_ms:
            return ""
        parts = []
        if self.held:
            n = len(self.held)
            counts: dict[int, int] = {}
            for h in self.held:
                counts[h] = counts.get(h, 0) + 1
            parts.append("frames held " + ", ".join(
                f"{k} refresh{'es' if k != 1 else ''} {100 * v / n:.0f}%" for k, v in sorted(counts.items())))
        if self.delay_ms:
            d = sorted(self.delay_ms)
            parts.append(f"Present to screen {d[len(d) // 2]:.1f} ms median, {d[int(0.95 * (len(d) - 1))]:.1f} ms "
                         "slowest 5%")
        if self.period_s:
            parts.append(f"display {1 / self.period_s:.2f} Hz")
        if self.unseen:
            parts.append(f"{self.unseen} frames not seen by a sample")
        return "; ".join(parts)
