"""In-video settings menu: keyboard-driven, drawn as an image that the GL window blits over the video.

Rows come from settings.SPECS, so a new setting shows up here without changes to this file. The main page has the
quick picks (Spec.quick choices that apply live: the launcher's big choices) and a row for each section (Spec.group,
as on the launcher's Settings page); Enter or Right opens a section's page, Esc or Backspace goes back. Drawing
happens only when something on the menu changes (the window caches the image), so an open menu costs nothing per
frame.
"""
from __future__ import annotations

from typing import Callable

import cv2
import numpy as np

from .settings import SPECS, Settings, Spec

FONT = cv2.FONT_HERSHEY_SIMPLEX
TITLE_FONT = cv2.FONT_HERSHEY_DUPLEX
# BGR
BG = (34, 30, 28)
ROW_SEL = (70, 58, 48)
ACCENT = (40, 170, 255)
TEXT = (238, 238, 238)
MUTED = (150, 150, 150)
IMPACT = (200, 200, 200)
DIM = (95, 95, 95)
RULE = (70, 66, 62)

# A row: ("setting", Spec), ("section", its name) or ("restart", None)
Row = tuple[str, object]


class Menu:
    """allowed(key, value): whether a choice can be picked now (e.g. RTX VSR when the SDK isn't installed).
    disabled(key): a reason to grey out a whole row, or None. restart_reasons(): labels of settings changed
    since start that need a restart. Changes go through Settings, whose watchers apply them.

    page: the section shown, or None for the main page. It stays when the menu closes, so the player comes back to
    where they were."""

    def __init__(self, settings: Settings, specs: tuple[Spec, ...] = SPECS,
                 allowed: Callable[[str, object], bool] | None = None,
                 disabled: Callable[[str], str | None] | None = None,
                 restart_reasons: Callable[[], list[str]] | None = None) -> None:
        self.settings, self.specs = settings, specs
        self.allowed = allowed or (lambda k, v: True)
        self.disabled = disabled or (lambda k: None)
        self.restart_reasons = restart_reasons or (lambda: [])
        self.open = False
        self.page: str | None = None
        self.index = 0
        self._sel: str | None = None  # the highlighted row's key, and the index it was at (see _follow)
        self._at = 0
        self._left: dict[str, str] = {}  # each section's highlighted row when the player last left it
        self._size_key, self._size = None, (0, 0)  # render(): the longest page and the tallest help, for these rows

    def sections(self) -> list[str]:
        """The sections that have a setting that applies right now, in SPECS order."""
        values = self.settings.values()
        out: list[str] = []
        for s in self.specs:
            if s.group and s.group not in out and s.applies_to(values):
                out.append(s.group)
        return out

    def page_rows(self, page: str | None) -> list[Row]:
        """A page's rows: the settings that apply right now (VFG quality only while NVIDIA VFG is picked, and so on),
        on the main page the quick picks and the sections, and Restart now when something needs it."""
        values = self.settings.values()
        if page is None:
            rows: list[Row] = [("setting", s) for s in self.specs
                               if (s.quick and s.live or not s.group) and not s.hidden and s.applies_to(values)]
            rows += [("section", g) for g in self.sections()]
        else:
            rows = [("setting", s) for s in self.specs if s.group == page and not s.hidden and s.applies_to(values)]
        if self.restart_reasons():
            rows.append(("restart", None))
        return rows

    def rows(self) -> list[Row]:
        """The rows of the page shown (the main page, if its section has gone: nothing in it applies any more)."""
        if self.page is not None and self.page not in self.sections():
            self.page = None
        return self.page_rows(self.page)

    @staticmethod
    def _row_key(row: Row) -> str:
        kind, item = row
        return item.key if kind == "setting" else f"section:{item}" if kind == "section" else kind

    def _follow(self, rows: list[Row]) -> None:
        """Keeps the highlight on the same row when rows appear or go (a model or upscaler change)."""
        keys = [self._row_key(r) for r in rows]
        if self.index == self._at and self._sel in keys:  # (index set from outside: that row, as given)
            self.index = keys.index(self._sel)
        self.index = min(self.index, len(rows) - 1)
        self._sel, self._at = keys[self.index], self.index

    def _show(self, page: str | None) -> None:
        """Opens a section's page (on the row it was left on), or goes back to the main page (on that section)."""
        if self.page is not None:
            self._left[self.page] = self._sel
        want = f"section:{self.page}" if page is None else self._left.get(page)
        self.page = page
        keys = [self._row_key(r) for r in self.rows()]
        self.index = keys.index(want) if want in keys else 0
        self._sel, self._at = keys[self.index], self.index

    def key(self, name: str) -> str | None:
        """name: up / down / left / right / enter / back. Returns "close", "restart" or None."""
        rows = self.rows()
        self._follow(rows)
        if name == "back":
            if self.page is None:
                return "close"
            self._show(None)
            return None
        if name in ("up", "down"):
            self.index = (self.index + (-1 if name == "up" else 1)) % len(rows)
            self._sel, self._at = self._row_key(rows[self.index]), self.index
            return None
        kind, item = rows[self.index]
        if kind == "restart":
            return "restart" if name == "enter" else None
        if kind == "section":
            if name in ("enter", "right"):
                self._show(item)
            return None
        if self.disabled(item.key):
            return None
        if name in ("left", "right", "enter"):
            self.settings.step(item.key, -1 if name == "left" else 1, self.allowed)
        return None

    def state(self) -> tuple:
        """Everything the drawing depends on (the window redraws only when this changes)."""
        return (self.page, self.index, tuple(self.settings.values().items()), tuple(self.restart_reasons()),
                tuple(self.disabled(s.key) for s in self.specs))

    def _section_help(self, section: str) -> str:
        labels = [s.label for kind, s in self.page_rows(section) if kind == "setting"]
        return "In here: " + (", ".join(labels[:-1]) + " and " + labels[-1] if len(labels) > 1 else labels[0]) + "."

    def render(self, ui: float = 1.0) -> np.ndarray:
        """The menu as a BGR image; ui scales it (1.0 suits a 1080p-tall window)."""
        def px(v: float) -> int:
            return max(1, int(round(v * ui)))

        w, pad, row_h, head_h = px(780), px(26), px(46), px(78)
        fs, small = 0.66 * ui, 0.5 * ui
        th = 1 if ui < 1.4 else 2
        rows = self.rows()
        self._follow(rows)

        def foot(kind: str, item) -> tuple[list[str], list[str]]:
            """A row's help under the list, and under that its rough effect on the picture and on speed."""
            impact: list[str] = []
            if kind == "restart":
                text = "Restarts the pipeline with the new settings. The stream keeps running; takes a few seconds."
            elif kind == "section":
                text = self._section_help(item)
            else:
                text = self.disabled(item.key) or item.help
                if not self.disabled(item.key):
                    for k, name in (("picture", "Picture"), ("speed", "Speed")):
                        if item.impact.get(k):
                            impact += _wrap(f"{name}: {item.impact[k]}", w - 2 * pad, small, th)[:2]
            return _wrap(text, w - 2 * pad, small, th)[:4], impact

        def foot_size(lines: tuple[list[str], list[str]]) -> int:
            return (len(lines[0]) + len(lines[1])) * px(26) + (px(8) if lines[1] else 0)

        help_lines, impact_lines = foot(*rows[self.index])
        # The list is as long as the longest page and the help area as tall as the longest help, whatever page and row
        # are shown, so the menu keeps its size and the rows stay where they are while the player goes through it.
        pages = {p: self.page_rows(p) for p in [None] + self.sections()}
        key = (round(ui, 3), tuple((p, tuple((self._row_key(r), self.disabled(r[1].key) if r[0] == "setting" else None)
                                             for r in rs)) for p, rs in pages.items()))
        if self._size_key != key:
            self._size_key = key
            self._size = (max(len(rs) for rs in pages.values()),
                          max(foot_size(foot(*r)) for rs in pages.values() for r in rs))
        n, foot_body = self._size
        foot_h = px(18) + foot_body + px(46)
        if head_h + n * row_h + foot_h > px(1040):  # many rows: closer together, so it fits the window
            row_h = max(px(30), (px(1040) - head_h - foot_h) // n)
        h = head_h + n * row_h + foot_h
        img = np.full((h, w, 3), BG, np.uint8)

        title_scale = 1.05 * ui
        if self.page is None:
            cv2.putText(img, "Settings", (pad, px(50)), TITLE_FONT, title_scale, TEXT, th + 1, cv2.LINE_AA)
            _text_right(img, "M / Esc to close", w - pad, px(48), small, MUTED, th)
        else:  # Settings > Smoothness
            x = pad
            for part, color in (("Settings  >  ", MUTED), (self.page, TEXT)):
                cv2.putText(img, part, (x, px(50)), TITLE_FONT, title_scale, color, th + 1, cv2.LINE_AA)
                x += cv2.getTextSize(part, TITLE_FONT, title_scale, th + 1)[0][0]
            _text_right(img, "Esc: back", w - pad, px(48), small, MUTED, th)
        cv2.line(img, (pad, head_h - px(10)), (w - pad, head_h - px(10)), RULE, max(1, px(1.5)))

        for i, (kind, item) in enumerate(rows):
            y0 = head_h + i * row_h
            base = y0 + int(row_h * 0.66)
            selected = i == self.index
            if selected:
                cv2.rectangle(img, (0, y0), (w, y0 + row_h - px(2)), ROW_SEL, -1)
                cv2.rectangle(img, (0, y0), (px(6), y0 + row_h - px(2)), ACCENT, -1)
            if kind == "restart":
                reasons = ", ".join(self.restart_reasons())
                cv2.putText(img, f"Restart now  (applies: {reasons})", (pad, base), FONT, fs, ACCENT, th,
                            cv2.LINE_AA)
                continue
            if kind == "section":
                if i and rows[i - 1][0] == "setting":  # the quick picks above, the sections below
                    cv2.line(img, (pad, y0), (w - pad, y0), RULE, max(1, px(1.5)))
                cv2.putText(img, item, (pad, base), FONT, fs, TEXT, th, cv2.LINE_AA)
                _text_right(img, ">", w - pad, base, fs, ACCENT if selected else MUTED, th)
                continue
            spec = item
            off = self.disabled(spec.key) is not None
            color = DIM if off else TEXT
            cv2.putText(img, spec.label, (pad, base), FONT, fs, color, th, cv2.LINE_AA)
            tag_x = pad + cv2.getTextSize(spec.label, FONT, fs, th)[0][0] + px(12)
            if spec.experimental:  # built without the hardware to test it (as the launcher's EXPERIMENTAL badge)
                cv2.putText(img, "experimental", (tag_x, base), FONT, small * 0.9, DIM if off else ACCENT, th,
                            cv2.LINE_AA)
                tag_x += cv2.getTextSize("experimental", FONT, small * 0.9, th)[0][0] + px(10)
            if not spec.live:
                cv2.putText(img, "restart", (tag_x, base), FONT, small * 0.9, DIM if off else MUTED,
                            th, cv2.LINE_AA)
            value = "not available" if off else spec.name(getattr(self.settings, spec.key))
            right = w - pad
            if selected and not off:
                _text_right(img, ">", right, base, fs, ACCENT, th)
                right -= px(30)
                vw = _text_right(img, value, right, base, fs, TEXT, th)
                _text_right(img, "<", right - vw - px(14), base, fs, ACCENT, th)
            else:
                _text_right(img, value, right - (px(30) if not off else 0), base, fs, MUTED if not off else DIM, th)

        y = head_h + n * row_h
        cv2.line(img, (pad, y + px(4)), (w - pad, y + px(4)), RULE, max(1, px(1.5)))
        y += px(18)
        for line in help_lines:
            y += px(26)
            cv2.putText(img, line, (pad, y), FONT, small, MUTED, th, cv2.LINE_AA)
        y += px(8) if impact_lines else 0
        for line in impact_lines:
            y += px(26)
            cv2.putText(img, line, (pad, y), FONT, small, IMPACT, th, cv2.LINE_AA)
        cv2.putText(img, KEYS[self.page is not None], (pad, h - px(16)), FONT, small * 0.9, DIM, th, cv2.LINE_AA)
        return img


# The keys, at the bottom: on the main page, and on a section's page
KEYS = ("Up/Down select   Left/Right change   Enter open   M close   Q quit",
        "Up/Down select   Left/Right change   Esc back   M close   Q quit")


def _text_right(img: np.ndarray, text: str, right: int, base: int, scale: float, color, th: int) -> int:
    tw = cv2.getTextSize(text, FONT, scale, th)[0][0]
    cv2.putText(img, text, (right - tw, base), FONT, scale, color, th, cv2.LINE_AA)
    return tw


def _wrap(text: str, width: int, scale: float, th: int) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if line and cv2.getTextSize(trial, FONT, scale, th)[0][0] > width:
            lines.append(line)
            line = word
        else:
            line = trial
    return lines + ([line] if line else [])
