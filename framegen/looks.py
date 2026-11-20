"""Looks: optional effects on the final picture, stacked like the nodes of a color grading app.

A look is a name and a list of nodes, applied in order. Each node is one effect (color, lift / gamma / gain, split
tone, black & white, LUT, bloom, halation, blur, clarity, chromatic aberration, sharpen, film grain, vignette,
scanlines, letterbox, color-blind assist), with an on/off switch, an amount ("mix": how much of its result replaces
what came in) and its own settings. As JSON (the saved files, and what the launcher sends):

    {"name": "Cinematic", "nodes": [{"type": "color", "on": true, "mix": 1.0, "contrast": 0.25}, ...]}

Settings left out are the type's defaults. framegen/looks_gl.py draws a look in the video window, after upscaling, and
at the stream's size for instant replay clips ("Looks in clips"). This module is the rest:

- TYPES: the node types and their settings, the one table the launcher's Looks page, the share codes and both
  renderers use. A type's `code` and its settings' order are part of the share-code format: never change or reuse
  them, only add.
- BUILTIN: the looks SmoovPlay comes with. Library: those plus the user's own (%APPDATA%\\framegen\\looks\\*.json,
  ids "my:<file name>"), and their .cube LUTs (looks\\luts).
- encode() / decode(): share codes, "SPL1-" and URL-safe base64 of the settings that differ from the defaults
  (quantised to each setting's step), with a checksum. A look of five nodes is about 60 characters.
- render(): the reference renderer, in NumPy, that tests/test_looks.py holds the shaders to. stages() splits a look
  into the passes both renderers draw: nodes that only need their own pixel run together in one pass; a node that
  reads other pixels (blur, bloom, sharpen...) starts a new pass, which reads the one before it.

`python -m framegen.looks schema|encode|decode|import|save|delete|add-lut|preview` is the launcher's way in (main()).
"""
from __future__ import annotations

import base64
import json
import math
import os
import re
import sys
import zlib
from dataclasses import dataclass
from pathlib import Path, PureWindowsPath

PREFIX = "SPL1-"
MAX_NODES = 16
MAX_LUTS = 4  # LUT nodes a look can use at once (texture units)
MAX_NAME = 48  # bytes of UTF-8 in a share code
W709 = (0.2126, 0.7152, 0.0722)  # luma weights


def looks_dir() -> Path:
    env = os.environ.get("SMOOVPLAY_LOOKS_DIR")
    return Path(env) if env else Path(os.environ.get("APPDATA", Path.home())) / "framegen" / "looks"


# ---------------------------------------------------------------------------------------------- the node types
@dataclass(frozen=True)
class Param:
    key: str
    label: str
    lo: float
    hi: float
    default: float | str
    step: float = 0.01
    unit: str = ""
    names: tuple = ()  # a choice: the names of lo, lo + 1, ...
    text: bool = False  # a file name (the LUT's), not a number

    def clean(self, v):
        if self.text:
            return _lut_name(v)
        try:
            v = float(v)
        except (TypeError, ValueError):
            return self.default
        if not math.isfinite(v):
            return self.default
        q = round((min(max(v, self.lo), self.hi) - self.lo) / self.step)
        return round(self.lo + q * self.step, 6)

    def q(self, v) -> int:
        return round((v - self.lo) / self.step)


MIX = Param("mix", "Amount", 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class NodeType:
    key: str
    code: int  # its number in share codes
    label: str
    kind: str  # "point": only its own pixel; "glow", "blur", "clarity": a blurred copy; "ca", "sharpen": neighbours
    help: str
    params: tuple[Param, ...]

    @property
    def fields(self) -> tuple[Param, ...]:
        return (MIX, *self.params)


def _p(key, label, lo, hi, default, step=0.01, unit="", names=()):
    return Param(key, label, float(lo), float(hi), float(default), float(step), unit, tuple(names))


_HUE = dict(lo=0, hi=360, step=1, unit="°")

TYPES_LIST: tuple[NodeType, ...] = (
    NodeType("color", 1, "Color", "point",
             "The basics: brightness, contrast, how colorful, and warm or cool.",
             (_p("exposure", "Exposure", -3, 3, 0, 0.05, "stops"), _p("contrast", "Contrast", -1, 1, 0),
              _p("shadows", "Shadows", -1, 1, 0), _p("highlights", "Highlights", -1, 1, 0),
              _p("fade", "Fade", 0, 1, 0), _p("saturation", "Saturation", 0, 2, 1),
              _p("vibrance", "Vibrance", -1, 1, 0), _p("temperature", "Temperature", -1, 1, 0),
              _p("tint", "Tint", -1, 1, 0))),
    NodeType("wheels", 2, "Lift / gamma / gain", "point",
             "Grading wheels: lift moves the darks, gamma the middle, gain the brights, each brighter or darker and "
             "toward a color.",
             (_p("lift", "Lift", -1, 1, 0), _p("lift_hue", "Lift color", default=210, **_HUE),
              _p("lift_amount", "Lift color amount", 0, 1, 0),
              _p("gamma", "Gamma", -1, 1, 0), _p("gamma_hue", "Gamma color", default=40, **_HUE),
              _p("gamma_amount", "Gamma color amount", 0, 1, 0),
              _p("gain", "Gain", -1, 1, 0), _p("gain_hue", "Gain color", default=40, **_HUE),
              _p("gain_amount", "Gain color amount", 0, 1, 0))),
    NodeType("split", 3, "Split tone", "point",
             "One color in the shadows and another in the highlights, like teal and orange.",
             (_p("shadow_hue", "Shadow color", default=200, **_HUE), _p("shadow_amount", "Shadow amount", 0, 1, 0),
              _p("highlight_hue", "Highlight color", default=35, **_HUE),
              _p("highlight_amount", "Highlight amount", 0, 1, 0), _p("balance", "Balance", -1, 1, 0))),
    NodeType("mono", 4, "Black & white", "point",
             "Black and white, with how bright each color comes out, and an optional tint like sepia.",
             (_p("red", "Reds", -1, 2, 0.30), _p("green", "Greens", -1, 2, 0.59), _p("blue", "Blues", -1, 2, 0.11),
              _p("tone_hue", "Tint color", default=35, **_HUE), _p("tone_amount", "Tint amount", 0, 1, 0))),
    NodeType("lut", 5, "LUT", "point",
             "A color grade from a .cube LUT file, as made in DaVinci Resolve, Photoshop and other apps. Amount "
             "blends it with the picture.",
             (Param("file", "LUT file", 0, 0, "", text=True),)),
    NodeType("bloom", 6, "Bloom", "glow",
             "A soft glow around bright parts of the picture.",
             (_p("amount", "Strength", 0, 2, 0.5), _p("threshold", "Threshold", 0, 1, 0.7),
              _p("size", "Size", 0, 1, 0.6), _p("warmth", "Warmth", -1, 1, 0))),
    NodeType("halation", 7, "Halation", "glow",
             "The red-orange glow film gets around bright edges.",
             (_p("amount", "Strength", 0, 2, 0.6), _p("threshold", "Threshold", 0, 1, 0.6),
              _p("size", "Size", 0, 1, 0.3), _p("hue", "Color", default=10, **_HUE))),
    NodeType("blur", 8, "Blur", "blur",
             "Softens the picture. Amount blends it with the sharp one, for a dreamy, soft-focus look.",
             (_p("size", "Size", 0, 1, 0.3),)),
    NodeType("clarity", 9, "Clarity", "clarity",
             "Brings out texture and detail (or, below zero, smooths it), without the halos of sharpening.",
             (_p("amount", "Strength", -1, 1, 0.3), _p("size", "Size", 0, 1, 0.4))),
    NodeType("ca", 10, "Chromatic aberration", "ca",
             "Color fringes toward the edges of the picture, like a cheap lens.",
             (_p("amount", "Strength", 0, 1, 0.3),)),
    NodeType("sharpen", 11, "Sharpen", "sharpen",
             "Crisper edges. Below zero, softer.",
             (_p("amount", "Strength", -1, 2, 0.5),)),
    NodeType("grain", 12, "Film grain", "point",
             "Moving film grain, mostly in the mid-tones.",
             (_p("amount", "Strength", 0, 1, 0.3), _p("size", "Size", 0.5, 3, 1.2, 0.05, "px"),
              _p("color", "Color", 0, 1, 0.2))),
    NodeType("vignette", 13, "Vignette", "point",
             "Darker corners (or, below zero, brighter ones).",
             (_p("amount", "Strength", -1, 1, 0.35), _p("size", "Size", 0, 1, 0.5),
              _p("softness", "Softness", 0, 1, 0.5), _p("roundness", "Roundness", 0, 1, 0.5))),
    NodeType("scanlines", 14, "Scanlines", "point",
             "The lines of an old TV.",
             (_p("amount", "Strength", 0, 1, 0.3), _p("size", "Size", 1, 8, 3, 0.5, "px"))),
    NodeType("letterbox", 15, "Letterbox", "point",
             "Black bars for a wider, film-shaped picture. They cover part of the game's picture, HUD included.",
             (_p("aspect", "Shape", 1.33, 3, 2.39, 0.01, ":1"),)),
    NodeType("assist", 16, "Color-blind assist", "point",
             "Moves the colors you can't tell apart toward ones you can (daltonization).",
             (_p("kind", "Type", 0, 2, 1, 1,
                 names=("Protanopia (red)", "Deuteranopia (green)", "Tritanopia (blue)")),)),
)
TYPES = {t.key: t for t in TYPES_LIST}
BY_CODE = {t.code: t for t in TYPES_LIST}


def _lut_name(v) -> str:
    """A LUT's file name, without folders (a share code or a file can't point anywhere else). Windows' rules on every
    system (the same as before there): a share code made on Windows can carry backslashes, which elsewhere aren't
    folder separators."""
    name = PureWindowsPath(str(v or "")).name.strip()
    return name[:120] if name.lower().endswith(".cube") else ""


# ---------------------------------------------------------------------------------------------- looks as data
def node(type_: str, on: bool = True, mix: float = 1.0, **params) -> dict:
    return {"type": type_, "on": on, "mix": mix, **params}


def normalize(look) -> dict:
    """A clean copy: known types only, every setting present, in range and on its step. Raises ValueError if it
    isn't a look at all."""
    if not isinstance(look, dict) or not isinstance(look.get("nodes", []), list):
        raise ValueError("not a look")
    name = " ".join(str(look.get("name") or "My look").split())[:60] or "My look"
    nodes = []
    for n in look.get("nodes", []):
        t = TYPES.get(n.get("type")) if isinstance(n, dict) else None
        if t is None:
            continue
        out = {"type": t.key, "on": bool(n.get("on", True))}
        for f in t.fields:
            out[f.key] = f.clean(n.get(f.key, f.default))
        nodes.append(out)
        if len(nodes) == MAX_NODES:
            break
    return {"name": name, "nodes": nodes}


def active_nodes(look: dict, has_lut=lambda name: True) -> list[dict]:
    """The nodes that do something: on, with an amount above zero, and for a LUT, a file that's there."""
    return [n for n in look["nodes"] if n["on"] and n["mix"] > 0
            and (n["type"] != "lut" or (n["file"] and has_lut(n["file"])))]


@dataclass
class Stage:
    start: dict | None  # a node that reads other pixels of the pass before (None: the first pass)
    nodes: list  # nodes that need only their own pixel, after it


def stages(look: dict, has_lut=lambda name: True) -> list[Stage]:
    """The passes that draw a look; [] if it does nothing. The first never starts with a node that reads other
    pixels: it brings the picture to the output's size (and the right way up), so every later pass reads an image of
    exactly the output's size."""
    nodes = active_nodes(look, has_lut)
    if not nodes:
        return []
    out = [Stage(None, [])]
    luts = 0
    for n in nodes:
        if n["type"] == "lut":
            luts += 1
            if luts > MAX_LUTS:
                continue
        if TYPES[n["type"]].kind == "point":
            out[-1].nodes.append(n)
        else:
            out.append(Stage(n, []))
    return out


# ---- what each node's shader gets (both renderers use these, so they can't drift apart)
def _hsv(deg: float) -> tuple[float, float, float]:
    h = (deg % 360.0) / 60.0
    x = 1.0 - abs(h % 2.0 - 1.0)
    return ((1, x, 0), (x, 1, 0), (0, 1, x), (0, x, 1), (x, 0, 1), (1, 0, x))[int(h) % 6]


def hue_vec(deg: float) -> tuple[float, float, float]:
    """A push toward a hue that leaves brightness (luma) alone; its largest part is 1."""
    rgb = _hsv(deg)
    y = sum(c * w for c, w in zip(rgb, W709))
    v = [c - y for c in rgb]
    m = max(abs(c) for c in v)
    return tuple(c / m for c in v)


def _scale(v, k):
    return tuple(k * c for c in v)


def _add(*vs):
    return tuple(sum(c) for c in zip(*vs))


# Machado, Oliveira & Fernandes (2009), severity 1, linear RGB; Fidaner et al.'s daltonization moves the lost part
# into channels that are still seen.
_CVD = (((0.152286, 1.052583, -0.204868), (0.114503, 0.786281, 0.099216), (-0.003882, -0.048116, 1.051998)),
        ((0.367322, 0.860646, -0.227968), (0.280085, 0.672501, 0.047413), (-0.011820, 0.042940, 0.968881)),
        ((1.255528, -0.076749, -0.178779), (-0.078411, 0.930809, 0.147602), (0.004733, 0.691367, 0.303900)))
_SHIFT = (((0, 0, 0), (0.7, 1, 0), (0.7, 0, 1)), ((0, 0, 0), (0.7, 1, 0), (0.7, 0, 1)),
          ((1, 0, 0.7), (0, 1, 0.7), (0, 0, 0)))


def assist_matrix(kind: int) -> tuple:
    """Row-major 3x3, linear RGB: I + E (I - S)."""
    s, e = _CVD[kind], _SHIFT[kind]
    lost = [[(1.0 if i == j else 0.0) - s[i][j] for j in range(3)] for i in range(3)]
    return tuple((1.0 if i == j else 0.0) + sum(e[i][k] * lost[k][j] for k in range(3))
                 for i in range(3) for j in range(3))


def uniforms(n: dict, lut=None) -> list[tuple[str, object]]:
    """A node's shader inputs, in the order its shader function takes them: floats, 3-tuples (vec3) and 9-tuples
    (mat3, row-major). lut: its Lut, for a LUT node."""
    t = n["type"]
    if t == "color":
        temp, tint = n["temperature"], n["tint"]
        wb = (1 + 0.18 * temp, 1 - 0.12 * tint, 1 - 0.18 * temp)
        y = sum(c * w for c, w in zip(wb, W709))
        return [("wb", _scale(wb, 1 / y)), ("gain", 2 ** (n["exposure"] / 2.2)), ("k", n["contrast"]),
                ("sh", 0.945 * n["shadows"]), ("hi", 0.945 * n["highlights"]), ("fade", 0.16 * n["fade"]),
                ("sat", n["saturation"]), ("vib", n["vibrance"])]
    if t == "wheels":
        one = (1.0, 1.0, 1.0)
        lift = _add(_scale(one, 0.25 * n["lift"]), _scale(hue_vec(n["lift_hue"]), 0.15 * n["lift_amount"]))
        gain = _add(one, _scale(one, 0.5 * n["gain"]), _scale(hue_vec(n["gain_hue"]), 0.25 * n["gain_amount"]))
        g = _add(_scale(one, 0.5 * n["gamma"]), _scale(hue_vec(n["gamma_hue"]), 0.3 * n["gamma_amount"]))
        return [("lift", lift), ("gain", gain), ("power", tuple(2.0 ** -c for c in g))]
    if t == "split":
        return [("sh", _scale(hue_vec(n["shadow_hue"]), 0.2 * n["shadow_amount"])),
                ("hi", _scale(hue_vec(n["highlight_hue"]), 0.2 * n["highlight_amount"])), ("bal", n["balance"])]
    if t == "mono":
        w = (n["red"], n["green"], n["blue"])
        total = sum(w)
        w = _scale(w, 1 / total) if total > 0.05 else W709
        return [("w", w), ("tone", _scale(hue_vec(n["tone_hue"]), 0.3 * n["tone_amount"]))]
    if t == "lut":
        dmin, dmax = (lut.dmin, lut.dmax) if lut is not None else ((0.0,) * 3, (1.0,) * 3)
        return [("dmin", dmin), ("dscale", tuple(1 / max(b - a, 1e-6) for a, b in zip(dmin, dmax))),
                ("n", float(lut.size if lut is not None else 2))]
    if t in ("bloom", "halation"):
        if t == "bloom":
            tint = (1 + 0.25 * n["warmth"], 1.0, 1 - 0.25 * n["warmth"])
        else:
            tint = tuple(0.15 + 0.85 * c for c in _hsv(n["hue"]))
        return [("amount", n["amount"]), ("tint", _scale(tint, 1 / max(tint)))]
    if t == "clarity":
        return [("amount", n["amount"])]
    if t == "ca":
        return [("k", 0.012 * n["amount"])]
    if t == "sharpen":
        return [("amount", n["amount"])]
    if t == "grain":
        return [("a", 0.15 * n["amount"]), ("size", n["size"]), ("col", n["color"])]
    if t == "vignette":
        a = n["amount"]
        return [("a", a if a >= 0 else 0.5 * a), ("r0", 1.0 - 0.6 * n["size"]), ("soft", 0.05 + 0.5 * n["softness"]),
                ("round", n["roundness"])]
    if t == "scanlines":
        return [("a", 0.6 * n["amount"]), ("size", n["size"])]
    if t == "letterbox":
        return [("aspect", n["aspect"])]
    if t == "assist":
        return [("m", assist_matrix(int(n["kind"])))]
    return []  # blur: everything is in its blurred copy


def glow(n: dict) -> tuple[float | None, float]:
    """For a node that uses a blurred copy: (its bright-pass threshold, or None for a plain blur; how much the
    wider levels count)."""
    thr = n["threshold"] if TYPES[n["type"]].kind == "glow" else None
    return thr, 0.15 + 0.8 * n["size"]


# ---------------------------------------------------------------------------------------------- built-in looks
def _look(name: str, *nodes: dict) -> dict:
    return normalize({"name": name, "nodes": list(nodes)})


BUILTIN: dict[str, dict] = {
    "cinematic": _look("Cinematic",
                       node("color", contrast=0.25, highlights=-0.3, shadows=0.1, saturation=0.9, temperature=0.05),
                       node("split", shadow_hue=195, shadow_amount=0.5, highlight_hue=30, highlight_amount=0.45),
                       node("bloom", amount=0.25, threshold=0.75, size=0.6),
                       node("vignette", amount=0.3, size=0.5, softness=0.6),
                       node("letterbox", on=False, aspect=2.39)),
    "film": _look("Film",
                  node("color", contrast=0.15, highlights=-0.2, fade=0.4, saturation=0.85, temperature=0.1),
                  node("split", shadow_hue=200, shadow_amount=0.2, highlight_hue=40, highlight_amount=0.25),
                  node("halation", amount=0.5, threshold=0.65, size=0.3, hue=12),
                  node("grain", amount=0.35, size=1.3, color=0.25),
                  node("vignette", amount=0.2)),
    "vivid": _look("Vivid",
                   node("color", contrast=0.2, shadows=0.15, saturation=1.1, vibrance=0.45),
                   node("clarity", amount=0.35, size=0.4),
                   node("sharpen", amount=0.3)),
    "noir": _look("Noir",
                  node("mono", red=0.45, green=0.45, blue=0.1),
                  node("color", exposure=-0.1, contrast=0.5, shadows=-0.2),
                  node("bloom", amount=0.2, threshold=0.8, size=0.5),
                  node("grain", amount=0.4, size=1.4, color=0),
                  node("vignette", amount=0.45, size=0.55, softness=0.5)),
    "golden": _look("Golden hour",
                    node("color", contrast=0.1, highlights=-0.15, saturation=1.05, temperature=0.45, tint=0.1),
                    node("split", shadow_hue=260, shadow_amount=0.15, highlight_hue=40, highlight_amount=0.4),
                    node("bloom", amount=0.45, threshold=0.6, size=0.75, warmth=0.6),
                    node("vignette", amount=0.25)),
    "neon": _look("Neon night",
                  node("color", exposure=-0.15, contrast=0.3, saturation=1.25, temperature=-0.2),
                  node("split", shadow_hue=240, shadow_amount=0.4, highlight_hue=320, highlight_amount=0.35),
                  node("bloom", amount=0.8, threshold=0.55, size=0.7),
                  node("ca", amount=0.35),
                  node("vignette", amount=0.35)),
    "vhs": _look("Retro VHS",
                 node("color", contrast=-0.1, fade=0.5, saturation=0.8, temperature=0.05, tint=0.15),
                 node("blur", mix=0.35, size=0.1),
                 node("ca", amount=0.8),
                 node("scanlines", amount=0.35, size=3),
                 node("grain", amount=0.5, size=1.0, color=0.6)),
    "bleach": _look("Bleach bypass",
                    node("color", contrast=0.45, shadows=-0.15, highlights=0.1, saturation=0.45),
                    node("grain", amount=0.2),
                    node("sharpen", amount=0.2)),
    "dreamy": _look("Dreamy",
                    node("color", exposure=0.1, contrast=-0.15, fade=0.3, saturation=0.9),
                    node("split", shadow_hue=220, shadow_amount=0.15, highlight_hue=330, highlight_amount=0.2),
                    node("blur", mix=0.25, size=0.5),
                    node("bloom", amount=0.6, threshold=0.45, size=0.85)),
    "arctic": _look("Arctic",
                    node("color", contrast=0.2, highlights=0.1, saturation=0.75, temperature=-0.5),
                    node("split", shadow_hue=210, shadow_amount=0.3, highlight_hue=190, highlight_amount=0.2),
                    node("sharpen", amount=0.25)),
    "sepia": _look("Sepia",
                   node("mono", tone_hue=35, tone_amount=0.7),
                   node("color", contrast=0.2, fade=0.25),
                   node("grain", amount=0.35),
                   node("vignette", amount=0.4)),
    "assist-protan": _look("Color assist: protanopia", node("assist", kind=0)),
    "assist-deutan": _look("Color assist: deuteranopia", node("assist", kind=1)),
    "assist-tritan": _look("Color assist: tritanopia", node("assist", kind=2)),
}
DEFAULT_LOOK = "cinematic"

# The "Which look" setting's choices and names (framegen/settings.py): the built-in looks, then the user's, by name.
# Library.reload() keeps them up to date in place, so the setting, the menu and the launcher see new looks at once.
CHOICES: list[str] = list(BUILTIN)
NAMES: dict[str, str] = {k: v["name"] for k, v in BUILTIN.items()}


# ---------------------------------------------------------------------------------------------- share codes
class CodeError(ValueError):
    """A share code that can't be read; the message says why, for the user."""


def _varint(out: bytearray, v: int) -> None:
    while True:
        b = v & 0x7F
        v >>= 7
        out.append(b | (0x80 if v else 0))
        if not v:
            return


def _zz(v: int) -> int:
    return v * 2 if v >= 0 else -v * 2 - 1


def _unzz(v: int) -> int:
    return v // 2 if v % 2 == 0 else -(v + 1) // 2


def _cut_utf8(text: str, limit: int) -> bytes:
    b = text.encode("utf-8")[:limit]
    return b.decode("utf-8", errors="ignore").encode("utf-8")


def encode(look: dict) -> str:
    """The look's share code."""
    look = normalize(look)
    out = bytearray()
    name = _cut_utf8(look["name"], MAX_NAME)
    _varint(out, len(name))
    out += name
    _varint(out, len(look["nodes"]))
    for n in look["nodes"]:
        t = TYPES[n["type"]]
        _varint(out, t.code << 1 | (0 if n["on"] else 1))
        mask, vals = 0, bytearray()
        for i, f in enumerate(t.fields):
            v = n[f.key]
            if f.text:
                if v:
                    mask |= 1 << i
                    b = _cut_utf8(v, 120)
                    _varint(vals, len(b))
                    vals += b
            elif f.q(v) != f.q(f.default):
                mask |= 1 << i
                _varint(vals, _zz(f.q(v) - f.q(f.default)))
        _varint(out, mask)
        out += vals
    body = bytes(out)
    pack = zlib.compressobj(9, zlib.DEFLATED, -15)
    packed = pack.compress(body) + pack.flush()
    data = bytes([1]) + packed if len(packed) < len(body) else bytes([0]) + body
    data += (zlib.crc32(data) & 0xFFFF).to_bytes(2, "big")
    return PREFIX + base64.urlsafe_b64encode(data).rstrip(b"=").decode("ascii")


def decode(code: str) -> dict:
    """The look in a share code. Raises CodeError, with a message for the user."""
    text = "".join(str(code or "").split())  # chat apps wrap long lines
    m = re.fullmatch(r"(?i)SPL(\d+)-([A-Za-z0-9_-]+)", text)
    if not m:
        raise CodeError("That isn't a SmoovPlay look code. Look codes start with SPL1-.")
    if m.group(1) != "1":
        raise CodeError("That look code is from a newer version of SmoovPlay. Update SmoovPlay to use it.")
    try:
        data = base64.urlsafe_b64decode(m.group(2) + "=" * (-len(m.group(2)) % 4))
    except ValueError:
        data = b""
    if len(data) < 4 or (zlib.crc32(data[:-2]) & 0xFFFF).to_bytes(2, "big") != data[-2:]:
        raise CodeError("That look code is incomplete or has a typo. Copy it again, all of it.")
    flags, body = data[0], data[1:-2]
    if flags & ~1:
        raise CodeError("That look code is from a newer version of SmoovPlay. Update SmoovPlay to use it.")
    try:
        if flags & 1:
            body = zlib.decompress(body, -15)
        return _parse(body)
    except CodeError:
        raise
    except (IndexError, ValueError, zlib.error):
        raise CodeError("That look code is damaged. Copy it again, all of it.") from None


def _parse(body: bytes) -> dict:
    pos = 0

    def varint() -> int:
        nonlocal pos
        v = shift = 0
        while True:
            b = body[pos]
            pos += 1
            v |= (b & 0x7F) << shift
            shift += 7
            if not b & 0x80:
                return v
            if shift > 63:
                raise ValueError("varint too long")

    def text() -> str:
        nonlocal pos
        n = varint()
        s = body[pos:pos + n].decode("utf-8")
        if len(body) < pos + n:
            raise ValueError("cut short")
        pos += n
        return s

    name = text()
    nodes = []
    for _ in range(varint()):
        head = varint()
        t = BY_CODE.get(head >> 1)
        if t is None:
            raise CodeError("That look uses an effect this version of SmoovPlay doesn't have. Update SmoovPlay "
                            "to use it.")
        n = node(t.key, on=not head & 1)
        mask = varint()
        i = 0
        while mask >> i:
            if mask >> i & 1:
                f = t.fields[i] if i < len(t.fields) else None
                if f is not None and f.text:
                    n[f.key] = text()
                else:
                    d = _unzz(varint())  # (a setting from a newer version: skipped)
                    if f is not None:
                        n[f.key] = f.lo + (f.q(f.default) + d) * f.step
            i += 1
        nodes.append(n)
    if pos != len(body):
        raise ValueError("extra bytes")
    return normalize({"name": name, "nodes": nodes})


# ---------------------------------------------------------------------------------------------- LUT files
@dataclass
class Lut:
    size: int
    table: object  # float32 (N, N, N, 3), indexed [b, g, r]
    dmin: tuple
    dmax: tuple
    title: str = ""


def load_cube(path) -> Lut:
    """A 3D .cube LUT (Adobe/Resolve format). Raises ValueError with a message for the user."""
    import numpy as np
    size, dmin, dmax, title, rows = 0, (0.0,) * 3, (1.0,) * 3, "", []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError as e:
        raise ValueError(f"couldn't read it ({e.strerror or e})") from None
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip()
        if not line:
            continue
        word = line.split(None, 1)[0].upper()
        if word == "TITLE":
            title = line.split(None, 1)[1].strip().strip('"') if " " in line else ""
        elif word == "LUT_3D_SIZE":
            size = int(line.split()[1])
        elif word == "LUT_1D_SIZE":
            raise ValueError("it's a 1D LUT; SmoovPlay uses 3D LUTs (LUT_3D_SIZE)")
        elif word == "DOMAIN_MIN":
            dmin = tuple(float(v) for v in line.split()[1:4])
        elif word == "DOMAIN_MAX":
            dmax = tuple(float(v) for v in line.split()[1:4])
        elif word[0].isdigit() or word[0] in "-+.":
            rows.append(line.split()[:3])
        # (other keywords, e.g. LUT_3D_INPUT_RANGE, are ignored)
    if not 2 <= size <= 129:
        raise ValueError("it has no LUT_3D_SIZE between 2 and 129")
    if len(rows) != size ** 3:
        raise ValueError(f"it should have {size ** 3} rows of colors for size {size}, but has {len(rows)}")
    try:
        table = np.array(rows, dtype=np.float32).reshape(size, size, size, 3)
    except ValueError:
        raise ValueError("some of its rows aren't three numbers") from None
    if not np.isfinite(table).all():
        raise ValueError("it has colors that aren't numbers")
    return Lut(size, table, dmin, dmax, title)


# ---------------------------------------------------------------------------------------------- the library
def _slug(name: str) -> str:
    s = re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")[:40]
    return s or "look"


class Library:
    """The built-in looks and the user's (folder: %APPDATA%\\framegen\\looks). Ids: the built-in keys, and
    "my:<file name without .json>" for the user's."""

    def __init__(self, folder=None) -> None:
        self.folder = Path(folder) if folder is not None else looks_dir()
        self.user: dict[str, dict] = {}
        self._luts: dict[str, tuple[float, Lut | None]] = {}
        self.reload()

    @property
    def lut_folder(self) -> Path:
        return self.folder / "luts"

    def reload(self) -> None:
        """Read the user's looks again (the launcher saved or deleted one), and update the setting's choices."""
        user = {}
        try:
            files = sorted(self.folder.glob("*.json"))
        except OSError:
            files = []
        for p in files:
            try:
                user["my:" + p.stem] = normalize(json.loads(p.read_text(encoding="utf-8")))
            except (OSError, ValueError) as e:
                print(f"[looks] skipped {p.name}: {e}")
        self.user = user
        CHOICES[:] = list(BUILTIN) + sorted(user, key=lambda i: (user[i]["name"].lower(), i))
        NAMES.clear()
        NAMES.update({k: v["name"] for k, v in BUILTIN.items()})
        NAMES.update({k: v["name"] for k, v in user.items()})

    def ids(self) -> list[str]:
        return list(CHOICES)

    def get(self, look_id: str) -> dict | None:
        return BUILTIN.get(look_id) or self.user.get(look_id)

    def path(self, look_id: str) -> Path | None:
        return self.folder / (look_id[3:] + ".json") if look_id.startswith("my:") and "/" not in look_id \
            and "\\" not in look_id else None

    def save(self, look: dict, look_id: str | None = None) -> str:
        """Writes a user look (a new file unless look_id is one of the user's); returns its id."""
        look = normalize(look)
        path = self.path(look_id) if look_id else None
        if path is None:
            stem, i = _slug(look["name"]), 1
            while (self.folder / f"{stem}.json").exists():
                i += 1
                stem = f"{_slug(look['name'])}-{i}"
            path = self.folder / f"{stem}.json"
        self.folder.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".json.part")
        tmp.write_text(json.dumps({"version": 1, **look}, indent=1), encoding="utf-8")
        os.replace(tmp, path)
        self.reload()
        return "my:" + path.stem

    def delete(self, look_id: str) -> bool:
        path = self.path(look_id)
        if path is None or not path.exists():
            return False
        path.unlink()
        self.reload()
        return True

    def luts(self) -> list[str]:
        try:
            return sorted(p.name for p in self.lut_folder.glob("*.cube"))
        except OSError:
            return []

    def lut(self, name: str) -> Lut | None:
        """A LUT from the luts folder, read once (and again if the file changes); None if it's missing or broken."""
        path = self.lut_folder / _lut_name(name) if _lut_name(name) else None
        try:
            mtime = path.stat().st_mtime if path is not None else None
        except OSError:
            mtime = None
        if mtime is None:
            return None
        got = self._luts.get(name)
        if got is None or got[0] != mtime:
            try:
                got = (mtime, load_cube(path))
            except ValueError as e:
                print(f"[looks] LUT {name}: {e}")
                got = (mtime, None)
            self._luts[name] = got
        return got[1]

    def add_lut(self, src) -> str:
        """Copies a .cube file into the luts folder (after checking it); returns its name. ValueError if it's not one
        SmoovPlay can use."""
        src = Path(src)
        if src.suffix.lower() != ".cube":
            raise ValueError("SmoovPlay reads LUTs in the .cube format")
        load_cube(src)
        self.lut_folder.mkdir(parents=True, exist_ok=True)
        name = _lut_name(src.name)
        dst = self.lut_folder / name
        if src.resolve() != dst.resolve():
            dst.write_bytes(src.read_bytes())
        self._luts.pop(name, None)
        return name


Library()  # the setting's choices include the user's looks from the start


# ---------------------------------------------------------------------------------------------- reference renderer
def _f16(a):
    import numpy as np
    return a.astype(np.float16).astype(np.float32)


def _resize(img, h: int, w: int):
    """Bilinear, as GL samples a texture at pixel centers (clamped at the edges)."""
    import numpy as np
    ih, iw = img.shape[:2]
    ys = np.clip((np.arange(h, dtype=np.float32) + 0.5) * (ih / h) - 0.5, 0, ih - 1)
    xs = np.clip((np.arange(w, dtype=np.float32) + 0.5) * (iw / w) - 0.5, 0, iw - 1)
    return _bilerp(img, ys[:, None], xs[None, :])


def _bilerp(img, ys, xs):
    """img sampled at (ys, xs), in pixel units (centers at integers), clamped to the edges."""
    import numpy as np
    ih, iw = img.shape[:2]
    ys = np.clip(ys, 0, ih - 1)
    xs = np.clip(xs, 0, iw - 1)
    y0 = np.floor(ys).astype(np.int64)
    x0 = np.floor(xs).astype(np.int64)
    y1, x1 = np.minimum(y0 + 1, ih - 1), np.minimum(x0 + 1, iw - 1)
    fy, fx = (ys - y0)[..., None].astype(np.float32), (xs - x0)[..., None].astype(np.float32)
    top = img[y0, x0] * (1 - fx) + img[y0, x1] * fx
    bot = img[y1, x0] * (1 - fx) + img[y1, x1] * fx
    return top * (1 - fy) + bot * fy


def _sample_uv(img, u, v):
    h, w = img.shape[:2]
    return _bilerp(img, v * h - 0.5, u * w - 0.5)


def _bright(c, thr: float):
    import numpy as np
    br = c.max(axis=-1, keepdims=True)
    knee = 0.1
    soft = np.clip(br - thr + knee, 0, 2 * knee)
    soft = soft * soft / (4 * knee + 1e-5)
    return c * (np.maximum(soft, br - thr) / np.maximum(br, 1e-4))


LEVELS = 6


def pyramid_sizes(w: int, h: int) -> list[tuple[int, int]]:
    out = []
    while len(out) < LEVELS and w >= 2 and h >= 2:
        w, h = w // 2, h // 2
        out.append((w, h))
    return out


def _pyramid(c, thr, r):
    import numpy as np
    x = _bright(c, thr) if thr is not None else c
    levels = []
    for w, h in pyramid_sizes(c.shape[1], c.shape[0]):
        p = np.pad(x, ((1, 1), (1, 1), (0, 0)), mode="edge")
        s = sum(p[dy:dy + 2 * h:2, dx:dx + 2 * w:2] for dy in range(4) for dx in range(4)) / 16
        x = _f16(s)
        levels.append(x)
    if not levels:
        return x
    u = levels[-1]
    for lv in reversed(levels[:-1]):
        u = _f16(lv * (1 - r) + _resize(u, *lv.shape[:2]) * r)
    return u


def _luma(c):
    import numpy as np
    return (c @ np.array(W709, np.float32))[..., None]


def _smooth(e0, e1, x):
    import numpy as np
    t = np.clip((x - e0) / (e1 - e0), 0, 1)
    return t * t * (3 - 2 * t)


def _pcg(v):
    """PCG hash (Jarzynski & Olano 2020), 32-bit unsigned, wrapping like GLSL's uint."""
    import numpy as np
    with np.errstate(over="ignore"):
        s = v * np.uint32(747796405) + np.uint32(2891336453)
        w = ((s >> ((s >> np.uint32(28)) + np.uint32(4))) ^ s) * np.uint32(277803737)
        return (w >> np.uint32(22)) ^ w


def _hash(x, y, z):
    import numpy as np
    return _pcg(x + _pcg(y + _pcg(z))).astype(np.float32) * np.float32(1 / 4294967296)


def _vnoise(ix, iy, f, z):
    import numpy as np
    one = np.uint32(1)
    a, b = _hash(ix, iy, z), _hash(ix + one, iy, z)
    c, d = _hash(ix, iy + one, z), _hash(ix + one, iy + one, z)
    fx, fy = f
    top = a * (1 - fx) + b * fx
    bot = c * (1 - fx) + d * fx
    return (top * (1 - fy) + bot * fy - 0.5) * 2


def _point(n: dict, u: dict, c, ctx):
    """f(c) for a node that needs only its own pixel (c: HxWx3 float32 RGB)."""
    import numpy as np
    t = n["type"]
    if t == "color":
        c = np.clip(c * np.array(u["wb"], np.float32), 0, 1)
        c = np.clip(c * np.float32(u["gain"]), 0, 1)
        s = c * c * (3 - 2 * c)
        c = c + np.float32(u["k"]) * (s - c)
        x1 = 1 - c
        c = c + np.float32(u["sh"]) * c * x1 * x1 + np.float32(u["hi"]) * c * c * x1
        x1 = 1 - c
        c = c + np.float32(u["fade"]) * x1 * x1
        y = _luma(c)
        spread = c.max(-1, keepdims=True) - c.min(-1, keepdims=True)
        f = np.float32(u["sat"]) * np.maximum(0, 1 + np.float32(u["vib"]) * (1 - spread))
        return y + (c - y) * f
    if t == "wheels":
        return np.maximum(np.array(u["gain"], np.float32) * (c + np.array(u["lift"], np.float32) * (1 - c)),
                          0) ** np.array(u["power"], np.float32)
    if t == "split":
        y = _luma(c)
        tt = np.clip(y + 0.5 * np.float32(u["bal"]), 0, 1)
        wh = tt * tt * (3 - 2 * tt)
        return (c + (1 - wh) * _smooth(0.0, 0.25, y) * np.array(u["sh"], np.float32)
                + wh * (1 - _smooth(0.9, 1.0, y)) * np.array(u["hi"], np.float32))
    if t == "mono":
        y = np.clip((c @ np.array(u["w"], np.float32))[..., None], 0, 1)
        return np.repeat(y, 3, -1) + np.array(u["tone"], np.float32) * (4 * y * (1 - y))
    if t == "lut":
        lut = ctx["luts"](n["file"])
        x = np.clip((c - np.array(u["dmin"], np.float32)) * np.array(u["dscale"], np.float32), 0, 1)
        return _trilinear(lut.table, x * (lut.size - 1))
    if t == "grain":
        px = max(np.float32(0.5), np.float32(u["size"]) * np.float32(c.shape[0]) / np.float32(1080))
        gx, gy = ctx["qx"] / np.float32(px), ctx["qy"] / np.float32(px)
        fx, fy = np.floor(gx), np.floor(gy)
        f = (gx - fx, gy - fy)
        f = tuple((a * a * (3 - 2 * a))[..., None] for a in f)
        ix, iy = fx.astype(np.uint32)[..., None], fy.astype(np.uint32)[..., None]
        z = np.uint32(ctx["frame"] * 4)
        n3 = np.concatenate([_vnoise(ix, iy, f, z + np.uint32(k)) for k in range(3)], -1)
        nm = _vnoise(ix, iy, f, z + np.uint32(3))
        col = np.float32(u["col"])
        nn = nm * (1 - col) + n3 * col
        y = np.clip(_luma(c), 0, 1)
        return c + np.float32(u["a"]) * nn * (0.25 + 3 * y * (1 - y))
    if t == "vignette":
        h, w = c.shape[:2]
        ax = 1.0 + (w / h - 1.0) * u["round"]
        dx, dy = (ctx["u"] - 0.5) * 2 * ax, (ctx["v"] - 0.5) * 2
        d = np.sqrt(dx * dx + dy * dy) / math.hypot(ax, 1.0)
        v = _smooth(u["r0"] - u["soft"], u["r0"] + u["soft"], d)[..., None]
        return c * (1 - np.float32(u["a"]) * v)
    if t == "scanlines":
        # At least 2 pixels: with 1, every row samples the same point of the wave and there are no lines at all
        # (Size 1 at 1080p, or the Looks page's small preview).
        period = max(np.float32(2), np.float32(u["size"]) * np.float32(c.shape[0]) / np.float32(1080))
        x = ctx["qy"] / period
        s = 0.5 + 0.5 * np.cos(np.float32(6.2831853) * (x - np.floor(x)))
        return c * (1 - np.float32(u["a"]) * s[..., None])
    if t == "letterbox":
        h, w = c.shape[:2]
        a = u["aspect"]
        if w / h < a:
            e = np.abs(ctx["qy"] - 0.5 * h) - 0.5 * w / a
        else:
            e = np.abs(ctx["qx"] - 0.5 * w) - 0.5 * h * a
        return c * (1 - np.clip(e + 0.5, 0, 1))[..., None]
    if t == "assist":
        m = np.array(u["m"], np.float32).reshape(3, 3)
        lin = c ** np.float32(2.2)
        return np.clip(lin @ m.T, 0, 1) ** np.float32(1 / 2.2)
    raise ValueError(t)


def _trilinear(table, x):
    """table [b, g, r] sampled at x (HxWx3, r g b in table units)."""
    import numpy as np
    n = table.shape[0]
    i0 = np.clip(np.floor(x).astype(np.int64), 0, n - 2) if n > 1 else np.zeros_like(x, np.int64)
    f = (x - i0).astype(np.float32)
    r0, g0, b0 = i0[..., 0], i0[..., 1], i0[..., 2]
    fr, fg, fb = f[..., 0:1], f[..., 1:2], f[..., 2:3]
    out = 0
    for db, wb in ((0, 1 - fb), (1, fb)):
        for dg, wg in ((0, 1 - fg), (1, fg)):
            for dr, wr in ((0, 1 - fr), (1, fr)):
                out = out + table[b0 + db, g0 + dg, r0 + dr] * (wb * wg * wr)
    return out


def _start(n: dict, u: dict, c, ctx):
    """f(c) for a node that reads other pixels of its input c."""
    import numpy as np
    t = n["type"]
    h, w = c.shape[:2]
    if t in ("bloom", "halation", "blur", "clarity"):
        thr, r = glow(n)
        g = _resize(_pyramid(c, thr, r), h, w)
        if t == "blur":
            return g
        if t == "clarity":
            return c + np.float32(u["amount"]) * (c - g)
        return c + np.float32(u["amount"]) * g * np.array(u["tint"], np.float32)
    if t == "ca":
        du, dv = (ctx["u"] - 0.5) * u["k"], (ctx["v"] - 0.5) * u["k"]
        red = _sample_uv(c, ctx["u"] + du, ctx["v"] + dv)[..., 0]
        blue = _sample_uv(c, ctx["u"] - du, ctx["v"] - dv)[..., 2]
        return np.stack([red, c[..., 1], blue], -1)
    if t == "sharpen":
        p = np.pad(c, ((1, 1), (1, 1), (0, 0)), mode="edge")
        nb = p[1:-1, 2:] + p[1:-1, :-2] + p[2:, 1:-1] + p[:-2, 1:-1]
        return c + np.float32(u["amount"]) * (c - 0.25 * nb)
    raise ValueError(t)


def render(look: dict, bgr, frame: int = 0, library: Library | None = None):
    """The look on an HxWx3 uint8 BGR image, as the video window draws it at that size."""
    import numpy as np
    lib = library or _default_library()
    passes = stages(look, lambda name: lib.lut(name) is not None)
    if not passes:
        return bgr.copy()
    h, w = bgr.shape[:2]
    c = bgr[..., ::-1].astype(np.float32) / 255
    qx = (np.arange(w, dtype=np.float32) + 0.5)[None, :].repeat(h, 0)
    qy = (np.arange(h, dtype=np.float32) + 0.5)[:, None].repeat(w, 1)
    ctx = {"qx": qx, "qy": qy, "u": qx / np.float32(w), "v": qy / np.float32(h), "frame": frame, "luts": lib.lut}
    for i, st in enumerate(passes):
        if i:
            c = _f16(c)  # the passes in between are half floats
        if st.start is not None:
            c = _mix(c, _start(st.start, dict(uniforms(st.start)), c, ctx), st.start["mix"])
        for n in st.nodes:
            lut = lib.lut(n["file"]) if n["type"] == "lut" else None
            c = _mix(c, _point(n, dict(uniforms(n, lut)), c, ctx), n["mix"])
    return (np.clip(c, 0, 1) * 255 + 0.5).astype(np.uint8)[..., ::-1].copy()


def _mix(a, b, m: float):
    import numpy as np
    m = np.float32(m)
    return np.clip(a * (1 - m) + b * m, 0, 1)


_LIB: list[Library] = []


def _default_library() -> Library:
    if not _LIB:
        _LIB.append(Library())
    return _LIB[0]


# ---------------------------------------------------------------------------------------------- what's on now
class LookState:
    """Which look the video shows: the "Look" settings (look_on, look), or, while the launcher's Looks page edits
    one, that one (preview). current() is the look to draw or None; `rev` changes whenever it does."""

    def __init__(self, settings, library: Library | None = None) -> None:
        self.settings, self.library = settings, library or _default_library()
        self.preview: dict | None = None
        self.rev = 0
        self._unwatch = settings.watch(self._changed)

    def _changed(self, key: str) -> None:
        if key in ("look", "look_on"):
            self.rev += 1

    def current(self) -> dict | None:
        if self.preview is not None:
            return self.preview
        if not self.settings.look_on:
            return None
        return self.library.get(self.settings.look) or BUILTIN[DEFAULT_LOOK]

    def name(self) -> str:
        look = self.current()
        return look["name"] if look is not None else ""

    def show(self, look: dict | None) -> None:
        """The launcher's live preview (None: back to the settings' look)."""
        self.preview = None if look is None else normalize(look)
        self.rev += 1

    def reload(self) -> None:
        """The user's looks changed on disk."""
        self.library.reload()
        if self.settings.look not in CHOICES:
            self.settings.set("look", DEFAULT_LOOK)
        self.rev += 1

    def toggle(self) -> None:
        """K: the look on or off."""
        self.preview = None
        self.settings.set("look_on", not self.settings.look_on)

    def next(self, delta: int = 1) -> None:
        """J: the next look (and on)."""
        self.preview = None
        if not self.settings.look_on:
            self.settings.set("look_on", True)
            return
        self.settings.step("look", delta)

    def close(self) -> None:
        self._unwatch()


# ---------------------------------------------------------------------------------------------- the launcher's way in
def schema(library: Library | None = None) -> dict:
    lib = library or Library()

    def entry(look_id, look, builtin):
        out = {"id": look_id, "name": look["name"], "builtin": builtin, "look": look, "code": encode(look)}
        if not builtin:
            out["path"] = str(lib.path(look_id))
        return out

    return {
        "folder": str(lib.folder), "lut_folder": str(lib.lut_folder), "luts": lib.luts(),
        "max_nodes": MAX_NODES, "default": DEFAULT_LOOK,
        "types": [{"key": t.key, "label": t.label, "kind": t.kind, "help": t.help,
                   "params": [{"key": f.key, "label": f.label, "min": f.lo, "max": f.hi, "default": f.default,
                               "step": f.step, "unit": f.unit, "names": list(f.names), "text": f.text}
                              for f in t.fields]}
                  for t in TYPES_LIST],
        "looks": [entry(k, v, True) for k, v in BUILTIN.items()]
                 + [entry(k, lib.user[k], False) for k in CHOICES if k in lib.user],
        "samples": _samples(),  # the preview's pictures (framegen/looks_preview.py)
    }


def _samples() -> list[dict]:
    from .looks_preview import samples
    return samples()


def main(argv: list[str] | None = None) -> int:
    """schema | encode [FILE] (a look's JSON file, or stdin) | decode CODE | import CODE (saves it as one of the
    user's looks) | save [ID] (look JSON on stdin) | delete ID | add-lut FILE | preview LOOK IN OUT (LOOK: an id or
    a code). Prints one line of JSON; errors as {"error": "..."}, with exit code 1."""
    import argparse
    p = argparse.ArgumentParser(prog="python -m framegen.looks")
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("schema")
    sub.add_parser("encode").add_argument("file", nargs="?")
    sub.add_parser("decode").add_argument("code")
    sub.add_parser("import").add_argument("code")
    sub.add_parser("save").add_argument("id", nargs="?")
    sub.add_parser("delete").add_argument("id")
    sub.add_parser("add-lut").add_argument("file")
    pv = sub.add_parser("preview")
    pv.add_argument("look")
    pv.add_argument("src")
    pv.add_argument("out")
    pv.add_argument("--frame", type=int, default=0)
    a = p.parse_args(argv)

    def say(obj) -> int:
        print(json.dumps(obj))  # (ASCII: the launcher reads it whatever the console code page)
        return 1 if "error" in obj else 0

    try:
        if a.cmd == "schema":
            return say(schema())
        if a.cmd == "encode":
            text = Path(a.file).read_text(encoding="utf-8") if a.file else sys.stdin.read()
            return say({"code": encode(json.loads(text))})
        if a.cmd == "decode":
            return say({"look": decode(a.code)})
        lib = Library()
        if a.cmd == "import":
            look = decode(a.code)
            return say({"id": lib.save(look), "look": look})
        if a.cmd == "save":
            return say({"id": lib.save(json.loads(sys.stdin.read()), a.id)})
        if a.cmd == "delete":
            return say({"deleted": lib.delete(a.id)})
        if a.cmd == "add-lut":
            return say({"name": lib.add_lut(a.file)})
        from .imagefile import read_image, write_image
        look = lib.get(a.look) or decode(a.look)
        img = read_image(a.src)
        if img is None:
            return say({"error": f"couldn't read {a.src}"})
        if not write_image(a.out, render(look, img, a.frame, lib)):
            return say({"error": f"couldn't write {a.out}"})
        return say({"out": a.out})
    except ValueError as e:  # CodeError, a bad LUT, bad JSON
        return say({"error": str(e)})
    except OSError as e:
        return say({"error": f"{e.strerror or e}"})


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    raise SystemExit(main())
