"""HDR from PS5 Remote Play: settings "hdr" (Auto by default: HDR output on a display with Windows' HDR on, "sdr" on
any other; framegen/chiaki_settings.py picks chiaki-ng's codec to match). With it off, or a stream without HDR,
nothing here runs.

A PS5 streaming in HDR (chiaki-ng's H.265 HDR codec) sends 10-bit H.265 in HDR10: BT.2020 colours and PQ (SMPTE
ST 2084) brightness, up to 10,000 nits. The rest of SmoovPlay works on 8-bit SDR frames, where that comes out washed
out, so framegen/chiaki_tap.py hands each decoded frame to a StreamConverter, which makes the pipeline's frames:

  "sdr"   (HDR stream, shown in SDR) 8-bit SDR, tone-mapped on the GPU as each frame is decoded: PQ to light, BT.2390's
          roll-off for highlights above SDR's range, BT.2020 to BT.709 colours, then SDR gamma. Everything after that
          (RIFE, upscaling, looks, instant replay) is as usual. Any display.
  "hdr10" (HDR output, Experimental) frames stay HDR10 (PQ, BT.2020) as half floats through RIFE and the upscaler,
          and the window presents them through a 10-bit DXGI swap chain in HDR10 colour space (framegen/
          dxgi_present.py), for an HDR display with Windows' HDR on. Without one, run.py uses "sdr" instead.

Frame formats (what the pipeline carries, on the GPU):
  sdr8  HxWx3 uint8 BGR, 0..255: the usual frames (BT.709 gamma)
  pq16  HxWx3 float16 BGR, 0..1 = PQ-encoded BT.2020 R'G'B', 0 to 10,000 nits. Half floats hold 11 significant bits,
        finer than the stream's 10 everywhere in 0..1, and OpenGL, CUDA and PyTorch all take them as they are.

The maths are plain PyTorch (any device), so tests/test_hdr.py checks them on the CPU. References: ITU-R BT.2100
(PQ, HLG, BT.2020 matrix coefficients), BT.2390 (the EETF roll-off), BT.2408 (203 nits for SDR white and graphics).
"""
from __future__ import annotations

import os
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

# ---------------------------------------------------------------------------------------------------- constants
# PQ (SMPTE ST 2084 / BT.2100)
M1, M2 = 2610 / 16384, 2523 / 4096 * 128
C1, C2, C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32
PQ_MAX_NITS = 10000.0
# HLG (BT.2100)
HLG_A, HLG_B, HLG_C = 0.17883277, 0.28466892, 0.55991073
HLG_DISPLAY_NITS, HLG_GAMMA = 1000.0, 1.2

SDR_WHITE_NITS = float(os.environ.get("FRAMEGEN_HDR_SDR_NITS", "203"))  # BT.2408: SDR white = HDR reference white
# The brightest a game's highlights go, when the stream doesn't say (PS5 games are mostly mastered for about this).
SOURCE_PEAK_NITS = float(os.environ.get("FRAMEGEN_HDR_PEAK", "1000"))
SDR_GAMMA = 2.4  # BT.1886: what an SDR stream of the same game is encoded for, so "sdr" looks like one

# AVColorTransferCharacteristic, AVColorPrimaries, AVColorSpace (matrix) and AVColorRange values (FFmpeg pixfmt.h)
TRC_BT709, TRC_PQ, TRC_HLG = 1, 16, 18
PRI_BT709, PRI_BT2020 = 1, 9
SPC_BT709, SPC_BT2020_NCL, SPC_BT2020_CL = 1, 9, 10
RANGE_FULL = 2

# Chromaticities (x, y)
BT709_XY = ((0.640, 0.330), (0.300, 0.600), (0.150, 0.060))
BT2020_XY = ((0.708, 0.292), (0.170, 0.797), (0.131, 0.046))
D65 = (0.3127, 0.3290)
KR_KB = {SPC_BT709: (0.2126, 0.0722), SPC_BT2020_NCL: (0.2627, 0.0593), SPC_BT2020_CL: (0.2627, 0.0593)}
KR_KB_601 = (0.299, 0.114)


def rgb_to_xyz(primaries, white=D65) -> np.ndarray:
    """3x3 matrix from linear RGB with these primaries to CIE XYZ (Y = 1 for white)."""
    m = np.array([[x / y, 1.0, (1 - x - y) / y] for x, y in primaries]).T
    wx, wy = white
    s = np.linalg.solve(m, np.array([wx / wy, 1.0, (1 - wx - wy) / wy]))
    return m * s


BT2020_TO_709 = np.linalg.solve(rgb_to_xyz(BT709_XY), rgb_to_xyz(BT2020_XY))
BT709_TO_2020 = np.linalg.inv(BT2020_TO_709)
BT2020_LUMA = rgb_to_xyz(BT2020_XY)[1]  # (0.2627, 0.6780, 0.0593)


# ------------------------------------------------------------------------------------------- transfer functions
def pq_eotf(e: torch.Tensor) -> torch.Tensor:
    """PQ signal 0..1 -> light, 0..1 of 10,000 nits."""
    p = e.clamp(0.0, 1.0).pow(1.0 / M2)
    return ((p - C1).clamp_min(0.0) / (C2 - C3 * p)).pow(1.0 / M1)


def pq_oetf(y: torch.Tensor) -> torch.Tensor:
    """Light 0..1 of 10,000 nits -> PQ signal 0..1 (the EOTF's inverse)."""
    p = y.clamp(0.0, 1.0).pow(M1)
    return ((C1 + C2 * p) / (1.0 + C3 * p)).pow(M2)


def pq_of_nits(nits: float) -> float:
    return float(pq_oetf(torch.tensor(nits / PQ_MAX_NITS, dtype=torch.float64)))


def srgb_eotf(e: torch.Tensor) -> torch.Tensor:
    """sRGB signal 0..1 -> light 0..1: the piecewise curve Windows shows SDR apps with on an HDR display. Its shadows
    are much brighter than BT.1886's gamma 2.4 (a tenth of the way up: 2.5x)."""
    return torch.where(e <= 0.04045, e / 12.92, ((e.clamp_min(0.0) + 0.055) / 1.055).pow(2.4))


def hlg_to_nits(rgb: torch.Tensor, display_nits: float = HLG_DISPLAY_NITS) -> torch.Tensor:
    """HLG R'G'B' (..., 3, BT.2020) -> display light in nits: inverse OETF, then the OOTF for a display this bright."""
    e = rgb.clamp(0.0, 1.0)
    scene = torch.where(e <= 0.5, e * e / 3.0, (torch.exp((e - HLG_C) / HLG_A) + HLG_B) / 12.0)
    ys = (scene * scene.new_tensor(BT2020_LUMA)).sum(-1, keepdim=True)
    return display_nits * ys.clamp_min(1e-9).pow(HLG_GAMMA - 1.0) * scene


def bt2390_eetf(e: torch.Tensor, src_peak_nits: float, dst_peak_nits: float) -> torch.Tensor:
    """BT.2390's roll-off, on PQ signals: below the knee as is, above it a smooth curve that lands the source's peak
    on the target's. Source and target black are both taken as 0 (the PS5's and a display's black are near enough
    for an EETF that's about highlights)."""
    src, dst = pq_of_nits(src_peak_nits), pq_of_nits(dst_peak_nits)
    max_lum = dst / src
    if max_lum >= 1.0:
        return e
    ks = max(1.5 * max_lum - 0.5, 0.0)
    x = (e / src).clamp_min(0.0)
    t = ((x - ks) / (1.0 - ks)).clamp(0.0, 1.0)
    t2, t3 = t * t, t * t * t
    p = (2 * t3 - 3 * t2 + 1) * ks + (t3 - 2 * t2 + t) * (1.0 - ks) + (-2 * t3 + 3 * t2) * max_lum
    return torch.where(x < ks, x, torch.where(x >= 1.0, x.new_tensor(max_lum), p)) * src


def tone_map_nits(rgb: torch.Tensor, src_peak_nits: float, dst_peak_nits: float) -> torch.Tensor:
    """Linear RGB in nits (..., 3) -> the same with highlights rolled off to dst_peak_nits (BT.2390 on the brightest
    channel, all three scaled alike, so colours keep their hue)."""
    if dst_peak_nits >= src_peak_nits:
        return rgb.clamp(0.0, dst_peak_nits)
    m = rgb.amax(-1, keepdim=True).clamp_min(1e-6)
    e = pq_oetf(m / PQ_MAX_NITS)
    mapped = pq_eotf(bt2390_eetf(e, src_peak_nits, dst_peak_nits)) * PQ_MAX_NITS
    return rgb.clamp_min(0.0) * (mapped / m)


def _matrix(rgb: torch.Tensor, m: np.ndarray) -> torch.Tensor:
    """(..., 3) RGB times a 3x3 matrix."""
    return rgb @ rgb.new_tensor(m).T


# ------------------------------------------------------------------------------------------------- frames
@dataclass(frozen=True)
class FrameInfo:
    """What a decoded frame's samples mean (from its colour tags)."""
    bits: int = 10
    full_range: bool = False
    matrix: int = SPC_BT2020_NCL
    transfer: int = TRC_PQ
    primaries: int = PRI_BT2020
    tagged: bool = True  # the frame said its transfer; False: taken from its bit depth (describe() says so)

    @property
    def hdr(self) -> bool:
        return self.transfer in (TRC_PQ, TRC_HLG)

    @classmethod
    def of(cls, fr, bits: int) -> "FrameInfo":
        """From a PyAV frame. Untagged 10-bit HEVC from a PS5 in HDR mode is HDR10; untagged 8-bit is BT.709."""
        trc, pri, spc = fr.color_trc, fr.color_primaries, fr.colorspace
        hdr_default = bits > 8
        tagged = trc in (TRC_PQ, TRC_HLG, TRC_BT709, 6, 14, 15)
        if not tagged:  # unknown or unspecified
            trc = TRC_PQ if hdr_default else TRC_BT709
        if pri not in (PRI_BT709, PRI_BT2020, 5, 6):
            pri = PRI_BT2020 if trc in (TRC_PQ, TRC_HLG) else PRI_BT709
        if spc not in KR_KB and spc not in (5, 6):
            spc = SPC_BT2020_NCL if pri == PRI_BT2020 else SPC_BT709
        return cls(bits, fr.color_range == RANGE_FULL, spc, trc, pri, tagged)


# A capture card's 10-bit P010 video (framegen/capture.py HdrCardSource), HDR10 or SDR as the card says the console
# sends it (its colour tags don't change: framegen/elgato.py). Both limited range, as HDMI video is: an Elgato HD60 X's
# HDR10 matched a PS5's Remote Play HDR stream of the same screen to about 2 codes, and its SDR, shown as Windows
# shows SDR, matched Remote Play's picture of it (black at 64; white text overshoots past 940 at its edges, but read
# as full range the picture came out washed out).
P010_HDR10 = FrameInfo(10, False, SPC_BT2020_NCL, TRC_PQ, PRI_BT2020)
P010_SDR = FrameInfo(10, False, SPC_BT709, TRC_BT709, PRI_BT709)


def yuv_to_rgb(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, info: FrameInfo) -> torch.Tensor:
    """Y'CbCr code values as floats (y HxW, u and v at half size, in the stream's own `bits`) -> R'G'B' (H, W, 3),
    0..1, in the stream's own transfer and primaries."""
    scale = float(1 << (info.bits - 8))
    if info.full_range:
        lo, ys, cs = 0.0, 255.0 * scale, 255.0 * scale
    else:
        lo, ys, cs = 16.0 * scale, 219.0 * scale, 224.0 * scale
    mid = 128.0 * scale
    kr, kb = KR_KB.get(info.matrix, KR_KB_601)
    kg = 1.0 - kr - kb
    lum = (y - lo) / ys
    c = torch.stack([u, v]).unsqueeze(0)
    c = F.interpolate(c, size=y.shape, mode="bilinear", align_corners=False)[0]
    cb, cr = (c[0] - mid) / cs, (c[1] - mid) / cs
    r = lum + 2 * (1 - kr) * cr
    b = lum + 2 * (1 - kb) * cb
    g = (lum - kr * r - kb * b) / kg
    return torch.stack([r, g, b], -1).clamp_(0.0, 1.0)


def rgb_to_nits(rgb: torch.Tensor, info: FrameInfo, sdr_nits: float = SDR_WHITE_NITS,
                srgb: bool = False) -> torch.Tensor:
    """R'G'B' in the stream's transfer -> linear light in nits, still in the stream's primaries. srgb: SDR as Windows
    shows SDR apps (srgb_eotf), not as BT.1886 video."""
    if info.transfer == TRC_PQ:
        return pq_eotf(rgb) * PQ_MAX_NITS
    if info.transfer == TRC_HLG:
        return hlg_to_nits(rgb)
    return (srgb_eotf(rgb) if srgb else rgb.pow(SDR_GAMMA)) * sdr_nits


def nits_to_sdr8(rgb: torch.Tensor, primaries: int, src_peak: float, sdr_nits: float = SDR_WHITE_NITS) -> torch.Tensor:
    """Linear light in nits (H, W, 3), `primaries` -> HxWx3 uint8 BGR SDR (BT.709): highlights rolled off to SDR
    white, BT.2020 colours mapped into BT.709 (what's outside is clipped), SDR gamma."""
    rgb = tone_map_nits(rgb, src_peak, sdr_nits) / sdr_nits
    if primaries == PRI_BT2020:
        rgb = _matrix(rgb, BT2020_TO_709)
    e = rgb.clamp_(0.0, 1.0).pow_(1.0 / SDR_GAMMA)
    return e.flip(-1).mul_(255.0).round_().to(torch.uint8)


def nits_to_pq16(rgb: torch.Tensor, primaries: int, src_peak: float, display_peak: float | None = None) -> torch.Tensor:
    """Linear light in nits (H, W, 3), `primaries` -> HxWx3 float16 BGR PQ BT.2020. display_peak: roll highlights
    off to this (an HDR display dimmer than the game's peak), else they go as they are."""
    if primaries != PRI_BT2020:
        rgb = _matrix(rgb, BT709_TO_2020)
    if display_peak is not None and display_peak < src_peak:
        rgb = tone_map_nits(rgb, src_peak, display_peak)
    return pq_oetf(rgb.clamp_min(0.0) / PQ_MAX_NITS).flip(-1).to(torch.float16)


def convert_yuv(y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, info: FrameInfo, out: str,
                src_peak: float = SOURCE_PEAK_NITS, display_peak: float | None = None,
                sdr_nits: float = SDR_WHITE_NITS) -> torch.Tensor:
    """Y'CbCr code values (floats, the stream's bit depth) -> the pipeline's frame: out "sdr8" or "pq16"."""
    rgb = yuv_to_rgb(y, u, v, info)
    if out == "pq16" and info.transfer == TRC_PQ and info.primaries == PRI_BT2020 and (
            display_peak is None or display_peak >= src_peak):
        return rgb.flip(-1).to(torch.float16)  # already HDR10: nothing to change but the layout
    if out == "sdr8" and not info.hdr and info.primaries != PRI_BT2020:
        return rgb.flip(-1).mul_(255.0).round_().to(torch.uint8)  # SDR in a 10-bit stream: just 8 bits
    nits = rgb_to_nits(rgb, info, sdr_nits)
    if out == "sdr8":
        return nits_to_sdr8(nits, info.primaries, src_peak, sdr_nits)
    return nits_to_pq16(nits, info.primaries, src_peak, display_peak)


def pq16_to_sdr8_exact(frame: torch.Tensor, src_peak: float = SOURCE_PEAK_NITS,
                       sdr_nits: float = SDR_WHITE_NITS) -> torch.Tensor:
    """An HDR10 frame (pq16) -> sdr8, the same way "sdr" makes them (the maths, per pixel; pq16_to_sdr8 is the fast
    one)."""
    nits = pq_eotf(frame.float().flip(-1)) * PQ_MAX_NITS
    return nits_to_sdr8(nits, PRI_BT2020, src_peak, sdr_nits)


# ------------------------------------------------------------------------------------------------- 3D LUTs
class Lut3D:
    """A function of three values (three out), sampled on a grid once and then applied to whole frames with one
    trilinear grid_sample: the tone mapping is about 45 small GPU steps per pixel done directly (2.5 ms a 1080p
    frame on an RTX 4070 Ti SUPER), and a handful this way. What a LUT holds should be smooth: SDR's LUTs hold linear
    light before it's clipped to BT.709, and the clip and gamma (steep near black, where a LUT's straight lines
    between points would miss by tens of levels) are done per pixel after it. tests/test_hdr.py checks it against
    the maths.

    fn(points) -> outputs: (N, 3) float64 in, (N, 3) out. Axes: (lo, hi, size) each; input values outside are
    clamped to the grid's edge."""

    def __init__(self, fn, axes, device) -> None:
        self.axes = axes
        grids = torch.meshgrid(*[torch.linspace(lo, hi, n, dtype=torch.float64) for lo, hi, n in axes], indexing="ij")
        pts = torch.stack(grids, -1).reshape(-1, 3)
        out = fn(pts).float()
        self.lut = out.T.reshape(1, 3, *[n for _, _, n in axes]).contiguous().to(device)
        # value -> grid_sample's -1..1, as a * value + b per axis
        self.ab = [(2.0 / (hi - lo), -1.0 - 2.0 * lo / (hi - lo)) for lo, hi, _ in axes]

    def __call__(self, x0: torch.Tensor, x1: torch.Tensor, x2: torch.Tensor) -> torch.Tensor:
        """Three (H, W) float tensors (the axes' values) -> (H, W, 3) float32."""
        (a0, b0), (a1, b1), (a2, b2) = self.ab
        # grid_sample's last dimension is (x, y, z) = the LUT's (last, middle, first) axis
        g = torch.stack([x2 * a2 + b2, x1 * a1 + b1, x0 * a0 + b0], -1)[None, None]
        out = F.grid_sample(self.lut, g, mode="bilinear", padding_mode="border", align_corners=True)
        return out[0, :, 0].permute(1, 2, 0)


def _ycbcr_axes(info: FrameInfo, n_y: int = 129, n_c: int = 129):
    """A LUT's axes in the stream's own code values: Y' over its nominal range, Cb and Cr likewise."""
    scale = float(1 << (info.bits - 8))
    if info.full_range:
        return ((0.0, 255.0 * scale, n_y), (0.0, 255.0 * scale, n_c), (0.0, 255.0 * scale, n_c))
    return ((16.0 * scale, 235.0 * scale, n_y), (16.0 * scale, 240.0 * scale, n_c), (16.0 * scale, 240.0 * scale, n_c))


def _pixels(fn_frame):
    """A per-frame conversion (y (H,W), u, v at half size -> (H, W, 3)) as a per-point one for Lut3D: points go in
    as a one-row "frame", with chroma at full size (the LUT works per pixel, after chroma is upsampled)."""
    def fn(pts: torch.Tensor) -> torch.Tensor:
        y, u, v = pts[:, 0][None], pts[:, 1][None], pts[:, 2][None]
        return fn_frame(y, u, v).reshape(-1, 3).double()
    return fn


def _upsample_uv(u: torch.Tensor, v: torch.Tensor, size) -> tuple[torch.Tensor, torch.Tensor]:
    c = F.interpolate(torch.stack([u, v]).unsqueeze(0), size=size, mode="bilinear", align_corners=False)[0]
    return c[0], c[1]


def _yuv_to_rgb_full(y, u, v, info: FrameInfo) -> torch.Tensor:
    """yuv_to_rgb with chroma already at full size (for building LUTs)."""
    scale = float(1 << (info.bits - 8))
    lo, ys, cs = (0.0, 255.0 * scale, 255.0 * scale) if info.full_range else (16.0 * scale, 219.0 * scale,
                                                                             224.0 * scale)
    kr, kb = KR_KB.get(info.matrix, KR_KB_601)
    kg = 1.0 - kr - kb
    lum = (y - lo) / ys
    cb, cr = (u - 128.0 * scale) / cs, (v - 128.0 * scale) / cs
    r = lum + 2 * (1 - kr) * cr
    b = lum + 2 * (1 - kb) * cb
    g = (lum - kr * r - kb * b) / kg
    return torch.stack([r, g, b], -1).clamp_(0.0, 1.0)


def _yuv_lut(info: FrameInfo, out: str, src_peak: float, display_peak: float | None, sdr_nits: float, device,
             srgb: bool = False) -> Lut3D:
    """A LUT from Y'CbCr codes (chroma at full size) to: sdr8, linear BT.709 BGR (1 = SDR white) before it's clipped
    (_sdr_finish does the rest); pq16, BGR PQ 0..1. srgb: see rgb_to_nits."""
    def frame_fn(y, u, v):
        rgb = _yuv_to_rgb_full(y, u, v, info)
        nits = rgb_to_nits(rgb, info, sdr_nits, srgb)
        if out == "sdr8":
            rgb = tone_map_nits(nits, src_peak, sdr_nits) / sdr_nits
            if info.primaries == PRI_BT2020:
                rgb = _matrix(rgb, BT2020_TO_709)
            return rgb.flip(-1)
        if info.primaries != PRI_BT2020:
            nits = _matrix(nits, BT709_TO_2020)
        if display_peak is not None and display_peak < src_peak:
            nits = tone_map_nits(nits, src_peak, display_peak)
        return pq_oetf(nits.clamp_min(0.0) / PQ_MAX_NITS).flip(-1)
    return Lut3D(_pixels(frame_fn), _ycbcr_axes(info), device)


def _sdr_finish(lin: torch.Tensor) -> torch.Tensor:
    """Linear BT.709 (1 = SDR white), from a LUT -> uint8 SDR: clipped, then gamma, exactly, per pixel."""
    return lin.clamp_(0.0, 1.0).pow_(1.0 / SDR_GAMMA).mul_(255.0).round_().to(torch.uint8)


_PQ_SDR_LUTS: dict = {}


def pq16_to_sdr8(frame: torch.Tensor, src_peak: float = SOURCE_PEAK_NITS,
                 sdr_nits: float = SDR_WHITE_NITS) -> torch.Tensor:
    """An HDR10 frame (pq16) -> sdr8, the same way "sdr" makes them: for instant replay's clips while the window
    shows HDR. Through a 3D LUT over PQ R'G'B' (see Lut3D)."""
    key = (frame.device, src_peak, sdr_nits)
    lut = _PQ_SDR_LUTS.get(key)
    if lut is None:
        def fn(pts):
            nits = pq_eotf(pts) * PQ_MAX_NITS  # (N, 3) RGB
            return _matrix(tone_map_nits(nits, src_peak, sdr_nits) / sdr_nits, BT2020_TO_709).flip(-1)
        _PQ_SDR_LUTS[key] = lut = Lut3D(fn, ((0.0, 1.0, 129),) * 3, frame.device)
    f = frame.float()
    return _sdr_finish(lut(f[..., 2], f[..., 1], f[..., 0]))


# pq16 -> 8-bit levels for the checks tuned on SDR frames (repeat and cut detection, the delay test, test mode's
# scores): SDR white's PQ value goes to 255, so the SDR range of an HDR picture spreads over the levels about as an
# SDR frame's does (highlights above it all read 255). One multiply: no tone mapping on every frame for a check.
LEVEL8_SCALE = 255.0 / pq_of_nits(SDR_WHITE_NITS)


def level8(frame: torch.Tensor) -> torch.Tensor:
    """Any pipeline frame (sdr8 or pq16) as 0..255 floats, for checks tuned in 8-bit levels."""
    return (frame.float() * LEVEL8_SCALE).clamp_(0.0, 255.0) if frame.is_floating_point() else frame.float()


def to_u8(frame: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """level8 as uint8 (into `out` if given): a pq16 frame's 8-bit stand-in for those checks."""
    if not frame.is_floating_point():
        return frame if out is None else out.copy_(frame)
    v = frame.mul(LEVEL8_SCALE).clamp_(0.0, 255.0)
    return v.to(torch.uint8) if out is None else out.copy_(v)


# ------------------------------------------------------------------------------ SDR graphics on an HDR picture
_SDR_LUT: dict[float, np.ndarray] = {}


def sdr_to_pq16_image(bgr: np.ndarray, nits: float = SDR_WHITE_NITS) -> np.ndarray:
    """An 8-bit BGR image (the menu, messages, the stats line) -> uint16 BGR PQ BT.2020, white at `nits` (BT.2408's
    graphics white): drawn over an HDR10 picture it looks as it does on an SDR one, not blinding."""
    lut = _SDR_LUT.get(nits)
    if lut is None:
        lin = (np.arange(256) / 255.0) ** 2.2 * nits / PQ_MAX_NITS  # sRGB-ish, as the menu is drawn for a PC monitor
        _SDR_LUT[nits] = lut = lin.astype(np.float32)
    lin = lut[bgr[..., ::-1]]  # RGB, linear
    flat = lin.reshape(-1, 3)
    if np.ptp(flat, axis=1).any():  # colour: BT.709 -> BT.2020 (grey stays grey, so most of the menu skips this)
        flat = flat @ BT709_TO_2020.T.astype(np.float32)
    p = np.clip(flat, 0.0, 1.0) ** M1
    pq = ((C1 + C2 * p) / (1.0 + C3 * p)) ** M2
    return (pq.reshape(lin.shape)[..., ::-1] * 65535.0 + 0.5).astype(np.uint16)


# --------------------------------------------------------------------------- test streams (scripts/frametap_fake.py)
def expand_sdr(bgr: torch.Tensor, peak_nits: float = SOURCE_PEAK_NITS, knee: float = 0.6) -> torch.Tensor:
    """An 8-bit SDR frame (HxWx3 BGR) -> linear BT.2020 light in nits, (H, W, 3) RGB, the way a game's HDR might
    look: SDR white at 203 nits, and the brightest parts (above `knee` of SDR's light) pushed up to peak_nits, so a
    test stream has highlights to roll off. For testing only: not how any game makes its HDR."""
    lin = (bgr.float().flip(-1) / 255.0).pow(SDR_GAMMA)
    boost = ((lin.amax(-1, keepdim=True) - knee) / (1 - knee)).clamp_min(0.0).pow(2.0)
    nits = lin * SDR_WHITE_NITS * (1.0 + boost * (peak_nits / SDR_WHITE_NITS - 1.0))
    return _matrix(nits, BT709_TO_2020)


def nits_to_yuv10(rgb: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Linear BT.2020 light in nits (H, W, 3) -> HDR10's 10-bit limited-range Y', Cb, Cr planes (4:2:0, int16),
    as an encoder takes them."""
    e = pq_oetf(rgb.clamp_min(0.0) / PQ_MAX_NITS)
    kr, kb = KR_KB[SPC_BT2020_NCL]
    y = kr * e[..., 0] + (1 - kr - kb) * e[..., 1] + kb * e[..., 2]
    cb, cr = (e[..., 2] - y) / (2 * (1 - kb)), (e[..., 0] - y) / (2 * (1 - kr))

    def half(c):
        return F.avg_pool2d(c[None, None], 2)[0, 0]

    return ((64 + 876 * y).round().to(torch.int16), (512 + 896 * half(cb)).round().to(torch.int16),
            (512 + 896 * half(cr)).round().to(torch.int16))


# ---------------------------------------------------------------------------------------------- decoded frames
def _u16(t: torch.Tensor) -> torch.Tensor:
    """16-bit samples, however DLPack typed them, as int32 0..65535."""
    if t.dtype == torch.uint8:  # bytes: pairs, little-endian
        t = t.contiguous().view(torch.int16)
    if t.dtype == torch.int16:
        return t.to(torch.int32) & 0xFFFF
    return t.to(torch.int32)


class StreamConverter:
    """Decoded frames (PyAV) -> the pipeline's frames: out "sdr8" (uint8 BGR) or "pq16" (float16 BGR, PQ BT.2020),
    as PyTorch tensors on `device`. NVDEC frames (P010 on the GPU) never leave it; software-decoded ones are uploaded
    as their planes, a third of the bytes of a BGR frame."""

    def __init__(self, out: str = "sdr8", device: str = "cuda", src_peak: float = SOURCE_PEAK_NITS,
                 display_peak: float | None = None, sdr_nits: float | None = None) -> None:
        """sdr_nits: for an SDR stream in HDR output (pq16), show it as Windows shows SDR apps: white at this (Windows'
        SDR content brightness) and the sRGB curve, so it looks as it did as an SDR window. Else BT.1886 video with
        white at SDR_WHITE_NITS."""
        if out not in ("sdr8", "pq16"):
            raise ValueError(out)
        self.out, self.device = out, torch.device(device)
        self.src_peak, self.display_peak = src_peak, display_peak
        self.sdr_nits = sdr_nits
        self.info: FrameInfo | None = None  # the last frame's, for the log
        self.said = False
        self.lut: Lut3D | None = None  # for self.info
        self._luts: dict = {}  # one per FrameInfo seen, so a stream that switches between HDR and SDR (a capture
        # card's console turning its HDR on and off) doesn't make them again each time

    def __call__(self, fr) -> torch.Tensor:
        fmt = fr.format.name
        if fmt == "cuda":  # NVDEC: NV12 (8-bit) or P010 (10-bit, the samples in the top bits), on the GPU
            y = torch.from_dlpack(fr.planes[0].__dlpack__())
            uv = torch.from_dlpack(fr.planes[1].__dlpack__())
            ten = (fr.sw_format.name if fr.sw_format is not None else "") in ("p010le", "p016le") or \
                y.dtype in (torch.int16, torch.uint16)
            if ten:
                y, uv = _u16(y), _u16(uv)
                if uv.dim() == 2:  # interleaved as bytes: W bytes per row = W/2 pairs of 16-bit U, V
                    uv = uv.view(uv.shape[0], -1, 2)
                bits, yf, uf, vf = 10, (y >> 6).float(), (uv[..., 0] >> 6).float(), (uv[..., 1] >> 6).float()
            else:
                bits, yf, uf, vf = 8, y.float(), uv[..., 0].float(), uv[..., 1].float()
        else:
            bits, (yf, uf, vf) = self._planes(fr)
        return self.convert(yf, uf, vf, FrameInfo.of(fr, bits))

    def convert(self, y: torch.Tensor, u: torch.Tensor, v: torch.Tensor, info: FrameInfo) -> torch.Tensor:
        """Y'CbCr code planes (floats, chroma at half size) -> the pipeline's frame. HDR10 kept as HDR10 is just the
        matrix; anything else goes through a 3D LUT of convert_yuv's maths. `info` may change from frame to frame."""
        if info != self.info:
            self.info, self.said, self.lut = info, False, self._luts.get(info)
        if self.out == "pq16" and info.transfer == TRC_PQ and info.primaries == PRI_BT2020 and (
                self.display_peak is None or self.display_peak >= self.src_peak):
            return yuv_to_rgb(y, u, v, info).flip(-1).to(torch.float16)
        if self.out == "sdr8" and not info.hdr and info.primaries != PRI_BT2020:  # SDR in 10 bits: just 8 bits
            return yuv_to_rgb(y, u, v, info).flip(-1).mul_(255.0).round_().to(torch.uint8)
        if self.lut is None:
            # an SDR stream to HDR output as Windows shows SDR apps (sdr_nits); otherwise HDR's reference white (the
            # level HDR is fitted to for SDR output)
            windows = self.sdr_nits is not None and self.out == "pq16" and not info.hdr
            self.lut = self._luts[info] = _yuv_lut(info, self.out, self.src_peak, self.display_peak,
                                                   self.sdr_nits if windows else SDR_WHITE_NITS, y.device, windows)
        uf, vf = _upsample_uv(u, v, y.shape)
        out = self.lut(y, uf, vf)
        return _sdr_finish(out) if self.out == "sdr8" else out.to(torch.float16)

    def _planes(self, fr):
        """A software-decoded frame's Y, U, V planes as float tensors on the device."""
        name = fr.format.name
        if name in ("yuv420p10le", "yuv420p"):
            bits, dt = (10, np.uint16) if name.endswith("10le") else (8, np.uint8)
        elif name in ("p010le", "nv12"):
            fr = fr.reformat(format="yuv420p10le" if name == "p010le" else "yuv420p")
            return self._planes(fr)
        else:  # anything else: through FFmpeg's own conversion to 10 bits
            return self._planes(fr.reformat(format="yuv420p10le"))
        out = []
        for i, p in enumerate(fr.planes):
            h = fr.height if i == 0 else (fr.height + 1) // 2
            w = fr.width if i == 0 else (fr.width + 1) // 2
            stride = p.line_size // np.dtype(dt).itemsize
            a = np.ascontiguousarray(np.frombuffer(p, dt, count=h * stride).reshape(h, stride)[:, :w])
            if dt is np.uint16:
                a = a.view(np.int16)  # 10-bit values fit; PyTorch takes int16 from NumPy everywhere
            out.append(torch.from_numpy(a).to(self.device).float())
        return bits, out

    def describe(self) -> str:
        i = self.info
        if i is None:
            return "no frame yet"
        trc = {TRC_PQ: "PQ", TRC_HLG: "HLG"}.get(i.transfer, "SDR gamma")
        pri = {PRI_BT2020: "BT.2020", PRI_BT709: "BT.709"}.get(i.primaries, f"primaries {i.primaries}")
        guessed = "" if i.tagged else f" (no colour tags: taken as {'HDR10' if i.hdr else 'SDR'})"
        return f"{i.bits}-bit {trc} {pri}{' full range' if i.full_range else ''}{guessed}"
