"""NVIDIA Video Frame Generation (VFG) as an interpolation engine, in place of RIFE.

VFG is a filter in NVIDIA's Video Effects SDK, used here through NVIDIA's official `nvidia-vfx` Python package.
It is NOT part of this repo or of SmoovPlay's downloads: each user installs it from NVIDIA after accepting
NVIDIA's licence, through the launcher's Settings page or framegen/vfg_setup.py, into its own folder
(FRAMEGEN_VFX_PATH). It runs on RTX 40 and 50 series cards only (driver 570.65 or newer).

NVIDIA's licence for it forbids publishing its benchmark results without NVIDIA's written permission, and
using its output to train or tune other models. So: VFG timings and scores stay on the user's own PC (the
overlay, Auto-detect), never in the README, release notes, diagnostics or anything we publish.

The interface matches RifeInterpolator, so the pipeline and scheduler don't change: load() a frame into
`in_u8`, prepare() it into a slot, interpolate(newest_slot, t) into `out_u8`. Our own cut detector decides
when not to interpolate, so VFG's automatic shot-change detection is off.
"""
from __future__ import annotations

import torch

from .vfg_setup import MIN_COMPUTE, add_to_path, installed

QUALITY = {"low": 0, "medium": 1, "high": 2}
NOT_SET_UP = "NVIDIA VFG isn't set up on this PC. SmoovPlay's Settings page downloads it from NVIDIA."


def import_vfx():
    add_to_path()
    import nvvfx
    return nvvfx


_available: str | None = None


def vfg_unavailable() -> str:
    """Why VFG can't run on this PC ("" if it can). Cached. Quick: it finds the package without importing it, so
    a broken install shows up when a session builds VfgInterpolator (run.py then falls back to RIFE)."""
    global _available
    if _available is None:
        _available = _check()
    return _available


def _check() -> str:
    try:
        if not torch.cuda.is_available():
            return "No NVIDIA graphics card was found."
        if torch.cuda.get_device_capability(0) < MIN_COMPUTE:
            return "NVIDIA VFG needs an RTX 40 or 50 series card."
        return "" if installed() else NOT_SET_UP
    except Exception as e:  # noqa: BLE001 - a broken install must never stop RIFE from running
        return f"NVIDIA VFG couldn't be checked: {e}"


class VfgInterpolator:
    """VFG behind RifeInterpolator's interface. encoding: "rgb10a2" (packed 10-bit, what NVIDIA recommends: no
    float round trip, and the whole run stays on our stream) or "rgb8" (float32 planes, the simpler path)."""

    def __init__(self, width: int, height: int, quality: str = "medium", device: str = "cuda",
                 encoding: str = "rgb10a2", hdr: bool = False) -> None:
        nvvfx = import_vfx()
        self.w, self.h = width, height
        self.device = torch.device(device)
        self.quality, self.encoding = quality, encoding
        self.version, self.scale, self.trt = f"VFG {quality}", 1.0, None  # what run.py's label reads
        self.private = True  # the pipeline keeps timings and scores out of logs and reports (NVIDIA's licence)
        enc = nvvfx.ImageEncoding.RGB10A2 if encoding == "rgb10a2" else nvvfx.ImageEncoding.RGB8
        self.fx = nvvfx.VideoFrameGeneration(mode=QUALITY[quality], automatic_shot_change_detection_enabled=False,
                                             device=self.device.index or 0, image_encoding=enc)
        self.fx.input_width, self.fx.input_height = width, height
        self.fx.load()
        # hdr: frames are float16 PQ (framegen/hdr.py pq16); in_u8 is then their 8-bit stand-in for the pipeline's
        # checks, and out_u8 holds float16 frames. RGB10A2 carries the stream's 10 bits as they are.
        self.hdr = hdr
        self.frame_dtype = torch.float16 if hdr else torch.uint8
        self.inp = torch.zeros((height, width, 3), dtype=self.frame_dtype, device=self.device)
        self._out_u8 = torch.zeros((height, width, 3), dtype=self.frame_dtype, device=self.device)
        self.out_rgb10a2 = torch.zeros((height, width), dtype=torch.int32, device=self.device) if encoding == "rgb10a2" else None
        self._out_u8_dirty = False
        self.rgb10a2_out = False
        self.in_u8 = torch.zeros((height, width, 3), dtype=torch.uint8, device=self.device) if hdr else self.inp
        if encoding == "rgb10a2":
            self.slots = [torch.zeros((height, width), dtype=torch.int32, device=self.device) for _ in range(2)]
        else:
            self.slots = [torch.zeros((3, height, width), dtype=torch.float32, device=self.device) for _ in range(2)]
        # Steady HUD (framegen/steady.py, set_steady): made the first time it's turned on, with the real frames kept
        # as (1, 3, h, w) RGB halves beside VFG's own copies.
        self.steady, self.hud, self.real = 0.0, None, None
        self._slot: int | None = None  # the newest prepared slot
        self.warm_up()

    @torch.no_grad()
    def warm_up(self) -> None:
        """Run one dummy interpolation pass at startup so CUDA/VFX kernels and allocations are ready before tick 1."""
        prev, cur = self.slots[0], self.slots[1]
        if self.encoding == "rgb10a2":
            prev, cur = prev.view(torch.uint32), cur.view(torch.uint32)
        stream = torch.cuda.current_stream(self.device)
        self.fx.run_at_timestep(prev, cur, 0.5, stream_ptr=stream.cuda_stream)
        stream.synchronize()

    @property
    def out_u8(self) -> torch.Tensor:
        """HxWx3 BGR output buffer, unpacked lazily when RGB10A2 passthrough is active."""
        if self._out_u8_dirty:
            self._out_u8.copy_(unpack_rgb10a2(self.out_rgb10a2, hdr=self.hdr))
            self._out_u8_dirty = False
        return self._out_u8

    # ---- the RifeInterpolator interface --------------------------------------------------------------------
    def load(self, frame_u8: torch.Tensor) -> None:
        """Copy an HxWx3 BGR frame (GPU or pinned CPU; uint8, or float16 if hdr) into the static input."""
        self.inp.copy_(frame_u8, non_blocking=True)
        if self.hdr:
            from .hdr import to_u8
            to_u8(self.inp, self.in_u8)

    @torch.no_grad()
    def prepare(self, slot: int) -> None:
        """Convert the input into slot `slot` (0/1) in VFG's input format; with Steady HUD on, also where it acts
        between the previous real frame and this one."""
        if self.encoding == "rgb10a2":
            pack_rgb10a2(self.inp, out=self.slots[slot])
        elif self.hdr:
            self.slots[slot].copy_(self.inp.permute(2, 0, 1).flip(0).float())
        else:
            self.slots[slot].copy_(self.inp.permute(2, 0, 1).flip(0).float().div_(255.0))
        if self.steady > 0:
            self.real[slot].copy_(_rgb(self.inp, self.hdr))
            self.hud.mask(1 - slot, slot, self.real[1 - slot], self.real[slot])
        self._slot = slot

    @torch.no_grad()
    def interpolate(self, newest_slot: int, t: float = 0.5) -> torch.Tensor:
        """Frame at fraction t from slot (1-newest_slot) to newest_slot -> `out_u8` (or `out_rgb10a2` when
        rgb10a2_out is enabled, overwritten each call)."""
        prev, cur = self.slots[1 - newest_slot], self.slots[newest_slot]
        stream = torch.cuda.current_stream(self.device).cuda_stream
        if self.encoding == "rgb10a2":
            prev, cur = prev.view(torch.uint32), cur.view(torch.uint32)
        out = self.fx.run_at_timestep(prev, cur, t, stream_ptr=stream)
        img = torch.from_dlpack(out.image)  # SDK-owned memory: copied before the next run
        if self.encoding == "rgb10a2":
            if self.rgb10a2_out and not self.hdr and self.steady <= 0:
                self.out_rgb10a2.copy_(img.view(torch.int32))
                self._out_u8_dirty = True
                return self.out_rgb10a2
            self._out_u8.copy_(unpack_rgb10a2(img.view(torch.int32), hdr=self.hdr))
            self._out_u8_dirty = False
        elif self.hdr:
            self._out_u8.copy_(img.clamp(0.0, 1.0).flip(0).permute(1, 2, 0))
            self._out_u8_dirty = False
        else:
            self._out_u8.copy_(img.clamp(0.0, 1.0).mul(255.0).round_().to(torch.uint8).flip(0).permute(1, 2, 0))
            self._out_u8_dirty = False
        if self.steady > 0:  # Steady HUD: the real pixels where the HUD is
            self.hud.t_now.fill_(t)
            mid = self.hud.blend(_rgb(self._out_u8, self.hdr), self.real[1 - newest_slot], self.real[newest_slot])
            _store(mid, self._out_u8)
        return self._out_u8

    @torch.no_grad()
    def set_steady(self, threshold: float) -> None:
        """Steady HUD (framegen/steady.py) on (threshold: largest real-frame difference, 0..1, that counts as still)
        or off (0). Turned on between two real frames, it takes them from VFG's own copies."""
        if threshold > 0:
            if self.hud is None:
                from .steady import SteadyHud
                self.hud = SteadyHud(self.h, self.w, self.device)
                self.real = [torch.zeros((1, 3, self.h, self.w), dtype=torch.half, device=self.device)
                             for _ in range(2)]
            self.hud.still_thr.fill_(threshold)
            if self.steady <= 0 and self._slot is not None:  # turned on mid-pair: the current pair's mask now
                for s in (0, 1):
                    self.real[s].copy_(self._slot_rgb(s))
                old = 1 - self._slot
                self.hud.edges(old, self.real[old])
                self.hud.mask(old, self._slot, self.real[old], self.real[self._slot])
        self.steady = threshold

    def _slot_rgb(self, slot: int) -> torch.Tensor:
        """A prepared real frame back from VFG's input format, as (1, 3, h, w) RGB."""
        if self.encoding == "rgb10a2":
            return _rgb(unpack_rgb10a2(self.slots[slot], hdr=self.hdr), self.hdr)
        return self.slots[slot].unsqueeze(0)  # (already RGB, 0..1)

    def close(self) -> None:
        close = getattr(self.fx, "close", None)
        if close is not None:
            close()


def _rgb(bgr: torch.Tensor, hdr: bool) -> torch.Tensor:
    """An HxWx3 BGR frame (uint8, or float16 PQ with hdr) as (1, 3, H, W) RGB, 0..1, for Steady HUD."""
    x = bgr.permute(2, 0, 1).flip(0).unsqueeze(0).float()
    return x if hdr else x.div_(255.0)


def _store(rgb: torch.Tensor, out: torch.Tensor) -> None:
    """(1, 3, H, W) RGB 0..1 back into an HxWx3 BGR frame, in its own type (uint8, or float16 with hdr)."""
    x = rgb[0].clamp(0.0, 1.0).flip(0).permute(1, 2, 0)
    out.copy_(x if out.is_floating_point() else x.mul(255.0).round_().to(torch.uint8))


# ---- packed 10-bit RGB: R in bits 0-9, G 10-19, B 20-29, alpha 30-31 (left 0, so it fits a signed int32) ------
def pack_rgb10a2(bgr: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """HxWx3 BGR -> HxW int32 holding packed RGB10A2. uint8 values are widened exactly (v10 = v8 * 1023 / 255);
    float ones (0..1, hdr's pq16) rounded to 10 bits."""
    if bgr.is_floating_point():
        v = bgr.float().clamp(0.0, 1.0).mul_(1023.0).round_().to(torch.int32)
    else:
        v = bgr.to(torch.int32)
        v = (v << 2) | (v >> 6)  # 8 -> 10 bits, 255 -> 1023
    if out is not None:
        return torch.bitwise_or(v[..., 2] | (v[..., 1] << 10), v[..., 0] << 20, out=out)
    return v[..., 2] | (v[..., 1] << 10) | (v[..., 0] << 20)


def unpack_rgb10a2(packed: torch.Tensor, hdr: bool = False) -> torch.Tensor:
    """HxW int32 (packed RGB10A2) -> HxWx3 uint8 BGR, rounded; with hdr, float16 0..1."""
    r, g, b = packed & 1023, (packed >> 10) & 1023, (packed >> 20) & 1023
    bgr = torch.stack([b, g, r], -1).float()
    if hdr:
        return bgr.div_(1023.0).to(torch.float16)
    return bgr.mul_(255.0 / 1023.0).round_().to(torch.uint8)
