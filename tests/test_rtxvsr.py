"""RTX VSR (framegen/rtxvsr.py): which way it runs, NVIDIA's quality levels, and the packed 10-bit frames NVIDIA's
package takes. The upscaling itself needs an RTX card and NVIDIA's package: checked by hand, not here."""
import torch

from framegen import rtxvsr
from framegen.rtxvsr import CLEAN, VfxVsr, pack_bgr10, pack_rgba10


def _fields(packed):
    p = packed.to(torch.int64) & 0xFFFFFFFF
    return p & 1023, (p >> 10) & 1023, (p >> 20) & 1023, p >> 30


def test_8_bit_frames_pack_into_10_bits_as_gl_reads_them():
    # RGB10_A2 read as GL_UNSIGNED_INT_2_10_10_10_REV: red in the low bits, as NVIDIA packs it
    rgba = torch.zeros((2, 3, 4), dtype=torch.uint8)
    rgba[0, 0, :3] = torch.tensor([255, 0, 0])
    rgba[0, 1, :3] = torch.tensor([0, 255, 0])
    rgba[0, 2, :3] = torch.tensor([0, 0, 255])
    rgba[1, 0, :3] = torch.tensor([1, 128, 254])
    out = torch.empty((2, 3), dtype=torch.int32)
    r, g, b, a = _fields(pack_rgba10(rgba, out))
    assert r[0].tolist() == [1023, 0, 0] and g[0].tolist() == [0, 1023, 0] and b[0].tolist() == [0, 0, 1023]
    assert (r[1, 0], g[1, 0], b[1, 0]) == (4, 514, 1019)  # v * 1023 / 255, exactly
    assert (a == 3).all()  # opaque


def test_every_8_bit_value_comes_back_from_10_bits():
    v = torch.arange(256, dtype=torch.uint8)
    rgba = torch.stack([v, v.flip(0), v, v], -1).reshape(16, 16, 4)
    r, g, _, _ = _fields(pack_rgba10(rgba, torch.empty((16, 16), dtype=torch.int32)))
    assert torch.equal((r.float() * 255 / 1023).round().to(torch.uint8).flatten(), v)  # as GL reads it, 8 bits back
    assert torch.equal((g.float() * 255 / 1023).round().to(torch.uint8).flatten(), v.flip(0))


def test_float_frames_round_to_10_bits():
    rgba = torch.tensor([[[0.0, 0.5, 1.0, 1.0], [1.2, -0.1, 0.25, 1.0]]], dtype=torch.float16)
    r, g, b, _ = _fields(pack_rgba10(rgba, torch.empty((1, 2), dtype=torch.int32)))
    assert r[0].tolist() == [0, 1023] and g[0].tolist() == [512, 0] and b[0].tolist() == [1023, 256]


def test_quality_levels_and_the_models_for_clean_video():
    assert [VfxVsr.level(q, False) for q in (1, 2, 3, 4)] == [1, 2, 3, 4]  # NVIDIA's LOW..ULTRA
    assert [VfxVsr.level(q, True) for q in (1, 2, 3, 4)] == [16, 17, 18, 19]  # HIGHBITRATE_LOW..ULTRA
    assert CLEAN == 15 and VfxVsr.level(0, False) == 1 and VfxVsr.level(9, True) == 19


def test_nvidias_package_first_the_sdk_for_developers(monkeypatch):
    monkeypatch.delenv("FRAMEGEN_VSR", raising=False)
    monkeypatch.setattr(rtxvsr, "vfx_reason", lambda: None)
    monkeypatch.setattr(rtxvsr, "sdk_reason", lambda: None)
    assert rtxvsr.backend() == ("vfx", None)
    monkeypatch.setenv("FRAMEGEN_VSR", "sdk")
    assert rtxvsr.backend() == ("sdk", None)
    monkeypatch.setattr(rtxvsr, "sdk_reason", lambda: "NV_RTX_VIDEO_SDK isn't set")
    assert rtxvsr.backend() == ("vfx", None)  # (only one there: that one)
    monkeypatch.setattr(rtxvsr, "vfx_reason", lambda: "it comes with NVIDIA VFG's download")
    name, why = rtxvsr.backend()
    assert name is None and why == "it comes with NVIDIA VFG's download"  # what people see is the package's reason
    assert rtxvsr.unavailable_reason() == why


def test_a_broken_package_check_never_stops_the_window(monkeypatch):
    monkeypatch.delenv("FRAMEGEN_VSR", raising=False)

    def broken():
        raise OSError("bad install")

    monkeypatch.setattr(rtxvsr, "vfx_reason", broken)
    monkeypatch.setattr(rtxvsr, "sdk_reason", lambda: "NV_RTX_VIDEO_SDK isn't set")
    name, why = rtxvsr.backend()
    assert name is None and "couldn't be checked" in why


def test_only_the_sdk_way_may_log_its_timings():
    assert VfxVsr.private and not rtxvsr.RtxVsr.private  # NVIDIA's package's licence: no published benchmarks
    assert VfxVsr.out_format == "rgb10a2" and rtxvsr.RtxVsr.out_format == "rgba8"
    assert VfxVsr.in_format == "rgb10a2" and rtxvsr.RtxVsr.in_format == "rgba8"


def test_pack_bgr10_matches_pack_rgba10_for_all_8bit_and_float():
    v = torch.arange(256, dtype=torch.uint8)
    rgba = torch.stack([v, v.flip(0), ((v.int() * 7) % 256).to(torch.uint8), v], -1).reshape(16, 16, 4)
    bgr = rgba[..., [2, 1, 0]]
    out_rgba = torch.empty((16, 16), dtype=torch.int32)
    out_bgr = torch.empty((16, 16), dtype=torch.int32)
    assert torch.equal(pack_bgr10(bgr, out_bgr), pack_rgba10(rgba, out_rgba))

    f_rgba = torch.tensor([[[0.0, 0.5, 1.0, 1.0], [1.2, -0.1, 0.25, 1.0]]], dtype=torch.float16)
    f_bgr = f_rgba[..., [2, 1, 0]]
    assert torch.equal(pack_bgr10(f_bgr, torch.empty((1, 2), dtype=torch.int32)),
                       pack_rgba10(f_rgba, torch.empty((1, 2), dtype=torch.int32)))


def test_vfx_vsr_accepts_rgb10a2_bgr_and_rgba(monkeypatch):
    if not torch.cuda.is_available():
        return
    import ctypes
    from types import SimpleNamespace
    from framegen import vfg

    seen = []

    class _FakeVsrFx:
        def __init__(self, **kw):
            self.output_width = 0
            self.output_height = 0

        def load(self):
            pass

        def run(self, src, non_blocking=True, stream_ptr=None):
            seen.append(src.clone())
            return SimpleNamespace(image=torch.zeros((self.output_height, self.output_width),
                                                     dtype=torch.int32, device="cuda"))

        def close(self):
            pass

    fake_vfx = SimpleNamespace(ImageEncoding=SimpleNamespace(RGB10A2=0), VideoSuperRes=_FakeVsrFx)
    monkeypatch.setattr(vfg, "import_vfx", lambda: fake_vfx)
    monkeypatch.setattr(rtxvsr, "vfx_reason", lambda: None)
    monkeypatch.setattr(rtxvsr, "_cudart", lambda: SimpleNamespace(
        cudaMemcpy2DToArrayAsync=SimpleNamespace(argtypes=None),
        __call__=None,
    ))
    vsr = VfxVsr(8, 4, quality=2)
    vsr.rt = SimpleNamespace(cudaMemcpy2DToArrayAsync=lambda *a: 0)

    # 1. Direct RGB10A2 passthrough (preserves arbitrary 10-bit value 513)
    rgb10 = torch.full((4, 8), 513, dtype=torch.int32, device="cuda")
    vsr(rgb10, ctypes.c_void_p(1), 16, 8)
    assert seen[-1].dtype == torch.uint32 and int(seen[-1][0, 0]) == 513

    # 2. Direct BGR uint8
    bgr = torch.zeros((4, 8, 3), dtype=torch.uint8, device="cuda")
    bgr[..., 2] = 255
    vsr(bgr, ctypes.c_void_p(1), 16, 8)
    assert int(seen[-1][0, 0]) & 1023 == 1023

    # 3. RGBA uint8
    rgba = torch.zeros((4, 8, 4), dtype=torch.uint8, device="cuda")
    rgba[..., 0] = 255
    vsr(rgba, ctypes.c_void_p(1), 16, 8)
    assert int(seen[-1][0, 0]) & 1023 == 1023


def test_vfx_vsr_prewarms_on_init_and_dedups_same_size(monkeypatch):
    if not torch.cuda.is_available():
        return
    from types import SimpleNamespace
    from framegen import vfg

    runs = []

    class _FakeVsrWarm:
        def __init__(self, **kw):
            self.output_width = 0
            self.output_height = 0

        def load(self):
            pass

        def run(self, src, non_blocking=True, stream_ptr=None):
            runs.append((self.output_width, self.output_height))
            return SimpleNamespace(image=torch.zeros((self.output_height, self.output_width),
                                                     dtype=torch.int32, device="cuda"))

        def close(self):
            pass

    fake_vfx = SimpleNamespace(ImageEncoding=SimpleNamespace(RGB10A2=0), VideoSuperRes=_FakeVsrWarm)
    monkeypatch.setattr(vfg, "import_vfx", lambda: fake_vfx)
    monkeypatch.setattr(rtxvsr, "vfx_reason", lambda: None)
    monkeypatch.setattr(rtxvsr, "_cudart", lambda: SimpleNamespace(
        cudaMemcpy2DToArrayAsync=SimpleNamespace(argtypes=None),
    ))
    vsr = VfxVsr(8, 4, quality=2)
    assert runs == [(16, 8)]
    vsr.warm_up(16, 8)  # same size: no-op
    assert runs == [(16, 8)]
    vsr.warm_up(12, 6)  # new display size: warms once at new size
    assert runs == [(16, 8), (12, 6)]
    assert vsr.fx.output_width == 12 and vsr.fx.output_height == 6
