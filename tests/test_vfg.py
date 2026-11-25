import torch

from framegen.vfg import pack_rgb10a2, unpack_rgb10a2


def test_rgb10a2_round_trip_is_exact_for_every_8bit_value():
    v = torch.arange(256)
    bgr = torch.stack([v, v.flip(0), (v * 7) % 256], -1).to(torch.uint8).view(16, 16, 3)  # B, G, R differ
    packed = pack_rgb10a2(bgr)
    assert packed.dtype == torch.int32 and packed.shape == (16, 16)
    assert int(packed.min()) >= 0  # alpha bits left 0, so it fits a signed int32 and views as uint32 unchanged
    assert torch.equal(unpack_rgb10a2(packed), bgr)


def test_rgb10a2_layout_is_red_in_the_low_bits():
    px = torch.tensor([[[0, 0, 255]]], dtype=torch.uint8)  # pure red in BGR
    assert int(pack_rgb10a2(px)) == 1023
    px = torch.tensor([[[255, 0, 0]]], dtype=torch.uint8)  # pure blue
    assert int(pack_rgb10a2(px)) == 1023 << 20


def test_pack_rgb10a2_in_place_matches_allocating():
    v = torch.arange(256)
    bgr = torch.stack([v, v.flip(0), (v * 7) % 256], -1).to(torch.uint8).view(16, 16, 3)
    out = torch.empty((16, 16), dtype=torch.int32)
    res = pack_rgb10a2(bgr, out=out)
    assert res is out
    assert torch.equal(out, pack_rgb10a2(bgr))


def test_vfg_rgb10a2_passthrough_and_lazy_out_u8(monkeypatch):
    if not torch.cuda.is_available():
        return
    from types import SimpleNamespace
    from framegen import vfg

    class _FakeVfg10:
        def __init__(self, **kw):
            pass

        def load(self):
            pass

        def run_at_timestep(self, prev, cur, t, stream_ptr=None):
            # Emit a non-8-bit 10-bit value (513 in R) to verify no 10->8->10 quantization
            out = torch.full(cur.shape, 513, dtype=torch.int32, device=cur.device)
            return SimpleNamespace(image=out)

    fake = SimpleNamespace(ImageEncoding=SimpleNamespace(RGB10A2=0, RGB8=1), VideoFrameGeneration=_FakeVfg10)
    monkeypatch.setattr(vfg, "import_vfx", lambda: fake)
    it = vfg.VfgInterpolator(8, 4, encoding="rgb10a2")
    frame = torch.full((4, 8, 3), 128, dtype=torch.uint8, device="cuda")
    it.load(frame)
    it.prepare(0)
    it.prepare(1)

    it.rgb10a2_out = True
    out = it.interpolate(1, 0.5)
    assert out.dtype == torch.int32 and out.shape == (4, 8)
    assert int(out[0, 0]) == 513
    assert it._out_u8_dirty is True
    bgr = it.out_u8
    assert it._out_u8_dirty is False
    assert bgr.dtype == torch.uint8 and bgr.shape == (4, 8, 3)

    it.rgb10a2_out = False
    out_bgr = it.interpolate(1, 0.5)
    assert out_bgr.dtype == torch.uint8 and out_bgr.shape == (4, 8, 3)
    assert it._out_u8_dirty is False


def test_vfg_prewarms_on_init(monkeypatch):
    if not torch.cuda.is_available():
        return
    from types import SimpleNamespace
    from framegen import vfg

    calls = []

    class _FakeVfgWarm:
        def __init__(self, **kw):
            pass

        def load(self):
            pass

        def run_at_timestep(self, prev, cur, t, stream_ptr=None):
            calls.append((prev.dtype, cur.dtype, t))
            return SimpleNamespace(image=torch.zeros(cur.shape, dtype=torch.int32, device=cur.device))

    fake = SimpleNamespace(ImageEncoding=SimpleNamespace(RGB10A2=0, RGB8=1), VideoFrameGeneration=_FakeVfgWarm)
    monkeypatch.setattr(vfg, "import_vfx", lambda: fake)
    it = vfg.VfgInterpolator(8, 4, encoding="rgb10a2")
    assert len(calls) == 1
    assert calls[0] == (torch.uint32, torch.uint32, 0.5)
    assert it._slot is None
    assert it._out_u8_dirty is False
