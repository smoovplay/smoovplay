"""HDR maths (framegen/hdr.py), on the CPU: PQ, the BT.2390 roll-off, BT.2020 colours, and decoded 10-bit frames
turned into the pipeline's SDR (sdr8) and HDR10 (pq16) frames."""
from fractions import Fraction

import av
import numpy as np
import pytest
import torch

from framegen import hdr


def t(x):
    return torch.tensor(x, dtype=torch.float64)


def test_pq_known_values():
    # SMPTE ST 2084 / BT.2408 reference points
    assert hdr.pq_of_nits(10000) == pytest.approx(1.0, abs=1e-9)
    assert hdr.pq_of_nits(1000) == pytest.approx(0.7518, abs=2e-4)
    assert hdr.pq_of_nits(203) == pytest.approx(0.5807, abs=2e-4)
    assert hdr.pq_of_nits(100) == pytest.approx(0.5081, abs=2e-4)
    assert hdr.pq_of_nits(0) < 1e-5


def test_pq_round_trip():
    y = t(np.linspace(0, 1, 1001))
    assert torch.allclose(hdr.pq_eotf(hdr.pq_oetf(y)), y, atol=1e-9)


def test_bt2020_to_709_matrix():
    want = [[1.6605, -0.5876, -0.0728], [-0.1246, 1.1329, -0.0083], [-0.0182, -0.1006, 1.1187]]  # BT.2087
    assert np.allclose(hdr.BT2020_TO_709, want, atol=2e-4)
    assert np.allclose(hdr.BT2020_TO_709.sum(1), 1.0)  # white stays white
    assert np.allclose(hdr.BT2020_LUMA, [0.2627, 0.6780, 0.0593], atol=2e-4)


def test_eetf():
    src, dst = 1000.0, 203.0
    e = t(np.linspace(0, hdr.pq_of_nits(src), 2001))
    out = hdr.bt2390_eetf(e, src, dst)
    assert torch.all(out[1:] >= out[:-1])  # never darker for a brighter input
    assert float(out[-1]) == pytest.approx(hdr.pq_of_nits(dst), abs=1e-6)  # the game's peak lands on SDR white
    ks = 1.5 * hdr.pq_of_nits(dst) / hdr.pq_of_nits(src) - 0.5
    below = e / hdr.pq_of_nits(src) < ks
    assert torch.allclose(out[below], e[below], atol=1e-12)  # shadows and mid-tones untouched
    assert torch.max(torch.abs(out[1:] - out[:-1])) < 1e-3  # no jump at the knee
    # a target at least as bright as the source changes nothing
    assert torch.equal(hdr.bt2390_eetf(e, 600.0, 1000.0), e)


def code10(nits_rgb, matrix=hdr.SPC_BT2020_NCL):
    """Linear BT.2020 light (nits) -> 10-bit limited-range PQ Y'CbCr codes for one pixel."""
    rgb = hdr.pq_oetf(t(nits_rgb) / 10000).numpy()
    kr, kb = hdr.KR_KB[matrix]
    y = kr * rgb[0] + (1 - kr - kb) * rgb[1] + kb * rgb[2]
    cb, cr = (rgb[2] - y) / (2 * (1 - kb)), (rgb[0] - y) / (2 * (1 - kr))
    return 64 + 876 * y, 512 + 896 * cb, 512 + 896 * cr


def flat(codes, h=4, w=4):
    y, u, v = codes
    return (torch.full((h, w), float(y)), torch.full((h // 2, w // 2), float(u)), torch.full((h // 2, w // 2), float(v)))


PQ = hdr.FrameInfo()


def test_yuv_to_rgb_10bit_limited():
    black = hdr.yuv_to_rgb(*flat((64, 512, 512)), PQ)
    white = hdr.yuv_to_rgb(*flat((940, 512, 512)), PQ)
    assert torch.allclose(black, torch.zeros_like(black), atol=1e-6)
    assert torch.allclose(white, torch.ones_like(white), atol=1e-6)
    kr, kb = hdr.KR_KB[hdr.SPC_BT2020_NCL]
    red = hdr.yuv_to_rgb(*flat((64 + 876 * kr, 512 - 896 * kr / (2 * (1 - kb)), 512 + 896 * 0.5)), PQ)[0, 0]
    assert torch.allclose(red, torch.tensor([1.0, 0.0, 0.0]), atol=1e-4)


def test_sdr8_greys():
    def grey(nits):
        return hdr.convert_yuv(*flat(code10([nits] * 3)), PQ, "sdr8", src_peak=1000)[0, 0]

    assert grey(0).tolist() == [0, 0, 0]
    assert grey(1000).tolist() == [255, 255, 255]  # the game's peak: SDR's white
    levels = [int(grey(n)[0]) for n in (1, 5, 20, 50, 100, 203, 400, 1000)]
    assert levels == sorted(levels) and len(set(levels)) == len(levels)
    assert 200 <= levels[5] <= 240  # HDR's reference white: bright, with room left for highlights
    g = grey(100)
    assert g[0] == g[1] == g[2]  # grey stays grey


def test_sdr8_colours_go_to_709():
    # a BT.709 red, sent as BT.2020 light: back to (about) pure red in SDR
    rgb2020 = hdr.BT709_TO_2020 @ np.array([50.0, 0.0, 0.0])
    b, g, r = hdr.convert_yuv(*flat(code10(rgb2020)), PQ, "sdr8").reshape(-1, 3)[0].tolist()
    assert abs(r - round((50 / 203) ** (1 / 2.4) * 255)) <= 1 and g <= 3 and b <= 3


def test_pq16_passthrough():
    codes = flat(code10([120.0, 40.0, 10.0]))
    rgb = hdr.yuv_to_rgb(*codes, PQ)
    out = hdr.convert_yuv(*codes, PQ, "pq16")
    assert out.dtype == torch.float16
    assert torch.allclose(out.float(), rgb.flip(-1), atol=1e-3)  # BGR, as it came
    back = hdr.pq_eotf(out[0, 0].double().flip(-1)) * 10000
    assert np.allclose(back.numpy(), [120.0, 40.0, 10.0], rtol=0.02, atol=0.5)


def test_pq16_rolls_off_to_display_peak():
    out = hdr.convert_yuv(*flat(code10([4000.0] * 3)), PQ, "pq16", src_peak=4000, display_peak=600)
    nits = float(hdr.pq_eotf(out[0, 0, 0].double()) * 10000)
    assert nits == pytest.approx(600, rel=0.02)


def test_sdr_stream_to_pq16_is_sdr_white():
    info = hdr.FrameInfo(bits=10, matrix=hdr.SPC_BT709, transfer=hdr.TRC_BT709, primaries=hdr.PRI_BT709)
    out = hdr.convert_yuv(*flat((940, 512, 512)), info, "pq16")
    assert float(out[0, 0, 0]) == pytest.approx(hdr.pq_of_nits(hdr.SDR_WHITE_NITS), abs=1e-3)
    sdr = hdr.convert_yuv(*flat((940, 512, 512)), info, "sdr8")
    assert sdr[0, 0].tolist() == [255, 255, 255]


def test_replay_matches_window():
    """pq16_to_sdr8 (instant replay while the window shows HDR) gives what "sdr" shows, within a level."""
    rng = np.random.default_rng(1)
    y = torch.from_numpy(rng.uniform(64, 940, (8, 8)))
    u = torch.from_numpy(rng.uniform(300, 700, (4, 4)))
    v = torch.from_numpy(rng.uniform(300, 700, (4, 4)))
    direct = hdr.convert_yuv(y, u, v, PQ, "sdr8").int()
    via = hdr.pq16_to_sdr8(hdr.convert_yuv(y, u, v, PQ, "pq16")).int()
    assert (direct - via).abs().max() <= 2


@pytest.mark.parametrize("grey", [True, False])
def test_luts_match_the_maths(grey):
    """The 3D LUTs that run on every frame (StreamConverter.convert, pq16_to_sdr8) against the per-pixel maths, on
    random pixels: near grey (as most of a game's are) or any colour at all (the worst case)."""
    rng = np.random.default_rng(2)
    lo, hi = (470, 554) if grey else (64, 960)
    y = torch.from_numpy(rng.uniform(64, 940, (128, 128)))
    u, v = (torch.from_numpy(rng.uniform(lo, hi, (64, 64))) for _ in range(2))
    lut = hdr.StreamConverter("sdr8", device="cpu").convert(y.float(), u.float(), v.float(), PQ)
    d = (lut.int() - hdr.convert_yuv(y, u, v, PQ, "sdr8").int()).abs().float()
    assert d.mean() < 0.3 and d.quantile(0.99) <= 4 and (d.max() <= 6 or not grey)
    pq = hdr.convert_yuv(y, u, v, PQ, "pq16")
    r = (hdr.pq16_to_sdr8(pq).int() - hdr.pq16_to_sdr8_exact(pq).int()).abs().float()
    assert r.mean() < 0.1 and r.quantile(0.99) <= 1
    for peak in (400.0, 600.0):  # HDR10 to a display dimmer than the game's highlights: rolled off
        got = hdr.StreamConverter("pq16", device="cpu", display_peak=peak).convert(y.float(), u.float(), v.float(), PQ)
        e = (got.double() - hdr.convert_yuv(y, u, v, PQ, "pq16", display_peak=peak).double()).abs() * 1023
        assert e.mean() < 0.3 and e.max() <= 5  # in 10-bit codes


def test_hlg_white():
    nits = hdr.hlg_to_nits(torch.ones(1, 3, dtype=torch.float64))
    assert float(nits[0, 0]) == pytest.approx(1000.0, rel=1e-3)  # HLG's peak on a 1,000-nit display


def test_sdr_graphics_on_hdr():
    img = np.array([[[255, 255, 255], [0, 0, 0], [128, 128, 128], [255, 170, 40]]], np.uint8)  # BGR
    out = hdr.sdr_to_pq16_image(img)
    assert out.dtype == np.uint16
    assert np.allclose(out[0, 0], round(hdr.pq_of_nits(203) * 65535), atol=1)
    assert out[0, 1].max() < 100
    assert out[0, 2, 0] == out[0, 2, 1] == out[0, 2, 2]
    b, g, r = out[0, 3].astype(int)  # the accent blue keeps blue the strongest
    assert b > g > r


def p10_frame(y, u, v, tag=True):
    """A yuv420p10le PyAV frame from code planes."""
    h, w = y.shape
    arr = np.concatenate([y, u.reshape(h // 4, w), v.reshape(h // 4, w)]).astype(np.uint16)
    fr = av.VideoFrame.from_ndarray(arr, format="yuv420p10le")
    if tag:
        fr.color_trc, fr.color_primaries, fr.colorspace = hdr.TRC_PQ, hdr.PRI_BT2020, hdr.SPC_BT2020_NCL
    return fr


@pytest.mark.parametrize("tag", [True, False])
def test_stream_converter_software_frames(tag):
    yc, uc, vc = (round(c) for c in code10([203.0] * 3))
    fr = p10_frame(np.full((8, 16), yc), np.full((4, 8), uc), np.full((4, 8), vc), tag)
    conv = hdr.StreamConverter("sdr8", device="cpu")
    out = conv(fr)
    assert out.shape == (8, 16, 3) and out.dtype == torch.uint8
    want = hdr.convert_yuv(*flat((yc, uc, vc), 8, 16), PQ, "sdr8")
    assert torch.equal(out, want)  # untagged 10-bit is taken as HDR10, as a PS5 sends it
    # ...and the log says which it was (the stream's own tags, or taken from the bit depth)
    assert conv.describe() == "10-bit PQ BT.2020" + ("" if tag else " (no colour tags: taken as HDR10)")
    assert hdr.StreamConverter("pq16", device="cpu")(fr).dtype == torch.float16


def test_stream_converter_p010_on_gpu_layout():
    """NVDEC's P010 (samples in the top 10 bits), as DLPack may type it: uint16, int16 or bytes."""
    y10 = torch.tensor([[64, 940], [502, 700]])
    for as_type in ("u16", "i16", "bytes"):
        y = y10 << 6
        if as_type == "u16":
            y = y.to(torch.uint16)
        elif as_type == "i16":
            y = y.to(torch.int16)
        else:
            y = y.to(torch.int16).view(torch.uint8)
        assert torch.equal(hdr._u16(y) >> 6, y10)


def test_libx265_main10_round_trip():
    """A real HDR10 stream (libx265 Main10, PQ, BT.2020, CPU), decoded and converted as Remote Play's is."""
    if "libx265" not in av.codecs_available:
        pytest.skip("no libx265 in this PyAV")
    w = h = 64
    enc = av.CodecContext.create("libx265", "w")
    enc.width, enc.height, enc.pix_fmt = w, h, "yuv420p10le"
    enc.time_base, enc.framerate = Fraction(1, 60), Fraction(60)
    enc.options = {"preset": "ultrafast", "x265-params": "log-level=none:colorprim=bt2020:transfer=smpte2084:"
                   "colormatrix=bt2020nc:lossless=1"}
    enc.open()
    yc, uc, vc = (round(c) for c in code10([203.0] * 3))
    packets = []
    for i in range(3):
        fr = p10_frame(np.full((h, w), yc), np.full((h // 2, w // 2), uc), np.full((h // 2, w // 2), vc))
        fr.pts = i
        packets += enc.encode(fr)
    packets += enc.encode(None)
    dec = av.CodecContext.create("hevc", "r")
    frames = [f for p in packets for f in dec.decode(p)]
    assert frames and frames[0].format.name == "yuv420p10le"
    assert frames[0].color_trc == hdr.TRC_PQ
    out = hdr.StreamConverter("sdr8", device="cpu")(frames[0])
    want = hdr.convert_yuv(*flat((yc, uc, vc), h, w), PQ, "sdr8")
    assert torch.equal(out, want)


# ---------------------------------------------------------------- HDR output: the pipeline carrying float16 frames
from framegen.interpolator import DEFAULT_MODEL_DIR  # noqa: E402

needs_weights = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / "flownet_v4.22.lite.pkl").exists(),
                                   reason="RIFE weights not downloaded")


def scene(i, w=64, h=64):
    """A moving bright square on a gradient, as 8-bit BGR."""
    img = np.tile(np.linspace(20, 120, w, dtype=np.float32)[None, :, None], (h, 1, 3))
    x = 8 + 4 * i
    img[20:40, x:x + 16] = (60, 160, 250)
    return img.astype(np.uint8)


@needs_weights
def test_rife_sees_hdr_frames_as_it_sees_sdr_ones():
    """With hdr, RIFE takes and gives 0..1 half floats; the model's input is the same values either way."""
    from framegen.interpolator import RifeInterpolator
    a, b = scene(0), scene(2)
    outs = {}
    for hdr_on in (False, True):
        r = RifeInterpolator(64, 64, "4.22.lite", 1.0, fp16=False, cuda_graph=False, device="cpu", hdr=hdr_on)
        for slot, img in ((0, a), (1, b)):
            f = torch.from_numpy(img)
            r.load(f.float().div(255).half() if hdr_on else f)
            r.prepare(slot)
        out = r.interpolate(1, 0.5)
        assert out.dtype == (torch.float16 if hdr_on else torch.uint8)
        outs[hdr_on] = out.float() * (255 if hdr_on else 1)
        if hdr_on:  # the 8-bit stand-in for the pipeline's checks
            assert r.in_u8.dtype == torch.uint8 and r.in_u8.shape == (64, 64, 3)
    assert (outs[True] - outs[False]).abs().max() <= 1.5  # half-float input vs 8 bits: rounding only


@needs_weights
def test_pipeline_carries_hdr10_frames():
    """A 30 fps game at 60 Hz, in HDR10 frames, through Pipeline.run: what reaches the window stays float16, the
    real frames exactly as they came, the in-between ones between them."""
    import time

    from framegen.interpolator import RifeInterpolator
    from framegen.io import Tick
    from framegen.pipeline import Pipeline

    frames = [torch.from_numpy(scene(i // 2)).float().div(255).mul(0.75).half() for i in range(16)]

    def ticks():  # each arriving as it's taken, so none counts as late however slow the CPU is
        for i, f in enumerate(frames):
            yield Tick(f, time.perf_counter(), i, new_hint=(i % 2 == 0))

    class Sink:
        sinks, gpu = [1], False
        got = []

        def write(self, frame, text=None):
            Sink.got.append((frame.numpy() if torch.is_tensor(frame) else frame).copy())
            return None

    r = RifeInterpolator(64, 64, "4.22.lite", 1.0, fp16=False, cuda_graph=False, device="cpu", hdr=True)
    st = Pipeline(r).run(ticks(), Sink())
    assert st.actions["interp"] > 0
    assert all(g.dtype == np.float16 and g.shape == (64, 64, 3) for g in Sink.got)
    real = {f.numpy().tobytes() for f in frames}
    made = [g for g in Sink.got if g.tobytes() not in real]
    assert made, "no in-between frames"
    assert all(0.0 <= g.min() and g.max() <= 0.76 for g in made)


def test_vfg_packs_hdr_frames_in_10_bits():
    from framegen.vfg import pack_rgb10a2, unpack_rgb10a2
    v = torch.tensor([[[0.0, 0.5, 1.0], [0.25, 0.75, 0.001]]], dtype=torch.float16)  # BGR
    packed = pack_rgb10a2(v)
    assert int(packed[0, 0]) == 1023 | (512 << 10) | (0 << 20)  # R=1.0 B=0.0 (0.5 rounds to 512 of 1023)
    back = unpack_rgb10a2(packed, hdr=True)
    assert back.dtype == torch.float16 and (back.float() - v.float()).abs().max() <= 0.5 / 1023 + 1e-3


def test_replay_turns_hdr_frames_into_sdr():
    from framegen.replay import Recorder
    r = Recorder.__new__(Recorder)  # only _put's conversion: no encoders
    seen = []
    r._zero = lambda t: None
    r._convert = lambda name, frame: seen.append(frame) or None
    r._warn = lambda *a: None
    r.hdr_to_sdr = hdr.pq16_to_sdr8
    try:
        r._put("after", torch.full((4, 4, 3), hdr.pq_of_nits(203), dtype=torch.float16), 0.0)
    except Exception:  # (whatever _put does after converting needs a real Recorder)
        pass
    assert seen and seen[0].dtype == torch.uint8 and 200 <= int(seen[0][0, 0, 0]) <= 240
