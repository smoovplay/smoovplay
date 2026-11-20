"""DirectML engine (framegen/dml.py), for AMD and Intel cards: the conversion to ONNX, checked on the CPU against
PyTorch, plus setup and Auto-detect without NVIDIA's CUDA. Nothing here needs a graphics card; running the converted
graphs on one through DirectML is scripts/check_dml.py's job."""
from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

ort = pytest.importorskip("onnxruntime")

from framegen import dml  # noqa: E402
from framegen.autodetect import Display, recommend  # noqa: E402
from framegen.interpolator import DEFAULT_MODEL_DIR, RifeInterpolator  # noqa: E402

MODEL = "4.22.lite"
needs_weights = pytest.mark.skipif(not (DEFAULT_MODEL_DIR / f"flownet_v{MODEL}.pkl").exists(),
                                   reason="RIFE weights not downloaded")


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    mse = np.mean((a.astype(np.float32) - b.astype(np.float32)) ** 2)
    return float("inf") if mse == 0 else 10 * np.log10(255.0**2 / mse)


def two_frames(w: int, h: int, shift: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """A smooth coloured pattern and the same pattern moved `shift` pixels right: something to find motion in."""
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

    def at(x):
        return np.stack([np.sin(x / 9) * 70 + np.cos(yy / 7) * 50 + 128,
                         np.cos(x / 13) * 80 + 128 + 0 * yy,
                         np.sin((x + yy) / 11) * 90 + 128], -1).clip(0, 255).astype(np.uint8)

    return at(xx), at(xx - shift)


@needs_weights
def test_converted_graphs_draw_what_pytorch_draws(tmp_path, monkeypatch):
    monkeypatch.setattr(dml, "MODEL_DIR", tmp_path)
    w, h = 160, 96
    enc_path, flow_path, mask_path = dml.convert(MODEL, 1.0, w, h)
    assert enc_path.exists() and flow_path.exists() and mask_path.exists()
    assert not list(tmp_path.glob("*.part"))

    a, b = two_frames(w, h)
    cpu = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(str(enc_path), providers=cpu)
    flow = ort.InferenceSession(str(flow_path), providers=cpu)
    s0, s1 = enc.run(None, {"frame": a})[0], enc.run(None, {"frame": b})[0]
    no_hud = np.zeros((1, 1) + tuple(s0.shape[2:]), np.float32)
    got = flow.run(None, {"src0": s0, "src1": s1, "t": np.full((1, 1, 1, 1), 0.5, np.float32), "w": no_hud})[0]

    ref = RifeInterpolator(w, h, MODEL, 1.0, fp16=False, cuda_graph=False, device="cpu")
    ref.load(torch.from_numpy(a))
    ref.prepare(0)
    ref.load(torch.from_numpy(b))
    ref.prepare(1)
    want = ref.interpolate(1, 0.5).numpy()

    assert got.shape == (h, w, 3) and got.dtype == np.uint8
    assert psnr(got, want) > 40  # fp16 convolutions against fp32: tiny rounding differences only
    assert psnr(got, a) < psnr(got, want)  # and it's an in-between frame, not a copy of a real one


def hud_frames(w: int, h: int, shift: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """two_frames() with a still white box (a HUD) at rows 8-23, cols 8-39, the same in both frames."""
    a, b = (f.copy() for f in two_frames(w, h, shift))
    a[8:24, 8:40] = b[8:24, 8:40] = 255
    return a, b


@needs_weights
def test_steady_hud_graphs_match_the_pytorch_path(tmp_path, monkeypatch):
    """The mask graph finds the still HUD's edges and the flow graph puts the real pixels there, as RIFE on CUDA does
    (framegen/steady.py); the moving background stays the model's."""
    from framegen.steady import STEADY_HUD_THRESHOLD, SteadyHud
    monkeypatch.setattr(dml, "MODEL_DIR", tmp_path)
    w, h = 160, 96
    enc_path, flow_path, mask_path = dml.convert(MODEL, 1.0, w, h)
    a, b = hud_frames(w, h)
    cpu = ["CPUExecutionProvider"]
    enc = ort.InferenceSession(str(enc_path), providers=cpu)
    flow = ort.InferenceSession(str(flow_path), providers=cpu)
    mask = ort.InferenceSession(str(mask_path), providers=cpu)
    s0, s1 = enc.run(None, {"frame": a})[0], enc.run(None, {"frame": b})[0]
    thr = np.full((1, 1, 1, 1), STEADY_HUD_THRESHOLD, np.float32)
    t = np.full((1, 1, 1, 1), 0.5, np.float32)
    got_w = mask.run(None, {"src0": s0, "src1": s1, "thr": thr})[0]

    hud = SteadyHud(s0.shape[2], s0.shape[3], "cpu")  # the reference mask, from the same padded RGB planes
    r0, r1 = torch.from_numpy(s0[:, :3]), torch.from_numpy(s1[:, :3])
    hud.edges(0, r0)
    hud.mask(0, 1, r0, r1)
    assert got_w.shape == tuple(hud.w.shape)
    assert np.abs(got_w - hud.w.numpy()).max() < 0.02  # fp32 against the CUDA path's fp16 maths
    assert got_w[0, 0, 8, 20] == 1.0 and got_w[0, 0, 40, 100] == 0.0  # the box's top edge; the moving background

    off = flow.run(None, {"src0": s0, "src1": s1, "t": t, "w": np.zeros_like(got_w)})[0]
    on = flow.run(None, {"src0": s0, "src1": s1, "t": t, "w": got_w})[0]
    edge = (slice(8, 10), slice(10, 38))  # just inside the box's top edge
    assert np.array_equal(on[edge], a[edge])  # the HUD as the game drew it, in both real frames
    assert np.array_equal(on[40:90, 60:150], off[40:90, 60:150])  # away from the HUD: the model's frame


@needs_weights
def test_conversion_is_cached_and_the_encoder_shared(tmp_path, monkeypatch):
    monkeypatch.setattr(dml, "MODEL_DIR", tmp_path)
    exported = []
    monkeypatch.setattr(dml, "_export", lambda module, args, path, *rest: (exported.append(path.name),
                                                                           path.write_bytes(b"x")))
    dml.convert(MODEL, 1.0, 64, 64)
    dml.convert(MODEL, 1.0, 64, 64)  # cached: nothing more
    dml.convert(MODEL, 0.5, 64, 64)  # half motion detail: its own flow graph, the same encoder and mask
    assert len(exported) == 4
    assert sum("encode" in n for n in exported) == 1 and sum("flow" in n for n in exported) == 2
    assert sum("mask" in n for n in exported) == 1


def test_unavailable_without_the_directml_package(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["CPUExecutionProvider"])
    assert "onnxruntime-directml" in dml.unavailable_reason()


def test_unavailable_without_a_graphics_adapter(monkeypatch):
    monkeypatch.setattr(ort, "get_available_providers", lambda: ["DmlExecutionProvider", "CPUExecutionProvider"])
    monkeypatch.setattr(dml, "adapters", lambda: [])
    assert "adapter" in dml.unavailable_reason()


def test_auto_detect_suggests_settings_for_an_amd_card():
    timing = {"full": {"ms": 16.0, "engine": "dml", "estimated": False},
              "half": {"ms": 11.0, "engine": "dml", "estimated": False}}
    r = recommend({}, timing, Display(2560, 1440, 144, 144), gpu="AMD Radeon RX 7800 XT", trt=False,
                  vsr="it needs an NVIDIA RTX card")
    assert "DirectML" in r["reasons"]["engine"]
    assert r["settings"]["scaler"] == "fsr"
    assert r["settings"]["hz"] == 60  # 16 ms (plus FSR) doesn't fit 90 fps's 11.1 ms


def _prepare(monkeypatch, why):
    from framegen import interpolator, prepare
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(dml, "unavailable_reason", lambda: why)
    monkeypatch.setattr(dml, "pick_adapter", lambda: {"index": 0, "name": "AMD Radeon RX 7800 XT", "vram_gb": 16.0})
    converted = []
    monkeypatch.setattr(dml, "convert", lambda model, scale, w, h: converted.append((model, scale, w, h)))
    monkeypatch.setattr(interpolator, "ensure_weights", lambda model: None)
    monkeypatch.setattr(prepare, "pick_settings", lambda: converted.append("settings"))
    monkeypatch.setattr(sys, "argv", ["prepare"])
    return prepare.main(), converted


def test_setup_prepares_directml_without_cuda(monkeypatch, capsys):
    rc, converted = _prepare(monkeypatch, None)
    assert rc == 0
    assert converted == [(MODEL, 1.0, 1920, 1080), (MODEL, 0.5, 1920, 1080), "settings"]
    assert "DirectML (Experimental)" in capsys.readouterr().out


def test_setup_fails_clearly_when_neither_engine_can_run(monkeypatch, capsys):
    rc, converted = _prepare(monkeypatch, "no DirectX 12 graphics adapter found")
    assert rc == 2 and converted == []
    assert "no DirectX 12 graphics adapter found" in capsys.readouterr().out
