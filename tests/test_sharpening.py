"""Settings "Sharpening" (#58): AMD's RCAS after any upscaler, or none. It took over "FSR sharpening" (fsr_sharpness,
up to 0.3.14), and a saved level is carried over so nothing looks different after updating: still only while FSR 1
enlarges the picture, until the player changes Sharpening (sharpen_fsr_only). The shader tests need OpenGL 4.5 (4.1
on a Mac) and a hidden window, and skip without them."""
import json
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from framegen.settings import BY_KEY, SHARPEN_STOPS, Settings

ROOT = Path(__file__).resolve().parent.parent


def _old_file(tmp_path, **values):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hz": 90, **values}), encoding="utf-8")
    return path


# ---- carried over from FSR sharpening
@pytest.mark.parametrize("old, level", [(0.0, "max"), (0.25, "high"), (0.5, "medium"), (1.0, "low"), (2.0, "minimal"),
                                        (1, "low")])  # (JSON may hold 1.0 as 1)
def test_an_fsr_players_level_carries_over_for_fsr_only(tmp_path, old, level):
    path = _old_file(tmp_path, scaler="fsr", fsr_sharpness=old)
    st = Settings.load(path)
    assert (st.sharpening, st.sharpen_fsr_only) == (level, True)
    saved = json.loads(path.read_text(encoding="utf-8"))  # written straight away, for the launcher's page
    assert saved["sharpening"] == level and saved["sharpen_fsr_only"] is True and saved["hz"] == 90
    assert "fsr_sharpness" not in saved
    assert (Settings.load(path).sharpening, Settings.load(path).sharpen_fsr_only) == (level, True)  # and stays


@pytest.mark.parametrize("scaler", ["vsr", "bilinear"])
def test_rtx_vsr_and_bilinear_players_start_at_off(tmp_path, scaler):
    st = Settings.load(_old_file(tmp_path, scaler=scaler, fsr_sharpness=0.0))
    assert (st.sharpening, st.sharpen_fsr_only, st.scaler) == ("off", False, scaler)


def test_an_unknown_upscaler_counts_as_fsr_the_default(tmp_path):
    st = Settings.load(_old_file(tmp_path, scaler="nope", fsr_sharpness=0.25))
    assert (st.sharpening, st.sharpen_fsr_only) == ("high", True)


@pytest.mark.parametrize("values", [{"sharpening": "low", "fsr_sharpness": 0.0}, {"fsr_sharpness": True},
                                    {"fsr_sharpness": 0.7}, {"fsr_sharpness": "max"}, {}])
def test_nothing_else_is_carried_over_or_rewritten(tmp_path, values):
    path = _old_file(tmp_path, **values)
    before = path.read_text(encoding="utf-8")
    st = Settings.load(path)
    assert path.read_text(encoding="utf-8") == before  # loading alone doesn't write
    want = values.get("sharpening", BY_KEY["sharpening"].default)
    assert (st.sharpening, st.sharpen_fsr_only) == (want, "sharpening" not in values)


def test_a_new_install_sharpens_as_fsr_sharpening_did(tmp_path):
    st = Settings.load(tmp_path / "absent.json")
    assert (st.sharpening, st.sharpen_fsr_only) == ("medium", True)
    assert not (tmp_path / "absent.json").exists()


def test_changing_sharpening_ends_the_carried_over_way(tmp_path):
    path = _old_file(tmp_path, scaler="fsr", fsr_sharpness=0.5)
    st = Settings.load(path)
    heard = []
    st.watch(heard.append)
    st.set("hz", 60)  # anything else: still carried over
    assert st.sharpen_fsr_only is True
    st.set("sharpening", "medium")  # picking it, even the same level: from now on wherever the picture is drawn
    assert st.sharpen_fsr_only is False and json.loads(path.read_text())["sharpen_fsr_only"] is False
    assert heard == ["hz", "sharpening"]  # once per change, for the key that was set


def test_the_command_line_sharpens_everywhere_for_that_run(tmp_path):
    path = _old_file(tmp_path, scaler="fsr", fsr_sharpness=0.5)
    st = Settings.load(path)
    st.override("sharpening", "max")
    assert (st.sharpening, st.sharpen_fsr_only) == ("max", False)
    assert json.loads(path.read_text())["sharpen_fsr_only"] is True  # not saved


def test_the_launchers_list_carries_an_old_file_over_and_leaves_hidden_state_out(tmp_path):
    path = _old_file(tmp_path, scaler="fsr", fsr_sharpness=0.25)
    env = dict(os.environ, SMOOVPLAY_SETTINGS=str(path), PYTHONNOUSERSITE="1")
    out = subprocess.run([sys.executable, "-m", "framegen.settings"], cwd=ROOT, env=env, capture_output=True,
                         text=True, timeout=120, check=True).stdout
    keys = [s["key"] for s in json.loads(out.strip().splitlines()[-1])["specs"]]
    assert "sharpening" in keys and "sharpen_fsr_only" not in keys and "fsr_sharpness" not in keys
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["sharpening"] == "high" and saved["sharpen_fsr_only"] is True


# ---- when the video window sharpens
def _window(sharpening="medium", fsr_only=False, fsr=True):
    from framegen.io import GLWindowSink
    me = SimpleNamespace(settings=Settings({"sharpening": sharpening, "sharpen_fsr_only": fsr_only}),
                         fsr=object() if fsr else None)
    return lambda enlarging_with_fsr: GLWindowSink._sharpness(me, enlarging_with_fsr)


def test_sharpening_applies_after_every_upscaler_and_with_none():
    sharp = _window("high")
    assert sharp(True) == sharp(False) == SHARPEN_STOPS["high"] == 0.25


def test_carried_over_it_applies_only_while_fsr_enlarges():
    sharp = _window("low", fsr_only=True)
    assert sharp(True) == 1.0 and sharp(False) is None


def test_off_or_without_fsrs_shaders_nothing_is_sharpened():
    assert _window("off")(True) is None and _window("off")(False) is None
    assert _window("max", fsr=False)(True) is None  # (RCAS is FSR 1's: the row is greyed out then)


# ---- the shaders
@pytest.fixture(scope="module")
def GL():
    glfw = pytest.importorskip("glfw")
    from OpenGL import GL as gl

    from framegen.glsl import context_hints
    if not glfw.init():
        pytest.skip("no OpenGL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    if sys.platform == "darwin":
        context_hints(glfw)  # 4.1 core
    else:
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 4)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 5)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
    win = glfw.create_window(64, 64, "test", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("no OpenGL 4.5 context")
    glfw.make_context_current(win)
    yield gl
    glfw.destroy_window(win)
    glfw.terminate()


def _target(GL, w, h, img=None):
    """An RGBA8 texture (img: HxWx4 uint8, row 0 = the image's top, as the window keeps frames) and its framebuffer."""
    tex = GL.glGenTextures(1)
    GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
    for p in (GL.GL_TEXTURE_MIN_FILTER, GL.GL_TEXTURE_MAG_FILTER):
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_LINEAR)
    for p in (GL.GL_TEXTURE_WRAP_S, GL.GL_TEXTURE_WRAP_T):
        GL.glTexParameteri(GL.GL_TEXTURE_2D, p, GL.GL_CLAMP_TO_EDGE)
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE,
                    None if img is None else np.ascontiguousarray(img))
    fbo = GL.glGenFramebuffers(1)
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, tex, 0)
    return tex, fbo


def _read(GL, fbo, w, h):
    """The framebuffer as an image, row 0 = memory row 0 (for an upright drawing that's the picture's bottom)."""
    GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    return np.frombuffer(GL.glReadPixels(0, 0, w, h, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(h, w, 4)


def _picture(w, h, seed=3):
    """Edges and detail to sharpen: blocks, a gradient and noise."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    base = (((x // 7 + y // 5) % 2) * 120 + x * 100 // w).astype(np.int32)
    img = np.clip(base[..., None] + rng.integers(-12, 13, (h, w, 3)), 0, 255).astype(np.uint8)
    return np.dstack([img, np.full((h, w), 255, np.uint8)])


def test_sharpening_on_its_own_is_fsrs_own_rcas(GL):
    """FSR 1 = EASU then RCAS. EASU alone (Sharpening off), sharpened on its own, must give exactly what FSR gives
    with its own RCAS, so Sharpening after RTX VSR or bilinear is the same sharpening FSR players have."""
    from framegen.fsr import FsrScaler
    sw, sh, dw, dh = 96, 54, 160, 90
    src, _ = _target(GL, sw, sh, _picture(sw, sh))
    fsr = FsrScaler(GL, None)
    try:
        for stops in SHARPEN_STOPS.values():
            out, out_fbo = _target(GL, dw, dh)
            fsr.sharpness = stops
            fsr.draw(src, sw, sh, out_fbo, 0, 0, dw, dh)
            whole = _read(GL, out_fbo, dw, dh)

            fsr.sharpness = None  # EASU only: upright, so read back bottom row first; flipped to the image's order
            fsr.draw(src, sw, sh, out_fbo, 0, 0, dw, dh)
            easu = _read(GL, out_fbo, dw, dh)[::-1]
            staged, _ = _target(GL, dw, dh, easu)
            fsr.sharpen(staged, out_fbo, 0, 0, dw, dh, stops)
            apart = _read(GL, out_fbo, dw, dh)
            assert np.array_equal(whole, apart), stops
            assert not np.array_equal(easu[::-1], whole)  # (and it did sharpen)
            GL.glDeleteTextures([out, staged])
    finally:
        fsr.close()


def test_a_clips_frame_keeps_its_row_order_and_more_is_sharper(GL):
    from framegen.fsr import FsrScaler
    w, h = 80, 48
    img = _picture(w, h, seed=5)
    src, _ = _target(GL, w, h, img)
    out, out_fbo = _target(GL, w, h)
    fsr = FsrScaler(GL, None)
    try:
        detail = {}
        for name, stops in SHARPEN_STOPS.items():
            fsr.sharpen(src, out_fbo, 0, 0, w, h, stops, flip=False)  # a clip's frame, read back as it's stored
            kept = _read(GL, out_fbo, w, h)
            fsr.sharpen(src, out_fbo, 0, 0, w, h, stops)  # on the screen: upright
            assert np.array_equal(_read(GL, out_fbo, w, h)[::-1], kept), name
            lap = np.abs(np.diff(kept[..., :3].astype(int), 2, axis=1)).mean()  # local contrast
            detail[name] = lap
            assert np.abs(kept[..., :3].astype(int) - img[..., :3]).mean() < 40, name  # the same picture
        order = ["minimal", "low", "medium", "high", "max"]
        assert [detail[k] for k in order] == sorted(detail[k] for k in order)  # more is crisper
        base = np.abs(np.diff(img[..., :3].astype(int), 2, axis=1)).mean()
        assert detail["minimal"] > base
    finally:
        fsr.close()
        GL.glDeleteTextures([src, out])
