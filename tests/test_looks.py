"""Looks (framegen/looks.py, framegen/looks_gl.py): share codes, saved looks, the reference renderer, and the shaders
held to it (with a hidden OpenGL window, when there is one)."""
import io
import json
import re

import numpy as np
import pytest

from framegen import looks
from framegen.looks import BUILTIN, CHOICES, TYPES, TYPES_LIST, CodeError, Library, LookState, decode, encode, node, \
    normalize, render
from framegen.settings import BY_KEY, Settings

H, W = 72, 128


def _img() -> np.ndarray:
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    a = np.stack([128 + 120 * np.sin(xx / 9), 128 + 120 * np.sin(yy / 7 + 1), 128 + 120 * np.cos((xx + yy) / 13)], -1)
    a[20:30, 40:60] = 255  # something bright, for bloom
    return np.clip(a, 0, 255).astype(np.uint8)


def _random_look(rng, lut: str = "") -> dict:
    nodes = []
    for _ in range(rng.integers(1, 7)):
        t = TYPES_LIST[rng.integers(len(TYPES_LIST))]
        if t.key == "lut" and not lut:
            continue
        n = node(t.key, on=bool(rng.random() > 0.1), mix=float(rng.random()))
        for f in t.params:
            n[f.key] = lut if f.text else float(rng.uniform(f.lo, f.hi))
        nodes.append(n)
    return normalize({"name": "Random", "nodes": nodes})


@pytest.fixture
def lib(tmp_path):
    lib = Library(tmp_path)
    yield lib
    Library()  # the setting's choices back to the real folder's


def _cube(path, size=9, fn=lambda r, g, b: (r, g, b)):
    g = np.linspace(0, 1, size)
    b_, g_, r_ = np.meshgrid(g, g, g, indexing="ij")
    rows = np.stack(fn(r_, g_, b_), -1).reshape(-1, 3)
    path.write_text(f"TITLE \"test\"\nLUT_3D_SIZE {size}\n# a comment\n"
                    + "\n".join(f"{a:.6f} {b:.6f} {c:.6f}" for a, b, c in rows), encoding="utf-8")


# ---- share codes
def test_every_builtin_look_survives_its_code_and_codes_are_short():
    for key, look in BUILTIN.items():
        code = encode(look)
        assert code.startswith("SPL1-") and len(code) <= 64, (key, code)
        assert decode(code) == look, key


def test_random_looks_survive_their_codes():
    rng = np.random.default_rng(1)
    for _ in range(200):
        look = _random_look(rng, lut="My grade.cube")
        assert decode(encode(look)) == look


def test_codes_survive_chat_apps_and_say_what_is_wrong():
    code = encode(BUILTIN["film"])
    assert decode(f"  {code[:20]}\n{code[20:]} \n") == BUILTIN["film"]  # wrapped and padded
    assert decode(code.lower().replace("spl1-", "SPL1-", 1)[:5] + code[5:]) == BUILTIN["film"]
    for bad, says in (("hello", "start with SPL1-"), ("SPL2-AAAA", "newer version"),
                      (code[:-3], "incomplete or has a typo"), (code[:12] + ("A" if code[12] != "A" else "B")
                                                               + code[13:], "incomplete or has a typo")):
        with pytest.raises(CodeError, match=says):
            decode(bad)


def test_codes_from_a_newer_version():
    import base64
    import zlib

    def make(body: bytes) -> str:
        data = bytes([0]) + body
        data += (zlib.crc32(data) & 0xFFFF).to_bytes(2, "big")
        return "SPL1-" + base64.urlsafe_b64encode(data).rstrip(b"=").decode()

    # a vignette with a setting this version doesn't have (bit 6, past roundness): skipped
    assert decode(make(b"\x01X\x01" + bytes([13 << 1]) + bytes([1 << 6]) + b"\x04"))["nodes"] == \
        [normalize({"nodes": [node("vignette")]})["nodes"][0]]
    with pytest.raises(CodeError, match="doesn't have"):  # an effect it doesn't have
        decode(make(b"\x01X\x01" + bytes([99 << 1]) + b"\x00"))


def test_normalize_cleans_up():
    look = normalize({"name": "  A   look ", "nodes": [
        {"type": "color", "contrast": 7, "saturation": "x", "exposure": 0.123},
        {"type": "warp-drive"}, "junk",
        {"type": "lut", "file": "..\\..\\secret\\grade.cube"},
        {"type": "lut", "file": "notalut.png"}] + [{"type": "grain"}] * 30})
    color = look["nodes"][0]
    assert look["name"] == "A look" and color["contrast"] == 1.0 and color["saturation"] == 1.0
    assert color["exposure"] == pytest.approx(0.1) and color["on"] and color["mix"] == 1.0
    assert look["nodes"][1]["file"] == "grade.cube" and look["nodes"][2]["file"] == ""  # no folders, .cube only
    assert len(look["nodes"]) == looks.MAX_NODES
    with pytest.raises(ValueError):
        normalize("not a look")


def test_types_are_numbered_once_and_their_settings_are_sane():
    codes = [t.code for t in TYPES_LIST]
    assert len(set(codes)) == len(codes) and len(TYPES) == len(TYPES_LIST)
    for t in TYPES_LIST:
        assert t.kind in ("point", "glow", "blur", "clarity", "ca", "sharpen") and t.help
        assert len(t.fields) < 32  # the share code's mask
        keys = [f.key for f in t.fields]
        assert len(set(keys)) == len(keys) and "type" not in keys and "on" not in keys, t.key
        for f in t.fields:
            assert f.text or f.lo <= f.default <= f.hi, (t.key, f.key)


# ---- saved looks
def test_saving_loading_and_deleting_looks(lib, tmp_path):
    look_id = lib.save({"name": "Moody blues", "nodes": [node("color", temperature=-0.4)]})
    assert look_id == "my:moody-blues" and (tmp_path / "moody-blues.json").exists()
    again = lib.save({"name": "Moody blues", "nodes": []})
    assert again == "my:moody-blues-2"
    assert CHOICES[:len(BUILTIN)] == list(BUILTIN) and CHOICES[len(BUILTIN):] == ["my:moody-blues", "my:moody-blues-2"]
    assert lib.get(look_id)["nodes"][0]["temperature"] == -0.4 and looks.NAMES[look_id] == "Moody blues"
    assert lib.save({"name": "Renamed", "nodes": []}, look_id) == look_id  # the same file
    assert lib.get(look_id)["name"] == "Renamed"
    (tmp_path / "broken.json").write_text("{nope", encoding="utf-8")
    lib.reload()  # a broken file is skipped
    assert "my:broken" not in CHOICES
    assert lib.delete(look_id) and look_id not in CHOICES and not lib.delete(look_id)
    assert lib.path("my:..\\..\\x") is None and not lib.delete("cinematic")


def test_the_look_settings(lib):
    spec = BY_KEY["look"]
    assert spec.choices is CHOICES and spec.default == looks.DEFAULT_LOOK and spec.applies == ("look_on", (True,))
    assert BY_KEY["look_on"].default is False and BY_KEY["look_clips"].default is False
    assert BY_KEY["look_clips"].applies[0] == "replay" and BY_KEY["look_clips"].group == "Instant replay"
    my = lib.save({"name": "Mine", "nodes": [node("grain")]})
    st = Settings({"look": my, "look_on": True})
    assert st.look == my  # a saved look of the user's is a valid choice


def test_k_and_j_and_the_launchers_preview(lib):
    st = Settings()
    ls = LookState(st, lib)
    assert ls.current() is None and st.look == "cinematic"
    rev = ls.rev
    ls.toggle()  # K
    assert st.look_on and ls.current() is BUILTIN["cinematic"] and ls.rev > rev
    ls.next()  # J
    assert st.look == "film" and ls.name() == "Film"
    ls.next(-1)
    assert st.look == "cinematic"
    ls.toggle()
    assert ls.current() is None
    ls.next()  # J while off: turns it on, as it was
    assert st.look_on and st.look == "cinematic"
    ls.show({"name": "Editing", "nodes": [node("vignette")]})  # the Looks page, live
    assert ls.name() == "Editing"
    ls.show(None)
    assert ls.name() == "Cinematic"
    my = lib.save({"name": "Gone soon", "nodes": []})
    st.set("look", my)
    lib.delete(my)
    ls.reload()  # its file was deleted: back to the default
    assert st.look == looks.DEFAULT_LOOK
    ls.close()


# ---- LUT files
def test_cube_files(lib, tmp_path):
    src = tmp_path / "src" / "Warm.cube"
    src.parent.mkdir()
    _cube(src, fn=lambda r, g, b: (np.sqrt(r), g, b * 0.5))
    assert lib.add_lut(src) == "Warm.cube" and lib.luts() == ["Warm.cube"]
    lut = lib.lut("Warm.cube")
    assert lut.size == 9 and lut.title == "test" and lut.table.shape == (9, 9, 9, 3)
    assert lut.table[0, 0, 8].tolist() == [1.0, 0.0, 0.0] and lut.table[8, 0, 0, 2] == pytest.approx(0.5)
    bad = tmp_path / "bad.cube"
    for text, says in (("LUT_1D_SIZE 4\n0 0 0\n", "1D LUT"), ("LUT_3D_SIZE 2\n0 0 0\n", "should have 8 rows"),
                       ("hello\n", "no LUT_3D_SIZE")):
        bad.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError, match=says):
            looks.load_cube(bad)
    with pytest.raises(ValueError, match=".cube"):
        lib.add_lut(tmp_path / "x.png")
    assert lib.lut("Missing.cube") is None


def test_a_lut_node(lib, tmp_path):
    _cube(lib.lut_folder / "Identity.cube" if lib.lut_folder.mkdir() is None else None)
    img = _img()
    same = render(normalize({"nodes": [node("lut", file="Identity.cube")]}), img, library=lib)
    assert np.abs(same.astype(int) - img).max() <= 1
    _cube(lib.lut_folder / "Invert.cube", fn=lambda r, g, b: (1 - r, 1 - g, 1 - b))
    inv = render(normalize({"nodes": [node("lut", file="Invert.cube", mix=0.5)]}), img, library=lib)
    assert np.abs(inv.astype(int) - 127.5).max() <= 1.5  # half inverted: gray
    gone = normalize({"nodes": [node("lut", file="Gone.cube")]})
    assert looks.stages(gone, lambda n: lib.lut(n) is not None) == [] and (render(gone, img, library=lib) == img).all()


# ---- the reference renderer
def test_what_the_nodes_do():
    img = _img()

    def one(t, **kw):
        return render(normalize({"nodes": [node(t, **kw)]}), img).astype(int)

    for t in ("color", "wheels", "split"):  # at their defaults: nothing
        assert np.abs(one(t) - img).max() <= 1, t
    assert np.abs(one("color", mix=0.0) - img).max() == 0
    gray = one("mono")
    assert (gray.max(-1) - gray.min(-1)).max() <= 1  # black and white
    bars = one("letterbox", aspect=2.39)
    assert bars[:5].max() == 0 and bars[-5:].max() == 0 and bars[H // 2].max() > 0
    vig = one("vignette", amount=1.0)
    assert vig[0, 0].sum() < img[0, 0].sum() * 0.6 and np.abs(vig[H // 2, W // 2] - img[H // 2, W // 2]).max() <= 1
    warm = one("color", temperature=1.0)
    assert warm[..., 2].mean() > img[..., 2].mean() + 5 and warm[..., 0].mean() < img[..., 0].mean() - 5  # BGR
    bloom = one("bloom", amount=1.0, threshold=0.5)
    assert bloom[32, 50].sum() >= img[32, 50].sum() and bloom.mean() > img.mean()
    gray_img = np.full((8, 8, 3), 120, np.uint8)
    for k in range(3):  # a gray stays gray whatever the color blindness
        out = render(normalize({"nodes": [node("assist", kind=k)]}), gray_img).astype(int)
        assert np.abs(out - 120).max() <= 1
    a, b = (render(normalize({"nodes": [node("grain", amount=1)]}), img, frame=f) for f in (1, 2))
    assert (a != b).mean() > 0.5  # the grain moves


def test_every_builtin_look_changes_the_picture_and_stays_in_range():
    img = _img()
    for key, look in BUILTIN.items():
        out = render(look, img)
        assert out.shape == img.shape and out.dtype == np.uint8
        assert np.abs(out.astype(int) - img).mean() > 3, key


def test_passes():
    st = looks.stages(BUILTIN["dreamy"])  # color, split | blur | bloom
    assert [s.start["type"] if s.start else None for s in st] == [None, "blur", "bloom"]
    assert [n["type"] for n in st[0].nodes] == ["color", "split"]
    assert looks.stages(normalize({"nodes": [node("sharpen")]}))[0].start is None  # the first pass: never a start
    assert looks.stages(normalize({"nodes": [node("color", on=False), node("grain", mix=0)]})) == []


# ---- the launcher's way in
def test_the_command_line(lib, monkeypatch, capsys, tmp_path):
    monkeypatch.setenv("SMOOVPLAY_LOOKS_DIR", str(tmp_path))

    def run(*argv, stdin=""):
        monkeypatch.setattr("sys.stdin", io.StringIO(stdin))
        rc = looks.main(list(argv))
        return rc, json.loads(capsys.readouterr().out)

    rc, out = run("schema")
    assert rc == 0 and [t["key"] for t in out["types"]] == list(TYPES) and out["folder"] == str(tmp_path)
    assert out["types"][0]["params"][0]["key"] == "mix" and len(out["looks"]) == len(BUILTIN)
    rc, out = run("encode", stdin=json.dumps(BUILTIN["noir"]))
    assert rc == 0 and out["code"] == encode(BUILTIN["noir"])
    rc, out = run("decode", out["code"])
    assert rc == 0 and out["look"] == BUILTIN["noir"]
    rc, out = run("decode", "SPL1-garbage")
    assert rc == 1 and "typo" in out["error"]
    rc, out = run("save", stdin=json.dumps({"name": "From the launcher", "nodes": []}))
    assert rc == 0 and out["id"] == "my:from-the-launcher"
    rc, out = run("schema")
    assert out["looks"][-1]["id"] == "my:from-the-launcher" and out["looks"][-1]["path"].endswith(".json")
    rc, out = run("encode", out["looks"][-1]["path"])  # a saved look's file
    assert rc == 0 and decode(out["code"])["name"] == "From the launcher"
    rc, out = run("import", encode(BUILTIN["vhs"]))  # a pasted code becomes one of the user's looks
    assert rc == 0 and out["id"] == "my:retro-vhs" and out["look"] == BUILTIN["vhs"]
    rc, out = run("delete", "my:from-the-launcher")
    assert out == {"deleted": True}


def test_the_remote_link_takes_look_commands():
    from framegen.remote import Remote
    seen = []
    r = Remote(Settings(), lambda k, v: True, lambda k: None, lambda: [], commands=io.StringIO(
        'look-show {"name": "x", "nodes": []}\nlook-hide\nlooks-reload\nlook-show oops\n'), out=io.StringIO(),
        handlers={"look-show": lambda t: seen.append(("show", json.loads(t))), "look-hide": lambda t: seen.append("hide"),
                  "looks-reload": lambda t: seen.append("reload")})
    import time
    t_end = time.perf_counter() + 2
    while len(seen) < 3 and time.perf_counter() < t_end:
        r.poll()
        time.sleep(0.01)
    r.poll()  # the bad JSON is reported, not raised
    assert seen == [("show", {"name": "x", "nodes": []}), "hide", "reload"]


def test_looks_in_clips_only_on_the_smoovplay_side(tmp_path):
    from test_replay import _feed, _probe
    from framegen import replay
    import av
    r = replay.Recorder(3, out_dir=tmp_path, encoder="libx264", device="cpu", drop_when_behind=False)
    try:
        r.look = lambda f: np.zeros_like(f)  # a look that makes everything black
        _feed(r, 4)
        clip, both = r.save("both", wait=True)
        c = av.open(str(clip))
        assert next(c.decode(video=0)).to_ndarray(format="bgr24").max() < 30  # the look is in the clip
        c.close()
        c = av.open(str(both))
        img = next(f for i, f in enumerate(c.decode(video=0)) if i == 30).to_ndarray(format="bgr24")
        c.close()
        assert img[60:90, 4:30].mean() > 50 and img[60:90, -30:-4].max() < 40  # before as it was; after with it
        assert _probe(both)["video"][2] == 160
    finally:
        r.close()


# ---- the shaders, held to the reference (needs OpenGL 4.5: skipped without a graphics card)
@pytest.fixture(scope="module")
def gl():
    glfw = pytest.importorskip("glfw")
    from OpenGL import GL
    if not glfw.init():
        pytest.skip("no OpenGL")
    glfw.window_hint(glfw.VISIBLE, glfw.FALSE)
    win = glfw.create_window(64, 64, "test", None, None)
    if not win:
        glfw.terminate()
        pytest.skip("no OpenGL 4.5 context")
    glfw.make_context_current(win)
    version = (GL.glGetString(GL.GL_VERSION) or b"").decode(errors="replace")
    m = re.match(r"(\d+)\.(\d+)", version)
    if not m or (int(m[1]), int(m[2])) < (4, 5):  # e.g. macOS: 2.1, or 4.1 at most
        glfw.destroy_window(win)
        glfw.terminate()
        pytest.skip(f"no OpenGL 4.5 context (got {version})")
    GL.glPixelStorei(GL.GL_UNPACK_ALIGNMENT, 1)
    GL.glPixelStorei(GL.GL_PACK_ALIGNMENT, 1)
    yield GL
    glfw.destroy_window(win)
    glfw.terminate()


class _Gl:
    def __init__(self, GL, lut_of):
        from framegen.looks_gl import LookRenderer
        self.GL, self.r, self.rev = GL, LookRenderer(GL, lut_of), 0
        self.out, self.fbo = self._tex(None, W, H)

    def _tex(self, img, w, h):
        GL = self.GL
        t = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, t)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_LINEAR), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_LINEAR),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_BGR, GL.GL_UNSIGNED_BYTE,
                        None if img is None else np.ascontiguousarray(img))
        fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, t, 0)
        return t, fbo

    def render(self, look, img, frame=0, flip_in=False):
        GL = self.GL
        src, _ = self._tex(img[::-1].copy() if flip_in else img, W, H)
        self.rev += 1
        plan = self.r.plan(look, self.rev)
        if plan is None:
            return img
        self.r.draw(plan, src, flip_in, self.fbo, 0, 0, W, H, flip_out=False, frame=frame, purpose="test")
        GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
        out = np.frombuffer(GL.glReadPixels(0, 0, W, H, GL.GL_BGR, GL.GL_UNSIGNED_BYTE), np.uint8).reshape(H, W, 3)
        GL.glDeleteTextures([src])
        return out


def test_the_shaders_draw_what_the_reference_does(gl, lib):
    lib.lut_folder.mkdir()
    _cube(lib.lut_folder / "Test.cube", 17, lambda r, g, b: (np.sqrt(r), g ** 1.5, 0.5 * b + 0.25 * r))
    g = _Gl(gl, lib.lut)
    img = _img()
    cases = dict(BUILTIN)
    rng = np.random.default_rng(7)
    for t in TYPES_LIST:  # each node alone, at random settings
        n = node(t.key, mix=0.8)
        for f in t.params:
            n[f.key] = "Test.cube" if f.text else float(rng.uniform(f.lo, f.hi))
        cases["only " + t.key] = normalize({"nodes": [n]})
    for i in range(30):
        cases[f"random {i}"] = _random_look(rng, lut="Test.cube")
    for i, (key, look) in enumerate(cases.items()):
        d = np.abs(g.render(look, img, frame=i).astype(int) - render(look, img, i, lib))
        # Half floats between passes, the GPU's texture filtering and its maths: within a level or two almost
        # everywhere. (Near black, assist's curve can turn a 1e-4 difference into a few levels, on a pixel or two.)
        assert d.mean() < 0.6 and (d > 2).mean() < 0.002 and d.max() <= 12, (key, d.max(), (d > 2).mean())
    look = BUILTIN["film"]
    assert np.abs(g.render(look, img, 5, flip_in=True).astype(int) - render(look, img, 5, lib)).max() <= 1
    assert g.r.broken is None
    g.r.close()
