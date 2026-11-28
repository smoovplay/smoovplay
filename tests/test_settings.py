import json

import numpy as np
import pytest

from framegen.menu import Menu
from framegen.settings import BY_KEY, SPECS, Settings


def test_defaults_are_valid_choices():
    for s in SPECS:
        assert s.default in s.choices, s.key
        assert s.help, s.key
    assert len(BY_KEY) == len(SPECS)  # unique keys


def test_load_save_roundtrip_and_bad_values_ignored(tmp_path):
    path = tmp_path / "settings.json"
    path.write_text(json.dumps({"hz": 90, "scaler": "nope", "unknown": 1, "sharpening": "high"}))
    st = Settings.load(path)
    assert st.hz == 90 and st.scaler == "fsr" and st.sharpening == "high"
    st.set("scaler", "bilinear")
    assert json.loads(path.read_text())["scaler"] == "bilinear"
    assert Settings.load(path).scaler == "bilinear"


def test_missing_or_corrupt_file_gives_defaults(tmp_path):
    assert Settings.load(tmp_path / "absent.json").hz == BY_KEY["hz"].default
    bad = tmp_path / "bad.json"
    bad.write_text("{not json")
    assert Settings.load(bad).interp is True


def test_override_is_not_saved(tmp_path):
    path = tmp_path / "s.json"
    st = Settings.load(path)
    st.override("hz", 120)
    st.set("overlay", False)  # saving something else must not persist the override
    saved = json.loads(path.read_text())
    assert st.hz == 120 and saved["hz"] == BY_KEY["hz"].default and saved["overlay"] is False


def test_test_mode_is_this_session_only(tmp_path):
    path = tmp_path / "s.json"
    path.write_text(json.dumps({"simulate_30": True}))  # e.g. hand-edited: still starts off
    st = Settings.load(path)
    assert st.simulate_30 is False
    seen = []
    st.watch(seen.append)
    st.set("simulate_30", True)  # from the menu: applies (and is heard) but isn't written
    assert st.simulate_30 is True and seen == ["simulate_30"]
    st.set("overlay", False)
    saved = json.loads(path.read_text())
    assert saved["simulate_30"] is False and saved["overlay"] is False
    assert Settings.load(path).simulate_30 is False


def test_results_report_reads_plainly():
    from framegen.pipeline import Stats

    st = Stats(proc_interp_ms=[8.0] * 100, busy_ms=[9.0] * 200, real_delay_ms=[20.0] * 100,
               psnr_interp=[34.0] * 50 + [float("inf")], psnr_repeat=[26.0] * 50 + [float("inf")],
               off_interp=[0.062] * 50, off_repeat=[0.12] * 50,
               hz_used={60, 120}, quarter_shown=99, quarter_skipped=1, simulated=200)
    text = "\n".join(st.results())
    assert "Output: 60 fps and 120 fps" in text
    # Quality in plain terms (the share of each frame visibly off), not dB; the dB stay in the log
    assert "6.2% of each AI frame was visibly off" in text and "against 12% when the last frame is repeated" in text
    assert "50 frames measured" in text and " dB" not in text
    assert "PSNR" in st.summary() and "visibly off (more than 12 levels): RIFE 6.20% | repeat 12.00%" in st.summary()
    assert "Verdict: this PC keeps up at 120 fps" in text
    st.behind = 50  # a quarter of the frames waited, but the extra frames were all but all shown: still keeping up
    assert "keeps up at 120 fps" in st.results()[-1]
    st.quarter_shown, st.quarter_skipped = 60, 40
    assert "falling behind at 120 fps" in st.results()[-1]
    # 60 fps: frames that waited (a quarter here) say nothing, as with VSync each one waits for the screen's refresh
    # however fast the PC is; frames that took longer to make than a frame lasts do
    st.hz_used, st.quarter_shown, st.quarter_skipped = {60}, 0, 0
    assert st.results()[-1] == "Verdict: this PC keeps up at 60 fps"
    assert "Running behind" not in "\n".join(st.results())
    st.proc_interp_ms = [8.0] * 90 + [20.0] * 10
    assert "Slow frames: 10.0% (10 of 100) took longer to make than a 60 fps frame lasts" in "\n".join(st.results())
    assert "falling behind at 60 fps" in st.results()[-1]
    st.proc_interp_ms, st.proc_other_ms = [8.0] * 97 + [20.0] * 3, [1.0] * 100  # cheap ticks count too: 1.5%
    assert "mostly keeps up at 60 fps" in st.results()[-1]
    st.proc_interp_ms, st.proc_other_ms = [8.0] * 100, []
    st.hz_used, st.quarter_shown, st.quarter_skipped = {90}, 0, 0
    st.psnr_interp, st.off_interp, st.off_repeat = [], [], []
    assert any("not measured yet" in line for line in st.results())


def test_invalid_value_rejected():
    with pytest.raises(ValueError):
        Settings().set("hz", 75)


def test_step_wraps_and_skips_disallowed():
    st = Settings()
    assert st.step("hz", 1) == 90
    assert st.step("hz", 1) == 120
    assert st.step("hz", 1) == 60  # wraps
    no_vsr = lambda k, v: v != "vsr"  # noqa: E731
    assert st.step("scaler", 1, no_vsr) == "bilinear"
    assert st.step("scaler", 1, no_vsr) == "fsr"


def test_watchers_fire_on_change_only():
    st = Settings()
    seen = []
    stop = st.watch(seen.append)
    st.set("hz", 90)
    st.set("hz", 90)  # unchanged: no call
    st.override("interp", False)
    stop()
    st.set("hz", 60)
    assert seen == ["hz", "interp"]


def _keys(m):
    return [Menu._row_key(r) for r in m.rows()]


def _open(m, section):
    """From the main page, down to a section's row and into it."""
    while _keys(m)[m.index] != f"section:{section}":
        m.key("down")
    m.key("enter")
    assert m.page == section


def test_menu_navigation_changes_settings():
    st = Settings()
    m = Menu(st)
    m.index = _keys(m).index("hz")  # a quick pick, on the main page
    m.key("right")
    assert st.hz == 90
    m.key("left")
    assert st.hz == 60
    m.key("up")
    assert m.index == _keys(m).index("hz") - 1
    assert m.key("back") == "close"


def test_the_main_page_has_the_quick_picks_and_a_row_for_each_section():
    m = Menu(Settings())
    assert _keys(m) == ["interp", "hz", "scaler", "section:Smoothness", "section:Picture", "section:Looks",
                        "section:Display", "section:Instant replay", "section:PS5 Remote Play"]


def test_sections_open_and_go_back_where_the_player_was():
    st = Settings()
    m = Menu(st)
    _open(m, "Smoothness")
    assert _keys(m)[0] == "interp" and "even_dips" in _keys(m) and "scaler" not in _keys(m)
    while _keys(m)[m.index] != "even_dips":
        m.key("down")
    m.key("right")
    assert st.even_dips is True
    assert m.key("back") is None and m.page is None  # back to the main page...
    assert _keys(m)[m.index] == "section:Smoothness"  # ...on the section just left
    m.key("right")  # Right opens a section too, on the row it was left on
    assert m.page == "Smoothness" and _keys(m)[m.index] == "even_dips"
    m.key("back")
    assert m.key("left") is None and m.page is None  # Left on a section's row does nothing
    assert m.key("back") == "close"


def test_every_setting_is_on_a_page():
    for vals in ({}, {"model": "vfg"}, {"scaler": "vsr", "replay": 60, "overlay": True}):
        st = Settings(vals)
        m = Menu(st)
        on_pages = {s.key for p in [None] + m.sections() for kind, s in m.page_rows(p) if kind == "setting"}
        assert on_pages == {s.key for s in SPECS if s.applies_to(st.values()) and not s.hidden}, vals


def test_menu_disabled_row_and_restart_row():
    st = Settings()
    pending: list[str] = []
    m = Menu(st, disabled=lambda k: "n/a" if k == "sharpening" else None, restart_reasons=lambda: pending)
    _open(m, "Picture")
    m.index = _keys(m).index("sharpening")
    m.key("right")
    assert st.sharpening == BY_KEY["sharpening"].default  # greyed out: no change
    assert _keys(m) == ["scaler", "sharpening", "hdr", "capture_size"]  # RTX VSR quality: hidden
    pending.append("AI engine")
    assert m.rows()[-1][0] == "restart"
    m.index = len(m.rows()) - 1
    assert m.key("right") is None
    assert m.key("enter") == "restart"


def test_menu_renders_at_any_scale():
    st = Settings()
    m = Menu(st, restart_reasons=lambda: ["AI engine"])
    for ui in (0.5, 1.0, 1.5, 2.0):
        img = m.render(ui)
        assert img.dtype == np.uint8 and img.ndim == 3 and img.shape[2] == 3
        assert img.shape[1] == round(780 * ui)
    m.index = len(m.rows()) - 1  # the restart row selected
    m.render(1.0)
    m.index = 0
    _open(m, "Smoothness")
    for ui in (0.5, 1.0, 1.5, 2.0):
        assert m.render(ui).shape[1] == round(780 * ui)


def test_specs_json_is_what_the_launcher_needs():
    from framegen.settings import specs_json
    data = json.loads(json.dumps(specs_json()))  # JSON-safe
    assert data["path"].endswith("settings.json")
    keys = [s["key"] for s in data["specs"]]
    assert keys == [s.key for s in SPECS if not s.hidden] and "sharpen_fsr_only" not in keys
    for s in data["specs"]:
        assert s["group"] and s["label"] and s["help"], s["key"]
        assert s["default"] in [c["value"] for c in s["choices"]], s["key"]
    sim = next(s for s in data["specs"] if s["key"] == "simulate_30")
    assert sim["saved"] is False
    sharp = next(s for s in data["specs"] if s["key"] == "sharpening")
    assert [c["name"] for c in sharp["choices"]] == ["Off", "Minimal", "Low", "Medium", "High", "Max"]
    assert sharp["also"] == {"sharpen_fsr_only": False}  # the launcher writes it too (SettingsPage.cs ChangeSetting)
    model = next(s for s in data["specs"] if s["key"] == "model")
    assert model["live"] is False and model["later"] and model["choices"][0]["name"].startswith("RIFE")
    scaler = next(s for s in data["specs"] if s["key"] == "scaler")
    assert scaler["later"] and all(c["hint"] for c in scaler["choices"])  # a line under each choice
    assert next(c for c in sharp["choices"] if c["name"] == "High")["hint"] == ""  # not every choice needs one


def test_hints_are_for_real_choices_and_short():
    for s in SPECS:
        assert set(s.hints) <= set(s.choices), s.key
        for v, text in s.hints.items():
            assert len(text) <= 140, (s.key, v)  # two or three lines in the dropdown


def test_help_fits_the_in_video_menu():
    from framegen.menu import _wrap
    for s in SPECS:  # the menu shows up to 4 lines of help (menu.render: 780 px wide, 26 px padding, scale 0.5)
        assert len(_wrap(s.help, 780 - 2 * 26, 0.5, 1)) <= 4, s.key


def test_the_in_video_menu_stays_still_while_going_through_it():
    # Its size is the same on every row of every page (the list fits the longest page, the help area the longest
    # help), so the rows never move.
    from framegen.menu import Menu
    from framegen.settings import Settings
    for vals in ({}, {"model": "vfg"}, {"overlay": False}):
        m = Menu(Settings(vals))
        sizes = set()
        for page in [None] + m.sections():
            m.page = page
            for i in range(len(m.rows())):
                m.index = i
                sizes.add(m.render(1.0).shape)
        assert len(sizes) == 1, (vals, sizes)


def test_the_in_video_menu_fits_a_window_whatever_is_picked():
    from framegen.menu import Menu
    from framegen.settings import Settings
    for vals in ({}, {"model": "vfg"}, {"fullscreen": True}, {"scaler": "vsr"}):
        m = Menu(Settings(vals))
        for page in [None] + m.sections():
            m.page = page
            for i in range(len(m.rows())):
                m.index = i
                assert m.render(1.0).shape[0] <= 1060, (vals, page, i)  # drawn at 1.0 in a 1080 px tall picture
    # and with settings to come: a page has room for 15 rows before they're drawn closer together
    from dataclasses import replace
    extra = SPECS + tuple(replace(BY_KEY[k], group="Smoothness")  # (as stand-ins)
                          for k in ("overlay", "vsync", "fullscreen", "audio_sync", "minimize_chiaki"))
    m = Menu(Settings({}), specs=extra)
    m.page = "Smoothness"
    assert len(m.rows()) == 15 and m.render(1.0).shape[0] <= 1060


def test_the_menu_text_fits():
    from framegen.menu import KEYS, _wrap
    m = Menu(Settings({"model": "vfg", "scaler": "vsr", "replay": 60}))
    for section in m.sections():
        assert len(_wrap(m._section_help(section), 780 - 2 * 26, 0.5, 1)) <= 4, section  # 4 lines of help
    for keys in KEYS:
        assert len(_wrap(keys, 780 - 2 * 26, 0.45, 1)) == 1, keys


def test_impact_lines_are_plain_short_and_never_about_vfg():
    from framegen.menu import _wrap
    from framegen.settings import specs_json
    for s in SPECS:
        assert set(s.impact) <= {"picture", "speed"}, s.key
        for k, text in s.impact.items():
            assert text.isascii(), (s.key, k)  # the in-video menu's font draws ASCII only
            assert len(_wrap(f"{k.title()}: {text}", 780 - 2 * 26, 0.5, 1)) <= 2, (s.key, k)  # menu.render: 2 lines
    for key in ("model", "vfg_quality"):  # NVIDIA's VFG licence: no published speed or quality figures for it
        assert not BY_KEY[key].impact, key
    data = specs_json()
    assert data["impact_pc"] and next(s for s in data["specs"] if s["key"] == "steady_hud")["impact"]["speed"]
    from dataclasses import replace
    from framegen.menu import IMPACT
    shown = []
    for specs in (SPECS, tuple(replace(s, impact={}) for s in SPECS)):
        menu = Menu(Settings(), specs)
        menu.page = "Smoothness"
        menu.index = _keys(menu).index("engine")
        img = menu.render(1.0)
        assert img.shape[0] <= 1080  # fits a 1080p-tall window, rows closer together if need be
        shown.append(int((img == np.array(IMPACT, np.uint8)).all(-1).sum()))
    assert shown[0] > 200 and shown[1] == 0  # the menu shows the highlighted setting's impact lines


def test_model_choices_are_real_models():
    from framegen.interpolator import MODELS
    assert set(BY_KEY["model"].choices) - {"vfg"} <= set(MODELS)  # vfg: NVIDIA VFG, framegen/vfg.py


def _remote(tmp_path, commands, allowed=lambda k, v: True, disabled=lambda k: None, reasons=lambda: []):
    import io
    import time

    from framegen.remote import PREFIX, Remote
    st = Settings.load(tmp_path / "settings.json")
    out = io.StringIO()
    r = Remote(st, allowed, disabled, reasons, commands=io.StringIO("".join(c + "\n" for c in commands)), out=out)
    for _ in range(200):  # let the reader thread queue them all
        if r._q.qsize() == len(commands):
            break
        time.sleep(0.005)

    def states():
        return [json.loads(line[len(PREFIX):]) for line in out.getvalue().splitlines() if line.startswith(PREFIX)]
    return st, r, states


def test_remote_set_applies_saves_and_reports(tmp_path):
    st, r, states = _remote(tmp_path, ["set hz 90", "set vsr_quality 3.0", "set sharpening \"low\"",
                                       "set overlay false"])
    assert r.poll() is None
    assert (st.hz, st.vsr_quality, st.sharpening, st.overlay) == (90, 3, "low", False)
    assert isinstance(st.vsr_quality, int)  # the spec's own type, so the menu still finds it
    saved = json.loads((tmp_path / "settings.json").read_text())
    assert saved["hz"] == 90 and saved["overlay"] is False
    assert saved["sharpening"] == "low" and saved["sharpen_fsr_only"] is False  # set from the launcher: everywhere
    last = states()[-1]
    assert last["values"]["hz"] == 90 and last["restart"] == []
    assert len(states()) == 4  # one per change


def test_remote_refuses_bad_and_unavailable_values(tmp_path, capsys):
    allowed = lambda k, v: not (k == "scaler" and v == "vsr")  # noqa: E731
    disabled = lambda k: "no RTX VSR here" if k == "vsr_quality" else None  # noqa: E731
    st, r, states = _remote(tmp_path, ["set hz 75", "set nope 1", "set scaler \"vsr\"", "set vsr_quality 4",
                                       "set hz {broken", "bogus"], allowed, disabled)
    r.poll()
    assert (st.hz, st.scaler, st.vsr_quality) == (60, "fsr", 2)
    assert not (tmp_path / "settings.json").exists()
    assert len(states()) == 5  # each refused set re-sends the real values
    last = states()[-1]
    assert "vsr" not in last["choices"]["scaler"] and last["disabled"] == {"vsr_quality": "no RTX VSR here"}
    out = capsys.readouterr().out
    assert "can't set vsr_quality: no RTX VSR here" in out and "unknown command: bogus" in out


def test_remote_restart_and_state(tmp_path):
    st, r, states = _remote(tmp_path, ["set engine \"torch\"", "restart", "state"], reasons=lambda: ["AI engine"])
    assert r.poll() == "restart"
    assert st.engine == "torch"
    assert [s["restart"] for s in states()] == [["AI engine"], ["AI engine"]]


def test_the_launchers_stop_quits_as_esc_does(tmp_path):
    _, r, _ = _remote(tmp_path, ["restart", "quit", "restart"])
    assert r.poll() == "quit"  # a Stop wins over a restart that came with it


def test_rows_that_dont_apply_are_hidden_and_the_highlight_stays_put():
    from framegen.menu import Menu
    from framegen.settings import SPECS, Settings

    st = Settings()
    menu = Menu(st, SPECS)
    keys = lambda: [Menu._row_key(r) for p in [None] + menu.sections() for r in menu.page_rows(p)]  # noqa: E731
    assert "vfg_quality" not in keys() and "sharpening" in keys() and "vsr_quality" not in keys()
    _open(menu, "Picture")
    while menu.rows()[menu.index][1].key != "scaler":
        menu.key("down")
    menu.key("right")  # FSR 1 -> RTX VSR: RTX VSR quality comes, and Sharpening stays (it's for every upscaler)
    assert st.scaler == "vsr"
    assert "sharpening" in keys() and "vsr_quality" in keys()
    menu.key("enter")  # still on the upscaler row, though the rows above and below changed
    assert st.scaler == "bilinear"
    st.set("model", "vfg")
    assert "vfg_quality" in keys() and not {"engine", "motion"} & set(keys())
    assert "steady_hud" in keys()  # Steady HUD works with NVIDIA VFG too (framegen/steady.py)
    menu.render()  # and it still draws, with the highlight where it was
    assert menu.rows()[menu.index][1].key == "scaler"


def test_specs_json_says_what_applies_when():
    from framegen.settings import specs_json

    specs = {s["key"]: s for s in specs_json()["specs"]}
    assert specs["vfg_quality"]["applies"] == {"key": "model", "values": ["vfg"]}
    assert specs["fullscreen"]["applies"] is None
    assert [k for k, s in specs.items() if s["quick"]] == ["interp", "hz", "model", "scaler"]
    assert [s["group"] for s in specs.values()][:3] == ["Smoothness"] * 3


def test_capture_card_resolution_says_what_it_costs():
    from framegen.settings import CAPTURE_SIZES
    s = BY_KEY["capture_size"]
    assert s.default == "1080p" and CAPTURE_SIZES[s.default] == (1920, 1080) and not s.live and s.experimental
    assert set(CAPTURE_SIZES) == set(s.choices) and s.name("2160p") == "4K"
    assert "twice" in s.hints["1440p"] and "four times" in s.hints["2160p"] and "Recommended" in s.hints["1080p"]
    assert all("sluggish" in t for t in (s.hints["1440p"], s.hints["2160p"], s.help, s.impact["speed"]))
    assert "gains nothing" in s.help and "Capture cards only" in s.help


def test_settings_for_one_source_say_why_not_with_the_others():
    from framegen.settings import specs_json
    size, mini = BY_KEY["capture_size"], BY_KEY["minimize_chiaki"]
    assert size.unused_with("capture") is None and mini.unused_with("chiaki") is None
    assert "1080p at most" in size.unused_with("chiaki")  # Remote Play: greyed out, and why
    assert "1080p" in size.unused_with("window") and "own size" in size.unused_with("tap")
    assert size.unused_with("file") == "Only used with a capture card."
    assert mini.unused_with("tap") == mini.unused_with("capture") == "Only used with PS5 Remote Play."
    assert BY_KEY["fill"].unused_with("tap") is None  # most settings: every source
    rows = {d["key"]: d for d in specs_json()["specs"]}
    assert rows["capture_size"]["sources"] == ["capture"] and rows["capture_size"]["elsewhere"][""]
    assert rows["fill"]["sources"] is None and rows["fill"]["elsewhere"] is None
