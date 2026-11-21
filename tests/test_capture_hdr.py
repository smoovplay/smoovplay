"""HDR through a capture card (framegen/capture.py HdrCardSource, open_hdr; framegen/elgato.py; framegen/mf_capture.py):
P010 frames read as HDR10 or SDR as the card says, on the CPU. The card itself is stood in for: its Media Foundation
reader by FakeReader, its HID by FakeCard; the register bytes are as a real Elgato HD60 X Rev. 2 sent them."""
import time
import types

import numpy as np
import pytest
import torch

from framegen import capture as c
from framegen import elgato, hdr, mf_capture


def p010(y, u, v, stride=None):
    """A P010 frame's bytes, as Media Foundation hands them over, from 10-bit code planes (chroma at half size)."""
    h, w = y.shape
    stride = stride or w * 2
    buf = np.zeros((h * 3 // 2, stride // 2), np.uint16)
    buf[:h, :w] = np.asarray(y, np.uint16) << 6
    buf[h:, :w] = (np.stack([u, v], -1).reshape(h // 2, w).astype(np.uint16)) << 6
    return buf.view(np.uint8).ravel()


class FakeReader:
    """Stands in for mf_capture.P010Reader: hands over the frames given, then stops."""

    def __init__(self, frames, w, h, stride=None, fps=60.0):
        self.w, self.h, self.fps, self.stride = w, h, fps, stride or w * 2
        self.frames, self.dropped, self.error, self.closed = list(frames), 0, "no more test frames", False

    def get(self, timeout=0.5):
        if not self.frames:
            return None
        return self.frames.pop(0), time.perf_counter(), 0.0

    def close(self):
        self.closed = True


def planes(y, u, v, h=4, w=8):
    return np.full((h, w), y), np.full((h // 2, w // 2), u), np.full((h // 2, w // 2), v)


def frames_of(src):
    from framegen.io import Waiting
    return [t.frame for t in src if not isinstance(t, Waiting)]  # (a Waiting while a table is made on the CPU)


def run_source(frames, hdr_of, out="sdr8", stride=None, h=4, w=8):
    src = c.HdrCardSource(FakeReader(frames, w, h, stride), "Capture card (test)", hdr_of, device="cpu")
    src.set_hdr(out)
    got = frames_of(src)
    src.close()
    return src, got


GREY = (502, 512, 512)  # half way up in 10-bit limited range
RED = (64 + round(876 * 0.2126), 512 - round(896 * 0.2126 / (2 * (1 - 0.0722))), 512 + 448)  # BT.709 full red


def expect(codes, info, out, h=4, w=8):
    y, u, v = (torch.from_numpy(p.astype(np.float32)) for p in planes(*codes, h, w))
    return hdr.convert_yuv(y, u, v, info, out)


@pytest.mark.parametrize("stride", [None, 8 * 2 + 64])  # a card may pad its rows
def test_frames_are_read_as_the_card_says(stride, capsys):
    """The same codes read as SDR (the console in SDR) and as HDR10 (in HDR): each matches hdr.convert_yuv's maths,
    and they differ the way the user sees it: SDR taken for HDR10 is the wrong picture."""
    says = iter([False, True, False, True])
    src, got = run_source([p010(*planes(*GREY), stride)] * 4, lambda: next(says), stride=stride)
    sdr, as_hdr = expect(GREY, hdr.P010_SDR, "sdr8"), expect(GREY, hdr.P010_HDR10, "sdr8")
    assert torch.equal(got[0], sdr) and torch.equal(got[2], sdr)  # SDR to SDR: exactly, no table
    assert (got[1].int() - as_hdr.int()).abs().max() <= 2 and torch.equal(got[1], got[3])
    assert abs(int(got[0][0, 0, 0]) - 128) <= 1  # SDR half grey stays half grey
    assert abs(int(got[1][0, 0, 0]) - int(got[0][0, 0, 0])) > 20
    assert len(src.convert._luts) == 1  # HDR10's table, made once: switching back to it doesn't make it again
    assert "shown in SDR" in capsys.readouterr().out


def test_sdr_red_read_as_sdr_stays_red():
    _, got = run_source([p010(*planes(*RED))], lambda: False)
    b, g, r = got[0][0, 0].tolist()
    assert r >= 250 and g <= 3 and b <= 3


def test_the_hd60_xs_sdr_is_limited_range():
    """A PS5 menu through the card in SDR (HD60 X Rev. 2): black at 64, white text overshooting past 940 at its
    edges. Limited range, as the card tells Windows: read as full range it came out washed out next to the PS5's own
    Remote Play picture of the same screen."""
    _, got = run_source([p010(*planes(1023, 512, 512)), p010(*planes(64, 512, 512))], lambda: False)
    assert got[0][0, 0].tolist() == [255, 255, 255] and got[1][0, 0].tolist() == [0, 0, 0]


def test_hdr10_to_hdr_output_is_passed_through():
    codes = (600, 480, 560)
    _, got = run_source([p010(*planes(*codes))], lambda: True, out="pq16")
    y, u, v = (torch.from_numpy(p.astype(np.float32)) for p in planes(*codes))
    want = hdr.yuv_to_rgb(y, u, v, hdr.P010_HDR10).flip(-1).to(torch.float16)
    assert got[0].dtype == torch.float16 and torch.equal(got[0], want)


def test_the_consoles_sdr_in_hdr_output_looks_as_windows_shows_sdr():
    """The console's SDR in HDR output, when run.py knows Windows' SDR content brightness (240 nits on the PC it was
    tried on): white at that, and the sRGB curve Windows shows SDR apps with, so it looks as it did when the card was
    read in 8 bits (an SDR window). BT.1886's gamma 2.4 at 203 nits made a PS5 menu's dark background 7x darker.
    HDR10 isn't touched."""
    white, dark = (940, 512, 512), (64 + round(876 * 0.04), 512, 512)  # a PS5 menu's background, 4% up

    def nits(frame):
        return float(hdr.pq_eotf(frame[0, 0, 0].double()) * 10000)

    for sdr_nits in (None, 240.0):
        src = c.HdrCardSource(FakeReader([p010(*planes(*code)) for code in (white, dark, GREY)], 8, 4),
                              "Capture card (test)", iter([False, False, True]).__next__, device="cpu")
        src.set_hdr("pq16", sdr_nits=sdr_nits)
        w, d, hdr10 = frames_of(src)
        src.close()
        if sdr_nits is None:  # (as video: BT.1886 at HDR's reference white)
            assert nits(w) == pytest.approx(hdr.SDR_WHITE_NITS, rel=0.01)
            assert nits(d) == pytest.approx(0.04 ** 2.4 * hdr.SDR_WHITE_NITS, rel=0.1)
        else:
            assert nits(w) == pytest.approx(240.0, rel=0.01)
            assert nits(d) == pytest.approx(0.04 / 12.92 * 240.0, rel=0.1)  # 0.74 nits, not 0.09
        assert torch.equal(hdr10, expect(GREY, hdr.P010_HDR10, "pq16"))


def test_a_card_that_cant_say_is_read_as_hdr10():
    _, got = run_source([p010(*planes(*GREY))], None)
    assert (got[0].int() - expect(GREY, hdr.P010_HDR10, "sdr8").int()).abs().max() <= 2


# ---- the Elgato HD60 X's input register (framegen/elgato.py)
HDR_BYTES = bytes.fromhex("00000f70083c560001") + bytes(23)  # 3840x2160 60 Hz, HDR: as a real HD60 X sent it
SDR_BYTES = bytes.fromhex("00000f70083c560000") + bytes(23)


def test_input_register():
    assert elgato.parse_input_status(HDR_BYTES) == elgato.InputStatus(3840, 2160, 60, True)
    assert elgato.parse_input_status(SDR_BYTES) == elgato.InputStatus(3840, 2160, 60, False)
    assert elgato.parse_input_status(bytes(32)) == elgato.NO_SIGNAL and not elgato.NO_SIGNAL.signal
    assert elgato.parse_input_status(bytes(4)) is None  # too short
    assert elgato.parse_input_status(bytes.fromhex("0000100000003c0000") + bytes(23)) is None  # 4096 x 0: not this
    assert elgato.parse_input_status(HDR_BYTES[:8] + b"\x07" + bytes(23)) is None  # not a yes/no


@pytest.fixture(autouse=True)
def restore_note(tmp_path, monkeypatch):
    """elgato's note of tone mapping to put back, somewhere of the test's own."""
    note = tmp_path / "elgato-tone-mapping.txt"
    monkeypatch.setattr(elgato, "RESTORE_NOTE", note)
    return note


class FakeCard:
    """Stands in for elgato.Hd60x: the input statuses given, one per read (the last one stays), and the tone mapping
    register; `refuse` writes don't take (as one didn't on a real card as a game closed)."""

    def __init__(self, statuses, tone=False, refuse=0):
        self.statuses, self.tone, self.closed, self.writes, self.refuse = list(statuses), tone, False, [], refuse

    def input_status(self):
        return self.statuses.pop(0) if len(self.statuses) > 1 else self.statuses[0]

    def tone_mapping(self):
        return self.tone

    def set_tone_mapping(self, on):
        self.writes.append(on)
        if self.refuse:
            self.refuse -= 1
            return False
        self.tone = on
        return True

    def close(self):
        self.closed = True


HDR_ON = elgato.InputStatus(3840, 2160, 60, True)
SDR_ON = elgato.InputStatus(3840, 2160, 60, False)


def wait_until(cond, s=2.0):
    t_end = time.time() + s
    while not cond() and time.time() < t_end:
        time.sleep(0.01)
    time.sleep(0.05)


def test_hdr_watch_follows_the_console_through_a_switch():
    """HDR on, then a moment without a signal (the switch), a read that fails, then SDR: the moment and the failed
    read change nothing, SDR does, and it's said."""
    card, logs = FakeCard([elgato.NO_SIGNAL, None, SDR_ON]), []
    watch = elgato.HdrWatch(card, HDR_ON, poll_s=0.005, log=logs.append)
    assert watch() is True
    wait_until(lambda: not watch())
    watch.close()
    assert watch() is False and card.closed and card.writes == []
    assert logs == ["[capture] the console switched to SDR", "[capture] the picture is now read as SDR"]
    assert watch.describe() == "the console sends 3840x2160 at 60 Hz in SDR"


def test_with_the_cards_tone_mapping_on_hdr_comes_as_sdr():
    """The HD60 X's own tone mapping on (its default): HDR coming in leaves the card as SDR, so frames are read as
    SDR. For HDR output, keep_hdr turns it off (frames HDR10 from then on), and close puts it back as it was."""
    card, logs = FakeCard([HDR_ON], tone=True), []
    watch = elgato.HdrWatch(card, HDR_ON, tone_mapping=True, poll_s=0.005, log=logs.append)
    assert watch() is False and "tone mapping is on" in watch.describe()
    assert watch.keep_hdr() and watch() is True and card.writes == [False]
    wait_until(lambda: False, 0.05)
    assert watch() is True  # (the card reads back off: stays HDR10)
    watch.close()
    assert card.writes == [False, True] and card.tone is True
    assert any("tone mapping is off while SmoovPlay plays" in line for line in logs)


def test_tone_mapping_is_put_back_even_when_the_card_doesnt_take_it_at_once(restore_note):
    card, logs = FakeCard([HDR_ON], tone=True), []
    watch = elgato.HdrWatch(card, HDR_ON, tone_mapping=True, poll_s=0.005, log=logs.append)
    assert watch.keep_hdr() and restore_note.read_text() == "1"  # noted before it's changed
    card.refuse = 2
    watch._put_back(wait_s=0.01)
    watch.close()
    assert card.tone is True and card.writes == [False, True, True, True] and not restore_note.exists()
    assert "[capture] the card's own HDR tone mapping is back on" in logs


def test_a_smoovplay_that_couldnt_put_it_back_is_made_good_next_time(restore_note, monkeypatch):
    card, logs = FakeCard([HDR_ON], tone=True, refuse=99), []
    watch = elgato.HdrWatch(card, HDR_ON, tone_mapping=True, poll_s=0.005, log=logs.append)
    card.refuse = 0
    assert watch.keep_hdr()
    card.refuse = 99
    assert not watch._put_back(tries=2, wait_s=0.01)
    watch.close()
    assert card.tone is False and restore_note.read_text() == "1"
    assert any("couldn't turn the card's own HDR tone mapping back on" in line for line in logs)
    # the next SmoovPlay to open the card puts it back first
    card.refuse, card.statuses = 0, [HDR_ON]
    monkeypatch.setattr(elgato.Hd60x, "open", classmethod(lambda cls: card))
    later = elgato.HdrWatch.open("Elgato HD60 X", log=logs.append)
    later.close()
    assert card.tone is True and not restore_note.exists() and later.tone_mapping is True
    assert any("as an earlier SmoovPlay left it off" in line for line in logs)


def test_tone_mapping_turned_on_by_elgatos_app_mid_game():
    card, logs = FakeCard([HDR_ON]), []
    watch = elgato.HdrWatch(card, HDR_ON, poll_s=0.005, log=logs.append)
    assert watch() is True and watch.keep_hdr() and card.writes == []  # already off: nothing written
    card.tone = True
    wait_until(lambda: not watch())
    watch.close()
    assert watch() is False and card.writes == []  # not ours to put back
    assert "[capture] the card's own HDR tone mapping was turned on" in logs


def test_only_an_hd60_x_is_asked(monkeypatch):
    monkeypatch.setattr(elgato.Hd60x, "open", classmethod(lambda cls: pytest.fail("asked the HID")))
    assert elgato.HdrWatch.open("USB3.0 Capture") is None
    monkeypatch.setattr(elgato.Hd60x, "open", classmethod(lambda cls: None))
    assert elgato.HdrWatch.open("Elgato HD60 X") is None  # no (or more than one) HD60 X HID

    class Odd(FakeCard):  # another firmware's layout: can't tell
        pass
    card = Odd([None])
    monkeypatch.setattr(elgato.Hd60x, "open", classmethod(lambda cls: card))
    assert elgato.HdrWatch.open("Elgato HD60 X") is None and card.closed


# ---- choosing the 10-bit path (capture.open_hdr, run.open_capture)
def test_p010_mode_choice():
    modes = [mf_capture.MfMode("nv12", 1920, 1080, 120.0), mf_capture.MfMode("p010", 1920, 1080, 60.0),
             mf_capture.MfMode("p010", 1920, 1080, 30.0), mf_capture.MfMode("p010", 1280, 720, 60.0)]
    assert mf_capture.pick_p010(modes, 1920, 1080, 60.0) == modes[1]
    assert mf_capture.pick_p010(modes, 1920, 1080, 90.0) == modes[1]  # the fastest there is
    assert mf_capture.pick_p010(modes, 1280, 720, 30.0) == modes[3]
    assert mf_capture.pick_p010(modes, 2560, 1440, 60.0) is None


def test_auto_reads_10_bits_only_from_a_card_that_says(monkeypatch):
    monkeypatch.setattr(elgato.HdrWatch, "open", classmethod(lambda cls, name, log=print: None))
    monkeypatch.setattr(mf_capture.P010Reader, "open", classmethod(lambda cls, *a, **k: pytest.fail("opened P010")))
    assert c.open_hdr("USB3.0 Capture", 1920, 1080, 60.0, "auto") == (None, "")


class FakeWatch:
    """Stands in for elgato.HdrWatch: the console in SDR."""

    def __init__(self):
        self.closed, self.kept = False, 0

    def __call__(self):
        return False

    def describe(self):
        return "the console sends 3840x2160 at 60 Hz in SDR"

    def keep_hdr(self):
        self.kept += 1
        return True

    def close(self):
        self.closed = True


@pytest.mark.parametrize("out, kept", [("pq16", 1), ("sdr8", 0), (None, 0)])
def test_hdr_output_turns_the_cards_tone_mapping_off(out, kept):
    """Only for HDR output: shown in SDR, the card's own HDR-to-SDR is left as the player set it."""
    watch = FakeWatch()
    src = c.HdrCardSource(FakeReader([], 8, 4), "Capture card (test)", watch, device="cpu")
    src.set_hdr(out)
    src.close()
    assert watch.kept == kept and watch.closed


@pytest.mark.parametrize("setting, says", [("hdr10", False), ("auto", True)])
def test_open_hdr(monkeypatch, setting, says):
    """With an HD60 X (Auto or On) frames are read as it says (here SDR); with a card that can't say (On only), HDR10."""
    watch = FakeWatch() if says else None
    monkeypatch.setattr(elgato.HdrWatch, "open", classmethod(lambda cls, name, log=print: watch))
    reader = FakeReader([p010(*planes(*GREY))], 8, 4)
    monkeypatch.setattr(mf_capture.P010Reader, "open", classmethod(lambda cls, *a, **k: (reader, "")))
    src, why = c.open_hdr("Elgato HD60 X", 8, 4, 60.0, setting, device="cpu")
    assert why == "" and isinstance(src, c.HdrCardSource) and src.hdr_stream
    src.set_hdr("sdr8")
    frame, = frames_of(src)
    src.close()
    info = hdr.P010_SDR if says else hdr.P010_HDR10
    assert (frame.int() - expect(GREY, info, "sdr8").int()).abs().max() <= 2
    assert reader.closed and (watch is None or watch.closed)


def test_no_p010_at_that_size_lets_the_card_go(monkeypatch):
    watch = FakeWatch()
    monkeypatch.setattr(elgato.HdrWatch, "open", classmethod(lambda cls, name, log=print: watch))
    monkeypatch.setattr(mf_capture.P010Reader, "open", classmethod(
        lambda cls, *a, **k: (None, "Elgato HD60 X has no 10-bit (P010) mode at 2560x1440 (it has 1920x1080)")))
    src, why = c.open_hdr("Elgato HD60 X", 2560, 1440, 60.0, "auto")
    assert src is None and "2560x1440" in why and watch.closed


def _args(name):
    return types.SimpleNamespace(capture_backend="pyav", capture_colour="auto", capture_audio="off", capture=name,
                                 capture_format="auto", capture_fps=60.0)


def test_run_asks_for_10_bits_only_with_hdr_on(monkeypatch, capsys):
    import run
    name = "Elgato HD60 X"
    monkeypatch.setattr(c, "list_devices", lambda: [c.Device(name, "video")])
    monkeypatch.setattr(c.CardSource, "open", classmethod(lambda cls, *a, **kw: "8-bit"))
    asked = []

    def fake_open_hdr(device, w, h, fps, setting, **kw):
        asked.append(setting)
        return (None, "no P010 here") if setting == "sdr" else ("10-bit", "")

    monkeypatch.setattr(c, "open_hdr", fake_open_hdr)
    assert run.open_capture(_args(name), 1920, 1080, None) == "8-bit" and asked == []  # HDR off (the default here)
    assert run.open_capture(_args(name), 1920, 1080, None, hdr="auto") == "10-bit"
    assert run.open_capture(_args(name), 1920, 1080, None, hdr="sdr") == "8-bit"
    assert "no 10-bit HDR capture: no P010 here; using 8 bits" in capsys.readouterr().out
    forced = _args(name)
    forced.capture_format = "nv12"
    monkeypatch.setattr(c, "list_modes", lambda n: [c.Mode("nv12", True, 1920, 1080, 60.0, 60.0)])
    assert run.open_capture(forced, 1920, 1080, None, hdr="auto") == "8-bit" and asked == ["auto", "sdr"]


def test_the_note_on_a_cards_hdr_modes_says_what_to_do():
    _, note = c.pick_mode([c.Mode("nv12", True, 1920, 1080, 60.0, 60.0), c.Mode("p010", False, 1920, 1080, 60, 60)])
    assert "10-bit HDR" in note and "turn HDR off on the console" in note and "HD60 X" in note
