"""chiaki-ng's PS5 codec kept in step with SmoovPlay's HDR setting (framegen/chiaki_settings.py). The registry tests use
a key of their own under HKEY_CURRENT_USER\\Software, deleted afterwards; chiaki-ng's own settings aren't touched."""
import sys
import uuid

import pytest

from framegen import chiaki_settings as cs
from framegen.chiaki_settings import HDR, wanted


def test_auto_turns_hdr_on_with_an_hdr_display():
    assert wanted("auto", None, hdr_display=True, nvidia=True) == HDR  # not set yet: chiaki-ng's default, H.265
    assert wanted("auto", "h265", hdr_display=True, nvidia=True) == HDR
    assert wanted("auto", "h264", hdr_display=True, nvidia=True) == HDR  # HDR needs H.265
    assert wanted("auto", HDR, hdr_display=True, nvidia=True) is None  # already


def test_auto_never_switches_back():
    # without an HDR display, SmoovPlay fits an HDR stream to it anyway, and chiaki-ng's settings are shared with any
    # separate install: whatever is set there stays
    for current in (None, "h264", "h265", HDR):
        assert wanted("auto", current, hdr_display=False, nvidia=True) is None


def test_on_asks_for_hdr_and_off_leaves_it_alone():
    for on in ("sdr", "hdr10"):
        assert wanted(on, "h265", hdr_display=False, nvidia=True) == HDR
        assert wanted(on, HDR, hdr_display=False, nvidia=True) is None
    for current in (None, "h264", "h265", HDR):
        assert wanted("off", current, hdr_display=True, nvidia=True) is None


def test_without_an_nvidia_card_no_hdr():
    # SmoovPlay's HDR needs CUDA: an HDR stream would only look washed out
    assert wanted("auto", "h265", hdr_display=True, nvidia=False) is None
    assert wanted("auto", HDR, hdr_display=True, nvidia=False) is None  # (left as it is, as everywhere)
    assert wanted("hdr10", "h265", hdr_display=True, nvidia=False) is None


@pytest.fixture
def key():
    if sys.platform != "win32":
        pytest.skip("Windows registry")
    import winreg
    path = rf"Software\SmoovPlayTest-{uuid.uuid4().hex[:8]}\settings"
    yield path
    for p in (path, path.rsplit("\\", 1)[0]):
        try:
            winreg.DeleteKey(winreg.HKEY_CURRENT_USER, p)
        except OSError:
            pass


def _read(path):
    import winreg
    out = {}
    with winreg.OpenKey(winreg.HKEY_CURRENT_USER, path) as k:
        for name in cs.NAMES:
            try:
                out[name] = winreg.QueryValueEx(k, name)
            except FileNotFoundError:
                out[name] = None
    return out


def test_match_codec_writes_both_codecs_as_text(key, monkeypatch, capsys):
    import winreg
    monkeypatch.delenv("FRAMEGEN_CHIAKI_CODEC", raising=False)
    monkeypatch.setattr(cs, "_hdr_display", lambda: True)
    assert cs.match_codec("auto", nvidia=True, key=key) == HDR
    assert _read(key) == {n: (HDR, winreg.REG_SZ) for n in cs.NAMES}
    out = capsys.readouterr().out
    assert "set to H.265 HDR (was H.265), as this display has Windows' HDR on" in out
    assert cs.match_codec("auto", nvidia=True, key=key) is None  # already: nothing said
    assert capsys.readouterr().out == ""
    monkeypatch.setattr(cs, "_hdr_display", lambda: False)  # Windows' HDR turned off since
    assert cs.match_codec("auto", nvidia=True, key=key) is None
    assert _read(key) == {n: (HDR, winreg.REG_SZ) for n in cs.NAMES}  # never switched back
    assert capsys.readouterr().out == ""


def test_match_codec_leaves_other_values_and_makes_no_key_for_nothing(key, monkeypatch):
    import winreg
    monkeypatch.delenv("FRAMEGEN_CHIAKI_CODEC", raising=False)
    monkeypatch.setattr(cs, "_hdr_display", lambda: False)
    assert cs.match_codec("auto", nvidia=True, key=key) is None  # nothing to write
    with pytest.raises(FileNotFoundError):  # so chiaki-ng's key isn't made before chiaki-ng ever ran
        winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
    with winreg.CreateKeyEx(winreg.HKEY_CURRENT_USER, key, 0, winreg.KEY_WRITE) as k:  # a separate install's choice
        winreg.SetValueEx(k, "codec_local_ps5", 0, winreg.REG_SZ, "h264")
        winreg.SetValueEx(k, "codec_remote_ps5", 0, winreg.REG_SZ, HDR)
    monkeypatch.setattr(cs, "_hdr_display", lambda: True)
    assert cs.match_codec("auto", nvidia=True, key=key) == HDR
    assert _read(key) == {n: (HDR, winreg.REG_SZ) for n in cs.NAMES}  # only H.264 changed (HDR needs H.265)


def test_match_codec_off_and_opt_out_touch_nothing(key, monkeypatch):
    monkeypatch.setattr(cs, "_hdr_display", lambda: True)
    assert cs.match_codec("off", nvidia=True, key=key) is None
    monkeypatch.setenv("FRAMEGEN_CHIAKI_CODEC", "0")
    assert cs.match_codec("auto", nvidia=True, key=key) is None
    import winreg
    with pytest.raises(FileNotFoundError):  # the key was never even made
        winreg.OpenKey(winreg.HKEY_CURRENT_USER, key)
