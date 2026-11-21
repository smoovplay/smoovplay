"""Auto-detect's choices (framegen.autodetect.recommend) for made-up PCs. No GPU needed."""
import re

from framegen.autodetect import Display, recommend, vfg_offer
from framegen.settings import BY_KEY, SPECS

DEFAULTS = {s.key: s.default for s in SPECS}


def timing(full: float, half: float, full_engine: str = "trt", estimated: bool = False) -> dict:
    return {"full": {"ms": full, "engine": full_engine, "estimated": estimated},
            "half": {"ms": half, "engine": "torch", "estimated": False}}


THIS_PC = timing(9.3, 6.8)  # RTX 4070 Ti SUPER, what the test measures there
UW120 = Display(3440, 1440, 120, 120)


def pick(t=THIS_PC, display=UW120, current=None, **kw) -> dict:
    r = recommend({**DEFAULTS, **(current or {})}, t, display, trt=True, **kw)
    for k, v in r["settings"].items():
        assert v in BY_KEY[k].choices
        assert r["reasons"][k]
    return r


def test_tested_card_gets_the_mode_it_was_tuned_for():
    r = pick()
    assert r["settings"] == {"hz": 90, "motion": "auto", "engine": "auto", "scaler": "fsr"}
    assert "120 fps would only fit with half motion detail" in r["reasons"]["hz"]


def test_display_caps_the_frame_rate():
    assert pick(display=Display(2560, 1440, 60, 60))["settings"]["hz"] == 60
    assert pick(display=Display(2560, 1440, 100, 100))["settings"]["hz"] == 90
    assert pick(t=timing(5.0, 3.5), display=Display(2560, 1440, 119, 119))["settings"]["hz"] == 120  # 119.88 Hz


def test_display_set_below_what_it_can_do_gets_a_note():
    r = pick(display=Display(2560, 1440, 60, 144))
    assert r["settings"]["hz"] == 60
    assert any("can run at 144 Hz" in n for n in r["notes"])


def test_fast_card_keeps_full_detail_at_120():
    r = pick(t=timing(5.0, 3.5), display=Display(2560, 1440, 144, 144))
    assert r["settings"]["hz"] == 120 and r["settings"]["motion"] == "full"
    assert any("G-Sync" in n for n in r["notes"])  # 120 into 144 Hz


def test_slower_cards_step_down():
    mid = pick(t=timing(12.0, 8.5), display=Display(2560, 1440, 144, 144))["settings"]
    assert (mid["hz"], mid["motion"]) == (90, "half")
    slow = pick(t=timing(20.0, 13.0), display=Display(2560, 1440, 144, 144))
    assert (slow["settings"]["hz"], slow["settings"]["motion"]) == (60, "half")
    too_slow = pick(t=timing(40.0, 30.0))
    assert too_slow["settings"]["hz"] == 60 and "Even 60 fps is tight" in too_slow["reasons"]["hz"]


def test_laptop_leaves_more_room():
    t = timing(8.9, 6.6)  # fits 90 fps on a desktop card, not with a laptop's margin
    assert pick(t=t)["settings"]["hz"] == 90
    r = pick(t=t, gpu="NVIDIA GeForce RTX 4080 Laptop GPU")
    assert (r["settings"]["hz"], r["settings"]["motion"]) == (90, "half")
    assert any("Laptop" in n for n in r["notes"])


def test_rtx_vsr_kept_only_where_it_fits():
    r = pick(current={"scaler": "vsr"})  # 90 fps leaves ~0.7 ms: not enough (in testing it cost a quarter of the frames)
    assert r["settings"]["scaler"] == "fsr" and "vsr_quality" not in r["settings"]
    r = pick(current={"scaler": "vsr"}, display=Display(2560, 1440, 60, 60))
    assert r["settings"]["scaler"] == "vsr" and r["settings"]["vsr_quality"] == 4
    r = pick(current={"scaler": "vsr"}, display=Display(2560, 1440, 60, 60), vsr="RTX Video SDK not set up")
    assert r["settings"]["scaler"] == "fsr" and "isn't set up" in r["reasons"]["scaler"]


def test_fsr_is_never_swapped_for_vsr():
    assert pick(t=timing(3.0, 2.0), display=Display(2560, 1440, 60, 60))["settings"]["scaler"] == "fsr"


def test_without_tensorrt_or_display():
    r = recommend(DEFAULTS, timing(10.4, 6.8, "torch"), None, trt=False)
    assert r["settings"]["hz"] == 60 and r["settings"]["engine"] == "auto"
    assert "TensorRT isn't installed" in r["reasons"]["engine"]
    assert any("assumes 60 Hz" in n for n in r["notes"])
    r = pick(t=timing(9.3, 6.8, estimated=True))
    assert any("estimated" in n for n in r["notes"])


def test_hdr_off_from_before_gets_auto():
    r = pick(current={"hdr": "off"}, hdr_display=True)
    assert r["settings"]["hdr"] == "auto" and "stays HDR all the way" in r["reasons"]["hdr"]
    r = pick(current={"hdr": "off"}, hdr_display=False)
    assert r["settings"]["hdr"] == "auto" and "fitted to it" in r["reasons"]["hdr"]
    r = pick(current={"hdr": "off"}, vfg={"low": {"ms": 3.0}, "medium": {"ms": 4.0}, "high": {"ms": 5.0}})
    assert r["settings"]["model"] == "vfg" and r["settings"]["hdr"] == "auto"  # with NVIDIA VFG picked too


def test_hdr_choices_made_on_purpose_are_kept():
    for hdr in ("auto", "sdr", "hdr10"):
        assert "hdr" not in pick(current={"hdr": hdr}, hdr_display=True)["settings"]
    # without an NVIDIA card (DirectML), SmoovPlay's HDR doesn't work yet: nothing suggested
    r = recommend({**DEFAULTS, "hdr": "off"}, timing(30.0, 20.0, "dml"), UW120, trt=False, hdr_display=True)
    assert "hdr" not in r["settings"]


def test_busy_card_note():
    assert any("busy" in n for n in pick(busy=85)["notes"])
    assert not any("busy" in n for n in pick(busy=3)["notes"])


def test_rtx_40_and_50_cards_without_vfg_are_offered_it():
    text = vfg_offer(True, "NVIDIA GeForce RTX 5070 Ti", "621.24")
    assert text.startswith("Your RTX 5070 Ti can also run NVIDIA VFG")
    assert "NVIDIA's licence" in text and "Auto-detect times it" in text
    assert "driver" not in text  # new enough
    assert not re.search(r"\d ?ms\b", text) and "faster" not in text  # NVIDIA's licence: no speed or quality claims


def test_vfg_offer_asks_for_a_newer_driver_first():
    assert "570.65 or newer, and this PC has 566.36" in vfg_offer(True, "NVIDIA GeForce RTX 4070", "566.36")
    assert "driver" not in vfg_offer(True, "NVIDIA GeForce RTX 4070", "570.65")
    assert "driver" not in vfg_offer(True, "NVIDIA GeForce RTX 4070", None)  # unknown: VFG itself will say
    assert "driver" not in vfg_offer(True, "NVIDIA GeForce RTX 4070", "r570")


def test_no_vfg_offer_when_set_up_or_the_card_cant_run_it():
    # not_set_up is False when it's set up, on RTX 20/30 cards, without an NVIDIA card, or when the check failed
    assert vfg_offer(False, "NVIDIA GeForce RTX 3080", "621.24") == ""
    assert vfg_offer(False, "NVIDIA GeForce RTX 5070 Ti", "621.24") == ""


def _detect_on(monkeypatch, vfg_why: str, driver: str = "621.24") -> dict:
    """detect() on a made-up PC (no GPU used): an RTX 4070 whose VFG state is vfg_why."""
    from types import SimpleNamespace

    import torch

    from framegen import autodetect, rtxvsr, trt, vfg
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "get_device_name", lambda i=0: "NVIDIA GeForce RTX 4070")
    monkeypatch.setattr(torch.cuda, "get_device_properties", lambda i=0: SimpleNamespace(total_memory=12 * 2**30))
    monkeypatch.setattr(autodetect, "primary_display", lambda: UW120)
    monkeypatch.setattr(autodetect, "nvidia_smi", lambda samples=3: {"driver": driver, "busy": 0})
    monkeypatch.setattr(trt, "trt_available", lambda: True)
    monkeypatch.setattr(rtxvsr, "unavailable_reason", lambda: None)
    monkeypatch.setattr(vfg, "vfg_unavailable", lambda: vfg_why)
    return autodetect.detect(DEFAULTS, log=lambda m: None, timing=THIS_PC)


def test_detect_offers_vfg_only_where_it_can_be_set_up(monkeypatch):
    from framegen.vfg import NOT_SET_UP
    r = _detect_on(monkeypatch, NOT_SET_UP)
    assert r["vfg_offer"].startswith("Your RTX 4070 can also run NVIDIA VFG") and r["vfg_offer_ready"] is True
    assert next(f for f in r["found"] if f["label"] == "NVIDIA VFG")["detail"] == "Optional: see below."
    r = _detect_on(monkeypatch, NOT_SET_UP, driver="566.36")  # too old: offered, but the button waits for the driver
    assert "566.36" in r["vfg_offer"] and r["vfg_offer_ready"] is False
    for why in ("NVIDIA VFG needs an RTX 40 or 50 series card.", "No NVIDIA graphics card was found.",
                "NVIDIA VFG couldn't be checked: boom"):
        r = _detect_on(monkeypatch, why)
        assert "vfg_offer" not in r and "vfg_offer_ready" not in r, why
