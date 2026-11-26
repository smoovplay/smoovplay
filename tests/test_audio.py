"""A capture card's sound (framegen/audio.py, framegen/capture.py): held back to meet the picture, picked by name, and
read from a recording standing in for a card. No sound is played: a stand-in output records what it's given."""
import time

import av
import numpy as np
import pytest

from framegen import capture as c
from framegen.audio import DelayLine

RATE, CH = 48000, 2
FRAME = CH * 2


def _pcm(seconds, value=0):
    return np.full(int(round(seconds * RATE)) * CH, value, np.int16).tobytes()


def _blocks_of(line, seconds, block_s=0.01):
    n = int(round(block_s * RATE)) * FRAME
    return b"".join(line.pull(n) for _ in range(int(round(seconds / block_s))))


class _Clock:
    """Stands in for time.perf_counter, so waits between packets are the simulated ones."""
    t = 0.0

    def __call__(self):
        return self.t


def _play_events(line, clock, events, pcm=None, block_s=0.01):
    """(time, "in" | "out") in order: a packet pushed, or a block pulled by the output. The output, as played."""
    n, out = int(round(block_s * RATE)) * FRAME, []
    for t, what in events:
        clock.t = t
        if what == "in":
            line.push(pcm(t) if callable(pcm) else pcm or _pcm(0.02, 100))
        else:
            out.append(line.pull(n))
    return np.frombuffer(b"".join(out), np.int16).reshape(-1, CH)


def _card(secs, hold_ms=0.0, every_s=5.0, seed=2, first_at=None):
    """A card's 20 ms packets, as the reader thread hands them over: held up now and then (hold_ms, every every_s;
    the packets meanwhile all come at once at the end), and the output's 10 ms blocks."""
    import random
    rnd = random.Random(seed)
    ev, held, nxt = [], 0, every_s if first_at is None else first_at
    for i in range(int(secs / 0.02)):
        t = i * 0.02 + abs(rnd.gauss(0, 0.002))
        if hold_ms and nxt <= t < nxt + hold_ms / 1000:
            held += 1
            continue
        if held:
            ev += [(nxt + hold_ms / 1000, "in")] * held
            held, nxt = 0, nxt + every_s
        ev.append((t, "in"))
    ev += [(i * 0.01, "out") for i in range(int(secs / 0.01))]
    return sorted(ev, key=lambda e: e[0])


def test_hold_ups_reading_the_card_leave_no_gaps():
    # Issue 17 again: "still a little bit of popping" in a heavy game. The thread that reads the card is held up now
    # and then, and the packets meanwhile come at once: with only a packet and 10 ms queued, each hold-up of 20 ms or
    # more was a gap. The cushion now grows to the longest recent wait.
    for hold in (20, 40, 60, 100):
        clock = _Clock()
        line = DelayLine(RATE, CH, clock=clock)
        line.set_delay(0.015)
        _play_events(line, clock, _card(120, hold))
        assert line.underruns <= 3, (hold, line.underruns)  # the first hold-up only, before the cushion grew
        assert line.jumps == 0, (hold, line.skips, line.refills)  # the burst after each one is played off gently
        assert line.kept_s <= hold / 1000 + 0.035 and line.kept_s <= 0.15 + 1e-3
        assert abs(line.longest_wait_s - (hold + 20) / 1000) < 0.008


def test_the_cushion_forgets_old_hold_ups_and_a_stop_isnt_one():
    clock = _Clock()
    line = DelayLine(RATE, CH, clock=clock)
    _play_events(line, clock, _card(70, 60, every_s=100, first_at=2.0))  # one hold-up, near the start
    assert line.kept_s < 0.04  # a minute later, back to a packet and the cushion
    clock = _Clock()
    line = DelayLine(RATE, CH, clock=clock)
    ev = [e for e in _card(10) if not 4.0 <= e[0] < 6.0 or e[1] == "out"]  # the sound stops for 2 s
    _play_events(line, clock, ev)
    assert line.stops == 1 and line.kept_s < 0.04


def test_a_change_in_the_pictures_delay_is_eased_not_jumped():
    clock = _Clock()
    line = DelayLine(RATE, CH, clock=clock)
    line.set_delay(0.04)
    ev = _card(30)
    _play_events(line, clock, [e for e in ev if e[0] < 5])
    line.set_delay(0.12)  # 80 ms more, e.g. a 30 fps game starting to dip
    _play_events(line, clock, [e for e in ev if 5 <= e[0] < 20])
    assert abs(line.queued_s - 0.12) < 0.02 and line.jumps == 0 and line.underruns == 0
    line.set_delay(0.04)  # and back
    _play_events(line, clock, [e for e in ev if e[0] >= 20])
    assert abs(line.queued_s - 0.04) < 0.02 and line.jumps == 0


def _sine(t0_holder):
    """A 220 Hz tone, continuous across packets (as a card's sound is), loud."""
    state = {"n": 0}

    def make(_t):
        n = int(0.02 * RATE)
        i = np.arange(state["n"], state["n"] + n)
        state["n"] += n
        v = (12000 * np.sin(2 * np.pi * 220 * i / RATE)).astype(np.int16)
        return np.repeat(v, CH).tobytes()
    return make


def test_jumps_and_gaps_are_eased_so_they_dont_click():
    natural = 12000 * 2 * np.pi * 220 / RATE  # the tone's own biggest step from one sample to the next (~350)
    clock = _Clock()
    line = DelayLine(RATE, CH, clock=clock)
    line.set_delay(0.3)
    tone = _sine(None)
    ev = _card(6)
    out = [_play_events(line, clock, [e for e in ev if e[0] < 3], tone)]
    line.set_delay(0.0)  # far less: the oldest sound is skipped
    out.append(_play_events(line, clock, [e for e in ev if 3 <= e[0] < 4], tone))
    out.append(_play_events(line, clock, [e for e in ev if 4 <= e[0] < 6 and e[1] == "out"]))  # nothing comes: dry
    out.append(_play_events(line, clock, [(6 + e[0], e[1]) for e in _card(2)], tone))  # and back
    v = np.concatenate(out)[:, 0].astype(int)
    assert line.skips >= 1 and line.underruns == 1  # one gap while nothing came, not one per block
    assert np.abs(np.diff(v)).max() < natural + 24000 / line._fade + 50  # eased: never a step like a click


def test_the_log_says_what_went_wrong():
    from framegen.audio import sound_report
    s = {"ran_dry": 3, "output_short": 1, "skips": 0, "refills": 1, "stops": 0, "longest_wait_ms": 46,
         "kept_ms": 56, "longest_call_ms": 31, "output_ms": 40}
    full = sound_report(s)
    assert "ran dry 3 times" in full and "Windows' output ran short 1 times" in full and "46 ms" in full
    assert "Windows' buffer 40 ms" in full and "stopped" not in full
    assert sound_report(s, dict(s)) is None  # nothing new: no line
    assert "ran dry 1 times" in sound_report({**s, "ran_dry": 4}, s)


def test_the_player_counts_windows_running_short():
    import types
    from framegen.audio import Player
    p = Player.__new__(Player)  # (no sound device: the callback alone)
    p.line, p.short, p.longest_call_s, p._called, p.latency = DelayLine(RATE, CH), 0, 0.0, None, 0.04
    buf = bytearray(480 * FRAME)
    ok = types.SimpleNamespace(output_underflow=False)
    p._play(buf, 480, None, ok)
    time.sleep(0.02)
    p._play(buf, 480, None, types.SimpleNamespace(output_underflow=True))  # Windows says so
    assert p.short == 1
    time.sleep(0.06)
    p._play(buf, 480, None, ok)  # it doesn't, but the call came later than the 40 ms buffer lasts
    st = p.stats()
    assert st["output_short"] == 2 and st["longest_call_ms"] >= 50 and st["output_ms"] == 40


def test_the_capture_log_reports_once_a_minute_only_when_something_happened(capsys):
    import types
    stats = {"ran_dry": 0, "output_short": 0, "skips": 0, "refills": 0, "stops": 0, "longest_wait_ms": 21,
             "kept_ms": 31}
    src = types.SimpleNamespace(player=types.SimpleNamespace(stats=lambda: dict(stats)), _sound_seen=None)
    c.CardSource._report_sound(src, final=False)
    assert capsys.readouterr().out == ""
    stats["ran_dry"] = 2
    c.CardSource._report_sound(src, final=False)
    c.CardSource._report_sound(src, final=False)
    out = capsys.readouterr().out
    assert out.count("[capture] sound, last minute: ran dry 2 times") == 1
    c.CardSource._report_sound(src, final=True)
    assert "[capture] sound this session: ran dry 2 times" in capsys.readouterr().out


def _clocks(late_a: float, late_v: float, n: int = 120):
    import types
    src = types.SimpleNamespace(_late_a=[late_a] * n, _late_v=[late_v] * n, _delay_s=None, _late_cal=None,
                                player=types.SimpleNamespace(set_delay=lambda s: setattr(src, "set", s)))
    for name in ("sound_lates", "sound_late_raw_s", "sound_late_s"):
        setattr(src, name, types.MethodType(getattr(c.CardSource, name), src))
    return src


def test_the_clock_difference_is_measured_once_at_the_start_and_not_followed():
    """Elgato HD60 X logs: the picture's timestamps drift about 40 ms a minute against the sound's, and following
    that held the sound back 964 ms, then (bounded) a constant 80 ms too late."""
    src = _clocks(0.023, 0.025)
    assert src.sound_late_s() == pytest.approx(-0.002)
    src._late_v = [0.155] * 120  # minutes later: the picture's clock has drifted
    assert src.sound_late_s() == pytest.approx(-0.002)
    c.CardSource.set_audio_delay(src, 40)
    assert src.set == pytest.approx(0.042)
    assert _clocks(0.0, 0.93).sound_late_s() == pytest.approx(-c.MAX_SOUND_LATE_S)  # a wild start is bounded
    assert _clocks(0.0, 0.5, n=20).sound_late_s() == 0.0  # not enough to go on yet


def test_the_log_says_when_the_cards_clocks_have_drifted_once_and_again_only_on_a_jump(capsys):
    src = _clocks(0.023, 0.025)
    src.sound_late_s()
    c.CardSource._report_clocks(src)
    assert capsys.readouterr().out == ""
    for late in (0.155, 0.198, 0.241):  # the steady drift: said once
        src._late_v = [late] * 120
        c.CardSource._report_clocks(src)
    out = capsys.readouterr().out
    assert out.count("[capture] sound clock:") == 1 and "-130 ms off from the start" in out and "-2 ms" in out
    src._late_v = [1.0] * 120  # a jump
    c.CardSource._report_clocks(src)
    assert capsys.readouterr().out.count("[capture] sound clock:") == 1


def test_silence_until_the_delay_then_the_sound_in_order():
    line = DelayLine(RATE, CH)
    line.set_delay(0.1)
    out = b""
    for i in range(30):  # 10 ms in, 10 ms out
        line.push(_pcm(0.01, i + 1))
        out += _blocks_of(line, 0.01)
    v = np.frombuffer(out, np.int16)[::CH]
    first = int(np.argmax(v != 0)) / RATE
    assert 0.06 <= first <= 0.11  # silence while it fills (to within a jump), then...
    sound = v[v != 0]
    assert np.all(np.diff(sound.astype(int)) >= 0) and sound[0] == 1  # ...from the start, in order
    assert abs(line.queued_s - 0.1) < 0.045


def test_the_delay_holds_when_the_card_clock_runs_fast_or_slow():
    for speed in (1.002, 0.998):  # 0.2%: far more than real clocks differ
        line = DelayLine(RATE, CH)
        line.set_delay(0.05)
        got = 0.0
        for i in range(3000):  # 30 s
            got += 0.01 * speed
            line.push(_pcm(round(got, 4) - round(got - 0.01 * speed, 4), 100))
            queued = line.queued_s  # (before the output takes its block)
            line.pull(int(0.01 * RATE) * FRAME)
        assert abs(queued - 0.05) < 0.006, (speed, queued)
        assert line.jumps == 0  # nudged back, by playing a little faster or slower


def test_a_short_delay_never_runs_dry_between_packets():
    # Issue 17: a card's sound in 20 ms packets, a little late now and then, with less delay asked for than a packet
    # lasts (the picture's delay, less the output's own): the queue ran dry every packet or so, a click each time.
    import random
    rnd = random.Random(1)
    for asked in (0.0, 0.005, 0.015):
        clock = _Clock()
        line = DelayLine(RATE, CH, clock=clock)
        line.set_delay(asked)
        events = sorted([(i * 0.02 + abs(rnd.gauss(0, 0.004)), "in") for i in range(1500)]  # 30 s
                        + [(i * 0.010001, "out") for i in range(3000)])
        _play_events(line, clock, events)
        assert line.underruns <= 1, (asked, line.underruns)  # (the first block, before anything came)
        # the longest wait between packets and the cushion: sound at most ~45 ms behind the picture
        assert 0.025 <= line.kept_s <= 0.045, line.kept_s
        assert line.jumps == 0
    line = DelayLine(RATE, CH)
    line.set_delay(0.08)
    line.push(_pcm(0.02))
    assert abs(line.kept_s - 0.08) < 1e-3  # a longer delay is kept as asked


def test_a_big_change_jumps():
    line = DelayLine(RATE, CH)
    line.set_delay(0.3)
    for _ in range(60):
        line.push(_pcm(0.01, 7))
        line.pull(int(0.01 * RATE) * FRAME)
    line.set_delay(0.03)
    line.push(_pcm(0.01, 7))
    line.pull(int(0.01 * RATE) * FRAME)
    assert line.queued_s < 0.05 and line.jumps >= 1
    n = int(0.01 * RATE) * FRAME
    assert len(line.pull(n)) == n


def test_the_cards_sound_device_is_found_by_name():
    D = c.Device
    devs = [D("USB3.0 Capture", "video"), D("Digital Audio Interface (USB3.0 Capture)", "audio"),
            D("Microphone (Realtek Audio)", "audio"), D("Cam Link 4K", "video"), D("Cam Link 4K", "audio"),
            D("Game Capture HD60 X", "video"), D("Game Capture HD60 X Audio", "audio")]
    assert c.pick_audio("USB3.0 Capture", devs) == "Digital Audio Interface (USB3.0 Capture)"
    assert c.pick_audio("Cam Link 4K", devs) == "Cam Link 4K"
    assert c.pick_audio("Game Capture HD60 X", devs) == "Game Capture HD60 X Audio"
    assert c.pick_audio("USB Video", devs) is None  # generic words only: no guess
    assert c.pick_audio("Elgato 4K X", [D("Elgato 4K X", "video"), D("Line (Elgato 4K X)", "audio"),
                                         D("Line (Elgato Wave)", "audio")]) == "Line (Elgato 4K X)"
    # AVerMedia (issue 23): "-Video"/"-Audio" endings, and a "Not To Use" device named like the real one
    avermedia = [D("Live Gamer Ultra 2.1-Video", "video"), D("HDMI/Line In (Live Gamer Ultra 2.1-Audio)", "audio"),
                 D("Virtual Microphone (AVerMedia Streaming Center)", "audio"),
                 D("Not To Use (Live Gamer Ultra 2.1-Audio)", "audio")]
    assert c.pick_audio("Live Gamer Ultra 2.1-Video", avermedia) == "HDMI/Line In (Live Gamer Ultra 2.1-Audio)"
    assert c.pick_audio("Live Gamer Ultra 2.1-Video", avermedia[:1] + avermedia[2:]) is None  # never "Not To Use"


class _Out:
    """Stands in for framegen.audio.Player."""

    def __init__(self, rate, channels):
        self.rate, self.channels, self.pushed, self.delays = rate, channels, [], []

    def push(self, pcm):
        self.pushed.append((time.perf_counter(), pcm))

    def set_delay(self, s):
        self.delays.append(s)

    def close(self):
        pass


def _clip(path, seconds=1.0, fps=60, flash=0.5):
    """A recording with a flash in the picture and a click in the sound at the same moment."""
    out = av.open(path, "w", format="matroska")
    vs = out.add_stream("mpeg4", rate=fps)
    vs.width, vs.height, vs.pix_fmt = 64, 48, "yuv420p"
    aus = out.add_stream("pcm_s16le", rate=RATE, layout="stereo")
    per = RATE // fps
    for i in range(int(seconds * fps)):
        img = np.full((48, 64, 3), 230 if i == int(flash * fps) else 20, np.uint8)
        f = av.VideoFrame.from_ndarray(img, format="bgr24")
        f.pts = i
        for p in vs.encode(f):
            out.mux(p)
        a = np.zeros((1, per * CH), np.int16)
        if i == int(flash * fps):
            a[:] = 20000
        af = av.AudioFrame.from_ndarray(a, format="s16", layout="stereo")
        af.sample_rate, af.pts = RATE, i * per
        for p in aus.encode(af):
            out.mux(p)
    for st in (vs, aus):
        for p in st.encode(None):
            out.mux(p)
    out.close()


def test_a_recordings_sound_comes_out_with_its_picture(tmp_path):
    path = str(tmp_path / "clip.mkv")
    _clip(path)
    outs = []
    src = c.CardSource.video_file(path, device="cpu", colour=(False, False),
                                  audio=lambda r, ch: outs.append(_Out(r, ch)) or outs[-1])
    assert src.can_delay_audio() and not src.can_press()
    src.set_audio_delay(40)
    from framegen.io import Waiting
    ticks = [t for t in src if not isinstance(t, Waiting)]  # (a Waiting while the clip opens, when run on its own)
    src.close()
    out = outs[0]
    assert (out.rate, out.channels) == (RATE, CH) and out.delays == [pytest.approx(0.04)]
    flash = next(t for t in ticks if t.frame.float().mean() > 120)
    click = next(t for t, pcm in out.pushed if np.abs(np.frombuffer(pcm, np.int16)).max() > 10000)
    assert abs(click - flash.t_arrival) < 0.04  # out with the picture, not when the file was read ahead
    total = sum(len(pcm) for _, pcm in out.pushed) / FRAME / RATE
    assert 0.9 < total < 1.1


def test_no_sound_wanted_or_none_there(tmp_path):
    path = str(tmp_path / "clip.mkv")
    _clip(path, seconds=0.2)
    src = c.CardSource.video_file(path, device="cpu", colour=(False, False))  # audio=None: the picture only
    assert not src.can_delay_audio() and src.astream is None
    assert len(list(src)) == 12
    src.close()


def test_a_card_with_no_sound_device_of_its_own_tries_its_video_device_for_sound(monkeypatch):
    """The AVerMedia GC573 (issue 9): its sound is a pin on the video device, so Windows lists no sound device."""
    import types
    import run
    from framegen import capture as c
    name = "AVerMedia HD Capture GC573 1"
    monkeypatch.setattr(c, "list_devices", lambda: [c.Device(name, "video"), c.Device("OBS Virtual Camera", "video"),
                                                    c.Device("Microphone (USB Audio)", "audio")])
    monkeypatch.setattr(c.CardSource, "open", classmethod(lambda cls, *args, **kw: kw))
    monkeypatch.setattr(run, "sound_player", lambda: object())
    a = types.SimpleNamespace(capture_backend="pyav", capture_colour="auto", capture_audio="auto", capture=name,
                              capture_format="auto", capture_fps=60.0)
    assert run.open_capture(a, 1920, 1080, None)["audio_device"] == name


def test_no_sound_on_the_video_device_plays_the_picture_alone(monkeypatch, capsys):
    import av
    from framegen import capture as c
    name = "AVerMedia HD Capture GC573 1"
    tried = []

    def fake_open(url, **kw):
        tried.append(url)
        raise av.error.OSError(5, "Could not find audio only device with name")

    monkeypatch.setattr(av, "open", fake_open)
    monkeypatch.setattr(c, "_open_card", lambda device, opts: "picture only")
    monkeypatch.setattr(c.CardSource, "__init__", lambda self, container, *a, **kw: setattr(self, "container", container))
    src = c.CardSource.open(name, mode=c.Mode("nv12", True, 1920, 1080, 60.0, 60.0), audio_device=name, audio=object())
    assert tried == [f"video={name}:audio={name}"] and src.container == "picture only"
    assert "no sound from the card itself" in capsys.readouterr().out
