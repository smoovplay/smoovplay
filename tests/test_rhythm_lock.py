"""Game frame rate 30 fps (cadence.RhythmLock): every other tick is new, whatever the checks make of a still, dark or
busy scene; the checks only pick which tick of the two, and a hitch moves it. Plus the frame log run.py keeps for
each Remote Play session from the launcher."""
import random

from framegen.cadence import RhythmLock


def _sizes(n, phase=0, new=650, repeat=200, jitter=0.25, seed=1):
    """A 30 fps game's compressed frame sizes at 60 Hz, as tiny as in a still scene (0.6 KB new, 0.2 KB repeats)."""
    rnd = random.Random(seed)
    return [int((new if i % 2 == phase else repeat) * rnd.uniform(1 - jitter, 1 + jitter)) for i in range(n)]


def test_tiny_frames_that_fool_the_size_line_still_alternate():
    lock = RhythmLock()
    rnd = random.Random(2)
    sizes = _sizes(600)
    # the size check's own calls, with 40% of repeats misread as new (a game at 30 read as about 42 fps)
    calls = [i % 2 == 0 or rnd.random() < 0.4 for i in range(600)]
    out = [lock.update(c, s) for c, s in zip(calls, sizes)]
    assert out[40:] == [i % 2 == 0 for i in range(40, 600)]
    assert lock.moves == 0
    assert sum(out[40:]) == 280  # exactly 30 fps


def test_a_hitch_moves_the_rhythm_by_a_tick():
    lock = RhythmLock()
    sizes = _sizes(200, phase=0) + _sizes(300, phase=1, seed=3)[200:]  # a frame held for 3 ticks at 200
    out = [lock.update(None, s) for s in sizes]
    assert out[150:200] == [i % 2 == 0 for i in range(150, 200)]
    assert out[260:] == [i % 2 == 1 for i in range(260, 300)]
    assert lock.moves == 1


def test_without_sizes_the_checks_calls_vote():
    lock = RhythmLock()
    rnd = random.Random(4)
    calls = [(i % 2 == 1) if rnd.random() > 0.2 else (i % 2 == 0) for i in range(300)]  # 20% wrong either way
    out = [lock.update(c) for c in calls]
    assert out[40:] == [i % 2 == 1 for i in range(40, 300)]


def test_keyframes_say_nothing():
    lock = RhythmLock()
    sizes = _sizes(100)
    out = [lock.update(None, 50_000 if i % 20 == 7 else s, keyframe=i % 20 == 7) for i, s in enumerate(sizes)]
    assert out[30:] == [i % 2 == 0 for i in range(30, 100)]


def test_the_first_ticks_go_by_the_checks():
    lock = RhythmLock()
    assert [lock.update(c) for c in (True, True, False)] == [True, True, False]
    assert lock.phase is None


def test_the_frame_log_sits_next_to_the_session_and_keeps_ten(tmp_path):
    import run
    for k in range(12):
        (tmp_path / f"frames-20261117-1400{k:02d}.csv").write_text("x")
    path = run.frame_log(str(tmp_path / "results-20261117-150000.txt"))
    assert path == str(tmp_path / "frames-20261117-150000.csv")
    left = sorted(p.name for p in tmp_path.glob("frames-*.csv"))
    assert len(left) == 9 and left[0] == "frames-20261117-140003.csv"  # 9 old + this one = 10
    assert run.frame_log(None) is None


def test_lost_frames_dont_move_the_rhythm_when_counted_by_the_streams_numbers():
    lock = RhythmLock()
    sizes = _sizes(400)
    seqs = list(range(400))
    keep = [i for i in range(400) if i not in (101, 151, 152, 153, 251)]  # frames lost on the way
    out = [(i, lock.update(None, sizes[i], seq=seqs[i])) for i in keep]
    assert all(new == (i % 2 == 0) for i, new in out if i >= 40)
    assert lock.moves == 0


def test_a_capture_cards_frames_dropped_from_the_queue_dont_move_the_rhythm():
    """A capture card's frames are numbered as they come off it (framegen/capture.py), so the ones SmoovPlay's queue
    drops when the PC falls behind leave gaps in the count. Counted as they arrived, each drop shifted the beat by a
    tick, and moving to the new beat a moment later showed as a hitch (issue 20, against Simulate a 30 fps game)."""
    rnd = random.Random(7)
    calls = [(i % 2 == 0) if rnd.random() > 0.15 else (i % 2 == 1) for i in range(600)]  # 15% misread
    keep = [i for i in range(600) if i not in (100, 230, 231, 377, 501)]
    lock = RhythmLock()
    out = [(i, lock.update(calls[i], seq=i)) for i in keep]
    assert all(new == (i % 2 == 0) for i, new in out if i >= 40)
    assert lock.moves == 0
    as_they_came = RhythmLock()
    for i in keep:
        as_they_came.update(calls[i])
    assert as_they_came.moves >= 2  # what it did before


def test_run_py_accepts_game_fps_flag():
    import subprocess
    import sys
    out = subprocess.check_output([sys.executable, "run.py", "--help"], text=True)
    assert "--game-fps {auto,30}" in out
