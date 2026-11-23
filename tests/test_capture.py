"""Capture cards (framegen.capture): reading DirectShow's listings, picking a format, converting raw frames, and the
whole source with a raw video file standing in for a card. CPU only: no card, no GPU."""

import cv2
import numpy as np
import pytest
import torch

from framegen import capture as c

# FFmpeg 5+ (each device tagged with its type), as PyAV's log gives it
DEVICES_NEW = '''[dshow @ 0000020a] "USB3.0 Capture" (video)
[dshow @ 0000020a]   Alternative name "@device_pnp_\\\\?\\usb#vid_345f&pid_2130&mi_00#7&1b2&0&0000#{65e8773d}\\global"
[dshow @ 0000020a] "Game Capture HD60 X" (video)
[dshow @ 0000020a]   Alternative name "@device_pnp_\\\\?\\usb#vid_0fd9&pid_0082"
[dshow @ 0000020a] "Virtual Camera" (none)
[dshow @ 0000020a]   Alternative name "@device_sw_{860BB310}"
[dshow @ 0000020a] "Digital Audio Interface (USB3.0 Capture)" (audio)
[dshow @ 0000020a]   Alternative name "@device_cm_{33D9A762}\\wave_{1}"
'''
# FFmpeg 4 (headings instead of tags)
DEVICES_OLD = '''[dshow @ 00000001] DirectShow video devices (some may be both video and audio devices)
[dshow @ 00000001]  "USB Video"
[dshow @ 00000001]     Alternative name "@device_pnp_x"
[dshow @ 00000001] DirectShow audio devices
[dshow @ 00000001]  "Line In (USB Audio)"
[dshow @ 00000001]     Alternative name "@device_cm_y"
'''
# -list_options for an MS2130-style stick: YUY2 and MJPEG at 1080p, NV12 only at 4K30, colour info on some lines
OPTIONS = '''DirectShow video device options (from video devices)
 Pin "Capture" (alternative pin name "0")
  pixel_format=yuyv422  min s=1920x1080 fps=60 max s=1920x1080 fps=60 (tv, bt709/bt709/bt709, topleft)
  pixel_format=yuyv422  min s=1920x1080 fps=30 max s=1920x1080 fps=30
  pixel_format=yuyv422  min s=1280x720 fps=60 max s=1280x720 fps=60
  vcodec=mjpeg  min s=1920x1080 fps=60 max s=1920x1080 fps=60
  vcodec=mjpeg  min s=3840x2160 fps=30 max s=3840x2160 fps=30
  pixel_format=nv12  min s=3840x2160 fps=30 max s=3840x2160 fps=30 (pc, unknown/unknown/unknown, left)
  unknown compression type 0x3231564E  min s=640x480 fps=30 max s=640x480 fps=30
'''


def test_devices_from_either_ffmpeg_format():
    new = c.parse_devices(DEVICES_NEW)
    assert [(d.name, d.kind) for d in new] == [("USB3.0 Capture", "video"), ("Game Capture HD60 X", "video"),
                                               ("Virtual Camera", "unknown"),
                                               ("Digital Audio Interface (USB3.0 Capture)", "audio")]
    assert new[0].alt.startswith("@device_pnp_")
    old = c.parse_devices(DEVICES_OLD)
    assert [(d.name, d.kind) for d in old] == [("USB Video", "video"), ("Line In (USB Audio)", "audio")]


def test_devices_with_several_kinds_of_output_are_listed():
    # FFmpeg joins a device's kinds with ", " (libavdevice's " (%s" then ", %s"): a card that also carries its sound,
    # or has a stream FFmpeg has no name for, was dropped from the list when only "video,audio" was read
    text = '''"Game Capture HD60 X" (video, audio)
  Alternative name "@device_pnp_a"
"Elgato Capture (2)" (video, unknown)
"Line (Card)" (audio, unknown)
"Odd Card" (unknown)
"Board" (audio, video)
'''
    assert [(d.name, d.kind) for d in c.parse_devices(text)] == [
        ("Game Capture HD60 X", "video"), ("Elgato Capture (2)", "video"), ("Line (Card)", "audio"),
        ("Odd Card", "unknown"), ("Board", "video")]
    assert c.parse_devices(text)[0].alt == "@device_pnp_a"


def test_a_cards_hdr_modes_are_noted_but_not_used():
    # FFmpeg's DirectShow input can't read P010 and lists it as an unknown compression type (HD60 X in HDR)
    text = (OPTIONS + "  unknown compression type 0x30313050  min s=1920x1080 fps=60 max s=1920x1080 fps=60\n")
    modes = c.parse_modes(text)
    assert c.Mode("p010", False, 1920, 1080, 60.0, 60.0) in modes
    best, note = c.pick_mode(modes)
    assert best.fmt == "yuyv422" and "10-bit HDR" in note and "turn HDR off" in note


def test_a_card_on_usb_2_is_pointed_to_usb_3():
    slow = [c.Mode("nv12", True, 1280, 720, 30.0, 30.0), c.Mode("nv12", True, 640, 480, 30.0, 30.0)]
    assert "USB 3" in c.pick_mode(slow)[1]
    assert "USB 3" not in c.pick_mode(slow + [c.Mode("nv12", True, 1920, 1080, 60.0, 60.0)])[1]


def test_open_failures_are_said_in_plain_words():
    busy = c.open_failure("Game Capture HD60 X", "Could not run graph (sometimes caused by a device already in use "
                                                 "by other application)\n", "[Errno 5] I/O error")
    assert "in use by another app" in busy and "one app at a time" in busy
    assert "Privacy" in c.open_failure("Card", "", "0x80070005")
    gone = "Could not find video device with name [Card] among source devices of type video."
    assert "reconnect" in c.open_failure("Card", gone, "")
    assert c.open_failure("Card", "", "something else") == ""


def test_modes_are_read_with_their_colour_info():
    modes = c.parse_modes(OPTIONS)
    assert len(modes) == 6  # the unknown compression type is skipped
    first = modes[0]
    assert (first.fmt, first.raw, first.w, first.h, first.fps_max, first.range, first.space) == \
        ("yuyv422", True, 1920, 1080, 60.0, "tv", "bt709")
    assert modes[3] == c.Mode("mjpeg", False, 1920, 1080, 60.0, 60.0)
    assert modes[5].range == "pc" and modes[5].space == ""


def test_pick_prefers_raw_at_the_size_and_rate_asked_for():
    modes = c.parse_modes(OPTIONS)
    best, note = c.pick_mode(modes)
    assert (best.fmt, best.w, best.fps_max) == ("yuyv422", 1920, 60.0) and note == ""
    # NV12 beats YUY2 at the same size and rate
    nv12 = c.Mode("nv12", True, 1920, 1080, 60, 60)
    assert c.pick_mode(modes + [nv12])[0] == nv12
    # only MJPEG reaches 60 at 1080p: it's used, with a warning; a raw 30 fps mode isn't preferred over it
    no_raw60 = [m for m in modes if not (m.raw and m.fps_max == 60 and m.w == 1920)]
    best, note = c.pick_mode(no_raw60)
    assert best.fmt == "mjpeg" and "compressed" in note
    # no 1080p at all: the largest size under it
    best, note = c.pick_mode([m for m in modes if m.w != 1920])
    assert (best.w, best.h) == (1280, 720) and "1280x720" in note
    assert c.pick_mode([])[0] is None
    assert c.pick_mode([c.Mode("h264", False, 1920, 1080, 60, 60)])[0] is None


def test_open_options():
    o = c.open_options(c.Mode("nv12", True, 1920, 1080, 59.94, 59.94))
    assert o["video_size"] == "1920x1080" and o["pixel_format"] == "nv12" and o["framerate"] == "59.94"
    assert int(o["rtbufsize"]) == 1920 * 1080 * 3 // 2 * 4
    assert c.open_options(c.Mode("mjpeg", False, 1920, 1080, 30, 60))["vcodec"] == "mjpeg"


# ---- conversion
def _blocks(h=64, w=96, block=16, seed=1):
    """An image of flat 16x16 colour blocks: chroma upsampling differences stay at block edges, compared at centres."""
    rng = np.random.default_rng(seed)
    small = rng.integers(20, 236, (h // block, w // block, 3), dtype=np.uint8)
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST), block


def _centres(img, block):
    return img[block // 2::block, block // 2::block]


def _to_nv12(bgr):
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)  # BT.601 limited, as OpenCV's decoders expect
    h, w = bgr.shape[:2]
    y, u, v = i420[:h], i420[h:h + h // 4].reshape(h // 2, w // 2), i420[h + h // 4:].reshape(h // 2, w // 2)
    return np.concatenate([y.ravel(), np.stack([u, v], -1).ravel()])


def _to_422(bgr, order):
    h, w = bgr.shape[:2]
    i420 = cv2.cvtColor(bgr, cv2.COLOR_BGR2YUV_I420)
    y = i420[:h]
    u = cv2.resize(i420[h:h + h // 4].reshape(h // 2, w // 2), (w // 2, h), interpolation=cv2.INTER_NEAREST)
    v = cv2.resize(i420[h + h // 4:].reshape(h // 2, w // 2), (w // 2, h), interpolation=cv2.INTER_NEAREST)
    parts = {"Y0": y[:, 0::2], "Y1": y[:, 1::2], "U": u, "V": v}
    return np.stack([parts[k] for k in order], -1).ravel()


@pytest.mark.parametrize("fmt", ["nv12", "yuyv422", "uyvy422"])
def test_gpu_conversion_matches_opencv(fmt):
    bgr, block = _blocks()
    h, w = bgr.shape[:2]
    raw = _to_nv12(bgr) if fmt == "nv12" else _to_422(bgr, ["Y0", "U", "Y1", "V"] if fmt == "yuyv422"
                                                        else ["U", "Y0", "V", "Y1"])
    ours = c.raw_to_bgr(torch.from_numpy(raw), fmt, w, h, bt709=False, full_range=False).numpy()
    cv = c.raw_to_bgr_cpu(raw, fmt, w, h)
    assert ours.shape == (h, w, 3)
    assert np.abs(_centres(ours, block).astype(int) - _centres(cv, block)).max() <= 2
    assert np.abs(_centres(ours, block).astype(int) - _centres(bgr, block)).max() <= 4  # back where it started


def test_bt709_and_full_range_round_trip():
    bgr, block = _blocks(seed=2)
    h, w = bgr.shape[:2]
    for bt709, full in ((True, False), (True, True), (False, True)):
        kr, kb = (0.2126, 0.0722) if bt709 else (0.299, 0.114)
        b, g, r = (bgr[..., i].astype(np.float64) for i in range(3))
        y = kr * r + (1 - kr - kb) * g + kb * b
        u, v = (b - y) / (2 * (1 - kb)), (r - y) / (2 * (1 - kr))
        if full:
            yq, uq, vq = y, u + 128, v + 128
        else:
            yq, uq, vq = 16 + y * 219 / 255, 128 + u * 224 / 255, 128 + v * 224 / 255
        q = [np.clip(np.round(p), 0, 255).astype(np.uint8) for p in (yq, uq, vq)]
        raw = np.concatenate([q[0].ravel(), np.stack([q[1][::2, ::2], q[2][::2, ::2]], -1).ravel()])
        out = c.raw_to_bgr(torch.from_numpy(raw), "nv12", w, h, bt709, full).numpy()
        assert np.abs(_centres(out, block).astype(int) - _centres(bgr, block)).max() <= 2, (bt709, full)


# ---- the source, with a raw file standing in for a card
def _raw_file(tmp_path, frames, fmt="nv12"):
    path = tmp_path / f"card.{fmt}"
    with open(path, "wb") as f:
        for bgr in frames:
            f.write((_to_nv12(bgr) if fmt == "nv12" else _to_422(bgr, ["Y0", "U", "Y1", "V"])).tobytes())
    return str(path)


@pytest.mark.parametrize("device", ["cpu", None])
def test_source_plays_a_raw_file_like_a_card(tmp_path, device):
    a, block = _blocks(seed=3)
    b, _ = _blocks(seed=4)
    h, w = a.shape[:2]
    path = _raw_file(tmp_path, [a, a, b, b, b, a])
    src = c.CardSource.raw_file(path, w, h, "nv12", fps=240, device=device, colour=(False, False))
    assert (src.w, src.h, src.fmt, src.raw, src.info.kind) == (w, h, "nv12", True, "capture")
    ticks = list(src)
    src.close()
    assert [t.index for t in ticks] == list(range(6))
    assert all(t.t_ready >= t.t_arrival for t in ticks)
    assert [round(t.t_source * 240) for t in ticks] == list(range(6))  # the "card's" clock, frame by frame
    assert [t.extra["seq"] for t in ticks] == list(range(6))  # numbered as they came off the card
    frames = [t.frame.numpy() if device else t.frame for t in ticks]
    for got, want in zip(frames, [a, a, b, b, b, a]):
        assert np.abs(_centres(got, block).astype(int) - _centres(want, block)).max() <= 4
    assert np.array_equal(frames[0], frames[1])  # a repeat stays bit-identical through the conversion


@pytest.mark.parametrize("found, said", [(None, "CPU (OpenCV)"), ("cpu", "CPU (torch)")])
def test_an_auto_device_is_used_and_said_as_worked_out(tmp_path, monkeypatch, capsys, found, said):
    monkeypatch.setattr(c, "_cuda_device", lambda: found)
    a, _ = _blocks()
    src = c.CardSource.raw_file(_raw_file(tmp_path, [a]), a.shape[1], a.shape[0], "nv12")  # device="auto"
    assert len(list(src)) == 1
    src.close()
    assert src.device == found and f"converted on {said}" in capsys.readouterr().out


def test_source_test_mode_and_yuy2(tmp_path):
    from framegen.sources import TestMode
    frames = [_blocks(seed=s)[0] for s in range(10, 18)]
    h, w = frames[0].shape[:2]
    path = _raw_file(tmp_path, frames, fmt="yuyv422")
    src = c.CardSource.raw_file(path, w, h, "yuyv422", fps=240, device="cpu", test=TestMode(True))
    ticks = list(src)
    src.close()
    assert len(ticks) == 8  # test mode keeps every tick, repeating every other frame's content
    assert np.array_equal(ticks[0].frame.numpy(), ticks[1].frame.numpy())


def test_a_video_file_stands_in_through_the_raw_path(tmp_path):
    frames = [_blocks(seed=s)[0] for s in (20, 20, 21)]
    h, w = frames[0].shape[:2]
    path = str(tmp_path / "clip.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 60, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    src = c.CardSource.video_file(path, device="cpu", colour=(False, False))
    assert (src.raw, src.fmt, src.decode_to) == (True, "nv12", "nv12")  # decoded, then sent on as NV12
    ticks = list(src)
    src.close()
    assert len(ticks) == 3
    for t, want in zip(ticks, frames):
        assert np.abs(t.frame.numpy().astype(int) - want).mean() < 6  # MJPEG and 4:2:0 lose a little


def test_mjpeg_falls_back_to_the_cpu_when_the_gpu_cant(tmp_path, monkeypatch, capsys):
    import av
    frames = [_blocks(seed=s)[0] for s in (30, 31)]
    h, w = frames[0].shape[:2]
    path = str(tmp_path / "card.avi")
    out = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"MJPG"), 60, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    src = c.CardSource(av.open(path), "an MJPEG card", live=False, device="cpu")
    assert (src.raw, src.fmt) == (False, "mjpeg")
    src._gpu_jpeg = True  # as on a PC with CUDA, where NVDEC is tried first

    def no_gpu(self):
        raise RuntimeError("no NVDEC here")
    monkeypatch.setattr(c.CardSource, "_gpu_decoder", no_gpu)
    ticks = list(src)
    src.close()
    assert len(ticks) == 2 and not src._gpu_jpeg
    assert "decoding on the CPU instead" in capsys.readouterr().out
    assert np.abs(ticks[0].frame.numpy().astype(int) - frames[0]).mean() < 6


def _mjpeg_file(tmp_path, frames, damage=()):
    """An MJPEG AVI standing in for a card's stream, with the frames numbered in `damage` broken (every byte zero),
    as a cheap card can send: FFmpeg's decoder can't read them."""
    h, w = frames[0].shape[:2]
    path = tmp_path / "card.avi"
    out = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 60, (w, h))
    for f in frames:
        out.write(f)
    out.release()
    buf = bytearray(path.read_bytes())
    jpegs, i = [], buf.find(b"00dc")
    while i >= 0:  # each frame's chunk: "00dc", its size, then the JPEG (the index at the end names them too)
        if buf[i + 8:i + 10] == b"\xff\xd8":
            jpegs.append((i + 8, int.from_bytes(buf[i + 4:i + 8], "little")))
        i = buf.find(b"00dc", i + 4)
    assert len(jpegs) == len(frames)
    for k in damage:
        start, size = jpegs[k]
        buf[start:start + size] = bytes(size)
    path.write_bytes(buf)
    return str(path)


def _frames_of(src):
    from framegen.io import Waiting
    ticks = [t for t in src if not isinstance(t, Waiting)]
    src.close()
    return ticks


def test_a_damaged_mjpeg_frame_is_skipped_and_capture_goes_on(tmp_path, capsys):
    """A cheap card's broken JPEG (often the first one) ended capture before a single frame was shown."""
    import av
    frames = [_blocks(seed=s)[0] for s in (40, 41, 42, 43)]
    src = c.CardSource(av.open(_mjpeg_file(tmp_path, frames, damage=(0, 2))), "an MJPEG card", live=False,
                       device="cpu")
    ticks = _frames_of(src)
    out = capsys.readouterr().out
    assert [t.extra["seq"] for t in ticks] == [1, 3]  # numbered as the card sent them, the broken ones too
    assert np.abs(ticks[0].frame.numpy().astype(int) - frames[1]).mean() < 6
    assert src.damaged == 2 and "stopped" not in out
    assert out.count("the card sent a damaged MJPEG frame: skipped") == 1  # said once; the count comes at the end
    assert "dropped frames: 0, damaged frames from the card skipped: 2" in out


@pytest.mark.parametrize("damaged, seqs", [(0, [1, 2]), (1, [0, 2])])  # the first frame, as a USB stick sent one
def test_on_the_gpu_a_damaged_frame_doesnt_send_mjpeg_to_the_cpu(tmp_path, monkeypatch, capsys, damaged, seqs):
    import av
    frames = [_blocks(seed=s)[0] for s in (44, 45, 46)]
    src = c.CardSource(av.open(_mjpeg_file(tmp_path, frames, damage=(damaged,))), "an MJPEG card", live=False,
                       device="cpu")
    src._gpu_jpeg = True  # as on a PC with CUDA, where NVDEC is tried first
    nvdec = av.CodecContext.create("mjpeg", "r")  # standing in for NVDEC, which can't read a broken frame either

    def gpu(self, packet):
        decoded = nvdec.decode(packet)
        return torch.from_numpy(decoded[-1].to_ndarray(format="bgr24")) if decoded else None
    monkeypatch.setattr(c.CardSource, "_convert_gpu", gpu)
    ticks = _frames_of(src)
    assert [t.extra["seq"] for t in ticks] == seqs
    assert src._gpu_jpeg and src.damaged == 1  # the CPU couldn't read it either: the frame, so NVDEC stays on
    assert "decoding on the CPU instead" not in capsys.readouterr().out


def test_capture_stops_saying_why_when_no_frame_can_be_read(tmp_path, monkeypatch, capsys):
    import av
    monkeypatch.setattr(c, "DAMAGED_GIVE_UP_S", 0.03)
    frames = [_blocks(seed=s)[0] for s in range(50, 60)]
    src = c.CardSource(av.open(_mjpeg_file(tmp_path, frames, damage=range(10))), "an MJPEG card", live=False,
                       device="cpu")
    assert _frames_of(src) == []
    assert "[capture] stopped: none of the card's MJPEG frames could be read for 0.03 s" in capsys.readouterr().out


def test_frames_keep_the_cards_numbers_when_the_queue_drops_some(tmp_path):
    """A frame dropped from the queue (the PC behind for a moment) leaves a gap in the numbers rather than shifting
    them: Game frame rate 30 fps counts by them (cadence.RhythmLock), and a shift made its beat move, a hitch."""
    import time
    frames = [_blocks(seed=s)[0] for s in range(30, 50)]
    h, w = frames[0].shape[:2]
    path = _raw_file(tmp_path, frames)
    src = c.CardSource.raw_file(path, w, h, "nv12", fps=240, device="cpu", colour=(False, False), queue_size=1)
    ticks = []
    for t in src:
        ticks.append(t)
        time.sleep(0.02)  # slower than the "card" (240 fps): the queue drops frames
    src.close()
    seqs = [t.extra["seq"] for t in ticks]
    assert len(ticks) < len(frames)
    assert seqs == sorted(set(seqs)) and seqs == [round(t.t_source * 240) for t in ticks]
