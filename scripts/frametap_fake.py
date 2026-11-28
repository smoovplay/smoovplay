"""A fake PS5 for testing the chiaki frame tap without a console.

Encodes a video into a 60 Hz H.264/HEVC stream the way a PS5 sends a 30 fps game (every frame twice, a long
GOP, keyframes only on request) and writes it into the frame tap's shared memory, paced in real time:

    .venv\\Scripts\\python scripts\\frametap_fake.py testdata\\clip_1080p60.mp4 --seconds 20
    .venv\\Scripts\\python run.py --chiaki --display          # in another terminal, before or after

--loss drops random frames (the next one is flagged frames_lost=1, then a keyframe is sent and flagged
recovered, roughly what the PS5 does). --jitter adds random arrival delay. Answers the reader's keyframe requests.
The clip's sound (or, without any, a quiet tone with a click each second) goes into the sound ring, 10 ms at a time
as it's due, as chiaki-ng's does, for instant replay; --no-sound leaves it out (an older chiaki-ng).

--hdr sends what a PS5 in HDR does (chiaki-ng's H.265 HDR codec): 10-bit HEVC in HDR10 (PQ, BT.2020), made from the
SDR clip by framegen.hdr.expand_sdr (SDR white at 203 nits, highlights up to --peak nits), for SmoovPlay's HDR setting.

On a Mac it's the same with .venv/bin/python. The tap is POSIX shared memory there, so --tap NAME (and run.py's
--chiaki-tap NAME) can have at most 24 characters after any Local\\, leaving room for the sound ring's "-audio".
Without NVENC it encodes with libx264 or libx265, which keep up at 1080p60 on Apple's chips. The exception is --hdr,
whose 10 bits go through VideoToolbox's encoder.
"""
from __future__ import annotations

import argparse
import random
import sys
import time
from fractions import Fraction
from pathlib import Path

import av
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from framegen.chiaki_tap import (AUDIO_SUFFIX, FEATURE_AUDIO_TAP, NAME, PARAM_NALS, AudioTap,  # noqa: E402
                                 FrameTap, nal_units, param_sets)

TICK = 1 / 60
SOUND_RATE = 48000


def sound_of(path: str, start: float, seconds: float) -> np.ndarray:
    """The clip's sound from `start`, as 48 kHz stereo 16-bit samples (n, 2); without any, a quiet 440 Hz tone with a
    click on each whole second."""
    n = int(seconds * SOUND_RATE)
    try:
        cont = av.open(path)
        if cont.streams.audio:
            res = av.AudioResampler(format="s16", layout="stereo", rate=SOUND_RATE)
            parts = []
            for fr in cont.decode(cont.streams.audio[0]):
                if fr.time is not None and fr.time + fr.samples / fr.sample_rate < start:
                    continue
                if fr.time is not None and fr.time >= start + seconds:
                    break
                parts += [o.to_ndarray().reshape(-1, 2) for o in res.resample(fr)]
            cont.close()
            if parts:
                return np.concatenate(parts)[:n]
        cont.close()
    except Exception as e:  # noqa: BLE001 - fall back to the tone
        print(f"[fake ps5] no sound from the clip ({e})")
    t = np.arange(n)
    tone = (np.sin(2 * np.pi * 440 * t / SOUND_RATE) * 2000).astype(np.int16)
    tone[(t % SOUND_RATE) < SOUND_RATE // 100] = 12000  # a 10 ms click each second
    return np.repeat(tone[:, None], 2, axis=1)


def encoder(codec: str, w: int, h: int, mbps: float, sw: bool = False, hdr: bool = False):
    names = {"h264": ["h264_nvenc", "libx264"], "hevc": ["hevc_nvenc", "libx265"]}[codec]
    if hdr and sys.platform == "darwin":  # libx265's 10 bits take about 35 ms a 1080p frame here, too slow for 60 Hz
        names.insert(-1, "hevc_videotoolbox")
    if sw:
        names = names[-1:]
    for name in names:
        try:
            ctx = av.CodecContext.create(name, "w")
        except Exception:
            continue
        ctx.width, ctx.height = w, h
        ctx.pix_fmt = ("p010le" if "nvenc" in name or "videotoolbox" in name else "yuv420p10le") if hdr else "yuv420p"
        if hdr:  # HDR10's tags, as a PS5 sends them
            ctx.color_primaries, ctx.color_trc, ctx.colorspace, ctx.color_range = 9, 16, 9, 1
        ctx.time_base, ctx.framerate = Fraction(1, 60), Fraction(60)
        ctx.bit_rate = int(mbps * 1e6)
        ctx.gop_size = 100000  # like Remote Play: no periodic keyframes
        ctx.max_b_frames = 0
        if "nvenc" in name:
            # delay=0: output each frame's packet right away, so loss/keyframe flags land on the right frame
            ctx.options = {"preset": "p4", "tune": "ull", "zerolatency": "1", "rc": "cbr", "forced-idr": "1",
                           "delay": "0", **({"profile": "main10"} if hdr else {})}
        elif "videotoolbox" in name:
            # low_delay turns on Apple's low-latency rate control, which puts out each frame's packet right away, so
            # loss and keyframe flags land on the right frame. Without it packets come a few frames late, two at a time.
            ctx.flags |= av.codec.context.Flags.low_delay
            ctx.options = {"realtime": "1", "prio_speed": "1", **({"profile": "main10"} if hdr else {})}
        elif name == "libx264":
            ctx.options = {"preset": "veryfast", "tune": "zerolatency", "forced-idr": "1"}
        else:
            ctx.options = {"preset": "veryfast", "tune": "zerolatency", "x265-params": "repeat-headers=1" + (
                ":colorprim=bt2020:transfer=smpte2084:colormatrix=bt2020nc:range=limited:hdr10=1" if hdr else "")}
            if sys.platform == "darwin":
                # A Mac has no NVENC, so libx265 answers the reader's keyframe requests (at the start, and after
                # --loss). Its forced intra frame is otherwise a TRAIL picture that still lists earlier frames as
                # references, so FFmpeg skips every frame after it until the next IRAP, up to 250 frames later. An
                # IDR repairs the stream, as NVENC's forced-idr does on Windows.
                ctx.options["forced-idr"] = "1"
        try:
            ctx.open()
        except Exception:
            continue
        return ctx
    raise RuntimeError(f"no {codec} encoder available")


def make_hdr(enc, w: int, h: int, peak: float, sw: bool):
    """fn(SDR frame) -> the same picture as an HDR10 frame for `enc` (framegen.hdr.expand_sdr). It runs on the GPU
    unless sw, so on CUDA, or on Apple's GPU on a Mac, where the CPU is three times too slow for a 1080p stream."""
    import torch

    from framegen import gpu
    from framegen import hdr as H
    dev = gpu.device() if not sw else "cpu"
    pix = enc.pix_fmt

    def put(frame, arrays):
        for plane, arr in zip(frame.planes, arrays):
            rows = np.frombuffer(plane, np.uint16, count=arr.shape[0] * (plane.line_size // 2))
            rows.reshape(arr.shape[0], plane.line_size // 2)[:, :arr.shape[1]] = arr

    def to_hdr(fr):
        bgr = torch.from_numpy(fr.to_ndarray(format="bgr24")).to(dev)
        y, u, v = (p.cpu().numpy().astype(np.uint16) for p in H.nits_to_yuv10(H.expand_sdr(bgr, peak)))
        out = av.VideoFrame(w, h, pix)
        if pix == "p010le":  # the 10 bits at the top of 16, chroma interleaved
            put(out, (y << 6, np.stack([u, v], -1).reshape(u.shape[0], -1) << 6))
        else:
            put(out, (y, u, v))
        tag(out)
        return out

    return to_hdr


def copy_frame(img):
    """A copy of a 10-bit frame (PyAV can't make P010 into a NumPy array and back)."""
    out = av.VideoFrame(img.width, img.height, img.format.name)
    for src, dst in zip(img.planes, out.planes):
        np.frombuffer(dst, np.uint8)[:] = np.frombuffer(src, np.uint8)
    tag(out)
    return out


def tag(fr) -> None:
    """HDR10's colour tags on a frame: BT.2020 primaries and matrix, PQ, limited range."""
    fr.color_primaries, fr.color_trc, fr.colorspace, fr.color_range = 9, 16, 9, 1


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("video")
    ap.add_argument("--start", type=float, default=0.0)
    ap.add_argument("--seconds", type=float, default=20.0)
    ap.add_argument("--codec", default="h264", choices=["h264", "hevc"])
    ap.add_argument("--mbps", type=float, default=15.0)
    ap.add_argument("--loss", type=float, default=0.0, help="probability a frame is lost in the network")
    ap.add_argument("--jitter", type=float, default=0.0, help="arrival jitter, ms (std dev)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds to wait before starting the stream")
    ap.add_argument("--tap", default=NAME, help="shared-memory name (use another one while chiaki-ng is live)")
    ap.add_argument("--chiaki-like", action="store_true",
                    help="send the codec header once, as its own packet, and never again (what chiaki-ng does)")
    ap.add_argument("--sw", action="store_true", help="encode on the CPU (libx264/libx265), leaving the GPU alone")
    ap.add_argument("--no-sound", action="store_true", help="no sound ring (a chiaki-ng from before it)")
    ap.add_argument("--hdr", action="store_true",
                    help="an HDR stream (HEVC Main10, HDR10), as chiaki-ng's H.265 HDR codec gets from a PS5")
    ap.add_argument("--peak", type=float, default=1000.0, help="--hdr: the brightest highlights, in nits")
    ap.add_argument("--native", action="store_true",
                    help="a 60 fps game: send every frame of a 60 fps file once, instead of a 30 fps game's repeats")
    ap.add_argument("--game-fps", type=float, default=None, metavar="F",
                    help="a game at F fps (40-60) from a 60 fps file: its frames evenly spaced in the game, each on "
                         "the next 60 Hz tick, so some stay up twice as long (the uneven arrival Even out 40-60 fps "
                         "fixes)")
    a = ap.parse_args()
    if a.hdr:
        a.codec = "hevc"  # a PS5's HDR is H.265 only
    if a.game_fps:
        a.native = True  # one 60 Hz tick per source frame; the game's repeats are made below
    time.sleep(a.delay)
    rng = random.Random(a.seed)

    cont = av.open(a.video)
    vs = cont.streams.video[0]
    fps = float(vs.average_rate or 30)
    step = 2 if fps > 45 and not a.native else 1  # a 60 fps file: keep every other frame as the "30 fps game"
    reps = 1 if a.native else 2
    if a.start:
        cont.seek(int(a.start / vs.time_base), stream=vs)
    w, h = vs.codec_context.width, vs.codec_context.height
    enc = encoder(a.codec, w, h, a.mbps, a.sw, a.hdr)
    print(f"[fake ps5] {enc.name} {w}x{h} {a.mbps} Mbps, {fps:.0f} fps source -> 60 Hz stream"
          + ("" if a.native else " with repeats") + (f", HDR10 (highlights to {a.peak:.0f} nits)" if a.hdr else ""))
    to_hdr = make_hdr(enc, w, h, a.peak, a.sw) if a.hdr else None

    tap = FrameTap(a.tap, a.tap + "-frame")
    tap.begin({"h264": 0, "hevc": 2 if a.hdr else 1}[a.codec], w, h, 60)
    sound, pcm, sent_s = None, None, 0
    if not a.no_sound:
        pcm = sound_of(a.video, a.start, a.seconds)
        sound = AudioTap(a.tap + AUDIO_SUFFIX)
        sound.begin(SOUND_RATE, 2)
        tap.set_features(FEATURE_AUDIO_TAP)
    header_sent = False
    t0 = time.perf_counter() + 0.2
    tick, pts, lost_pending, force_key, recover_next = 0, 0, 0, True, False
    sizes = {"new": [], "repeat": [], "key": []}
    served, dropped = 0, 0
    prev = held = None  # --game-fps: the source frame before this one, and the game frame on screen
    try:
        for i, fr in enumerate(cont.decode(vs)):
            if fr.time is not None and fr.time < a.start:
                continue
            if fr.time is not None and fr.time >= a.start + a.seconds:
                break
            if i % step:
                continue
            img = to_hdr(fr) if to_hdr else fr.reformat(format="yuv420p")
            game_repeat = False
            if a.game_fps:  # tick i shows game frame g, made at g/F s: source frame round(g*60/F), i or i-1
                g = int(i * a.game_fps / 60)
                game_repeat = i > 0 and g == int((i - 1) * a.game_fps / 60)
                if game_repeat:
                    img = held
                elif round(g * 60 / a.game_fps) < i and prev is not None:
                    img = prev
                prev, held = (to_hdr(fr) if to_hdr else fr.reformat(format="yuv420p")), img
            for rep in range(reps):
                rep = rep or game_repeat
                if tap.take_idr_request():
                    force_key = True
                    served += 1
                if rep:  # a fresh frame (the encoder may still hold the last one)
                    out = copy_frame(img) if to_hdr else av.VideoFrame.from_ndarray(img.to_ndarray(), format="yuv420p")
                else:
                    out = img
                out.pts = pts
                pts += 1
                if force_key:
                    out.pict_type = av.video.frame.PictureType.I
                packets = enc.encode(out)
                data = b"".join(bytes(p) for p in packets)
                if not data:
                    continue
                key = force_key
                force_key = False
                t_due = t0 + tick * TICK + max(0.0, rng.gauss(0, a.jitter / 1000)) if a.jitter else t0 + tick * TICK
                tick += 1
                if (wait := t_due - time.perf_counter()) > 0:
                    time.sleep(wait)
                if sound is not None:  # the sound that's due by now, in 10 ms chunks
                    due = min(len(pcm), int((time.perf_counter() - t0) * SOUND_RATE))
                    while sent_s + 480 <= due:
                        sound.push(pcm[sent_s:sent_s + 480].tobytes())
                        sent_s += 480
                if a.loss and not key and rng.random() < a.loss:
                    lost_pending += 1
                    dropped += 1
                    force_key, recover_next = True, True  # the PS5 answers a loss with a recovery frame
                    continue
                if a.chiaki_like:
                    if not header_sent:
                        tap.push(param_sets(data, a.codec))
                        header_sent = True
                    data = b"".join(u for t, u in nal_units(data, a.codec) if t not in PARAM_NALS[a.codec])
                tap.push(data, frames_lost=lost_pending, recovered=recover_next and key)
                if key:
                    recover_next = False
                lost_pending = 0
                sizes["key" if key else ("repeat" if rep else "new")].append(len(data))
        # The reader stops at a dead header with whatever it hasn't read still in the ring, so on a slow machine
        # ending right after the last frame loses it the last few. Give it a moment to take them first.
        tap.wait_read(1.0)
    finally:
        tap.end()
        if sound is not None:
            sound.end()
    med = {k: int(np.median(v)) if v else 0 for k, v in sizes.items()}
    print(f"[fake ps5] sent {sum(map(len, sizes.values()))} frames in {time.perf_counter() - t0:.1f} s; "
          f"median bytes: new {med['new']}, repeat {med['repeat']}, keyframe {med['key']}; "
          f"keyframe requests answered {served}; frames lost {dropped}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
