# chiaki-ng frame tap: PS5 Remote Play into the pipeline

[chiaki-ng](https://github.com/streetpea/chiaki-ng) is an open-source PS5 Remote Play client. This folder adds a
small "frame tap" to it: every compressed video frame the PS5 sends is also copied into shared memory, where
`run.py --chiaki` decodes it, interpolates it and shows it. chiaki-ng keeps doing everything else: pairing, the
controller (it reads it even when its window is in the background), audio, and asking the PS5 to resend after a
network loss.

```
PS5 ─HEVC/H.264 over LAN─▶ chiaki-ng ─▶ its own decoder + window (stays behind ours)
                               │
                               └─ frame tap: each frame + size + loss flags ─▶ shared memory "Local\chiaki-frametap"
                                                                                  │
                         run.py --chiaki: decode ─▶ repeat check by size ─▶ scheduler ─▶ RIFE ─▶ window
                                ▲
                                └── "send a keyframe" flag, when it joins mid-stream or falls behind
```

What the stream gives that a capture card can't:

- **Repeats from frame size.** After compression a repeated frame isn't an exact copy, so the image-based
  repeat check calls most of them new (tested: 464 of 470 ticks went wrong). But a repeat costs the encoder
  almost nothing, ~3 KB against ~55 KB for a new H.264 frame, ~12 KB against ~47 KB for HEVC. The pipeline uses
  the size.
- **Loss flags.** Frames that follow a network loss are held instead of interpolated, so corruption isn't
  smeared into the generated frame.

Other apps can write the same shared memory, compressed or decoded frames: see [docs/frame-tap.md](../docs/frame-tap.md).

## Files

| | |
|---|---|
| `frametap.c`, `frametap.h` | the tap's shared memory, compiled into chiaki-ng (Windows; a no-op elsewhere). Does nothing unless `CHIAKI_FRAMETAP=1` |
| `frametap_session.cpp`, `frametap_session.h` | what the tap does inside chiaki-ng's StreamSession: copying frames and sound out, keyframe requests, the delay test's presses, the sound delay |
| `apply_patch.py` | adds the tap to a chiaki-ng checkout: copies the files above in, then one-line edits to chiaki-ng's own files (six calls in streamsession.cpp, a friend line in streamsession.h, the files in CMakeLists.txt, and one check in qmlmainwindow.cpp: `CHIAKI_FRAMETAP_RENDERER=opengl` makes chiaki-ng draw with OpenGL for that session, without its fallback dialog or changing the saved setting, which SmoovPlay's launcher asks for without an NVIDIA card). It fails loudly if chiaki-ng moved the code, and refuses a checkout with an older tap patch: `git checkout -- gui/src/streamsession.cpp gui/CMakeLists.txt` first |
| `selftest.py`, `selftest.c` | builds a small writer with MSYS2's gcc and checks it against the Python reader byte for byte, including the press, the sound delay and the sound ring. Uses its own tap name, so it runs while chiaki-ng streams |
| `build_windows.sh` | full Windows build in MSYS2, mirroring chiaki-ng's own CI, plus the patch |
| `run_chiaki_frametap.bat` | starts the built chiaki-ng with the tap on |
| `../framegen/chiaki_tap.py` | the reader: `FrameTap` (shared memory), `AudioTap` (the sound ring), `ChiakiSource` (decode, size-based repeats) |
| `../scripts/frametap_fake.py` | a fake PS5 for testing without a console: streams a video file through the tap |

## Build (done once)

You need [MSYS2](https://www.msys2.org/) and a chiaki-ng v1.10.1 source checkout. From PowerShell, in the repo
root (`CHERE_INVOKING` keeps MSYS2's login shell in the current folder). The example assumes MSYS2 in its default
folder, `C:\msys64`, the source in `C:\src\chiaki-ng` and the build going to `C:\src\chiaki-ng-frametap`; use
your own folders, written as MSYS2 paths:

```powershell
$env:MSYSTEM = 'MINGW64'; $env:CHERE_INVOKING = '1'
C:\msys64\usr\bin\bash.exe -lc "bash chiaki_tap/build_windows.sh /c/src/chiaki-ng /c/src/chiaki-ng-frametap"
```

Re-running only rebuilds chiaki-ng. For a newer chiaki-ng, check out its tag in the source folder first; if
`apply_patch.py` stops with "expected the anchor exactly once", the code it hooks into moved: each edit is one
line, so find where that line went in the new streamsession.cpp and update its anchor.

## Testing with a PS5

1. **PS5:** Settings → System → Remote Play → turn on Enable Remote Play. Wire the PS5 and the PC to the router
   with Ethernet if you can. Pick a game that runs at 30 fps (or its 30 fps quality/fidelity mode).
2. **Start chiaki-ng:** run `run_chiaki_frametap.bat` in the build folder. The first time, register the
   PS5: sign in to PSN in chiaki-ng (or paste your account ID), then enter the 8-digit code from
   PS5 Settings → System → Remote Play → Link Device.
3. **chiaki-ng settings:** resolution 1080p, FPS **60** (the pipeline expects the PS5's 60 Hz output with
   repeats), bitrate as high as your network allows. The codec is set by the pipeline before chiaki-ng
   connects ([`framegen/chiaki_settings.py`](../framegen/chiaki_settings.py)): switched to H.265 HDR with SmoovPlay's
   HDR on Auto and Windows' HDR on (or HDR set to `sdr` or `hdr10`), and never switched back; otherwise it's yours to
   pick.
4. **Start the pipeline**, ideally before connecting in chiaki-ng:
   ```powershell
   .venv\Scripts\python run.py --chiaki --display --chiaki-log out\session_GAME.csv
   ```
   It waits for the stream, then opens its window and minimises chiaki-ng's (which keeps streaming and reading
   the controller; restored on exit). Press **M** for the settings menu: frame rate (60/90/120), upscaler,
   AI engine and so on, saved for next time (see the main README). Hotkeys: `f` fullscreen, `i` interpolation
   on/off (instant A/B), `u` next upscaler (matters in fullscreen), `s` stats overlay, `l` delay test, `q` quit.

   chiaki-ng passes on the stream's codec header only once, as the stream's first packet. If no run.py has
   seen this stream start, the pipeline can't decode it and says so: end the stream in chiaki-ng (Ctrl+O, the
   x) and connect again. Once one run has seen it, later runs reuse the saved header and join without that.

   **No 30 fps game to hand?** Play any 60 fps game as if it were 30 fps, and score every generated frame
   against the real frame that was left out:
   ```powershell
   .venv\Scripts\python run.py --chiaki --test-mode --score --display --chiaki-log out\session_GAME_as30.csv
   ```
   Tested with the fake PS5 streaming a 60 fps recording: RIFE 35.5 dB against the real frames, where showing
   the repeat (what a 30 fps game does) scores 26.5 dB.
5. **Afterwards**, attach the console summary and the CSV to a GitHub issue. The CSV has one row per tick: frame size, loss flags,
   decode time, what the size check and the image check each said, and what the pipeline did. That's what
   tunes the size threshold on real PS5 streams.

Decoding runs on the GPU by default (`--chiaki-decoder cuda`, NVDEC); `--chiaki-decoder sw` is the fallback.

## Back the other way: button presses and the sound delay

The tap's header also carries two requests from the pipeline to chiaki-ng (the header's `features` field says
which ones this chiaki-ng understands):

- **Press a button** (the delay test, **L**): the pipeline asks for a button and how long to hold it. chiaki-ng
  adds it to the controller state it sends the PS5, notes the time it went out (the clock both sides share), and
  lets go after the hold. The pipeline then times the first frame that shows the answer.
- **Hold the sound back** (Sound in sync): chiaki-ng plays the sound as it arrives, while the picture spends a while
  in SmoovPlay. The pipeline writes that while (the median of the last ~2 s of frames, every 2 s) and chiaki-ng
  delays its sound by as much, 0–500 ms. If the pipeline stops (its heartbeat is over 2 s old), the delay drops
  to 0 by itself, so chiaki-ng on its own never plays late sound.

## The game's sound, for instant replay

chiaki-ng also copies the game's sound, as it arrives and before its own volume, into a second shared-memory ring
(`<tap name>-audio`: 512 slots of up to 8 KB of 16-bit PCM, each with its arrival time; layout in `frametap.h`).
A chunk too big for one slot is split by whole samples, each part timed by the samples before it. The header's
`features` gains a bit for it, so SmoovPlay knows a chiaki-ng from before this ring (whose instant replays are
silent). Instant replay (`framegen/replay.py`) reads it every 20 ms and times the sound by SmoovPlay's delay, so
it meets the picture.

`CHIAKI_FRAMETAP_NAME` gives the tap another shared-memory name (the self-test uses it).

## Testing without a PS5

```powershell
.venv\Scripts\python scripts\frametap_fake.py testdata\clip_1080p60.mp4 --seconds 20 --codec hevc --loss 0.01 --jitter 2 --chiaki-like
.venv\Scripts\python run.py --chiaki --display    # in another terminal
```

`--chiaki-like` sends the codec header once, as its own packet, the way chiaki-ng does. The clip's sound (or a
tone with a click each second) goes into the sound ring as it's due; `--no-sound` leaves it out. `--native` streams a
60 fps file as a 60 fps game (for test mode, `--test-mode`). While chiaki-ng is streaming, give the fake and run.py
another tap name (`--tap NAME` and `--chiaki-tap NAME`) so they don't collide with it.

## First PS5 session (10 Nov 2026)

- The stream is H.265 1920×1080 at 60 fps. chiaki-ng delivers it as expected: about 57 frames a second, no
  losses over Ethernet, about 2 ms to decode a frame.
- The PS5 sends the codec header out of band, once per stream, and answers a keyframe request with an intra
  frame that isn't marked as a keyframe in its NAL type. The reader now keeps the header (and saves it for later
  runs) and no longer waits for a keyframe after a gap.
- Only menus and near-still screens were played (true 60 fps, passed through), so the 30 fps cadence over
  Remote Play is still untested. On near-still screens every frame is about 500 bytes and the size check calls
  them all new; harmless there, since nothing moves.
