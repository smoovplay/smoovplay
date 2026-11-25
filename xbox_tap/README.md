# smoovplay-xbox (Native Xbox WebRTC → SmoovPlay FrameTap)

A standalone, zero-CGo Go client for **Xbox Remote Play (`xhome`)** and **Xbox Cloud Gaming (`xcloud`)** that bypasses Chromium/WebView2 completely:

- **0 ms video jitter buffer:** reassembles RFC 6184 H.264 RTP packets directly into Annex-B frames and writes them straight into SmoovPlay's `Local\smoovplay-tap` shared-memory ring (`version 1`, same layout as `chiaki_tap/frametap.h`).
- **Zero-decode resolution detection:** parses H.264 SPS NAL units directly in Go (1080p / 1440p) with no CPU decode spike on keyframes.
- **Instant keyframe recovery (1 RTT):** answers SmoovPlay's `idr_request` with both RTCP PLI and Xbox's `control` channel `videoKeyframeRequested`.
- **Native 125 Hz XInput + rumble + Delay Test (`L`):** reads XInput directly (`xinput1_4.dll`), sends Xbox's binary `input` DataChannel packets, and implements `FEATURE_PRESS` so SmoovPlay's built-in Delay Test (`L`) works natively on Xbox and SmoovPlay can run in **unoccluded DXGI fullscreen** instead of a click-through overlay.
- **Pure-Go Opus audio + sync:** decodes 48 kHz stereo Opus via `go-opus`, writes raw PCM into SmoovPlay's `-audio` ring (for Instant Replay), applies SmoovPlay's `audio_delay_ms` holdback, and plays via Windows `waveOut`.
- **Live WebRTC telemetry:** prints live RTT, 0 ms jitter buffer, fps, keyframes, packet loss, and audio packet counters every 2 seconds.

## Build

Requires Go 1.27+ (or any Go 1.21+ with `GOTOOLCHAIN=auto`). No C compiler, no CGo, no external DLLs.

On Windows:
```powershell
cd xbox_tap
go build -o smoovplay-xbox.exe ./cmd/smoovplay-xbox
```

Cross-compile for Windows from Linux/macOS:
```bash
cd xbox_tap
CGO_ENABLED=0 GOOS=windows GOARCH=amd64 go build -o smoovplay-xbox.exe ./cmd/smoovplay-xbox
```

## Usage

1. **Sign in once** (Microsoft Device Code OAuth — caches refresh token in `%LOCALAPPDATA%\SmoovPlay\xbox_tokens.json`):
   ```powershell
   .\smoovplay-xbox.exe login
   ```
2. **List your consoles or xCloud titles:**
   ```powershell
   .\smoovplay-xbox.exe consoles
   .\smoovplay-xbox.exe titles halo
   ```
3. **Start streaming into SmoovPlay:**
   ```powershell
   # Terminal 1: Xbox Remote Play (auto-picks your first console, or pass --console "My Xbox")
   .\smoovplay-xbox.exe play --home --res 1440p

   # Or xCloud:
   .\smoovplay-xbox.exe play --cloud <titleId> --res 1080p

   # Terminal 2: SmoovPlay in unoccluded low-delay DXGI fullscreen
   python run.py --source tap --display --present dxgi --fullscreen
   ```
