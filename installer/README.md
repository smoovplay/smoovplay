# SmoovPlay launcher and installer

Two Windows programs built from one C# source tree in `launcher\`. They use WPF on .NET Framework 4.8, which every Windows 10 and 11 PC has.

| File | What it is |
|---|---|
| `SmoovPlay-Setup.exe` (~140 MB) | The launcher with the app and chiaki-ng embedded. Walks through **Check your PC → Choose a folder → Install → Ready to play**. Installs per user (no admin rights), downloads the runtime from official sources, prepares the GPU and adds shortcuts. |
| `SmoovPlay.exe` (~0.3 MB) | The launcher, inside every install. Its left rail has five pages: **Play**, **Connect your PS5**, **PC check**, **Settings** and **Help**. Play starts the pipeline *before* chiaki-ng, so the stream's codec header is never missed. **Test with a 60 fps game**, under Play, does the same in test mode (`run.py --test-mode`) and shows a results card when it stops. While a stream plays, the launcher minimises to the taskbar, with a dot on its button (green: playing, amber: starting, red: stopped with an error), and comes back up when the session ends. **Settings** lists the pipeline's own settings (`python -m framegen.settings`): switches for on/off, dropdowns for the rest, each choice with a line on when to pick it. The interpolation model and upscaler dropdowns offer only what SmoovPlay supports today, and end with a note that other models may come later. **Auto-detect**, at the top, runs `python -m framegen.autodetect` (about 10 s; off while playing, and Play waits for it), shows what it found and what it suggests with a reason for each, and **Apply** sets the suggestions. Its output goes to `logs\autodetect.txt`. With nothing playing it edits the settings file; while a stream plays it sends each change to the running pipeline (`run.py --control`, see `framegen/remote.py`), which applies it straight away, and it follows changes made in the video window's menu. |

Source files:

| File | Contents |
|---|---|
| `Core.cs` | Entry point, paths, the install steps, play sessions, chiaki-ng handling, diagnostics, uninstall. |
| `Checks.cs` | The PC check and its fixes. |
| `Ui.xaml` | The window: layout, styles and animations. Embedded and loaded at run time, with no code-behind. |
| `Ui.cs` | The window's logic. |
| `SettingsPage.cs` | The Settings page, built from the pipeline's own list of settings. |
| `AutoDetect.cs` | Auto-detect: times this PC and suggests settings. |
| `Vfg.cs` | NVIDIA VFG setup and removal, with NVIDIA's licence. |
| `Feedback.cs` | The Experimental badge and Send feedback. |
| `Updater.cs` | Updates: finds the newest GitHub release, downloads and checks its setup, hands over to it. |
| `UpdateSignature.cs` | Signs and checks the `.sig` file that comes with each setup (also used by `tools\SignUpdate.cs`). |

## Build

```
.venv\Scripts\python installer\build.py                  # both exes, into installer\dist
.venv\Scripts\python installer\build.py --version 0.2.0  # a release's version (default: VERSION in build.py)
.venv\Scripts\python installer\build.py --launcher-only  # only SmoovPlay.exe
```

A full build also writes `SmoovPlay-Setup.exe.sig`, the update signature (see Updates).

Needs:

* Roslyn `csc.exe` from Visual Studio 2019+ Build Tools.
* WPF's assemblies from `C:\Windows\Microsoft.NET\Framework64\v4.0.30319\WPF`, which are part of Windows.
* The patched chiaki-ng build: `--chiaki-build <folder>`, or set `SMOOVPLAY_CHIAKI_BUILD`.
* Its source checkout, which supplies the licence texts and the patch as a git diff: `--chiaki-src <folder>`, or set `SMOOVPLAY_CHIAKI_SRC`.
* `models\flownet_v4.22.lite.pkl`.

`build.py` pins the embeddable Python's SHA-256, and setup refuses a download that doesn't match it.

## The PC check, and what could stop an install

Setup opens with the PC check, and the launcher's **PC check** page shows the same cards. A **Fail** blocks Continue; **Continue anyway** is there in case a check is wrong. Hovering a card shows more help.

| Card | Checks | If it's a problem |
|---|---|---|
| Graphics card | An NVIDIA card (WMI), with its memory from `nvidia-smi`; otherwise an AMD or Intel card is named. | An AMD Radeon or Intel Arc card: warning, Experimental (RIFE runs through DirectML, slower; expect 60 fps), with **Send feedback**. Built-in graphics: warning, likely too slow. No graphics card found, or an NVIDIA card older than RTX: fail. RTX 20, laptop, or under 6 GB: warning. |
| NVIDIA driver | The version: CUDA 12 needs 528 or newer, and 570+ is recommended (RTX 50 needs it). | **Get the driver** opens NVIDIA's driver page. NVIDIA's installer is theirs to run. Without an NVIDIA card: info, not needed. |
| This PC | 64-bit Windows 10 or 11, and memory. | Under 8 GB: warning. |
| Visual C++ runtime | `msvcp140.dll` 14.40 or newer and `vcruntime140_1.dll`. PyTorch won't load without them, and crashes with older ones. | **Install it for me**: downloads `vc_redist.x64.exe` from Microsoft, checks its Authenticode signature is valid and Microsoft's (WinVerifyTrust), then runs it `/passive` (Windows asks for permission). |
| Disk space | Free space on the install drive (about 12 GB needed). | Warning here. The folder step blocks Install until another drive is chosen. |
| Download servers | python.org, bootstrap.pypa.io, pypi.org, files.pythonhosted.org, download.pytorch.org and pypi.nvidia.com, through Windows' proxy settings. | All unreachable: fail. Some: warning. A system proxy is passed to pip (`--proxy`). |
| Display | The fastest refresh rate a monitor is set to, and the fastest it could be set to. | Set below what it supports: **Open display settings**. A 60 Hz monitor is fine (info). |
| Home network | Wired or Wi-Fi; on Wi-Fi, the band, 2.4, 5 or 6 GHz (from `netsh wlan`). | 2.4 GHz: warning, suggesting 5 or 6 GHz or a cable. |

The folder step also checks the folder:

* **Admin-only places** (Program Files, Windows) are refused.
* **Network and removable drives** are refused.
* **Long paths:** with Windows' long paths off, a path over 80 characters is refused, because the deepest package file sits about 155 characters below the install folder. The step offers **Use a shorter path** (`<drive>:\SmoovPlay`) and **Turn on long paths** (admin).
* **An existing install** is updated in place.

Not handled in the app:

* **SmartScreen.** "Windows protected your PC" appears before any of our code runs; the way through is More info, then Run anyway. Only code signing (and download history) removes it. The main README explains it to players, with how to check the file's SHA-256.
* **Antivirus.** Some products may quarantine an unsigned exe.
* **.NET Framework 4.8.** It's built into Windows 10 1903 and later.

## Connect your PS5

The launcher opens on this page the first time it runs. It has eight steps the user ticks off, and SmoovPlay remembers them in `%APPDATA%\framegen\smoovplay-launcher.txt`:

1. Turn on Remote Play.
2. Rest-mode settings (optional).
3. Sign in to PSN in chiaki-ng.
4. Register the PS5, automatically or with a PIN.
5. Set the stream to 60 fps. With HDR on Auto (the default), SmoovPlay switches chiaki-ng to the H.265 HDR codec when Play starts and Windows' HDR is on; it never switches it back.
6. Black picture? Turn off HDCP (only if needed). If the picture stays black, or the PS5 says the content can't be shown, turn off Enable HDCP (PS5: Settings → System → HDMI), and turn it back on afterwards for video apps like Netflix.
7. Connect a controller.
8. Press Play.

A capture card's guide has the same HDCP step; there it's always needed on a PS5.

**Open chiaki-ng** starts the bundled chiaki-ng with the frame tap on. A later Play then reuses that same chiaki-ng.

## What setup does

Every step can run again: **Retry** after a failure, or **Repair** in Help later, picks up where it stopped. Closing setup mid-way asks first, then pauses it.

1. Unpacks these:
   * `app\`: run.py, framegen and the RIFE weights.
   * `chiaki\`: chiaki-ng with the frame tap, its licences and the patch source.
   * `SmoovPlay.exe`.
2. Downloads the Python 3.12.10 embeddable package from python.org, verifies its SHA-256, and sets its `._pth` file to see `Lib\site-packages` and `..\..\app`. It then installs pip with `get-pip.py`.
3. Runs `pip install -r app\requirements-runtime.txt` with pinned versions, about 5 GB:
   * Sources: pypi.org, plus download.pytorch.org for CUDA PyTorch and pypi.nvidia.com for TensorRT's wheels.
   * setuptools and wheel go in first, and pip runs with `--no-build-isolation`, because pip's isolated build doesn't work in the embeddable Python.
   * The page shows download speed and time left.
4. `python -m framegen.prepare` builds the TensorRT engine for full motion detail; at half, Auto uses PyTorch, which is faster there. On a new install it then picks settings for the PC (`framegen.autodetect --apply --if-new`, about 10 s). It saves nothing if the graphics card is busy with something else, and never replaces an existing settings file.
5. Adds Start-menu and desktop shortcuts and an Apps & features entry (HKCU), then writes `runtime\ready.txt`.

None of PyTorch, TensorRT, CUDA or PyAV is ever redistributed by us.

## Install layout

```
SmoovPlay\                   default: %LOCALAPPDATA%\Programs\SmoovPlay
  SmoovPlay.exe
  LICENSE.txt, THIRD_PARTY_NOTICES.txt
  app\                       run.py, framegen\, models\ (weights; TensorRT engines in models\trt)
  chiaki\                    chiaki-ng + licenses\ + source\
  runtime\python\            embeddable Python with the packages
  logs\                      setup-*.log, session-*.log, results-*.txt (test mode reports, also in Copy diagnostics),
                             autodetect.txt (the last Auto-detect run)
```

Settings stay in `%APPDATA%\framegen\` and are kept on uninstall.

Uninstall (Help → Uninstall, or Apps & features) removes only what setup created, then the folder if it's empty. Setup never installs into a folder that already holds other files; it makes a `SmoovPlay` subfolder instead.

## Updates

The launcher finds new releases on GitHub and, with a click, installs them itself.

**Finding one.** When the launcher opens, it asks GitHub for the newest release in `RELEASES_REPO` (set in `build.py`; empty turns updates off). Drafts are skipped; pre-releases count, so betas can go out as pre-releases. Tags are versions: `v0.2.0`, `v0.2.0-beta.2`. The answer is cached for 6 hours in `%APPDATA%\framegen\smoovplay-update.txt` (shared by every install), and the request carries only the app's version, in its User-Agent. **Check for updates** under the version in the rail asks again straight away. A newer release shows **Update available: v0.2.0** there instead, and the Help page mentions it too.

**Installing it.** Clicking that line asks **Update to v0.2.0?** (Update now, What's new, Later). Update now:

1. Downloads `SmoovPlay-Setup.exe` from the release into `%TEMP%\SmoovPlay-update\`, with progress in the rail.
2. Checks it before running anything:
   * The SHA-256 must match both GitHub's asset digest and the signed one in `SmoovPlay-Setup.exe.sig`.
   * The `.sig` must be signed with SmoovPlay's update key, whose public half is built into every launcher.
   * The signed version must be the release's tag, and newer than the running app.
   * Nothing runs if any check fails, and the error offers the release page instead.
3. Closes chiaki-ng if it's running from this install, starts `SmoovPlay-Setup.exe --update --dir <install> --wait <launcher pid>`, and closes the launcher.
4. Setup waits for the launcher to exit, then runs its usual steps in place. Python and packages already there are kept, and pip only adds what changed. Engines stay cached, and settings, looks and clips are kept. Then it opens SmoovPlay again.

The next launcher start deletes the downloaded setup. A dev launcher (`SMOOVPLAY_APP`), or a build without the update key, only links to the release page.

**Publishing a release.** Build with `--version`, then upload **both** `installer\dist\SmoovPlay-Setup.exe` and `SmoovPlay-Setup.exe.sig` to a GitHub release tagged `v<version>`. A release without the `.sig` only gets a link to its page.

**The signing key.** The first build makes an RSA-3072 key at `%USERPROFILE%\.smoovplay\update-signing-key.xml` (`SMOOVPLAY_SIGNING_KEY` overrides it), outside the repo. Back it up somewhere safe and never commit it:

* Without it, installed copies can't update themselves. Running a new setup by hand still updates them.
* With it, anyone could sign a setup that installed copies would accept.

`installer\build\SignUpdate.exe sign <key> <file> <version>` signs a file by hand.

**While the repo is private** GitHub's API hides its releases from anyone not signed in, so installed copies find no update. `SMOOVPLAY_GITHUB_TOKEN=<token>` lets a test machine see them; it's sent only to api.github.com, never to the download host it redirects to.

Test switches:

* `SMOOVPLAY_UPDATE_CHECK=0` turns updates off.
* `SMOOVPLAY_RELEASES_REPO=owner/name` points the check at another repo.
* `SMOOVPLAY_RELEASES_API=http://127.0.0.1:<port>` sends it to a stand-in for GitHub's API.

## Command line

```
SmoovPlay-Setup.exe --unattended --dir "C:\Games\SmoovPlay" [--no-shortcuts] [--skip-prepare]   # log: <dir>\install.log, exit code 0 = ok
SmoovPlay-Setup.exe --update --dir "<install>" [--wait <pid>]   # update in place with the window (what Update now runs)
SmoovPlay.exe --page play|guide|check|settings|help    # open on a page
SmoovPlay.exe --diagnostics out.txt           # what "Copy diagnostics" copies (user folder replaced by %USERPROFILE%)
SmoovPlay.exe --uninstall [--yes]             # --yes: already confirmed (the Help page asks in-app)
SmoovPlay.exe --check-signature file.exe      # exit 0 if Microsoft-signed (tests the Visual C++ download check)
```

## Testing

Dev mode runs the launcher against a dev checkout:

* `SMOOVPLAY_PYTHON` and `SMOOVPLAY_APP`: the venv Python and the repo.
* `SMOOVPLAY_CHIAKI` and `SMOOVPLAY_CHIAKI_ARGS`: what "chiaki-ng" is. The fake PS5 stream can stand in for it.
* `SMOOVPLAY_RUN_ARGS`: extra `run.py` arguments.

Other test switches:

* `SMOOVPLAY_SHOTS=<folder>`: saves a PNG of the window on every page or state change. WPF renders it itself, so it's never a screen capture.
* `SMOOVPLAY_RECORD=<folder>`: records the window as video (`frames.mkv`, lossless, encoded on the GPU by ffmpeg on `PATH`) with each frame's time in `frames.txt`. Also rendered by WPF, never a screen capture. `SMOOVPLAY_RECORD_FPS` (default 30) and `SMOOVPLAY_RECORD_SCALE` (default 1.5) tune it. The frames arrive unevenly, so resample a recording to an even frame rate before editing it.
* `SMOOVPLAY_SETTINGS=<file>`: uses this settings file instead of the real one, to show chosen settings without changing yours. Give `run.py` the same file (`--settings` in `SMOOVPLAY_RUN_ARGS`), or the Settings page and the pipeline read different files.
* `SMOOVPLAY_PREFS=<file>`: keeps the guide's ticks out of the real profile.
* `SMOOVPLAY_SOURCES=all`: also offers the video sources that aren't ready yet (`python -m framegen.sources`), with a picker on the Play page (`source-<kind>`), a setup guide for each, and a PC check card for a capture card in place of the home network. `SMOOVPLAY_CAPTURE=<index or video file>` is the capture card (a file stands in for one, played in real time through the capture path), and `SMOOVPLAY_SOURCE_FILE=<video file>` is the "Video file" source.
* `FRAMEGEN_VFX_PATH=<folder>`: where NVIDIA VFG goes (a dev launcher only; an install always uses `runtime\vfx`). On the Settings page, under Interpolation model, **Set up NVIDIA VFG** shows NVIDIA's licence first, then runs `python -m framegen.vfg_setup --install --accept-nvidia-licence` into that folder (NVIDIA's pinned wheel from pypi.nvidia.com, output in `logs\vfg-setup.txt`) and picks it; **Remove** deletes it (only a folder with SmoovPlay's marker file). The NVIDIA VFG choice is greyed out until it's set up. Copy diagnostics leaves out a test report that includes NVIDIA VFG, because NVIDIA's licence forbids publishing its results.
* `SMOOVPLAY_DETECT_ARGS=--timing 9.3,6.8`: Auto-detect uses these times (full, half motion detail, in ms) instead of timing the AI model, so UI tests don't load the graphics card.
* `SMOOVPLAY_FAKE_CHECKS=vc,driver,display,lan,net,longpaths`: shows those checks as problems, to see the fix-it UI on a PC that passes. The fixes still act for real.
* `SMOOVPLAY_TRACE=<file>`: developer tracing.
* `SMOOVPLAY_SHOW_RESULTS=<results .txt>`: opens with that test report on the Play page, to see the results card without playing.

Every named element has a UI Automation id, so tests can drive the window: `Invoke` on buttons, `SelectionItem` on the rail, `Value` on the folder box. On the Settings page, switches are `Set_<key>` (`Toggle`) and dropdowns are `Set_<key>` (`ExpandCollapse`). A row of two to four choices is `Set_<key>`, and each choice `Set_<key>_<value>` (`SelectionItem`, its hint as `HelpText`); the frame-rate pick at the top is `Set_hz`, with `Set_hz_off` for frame generation off. A section's header is `SetFold_<group>` without spaces (`Invoke`; `ItemStatus` says open or closed); a setting that doesn't apply right now (FSR sharpening with RTX VSR picked, say) isn't shown. A dropdown's entries, `Set_<key>_<value>` (`SelectionItem`, e.g. `Set_hz_90`), exist once it has been expanded; look them up under the dropdown, not the window (the popup's own copy has no `SelectionItem`). `Set_<key>_later` is the "may come later" note, which can't be selected. Each entry's hint is its `HelpText`. Auto-detect: `BtnDetect`, `DetectProgress` while it runs, then `Detect_<key>` (a suggestion: "60 fps → 90 fps", or "Auto already set"), `BtnDetectApply` and `BtnDetectDismiss`.

Use a separate tap name so a real chiaki-ng isn't disturbed:

```
set SMOOVPLAY_PYTHON=%CD%\.venv\Scripts\python.exe
set SMOOVPLAY_APP=%CD%
set SMOOVPLAY_CHIAKI=%CD%\.venv\Scripts\python.exe
set SMOOVPLAY_CHIAKI_ARGS=scripts\frametap_fake.py testdata\clip_1080p60.mp4 --seconds 20 --delay 1 --tap Local\sp-test
set SMOOVPLAY_RUN_ARGS=--settings %TEMP%\sp-settings.json --no-minimize-chiaki --chiaki-tap Local\sp-test
set SMOOVPLAY_SETTINGS=%TEMP%\sp-settings.json
installer\dist\SmoovPlay.exe
```

For **Test with a 60 fps game**, add `--native` to the fake's arguments so it sends a 60 fps game.

## Known limits

* **Unsigned:** SmartScreen shows "Windows protected your PC" (More info, then Run anyway) until the exes are code-signed.
* **Size:** the setup is ~140 MB, because chiaki-ng is embedded (its FFmpeg DLL alone is 120 MB). Once the chiaki-ng fork publishes releases, setup can download chiaki-ng too.
* **Updates need a public release:** while the repo is private, installed copies don't see new releases (see Updates).
* **No RTX VSR:** the installed app lacks the RTX Video SDK, so it uses FSR 1.
* **One chiaki-ng at a time:** the launcher only sends video to a chiaki-ng it started itself. If another one is open, it offers to close it.
* **Idle cost:** decorative animations run at 30 fps, and pause while the window is minimised or in the background during a stream.
