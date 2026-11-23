# SmoovPlay on Linux: PS5 Remote Play

> [!NOTE]
> **Experimental.** Tested on one PC: an RTX 5070 with NVIDIA's 615.78 driver, on KDE Plasma (Wayland). The PS5 was
> wired to the router, playing GTA VI. Only the PS5 Remote Play path has been tried. The launcher, capture cards,
> app windows and AMD/Intel graphics are untested on Linux. Everything runs from a terminal.

The PS5 path is the same as on Windows: chiaki-ng with the frame tap ([`chiaki_tap/`](../chiaki_tap/)) copies each
frame the PS5 sends into shared memory, and `run.py --chiaki` decodes it on the graphics card (NVDEC), generates
frames with RIFE on TensorRT and shows them in its OpenGL window. On Linux the tap uses POSIX shared memory
(`/dev/shm/chiaki-frametap`) and a futex in place of Windows' named event. The layout is the same, and so is the
Windows code ([`frametap_posix.h`](../chiaki_tap/frametap_posix.h)).

## Measured

All measured on the same PC, from `run.py`'s own end-of-session report.

| | processing per generated frame (p50 / p95) | PS5 frame arriving → on screen (p50 / p95) |
|---|---|---|
| PS5, GTA VI, 1080p H.265, 92 s | 8.9 / 9.3 ms | 20.6 / 36.7 ms |
| `scripts/frametap_fake.py`, 1080p H.264, 17.7 s | 9.2 / 10.1 ms | 19.4 / 25.6 ms |

Both used RIFE 4.22 lite on TensorRT at 60 fps out. The live session also held chiaki-ng's sound back 24–29 ms to
meet the picture. Both ran on an earlier revision of the Linux tap, which restructured `frametap.c` where this one
shims it; the shared memory, futex and Python reader were the same.

## Set up

On an immutable distribution (tested on Bazzite) use a container. These steps use a Fedora 44 distrobox,
which shares your home folder and, with `--nvidia`, the NVIDIA driver:

```bash
distrobox create --nvidia --name smoovplay --image registry.fedoraproject.org/fedora-toolbox:44
distrobox enter smoovplay
```

1. **chiaki-ng with the tap**, from a chiaki-ng v1.10.1 checkout:
   ```bash
   git clone --recursive -b v1.10.1 https://github.com/streetpea/chiaki-ng.git ~/src/chiaki-ng
   bash chiaki_tap/build_linux.sh ~/src/chiaki-ng ~/src/chiaki-ng-frametap
   ```
   The tap-patched chiaki-ng is a separate build. An installed chiaki-ng (Flatpak or AppImage) can't feed SmoovPlay.
   It keeps its settings in `~/.config/Chiaki/`. A Flatpak chiaki-ng keeps its settings, including a registered PS5,
   in `~/.var/app/io.github.streetpea.Chiaki4deck/config/Chiaki/`, and copying that folder across saves registering
   again.
2. **SmoovPlay's Python packages.** TensorRT's wheels needed Python 3.12 here:
   ```bash
   sudo dnf install python3.12 portaudio
   python3.12 -m venv .venv
   .venv/bin/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
   .venv/bin/pip install numpy opencv-python-headless glfw PyOpenGL av sounddevice onnx onnxscript tensorrt-cu12
   ```
   That's the set tested here (`requirements.txt` lists `opencv-python`, which wasn't tried).

## Play

```bash
CHIAKI_FRAMETAP=1 ~/src/chiaki-ng-frametap/gui/chiaki &
PYGLFW_LIBRARY_VARIANT=wayland .venv/bin/python run.py --chiaki --display
```

In chiaki-ng: 1080p, 60 fps, H.265, then connect. The first start builds the TensorRT engine (about 25–40 s). The keys
are the same as on Windows (**M** menu, **I** frame generation on and off, **Esc** quits).

## Known issues

- **Use native Wayland.** In a container, GLFW hung in `create_window` over XWayland, so `PYGLFW_LIBRARY_VARIANT=wayland`
  is needed on a Wayland desktop. An Ubuntu-based CUDA image fell back to software OpenGL on Wayland because it lacked
  NVIDIA's `libnvidia-egl-gbm`. The Fedora image above works.
- **Black stream, then `Ctrl did not receive session id` in chiaki-ng's log:** no profile is signed in on the PS5.
- chiaki-ng ignores Ctrl+C; close its window, or `kill -9` it.
- Not ported yet: keeping chiaki-ng's window minimised, the screen staying on, MMCSS and GPU priority (`winutil.py`),
  and the DXGI presentation. Each one is skipped on Linux; nothing breaks.
