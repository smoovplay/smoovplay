#!/bin/bash
# Build chiaki-ng with the frame tap on Linux. Run in a Fedora 44 container (toolbox or distrobox; on an NVIDIA card,
# create it with distrobox's --nvidia so the driver's Vulkan and OpenGL come in):
#
#   bash chiaki_tap/build_linux.sh <chiaki-ng source> <build folder>
#
# Uses the versions chiaki-ng's own Linux builds pin (libplacebo v7.387.0 in its Flatpak manifest; FFmpeg n9.0.2 with
# its Vulkan patch and nv-codec-headers n13.1.15.0 in scripts/build-ffmpeg.sh), built from source into /usr/local:
# Fedora ships an older libplacebo, and its FFmpeg has no H.264/HEVC. Everything else comes from Fedora. Plus one step: apply_patch.py
# before compiling. Each dependency step is skipped if already done, so re-running only rebuilds chiaki-ng.
# Tested with chiaki-ng v1.10.1 on Fedora 44.
#
# Start it with CHIAKI_FRAMETAP=1 <build folder>/gui/chiaki, then run.py --chiaki (docs/linux.md).
set -euo pipefail

SRC="$(realpath "${1:?usage: build_linux.sh <chiaki-ng source> <build folder>}")"
OUT="$(realpath -m "${2:?usage: build_linux.sh <chiaki-ng source> <build folder>}")"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPS="$OUT/deps"
export PKG_CONFIG_PATH="/usr/local/lib64/pkgconfig:/usr/local/lib/pkgconfig${PKG_CONFIG_PATH:+:$PKG_CONFIG_PATH}"

step() { echo; echo "=== $*"; }

step "packages"
sudo dnf install -y --setopt=install_weak_deps=False \
    gcc gcc-c++ make cmake ninja-build meson git pkgconf-pkg-config nasm patch \
    python3 python3-protobuf python3-jinja2 protobuf-compiler \
    qt6-qtbase-devel qt6-qtbase-private-devel qt6-qtdeclarative-devel qt6-qtsvg-devel qt6-qtwebengine-devel \
    qt6-qtwebchannel-devel qt6-qtpositioning-devel qt6-qtwayland \
    opus-devel openssl-devel zlib-devel libidn2-devel SDL3-devel sdl2-compat-devel hidapi-devel fftw-devel \
    speexdsp-devel json-c-devel miniupnpc-devel libevent-devel libevdev-devel systemd-devel libdrm-devel \
    libva-devel vulkan-headers vulkan-loader-devel libshaderc-devel glslang-devel spirv-tools-devel lcms2-devel \
    libxkbcommon-devel mesa-libGL-devel mesa-libEGL-devel libunwind-devel
echo /usr/local/lib64 | sudo tee /etc/ld.so.conf.d/usr-local.conf >/dev/null
mkdir -p "$DEPS"

if ! pkg-config --exact-version=7.387.0 libplacebo 2>/dev/null; then
    step "libplacebo v7.387.0"
    [ -d "$DEPS/libplacebo" ] || git clone --recursive -b v7.387.0 https://github.com/haasn/libplacebo.git "$DEPS/libplacebo"
    (cd "$DEPS/libplacebo" && meson setup --wipe --prefix /usr/local -Dxxhash=disabled -Dvulkan=enabled \
        -Dshaderc=enabled build && ninja -C build && sudo ninja -C build install)
fi

if [ ! -f /usr/local/lib64/pkgconfig/ffnvcodec.pc ] && [ ! -f /usr/local/lib/pkgconfig/ffnvcodec.pc ]; then
    step "nv-codec-headers n13.1.15.0"
    [ -d "$DEPS/nv-codec-headers" ] || git clone -b n13.1.15.0 https://github.com/FFmpeg/nv-codec-headers.git \
        "$DEPS/nv-codec-headers"
    (cd "$DEPS/nv-codec-headers" && sudo make PREFIX=/usr/local install)
fi

if [ ! -f "$DEPS/ffmpeg/.installed" ]; then
    step "FFmpeg n9.0.2"
    if [ ! -d "$DEPS/ffmpeg" ]; then
        git clone --depth 1 -b n9.0.2 https://github.com/FFmpeg/FFmpeg.git "$DEPS/ffmpeg"
        (cd "$DEPS/ffmpeg" && git apply "$SRC/scripts/flatpak/0001-vulkan-ignore-frames-without-hw-context.patch")
    fi
    (cd "$DEPS/ffmpeg" && ./configure --prefix=/usr/local --disable-all --enable-static \
        --enable-avformat --enable-avcodec --enable-vulkan --enable-libdrm \
        --enable-decoder=h264 --enable-decoder=hevc \
        --enable-hwaccel=h264_vaapi --enable-hwaccel=hevc_vaapi --enable-hwaccel=h264_vulkan \
        --enable-hwaccel=hevc_vulkan --enable-decoder=h264_cuvid --enable-decoder=hevc_cuvid \
        --enable-hwaccel=h264_nvdec --enable-hwaccel=hevc_nvdec --enable-cuvid --enable-nvdec --enable-ffnvcodec &&
        make -j"$(nproc)" && sudo make install && touch .installed)
fi
sudo ldconfig

step "frame tap"
python3 "$HERE/apply_patch.py" "$SRC"

step "chiaki-ng"
(cd "$SRC" && git submodule update --init --recursive && rm -f third-party/nanopb/generator/proto/nanopb_pb2.py)
# --exclude-libs: Fedora's QtWebEngine loads the system FFmpeg; without it, chiaki-ng exports its static FFmpeg 9's
# av_* symbols (and the bundled curl's) and WebEngine's calls would bind to them.
cmake -S "$SRC" -B "$OUT" -GNinja -DCMAKE_BUILD_TYPE=Release -DCMAKE_PREFIX_PATH=/usr/local \
    -DCHIAKI_ENABLE_TESTS=ON -DCHIAKI_ENABLE_GUI=ON -DCHIAKI_GUI_ENABLE_SDL_GAMECONTROLLER=ON \
    -DCMAKE_EXE_LINKER_FLAGS="-Wl,--exclude-libs,ALL"
ninja -C "$OUT"
"$OUT/test/chiaki-unit"
echo; echo "built: $OUT/gui/chiaki (start it with CHIAKI_FRAMETAP=1)"
