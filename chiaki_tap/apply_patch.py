"""Add the frame tap to a chiaki-ng source checkout (tested against v1.10.1).

    python chiaki_tap/apply_patch.py <chiaki-ng source folder>

Copies the tap into gui/ (frametap.c/.h: the shared memory; frametap_session.cpp/.h: what it does inside chiaki-ng's
StreamSession) and makes one-line edits to chiaki-ng's own files: six calls in gui/src/streamsession.cpp, one
friend line in gui/include/streamsession.h (the tap reaches the session's sound output and controller state), the
new files in gui/CMakeLists.txt, and one check in gui/src/qmlmainwindow.cpp (CHIAKI_FRAMETAP_RENDERER=opengl: OpenGL
for that session, which SmoovPlay asks for on AMD and Intel graphics, where chiaki-ng's Vulkan has crashed). Each
edit is an exact-text replacement that must match once, so a chiaki-ng change that moves the code fails loudly here
instead of building something subtly wrong; with one-line edits, a new chiaki-ng is quick to fix up. Running it twice is a no-op. A source patched by an older version (with the tap's
code inside streamsession.cpp) has to be restored first:
git checkout -- gui/src/streamsession.cpp gui/CMakeLists.txt

With the tap built in, chiaki-ng behaves exactly as before unless CHIAKI_FRAMETAP=1 is set. With it set, besides
copying frames out, it presses buttons when SmoovPlay's delay test asks, holds the sound back by SmoovPlay's
picture delay when SmoovPlay asks (both through the shared memory; see frametap.h), and copies the game's sound as
it arrives into a second ring, for SmoovPlay's instant replay.
"""
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
MARK = "FrameTapSession::"  # this version's edits are in
OLD_MARK = "frametap"  # an older version's are (its code was inside streamsession.cpp)
FILES = {"frametap.h": "include", "frametap_posix.h": "include", "frametap_session.h": "include", "frametap.c": "src",
         "frametap_session.cpp": "src"}

EDITS = [
    ("#include <streamsession.h>\n",
     "#include <streamsession.h>\n#include <frametap_session.h>\n"),

    # Stream start: the tap sets its own video callback (copy out, then chiaki-ng's decoder), or chiaki-ng's as before.
    ("\t\tchiaki_session_set_video_sample_cb(&session, chiaki_ffmpeg_decoder_video_sample_cb, ffmpeg_decoder);\n",
     "\t\tif(!FrameTapSession::Begin(this, &session, chiaki_connect_info.video_profile))\n"
     "\t\t\tchiaki_session_set_video_sample_cb(&session, chiaki_ffmpeg_decoder_video_sample_cb, ffmpeg_decoder);\n"),

    ("\tchiaki_session_fini(&session);\n\tchiaki_opus_decoder_fini(&opus_decoder);\n",
     "\tchiaki_session_fini(&session);\n\tFrameTapSession::End();\n\tchiaki_opus_decoder_fini(&opus_decoder);\n"),

    # Sound: SmoovPlay can ask for it to be held back to meet its picture.
    ("\taudio_out_sample_size = sizeof(int16_t) * channels;\n",
     "\taudio_out_sample_size = sizeof(int16_t) * channels;\n\tFrameTapSession::AudioFormat(rate, audio_out_sample_size);\n"),

    ("queue_audio:\n\tQueueAudioOutData(buf);\n",
     "queue_audio:\n\tFrameTapSession::QueueAudio(this, buf);\n"),

    # Instant replay: the game's sound as it arrives, before chiaki-ng's volume (so a quiet chiaki-ng still records).
    ("void StreamSession::PushAudioFrame(int16_t *og_buf, size_t samples_count)\n{\n",
     "void StreamSession::PushAudioFrame(int16_t *og_buf, size_t samples_count)\n{\n"
     "\tFrameTapSession::PushAudio(this, og_buf, samples_count);\n"),
]

HEADER_EDIT = ("\tfriend class StreamSessionPrivate;\n",
               "\tfriend class StreamSessionPrivate;\n\tfriend class FrameTapSession;\n")

# SmoovPlay draws the stream itself, and asks for OpenGL on cards where chiaki-ng's Vulkan has crashed: this session
# only (chiaki-ng's own test switch below it shows a dialog; the saved setting, shared with a standalone chiaki-ng,
# stays as it is).
RENDERER_EDIT = (
    '    if (render_backend == RenderBackend::Vulkan) {\n'
    '        if (qEnvironmentVariableIsSet("CHIAKI_FORCE_VULKAN_FALLBACK")) {\n',
    '    if (render_backend == RenderBackend::Vulkan) {\n'
    '        if (qEnvironmentVariable("CHIAKI_FRAMETAP_RENDERER") == QLatin1String("opengl")) {\n'
    '            // SmoovPlay, which draws the stream itself, asks for OpenGL on cards where Vulkan has crashed: this '
    'session only, no dialog\n'
    '            fallbackToOpenGL(QStringLiteral("SmoovPlay asked for OpenGL (CHIAKI_FRAMETAP_RENDERER)"));\n'
    '            pending_renderer_fallback_reason.clear();\n'
    '            goto renderer_backend_ready;\n'
    '        }\n'
    '        if (qEnvironmentVariableIsSet("CHIAKI_FORCE_VULKAN_FALLBACK")) {\n')

CMAKE_EDIT = ("\tsrc/streamsession.cpp\n",
              "\tsrc/streamsession.cpp\n\tinclude/frametap.h\n\tsrc/frametap.c\n"
              "\tinclude/frametap_session.h\n\tsrc/frametap_session.cpp\n")


def replace_once(text: str, old: str, new: str, where: str) -> str:
    n = text.count(old)
    if n != 1:
        sys.exit(f"{where}: expected the anchor exactly once, found {n}:\n{old}")
    return text.replace(old, new)


def patch_file(path: Path, edits: list[tuple[str, str]], done_marker: str) -> None:
    with open(path, encoding="utf-8", newline="") as f:
        text = f.read()
    if done_marker in text:
        print(f"{path.name}: already patched")
        return
    if path.name in ("streamsession.cpp", "CMakeLists.txt") and OLD_MARK in text:
        sys.exit(f"{path.name} has an older frame tap: restore it first "
                 "(git checkout -- gui/src/streamsession.cpp gui/CMakeLists.txt)")
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    for old, new in edits:
        text = replace_once(text, old, new, path.name)
    with open(path, "w", encoding="utf-8", newline="") as f:
        f.write(text.replace("\n", "\r\n") if crlf else text)
    print(f"{path.name}: patched")


def main() -> int:
    root = Path(sys.argv[1])
    gui = root / "gui"
    session = (gui / "src" / "streamsession.cpp").read_text(encoding="utf-8")
    if OLD_MARK in session and MARK not in session:  # before copying anything in
        sys.exit("streamsession.cpp has an older frame tap: restore it first "
                 "(git checkout -- gui/src/streamsession.cpp gui/CMakeLists.txt)")
    for name, folder in FILES.items():
        shutil.copy(HERE / name, gui / folder / name)
    patch_file(gui / "src" / "streamsession.cpp", EDITS, MARK)
    patch_file(gui / "include" / "streamsession.h", [HEADER_EDIT], "friend class FrameTapSession;")
    patch_file(gui / "CMakeLists.txt", [CMAKE_EDIT], "src/frametap_session.cpp")
    patch_file(gui / "src" / "qmlmainwindow.cpp", [RENDERER_EDIT], "CHIAKI_FRAMETAP_RENDERER")
    return 0


if __name__ == "__main__":
    sys.exit(main())
