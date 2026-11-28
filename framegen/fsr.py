"""AMD FidelityFX Super Resolution 1 (FSR 1) for the display window.

Two OpenGL fragment-shader passes where the frame is drawn to the screen:
EASU (edge-adaptive upscale to the window size) and RCAS (sharpening).
Because it runs at draw time it scales real and generated frames alike, and it doesn't touch the
interpolation's CUDA work. The shader code is AMD's own (framegen/shaders/fsr1, MIT licence).
RCAS also runs on its own (FsrScaler.sharpen): Settings "Sharpening" after RTX VSR, bilinear or no upscaling.
"""
from __future__ import annotations

from pathlib import Path

SHADER_DIR = Path(__file__).resolve().parent / "shaders" / "fsr1"

_VERT = """#version 450
void main() {  // one triangle covering the viewport
    vec2 p = vec2((gl_VertexID << 1) & 2, gl_VertexID & 2);
    gl_Position = vec4(p * 2.0 - 1.0, 0.0, 1.0);
}
"""

_EASU = """
layout(binding = 0) uniform sampler2D src;
uniform vec2 in_size;   // source image, pixels
uniform vec2 out_size;  // upscaled image, pixels
out vec4 color;
AF4 FsrEasuRF(AF2 p) { return textureGather(src, p, 0); }
AF4 FsrEasuGF(AF2 p) { return textureGather(src, p, 1); }
AF4 FsrEasuBF(AF2 p) { return textureGather(src, p, 2); }
void main() {
    AU4 c0, c1, c2, c3;
    FsrEasuCon(c0, c1, c2, c3, in_size.x, in_size.y, in_size.x, in_size.y, out_size.x, out_size.y);
    AF3 c;
    FsrEasuF(c, AU2(gl_FragCoord.xy), c0, c1, c2, c3);
    color = vec4(c, 1.0);
}
"""

# Images keep row 0 at the top (texture row 0); the window's row 0 is the bottom, so RCAS flips while it draws
# (flip 1). A frame read back from the graphics card (an instant replay clip's) keeps row 0 at the top (flip 0).
_RCAS = """
layout(binding = 0) uniform sampler2D mid;
uniform ivec2 origin;     // window pixel of the image's bottom-left corner
uniform float sharpness;  // stops below maximum sharpness: 0 = max, each +1 halves it
uniform int flip;         // 1: upright on the screen; 0: in the image's own row order
out vec4 color;
AF4 FsrRcasLoadF(ASU2 p) { return texelFetch(mid, clamp(p, ASU2(0), textureSize(mid, 0) - 1), 0); }
void FsrRcasInputF(inout AF1 r, inout AF1 g, inout AF1 b) {}
void main() {
    ivec2 q = ivec2(gl_FragCoord.xy) - origin;
    AU2 ip = AU2(q.x, flip != 0 ? textureSize(mid, 0).y - 1 - q.y : q.y);
    AU4 con;
    FsrRcasCon(con, sharpness);
    AF1 r, g, b;
    FsrRcasF(r, g, b, ip, con);
    color = vec4(r, g, b, 1.0);
}
"""

# EASU's picture as it is, with Sharpening off: RCAS's flip and placement, and no sharpening.
_COPY = """#version 450
layout(binding = 0) uniform sampler2D mid;
uniform ivec2 origin;
uniform int flip;
out vec4 color;
void main() {
    ivec2 q = ivec2(gl_FragCoord.xy) - origin;
    color = vec4(texelFetch(mid, ivec2(q.x, flip != 0 ? textureSize(mid, 0).y - 1 - q.y : q.y), 0).rgb, 1.0);
}
"""


def _source(defines: str, body: str) -> str:
    a = (SHADER_DIR / "ffx_a.h").read_text(encoding="utf-8")
    f = (SHADER_DIR / "ffx_fsr1.h").read_text(encoding="utf-8")
    return "#version 450\n#define A_GPU 1\n#define A_GLSL 1\n" + a + defines + f + body


class FsrScaler:
    """FSR 1 upscaling of a GL texture into a rectangle of a framebuffer, and RCAS sharpening on its own. Needs a
    current GL 4.5 context (on a Mac, the 4.1 core one framegen/glsl.py asks for).
    sharpness: RCAS's, in stops below its maximum (settings.SHARPEN_STOPS), or None for none."""

    def __init__(self, GL, sharpness: float | None = 0.5, denoise: bool = True, hdr: bool = False) -> None:
        """hdr: the picture is HDR10 (PQ values, framegen/hdr.py), so EASU's output is kept in half floats rather
        than 8 bits. FSR 1 is made for perceptual values, which PQ's are, as SDR's gamma ones are."""
        self.GL, self.sharpness, self.hdr = GL, sharpness, hdr
        rcas_defs = "#define FSR_RCAS_F 1\n" + ("#define FSR_RCAS_DENOISE 1\n" if denoise else "")
        self.easu = self._program(_source("#define FSR_EASU_F 1\n", _EASU))
        self.rcas = self._program(_source(rcas_defs, _RCAS))
        self.copy = self._program(_COPY)
        self.u_in = GL.glGetUniformLocation(self.easu, "in_size")
        self.u_out = GL.glGetUniformLocation(self.easu, "out_size")
        self.u_origin = GL.glGetUniformLocation(self.rcas, "origin")
        self.u_sharp = GL.glGetUniformLocation(self.rcas, "sharpness")
        self.u_flip = GL.glGetUniformLocation(self.rcas, "flip")
        self.u_copy_origin = GL.glGetUniformLocation(self.copy, "origin")
        self.u_copy_flip = GL.glGetUniformLocation(self.copy, "flip")
        self.vao = GL.glGenVertexArrays(1)
        self.mid_tex = self.mid_fbo = None
        self.mid_size = (0, 0)

    def _program(self, frag: str) -> int:
        from .glsl import program
        return program(self.GL, _VERT, frag)

    def stage(self, w: int, h: int) -> tuple[int, int]:
        """The texture RCAS reads, at w x h, and its framebuffer: for a picture resized another way first (bilinear)
        and then sharpened with sharpen(). Row 0 is the image's top. Shared with draw(), which fills it with EASU."""
        self._mid(w, h)
        return self.mid_tex, self.mid_fbo

    def _mid(self, w: int, h: int) -> None:
        """The EASU output: an RGBA8 (HDR: RGBA16F) texture at the upscaled size, re-made when the window size
        changes."""
        GL = self.GL
        if self.mid_size == (w, h):
            return
        if self.mid_tex is not None:
            GL.glDeleteFramebuffers(1, [self.mid_fbo])
            GL.glDeleteTextures([self.mid_tex])
        self.mid_tex = GL.glGenTextures(1)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.mid_tex)
        for p, v in ((GL.GL_TEXTURE_MIN_FILTER, GL.GL_NEAREST), (GL.GL_TEXTURE_MAG_FILTER, GL.GL_NEAREST),
                     (GL.GL_TEXTURE_WRAP_S, GL.GL_CLAMP_TO_EDGE), (GL.GL_TEXTURE_WRAP_T, GL.GL_CLAMP_TO_EDGE)):
            GL.glTexParameteri(GL.GL_TEXTURE_2D, p, v)
        if self.hdr:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_HALF_FLOAT, None)
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        self.mid_fbo = GL.glGenFramebuffers(1)
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.mid_fbo)
        GL.glFramebufferTexture2D(GL.GL_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.mid_tex, 0)
        self.mid_size = (w, h)

    def draw(self, src_tex: int, sw: int, sh: int, dst_fbo: int, x0: int, y0: int, dw: int, dh: int) -> None:
        """Upscale src_tex (sw x sh, row 0 = image top) into dst_fbo's rectangle at (x0, y0) of size dw x dh,
        upright on screen. Leaves dst_fbo bound as the framebuffer."""
        GL = self.GL
        self._mid(dw, dh)
        GL.glBindVertexArray(self.vao)
        GL.glActiveTexture(GL.GL_TEXTURE0)

        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, self.mid_fbo)
        GL.glViewport(0, 0, dw, dh)
        GL.glUseProgram(self.easu)
        GL.glUniform2f(self.u_in, float(sw), float(sh))
        GL.glUniform2f(self.u_out, float(dw), float(dh))
        GL.glBindTexture(GL.GL_TEXTURE_2D, src_tex)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

        self._finish(self.mid_tex, dst_fbo, x0, y0, dw, dh, self.sharpness, True)

    def sharpen(self, src_tex: int, dst_fbo: int, x0: int, y0: int, dw: int, dh: int, sharpness: float,
                flip: bool = True) -> None:
        """RCAS alone, on a picture that's already its final size (src_tex is dw x dh, row 0 = image top: RTX VSR's
        output, the frame itself, or stage()'s texture), into dst_fbo's rectangle at (x0, y0). flip: upright on the
        screen; False keeps the image's row order (a clip's frame, read back). Leaves dst_fbo bound."""
        self.GL.glBindVertexArray(self.vao)
        self.GL.glActiveTexture(self.GL.GL_TEXTURE0)
        self._finish(src_tex, dst_fbo, x0, y0, dw, dh, sharpness, flip)

    def _finish(self, tex: int, dst_fbo: int, x0: int, y0: int, dw: int, dh: int, sharpness: float | None,
                flip: bool) -> None:
        """tex (dw x dh) into dst_fbo at (x0, y0): sharpened with RCAS, or as it is when sharpness is None."""
        GL = self.GL
        GL.glBindFramebuffer(GL.GL_FRAMEBUFFER, dst_fbo)
        GL.glViewport(x0, y0, dw, dh)
        if sharpness is None:
            GL.glUseProgram(self.copy)
            GL.glUniform2i(self.u_copy_origin, x0, y0)
            GL.glUniform1i(self.u_copy_flip, int(flip))
        else:
            GL.glUseProgram(self.rcas)
            GL.glUniform2i(self.u_origin, x0, y0)
            GL.glUniform1f(self.u_sharp, float(sharpness))
            GL.glUniform1i(self.u_flip, int(flip))
        GL.glBindTexture(GL.GL_TEXTURE_2D, tex)
        GL.glDrawArrays(GL.GL_TRIANGLES, 0, 3)

        GL.glUseProgram(0)
        GL.glBindVertexArray(0)

    def close(self) -> None:
        GL = self.GL
        if self.mid_tex is not None:
            GL.glDeleteFramebuffers(1, [self.mid_fbo])
            GL.glDeleteTextures([self.mid_tex])
        GL.glDeleteProgram(self.easu)
        GL.glDeleteProgram(self.rcas)
        GL.glDeleteProgram(self.copy)
        GL.glDeleteVertexArrays(1, [self.vao])


class GpuTimer:
    """Time GL work with timer queries, read back a few frames later so it never stalls the pipeline."""

    def __init__(self, GL, ring: int = 4) -> None:
        self.GL, self.ring = GL, ring
        self.queries = list(GL.glGenQueries(ring))
        self.pending: list[tuple[int, str]] = []  # (query, label) in flight
        self.ms: dict[str, list[float]] = {}

    def begin(self) -> int | None:
        GL = self.GL
        self._collect()
        if len(self.pending) >= self.ring:
            return None  # all in flight: skip timing this frame
        busy = {q for q, _ in self.pending}
        q = next(q for q in self.queries if q not in busy)
        GL.glBeginQuery(GL.GL_TIME_ELAPSED, q)
        return q

    def end(self, q: int | None, label: str) -> None:
        if q is not None:
            self.GL.glEndQuery(self.GL.GL_TIME_ELAPSED)
            self.pending.append((q, label))

    def _collect(self) -> None:
        GL = self.GL
        while self.pending:
            q, label = self.pending[0]
            if not GL.glGetQueryObjectiv(q, GL.GL_QUERY_RESULT_AVAILABLE):
                return
            ns = GL.glGetQueryObjectuiv(q, GL.GL_QUERY_RESULT)  # 32-bit: fine below 4 s (PyOpenGL's ui64v is broken)
            self.ms.setdefault(label, []).append(int(ns) / 1e6)
            self.pending.pop(0)

    def close(self) -> None:
        self.GL.glDeleteQueries(len(self.queries), self.queries)
