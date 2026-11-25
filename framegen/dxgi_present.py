"""Window presentation through a DXGI flip-model swap chain: settings "present" = "dxgi" (Experimental).

Why: OpenGL's own SwapBuffers in a window goes through the Windows compositor as a copy. PresentMon on an RTX 4070
Ti SUPER (scripts/presentmon_check.ps1) showed "Composed: Copy with GPU GDI": about 12 ms more until a frame is on
screen than fullscreen, 3-4% of frames never shown, and frames snapped to the display's refresh, so no G-Sync or
FreeSync in a window. A flip-model swap chain (what games use) lets Windows hand frames to the display directly
("Independent Flip"), in a window too.

How: WGL_NV_DX_interop2 shares the swap chain's back buffer with OpenGL as a renderbuffer. The window still draws
everything with OpenGL, into an offscreen framebuffer (fbo); present() copies that into the back buffer, flipped
(Direct3D's rows run top-down, OpenGL's bottom-up), and calls IDXGISwapChain::Present. The visible window has no
OpenGL context of its own (GLFW NO_API), so OpenGL and DXGI never both present to it; the caller keeps OpenGL on a
hidden window.

NVIDIA only (the interop extension), with Direct3D 11 on the NVIDIA adapter OpenGL runs on. Any failure while
setting up raises, and the caller falls back to OpenGL's own presentation. COM is called through ctypes; the vtable
slots below are from the Windows SDK headers (dxgi.h, dxgi1_2.h, dxgi1_5.h, d3d11.h).

HDR output (settings "hdr" = "hdr10", framegen/hdr.py): hdr=True makes the swap chain 10-bit (R10G10B10A2) in HDR10's
colour space (PQ, BT.2020: DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020), so Windows shows the PQ values OpenGL draws as
they are, on a display with Windows' HDR on. It refuses (OSError) when the display can't take that colour space, unless
FRAMEGEN_HDR_FORCE=1 (testing on an SDR display: the 10-bit chain is made anyway, and read_back() shows what it holds).
display_hdr() says whether a monitor has Windows' HDR on, and how bright it goes.

Testing: FRAMEGEN_PRESENT_SNAP=<folder> saves the back buffer as Direct3D is about to show it (read back, so it's
what the window gets, upright or not) every SNAP_EVERY presents and after each resize (scripts/dxgi_smoke.py).

How OpenGL's picture gets into the swap chain (FRAMEGEN_DXGI_MODE, for testing):
  copy    (the default) OpenGL draws into a texture of our own (shared with the interop), and Direct3D copies that
          into the back buffer before Present: one extra GPU copy. Direct3D 11's buffer 0 is always the buffer to draw
          into next (it moves round the swap chain's buffers with each Present), so the copy lands where it should.
  direct  OpenGL draws straight into the swap chain's back buffer, shared with the interop. Read back before each
          Present, that held the new frame only every other Present (2 buffers; 1 in 3 with 3), and on the others a
          buffer OpenGL never drew into (black): the interop keeps the buffer that was buffer 0 when it was shared.
          Kept to compare (the fullscreen-stall tests ran both).
  FRAMEGEN_DXGI_BUFFERS=2|3  the swap chain's buffer count.
A lock or Present over 250 ms is logged separately, to show which one waits.
"""
from __future__ import annotations

import ctypes
import os
import sys
import time
import uuid
from ctypes import POINTER, byref, c_char_p, c_int, c_long, c_size_t, c_uint, c_ulong, c_void_p

if sys.platform == "win32":
    from ctypes import WINFUNCTYPE
else:  # importable elsewhere (its structures and the presentation choice are tested on macOS); never called there
    from ctypes import CFUNCTYPE as WINFUNCTYPE

from .display_sync import DisplayTimes

# ---------------------------------------------------------------------------------------------- Windows constants
DXGI_FORMAT_UNKNOWN = 0
DXGI_FORMAT_B8G8R8A8_UNORM = 87
DXGI_FORMAT_R10G10B10A2_UNORM = 24
DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709 = 0
DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020 = 12
DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT = 1
DXGI_HDR_METADATA_TYPE_HDR10 = 1
HDR_FORCE = os.environ.get("FRAMEGEN_HDR_FORCE", "") == "1"  # testing: HDR output on a display without HDR
DXGI_USAGE_RENDER_TARGET_OUTPUT = 0x20
DXGI_SCALING_STRETCH = 0
DXGI_SWAP_EFFECT_FLIP_DISCARD = 4
DXGI_ALPHA_MODE_IGNORE = 3
DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING = 2048
DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT = 0x40
WAIT_OBJECT_0 = 0
WAIT_MS = 100  # longest wait for the swap chain to take a frame: never blocks forever (watchdog, minimised window)
DXGI_PRESENT_ALLOW_TEARING = 0x200
DXGI_FEATURE_PRESENT_ALLOW_TEARING = 0
DXGI_MWA_NO_ALT_ENTER = 2
DXGI_ERROR_NOT_FOUND = ctypes.c_long(0x887A0002).value
D3D_DRIVER_TYPE_UNKNOWN = 0  # an adapter is given
D3D11_CREATE_DEVICE_BGRA_SUPPORT = 0x20
D3D11_SDK_VERSION = 7
NVIDIA = 0x10DE
WGL_ACCESS_WRITE_DISCARD_NV = 0x0002
GL_RENDERBUFFER = 0x8D41
D3D11_USAGE_DEFAULT = 0
D3D11_USAGE_STAGING = 3
D3D11_BIND_SHADER_RESOURCE = 0x8
D3D11_BIND_RENDER_TARGET = 0x20
D3D11_CPU_ACCESS_READ = 0x20000
D3D11_MAP_READ = 1
SNAP_DIR = os.environ.get("FRAMEGEN_PRESENT_SNAP")
SNAP_EVERY, SNAP_MAX = 90, 80
MODE = os.environ.get("FRAMEGEN_DXGI_MODE", "copy")  # see the module docstring
BUFFERS = min(3, max(2, int(os.environ.get("FRAMEGEN_DXGI_BUFFERS", "2") or 2)))
SLOW_MS = 250.0

IID_IDXGIFactory1 = "770aae78-f26f-4dba-a829-253c83d1b387"
IID_IDXGIFactory2 = "50c83a1c-e072-4c48-87b0-3630fa36a6d0"
IID_IDXGIFactory5 = "7632e1f5-ee65-4dca-87fd-84cd75f8838d"
IID_IDXGIDevice1 = "77db970f-6276-48ba-ba28-070143b4392c"
IID_ID3D11Texture2D = "6f15aaf2-d208-4e89-9ab4-489535d34f9c"
IID_IDXGISwapChain2 = "a8be2ac4-199f-4946-b331-79599fb98de7"
IID_IDXGISwapChain3 = "94d99bdb-f1f8-4ab0-b236-7da0170edab1"
IID_IDXGISwapChain4 = "3d585d5a-bd4a-489e-b1f4-3dbcb6452ffb"
IID_IDXGIOutput6 = "068346e8-aaec-4b84-add7-137f513f77a1"

# vtable slots (0-2 are IUnknown's QueryInterface, AddRef, Release)
QUERY_INTERFACE, RELEASE = 0, 2
FACTORY_ENUM_ADAPTERS1, FACTORY_MAKE_WINDOW_ASSOCIATION = 12, 8
ADAPTER_ENUM_OUTPUTS = 7
OUTPUT6_GET_DESC1 = 27
SWAP_CHAIN3_CHECK_COLOR_SPACE_SUPPORT, SWAP_CHAIN3_SET_COLOR_SPACE1 = 37, 38
SWAP_CHAIN4_SET_HDR_METADATA = 40
FACTORY2_CREATE_SWAP_CHAIN_FOR_HWND = 15
FACTORY5_CHECK_FEATURE_SUPPORT = 28
ADAPTER1_GET_DESC1 = 10
DEVICE1_SET_MAXIMUM_FRAME_LATENCY = 12
SWAP_CHAIN_PRESENT, SWAP_CHAIN_GET_BUFFER, SWAP_CHAIN_RESIZE_BUFFERS = 8, 9, 13
SWAP_CHAIN_GET_FRAME_STATISTICS, SWAP_CHAIN_GET_LAST_PRESENT_COUNT = 16, 17
SWAP_CHAIN2_SET_MAXIMUM_FRAME_LATENCY, SWAP_CHAIN2_GET_FRAME_LATENCY_WAITABLE_OBJECT = 31, 33
DEVICE_CREATE_TEXTURE2D = 5
CONTEXT_MAP, CONTEXT_UNMAP, CONTEXT_COPY_RESOURCE = 14, 15, 47


class GUID(ctypes.Structure):
    _fields_ = [("data", ctypes.c_ubyte * 16)]

    @classmethod
    def of(cls, text: str) -> "GUID":
        return cls.from_buffer_copy(uuid.UUID(text).bytes_le)


class SwapChainDesc1(ctypes.Structure):  # DXGI_SWAP_CHAIN_DESC1
    _fields_ = [("Width", c_uint), ("Height", c_uint), ("Format", c_uint), ("Stereo", c_int),
                ("SampleCount", c_uint), ("SampleQuality", c_uint), ("BufferUsage", c_uint), ("BufferCount", c_uint),
                ("Scaling", c_uint), ("SwapEffect", c_uint), ("AlphaMode", c_uint), ("Flags", c_uint)]


class FrameStatistics(ctypes.Structure):  # DXGI_FRAME_STATISTICS
    _fields_ = [("PresentCount", c_uint), ("PresentRefreshCount", c_uint), ("SyncRefreshCount", c_uint),
                ("SyncQPCTime", ctypes.c_int64), ("SyncGPUTime", ctypes.c_int64)]


class Texture2DDesc(ctypes.Structure):  # D3D11_TEXTURE2D_DESC
    _fields_ = [("Width", c_uint), ("Height", c_uint), ("MipLevels", c_uint), ("ArraySize", c_uint), ("Format", c_uint),
                ("SampleCount", c_uint), ("SampleQuality", c_uint), ("Usage", c_uint), ("BindFlags", c_uint),
                ("CPUAccessFlags", c_uint), ("MiscFlags", c_uint)]


class MappedSubresource(ctypes.Structure):  # D3D11_MAPPED_SUBRESOURCE
    _fields_ = [("pData", c_void_p), ("RowPitch", c_uint), ("DepthPitch", c_uint)]


class OutputDesc1(ctypes.Structure):  # DXGI_OUTPUT_DESC1
    _fields_ = [("DeviceName", ctypes.c_wchar * 32), ("Left", c_long), ("Top", c_long), ("Right", c_long),
                ("Bottom", c_long), ("AttachedToDesktop", c_int), ("Rotation", c_uint), ("Monitor", c_void_p),
                ("BitsPerColor", c_uint), ("ColorSpace", c_uint), ("Primaries", ctypes.c_float * 8),
                ("MinLuminance", ctypes.c_float), ("MaxLuminance", ctypes.c_float),
                ("MaxFullFrameLuminance", ctypes.c_float)]


class HdrMetadataHdr10(ctypes.Structure):  # DXGI_HDR_METADATA_HDR10
    _fields_ = [("Primaries", ctypes.c_uint16 * 8), ("MaxMasteringLuminance", c_uint),
                ("MinMasteringLuminance", c_uint), ("MaxContentLightLevel", ctypes.c_uint16),
                ("MaxFrameAverageLightLevel", ctypes.c_uint16)]

    @classmethod
    def hdr10(cls, peak_nits: float, average_nits: float = 400.0) -> "HdrMetadataHdr10":
        """BT.2020 primaries and D65 white (in 0.00002 units), mastered up to peak_nits (whole nits; black in
        0.0001-nit units)."""
        xy = (0.708, 0.292, 0.170, 0.797, 0.131, 0.046, 0.3127, 0.3290)
        peak = int(min(max(peak_nits, 1), 10000))
        return cls((ctypes.c_uint16 * 8)(*(round(v * 50000) for v in xy)), peak, 1,
                   min(peak, 65535), int(min(average_nits, peak)))


class Luid(ctypes.Structure):
    _fields_ = [("LowPart", c_uint), ("HighPart", c_long)]


class AdapterDesc1(ctypes.Structure):  # DXGI_ADAPTER_DESC1
    _fields_ = [("Description", ctypes.c_wchar * 128), ("VendorId", c_uint), ("DeviceId", c_uint),
                ("SubSysId", c_uint), ("Revision", c_uint), ("DedicatedVideoMemory", c_size_t),
                ("DedicatedSystemMemory", c_size_t), ("SharedSystemMemory", c_size_t), ("AdapterLuid", Luid),
                ("Flags", c_uint)]


def _com(obj: c_void_p, slot: int, argtypes: list, *args, restype=c_long):
    """Calls method `slot` of a COM object. HRESULTs come back as numbers (negative: failed)."""
    vtbl = ctypes.cast(obj, POINTER(POINTER(c_void_p)))[0]
    return WINFUNCTYPE(restype, c_void_p, *argtypes)(vtbl[slot])(obj, *args)


def _check(hr: int, what: str) -> None:
    if hr < 0:
        raise OSError(f"{what} failed (HRESULT 0x{hr & 0xFFFFFFFF:08X})")


def _release(obj: c_void_p | None) -> None:
    if obj:
        _com(obj, RELEASE, [], restype=c_ulong)


def _query(obj: c_void_p, iid: str) -> c_void_p | None:
    out = c_void_p()
    return out if _com(obj, QUERY_INTERFACE, [POINTER(GUID), POINTER(c_void_p)], byref(GUID.of(iid)),
                       byref(out)) >= 0 else None


class _Wgl:
    """The WGL_NV_DX_interop2 functions, from the current OpenGL context."""

    def __init__(self) -> None:
        gl = ctypes.WinDLL("opengl32")
        gl.wglGetProcAddress.restype, gl.wglGetProcAddress.argtypes = c_void_p, [c_char_p]
        gl.wglGetCurrentDC.restype = c_void_p

        def fn(name: str, restype, *argtypes):
            p = gl.wglGetProcAddress(name.encode())
            if not p:
                raise OSError(f"{name} isn't available")
            return WINFUNCTYPE(restype, *argtypes)(p)

        ext = fn("wglGetExtensionsStringARB", c_char_p, c_void_p)(gl.wglGetCurrentDC()) or b""
        if b"WGL_NV_DX_interop2" not in ext.split():
            raise OSError("the graphics driver doesn't offer WGL_NV_DX_interop2")
        HANDLE = c_void_p
        self.open_device = fn("wglDXOpenDeviceNV", HANDLE, c_void_p)
        self.close_device = fn("wglDXCloseDeviceNV", c_int, HANDLE)
        self.register = fn("wglDXRegisterObjectNV", HANDLE, HANDLE, c_void_p, c_uint, c_uint, c_uint)
        self.unregister = fn("wglDXUnregisterObjectNV", c_int, HANDLE, HANDLE)
        self.lock = fn("wglDXLockObjectsNV", c_int, HANDLE, c_int, POINTER(HANDLE))
        self.unlock = fn("wglDXUnlockObjectsNV", c_int, HANDLE, c_int, POINTER(HANDLE))


def _nvidia_adapter(factory: c_void_p) -> c_void_p:
    """The first NVIDIA adapter: the interop works only with a Direct3D device on the GPU OpenGL runs on (on a
    laptop the default adapter is often the integrated one)."""
    i = 0
    while True:
        adapter = c_void_p()
        hr = _com(factory, FACTORY_ENUM_ADAPTERS1, [c_uint, POINTER(c_void_p)], i, byref(adapter))
        if hr == DXGI_ERROR_NOT_FOUND:
            raise OSError("no NVIDIA graphics card found by DXGI")
        _check(hr, "EnumAdapters1")
        desc = AdapterDesc1()
        if _com(adapter, ADAPTER1_GET_DESC1, [POINTER(AdapterDesc1)], byref(desc)) >= 0 and desc.VendorId == NVIDIA:
            return adapter
        _release(adapter)
        i += 1


def outputs() -> list[dict]:
    """Every display Windows has, from DXGI: {"name", "monitor" (HMONITOR), "hdr" (Windows' HDR is on for it),
    "bits", "max_nits", "min_nits", "full_frame_nits", "rect" (left, top, right, bottom)}. [] if DXGI can't say."""
    found = []
    try:
        dxgi = ctypes.WinDLL("dxgi")
        factory = c_void_p()
        _check(dxgi.CreateDXGIFactory1(byref(GUID.of(IID_IDXGIFactory1)), byref(factory)), "CreateDXGIFactory1")
    except OSError:
        return found
    try:
        i = 0
        while True:
            adapter = c_void_p()
            if _com(factory, FACTORY_ENUM_ADAPTERS1, [c_uint, POINTER(c_void_p)], i, byref(adapter)) < 0:
                break
            try:
                j = 0
                while True:
                    out = c_void_p()
                    if _com(adapter, ADAPTER_ENUM_OUTPUTS, [c_uint, POINTER(c_void_p)], j, byref(out)) < 0:
                        break
                    out6 = _query(out, IID_IDXGIOutput6)
                    _release(out)
                    if out6 is not None:
                        d = OutputDesc1()
                        if _com(out6, OUTPUT6_GET_DESC1, [POINTER(OutputDesc1)], byref(d)) >= 0:
                            found.append({"name": d.DeviceName, "monitor": d.Monitor or 0,
                                          "hdr": d.ColorSpace == DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020,
                                          "bits": d.BitsPerColor, "max_nits": float(d.MaxLuminance),
                                          "min_nits": float(d.MinLuminance),
                                          "full_frame_nits": float(d.MaxFullFrameLuminance),
                                          "rect": (d.Left, d.Top, d.Right, d.Bottom)})
                        _release(out6)
                    j += 1
            finally:
                _release(adapter)
            i += 1
    finally:
        _release(factory)
    return found


class _Point(ctypes.Structure):
    _fields_ = [("x", c_long), ("y", c_long)]


def display_hdr(hwnd=None) -> dict | None:
    """The display a window is on (most of it), or without one the main display, from outputs(); None if unknown."""
    u32 = ctypes.WinDLL("user32")
    if hwnd is not None:
        u32.MonitorFromWindow.argtypes, u32.MonitorFromWindow.restype = [c_void_p, c_uint], c_void_p
        mon = u32.MonitorFromWindow(c_void_p(hwnd_value(hwnd)), 2)  # MONITOR_DEFAULTTONEAREST
    else:
        u32.MonitorFromPoint.argtypes, u32.MonitorFromPoint.restype = [_Point, c_uint], c_void_p
        mon = u32.MonitorFromPoint(_Point(0, 0), 1)  # MONITOR_DEFAULTTOPRIMARY
    for o in outputs():
        if o["monitor"] == (mon or 0):
            return o
    return None


def _qpc() -> tuple[int, int]:
    k32 = ctypes.WinDLL("kernel32")
    c, f = ctypes.c_int64(), ctypes.c_int64()
    k32.QueryPerformanceCounter(byref(c))
    k32.QueryPerformanceFrequency(byref(f))
    return c.value, f.value or 1


try:
    _QPF = _qpc()[1]
except (OSError, AttributeError):  # (not Windows: the module still imports, for its tests)
    _QPF = 1


def _qpc_offset() -> float:
    """time.perf_counter() minus QueryPerformanceCounter, in seconds: frame statistics' times are QPC ticks."""
    try:
        c, f = _qpc()
        return time.perf_counter() - c / f
    except (OSError, AttributeError):
        return 0.0


def _wait_for(handle, timeout_ms: int) -> bool:
    k32 = ctypes.WinDLL("kernel32")
    k32.WaitForSingleObjectEx.argtypes, k32.WaitForSingleObjectEx.restype = [c_void_p, c_uint, c_int], c_uint
    return k32.WaitForSingleObjectEx(c_void_p(handle), int(timeout_ms), 1) == WAIT_OBJECT_0


def _close_handle(handle) -> None:
    k32 = ctypes.WinDLL("kernel32")
    k32.CloseHandle.argtypes = [c_void_p]
    k32.CloseHandle(c_void_p(handle))


class DxgiPresenter:
    """A flip-model swap chain on `hwnd`, fed by OpenGL. Draw into `fbo` (sized with resize()), then present().

    Needs the OpenGL context that will draw to be current. Raises OSError (or RuntimeError) if anything is missing.
    """

    def __init__(self, GL, hwnd: int, w: int, h: int, hdr: bool = False, peak_nits: float = 1000.0) -> None:
        """hdr: a 10-bit swap chain in HDR10's colour space (see the module docstring); peak_nits goes in its HDR
        metadata, as the brightest the picture gets."""
        self.GL, self.hwnd = GL, hwnd
        self.hdr, self.peak_nits = hdr, peak_nits
        self.format = DXGI_FORMAT_R10G10B10A2_UNORM if hdr else DXGI_FORMAT_B8G8R8A8_UNORM
        self.color_space = DXGI_COLOR_SPACE_RGB_FULL_G22_NONE_P709  # HDR10's once _set_hdr_color_space sets it
        self.w = self.h = 0
        self.factory = self.factory2 = self.adapter = self.device = self.context = self.swap_chain = None
        self.back = self.shared = self.obj = self.hdev = None
        self.tex = self.fbo = self.rb = self.back_fbo = None
        self.presents, self.snaps, self.snap_next = 0, 0, False
        self.waitable = None  # the swap chain's frame latency waitable object (a HANDLE), when DXGI 1.3 gives one
        self.times = DisplayTimes()  # when frames actually reached the screen (GetFrameStatistics)
        self.refresh_hz = 60.0  # the display's, for times between statistics samples (the window sets it)
        self._qpc_offset = _qpc_offset()
        self.mode = MODE if MODE in ("direct", "copy") else "copy"
        try:
            self._open(w, h)
        except BaseException:
            self.close()
            raise

    def _open(self, w: int, h: int) -> None:
        GL = self.GL
        self.wgl = _Wgl()
        dxgi, d3d11 = ctypes.WinDLL("dxgi"), ctypes.WinDLL("d3d11")
        self.factory = c_void_p()
        _check(dxgi.CreateDXGIFactory1(byref(GUID.of(IID_IDXGIFactory1)), byref(self.factory)), "CreateDXGIFactory1")
        self.adapter = _nvidia_adapter(self.factory)
        self.device, self.context, level = c_void_p(), c_void_p(), c_uint()
        d3d11.D3D11CreateDevice.argtypes = [c_void_p, c_uint, c_void_p, c_uint, c_void_p, c_uint, c_uint,
                                            POINTER(c_void_p), POINTER(c_uint), POINTER(c_void_p)]
        _check(d3d11.D3D11CreateDevice(self.adapter, D3D_DRIVER_TYPE_UNKNOWN, None, D3D11_CREATE_DEVICE_BGRA_SUPPORT,
                                       None, 0, D3D11_SDK_VERSION, byref(self.device), byref(level),
                                       byref(self.context)), "D3D11CreateDevice")
        self.factory2 = _query(self.factory, IID_IDXGIFactory2)
        if self.factory2 is None:
            raise OSError("DXGI 1.2 isn't available")
        # Tearing (vsync off, in a window) needs DXGI 1.5 and a driver that allows it; without it, vsync off still
        # presents at once but Windows waits for the next refresh to show it.
        self.tearing = False
        factory5 = _query(self.factory, IID_IDXGIFactory5)
        if factory5 is not None:
            allow = c_int(0)
            if _com(factory5, FACTORY5_CHECK_FEATURE_SUPPORT, [c_uint, c_void_p, c_uint],
                    DXGI_FEATURE_PRESENT_ALLOW_TEARING, byref(allow), ctypes.sizeof(allow)) >= 0:
                self.tearing = bool(allow.value)
            _release(factory5)
        # A frame latency waitable object (DXGI 1.3): wait() returns when the swap chain can take the next frame, so
        # with a sync interval each Present lines up with a refresh and the display, not a timer, spaces the frames.
        tearing = DXGI_SWAP_CHAIN_FLAG_ALLOW_TEARING if self.tearing else 0
        hr = 0
        for self.flags in (tearing | DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT, tearing):
            desc = SwapChainDesc1(Width=max(w, 1), Height=max(h, 1), Format=self.format, SampleCount=1,
                                  BufferUsage=DXGI_USAGE_RENDER_TARGET_OUTPUT, BufferCount=BUFFERS,
                                  Scaling=DXGI_SCALING_STRETCH,
                                  SwapEffect=DXGI_SWAP_EFFECT_FLIP_DISCARD, AlphaMode=DXGI_ALPHA_MODE_IGNORE,
                                  Flags=self.flags)
            self.swap_chain = c_void_p()
            hr = _com(self.factory2, FACTORY2_CREATE_SWAP_CHAIN_FOR_HWND,
                      [c_void_p, c_void_p, POINTER(SwapChainDesc1), c_void_p, c_void_p, POINTER(c_void_p)],
                      self.device, c_void_p(hwnd_value(self.hwnd)), byref(desc), None, None, byref(self.swap_chain))
            if hr >= 0:
                break
        _check(hr, "CreateSwapChainForHwnd")
        if self.flags & DXGI_SWAP_CHAIN_FLAG_FRAME_LATENCY_WAITABLE_OBJECT:
            self._open_waitable()
        _com(self.factory2, FACTORY_MAKE_WINDOW_ASSOCIATION, [c_void_p, c_uint], c_void_p(hwnd_value(self.hwnd)),
             DXGI_MWA_NO_ALT_ENTER)  # Alt+Enter is ours (F), not DXGI's exclusive fullscreen
        if self.hdr:
            self._set_hdr_color_space()
        device1 = _query(self.device, IID_IDXGIDevice1)
        if device1 is not None and not self.waitable:  # at most one frame queued ahead (else the waitable chain's)
            _com(device1, DEVICE1_SET_MAXIMUM_FRAME_LATENCY, [c_uint], 1)
            _release(device1)
        self.hdev = self.wgl.open_device(self.device)
        if not self.hdev:
            raise OSError("wglDXOpenDeviceNV failed")
        self.tex, self.fbo = GL.glGenTextures(1), GL.glGenFramebuffers(1)
        self.rb, self.back_fbo = GL.glGenRenderbuffers(1), GL.glGenFramebuffers(1)
        self.resize(max(w, 1), max(h, 1))
        self.wait()  # consume the waitable object's initial signal so this check Present doesn't leave its count at 2
        self.present(1, check=True)  # fails here, not mid-session, if the driver can't do it

    def _open_waitable(self) -> None:
        """At most one frame queued ahead, and the object that says when the next one can go (else self.waitable
        stays None and Present itself blocks, as before)."""
        sc2 = _query(self.swap_chain, IID_IDXGISwapChain2)
        if sc2 is None:
            return
        try:
            _com(sc2, SWAP_CHAIN2_SET_MAXIMUM_FRAME_LATENCY, [c_uint], 1)
            self.waitable = _com(sc2, SWAP_CHAIN2_GET_FRAME_LATENCY_WAITABLE_OBJECT, [], restype=c_void_p) or None
        finally:
            _release(sc2)

    def wait(self, timeout_ms: int = WAIT_MS) -> bool:
        """Waits until the swap chain can take the next frame (the one before is on screen or about to be): with a
        sync interval, the display's refresh paces the caller. Never longer than timeout_ms. True if it was ready.
        (ctypes lets go of the GIL meanwhile, so the source's reader thread keeps running.)"""
        if not self.waitable:
            return True
        return _wait_for(self.waitable, timeout_ms)

    def _measure(self, t_sent: float) -> None:
        """Notes this Present's number and time, and when the latest frame to reach the screen got there (frame
        statistics; skipped quietly when Windows has none, e.g. the window is hidden or the display changed mode)."""
        count = c_uint(0)
        if _com(self.swap_chain, SWAP_CHAIN_GET_LAST_PRESENT_COUNT, [POINTER(c_uint)], byref(count)) >= 0:
            self.times.sent(count.value, t_sent)
        fs = FrameStatistics()
        if _com(self.swap_chain, SWAP_CHAIN_GET_FRAME_STATISTICS, [POINTER(FrameStatistics)], byref(fs)) < 0:
            return
        if fs.SyncQPCTime > 0 and fs.PresentCount:
            self.times.observe(fs.PresentCount, fs.PresentRefreshCount, fs.SyncRefreshCount,
                               fs.SyncQPCTime / _QPF + self._qpc_offset, 1.0 / max(self.refresh_hz, 1.0))

    def _set_hdr_color_space(self) -> None:
        """HDR10's colour space (PQ, BT.2020) and its metadata on the swap chain. Raises OSError if the display can't
        show it (Windows' HDR off), unless FRAMEGEN_HDR_FORCE=1."""
        sc3 = _query(self.swap_chain, IID_IDXGISwapChain3)
        if sc3 is None:
            raise OSError("DXGI 1.4 (IDXGISwapChain3) isn't available, so no HDR colour space")
        try:
            support = c_uint(0)
            _check(_com(sc3, SWAP_CHAIN3_CHECK_COLOR_SPACE_SUPPORT, [c_uint, POINTER(c_uint)],
                        DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020, byref(support)), "CheckColorSpaceSupport")
            ok = bool(support.value & DXGI_SWAP_CHAIN_COLOR_SPACE_SUPPORT_FLAG_PRESENT)
            if not ok and not HDR_FORCE:
                raise OSError("this display can't show HDR10 (is Use HDR on in Windows' display settings?)")
            hr = _com(sc3, SWAP_CHAIN3_SET_COLOR_SPACE1, [c_uint], DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020)
            if hr >= 0:
                self.color_space = DXGI_COLOR_SPACE_RGB_FULL_G2084_NONE_P2020
            elif not HDR_FORCE:
                _check(hr, "SetColorSpace1")
            else:
                print(f"[display] (testing) HDR10 colour space refused (0x{hr & 0xFFFFFFFF:08X}); 10-bit chain kept")
        finally:
            _release(sc3)
        sc4 = _query(self.swap_chain, IID_IDXGISwapChain4)
        if sc4 is not None:  # optional: lets the display fit the picture to itself
            meta = HdrMetadataHdr10.hdr10(self.peak_nits)
            _com(sc4, SWAP_CHAIN4_SET_HDR_METADATA, [c_uint, c_uint, c_void_p], DXGI_HDR_METADATA_TYPE_HDR10,
                 ctypes.sizeof(meta), byref(meta))
            _release(sc4)

    def _attach(self, w: int, h: int) -> None:
        """Shares a w x h buffer with OpenGL: a texture of our own that present() copies into the back buffer
        ("copy"), or the swap chain's back buffer itself ("direct": see the module docstring). Direct3D 11 keeps
        buffer 0 as the one to draw into next, whichever buffer that is, so self.back lasts until the next resize."""
        GL = self.GL
        self.back = c_void_p()
        _check(_com(self.swap_chain, SWAP_CHAIN_GET_BUFFER, [c_uint, POINTER(GUID), POINTER(c_void_p)], 0,
                    byref(GUID.of(IID_ID3D11Texture2D)), byref(self.back)), "GetBuffer")
        target = self.back
        if self.mode == "copy":
            desc = Texture2DDesc(Width=w, Height=h, MipLevels=1, ArraySize=1, Format=self.format,
                                 SampleCount=1, Usage=D3D11_USAGE_DEFAULT,
                                 BindFlags=D3D11_BIND_RENDER_TARGET | D3D11_BIND_SHADER_RESOURCE)
            self.shared = c_void_p()
            _check(_com(self.device, DEVICE_CREATE_TEXTURE2D, [POINTER(Texture2DDesc), c_void_p, POINTER(c_void_p)],
                        byref(desc), None, byref(self.shared)), "CreateTexture2D (shared)")
            target = self.shared
        self.obj = self.wgl.register(self.hdev, target, int(self.rb), GL_RENDERBUFFER, WGL_ACCESS_WRITE_DISCARD_NV)
        if not self.obj:
            raise OSError("wglDXRegisterObjectNV failed")
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.back_fbo)
        GL.glFramebufferRenderbuffer(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_RENDERBUFFER, self.rb)

    def _detach(self) -> None:
        if self.obj:
            self.wgl.unregister(self.hdev, self.obj)
            self.obj = None
        _release(self.shared)
        _release(self.back)
        self.back = self.shared = None

    def resize(self, w: int, h: int) -> None:
        """Matches the swap chain and `fbo` to the window's size (in pixels); cheap when it hasn't changed."""
        if (w, h) == (self.w, self.h) or w <= 0 or h <= 0:
            return
        GL = self.GL
        self._detach()
        if self.w:  # not the first time: the buffers were made at the old size
            _check(_com(self.swap_chain, SWAP_CHAIN_RESIZE_BUFFERS, [c_uint, c_uint, c_uint, c_uint, c_uint], 0, w, h,
                        DXGI_FORMAT_UNKNOWN, self.flags), "ResizeBuffers")
        self._attach(w, h)
        GL.glBindTexture(GL.GL_TEXTURE_2D, self.tex)
        if self.hdr:  # half floats until the last copy, so the 10 bits on screen are the picture's own
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA16F, w, h, 0, GL.GL_RGBA, GL.GL_HALF_FLOAT, None)
        else:
            GL.glTexImage2D(GL.GL_TEXTURE_2D, 0, GL.GL_RGBA8, w, h, 0, GL.GL_RGBA, GL.GL_UNSIGNED_BYTE, None)
        GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.fbo)
        GL.glFramebufferTexture2D(GL.GL_DRAW_FRAMEBUFFER, GL.GL_COLOR_ATTACHMENT0, GL.GL_TEXTURE_2D, self.tex, 0)
        GL.glClear(GL.GL_COLOR_BUFFER_BIT)  # black until the first frame, not whatever the memory held
        self.snap_next = self.w > 0  # (testing) the first frame at the new size, after it's drawn
        self.w, self.h = w, h

    def present(self, interval: int, check: bool = False, tear: bool = True) -> None:
        """Copies `fbo` into the back buffer (upright) and shows it. interval: the sync interval, the refreshes this
        frame waits for (and stays up for) after the one before (framegen/display_sync.py); 0: at once, tearing if
        allowed (and tear: the caller may turn it off, e.g. in a window that never tore before). Call wait() first."""
        interval = max(0, min(4, int(interval)))
        GL, handles = self.GL, (c_void_p * 1)(self.obj)
        t0 = time.perf_counter()
        if not self.wgl.lock(self.hdev, 1, handles):
            raise OSError("wglDXLockObjectsNV failed")
        self._say_if_slow("lock", t0)
        try:
            GL.glBindFramebuffer(GL.GL_READ_FRAMEBUFFER, self.fbo)
            GL.glBindFramebuffer(GL.GL_DRAW_FRAMEBUFFER, self.back_fbo)
            if check and GL.glCheckFramebufferStatus(GL.GL_DRAW_FRAMEBUFFER) != GL.GL_FRAMEBUFFER_COMPLETE:
                raise OSError("the shared back buffer can't be drawn into")
            GL.glBlitFramebuffer(0, 0, self.w, self.h, 0, self.h, self.w, 0, GL.GL_COLOR_BUFFER_BIT, GL.GL_NEAREST)
        finally:
            t0 = time.perf_counter()
            self.wgl.unlock(self.hdev, 1, handles)
            self._say_if_slow("unlock", t0)
        if self.shared is not None:  # "copy": our texture into the back buffer, on the GPU (after OpenGL's blit)
            _com(self.context, CONTEXT_COPY_RESOURCE, [c_void_p, c_void_p], self.back, self.shared, restype=None)
        self.presents += 1
        if SNAP_DIR and not check and self.snaps < SNAP_MAX and (self.snap_next or self.presents % SNAP_EVERY == 0):
            self._snap(interval)
        # (ALLOW_TEARING with a sync interval is an invalid call)
        flags = DXGI_PRESENT_ALLOW_TEARING if interval == 0 and self.tearing and tear else 0
        t0 = time.perf_counter()
        hr = _com(self.swap_chain, SWAP_CHAIN_PRESENT, [c_uint, c_uint], interval, flags)
        self._say_if_slow("Present call", t0)
        _check(hr, "Present")
        if not check:
            self._measure(t0)

    def _say_if_slow(self, what: str, t0: float) -> None:
        ms = (time.perf_counter() - t0) * 1000
        if ms > SLOW_MS:
            print(f"[display] DXGI {what} took {ms:.0f} ms ({self.mode}, {BUFFERS} buffers, {self.w}x{self.h})")

    def read_back(self):
        """The back buffer as Direct3D will show it (rows top-down), as BGR: uint8, or with hdr uint16 10-bit values
        (0..1023). Waits for the GPU: for testing only, between drawing and Present."""
        import numpy as np
        desc = Texture2DDesc(Width=self.w, Height=self.h, MipLevels=1, ArraySize=1, Format=self.format,
                             SampleCount=1, Usage=D3D11_USAGE_STAGING, CPUAccessFlags=D3D11_CPU_ACCESS_READ)
        staging = c_void_p()
        _check(_com(self.device, DEVICE_CREATE_TEXTURE2D, [POINTER(Texture2DDesc), c_void_p, POINTER(c_void_p)],
                    byref(desc), None, byref(staging)), "CreateTexture2D")
        try:
            _com(self.context, CONTEXT_COPY_RESOURCE, [c_void_p, c_void_p], staging, self.back, restype=None)
            m = MappedSubresource()
            _check(_com(self.context, CONTEXT_MAP, [c_void_p, c_uint, c_uint, c_uint, POINTER(MappedSubresource)],
                        staging, 0, D3D11_MAP_READ, 0, byref(m)), "Map")
            try:
                buf = (ctypes.c_ubyte * (m.RowPitch * self.h)).from_address(m.pData)
                if self.hdr:  # R in bits 0-9, G 10-19, B 20-29
                    v = np.frombuffer(buf, np.uint32).reshape(self.h, m.RowPitch // 4)[:, :self.w]
                    return np.stack([(v >> 20) & 1023, (v >> 10) & 1023, v & 1023], -1).astype(np.uint16)
                return np.frombuffer(buf, np.uint8).reshape(self.h, m.RowPitch // 4, 4)[:, :self.w, :3].copy()
            finally:
                _com(self.context, CONTEXT_UNMAP, [c_void_p, c_uint], staging, 0, restype=None)
        finally:
            _release(staging)

    def _snap(self, interval: int) -> None:
        from .imagefile import write_image
        self.snap_next = False
        try:
            os.makedirs(SNAP_DIR, exist_ok=True)
            name = f"dxgi_{self.presents:06d}_{self.w}x{self.h}_sync-{interval}.png"
            img = self.read_back()
            write_image(os.path.join(SNAP_DIR, name), img << 6 if self.hdr else img)  # (HDR: a 16-bit PNG of PQ)
            self.snaps += 1
        except Exception as e:  # noqa: BLE001 - testing aid: say so, keep presenting
            print(f"[display] couldn't save the back buffer: {e}")
            self.snaps = SNAP_MAX

    def close(self) -> None:
        GL = self.GL
        if self.waitable:
            _close_handle(self.waitable)
            self.waitable = None
        try:
            self._detach()
            if self.hdev:
                self.wgl.close_device(self.hdev)
                self.hdev = None
        except Exception:  # noqa: BLE001 - closing: free what we can
            pass
        for fb in (self.fbo, self.back_fbo):
            if fb:
                GL.glDeleteFramebuffers(1, [fb])
        if self.rb:
            GL.glDeleteRenderbuffers(1, [self.rb])
        if self.tex:
            GL.glDeleteTextures([self.tex])
        self.fbo = self.back_fbo = self.rb = self.tex = None
        for name in ("swap_chain", "factory2", "context", "device", "adapter", "factory"):
            _release(getattr(self, name))
            setattr(self, name, None)


def hwnd_value(hwnd) -> int:
    """glfw.get_win32_window gives an int or a ctypes pointer, depending on the pyGLFW version."""
    if isinstance(hwnd, int):
        return hwnd
    return ctypes.cast(hwnd, c_void_p).value or 0
