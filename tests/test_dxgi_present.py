"""DXGI presentation's Windows structures, checked against the C layouts (CPU only: no window, no device)."""

import ctypes
import sys

import pytest

pytestmark = pytest.mark.skipif(sys.platform != "win32", reason="DXGI is Windows-only")

from framegen import dxgi_present as d  # noqa: E402


def test_structures_match_the_windows_sdk():
    assert ctypes.sizeof(d.SwapChainDesc1) == 48  # twelve 4-byte fields
    assert ctypes.sizeof(d.AdapterDesc1) == 312  # 64-bit Python
    assert d.AdapterDesc1.VendorId.offset == 256
    assert d.AdapterDesc1.AdapterLuid.offset == 296 and d.AdapterDesc1.Flags.offset == 304
    assert ctypes.sizeof(d.Texture2DDesc) == 44  # eleven 4-byte fields (SampleDesc is two)
    assert ctypes.sizeof(d.MappedSubresource) == 16 and d.MappedSubresource.RowPitch.offset == 8


def test_guids_are_laid_out_like_windows():
    g = d.GUID.of(d.IID_IDXGIFactory1)  # 770aae78-f26f-4dba-a829-253c83d1b387
    assert bytes(g.data).hex() == "78ae0a776ff2ba4da829253c83d1b387"


def test_hresults():
    assert d.DXGI_ERROR_NOT_FOUND == -2005270526
    with pytest.raises(OSError, match="0x887A0002"):
        d._check(d.DXGI_ERROR_NOT_FOUND, "EnumAdapters1")
    d._check(0x087A0001, "Present")  # DXGI_STATUS_OCCLUDED: a success code


def test_hwnd_from_either_pyglfw_form():
    assert d.hwnd_value(0x1234) == 0x1234
    assert d.hwnd_value(ctypes.c_void_p(0x5678)) == 0x5678


def test_hdr_structures_match_the_windows_sdk():
    assert ctypes.sizeof(d.OutputDesc1) == 152  # 64-bit Python
    assert d.OutputDesc1.Monitor.offset == 88 and d.OutputDesc1.ColorSpace.offset == 100
    assert d.OutputDesc1.MinLuminance.offset == 136
    assert ctypes.sizeof(d.HdrMetadataHdr10) == 28
    m = d.HdrMetadataHdr10.hdr10(1000)
    assert list(m.Primaries) == [35400, 14600, 8500, 39850, 6550, 2300, 15635, 16450]
    assert m.MaxMasteringLuminance == 1000 and m.MaxContentLightLevel == 1000 and m.MaxFrameAverageLightLevel == 400


def test_outputs_can_be_listed():
    outs = d.outputs()  # every display, with whether Windows' HDR is on for it (no window, no device)
    assert all({"hdr", "max_nits", "monitor"} <= set(o) for o in outs)


def test_open_waits_before_initial_check_present(monkeypatch):
    """_open() must consume the waitable object's initial signal before its check Present, else the semaphore count
    becomes 2 instead of 1 and adds an extra frame of latency."""
    from types import SimpleNamespace

    calls = []
    p = d.DxgiPresenter.__new__(d.DxgiPresenter)
    p.GL = SimpleNamespace(glGenTextures=lambda n: 1, glGenFramebuffers=lambda n: 2, glGenRenderbuffers=lambda n: 3)
    p.hwnd, p.format, p.hdr, p.waitable, p.tearing = 0x1234, d.DXGI_FORMAT_B8G8R8A8_UNORM, False, None, False
    p.factory = p.factory2 = p.adapter = p.device = p.context = p.swap_chain = None
    p._open_waitable = lambda: (setattr(p, "waitable", 0xABCD), calls.append("open_waitable"))
    p.resize = lambda w, h: calls.append("resize")
    p.wait = lambda timeout_ms=d.WAIT_MS: (calls.append("wait"), True)[1]
    p.present = lambda interval, check=False, tear=True: calls.append(("present", interval, check))

    fake_dll = SimpleNamespace(
        CreateDXGIFactory1=lambda *a: 0,
        D3D11CreateDevice=SimpleNamespace(argtypes=None, __call__=lambda *a: 0),
    )

    class _D3D:
        class _Create:
            argtypes = None
            def __call__(self, *a): return 0
        D3D11CreateDevice = _Create()

    monkeypatch.setattr(d, "_Wgl", lambda: SimpleNamespace(open_device=lambda dev: 0x99))
    monkeypatch.setattr(d.ctypes, "WinDLL", lambda name: _D3D() if name == "d3d11" else fake_dll)
    monkeypatch.setattr(d, "_nvidia_adapter", lambda factory: 0x11)
    monkeypatch.setattr(d, "_query", lambda obj, iid: 0x22 if iid == d.IID_IDXGIFactory2 else None)
    monkeypatch.setattr(d, "_com", lambda *a, **kw: 0)
    monkeypatch.setattr(d, "_release", lambda obj: None)

    p._open(1920, 1080)
    assert calls.index("open_waitable") < calls.index("wait") < calls.index(("present", 1, True))
