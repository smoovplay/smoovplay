"""Tests for framegen/webrtc_tap.py: RFC 6455 framing, RTP loss inference, and end-to-end localhost bridge."""
from __future__ import annotations

import base64
import hashlib
import json
import os
import socket
import struct
import time

import pytest

from framegen.chiaki_tap import FEATURE_AUDIO_DELAY, FrameTap
from framegen.webrtc_tap import (
    FLAG_KEYFRAME,
    WebRtcTapBridge,
    _FRAME_HDR,
    _WS_GUID,
    format_webrtc_stats,
    infer_lost_frames,
    ws_encode_frame,
    ws_recv_message,
)


def _client_handshake(sock: socket.socket) -> None:
    key = base64.b64encode(os.urandom(16)).decode("ascii")
    req = (
        "GET / HTTP/1.1\r\n"
        "Host: 127.0.0.1\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Key: {key}\r\n"
        "Sec-WebSocket-Version: 13\r\n\r\n"
    ).encode("ascii")
    sock.sendall(req)
    raw = bytearray()
    while b"\r\n\r\n" not in raw:
        raw.extend(sock.recv(4096))
    text = raw.decode("iso-8859-1")
    expected = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")
    assert "101 Switching Protocols" in text
    assert f"Sec-WebSocket-Accept: {expected}" in text


def test_infer_lost_frames_detects_gaps_and_handles_wraparound():
    # 60 fps on 90 kHz RTP clock = 1500 ticks per frame
    assert infer_lost_frames(None, 1500, 60) == 0
    assert infer_lost_frames(1500, 3000, 60) == 0
    assert infer_lost_frames(3000, 4800, 60) == 0  # small jitter (< 1.6x)
    assert infer_lost_frames(3000, 6000, 60) == 1  # 1 frame dropped
    assert infer_lost_frames(3000, 9000, 60) == 3  # 3 frames dropped
    # uint32 wraparound: 0xFFFFFFFF - 499 -> 1000 is delta 1500 (no loss)
    assert infer_lost_frames(0xFFFFFFFF - 499, 1000, 60) == 0
    # uint32 wraparound with 1 dropped frame (delta 3000)
    assert infer_lost_frames(0xFFFFFFFF - 499, 2500, 60) == 1
    # Zero timestamps ignored
    assert infer_lost_frames(0, 3000, 60) == 0
    assert infer_lost_frames(3000, 0, 60) == 0


def test_ws_framing_roundtrip_sizes_fragmentation_and_ping():
    import threading
    a, b = socket.socketpair()
    try:
        # Small, medium (>125), and large (>65535) masked binary payloads
        for size in (0, 17, 300, 70000):
            payload = os.urandom(size)
            mask = os.urandom(4)
            frame = ws_encode_frame(0x2, payload, mask=mask)
            t = threading.Thread(target=a.sendall, args=(frame,))
            t.start()
            assert ws_recv_message(b) == (0x2, payload)
            t.join(timeout=2.0)

        # Ping followed by a 2-fragment text message
        mask = b"\x12\x34\x56\x78"
        a.sendall(ws_encode_frame(0x9, b"hi", mask=mask))
        # Fragment 1 (FIN=0, opcode=1) + Fragment 2 (FIN=1, opcode=0)
        frag1 = bytearray(ws_encode_frame(0x1, b"hello ", mask=mask))
        frag1[0] &= 0x7F  # clear FIN
        frag2 = ws_encode_frame(0x0, b"world", mask=mask)
        a.sendall(bytes(frag1) + frag2)
        assert ws_recv_message(b) == (0x1, b"hello world")
        # Server auto-replied with Pong (unmasked 0x8A, len=2, b"hi")
        assert a.recv(16) == ws_encode_frame(0xA, b"hi")
    finally:
        a.close()
        b.close()


def test_webrtc_tap_bridge_end_to_end_and_backchannel():
    tap_name = f"Local\\smoovplay-webrtc-test-{os.getpid()}"
    tap_reader = FrameTap(tap_name, tap_name + "-frame")
    bridge = WebRtcTapBridge(tap_name=tap_name, port=0)
    port = bridge.start()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    sock.settimeout(5.0)
    try:
        _client_handshake(sock)
        mask = b"\xAA\x55\x0F\xF0"

        # 1. Start stream
        start_msg = json.dumps({
            "type": "start", "codec": "h264", "width": 1920, "height": 1080, "fps": 60, "audio_sync": True
        }).encode("utf-8")
        sock.sendall(ws_encode_frame(0x1, start_msg, mask=mask))

        # Wait for tap header to go live
        deadline = time.perf_counter() + 2.0
        while not (h := tap_reader.header()).live and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert h.live and h.codec == 0 and (h.width, h.height, h.max_fps) == (1920, 1080, 60)
        assert h.features & FEATURE_AUDIO_DELAY

        # 2. Send a 90 KB keyframe + a normal delta + a delta after a 2-frame RTP gap
        kf = b"\x00\x00\x00\x01\x67" + os.urandom(90_000)
        d1 = b"\x00\x00\x00\x01\x41" + os.urandom(12_000)
        d2 = b"\x00\x00\x00\x01\x65" + os.urandom(20_000)
        for flags, rtp, payload in (
            (FLAG_KEYFRAME, 90000, kf),
            (0, 91500, d1),
            (FLAG_KEYFRAME, 96000, d2),  # 4500 ticks = 2 frames lost + keyframe -> recovered=True
        ):
            frame = _FRAME_HDR.pack(flags, rtp, 0) + payload
            sock.sendall(ws_encode_frame(0x2, frame, mask=mask))

        deadline = time.perf_counter() + 2.0
        while tap_reader.header().write_seq < 3 and time.perf_counter() < deadline:
            time.sleep(0.01)
        h = tap_reader.header()
        assert h.write_seq == 3

        p0 = tap_reader.read(0, h.qpc_freq)
        p1 = tap_reader.read(1, h.qpc_freq)
        p2 = tap_reader.read(2, h.qpc_freq)
        assert p0 is not None and p0.data == kf and p0.frames_lost == 0
        assert p1 is not None and p1.data == d1 and p1.frames_lost == 0
        assert p2 is not None and p2.data == d2 and p2.frames_lost == 2 and p2.recovered is True

        # 3. Backchannel: audio_delay_ms and idr_request reach the client
        tap_reader.set_audio_delay(37)
        tap_reader.request_idr()
        got = []
        deadline = time.perf_counter() + 2.0
        while len(got) < 3 and time.perf_counter() < deadline:
            msg = ws_recv_message(sock)
            assert msg is not None and msg[0] == 0x1
            got.append(json.loads(msg[1].decode("utf-8")))
            if any(m == {"type": "audio_delay", "ms": 37} for m in got) and any(m == {"type": "idr"} for m in got):
                break
        assert {"type": "audio_delay", "ms": 37} in got
        assert {"type": "idr"} in got

        # 4. Stop stream
        sock.sendall(ws_encode_frame(0x1, json.dumps({"type": "stop"}).encode("utf-8"), mask=mask))
        deadline = time.perf_counter() + 2.0
        while tap_reader.header().live and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert not tap_reader.header().live
    finally:
        sock.close()
        bridge.stop()
        tap_reader.close()


def test_find_webview_overlay_tracks_client_below_toolbar(monkeypatch):
    from types import SimpleNamespace
    from framegen import webrtc_tap, window

    mon = window.Rect(0, 0, 2560, 1440)
    monkeypatch.setattr(window, "pick", lambda q: SimpleNamespace(hwnd=999) if "SmoovPlayXboxWebView" in q else None)
    monkeypatch.setattr(window, "monitor_of", lambda hwnd: (mon, r"\\.\DISPLAY1", 120))
    monkeypatch.setattr(webrtc_tap, "_window_dpi", lambda hwnd: 144)  # 150% DPI -> 32 DIP = 48 px

    rect = [window.Rect(100, 200, 1280, 768)]
    monkeypatch.setattr(window, "client_rect", lambda hwnd: rect[0])

    where = webrtc_tap.find_webview_overlay()
    assert where is not None
    c, m = where()
    assert c == window.Rect(100, 248, 1280, 720) and m == mon

    rect[0] = None  # minimised
    assert where() is None


def test_idle_client_does_not_block_active_stream():
    tap_name = f"Local\\smoovplay-webrtc-idle-{os.getpid()}"
    tap_reader = FrameTap(tap_name, tap_name + "-frame")
    bridge = WebRtcTapBridge(tap_name=tap_name, port=0)
    port = bridge.start()
    idle = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    active = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        _client_handshake(idle)
        _client_handshake(active)
        mask = b"\x11\x22\x33\x44"
        start_msg = json.dumps({
            "type": "start", "codec": "h264", "width": 1920, "height": 1080, "fps": 60,
        }).encode("utf-8")
        active.sendall(ws_encode_frame(0x1, start_msg, mask=mask))
        deadline = time.perf_counter() + 2.0
        while not tap_reader.header().live and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert tap_reader.header().live

        kf = b"\x00\x00\x00\x01\x67" + os.urandom(1024)
        active.sendall(ws_encode_frame(0x2, _FRAME_HDR.pack(FLAG_KEYFRAME, 90000, 0) + kf, mask=mask))
        deadline = time.perf_counter() + 2.0
        while tap_reader.header().write_seq < 1 and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert tap_reader.header().write_seq == 1

        # Disconnecting the idle client must NOT end the active stream
        idle.close()
        time.sleep(0.1)
        assert tap_reader.header().live
    finally:
        active.close()
        bridge.stop()
        tap_reader.close()


def test_format_webrtc_stats():
    assert format_webrtc_stats({
        "rtt_ms": 14.2,
        "jb_ms": 4.1,
        "jb_target_ms": 0.0,
        "jb_min_ms": 2.0,
        "jitter_ms": 1.3,
        "audio_jb_ms": 12.5,
        "fps": 59.9,
        "packets_lost": 0,
        "frames_dropped": 0,
    }) == "rtt=14.2ms jb=4.1ms (target=0.0ms min=2.0ms) jitter=1.3ms audio_jb=12.5ms fps=60 lost=0"

    assert format_webrtc_stats({
        "rtt_ms": 8.0,
        "jb_ms": 3.5,
        "frames_dropped": 2,
    }) == "rtt=8.0ms jb=3.5ms dropped=2"

    # Booleans, None, and non-finite numbers are ignored
    assert format_webrtc_stats({
        "rtt_ms": float("nan"),
        "jb_ms": True,
        "jitter_ms": None,
        "fps": float("inf"),
    }) == ""


def test_webrtc_tap_bridge_logs_stats(capsys):
    class _DummyTap:
        def begin(self, *args, **kwargs):
            pass

        def push(self, *args, **kwargs):
            pass

        def end(self):
            pass

        def close(self):
            pass

        def take_idr_request(self):
            return False

        def audio_delay_ms(self):
            return 0

    bridge = WebRtcTapBridge(port=0, tap=_DummyTap())
    port = bridge.start()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        _client_handshake(sock)
        mask = b"\x01\x02\x03\x04"
        stats_msg = {
            "type": "stats",
            "rtt_ms": 14.2,
            "jb_ms": 4.1,
            "jb_target_ms": 0.0,
            "jb_min_ms": 2.0,
            "jitter_ms": 1.3,
            "audio_jb_ms": 12.5,
            "fps": 60,
            "packets_lost": 0,
            "frames_dropped": 0,
        }
        sock.sendall(ws_encode_frame(0x1, json.dumps(stats_msg).encode("utf-8"), mask=mask))
        deadline = time.perf_counter() + 2.0
        while bridge.last_stats is None and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert bridge.last_stats == stats_msg
        out = capsys.readouterr().out
        assert "[webrtc-tap] webrtc: rtt=14.2ms jb=4.1ms (target=0.0ms min=2.0ms) jitter=1.3ms audio_jb=12.5ms fps=60 lost=0" in out
    finally:
        sock.close()
        bridge.stop()


def test_keyframe_skips_probe_when_param_sets_unchanged(monkeypatch):
    from framegen import webrtc_tap

    begins = []
    pushes = []

    class _DummyTap:
        def begin(self, codec, width, height, max_fps=60, features=0):
            begins.append((width, height))

        def push(self, payload, frames_lost=0, recovered=False):
            pushes.append(payload)

        def end(self):
            pass

        def close(self):
            pass

        def take_idr_request(self):
            return False

        def audio_delay_ms(self):
            return 0

    probe_calls = []

    def fake_probe(codec, payload):
        probe_calls.append(payload)
        return (2560, 1440) if b"\x67\x42\x00\x33" in payload else (1920, 1080)

    monkeypatch.setattr(webrtc_tap, "_probe_resolution", fake_probe)

    bridge = WebRtcTapBridge(port=0, tap=_DummyTap())
    port = bridge.start()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        _client_handshake(sock)
        mask = b"\x05\x06\x07\x08"
        sock.sendall(ws_encode_frame(
            0x1,
            json.dumps({"type": "start", "codec": "h264", "width": 1920, "height": 1080, "fps": 60, "probe": True}).encode(),
            mask=mask,
        ))
        kf_1080_a = b"\x00\x00\x00\x01\x67\x42\x00\x28\x00\x00\x00\x01\x68\xce\x00\x00\x00\x01\x65\x11"
        kf_1080_b = b"\x00\x00\x00\x01\x67\x42\x00\x28\x00\x00\x00\x01\x68\xce\x00\x00\x00\x01\x65\x22"
        kf_1440 = b"\x00\x00\x00\x01\x67\x42\x00\x33\x00\x00\x00\x01\x68\xce\x00\x00\x00\x01\x65\x33"
        for idx, payload in enumerate((kf_1080_a, kf_1080_b, kf_1440)):
            sock.sendall(ws_encode_frame(0x2, _FRAME_HDR.pack(FLAG_KEYFRAME, 90000 + idx * 1500, 0) + payload, mask=mask))

        deadline = time.perf_counter() + 2.0
        while len(pushes) < 3 and time.perf_counter() < deadline:
            time.sleep(0.01)

        assert len(pushes) == 3
        assert len(probe_calls) == 2  # kf_1080_a probed, kf_1080_b skipped, kf_1440 probed
        assert begins == [(1920, 1080), (2560, 1440)]
    finally:
        sock.close()
        bridge.stop()


def test_stop_does_not_clobber_tap_when_no_browser_streamed():
    end_calls = 0

    class _SpyTap:
        def begin(self, *args, **kwargs):
            pass

        def push(self, *args, **kwargs):
            pass

        def end(self):
            nonlocal end_calls
            end_calls += 1

        def close(self):
            pass

        def take_idr_request(self):
            return False

        def audio_delay_ms(self):
            return 0

    # 1. No client ever connected -> stop() must NOT call tap.end()
    bridge = WebRtcTapBridge(port=0, tap=_SpyTap())
    bridge.start()
    bridge.stop()
    assert end_calls == 0

    # 2. Idle client connected (no "start" / frames) -> stop() must NOT call tap.end()
    bridge = WebRtcTapBridge(port=0, tap=_SpyTap())
    port = bridge.start()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        _client_handshake(sock)
        time.sleep(0.05)
    finally:
        bridge.stop()
        sock.close()
    assert end_calls == 0

    # 3. Lazy default tap (tap=None): never even opens FrameTap when no client streams
    lazy_bridge = WebRtcTapBridge(port=0)
    lazy_bridge.start()
    assert lazy_bridge.tap is None
    lazy_bridge.stop()
    assert lazy_bridge.tap is None


def test_stop_does_not_double_end_after_browser_already_stopped():
    end_calls = 0

    class _SpyTap:
        def begin(self, *args, **kwargs):
            pass

        def push(self, *args, **kwargs):
            pass

        def end(self):
            nonlocal end_calls
            end_calls += 1

        def close(self):
            pass

        def take_idr_request(self):
            return False

        def audio_delay_ms(self):
            return 0

    bridge = WebRtcTapBridge(port=0, tap=_SpyTap())
    port = bridge.start()
    sock = socket.create_connection(("127.0.0.1", port), timeout=5.0)
    try:
        _client_handshake(sock)
        mask = b"\x09\x0A\x0B\x0C"
        sock.sendall(ws_encode_frame(
            0x1,
            json.dumps({"type": "start", "codec": "h264", "width": 1920, "height": 1080, "fps": 60}).encode(),
            mask=mask,
        ))
        deadline = time.perf_counter() + 2.0
        while not bridge.active and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert bridge.active

        # Client sends explicit stop -> tap.end() called once
        sock.sendall(ws_encode_frame(0x1, json.dumps({"type": "stop"}).encode(), mask=mask))
        deadline = time.perf_counter() + 2.0
        while bridge.active and time.perf_counter() < deadline:
            time.sleep(0.01)
        assert not bridge.active
        assert end_calls == 1
    finally:
        sock.close()
        bridge.stop()
    # bridge.stop() must NOT call tap.end() a second time (which could clobber a native writer)
    assert end_calls == 1
