"""WebRTC Encoded Streams -> SmoovPlay v1 compressed frame tap bridge.

Receives raw compressed H.264 / H.265 frames over a local WebSocket (127.0.0.1) from a browser userscript
(scripts/xbox_webrtc_tap.user.js for xbox.com/play and Better xCloud) or Electron client (Greenlight) and writes
them straight into SmoovPlay's shared-memory frame tap (docs/frame-tap.md, version 1 compressed), so SmoovPlay
decodes on NVDEC without screen capture, DWM composition, or Clock60 resampling.

Zero external dependencies: uses a small RFC 6455 WebSocket server built on Python's stdlib sockets + numpy for
fast frame unmasking.

Wire protocol (client -> bridge):
  - Text frame (JSON):
      {"type": "start", "codec": "h264"|"h265"|"h265_hdr", "width": 1920, "height": 1080, "fps": 60,
       "audio_sync": true}
      {"type": "stats", "rtt_ms": ..., "jb_ms": ..., "jb_target_ms": ..., "jb_min_ms": ...,
       "jitter_ms": ..., "audio_jb_ms": ..., "fps": ..., "packets_lost": ..., "frames_dropped": ...}
      {"type": "stop"}
  - Binary frame (12-byte little-endian header + raw Annex-B compressed bitstream):
      u32 flags          bit 0: keyframe (IDR), bit 1: recovered
      u32 rtp_timestamp  90 kHz RTP timestamp (0 if unknown; used to infer dropped frames from gaps)
      u32 frames_lost    explicit lost-frame count before this frame (0 if unknown)
      u8[] payload       raw Annex-B H.264 / H.265 NAL units

Wire protocol (bridge -> client, JSON text frames):
  - {"type": "audio_delay", "ms": <int>}   when SmoovPlay updates audio_delay_ms
  - {"type": "idr"}                        when SmoovPlay requests a keyframe (idr_request)

Usage:
    python -m framegen.webrtc_tap [--port 8088] [--tap Local\\smoovplay-tap]
    python run.py --source tap --display   # in another terminal (either order)
"""
from __future__ import annotations

import argparse
import base64
import ctypes
import hashlib
import json
import math
import socket
import struct
import sys
import threading
import time

import numpy as np

from .chiaki_tap import FEATURE_AUDIO_DELAY, FrameTap, param_sets
from .tap import DEFAULT_NAME

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8088  # explicitly permitted by play.xbox.com's CSP connect-src
_WS_GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"
_FRAME_HDR = struct.Struct("<III")
FLAG_KEYFRAME = 1
FLAG_RECOVERED = 2
CODEC_IDS = {"h264": 0, "avc": 0, "h265": 1, "hevc": 1, "h265_hdr": 2, "hevc_hdr": 2}
MAX_PAYLOAD = 4 * 1024 * 1024  # 4 MiB safety cap per WebSocket message


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    buf = bytearray(n)
    view = memoryview(buf)
    got = 0
    while got < n:
        r = sock.recv_into(view[got:], n - got)
        if r == 0:
            raise ConnectionError("peer closed connection")
        got += r
    return bytes(buf)


def _unmask(payload: bytes, mask: bytes) -> bytes:
    if not payload:
        return b""
    arr = np.frombuffer(payload, dtype=np.uint8).copy()
    m = np.frombuffer(mask, dtype=np.uint8)
    # Vectorised 4-byte repeating XOR: ~5 us for a 50 KB H.264 frame
    n4 = (arr.size // 4) * 4
    if n4:
        arr[:n4].reshape(-1, 4)[:] ^= m
    for i in range(n4, arr.size):
        arr[i] ^= m[i & 3]
    return arr.tobytes()


def ws_handshake(sock: socket.socket) -> dict[str, str]:
    """Perform an RFC 6455 server handshake on `sock`. Returns lower-cased request headers."""
    raw = bytearray()
    while b"\r\n\r\n" not in raw:
        chunk = sock.recv(4096)
        if not chunk:
            raise ConnectionError("closed during WebSocket handshake")
        raw.extend(chunk)
        if len(raw) > 16384:
            raise ValueError("HTTP header too large")
    head = raw.split(b"\r\n\r\n", 1)[0].decode("iso-8859-1", errors="replace")
    headers: dict[str, str] = {}
    for line in head.split("\r\n")[1:]:
        if ":" in line:
            k, v = line.split(":", 1)
            headers[k.strip().lower()] = v.strip()
    key = headers.get("sec-websocket-key")
    if not key:
        sock.sendall(b"HTTP/1.1 400 Bad Request\r\nContent-Length: 0\r\n\r\n")
        raise ValueError("missing Sec-WebSocket-Key")
    accept = base64.b64encode(hashlib.sha1((key + _WS_GUID).encode("ascii")).digest()).decode("ascii")
    resp = (
        "HTTP/1.1 101 Switching Protocols\r\n"
        "Upgrade: websocket\r\n"
        "Connection: Upgrade\r\n"
        f"Sec-WebSocket-Accept: {accept}\r\n\r\n"
    ).encode("ascii")
    sock.sendall(resp)
    return headers


def ws_encode_frame(opcode: int, payload: bytes, mask: bytes | None = None) -> bytes:
    """Encode a single RFC 6455 frame (unmasked for server->client, or masked when `mask` is 4 bytes)."""
    b0 = 0x80 | (opcode & 0x0F)
    n = len(payload)
    mbit = 0x80 if mask is not None else 0
    if n < 126:
        hdr = struct.pack("!BB", b0, mbit | n)
    elif n < 65536:
        hdr = struct.pack("!BBH", b0, mbit | 126, n)
    else:
        hdr = struct.pack("!BBQ", b0, mbit | 127, n)
    if mask is not None:
        return hdr + mask + _unmask(payload, mask)
    return hdr + payload


def ws_recv_message(sock: socket.socket) -> tuple[int, bytes] | None:
    """Read one complete WebSocket message (reassembling fragments, auto-replying to Ping).
    Returns (opcode, payload) for text (1) or binary (2), or None on Close."""
    msg_opcode = 0
    chunks: list[bytes] = []
    total = 0
    while True:
        b0, b1 = _recv_exact(sock, 2)
        fin = bool(b0 & 0x80)
        opcode = b0 & 0x0F
        masked = bool(b1 & 0x80)
        length = b1 & 0x7F
        if length == 126:
            (length,) = struct.unpack("!H", _recv_exact(sock, 2))
        elif length == 127:
            (length,) = struct.unpack("!Q", _recv_exact(sock, 8))
        if length > MAX_PAYLOAD or total + length > MAX_PAYLOAD:
            raise ValueError(f"WebSocket payload too large ({total + length} bytes)")
        mask = _recv_exact(sock, 4) if masked else None
        data = _recv_exact(sock, length) if length else b""
        if mask is not None:
            data = _unmask(data, mask)

        if opcode == 0x8:  # Close
            try:
                sock.sendall(ws_encode_frame(0x8, b""))
            except OSError:
                pass
            return None
        if opcode == 0x9:  # Ping -> Pong
            sock.sendall(ws_encode_frame(0xA, data))
            continue
        if opcode == 0xA:  # Pong
            continue
        if opcode in (0x1, 0x2):
            msg_opcode = opcode
            chunks = [data]
            total = len(data)
        elif opcode == 0x0:  # Continuation
            chunks.append(data)
            total += len(data)
        else:
            continue
        if fin and msg_opcode:
            return msg_opcode, b"".join(chunks) if len(chunks) > 1 else chunks[0]


def _probe_resolution(codec_id: int, payload: bytes) -> tuple[int, int] | None:
    """Decode the first keyframe with PyAV (CPU) to read the bitstream's exact (width, height) — e.g. 2560x1440
    for Xbox Remote Play 1440p Max Quality, before opening the tap header. Returns None if not a decodable keyframe."""
    try:
        import av
        name = "hevc" if codec_id in (1, 2) else "h264"
        ctx = av.CodecContext.create(name, "r")
        for fr in ctx.decode(av.Packet(payload)):
            if fr.width > 0 and fr.height > 0:
                return int(fr.width), int(fr.height)
    except Exception:  # noqa: BLE001 - synthetic payloads or delta frames before first IDR
        pass
    return None


def infer_lost_frames(prev_rtp: int | None, rtp: int, fps: int) -> int:
    """Infer lost frames from 90 kHz RTP timestamp gaps (uint32 wrap-safe). Returns 0 when cadence is normal."""
    if prev_rtp is None or rtp == 0 or prev_rtp == 0 or fps <= 0:
        return 0
    delta = (rtp - prev_rtp) & 0xFFFFFFFF
    if delta == 0 or delta > 90000 * 5:  # duplicate timestamp or >5 s discontinuity (handled as stream jump)
        return 0
    step = 90000.0 / fps
    missed = int(round(delta / step)) - 1
    return max(0, missed) if delta > step * 1.6 else 0


def _finite_num(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        f = float(value)
        if math.isfinite(f):
            return f
    return None


def format_webrtc_stats(stats: dict) -> str:
    """Format a live WebRTC RTT / jitter-buffer stats message from the tap script into one log line."""
    parts: list[str] = []
    rtt = _finite_num(stats.get("rtt_ms"))
    if rtt is not None:
        parts.append(f"rtt={rtt:.1f}ms")
    jb = _finite_num(stats.get("jb_ms"))
    if jb is not None:
        jb_str = f"jb={jb:.1f}ms"
        sub: list[str] = []
        jb_target = _finite_num(stats.get("jb_target_ms"))
        if jb_target is not None:
            sub.append(f"target={jb_target:.1f}ms")
        jb_min = _finite_num(stats.get("jb_min_ms"))
        if jb_min is not None:
            sub.append(f"min={jb_min:.1f}ms")
        if sub:
            jb_str += f" ({' '.join(sub)})"
        parts.append(jb_str)
    jitter = _finite_num(stats.get("jitter_ms"))
    if jitter is not None:
        parts.append(f"jitter={jitter:.1f}ms")
    audio_jb = _finite_num(stats.get("audio_jb_ms"))
    if audio_jb is not None:
        parts.append(f"audio_jb={audio_jb:.1f}ms")
    fps = _finite_num(stats.get("fps"))
    if fps is not None:
        parts.append(f"fps={round(fps)}")
    lost = _finite_num(stats.get("packets_lost"))
    if lost is not None:
        parts.append(f"lost={int(lost)}")
    dropped = _finite_num(stats.get("frames_dropped"))
    if dropped is not None and dropped > 0:
        parts.append(f"dropped={int(dropped)}")
    return " ".join(parts)


class WebRtcTapBridge:
    """Listens on 127.0.0.1:<port> for WebRTC Encoded Streams and writes them into a SmoovPlay FrameTap."""

    def __init__(self, tap_name: str = DEFAULT_NAME, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT,
                 tap: FrameTap | None = None) -> None:
        self.tap_name = tap_name
        self.host = host
        self.port = port
        self._owns_tap = tap is None
        self.tap: FrameTap | None = tap
        self._srv: socket.socket | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._clients: set[socket.socket] = set()
        self._client_threads: set[threading.Thread] = set()
        self._clients_lock = threading.Lock()
        self.frames_received = 0
        self.bytes_received = 0
        self.active = False
        self.last_stats: dict | None = None

    def _ensure_tap(self) -> FrameTap:
        if self.tap is None:
            self.tap = FrameTap(self.tap_name, self.tap_name + "-frame")
        return self.tap

    def start(self) -> int:
        """Bind and start serving in a background daemon thread. Returns the bound TCP port."""
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(2)
        srv.settimeout(0.25)
        self._srv = srv
        self.port = srv.getsockname()[1]
        self._stop.clear()
        self._thread = threading.Thread(target=self._accept_loop, name="webrtc-tap", daemon=True)
        self._thread.start()
        return self.port

    def stop(self) -> None:
        self._stop.set()
        with self._clients_lock:
            clients = list(self._clients)
            threads = list(self._client_threads)
        for c in clients:
            try:
                c.close()
            except OSError:
                pass
        if self._srv is not None:
            try:
                self._srv.close()
            except OSError:
                pass
            self._srv = None
        if self._thread is not None:
            self._thread.join(timeout=2.0)
            self._thread = None
        for t in threads:
            if t is not threading.current_thread():
                t.join(timeout=1.0)
        # Only end the shared-memory tap if a browser client is still actively streaming through
        # this bridge; otherwise we would clobber an external native writer (e.g. smoovplay-xbox.exe).
        if self.active and self.tap is not None:
            self.active = False
            try:
                self.tap.end()
            except Exception:  # noqa: BLE001
                pass
        if self._owns_tap and self.tap is not None:
            try:
                self.tap.close()
            except Exception:  # noqa: BLE001
                pass
            self.tap = None

    def _accept_loop(self) -> None:
        assert self._srv is not None
        while not self._stop.is_set():
            try:
                conn, addr = self._srv.accept()
            except socket.timeout:
                continue
            except OSError:
                break
            try:
                conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            except OSError:
                pass
            t = threading.Thread(target=self._handle_client, args=(conn, addr),
                                 name="webrtc-tap-client", daemon=True)
            with self._clients_lock:
                self._clients.add(conn)
                self._client_threads.add(t)
            t.start()

    def _handle_client(self, conn: socket.socket, addr) -> None:
        try:
            self._serve_client(conn, addr)
        except Exception as e:  # noqa: BLE001
            if not self._stop.is_set():
                print(f"[webrtc-tap] client {addr[0]}:{addr[1]} disconnected: {e}")
        finally:
            me = threading.current_thread()
            with self._clients_lock:
                self._clients.discard(conn)
                self._client_threads.discard(me)
            try:
                conn.close()
            except OSError:
                pass

    def _serve_client(self, conn: socket.socket, addr) -> None:
        ws_handshake(conn)
        conn.settimeout(1.0)
        send_lock = threading.Lock()
        client_done = threading.Event()
        started = False
        pending_start: dict | None = None
        codec = 0
        codec_name = "h264"
        width, height, fps, feats = 1920, 1080, 60, 0
        prev_rtp: int | None = None
        last_audio_delay = -1
        last_ps = b""

        def begin_stream(w: int, h: int) -> None:
            nonlocal started, width, height, prev_rtp, last_audio_delay
            width, height = w, h
            self._ensure_tap().begin(codec, width, height, max_fps=fps, features=feats)
            started, self.active, prev_rtp, last_audio_delay = True, True, None, -1
            print(f"[webrtc-tap] stream started: {codec_name.upper()} {width}x{height} @ {fps} fps -> {self.tap_name}")

        def send_json(obj: dict) -> None:
            frame = ws_encode_frame(0x1, json.dumps(obj).encode("utf-8"))
            with send_lock:
                conn.sendall(frame)

        def poll_backchannel() -> None:
            nonlocal last_audio_delay
            while not self._stop.is_set() and not client_done.is_set():
                try:
                    tap = self.tap
                    if started and tap is not None:
                        if tap.take_idr_request():
                            send_json({"type": "idr"})
                        delay = tap.audio_delay_ms()
                        if delay != last_audio_delay:
                            last_audio_delay = delay
                            send_json({"type": "audio_delay", "ms": delay})
                except OSError:
                    break
                client_done.wait(0.05)

        back = threading.Thread(target=poll_backchannel, name="webrtc-tap-back", daemon=True)
        back.start()
        print(f"[webrtc-tap] client connected from {addr[0]}:{addr[1]}")
        try:
            while not self._stop.is_set():
                try:
                    msg = ws_recv_message(conn)
                except socket.timeout:
                    continue
                if msg is None:
                    break
                opcode, data = msg
                if opcode == 0x1:
                    ctrl = json.loads(data.decode("utf-8"))
                    kind = ctrl.get("type")
                    if kind == "start":
                        codec_name = str(ctrl.get("codec", "h264")).lower()
                        codec = CODEC_IDS.get(codec_name, 0)
                        w = max(16, int(ctrl.get("width", 1920)))
                        h = max(16, int(ctrl.get("height", 1080)))
                        fps = max(1, min(240, int(ctrl.get("fps", 60))))
                        feats = FEATURE_AUDIO_DELAY if ctrl.get("audio_sync") else 0
                        last_ps = b""
                        if ctrl.get("probe", False):
                            pending_start = {"width": w, "height": h}
                        else:
                            pending_start = None
                            begin_stream(w, h)
                    elif kind == "stats":
                        self.last_stats = ctrl
                        line = format_webrtc_stats(ctrl)
                        if line:
                            print(f"[webrtc-tap] webrtc: {line}")
                    elif kind == "stop":
                        pending_start = None
                        last_ps = b""
                        if started:
                            self.tap.end()
                            started, self.active, prev_rtp = False, False, None
                            print("[webrtc-tap] stream stopped")
                elif opcode == 0x2:
                    if len(data) < _FRAME_HDR.size:
                        continue
                    flags, rtp, lost = _FRAME_HDR.unpack_from(data, 0)
                    payload = data[_FRAME_HDR.size:]
                    if not payload:
                        continue
                    is_key = bool(flags & FLAG_KEYFRAME)
                    if not started:
                        if is_key:
                            ps = param_sets(payload, "hevc" if codec in (1, 2) else "h264")
                            probed = _probe_resolution(codec, payload)
                            if probed is not None:
                                begin_stream(*probed)
                            elif pending_start is not None:
                                begin_stream(pending_start["width"], pending_start["height"])
                            else:
                                begin_stream(1920, 1080)
                            if ps:
                                last_ps = ps
                            pending_start = None
                        else:
                            continue  # wait for the first keyframe so resolution & NVDEC start cleanly
                    elif is_key:
                        # Avoid CPU-decoding every keyframe: only re-probe when SPS/PPS actually changes.
                        ps = param_sets(payload, "hevc" if codec in (1, 2) else "h264")
                        if ps and ps != last_ps:
                            last_ps = ps
                            probed = _probe_resolution(codec, payload)
                            if probed is not None and probed != (width, height):
                                print(f"[webrtc-tap] resolution changed: {width}x{height} -> {probed[0]}x{probed[1]}")
                                begin_stream(*probed)
                    if lost == 0:
                        lost = infer_lost_frames(prev_rtp, rtp, fps)
                    if rtp:
                        prev_rtp = rtp
                    recovered = bool(flags & FLAG_RECOVERED) or (lost > 0 and is_key)
                    self.tap.push(payload, frames_lost=lost, recovered=recovered)
                    self.frames_received += 1
                    self.bytes_received += len(payload)
        finally:
            client_done.set()
            back.join(timeout=1.0)
            if started:
                self.active = False
                try:
                    self.tap.end()
                except Exception:  # noqa: BLE001
                    pass


def _window_dpi(hwnd: int) -> int:
    """DPI of `hwnd` (96 at 100% scaling, 144 at 150%, etc.), for converting WPF DIPs to screen pixels."""
    if sys.platform != "win32":
        return 96
    try:
        u32 = ctypes.WinDLL("user32")
        u32.GetDpiForWindow.argtypes, u32.GetDpiForWindow.restype = [ctypes.c_void_p], ctypes.c_uint
        dpi = int(u32.GetDpiForWindow(ctypes.c_void_p(hwnd)))
        return dpi if dpi > 0 else 96
    except (OSError, AttributeError):
        return 96


def find_webview_overlay(exe: str = "SmoovPlayXboxWebView.exe", top_dip: int = 32):
    """If the SmoovPlay Xbox WebView2 host is open, returns an `overlay_of` callable () -> (client, monitor)
    for GLWindowSink so SmoovPlay sits click-through over the WebView2 video viewport while XboxWebView keeps
    focus for controller input (and its top toolbar stays clickable). Returns None if the host isn't running."""
    if sys.platform != "win32":
        return None
    from . import window
    target = window.pick(f"exe:{exe}")
    if target is None:
        return None
    hwnd = target.hwnd

    def where():
        c = window.client_rect(hwnd)
        if c is None or c.w <= 0 or c.h <= 0:
            return None
        bar = round(top_dip * _window_dpi(hwnd) / 96.0)
        if c.h > bar + 32:
            c = window.Rect(c.x, c.y + bar, c.w, c.h - bar)
        mon = window.monitor_of(hwnd)[0]
        return c, mon

    return where


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--port", type=int, default=DEFAULT_PORT, help="localhost WebSocket port (default %(default)s)")
    ap.add_argument("--tap", default=DEFAULT_NAME, help="shared-memory tap name (default %(default)s)")
    a = ap.parse_args(argv)
    bridge = WebRtcTapBridge(tap_name=a.tap, port=a.port)
    port = bridge.start()
    print(f"[webrtc-tap] listening on ws://{DEFAULT_HOST}:{port} -> {a.tap}")
    print("[webrtc-tap] start SmoovPlay with: python run.py --source tap --display")
    try:
        while True:
            time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        bridge.stop()
        print(f"[webrtc-tap] stopped ({bridge.frames_received} frames, {bridge.bytes_received / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
