// ==UserScript==
// @name         SmoovPlay Xbox WebRTC Frame Tap
// @namespace    https://github.com/smoovplay
// @version      0.1.0
// @description  Streams raw compressed WebRTC H.264 frames from xbox.com/play (or Better xCloud / Greenlight) directly into SmoovPlay's low-latency frame tap before browser decode.
// @match        https://play.xbox.com/*
// @match        https://www.xbox.com/*
// @run-at       document-start
// @grant        none
// ==/UserScript==

(function () {
  "use strict";

  const WS_URL = window.SMOOVPLAY_TAP_URL || "ws://127.0.0.1:8088";
  const FLAG_KEYFRAME = 1;
  const FLAG_RECOVERED = 2;
  // Set window.SMOOVPLAY_SKIP_BROWSER_DECODE = true to skip browser video decoding while SmoovPlay is connected.
  const skipDecode = () => Boolean(window.SMOOVPLAY_SKIP_BROWSER_DECODE);
  // Route WebRTC audio straight to AudioContext / dedicated <audio> by default so audio plays even if
  // play.xbox.com starts muted or Skip browser decode stalls <video>'s A/V sync clock.
  const directAudioEnabled = () => window.SMOOVPLAY_DIRECT_AUDIO !== false;

  // Target 0 ms WebRTC jitter buffer / playout delay by default; set to false to keep browser defaults.
  const zeroJitterBuffer = () => window.SMOOVPLAY_ZERO_JITTER_BUFFER !== false;
  // Live WebRTC RTT / jitter-buffer stats logging interval in ms (0 disables).
  const statsIntervalMs = () => {
    const raw = window.SMOOVPLAY_STATS_INTERVAL_MS;
    if (raw === undefined) return 2000;
    const n = Number(raw);
    return Number.isFinite(n) && n >= 0 ? n : 2000;
  };

  let ws = null;
  let wsOpen = false;
  let started = false;
  let streamWidth = 1920;
  let streamHeight = 1080;
  let streamFps = 60;
  let audioCtx = null;
  let audioDelayNode = null;
  let audioPuller = null;
  let audioPc = null;
  let currentAudioDelaySec = 0;
  let directAudioActive = false;
  let framesSent = 0;
  let lastKeyframe = null; // { data: Uint8Array, timestamp: number } for instant mid-stream attach
  let videoPc = null;
  let videoReceiver = null;
  let audioReceiver = null;
  let statsTimer = null;
  let statsInFlight = false;
  let prevStats = null;
  const seenMedia = new Set();

  function log(...args) {
    console.log("[smoovplay-tap]", ...args);
  }

  function applyLowLatencyReceiver(receiver, label) {
    if (!receiver || !zeroJitterBuffer()) return;
    const applied = [];
    if ("jitterBufferTarget" in receiver) {
      try {
        if (receiver.jitterBufferTarget !== 0) {
          receiver.jitterBufferTarget = 0;
          applied.push("jitterBufferTarget=0");
        }
      } catch (err) {
        log(`failed to set ${label} jitterBufferTarget=0:`, err);
      }
    }
    if ("playoutDelayHint" in receiver) {
      try {
        if (receiver.playoutDelayHint !== 0) {
          receiver.playoutDelayHint = 0;
          applied.push("playoutDelayHint=0");
        }
      } catch (err) {
        log(`failed to set ${label} playoutDelayHint=0:`, err);
      }
    }
    if (applied.length > 0) {
      log(`${label} receiver low-latency:`, applied.join(", "));
    }
  }

  function round1(val) {
    return typeof val === "number" && Number.isFinite(val) ? Math.round(val * 10) / 10 : null;
  }

  function intervalAvgMs(curSum, prevSum, curCount, prevCount) {
    if (typeof curSum !== "number" || typeof curCount !== "number" || curCount <= 0) return null;
    if (typeof prevSum === "number" && typeof prevCount === "number") {
      if (curCount > prevCount && curSum >= prevSum) {
        return round1(((curSum - prevSum) / (curCount - prevCount)) * 1000);
      }
      if (curCount === prevCount) return null;
    }
    return round1((curSum / curCount) * 1000);
  }

  function formatStatsLine(s) {
    const parts = [];
    if (s.rtt_ms != null) parts.push(`rtt=${s.rtt_ms.toFixed(1)}ms`);
    if (s.jb_ms != null) {
      let jb = `jb=${s.jb_ms.toFixed(1)}ms`;
      const sub = [];
      if (s.jb_target_ms != null) sub.push(`target=${s.jb_target_ms.toFixed(1)}ms`);
      if (s.jb_min_ms != null) sub.push(`min=${s.jb_min_ms.toFixed(1)}ms`);
      if (sub.length > 0) jb += ` (${sub.join(" ")})`;
      parts.push(jb);
    }
    if (s.jitter_ms != null) parts.push(`jitter=${s.jitter_ms.toFixed(1)}ms`);
    if (s.audio_jb_ms != null) parts.push(`audio_jb=${s.audio_jb_ms.toFixed(1)}ms`);
    if (s.fps != null) parts.push(`fps=${Math.round(s.fps)}`);
    if (s.packets_lost != null) parts.push(`lost=${s.packets_lost}`);
    if (s.frames_dropped != null && s.frames_dropped > 0) parts.push(`dropped=${s.frames_dropped}`);
    return parts.join(" ");
  }

  async function pollWebRtcStats(pc) {
    if (!pc || pc !== videoPc || statsInFlight) return;
    statsInFlight = true;
    try {
      // Re-enforce 0 ms jitter buffer hints in case the page overwrites them mid-stream.
      applyLowLatencyReceiver(videoReceiver, "video");
      applyLowLatencyReceiver(audioReceiver, "audio");

      const report = await pc.getStats();
      let selectedPairId = null;
      let pair = null;
      let remoteOutboundRtt = null;
      let videoIn = null;
      let audioIn = null;

      report.forEach((s) => {
        if (s.type === "transport" && s.selectedCandidatePairId) {
          selectedPairId = s.selectedCandidatePairId;
        } else if (s.type === "inbound-rtp") {
          const kind = s.kind || s.mediaType;
          if (kind === "video") videoIn = s;
          else if (kind === "audio") audioIn = s;
        } else if (s.type === "remote-outbound-rtp" && typeof s.roundTripTime === "number") {
          remoteOutboundRtt = s.roundTripTime;
        }
      });

      if (selectedPairId && typeof report.get === "function") {
        pair = report.get(selectedPairId) || null;
      }
      if (!pair) {
        report.forEach((s) => {
          if (s.type !== "candidate-pair" || s.state !== "succeeded") return;
          if (!pair || (s.nominated && !pair.nominated)) {
            pair = s;
          }
        });
      }

      let rttMs = null;
      if (pair && typeof pair.currentRoundTripTime === "number") {
        rttMs = round1(pair.currentRoundTripTime * 1000);
      } else if (remoteOutboundRtt !== null) {
        rttMs = round1(remoteOutboundRtt * 1000);
      }

      const prev = prevStats;
      const stats = {
        rtt_ms: rttMs,
        jb_ms: videoIn
          ? intervalAvgMs(
              videoIn.jitterBufferDelay,
              prev && prev.vJbDelay,
              videoIn.jitterBufferEmittedCount,
              prev && prev.vJbEmitted
            )
          : null,
        jb_target_ms: videoIn
          ? intervalAvgMs(
              videoIn.jitterBufferTargetDelay,
              prev && prev.vJbTargetDelay,
              videoIn.jitterBufferEmittedCount,
              prev && prev.vJbEmitted
            )
          : null,
        jb_min_ms: videoIn
          ? intervalAvgMs(
              videoIn.jitterBufferMinimumDelay,
              prev && prev.vJbMinDelay,
              videoIn.jitterBufferEmittedCount,
              prev && prev.vJbEmitted
            )
          : null,
        jitter_ms: videoIn && typeof videoIn.jitter === "number" ? round1(videoIn.jitter * 1000) : null,
        audio_jb_ms: audioIn
          ? intervalAvgMs(
              audioIn.jitterBufferDelay,
              prev && prev.aJbDelay,
              audioIn.jitterBufferEmittedCount,
              prev && prev.aJbEmitted
            )
          : null,
        fps: videoIn && typeof videoIn.framesPerSecond === "number" ? round1(videoIn.framesPerSecond) : null,
        packets_lost: videoIn && typeof videoIn.packetsLost === "number" ? videoIn.packetsLost : null,
        frames_dropped: videoIn && typeof videoIn.framesDropped === "number" ? videoIn.framesDropped : null,
      };

      if (videoReceiver && "jitterBufferTarget" in videoReceiver) {
        stats.jb_target_prop = videoReceiver.jitterBufferTarget;
      }
      if (videoReceiver && "playoutDelayHint" in videoReceiver) {
        stats.playout_delay_hint = videoReceiver.playoutDelayHint;
      }

      prevStats = {
        vJbDelay: videoIn ? videoIn.jitterBufferDelay : undefined,
        vJbTargetDelay: videoIn ? videoIn.jitterBufferTargetDelay : undefined,
        vJbMinDelay: videoIn ? videoIn.jitterBufferMinimumDelay : undefined,
        vJbEmitted: videoIn ? videoIn.jitterBufferEmittedCount : undefined,
        aJbDelay: audioIn ? audioIn.jitterBufferDelay : undefined,
        aJbEmitted: audioIn ? audioIn.jitterBufferEmittedCount : undefined,
      };

      const line = formatStatsLine(stats);
      if (line) {
        log("webrtc:", line);
        if (wsOpen && ws && ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify(Object.assign({ type: "stats" }, stats)));
        }
      }
    } catch (err) {
      log("stats poll failed:", err && err.message ? err.message : err);
    } finally {
      statsInFlight = false;
    }
  }

  function stopStatsLogging() {
    if (statsTimer !== null) {
      clearInterval(statsTimer);
      statsTimer = null;
    }
    statsInFlight = false;
    prevStats = null;
  }

  function startStatsLogging(pc) {
    stopStatsLogging();
    const interval = statsIntervalMs();
    if (!pc || typeof pc.getStats !== "function" || interval <= 0) return;
    statsTimer = setInterval(() => {
      pollWebRtcStats(pc);
    }, interval);
  }

  function muteForeignMediaElement(el) {
    if (!el || el.__smoovplayAudio) return;
    seenMedia.add(el);
    if (!el.__smoovplayMuteHooked) {
      el.__smoovplayMuteHooked = true;
      el.addEventListener("volumechange", () => {
        if (directAudioActive && !el.muted) el.muted = true;
      });
    }
    if (directAudioActive && !el.muted) {
      el.muted = true;
    }
  }

  function muteAllPageMedia() {
    if (!directAudioActive) return;
    for (const el of seenMedia) {
      if (!el.__smoovplayAudio && !el.muted) el.muted = true;
    }
    if (document.querySelectorAll) {
      for (const el of document.querySelectorAll("video, audio")) {
        muteForeignMediaElement(el);
      }
    }
  }

  function resumeAudioIfNeeded() {
    if (audioCtx && audioCtx.state === "suspended") {
      audioCtx.resume().catch(() => {});
    }
    if (audioPuller && audioPuller.paused) {
      audioPuller.play().catch(() => {});
    }
  }

  if (window.HTMLMediaElement && HTMLMediaElement.prototype.play) {
    const origPlay = HTMLMediaElement.prototype.play;
    HTMLMediaElement.prototype.play = function (...args) {
      muteForeignMediaElement(this);
      return origPlay.apply(this, args);
    };
  }
  for (const evt of ["pointerdown", "keydown", "click"]) {
    window.addEventListener(evt, resumeAudioIfNeeded, { capture: true, passive: true });
  }

  function connect() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return;
    }
    try {
      ws = new WebSocket(WS_URL);
      ws.binaryType = "arraybuffer";
    } catch (err) {
      log("WebSocket creation failed (CSP?):", err);
      setTimeout(connect, 1000);
      return;
    }
    ws.onopen = () => {
      wsOpen = true;
      started = false;
      log("connected to", WS_URL);
      // If Xbox is already streaming, replay the latest keyframe immediately so SmoovPlay attaches at once
      // without waiting for the next keyframe or refreshing the tab.
      if (lastKeyframe) {
        ensureStarted();
        sendRawFrame(lastKeyframe.data, lastKeyframe.timestamp, FLAG_KEYFRAME);
        log("replayed cached keyframe for instant mid-stream attach");
      }
    };
    ws.onmessage = (ev) => {
      if (typeof ev.data !== "string") return;
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === "audio_delay") {
          const sec = Math.max(0, Math.min(0.5, (Number(msg.ms) || 0) / 1000));
          currentAudioDelaySec = sec;
          if (audioDelayNode) {
            audioDelayNode.delayTime.value = sec;
          }
        }
      } catch {
        // ignore malformed control message
      }
    };
    ws.onclose = () => {
      wsOpen = false;
      started = false;
      if (videoPc !== null) {
        setTimeout(connect, 500);
      }
    };
    ws.onerror = () => {
      try { ws.close(); } catch { /* noop */ }
    };
  }

  function detectVideoSize() {
    const videos = document.querySelectorAll("video");
    for (const v of videos) {
      if (v.videoWidth > 0 && v.videoHeight > 0) {
        return [v.videoWidth, v.videoHeight];
      }
    }
    return [streamWidth, streamHeight];
  }

  function ensureStarted() {
    if (!wsOpen || started) return;
    const [w, h] = detectVideoSize();
    streamWidth = w;
    streamHeight = h;
    ws.send(JSON.stringify({
      type: "start",
      codec: "h264",
      width: streamWidth,
      height: streamHeight,
      fps: streamFps,
      audio_sync: Boolean(audioDelayNode || window.SMOOVPLAY_AUDIO_SYNC),
      probe: true,
    }));
    started = true;
    log(`stream started (${streamWidth}x${streamHeight} @ ${streamFps} fps)`);
  }

  function sendRawFrame(payload, timestamp, flags) {
    if (!wsOpen || payload.byteLength === 0) return;
    // Drop frames if the local socket is somehow backed up rather than adding latency.
    if (ws.bufferedAmount > 256 * 1024) return;
    ensureStarted();
    const out = new Uint8Array(12 + payload.byteLength);
    const view = new DataView(out.buffer);
    view.setUint32(0, flags, true);
    view.setUint32(4, timestamp >>> 0, true);
    view.setUint32(8, 0, true); // frames_lost (bridge infers from RTP timestamp gaps)
    out.set(payload, 12);
    ws.send(out.buffer);
    framesSent += 1;
  }

  function sendEncodedFrame(frame) {
    const payload = new Uint8Array(frame.data);
    if (payload.byteLength === 0) return;
    const isKey = frame.type === "key";
    if (isKey) {
      lastKeyframe = { data: payload.slice(), timestamp: frame.timestamp >>> 0 };
    }
    if (!wsOpen) return;
    sendRawFrame(payload, frame.timestamp >>> 0, isKey ? FLAG_KEYFRAME : 0);
  }

  function attachVideoReceiver(receiver) {
    if (receiver.__smoovplayTapped) return;
    receiver.__smoovplayTapped = true;
    videoReceiver = receiver;
    applyLowLatencyReceiver(receiver, "video");
    if (typeof receiver.createEncodedStreams !== "function") {
      log("receiver.createEncodedStreams() is not available in this browser");
      return;
    }
    try {
      const { readable, writable } = receiver.createEncodedStreams();
      const transform = new TransformStream({
        transform(encodedFrame, controller) {
          sendEncodedFrame(encodedFrame);
          if (!wsOpen || !skipDecode() || encodedFrame.type === "key") {
            controller.enqueue(encodedFrame);
          }
        },
      });
      readable.pipeThrough(transform).pipeTo(writable).catch((err) => {
        log("encoded stream ended:", err && err.message ? err.message : err);
      });
      log("attached WebRTC encoded video stream tap");
      connect();
    } catch (err) {
      log("failed to attach encoded stream:", err);
    }
  }

  function passThroughAudioReceiver(receiver) {
    if (!receiver || receiver.__smoovplayAudioTapped) return;
    receiver.__smoovplayAudioTapped = true;
    audioReceiver = receiver;
    applyLowLatencyReceiver(receiver, "audio");
    if (typeof receiver.createEncodedStreams !== "function") return;
    try {
      const { readable, writable } = receiver.createEncodedStreams();
      readable.pipeTo(writable).catch((err) => {
        log("encoded audio stream ended:", err && err.message ? err.message : err);
      });
      log("attached WebRTC encoded audio pass-through");
    } catch (err) {
      log("failed to pass through encoded audio stream:", err);
    }
  }

  function detachAudio() {
    directAudioActive = false;
    audioDelayNode = null;
    audioPc = null;
    if (audioPuller) {
      try {
        audioPuller.pause();
        audioPuller.srcObject = null;
        if (audioPuller.parentNode) audioPuller.parentNode.removeChild(audioPuller);
      } catch { /* noop */ }
      audioPuller = null;
    }
    if (audioCtx) {
      try { audioCtx.close(); } catch { /* noop */ }
      audioCtx = null;
    }
  }

  function attachAudioTrack(track, pc) {
    if (!directAudioEnabled()) return;
    detachAudio();
    try {
      const stream = new MediaStream([track]);
      // Play the WebRTC audio track directly on a dedicated <audio> element so it stays
      // completely independent of <video> (Skip browser decode won't stall A/V sync) and
      // bypasses play.xbox.com starting muted.
      const el = document.createElement("audio");
      el.__smoovplayAudio = true;
      el.autoplay = true;
      el.muted = false;
      el.volume = 1.0;
      el.srcObject = stream;
      el.style.display = "none";
      if (document.body) {
        document.body.appendChild(el);
      } else if (document.documentElement) {
        document.documentElement.appendChild(el);
      }
      audioPuller = el;
      audioPc = pc;
      window.__smoovplayAudio = el;

      if (window.SMOOVPLAY_AUDIO_SYNC) {
        const AudioCtx = window.AudioContext || window.webkitAudioContext;
        if (AudioCtx) {
          try {
            const ctx = new AudioCtx();
            const src = ctx.createMediaStreamSource(stream);
            const delay = ctx.createDelay(0.5);
            delay.delayTime.value = currentAudioDelaySec;
            src.connect(delay);
            delay.connect(ctx.destination);
            audioCtx = ctx;
            audioDelayNode = delay;
            el.muted = true;
            if (ctx.state === "suspended") {
              ctx.resume().catch(() => {});
            }
            log("routed WebRTC audio via AudioContext + DelayNode");
          } catch (err) {
            log("AudioContext failed, using dedicated <audio>:", err);
            el.muted = false;
            el.volume = 1.0;
          }
        }
      } else {
        log("routed WebRTC audio via dedicated <audio> element (independent of <video>)");
      }

      directAudioActive = true;
      el.play().catch((err) => log("dedicated <audio> play() failed:", err));
      muteAllPageMedia();

      track.addEventListener("unmute", () => {
        resumeAudioIfNeeded();
        muteAllPageMedia();
      });
      track.addEventListener("ended", () => {
        if (audioPuller === el) {
          detachAudio();
        }
      });
    } catch (err) {
      log("direct audio routing failed:", err);
    }
  }

  const NativePC = window.RTCPeerConnection;
  if (!NativePC) {
    log("RTCPeerConnection not present");
    return;
  }

  function SmoovPlayRTCPeerConnection(config, constraints) {
    const nextCfg = Object.assign({}, config || {}, { encodedInsertableStreams: true });
    const pc = constraints !== undefined ? new NativePC(nextCfg, constraints) : new NativePC(nextCfg);
    pc.addEventListener("track", (ev) => {
      if (ev.track && ev.track.kind === "video" && ev.receiver) {
        videoPc = pc;
        attachVideoReceiver(ev.receiver);
        startStatsLogging(pc);
      } else if (ev.track && ev.track.kind === "audio") {
        if (ev.receiver) passThroughAudioReceiver(ev.receiver);
        attachAudioTrack(ev.track, pc);
      }
    });
    pc.addEventListener("connectionstatechange", () => {
      if (pc.connectionState === "closed" || pc.connectionState === "disconnected") {
        if (pc === audioPc) {
          audioReceiver = null;
          detachAudio();
        }
        if (pc === videoPc) {
          stopStatsLogging();
          videoPc = null;
          videoReceiver = null;
          lastKeyframe = null;
          if (wsOpen && started) {
            ws.send(JSON.stringify({ type: "stop" }));
            started = false;
          }
          if (ws) {
            try { ws.close(); } catch { /* noop */ }
            ws = null;
          }
        }
      }
    });
    return pc;
  }
  SmoovPlayRTCPeerConnection.prototype = NativePC.prototype;
  Object.setPrototypeOf(SmoovPlayRTCPeerConnection, NativePC);
  window.RTCPeerConnection = SmoovPlayRTCPeerConnection;

  log("hook installed; waiting for WebRTC video track ->", WS_URL);
})();
