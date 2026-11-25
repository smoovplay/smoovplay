package gssv

import (
	"strings"
	"testing"
)

func TestPrioritizeH264AndBitrate(t *testing.T) {
	sdp := strings.Join([]string{
		"v=0",
		"m=video 9 UDP/TLS/RTP/SAVPF 96 97 98 99",
		"c=IN IP4 0.0.0.0",
		"a=rtpmap:96 VP8/90000",
		"a=rtpmap:97 H264/90000",
		"a=fmtp:97 level-asymmetry-allowed=1;profile-level-id=42e01f",
		"a=rtpmap:98 H264/90000",
		"a=fmtp:98 level-asymmetry-allowed=1;profile-level-id=64002a",
		"a=rtpmap:99 H264/90000",
		"a=fmtp:99 level-asymmetry-allowed=1;profile-level-id=4d002a",
	}, "\r\n")

	out := PrioritizeH264AndBitrate(sdp, 20000)
	if !strings.Contains(out, "m=video 9 UDP/TLS/RTP/SAVPF 98 99 97 96") {
		t.Fatalf("expected H.264 High(98) > Main(99) > Base(97) > VP8(96), got:\n%s", out)
	}
	if !strings.Contains(out, "b=AS:20000\r\nb=TIAS:20000000") {
		t.Fatalf("missing injected bitrate lines:\n%s", out)
	}
}

func TestExtractSDPAndICEStripsAPrefix(t *testing.T) {
	sdpResp := []byte(`{"exchangeResponse":"{\"sdp\":\"v=0\\r\\nm=video 9\"}"}`)
	sdp, ok := ExtractSDP(sdpResp)
	if !ok || !strings.HasPrefix(sdp, "v=0") {
		t.Fatalf("ExtractSDP = (%q, %v)", sdp, ok)
	}

	iceResp := []byte(`{"exchangeResponse":"[{\"candidate\":\"a=candidate:1 1 udp 2122260223 192.168.1.50 50000 typ host\",\"sdpMLineIndex\":0,\"sdpMid\":\"0\"}]"}`)
	cands := ExtractICECandidates(iceResp)
	if len(cands) != 1 || cands[0].Candidate != "candidate:1 1 udp 2122260223 192.168.1.50 50000 typ host" {
		t.Fatalf("unexpected ICE candidates: %+v", cands)
	}
}
