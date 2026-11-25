package gssv

import (
	"encoding/json"
	"fmt"
	"strconv"
	"strings"
)

// PrioritizeH264AndBitrate reorders the video m-line payload types to put H.264 High/Main/Baseline
// first and injects b=AS / b=TIAS bitrate caps so Xbox negotiates high-bitrate H.264.
func PrioritizeH264AndBitrate(sdp string, maxKbps int) string {
	sep := "\r\n"
	if !strings.Contains(sdp, "\r\n") {
		sep = "\n"
	}
	lines := strings.Split(sdp, sep)
	h264PTs := map[string]string{} // pt -> fmtp line (lowercased)
	for _, l := range lines {
		if strings.HasPrefix(l, "a=rtpmap:") && strings.Contains(strings.ToUpper(l), "H264/90000") {
			rest := strings.TrimPrefix(l, "a=rtpmap:")
			if sp := strings.IndexByte(rest, ' '); sp > 0 {
				h264PTs[rest[:sp]] = ""
			}
		}
	}
	for _, l := range lines {
		if strings.HasPrefix(l, "a=fmtp:") {
			rest := strings.TrimPrefix(l, "a=fmtp:")
			if sp := strings.IndexByte(rest, ' '); sp > 0 {
				pt := rest[:sp]
				if _, ok := h264PTs[pt]; ok {
					h264PTs[pt] = strings.ToLower(rest[sp+1:])
				}
			}
		}
	}

	out := make([]string, 0, len(lines)+4)
	inVideo := false
	for _, l := range lines {
		if strings.HasPrefix(l, "m=") {
			inVideo = strings.HasPrefix(l, "m=video ")
			if inVideo && len(h264PTs) > 0 {
				fields := strings.Fields(l)
				if len(fields) > 3 {
					head := fields[:3]
					pts := fields[3:]
					var high, mainP, base, other []string
					for _, pt := range pts {
						fmtp, isH264 := h264PTs[pt]
						if !isH264 {
							other = append(other, pt)
							continue
						}
						switch {
						case strings.Contains(fmtp, "profile-level-id=64"):
							high = append(high, pt)
						case strings.Contains(fmtp, "profile-level-id=4d"):
							mainP = append(mainP, pt)
						default:
							base = append(base, pt)
						}
					}
					reordered := append(append(append(append(head, high...), mainP...), base...), other...)
					l = strings.Join(reordered, " ")
				}
			}
			out = append(out, l)
			continue
		}
		if inVideo && maxKbps > 0 && strings.HasPrefix(l, "c=") {
			out = append(out, l)
			out = append(out, fmt.Sprintf("b=AS:%d", maxKbps))
			out = append(out, fmt.Sprintf("b=TIAS:%d", maxKbps*1000))
			continue
		}
		if inVideo && (strings.HasPrefix(l, "b=AS:") || strings.HasPrefix(l, "b=TIAS:")) {
			continue
		}
		out = append(out, l)
	}
	return strings.Join(out, sep)
}

// NormalizeOutboundCandidate strips leading "a=" and rejects empty or end-of-candidates lines.
func NormalizeOutboundCandidate(cand string) (string, bool) {
	c := strings.TrimSpace(cand)
	c = strings.TrimPrefix(c, "a=")
	if c == "" || c == "end-of-candidates" || !strings.HasPrefix(c, "candidate:") {
		return "", false
	}
	return c, true
}

// ExtractSDP unwraps GSSV's nested JSON/string SDP response envelopes.
func ExtractSDP(data []byte) (string, bool) {
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return "", false
	}
	return extractSDPAny(raw)
}

func extractSDPAny(v any) (string, bool) {
	switch x := v.(type) {
	case map[string]any:
		if s, ok := x["sdp"].(string); ok && s != "" {
			return s, true
		}
		if ex, ok := x["exchangeResponse"]; ok {
			return extractSDPAny(ex)
		}
	case string:
		var nested any
		if json.Unmarshal([]byte(x), &nested) == nil {
			return extractSDPAny(nested)
		}
	}
	return "", false
}

// ExtractICECandidates unwraps GSSV's nested JSON/string ICE candidate responses and strips "a=".
func ExtractICECandidates(data []byte) []ICECandidate {
	var raw any
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil
	}
	return extractICEAny(raw)
}

func extractICEAny(v any) []ICECandidate {
	switch x := v.(type) {
	case []any:
		var out []ICECandidate
		for _, el := range x {
			if m, ok := el.(map[string]any); ok {
				if c, ok2 := decodeICEMap(m); ok2 {
					out = append(out, c)
				}
			}
		}
		return out
	case map[string]any:
		for _, key := range []string{"exchangeResponse", "iceCandidates", "candidates"} {
			if sub, ok := x[key]; ok {
				if cands := extractICEAny(sub); len(cands) > 0 {
					return cands
				}
			}
		}
		if c, ok := decodeICEMap(x); ok {
			return []ICECandidate{c}
		}
	case string:
		var nested any
		if json.Unmarshal([]byte(x), &nested) == nil {
			return extractICEAny(nested)
		}
	}
	return nil
}

func decodeICEMap(m map[string]any) (ICECandidate, bool) {
	cand, _ := m["candidate"].(string)
	cand = strings.TrimSpace(strings.TrimPrefix(strings.TrimSpace(cand), "a="))
	if cand == "" || cand == "end-of-candidates" || !strings.HasPrefix(cand, "candidate:") {
		return ICECandidate{}, false
	}
	mid, _ := m["sdpMid"].(string)
	ufrag, _ := m["usernameFragment"].(string)
	idx := 0
	switch v := m["sdpMLineIndex"].(type) {
	case float64:
		idx = int(v)
	case string:
		if n, err := strconv.Atoi(v); err == nil {
			idx = n
		}
	}
	return ICECandidate{
		Candidate:        cand,
		SDPMLineIndex:    idx,
		SDPMid:           mid,
		UsernameFragment: ufrag,
	}, true
}
