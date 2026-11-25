// Package gssv implements the Xbox GSSV REST signaling API for both
// xHome (Remote Play) and xCloud (Cloud Gaming).
package gssv

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"strings"
	"time"
)

// StreamKind is "home" (Remote Play) or "cloud" (xCloud).
type StreamKind string

const (
	KindHome  StreamKind = "home"
	KindCloud StreamKind = "cloud"
)

// Console is a remote-play-enabled Xbox console returned by /v6/servers/home.
type Console struct {
	DeviceName  string `json:"deviceName"`
	ServerID    string `json:"serverId"`
	PowerState  string `json:"powerState"`
	ConsoleType string `json:"consoleType"`
	PlayPath    string `json:"playPath"`
}

// CloudTitle is a playable xCloud title from /v2/titles.
type CloudTitle struct {
	TitleID string `json:"titleId"`
	Details struct {
		ProductID      string `json:"productId"`
		Name           string `json:"name"`
		HasEntitlement bool   `json:"hasEntitlement"`
	} `json:"details"`
}

// SessionStartResponse is returned by POST /v5/sessions/{home|cloud}/play.
type SessionStartResponse struct {
	SessionID   string `json:"sessionId"`
	SessionPath string `json:"sessionPath"`
	State       string `json:"state"`
}

// StateResponse is returned by GET /{sessionPath}/state.
type StateResponse struct {
	State        string `json:"state"`
	ErrorDetails *struct {
		Code    string `json:"code"`
		Message string `json:"message"`
	} `json:"errorDetails,omitempty"`
}

// ICECandidate represents one WebRTC ICE candidate exchanged over GSSV HTTP.
type ICECandidate struct {
	Candidate        string `json:"candidate"`
	SDPMLineIndex    int    `json:"sdpMLineIndex"`
	SDPMid           string `json:"sdpMid"`
	UsernameFragment string `json:"usernameFragment,omitempty"`
}

// Preferences configures device spoofing & resolution tier for Xbox signaling.
type Preferences struct {
	Width  int
	Height int
	Locale string
	OSName string // "windows" for 1080p/1440p desktop tier
}

func (p Preferences) normalized() Preferences {
	if p.Width <= 0 {
		p.Width = 1920
	}
	if p.Height <= 0 {
		p.Height = 1080
	}
	if p.Locale == "" {
		p.Locale = "en-US"
	}
	if p.OSName == "" {
		p.OSName = "windows"
	}
	return p
}

// MakeDeviceInfo builds the X-MS-Device-Info header required by Microsoft for 1080p/1440p streams.
func MakeDeviceInfo(p Preferences) string {
	p = p.normalized()
	return fmt.Sprintf(
		`{"appInfo":{"env":{"clientAppId":"www.xbox.com","clientAppType":"browser","clientAppVersion":"21.1.98","clientSdkVersion":"8.5.3","httpEnvironment":"prod","sdkInstallId":""}},"dev":{"hw":{"make":"Microsoft","model":"unknown","sdktype":"web"},"os":{"name":"%s","ver":"22631.2715","platform":"desktop"},"displayInfo":{"dimensions":{"widthInPixels":%d,"heightInPixels":%d},"pixelDensity":{"dpiX":2,"dpiY":2}},"browser":{"browserName":"chrome","browserVersion":"119.0"}}}`,
		p.OSName, p.Width, p.Height,
	)
}

// Client talks to one GSSV host (xhome or xcloud) with a gsToken.
type Client struct {
	BaseHost   string
	GSToken    string
	Prefs      Preferences
	HTTP       *http.Client
	deviceInfo string
}

// NewClient creates a GSSV client.
func NewClient(baseHost, gsToken string, prefs Preferences) *Client {
	p := prefs.normalized()
	return &Client{
		BaseHost:   strings.TrimRight(baseHost, "/"),
		GSToken:    gsToken,
		Prefs:      p,
		HTTP:       &http.Client{Timeout: 30 * time.Second},
		deviceInfo: MakeDeviceInfo(p),
	}
}

// ListConsoles fetches the user's Xbox consoles via GET /v6/servers/home.
func (c *Client) ListConsoles(ctx context.Context) ([]Console, error) {
	var resp struct {
		Results []Console `json:"results"`
	}
	if err := c.doJSON(ctx, http.MethodGet, "/v6/servers/home", nil, &resp); err != nil {
		return nil, err
	}
	return resp.Results, nil
}

// ListCloudTitles fetches playable titles via GET /v2/titles.
func (c *Client) ListCloudTitles(ctx context.Context) ([]CloudTitle, error) {
	var resp struct {
		Results []CloudTitle `json:"results"`
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodGet, c.BaseHost+"/v2/titles", nil)
	if err != nil {
		return nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Authorization", "Bearer "+c.GSToken)
	httpResp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer httpResp.Body.Close()
	body, _ := io.ReadAll(httpResp.Body)
	if httpResp.StatusCode < 200 || httpResp.StatusCode >= 300 {
		return nil, fmt.Errorf("GET /v2/titles HTTP %d: %s", httpResp.StatusCode, string(body))
	}
	if err := json.Unmarshal(body, &resp); err != nil {
		return nil, err
	}
	return resp.Results, nil
}

// StartSession initiates an xHome (targetID=serverId) or xCloud (targetID=titleId) session.
func (c *Client) StartSession(ctx context.Context, kind StreamKind, targetID string) (*SessionStartResponse, error) {
	p := c.Prefs.normalized()
	body := map[string]any{
		"clientSessionId":   "",
		"titleId":           "",
		"systemUpdateGroup": "",
		"serverId":          "",
		"settings": map[string]any{
			"nanoVersion":                  "V3;WebrtcTransport.dll",
			"enableOptionalDataCollection": false,
			"enableTextToSpeech":           false,
			"highContrast":                 0,
			"locale":                       p.Locale,
			"useIceConnection":             false,
			"timezoneOffsetMinutes":        120,
			"sdkType":                      "web",
			"osName":                       p.OSName,
		},
		"fallbackRegionNames": []string{},
	}
	if kind == KindCloud {
		body["titleId"] = targetID
	} else {
		body["serverId"] = targetID
	}
	var out SessionStartResponse
	if err := c.doJSON(ctx, http.MethodPost, fmt.Sprintf("/v5/sessions/%s/play", kind), body, &out); err != nil {
		return nil, err
	}
	out.SessionPath = "/" + strings.TrimLeft(out.SessionPath, "/")
	return &out, nil
}

// WaitForState polls /{sessionPath}/state until one of targetStates is reached.
func (c *Client) WaitForState(ctx context.Context, sessionPath string, targetStates []string, pollInterval time.Duration) (string, error) {
	want := make(map[string]bool, len(targetStates))
	for _, s := range targetStates {
		want[s] = true
	}
	for {
		var st StateResponse
		if err := c.doJSON(ctx, http.MethodGet, sessionPath+"/state", nil, &st); err != nil {
			return "", err
		}
		if st.ErrorDetails != nil && st.ErrorDetails.Code != "" {
			return st.State, fmt.Errorf("session error %s: %s", st.ErrorDetails.Code, st.ErrorDetails.Message)
		}
		if want[st.State] {
			return st.State, nil
		}
		select {
		case <-ctx.Done():
			return st.State, ctx.Err()
		case <-time.After(pollInterval):
		}
	}
}

// SendMSALAuth posts the xCloud Long Play Token to /{sessionPath}/connect.
func (c *Client) SendMSALAuth(ctx context.Context, sessionPath, lptToken string) error {
	return c.doJSON(ctx, http.MethodPost, sessionPath+"/connect", map[string]string{"userToken": lptToken}, nil)
}

// ExchangeSDP posts the local SDP offer and polls until the server's SDP answer arrives.
func (c *Client) ExchangeSDP(ctx context.Context, sessionPath, localSDP string) (string, error) {
	body := map[string]any{
		"messageType": "offer",
		"sdp":         localSDP,
		"requestId":   "1",
		"configuration": map[string]any{
			"chatConfiguration": map[string]any{
				"bytesPerSample":         2,
				"expectedClipDurationMs": 20,
				"format":                 map[string]string{"codec": "opus", "container": "webm"},
				"numChannels":            1,
				"sampleFrequencyHz":      24000,
			},
			"chat":            map[string]int{"minVersion": 1, "maxVersion": 1},
			"control":         map[string]int{"minVersion": 1, "maxVersion": 3},
			"input":           map[string]int{"minVersion": 1, "maxVersion": 9},
			"message":         map[string]int{"minVersion": 1, "maxVersion": 1},
			"reliableinput":   map[string]int{"minVersion": 9, "maxVersion": 9},
			"unreliableinput": map[string]int{"minVersion": 9, "maxVersion": 9},
		},
	}
	status, data, err := c.doRaw(ctx, http.MethodPost, sessionPath+"/sdp", body)
	if err != nil {
		return "", err
	}
	if status != http.StatusNoContent && len(data) > 0 {
		if sdp, ok := ExtractSDP(data); ok {
			return sdp, nil
		}
	}
	for i := 0; i < 60; i++ {
		select {
		case <-ctx.Done():
			return "", ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
		st, d, err := c.doRaw(ctx, http.MethodGet, sessionPath+"/sdp", nil)
		if err != nil {
			return "", err
		}
		if st == http.StatusNoContent || len(d) == 0 {
			continue
		}
		if sdp, ok := ExtractSDP(d); ok {
			return sdp, nil
		}
	}
	return "", errors.New("timed out waiting for server SDP answer")
}

// ExchangeICE posts local ICE candidates and polls until the server returns remote candidates.
func (c *Client) ExchangeICE(ctx context.Context, sessionPath string, local []ICECandidate) ([]ICECandidate, error) {
	encoded := make([]string, 0, len(local))
	seen := make(map[string]bool, len(local))
	for _, cand := range local {
		norm, ok := NormalizeOutboundCandidate(cand.Candidate)
		if !ok {
			continue
		}
		key := fmt.Sprintf("%s|%d|%s", cand.SDPMid, cand.SDPMLineIndex, norm)
		if seen[key] {
			continue
		}
		seen[key] = true
		wire := ICECandidate{
			Candidate:        norm,
			SDPMLineIndex:    cand.SDPMLineIndex,
			SDPMid:           cand.SDPMid,
			UsernameFragment: cand.UsernameFragment,
		}
		b, err := json.Marshal(wire)
		if err != nil {
			continue
		}
		encoded = append(encoded, string(b))
	}
	if _, _, err := c.doRaw(ctx, http.MethodPost, sessionPath+"/ice", map[string]any{"candidates": encoded}); err != nil {
		return nil, err
	}
	for i := 0; i < 30; i++ {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(500 * time.Millisecond):
		}
		st, d, err := c.doRaw(ctx, http.MethodGet, sessionPath+"/ice", nil)
		if err != nil {
			return nil, err
		}
		if st == http.StatusNoContent || len(d) == 0 {
			continue
		}
		if cands := ExtractICECandidates(d); len(cands) > 0 {
			return cands, nil
		}
	}
	return nil, errors.New("timed out waiting for server ICE candidates")
}

// Keepalive sends POST /{sessionPath}/keepalive.
func (c *Client) Keepalive(ctx context.Context, sessionPath string) error {
	_, _, err := c.doRaw(ctx, http.MethodPost, sessionPath+"/keepalive", map[string]any{})
	return err
}

// StopSession sends DELETE /{sessionPath}.
func (c *Client) StopSession(ctx context.Context, sessionPath string) error {
	_, _, err := c.doRaw(ctx, http.MethodDelete, sessionPath, nil)
	return err
}

func (c *Client) doJSON(ctx context.Context, method, path string, body any, dst any) error {
	_, data, err := c.doRaw(ctx, method, path, body)
	if err != nil {
		return err
	}
	if dst == nil || len(data) == 0 {
		return nil
	}
	return json.Unmarshal(data, dst)
}

func (c *Client) doRaw(ctx context.Context, method, path string, body any) (int, []byte, error) {
	var r io.Reader
	if body != nil {
		b, err := json.Marshal(body)
		if err != nil {
			return 0, nil, err
		}
		r = bytes.NewReader(b)
	}
	req, err := http.NewRequestWithContext(ctx, method, c.BaseHost+path, r)
	if err != nil {
		return 0, nil, err
	}
	req.Header.Set("Accept", "application/json")
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("X-Gssv-Client", "XboxComBrowser")
	req.Header.Set("X-MS-Device-Info", c.deviceInfo)
	req.Header.Set("Authorization", "Bearer "+c.GSToken)
	req.Header.Set("User-Agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")

	resp, err := c.HTTP.Do(req)
	if err != nil {
		return 0, nil, err
	}
	defer resp.Body.Close()
	data, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return resp.StatusCode, data, fmt.Errorf("%s %s HTTP %d: %s", method, path, resp.StatusCode, string(data))
	}
	return resp.StatusCode, data, nil
}
