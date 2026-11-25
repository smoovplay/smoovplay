// Package auth implements Microsoft Device Code OAuth, token caching,
// Xbox Live XSTS exchange, GSSV stream token (gsToken) exchange for both
// xHome (Remote Play) and xCloud, and xCloud Long Play Token (LPT) acquisition.
package auth

import (
	"bytes"
	"context"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/url"
	"os"
	"path/filepath"
	"strings"
	"time"
)

const (
	ClientID   = "1f907974-e22b-4810-a9de-d9647380c97e"
	OAuthScope = "xboxlive.signin openid profile offline_access"

	deviceCodeURL = "https://login.microsoftonline.com/consumers/oauth2/v2.0/devicecode"
	tokenURL      = "https://login.microsoftonline.com/consumers/oauth2/v2.0/token"
	liveTokenURL  = "https://login.live.com/oauth20_token.srf"
	xblUserURL    = "https://user.auth.xboxlive.com/user/authenticate"
	xstsAuthURL   = "https://xsts.auth.xboxlive.com/xsts/authorize"

	xhomeLoginHost   = "xhome.gssv-play-prod.xboxlive.com"
	xgpuwebLoginHost = "xgpuweb.gssv-play-prod.xboxlive.com"
	f2pLoginHost     = "xgpuwebf2p.gssv-play-prod.xboxlive.com"
)

// Region describes one GSSV streaming region returned during login.
type Region struct {
	Name      string `json:"name"`
	BaseURI   string `json:"baseUri"`
	IsDefault bool   `json:"isDefault"`
}

// StreamToken holds one GSSV offering's bearer token and resolved host.
type StreamToken struct {
	GSToken string   `json:"gsToken"`
	Host    string   `json:"host"`
	Regions []Region `json:"regions,omitempty"`
}

// Tokens holds persisted Microsoft OAuth + derived GSSV tokens.
type Tokens struct {
	RefreshToken string       `json:"refresh_token"`
	AccessToken  string       `json:"access_token,omitempty"`
	AccessExpiry time.Time    `json:"access_expiry,omitempty"`
	XHome        *StreamToken `json:"xhome,omitempty"`
	XCloud       *StreamToken `json:"xcloud,omitempty"`
	XCloudF2P    *StreamToken `json:"xcloud_f2p,omitempty"`
	StreamExpiry time.Time    `json:"stream_expiry,omitempty"`
}

// DeviceCodeResponse is returned by Microsoft's device code endpoint.
type DeviceCodeResponse struct {
	DeviceCode              string `json:"device_code"`
	UserCode                string `json:"user_code"`
	VerificationURI         string `json:"verification_uri"`
	VerificationURIComplete string `json:"verification_uri_complete"`
	ExpiresIn               int    `json:"expires_in"`
	Interval                int    `json:"interval"`
	Message                 string `json:"message"`
}

// DefaultTokenPath returns %LOCALAPPDATA%\SmoovPlay\xbox_tokens.json (or ~/.config/smoovplay/xbox_tokens.json).
func DefaultTokenPath() string {
	if dir := os.Getenv("LOCALAPPDATA"); dir != "" {
		return filepath.Join(dir, "SmoovPlay", "xbox_tokens.json")
	}
	if cfg, err := os.UserConfigDir(); err == nil {
		return filepath.Join(cfg, "smoovplay", "xbox_tokens.json")
	}
	return "xbox_tokens.json"
}

// LoadTokens loads cached tokens from path, or returns nil, nil if not found.
func LoadTokens(path string) (*Tokens, error) {
	data, err := os.ReadFile(path)
	if errors.Is(err, os.ErrNotExist) {
		return nil, nil
	}
	if err != nil {
		return nil, err
	}
	var t Tokens
	if err := json.Unmarshal(data, &t); err != nil {
		return nil, err
	}
	return &t, nil
}

// SaveTokens writes tokens to path with 0600 permissions.
func SaveTokens(path string, t *Tokens) error {
	if err := os.MkdirAll(filepath.Dir(path), 0o700); err != nil {
		return err
	}
	data, err := json.MarshalIndent(t, "", "  ")
	if err != nil {
		return err
	}
	return os.WriteFile(path, data, 0o600)
}

// Client handles Microsoft / Xbox Live / GSSV authentication.
type Client struct {
	HTTP *http.Client
}

// NewClient returns an auth client with sensible timeouts.
func NewClient() *Client {
	return &Client{HTTP: &http.Client{Timeout: 30 * time.Second}}
}

// StartDeviceCode initiates the OAuth device code flow (no browser embedding required).
func (c *Client) StartDeviceCode(ctx context.Context) (*DeviceCodeResponse, error) {
	form := url.Values{
		"client_id": {ClientID},
		"scope":     {OAuthScope},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, deviceCodeURL, strings.NewReader(form.Encode()))
	if err != nil {
		return nil, err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return nil, err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	if resp.StatusCode != http.StatusOK {
		return nil, fmt.Errorf("devicecode HTTP %d: %s", resp.StatusCode, string(body))
	}
	var out DeviceCodeResponse
	if err := json.Unmarshal(body, &out); err != nil {
		return nil, err
	}
	if out.Interval <= 0 {
		out.Interval = 5
	}
	return &out, nil
}

type oauthTokenResp struct {
	AccessToken  string `json:"access_token"`
	RefreshToken string `json:"refresh_token"`
	ExpiresIn    int    `json:"expires_in"`
	Error        string `json:"error"`
	ErrorDesc    string `json:"error_description"`
}

// PollDeviceCode polls until the user completes login or the code expires.
func (c *Client) PollDeviceCode(ctx context.Context, dc *DeviceCodeResponse) (*Tokens, error) {
	interval := time.Duration(dc.Interval) * time.Second
	deadline := time.Now().Add(time.Duration(dc.ExpiresIn) * time.Second)
	for time.Now().Before(deadline) {
		select {
		case <-ctx.Done():
			return nil, ctx.Err()
		case <-time.After(interval):
		}
		form := url.Values{
			"grant_type":  {"urn:ietf:params:oauth:grant-type:device_code"},
			"device_code": {dc.DeviceCode},
			"client_id":   {ClientID},
		}
		req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
		if err != nil {
			return nil, err
		}
		req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
		resp, err := c.HTTP.Do(req)
		if err != nil {
			return nil, err
		}
		body, _ := io.ReadAll(resp.Body)
		resp.Body.Close()

		var tr oauthTokenResp
		_ = json.Unmarshal(body, &tr)
		if resp.StatusCode == http.StatusOK && tr.AccessToken != "" {
			return &Tokens{
				RefreshToken: tr.RefreshToken,
				AccessToken:  tr.AccessToken,
				AccessExpiry: time.Now().Add(time.Duration(tr.ExpiresIn-60) * time.Second),
			}, nil
		}
		switch tr.Error {
		case "authorization_pending":
			continue
		case "slow_down":
			interval += 2 * time.Second
			continue
		default:
			return nil, fmt.Errorf("device code auth failed (%s): %s", tr.Error, tr.ErrorDesc)
		}
	}
	return nil, errors.New("device code expired before sign-in completed")
}

// RefreshMSA refreshes the MSA access token using the stored refresh token.
func (c *Client) RefreshMSA(ctx context.Context, tokens *Tokens) error {
	if tokens == nil || tokens.RefreshToken == "" {
		return errors.New("no refresh token cached; run login first")
	}
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {tokens.RefreshToken},
		"client_id":     {ClientID},
		"scope":         {OAuthScope},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, tokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var tr oauthTokenResp
	if err := json.Unmarshal(body, &tr); err != nil || resp.StatusCode != http.StatusOK || tr.AccessToken == "" {
		return fmt.Errorf("refresh token HTTP %d: %s", resp.StatusCode, string(body))
	}
	tokens.AccessToken = tr.AccessToken
	if tr.RefreshToken != "" {
		tokens.RefreshToken = tr.RefreshToken
	}
	tokens.AccessExpiry = time.Now().Add(time.Duration(tr.ExpiresIn-60) * time.Second)
	return nil
}

// FetchLongPlayToken fetches the short-lived MSAL/LPT token required for xCloud /connect.
func (c *Client) FetchLongPlayToken(ctx context.Context, refreshToken string) (string, error) {
	form := url.Values{
		"grant_type":    {"refresh_token"},
		"refresh_token": {refreshToken},
		"client_id":     {ClientID},
		"scope":         {"service::http://Passport.NET/purpose::PURPOSE_XBOX_CLOUD_CONSOLE_TRANSFER_TOKEN"},
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, liveTokenURL, strings.NewReader(form.Encode()))
	if err != nil {
		return "", err
	}
	req.Header.Set("Content-Type", "application/x-www-form-urlencoded")
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return "", err
	}
	defer resp.Body.Close()
	body, _ := io.ReadAll(resp.Body)
	var tr oauthTokenResp
	if err := json.Unmarshal(body, &tr); err != nil || resp.StatusCode != http.StatusOK || tr.AccessToken == "" {
		return "", fmt.Errorf("LPT HTTP %d: %s", resp.StatusCode, string(body))
	}
	return tr.AccessToken, nil
}

// EnsureStreamTokens refreshes MSA and exchanges XBL -> XSTS -> GSSV gsTokens if expired.
func (c *Client) EnsureStreamTokens(ctx context.Context, tokens *Tokens) error {
	if tokens.XHome != nil && time.Now().Before(tokens.StreamExpiry) {
		return nil
	}
	if tokens.AccessToken == "" || time.Now().After(tokens.AccessExpiry) {
		if err := c.RefreshMSA(ctx, tokens); err != nil {
			return err
		}
	}
	userToken, err := c.getXBLUserToken(ctx, tokens.AccessToken)
	if err != nil {
		return err
	}
	gssvXSTS, err := c.getXSTSToken(ctx, userToken, "http://gssv.xboxlive.com/")
	if err != nil {
		return err
	}
	xhome, err := c.getStreamToken(ctx, xhomeLoginHost, gssvXSTS, "xhome")
	if err != nil {
		return fmt.Errorf("xhome login: %w", err)
	}
	tokens.XHome = xhome
	// xCloud offerings are best-effort (user may only have Remote Play without Game Pass Ultimate).
	if xc, err := c.getStreamToken(ctx, xgpuwebLoginHost, gssvXSTS, "xgpuweb"); err == nil {
		tokens.XCloud = xc
	}
	if f2p, err := c.getStreamToken(ctx, f2pLoginHost, gssvXSTS, "xgpuwebf2p"); err == nil {
		tokens.XCloudF2P = f2p
	}
	tokens.StreamExpiry = time.Now().Add(50 * time.Minute)
	return nil
}

func (c *Client) getXBLUserToken(ctx context.Context, msaAccessToken string) (string, error) {
	payload := map[string]any{
		"Properties": map[string]any{
			"AuthMethod": "RPS",
			"SiteName":   "user.auth.xboxlive.com",
			"RpsTicket":  "d=" + msaAccessToken,
		},
		"RelyingParty": "http://auth.xboxlive.com",
		"TokenType":    "JWT",
	}
	var out struct {
		Token string `json:"Token"`
	}
	if err := c.postJSON(ctx, xblUserURL, payload, nil, &out); err != nil {
		return "", fmt.Errorf("xbl user auth: %w", err)
	}
	if out.Token == "" {
		return "", errors.New("xbl user auth returned empty token")
	}
	return out.Token, nil
}

func (c *Client) getXSTSToken(ctx context.Context, userToken, relyingParty string) (string, error) {
	payload := map[string]any{
		"Properties": map[string]any{
			"SandboxId":  "RETAIL",
			"UserTokens": []string{userToken},
		},
		"RelyingParty": relyingParty,
		"TokenType":    "JWT",
	}
	var out struct {
		Token string `json:"Token"`
	}
	if err := c.postJSON(ctx, xstsAuthURL, payload, nil, &out); err != nil {
		return "", fmt.Errorf("xsts (%s): %w", relyingParty, err)
	}
	if out.Token == "" {
		return "", errors.New("xsts returned empty token")
	}
	return out.Token, nil
}

func (c *Client) getStreamToken(ctx context.Context, host, xstsToken, offering string) (*StreamToken, error) {
	payload := map[string]string{
		"token":      xstsToken,
		"offeringId": offering,
	}
	var resp struct {
		GSToken          string `json:"gsToken"`
		OfferingSettings struct {
			Regions []Region `json:"regions"`
		} `json:"offeringSettings"`
	}
	headers := map[string]string{"x-gssv-client": "XboxComBrowser"}
	if err := c.postJSON(ctx, "https://"+host+"/v2/login/user", payload, headers, &resp); err != nil {
		return nil, err
	}
	if resp.GSToken == "" {
		return nil, errors.New("empty gsToken")
	}
	resolved := "https://" + host
	for _, r := range resp.OfferingSettings.Regions {
		if r.IsDefault && r.BaseURI != "" {
			resolved = strings.TrimRight(r.BaseURI, "/")
			break
		}
	}
	return &StreamToken{
		GSToken: resp.GSToken,
		Host:    resolved,
		Regions: resp.OfferingSettings.Regions,
	}, nil
}

func (c *Client) postJSON(ctx context.Context, endpoint string, body any, headers map[string]string, dst any) error {
	raw, err := json.Marshal(body)
	if err != nil {
		return err
	}
	req, err := http.NewRequestWithContext(ctx, http.MethodPost, endpoint, bytes.NewReader(raw))
	if err != nil {
		return err
	}
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("Accept", "application/json")
	for k, v := range headers {
		req.Header.Set(k, v)
	}
	resp, err := c.HTTP.Do(req)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	respBody, _ := io.ReadAll(resp.Body)
	if resp.StatusCode < 200 || resp.StatusCode >= 300 {
		return fmt.Errorf("HTTP %d: %s", resp.StatusCode, string(respBody))
	}
	return json.Unmarshal(respBody, dst)
}
