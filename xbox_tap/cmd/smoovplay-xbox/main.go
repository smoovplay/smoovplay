// smoovplay-xbox: native zero-jitter-buffer Xbox Remote Play & xCloud WebRTC client
// that writes RTP-reassembled H.264 straight into SmoovPlay's FrameTap shared memory,
// forwards XInput at 125 Hz, and plays pure-Go decoded Opus audio in sync with SmoovPlay.
package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"os"
	"os/signal"
	"strings"
	"syscall"
	"time"

	"github.com/noveblue53/smoovplay/xbox_tap/internal/auth"
	"github.com/noveblue53/smoovplay/xbox_tap/internal/frametap"
	"github.com/noveblue53/smoovplay/xbox_tap/internal/gssv"
	"github.com/noveblue53/smoovplay/xbox_tap/internal/session"
)

func main() {
	log.SetFlags(log.Ltime)
	if len(os.Args) < 2 {
		printUsage()
		os.Exit(1)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), os.Interrupt, syscall.SIGTERM)
	defer cancel()

	var err error
	switch os.Args[1] {
	case "login":
		err = cmdLogin(ctx, os.Args[2:])
	case "consoles":
		err = cmdConsoles(ctx, os.Args[2:])
	case "titles":
		err = cmdTitles(ctx, os.Args[2:])
	case "play":
		err = cmdPlay(ctx, os.Args[2:])
	case "-h", "--help", "help":
		printUsage()
		return
	default:
		fmt.Fprintf(os.Stderr, "unknown command %q\n\n", os.Args[1])
		printUsage()
		os.Exit(2)
	}
	if err != nil {
		fmt.Fprintf(os.Stderr, "error: %v\n", err)
		os.Exit(1)
	}
}

func printUsage() {
	fmt.Println(`smoovplay-xbox — native Xbox Remote Play & xCloud client for SmoovPlay FrameTap

Commands:
  login                                  Sign in with Microsoft Device Code OAuth
  consoles                               List your Xbox consoles for Remote Play
  titles [filter]                        List entitled xCloud titles (optional substring filter)
  play [flags]                           Stream Xbox Remote Play or xCloud into FrameTap

Play flags:
  --home [console]                       Stream from your Xbox (auto-picks if only one, or match name/serverId)
  --cloud <titleId>                      Stream an xCloud title by titleId
  --res 1080p|1440p|720p                 Preferred resolution (default 1080p)
  --bitrate <kbps>                       SDP max video bitrate in kbps (default 20000)
  --tap <name>                           SmoovPlay FrameTap name (default Local\smoovplay-tap)
  --no-audio                             Disable local Opus audio playback
  --stats <duration>                     Live WebRTC RTT/fps log interval (default 2s, 0 disables)
  --tokens <path>                        Token cache file path

Example:
  smoovplay-xbox login
  smoovplay-xbox play --home
  python run.py --source tap --display --present dxgi --fullscreen`)
}

func loadAndEnsureAuth(ctx context.Context, tokenPath string) (*auth.Client, *auth.Tokens, error) {
	ac := auth.NewClient()
	tokens, err := auth.LoadTokens(tokenPath)
	if err != nil {
		return nil, nil, fmt.Errorf("load tokens: %w", err)
	}
	if tokens == nil || tokens.RefreshToken == "" {
		return nil, nil, fmt.Errorf("not signed in — run `smoovplay-xbox login` first")
	}
	if err := ac.EnsureStreamTokens(ctx, tokens); err != nil {
		return nil, nil, err
	}
	_ = auth.SaveTokens(tokenPath, tokens)
	return ac, tokens, nil
}

func cmdLogin(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("login", flag.ExitOnError)
	tokenPath := fs.String("tokens", auth.DefaultTokenPath(), "token cache file")
	_ = fs.Parse(args)

	ac := auth.NewClient()
	dc, err := ac.StartDeviceCode(ctx)
	if err != nil {
		return err
	}
	fmt.Println()
	if dc.Message != "" {
		fmt.Println(dc.Message)
	} else {
		fmt.Printf("Open %s and enter code: %s\n", dc.VerificationURI, dc.UserCode)
	}
	if dc.VerificationURIComplete != "" {
		fmt.Printf("Direct link: %s\n", dc.VerificationURIComplete)
	}
	fmt.Println("Waiting for sign-in...")

	tokens, err := ac.PollDeviceCode(ctx, dc)
	if err != nil {
		return err
	}
	if err := ac.EnsureStreamTokens(ctx, tokens); err != nil {
		return err
	}
	if err := auth.SaveTokens(*tokenPath, tokens); err != nil {
		return fmt.Errorf("save tokens: %w", err)
	}
	fmt.Printf("Signed in! Tokens saved to %s\n", *tokenPath)
	if tokens.XCloud != nil {
		fmt.Printf("  xHome:  ready (%s)\n  xCloud: ready (%s)\n", tokens.XHome.Host, tokens.XCloud.Host)
	} else {
		fmt.Printf("  xHome:  ready (%s)\n  xCloud: not available (no Game Pass Ultimate)\n", tokens.XHome.Host)
	}
	return nil
}

func cmdConsoles(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("consoles", flag.ExitOnError)
	tokenPath := fs.String("tokens", auth.DefaultTokenPath(), "token cache file")
	_ = fs.Parse(args)

	_, tokens, err := loadAndEnsureAuth(ctx, *tokenPath)
	if err != nil {
		return err
	}
	client := gssv.NewClient(tokens.XHome.Host, tokens.XHome.GSToken, gssv.Preferences{})
	consoles, err := client.ListConsoles(ctx)
	if err != nil {
		return err
	}
	if len(consoles) == 0 {
		fmt.Println("No Remote Play consoles found on this account.")
		return nil
	}
	for _, c := range consoles {
		fmt.Printf("%-24s  id=%-20s  type=%-14s  power=%s\n", c.DeviceName, c.ServerID, c.ConsoleType, c.PowerState)
	}
	return nil
}

func cmdTitles(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("titles", flag.ExitOnError)
	tokenPath := fs.String("tokens", auth.DefaultTokenPath(), "token cache file")
	_ = fs.Parse(args)

	filter := strings.ToLower(strings.TrimSpace(strings.Join(fs.Args(), " ")))
	_, tokens, err := loadAndEnsureAuth(ctx, *tokenPath)
	if err != nil {
		return err
	}
	st := tokens.XCloud
	if st == nil {
		st = tokens.XCloudF2P
	}
	if st == nil {
		return fmt.Errorf("xCloud token not available for this account")
	}
	client := gssv.NewClient(st.Host, st.GSToken, gssv.Preferences{})
	titles, err := client.ListCloudTitles(ctx)
	if err != nil {
		return err
	}
	count := 0
	for _, t := range titles {
		if !t.Details.HasEntitlement {
			continue
		}
		if filter != "" && !strings.Contains(strings.ToLower(t.Details.Name), filter) &&
			!strings.Contains(strings.ToLower(t.TitleID), filter) {
			continue
		}
		fmt.Printf("%-36s  %s\n", t.TitleID, t.Details.Name)
		count++
	}
	if count == 0 {
		fmt.Println("No matching entitled xCloud titles found.")
	}
	return nil
}

func cmdPlay(ctx context.Context, args []string) error {
	fs := flag.NewFlagSet("play", flag.ExitOnError)
	homeFlag := fs.Bool("home", false, "stream from your Xbox console (Remote Play)")
	consoleSel := fs.String("console", "", "console name or serverId (optional with --home)")
	cloudID := fs.String("cloud", "", "xCloud titleId to stream")
	res := fs.String("res", "1080p", "preferred resolution: 720p, 1080p, 1440p")
	bitrate := fs.Int("bitrate", 20000, "max video bitrate in kbps")
	tapName := fs.String("tap", frametap.DefaultName, "SmoovPlay FrameTap shared-memory name")
	noAudio := fs.Bool("no-audio", false, "disable local Opus audio playback")
	statsEvery := fs.Duration("stats", 2*time.Second, "live WebRTC stats log interval (0 disables)")
	tokenPath := fs.String("tokens", auth.DefaultTokenPath(), "token cache file")
	_ = fs.Parse(args)

	if !*homeFlag && *cloudID == "" {
		*homeFlag = true // default to Remote Play if neither flag is passed
	}
	if *consoleSel == "" && fs.NArg() > 0 && *homeFlag {
		*consoleSel = fs.Arg(0)
	}

	w, h := parseResolution(*res)
	prefs := gssv.Preferences{Width: w, Height: h, Locale: "en-US", OSName: "windows"}

	ac, tokens, err := loadAndEnsureAuth(ctx, *tokenPath)
	if err != nil {
		return err
	}

	cfg := session.Config{
		Width:          w,
		Height:         h,
		FPS:            60,
		MaxBitrateKbps: *bitrate,
		TapName:        *tapName,
		EnableAudio:    !*noAudio,
		StatsInterval:  *statsEvery,
	}

	var client *gssv.Client
	if *cloudID != "" {
		st := tokens.XCloud
		if st == nil {
			st = tokens.XCloudF2P
		}
		if st == nil {
			return fmt.Errorf("xCloud token not available for this account")
		}
		lpt, err := ac.FetchLongPlayToken(ctx, tokens.RefreshToken)
		if err != nil {
			return fmt.Errorf("fetch xCloud LPT: %w", err)
		}
		cfg.Kind = gssv.KindCloud
		cfg.TargetID = *cloudID
		cfg.LPTToken = lpt
		client = gssv.NewClient(st.Host, st.GSToken, prefs)
	} else {
		client = gssv.NewClient(tokens.XHome.Host, tokens.XHome.GSToken, prefs)
		consoles, err := client.ListConsoles(ctx)
		if err != nil {
			return fmt.Errorf("list consoles: %w", err)
		}
		if len(consoles) == 0 {
			return fmt.Errorf("no Remote Play consoles found")
		}
		chosen, err := pickConsole(consoles, *consoleSel)
		if err != nil {
			return err
		}
		log.Printf("[xbox-tap] selected console: %s (%s, %s)", chosen.DeviceName, chosen.ServerID, chosen.PowerState)
		cfg.Kind = gssv.KindHome
		cfg.TargetID = chosen.ServerID
	}

	setHighPriority()
	log.Printf("[xbox-tap] start SmoovPlay with: python run.py --source tap --display --present dxgi --fullscreen")
	return session.Run(ctx, client, cfg)
}

func parseResolution(s string) (int, int) {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "720p":
		return 1280, 720
	case "1440p":
		return 2560, 1440
	default:
		return 1920, 1080
	}
}

func pickConsole(consoles []gssv.Console, query string) (gssv.Console, error) {
	q := strings.ToLower(strings.TrimSpace(query))
	if q == "" {
		return consoles[0], nil
	}
	for _, c := range consoles {
		if strings.EqualFold(c.ServerID, q) || strings.EqualFold(c.DeviceName, q) ||
			strings.Contains(strings.ToLower(c.DeviceName), q) {
			return c, nil
		}
	}
	return gssv.Console{}, fmt.Errorf("console %q not found (run `smoovplay-xbox consoles` to list)", query)
}
