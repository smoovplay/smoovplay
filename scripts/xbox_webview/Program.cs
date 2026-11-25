using System;
using System.IO;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using Microsoft.Web.WebView2.Core;
using Microsoft.Web.WebView2.Wpf;

namespace SmoovPlayXboxWebView;

public static class Program
{
    [STAThread]
    public static void Main(string[] args)
    {
        var url = args.Length > 0 && args[0].StartsWith("http", StringComparison.OrdinalIgnoreCase)
            ? args[0]
            : "https://play.xbox.com";
        var scriptPath = FindTapScript();
        var app = new Application();
        app.Run(new XboxWindow(url, scriptPath));
    }

    private static string FindTapScript()
    {
        // Walk up from the executable (and cwd) to find scripts/xbox_webrtc_tap.user.js
        foreach (var start in new[] { AppContext.BaseDirectory, Directory.GetCurrentDirectory() })
        {
            var dir = new DirectoryInfo(start);
            for (int i = 0; i < 6 && dir != null; i++, dir = dir.Parent)
            {
                var candidate = Path.Combine(dir.FullName, "scripts", "xbox_webrtc_tap.user.js");
                if (File.Exists(candidate)) return candidate;
                candidate = Path.Combine(dir.FullName, "xbox_webrtc_tap.user.js");
                if (File.Exists(candidate)) return candidate;
            }
        }
        return "";
    }
}

public sealed class XboxWindow : Window
{
    private readonly WebView2 _web = new();
    private readonly TextBlock _status = new()
    {
        Foreground = new SolidColorBrush(Color.FromRgb(180, 220, 180)),
        VerticalAlignment = VerticalAlignment.Center,
        Margin = new Thickness(10, 0, 10, 0),
        FontSize = 12,
    };
    private readonly string _startUrl;
    private readonly string _scriptPath;

    public XboxWindow(string startUrl, string scriptPath)
    {
        _startUrl = startUrl;
        _scriptPath = scriptPath;
        Title = "SmoovPlay Xbox (WebView2 Frame Tap)";
        Width = 1280;
        Height = 780;
        Background = new SolidColorBrush(Color.FromRgb(18, 18, 22));

        var bar = new DockPanel
        {
            Background = new SolidColorBrush(Color.FromRgb(26, 26, 32)),
            Height = 32,
        };
        var skipBox = new CheckBox
        {
            Content = "Skip browser decode",
            Foreground = Brushes.White,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(8, 0, 10, 0),
            ToolTip = "Stop WebView2 from decoding video frames once SmoovPlay is connected (saves GPU/CPU).",
        };
        skipBox.Checked += (_, _) => SetSkipDecode(true);
        skipBox.Unchecked += (_, _) => SetSkipDecode(false);
        DockPanel.SetDock(skipBox, Dock.Right);

        var muteBox = new CheckBox
        {
            Content = "Mute",
            Foreground = Brushes.White,
            VerticalAlignment = VerticalAlignment.Center,
            Margin = new Thickness(8, 0, 10, 0),
            ToolTip = "Mute or unmute WebView2 audio output.",
        };
        muteBox.Checked += (_, _) => SetMuted(true);
        muteBox.Unchecked += (_, _) => SetMuted(false);
        DockPanel.SetDock(muteBox, Dock.Right);

        var devtoolsBtn = new Button
        {
            Content = "DevTools",
            Margin = new Thickness(4, 4, 8, 4),
            Padding = new Thickness(8, 0, 8, 0),
        };
        devtoolsBtn.Click += (_, _) => _web.CoreWebView2?.OpenDevToolsWindow();
        DockPanel.SetDock(devtoolsBtn, Dock.Right);

        bar.Children.Add(devtoolsBtn);
        bar.Children.Add(skipBox);
        bar.Children.Add(muteBox);
        bar.Children.Add(_status);

        var root = new DockPanel();
        DockPanel.SetDock(bar, Dock.Top);
        root.Children.Add(bar);
        root.Children.Add(_web);
        Content = root;

        Loaded += async (_, _) => await InitAsync();
    }

    private void SetMuted(bool on)
    {
        if (_web.CoreWebView2 == null) return;
        _web.CoreWebView2.IsMuted = on;
    }

    private async void SetSkipDecode(bool on)
    {
        if (_web.CoreWebView2 == null) return;
        await _web.CoreWebView2.ExecuteScriptAsync($"window.SMOOVPLAY_SKIP_BROWSER_DECODE = {(on ? "true" : "false")};");
    }

    private async System.Threading.Tasks.Task InitAsync()
    {
        try
        {
            var dataDir = Path.Combine(
                Environment.GetFolderPath(Environment.SpecialFolder.LocalApplicationData),
                "SmoovPlay", "XboxWebView");
            // Keep WebRTC & gamepad active when this window is minimised or behind SmoovPlay.
            var opts = new CoreWebView2EnvironmentOptions(
                "--disable-background-timer-throttling --disable-renderer-backgrounding --disable-backgrounding-occluded-windows --autoplay-policy=no-user-gesture-required");
            var env = await CoreWebView2Environment.CreateAsync(null, dataDir, opts);
            await _web.EnsureCoreWebView2Async(env);
            _web.CoreWebView2.IsMuted = false;

            if (!string.IsNullOrEmpty(_scriptPath) && File.Exists(_scriptPath))
            {
                var js = await File.ReadAllTextAsync(_scriptPath);
                await _web.CoreWebView2.AddScriptToExecuteOnDocumentCreatedAsync(js);
                _status.Text = "WebRTC Frame Tap active (ws://127.0.0.1:8088) — start SmoovPlay with --source tap";
            }
            else
            {
                _status.Text = "Warning: xbox_webrtc_tap.user.js not found";
                _status.Foreground = Brushes.Orange;
            }

            _web.CoreWebView2.Navigate(_startUrl);
        }
        catch (Exception ex)
        {
            _status.Text = $"WebView2 init failed: {ex.Message}";
            _status.Foreground = Brushes.IndianRed;
        }
    }
}
