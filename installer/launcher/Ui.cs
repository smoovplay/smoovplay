// The SmoovPlay window (layout and styles in Ui.xaml). Setup mode walks through PC check -> folder -> install ->
// done; launcher mode has Play, the PS5 guide, the PC check, Settings (SettingsPage.cs) and Help in the left rail.
// While a stream plays the launcher waits in the taskbar, its button showing a coloured dot, and comes back up when
// the session ends or fails.
//
// SMOOVPLAY_SHOTS=<folder> saves a PNG of the window (rendered by WPF itself, never a screen capture) whenever the
// page or state changes, for testing and docs.
// SMOOVPLAY_RECORD=<folder> records it frame by frame the same way, for demo videos (see StartRecorder).
// SMOOVPLAY_SHOW_RESULTS=<results .txt> opens with that test report on the Play page, for testing the card.
// SMOOVPLAY_SIZE=<width>x<height> opens at that size, centred (and saves neither), for snapshots at several sizes.

using System;
using System.Collections.Concurrent;
using System.Collections.Generic;
using System.ComponentModel;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Runtime.InteropServices;
using System.Text;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Markup;
using System.Windows.Media;
using System.Windows.Media.Animation;
using System.Windows.Media.Imaging;
using System.Windows.Shapes;
using System.Windows.Shell;
using System.Windows.Threading;
using Path = System.IO.Path;
using ShapePath = System.Windows.Shapes.Path;

namespace SmoovPlay
{
    partial class Ui
    {
        public readonly Window Window;
        readonly Options opts;
        readonly bool setupMode;  // SmoovPlay-Setup.exe: the app is inside it
        Paths paths;
        Installer installer;
        Session session;
        SourceDef source = Sources.RemotePlay;  // where the video comes from (the Play page's picker)
        bool allowClose, finishSetup;
        string page;
        readonly HashSet<string> prefs;
        readonly string shots = Environment.GetEnvironmentVariable("SMOOVPLAY_SHOTS");
        int shotNo;
        string updateUrl, updateTag;  // the newer release's page and tag, once UpdateCheck finds one
        bool updating;  // downloading it (Updater)
        bool updateMode;  // setup opened by the launcher to update an install (--update)

        static readonly string[] SetupSteps = { "Check your PC", "Choose a folder", "Install", "Ready to play" };
        static readonly string[] SetupStepHints = { "A few seconds", "About 12 GB", "10 to 20 minutes", "" };

        public Ui(Options opts)
        {
            this.opts = opts;
            setupMode = Payload.Present;
            using (var s = Assembly.GetExecutingAssembly().GetManifestResourceStream("Ui.xaml"))
                Window = (Window)XamlReader.Load(s);
            paths = setupMode ? Paths.ForInstall(opts.Dir ?? Paths.ExistingInstall ?? Paths.DefaultInstall) : Paths.Detect();
            Proc.VfxPath = paths.Vfx;
            prefs = Prefs.Load();
            Wire();
            StartRecorder();
            if (setupMode && opts.Update) StartUpdate();
            else if (setupMode) StartSetup();
            else StartLauncher();
        }

        // ------------------------------------------------------------------------------------------- plumbing
        T E<T>(string name) where T : class
        {
            var o = Window.FindName(name) as T;
            if (o == null) throw new InvalidOperationException("Ui.xaml has no " + typeof(T).Name + " named " + name);
            return o;
        }
        FrameworkElement E(string name) { return E<FrameworkElement>(name); }
        TextBlock T(string name) { return E<TextBlock>(name); }
        Button B(string name) { return E<Button>(name); }
        Brush R(string key) { return (Brush)Window.FindResource(key); }
        Style S(string key) { return (Style)Window.FindResource(key); }
        static Color C(string hex) { return (Color)ColorConverter.ConvertFromString(hex); }
        static SolidColorBrush Solid(string hex) { return new SolidColorBrush(C(hex)); }
        void OnUi(Action a) { try { Window.Dispatcher.BeginInvoke(a); } catch (Exception) { } }
        void Click(string name, Action a) { B(name).Click += (s, e) => a(); }
        static void Show(UIElement e, bool on) { e.Visibility = on ? Visibility.Visible : Visibility.Collapsed; }

        static readonly IEasingFunction EaseOut = new CubicEase { EasingMode = EasingMode.EaseOut };
        static DoubleAnimation Anim(double? from, double to, double ms, double delayMs = 0, IEasingFunction ease = null)
        {
            var a = new DoubleAnimation { To = to, Duration = TimeSpan.FromMilliseconds(ms), BeginTime = TimeSpan.FromMilliseconds(delayMs), EasingFunction = ease ?? EaseOut };
            if (from.HasValue) a.From = from.Value;
            return a;
        }
        // Endless animations (glows, spinners). They're paused while the window is minimised, or in the background
        // while a stream plays, so the launcher never redraws itself behind a game; and they run at 30 fps.
        readonly Dictionary<Tuple<IAnimatable, DependencyProperty>, Func<AnimationTimeline>> ambient =
            new Dictionary<Tuple<IAnimatable, DependencyProperty>, Func<AnimationTimeline>>();
        bool ambientPaused;

        void Forever(IAnimatable target, DependencyProperty prop, double from, double to, double ms, bool reverse)
        {
            Func<AnimationTimeline> make = () =>
            {
                var a = new DoubleAnimation(from, to, TimeSpan.FromMilliseconds(ms))
                { RepeatBehavior = RepeatBehavior.Forever, AutoReverse = reverse, EasingFunction = reverse ? new SineEase { EasingMode = EasingMode.EaseInOut } : null };
                Timeline.SetDesiredFrameRate(a, 30);
                return a;
            };
            ambient[Tuple.Create(target, prop)] = make;
            if (!ambientPaused) target.BeginAnimation(prop, make());
        }

        void StopForever(IAnimatable target, DependencyProperty prop)
        {
            if (ambient.Remove(Tuple.Create(target, prop))) target.BeginAnimation(prop, null);
        }

        void UpdateAmbient()
        {
            bool pause = Window.WindowState == WindowState.Minimized || (!Window.IsActive && session != null && session.Running);
            if (pause == ambientPaused) return;
            ambientPaused = pause;
            foreach (var kv in ambient) kv.Key.Item1.BeginAnimation(kv.Key.Item2, pause ? null : kv.Value());
            Program.Trace("ambient animations " + (pause ? "paused" : "running"));
        }

        /// <summary>The window resizes from its edges (WindowChrome in Ui.xaml; the see-through margin round it, where
        /// the shadow is, takes the pointer there), between its MinWidth/MinHeight and the screen, and opens at the
        /// size and place it last had (the place only if it's still on a screen: one may have been unplugged since).
        /// It doesn't maximise: the layout is made for a window, and it keeps its shadow.</summary>
        void WireSize()
        {
            string forced = Environment.GetEnvironmentVariable("SMOOVPLAY_SIZE");
            string saved = forced ?? prefs.Where(p => p.StartsWith("window:")).Select(p => p.Substring(7)).FirstOrDefault();
            var m = Regex.Match(saved ?? "", @"^\s*(\d+)\s*x\s*(\d+)\s*$");
            if (m.Success)
            {
                var area = SystemParameters.WorkArea;
                Window.Width = Math.Max(Window.MinWidth, Math.Min(int.Parse(m.Groups[1].Value), area.Width));
                Window.Height = Math.Max(Window.MinHeight, Math.Min(int.Parse(m.Groups[2].Value), area.Height));
            }
            Window.StateChanged += (s, e) => { if (Window.WindowState == WindowState.Maximized) Window.WindowState = WindowState.Normal; };
            if (forced != null) return;
            var at = Regex.Match(prefs.Where(p => p.StartsWith("window-at:")).Select(p => p.Substring(10)).FirstOrDefault() ?? "", @"^(-?\d+),(-?\d+)$");
            if (at.Success)
            {
                double left = int.Parse(at.Groups[1].Value), top = int.Parse(at.Groups[2].Value);
                double width = double.IsNaN(Window.Width) ? Window.MinWidth : Window.Width;
                if (OnScreen(left + width / 2, top + 48))  // the middle of the header, where it's dragged by
                {
                    Window.WindowStartupLocation = WindowStartupLocation.Manual;
                    Window.Left = left;
                    Window.Top = top;
                }
            }
            var save = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(600) };
            save.Tick += (s, e) =>
            {
                save.Stop();
                if (Window.WindowState != WindowState.Normal) return;  // (minimised, it's at -32000)
                string size = "window:" + (int)Math.Round(Window.ActualWidth) + "x" + (int)Math.Round(Window.ActualHeight);
                string place = "window-at:" + (int)Math.Round(Window.Left) + "," + (int)Math.Round(Window.Top);
                if (prefs.Contains(size) && prefs.Contains(place)) return;
                prefs.RemoveWhere(p => p.StartsWith("window:") || p.StartsWith("window-at:"));
                prefs.Add(size);
                prefs.Add(place);
                Prefs.Save(prefs);
            };
            EventHandler moved = (s, e) => { if (Window.IsLoaded) { save.Stop(); save.Start(); } };
            Window.SizeChanged += (s, e) => moved(s, e);
            Window.LocationChanged += moved;
        }

        /// <summary>Whether a point (in WPF's units, which are the main display's scaled pixels) is on a screen now.</summary>
        static bool OnScreen(double x, double y)
        {
            try
            {
                double scale;
                using (var g = System.Drawing.Graphics.FromHwnd(IntPtr.Zero)) scale = g.DpiX / 96.0;
                return MonitorFromPoint(new NativePoint { X = (int)Math.Round(x * scale), Y = (int)Math.Round(y * scale) }, 0) != IntPtr.Zero;  // 0: MONITOR_DEFAULTTONULL
            }
            catch (Exception) { return false; }
        }

        [StructLayout(LayoutKind.Sequential)] struct NativePoint { public int X, Y; }
        [DllImport("user32.dll")] static extern IntPtr MonitorFromPoint(NativePoint pt, uint flags);

        /// <summary>The fade at the foot of a scrolling page (its OpacityMask in Ui.xaml) says there's more below, so
        /// it shows only while there is: in a tall window, or scrolled to the end, the last card isn't faded.</summary>
        void WireFades()
        {
            foreach (var sv in Logical<ScrollViewer>(Window).Where(s => s.OpacityMask != null).ToList())
            {
                var mask = sv.OpacityMask;
                sv.ScrollChanged += (s, e) => sv.OpacityMask = sv.VerticalOffset < sv.ScrollableHeight - 1 ? mask : null;
            }
        }

        static IEnumerable<T> Logical<T>(DependencyObject root) where T : DependencyObject
        {
            foreach (var child in LogicalTreeHelper.GetChildren(root).OfType<DependencyObject>())
            {
                var t = child as T;
                if (t != null) yield return t;
                foreach (var d in Logical<T>(child)) yield return d;
            }
        }

        void Wire()
        {
            Window.Title = setupMode ? "SmoovPlay Setup" : "SmoovPlay";
            try
            {
                using (var ico = System.Drawing.Icon.ExtractAssociatedIcon(Assembly.GetExecutingAssembly().Location))
                    Window.Icon = Imaging.CreateBitmapSourceFromHIcon(ico.Handle, Int32Rect.Empty, BitmapSizeOptions.FromEmptyOptions());
            }
            catch (Exception) { }
            Window.TaskbarItemInfo = new TaskbarItemInfo();
            var root = E<Grid>("Root");
            root.SizeChanged += (s, e) => root.Clip = new RectangleGeometry(new Rect(0, 0, root.ActualWidth, root.ActualHeight), 16, 16);
            Window.MouseLeftButtonDown += (s, e) => { try { if (e.ButtonState == MouseButtonState.Pressed) Window.DragMove(); } catch (Exception) { } };
            WireSize();
            WireFades();
            Click("BtnMin", () => Window.WindowState = WindowState.Minimized);
            Click("BtnClose", () => Window.Close());
            Window.Closing += OnClosing;
            EventHandler displayChanged = (s, e) => OnUi(() => { settingSpecsStale = true; if (page == "PageSettings") LoadSettingSpecs(true); });
            Microsoft.Win32.SystemEvents.DisplaySettingsChanged += displayChanged;
            Window.Closed += (s, e) => Microsoft.Win32.SystemEvents.DisplaySettingsChanged -= displayChanged;
            Window.Activated += (s, e) => UpdateAmbient();
            Window.Deactivated += (s, e) => UpdateAmbient();
            Window.StateChanged += (s, e) => UpdateAmbient();
            Window.PreviewKeyDown += OnKeysPressed;  // App window mode's keys, being changed (BuildKeys)
            Window.KeyDown += (s, e) =>
            {
                if (E("Dialog").Visibility == Visibility.Visible && e.Key == Key.Escape) { CloseDialog(0); e.Handled = true; }
            };
            T("RailVersion").Text = "v" + BuildInfo.Version + (paths.Dev ? "  ·  dev" : "");

            // a slow breathing glow in the corner of the rail, and the shimmer on the progress bar
            var glow = E("RailGlow");
            Forever(glow, UIElement.OpacityProperty, 0.55, 1, 4200, true);
            Forever(glow, Canvas.LeftProperty, -210, -170, 7000, true);
            var shimmer = E("BarShimmer");
            var shimmerX = new TranslateTransform();
            shimmer.RenderTransform = shimmerX;
            Forever(shimmerX, TranslateTransform.XProperty, -160, 700, 1700, false);

            // check page
            Click("BtnRecheck", () => RunChecks());
            Click("BtnCheckNext", CheckNext);
            Click("BtnCheckAnyway", CheckNext);

            // folder page
            var dir = E<TextBox>("DirBox");
            var dirTimer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(250) };
            dirTimer.Tick += (s, e) => { dirTimer.Stop(); ValidateDir(); };
            dir.TextChanged += (s, e) => { dirTimer.Stop(); dirTimer.Start(); };
            Click("BtnBrowse", Browse);
            Click("BtnLocBack", () => { ShowPage("PageCheck"); SetStep(0); });
            Click("BtnInstall", () => StartInstall(true, false));
            Click("BtnShortPath", () =>
            {
                string root2 = SafeRoot(dir.Text) ?? Path.GetPathRoot(Paths.DefaultInstall);
                dir.Text = Path.Combine(root2, "SmoovPlay");
            });
            Click("BtnLongPaths", () => Busy("BtnLongPaths", SystemCheck.EnableLongPaths, err =>
            {
                if (err == null) Toast("Long paths are on", false); else Toast(err, true);
                ValidateDir();
            }));
            E("DriveBar").SizeChanged += (s, e) => LayoutDriveBar(false);

            // install page
            Click("BtnCancel", () => { var i = installer; if (i != null) { B("BtnCancel").IsEnabled = false; i.Cancel(); } });
            Click("BtnRetry", () => StartInstall(lastFromPayload, false));
            Click("BtnCopyLog", CopyDiagnostics);
            var installDetails = E<ToggleButton>("BtnInstallDetails");
            installDetails.Checked += (s, e) => { Show(E("InstallLog"), true); installDetails.Content = "Hide details"; Snap("install-details"); };
            installDetails.Unchecked += (s, e) => { Show(E("InstallLog"), false); installDetails.Content = "Show details"; };
            installLog = new LogPump(E<TextBox>("InstallLog"));

            // done page
            Click("BtnOpen", OpenAfterInstall);

            // play page
            Click("PlayButton", () => { if (session != null && session.Running) session.Stop(); else OnPlay(false); });
            Click("BtnPlayTest", () => OnPlay(true));
            Click("BtnResultsCopy", CopyResults);
            Click("BtnResultsOpen", () => { if (resultsFile != null && File.Exists(resultsFile)) Proc.Open(resultsFile); });
            Click("BtnResultsClose", () => ShowResultsCard(false));
            var playDetails = E<ToggleButton>("BtnPlayDetails");
            playDetails.Checked += (s, e) => { Show(E("PlayLog"), true); playDetails.Content = "Hide details"; };
            playDetails.Unchecked += (s, e) => { Show(E("PlayLog"), false); playDetails.Content = "Show details"; };
            playLog = new LogPump(E<TextBox>("PlayLog"));
            Click("BtnPlayLogs", OpenLogs);
            Click("BtnPlayDiag", CopyDiagnostics);

            // guide page
            Click("BtnGuideDocs", () => Proc.Open(ChiakiDocs));
            Click("BtnGuideChiaki", OpenChiaki);
            Click("BtnGuidePlay", () => Nav("PagePlay"));

            // settings page
            Click("BtnSetRestart", RestartFromSettings);
            Click("BtnDetect", RunDetect);

            // looks page
            Click("BtnLookNew", NewLook);
            Click("BtnLookPaste", PasteLookCode);
            WireLookPreview();
            E("SetScroll").SizeChanged += (s, e) => LayoutSections(false);

            // help page
            Click("HelpDiag", CopyDiagnostics);
            Click("HelpLogs", OpenLogs);
            Click("HelpRepair", AskRepair);
            Click("HelpUninstall", AskUninstall);
            Click("HelpDocs", () => Proc.Open(ChiakiDocs));
            Click("HelpFeedback", () => SendFeedback(""));
            Click("RailUpdate", AskUpdate);
            Click("RailCheck", CheckForUpdates);
            Click("ToastAction", ToastActionClicked);
            Click("HelpLicences", () =>
            {
                string f = Path.Combine(paths.Install, "THIRD_PARTY_NOTICES.txt");
                if (!File.Exists(f)) f = Path.Combine(paths.App, "installer", "THIRD_PARTY_NOTICES.txt");
                if (File.Exists(f)) Proc.Open(f); else Toast("The licence files are in the install folder", false);
            });
            Click("HelpSource", () =>
            {
                string d = Path.Combine(paths.Install, "chiaki", "source");
                if (Directory.Exists(d)) Process.Start("explorer.exe", "\"" + d + "\"");
                else Toast("The chiaki-ng source is in the install folder's chiaki\\source", false);
            });

            // dialog
            Click("DialogBtn1", () => CloseDialog(1));
            Click("DialogBtn2", () => CloseDialog(2));
            Click("DialogBtn3", () => CloseDialog(3));

            // rail navigation (launcher)
            foreach (var pair in new[] { Tuple.Create("NavPlay", "PagePlay"), Tuple.Create("NavGuide", "PageGuide"),
                                         Tuple.Create("NavCheck", "PageCheck"), Tuple.Create("NavSettings", "PageSettings"),
                                         Tuple.Create("NavLooks", "PageLooks"), Tuple.Create("NavHelp", "PageHelp") })
            {
                var target = pair.Item2;
                E<RadioButton>(pair.Item1).Checked += (s, e) => { if (!navigating) OnNav(target); };
            }
        }

        const string ChiakiDocs = "https://chiaki-ng.github.io/chiaki-ng/";

        // ------------------------------------------------------------------------------------------- pages
        void ShowPage(string name)
        {
            foreach (var p in new[] { "PageCheck", "PageLocation", "PageInstall", "PageDone", "PagePlay", "PageGuide", "PageSettings", "PageLooks", "PageHelp" })
                if (p != name) Show(E(p), false);
            var el = E(name);
            bool changed = page != name;
            if (changed && page == "PageLooks") OnLooksLeave();
            page = name;
            Show(el, true);
            if (changed)
            {
                var tt = new TranslateTransform();
                el.RenderTransform = tt;
                el.BeginAnimation(UIElement.OpacityProperty, Anim(0, 1, 260));
                tt.BeginAnimation(TranslateTransform.YProperty, Anim(10, 0, 320));
            }
            Snap(name.Substring(4).ToLowerInvariant());
        }

        bool navigating;
        /// <summary>Launcher: show a page and select its rail button.</summary>
        void Nav(string name)
        {
            navigating = true;
            try
            {
                string nav = name == "PagePlay" ? "NavPlay" : name == "PageGuide" ? "NavGuide" : name == "PageCheck" ? "NavCheck"
                    : name == "PageSettings" ? "NavSettings" : name == "PageLooks" ? "NavLooks" : name == "PageHelp" ? "NavHelp" : null;
                if (nav != null) E<RadioButton>(nav).IsChecked = true;
            }
            finally { navigating = false; }
            OnNav(name);
        }

        void OnNav(string name)
        {
            if (name == "PageCheck") { ShowPage(name); if (checks == null) RunChecks(); return; }
            if (name == "PagePlay") BuildChips();
            if (name == "PageGuide") BuildGuide();
            if (name == "PageSettings") { ShowSettings(); return; }
            if (name == "PageLooks") { ShowLooks(); return; }
            ShowPage(name);
        }

        void SetNavEnabled(bool on)
        {
            foreach (var n in new[] { "NavPlay", "NavGuide", "NavCheck", "NavSettings", "NavLooks", "NavHelp" }) E(n).IsEnabled = on;
        }

        // ---- the setup stepper in the rail
        readonly List<Tuple<Border, TextBlock, TextBlock, Rectangle, Ellipse>> steps = new List<Tuple<Border, TextBlock, TextBlock, Rectangle, Ellipse>>();

        void BuildStepper()
        {
            var host = E<StackPanel>("Stepper");
            host.Children.Clear();
            steps.Clear();
            for (int i = 0; i < SetupSteps.Length; i++)
            {
                var g = new Grid { Height = i < SetupSteps.Length - 1 ? 62 : 40 };
                g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(30) });
                g.ColumnDefinitions.Add(new ColumnDefinition());
                var line = new Rectangle { Width = 2, Margin = new Thickness(0, 31, 0, 3), RadiusX = 1, RadiusY = 1, Fill = Solid("#262C36"), HorizontalAlignment = HorizontalAlignment.Center };
                if (i == SetupSteps.Length - 1) line.Visibility = Visibility.Collapsed;
                var halo = new Ellipse { Width = 36, Height = 36, Margin = new Thickness(-5), Stroke = Solid("#55FFB02E"), StrokeThickness = 1.5, VerticalAlignment = VerticalAlignment.Top, Opacity = 0 };
                var circle = new Border { Width = 26, Height = 26, CornerRadius = new CornerRadius(13), VerticalAlignment = VerticalAlignment.Top, BorderThickness = new Thickness(1.5) };
                var num = new TextBlock { FontSize = 12, FontWeight = FontWeights.SemiBold, HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center };
                circle.Child = num;
                var labels = new StackPanel { Margin = new Thickness(10, 3, 0, 0) };
                var label = new TextBlock { Text = SetupSteps[i], FontSize = 13.5 };
                string hintText = updateMode && i == 2 ? "A minute or two" : SetupStepHints[i];
                var hint = new TextBlock { Text = hintText, FontSize = 11.5, Foreground = R("Muted"), Margin = new Thickness(0, 2, 0, 0) };
                labels.Children.Add(label);
                if (hintText.Length > 0) labels.Children.Add(hint);
                Grid.SetColumn(labels, 1);
                var col0 = new Grid();
                col0.Children.Add(line);
                col0.Children.Add(halo);
                col0.Children.Add(circle);
                g.Children.Add(col0);
                g.Children.Add(labels);
                host.Children.Add(g);
                steps.Add(Tuple.Create(circle, num, label, line, halo));
            }
        }

        void SetStep(int active)
        {
            for (int i = 0; i < steps.Count; i++)
            {
                var st = steps[i];
                Border circle = st.Item1; TextBlock num = st.Item2, label = st.Item3; Rectangle line = st.Item4; Ellipse halo = st.Item5;
                if (i < active)
                {
                    circle.Background = R("Good"); circle.BorderBrush = R("Good");
                    num.Text = "\uE73E"; num.FontFamily = (FontFamily)Window.FindResource("Icons"); num.FontSize = 11; num.Foreground = R("Ink");
                    label.Foreground = R("Text2"); label.FontWeight = FontWeights.Normal;
                    line.Fill = Solid("#2E5A45");
                    StopForever(halo, UIElement.OpacityProperty); halo.Opacity = 0;
                }
                else if (i == active)
                {
                    circle.Background = R("AccentGrad"); circle.BorderBrush = Brushes.Transparent;
                    num.Text = (i + 1).ToString(); num.FontFamily = Window.FontFamily; num.FontSize = 12; num.Foreground = R("Ink");
                    label.Foreground = R("Text"); label.FontWeight = FontWeights.SemiBold;
                    line.Fill = Solid("#262C36");
                    Forever(halo, UIElement.OpacityProperty, 0.25, 1, 1300, true);
                }
                else
                {
                    circle.Background = Brushes.Transparent; circle.BorderBrush = Solid("#323a47");
                    num.Text = (i + 1).ToString(); num.FontFamily = Window.FontFamily; num.FontSize = 12; num.Foreground = R("Muted");
                    label.Foreground = R("Muted"); label.FontWeight = FontWeights.Normal;
                    line.Fill = Solid("#262C36");
                    StopForever(halo, UIElement.OpacityProperty); halo.Opacity = 0;
                }
            }
        }

        // ------------------------------------------------------------------------------------------- start
        void StartSetup()
        {
            Show(E("Stepper"), true);
            Show(E("Nav"), false);
            T("RailFooterTitle").Text = "About 5 GB to download";
            T("RailFooterText").Text = "Python, PyTorch and TensorRT come straight from their official servers. Free and open source (MIT).";
            BuildStepper();
            SetStep(0);
            E<TextBox>("DirBox").Text = paths.Install;
            ShowPage("PageCheck");
            RunChecks();
        }

        /// <summary>Setup, opened by the launcher to update this install (--update --dir --wait, Updater.cs): straight to
        /// installing once the launcher has closed, then SmoovPlay opens again.</summary>
        void StartUpdate()
        {
            updateMode = true;
            Show(E("Stepper"), true);
            Show(E("Nav"), false);
            T("RailFooterTitle").Text = "Updating SmoovPlay";
            T("RailFooterText").Text = "Your settings, looks and clips stay. Only parts this version changes are downloaded again.";
            BuildStepper();
            SetStep(2);
            E<TextBox>("DirBox").Text = paths.Install;
            E<CheckBox>("ToggleDesktop").IsChecked = File.Exists(Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.DesktopDirectory), "SmoovPlay.lnk"));
            T("InstallOverline").Text = "UPDATING TO " + V(BuildInfo.Version);
            T("InstallTitle").Text = "Updating SmoovPlay";
            T("InstallSub").Text = "A minute or two. SmoovPlay opens again when it's done, with your settings, looks and clips.";
            ShowPage("PageInstall");
            int wait = opts.Wait;
            Task.Run(() =>
            {
                Program.WaitFor(wait, 30);  // the launcher that started this; files it holds can't be replaced
                OnUi(() => StartInstall(true, false));
            });
        }

        static string V(string tag) { return tag.StartsWith("v") ? tag : "v" + tag; }

        /// <summary>GitHub has a newer release: a link under the version in the rail, and a note on the Help page.</summary>
        void ShowUpdate(string tag, string url)
        {
            if (updating) return;
            bool same = tag == updateTag;
            updateUrl = url;
            updateTag = tag;
            var link = B("RailUpdate");
            link.Content = "Update available: " + V(tag);
            link.ToolTip = Updater.Unavailable(paths) == null
                ? "Installs it: SmoovPlay closes, updates in a minute or two and opens again, with your settings."
                : "Opens the release page. Download the new setup and run it: it updates this install and keeps your settings.";
            Show(link, true);
            Show(B("RailCheck"), false);
            var help = T("HelpVersion");  // next to the version: the install path at the end gets cut off
            if (!same) help.Text = $"SmoovPlay {BuildInfo.Version} ({tag} is out)  ·  Python {BuildInfo.PythonVersion}  ·  {paths.Install}";
            Snap("update");
        }

        /// <summary>The rail's "Check for updates": asks GitHub now, whatever the daily check last found.</summary>
        void CheckForUpdates()
        {
            var link = B("RailCheck");
            if ((link.Tag as string) == "busy") return;
            link.Tag = "busy";
            link.Content = "Checking...";
            UpdateCheck.Now((tag, error) => OnUi(() =>
            {
                link.Tag = null;
                if (error == null && tag != null && UpdateCheck.Compare(tag, BuildInfo.Version) > 0)
                {
                    link.Content = "Check for updates";
                    ShowUpdate(tag, GitHub.Page(GitHub.Repo, tag));
                    return;
                }
                link.Content = error != null ? "Couldn't check" : "Up to date";
                link.ToolTip = error != null ? "GitHub didn't answer (" + Shorten(error, 120) + "). Click to try again." : V(BuildInfo.Version) + " is the newest.";
                var t = new DispatcherTimer { Interval = TimeSpan.FromSeconds(error != null ? 8 : 4) };
                t.Tick += (s, e) =>
                {
                    t.Stop();
                    if ((link.Tag as string) == "busy") return;
                    link.Content = "Check for updates";
                    link.ToolTip = "Asks GitHub for a newer SmoovPlay now (it also checks by itself when it opens).";
                };
                t.Start();
            }));
        }

        void StartLauncher()
        {
            Show(E("Stepper"), false);
            Show(E("Nav"), true);
            BuildKeys();
            T("HelpVersion").Text = $"SmoovPlay {BuildInfo.Version}  ·  Python {BuildInfo.PythonVersion}  ·  {paths.Install}";
            if (!paths.Dev) Task.Run(() => Updater.CleanUp());  // an update this one came from
            UpdateCheck.Start((tag, url) => Window.Dispatcher.BeginInvoke(new Action(() => ShowUpdate(tag, url))));
            Show(B("RailCheck"), GitHub.Repo != null);
            WatchSaving();  // before & after videos a session left being made, even before this launcher opened
            B("HelpRepair").IsEnabled = !paths.Dev;
            B("HelpUninstall").IsEnabled = !paths.Dev;
            T("CheckOverline").Text = "PC CHECK";
            T("CheckTitle").Text = "Your PC";
            T("CheckSub").Text = "What SmoovPlay found. Anything that needs attention can be fixed from here.";
            SetPlayState(PlayState.Idle, null);
            string showResults = Environment.GetEnvironmentVariable("SMOOVPLAY_SHOW_RESULTS");  // testing and docs
            if (!string.IsNullOrEmpty(showResults) && File.Exists(showResults)) ShowResults(showResults);
            if (!paths.RuntimeReady) { ShowFinishSetup(); return; }
            LoadSettingSpecs();
            LoadSources();
            LoadVfgStatus();  // for the chips and the Settings page
            string want = opts.Page;
            if (want == null) want = prefs.Contains("guide-seen") ? "play" : "guide";
            if (!prefs.Contains("guide-seen")) { prefs.Add("guide-seen"); Prefs.Save(prefs); }
            Nav(want == "guide" ? "PageGuide" : want == "check" ? "PageCheck" : want == "settings" ? "PageSettings" : want == "looks" ? "PageLooks" : want == "help" ? "PageHelp" : "PagePlay");
        }

        /// <summary>An install whose downloads never finished (or were removed): check, then finish.</summary>
        void ShowFinishSetup()
        {
            finishSetup = true;
            T("CheckOverline").Text = "FINISH SETUP";
            T("CheckTitle").Text = "Let's finish setting up";
            T("CheckSub").Text = "SmoovPlay still needs Python, PyTorch and TensorRT (about 5 GB). First, a quick look at your PC.";
            T("BtnCheckNextText").Text = "Finish setup";
            Nav("PageCheck");
        }

        // ------------------------------------------------------------------------------------------- PC check
        List<Check> checks;
        readonly Dictionary<string, CheckCard> cards = new Dictionary<string, CheckCard>();
        int checkRun;

        class CheckCard
        {
            public Border Root, Badge;
            public SolidColorBrush CircleFill, GlyphFill, BadgeFill;
            public TextBlock BadgeGlyph, Value, Detail;
            public Ellipse Spinner;
            public RotateTransform SpinRot;
            public Button Fix;
            public Check Check;
        }

        string CheckDir()
        {
            if (!setupMode) return paths.Install;
            string raw = E<TextBox>("DirBox").Text.Trim();
            try { return raw.Length > 0 ? Path.GetFullPath(raw) : paths.Install; } catch (Exception) { return paths.Install; }
        }

        void RunChecks()
        {
            int run = ++checkRun;
            bool fresh = setupMode || finishSetup;
            string dir = CheckDir();
            checks = SystemCheck.Plan(source.Kind);
            var grid = E<UniformGrid>("CheckCards");
            grid.Children.Clear();
            foreach (var old in cards.Values) StopForever(old.SpinRot, RotateTransform.AngleProperty);
            cards.Clear();
            for (int i = 0; i < checks.Count; i++)
            {
                var card = MakeCard(checks[i]);
                cards[checks[i].Id] = card;
                grid.Children.Add(card.Root);
                var tt = new TranslateTransform();
                card.Root.RenderTransform = tt;
                card.Root.Opacity = 0;
                card.Root.BeginAnimation(UIElement.OpacityProperty, Anim(0, 1, 300, i * 45));
                tt.BeginAnimation(TranslateTransform.YProperty, Anim(14, 0, 420, i * 45));
            }
            B("BtnRecheck").IsEnabled = false;
            B("BtnCheckNext").IsEnabled = false;
            Show(B("BtnCheckNext"), setupMode || finishSetup);
            Show(B("BtnCheckAnyway"), false);
            Summary("Checking your PC...", "", "#22FFB02E", "Accent", "\uE9D9");
            var started = DateTime.UtcNow;
            var gpu = Task.Run(() => SystemCheck.QueryGpu());
            var all = new List<Task>();
            for (int i = 0; i < checks.Count; i++)
            {
                var c = checks[i];
                int order = i;
                all.Add(Task.Run(async () =>
                {
                    if (c.Id == "gpu" || c.Id == "driver") await gpu;
                    SystemCheck.Run(c, dir, fresh);
                    // results land one after another, so the page reads top to bottom
                    var due = started + TimeSpan.FromMilliseconds(450 + order * 120);
                    var wait = due - DateTime.UtcNow;
                    if (wait > TimeSpan.Zero) await Task.Delay(wait);
                    OnUi(() => { if (run == checkRun) SetCard(cards[c.Id], c); });
                }));
            }
            Task.WhenAll(all).ContinueWith(t => OnUi(() => { if (run == checkRun) ChecksDone(); }));
        }

        CheckCard MakeCard(Check c)
        {
            var card = new CheckCard { Check = c };
            card.Root = new Border { Style = S("Card"), Margin = new Thickness(5), Padding = new Thickness(14, 12, 14, 12) };
            var g = new Grid();
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(50) });
            g.ColumnDefinitions.Add(new ColumnDefinition());

            var icon = new Grid { Width = 38, Height = 38, VerticalAlignment = VerticalAlignment.Top, HorizontalAlignment = HorizontalAlignment.Left };
            card.CircleFill = Solid("#1E232C");
            card.GlyphFill = Solid("#8D96A5");
            icon.Children.Add(new Ellipse { Fill = card.CircleFill });
            icon.Children.Add(new TextBlock { Style = S("Icon"), Text = c.Glyph, FontSize = 16, Foreground = card.GlyphFill });
            var spinRot = new RotateTransform();
            card.SpinRot = spinRot;
            card.Spinner = new Ellipse
            {
                Margin = new Thickness(-3), Stroke = R("AccentGrad"), StrokeThickness = 2, StrokeDashArray = new DoubleCollection { 7, 14 },
                StrokeDashCap = PenLineCap.Round, RenderTransformOrigin = new Point(0.5, 0.5), RenderTransform = spinRot,
            };
            Forever(spinRot, RotateTransform.AngleProperty, 0, 360, 900, false);
            icon.Children.Add(card.Spinner);
            card.BadgeFill = Solid("#3DDC8E");
            card.BadgeGlyph = new TextBlock { FontSize = 8, Foreground = R("Ink"), HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center, FontWeight = FontWeights.Bold };
            card.Badge = new Border
            {
                Width = 18, Height = 18, CornerRadius = new CornerRadius(9), Background = card.BadgeFill, BorderBrush = R("CardBg"), BorderThickness = new Thickness(2),
                HorizontalAlignment = HorizontalAlignment.Right, VerticalAlignment = VerticalAlignment.Bottom, Margin = new Thickness(0, 0, -4, -4),
                Child = card.BadgeGlyph, RenderTransformOrigin = new Point(0.5, 0.5), RenderTransform = new ScaleTransform(0, 0),
            };
            icon.Children.Add(card.Badge);
            g.Children.Add(icon);

            var col = new StackPanel();
            Grid.SetColumn(col, 1);
            var head = new Grid { Height = 16 };
            head.ColumnDefinitions.Add(new ColumnDefinition());
            head.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            head.Children.Add(new TextBlock { Text = c.Title.ToUpperInvariant(), FontSize = 11, FontWeight = FontWeights.SemiBold, Foreground = R("Muted"), VerticalAlignment = VerticalAlignment.Center });
            card.Fix = new Button { Style = S("Pill"), Margin = new Thickness(6, -7, 0, -7), Visibility = Visibility.Collapsed, VerticalAlignment = VerticalAlignment.Center };
            card.Fix.Click += (s, e) => RunFix(card);
            Grid.SetColumn(card.Fix, 1);
            head.Children.Add(card.Fix);
            col.Children.Add(head);
            card.Value = new TextBlock { Text = "Checking...", FontSize = 14.5, FontWeight = FontWeights.SemiBold, Foreground = R("Text2"), Margin = new Thickness(0, 4, 0, 0), TextTrimming = TextTrimming.CharacterEllipsis };
            col.Children.Add(card.Value);
            card.Detail = new TextBlock { Style = S("Small"), Margin = new Thickness(0, 3, 0, 0), Height = 34, TextTrimming = TextTrimming.WordEllipsis };
            col.Children.Add(card.Detail);
            g.Children.Add(col);
            card.Root.Child = g;
            AutomationProperties.SetAutomationId(card.Root, "check-" + c.Id);
            return card;
        }

        void SetCard(CheckCard card, Check c)
        {
            card.Check = c;
            string tone = c.Level == Level.Ok ? "#3DDC8E" : c.Level == Level.Info ? "#6DB3FF" : c.Level == Level.Warn ? "#FFC24B" : c.Level == Level.Fail ? "#FF6464" : "#8D96A5";
            var col = C(tone);
            bool checking = c.Level == Level.Checking;
            card.CircleFill.BeginAnimation(SolidColorBrush.ColorProperty, new ColorAnimation(checking ? C("#1E232C") : Color.FromArgb(0x24, col.R, col.G, col.B), TimeSpan.FromMilliseconds(300)));
            card.GlyphFill.BeginAnimation(SolidColorBrush.ColorProperty, new ColorAnimation(checking ? C("#8D96A5") : col, TimeSpan.FromMilliseconds(300)));
            card.Spinner.BeginAnimation(UIElement.OpacityProperty, Anim(null, checking ? 1 : 0, 200));
            if (checking) Forever(card.SpinRot, RotateTransform.AngleProperty, 0, 360, 900, false);
            else StopForever(card.SpinRot, RotateTransform.AngleProperty);
            card.Value.Text = checking ? (c.Value.Length > 0 ? c.Value : "Checking...") : c.Value;
            card.Value.Foreground = checking ? R("Text2") : R("Text");
            card.Detail.Text = c.Detail;
            var sc = (ScaleTransform)card.Badge.RenderTransform;
            if (checking)
            {
                sc.BeginAnimation(ScaleTransform.ScaleXProperty, Anim(null, 0, 150));
                sc.BeginAnimation(ScaleTransform.ScaleYProperty, Anim(null, 0, 150));
            }
            else
            {
                card.BadgeFill.Color = col;
                bool iconGlyph = c.Level == Level.Ok || c.Level == Level.Fail;
                card.BadgeGlyph.FontFamily = iconGlyph ? (FontFamily)Window.FindResource("Icons") : new FontFamily("Segoe UI");
                card.BadgeGlyph.FontSize = iconGlyph ? 7.5 : 10;
                card.BadgeGlyph.Text = c.Level == Level.Ok ? "\uE73E" : c.Level == Level.Fail ? "\uE711" : c.Level == Level.Warn ? "!" : "i";
                var pop = new BackEase { EasingMode = EasingMode.EaseOut, Amplitude = 0.6 };
                sc.BeginAnimation(ScaleTransform.ScaleXProperty, Anim(0, 1, 380, 0, pop));
                sc.BeginAnimation(ScaleTransform.ScaleYProperty, Anim(0, 1, 380, 0, pop));
            }
            Show(card.Fix, !checking && c.FixLabel != null);
            card.Fix.Content = c.FixLabel;
            card.Fix.IsEnabled = true;
            if (card.Fix.Visibility == Visibility.Visible) AutomationProperties.SetAutomationId(card.Fix, "fix-" + c.Id);
            var tip = new StackPanel { MaxWidth = 300 };
            tip.Children.Add(new TextBlock { Text = c.Title + (c.Value.Length > 0 ? ": " + c.Value : ""), FontWeight = FontWeights.SemiBold, TextWrapping = TextWrapping.Wrap });
            if (c.Detail.Length > 0) tip.Children.Add(new TextBlock { Text = c.Detail, Margin = new Thickness(0, 4, 0, 0), TextWrapping = TextWrapping.Wrap, Foreground = R("Text2") });
            if (c.Tip != null) tip.Children.Add(new TextBlock { Text = c.Tip, Margin = new Thickness(0, 6, 0, 0), TextWrapping = TextWrapping.Wrap, Foreground = R("Text2") });
            card.Root.ToolTip = checking ? null : tip;
        }

        void Summary(string title, string detail, string dot, string fgKey, string glyph)
        {
            T("CheckSummary").Text = title;
            T("CheckSummaryDetail").Text = detail;
            Show(T("CheckSummaryDetail"), detail.Length > 0);
            E<Ellipse>("CheckSummaryDot").Fill = Solid(dot);
            T("CheckSummaryIcon").Foreground = R(fgKey);
            T("CheckSummaryIcon").Text = glyph;
        }

        void ChecksDone()
        {
            B("BtnRecheck").IsEnabled = true;
            var fails = checks.Where(c => c.Level == Level.Fail).ToList();
            var warns = checks.Where(c => c.Level == Level.Warn).ToList();
            if (fails.Count == 0 && warns.Count == 0)
                Summary("Your PC is ready for SmoovPlay", "Everything it needs is here.", "#243DDC8E", "Good", "\uE73E");
            else if (fails.Count == 0)
                Summary(warns.Count == 1 ? "Ready, with one suggestion" : $"Ready, with {warns.Count} suggestions",
                        string.Join(", ", warns.Select(c => c.Title)) + ": hover a card for help.", "#24FFC24B", "Warn", "!");
            else
                Summary(fails.Count == 1 ? "One thing to sort out first" : $"{fails.Count} things to sort out first",
                        string.Join(", ", fails.Select(c => c.Title)), "#24FF6464", "Bad", "\uE711");
            if (T("CheckSummaryIcon").Text == "!") T("CheckSummaryIcon").FontFamily = new FontFamily("Segoe UI");
            else T("CheckSummaryIcon").FontFamily = (FontFamily)Window.FindResource("Icons");
            B("BtnCheckNext").IsEnabled = fails.Count == 0;
            Show(B("BtnCheckAnyway"), (setupMode || finishSetup) && fails.Count > 0);
            Snap("checks-done");
        }

        void RunFix(CheckCard card)
        {
            var c = card.Check;
            switch (c.FixId)
            {
                case "driver":
                    Proc.Open(SystemCheck.DriverUrl);
                    Toast("Opened NVIDIA's driver page. Install the driver, then press Check again", false);
                    break;
                case "othergpu":
                    SendFeedback("AMD and Intel graphics (DirectML)");
                    break;
                case "display":
                    Proc.Open("ms-settings:display-advanced");
                    Toast("In Windows' display settings, raise the refresh rate, then press Check again", false);
                    break;
                case "vc":
                    card.Fix.IsEnabled = false;
                    var working = new Check(c.Id, c.Title, c.Glyph).Set(Level.Checking, "Installing...", "Downloading from Microsoft...");
                    SetCard(card, working);
                    string dir = CheckDir();
                    Task.Run(() =>
                    {
                        string err = SystemCheck.InstallVcRuntime(s => OnUi(() => card.Detail.Text = s));
                        var again = new Check(c.Id, c.Title, c.Glyph);
                        SystemCheck.Run(again, dir, true);
                        OnUi(() =>
                        {
                            int i = checks.FindIndex(x => x.Id == c.Id);
                            if (i >= 0) checks[i] = again;
                            if (err != null && again.Level != Level.Ok) again.Detail = err;
                            SetCard(card, again);
                            ChecksDone();
                            Toast(again.Level == Level.Ok ? "Visual C++ runtime installed" : "Couldn't install it: " + err, again.Level != Level.Ok);
                        });
                    });
                    break;
            }
        }

        void CheckNext()
        {
            if (finishSetup) { StartInstall(false, false); return; }
            ShowPage("PageLocation");
            SetStep(1);
            ValidateDir();
        }

        // ------------------------------------------------------------------------------------------- folder
        double driveUsedFrac, driveNeedFrac;

        static string SafeRoot(string dir)
        {
            try { return Path.GetPathRoot(Path.GetFullPath(dir.Trim())); } catch (Exception) { return null; }
        }

        /// <summary>The folder setup will really use: never mixes SmoovPlay into a folder that has other things in it
        /// (uninstall removes what setup made), so a non-empty folder gets a SmoovPlay folder inside.</summary>
        static string ResolveDir(string raw, out bool sub)
        {
            sub = false;
            string dir = Path.GetFullPath(raw.Trim()).TrimEnd('\\');
            if (dir.Length <= 3) { sub = true; return Path.Combine(dir + "\\", "SmoovPlay"); }  // a drive root
            if (Directory.Exists(dir) && Directory.EnumerateFileSystemEntries(dir).Any() && !Paths.ForInstall(dir).IsInstall)
            {
                sub = true;
                return Path.Combine(dir, "SmoovPlay");
            }
            return dir;
        }

        string resolvedDir;

        void ValidateDir()
        {
            string raw = E<TextBox>("DirBox").Text;
            bool tooLong;
            string problem = SystemCheck.FolderProblem(raw, out tooLong), dir = null;
            bool sub = false;
            if (problem == null)
            {
                try { dir = ResolveDir(raw, out sub); problem = SystemCheck.FolderProblem(dir, out tooLong); }
                catch (Exception e) { problem = e.Message; }
            }
            if (problem == null && dir != null) problem = SystemCheck.WriteProblem(dir);
            string root = SafeRoot(raw);
            double free = 0, total = 0;
            try
            {
                var d = new DriveInfo(root);
                free = d.AvailableFreeSpace / 1e9; total = d.TotalSize / 1e9;
                T("DriveName").Text = root.TrimEnd('\\') + (d.VolumeLabel.Length > 0 ? "  " + d.VolumeLabel : "");
                T("DriveFree").Text = $"{free:0} GB free of {total:0} GB";
            }
            catch (Exception)
            {
                T("DriveName").Text = root != null ? root.TrimEnd('\\') : "No drive";
                T("DriveFree").Text = "";
            }
            bool existing = dir != null && Paths.ForInstall(dir).IsInstall;
            if (problem == null && free < SystemCheck.NeedGB && !existing)
                problem = $"Only {free:0} GB free on {root.TrimEnd('\\')}. SmoovPlay needs about {SystemCheck.NeedGB:0} GB: choose another drive, or free some space.";
            driveUsedFrac = total > 0 ? (total - free) / total : 0;
            driveNeedFrac = total > 0 ? Math.Min(SystemCheck.NeedGB, Math.Max(free, 0)) / total : 0;
            LayoutDriveBar(true);

            Show(E("DirProblem"), problem != null);
            T("DirProblemText").Text = problem ?? "";
            Show(B("BtnShortPath"), tooLong);
            Show(B("BtnLongPaths"), tooLong);
            var note = T("DirExisting");
            if (problem == null && existing)
            {
                note.Text = "SmoovPlay is already installed here. Setup updates it in place and keeps your settings.";
                note.Foreground = R("Good");
                Show(note, true);
            }
            else if (problem == null && sub)
            {
                note.Text = "Installs into " + dir;
                note.Foreground = R("Text2");
                Show(note, true);
            }
            else Show(note, false);
            resolvedDir = problem == null ? dir : null;
            B("BtnInstall").IsEnabled = problem == null;
            T("BtnInstallText").Text = existing ? "Update" : "Install";
            if (page == "PageLocation") SnapLater("location", 800);
        }

        void LayoutDriveBar(bool animate)
        {
            double w = E("DriveBar").ActualWidth;
            if (w <= 0) return;
            double used = Math.Max(4, w * driveUsedFrac), need = Math.Max(4, w * driveNeedFrac);
            if (used + need + 2 > w) used = Math.Max(0, w - need - 2);
            E("DriveUsed").BeginAnimation(FrameworkElement.WidthProperty, Anim(null, used, animate ? 500 : 1));
            E("DriveNeed").BeginAnimation(FrameworkElement.WidthProperty, Anim(null, need, animate ? 600 : 1, animate ? 120 : 0));
        }

        void Browse()
        {
            var box = E<TextBox>("DirBox");
            string start = box.Text.Trim();
            try
            {
                while (start.Length > 3 && !Directory.Exists(start)) start = Path.GetDirectoryName(start);
            }
            catch (Exception) { start = null; }
            string picked = FolderPicker.Pick(new WindowInteropHelper(Window).Handle, "Where should SmoovPlay go? It needs about 12 GB.", start);
            if (picked == null) return;
            string name = Path.GetFileName(picked.TrimEnd('\\'));
            bool isInstall = Paths.ForInstall(picked).IsInstall;
            box.Text = isInstall || name.Equals("SmoovPlay", StringComparison.OrdinalIgnoreCase) ? picked : Path.Combine(picked, "SmoovPlay");
            ValidateDir();
        }

        // ------------------------------------------------------------------------------------------- install
        LogPump installLog, playLog;
        bool lastFromPayload;
        readonly List<Tuple<Grid, TextBlock, TextBlock, Grid>> stepRows = new List<Tuple<Grid, TextBlock, TextBlock, Grid>>();
        int activeStep = -1;
        DispatcherTimer tipTimer;
        DateTime lastBarUpdate;
        readonly Queue<Tuple<DateTime, double>> speedSamples = new Queue<Tuple<DateTime, double>>();
        double downloaded;

        /// <param name="fromPayload">Setup: unpack the app into the chosen folder. Otherwise (launcher), repair the runtime.</param>
        void StartInstall(bool fromPayload, bool redownload)
        {
            if (fromPayload && setupMode)
            {
                ValidateDir();
                if (resolvedDir == null) return;
                paths = Paths.ForInstall(resolvedDir);
                Proc.VfxPath = paths.Vfx;
            }
            lastFromPayload = fromPayload;
            bool repair = !fromPayload;
            if (setupMode) SetStep(2); else SetNavEnabled(false);
            T("InstallOverline").Text = updateMode ? "UPDATING TO " + V(BuildInfo.Version) : setupMode ? "STEP 3 OF 3  ·  INSTALLING" : repair && !finishSetup ? "REPAIR" : "FINISH SETUP";
            T("InstallTitle").Text = updateMode ? "Updating SmoovPlay" : !repair ? "Setting things up" : finishSetup ? "Finishing setup" : redownload ? "Repairing SmoovPlay" : "Checking SmoovPlay";
            T("InstallPercent").Text = "0";
            T("InstallSpeed").Text = "";
            E("BarFill").BeginAnimation(FrameworkElement.WidthProperty, null);
            E("BarFill").Width = 0;
            E("BarGlow").BeginAnimation(FrameworkElement.WidthProperty, null);
            E("BarGlow").Width = 0;
            Show(E("ErrorCard"), false);
            Show(B("BtnRetry"), false);
            Show(B("BtnCopyLog"), false);
            Show(B("BtnCancel"), true);
            B("BtnCancel").IsEnabled = true;
            installLog.Clear();
            BuildInstallSteps(fromPayload);
            speedSamples.Clear();
            downloaded = 0;
            StartTips();
            ShowPage("PageInstall");
            Window.TaskbarItemInfo.ProgressState = TaskbarItemProgressState.Normal;
            Window.TaskbarItemInfo.ProgressValue = 0;

            var inst = new Installer(paths, fromPayload, fromPayload && !paths.Dev, true, E<CheckBox>("ToggleDesktop").IsChecked == true, redownload);
            installer = inst;
            Directory.CreateDirectory(paths.Logs);
            var log = new StreamWriter(Path.Combine(paths.Logs, $"setup-{DateTime.Now:yyyyMMdd-HHmmss}.log"), false, Encoding.UTF8) { AutoFlush = true };
            var gate = new object();
            inst.Log = line => { lock (gate) log.WriteLine(line); installLog.Add(line); };
            inst.Progress = (f, s) => OnUi(() => SetProgress(f, s));
            inst.StepStarted = i => OnUi(() => SetInstallStep(i));
            inst.Downloaded = b => { lock (speedSamples) downloaded = b; };
            Task.Run(() =>
            {
                Exception err = null;
                try { inst.Run(); }
                catch (Exception ex) { err = ex; }
                lock (gate) { if (err != null) log.WriteLine("ERROR: " + Installer.Message(err)); log.Dispose(); }
                OnUi(() => InstallFinished(err, fromPayload));
            });
        }

        void BuildInstallSteps(bool fromPayload)
        {
            var host = E<StackPanel>("InstallSteps");
            host.Children.Clear();
            stepRows.Clear();
            activeStep = -1;
            for (int i = 0; i < Installer.StepNames.Length; i++)
            {
                var row = new Grid { Margin = new Thickness(0, 5, 0, 5) };
                row.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(34) });
                row.ColumnDefinitions.Add(new ColumnDefinition());
                var ind = new Grid { Width = 22, Height = 22, VerticalAlignment = VerticalAlignment.Top, HorizontalAlignment = HorizontalAlignment.Left };
                var text = new StackPanel();
                Grid.SetColumn(text, 1);
                var name = new TextBlock { Text = Installer.StepNames[i], FontSize = 13.5, Margin = new Thickness(0, 2, 0, 0) };
                var sub = new TextBlock { FontSize = 12, Foreground = R("Text2"), Margin = new Thickness(0, 2, 0, 0), TextTrimming = TextTrimming.CharacterEllipsis, Visibility = Visibility.Collapsed };
                text.Children.Add(name);
                text.Children.Add(sub);
                row.Children.Add(ind);
                row.Children.Add(text);
                if (i == 0 && !fromPayload) row.Visibility = Visibility.Collapsed;
                host.Children.Add(row);
                stepRows.Add(Tuple.Create(row, name, sub, ind));
                StepIndicator(i, "todo");
            }
        }

        readonly Dictionary<int, RotateTransform> stepSpin = new Dictionary<int, RotateTransform>();

        void StepIndicator(int i, string state)
        {
            RotateTransform old;
            if (stepSpin.TryGetValue(i, out old)) { StopForever(old, RotateTransform.AngleProperty); stepSpin.Remove(i); }
            var r = stepRows[i];
            var ind = r.Item4;
            ind.Children.Clear();
            var name = r.Item2;
            Show(r.Item3, state == "active" || state == "failed");
            if (state == "todo")
            {
                ind.Children.Add(new Ellipse { Stroke = Solid("#323a47"), StrokeThickness = 1.5, Margin = new Thickness(2) });
                name.Foreground = R("Muted"); name.FontWeight = FontWeights.Normal;
            }
            else if (state == "active")
            {
                var rot = new RotateTransform();
                ind.Children.Add(new Ellipse { Stroke = Solid("#2A2F38"), StrokeThickness = 2.5 });
                ind.Children.Add(new Ellipse
                {
                    Stroke = R("AccentGrad"), StrokeThickness = 2.5, StrokeDashArray = new DoubleCollection { 3, 5.5 }, StrokeDashCap = PenLineCap.Round,
                    RenderTransformOrigin = new Point(0.5, 0.5), RenderTransform = rot,
                });
                ind.Children.Add(new Ellipse { Width = 6, Height = 6, Fill = R("Accent") });
                Forever(rot, RotateTransform.AngleProperty, 0, 360, 1000, false);
                stepSpin[i] = rot;
                name.Foreground = R("Text"); name.FontWeight = FontWeights.SemiBold;
            }
            else if (state == "done" || state == "failed")
            {
                bool ok = state == "done";
                var dot = new Border { CornerRadius = new CornerRadius(11), Background = ok ? R("Good") : R("Bad"), RenderTransformOrigin = new Point(0.5, 0.5) };
                dot.Child = new TextBlock { Style = S("Icon"), Text = ok ? "\uE73E" : "\uE711", FontSize = 10, Foreground = R("Ink") };
                var sc = new ScaleTransform();
                dot.RenderTransform = sc;
                ind.Children.Add(dot);
                var pop = new BackEase { EasingMode = EasingMode.EaseOut, Amplitude = 0.5 };
                sc.BeginAnimation(ScaleTransform.ScaleXProperty, Anim(0.3, 1, 340, 0, pop));
                sc.BeginAnimation(ScaleTransform.ScaleYProperty, Anim(0.3, 1, 340, 0, pop));
                name.Foreground = ok ? R("Text2") : R("Text"); name.FontWeight = ok ? FontWeights.Normal : FontWeights.SemiBold;
            }
        }

        static readonly string[] StepHints =
            { "Unpacking the app and chiaki-ng", "A private copy of Python, just for SmoovPlay", "About 5 GB: the bulk of the wait",
              "Once, so it starts fast every time after", "Shortcuts and the Apps & features entry" };

        void SetInstallStep(int i)
        {
            for (int k = Math.Max(0, activeStep); k < Math.Min(i, stepRows.Count); k++) StepIndicator(k, "done");
            activeStep = i;
            if (i < stepRows.Count)
            {
                StepIndicator(i, "active");
                stepRows[i].Item3.Text = updateMode && i == 2 ? "Only what this version changed" : StepHints[i];  // until the step reports its own progress
            }
            if (i == 3) T("InstallSpeed").Text = "A few minutes left";
            Snap("install-step" + i);
        }

        void SetProgress(double f, string status)
        {
            f = Math.Max(0, Math.Min(1, f));
            T("InstallPercent").Text = ((int)Math.Floor(f * 100)).ToString();
            if (activeStep >= 0 && activeStep < stepRows.Count) stepRows[activeStep].Item3.Text = status;
            Window.TaskbarItemInfo.ProgressValue = f;
            var now = DateTime.UtcNow;
            if ((now - lastBarUpdate).TotalMilliseconds > 120 || f >= 1)
            {
                lastBarUpdate = now;
                double w = E("BarTrack").ActualWidth * f;
                E("BarFill").BeginAnimation(FrameworkElement.WidthProperty, Anim(null, w, 400));
                E("BarGlow").BeginAnimation(FrameworkElement.WidthProperty, Anim(null, w, 400));
                if (activeStep == 2) UpdateSpeed(now);
            }
        }

        void UpdateSpeed(DateTime now)
        {
            double got;
            lock (speedSamples) got = downloaded;
            speedSamples.Enqueue(Tuple.Create(now, got));
            while (speedSamples.Count > 2 && (now - speedSamples.Peek().Item1).TotalSeconds > 4) speedSamples.Dequeue();
            var first = speedSamples.Peek();
            double dt = (now - first.Item1).TotalSeconds;
            if (dt < 1) return;
            double bps = (got - first.Item2) / dt;
            if (bps <= 1e5) { T("InstallSpeed").Text = "Installing..."; return; }
            double left = Math.Max(0, Installer.ExpectedDownloadBytes - got) / bps + 180;  // plus unpacking and the GPU step
            T("InstallSpeed").Text = $"{bps / 1e6:0.0} MB/s  ·  about {Math.Max(1, Math.Round(left / 60)):0} min left";
        }

        static readonly string[][] Tips =
        {
            new[] { "Turn on Remote Play", "On the PS5: Settings → System → Remote Play → Enable Remote Play." },
            new[] { "Wire it up if you can", "Remote Play is smoothest with the PC and the PS5 both plugged into the router." },
            new[] { "Stream at 60 fps", "In chiaki-ng's settings: 1080p and 60 fps. Playing in HDR? Pick the H.265 HDR codec and turn on HDR (Experimental) in SmoovPlay's Settings. SmoovPlay warns you if it's set differently." },
            new[] { "Pick your frame rate", "On SmoovPlay's Settings page, or with M in the video window: 60, 90 or 120 fps, the upscaler and more." },
            new[] { "Your controller just works", "Plug it into the PC or pair it over Bluetooth. It keeps working while chiaki-ng is minimised." },
        };
        int tip;

        void StartTips()
        {
            var dots = E<StackPanel>("TipDots");
            dots.Children.Clear();
            for (int i = 0; i < Tips.Length; i++)
                dots.Children.Add(new Border { Height = 6, Width = 6, CornerRadius = new CornerRadius(3), Margin = new Thickness(0, 0, 5, 0), Background = Solid("#2E3542") });
            tip = -1;
            NextTip();
            if (tipTimer == null)
            {
                tipTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(8) };
                tipTimer.Tick += (s, e) =>
                {
                    var body = E("TipBody");
                    var fade = Anim(null, 0, 200);
                    fade.Completed += (s2, e2) => { NextTip(); body.BeginAnimation(UIElement.OpacityProperty, Anim(0, 1, 300)); };
                    body.BeginAnimation(UIElement.OpacityProperty, fade);
                };
            }
            tipTimer.Start();
        }

        void NextTip()
        {
            tip = (tip + 1) % Tips.Length;
            T("TipTitle").Text = Tips[tip][0];
            T("TipText").Text = Tips[tip][1];
            var dots = E<StackPanel>("TipDots");
            for (int i = 0; i < dots.Children.Count; i++)
            {
                var d = (Border)dots.Children[i];
                d.Background = i == tip ? R("AccentGradH") : Solid("#2E3542");
                d.BeginAnimation(FrameworkElement.WidthProperty, Anim(null, i == tip ? 18 : 6, 250));
            }
        }

        void InstallFinished(Exception err, bool fromPayload)
        {
            installer = null;
            if (tipTimer != null) tipTimer.Stop();
            if (!setupMode) SetNavEnabled(true);
            if (err != null)
            {
                bool cancelled = err is OperationCanceledException;
                if (activeStep >= 0 && activeStep < stepRows.Count) StepIndicator(activeStep, "failed");
                T("ErrorTitle").Text = cancelled ? "Setup paused" : "Setup stopped";
                T("ErrorText").Text = (cancelled ? "" : Installer.Message(err) + " ") + "Retry picks up where it stopped.";
                Show(E("ErrorCard"), true);
                Show(B("BtnRetry"), true);
                Show(B("BtnCopyLog"), true);
                Show(B("BtnCancel"), false);
                if (!cancelled) E<ToggleButton>("BtnInstallDetails").IsChecked = true;
                T("InstallSpeed").Text = "";
                Window.TaskbarItemInfo.ProgressState = TaskbarItemProgressState.Error;
                Flash();
                Snap("install-failed");
                return;
            }
            Window.TaskbarItemInfo.ProgressState = TaskbarItemProgressState.None;
            if (updateMode)
            {
                SetStep(4);
                T("DoneSub").Text = "Updated to " + V(BuildInfo.Version) + ". Opening SmoovPlay...";
                T("BtnOpenText").Text = "Open SmoovPlay";
                var reopen = new DispatcherTimer { Interval = TimeSpan.FromSeconds(2) };
                reopen.Tick += (s, e) => { reopen.Stop(); OpenAfterInstall(); };
                reopen.Start();
            }
            else if (setupMode)
            {
                SetStep(4);
                T("DoneSub").Text = "It's in your Start menu too. Next, SmoovPlay walks you through connecting your PS5: about five minutes, once.";
                T("BtnOpenText").Text = "Open SmoovPlay";
            }
            else
            {
                T("DoneSub").Text = finishSetup ? "Everything's downloaded and your graphics card is ready." : "Every part of SmoovPlay is in place.";
                T("BtnOpenText").Text = "Let's play";
            }
            finishSetup = false;
            ShowPage("PageDone");
            Celebrate();
            Flash();
        }

        void OpenAfterInstall()
        {
            if (setupMode)
            {
                try { Process.Start(new ProcessStartInfo(Path.Combine(paths.Install, "SmoovPlay.exe")) { WorkingDirectory = paths.Install }); }
                catch (Exception e) { Toast("Couldn't open it: " + e.Message, true); return; }
                allowClose = true;
                Window.Close();
                return;
            }
            T("CheckOverline").Text = "PC CHECK";
            T("CheckTitle").Text = "Your PC";
            T("CheckSub").Text = "What SmoovPlay found. Anything that needs attention can be fixed from here.";
            Nav(prefs.Contains("guide-done") ? "PagePlay" : "PageGuide");
        }

        void Celebrate()
        {
            var ring = E<Ellipse>("DoneRing");
            double len = (116 - 5) * Math.PI / 5;  // circumference in stroke widths
            ring.StrokeDashArray = new DoubleCollection { len, len };
            ring.BeginAnimation(Shape.StrokeDashOffsetProperty, Anim(len, 0, 700, 100));
            var check = E<ShapePath>("DoneCheck");
            double clen = 72.0 / 7;
            check.StrokeDashArray = new DoubleCollection { clen, clen };
            check.Opacity = 0;
            check.BeginAnimation(UIElement.OpacityProperty, Anim(0, 1, 1, 650));
            check.BeginAnimation(Shape.StrokeDashOffsetProperty, Anim(clen, 0, 420, 650));
            var glow = E("DoneGlow");
            glow.BeginAnimation(UIElement.OpacityProperty, new DoubleAnimation(0, 0.4, TimeSpan.FromMilliseconds(1600))
            { BeginTime = TimeSpan.FromMilliseconds(700), AutoReverse = true, RepeatBehavior = RepeatBehavior.Forever, EasingFunction = new SineEase() });
            // a little burst of confetti in the brand colours
            var burst = E<Canvas>("DoneBurst");
            burst.Children.Clear();
            var rnd = new Random(7);
            string[] colours = { "#FFCB5C", "#FF8A1F", "#3DDC8E", "#6DB3FF", "#FFB02E", "#F3F5F9" };
            for (int i = 0; i < 26; i++)
            {
                double a = rnd.NextDouble() * Math.PI * 2, dist = 80 + rnd.NextDouble() * 90;
                var bit = new Rectangle
                {
                    Width = 5 + rnd.Next(4), Height = 9 + rnd.Next(5), RadiusX = 2, RadiusY = 2, Fill = Solid(colours[i % colours.Length]), Opacity = 0,
                    RenderTransformOrigin = new Point(0.5, 0.5),
                };
                var move = new TranslateTransform(66, 66);
                var spin = new RotateTransform(rnd.Next(360));
                var tg = new TransformGroup();
                tg.Children.Add(spin);
                tg.Children.Add(move);
                bit.RenderTransform = tg;
                burst.Children.Add(bit);
                double delay = 900 + rnd.Next(120), dur = 900 + rnd.Next(500);
                move.BeginAnimation(TranslateTransform.XProperty, Anim(66, 66 + Math.Cos(a) * dist, dur, delay));
                move.BeginAnimation(TranslateTransform.YProperty, Anim(66, 66 + Math.Sin(a) * dist + 40, dur, delay, new QuadraticEase { EasingMode = EasingMode.EaseOut }));
                spin.BeginAnimation(RotateTransform.AngleProperty, Anim(null, spin.Angle + rnd.Next(-400, 400), dur, delay));
                var fade = new DoubleAnimationUsingKeyFrames { BeginTime = TimeSpan.FromMilliseconds(delay) };
                fade.KeyFrames.Add(new LinearDoubleKeyFrame(1, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(60))));
                fade.KeyFrames.Add(new LinearDoubleKeyFrame(1, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(dur * 0.6))));
                fade.KeyFrames.Add(new LinearDoubleKeyFrame(0, KeyTime.FromTimeSpan(TimeSpan.FromMilliseconds(dur))));
                bit.BeginAnimation(UIElement.OpacityProperty, fade);
            }
            SnapLater("done-animated", 2300);
        }

        // ------------------------------------------------------------------------------------------- sources
        /// <summary>The video sources come from the pipeline (Sources in Core.cs), in the background; Remote Play until then.</summary>
        void LoadSources()
        {
            Task.Run(() =>
            {
                Sources.All(paths);
                OnUi(() => SetSource(Sources.Chosen(paths, prefs), false));
            });
        }

        void SetSource(SourceDef s, bool save)
        {
            bool changed = s.Kind != source.Kind;
            source = s;
            if (save) { Sources.Choose(prefs, s.Kind); Prefs.Save(prefs); }
            E<RadioButton>("NavGuide").Content = s.RemotePlay ? "Connect your PS5" : "Set up";
            BuildSources();
            if (!changed) return;
            checks = null;  // the PC check's last card depends on the source
            if (settingRows.Count > 0) RefreshSettings();  // settings only used with some sources (Spec.sources)
            BuildChips();
            if (playState == PlayState.Idle) SetPlayState(PlayState.Idle, null);
            Show(B("BtnPlayTest"), !(session != null && session.Running) && source.TestMode);
            if (page == "PageGuide") BuildGuide();
            if (page == "PageCheck") RunChecks();
            SnapLater("source-" + s.Kind, 600);
        }

        /// <summary>A row of choices above the chips, when there's more than one source to choose from.</summary>
        void BuildSources()
        {
            var host = E<WrapPanel>("PlaySources");
            host.Children.Clear();
            var offered = Sources.Offered(paths);
            Show(host, offered.Count > 1);
            if (offered.Count < 2) return;
            foreach (var s in offered)
            {
                bool on = s.Kind == source.Kind;
                var label = new StackPanel { Orientation = Orientation.Horizontal };
                label.Children.Add(new TextBlock { Text = s.Name, FontSize = 12.5, FontWeight = on ? FontWeights.SemiBold : FontWeights.Normal, Foreground = on ? R("Accent") : R("Text2"), VerticalAlignment = VerticalAlignment.Center });
                if (s.Experimental) label.Children.Add(ExperimentalBadge());
                var face = new Border
                {
                    CornerRadius = new CornerRadius(15), BorderThickness = new Thickness(1), Padding = new Thickness(14, 6, 14, 6),
                    Background = on ? Solid("#26FFB02E") : Solid("#171B22"), BorderBrush = on ? R("Accent") : R("Line2"),
                    Child = label,
                };
                var b = new Button { Style = S("Bare"), Content = face, Margin = new Thickness(4), Cursor = Cursors.Hand,
                                     ToolTip = s.Experimental ? s.Guide + "\n\n" + ExperimentalWhy : s.Guide };
                AutomationProperties.SetAutomationId(b, "source-" + s.Kind);
                AutomationProperties.SetName(b, (on ? "Playing from " : "Play from ") + s.Name);
                var pick = s;
                b.Click += (o, e) =>
                {
                    if (session != null && session.Running) { Toast("Stop playing first, then choose where the video comes from", true); return; }
                    SetSource(pick, true);
                };
                host.Children.Add(b);
            }
            if (source.Experimental) host.Children.Add(FeedbackPill(source.Name, "source-feedback"));
            BuildDevices();
        }

        /// <summary>Playing from a capture card: which card, and which sound (CaptureDevices), under the source pills. From
        /// an app's window: which window (AppWindows).</summary>
        void BuildDevices()
        {
            BuildWindowKeys();
            var host = E<WrapPanel>("PlayDevices");
            host.Children.Clear();
            bool capture = source.Kind == "capture", window = source.Kind == "window";
            Show(host, capture || window);
            if (window) { BuildWindows(host); return; }
            if (!capture) return;
            if (!CaptureDevices.Loaded)
            {
                host.Children.Add(new TextBlock { Text = "Looking for capture devices…", FontSize = 12.5, Foreground = R("Text2"), Margin = new Thickness(4, 8, 4, 0) });
                Task.Run(() => { CaptureDevices.Load(paths); OnUi(BuildDevices); });
                return;
            }
            if (CaptureDevices.Video.Count == 0)
            {
                // Three different things: the look failed, Windows has sound devices only, or nothing at all
                string why = CaptureDevices.Failed
                    ? "Couldn't list capture devices: " + CaptureDevices.Error + ". Press Look again; if it keeps happening, please use Send feedback."
                    : CaptureDevices.Audio.Count > 0
                    ? $"No video capture device found (Windows lists {CaptureDevices.Audio.Count} sound {(CaptureDevices.Audio.Count == 1 ? "device" : "devices")} only). Connect the card (USB 3, a blue port), then press Look again."
                    : "No capture device found. Connect the card (USB 3, a blue port), then press Look again.";
                host.Children.Add(new TextBlock { Text = why, FontSize = 12.5, Foreground = R("Warn"), Margin = new Thickness(4, 8, 4, 0), TextWrapping = TextWrapping.Wrap, MaxWidth = 420 });
                host.Children.Add(LookAgain());
                return;
            }
            string video = CaptureDevices.ChosenVideo(prefs), audio = CaptureDevices.ChosenAudio(prefs);
            var card = DeviceDrop("capture-card", "Capture card", CaptureDevices.Video.Select(n => new KeyValuePair<string, string>(n, n)), video, v =>
            {
                CaptureDevices.Choose(prefs, "capture-video:", v);
                Prefs.Save(prefs);
            });
            var sounds = new List<KeyValuePair<string, string>> { new KeyValuePair<string, string>("", "The card's own (found by name)") };
            sounds.AddRange(CaptureDevices.Audio.Select(n => new KeyValuePair<string, string>(n, n)));
            sounds.Add(new KeyValuePair<string, string>("off", "No sound (listen on the card's HDMI output)"));
            var sound = DeviceDrop("capture-sound", "Sound", sounds, audio ?? "", v =>
            {
                CaptureDevices.Choose(prefs, "capture-audio:", v == "" ? null : v);
                Prefs.Save(prefs);
            });
            host.Children.Add(Labelled("Card", card));
            host.Children.Add(Labelled("Sound", sound));
            host.Children.Add(LookAgain());
        }

        void BuildWindows(WrapPanel host)
        {
            if (!AppWindows.Loaded)
            {
                host.Children.Add(new TextBlock { Text = "Looking for app windows…", FontSize = 12.5, Foreground = R("Text2"), Margin = new Thickness(4, 8, 4, 0) });
                Task.Run(() => { AppWindows.Load(paths); OnUi(BuildDevices); });
                return;
            }
            string hint = AppWindows.MissedHint();
            if (AppWindows.List.Count == 0)
            {
                host.Children.Add(new TextBlock { Text = hint ?? "No app window found. Open the app you stream with (PS Remote Play, the Xbox app, Moonlight or a browser), then press Look again.", FontSize = 12.5, Foreground = R("Warn"), Margin = new Thickness(4, 8, 4, 0), TextWrapping = TextWrapping.Wrap, MaxWidth = 420 });
                host.Children.Add(LookAgain());
                return;
            }
            var chosen = AppWindows.Chosen(prefs);
            var app = DeviceDrop("window-app", "App window", AppWindows.List.Select(w => new KeyValuePair<string, string>(w.Hwnd.ToString(), AppWindows.Label(w))),
                                 chosen != null ? chosen.Hwnd.ToString() : "", v =>
            {
                var w = AppWindows.List.FirstOrDefault(x => x.Hwnd.ToString() == v);
                CaptureDevices.Choose(prefs, "window-app:", w != null ? w.Exe : null);
                CaptureDevices.Choose(prefs, "window-hwnd:", v);
                Prefs.Save(prefs);
            });
            app.MaxWidth = 420;
            host.Children.Add(Labelled("App", app));
            host.Children.Add(LookAgain());
            if (hint != null)  // e.g. a browser is listed, but the Xbox app is minimised
                host.Children.Add(new TextBlock { Text = hint, FontSize = 12.5, Foreground = R("Text2"), Margin = new Thickness(4, 8, 4, 0), TextWrapping = TextWrapping.Wrap, MaxWidth = 420 });
        }

        // ---- App window mode's keys (WindowKeys.cs) --------------------------------------------------------------
        // The app keeps the keyboard, so SmoovPlay's keys come through Windows from anywhere: listed here, with what
        // each gets on this PC (another program can have one), and changed by clicking one and pressing new keys.
        bool keysEditing;
        string keyListening;  // the action whose new keys are being pressed, or null
        string keysProblem;   // why the last keys pressed can't be used

        void BuildWindowKeys()
        {
            var host = E<StackPanel>("PlayKeys");
            host.Children.Clear();
            bool show = source != null && source.Kind == "window";
            Show(host, show);
            bool playing = session != null && session.Running;
            if (!show || playing) { keysEditing = false; keyListening = null; keysProblem = null; }
            if (!show) return;
            var taken = new List<string>();
            var got = playing ? WindowKeys.Playing(prefs) : WindowKeys.Resolve(prefs, taken);
            if (playing) taken = WindowKeys.Live != null ? WindowKeys.LiveTaken : taken;
            var picked = WindowKeys.Picked(prefs);

            var card = new Border { Background = R("CardBg"), BorderBrush = R("Line2"), BorderThickness = new Thickness(1), CornerRadius = new CornerRadius(10),
                                    Padding = new Thickness(16, 12, 16, 12), Width = 460, HorizontalAlignment = HorizontalAlignment.Center };
            AutomationProperties.SetAutomationId(card, "window-keys");
            var sp = new StackPanel();
            card.Child = sp;
            var head = new DockPanel();
            var change = new Button { Style = S("Bare"), Cursor = Cursors.Hand, VerticalAlignment = VerticalAlignment.Center,
                                      Content = new TextBlock { Text = keysEditing ? "Done" : "Change keys", FontSize = 12.5, Foreground = R("Accent"), TextDecorations = TextDecorations.Underline } };
            AutomationProperties.SetAutomationId(change, "window-keys-change");
            change.Click += (o, e) =>
            {
                if (session != null && session.Running) { Toast("Stop playing first, then change the keys", true); return; }
                keysEditing = !keysEditing;
                keyListening = keysProblem = null;
                BuildWindowKeys();
            };
            DockPanel.SetDock(change, Dock.Right);
            head.Children.Add(change);
            head.Children.Add(new TextBlock { Text = "SmoovPlay's keys while you play", FontSize = 13.5, FontWeight = FontWeights.SemiBold, Foreground = R("Text") });
            sp.Children.Add(head);
            sp.Children.Add(new TextBlock { Text = "The app keeps the keyboard, so M, F and Esc go to it. These work from anywhere:", FontSize = 12, Foreground = R("Text2"),
                                            TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 3, 0, 8) });

            var grid = new Grid();
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(1, GridUnitType.Star) });
            grid.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            int row = 0;
            foreach (var a in WindowKeys.Actions)
            {
                grid.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });
                string k = got.ContainsKey(a.Id) ? got[a.Id] : null;
                bool listening = keyListening == a.Id;
                var chip = new Border  // a key cap, as on the Help page
                {
                    Height = 26, MinWidth = 150, CornerRadius = new CornerRadius(6), Background = Solid("#20252E"), BorderBrush = listening ? R("Accent") : Solid("#323A47"),
                    BorderThickness = new Thickness(1, 1, 1, 3), Padding = new Thickness(10, 0, 10, 0), Margin = new Thickness(0, 3, 14, 3),
                    Child = new TextBlock { Text = listening ? "Press the new keys…" : k ?? "None free", FontSize = 12, FontWeight = FontWeights.SemiBold, VerticalAlignment = VerticalAlignment.Center,
                                            Foreground = listening ? R("Accent") : k == null ? R("Warn") : R("Text") },
                };
                UIElement first = chip;
                if (keysEditing)
                {
                    string id = a.Id;
                    var b = new Button { Style = S("Bare"), Cursor = Cursors.Hand, Content = chip, HorizontalAlignment = HorizontalAlignment.Left,
                                         ToolTip = "Click, then press the keys you'd like (Esc cancels)" };
                    AutomationProperties.SetAutomationId(b, "window-key-" + id);
                    AutomationProperties.SetName(b, a.Label + " keys");
                    b.Click += (o, e) => { keyListening = keyListening == id ? null : id; keysProblem = null; BuildWindowKeys(); };
                    first = b;
                }
                Grid.SetRow(first, row);
                grid.Children.Add(first);
                var label = new TextBlock { Text = a.Label, FontSize = 12.5, Foreground = R("Text2"), VerticalAlignment = VerticalAlignment.Center };
                Grid.SetRow(label, row);
                Grid.SetColumn(label, 1);
                grid.Children.Add(label);
                if (keysEditing && picked.ContainsKey(a.Id))
                {
                    string id = a.Id;
                    var reset = new Button { Style = S("Bare"), Cursor = Cursors.Hand, VerticalAlignment = VerticalAlignment.Center,
                                             Content = new TextBlock { Text = "Default", FontSize = 12, Foreground = R("Accent"), TextDecorations = TextDecorations.Underline } };
                    AutomationProperties.SetAutomationId(reset, "window-key-default-" + id);
                    reset.Click += (o, e) => { WindowKeys.Pick(prefs, id, null); Prefs.Save(prefs); keyListening = keysProblem = null; BuildWindowKeys(); };
                    Grid.SetRow(reset, row);
                    Grid.SetColumn(reset, 2);
                    grid.Children.Add(reset);
                }
                row++;
            }
            sp.Children.Add(grid);

            // keys another program has: what SmoovPlay uses instead
            foreach (var a in WindowKeys.Actions)
            {
                string k = got.ContainsKey(a.Id) ? got[a.Id] : null;
                var tried = picked.ContainsKey(a.Id) ? new[] { picked[a.Id] } : a.Defaults;
                var missed = taken.Where(t => tried.Contains(t)).Distinct().ToList();
                if (missed.Count == 0) continue;
                string what = string.Join(" and ", missed) + (missed.Count == 1 ? " is" : " are") + " used by another program on this PC, so ";
                string text = k != null ? what + a.Label.ToLowerInvariant() + " uses " + k + "."
                                        : what + a.Label.ToLowerInvariant() + " has no keys: press Change keys to pick others" + (a.Id == "quit" ? " (Stop here works too)." : ".");
                sp.Children.Add(new TextBlock { Text = text, FontSize = 12, Foreground = R("Warn"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 6, 0, 0) });
            }
            if (keysProblem != null)
                sp.Children.Add(new TextBlock { Text = keysProblem, FontSize = 12, Foreground = R("Bad"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 6, 0, 0) });
            if (keysEditing)
            {
                var foot = new DockPanel { Margin = new Thickness(0, 8, 0, 0) };
                if (picked.Count > 0)
                {
                    var all = new Button { Style = S("Bare"), Cursor = Cursors.Hand, VerticalAlignment = VerticalAlignment.Top,
                                           Content = new TextBlock { Text = "Reset all", FontSize = 12, Foreground = R("Accent"), TextDecorations = TextDecorations.Underline } };
                    AutomationProperties.SetAutomationId(all, "window-keys-reset");
                    all.Click += (o, e) => { WindowKeys.ResetAll(prefs); Prefs.Save(prefs); keyListening = keysProblem = null; BuildWindowKeys(); };
                    DockPanel.SetDock(all, Dock.Right);
                    foot.Children.Add(all);
                }
                foot.Children.Add(new TextBlock { Text = "Click one, then press its new keys with Ctrl, Alt or Win. If nothing happens, another program has those keys.",
                                                  FontSize = 12, Foreground = R("Muted"), TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 0, 12, 0) });
                sp.Children.Add(foot);
            }
            else
                sp.Children.Add(new TextBlock { Text = "Stop here ends it too, and the Settings page changes things while you play.", FontSize = 12, Foreground = R("Muted"),
                                                TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 8, 0, 0) });
            host.Children.Add(card);
        }

        /// <summary>The keys pressed for the action being changed (Window.PreviewKeyDown): checked, then kept.</summary>
        void OnKeysPressed(object sender, KeyEventArgs e)
        {
            if (keyListening == null || E("PagePlay").Visibility != Visibility.Visible) return;
            var key = e.Key == Key.System ? e.SystemKey : e.Key;
            var mods = Keyboard.Modifiers;
            e.Handled = true;
            if (key == Key.Escape && mods == ModifierKeys.None) { keyListening = keysProblem = null; BuildWindowKeys(); return; }
            string name = WindowKeys.FromKey(key, mods);
            if (name == null) return;  // only Ctrl, Alt, Shift or Win so far
            string why = WindowKeys.Problem(name, keyListening, prefs);
            if (why != null) { keysProblem = why; BuildWindowKeys(); return; }  // still listening
            WindowKeys.Pick(prefs, keyListening, WindowKeys.Find(keyListening).Defaults[0] == name ? null : name);
            Prefs.Save(prefs);
            keyListening = keysProblem = null;
            BuildWindowKeys();
        }

        UIElement Labelled(string label, UIElement child)
        {
            var sp = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(6, 6, 6, 0) };
            sp.Children.Add(new TextBlock { Text = label, FontSize = 12.5, Foreground = R("Text2"), VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 8, 0) });
            sp.Children.Add(child);
            return sp;
        }

        ComboBox DeviceDrop(string id, string name, IEnumerable<KeyValuePair<string, string>> choices, string selected, Action<string> picked)
        {
            var drop = new ComboBox { Style = S("Drop"), MinWidth = 200, MaxWidth = 300, VerticalAlignment = VerticalAlignment.Center };
            AutomationProperties.SetAutomationId(drop, id);
            AutomationProperties.SetName(drop, name);
            foreach (var c in choices)
            {
                var item = new ComboBoxItem { Style = S("DropItem"), Content = c.Value, Tag = c.Key, ToolTip = c.Value };
                drop.Items.Add(item);
                if (c.Key == selected) drop.SelectedItem = item;
            }
            if (drop.SelectedItem == null && drop.Items.Count > 0) drop.SelectedIndex = 0;
            drop.SelectionChanged += (o, e) =>
            {
                if (session != null && session.Running) { Toast("Stop playing first, then choose the card", true); return; }
                if (drop.SelectedItem is ComboBoxItem it) picked((string)it.Tag);
            };
            return drop;
        }

        UIElement LookAgain()
        {
            var b = new Button { Style = S("Bare"), Margin = new Thickness(6, 6, 6, 0), Cursor = Cursors.Hand, VerticalAlignment = VerticalAlignment.Center,
                                 Content = new TextBlock { Text = "Look again", FontSize = 12.5, Foreground = R("Accent"), TextDecorations = TextDecorations.Underline } };
            AutomationProperties.SetAutomationId(b, "capture-refresh");
            b.Click += (o, e) =>
            {
                CaptureDevices.Loaded = AppWindows.Loaded = false;
                BuildDevices();
            };
            return b;
        }

        // ------------------------------------------------------------------------------------------- play
        void BuildChips()
        {
            var chips = E<WrapPanel>("PlayChips");
            chips.Children.Clear();
            string hz = SettingText("hz") ?? "60", scaler = SettingText("scaler") ?? "fsr", engine = SettingText("engine") ?? "auto", interp = SettingText("interp") ?? "true";
            bool vfg = SettingText("model") == "vfg" && (liveState != null || VfgReady);  // not set up: run.py uses RIFE
            bool on = interp != "false";
            Chip(chips, "\uEC4A", on ? hz + " fps" : "Frame generation off");
            Chip(chips, "\uE7B3", scaler == "vsr" ? "NVIDIA RTX VSR" : scaler == "bilinear" ? "Bilinear" : "AMD FSR 1");
            Chip(chips, "\uE950", vfg ? "NVIDIA VFG" : engine == "trt" ? "TensorRT" : engine == "torch" ? "PyTorch" : "Engine: auto");
            if (source != null && source.Kind != "window" && settingSpecs != null)
            {
                string mon = SettingText("monitor");
                var monSpec = settingSpecs.FirstOrDefault(s => s.Key == "monitor");
                if (!string.IsNullOrEmpty(mon) && mon != "primary" && monSpec != null && monSpec.Choices.Count > 1
                    && monSpec.Choices.Any(c => Same(c.Key, mon)))
                    Chip(chips, "\uE7F4", ChoiceName(monSpec, mon));
            }
            chips.ToolTip = "Change these on the Settings page, or press M in the video window.";
        }

        void Chip(Panel host, string glyph, string text)
        {
            var b = new Border { CornerRadius = new CornerRadius(15), Background = Solid("#171B22"), BorderBrush = R("Line2"), BorderThickness = new Thickness(1), Padding = new Thickness(12, 6, 14, 6), Margin = new Thickness(4) };
            var sp = new StackPanel { Orientation = Orientation.Horizontal };
            sp.Children.Add(new TextBlock { Style = S("Icon"), Text = glyph, FontSize = 12, Foreground = R("Accent") });
            sp.Children.Add(new TextBlock { Text = text, FontSize = 12.5, Margin = new Thickness(8, 0, 0, 0), VerticalAlignment = VerticalAlignment.Center });
            b.Child = sp;
            host.Children.Add(b);
        }

        /// <summary>test: "Test with a 60 fps game", played as if it were 30 fps, with a report at the end.</summary>
        void OnPlay(bool test)
        {
            if (!paths.RuntimeReady) { ShowFinishSetup(); return; }
            if (detecting) { Toast("Auto-detect is still testing the graphics card. A few more seconds", true); return; }
            if (!source.RemotePlay) { StartSession(test); return; }  // chiaki-ng isn't involved
            var others = Chiaki.Others(paths);
            if (others.Length == 0) { StartSession(test); return; }
            Ask("\uE7BA", "Warn", "chiaki-ng is already open",
                "SmoovPlay didn't start that copy, so it isn't passing the video on. Close it so SmoovPlay can open it again? A stream running in it ends.",
                "Close it and play", "Play anyway", "Cancel", r =>
                {
                    if (r == 1) Task.Run(() => { Chiaki.Close(others); OnUi(() => StartSession(test)); });
                    else if (r == 2) StartSession(test);
                });
        }

        void StartSession(bool test)
        {
            playLog.Clear();
            Show(E("PlayNote"), false);
            Show(B("BtnPlayDiag"), false);
            ShowResultsCard(false);
            var s = new Session(paths) { TestMode = test, Source = source.Kind };
            if (source.Kind == "capture")
            {
                s.CaptureVideo = CaptureDevices.ChosenVideo(prefs);
                s.CaptureAudio = CaptureDevices.ChosenAudio(prefs);
            }
            if (source.Kind == "window")
            {
                s.WindowApp = AppWindows.Query(AppWindows.Chosen(prefs));
                s.WindowKeysArg = WindowKeys.Arg(prefs);
                WindowKeys.Live = null;  // until this session says what it got
                keysEditing = false;
                keyListening = null;
            }
            session = s;
            liveState = null;
            restartSent = minimisedForPlay = false;
            s.OnLine = line => playLog.Add(line);
            s.OnSettings = json => OnUi(() => OnSettingsState(s, json));
            s.OnWindowKeys = json => OnUi(() =>
            {
                if (session != s) return;
                WindowKeys.FromSession(json);  // (before "[run] playing", whose text says them)
                BuildWindowKeys();
            });
            s.OnState = (st, d) => OnUi(() =>
            {
                if (session != s) return;
                SetPlayState(st, d);
                if (st == PlayState.Stopped || st == PlayState.Ended || st == PlayState.Error) AfterSession();
                if (source.Kind == "window") BuildWindowKeys();  // what was got while playing; then this PC's again
            });
            s.OnNote = n => OnUi(() => { T("PlayNoteText").Text = n; Show(E("PlayNote"), true); });
            SetPlayState(PlayState.Starting, null);
            Task.Run(() =>
            {
                try { s.Start(); }
                catch (Exception ex) { OnUi(() => SetPlayState(PlayState.Error, "Couldn't start: " + ex.Message)); }
            });
        }

        PlayState playState = PlayState.Idle;
        bool minimisedForPlay;  // once per session, when the video first appears (not again after an in-app restart)

        void SetPlayState(PlayState st, string detail)
        {
            bool changed = st != playState;
            playState = st;
            string title, sub;
            bool busy = st == PlayState.Starting || st == PlayState.Waiting || st == PlayState.Found || st == PlayState.Building;
            bool running = busy || st == PlayState.Playing;
            switch (st)
            {
                case PlayState.Starting: title = "Starting SmoovPlay"; sub = "Waking up the AI model..."; break;
                case PlayState.Waiting:
                    title = "Waiting for your stream";
                    sub = source.Kind == "tap" ? "Start the app that sends its frames to SmoovPlay. Its smooth video opens as soon as they come."
                        : "chiaki-ng is open. Connect to your PS5 there and start Remote Play.";
                    break;
                case PlayState.Found: title = "Stream found"; sub = (detail != null ? detail + ". " : "") + "Loading the AI model..."; break;
                case PlayState.Building:
                    if (detail != null) { title = "Applying your settings"; sub = "Restarting the AI model for " + detail + ". A few seconds; the stream keeps going."; }
                    else { title = "Optimising for this stream"; sub = "Once per stream size: a minute or two. Then it's instant."; }
                    break;
                case PlayState.Playing:
                    if (session != null && session.TestMode)
                    {
                        title = "Testing";
                        sub = "Your 60 fps game is playing as if it were 30 fps. Play a few minutes, with some fast motion, then press Q in the video window for the results.";
                    }
                    else if (source.Kind == "window")
                    {
                        title = "Playing";
                        string keys = WindowKeys.Sentence(WindowKeys.Playing(prefs));
                        sub = "The smooth picture is over the app's window, and the app keeps the keyboard and the controller. "
                              + (keys.Length > 0 ? keys + ", from anywhere (keys below)." : "Stop here ends it.") + " Settings here apply straight away.";
                    }
                    else { title = "Playing"; sub = "The smooth video is in its own window. There, M opens the menu, F goes fullscreen and Esc stops. Settings here apply to it straight away."; }
                    break;
                case PlayState.Ended: title = "The stream ended"; sub = "Press Play to connect again."; break;
                case PlayState.Stopped: title = "Stopped"; sub = "Press Play to start again."; break;
                case PlayState.Error:
                    title = "SmoovPlay stopped";
                    sub = Shorten(detail ?? "It stopped unexpectedly.", 180) + "  Copy diagnostics puts the details on the clipboard for a bug report.";
                    break;
                default:
                    title = "Ready when you are";
                    sub = source.RemotePlay ? "Press Play. chiaki-ng opens: connect to your PS5 there, and the smooth video appears in its own window."
                        : source.Kind == "capture" ? "Press Play. The smooth video from your capture card appears in its own window."
                        : source.Kind == "window" ? "Press Play. The smooth picture appears over the app's window, which keeps the keyboard and the controller. SmoovPlay's own keys are below."
                        : source.Kind == "tap" ? "Press Play, then start the app. Its smooth video appears in its own window."
                        : "Press Play to play the test video.";
                    break;
            }
            var t = T("PlayTitle");
            if (changed && t.Text != title)
            {
                t.BeginAnimation(UIElement.OpacityProperty, Anim(0.2, 1, 300));
                T("PlaySub").BeginAnimation(UIElement.OpacityProperty, Anim(0.2, 1, 380));
            }
            t.Text = title;
            T("PlaySub").Text = sub;
            Show(E<ShapePath>("PlayTri"), !running);
            Show(E("PlayStop"), running);
            AutomationProperties.SetName(B("PlayButton"), running ? "Stop" : "Play");
            B("PlayButton").ToolTip = running ? (source.RemotePlay ? "Stop SmoovPlay (chiaki-ng stays open)" : "Stop SmoovPlay") : null;
            E("PlayLive").Visibility = st == PlayState.Playing ? Visibility.Visible : Visibility.Hidden;
            var spin = E<Ellipse>("PlaySpin");
            if (busy)
            {
                if (!(spin.RenderTransform is RotateTransform) || spin.RenderTransform.IsFrozen) spin.RenderTransform = new RotateTransform();
                Forever((RotateTransform)spin.RenderTransform, RotateTransform.AngleProperty, 0, 360, 1400, false);
                spin.BeginAnimation(UIElement.OpacityProperty, Anim(null, 1, 250));
            }
            else spin.BeginAnimation(UIElement.OpacityProperty, Anim(null, 0, 250));
            var glow = E("PlayGlow");
            if (st == PlayState.Playing) Forever(glow, UIElement.OpacityProperty, 0.3, 0.75, 1100, true);
            else if (busy) Forever(glow, UIElement.OpacityProperty, 0.25, 0.55, 1600, true);
            else Forever(glow, UIElement.OpacityProperty, 0.25, 0.45, 3200, true);
            if (st == PlayState.Playing) Forever(E("PlayLiveDot"), UIElement.OpacityProperty, 0.3, 1, 700, true);
            if (st == PlayState.Error)
            {
                Show(B("BtnPlayDiag"), true);
                E<ToggleButton>("BtnPlayDetails").IsChecked = true;
            }
            bool over = st == PlayState.Ended || st == PlayState.Stopped || st == PlayState.Error;
            // Out of the way while the video plays (still in the taskbar, for the log and settings); back when it ends.
            if (st == PlayState.Playing && !minimisedForPlay && session != null)
            {
                minimisedForPlay = true;
                // Without activating anything: WPF's own minimise hands the keys to the next window in line, which
                // took them from the video window (SW_SHOWMINNOACTIVE).
                IntPtr h = IntPtr.Zero;
                try { h = new WindowInteropHelper(Window).Handle; } catch (Exception) { }
                if (h != IntPtr.Zero) ShowWindow(h, 7);
                else Window.WindowState = WindowState.Minimized;
            }
            if (over && changed) BringBack();
            if (over && changed) { liveState = null; restartSent = false; BuildChips(); }
            if (over && changed && detectAfterPlay && !detecting) { detectAfterPlay = false; RunDetect(); }  // (Vfg.cs)
            RefreshSettings();
            OnLooksState();
            UpdateTaskbar(st, title);
            Show(B("BtnPlayTest"), !running && source.TestMode);
            // A session that tested something (this button, or the video window's "Simulate a 30 fps game") left a report.
            if (over && changed && session != null && session.ResultsPath != null && File.Exists(session.ResultsPath))
                ShowResults(session.ResultsPath);
            UpdateAmbient();
            if (changed) Snap("play-" + st.ToString().ToLowerInvariant());
        }

        static string Shorten(string s, int n) { return s.Length <= n ? s : s.Substring(0, n - 3) + "..."; }

        string resultsFile;  // the report on the results card

        /// <summary>A test-mode report (run.py --results) as its verdict and a few numbers; Full report opens all of it.
        /// When a run had several parts (a setting that needs a restart, like 120 fps, starts a new one), the card
        /// shows the last.</summary>
        void ShowResults(string path)
        {
            string text = Diagnostics.ReadShared(path);
            if (text.Trim().Length == 0) return;
            resultsFile = path;
            var lines = text.Split('\n').Select(l => l.TrimEnd('\r')).ToArray();
            Func<string, string> last = prefix => lines.LastOrDefault(l => l.StartsWith(prefix)) ?? "";
            string verdict = lines.LastOrDefault(l => l.StartsWith("Verdict: "));
            verdict = verdict != null ? verdict.Substring("Verdict: ".Length) : "";
            T("ResultsVerdict").Text = verdict.Length > 0 ? char.ToUpper(verdict[0]) + verdict.Substring(1) + "."
                : "Not enough was played to judge. Next time, play for a minute or more.";
            var tiles = E<UniformGrid>("ResultsTiles");
            tiles.Children.Clear();
            string quality = last("Quality: ");
            // "Quality: 6.2% of each AI frame was visibly off ..., against 11% when the last frame is repeated ..."
            var q = Regex.Match(quality, @"([\d.]+%) of each AI frame was visibly off.*?against ([\d.]+%)");
            if (q.Success)
            {
                ResultTile(tiles, "With AI", q.Groups[1].Value, "visibly off", true);
                ResultTile(tiles, "Without AI", q.Groups[2].Value, "visibly off (last frame repeated)", false);
            }
            else if (quality.Contains("not measured")) ResultTile(tiles, "Quality", "n/a", "measured at 60 or 120 fps", false);
            var behind = Regex.Match(last("Running behind: "), @"([\d.]+)%");
            if (behind.Success) ResultTile(tiles, "Running behind", behind.Groups[1].Value + "%", "of frames", false);
            var ms = Regex.Match(last("AI frame time: "), @"([\d.]+) ms");
            if (ms.Success) ResultTile(tiles, "AI frame time", ms.Groups[1].Value + " ms", "out of 16.7 ms", false);
            tiles.Columns = Math.Max(1, tiles.Children.Count);
            Show(tiles, tiles.Children.Count > 0);
            var facts = new List<string>();
            string output = last("Output: ");
            if (output.Length > 0) facts.Add(output.Substring("Output: ".Length) + " output");
            var played = Regex.Match(last("Stream: "), @"played ([\d:]+)");
            if (played.Success) facts.Add(played.Groups[1].Value + " played");
            T("ResultsNote").Text = (facts.Count > 0 ? string.Join(", ", facts) + ". " : "")
                + "Visibly off: the share of each frame more than about 5% brighter or darker than the real one. Lower is "
                + "better; slow scenes score well either way.";
            ShowResultsCard(true);
            Snap("play-results");
        }

        /// <summary>The card needs the room: while it's up, the Play disc shrinks and the settings chips step aside.</summary>
        void ShowResultsCard(bool on)
        {
            Show(E("PlayResults"), on);
            Show(E("PlayChips"), !on);
            double s = on ? 0.72 : 1.0;
            E("PlayDiscArea").LayoutTransform = new ScaleTransform(s, s);
        }

        void ResultTile(Panel host, string label, string value, string hint, bool accent)
        {
            var sp = new StackPanel();
            sp.Children.Add(new TextBlock { Text = label, FontSize = 11.5, Foreground = R("Muted") });
            sp.Children.Add(new TextBlock { Text = value, FontSize = 19, FontWeight = FontWeights.SemiBold, Margin = new Thickness(0, 1, 0, 1), Foreground = accent ? R("Accent") : R("Text") });
            sp.Children.Add(new TextBlock { Text = hint, FontSize = 11.5, Foreground = R("Text2"), TextTrimming = TextTrimming.CharacterEllipsis });
            host.Children.Add(new Border
            {
                CornerRadius = new CornerRadius(9), Background = Solid("#171B22"), BorderBrush = R("Line2"), BorderThickness = new Thickness(1),
                Padding = new Thickness(11, 7, 8, 8), Margin = new Thickness(4, 0, 4, 0), Child = sp,
            });
        }

        void CopyResults()
        {
            string text = resultsFile != null ? Diagnostics.ReadShared(resultsFile).Trim() : "";
            if (text.Length == 0) { Toast("The results file is gone", true); return; }
            try { Clipboard.SetText(text); Toast("Results copied. Paste them into your test report", false); }
            catch (Exception e) { Toast("Couldn't copy: " + e.Message, true); }
        }

        void OpenChiaki()
        {
            if (Chiaki.Ours(paths) != null) { Toast("chiaki-ng is already open", false); return; }
            var others = Chiaki.Others(paths);
            if (others.Length > 0) { Toast("chiaki-ng is already open", false); return; }
            if (Chiaki.Start(paths, l => playLog.Add(l))) Toast("Opening chiaki-ng...", false);
            else Toast("chiaki-ng wasn't found in the install folder. Repair SmoovPlay from Help.", true);
        }

        // ------------------------------------------------------------------------------------------- setup guides
        // One per source: id (its tick is remembered as "guide:<id>"), title, text, and an optional button.
        static readonly string[][] Ps5Guide =
        {
            new[] { "remoteplay", "Turn on Remote Play", "On the PS5: Settings → System → Remote Play → turn on Enable Remote Play.", null },
            new[] { "rest", "Let it wake from rest mode (optional)", "Settings → System → Power Saving → Features Available in Rest Mode: turn on Stay Connected to the Internet and Enable Turning On PS5 from Network.", null },
            new[] { "psn", "Sign in to PSN in chiaki-ng", "Open chiaki-ng. When it asks, sign in to PlayStation Network in the browser window it opens, then paste the address you land on back into chiaki-ng.", "chiaki" },
            new[] { "register", "Register your PS5", "chiaki-ng finds consoles on your network. Click the one marked Automatic Registration Available and choose Yes. Not offered? Use a PIN from the PS5: Settings → System → Remote Play → Link Device.", null },
            new[] { "video", "Stream at 60 fps", "In chiaki-ng's settings, under Video: resolution 1080p (720p on a weak connection) and 60 fps. For HDR, pick the H.265 HDR codec and turn on HDR (Experimental) in SmoovPlay's Settings (Picture); otherwise pick a codec without HDR. SmoovPlay warns you if it's set differently.", null },
            new[] { "hdcp", "Black picture? Turn off HDCP", "Only if you need it: if the picture stays black, or the PS5 says the content can't be shown, go to Settings → System → HDMI on the PS5, turn off Enable HDCP and connect again. Video apps like Netflix need HDCP, so turn it back on when you're done playing.", null },
            new[] { "controller", "Connect your controller", "Plug it into the PC with USB or pair it over Bluetooth. It keeps working while chiaki-ng is minimised.", null },
            new[] { "play", "Press Play", "Always start from SmoovPlay: press Play, then connect to your PS5 in chiaki-ng. The smooth video opens in its own window.", "play" },
        };

        static readonly string[][] WindowGuide =
        {
            new[] { "window-app", "Open the app and start a game", "PS Remote Play, the Xbox app, Moonlight, or a browser for cloud gaming. If the app has a frame rate setting, pick 60 fps.", null },
            new[] { "window-screen", "Keep its window on screen", "Not minimised, and on the monitor plugged into the graphics card. SmoovPlay covers it with the smooth picture (or the whole monitor, in fullscreen) and leaves itself out of what it captures.", null },
            new[] { "window-sound", "Sound and controller", "The app keeps the controller, the keyboard and its own sound. Its sound comes a little ahead of the smooth picture.", null },
            new[] { "window-play", "Pick the window, then press Play", "Choose the app's window under the source buttons on the Play page. The app keeps the keyboard, so SmoovPlay's keys use Ctrl+Alt: Ctrl+Alt+M menu, Ctrl+Alt+F fullscreen, Ctrl+Alt+Esc quit. The Play page lists them all, and you can change them there.", "play" },
        };

        static readonly string[][] CaptureGuide =
        {
            new[] { "capture-wire", "Connect the card", "The console's HDMI output goes into the card's HDMI input, and the card into this PC: USB 3 (a blue port) or a PCIe slot. A card with an HDMI output can pass the picture on to your TV too.", null },
            new[] { "capture-hdcp", "Turn off HDCP on a PS5", "On the PS5: Settings → System → HDMI → turn off Enable HDCP. With it on, the card only gets a blank picture. Turn it back on when you're done: video apps like Netflix need it. Xbox and Switch games don't need this.", null },
            new[] { "capture-mode", "1080p, 60 Hz, HDR off", "Set the console's video output to 1080p at 60 Hz, with HDR off. SmoovPlay expects a 60 Hz signal, in which a 30 fps game shows each frame twice. Picked a higher Capture card resolution in Settings (Experimental)? Set the console to that instead.", null },
            new[] { "capture-sound", "Sound and controller", "Your controller stays connected to the console as usual. SmoovPlay plays the card's sound on this PC, held back to meet the smooth picture. If you'd rather listen through the card's HDMI output, pick no sound for the card on the Play page.", null },
            new[] { "capture-play", "Press Play", "The smooth video from the card opens in its own window.", "play" },
        };

        static readonly string[][] TapGuide =
        {
            new[] { "tap-app", "For app developers: an app that sends its frames to SmoovPlay", "An app built for it can hand its frames straight to SmoovPlay, saying which are repeats, so nothing is captured from the screen. The format is docs/frame-tap.md in SmoovPlay's GitHub repo. For the Xbox app, PS Remote Play, Moonlight or a browser, pick App window instead.", null },
            new[] { "tap-sound", "Sound and controller", "The app keeps the controller and plays its own sound.", null },
            new[] { "tap-play", "Press Play, then start the app", "Either can go first: SmoovPlay waits for the app's frames, then the smooth video opens in its own window.", "play" },
        };

        string[][] GuideSteps
        {
            get
            {
                if (source.RemotePlay) return Ps5Guide;
                if (source.Kind == "capture") return CaptureGuide;
                if (source.Kind == "window") return WindowGuide;
                if (source.Kind == "tap") return TapGuide;
                return new[] { new[] { source.Kind + "-play", source.Name, source.Guide ?? "", "play" } };
            }
        }

        void BuildGuide()
        {
            bool ps5 = source.RemotePlay;
            T("GuideOverline").Text = ps5 || source.Kind == "capture" ? "ONE-TIME SETUP  ·  ABOUT 5 MINUTES" : source.Kind == "window" ? "ONE-TIME SETUP  ·  ABOUT 2 MINUTES" : "TESTING";
            T("GuideTitle").Text = ps5 ? "Connect your PS5" : source.Kind == "capture" ? "Set up your capture card" : source.Kind == "window" ? "Set up window capture" : source.Name;
            T("GuideSub").Text = ps5 ? "SmoovPlay gets the picture from chiaki-ng, a free, unofficial Remote Play app that comes with it. Sony doesn't make or support it. Tick each step as you go; SmoovPlay remembers."
                : "SmoovPlay gets the picture from " + (source.Kind == "capture" ? "a capture card between your console and this PC" : source.Kind == "window" ? "another app's window on this PC" : "a video file") + ". Tick each step as you go; SmoovPlay remembers.";
            Show(B("BtnGuideDocs"), ps5);
            Show(B("BtnGuideChiaki"), ps5);
            var host = E<StackPanel>("GuideSteps");
            host.Children.Clear();
            for (int i = 0; i < GuideSteps.Length; i++)
            {
                var step = GuideSteps[i];
                string key = "guide:" + step[0];
                bool done = prefs.Contains(key);
                var card = new Border { Style = S("Card"), Padding = new Thickness(14, 12, 16, 12), Margin = new Thickness(0, 0, 0, 8) };
                var g = new Grid();
                g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(46) });
                g.ColumnDefinitions.Add(new ColumnDefinition());
                g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                var tick = new Button { Style = S("Bare"), Width = 30, Height = 30, VerticalAlignment = VerticalAlignment.Top, HorizontalAlignment = HorizontalAlignment.Left, Margin = new Thickness(0, 1, 0, 0) };
                AutomationProperties.SetAutomationId(tick, "guide-" + step[0]);
                AutomationProperties.SetName(tick, (done ? "Done: " : "Mark done: ") + step[1]);
                var circle = new Border { CornerRadius = new CornerRadius(15), BorderThickness = new Thickness(1.5) };
                var mark = new TextBlock { HorizontalAlignment = HorizontalAlignment.Center, VerticalAlignment = VerticalAlignment.Center };
                circle.Child = mark;
                tick.Content = circle;
                var texts = new StackPanel();
                Grid.SetColumn(texts, 1);
                var title = new TextBlock { Text = step[1], FontSize = 14.5, FontWeight = FontWeights.SemiBold, TextWrapping = TextWrapping.Wrap, Margin = new Thickness(0, 4, 0, 0) };
                var body = new TextBlock { Style = S("Small"), Text = step[2], Margin = new Thickness(0, 4, 0, 0) };
                texts.Children.Add(title);
                texts.Children.Add(body);
                int n = i + 1;
                Action paint = () =>
                {
                    bool d = prefs.Contains(key);
                    circle.Background = d ? R("Good") : Brushes.Transparent;
                    circle.BorderBrush = d ? R("Good") : Solid("#3A4250");
                    mark.Text = d ? "\uE73E" : n.ToString();
                    mark.FontFamily = d ? (FontFamily)Window.FindResource("Icons") : Window.FontFamily;
                    mark.FontSize = d ? 12 : 12.5;
                    mark.FontWeight = d ? FontWeights.Normal : FontWeights.SemiBold;
                    mark.Foreground = d ? R("Ink") : R("Text2");
                    title.Foreground = d ? R("Text2") : R("Text");
                    card.BorderBrush = d ? Solid("#24503A") : R("Line");
                    tick.ToolTip = d ? "Done. Click to untick." : "Mark as done";
                };
                paint();
                tick.Click += (s, e) =>
                {
                    if (prefs.Contains(key)) prefs.Remove(key); else prefs.Add(key);
                    if (source.RemotePlay)  // the PS5 guide, done: the launcher opens on Play
                    {
                        if (GuideSteps.All(x => prefs.Contains("guide:" + x[0]))) prefs.Add("guide-done"); else prefs.Remove("guide-done");
                    }
                    Prefs.Save(prefs);
                    paint();
                    if (prefs.Contains(key))
                    {
                        var sc = new ScaleTransform();
                        circle.RenderTransformOrigin = new Point(0.5, 0.5);
                        circle.RenderTransform = sc;
                        var pop = new BackEase { EasingMode = EasingMode.EaseOut, Amplitude = 0.7 };
                        sc.BeginAnimation(ScaleTransform.ScaleXProperty, Anim(0.6, 1, 360, 0, pop));
                        sc.BeginAnimation(ScaleTransform.ScaleYProperty, Anim(0.6, 1, 360, 0, pop));
                    }
                    GuideProgress(true);
                };
                g.Children.Add(tick);
                g.Children.Add(texts);
                if (step[3] != null)
                {
                    var act = new Button { Style = S("Pill"), Content = step[3] == "chiaki" ? "Open chiaki-ng" : "Go to Play", VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(12, 4, 0, 0) };
                    if (step[3] == "chiaki") act.Click += (s, e) => OpenChiaki();
                    else act.Click += (s, e) => Nav("PagePlay");
                    Grid.SetColumn(act, 2);
                    g.Children.Add(act);
                }
                card.Child = g;
                host.Children.Add(card);
            }
            GuideProgress(false);
        }

        void GuideProgress(bool animate)
        {
            int done = GuideSteps.Count(x => prefs.Contains("guide:" + x[0]));
            T("GuideCount").Text = done + "/" + GuideSteps.Length;
            var ring = E<Ellipse>("GuideRing");
            double len = (64 - 5) * Math.PI / 5;
            ring.StrokeDashArray = new DoubleCollection { len, len };
            double offset = len * (1 - (double)done / GuideSteps.Length);
            ring.BeginAnimation(Shape.StrokeDashOffsetProperty, Anim(null, offset, animate ? 500 : 1));
            ring.Opacity = done == 0 ? 0 : 1;
            if (done == GuideSteps.Length && animate) Toast("All set. Press Play and enjoy the smoothness", false);
        }

        // ------------------------------------------------------------------------------------------- help
        void BuildKeys()
        {
            var host = E<WrapPanel>("Keys");
            host.Children.Clear();
            foreach (var k in new[] { new[] { "M", "Menu" }, new[] { "F", "Fullscreen" }, new[] { "I", "Frame generation on/off" },
                                      new[] { "S", "Stats" }, new[] { "U", "Upscaler" }, new[] { "Q", "Stop" } })
            {
                var sp = new StackPanel { Orientation = Orientation.Horizontal, Margin = new Thickness(0, 6, 20, 0) };
                sp.Children.Add(new Border
                {
                    MinWidth = 26, Height = 26, CornerRadius = new CornerRadius(6), Background = Solid("#20252E"), BorderBrush = Solid("#323A47"),
                    BorderThickness = new Thickness(1, 1, 1, 3), Padding = new Thickness(7, 0, 7, 0),
                    Child = new TextBlock { Text = k[0], FontSize = 12, FontWeight = FontWeights.SemiBold, VerticalAlignment = VerticalAlignment.Center, HorizontalAlignment = HorizontalAlignment.Center },
                });
                sp.Children.Add(new TextBlock { Text = k[1], FontSize = 12.5, Foreground = R("Text2"), Margin = new Thickness(9, 0, 0, 0), VerticalAlignment = VerticalAlignment.Center });
                host.Children.Add(sp);
            }
        }

        void CopyDiagnostics()
        {
            string last = session != null ? session.LogPath : null;
            Toast("Collecting diagnostics...", false);
            Task.Run(() =>
            {
                string text;
                try { text = Diagnostics.Collect(paths, last); }
                catch (Exception e) { text = "Couldn't collect diagnostics: " + e.Message; }
                OnUi(() =>
                {
                    try { Clipboard.SetText(text); Toast("Diagnostics copied. Paste them into your bug report", false); }
                    catch (Exception e) { Toast("Couldn't copy: " + e.Message, true); }
                });
            });
        }

        void OpenLogs()
        {
            Directory.CreateDirectory(paths.Logs);
            Process.Start("explorer.exe", "\"" + paths.Logs + "\"");
        }

        void AskRepair()
        {
            if (session != null && session.Running) { Toast("Stop playing first", true); return; }
            Ask("\uE90F", "Accent", "Repair SmoovPlay",
                "A quick repair checks every part and puts back anything missing (a few minutes). A full repair downloads everything again (about 5 GB).",
                "Quick repair", "Full repair", "Cancel", r => { if (r == 1 || r == 2) UnlessSaving("Repairing", () => StartInstall(false, r == 2)); });
        }

        /// <summary>The "Update available" link: install it from here (Updater), or open the release page for a copy
        /// that can't update itself.</summary>
        void AskUpdate()
        {
            if (updateUrl == null || updating) return;
            if (Updater.Unavailable(paths) != null) { Proc.Open(updateUrl); return; }
            if (session != null && session.Running) { Toast("Stop playing first, then update", true); return; }
            bool chiaki = Updater.ChiakiInInstall(paths).Length > 0;
            Ask("\uE896", "Accent", "Update to " + V(updateTag) + "?",
                "SmoovPlay downloads it (about 140 MB), closes, updates in a minute or two and opens again. " +
                "Your settings, looks and clips stay." + (chiaki ? " chiaki-ng closes too, since it's updated with it." : ""),
                "Update now", "What's new", "Later", r =>
                {
                    if (r == 1) UnlessSaving("Updating", RunUpdate);
                    else if (r == 2) Proc.Open(updateUrl);
                });
        }

        void RunUpdate()
        {
            updating = true;
            var link = B("RailUpdate");
            link.Content = "Downloading update";
            string repo = GitHub.Repo, tag = updateTag;
            Task.Run(() =>
            {
                try
                {
                    var rel = GitHub.Newest(repo);  // afresh: the check's answer may be a day old
                    if (rel == null || UpdateCheck.Compare(rel.Tag, BuildInfo.Version) <= 0)
                        throw new Exception("GitHub no longer lists a newer release");
                    int shown = -1;
                    string setup = Updater.Fetch(rel, repo, f =>
                    {
                        int pc = (int)(f * 100);
                        if (pc == shown) return;
                        shown = pc;
                        OnUi(() => link.Content = $"Downloading update {pc}%");
                    });
                    OnUi(() => HandOver(setup));
                }
                catch (Exception e)
                {
                    string why = Installer.Message(e);
                    OnUi(() =>
                    {
                        updating = false;
                        link.Content = "Update available: " + V(tag);
                        Toast("Couldn't update: " + why + ". The release page has the setup too.", true, "Release page", () => Proc.Open(updateUrl));
                    });
                }
            });
        }

        /// <summary>The checked setup takes over: chiaki-ng from this install closes, the setup starts, we close.</summary>
        void HandOver(string setup)
        {
            if (session != null && session.Running)  // started playing while it downloaded
            {
                updating = false;
                ResetUpdateLink();
                Toast("The update is ready: stop playing, then click Update available again", false);
                return;
            }
            Chiaki.Close(Updater.ChiakiInInstall(paths));
            try { Updater.Launch(setup, paths); }
            catch (Exception e)
            {
                updating = false;
                ResetUpdateLink();
                Toast("Couldn't start the update: " + e.Message, true);
                return;
            }
            allowClose = true;
            Window.Close();
        }

        void ResetUpdateLink() { B("RailUpdate").Content = "Update available: " + V(updateTag); }

        void AskUninstall()
        {
            if (session != null && session.Running) { Toast("Stop playing first", true); return; }
            Ask("\uE74D", "Bad", "Uninstall SmoovPlay?", "This removes SmoovPlay from " + paths.Install + ". Your settings are kept, in case you come back.",
                "Uninstall", null, "Cancel", r =>
                {
                    if (r != 1) return;
                    UnlessSaving("Uninstalling", () =>
                    {
                        try
                        {
                            Process.Start(new ProcessStartInfo(Path.Combine(paths.Install, "SmoovPlay.exe"), "--uninstall --yes") { WorkingDirectory = Path.GetTempPath() });
                            allowClose = true;
                            Window.Close();
                        }
                        catch (Exception e) { Toast("Couldn't start the uninstaller: " + e.Message, true); }
                    });
                });
        }

        // ------------------------------------------------------------------------------------------- saving
        // Before & after videos carry on in their own process once play stops (Saving). Closing the launcher leaves
        // them be; updating, repairing and uninstalling would stop them, so those ask first.
        DispatcherTimer savingTimer;
        readonly Dictionary<int, string> savingKnown = new Dictionary<int, string>();  // being made: process id -> video
        bool closeWhenStopped;

        /// <summary>A session has ended: close, if that was asked for while it was stopping; otherwise say how any
        /// before & after videos it left being made get on.</summary>
        void AfterSession()
        {
            if (closeWhenStopped)
            {
                allowClose = true;
                Window.Close();
                return;
            }
            WatchSaving();
        }

        void WatchSaving()
        {
            if (savingTimer == null)
            {
                savingTimer = new DispatcherTimer { Interval = TimeSpan.FromSeconds(2) };
                savingTimer.Tick += (s, e) => CheckSaving();
            }
            savingTimer.Start();
            CheckSaving();
        }

        void CheckSaving()
        {
            if (session != null && session.Running) { savingTimer.Stop(); return; }  // a session reports its own
            var all = Saving.Read();
            foreach (var v in all.Where(x => !x.Running))
            {
                if (v.Failed != null) Toast("Couldn't make the before & after video: " + Shorten(v.Failed, 140), true);
                else if (!File.Exists(v.Out)) Toast("The before & after video was stopped before it was finished", true);
                savingKnown.Remove(v.Pid);
                Saving.Forget(v);
            }
            // Finished: it tidies its file away itself.
            var done = savingKnown.Where(k => !all.Any(v => v.Pid == k.Key)).ToList();
            foreach (var k in done) savingKnown.Remove(k.Key);
            var saved = done.Select(k => k.Value).Where(File.Exists).ToList();
            if (saved.Count > 0)
            {
                string last = saved[saved.Count - 1];
                Toast("Saved " + (saved.Count == 1 ? "the before & after video: " + Path.GetFileName(last) : saved.Count + " before & after videos"),
                    false, "Show", () => ShowFile(last));
            }
            var started = all.Where(v => v.Running && !savingKnown.ContainsKey(v.Pid)).ToList();
            foreach (var v in started) savingKnown[v.Pid] = v.Out;
            if (started.Count > 0)
                Toast((started.Count == 1 ? "A before & after video is" : started.Count + " before & after videos are")
                    + " being made, usually for a minute or two. It carries on if you close SmoovPlay.", false);
            if (savingKnown.Count == 0) savingTimer.Stop();
        }

        static void ShowFile(string path)
        {
            try { Process.Start(new ProcessStartInfo("explorer.exe", "/select,\"" + path + "\"") { UseShellExecute = false }); } catch (Exception) { }
        }

        /// <summary>Runs go, after asking first if a before & after video is still being made, which `doing`
        /// ("Updating") would stop.</summary>
        void UnlessSaving(string doing, Action go)
        {
            var making = Saving.Read().Where(v => v.Running).ToList();
            if (making.Count == 0) { go(); return; }
            Ask("", "Warn", making.Count == 1 ? "A video is still being saved" : "Videos are still being saved",
                (making.Count == 1 ? "A before & after video is" : making.Count + " before & after videos are")
                + " still being made, usually for a minute or two more. " + doing + " now stops it, and it won't be saved.",
                doing + " anyway", null, "Wait", r =>
                {
                    if (r != 1) return;
                    Saving.Stop(making);
                    foreach (var v in making) savingKnown.Remove(v.Pid);
                    go();
                });
        }

        // ------------------------------------------------------------------------------------------- dialog, toast
        Action<int> dialogDone;

        /// <summary>An in-window question. done gets 1, 2 or 3 for the buttons (left to right: b1 is the main one,
        /// shown on the right), or 0 if dismissed with Esc.</summary>
        void Ask(string glyph, string toneKey, string title, string text, string b1, string b2, string b3, Action<int> done)
        {
            dialogDone = done;
            T("DialogIcon").Text = glyph;
            T("DialogIcon").Foreground = R(toneKey);
            var tone = ((SolidColorBrush)R(toneKey)).Color;
            E<Ellipse>("DialogIconBg").Fill = new SolidColorBrush(Color.FromArgb(0x26, tone.R, tone.G, tone.B));
            T("DialogTitle").Text = title;
            T("DialogText").Text = text;
            B("DialogBtn1").Content = b1;
            B("DialogBtn2").Content = b2;
            B("DialogBtn3").Content = b3;
            Show(B("DialogBtn2"), b2 != null);
            Show(B("DialogBtn3"), b3 != null);
            var dlg = E("Dialog");
            dlg.BeginAnimation(UIElement.OpacityProperty, null);
            dlg.Opacity = 1;
            Show(dlg, true);
            E("DialogScrim").BeginAnimation(UIElement.OpacityProperty, Anim(0, 1, 180));
            var card = E("DialogCard");
            var sc = new ScaleTransform();
            card.RenderTransform = sc;
            card.BeginAnimation(UIElement.OpacityProperty, Anim(0, 1, 200));
            var ease = new BackEase { EasingMode = EasingMode.EaseOut, Amplitude = 0.3 };
            sc.BeginAnimation(ScaleTransform.ScaleXProperty, Anim(0.94, 1, 300, 0, ease));
            sc.BeginAnimation(ScaleTransform.ScaleYProperty, Anim(0.94, 1, 300, 0, ease));
            B("DialogBtn1").Focus();
            Snap("dialog");
        }

        void CloseDialog(int result)
        {
            var dlg = E("Dialog");
            if (dlg.Visibility != Visibility.Visible) return;
            var fade = Anim(null, 0, 150);
            fade.Completed += (s, e) => { if (dlg.Opacity == 0) { Show(dlg, false); dlg.BeginAnimation(UIElement.OpacityProperty, null); dlg.Opacity = 1; } };
            dlg.BeginAnimation(UIElement.OpacityProperty, fade);
            var done = dialogDone;
            dialogDone = null;
            if (done != null) done(result);
        }

        int toastNo;
        Action toastAction;
        void Toast(string text, bool bad) { Toast(text, bad, null, null); }

        /// <summary>A short message at the bottom of the window. With an action ("Use it now"), it carries a button for
        /// it, takes clicks and stays up longer; a newer toast replaces the offer.</summary>
        void Toast(string text, bool bad, string action, Action onAction)
        {
            int no = ++toastNo;
            T("ToastText").Text = text;
            T("ToastIcon").Text = bad ? "\uE7BA" : "\uE73E";
            T("ToastIcon").Foreground = bad ? R("Warn") : R("Good");
            bool offer = action != null && onAction != null;
            toastAction = offer ? onAction : null;
            B("ToastAction").Content = action;
            Show(E("ToastAction"), offer);
            var t = E("Toast");
            t.IsHitTestVisible = offer;
            var tt = new TranslateTransform();
            t.RenderTransform = tt;
            t.BeginAnimation(UIElement.OpacityProperty, Anim(null, 1, 180));
            tt.BeginAnimation(TranslateTransform.YProperty, Anim(10, 0, 260));
            var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds((offer ? 10000 : 2800) + text.Length * 20) };
            timer.Tick += (s, e) => { timer.Stop(); if (no == toastNo) HideToast(); };
            timer.Start();
            SnapLater("toast", 400);
        }

        void HideToast()
        {
            ++toastNo;
            toastAction = null;
            var t = E("Toast");
            t.IsHitTestVisible = false;
            t.BeginAnimation(UIElement.OpacityProperty, Anim(null, 0, 300));
        }

        void ToastActionClicked()
        {
            var a = toastAction;
            HideToast();
            if (a != null) a();
        }

        void Busy(string button, Func<string> work, Action<string> done)
        {
            var b = B(button);
            b.IsEnabled = false;
            Task.Run(() => { string r; try { r = work(); } catch (Exception e) { r = e.Message; } OnUi(() => { b.IsEnabled = true; done(r); }); });
        }

        [DllImport("user32.dll")] static extern bool FlashWindow(IntPtr hwnd, bool invert);
        [DllImport("user32.dll")] static extern bool IsIconic(IntPtr hwnd);
        [DllImport("user32.dll")] static extern bool ShowWindow(IntPtr hwnd, int cmd);

        /// <summary>Up from the taskbar when a session ends. Asks Windows too: a window minimised from outside
        /// (UI Automation, some tools) can leave WPF's WindowState at Normal.</summary>
        void BringBack()
        {
            IntPtr h = IntPtr.Zero;
            try { h = new WindowInteropHelper(Window).Handle; } catch (Exception) { }
            bool iconic = h != IntPtr.Zero && IsIconic(h);
            Program.Trace($"session over: window {Window.WindowState}, iconic {iconic}");
            if (Window.WindowState != WindowState.Minimized && !iconic) return;
            if (Window.WindowState == WindowState.Minimized) Window.WindowState = WindowState.Normal;
            if (h != IntPtr.Zero && IsIconic(h)) ShowWindow(h, 9);  // SW_RESTORE
            Window.Activate();
            Flash();  // if Windows kept it behind another window
        }
        /// <summary>Get attention in the taskbar when a long wait ends while the window is in the background.</summary>
        void Flash()
        {
            if (Window.IsActive) return;
            try { FlashWindow(new WindowInteropHelper(Window).Handle, true); } catch (Exception) { }
        }

        // ------------------------------------------------------------------------------------------- closing
        void OnClosing(object sender, CancelEventArgs e)
        {
            if (allowClose) return;
            if (installer != null)
            {
                e.Cancel = true;
                Ask("\uE769", "Accent", "Setup is still running", "Stop it now? Running SmoovPlay again picks up where it stopped.",
                    "Stop and close", null, "Keep going", r =>
                    {
                        if (r != 1) return;
                        var i = installer;
                        if (i != null) i.Cancel();
                        allowClose = true;
                        Window.Close();
                    });
                return;
            }
            if (session != null && session.Running)
            {
                e.Cancel = true;
                Ask("\uE768", "Accent", "SmoovPlay is running", "Stop it and close? chiaki-ng stays open.", "Stop and close", null, "Keep playing", r =>
                {
                    if (r != 1) return;
                    // Closes once the session has gone: it finishes a clip being saved first (a second or two).
                    closeWhenStopped = true;
                    var s = session;
                    s.Stop();
                    if (!s.Running) { allowClose = true; Window.Close(); }
                    else Window.Hide();
                });
            }
        }

        // ------------------------------------------------------------------------------------------- recording
        // SMOOVPLAY_RECORD=<folder> records the window as WPF renders it (never the screen), for demo videos:
        // <folder>\frames.mkv (lossless, encoded on the GPU by ffmpeg on PATH; the rounded corners come out black),
        // frames.txt (when each frame was taken, ms) and size.txt. SMOOVPLAY_RECORD_FPS (30) and _SCALE (1.5).
        void StartRecorder()
        {
            string dir = Environment.GetEnvironmentVariable("SMOOVPLAY_RECORD");
            if (string.IsNullOrEmpty(dir)) return;
            Func<string, double, double> num = (name, dflt) =>
            {
                double v;
                return double.TryParse(Environment.GetEnvironmentVariable(name), System.Globalization.NumberStyles.Float,
                    System.Globalization.CultureInfo.InvariantCulture, out v) && v > 0 ? v : dflt;
            };
            double fps = num("SMOOVPLAY_RECORD_FPS", 30), scale = num("SMOOVPLAY_RECORD_SCALE", 1.5);
            Directory.CreateDirectory(dir);
            var root = E<Grid>("Root");
            var queue = new System.Collections.Concurrent.BlockingCollection<byte[]>(240);
            var pool = new System.Collections.Concurrent.ConcurrentBag<byte[]>();  // frame buffers, reused
            RenderTargetBitmap bmp = null;
            var times = new StreamWriter(Path.Combine(dir, "frames.txt")) { AutoFlush = true };
            var clock = new Stopwatch();
            Process ff = null;
            int w = 0, h = 0, n = 0, dropped = 0;
            double next = 0, renderMs = 0;
            TimeSpan lastFrame = TimeSpan.Zero;
            EventHandler onRender = null;
            onRender = (s, e) =>
            {
                var re = e as RenderingEventArgs;
                if (re != null && re.RenderingTime == lastFrame) return;  // raised more than once per frame
                if (re != null) lastFrame = re.RenderingTime;
                double now = clock.Elapsed.TotalMilliseconds;
                if (now < next) return;
                next = Math.Max(next + 1000 / fps, now);
                try
                {
                    int fw = (int)Math.Round(root.ActualWidth * scale) & ~1, fh = (int)Math.Round(root.ActualHeight * scale) & ~1;
                    if (ff == null)
                    {
                        w = fw; h = fh;
                        File.WriteAllText(Path.Combine(dir, "size.txt"), w + "x" + h);
                        // Lossless on the GPU, so the UI thread never waits for the encoder. The rounded corners
                        // come out black (no alpha); whoever uses the video clips them.
                        ff = Process.Start(new ProcessStartInfo("ffmpeg",
                            $"-v error -y -f rawvideo -pix_fmt bgra -s {w}x{h} -r {fps} -i - -c:v hevc_nvenc -preset p4 -tune lossless -pix_fmt yuv444p \"{Path.Combine(dir, "frames.mkv")}\"")
                        { UseShellExecute = false, RedirectStandardInput = true, CreateNoWindow = true });
                        var stdin = ff.StandardInput.BaseStream;
                        var proc = ff;
                        new System.Threading.Thread(() =>
                        {
                            foreach (var b in queue.GetConsumingEnumerable()) { stdin.Write(b, 0, b.Length); pool.Add(b); }
                            stdin.Close();
                            proc.WaitForExit();  // keep the process alive until the file is complete
                        }) { IsBackground = false }.Start();
                    }
                    if (fw != w || fh != h) return;
                    var t0 = clock.Elapsed.TotalMilliseconds;
                    var dv = new DrawingVisual();
                    using (var dc = dv.RenderOpen())
                        dc.DrawRectangle(new VisualBrush(root), null, new Rect(0, 0, root.ActualWidth, root.ActualHeight));
                    if (bmp == null) bmp = new RenderTargetBitmap(w, h, 96 * scale, 96 * scale, PixelFormats.Pbgra32);
                    else bmp.Clear();
                    bmp.Render(dv);
                    byte[] buf;
                    if (!pool.TryTake(out buf)) buf = new byte[w * h * 4];
                    bmp.CopyPixels(buf, w * 4, 0);
                    if (!queue.TryAdd(buf)) { dropped++; pool.Add(buf); return; }
                    times.WriteLine(now.ToString("F1", System.Globalization.CultureInfo.InvariantCulture));
                    renderMs += clock.Elapsed.TotalMilliseconds - t0;
                    if (++n % 300 == 0)
                    {
                        Program.Trace($"recording: {n} frames, {renderMs / 300:F1} ms each to render, {dropped} dropped");
                        renderMs = 0;
                    }
                }
                catch (Exception ex)
                {
                    Program.Trace("recording stopped: " + ex.Message);
                    CompositionTarget.Rendering -= onRender;
                }
            };
            Window.ContentRendered += (s, e) =>
            {
                clock.Start();
                CompositionTarget.Rendering += onRender;
            };
            Window.Closed += (s, e) =>
            {
                CompositionTarget.Rendering -= onRender;
                queue.CompleteAdding();
                times.Dispose();
            };
        }

        // ------------------------------------------------------------------------------------------- snapshots
        void Snap(string name) { SnapLater(name, 700); }

        void SnapLater(string name, int ms)
        {
            if (string.IsNullOrEmpty(shots)) return;
            var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(ms) };
            timer.Tick += (s, e) =>
            {
                timer.Stop();
                try
                {
                    double scale = 1.5;
                    var bmp = new RenderTargetBitmap((int)(Window.ActualWidth * scale), (int)(Window.ActualHeight * scale), 96 * scale, 96 * scale, PixelFormats.Pbgra32);
                    bmp.Render(Window);
                    Directory.CreateDirectory(shots);
                    var enc = new PngBitmapEncoder();
                    enc.Frames.Add(BitmapFrame.Create(bmp));
                    string file = $"{(setupMode ? "setup" : "app")}-{++shotNo:00}-{name}.png";
                    using (var f = File.Create(Path.Combine(shots, file))) enc.Save(f);
                }
                catch (Exception ex) { Program.Trace("snapshot failed: " + ex.Message); }
            };
            timer.Start();
        }
    }

    /// <summary>Lines from a background process, added to a TextBox a few times a second (not once per line).</summary>
    class LogPump
    {
        readonly TextBox box;
        readonly ConcurrentQueue<string> queue = new ConcurrentQueue<string>();

        public LogPump(TextBox box)
        {
            this.box = box;
            var t = new DispatcherTimer(TimeSpan.FromMilliseconds(200), DispatcherPriority.Background, (s, e) => Flush(), box.Dispatcher);
            t.Start();
        }

        public void Add(string line) { queue.Enqueue(line); }

        public void Clear()
        {
            string x;
            while (queue.TryDequeue(out x)) { }
            box.Clear();
        }

        void Flush()
        {
            if (queue.IsEmpty) return;
            var sb = new StringBuilder();
            string line;
            while (queue.TryDequeue(out line)) sb.Append(line).Append(Environment.NewLine);
            bool atEnd = box.VerticalOffset + box.ViewportHeight >= box.ExtentHeight - 4;
            if (box.Text.Length > 80000) box.Text = box.Text.Substring(box.Text.Length - 40000);
            box.AppendText(sb.ToString());
            if (atEnd) box.ScrollToEnd();
        }
    }

    /// <summary>Windows' own folder picker (the modern one, not the old tree).</summary>
    static class FolderPicker
    {
        [ComImport, Guid("DC1C5A9C-E88A-4dde-A5A1-60F82A20AEF7")] class FileOpenDialogCom { }

        [ComImport, Guid("42f85136-db7e-439c-85f1-e4075d135fc8"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IFileOpenDialog
        {
            [PreserveSig] int Show(IntPtr parent);
            void SetFileTypes(uint c, IntPtr specs);
            void SetFileTypeIndex(uint i);
            void GetFileTypeIndex(out uint i);
            void Advise(IntPtr sink, out uint cookie);
            void Unadvise(uint cookie);
            void SetOptions(uint fos);
            void GetOptions(out uint fos);
            void SetDefaultFolder(IShellItem si);
            void SetFolder(IShellItem si);
            void GetFolder(out IShellItem si);
            void GetCurrentSelection(out IShellItem si);
            void SetFileName([MarshalAs(UnmanagedType.LPWStr)] string name);
            void GetFileName([MarshalAs(UnmanagedType.LPWStr)] out string name);
            void SetTitle([MarshalAs(UnmanagedType.LPWStr)] string title);
            void SetOkButtonLabel([MarshalAs(UnmanagedType.LPWStr)] string text);
            void SetFileNameLabel([MarshalAs(UnmanagedType.LPWStr)] string label);
            void GetResult(out IShellItem si);
            void AddPlace(IShellItem si, int fdap);
            void SetDefaultExtension([MarshalAs(UnmanagedType.LPWStr)] string ext);
            void Close(int hr);
            void SetClientGuid(ref Guid g);
            void ClearClientData();
            void SetFilter(IntPtr filter);
            void GetResults(out IntPtr items);
            void GetSelectedItems(out IntPtr items);
        }

        [ComImport, Guid("43826D1E-E718-42EE-BC55-A1E261C37BFE"), InterfaceType(ComInterfaceType.InterfaceIsIUnknown)]
        interface IShellItem
        {
            void BindToHandler(IntPtr pbc, ref Guid bhid, ref Guid riid, out IntPtr ppv);
            void GetParent(out IShellItem si);
            void GetDisplayName(uint sigdn, [MarshalAs(UnmanagedType.LPWStr)] out string name);
            void GetAttributes(uint mask, out uint attribs);
            void Compare(IShellItem si, uint hint, out int order);
        }

        [DllImport("shell32.dll", CharSet = CharSet.Unicode)]
        static extern int SHCreateItemFromParsingName(string path, IntPtr pbc, [In] ref Guid riid, out IShellItem item);

        public static string Pick(IntPtr owner, string title, string start)
        {
            IFileOpenDialog dlg = null;
            try
            {
                dlg = (IFileOpenDialog)new FileOpenDialogCom();
                uint o;
                dlg.GetOptions(out o);
                dlg.SetOptions(o | 0x20 | 0x40 | 0x800);  // FOS_PICKFOLDERS | FOS_FORCEFILESYSTEM | FOS_PATHMUSTEXIST
                dlg.SetTitle(title);
                dlg.SetOkButtonLabel("Choose");
                if (!string.IsNullOrEmpty(start) && Directory.Exists(start))
                {
                    var iid = typeof(IShellItem).GUID;
                    IShellItem si;
                    if (SHCreateItemFromParsingName(start, IntPtr.Zero, ref iid, out si) == 0) dlg.SetFolder(si);
                }
                if (dlg.Show(owner) != 0) return null;  // cancelled
                IShellItem res;
                dlg.GetResult(out res);
                string path;
                res.GetDisplayName(0x80058000, out path);  // SIGDN_FILESYSPATH
                return path;
            }
            catch (Exception) { return null; }
            finally { if (dlg != null) Marshal.ReleaseComObject(dlg); }
        }
    }
}
