// The launcher's Settings page. Its rows come from the pipeline's own list (python -m framegen.settings, see
// framegen/settings.py), so a new setting appears here with no launcher change.
//
// Layout: the few big choices first ("quick" in the list: frame rate, model, upscaler), as rows of choices you can
// see at a glance, then Auto-detect, then everything else in folding sections, one per group (one column, or two in
// a wide window). A section remembers whether it's open (launcher prefs, "set-open:<group>"). A setting that only
// matters with another one ("applies" in the list, e.g. RTX VSR quality with RTX VSR) is hidden while it doesn't.
// Two to four choices show as a row of choices with the picked one's hint under it; more as a dropdown.
//
// Nothing playing: a change is written to the settings file, which the next session reads (with the keys its "also"
// lists, as framegen/settings.py's Settings.set writes them).
// Playing: a change goes to the running pipeline (Session.Send, framegen/remote.py) and reaches the video straight
// away, exactly as if it were picked in the video window's menu (M). The page then follows the pipeline's
// "[settings-state]" lines, so changes made in that menu or with its keys show up here too, along with which choices
// this PC can use and restart-only changes that are waiting ("Restart now").
// Starting (before the video is up): the rows are locked, because the pipeline has already read the file but can't
// take commands yet.

using System;
using System.Collections.Generic;
using System.Globalization;
using System.IO;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Web.Script.Serialization;
using System.Windows;
using System.Windows.Automation;
using System.Windows.Controls;
using System.Windows.Controls.Primitives;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Threading;

namespace SmoovPlay
{
    partial class Ui
    {
        class SettingSpec
        {
            public string Key, Label, Group, Help, Later;
            public string Picture, Speed;  // rough effect on the picture and on speed or delay ("impact"), or null
            public bool Live, Saved, Quick;
            public bool Experimental;  // built without the hardware to test it: the EXPERIMENTAL badge and Send feedback
            public object Default;
            public readonly List<KeyValuePair<object, string>> Choices = new List<KeyValuePair<object, string>>();
            public readonly List<string> Hints = new List<string>();  // a line under each choice ("" = none)
            public string AppliesKey;  // only used while that setting is one of AppliesValues (null: always)
            public readonly List<object> AppliesValues = new List<object>();
            public readonly List<string> Sources = new List<string>();  // only used with these sources (none: all)
            public readonly Dictionary<string, string> Elsewhere = new Dictionary<string, string>();  // source -> why not ("": any)
            public readonly Dictionary<string, object> Also = new Dictionary<string, object>();  // set with this one
            public bool OnOff { get { return Choices.Count == 2 && Choices.All(c => c.Key is bool); } }

            public string Hint(object value)
            {
                int i = Choices.FindIndex(c => Same(c.Key, value));
                return i >= 0 ? Hints[i] : "";
            }

            public bool AppliesTo(Func<string, object> value)
            {
                if (Choices.Count <= 1) return false;
                return AppliesKey == null || AppliesValues.Any(a => Same(a, value(AppliesKey)));
            }

            /// <summary>Why it isn't used with frames from this source (a SourceDef.Kind), or null if it is
            /// (settings.py Spec.unused_with).</summary>
            public string UnusedWith(string source)
            {
                if (Sources.Count == 0 || Sources.Contains(source)) return null;
                string why;
                if (Elsewhere.TryGetValue(source, out why) && !string.IsNullOrEmpty(why)) return why;
                if (Elsewhere.TryGetValue("", out why) && !string.IsNullOrEmpty(why)) return why;
                return "Not used with this source.";
            }
        }

        class SettingRow
        {
            public SettingSpec Spec;
            public SettingSpec Off;          // the frame-rate pick only: Frame generation, shown as its "Off" choice
            public CheckBox Switch;          // on/off settings
            public List<RadioButton> Segs;   // 2 to 4 choices, Tag = the value (OffTag: frame generation off)
            public ComboBox Drop;            // more
            public List<ComboBoxItem> Items; // the dropdown's choices, Tag = the value (not the "later" note)
            public TextBlock Hint;           // under a row of choices: the picked one's hint
            public TextBlock Note;
            public FrameworkElement RestartTag;
            public FrameworkElement Element, Rule;  // the row, and the line above it (both hidden when it doesn't apply)
        }

        class SettingSection
        {
            public string Group;
            public Border Card;
            public FrameworkElement Body;
            public TextBlock Summary;
            public RotateTransform Chevron;
            public readonly List<SettingRow> Rows = new List<SettingRow>();
        }

        static readonly JavaScriptSerializer Json = new JavaScriptSerializer();
        static readonly object OffTag = new object();
        const double WideSettings = 820;  // from this content width, two columns of sections
        static readonly Dictionary<string, string> GroupIcons = new Dictionary<string, string>
        {
            { "Smoothness", "\uEC4A" }, { "Picture", "\uE7B3" }, { "Display", "\uE7F4" }, { "Instant replay", "\uE714" },
            { "Looks", "\uE790" }, { "PS5 Remote Play", "\uE7FC" },
        };
        List<SettingSpec> settingSpecs;
        string settingsError;
        bool loadingSpecs, syncingSettings, restartSent;
        bool? settingsWide;
        readonly Dictionary<string, SettingRow> settingRows = new Dictionary<string, SettingRow>();
        readonly List<SettingRow> quickRows = new List<SettingRow>();
        readonly List<SettingSection> settingSections = new List<SettingSection>();
        Dictionary<string, object> liveState;  // the running session's last state line; null when none is reporting

        // ---- the list of settings, from the pipeline
        /// <summary>force: again, e.g. after the Looks page added a look (a choice of "Which look"); the old list stays
        /// up meanwhile, and if loading fails.</summary>
        void LoadSettingSpecs(bool force = false)
        {
            if (vfgStatus == null) LoadVfgStatus();
            if (loadingSpecs || (settingSpecs != null && !force) || !paths.RuntimeReady) return;
            loadingSpecs = true;
            Task.Run(() =>
            {
                List<SettingSpec> specs = null;
                string err = null;
                try
                {
                    var lines = new List<string>();
                    int rc = Proc.Run(Proc.Python(paths.Python, "-m framegen.settings", paths.App), l => { lock (lines) lines.Add(l); }, null);
                    string json = lines.LastOrDefault(l => l.StartsWith("{"));
                    if (rc != 0 || json == null) err = lines.LastOrDefault(l => l.Trim().Length > 0) ?? ("exit code " + rc);
                    else specs = ParseSpecs(json);
                }
                catch (Exception e) { err = e.Message; }
                if (err != null) Program.Trace("settings list: " + err);
                OnUi(() =>
                {
                    loadingSpecs = false;
                    if (specs != null || !force) { settingSpecs = specs; settingsError = err; }
                    BuildSettings();
                });
            });
        }

        static List<SettingSpec> ParseSpecs(string json)
        {
            var root = (Dictionary<string, object>)Json.DeserializeObject(json);
            var list = new List<SettingSpec>();
            foreach (Dictionary<string, object> d in (object[])root["specs"])
            {
                var s = new SettingSpec
                {
                    Key = (string)d["key"], Label = (string)d["label"], Group = (string)d["group"], Help = (string)d["help"],
                    Live = (bool)d["live"], Saved = (bool)d["saved"], Default = d["default"],
                    Later = d.ContainsKey("later") ? d["later"] as string : null,
                    Quick = d.ContainsKey("quick") && d["quick"] is bool && (bool)d["quick"],
                    Experimental = d.ContainsKey("experimental") && d["experimental"] is bool && (bool)d["experimental"],
                };
                foreach (Dictionary<string, object> c in (object[])d["choices"])
                {
                    s.Choices.Add(new KeyValuePair<object, string>(c["value"], (string)c["name"]));
                    s.Hints.Add(c.ContainsKey("hint") ? c["hint"] as string ?? "" : "");
                }
                var applies = d.ContainsKey("applies") ? d["applies"] as Dictionary<string, object> : null;
                if (applies != null)
                {
                    s.AppliesKey = (string)applies["key"];
                    s.AppliesValues.AddRange((object[])applies["values"]);
                }
                if (d.ContainsKey("sources") && d["sources"] is object[])
                    s.Sources.AddRange(((object[])d["sources"]).OfType<string>());
                if (d.ContainsKey("elsewhere") && d["elsewhere"] is Dictionary<string, object>)
                    foreach (var kv in (Dictionary<string, object>)d["elsewhere"]) s.Elsewhere[kv.Key] = kv.Value as string;
                if (d.ContainsKey("also") && d["also"] is Dictionary<string, object>)
                    foreach (var kv in (Dictionary<string, object>)d["also"]) s.Also[kv.Key] = kv.Value;
                var impact = d.ContainsKey("impact") ? d["impact"] as Dictionary<string, object> : null;
                if (impact != null)
                {
                    s.Picture = impact.ContainsKey("picture") ? impact["picture"] as string : null;
                    s.Speed = impact.ContainsKey("speed") ? impact["speed"] as string : null;
                }
                list.Add(s);
            }
            if (root.ContainsKey("impact_pc")) impactPc = root["impact_pc"] as string;
            return list;
        }

        static string impactPc;  // where the impact figures were measured ("an RTX 4070 Ti SUPER")

        static string ImpactText(SettingSpec spec)
        {
            return (spec.Picture != null ? "\n\nPicture: " + spec.Picture : "") + (spec.Speed != null ? (spec.Picture != null ? "\n" : "\n\n") + "Speed: " + spec.Speed : "");
        }

        /// <summary>Under a section row's help: what it does to the picture and to speed or delay, roughly.</summary>
        FrameworkElement ImpactLines(SettingSpec spec)
        {
            var panel = new StackPanel { Margin = new Thickness(0, 8, 0, 0) };
            foreach (var line in new[] { new[] { "\uE7B3", "Picture", spec.Picture }, new[] { "\uEC4A", "Speed", spec.Speed } })
            {
                if (string.IsNullOrEmpty(line[2])) continue;
                var g = new Grid { Margin = new Thickness(0, panel.Children.Count > 0 ? 4 : 0, 0, 0) };
                g.ColumnDefinitions.Add(new ColumnDefinition { Width = new GridLength(22) });
                g.ColumnDefinitions.Add(new ColumnDefinition());
                g.Children.Add(new TextBlock
                {
                    Style = S("Icon"), Text = line[0], FontSize = 12, Foreground = R("Accent"), HorizontalAlignment = HorizontalAlignment.Left,
                    VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 0, 0),
                });
                var text = new TextBlock { Style = S("Small"), Foreground = R("Text") };
                text.Inlines.Add(new System.Windows.Documents.Run(line[1] + "  ") { FontWeight = FontWeights.SemiBold });
                text.Inlines.Add(new System.Windows.Documents.Run(line[2]) { Foreground = R("Text2") });
                Grid.SetColumn(text, 1);
                g.Children.Add(text);
                if (line[1] == "Speed" && impactPc != null && line[2].Any(char.IsDigit))
                    g.ToolTip = "Rough figures, measured on " + impactPc + ". Yours will differ.";
                panel.Children.Add(g);
            }
            return panel.Children.Count > 0 ? panel : null;
        }

        /// <summary>JSON values as the serializer reads them: numbers may be int or decimal (1 vs 1.0), so compare
        /// those as numbers.</summary>
        static bool Same(object a, object b)
        {
            if (a == null || b == null) return a == null && b == null;
            if (IsNumber(a) && IsNumber(b))
                return Convert.ToDouble(a, CultureInfo.InvariantCulture) == Convert.ToDouble(b, CultureInfo.InvariantCulture);
            return a.Equals(b);
        }
        static bool IsNumber(object o) { return o is int || o is long || o is decimal || o is double || o is float; }

        // ---- the settings file (when nothing is playing)
        static Dictionary<string, object> ReadSettingsFile()
        {
            try
            {
                if (File.Exists(Paths.SettingsFile))
                {
                    var d = Json.DeserializeObject(File.ReadAllText(Paths.SettingsFile)) as Dictionary<string, object>;
                    if (d != null) return d;
                }
            }
            catch (Exception) { }  // unreadable: the pipeline uses its defaults too
            return new Dictionary<string, object>();
        }

        /// <summary>Keeps every other key as it is, in the pipeline's own layout (indent 2, no byte-order mark: Python's
        /// json module refuses one).</summary>
        static void WriteSetting(string key, object value)
        {
            var d = ReadSettingsFile();
            d[key] = value;
            string text = "{\n" + string.Join(",\n", d.Select(kv => "  " + Json.Serialize(kv.Key) + ": " + Json.Serialize(kv.Value))) + "\n}";
            Directory.CreateDirectory(Path.GetDirectoryName(Paths.SettingsFile));
            File.WriteAllText(Paths.SettingsFile, text, new UTF8Encoding(false));
        }

        /// <summary>A setting's current value as plain text ("90", "fsr", "true"): the running session's if one is
        /// reporting, else the file's. Null if neither has it.</summary>
        string SettingText(string key)
        {
            var values = liveState != null ? liveState["values"] as Dictionary<string, object> : null;
            if (values != null && values.ContainsKey(key)) return Json.Serialize(values[key]).Trim('"');
            return Prefs.Setting(key);
        }

        // ---- the page
        void ShowSettings()
        {
            LoadSettingSpecs(settingSpecsStale);
            if (loadingSpecs) settingSpecsStale = false;
            ShowPage("PageSettings");
            RefreshSettings();
        }

        void BuildSettings()
        {
            var quick = E<StackPanel>("SetQuick");
            quick.Children.Clear();
            settingRows.Clear();
            quickRows.Clear();
            settingSections.Clear();
            if (settingSpecs != null)
            {
                var picks = settingSpecs.Where(s => s.Saved && s.Quick).ToList();
                if (picks.Count > 0) quick.Children.Add(BuildQuickPicks(picks));
                // (this-session-only ones, like test mode, have their own button on the Play page)
                foreach (var group in settingSpecs.Where(s => s.Saved && !s.Quick).GroupBy(s => s.Group))
                    settingSections.Add(BuildSection(group.Key, group.ToList()));
            }
            LayoutSections(true);
            RefreshSettings();
            BuildChips();
            if (page == "PageSettings") SnapLater("settings-built", 600);
        }

        /// <summary>The big choices, in one card. Frame generation and its frame rate are one pick: Off, 60, 90, 120.</summary>
        FrameworkElement BuildQuickPicks(List<SettingSpec> picks)
        {
            var interp = picks.FirstOrDefault(s => s.Key == "interp");
            var hz = picks.FirstOrDefault(s => s.Key == "hz");
            bool rate = interp != null && interp.OnOff && hz != null;
            var list = new StackPanel();
            foreach (var spec in picks)
            {
                if (rate && spec == interp) continue;  // it's the frame-rate pick's "Off"
                var row = BuildRow(spec, rate && spec == hz ? interp : null, true);
                if (list.Children.Count > 0) { row.Rule = Rule(); list.Children.Add(row.Rule); }
                list.Children.Add(row.Element);
                quickRows.Add(row);
            }
            return new Border { Style = S("Card"), Padding = new Thickness(0), Margin = new Thickness(0, 0, 0, 16), Child = list };
        }

        SettingSection BuildSection(string group, List<SettingSpec> specs)
        {
            var sec = new SettingSection { Group = group };
            bool open = prefs.Contains("set-open:" + group);

            var head = new Grid();
            head.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            head.ColumnDefinitions.Add(new ColumnDefinition());
            head.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            string glyph;
            if (!GroupIcons.TryGetValue(group, out glyph)) glyph = "\uE713";
            head.Children.Add(new Border
            {
                Width = 32, Height = 32, CornerRadius = new CornerRadius(9), Background = Solid("#1E222B"), Margin = new Thickness(0, 0, 14, 0),
                VerticalAlignment = VerticalAlignment.Center,
                Child = new TextBlock { Style = S("Icon"), Text = glyph, FontSize = 15, Foreground = R("Accent") },
            });
            var text = new StackPanel { VerticalAlignment = VerticalAlignment.Center };
            text.Children.Add(new TextBlock { Text = group, FontSize = 14, FontWeight = FontWeights.SemiBold });
            sec.Summary = new TextBlock { FontSize = 12, Foreground = R("Muted"), TextTrimming = TextTrimming.CharacterEllipsis, Margin = new Thickness(0, 2, 0, 0) };
            text.Children.Add(sec.Summary);
            Grid.SetColumn(text, 1);
            head.Children.Add(text);
            sec.Chevron = new RotateTransform(open ? 180 : 0);
            var chev = new TextBlock
            {
                Style = S("Icon"), Text = "\uE70D", FontSize = 12, Foreground = R("Text2"), Margin = new Thickness(14, 0, 2, 0),
                RenderTransformOrigin = new Point(0.5, 0.5), RenderTransform = sec.Chevron,
            };
            Grid.SetColumn(chev, 2);
            head.Children.Add(chev);
            var btn = new Button { Style = S("Fold"), Padding = new Thickness(16, 12, 16, 12), Content = head };
            AutomationProperties.SetName(btn, group);
            AutomationProperties.SetAutomationId(btn, "SetFold_" + group.Replace(" ", ""));
            AutomationProperties.SetItemStatus(btn, open ? "open" : "closed");

            var rows = new StackPanel();
            foreach (var spec in specs)
            {
                var row = BuildRow(spec, null, false);
                if (rows.Children.Count > 0) { row.Rule = Rule(); rows.Children.Add(row.Rule); }
                rows.Children.Add(row.Element);
                sec.Rows.Add(row);
            }
            var body = new StackPanel();
            body.Children.Add(new Border { Height = 1, Background = R("Line") });
            body.Children.Add(rows);
            sec.Body = body;
            Show(body, open);

            var all = new StackPanel();
            all.Children.Add(btn);
            all.Children.Add(body);
            sec.Card = new Border { Style = S("Card"), Padding = new Thickness(0), Margin = new Thickness(0, 0, 0, 14), Child = all };
            btn.Click += (s, e) =>
            {
                bool now = sec.Body.Visibility != Visibility.Visible;
                Show(sec.Body, now);
                sec.Chevron.BeginAnimation(RotateTransform.AngleProperty, Anim(null, now ? 180 : 0, 180));
                AutomationProperties.SetItemStatus(btn, now ? "open" : "closed");
                if (now) prefs.Add("set-open:" + group); else prefs.Remove("set-open:" + group);
                Prefs.Save(prefs);
                if (now) SnapLater("settings-" + group.ToLowerInvariant().Replace(' ', '-'), 400);
            };
            return sec;
        }

        /// <summary>The sections in one column, or two once there's room (on a size change, only when that flips).</summary>
        void LayoutSections(bool force)
        {
            var host = E<Grid>("SetRows");
            bool wide = E("SetScroll").ActualWidth - 14 >= WideSettings;
            if (!force && settingsWide == wide) return;
            settingsWide = wide;
            foreach (var col in host.Children.OfType<Panel>()) col.Children.Clear();
            host.Children.Clear();
            host.ColumnDefinitions.Clear();
            int n = wide ? 2 : 1;
            var cols = new StackPanel[n];
            for (int i = 0; i < n; i++)
            {
                host.ColumnDefinitions.Add(new ColumnDefinition());
                cols[i] = new StackPanel { Margin = new Thickness(i == 0 ? 0 : 7, 0, i == n - 1 ? 0 : 7, 0) };
                Grid.SetColumn(cols[i], i);
                host.Children.Add(cols[i]);
            }
            for (int i = 0; i < settingSections.Count; i++) cols[i % n].Children.Add(settingSections[i].Card);
        }

        Border Rule() { return new Border { Height = 1, Background = R("Line"), Margin = new Thickness(18, 0, 18, 0) }; }

        static TextBlock Tip(string text) { return new TextBlock { Text = text, TextWrapping = TextWrapping.Wrap, MaxWidth = 360 }; }

        /// <summary>quick: a big choice (its hint under the name, the choices on the right, the help in a tooltip).
        /// Otherwise a section row (name and help, its choices under them, or a switch or dropdown on the right).
        /// off: the frame-rate pick's Frame generation setting, which becomes its first choice, "Off".</summary>
        SettingRow BuildRow(SettingSpec spec, SettingSpec off, bool quick)
        {
            var row = new SettingRow { Spec = spec, Off = off };
            var g = new Grid { Margin = quick ? new Thickness(18, 16, 18, 16) : new Thickness(18, 14, 18, 14) };
            g.ColumnDefinitions.Add(new ColumnDefinition());
            g.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
            for (int i = 0; i < 5; i++) g.RowDefinitions.Add(new RowDefinition { Height = GridLength.Auto });

            string label = off != null ? "Frame rate" : spec.Label;
            string help = off != null ? off.Help + "\n\n" + spec.Help : spec.Help;
            var title = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Center, Margin = new Thickness(0, 0, 16, 0) };
            title.Children.Add(new TextBlock { Text = label, FontSize = quick ? 14.5 : 14, FontWeight = FontWeights.SemiBold, VerticalAlignment = VerticalAlignment.Center });
            if (quick)
            {
                var info = new TextBlock
                {
                    Style = S("Icon"), Text = "\uE946", FontSize = 12, Foreground = R("Muted"), Margin = new Thickness(8, 1, 0, 0), Background = Brushes.Transparent,
                    ToolTip = Tip(help + ImpactText(off ?? spec) + (string.IsNullOrEmpty(spec.Later) ? "" : "\n\n" + spec.Later)),
                };
                ToolTipService.SetInitialShowDelay(info, 150);
                ToolTipService.SetShowDuration(info, 60000);
                title.Children.Add(info);
            }
            if (spec.Experimental && off == null)
            {
                title.Children.Add(ExperimentalBadge());
                title.Children.Add(FeedbackPill(spec.Label, "setting-feedback-" + spec.Key));
            }
            if (!spec.Live)
            {
                bool nextPlay = spec.Key == "present" || spec.Key == "monitor";
                row.RestartTag = new Border
                {
                    CornerRadius = new CornerRadius(8), BorderBrush = R("Line2"), BorderThickness = new Thickness(1), Padding = new Thickness(7, 1, 7, 2),
                    Margin = new Thickness(10, 0, 0, 0), VerticalAlignment = VerticalAlignment.Center, Visibility = Visibility.Collapsed,
                    // Window presentation and Monitor are read when the video window opens; Restart now only rebuilds the AI engine
                    ToolTip = nextPlay ? "While playing, this one changes the next time you press Play."
                                       : "While playing, this one changes after a restart (Restart now, at the top).",
                    Child = new TextBlock { Text = nextPlay ? "next Play" : "after a restart", FontSize = 10.5, Foreground = R("Muted") },
                };
                title.Children.Add(row.RestartTag);
            }
            g.Children.Add(title);

            FrameworkElement control;
            bool segs = off != null || (!spec.OnOff && spec.Choices.Count <= 4 && spec.Key != "monitor");
            if (spec.OnOff && off == null)
            {
                var sw = new CheckBox { Style = S("Switch"), VerticalAlignment = VerticalAlignment.Center };
                AutomationProperties.SetName(sw, spec.Label);
                AutomationProperties.SetAutomationId(sw, "Set_" + spec.Key);
                sw.Checked += (s, e) => ChangeSetting(spec, true);
                sw.Unchecked += (s, e) => ChangeSetting(spec, false);
                row.Switch = sw;
                control = sw;
            }
            else if (segs)
            {
                var bar = new StackPanel { Orientation = Orientation.Horizontal };
                row.Segs = new List<RadioButton>();
                if (off != null) AddSeg(row, bar, OffTag, off.Choices.First(c => Same(c.Key, false)).Value);
                foreach (var c in spec.Choices) AddSeg(row, bar, c.Key, c.Value);
                control = new Border { Style = S("SegBar"), Child = bar, VerticalAlignment = VerticalAlignment.Center };
                AutomationProperties.SetName(control, label);
                AutomationProperties.SetAutomationId(control, "Set_" + spec.Key);
                row.Hint = new TextBlock { Style = S("Small"), Visibility = Visibility.Collapsed };
            }
            else
                control = BuildDrop(row, spec);
            if (quick || !segs)
            {
                Grid.SetColumn(control, 1);
                if (quick) Grid.SetRowSpan(control, 2);
                g.Children.Add(control);
            }
            else
            {
                control.HorizontalAlignment = HorizontalAlignment.Left;
                control.Margin = new Thickness(0, 10, 0, 0);
                Grid.SetRow(control, 2);
                Grid.SetColumnSpan(control, 2);
                g.Children.Add(control);
            }

            if (quick)
            {
                if (row.Hint != null)
                {
                    row.Hint.Margin = new Thickness(0, 4, 16, 0);
                    Grid.SetRow(row.Hint, 1);
                    g.Children.Add(row.Hint);
                }
            }
            else
            {
                var helpText = new StackPanel { Margin = new Thickness(0, 6, 0, 0) };
                helpText.Children.Add(new TextBlock { Style = S("Small"), Text = spec.Help });
                var impact = ImpactLines(spec);
                if (impact != null) helpText.Children.Add(impact);
                Grid.SetRow(helpText, 1);
                Grid.SetColumnSpan(helpText, 2);
                g.Children.Add(helpText);
                if (row.Hint != null && spec.Key == "fill")
                {
                    // Fill the screen: a drawing of the picked choice on this PC's screen, its hint beside it (FillPreview.cs)
                    var wrap = new Grid { Margin = new Thickness(0, 8, 0, 0) };
                    wrap.ColumnDefinitions.Add(new ColumnDefinition { Width = GridLength.Auto });
                    wrap.ColumnDefinitions.Add(new ColumnDefinition());
                    wrap.Children.Add(FillPreview());
                    row.Hint.Foreground = R("Text");
                    row.Hint.Margin = new Thickness(18, FillPad, 0, 0);
                    Grid.SetColumn(row.Hint, 1);
                    wrap.Children.Add(row.Hint);
                    Grid.SetRow(wrap, 3);
                    Grid.SetColumnSpan(wrap, 2);
                    g.Children.Add(wrap);
                }
                else if (row.Hint != null)
                {
                    row.Hint.Foreground = R("Text");
                    row.Hint.Margin = new Thickness(2, 8, 0, 0);
                    Grid.SetRow(row.Hint, 3);
                    Grid.SetColumnSpan(row.Hint, 2);
                    g.Children.Add(row.Hint);
                }
            }
            row.Note = new TextBlock { Style = S("Small"), Foreground = R("Warn"), Margin = new Thickness(0, 6, 0, 0), Visibility = Visibility.Collapsed };
            Grid.SetRow(row.Note, 4);
            Grid.SetColumnSpan(row.Note, 2);
            g.Children.Add(row.Note);
            if (spec.Key == "model" && spec.Choices.Any(c => Same(c.Key, "vfg")))
            {
                var box = VfgBox();
                var wrap = new StackPanel();  // under the note, in the same row
                g.Children.Remove(row.Note);
                wrap.Children.Add(row.Note);
                wrap.Children.Add(box);
                Grid.SetRow(wrap, 4);
                Grid.SetColumnSpan(wrap, 2);
                g.Children.Add(wrap);
            }
            row.Element = g;
            settingRows[spec.Key] = row;
            return row;
        }

        void AddSeg(SettingRow row, Panel bar, object value, string name)
        {
            var rb = new RadioButton { Style = S("Seg"), Content = name, Tag = value, GroupName = "set_" + row.Spec.Key };
            if (bar.Children.Count > 0) rb.Margin = new Thickness(2, 0, 0, 0);
            ToolTipService.SetShowOnDisabled(rb, true);  // why it's greyed out
            // Under its own segment: at the mouse, a tooltip stays where the mouse came in (moving left along the row,
            // the next segment's right edge), behind the mouse once it moves on.
            ToolTipService.SetPlacement(rb, PlacementMode.Bottom);
            string hint = SegHint(row, value);
            AutomationProperties.SetName(rb, name);
            AutomationProperties.SetHelpText(rb, hint);
            AutomationProperties.SetAutomationId(rb, "Set_" + row.Spec.Key + "_" + (value == OffTag ? "off" : Json.Serialize(value).Trim('"')));
            rb.Checked += (s, e) => PickSeg(row, value);
            if (row.Spec.Key == "fill")  // the drawing shows the choice under the mouse, then the picked one again
            {
                rb.MouseEnter += (s, e) => fillDraw?.Invoke(value as string);
                rb.MouseLeave += (s, e) => fillDraw?.Invoke(fillPicked);
            }
            row.Segs.Add(rb);
            bar.Children.Add(rb);
        }

        static string SegHint(SettingRow row, object value) { return value == OffTag ? row.Off.Hint(false) : row.Spec.Hint(value); }

        void PickSeg(SettingRow row, object value)
        {
            if (syncingSettings) return;
            if (row.Off == null) { ChangeSetting(row.Spec, value); return; }
            if (value == OffTag) { ChangeSetting(row.Off, false); return; }
            ChangeSetting(row.Spec, value);  // the rate first, so frame generation comes back on at the rate picked
            if (!Same(CurrentSetting(row.Off), true)) ChangeSetting(row.Off, true);
        }

        ComboBox BuildDrop(SettingRow row, SettingSpec spec)
        {
            var drop = new ComboBox { Style = S("Drop"), VerticalAlignment = VerticalAlignment.Center };
            AutomationProperties.SetName(drop, spec.Label);
            AutomationProperties.SetAutomationId(drop, "Set_" + spec.Key);
            row.Items = new List<ComboBoxItem>();
            for (int i = 0; i < spec.Choices.Count; i++)
            {
                var c = spec.Choices[i];
                var item = new ComboBoxItem { Style = S("DropItem"), Content = c.Value, Tag = c.Key };
                ToolTipService.SetShowOnDisabled(item, true);  // why it's greyed out
                AutomationProperties.SetName(item, c.Value);
                AutomationProperties.SetHelpText(item, spec.Hints[i]);  // DropItem shows it under the name
                AutomationProperties.SetAutomationId(item, "Set_" + spec.Key + "_" + Json.Serialize(c.Key).Trim('"'));
                row.Items.Add(item);
                drop.Items.Add(item);
            }
            if (!string.IsNullOrEmpty(spec.Later))  // choices that may come later: a note at the end, not a choice
            {
                var text = new StackPanel { Orientation = Orientation.Horizontal };
                text.Children.Add(new TextBlock { Style = S("Icon"), Text = "\uE946", FontSize = 12, Foreground = R("Muted"), VerticalAlignment = VerticalAlignment.Top, Margin = new Thickness(0, 2, 8, 0) });
                text.Children.Add(new TextBlock { Text = spec.Later, FontSize = 12, LineHeight = 17, FontStyle = FontStyles.Italic, TextWrapping = TextWrapping.Wrap, MaxWidth = 250 });
                var note = new ComboBoxItem
                {
                    Style = S("DropItem"), IsEnabled = false, Focusable = false,
                    Content = new Border { BorderBrush = R("Line2"), BorderThickness = new Thickness(0, 1, 0, 0), Padding = new Thickness(0, 9, 0, 0), Margin = new Thickness(0, -2, -24, 0), Child = text },
                };
                AutomationProperties.SetName(note, spec.Later);
                AutomationProperties.SetAutomationId(note, "Set_" + spec.Key + "_later");
                drop.Items.Add(note);
            }
            drop.SelectionChanged += (s, e) =>
            {
                var item = drop.SelectedItem as ComboBoxItem;
                if (item != null && item.Tag != null) ChangeSetting(spec, item.Tag);
            };
            // A closed dropdown under the pointer would otherwise take the wheel and change the setting while
            // you scroll the page.
            drop.PreviewMouseWheel += (s, e) =>
            {
                if (drop.IsDropDownOpen) return;
                e.Handled = true;
                var sv = E<ScrollViewer>("SetScroll");
                sv.ScrollToVerticalOffset(sv.VerticalOffset - e.Delta / 3.0);
            };
            drop.DropDownOpened += (s, e) => SnapDropDown(drop, spec.Key);
            drop.Loaded += (s, e) => RightAlignList(drop);
            row.Drop = drop;
            return drop;
        }

        /// <summary>A list with hints is wider than its box: it opens under the box, lined up with the box's right edge,
        /// so it stays over the page instead of running past the window.</summary>
        static void RightAlignList(ComboBox drop)
        {
            drop.ApplyTemplate();
            var popup = drop.Template.FindName("PART_Popup", drop) as Popup;
            if (popup == null || popup.Placement == PlacementMode.Custom) return;
            popup.Placement = PlacementMode.Custom;
            // (The popup adds its VerticalOffset itself.)
            popup.CustomPopupPlacementCallback = (size, target, offset) =>
                new[] { new CustomPopupPlacement(new Point(target.Width - size.Width, target.Height), PopupPrimaryAxis.Horizontal) };
        }

        bool SessionRunning
        {
            get
            {
                return playState == PlayState.Starting || playState == PlayState.Waiting || playState == PlayState.Found
                    || playState == PlayState.Building || playState == PlayState.Playing;
            }
        }

        /// <summary>Shows the current values, what can be changed right now, and which rows apply.</summary>
        void RefreshSettings()
        {
            bool live = liveState != null, locked = !live && SessionRunning;
            var values = live ? liveState["values"] as Dictionary<string, object> : null;
            var allowed = live ? liveState["choices"] as Dictionary<string, object> : null;
            var disabled = live ? liveState["disabled"] as Dictionary<string, object> : null;
            var restart = live ? (liveState["restart"] as object[] ?? new object[0]).Cast<string>().ToArray() : new string[0];
            var file = live ? null : ReadSettingsFile();
            var byKey = (settingSpecs ?? new List<SettingSpec>()).ToDictionary(s => s.Key);
            Func<SettingSpec, object> value = spec =>
            {
                if (values != null && values.ContainsKey(spec.Key)) return values[spec.Key];
                if (file != null && file.ContainsKey(spec.Key) && spec.Choices.Any(c => Same(c.Key, file[spec.Key]))) return file[spec.Key];
                return spec.Default;
            };
            Func<string, object> valueOf = key => byKey.ContainsKey(key) ? value(byKey[key]) : null;
            Func<string, string> why = key => disabled != null && disabled.ContainsKey(key) ? disabled[key] as string : null;

            syncingSettings = true;
            try
            {
                foreach (var row in settingRows.Values)
                {
                    var spec = row.Spec;
                    object v = value(spec);
                    // Nothing playing: greyed out if it isn't used with the source picked on the Play page (playing,
                    // the session says, in `disabled`). E.g. Capture card resolution with Remote Play.
                    string off = why(spec.Key) ?? (row.Off != null ? why(row.Off.Key) : null)
                                 ?? (live ? null : spec.UnusedWith(source.Kind));
                    bool on = !locked && off == null;
                    var ok = allowed != null && allowed.ContainsKey(spec.Key) ? allowed[spec.Key] as object[] : null;
                    Show(row.Element, spec.AppliesTo(valueOf));
                    if (row.Switch != null)
                    {
                        row.Switch.IsChecked = Same(v, true);
                        row.Switch.IsEnabled = on;
                    }
                    else if (row.Segs != null)
                    {
                        bool running = row.Off == null || Same(value(row.Off), true);  // frame generation on
                        foreach (var rb in row.Segs)
                        {
                            bool isOff = rb.Tag == OffTag;
                            bool can = isOff || ok == null || ok.Any(a => Same(a, rb.Tag));
                            string tip = can ? null : "Not available on this PC right now";
                            // Nothing playing: NVIDIA VFG only once it's set up (playing, the session says).
                            string vfgWhy = spec.Key == "model" && Same(rb.Tag, "vfg") && !live ? VfgWhyNot : null;
                            if (vfgWhy != null) { can = false; tip = vfgWhy; }
                            rb.IsEnabled = on && can;
                            string hint = SegHint(row, rb.Tag);
                            rb.ToolTip = tip ?? (hint.Length > 0 ? hint : null);
                            rb.IsChecked = isOff ? !running : running && Same(rb.Tag, v);
                        }
                        string picked = running ? spec.Hint(v) : row.Off.Hint(false);
                        row.Hint.Text = picked;
                        Show(row.Hint, picked.Length > 0);
                        if (spec.Key == "fill")
                        {
                            fillPicked = v as string ?? "fit";
                            fillDraw?.Invoke(fillPicked);
                        }
                    }
                    else
                    {
                        foreach (var item in row.Items)
                        {
                            item.IsEnabled = ok == null || ok.Any(a => Same(a, item.Tag));
                            item.ToolTip = item.IsEnabled ? null : "Not available on this PC right now";
                            string vfgWhy = spec.Key == "model" && Same(item.Tag, "vfg") && !live ? VfgWhyNot : null;
                            if (vfgWhy != null) { item.IsEnabled = false; item.ToolTip = vfgWhy; }
                        }
                        row.Drop.SelectedItem = row.Items.FirstOrDefault(item => Same(item.Tag, v));
                        row.Drop.IsEnabled = on;
                        if (!on) row.Drop.IsDropDownOpen = false;
                    }
                    row.Note.Text = off ?? "";
                    Show(row.Note, off != null);
                    if (row.RestartTag != null) Show(row.RestartTag, live);
                }
            }
            finally { syncingSettings = false; }

            FixRules(quickRows);
            foreach (var sec in settingSections)
            {
                FixRules(sec.Rows);
                var shown = sec.Rows.Where(r => r.Element.Visibility == Visibility.Visible).ToList();
                Show(sec.Card, shown.Count > 0);
                var changed = shown.Where(r => !Same(value(r.Spec), r.Spec.Default))
                    .Select(r => r.Spec.OnOff ? r.Spec.Label + (Same(value(r.Spec), true) ? " on" : " off") : r.Spec.Label + ": " + ChoiceName(r.Spec, value(r.Spec)))
                    .ToList();
                sec.Summary.Text = changed.Count > 0 ? string.Join("  ·  ", changed)
                    : shown.Count == 1 ? "1 setting, at its default" : shown.Count + " settings, all at their defaults";
                sec.Summary.Foreground = R(changed.Count > 0 ? "Text2" : "Muted");
                sec.Summary.ToolTip = changed.Count > 1 ? string.Join("\n", changed) : null;
            }

            T("SetStatus").Text =
                settingSpecs == null ? (settingsError != null ? "Couldn't load the settings: " + Shorten(settingsError, 160) + ". Repair SmoovPlay, on the Help page, puts back anything missing."
                                        : paths.RuntimeReady ? "Loading..." : "Finish setting up SmoovPlay first.")
                : live && playState == PlayState.Building ? "Restarting the AI model. Other changes still reach the video."
                : live ? "Changes reach the video straight away and are saved for next time. M in the video window opens the same settings."
                : locked ? "SmoovPlay is starting. You can change settings again once the video is on screen."
                : "Saved for every session. While you play, changes reach the video straight away.";
            Show(E("SetLive"), live);
            bool showRestart = live && restart.Length > 0 && !restartSent;
            if (showRestart)
                T("SetRestartText").Text = "Restart to apply: " + string.Join(", ", restart)
                    + ". The video pauses for a few seconds, or a minute or two if TensorRT builds a new engine.";
            Show(E("SetRestart"), showRestart);
            UpdateDetect();
            RenderVfg();
            UpdateFillPreview(valueOf("monitor") as string);
        }

        /// <summary>The line above each row: only between rows that are showing.</summary>
        static void FixRules(IEnumerable<SettingRow> rows)
        {
            bool above = false;
            foreach (var r in rows)
            {
                bool shown = r.Element.Visibility == Visibility.Visible;
                if (r.Rule != null) Show(r.Rule, shown && above);
                above |= shown;
            }
        }

        void ChangeSetting(SettingSpec spec, object value)
        {
            if (syncingSettings) return;
            if (liveState != null && session != null && session.Running)
            {
                // The state line that comes back confirms it, or puts the real value back if it couldn't be used.
                if (!session.Send("set " + spec.Key + " " + Json.Serialize(value))) { Toast("SmoovPlay isn't running any more", true); RefreshSettings(); }
                return;
            }
            if (SessionRunning) { RefreshSettings(); return; }  // locked meanwhile
            try
            {
                WriteSetting(spec.Key, value);
                foreach (var kv in spec.Also) WriteSetting(kv.Key, kv.Value);  // e.g. Sharpening: not carried over
            }
            catch (Exception e) { Toast("Couldn't save the setting: " + e.Message, true); }
            if (spec.Key == "monitor") checks = null;
            RefreshSettings();
            BuildChips();
        }

        void RestartFromSettings()
        {
            if (session == null || !session.Send("restart")) return;
            restartSent = true;  // until the next state line, which comes once the video is back
            RefreshSettings();
        }

        /// <summary>A "[settings-state]" line from the session (on the UI thread).</summary>
        void OnSettingsState(Session s, string json)
        {
            if (session != s) return;
            try { liveState = Json.DeserializeObject(json) as Dictionary<string, object>; }
            catch (Exception e) { Program.Trace("settings state: " + e.Message); return; }
            restartSent = false;
            RefreshSettings();
            BuildChips();
            OnLooksState();
            if (page == "PageSettings") SnapLater("settings-state", 400);
        }

        /// <summary>SMOOVPLAY_SHOTS: a dropdown's list is a popup (a window of its own), so the usual snapshot
        /// leaves it out. This one draws it over the window where it opened.</summary>
        void SnapDropDown(ComboBox drop, string key)
        {
            if (string.IsNullOrEmpty(shots)) return;
            var timer = new DispatcherTimer { Interval = TimeSpan.FromMilliseconds(400) };
            timer.Tick += (s, e) =>
            {
                timer.Stop();
                try
                {
                    var popup = drop.Template.FindName("PART_Popup", drop) as Popup;
                    var list = popup != null ? popup.Child as FrameworkElement : null;
                    if (list == null || !drop.IsDropDownOpen) return;
                    double scale = 1.5;
                    var bmp = new RenderTargetBitmap((int)(Window.ActualWidth * scale), (int)(Window.ActualHeight * scale), 96 * scale, 96 * scale, PixelFormats.Pbgra32);
                    bmp.Render(Window);
                    // The list on its own, 1:1 (room for its shadow), then drawn where it opened.
                    var lb = new RenderTargetBitmap((int)Math.Ceiling((list.ActualWidth + 16) * scale), (int)Math.Ceiling((list.ActualHeight + 16) * scale),
                                                    96 * scale, 96 * scale, PixelFormats.Pbgra32);
                    lb.Render(list);
                    var at = Window.PointFromScreen(list.PointToScreen(new Point(0, 0)));  // where it opened (it may be right-aligned)
                    var dv = new DrawingVisual();
                    using (var dc = dv.RenderOpen())
                        dc.DrawImage(lb, new Rect(at.X, at.Y, lb.PixelWidth / scale, lb.PixelHeight / scale));
                    bmp.Render(dv);
                    Directory.CreateDirectory(shots);
                    var enc = new PngBitmapEncoder();
                    enc.Frames.Add(BitmapFrame.Create(bmp));
                    using (var f = File.Create(Path.Combine(shots, $"app-{++shotNo:00}-dropdown-{key}.png"))) enc.Save(f);
                }
                catch (Exception ex) { Program.Trace("dropdown snapshot failed: " + ex.Message); }
            };
            timer.Start();
        }

        // ---- the taskbar button while playing: a coloured dot, and the state in its tooltip
        static ImageSource Dot(string hex)
        {
            var g = new DrawingGroup();
            g.Children.Add(new GeometryDrawing(Solid("#0E1014"), null, new EllipseGeometry(new Point(8, 8), 8, 8)));
            g.Children.Add(new GeometryDrawing(Solid(hex), null, new EllipseGeometry(new Point(8, 8), 5.5, 5.5)));
            var img = new DrawingImage(g);
            img.Freeze();
            return img;
        }

        void UpdateTaskbar(PlayState st, string title)
        {
            var tb = Window.TaskbarItemInfo;
            if (tb == null || installer != null) return;  // setup uses it for its progress
            string hex = st == PlayState.Playing ? "#3DDC8E" : st == PlayState.Error ? "#FF6464" : SessionRunning ? "#FFB02E" : null;
            tb.Overlay = hex != null ? Dot(hex) : null;
            tb.Description = hex != null ? "SmoovPlay: " + title : "SmoovPlay";
        }
    }
}
