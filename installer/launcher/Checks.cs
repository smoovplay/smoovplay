// The PC check that opens setup (and the launcher's "PC check" page), plus the fixes it can offer: installing the
// Microsoft Visual C++ runtime, pointing at the NVIDIA driver download, Windows display settings and long paths.

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.IO;
using System.Linq;
using System.Management;
using System.Text;
using System.Net;
using System.Net.Http;
using System.Net.NetworkInformation;
using System.Runtime.InteropServices;
using System.Text.RegularExpressions;
using System.Threading.Tasks;
using Microsoft.Win32;

namespace SmoovPlay
{
    enum Level { Checking, Ok, Info, Warn, Fail }

    /// <summary>One line of the PC check. Fail blocks setup (unless the user insists); Warn and Info don't.</summary>
    class Check
    {
        public string Id, Title, Value = "", Detail = "", Glyph;
        public Level Level = Level.Checking;
        public string FixLabel;   // a button on the card, if there's something to do about it
        public string FixId;      // what the button does (see Ui.RunFix)
        public string Tip;        // longer help, shown as the card's tooltip

        public Check(string id, string title, string glyph) { Id = id; Title = title; Glyph = glyph; }
        public Check Set(Level l, string value, string detail) { Level = l; Value = value ?? ""; Detail = detail ?? ""; return this; }
        public Check WithFix(string label, string id) { FixLabel = label; FixId = id; return this; }
    }

    static class SystemCheck
    {
        public static string GpuName = "unknown", Driver = "unknown";
        /// <summary>An AMD or Intel card, if there's no NVIDIA one: named in the check, which runs it through DirectML (Experimental).</summary>
        public static string OtherGpu;
        public static double VramGB;
        public static int DriverMajor;
        public const double NeedGB = 12;
        public const int MinDriver = 528, GoodDriver = 570;
        public const string VcRedistUrl = "https://aka.ms/vs/17/release/vc_redist.x64.exe";
        public const string DriverUrl = "https://www.nvidia.com/en-us/drivers/";
        /// <summary>The system proxy for the download servers, if Windows has one set (passed on to pip).</summary>
        public static string Proxy;

        /// <summary>The checks in display order, all still "checking". The last one depends on the video source
        /// (a SourceDef.Kind): the home network for PS5 Remote Play, the card for a capture card.</summary>
        public static List<Check> Plan(string source = "chiaki")
        {
            var list = new List<Check>
            {
                new Check("gpu", "Graphics card", "\uF211"),
                new Check("driver", "NVIDIA driver", "\uE90F"),
                new Check("windows", "This PC", "\uE977"),
                new Check("vc", "Visual C++ runtime", "\uE943"),
                new Check("disk", "Disk space", "\uEDA2"),
                new Check("net", "Download servers", "\uE774"),
                new Check("display", "Display", "\uE7F4"),
            };
            if (source == "chiaki") list.Add(new Check("lan", "Home network", "\uE839"));
            else if (source == "capture") list.Add(new Check("capture", "Capture card", "\uE8B2"));
            return list;
        }

        /// <summary>Runs one check (slow ones, like the download servers, take a few seconds).</summary>
        public static void Run(Check c, string installDir, bool freshInstall)
        {
            try
            {
                switch (c.Id)
                {
                    case "windows": Windows(c); if (c.Level != Level.Fail) Ram(c); break;
                    case "gpu": EnsureGpu(); Gpu(c); break;
                    case "driver": EnsureGpu(); DriverCheck(c); break;
                    case "vc": VcRuntime(c); break;
                    case "disk": Disk(c, installDir, freshInstall); break;
                    case "net": Network(c, freshInstall); break;
                    case "display": Display(c); break;
                    case "lan": Lan(c); break;
                    case "capture": Capture(c); break;
                }
            }
            catch (Exception e) { c.Set(Level.Warn, "Couldn't check", e.Message); }
            if (Faked(c.Id)) Fake(c);
        }

        // ---- testing: SMOOVPLAY_FAKE_CHECKS=vc,driver,display,lan,net,longpaths shows those as problems, to try the fixes'
        // looks on a PC that passes everything. (The fixes themselves still act for real.)
        static readonly string[] fakes = (Environment.GetEnvironmentVariable("SMOOVPLAY_FAKE_CHECKS") ?? "").Split(',');
        static bool Faked(string id) { return fakes.Contains(id); }

        static void Fake(Check c)
        {
            c.FixLabel = c.FixId = null;
            switch (c.Id)
            {
                case "vc": c.Set(Level.Fail, "Not installed", "PyTorch needs it. One click installs it from Microsoft.").WithFix("Install it for me", "vc"); break;
                case "driver": c.Set(Level.Warn, "551.86", $"Works, but {GoodDriver} or newer is recommended (and needed for RTX 50 cards).").WithFix("Get the driver", "driver"); break;
                case "display": c.Set(Level.Warn, "2560×1440 at 60 Hz", "It can run at 144 Hz, but Windows has it at 60 Hz. Raise it for smoother play.").WithFix("Open display settings", "display"); break;
                case "lan": c.Set(Level.Warn, "Wi-Fi, 2.4 GHz", "2.4 GHz often stutters. Use your router's 5 GHz or 6 GHz network, or a cable."); break;
                case "net": c.Set(Level.Fail, "No connection", "Setup needs the internet. Check the connection, then check again."); break;
            }
        }

        /// <summary>Everything, one after another (diagnostics, unattended setup).</summary>
        public static List<Check> RunAll(string installDir, bool freshInstall, string source = "chiaki")
        {
            var all = Plan(source);
            foreach (var c in all) Run(c, installDir, freshInstall);
            return all;
        }

        // ---------------------------------------------------------------- the checks
        static void Windows(Check c)
        {
            string build = Registry.GetValue(@"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Windows NT\CurrentVersion", "CurrentBuild", "0") as string ?? "0";
            int b; int.TryParse(build, out b);
            if (!Environment.Is64BitOperatingSystem || b < 10240)
                c.Set(Level.Fail, "Not supported", "SmoovPlay needs 64-bit Windows 10 or 11.");
            else
                c.Set(Level.Ok, b >= 22000 ? "Windows 11" : "Windows 10", "64-bit, build " + b + ".");
        }

        static DateTime gpuQueried;
        static readonly object gpuLock = new object();

        /// <summary>QueryGpu, unless it ran in the last few seconds (the GPU and driver cards share one query).</summary>
        static void EnsureGpu()
        {
            lock (gpuLock)
                if ((DateTime.UtcNow - gpuQueried).TotalSeconds > 5) QueryGpu();
        }

        /// <summary>Finds the NVIDIA card: name and driver from WMI, memory from nvidia-smi (WMI caps it at 4 GB).</summary>
        public static void QueryGpu()
        {
            lock (gpuLock) { QueryGpuLocked(); gpuQueried = DateTime.UtcNow; }
        }

        static void QueryGpuLocked()
        {
            string name = "unknown", driver = "unknown", other = null;
            int major = 0, otherRank = 0;
            double vram = 0;
            try
            {
                using (var s = new ManagementObjectSearcher("SELECT Name, DriverVersion FROM Win32_VideoController"))
                    foreach (ManagementObject mo in s.Get())
                    {
                        string n = (mo["Name"] as string) ?? "";
                        if (n.IndexOf("NVIDIA", StringComparison.OrdinalIgnoreCase) < 0)
                        {
                            // A Radeon or Arc card outranks integrated Intel graphics; virtual adapters don't count.
                            int rank = Regex.IsMatch(n, @"Radeon|\bArc\b", RegexOptions.IgnoreCase) ? 2
                                : n.IndexOf("Intel", StringComparison.OrdinalIgnoreCase) >= 0 ? 1 : 0;
                            if (rank > otherRank) { other = n; otherRank = rank; }
                            continue;
                        }
                        name = n;
                        string digits = ((mo["DriverVersion"] as string) ?? "").Replace(".", "");  // 32.0.15.7283 = 572.83
                        if (digits.Length >= 5)
                        {
                            string last5 = digits.Substring(digits.Length - 5);
                            driver = last5.Substring(0, 3) + "." + last5.Substring(3);
                            int.TryParse(last5.Substring(0, 3), out major);
                        }
                    }
            }
            catch (Exception) { }
            if (name != "unknown")
                try
                {
                    string smi = Path.Combine(Environment.GetFolderPath(Environment.SpecialFolder.System), "nvidia-smi.exe");
                    if (!File.Exists(smi)) smi = @"C:\Program Files\NVIDIA Corporation\NVSMI\nvidia-smi.exe";
                    if (File.Exists(smi))
                    {
                        var psi = new ProcessStartInfo(smi, "--query-gpu=memory.total --format=csv,noheader,nounits")
                        { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true };
                        using (var pr = Process.Start(psi))
                        {
                            string first = pr.StandardOutput.ReadToEnd().Split('\n')[0].Trim();
                            pr.WaitForExit(5000);
                            double mib;
                            if (double.TryParse(first, System.Globalization.NumberStyles.Float, System.Globalization.CultureInfo.InvariantCulture, out mib))
                                vram = Math.Round(mib / 1024);
                        }
                    }
                }
                catch (Exception) { }
            GpuName = name; Driver = driver; DriverMajor = major; VramGB = vram; OtherGpu = other;
        }

        static void Gpu(Check c)
        {
            if (GpuName == "unknown" && OtherGpu != null)
            {
                // No NVIDIA card: RIFE runs through DirectML (framegen/dml.py), on any DirectX 12 card.
                string shortOther = Regex.Replace(OtherGpu, @"\((R|TM)\)|^AMD\s+", "").Trim();
                bool card = Regex.IsMatch(OtherGpu, @"Radeon\s+(RX|Pro)|\bArc\b", RegexOptions.IgnoreCase);
                c.Set(Level.Warn, shortOther, card
                    ? "Experimental: without an NVIDIA card, RIFE runs through DirectML. It's slower than on an NVIDIA card, and not tried with a game yet. Expect 60 fps; 90 and 120 may not keep up."
                    : "Experimental: built-in graphics run RIFE through DirectML, but are likely too slow for smooth frame generation. A Radeon RX, Arc or NVIDIA RTX card is better.");
                c.Tip = "NVIDIA TensorRT, RTX VSR and NVIDIA VFG need an NVIDIA card; everything else works the same. Send feedback says how it went (Radeon RX 6000, 7000 and 9000 especially wanted).";
                c.WithFix("Send feedback", "othergpu");
                return;
            }
            if (GpuName == "unknown")
            {
                c.Set(Level.Fail, "No graphics card found", "SmoovPlay needs a DirectX 12 graphics card: NVIDIA RTX, or AMD Radeon or Intel Arc (Experimental).");
                c.Tip = "If this PC has an NVIDIA card, its driver may not be installed: get it from nvidia.com, then check again.";
                c.WithFix("Get the driver", "driver");
                return;
            }
            string shortName = Regex.Replace(GpuName, @"^NVIDIA\s+(GeForce\s+)?", "");
            string value = shortName + (VramGB > 0 ? $"  ·  {VramGB:0} GB" : "");
            var m = Regex.Match(GpuName, @"RTX\s*(\d)\d\d\d");
            int series = m.Success ? int.Parse(m.Groups[1].Value) * 10 : 0;
            bool laptop = GpuName.IndexOf("Laptop", StringComparison.OrdinalIgnoreCase) >= 0;
            if (laptop)
                c.Set(Level.Warn, value, "Laptop graphics aren't tested yet. Expect 60 fps; 90 and 120 may not keep up.");
            else if (series >= 30)
                c.Set(Level.Ok, value, $"RTX {series / 10}0 series: supported.");
            else if (GpuName.Contains("RTX"))
                c.Set(Level.Warn, value, "RTX 20 series: older than the cards it's tested on. 60 fps may be the most it manages.");
            else
                c.Set(Level.Fail, value, "Not an RTX card: too slow for real-time frame generation.");
            if (VramGB > 0 && VramGB < 6 && c.Level == Level.Ok)
                c.Set(Level.Warn, value, "Under 6 GB of graphics memory: 1080p streams may run short of memory.");
        }

        static void DriverCheck(Check c)
        {
            c.Tip = "Update with the NVIDIA app (or GeForce Experience), or download the Game Ready driver for your card from nvidia.com. " +
                    "A clean install isn't needed. Press Check again afterwards.";
            if (GpuName == "unknown" && OtherGpu != null)
            {
                c.Tip = "AMD Software (Adrenalin) or Intel's driver app keeps the driver up to date.";
                c.Set(Level.Info, "Not needed", "This check is for NVIDIA cards. Keep your graphics driver up to date: DirectML runs through it.");
                return;
            }
            if (GpuName == "unknown") { c.Set(Level.Info, "No NVIDIA card", "Checked once a card is found."); return; }
            if (DriverMajor == 0) { c.Set(Level.Warn, "Unknown version", "Couldn't read the driver version."); return; }
            if (DriverMajor < MinDriver)
                c.Set(Level.Fail, Driver, $"Too old for CUDA 12. Version {MinDriver} or newer is needed, and {GoodDriver} or newer is best.").WithFix("Get the driver", "driver");
            else if (DriverMajor < GoodDriver)
                c.Set(Level.Warn, Driver, $"Works, but {GoodDriver} or newer is recommended (and needed for RTX 50 cards).").WithFix("Get the driver", "driver");
            else
                c.Set(Level.Ok, Driver, "Up to date for SmoovPlay.");
        }

        /// <summary>PyTorch needs msvcp140.dll from the Visual C++ 2015-2022 runtime, and a recent one: binaries built with
        /// newer compilers crash with runtimes older than 14.40.</summary>
        static void VcRuntime(Check c)
        {
            c.Tip = "The Microsoft Visual C++ 2015-2022 Redistributable (x64). Many games install it; SmoovPlay can download it " +
                    "from Microsoft and install it for you (Windows asks for permission).";
            string sys32 = Environment.GetFolderPath(Environment.SpecialFolder.System);
            string msvcp = Path.Combine(sys32, "msvcp140.dll");
            if (!File.Exists(msvcp) || !File.Exists(Path.Combine(sys32, "vcruntime140_1.dll")))
            {
                c.Set(Level.Fail, "Not installed", "PyTorch needs it. One click installs it from Microsoft.").WithFix("Install it for me", "vc");
                return;
            }
            var v = FileVersionInfo.GetVersionInfo(msvcp);
            string ver = $"{v.FileMajorPart}.{v.FileMinorPart}";
            if (v.FileMajorPart < 14 || (v.FileMajorPart == 14 && v.FileMinorPart < 40))
                c.Set(Level.Fail, ver + " (too old)", "PyTorch needs 14.40 or newer. One click updates it.").WithFix("Update it for me", "vc");
            else
                c.Set(Level.Ok, "Installed", "Version " + ver + ".");
        }

        [StructLayout(LayoutKind.Sequential)]
        class MemStatus
        {
            public uint Length = (uint)Marshal.SizeOf(typeof(MemStatus));
            public uint Load;
            public ulong TotalPhys, AvailPhys, TotalPageFile, AvailPageFile, TotalVirtual, AvailVirtual, AvailExtendedVirtual;
        }
        [DllImport("kernel32.dll")] static extern bool GlobalMemoryStatusEx([In, Out] MemStatus m);

        /// <summary>Adds the memory to the Windows card ("Windows 11 · 32 GB memory").</summary>
        static void Ram(Check c)
        {
            var m = new MemStatus();
            if (!GlobalMemoryStatusEx(m)) return;
            double gb = Math.Round(m.TotalPhys / 1073741824.0);
            string value = c.Value + $"  ·  {gb:0} GB memory";
            if (gb >= 12) c.Set(Level.Ok, value, c.Detail);
            else if (gb >= 8) c.Set(Level.Info, value, c.Detail + " Close big apps if the stream stutters.");
            else c.Set(Level.Warn, value, "Under 8 GB of memory: SmoovPlay may run short while you play.");
        }

        public static double FreeGB(string dir)
        {
            var root = Path.GetPathRoot(Path.GetFullPath(dir));
            return new DriveInfo(root).AvailableFreeSpace / 1e9;
        }

        static void Disk(Check c, string installDir, bool freshInstall)
        {
            var root = Path.GetPathRoot(Path.GetFullPath(installDir)).TrimEnd('\\');
            double gb = FreeGB(installDir);
            if (gb >= NeedGB)
                c.Set(Level.Ok, $"{gb:0} GB free on {root}", $"SmoovPlay needs about {NeedGB:0} GB.");
            else if (freshInstall)
                c.Set(Level.Warn, $"{gb:0} GB free on {root}", $"About {NeedGB:0} GB is needed: pick another drive in the next step, or free some space.");
            else
                c.Set(Level.Warn, $"{gb:0} GB free on {root}", $"Repairs and updates need up to {NeedGB:0} GB free.");
        }

        public static readonly string[] Servers =
            { "https://www.python.org", "https://bootstrap.pypa.io", "https://pypi.org", "https://files.pythonhosted.org",
              "https://download.pytorch.org", "https://pypi.nvidia.com" };

        /// <summary>Can this PC reach every server setup downloads from? Uses Windows' proxy settings, like setup does.</summary>
        static void Network(Check c, bool freshInstall)
        {
            c.Tip = "Setup downloads about 5 GB from python.org, pypi.org, download.pytorch.org and pypi.nvidia.com. " +
                    "If some are blocked, try another network, or pause a VPN or firewall that filters downloads.";
            Proxy = null;
            try
            {
                var sys = WebRequest.GetSystemWebProxy();
                var probe = new Uri("https://pypi.org/");
                var via = sys.GetProxy(probe);
                if (via != null && via != probe) Proxy = via.GetLeftPart(UriPartial.Authority);
            }
            catch (Exception) { }
            var failed = new List<string>();
            using (var http = new HttpClient { Timeout = TimeSpan.FromSeconds(10) })
            {
                http.DefaultRequestHeaders.UserAgent.ParseAdd("SmoovPlay-installer/" + BuildInfo.Version);
                var tasks = Servers.Select(s => Task.Run(() =>
                {
                    try
                    {
                        using (var req = new HttpRequestMessage(HttpMethod.Head, s + "/"))
                        using (var resp = http.SendAsync(req).Result)
                            return (int)resp.StatusCode < 500 ? null : s;
                    }
                    catch (Exception) { return s; }
                })).ToArray();
                Task.WaitAll(tasks);
                failed.AddRange(tasks.Select(t => t.Result).Where(s => s != null));
            }
            string via2 = Proxy != null ? " (through your proxy)" : "";
            if (failed.Count == 0)
                c.Set(Level.Ok, "All reachable" + via2, freshInstall ? "Setup downloads about 5 GB from them." : "Repairs can download what they need.");
            else
            {
                string names = string.Join(", ", failed.Select(s => new Uri(s).Host));
                var lvl = failed.Count == Servers.Length ? Level.Fail : Level.Warn;
                if (!freshInstall) lvl = Level.Info;
                c.Set(lvl, failed.Count == Servers.Length ? "No connection" + via2 : $"{failed.Count} not reachable",
                      (failed.Count == Servers.Length ? "Setup needs the internet. " : "Not reachable: " + names + ". ") + "Check the connection, then check again.");
            }
        }

        // ---- display
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct DISPLAY_DEVICE
        {
            public int cb;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string DeviceName;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceString;
            public int StateFlags;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceID;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 128)] public string DeviceKey;
        }
        [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
        struct DEVMODE
        {
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmDeviceName;
            public short dmSpecVersion, dmDriverVersion, dmSize, dmDriverExtra;
            public int dmFields, dmPositionX, dmPositionY, dmDisplayOrientation, dmDisplayFixedOutput;
            public short dmColor, dmDuplex, dmYResolution, dmTTOption, dmCollate;
            [MarshalAs(UnmanagedType.ByValTStr, SizeConst = 32)] public string dmFormName;
            public short dmLogPixels;
            public int dmBitsPerPel, dmPelsWidth, dmPelsHeight, dmDisplayFlags, dmDisplayFrequency;
            public int dmICMMethod, dmICMIntent, dmMediaType, dmDitherType, dmReserved1, dmReserved2, dmPanningWidth, dmPanningHeight;
        }
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern bool EnumDisplayDevices(string dev, int i, ref DISPLAY_DEVICE dd, int flags);
        [DllImport("user32.dll", CharSet = CharSet.Unicode)] static extern bool EnumDisplaySettings(string dev, int mode, ref DEVMODE dm);

        static string MonitorId(string device)
        {
            var mon = new DISPLAY_DEVICE { cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)) };
            if (EnumDisplayDevices(device, 0, ref mon, 1) && !string.IsNullOrEmpty(mon.DeviceID))  // EDD_GET_DEVICE_INTERFACE_NAME
            {
                var parts = mon.DeviceID.Split(new[] { '#' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts.Length >= 3)
                {
                    string pnp = Regex.Replace(parts[1], @"[^A-Za-z0-9_-]", "");
                    var tokens = parts[2].Split(new[] { '&' }, StringSplitOptions.RemoveEmptyEntries);
                    string tail = tokens.Length > 0 ? Regex.Replace(tokens[tokens.Length - 1], @"[^A-Za-z0-9_-]", "") : "";
                    if (pnp.Length > 0 && tail.Length > 0) return pnp + "-" + tail;
                }
            }
            string shortDev = device.StartsWith(@"\\.\") ? device.Substring(4) : device;
            string clean = Regex.Replace(shortDev, @"[^A-Za-z0-9_-]", "");
            return clean.Length > 0 ? clean : "DISPLAY";
        }

        static bool MatchesMonitor(string device, string want)
        {
            if (string.IsNullOrEmpty(want)) return false;
            string shortDev = Regex.Replace(device.StartsWith(@"\\.\") ? device.Substring(4) : device, @"[^A-Za-z0-9_-]", "");
            string id = MonitorId(device);
            return string.Equals(id, want, StringComparison.OrdinalIgnoreCase)
                || string.Equals(id + "-" + shortDev, want, StringComparison.OrdinalIgnoreCase)
                || string.Equals(device, want, StringComparison.OrdinalIgnoreCase)
                || string.Equals(shortDev, want, StringComparison.OrdinalIgnoreCase);
        }

        /// <summary>The shape of the chosen display (width over height, as set now), or WidestScreenRatio() if none
        /// matches: the Settings page draws Fill the screen on it.</summary>
        public static double ScreenRatio(string monitor)
        {
            bool wantPrimary = string.IsNullOrEmpty(monitor) || string.Equals(monitor, "primary", StringComparison.OrdinalIgnoreCase);
            var dd = new DISPLAY_DEVICE { cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)) };
            for (int i = 0; EnumDisplayDevices(null, i, ref dd, 0); i++, dd.cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)))
            {
                if ((dd.StateFlags & 1) == 0) continue;  // DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
                bool match = wantPrimary ? (dd.StateFlags & 4) != 0 : MatchesMonitor(dd.DeviceName, monitor);
                if (!match) continue;
                var dm = new DEVMODE { dmSize = (short)Marshal.SizeOf(typeof(DEVMODE)) };
                if (EnumDisplaySettings(dd.DeviceName, -1, ref dm) && dm.dmPelsHeight > 0)
                    return (double)dm.dmPelsWidth / dm.dmPelsHeight;
            }
            return WidestScreenRatio();
        }

        /// <summary>The widest shape among the connected displays (width over height, as set now), or 16:9 when none
        /// can be read: the Settings page draws Fill the screen on it.</summary>
        public static double WidestScreenRatio()
        {
            double best = 0;
            var dd = new DISPLAY_DEVICE { cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)) };
            for (int i = 0; EnumDisplayDevices(null, i, ref dd, 0); i++, dd.cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)))
            {
                if ((dd.StateFlags & 1) == 0) continue;  // DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
                var dm = new DEVMODE { dmSize = (short)Marshal.SizeOf(typeof(DEVMODE)) };
                if (!EnumDisplaySettings(dd.DeviceName, -1, ref dm) || dm.dmPelsHeight <= 0) continue;
                best = Math.Max(best, (double)dm.dmPelsWidth / dm.dmPelsHeight);
            }
            return best > 0 ? best : 16.0 / 9.0;
        }

        static void DisplayMode(string device, out int now, out int w, out int h, out int possible)
        {
            now = w = h = possible = 0;
            var dm = new DEVMODE { dmSize = (short)Marshal.SizeOf(typeof(DEVMODE)) };
            if (!EnumDisplaySettings(device, -1, ref dm)) return;  // ENUM_CURRENT_SETTINGS
            now = dm.dmDisplayFrequency; w = dm.dmPelsWidth; h = dm.dmPelsHeight; possible = now;
            for (int m = 0; ; m++)
            {
                var mode = new DEVMODE { dmSize = (short)Marshal.SizeOf(typeof(DEVMODE)) };
                if (!EnumDisplaySettings(device, m, ref mode)) break;
                if (mode.dmPelsWidth == w && mode.dmPelsHeight == h) possible = Math.Max(possible, mode.dmDisplayFrequency);
            }
        }

        /// <summary>The chosen display's refresh rate (or the fastest connected display's), and the fastest it could be set to.</summary>
        static void Display(Check c)
        {
            string want = Prefs.Setting("monitor");
            bool pickSpecific = !string.IsNullOrEmpty(want) && !string.Equals(want, "primary", StringComparison.OrdinalIgnoreCase);
            var devices = new List<string>();
            string chosen = null;
            var dd = new DISPLAY_DEVICE { cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)) };
            for (int i = 0; EnumDisplayDevices(null, i, ref dd, 0); i++, dd.cb = Marshal.SizeOf(typeof(DISPLAY_DEVICE)))
            {
                if ((dd.StateFlags & 1) == 0) continue;  // DISPLAY_DEVICE_ATTACHED_TO_DESKTOP
                devices.Add(dd.DeviceName);
                if (pickSpecific && chosen == null && MatchesMonitor(dd.DeviceName, want))
                    chosen = dd.DeviceName;
            }
            int bestNow = 0, bestPossible = 0, wNow = 0, hNow = 0;
            if (chosen != null)
            {
                DisplayMode(chosen, out bestNow, out wNow, out hNow, out bestPossible);
            }
            else
            {
                foreach (var dev in devices)
                {
                    int now, w, h, possible;
                    DisplayMode(dev, out now, out w, out h, out possible);
                    if (now > bestNow) { bestNow = now; wNow = w; hNow = h; }
                    bestPossible = Math.Max(bestPossible, possible);
                }
            }
            if (bestNow == 0) { c.Set(Level.Info, "Unknown", ""); return; }
            string value = $"{wNow}×{hNow} at {bestNow} Hz";
            c.Tip = "SmoovPlay turns 30 fps games into 60, 90 or 120 fps. Each needs a display running at least that fast " +
                    "(or one with G-Sync/FreeSync). Pick the rate in SmoovPlay's menu (press M in the video window).";
            if (bestPossible > bestNow + 5)
                c.Set(Level.Warn, value, $"It can run at {bestPossible} Hz, but Windows has it at {bestNow} Hz. Raise it for smoother play.")
                 .WithFix("Open display settings", "display");
            else if (bestNow >= 115)
                c.Set(Level.Ok, value, "Fast enough for SmoovPlay's 120 fps mode.");
            else if (bestNow >= 85)
                c.Set(Level.Ok, value, "Fast enough for 90 fps. 120 fps needs a 120 Hz display.");
            else
                c.Set(Level.Info, value, "SmoovPlay will show 60 fps. A 120 Hz display makes it even smoother.");
        }

        /// <summary>Wired or Wi-Fi: Remote Play is much steadier over a cable.</summary>
        static void Lan(Check c)
        {
            c.Tip = "Remote Play is smoothest with the PC and the PS5 both wired to the router. On Wi-Fi, use 5 GHz or 6 GHz and stay close to the router.";
            var up = NetworkInterface.GetAllNetworkInterfaces()
                .Where(n => n.OperationalStatus == OperationalStatus.Up &&
                            n.NetworkInterfaceType != NetworkInterfaceType.Loopback && n.NetworkInterfaceType != NetworkInterfaceType.Tunnel)
                .Where(n => { try { return n.GetIPProperties().GatewayAddresses.Any(g => !g.Address.Equals(IPAddress.Any)); } catch (Exception) { return false; } })
                .ToList();
            if (up.Count == 0) { c.Set(Level.Warn, "Not connected", "Connect to the same network as your PS5."); return; }
            if (up.Any(n => n.NetworkInterfaceType == NetworkInterfaceType.Ethernet || n.NetworkInterfaceType == NetworkInterfaceType.GigabitEthernet))
            {
                c.Set(Level.Ok, "Wired (Ethernet)", "Best for Remote Play. A wired PS5 helps too.");
                return;
            }
            if (up.Any(n => n.NetworkInterfaceType == NetworkInterfaceType.Wireless80211))
            {
                string band = WifiBand();
                if (band == "6") c.Set(Level.Info, "Wi-Fi, 6 GHz", "Works well. A cable to the router is steadier still.");
                else if (band == "5") c.Set(Level.Info, "Wi-Fi, 5 GHz", "Works. A cable to the router is steadier still.");
                else if (band == "2.4") c.Set(Level.Warn, "Wi-Fi, 2.4 GHz", "2.4 GHz often stutters. Use your router's 5 GHz or 6 GHz network, or a cable.");
                else c.Set(Level.Info, "Wi-Fi", "Works. Use 5 GHz or 6 GHz, or a cable to the router for the steadiest stream.");
                return;
            }
            c.Set(Level.Info, up[0].NetworkInterfaceType.ToString(), "Remote Play is steadiest on a wired connection.");
        }

        /// <summary>Video capture devices Windows knows about. Capture cards show up as cameras or imaging devices;
        /// which one is the card is picked on the Play page (CaptureDevices).</summary>
        static void Capture(Check c)
        {
            c.Tip = "Connect the console's HDMI output to the card's input and the card to this PC with USB 3 (or fit it in a PCIe slot). On a PS5, turn off HDCP while you use it: Settings → System → HDMI.";
            var names = new List<string>();
            using (var q = new ManagementObjectSearcher("SELECT Name FROM Win32_PnPEntity WHERE PNPClass = 'Camera' OR PNPClass = 'Image' OR PNPClass = 'MEDIA'"))
                foreach (ManagementObject o in q.Get())
                {
                    string n = o["Name"] as string;
                    if (!string.IsNullOrEmpty(n) && Regex.IsMatch(n, CaptureDevices.CardPattern, RegexOptions.IgnoreCase))
                        names.Add(n);
                }
            if (names.Count == 0) c.Set(Level.Warn, "None found", "Connect the card with USB 3 (blue port) or fit it in a PCIe slot, then check again.");
            else c.Set(Level.Ok, names[0], names.Count > 1 ? $"And {names.Count - 1} more." : "Found.");
        }

        /// <summary>For diagnostics: the capture hardware Windows has, whether or not its driver works, with its USB
        /// vendor and product (VID:PID; never the serial number). Only devices named like a card or made by a capture
        /// card maker: other cameras and sound devices can carry their owner's name.</summary>
        public static string CaptureHardware()
        {
            var sb = new StringBuilder();
            try
            {
                using (var q = new ManagementObjectSearcher("SELECT Name, PNPClass, PNPDeviceID, ConfigManagerErrorCode, Present FROM Win32_PnPEntity " +
                                                            "WHERE PNPClass = 'Camera' OR PNPClass = 'Image' OR PNPClass = 'MEDIA' OR PNPClass = 'USB' OR PNPClass = 'AudioEndpoint'"))
                    foreach (ManagementObject o in q.Get())
                    {
                        string name = o["Name"] as string ?? "", id = o["PNPDeviceID"] as string ?? "";
                        var vidpid = Regex.Match(id, @"VID_([0-9A-F]{4})&PID_([0-9A-F]{4})", RegexOptions.IgnoreCase);
                        bool maker = vidpid.Success && Regex.IsMatch(vidpid.Groups[1].Value, CaptureMakers, RegexOptions.IgnoreCase);
                        if (!maker && !Regex.IsMatch(name, CaptureDevices.CardPattern, RegexOptions.IgnoreCase)) continue;
                        uint code = o["ConfigManagerErrorCode"] is uint u ? u : 0;
                        bool present = !(o["Present"] is bool pr) || pr;
                        sb.AppendLine($"Capture hardware: {Diagnostics.NoOwner(name)} [{o["PNPClass"]}"
                                      + (vidpid.Success ? $", {vidpid.Groups[1].Value.ToUpper()}:{vidpid.Groups[2].Value.ToUpper()}" : "") + "]: "
                                      + (code == 0 ? "working" : $"problem code {code}") + (present ? "" : ", not connected"));
                    }
            }
            catch (Exception e) { sb.AppendLine("Capture hardware: couldn't ask Windows (" + e.Message + ")"); }
            return sb.Length > 0 ? sb.ToString() : "Capture hardware: none named like a card" + Environment.NewLine;
        }

        // USB vendor IDs of capture card makers: Elgato, AVerMedia, MacroSilicon (MS2109/MS2130 sticks: two IDs), Magewell, Razer
        const string CaptureMakers = "^(0FD9|07CA|534D|345F|2935|1532)$";

        /// <summary>"2.4", "5" or "6" for the band the Wi-Fi is connected on, or null if unknown. Windows 11 prints a
        /// "Band : 6 GHz" line (the label is translated, the value isn't); older Windows only has the channel. The channel
        /// alone can't tell 6 GHz apart (its channels are numbered 1 to 233 again), so it's only the fallback.</summary>
        static string WifiBand()
        {
            try
            {
                var psi = new ProcessStartInfo("netsh", "wlan show interfaces") { UseShellExecute = false, CreateNoWindow = true, RedirectStandardOutput = true };
                using (var pr = Process.Start(psi))
                {
                    string text = pr.StandardOutput.ReadToEnd();
                    pr.WaitForExit(3000);
                    var b = Regex.Match(text, @":\s*(2[.,]4|5|6)\s*GHz\s*$", RegexOptions.Multiline | RegexOptions.IgnoreCase);
                    if (b.Success) return b.Groups[1].Value.Replace(',', '.');
                    var m = Regex.Match(text, @"^\s*(Channel|Kanal|Canal|Canale)\s*:\s*(\d+)", RegexOptions.Multiline | RegexOptions.IgnoreCase);
                    if (!m.Success) return null;
                    int ch = int.Parse(m.Groups[2].Value);
                    return ch > 14 ? "5" : ch > 0 ? "2.4" : null;
                }
            }
            catch (Exception) { return null; }
        }

        // ---------------------------------------------------------------- the install folder
        public static bool LongPathsEnabled
        {
            get
            {
                if (Faked("longpaths")) return false;
                try { return Convert.ToInt32(Registry.GetValue(@"HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem", "LongPathsEnabled", 0)) == 1; }
                catch (Exception) { return false; }
            }
        }

        /// <summary>The deepest file the packages put in the install folder is about 155 characters below it (measured:
        /// runtime\python\Lib\site-packages\ plus 121), so Windows' old 260-character limit, which applies while long
        /// paths are off, leaves room for an install folder of about 80 with a little to spare.</summary>
        public const int MaxDirLength = 80;

        /// <summary>Why SmoovPlay can't write to this folder, or null if it can. Tried for real (a folder made in it,
        /// or in the nearest one that exists, and removed straight away), as a folder Windows protects only shows by
        /// refusing: a Program Files on another drive, say, which FolderProblem's name check doesn't know.</summary>
        public static string WriteProblem(string dir)
        {
            const string NoAdmin = "Windows doesn't let SmoovPlay write to that folder without admin rights. Choose one in your user folder, or a folder of your own on any drive.";
            if (Faked("readonlydir")) return NoAdmin;
            string probe;
            try { probe = Path.GetFullPath(dir.Trim()); }
            catch (Exception) { return null; }  // FolderProblem says why
            while (probe != null && !Directory.Exists(probe)) probe = Path.GetDirectoryName(probe);
            if (probe == null) return null;
            string test = Path.Combine(probe, ".smoovplay-write-test-" + Guid.NewGuid().ToString("N").Substring(0, 8));
            try
            {
                Directory.CreateDirectory(test);
                File.WriteAllText(Path.Combine(test, "test.txt"), "");
                return null;
            }
            catch (UnauthorizedAccessException) { return NoAdmin; }
            catch (IOException e) { return "SmoovPlay can't write to that folder (" + e.Message.Trim().TrimEnd('.') + "). Choose another one."; }
            catch (Exception) { return null; }  // couldn't tell: the install says if it can't
            finally { try { if (Directory.Exists(test)) Directory.Delete(test, true); } catch (Exception) { } }
        }

        /// <summary>Why this folder won't do, or null if it's fine.</summary>
        public static string FolderProblem(string dir, out bool tooLong)
        {
            tooLong = false;
            if (string.IsNullOrWhiteSpace(dir)) return "Choose a folder.";
            string full;
            try { full = Path.GetFullPath(dir.Trim()); }
            catch (Exception) { return "That isn't a valid folder path."; }
            if (!Path.IsPathRooted(full) || full.StartsWith(@"\\")) return "Choose a folder on a drive of this PC.";
            try
            {
                var drive = new DriveInfo(Path.GetPathRoot(full));
                if (!drive.IsReady) return "That drive isn't available.";
                if (drive.DriveType != DriveType.Fixed) return "Choose a drive inside this PC (not a removable or network drive).";
            }
            catch (Exception) { return "That drive isn't available."; }
            if (full.Length > MaxDirLength && !LongPathsEnabled) { tooLong = true; return $"That path is long ({full.Length} characters): Windows' 260-character limit could break the install."; }
            string pf = Environment.GetFolderPath(Environment.SpecialFolder.ProgramFiles);
            if (full.StartsWith(pf, StringComparison.OrdinalIgnoreCase) || full.StartsWith(Environment.GetFolderPath(Environment.SpecialFolder.Windows), StringComparison.OrdinalIgnoreCase))
                return "That folder needs admin rights. Choose one in your user folder or on another drive.";
            return null;
        }

        // ---------------------------------------------------------------- fixes
        /// <summary>Downloads the Visual C++ runtime from Microsoft and installs it (Windows asks for permission).
        /// Returns null on success, or what went wrong.</summary>
        public static string InstallVcRuntime(Action<string> status)
        {
            string dir = Path.Combine(Path.GetTempPath(), "SmoovPlay-vc");
            Directory.CreateDirectory(dir);
            string exe = Path.Combine(dir, "vc_redist.x64.exe");
            try
            {
                status("Downloading from Microsoft...");
                using (var http = new HttpClient { Timeout = TimeSpan.FromMinutes(5) })
                {
                    http.DefaultRequestHeaders.UserAgent.ParseAdd("SmoovPlay-installer/" + BuildInfo.Version);
                    var bytes = http.GetByteArrayAsync(VcRedistUrl).Result;
                    File.WriteAllBytes(exe, bytes);
                }
                if (!SignedByMicrosoft(exe)) return "The download isn't signed by Microsoft, so it wasn't run.";
                status("Installing (Windows asks for permission)...");
                var psi = new ProcessStartInfo(exe, "/install /passive /norestart") { UseShellExecute = true, Verb = "runas" };
                using (var pr = Process.Start(psi))
                {
                    pr.WaitForExit();
                    // 3010: installed, restart later; 1638: a newer version is already installed
                    if (pr.ExitCode == 0 || pr.ExitCode == 3010 || pr.ExitCode == 1638) return null;
                    return "The installer stopped with code " + pr.ExitCode + ".";
                }
            }
            catch (System.ComponentModel.Win32Exception e) when (e.NativeErrorCode == 1223) { return "Cancelled at the Windows permission prompt."; }
            catch (Exception e) { return Installer.Message(e); }
            finally { try { Directory.Delete(dir, true); } catch (Exception) { } }
        }

        /// <summary>A valid Authenticode signature (WinVerifyTrust: the file matches its signature and the certificate
        /// chains to a trusted root) whose signer is Microsoft.</summary>
        public static bool SignedByMicrosoft(string file)
        {
            try
            {
                if (!Authenticode.Verify(file)) return false;
                var cert = System.Security.Cryptography.X509Certificates.X509Certificate.CreateFromSignedFile(file);
                return cert.Subject.Contains("O=Microsoft Corporation");
            }
            catch (Exception) { return false; }
        }

        static class Authenticode
        {
            [StructLayout(LayoutKind.Sequential, CharSet = CharSet.Unicode)]
            class FileInfo
            {
                public uint cbStruct = (uint)Marshal.SizeOf(typeof(FileInfo));
                public string pcwszFilePath;
                public IntPtr hFile = IntPtr.Zero, pgKnownSubject = IntPtr.Zero;
            }

            [StructLayout(LayoutKind.Sequential)]
            class TrustData
            {
                public uint cbStruct = (uint)Marshal.SizeOf(typeof(TrustData));
                public IntPtr pPolicyCallbackData = IntPtr.Zero, pSIPClientData = IntPtr.Zero;
                public uint dwUIChoice = 2;           // WTD_UI_NONE
                public uint fdwRevocationChecks = 0;  // WTD_REVOKE_NONE
                public uint dwUnionChoice = 1;        // WTD_CHOICE_FILE
                public IntPtr pFile;
                public uint dwStateAction = 0;
                public IntPtr hWVTStateData = IntPtr.Zero, pwszURLReference = IntPtr.Zero;
                public uint dwProvFlags = 0;
                public uint dwUIContext = 0;
                public IntPtr pSignatureSettings = IntPtr.Zero;
            }

            [DllImport("wintrust.dll", CharSet = CharSet.Unicode)]
            static extern int WinVerifyTrust(IntPtr hwnd, [MarshalAs(UnmanagedType.LPStruct)] Guid action, TrustData data);

            static readonly Guid VerifyV2 = new Guid("00AAC56B-CD44-11d0-8CC2-00C04FC295EE");  // WINTRUST_ACTION_GENERIC_VERIFY_V2

            public static bool Verify(string file)
            {
                var fi = new FileInfo { pcwszFilePath = file };
                IntPtr pfi = Marshal.AllocHGlobal(Marshal.SizeOf(typeof(FileInfo)));
                try
                {
                    Marshal.StructureToPtr(fi, pfi, false);
                    return WinVerifyTrust(new IntPtr(-1), VerifyV2, new TrustData { pFile = pfi }) == 0;  // INVALID_HANDLE_VALUE: no UI
                }
                finally
                {
                    Marshal.DestroyStructure(pfi, typeof(FileInfo));
                    Marshal.FreeHGlobal(pfi);
                }
            }
        }

        /// <summary>Turns on Windows' long paths (admin; Windows asks). Returns null on success.</summary>
        public static string EnableLongPaths()
        {
            try
            {
                var psi = new ProcessStartInfo("reg.exe", @"add HKLM\SYSTEM\CurrentControlSet\Control\FileSystem /v LongPathsEnabled /t REG_DWORD /d 1 /f")
                { UseShellExecute = true, Verb = "runas", WindowStyle = ProcessWindowStyle.Hidden };
                using (var pr = Process.Start(psi)) { pr.WaitForExit(); return pr.ExitCode == 0 ? null : "reg.exe stopped with code " + pr.ExitCode; }
            }
            catch (System.ComponentModel.Win32Exception e) when (e.NativeErrorCode == 1223) { return "Cancelled at the Windows permission prompt."; }
            catch (Exception e) { return e.Message; }
        }
    }
}
