// Fill the screen (framegen/fill.py) on the Settings page: a small drawing beside its choices of what the picked one
// (or the one under the mouse) does with the console's 16:9 picture on this PC's widest screen. Consoles send 16:9
// only, so every choice is the same picture, kept whole with bars, cropped, or stretched: the drawing shows which.
// The maths is fill.py's: Zoom fills the width and cuts the top and bottom, Panorama bends the picture across the
// screen with u = atan(t*x) / atan(t).

using System;
using System.Linq;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Media;
using System.Windows.Shapes;

namespace SmoovPlay
{
    partial class Ui
    {
        Action<string> fillDraw;  // redraws the preview for a choice ("fit", "zoom", "stretch", "panorama")
        string fillPicked = "fit";
        ContentControl fillHost;
        string fillMonitor = "\0";

        const double FillH = 46;  // the screen's height in the drawing
        double FillPad = 9;  // room above and below it, for Zoom's cut-off parts (more on a wider screen)

        /// <summary>The preview: the console's picture, an arrow, and the screen with the picked choice.</summary>
        FrameworkElement FillPreview()
        {
            fillHost = new ContentControl();
            System.Windows.Automation.AutomationProperties.SetName(fillHost, "Fill the screen preview");
            fillMonitor = "\0";
            UpdateFillPreview(SettingText("monitor"), true);
            return fillHost;
        }

        void UpdateFillPreview(string monitor, bool force = false)
        {
            if (fillHost == null || (!force && monitor == fillMonitor)) return;
            fillMonitor = monitor;
            double ratio = SystemCheck.ScreenRatio(monitor);
            bool own = ratio > 1.85;  // an ultrawide here; else an example one
            if (!own) ratio = 21.0 / 9.0;
            string shape = ratio >= 3.2 ? "32:9" : ratio >= 2.2 ? "21:9" : null;
            double sh = 30, sw = Math.Round(sh * 16 / 9), fw = Math.Round(FillH * ratio);
            FillPad = Math.Max(9, Math.Ceiling((fw * 9 / 16 - FillH) / 2) + 3);

            var src = new Canvas { Width = sw, Height = FillH + 2 * FillPad };
            var pic = new Canvas();
            DrawFillPicture(pic, p => new Point(p.X * sw, FillPad + (FillH - sh) / 2 + p.Y * sh));
            src.Children.Add(pic);
            src.Children.Add(Outline(new Rect(0, FillPad + (FillH - sh) / 2, sw, sh)));

            var screen = new Canvas { Width = fw, Height = FillH + 2 * FillPad };
            fillDraw = mode => DrawFill(screen, mode, fw, ratio);
            fillDraw(fillPicked);

            var row = new StackPanel { Orientation = Orientation.Horizontal, VerticalAlignment = VerticalAlignment.Top };
            row.Children.Add(Captioned(src, "From the console: 16:9"));
            row.Children.Add(new TextBlock { Text = "→", FontSize = 16, Foreground = R("Muted"), Margin = new Thickness(10, FillPad + FillH / 2 - 12, 10, 0) });
            row.Children.Add(Captioned(screen, own ? "Your " + (shape ?? "wide") + " screen" : "On a 21:9 ultrawide"));
            fillHost.Content = row;
            SettingRow fillRow;
            if (settingRows.TryGetValue("fill", out fillRow) && fillRow.Hint != null)
                fillRow.Hint.Margin = new Thickness(18, FillPad, 0, 0);
        }

        FrameworkElement Captioned(FrameworkElement drawing, string caption)
        {
            var p = new StackPanel();
            p.Children.Add(drawing);
            p.Children.Add(new TextBlock { Text = caption, FontSize = 10.5, Foreground = R("Muted"), Margin = new Thickness(0, 1, 0, 0) });
            return p;
        }

        Rectangle Outline(Rect r)
        {
            var o = new Rectangle { Width = r.Width, Height = r.Height, Stroke = R("Text2"), StrokeThickness = 1, SnapsToDevicePixels = true };
            Canvas.SetLeft(o, r.X);
            Canvas.SetTop(o, r.Y);
            return o;
        }

        /// <summary>The screen (fw x FillH, under FillPad of room) with the picture laid out as `mode` does.</summary>
        void DrawFill(Canvas c, string mode, double fw, double ratio)
        {
            c.Children.Clear();
            var frame = new Rect(0, FillPad, fw, FillH);
            var bars = new Rectangle { Width = fw, Height = FillH, Fill = new SolidColorBrush(Color.FromRgb(5, 6, 8)) };
            Canvas.SetTop(bars, FillPad);
            c.Children.Add(bars);
            Func<Point, Point> map;
            if (mode == "zoom")  // as wide as the screen, true to shape: the top and bottom go past it
            {
                double ph = fw * 9 / 16, top = FillPad + (FillH - ph) / 2;
                map = p => new Point(p.X * fw, top + p.Y * ph);
                var cut = new Canvas { Opacity = 0.4 };  // the parts that are cut off, faint, outside the screen
                DrawFillPicture(cut, map);
                c.Children.Add(cut);
                var whole = new Rectangle { Width = fw, Height = ph, Stroke = R("Muted"), StrokeThickness = 1, StrokeDashArray = new DoubleCollection { 3, 2 } };
                Canvas.SetTop(whole, top);
                c.Children.Add(whole);
            }
            else if (mode == "stretch")
                map = p => new Point(p.X * fw, FillPad + p.Y * FillH);
            else if (mode == "panorama" && ratio > 16.0 / 9 * 1.01)
            {
                double t = PanoramaT(ratio / (16.0 / 9)), at = Math.Atan(t);
                map = p => new Point(fw / 2 * (1 + Math.Tan((2 * p.X - 1) * at) / t), FillPad + p.Y * FillH);
            }
            else  // fit: the screen's height, bars at the sides
            {
                double pw = FillH * 16 / 9, left = (fw - pw) / 2;
                map = p => new Point(left + p.X * pw, FillPad + p.Y * FillH);
            }
            var inside = new Canvas { Clip = new RectangleGeometry(frame) };
            DrawFillPicture(inside, map);
            c.Children.Add(inside);
            c.Children.Add(Outline(frame));
        }

        /// <summary>fill.py's panorama_t: the curve's t, so the middle is 15% of the extra width wider than true.</summary>
        static double PanoramaT(double k)
        {
            double goal = (1.0 + 0.15 * (k - 1.0)) / k, lo = 1e-6, hi = 1e3;
            for (int i = 0; i < 80; i++)
            {
                double mid = (lo + hi) / 2;
                if (Math.Atan(mid) / mid > goal) lo = mid; else hi = mid;
            }
            return (lo + hi) / 2;
        }

        /// <summary>A stand-in game picture through `map` (the picture's 0..1 x and y to the drawing): sky and ground,
        /// evenly spaced lines (bunched or spread by Panorama), a HUD in the top left corner (which Zoom cuts) and a
        /// round sun (which Stretch makes oval).</summary>
        void DrawFillPicture(Canvas c, Func<Point, Point> map)
        {
            Func<double, double, double, double, Brush, Polygon> quad = (x0, y0, x1, y1, fill) =>
            {
                var poly = new Polygon { Fill = fill };
                int n = 24;  // the edges in steps, so a bent picture's are drawn bent
                for (int i = 0; i <= n; i++) poly.Points.Add(map(new Point(x0 + (x1 - x0) * i / n, y0)));
                for (int i = n; i >= 0; i--) poly.Points.Add(map(new Point(x0 + (x1 - x0) * i / n, y1)));
                return poly;
            };
            c.Children.Add(quad(0, 0, 1, 0.62, new SolidColorBrush(Color.FromRgb(0x2B, 0x3B, 0x55))));
            c.Children.Add(quad(0, 0.62, 1, 1, new SolidColorBrush(Color.FromRgb(0x3A, 0x34, 0x2A))));
            var line = new SolidColorBrush(Color.FromArgb(70, 255, 255, 255));
            for (int i = 1; i < 8; i++)
            {
                Point a = map(new Point(i / 8.0, 0)), b = map(new Point(i / 8.0, 1));
                c.Children.Add(new Line { X1 = a.X, Y1 = a.Y, X2 = b.X, Y2 = b.Y, Stroke = line, StrokeThickness = 1 });
            }
            var sun = new Polygon { Fill = R("Text"), Opacity = 0.92 };
            double r = 0.2;  // of the picture's height
            for (int i = 0; i < 48; i++)
            {
                double a = i * Math.PI * 2 / 48;
                sun.Points.Add(map(new Point(0.5 + r * 9 / 16 * Math.Cos(a), 0.5 + r * Math.Sin(a))));
            }
            c.Children.Add(sun);
            c.Children.Add(quad(0.04, 0.07, 0.24, 0.17, R("Accent")));  // the HUD
        }
    }
}
