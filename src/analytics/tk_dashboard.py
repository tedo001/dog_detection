"""
Native Tkinter analytics dashboard — statistics inside the app, no browser.

``DashboardWindow`` is a dark-themed Toplevel that reads the recorded sessions
(``data/sessions/*.json``) and draws the statistics directly on Tk Canvases:

    - KPI row .......... alerts fired, peak dogs / persons in frame,
                        frames processed, peak risk
    - Risk over time ... per-session risk timeline (single) or peak-risk per
                        session (all), with the alert threshold line
    - Detections ....... dogs vs persons (peak concurrent per session)
    - Alerts by hour ... when dogs turned dangerous
    - Risk distribution  share of frames by risk band
    - Sessions table ... every run and its stats

The aggregation helpers at the top are pure (no Tk) so they are unit-testable;
the drawing methods use only tk.Canvas primitives so there is no extra
dependency (no matplotlib, no web view).
"""

from math import ceil

from .recorder import load_sessions

# ── palette (matches app2/app3 dark theme) ────────────────────────────
BG = "#1e1e1e"
CARD = "#262626"
PANEL = "#232323"
BLUE = "#3987e5"
AQUA = "#199e70"
RED = "#d03b3b"
GRID = "#2f2f2d"
BASE = "#3f3f3c"
TXT = "#e8e8e8"
MUTED = "#898781"
AMBER = "#f59e0b"


# ── pure aggregation (no Tk — unit-testable) ─────────────────────────

def peak_dogs(s):
    return max((p["dogs"] for p in s.get("timeline", [])), default=0)


def peak_persons(s):
    return max((p["persons"] for p in s.get("timeline", [])), default=0)


def mmss(t):
    return f"{int(t // 60):02d}:{int(t % 60):02d}"


def sess_label(sid):
    try:
        return f"{sid[4:6]}-{sid[6:8]} {sid[9:11]}:{sid[11:13]}"
    except Exception:
        return str(sid)


def scope(sessions, sel):
    """Return the sessions in view for a selected session id (or all)."""
    if sel:
        return [s for s in sessions if s["session_id"] == sel]
    return sessions


def compute_kpis(S):
    if not S:
        return {"alerts": 0, "peak_dogs": 0, "peak_persons": 0,
                "frames": 0, "duration_s": 0.0, "peak_risk": 0.0, "avg_fps": 0.0}
    return {
        "alerts": sum(s["alerts_total"] for s in S),
        "peak_dogs": max(peak_dogs(s) for s in S),
        "peak_persons": max(peak_persons(s) for s in S),
        "frames": sum(s["frames_processed"] for s in S),
        "duration_s": sum(s["duration_s"] for s in S),
        "peak_risk": max(s["peak_risk"] for s in S),
        "avg_fps": sum(s["avg_fps"] for s in S) / len(S),
    }


def compute_hours(S):
    """Alerts per hour-of-day (24 buckets)."""
    hours = [0] * 24
    for s in S:
        try:
            h0 = int(s["started"][11:13])
            m0 = int(s["started"][14:16])
        except Exception:
            h0, m0 = 0, 0
        for a in s.get("alerts", []):
            hours[(h0 + int((m0 * 60 + a["t"]) // 3600)) % 24] += 1
    return hours


def compute_risk_hist(S):
    """Return (10 shares summing ~1, total frames)."""
    bins = [0] * 10
    total = 0
    for s in S:
        for p in s.get("timeline", []):
            bins[min(9, int(p["risk"] * 10))] += 1
            total += 1
    shares = [(b / total if total else 0.0) for b in bins]
    return shares, total


# ── Tk dashboard ─────────────────────────────────────────────────────

def open_dashboard(parent, sessions_dir="data/sessions"):
    """Convenience entry point used by app3."""
    return DashboardWindow(parent, sessions_dir=sessions_dir)


class DashboardWindow:
    """A Toplevel window that draws the analytics natively with tk.Canvas."""

    CW, CH = 540, 200          # chart canvas size

    def __init__(self, parent, sessions_dir="data/sessions"):
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.sessions_dir = sessions_dir
        self.sessions = []

        self.win = tk.Toplevel(parent)
        self.win.title("Dog Aggression Detection — Analytics Dashboard")
        self.win.configure(bg=BG)
        self.win.geometry("1180x900")

        self._build()
        self.refresh()

    # -- layout --
    def _build(self):
        tk, ttk = self.tk, self.ttk

        # top bar
        top = tk.Frame(self.win, bg=BG)
        top.pack(fill="x", padx=14, pady=(12, 6))
        tk.Label(top, text="Analytics Dashboard", bg=BG, fg=AMBER,
                 font=("Segoe UI", 14, "bold")).pack(side="left")
        self.info_lbl = tk.Label(top, text="", bg=BG, fg=MUTED,
                                 font=("Segoe UI", 9))
        self.info_lbl.pack(side="left", padx=12)

        tk.Button(top, text="↻ Refresh", command=self.refresh, bg="#3a3a3a",
                  fg="white", relief="flat", padx=10, pady=3,
                  cursor="hand2").pack(side="right")
        tk.Button(top, text="Export HTML", command=self._export_html,
                  bg="#3a3a3a", fg="white", relief="flat", padx=10, pady=3,
                  cursor="hand2").pack(side="right", padx=(0, 8))
        tk.Label(top, text="Session:", bg=BG, fg=TXT).pack(side="right", padx=(0, 6))
        self.sess_var = tk.StringVar(value="All sessions")
        self.sess_combo = ttk.Combobox(top, textvariable=self.sess_var,
                                       state="readonly", width=28)
        self.sess_combo.pack(side="right")
        self.sess_combo.bind("<<ComboboxSelected>>", lambda e: self._redraw())

        # KPI row
        kpi_row = tk.Frame(self.win, bg=BG)
        kpi_row.pack(fill="x", padx=14, pady=6)
        self.kpi = {}
        for key, label in [("alerts", "Alerts fired"),
                           ("peak_dogs", "Peak dogs in frame"),
                           ("peak_persons", "Peak persons in frame"),
                           ("frames", "Frames processed"),
                           ("peak_risk", "Peak risk")]:
            tile = tk.Frame(kpi_row, bg=CARD, padx=14, pady=10)
            tile.pack(side="left", fill="both", expand=True, padx=4)
            val = tk.Label(tile, text="0", bg=CARD, fg=TXT,
                           font=("Segoe UI", 20, "bold"))
            val.pack(anchor="w")
            tk.Label(tile, text=label, bg=CARD, fg=MUTED,
                     font=("Segoe UI", 9)).pack(anchor="w")
            sub = tk.Label(tile, text="", bg=CARD, fg="#6f6f6f",
                           font=("Segoe UI", 8))
            sub.pack(anchor="w")
            self.kpi[key] = (val, sub)

        # charts grid (2 x 2)
        grid = tk.Frame(self.win, bg=BG)
        grid.pack(fill="x", padx=14, pady=6)

        def chart_card(row, col, title):
            card = tk.Frame(grid, bg=CARD)
            card.grid(row=row, column=col, padx=4, pady=4, sticky="nsew")
            tk.Label(card, text=title, bg=CARD, fg=TXT,
                     font=("Segoe UI", 10, "bold")).pack(anchor="w", padx=8, pady=(6, 0))
            cv = tk.Canvas(card, width=self.CW, height=self.CH,
                           bg=CARD, highlightthickness=0)
            cv.pack(padx=8, pady=(2, 8))
            return cv

        grid.columnconfigure(0, weight=1)
        grid.columnconfigure(1, weight=1)
        self.risk_title = tk.StringVar(value="Risk over time")
        self.cv_risk = chart_card(0, 0, "Risk over time")
        self.cv_det = chart_card(0, 1, "Detections — dogs vs persons")
        self.cv_hour = chart_card(1, 0, "Alerts by hour of day")
        self.cv_hist = chart_card(1, 1, "Risk distribution")

        # sessions table
        tbl_frame = tk.Frame(self.win, bg=BG)
        tbl_frame.pack(fill="both", expand=True, padx=14, pady=(6, 12))
        tk.Label(tbl_frame, text="Sessions", bg=BG, fg=TXT,
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        cols = ("session", "started", "source", "model", "mode",
                "frames", "peak_dogs", "alerts", "peak_risk", "fps")
        headers = ("Session", "Started", "Source", "Model", "Mode",
                   "Frames", "Peak dogs", "Alerts", "Peak risk", "Avg FPS")
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("Dash.Treeview", background=PANEL, foreground=TXT,
                        fieldbackground=PANEL, rowheight=22, borderwidth=0)
        style.configure("Dash.Treeview.Heading", background=CARD, foreground=TXT)
        self.table = ttk.Treeview(tbl_frame, columns=cols, show="headings",
                                  height=8, style="Dash.Treeview")
        for c, h in zip(cols, headers):
            self.table.heading(c, text=h)
            self.table.column(c, width=90, anchor="center")
        self.table.column("started", width=150)
        self.table.pack(fill="both", expand=True, pady=(4, 0))

    # -- data / redraw --
    def refresh(self):
        from datetime import datetime
        self.sessions = load_sessions(self.sessions_dir)
        vals = ["All sessions"] + [
            f"{sess_label(s['session_id'])}  ·  {s['model']}  ·  "
            f"{s['alerts_total']} alert(s)"
            for s in reversed(self.sessions)]
        self.sess_combo["values"] = vals
        if self.sess_var.get() not in vals:
            self.sess_var.set("All sessions")
        self.info_lbl.config(
            text=f"{len(self.sessions)} session(s) · updated "
                 f"{datetime.now():%H:%M:%S}")
        self._redraw()

    def _selected_id(self):
        idx = self.sess_combo.current()
        if idx <= 0:
            return None
        return list(reversed(self.sessions))[idx - 1]["session_id"]

    def _redraw(self):
        S = scope(self.sessions, self._selected_id())
        single = self._selected_id() is not None and len(S) == 1

        k = compute_kpis(S)
        self.kpi["alerts"][0].config(text=f"{k['alerts']:,}")
        self.kpi["alerts"][1].config(text="aggression alerts")
        self.kpi["peak_dogs"][0].config(text=str(k["peak_dogs"]))
        self.kpi["peak_dogs"][1].config(text="most seen at once")
        self.kpi["peak_persons"][0].config(text=str(k["peak_persons"]))
        self.kpi["peak_persons"][1].config(text="most seen at once")
        self.kpi["frames"][0].config(text=f"{k['frames']:,}")
        self.kpi["frames"][1].config(text=f"{mmss(k['duration_s'])} monitored")
        self.kpi["peak_risk"][0].config(text=f"{k['peak_risk']:.2f}")
        self.kpi["peak_risk"][1].config(text=f"avg FPS {k['avg_fps']:.1f}")

        # risk chart
        if not S:
            self._empty(self.cv_risk, "No sessions recorded yet")
        elif single:
            tl = S[0]["timeline"]
            markers = []
            for a in S[0].get("alerts", []):
                idx = min(range(len(tl)), key=lambda i: abs(tl[i]["t"] - a["t"]))
                markers.append((idx, a["risk"]))
            self._line(self.cv_risk, [p["t"] for p in tl],
                       [p["risk"] for p in tl], BLUE,
                       threshold=S[0]["risk_threshold"], ymax=1.0,
                       xfmt=mmss, markers=markers)
        else:
            self._bars(self.cv_risk, [s["session_id"] for s in S],
                       [{"color": BLUE, "values": [s["peak_risk"] for s in S]}],
                       pct=True, catfmt=sess_label,
                       tick_every=max(1, len(S) // 6))

        # detections
        if not S:
            self._empty(self.cv_det, "No data")
        elif single:
            tl = S[0]["timeline"]
            self._line_multi(self.cv_det, [p["t"] for p in tl],
                             [{"color": BLUE, "values": [p["dogs"] for p in tl]},
                              {"color": AQUA, "values": [p["persons"] for p in tl]}],
                             xfmt=mmss)
        else:
            self._bars(self.cv_det, [s["session_id"] for s in S],
                       [{"color": BLUE, "values": [peak_dogs(s) for s in S]},
                        {"color": AQUA, "values": [peak_persons(s) for s in S]}],
                       int_ticks=True, catfmt=sess_label,
                       tick_every=max(1, len(S) // 5))

        # alerts by hour
        self._bars(self.cv_hour, list(range(24)),
                   [{"color": BLUE, "values": compute_hours(S)}],
                   int_ticks=True, catfmt=lambda h: f"{h:02d}", tick_every=4)

        # risk distribution
        shares, _ = compute_risk_hist(S)
        self._bars(self.cv_hist, list(range(10)),
                   [{"color": BLUE, "values": shares}],
                   pct=True, catfmt=lambda i: f"{i/10:.1f}", tick_every=2)

        # table (always all sessions)
        for row in self.table.get_children():
            self.table.delete(row)
        for s in reversed(self.sessions):
            self.table.insert("", "end", values=(
                s["session_id"], s["started"].replace("T", " "),
                s["source"], s["model"].replace(".pt", ""),
                s["alert_type"].upper(), f"{s['frames_processed']:,}",
                peak_dogs(s), s["alerts_total"], f"{s['peak_risk']:.2f}",
                f"{s['avg_fps']:.1f}"))

    def _export_html(self):
        from tkinter import messagebox
        try:
            import webbrowser
            from .dashboard import generate_dashboard
            path = generate_dashboard(sessions_dir=self.sessions_dir,
                                      open_browser=True)
            messagebox.showinfo("Exported", f"HTML dashboard:\n{path}")
        except Exception as e:
            messagebox.showerror("Export failed", str(e))

    # -- canvas drawing primitives --
    ML, MR, MT, MB = 44, 12, 12, 26

    def _empty(self, cv, text):
        cv.delete("all")
        cv.create_rectangle(0, 0, self.CW, self.CH, fill=CARD, outline="")
        cv.create_text(self.CW / 2, self.CH / 2, text=text, fill=MUTED,
                       font=("Segoe UI", 10))

    def _axes(self, cv, ymax, pct):
        ML, MR, MT, MB = self.ML, self.MR, self.MT, self.MB
        iw = self.CW - ML - MR
        ih = self.CH - MT - MB
        cv.delete("all")
        cv.create_rectangle(0, 0, self.CW, self.CH, fill=CARD, outline="")
        for i in range(5):
            y = MT + ih - ih * i / 4
            cv.create_line(ML, y, ML + iw, y, fill=GRID)
            if pct:
                lab = f"{int(round(ymax * i / 4 * 100))}%"
            elif ymax <= 2:
                lab = f"{ymax * i / 4:.2f}"
            else:
                lab = f"{int(ymax * i / 4)}"
            cv.create_text(ML - 4, y, text=lab, anchor="e", fill=MUTED,
                           font=("Segoe UI", 7))
        cv.create_line(ML, MT + ih, ML + iw, MT + ih, fill=BASE)
        return iw, ih

    def _ymax(self, vals, pct, int_ticks):
        m = max(vals) if vals else 0
        if pct:
            return 1.0
        if int_ticks:
            return max(4, ceil(max(m, 1) / 4) * 4)
        return (m * 1.15) if m > 0 else 1.0

    def _bars(self, cv, cats, series, pct=False, int_ticks=False,
              catfmt=str, tick_every=1):
        vals = [v for s in series for v in s["values"]]
        ymax = self._ymax(vals, pct, int_ticks)
        iw, ih = self._axes(cv, ymax, pct)
        ML, MT = self.ML, self.MT
        n = len(cats)
        if n == 0:
            return
        ns = len(series)
        band = iw / n
        bw = min(22.0, max(3.0, (band - 6) / ns - 2))
        gw = bw * ns + 2 * (ns - 1)
        for ci, cat in enumerate(cats):
            gx = ML + band * ci + (band - gw) / 2
            for si, s in enumerate(series):
                v = s["values"][ci]
                bh = (v / ymax) * ih if ymax > 0 else 0
                if bh <= 0:
                    continue
                x = gx + si * (bw + 2)
                cv.create_rectangle(x, MT + ih - bh, x + bw, MT + ih,
                                    fill=s["color"], outline="")
            if ci % tick_every == 0:
                cv.create_text(ML + band * ci + band / 2, MT + ih + 12,
                               text=catfmt(cat), fill=MUTED,
                               font=("Segoe UI", 7))

    def _line(self, cv, xs, values, color, threshold=None, ymax=1.0,
              xfmt=str, markers=None):
        self._line_multi(cv, xs, [{"color": color, "values": values}],
                         ymax=ymax, threshold=threshold, xfmt=xfmt,
                         area=True, markers=markers)

    def _line_multi(self, cv, xs, series, ymax=None, threshold=None,
                    xfmt=str, area=False, markers=None):
        vals = [v for s in series for v in s["values"]]
        if ymax is None:
            ymax = self._ymax(vals, False, True)
        iw, ih = self._axes(cv, ymax, False)
        ML, MT = self.ML, self.MT
        n = len(xs)
        if n < 2:
            cv.create_text(self.CW / 2, self.CH / 2, text="Not enough data",
                           fill=MUTED, font=("Segoe UI", 9))
            return

        def X(i):
            return ML + iw * i / (n - 1)

        def Y(v):
            return MT + ih - min(1.0, v / ymax) * ih

        if threshold is not None and threshold <= ymax:
            ty = Y(threshold)
            cv.create_line(ML, ty, ML + iw, ty, fill=RED, dash=(3, 2))

        for s in series:
            pts = []
            for i, v in enumerate(s["values"]):
                pts += [X(i), Y(v)]
            if area and len(series) == 1:
                cv.create_polygon(ML, MT + ih, *pts, ML + iw, MT + ih,
                                  fill=s["color"], stipple="gray25", outline="")
            cv.create_line(*pts, fill=s["color"], width=2)
            cv.create_oval(X(n - 1) - 3, Y(s["values"][-1]) - 3,
                           X(n - 1) + 3, Y(s["values"][-1]) + 3,
                           fill=s["color"], outline=CARD)

        for (mi, mv) in (markers or []):
            if 0 <= mi < n:
                cv.create_oval(X(mi) - 3, Y(mv) - 3, X(mi) + 3, Y(mv) + 3,
                               fill=RED, outline=CARD)

        ticks = min(6, n)
        for t in range(ticks):
            i = round(t * (n - 1) / max(1, ticks - 1))
            cv.create_text(X(i), MT + ih + 12, text=xfmt(xs[i]), fill=MUTED,
                           font=("Segoe UI", 7))
