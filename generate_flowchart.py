"""
Generate Dog Aggression Detection — Workflow Diagram PDF
Run: python generate_flowchart.py
Output: dog_detection_workflow.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import matplotlib.patheffects as pe

# ── colour palette ──────────────────────────────────────────
C_BG      = "#1e1e1e"
C_PANEL   = "#252525"
C_TITLE   = "#f59e0b"
C_BLUE    = "#0078d4"
C_GREEN   = "#16a34a"
C_RED     = "#dc2626"
C_PURPLE  = "#7c3aed"
C_TEAL    = "#0ea5e9"
C_GREY    = "#3a3a3a"
C_WHITE   = "#ffffff"
C_LIGHT   = "#cccccc"
C_ORANGE  = "#f59e0b"

fig, ax = plt.subplots(figsize=(18, 24))
fig.patch.set_facecolor(C_BG)
ax.set_facecolor(C_BG)
ax.set_xlim(0, 18)
ax.set_ylim(0, 24)
ax.axis("off")


def box(x, y, w, h, text, color, text_color=C_WHITE,
        fontsize=10, bold=False, radius=0.3, alpha=1.0, sub=""):
    patch = FancyBboxPatch((x, y), w, h,
                           boxstyle=f"round,pad=0,rounding_size={radius}",
                           facecolor=color, edgecolor="#444444",
                           linewidth=1.2, alpha=alpha, zorder=3)
    ax.add_patch(patch)
    weight = "bold" if bold else "normal"
    ty = y + h / 2 + (0.12 if sub else 0)
    ax.text(x + w / 2, ty, text,
            ha="center", va="center", color=text_color,
            fontsize=fontsize, fontweight=weight, zorder=4,
            wrap=True)
    if sub:
        ax.text(x + w / 2, y + h / 2 - 0.22, sub,
                ha="center", va="center", color="#aaaaaa",
                fontsize=7.5, zorder=4)


def diamond(cx, cy, w, h, text, color, text_color=C_WHITE, fontsize=9):
    xs = [cx, cx + w/2, cx, cx - w/2, cx]
    ys = [cy + h/2, cy, cy - h/2, cy, cy + h/2]
    ax.fill(xs, ys, color=color, zorder=3, edgecolor="#444444", linewidth=1.2)
    ax.text(cx, cy, text, ha="center", va="center",
            color=text_color, fontsize=fontsize, fontweight="bold", zorder=4)


def arrow(x1, y1, x2, y2, color="#888888", label="", lw=1.8):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color=color,
                                lw=lw, mutation_scale=14),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx + 0.1, my, label, color=color,
                fontsize=7.5, va="center", zorder=4)


def line(x1, y1, x2, y2, color="#555555", lw=1.4, style="-"):
    ax.plot([x1, x2], [y1, y2], color=color, lw=lw, ls=style, zorder=2)


def section_label(x, y, text):
    ax.text(x, y, text, color=C_ORANGE, fontsize=8.5,
            fontweight="bold", va="center", zorder=4)

# ══════════════════════════════════════════════════════════════
# TITLE
# ══════════════════════════════════════════════════════════════
ax.text(9, 23.4, "Dog Aggression Detection System",
        ha="center", va="center", color=C_ORANGE,
        fontsize=20, fontweight="bold", zorder=4)
ax.text(9, 23.0, "Workflow Diagram  —  YOLO26 + ESP32-CAM + HC-SR04",
        ha="center", va="center", color=C_LIGHT, fontsize=11, zorder=4)
line(1, 22.7, 17, 22.7, color="#444444", lw=1)

# ══════════════════════════════════════════════════════════════
# ROW 1 — INPUT SOURCES  (y≈21)
# ══════════════════════════════════════════════════════════════
section_label(0.3, 21.9, "① INPUT SOURCES")

box(0.4, 20.8, 3.2, 0.9, "Video File",       C_BLUE,   fontsize=10, bold=True)
box(4.4, 20.8, 3.2, 0.9, "Laptop Webcam",    C_BLUE,   fontsize=10, bold=True)
box(8.4, 20.8, 3.2, 0.9, "ESP32-CAM",        C_PURPLE, fontsize=10, bold=True,
    sub="http://<IP>:81/stream  (MJPEG)")

# ESP32 hardware block — side column
box(13.0, 19.6, 4.5, 3.0, "", C_GREY, radius=0.35)
ax.text(15.25, 22.3, "ESP32-CAM Board", color=C_ORANGE,
        ha="center", fontsize=10, fontweight="bold", zorder=4)
box(13.2, 21.4, 4.1, 0.65, "WiFi (same network as laptop)",
    "#2a2a2a", fontsize=8.5)
box(13.2, 20.65, 4.1, 0.65, "MJPEG Camera Stream  :81/stream",
    "#2a2a2a", fontsize=8.5)
box(13.2, 19.9, 4.1, 0.65, "HC-SR04 Distance API  /distance",
    "#2a2a2a", fontsize=8.5)

# arrows: sources → centre merge
arrow(2.0, 20.8, 6.2, 20.0, C_BLUE)
arrow(6.0, 20.8, 6.2, 20.0, C_BLUE)
arrow(10.0, 20.8, 6.4, 20.0, C_PURPLE)
arrow(13.0, 21.0, 11.6, 21.25, C_PURPLE, lw=1.4)

# ══════════════════════════════════════════════════════════════
# ROW 2 — MJPEG CAPTURE  (y≈19.4)
# ══════════════════════════════════════════════════════════════
section_label(0.3, 19.6, "② VIDEO CAPTURE")
box(3.5, 19.2, 5.5, 0.9, "MJPEGCapture  /  cv2.VideoCapture",
    C_GREY, fontsize=9.5, bold=True,
    sub="reads raw JPEG bytes, decodes each frame (no FFmpeg needed on Windows)")

arrow(6.25, 19.2, 6.25, 18.55, C_LIGHT)

# ══════════════════════════════════════════════════════════════
# ROW 3 — YOLO26 PIPELINE  (y≈17.0)
# ══════════════════════════════════════════════════════════════
section_label(0.3, 18.35, "③ AI DETECTION  (YOLO26)")

box(0.4, 16.8, 2.6, 1.4, "YOLO26\nDetector",
    C_GREEN, fontsize=9.5, bold=True,
    sub="yolo26n / s / m / l / x")
box(3.3, 16.8, 2.6, 1.4, "YOLO11m\nPose Model",
    C_GREEN, fontsize=9.5, bold=True,
    sub="human skeleton")
box(6.2, 16.8, 2.6, 1.4, "Track\n& Sustain",
    C_GREEN, fontsize=9.5, bold=True,
    sub="dog track IDs")
box(9.1, 16.8, 2.6, 1.4, "Risk Score\nCalculator",
    "#b45309", fontsize=9.5, bold=True,
    sub="0.0 → 1.0")

arrow(6.25, 18.2,  2.0, 18.2,  C_LIGHT)   # horiz to detector
line( 2.0, 18.2,  2.0, 18.2)
arrow(2.0,  18.2,  2.0, 18.2)
arrow(3.0,  17.5,  3.3, 17.5,  C_LIGHT)
arrow(5.9,  17.5,  6.2, 17.5,  C_LIGHT)
arrow(8.8,  17.5,  9.1, 17.5,  C_LIGHT)

# cleaner: chain arrows between boxes
for x1, x2, y_ in [(3.0,3.3,17.5),(5.9,6.2,17.5),(8.8,9.1,17.5)]:
    arrow(x1, y_, x2, y_, C_LIGHT)
# first arrow from capture down
ax.annotate("", xy=(2.0, 18.2), xytext=(6.25, 18.55),
            arrowprops=dict(arrowstyle="-|>", color=C_LIGHT, lw=1.8,
                            mutation_scale=14), zorder=2)
ax.annotate("", xy=(2.0, 18.2), xytext=(2.0, 18.2),
            arrowprops=dict(arrowstyle="-|>", color=C_LIGHT, lw=1.8,
                            mutation_scale=14), zorder=2)

# simpler direct arrows through pipeline
arrow(6.25, 19.2, 6.25, 18.55, C_LIGHT)
line(6.25, 18.55, 2.0, 18.55); arrow(2.0, 18.55, 2.0, 18.2, C_LIGHT)
arrow(3.0, 17.5, 3.3, 17.5, C_LIGHT)
arrow(5.9, 17.5, 6.2, 17.5, C_LIGHT)
arrow(8.8, 17.5, 9.1, 17.5, C_LIGHT)

# ══════════════════════════════════════════════════════════════
# ROW 4 — DECISION  (y≈15.8)
# ══════════════════════════════════════════════════════════════
arrow(10.4, 16.8, 10.4, 16.3, C_ORANGE)
diamond(10.4, 15.5, 3.6, 1.4,
        "Risk Score ≥ Threshold?", C_ORANGE, C_BG, fontsize=9)

# NO path
arrow(8.6, 15.5, 6.0, 15.5, C_GREY, label="NO")
box(3.2, 15.1, 2.6, 0.8, "Next Frame", C_GREY, fontsize=9)
arrow(4.5, 15.1, 4.5, 14.6, C_GREY)
line(4.5, 14.6, 2.0, 14.6); line(2.0, 14.6, 2.0, 17.5)
arrow(2.0, 17.5, 2.0, 18.2, C_GREY)   # loop back

# YES path
arrow(10.4, 14.8, 10.4, 14.3, C_RED, label="YES")

# ══════════════════════════════════════════════════════════════
# ROW 5 — ALERT TYPE DECISION  (y≈13.5)
# ══════════════════════════════════════════════════════════════
section_label(0.3, 14.1, "④ ALERT SYSTEM")
diamond(10.4, 13.2, 3.2, 1.2, "Alert Type?", C_PURPLE, C_WHITE, fontsize=9)

# NORMAL branch (left)
arrow(8.8, 13.2, 6.6, 13.2, C_BLUE, label="NORMAL")
box(4.0, 12.7, 2.4, 1.0, "Bell\nSound", C_BLUE, fontsize=9, bold=True)

# HR branch (right)
arrow(12.0, 13.2, 13.5, 13.2, C_RED, label="HR")
box(13.0, 12.7, 3.8, 1.0, "Bell + Red Border\nFlash + Lower Threshold",
    C_RED, fontsize=9, bold=True)

# both merge down
arrow(5.2,  12.7, 5.2,  12.1, C_BLUE)
arrow(14.9, 12.7, 14.9, 12.1, C_RED)

# ══════════════════════════════════════════════════════════════
# ROW 6 — ESP SENSOR PARALLEL  (y≈11.5)
# ══════════════════════════════════════════════════════════════
section_label(0.3, 12.0, "⑤ ESP ULTRASONIC SENSOR  (parallel)")

box(0.4, 10.6, 4.0, 1.2, "HC-SR04\nUltrasonic Sensor",
    C_TEAL, fontsize=9.5, bold=True, sub="TRIG=GPIO14  ECHO=GPIO13")
arrow(4.4, 11.2, 5.6, 11.2, C_TEAL)
box(5.6, 10.6, 3.5, 1.2, "ESP32 HTTP\n/distance API",
    C_TEAL, fontsize=9.5, bold=True, sub="JSON: {distance_cm: 42.5}")
arrow(9.1, 11.2, 10.3, 11.2, C_TEAL)
diamond(11.5, 11.2, 3.0, 1.0, "< Threshold?", C_TEAL, C_BG, fontsize=8.5)
arrow(13.0, 11.2, 14.5, 11.2, C_ORANGE, label="YES")
box(14.5, 10.75, 3.0, 0.9, "Log Proximity\nAlert", C_ORANGE, fontsize=8.5)

arrow(11.5, 10.7, 11.5, 10.2, C_GREY, label="NO")
ax.text(11.5, 10.0, "Continue polling (0.5 s)", color="#666666",
        ha="center", fontsize=8, zorder=4)

# ══════════════════════════════════════════════════════════════
# ROW 7 — OUTPUT  (y≈9)
# ══════════════════════════════════════════════════════════════
section_label(0.3, 9.6, "⑥ OUTPUT")

# merge alerts → output
line(5.2, 12.1, 5.2, 9.4); line(14.9, 12.1, 14.9, 9.4)
line(5.2, 9.4, 14.9, 9.4)
arrow(10.0, 9.4, 10.0, 9.2, C_LIGHT)

box(0.4,  8.2, 3.0, 0.9, "Live Stats\nFPS / Frames / Dogs",    C_GREY, fontsize=8.5)
box(3.7,  8.2, 3.0, 0.9, "Alert Log\nTimestamp + Risk Score",  C_GREY, fontsize=8.5)
box(7.0,  8.2, 3.0, 0.9, "Save Annotated\nOutput Video (.mp4)",C_GREY, fontsize=8.5)
box(10.3, 8.2, 3.0, 0.9, "DIST(cm)\nLive Distance Bar",        C_TEAL, fontsize=8.5)
box(13.6, 8.2, 3.9, 0.9, "Export Alerts\nJSON file",           C_GREY, fontsize=8.5)

for cx in [1.9, 5.2, 8.5, 11.8, 15.55]:
    arrow(10.0, 9.2, cx, 9.1, C_LIGHT)
    arrow(cx, 9.1, cx, 8.2+0.9, C_LIGHT)

# ══════════════════════════════════════════════════════════════
# ROW 8 — VIDEO FORWARD  (side note)
# ══════════════════════════════════════════════════════════════
box(0.4, 7.0, 3.8, 0.85, "Video Forward (+10s)\nSkip ahead in file",
    C_PURPLE, fontsize=8.5, sub="only for Video File source")

# ══════════════════════════════════════════════════════════════
# LEGEND
# ══════════════════════════════════════════════════════════════
line(0.4, 6.5, 17.6, 6.5, color="#444444", lw=1)
ax.text(9, 6.3, "LEGEND", ha="center", color=C_ORANGE,
        fontsize=9, fontweight="bold", zorder=4)

legend_items = [
    (C_BLUE,   "Video / Webcam input"),
    (C_PURPLE, "ESP32-CAM stream"),
    (C_GREEN,  "AI detection (YOLO26)"),
    (C_RED,    "HR Alert"),
    (C_TEAL,   "Ultrasonic sensor"),
    (C_ORANGE, "Decision point"),
    (C_GREY,   "Processing / Output"),
]
for i, (color, label) in enumerate(legend_items):
    col = i % 4
    row = i // 4
    lx = 0.6 + col * 4.4
    ly = 5.8 - row * 0.55
    patch = FancyBboxPatch((lx, ly - 0.18), 0.5, 0.36,
                           boxstyle="round,pad=0,rounding_size=0.1",
                           facecolor=color, edgecolor="#555555",
                           linewidth=1, zorder=3)
    ax.add_patch(patch)
    ax.text(lx + 0.65, ly, label, color=C_LIGHT,
            fontsize=8.5, va="center", zorder=4)

# footer
ax.text(9, 4.7, "Dog Aggression Detection  —  YOLO26 + ESP32-CAM + HC-SR04  |  Python / Tkinter / OpenCV",
        ha="center", color="#555555", fontsize=8, zorder=4)

plt.tight_layout(pad=0)
plt.savefig("dog_detection_workflow.pdf",
            facecolor=C_BG, bbox_inches="tight", dpi=180)
plt.savefig("dog_detection_workflow.png",
            facecolor=C_BG, bbox_inches="tight", dpi=180)
print("Saved: dog_detection_workflow.pdf")
print("Saved: dog_detection_workflow.png")
