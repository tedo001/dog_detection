"""
Dog Aggression Detection — YOLO26 Edition + Analytics Dashboard (app3.py).

This is app2.py plus an analytics/statistics service. It reuses the entire
app2 GUI (source selection, ESP32-CAM, ultrasonic sensor, alert types, video
controls) and adds:

  - Corrected live statistics — "Dogs" / "Persons" now show the count IN the
    current frame (and the run summary reports the PEAK seen at once), instead
    of a running per-frame sum that made 4 dogs read as hundreds.
  - Per-session recording — every run is saved to data/sessions/*.json
    (risk timeline, per-frame dog/person counts, alerts, model, settings).
  - Analytics dashboard — an "Open Analytics Dashboard" button builds a
    self-contained offline HTML report and opens it in the browser:
    alerts fired, peak dogs / persons in frame, frames processed, peak risk,
    risk over time, detections (dogs vs persons), alerts by hour, risk
    distribution, and a per-session table.

app2.py is left completely untouched — app3 only extends it.

Run with:
    python app3.py
"""

import os
import sys
import json
import time
import urllib.request
from datetime import datetime
from pathlib import Path

import tkinter as tk
from tkinter import messagebox
import cv2

# Reuse everything app2 already provides — app3 subclasses its app.
from app2 import (
    DogAggressionAppV2, MJPEGCapture,
    SOURCE_VIDEO, SOURCE_WEBCAM, SOURCE_ESP_CAM,
)


class DogAggressionAppV3(DogAggressionAppV2):
    """app2's YOLO26 GUI + analytics dashboard and corrected live counts."""

    def __init__(self, root):
        super().__init__(root)
        self.root.title("Dog Aggression Detection — YOLO26 + Dashboard")
        self._recorder = None

    # ── UI: add a Dashboard button beneath app2's alert log ──────────

    def _build_display(self, parent):
        super()._build_display(parent)
        tk.Button(
            parent, text="📊  Open Analytics Dashboard",
            command=self.open_dashboard,
            bg="#7c3aed", fg="white", font=("Segoe UI", 10, "bold"),
            relief="flat", padx=15, pady=8, cursor="hand2",
        ).pack(fill="x", pady=(6, 0))

    def open_dashboard(self):
        """Build the HTML analytics dashboard from saved sessions and open it."""
        try:
            from src.analytics import generate_dashboard, load_sessions
            n = len(load_sessions())
            if n == 0:
                messagebox.showinfo(
                    "Analytics",
                    "No sessions recorded yet.\n\n"
                    "Run a detection first — every run is saved automatically, "
                    "then reopen the dashboard.")
                return
            path = generate_dashboard(open_browser=True)
            self.log(f"[analytics] Dashboard ({n} session(s)): {path}")
        except Exception as e:
            messagebox.showerror("Analytics", f"Could not build dashboard:\n{e}")

    # ── detection worker: app2's loop + stat fix + session recording ──

    def _detection_worker(self):
        self._recorder = None
        try:
            src = self.source_type.get()
            model_name = self.yolo26_variant.get()
            alert_t = self.alert_type.get()

            self._update_status(f"Loading {model_name}...")
            self.model_status.set(f"Loading {model_name}")
            self.log(f"Source: {src.upper()} | Alert type: {alert_t.upper()}")
            self.log(f"Initialising YOLO26: {model_name}")

            from src.inference.predict import DogAggressionPipeline

            pose_model = "yolo11m-pose.pt" if self.pose_enabled.get() else None

            eff_risk = self.risk_threshold.get()
            if alert_t == "hr":
                eff_risk = max(0.10, eff_risk - 0.10)
                self.log(f"[HR] Effective risk threshold: {eff_risk:.2f}")

            pipeline = DogAggressionPipeline(
                detector_path=model_name,
                pose_model=pose_model,
                det_conf=self.det_conf.get(),
                risk_threshold=eff_risk,
                sustain_frames=self.sustain_frames.get(),
            )
            self.model_status.set(f"{model_name} (active)")

            # Open capture source (identical to app2)
            if src == SOURCE_VIDEO:
                cap = cv2.VideoCapture(self.video_path)
                is_live = False
            elif src == SOURCE_WEBCAM:
                cap = cv2.VideoCapture(int(self.cam_index.get()))
                is_live = True
            else:  # ESP_CAM
                ip = self.esp_ip.get().strip()
                stream_url = f"http://{ip}:81/stream"
                self.log(f"[ESP-CAM] Pinging {ip} ...")
                self._update_status(f"Checking ESP32 at {ip} ...")
                try:
                    with urllib.request.urlopen(f"http://{ip}/status", timeout=3) as r:
                        info = json.loads(r.read())
                    self.log(f"[ESP-CAM] Device online  RSSI={info.get('wifi_rssi','?')} dBm")
                except Exception as ping_err:
                    clean = str(ping_err).replace("<", "").replace(">", "")
                    raise RuntimeError(
                        f"ESP32 not reachable at {ip}\n\nDetails: {clean}\n\n"
                        f"Check:\n  1. ESP32 is powered and WiFi connected\n"
                        f"  2. IP address is correct\n"
                        f"  3. Laptop and ESP32 are on the same WiFi")
                self.log(f"[ESP-CAM] Opening MJPEG stream: {stream_url}")
                self._update_status("Connecting to ESP32-CAM stream ...")
                cap = MJPEGCapture(stream_url, timeout=5)
                is_live = True

            if not cap.isOpened():
                raise RuntimeError("Could not open video source.")

            if src == SOURCE_ESP_CAM:
                self._update_status("Waiting for first frame ...")
                deadline = time.time() + 6
                ret_test = False
                while time.time() < deadline:
                    ret_test, _ = cap.read()
                    if ret_test:
                        break
                if not ret_test:
                    cap.release()
                    raise RuntimeError(
                        f"ESP32-CAM connected but no frames received within 6 s.\n\n"
                        f"Stream URL: {stream_url}\n"
                        f"Make sure the MJPEG server on the ESP32 is running.")

            self._update_status("Processing...")

            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_live else 0
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # ── analytics recorder (fail-safe: only observes the pipeline) ──
            try:
                from src.analytics import SessionRecorder
                self._recorder = SessionRecorder(
                    model=model_name, source=src, alert_type=alert_t,
                    risk_threshold=eff_risk, det_conf=self.det_conf.get())
            except Exception:
                self._recorder = None

            writer = None
            output_path = None
            if self.save_output.get() and not is_live:
                Path("experiments").mkdir(exist_ok=True)
                tag = "hr" if alert_t == "hr" else "norm"
                output_path = (
                    f"experiments/yolo26_{model_name.replace('.pt', '')}_{tag}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4")
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            skip = self.skip_frames.get()
            frame_count = processed = total_alerts = 0
            # counts are instantaneous (this frame) + peak concurrent, NOT a
            # per-frame running sum — that made 4 dogs read as hundreds.
            cur_persons = cur_dogs = peak_persons = peak_dogs = 0
            start_time = time.time()

            while not self.stop_requested:
                if not is_live:
                    with self._fwd_lock:
                        fwd = self.forward_count
                        if fwd > 0:
                            new_pos = frame_count + int(fwd * fps * 10)
                            cap.set(cv2.CAP_PROP_POS_FRAMES, new_pos)
                            frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES))
                            self.log(f"[FWD] +{fwd*10}s  ->  frame {frame_count}")
                            self.forward_count = 0

                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if (frame_count - 1) % skip != 0:
                    if writer:
                        writer.write(frame)
                    continue

                results = pipeline.process_frame(frame)
                annotated = pipeline.draw_results(frame, results)

                cur_persons = len(pipeline._last_persons)
                cur_dogs = len(results)
                peak_persons = max(peak_persons, cur_persons)
                peak_dogs = max(peak_dogs, cur_dogs)

                if self._recorder:
                    self._recorder.record_frame(results, cur_persons, frame_count)

                new_alerts_now = [r for r in results if r.get("new_alert")]
                total_alerts += len(new_alerts_now)

                if new_alerts_now:
                    timestamp_s = frame_count / fps
                    for a in new_alerts_now:
                        entry = {
                            "time": f"{int(timestamp_s//60):02d}:{int(timestamp_s%60):02d}",
                            "frame": frame_count,
                            "track_id": a["track_id"],
                            "risk": round(a["risk"], 3),
                            "model": model_name,
                            "alert_type": alert_t,
                            "features": a.get("features", {}),
                        }
                        self.alerts.append(entry)
                        prefix = "[HR ALERT]" if alert_t == "hr" else "[ALERT]"
                        self.log(f"[{entry['time']}] {prefix} dog#{a['track_id']} "
                                 f"risk={a['risk']:.2f}  frame {frame_count}")
                    if self.alert_sound_on.get():
                        self._play_alert_sound(alert_t)
                    if alert_t == "hr":
                        self.root.after(0, self._flash_alert)

                if writer:
                    writer.write(annotated)

                processed += 1
                elapsed = time.time() - start_time
                current_fps = processed / elapsed if elapsed > 0 else 0

                self.stats_vars["frames"].set(f"{frame_count:,}")
                self.stats_vars["persons"].set(f"{cur_persons}")
                self.stats_vars["dogs"].set(f"{cur_dogs}")
                self.stats_vars["alerts"].set(f"{total_alerts:,}")
                self.stats_vars["fps"].set(f"{current_fps:.1f}")
                if total_frames > 0:
                    self.progress["value"] = (frame_count / total_frames) * 100
                self.root.after(0, self._display_frame, annotated)

            cap.release()
            if writer:
                writer.release()

            # save the session for the dashboard
            if self._recorder:
                saved = self._recorder.finalize(save=True)
                if saved and saved.get("_path"):
                    self.log(f"[analytics] Session saved: {saved['_path']}")

            total_time = time.time() - start_time
            if self.stop_requested:
                self._update_status(f"Stopped after {processed:,} frames")
            else:
                self._update_status(
                    f"Complete - {processed:,} frames in {total_time:.1f}s "
                    f"({total_alerts} alerts)")

            if output_path:
                self.log(f"Output saved: {output_path}")

            if not is_live:
                title = "Analysis Complete"
                model_line = f"Model: {model_name}  |  Alert: {alert_t.upper()}\n\n"
                if self.alerts:
                    self.export_btn.config(state="normal")
                    messagebox.showinfo(title, model_line +
                                        f"Found {len(self.alerts)} alert(s)!\n"
                                        f"Frames: {processed:,}  |  "
                                        f"Peak dogs in frame: {peak_dogs}")
                else:
                    messagebox.showinfo(title, model_line +
                                        f"No aggression risk detected.\n"
                                        f"Frames: {processed:,}  |  "
                                        f"Peak dogs in frame: {peak_dogs}")

        except Exception as e:
            self.log(f"ERROR: {e}")
            self.model_status.set("Error")
            messagebox.showerror("Error", f"Detection failed:\n{e}")
        finally:
            try:
                del pipeline
            except NameError:
                pass
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass

            self._recorder = None
            self.is_processing = False
            self.model_status.set("Idle")
            self.progress["value"] = 0
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")
            self.forward_btn.config(state="disabled")
            self.root.after(0, self._clear_display)


def main():
    root = tk.Tk()
    DogAggressionAppV3(root)
    root.mainloop()


if __name__ == "__main__":
    main()
