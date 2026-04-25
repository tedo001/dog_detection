"""
Dog Aggression Detection - Experimental GUI (Model Switcher)

Same as app.py but lets you pick any YOLO model from a dropdown
to compare detection performance across different versions.

Run with:
    python app2.py
"""

import time
import threading
import json
from pathlib import Path
from datetime import datetime

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk


YOLO_MODELS = [
    # YOLO26 - latest (2026)
    "yolo26n.pt",
    "yolo26s.pt",
    "yolo26m.pt",
    "yolo26l.pt",
    "yolo26x.pt",
    # YOLO11 - previous stable
    "yolo11n.pt",
    "yolo11s.pt",
    "yolo11m.pt",
    "yolo11l.pt",
    "yolo11x.pt",
    # older versions
    "yolo12m.pt",
    "yolov8n.pt",
    "yolov8m.pt",
    "yolov9c.pt",
    "yolov10m.pt",
]


class DogAggressionAppV2:
    """Experimental GUI with model selector."""

    def __init__(self, root):
        self.root = root
        self.root.title("Dog Aggression Detection - Experiment (Model Switcher)")
        self.root.geometry("1280x900")
        self.root.configure(bg="#1e1e1e")

        self.video_path = None
        self.is_processing = False
        self.stop_requested = False
        self.alerts = []

        self.detector_path = tk.StringVar(value="yolo26n.pt")
        self.pose_enabled = tk.BooleanVar(value=True)
        self.det_conf = tk.DoubleVar(value=0.35)
        self.risk_threshold = tk.DoubleVar(value=0.35)
        self.sustain_frames = tk.IntVar(value=2)
        self.skip_frames = tk.IntVar(value=1)
        self.save_output = tk.BooleanVar(value=True)

        self._build_ui()

    def _build_ui(self):
        # Title bar
        title_frame = tk.Frame(self.root, bg="#2d2d2d", height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="Dog Aggression Detection  [EXPERIMENT]",
            font=("Segoe UI", 16, "bold"),
            bg="#2d2d2d", fg="#f59e0b",
        ).pack(side="left", padx=20, pady=15)

        tk.Label(
            title_frame,
            text="YOLO26 + Model Switcher",
            font=("Segoe UI", 10),
            bg="#2d2d2d", fg="#888888",
        ).pack(side="left", pady=15)

        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        left_frame = tk.Frame(main_frame, bg="#252525", width=370)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        left_frame.pack_propagate(False)

        right_frame = tk.Frame(main_frame, bg="#1e1e1e")
        right_frame.pack(side="right", fill="both", expand=True)

        self._build_controls(left_frame)
        self._build_display(right_frame)

    def _build_controls(self, parent):
        self._section_header(parent, "1. Select Video")

        self.path_label = tk.Label(
            parent, text="No video selected",
            bg="#252525", fg="#888888",
            font=("Segoe UI", 9), wraplength=330,
            justify="left", anchor="w",
        )
        self.path_label.pack(fill="x", padx=15, pady=(0, 8))

        btn_frame = tk.Frame(parent, bg="#252525")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        tk.Button(
            btn_frame, text="Browse...", command=self.browse_video,
            bg="#0078d4", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=15, pady=5, cursor="hand2",
        ).pack(side="left", padx=(0, 5))

        tk.Button(
            btn_frame, text="Enter Path", command=self.enter_path,
            bg="#3a3a3a", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=15, pady=5, cursor="hand2",
        ).pack(side="left")

        # Model selector dropdown
        self._section_header(parent, "2. YOLO Model")

        tk.Label(parent, text="Select model to test:",
                 bg="#252525", fg="#cccccc", font=("Segoe UI", 9)
                 ).pack(anchor="w", padx=15)

        model_combo = ttk.Combobox(
            parent, textvariable=self.detector_path,
            values=YOLO_MODELS, state="normal",
            font=("Segoe UI", 9),
        )
        model_combo.pack(fill="x", padx=15, pady=(0, 4))

        tk.Label(
            parent,
            text="Or type any .pt path manually above",
            bg="#252525", fg="#666666", font=("Segoe UI", 8),
        ).pack(anchor="w", padx=15, pady=(0, 8))

        tk.Checkbutton(
            parent, text="Use pose model (human skeleton)",
            variable=self.pose_enabled,
            bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
            activebackground="#252525", activeforeground="white",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=15, pady=(0, 15))

        # Settings
        self._section_header(parent, "3. Settings")

        self._slider(parent, "Detection confidence", self.det_conf, 0.1, 0.95, 0.05)
        self._slider(parent, "Risk threshold", self.risk_threshold, 0.1, 0.95, 0.05)
        self._slider(parent, "Sustain frames (N)", self.sustain_frames, 1, 20, 1, is_int=True)
        self._slider(parent, "Skip frames", self.skip_frames, 1, 30, 1, is_int=True)

        tk.Checkbutton(
            parent, text="Save annotated output video",
            variable=self.save_output,
            bg="#252525", fg="#cccccc", selectcolor="#1e1e1e",
            activebackground="#252525", activeforeground="white",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=15, pady=(5, 15))

        self._section_header(parent, "4. Run")

        self.start_btn = tk.Button(
            parent, text="Start Detection", command=self.start_detection,
            bg="#16a34a", fg="white", font=("Segoe UI", 11, "bold"),
            relief="flat", padx=15, pady=10, cursor="hand2",
        )
        self.start_btn.pack(fill="x", padx=15, pady=(0, 5))

        self.stop_btn = tk.Button(
            parent, text="Stop", command=self.stop_detection,
            bg="#dc2626", fg="white", font=("Segoe UI", 10),
            relief="flat", padx=15, pady=5, state="disabled", cursor="hand2",
        )
        self.stop_btn.pack(fill="x", padx=15, pady=(0, 10))

        self.status_label = tk.Label(
            parent, text="Ready",
            bg="#252525", fg="#888888",
            font=("Segoe UI", 9), anchor="w",
        )
        self.status_label.pack(fill="x", padx=15, pady=5)

        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", padx=15, pady=5)

    def _build_display(self, parent):
        video_frame = tk.Frame(parent, bg="#000000", relief="solid", bd=1)
        video_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.video_label = tk.Label(
            video_frame, text="Video preview will appear here",
            bg="#000000", fg="#555555", font=("Segoe UI", 12),
        )
        self.video_label.pack(fill="both", expand=True)

        stats_frame = tk.Frame(parent, bg="#252525", height=80)
        stats_frame.pack(fill="x", pady=(0, 10))
        stats_frame.pack_propagate(False)

        self.stats_vars = {
            "frames": tk.StringVar(value="0"),
            "persons": tk.StringVar(value="0"),
            "dogs": tk.StringVar(value="0"),
            "alerts": tk.StringVar(value="0"),
            "fps": tk.StringVar(value="0.0"),
            "model": tk.StringVar(value="-"),
        }

        for key, label in [
            ("frames", "Frames"), ("persons", "Persons"),
            ("dogs", "Dogs"), ("alerts", "ALERTS"), ("fps", "FPS"),
        ]:
            col = tk.Frame(stats_frame, bg="#252525")
            col.pack(side="left", fill="both", expand=True, padx=5, pady=10)
            color = "#dc2626" if key == "alerts" else "#22c55e" if key == "fps" else "#ffffff"
            tk.Label(col, textvariable=self.stats_vars[key],
                     bg="#252525", fg=color,
                     font=("Segoe UI", 18, "bold")).pack()
            tk.Label(col, text=label, bg="#252525", fg="#888888",
                     font=("Segoe UI", 9)).pack()

        # Active model badge
        model_frame = tk.Frame(parent, bg="#1a1a1a", pady=4)
        model_frame.pack(fill="x", pady=(0, 5))
        tk.Label(model_frame, text="Active model: ", bg="#1a1a1a",
                 fg="#888888", font=("Segoe UI", 9)).pack(side="left", padx=10)
        tk.Label(model_frame, textvariable=self.stats_vars["model"],
                 bg="#1a1a1a", fg="#f59e0b",
                 font=("Segoe UI", 9, "bold")).pack(side="left")

        log_frame = tk.LabelFrame(
            parent, text="  Alert Log  ",
            bg="#252525", fg="#ffffff", font=("Segoe UI", 10, "bold"),
        )
        log_frame.pack(fill="x")

        self.log_text = scrolledtext.ScrolledText(
            log_frame, height=7,
            bg="#1a1a1a", fg="#ffffff",
            insertbackground="white",
            font=("Consolas", 9), relief="flat", wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        self.export_btn = tk.Button(
            log_frame, text="Export Alerts (JSON)",
            command=self.export_alerts,
            bg="#3a3a3a", fg="white", font=("Segoe UI", 9),
            relief="flat", padx=10, pady=3, state="disabled",
        )
        self.export_btn.pack(side="right", padx=5, pady=(0, 5))

    def _section_header(self, parent, text):
        tk.Label(
            parent, text=text,
            bg="#252525", fg="#f59e0b",
            font=("Segoe UI", 11, "bold"), anchor="w",
        ).pack(fill="x", padx=15, pady=(10, 8))

    def _slider(self, parent, label, variable, from_, to, resolution, is_int=False):
        container = tk.Frame(parent, bg="#252525")
        container.pack(fill="x", padx=15, pady=(3, 8))
        header = tk.Frame(container, bg="#252525")
        header.pack(fill="x")
        tk.Label(header, text=label, bg="#252525", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side="left")
        value_label = tk.Label(header, text="", bg="#252525", fg="#f59e0b",
                               font=("Segoe UI", 9, "bold"))
        value_label.pack(side="right")

        def update_label(*_):
            val = variable.get()
            value_label.config(text=f"{int(val) if is_int else round(val, 2)}")
        variable.trace_add("write", update_label)
        update_label()

        slider = ttk.Scale(container, from_=from_, to=to,
                           variable=variable, orient="horizontal")
        slider.pack(fill="x")
        if is_int:
            slider.config(command=lambda v: variable.set(int(float(v))))

    def browse_video(self):
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                       ("All files", "*.*")],
        )
        if path:
            self._set_video(path)

    def enter_path(self):
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Video Path")
        dialog.geometry("500x120")
        dialog.configure(bg="#252525")
        dialog.transient(self.root)
        dialog.grab_set()
        tk.Label(dialog, text="Enter full local video path:",
                 bg="#252525", fg="#ffffff", font=("Segoe UI", 10)
                 ).pack(pady=(15, 5))
        path_var = tk.StringVar()
        entry = tk.Entry(dialog, textvariable=path_var, width=60,
                         bg="#1e1e1e", fg="#ffffff", insertbackground="white",
                         font=("Segoe UI", 10))
        entry.pack(padx=15, pady=5, fill="x")
        entry.focus()

        def submit():
            p = path_var.get().strip().strip('"')
            if p and Path(p).exists():
                self._set_video(p)
                dialog.destroy()
            else:
                messagebox.showerror("Error", f"File not found:\n{p}")

        tk.Button(dialog, text="Load", command=submit,
                  bg="#0078d4", fg="white", relief="flat",
                  font=("Segoe UI", 10), padx=20, pady=5).pack(pady=10)
        entry.bind("<Return>", lambda e: submit())

    def _set_video(self, path):
        self.video_path = path
        self.path_label.config(text=path, fg="#22c55e")
        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            self._display_frame(frame)
        cap.release()
        self.log(f"Video loaded: {path}")

    def _display_frame(self, frame):
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 800, 500
        h, w = rgb.shape[:2]
        scale = min(label_w / w, label_h / h)
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)))
        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text="")

    def start_detection(self):
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return
        self.is_processing = True
        self.stop_requested = False
        self.alerts = []
        self.log_text.delete(1.0, "end")
        self.start_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.export_btn.config(state="disabled")
        for var in self.stats_vars.values():
            var.set("0")
        threading.Thread(target=self._detection_worker, daemon=True).start()

    def stop_detection(self):
        self.stop_requested = True
        self.log("Stop requested...")

    def _detection_worker(self):
        try:
            model_name = self.detector_path.get()
            self._update_status(f"Loading {model_name}...")
            self.stats_vars["model"].set(model_name)

            from src.inference.predict import DogAggressionPipeline

            pose_model = "yolo11m-pose.pt" if self.pose_enabled.get() else None
            pipeline = DogAggressionPipeline(
                detector_path=model_name,
                pose_model=pose_model,
                det_conf=self.det_conf.get(),
                risk_threshold=self.risk_threshold.get(),
                sustain_frames=self.sustain_frames.get(),
            )

            self._update_status("Processing video...")
            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            writer = None
            output_path = None
            if self.save_output.get():
                Path("experiments").mkdir(exist_ok=True)
                output_path = (
                    f"experiments/exp_{model_name.replace('.pt','')}_"
                    f"{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            skip = self.skip_frames.get()
            frame_count = 0
            processed = 0
            total_persons = 0
            total_dogs = 0
            total_alerts = 0
            start_time = time.time()

            while not self.stop_requested:
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

                total_persons += len(pipeline._last_persons)
                total_dogs += len(results)

                new_alerts_now = [r for r in results if r.get("new_alert")]
                total_alerts += len(new_alerts_now)

                if new_alerts_now:
                    timestamp_s = frame_count / fps
                    for a in new_alerts_now:
                        f = a.get("features", {})
                        entry = {
                            "time": f"{int(timestamp_s//60):02d}:{int(timestamp_s%60):02d}",
                            "frame": frame_count,
                            "track_id": a["track_id"],
                            "risk": round(a["risk"], 3),
                            "model": model_name,
                            "features": f,
                        }
                        self.alerts.append(entry)
                        self.log(
                            f"[{entry['time']}] ALERT dog#{a['track_id']} "
                            f"risk={a['risk']:.2f} model={model_name} frame {frame_count}"
                        )
                    self.root.after(0, self.root.bell)

                if writer:
                    writer.write(annotated)

                processed += 1
                elapsed = time.time() - start_time
                current_fps = processed / elapsed if elapsed > 0 else 0

                self.stats_vars["frames"].set(f"{frame_count:,}")
                self.stats_vars["persons"].set(f"{total_persons:,}")
                self.stats_vars["dogs"].set(f"{total_dogs:,}")
                self.stats_vars["alerts"].set(f"{total_alerts:,}")
                self.stats_vars["fps"].set(f"{current_fps:.1f}")
                self.progress["value"] = (frame_count / total_frames) * 100
                self.root.after(0, self._display_frame, annotated)

            cap.release()
            if writer:
                writer.release()

            total_time = time.time() - start_time
            if self.stop_requested:
                self._update_status(f"Stopped after {processed:,} frames")
            else:
                self._update_status(
                    f"Complete — {processed:,} frames in {total_time:.1f}s "
                    f"({total_alerts} alerts) [{model_name}]"
                )

            if output_path:
                self.log(f"Output saved: {output_path}")

            if self.alerts:
                self.export_btn.config(state="normal")
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Model: {model_name}\n\n"
                    f"Found {len(self.alerts)} alert(s)!\n"
                    f"Frames processed: {processed:,}\n"
                    f"Dogs detected: {total_dogs:,}",
                )
            else:
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Model: {model_name}\n\n"
                    f"No aggression risk detected.\n"
                    f"Frames processed: {processed:,}\n"
                    f"Dogs detected: {total_dogs:,}",
                )

        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Detection failed:\n{e}")
        finally:
            self.is_processing = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def log(self, msg):
        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.root.after(0, _append)

    def _update_status(self, msg):
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def export_alerts(self):
        if not self.alerts:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".json",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            initialfile=f"alerts_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.alerts, f, indent=2)
            messagebox.showinfo("Exported", f"Alerts saved to:\n{path}")


def main():
    root = tk.Tk()
    app = DogAggressionAppV2(root)
    root.mainloop()


if __name__ == "__main__":
    main()
