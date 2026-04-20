"""
Dog Aggression Detection - Desktop GUI Application

Native Tkinter-based GUI app that:
- Browse/select a local video file
- Runs the full detection + classification pipeline
- Shows annotated video with bounding boxes in real-time
- Logs aggression alerts with timestamps
- Exports annotated output video

Run with:
    python app.py
"""

import os
import time
import threading
import json
from pathlib import Path
from datetime import datetime
from queue import Queue

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, scrolledtext
import cv2
from PIL import Image, ImageTk


class DogAggressionApp:
    """Main GUI application."""

    def __init__(self, root):
        self.root = root
        self.root.title("Dog Aggression Detection System")
        self.root.geometry("1280x820")
        self.root.configure(bg="#1e1e1e")

        # State
        self.video_path = None
        self.pipeline = None
        self.is_processing = False
        self.stop_requested = False
        self.alerts = []
        self.frame_queue = Queue(maxsize=10)

        # Default model paths
        self.detector_path = tk.StringVar(value=self._find_latest_model(
            "experiments/detector_*/weights/best.pt"
        ))
        self.classifier_path = tk.StringVar(value=self._find_latest_model(
            "experiments/classifier_*/checkpoints/best.pt"
        ))
        self.det_conf = tk.DoubleVar(value=0.5)
        self.aggression_threshold = tk.DoubleVar(value=0.7)
        self.skip_frames = tk.IntVar(value=1)
        self.save_output = tk.BooleanVar(value=True)

        self._build_ui()

    def _find_latest_model(self, pattern):
        """Find most recent trained model matching pattern."""
        candidates = list(Path(".").glob(pattern))
        if not candidates:
            return ""
        return str(max(candidates, key=lambda p: p.stat().st_mtime))

    def _build_ui(self):
        """Build the complete GUI layout."""

        # --- Top: Title Bar ---
        title_frame = tk.Frame(self.root, bg="#2d2d2d", height=60)
        title_frame.pack(fill="x")
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="Dog Aggression Detection System",
            font=("Segoe UI", 16, "bold"),
            bg="#2d2d2d",
            fg="#ffffff",
        ).pack(side="left", padx=20, pady=15)

        tk.Label(
            title_frame,
            text="YOLOv8 + ResNet Two-Stage Pipeline",
            font=("Segoe UI", 10),
            bg="#2d2d2d",
            fg="#888888",
        ).pack(side="left", pady=15)

        # --- Main Content: Two Columns ---
        main_frame = tk.Frame(self.root, bg="#1e1e1e")
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)

        # Left: Controls
        left_frame = tk.Frame(main_frame, bg="#252525", width=360)
        left_frame.pack(side="left", fill="y", padx=(0, 10))
        left_frame.pack_propagate(False)

        # Right: Video Display + Alerts
        right_frame = tk.Frame(main_frame, bg="#1e1e1e")
        right_frame.pack(side="right", fill="both", expand=True)

        self._build_controls(left_frame)
        self._build_display(right_frame)

    def _build_controls(self, parent):
        """Left-side control panel."""

        # Video Selection
        self._section_header(parent, "1. Select Video")

        self.path_label = tk.Label(
            parent,
            text="No video selected",
            bg="#252525",
            fg="#888888",
            font=("Segoe UI", 9),
            wraplength=320,
            justify="left",
            anchor="w",
        )
        self.path_label.pack(fill="x", padx=15, pady=(0, 8))

        btn_frame = tk.Frame(parent, bg="#252525")
        btn_frame.pack(fill="x", padx=15, pady=(0, 15))

        tk.Button(
            btn_frame,
            text="Browse...",
            command=self.browse_video,
            bg="#0078d4",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            padx=15,
            pady=5,
            cursor="hand2",
        ).pack(side="left", padx=(0, 5))

        tk.Button(
            btn_frame,
            text="Enter Path",
            command=self.enter_path,
            bg="#3a3a3a",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            padx=15,
            pady=5,
            cursor="hand2",
        ).pack(side="left")

        # Model Paths
        self._section_header(parent, "2. Model Weights")

        tk.Label(parent, text="Detector (.pt):", bg="#252525", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=15)
        det_entry = tk.Entry(
            parent, textvariable=self.detector_path,
            bg="#1e1e1e", fg="#ffffff", insertbackground="white",
            relief="flat", font=("Segoe UI", 9),
        )
        det_entry.pack(fill="x", padx=15, pady=(0, 8))

        tk.Label(parent, text="Classifier (.pt):", bg="#252525", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(anchor="w", padx=15)
        cls_entry = tk.Entry(
            parent, textvariable=self.classifier_path,
            bg="#1e1e1e", fg="#ffffff", insertbackground="white",
            relief="flat", font=("Segoe UI", 9),
        )
        cls_entry.pack(fill="x", padx=15, pady=(0, 15))

        # Settings
        self._section_header(parent, "3. Settings")

        self._slider(parent, "Dog detection confidence", self.det_conf, 0.1, 0.95, 0.05)
        self._slider(parent, "Aggression threshold", self.aggression_threshold, 0.3, 0.95, 0.05)
        self._slider(parent, "Skip frames", self.skip_frames, 1, 30, 1, is_int=True)

        tk.Checkbutton(
            parent,
            text="Save annotated output video",
            variable=self.save_output,
            bg="#252525",
            fg="#cccccc",
            selectcolor="#1e1e1e",
            activebackground="#252525",
            activeforeground="white",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=15, pady=(5, 15))

        # Action Buttons
        self._section_header(parent, "4. Run")

        self.start_btn = tk.Button(
            parent,
            text="Start Detection",
            command=self.start_detection,
            bg="#16a34a",
            fg="white",
            font=("Segoe UI", 11, "bold"),
            relief="flat",
            padx=15,
            pady=10,
            cursor="hand2",
        )
        self.start_btn.pack(fill="x", padx=15, pady=(0, 5))

        self.stop_btn = tk.Button(
            parent,
            text="Stop",
            command=self.stop_detection,
            bg="#dc2626",
            fg="white",
            font=("Segoe UI", 10),
            relief="flat",
            padx=15,
            pady=5,
            state="disabled",
            cursor="hand2",
        )
        self.stop_btn.pack(fill="x", padx=15, pady=(0, 10))

        # Status
        self.status_label = tk.Label(
            parent,
            text="Ready",
            bg="#252525",
            fg="#888888",
            font=("Segoe UI", 9),
            anchor="w",
        )
        self.status_label.pack(fill="x", padx=15, pady=5)

        # Progress bar
        self.progress = ttk.Progressbar(parent, mode="determinate")
        self.progress.pack(fill="x", padx=15, pady=5)

    def _build_display(self, parent):
        """Right-side video display and alerts."""

        # Video Display
        video_frame = tk.Frame(parent, bg="#000000", relief="solid", bd=1)
        video_frame.pack(fill="both", expand=True, pady=(0, 10))

        self.video_label = tk.Label(
            video_frame,
            text="Video preview will appear here",
            bg="#000000",
            fg="#555555",
            font=("Segoe UI", 12),
        )
        self.video_label.pack(fill="both", expand=True)

        # Stats Bar
        stats_frame = tk.Frame(parent, bg="#252525", height=80)
        stats_frame.pack(fill="x", pady=(0, 10))
        stats_frame.pack_propagate(False)

        self.stats_vars = {
            "frames": tk.StringVar(value="0"),
            "dogs": tk.StringVar(value="0"),
            "alerts": tk.StringVar(value="0"),
            "fps": tk.StringVar(value="0.0"),
        }

        for i, (key, label) in enumerate([
            ("frames", "Frames"),
            ("dogs", "Dogs Detected"),
            ("alerts", "AGGRESSIVE"),
            ("fps", "FPS"),
        ]):
            col = tk.Frame(stats_frame, bg="#252525")
            col.pack(side="left", fill="both", expand=True, padx=5, pady=10)

            color = "#dc2626" if key == "alerts" else "#22c55e" if key == "fps" else "#ffffff"
            tk.Label(
                col, textvariable=self.stats_vars[key],
                bg="#252525", fg=color,
                font=("Segoe UI", 20, "bold"),
            ).pack()
            tk.Label(
                col, text=label,
                bg="#252525", fg="#888888",
                font=("Segoe UI", 9),
            ).pack()

        # Alerts Log
        log_frame = tk.LabelFrame(
            parent,
            text="  Alert Log  ",
            bg="#252525",
            fg="#ffffff",
            font=("Segoe UI", 10, "bold"),
        )
        log_frame.pack(fill="x")

        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=8,
            bg="#1a1a1a",
            fg="#ffffff",
            insertbackground="white",
            font=("Consolas", 9),
            relief="flat",
            wrap="word",
        )
        self.log_text.pack(fill="both", expand=True, padx=5, pady=5)

        # Export button
        self.export_btn = tk.Button(
            log_frame,
            text="Export Alerts (JSON)",
            command=self.export_alerts,
            bg="#3a3a3a",
            fg="white",
            font=("Segoe UI", 9),
            relief="flat",
            padx=10,
            pady=3,
            state="disabled",
        )
        self.export_btn.pack(side="right", padx=5, pady=(0, 5))

    def _section_header(self, parent, text):
        """Create a styled section header."""
        header = tk.Label(
            parent,
            text=text,
            bg="#252525",
            fg="#0078d4",
            font=("Segoe UI", 11, "bold"),
            anchor="w",
        )
        header.pack(fill="x", padx=15, pady=(10, 8))

    def _slider(self, parent, label, variable, from_, to, resolution, is_int=False):
        """Create a slider control with label."""
        container = tk.Frame(parent, bg="#252525")
        container.pack(fill="x", padx=15, pady=(3, 8))

        header = tk.Frame(container, bg="#252525")
        header.pack(fill="x")

        tk.Label(header, text=label, bg="#252525", fg="#cccccc",
                 font=("Segoe UI", 9)).pack(side="left")

        value_label = tk.Label(header, text="", bg="#252525", fg="#0078d4",
                               font=("Segoe UI", 9, "bold"))
        value_label.pack(side="right")

        def update_label(*_):
            val = variable.get()
            value_label.config(text=f"{int(val) if is_int else round(val, 2)}")
        variable.trace_add("write", update_label)
        update_label()

        slider = ttk.Scale(
            container, from_=from_, to=to,
            variable=variable, orient="horizontal",
        )
        slider.pack(fill="x")
        if is_int:
            slider.config(command=lambda v: variable.set(int(float(v))))

    def browse_video(self):
        """Open file dialog to select a video."""
        path = filedialog.askopenfilename(
            title="Select Video",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.wmv *.flv"),
                ("All files", "*.*"),
            ],
        )
        if path:
            self._set_video(path)

    def enter_path(self):
        """Open a dialog to enter a local path manually."""
        dialog = tk.Toplevel(self.root)
        dialog.title("Enter Video Path")
        dialog.geometry("500x120")
        dialog.configure(bg="#252525")
        dialog.transient(self.root)
        dialog.grab_set()

        tk.Label(
            dialog, text="Enter full local video path:",
            bg="#252525", fg="#ffffff",
            font=("Segoe UI", 10),
        ).pack(pady=(15, 5))

        path_var = tk.StringVar()
        entry = tk.Entry(
            dialog, textvariable=path_var, width=60,
            bg="#1e1e1e", fg="#ffffff", insertbackground="white",
            font=("Segoe UI", 10),
        )
        entry.pack(padx=15, pady=5, fill="x")
        entry.focus()

        def submit():
            p = path_var.get().strip().strip('"')
            if p and Path(p).exists():
                self._set_video(p)
                dialog.destroy()
            else:
                messagebox.showerror("Error", f"File not found:\n{p}")

        tk.Button(
            dialog, text="Load", command=submit,
            bg="#0078d4", fg="white", relief="flat",
            font=("Segoe UI", 10), padx=20, pady=5,
        ).pack(pady=10)

        entry.bind("<Return>", lambda e: submit())

    def _set_video(self, path):
        """Set the current video and show info."""
        self.video_path = path
        self.path_label.config(text=path, fg="#22c55e")

        cap = cv2.VideoCapture(path)
        ret, frame = cap.read()
        if ret:
            self._display_frame(frame)
        cap.release()

        self.log(f"Video loaded: {path}")

    def _display_frame(self, frame):
        """Display a frame in the video label."""
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        label_w = self.video_label.winfo_width()
        label_h = self.video_label.winfo_height()
        if label_w < 10 or label_h < 10:
            label_w, label_h = 800, 500

        h, w = rgb.shape[:2]
        scale = min(label_w / w, label_h / h)
        new_w, new_h = int(w * scale), int(h * scale)
        rgb = cv2.resize(rgb, (new_w, new_h))

        img = Image.fromarray(rgb)
        imgtk = ImageTk.PhotoImage(image=img)
        self.video_label.imgtk = imgtk
        self.video_label.config(image=imgtk, text="")

    def start_detection(self):
        """Start the detection processing in a background thread."""
        if not self.video_path:
            messagebox.showwarning("No Video", "Please select a video first.")
            return

        if not Path(self.detector_path.get()).exists():
            messagebox.showerror(
                "Missing Model",
                f"Detector weights not found:\n{self.detector_path.get()}\n\n"
                "Train the model first:\n  python train.py --stage detector"
            )
            return

        if not Path(self.classifier_path.get()).exists():
            messagebox.showerror(
                "Missing Model",
                f"Classifier weights not found:\n{self.classifier_path.get()}\n\n"
                "Train the model first:\n  python train.py --stage classifier"
            )
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

        thread = threading.Thread(target=self._detection_worker, daemon=True)
        thread.start()

    def stop_detection(self):
        """Request the detection thread to stop."""
        self.stop_requested = True
        self.log("Stop requested...")

    def _detection_worker(self):
        """Background thread that processes the video."""
        try:
            self._update_status("Loading models...")

            from src.inference.predict import DogAggressionPipeline
            pipeline = DogAggressionPipeline(
                detector_path=self.detector_path.get(),
                classifier_path=self.classifier_path.get(),
                det_conf=self.det_conf.get(),
                aggression_threshold=self.aggression_threshold.get(),
            )

            self._update_status("Processing video...")

            cap = cv2.VideoCapture(self.video_path)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

            # Output writer
            writer = None
            output_path = None
            if self.save_output.get():
                Path("experiments").mkdir(exist_ok=True)
                output_path = (
                    f"experiments/output_{datetime.now().strftime('%Y%m%d_%H%M%S')}.mp4"
                )
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

            skip = self.skip_frames.get()
            frame_count = 0
            processed = 0
            total_dogs = 0
            total_aggressive = 0
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

                total_dogs += len(results)
                alerts_now = [r for r in results if r["alert"]]
                total_aggressive += len(alerts_now)

                if alerts_now:
                    timestamp_s = frame_count / fps
                    for a in alerts_now:
                        entry = {
                            "time": f"{int(timestamp_s//60):02d}:{int(timestamp_s%60):02d}",
                            "frame": frame_count,
                            "confidence": round(a["behavior_confidence"], 3),
                        }
                        self.alerts.append(entry)
                        self.log(
                            f"[{entry['time']}] ALERT — Aggressive dog "
                            f"(conf: {entry['confidence']:.2%}, frame {frame_count})"
                        )

                if writer:
                    writer.write(annotated)

                processed += 1
                elapsed = time.time() - start_time
                current_fps = processed / elapsed if elapsed > 0 else 0

                # Update UI every 3 frames
                if processed % 3 == 0:
                    self.stats_vars["frames"].set(f"{frame_count:,}")
                    self.stats_vars["dogs"].set(f"{total_dogs:,}")
                    self.stats_vars["alerts"].set(f"{total_aggressive:,}")
                    self.stats_vars["fps"].set(f"{current_fps:.1f}")

                    progress_val = (frame_count / total_frames) * 100
                    self.progress["value"] = progress_val

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
                    f"({total_aggressive} alerts)"
                )

            if output_path:
                self.log(f"Output saved: {output_path}")

            if self.alerts:
                self.export_btn.config(state="normal")
                messagebox.showinfo(
                    "Analysis Complete",
                    f"Found {len(self.alerts)} aggressive detection(s)!\n\n"
                    f"Frames processed: {processed:,}\n"
                    f"Total dogs: {total_dogs:,}\n"
                    f"Alerts: {total_aggressive:,}"
                )
            else:
                messagebox.showinfo(
                    "Analysis Complete",
                    f"No aggressive behavior detected.\n\n"
                    f"Frames processed: {processed:,}\n"
                    f"Dogs detected: {total_dogs:,}"
                )

        except Exception as e:
            self.log(f"ERROR: {e}")
            messagebox.showerror("Error", f"Detection failed:\n{e}")

        finally:
            self.is_processing = False
            self.start_btn.config(state="normal")
            self.stop_btn.config(state="disabled")

    def log(self, msg):
        """Append a message to the log text area."""
        def _append():
            self.log_text.insert("end", msg + "\n")
            self.log_text.see("end")
        self.root.after(0, _append)

    def _update_status(self, msg):
        """Update the status label."""
        self.root.after(0, lambda: self.status_label.config(text=msg))

    def export_alerts(self):
        """Export alert log to JSON file."""
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
    app = DogAggressionApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()
