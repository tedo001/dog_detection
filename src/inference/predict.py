"""
End-to-End Inference Pipeline - geometric risk rules.

Pipeline:
    1. Stock YOLO11 detects persons + dogs (COCO classes 0 and 16).
    2. Each dog is tracked across frames by greedy IoU matching.
    3. Per-dog risk is computed from interpretable geometric features:
         - proximity to nearest person
         - velocity component toward nearest person
         - aspect-ratio change (proxy for lunging / posture change)
         - multi-dog context
    4. Risk > threshold -> alert.

No behavior classifier needed. No custom training needed.
"""

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.models.detector import DogDetector


# -------------------- tracker --------------------

def _iou(a, b):
    ax1, ay1, ax2, ay2 = a["x1"], a["y1"], a["x2"], a["y2"]
    bx1, by1, bx2, by2 = b["x1"], b["y1"], b["x2"], b["y2"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0, ix2 - ix1), max(0, iy2 - iy1)
    inter = iw * ih
    if inter == 0:
        return 0.0
    area_a = (ax2 - ax1) * (ay2 - ay1)
    area_b = (bx2 - bx1) * (by2 - by1)
    return inter / (area_a + area_b - inter + 1e-6)


def _box_center(b):
    return ((b["x1"] + b["x2"]) / 2.0, (b["y1"] + b["y2"]) / 2.0)


def _box_aspect_ratio(b):
    w = max(1, b["x2"] - b["x1"])
    h = max(1, b["y2"] - b["y1"])
    return w / h


class Track:
    """State for one tracked dog."""

    _next_id = 1

    def __init__(self, box, history_len=15):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.history = deque([box], maxlen=history_len)
        self.missed = 0
        self.risk = 0.0

    def update(self, box):
        self.box = box
        self.history.append(box)
        self.missed = 0

    def mark_missed(self):
        self.missed += 1


class IoUTracker:
    """Greedy IoU tracker - good enough for dog-scale object counts."""

    def __init__(self, iou_threshold=0.2, max_missed=15):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections):
        unmatched_dets = list(range(len(detections)))
        unmatched_tracks = list(range(len(self.tracks)))

        # Score every (track, det) pair and greedily match the best
        pairs = []
        for ti, track in enumerate(self.tracks):
            for di, det in enumerate(detections):
                pairs.append((_iou(track.box, det), ti, di))
        pairs.sort(reverse=True)

        used_t, used_d = set(), set()
        for iou, ti, di in pairs:
            if iou < self.iou_threshold:
                break
            if ti in used_t or di in used_d:
                continue
            self.tracks[ti].update(detections[di])
            used_t.add(ti)
            used_d.add(di)

        # Unmatched dets -> new tracks
        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(det))

        # Unmatched tracks -> mark missed, drop stale
        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.mark_missed()
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return self.tracks


# -------------------- risk scoring --------------------

def _compute_risk(track, persons, frame_shape, num_dogs):
    """
    Interpretable risk score in [0, 1] from geometric features.

    Features:
        - distance_risk   : closer dog-to-person -> higher
        - velocity_risk   : dog moving toward person -> higher
        - posture_risk    : aspect-ratio change (lunging) -> higher
        - pack_bonus      : more than one dog -> slight increase
    """
    if not persons:
        return 0.0, {"distance": 0, "velocity": 0, "posture": 0}

    h, w = frame_shape[:2]
    diag = math.hypot(w, h)

    dog_cx, dog_cy = _box_center(track.box)
    nearest = min(
        persons,
        key=lambda p: math.hypot(
            _box_center(p)[0] - dog_cx, _box_center(p)[1] - dog_cy
        ),
    )
    p_cx, p_cy = _box_center(nearest)

    # 1. Distance: <15% of frame diagonal = full risk
    distance = math.hypot(dog_cx - p_cx, dog_cy - p_cy) / diag
    distance_risk = max(0.0, 1.0 - distance / 0.15)

    # 2. Velocity toward person over last ~5 frames
    velocity_risk = 0.0
    if len(track.history) >= 5:
        old_cx, old_cy = _box_center(track.history[-5])
        dx, dy = dog_cx - old_cx, dog_cy - old_cy
        # Unit vector from old dog pos toward person
        pdx, pdy = p_cx - old_cx, p_cy - old_cy
        p_len = math.hypot(pdx, pdy) + 1e-6
        pdx, pdy = pdx / p_len, pdy / p_len
        # Component of motion in person's direction, normalized by diag
        velocity_toward = (dx * pdx + dy * pdy) / diag
        velocity_risk = min(1.0, max(0.0, velocity_toward / 0.03))

    # 3. Aspect-ratio change (lunging / rearing changes silhouette)
    posture_risk = 0.0
    if len(track.history) >= 5:
        recent_ars = [_box_aspect_ratio(b) for b in list(track.history)[-5:]]
        ar_range = max(recent_ars) - min(recent_ars)
        posture_risk = min(1.0, ar_range / 0.6)

    # 4. Multi-dog context
    pack_bonus = 0.1 if num_dogs > 1 else 0.0

    risk = (
        0.50 * distance_risk
        + 0.30 * velocity_risk
        + 0.20 * posture_risk
        + pack_bonus
    )
    return min(1.0, risk), {
        "distance": round(distance_risk, 2),
        "velocity": round(velocity_risk, 2),
        "posture": round(posture_risk, 2),
    }


# -------------------- pipeline --------------------

class DogAggressionPipeline:
    """
    Detect persons + dogs, track each dog, score geometric risk per dog.

    Output per dog: {x1, y1, x2, y2, track_id, risk, behavior, alert, features}
    Output per person: drawn as context in draw_results.
    """

    COLORS = {
        "person": (255, 180, 60),      # blue-ish (BGR)
        "dog_safe": (0, 220, 90),      # green
        "dog_caution": (40, 200, 240), # yellow
        "dog_alert": (60, 60, 255),    # red
    }

    def __init__(self, detector_path="yolo11m.pt", classifier_path=None,
                 det_conf=0.35, risk_threshold=0.6,
                 aggression_threshold=None, crop_size=224, device="auto"):
        # aggression_threshold kept for backward-compat with callers
        if aggression_threshold is not None:
            risk_threshold = aggression_threshold

        if detector_path in (None, "", "None"):
            detector_path = "yolo11m.pt"

        self.detector = DogDetector(model_path=detector_path, conf=det_conf)
        self.risk_threshold = risk_threshold
        self.tracker = IoUTracker()
        self._last_persons = []

        print(f"[pipeline] detector={detector_path} "
              f"conf={det_conf} risk_threshold={risk_threshold}")

    # ---- per-frame ----

    def process_frame(self, frame):
        det = self.detector.detect(frame)
        persons, dogs = det["persons"], det["dogs"]
        self._last_persons = persons

        tracks = self.tracker.update(dogs)

        results = []
        for t in tracks:
            if t.missed > 0:
                continue  # drew the dog box once; skip ghosts
            risk, features = _compute_risk(t, persons, frame.shape, len(dogs))
            t.risk = risk
            alert = risk >= self.risk_threshold
            results.append({
                "x1": t.box["x1"], "y1": t.box["y1"],
                "x2": t.box["x2"], "y2": t.box["y2"],
                "track_id": t.id,
                "dog_confidence": t.box["confidence"],
                "behavior": "aggressive" if alert else "non_aggressive",
                "behavior_confidence": risk,
                "risk": risk,
                "features": features,
                "alert": alert,
            })
        return results

    # ---- drawing ----

    def _dog_color(self, risk):
        if risk >= self.risk_threshold:
            return self.COLORS["dog_alert"]
        if risk >= self.risk_threshold * 0.6:
            return self.COLORS["dog_caution"]
        return self.COLORS["dog_safe"]

    def _draw_box(self, img, x1, y1, x2, y2, color, thickness=2, label=None):
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        if label:
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 4, y1), color, -1)
            cv2.putText(img, label, (x1 + 2, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    def draw_results(self, frame, results):
        annotated = frame.copy()

        # Persons (context)
        for p in self._last_persons:
            self._draw_box(
                annotated, p["x1"], p["y1"], p["x2"], p["y2"],
                self.COLORS["person"], thickness=2,
                label=f"person {p['confidence']:.2f}",
            )

        # Dogs (risk-colored)
        for r in results:
            color = self._dog_color(r["risk"])
            thickness = 3 if r["alert"] else 2
            label = f"dog#{r['track_id']} risk {r['risk']:.2f}"
            if r["alert"]:
                label = "ALERT " + label
            self._draw_box(
                annotated, r["x1"], r["y1"], r["x2"], r["y2"],
                color, thickness=thickness, label=label,
            )

        return annotated

    # ---- file/video helpers ----

    def predict_image(self, image_path, save_path=None):
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Cannot read image: {image_path}")
        results = self.process_frame(frame)
        annotated = self.draw_results(frame, results)
        if save_path:
            cv2.imwrite(str(save_path), annotated)
            print(f"[pipeline] Saved to {save_path}")
        return results, annotated

    def predict_video(self, video_path, output_path=None):
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise FileNotFoundError(f"Cannot open video: {video_path}")

        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_count, alert_frames = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.process_frame(frame)
            annotated = self.draw_results(frame, results)
            if any(r["alert"] for r in results):
                alert_frames += 1
                print(f"  Frame {frame_count}: ALERT")
            if writer:
                writer.write(annotated)
            frame_count += 1

        cap.release()
        if writer:
            writer.release()
        print(f"[pipeline] {frame_count} frames, {alert_frames} alert frames")

    def predict_webcam(self):
        cap = cv2.VideoCapture(0)
        print("[pipeline] Press 'q' to quit webcam")
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.process_frame(frame)
            annotated = self.draw_results(frame, results)
            cv2.imshow("Dog Aggression Detection", annotated)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--source", type=str, required=True)
    p.add_argument("--detector", type=str, default="yolo11m.pt")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--risk", type=float, default=0.6)
    args = p.parse_args()

    pipe = DogAggressionPipeline(
        detector_path=args.detector,
        det_conf=args.conf,
        risk_threshold=args.risk,
    )
    if args.source == "webcam":
        pipe.predict_webcam()
    elif args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        pipe.predict_video(args.source, output_path=args.output)
    else:
        pipe.predict_image(args.source, save_path=args.output)
