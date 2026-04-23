"""
End-to-End Inference Pipeline - geometric risk + temporal intelligence.

Pipeline:
    1. YOLO11 detects persons + dogs. YOLO11-Pose adds human keypoints.
    2. IoU tracker follows each dog across frames (risk history per track).
    3. Per-frame risk from geometric + pose features:
         - dog <-> nearest-person distance
         - dog velocity toward nearest person
         - dog box aspect-ratio change (lunging/rearing)
         - human pose: arms raised (defensive) or running (fleeing)
    4. Temporal intelligence on top of raw risk:
         - EMA smoothing per track (removes single-frame noise)
         - sustained alert: must hold above threshold for N frames
         - alert cooldown: suppress repeat alerts for the same track
"""

import math
from collections import deque
from pathlib import Path

import cv2
import numpy as np

from src.models.detector import (
    DogDetector,
    KP_NOSE, KP_L_SHOULDER, KP_R_SHOULDER, KP_L_ELBOW, KP_R_ELBOW,
    KP_L_WRIST, KP_R_WRIST, KP_L_HIP, KP_R_HIP,
    KP_L_KNEE, KP_R_KNEE, KP_L_ANKLE, KP_R_ANKLE,
)


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
    """Per-dog state with temporal risk memory."""

    _next_id = 1
    HISTORY_LEN = 30

    def __init__(self, box):
        self.id = Track._next_id
        Track._next_id += 1
        self.box = box
        self.box_history = deque([box], maxlen=self.HISTORY_LEN)
        self.risk_history = deque(maxlen=self.HISTORY_LEN)
        self.missed = 0
        self.risk_ema = 0.0
        self.sustained_frames = 0     # consecutive frames above threshold
        self.last_alert_frame = -10_000
        self.age = 0                  # total frames since birth

    def update(self, box):
        self.box = box
        self.box_history.append(box)
        self.missed = 0
        self.age += 1

    def mark_missed(self):
        self.missed += 1
        self.age += 1


class IoUTracker:
    """Greedy IoU tracker - good enough for dog-scale object counts."""

    def __init__(self, iou_threshold=0.2, max_missed=15):
        self.tracks = []
        self.iou_threshold = iou_threshold
        self.max_missed = max_missed

    def update(self, detections):
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

        for di, det in enumerate(detections):
            if di not in used_d:
                self.tracks.append(Track(det))

        for ti, track in enumerate(self.tracks):
            if ti not in used_t:
                track.mark_missed()
        self.tracks = [t for t in self.tracks if t.missed <= self.max_missed]

        return self.tracks


# -------------------- pose features --------------------

def _kp_xy(keypoints, idx, visibility_threshold=0.3):
    """Return (x, y) if visible else None."""
    x, y, v = keypoints[idx]
    if v < visibility_threshold:
        return None
    return x, y


def compute_pose_features(person):
    """
    Behavioral features derived from COCO-Pose keypoints.

    Returns dict with:
        arms_raised: wrists above shoulders (defensive / warding off)
        hands_forward: wrists in front of body (reaching / grabbing dog)
        crouched: hips closer to knees than expected (bracing)
    """
    features = {"arms_raised": 0.0, "hands_forward": 0.0, "crouched": 0.0}
    kps = person.get("keypoints")
    if kps is None:
        return features

    l_sh = _kp_xy(kps, KP_L_SHOULDER)
    r_sh = _kp_xy(kps, KP_R_SHOULDER)
    l_wr = _kp_xy(kps, KP_L_WRIST)
    r_wr = _kp_xy(kps, KP_R_WRIST)
    l_hip = _kp_xy(kps, KP_L_HIP)
    r_hip = _kp_xy(kps, KP_R_HIP)
    l_knee = _kp_xy(kps, KP_L_KNEE)
    r_knee = _kp_xy(kps, KP_R_KNEE)

    # Arms raised: wrist y < shoulder y (image y grows downward)
    raised_count = 0.0
    if l_wr and l_sh and l_wr[1] < l_sh[1]:
        raised_count += 1
    if r_wr and r_sh and r_wr[1] < r_sh[1]:
        raised_count += 1
    features["arms_raised"] = raised_count / 2.0  # 0.0, 0.5, or 1.0

    # Crouched: knees close to hips vertically (small hip-knee vertical gap
    # relative to shoulder-hip torso length)
    if l_hip and l_knee and l_sh:
        torso = max(1, abs(l_sh[1] - l_hip[1]))
        thigh = abs(l_hip[1] - l_knee[1])
        ratio = thigh / torso
        # Standing: thigh >= torso; crouched: thigh < 0.6 * torso
        if ratio < 0.6:
            features["crouched"] = max(features["crouched"], 1.0 - ratio / 0.6)

    return features


# -------------------- risk scoring --------------------

def _compute_risk(track, persons, frame_shape, num_dogs):
    """
    Interpretable risk score in [0, 1]. Weights:
        0.40 distance    - closer dog-to-person
        0.25 velocity    - dog moving toward person
        0.15 posture     - dog aspect-ratio change (lunging)
        0.10 human pose  - defensive posture (arms up / crouched)
        + 0.10 pack bonus when multiple dogs
    """
    features = {
        "distance": 0, "velocity": 0, "posture": 0, "human_pose": 0,
    }
    if not persons:
        return 0.0, features

    h, w = frame_shape[:2]
    diag = math.hypot(w, h)

    dog_cx, dog_cy = _box_center(track.box)

    # nearest person
    nearest = min(
        persons,
        key=lambda p: math.hypot(
            _box_center(p)[0] - dog_cx, _box_center(p)[1] - dog_cy
        ),
    )
    p_cx, p_cy = _box_center(nearest)

    # 1. Distance
    distance = math.hypot(dog_cx - p_cx, dog_cy - p_cy) / diag
    distance_risk = max(0.0, 1.0 - distance / 0.30)

    # 2. Velocity toward person
    velocity_risk = 0.0
    if len(track.box_history) >= 5:
        old_cx, old_cy = _box_center(track.box_history[-5])
        dx, dy = dog_cx - old_cx, dog_cy - old_cy
        pdx, pdy = p_cx - old_cx, p_cy - old_cy
        p_len = math.hypot(pdx, pdy) + 1e-6
        pdx, pdy = pdx / p_len, pdy / p_len
        velocity_toward = (dx * pdx + dy * pdy) / diag
        velocity_risk = min(1.0, max(0.0, velocity_toward / 0.02))

    # 3. Posture change (box aspect ratio variance over last 5 frames)
    posture_risk = 0.0
    if len(track.box_history) >= 5:
        recent_ars = [_box_aspect_ratio(b) for b in list(track.box_history)[-5:]]
        ar_range = max(recent_ars) - min(recent_ars)
        posture_risk = min(1.0, ar_range / 0.6)

    # 4. Human pose (defensive / bracing) from nearest person
    pose = compute_pose_features(nearest)
    pose_risk = max(pose["arms_raised"], pose["crouched"])

    # 5. Pack bonus
    pack_bonus = 0.1 if num_dogs > 1 else 0.0

    risk = (
        0.40 * distance_risk
        + 0.25 * velocity_risk
        + 0.15 * posture_risk
        + 0.10 * pose_risk
        + pack_bonus
    )

    features = {
        "distance": round(distance_risk, 2),
        "velocity": round(velocity_risk, 2),
        "posture": round(posture_risk, 2),
        "human_pose": round(pose_risk, 2),
    }
    return min(1.0, risk), features


# -------------------- pipeline --------------------

class DogAggressionPipeline:
    """
    End-to-end detection + tracking + temporally-smoothed risk scoring.
    """

    COLORS = {
        "person": (255, 180, 60),
        "dog_safe": (0, 220, 90),
        "dog_caution": (40, 200, 240),
        "dog_alert": (60, 60, 255),
        "skeleton": (220, 220, 220),
    }

    # COCO-Pose edges for skeleton drawing
    SKELETON = [
        (5, 7), (7, 9), (6, 8), (8, 10),        # arms
        (5, 6), (5, 11), (6, 12), (11, 12),     # torso
        (11, 13), (13, 15), (12, 14), (14, 16), # legs
        (0, 5), (0, 6),                         # head to shoulders
    ]

    def __init__(self, detector_path="yolo11m.pt", pose_model="yolo11m-pose.pt",
                 classifier_path=None, det_conf=0.35, risk_threshold=0.45,
                 aggression_threshold=None, smoothing_alpha=0.5,
                 sustain_frames=3, cooldown_frames=30,
                 crop_size=224, device="auto"):
        if aggression_threshold is not None:
            risk_threshold = aggression_threshold
        if detector_path in (None, "", "None"):
            detector_path = "yolo11m.pt"

        self.detector = DogDetector(
            model_path=detector_path,
            pose_model=pose_model,
            conf=det_conf,
        )
        self.risk_threshold = risk_threshold
        self.smoothing_alpha = smoothing_alpha
        self.sustain_frames = sustain_frames
        self.cooldown_frames = cooldown_frames
        self.tracker = IoUTracker()
        self.frame_idx = 0
        self._last_persons = []

        print(f"[pipeline] detector={detector_path} pose={pose_model} "
              f"conf={det_conf} risk_threshold={risk_threshold} "
              f"sustain={sustain_frames} cooldown={cooldown_frames}")

    # ---- per-frame ----

    def process_frame(self, frame):
        self.frame_idx += 1

        det = self.detector.detect(frame)
        persons, dogs = det["persons"], det["dogs"]
        self._last_persons = persons

        tracks = self.tracker.update(dogs)

        results = []
        for t in tracks:
            if t.missed > 0:
                continue

            raw_risk, features = _compute_risk(t, persons, frame.shape, len(dogs))

            # EMA smoothing (first frame: initialize to raw)
            if len(t.risk_history) == 0:
                t.risk_ema = raw_risk
            else:
                a = self.smoothing_alpha
                t.risk_ema = (1 - a) * t.risk_ema + a * raw_risk
            t.risk_history.append(t.risk_ema)

            # Sustained-alert counter
            if t.risk_ema >= self.risk_threshold:
                t.sustained_frames += 1
            else:
                t.sustained_frames = 0

            # Fire an alert only when sustained AND out of cooldown
            cooldown_ok = (self.frame_idx - t.last_alert_frame) > self.cooldown_frames
            alert = (t.sustained_frames >= self.sustain_frames) and cooldown_ok
            new_alert = alert and (t.sustained_frames == self.sustain_frames)
            if new_alert:
                t.last_alert_frame = self.frame_idx

            results.append({
                "x1": t.box["x1"], "y1": t.box["y1"],
                "x2": t.box["x2"], "y2": t.box["y2"],
                "track_id": t.id,
                "dog_confidence": t.box["confidence"],
                "risk_raw": raw_risk,
                "risk": t.risk_ema,
                "behavior_confidence": t.risk_ema,
                "features": features,
                "sustained": t.sustained_frames,
                "sustain_target": self.sustain_frames,
                "alert": alert,
                "new_alert": new_alert,
                "history": list(t.risk_history),
                "behavior": "aggressive" if alert else "non_aggressive",
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
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(img, (x1, y1 - th - 10), (x1 + tw + 6, y1), color, -1)
            cv2.putText(img, label, (x1 + 3, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    def _draw_skeleton(self, img, keypoints):
        if keypoints is None:
            return
        for a, b in self.SKELETON:
            xa, ya, va = keypoints[a]
            xb, yb, vb = keypoints[b]
            if va < 0.3 or vb < 0.3:
                continue
            cv2.line(img, (int(xa), int(ya)), (int(xb), int(yb)),
                     self.COLORS["skeleton"], 2)
        for x, y, v in keypoints:
            if v < 0.3:
                continue
            cv2.circle(img, (int(x), int(y)), 3,
                       self.COLORS["skeleton"], -1)

    def _draw_risk_bar(self, img, x, y, w, h, history):
        """Miniature sparkline of risk over the last N frames."""
        cv2.rectangle(img, (x, y), (x + w, y + h), (40, 40, 40), -1)
        if len(history) < 2:
            return
        step = w / (len(history) - 1)
        pts = []
        for i, r in enumerate(history):
            px = int(x + i * step)
            py = int(y + h - r * h)
            pts.append((px, py))
        for i in range(1, len(pts)):
            color = (60, 60, 255) if history[i] >= self.risk_threshold else (0, 220, 90)
            cv2.line(img, pts[i - 1], pts[i], color, 2)
        # threshold line
        ty = int(y + h - self.risk_threshold * h)
        cv2.line(img, (x, ty), (x + w, ty), (100, 100, 100), 1)

    def draw_results(self, frame, results):
        annotated = frame.copy()

        # Persons (with skeleton when available)
        for p in self._last_persons:
            self._draw_box(
                annotated, p["x1"], p["y1"], p["x2"], p["y2"],
                self.COLORS["person"], thickness=2,
                label=f"person {p['confidence']:.2f}",
            )
            if "keypoints" in p:
                self._draw_skeleton(annotated, p["keypoints"])

        # Dogs with risk color + sparkline
        for r in results:
            color = self._dog_color(r["risk"])
            thickness = 3 if r["alert"] else 2
            if r["alert"]:
                label = f"ALERT dog#{r['track_id']} risk {r['risk']:.2f}"
            else:
                label = (f"dog#{r['track_id']} risk {r['risk']:.2f} "
                         f"[{r['sustained']}/{r['sustain_target']}]")
            self._draw_box(
                annotated, r["x1"], r["y1"], r["x2"], r["y2"],
                color, thickness=thickness, label=label,
            )

            # Risk sparkline above the box
            bar_w, bar_h = 120, 30
            bx = r["x1"]
            by = max(0, r["y1"] - bar_h - 30)
            self._draw_risk_bar(annotated, bx, by, bar_w, bar_h, r["history"])

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

        frame_count, new_alerts = 0, 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            results = self.process_frame(frame)
            annotated = self.draw_results(frame, results)
            for r in results:
                if r.get("new_alert"):
                    new_alerts += 1
                    print(f"  Frame {frame_count}: SUSTAINED ALERT dog#{r['track_id']} "
                          f"risk={r['risk']:.2f}")
            if writer:
                writer.write(annotated)
            frame_count += 1

        cap.release()
        if writer:
            writer.release()
        print(f"[pipeline] {frame_count} frames, {new_alerts} sustained alert(s)")

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
    p.add_argument("--pose", type=str, default="yolo11m-pose.pt")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--conf", type=float, default=0.35)
    p.add_argument("--risk", type=float, default=0.45)
    p.add_argument("--sustain", type=int, default=3)
    p.add_argument("--cooldown", type=int, default=30)
    args = p.parse_args()

    pose_model = None if args.pose.lower() in ("none", "off", "") else args.pose
    pipe = DogAggressionPipeline(
        detector_path=args.detector,
        pose_model=pose_model,
        det_conf=args.conf,
        risk_threshold=args.risk,
        sustain_frames=args.sustain,
        cooldown_frames=args.cooldown,
    )
    if args.source == "webcam":
        pipe.predict_webcam()
    elif args.source.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
        pipe.predict_video(args.source, output_path=args.output)
    else:
        pipe.predict_image(args.source, save_path=args.output)
