"""
End-to-End Inference Pipeline

Detect dogs → Classify aggression (if classifier available) → Alert

Supports:
- Single image
- Directory of images
- Video file
- Webcam stream
- Detection-only mode (no classifier needed)
"""

import cv2
import torch
import numpy as np
from pathlib import Path

from src.models.detector import DogDetector


# COCO class ID for 'dog' — used when running with pretrained YOLO
COCO_DOG_CLASS_ID = 16


class DogAggressionPipeline:
    """
    Full inference pipeline: detect dogs, classify aggression, visualize.
    Works in two modes:
    - Full mode: detector + classifier (after training)
    - Detection-only mode: just detector (works with pretrained yolov8n.pt)
    """

    COLORS = {
        "non_aggressive": (0, 255, 0),   # green
        "aggressive": (0, 0, 255),        # red
        "detected": (0, 200, 255),        # yellow (detection only)
    }
    BEHAVIOR_NAMES = {0: "non_aggressive", 1: "aggressive"}

    def __init__(self, detector_path, classifier_path=None,
                 det_conf=0.5, aggression_threshold=0.7,
                 crop_size=224, device="auto"):
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load detector
        self.detector = DogDetector(model_path=detector_path, conf=det_conf)

        # Determine if this is a COCO-pretrained model (filters for dog class)
        self.coco_mode = self._is_coco_model()

        # Try loading classifier (optional)
        self.classifier = None
        self.has_classifier = False
        self.transform = None

        if classifier_path and Path(classifier_path).exists():
            try:
                from src.models.classifier import AggressionClassifier
                self.classifier = AggressionClassifier.load(
                    classifier_path, device=str(self.device)
                )
                self.classifier.eval()
                self.has_classifier = True

                from src.data.prepare import get_val_transforms
                self.transform = get_val_transforms(crop_size)
                print(f"[pipeline] Classifier loaded from {classifier_path}")
            except Exception as e:
                print(f"[pipeline] Classifier not available ({e}) — detection-only mode")

        mode = "full (detect + classify)" if self.has_classifier else "detection-only"
        print(f"[pipeline] Mode: {mode}")
        print(f"[pipeline] Device: {self.device}")
        print(f"[pipeline] Detection conf: {det_conf}")
        if self.has_classifier:
            print(f"[pipeline] Aggression threshold: {aggression_threshold}")
        self.aggression_threshold = aggression_threshold

    def _is_coco_model(self):
        """Check if detector is using COCO class names (pretrained model)."""
        names = self.detector.model.names
        return len(names) > 16 and names.get(16, "") == "dog"

    def process_frame(self, frame):
        """
        Process a single frame/image.

        Returns:
            List of dicts: [{x1, y1, x2, y2, dog_conf, behavior, behavior_conf, alert}]
        """
        # Get all detections
        results = self.detector.predict(frame)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])

                # In COCO mode, only keep dog detections (class 16)
                if self.coco_mode and cls_id != COCO_DOG_CLASS_ID:
                    continue

                det = {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                    "confidence": conf,
                    "class_id": cls_id,
                }
                detections.append(det)

        # Classify each detection
        output = []
        for det in detections:
            result_entry = {
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "dog_confidence": det["confidence"],
            }

            if self.has_classifier:
                crop = frame[det["y1"]:det["y2"], det["x1"]:det["x2"]]
                if crop.size == 0:
                    continue

                crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
                augmented = self.transform(image=crop_rgb)
                tensor = augmented["image"].unsqueeze(0).to(self.device)

                cls_id, conf = self.classifier.predict_single(tensor)
                behavior = self.BEHAVIOR_NAMES[cls_id]
                is_aggressive = (cls_id == 1 and conf >= self.aggression_threshold)

                result_entry["behavior"] = "aggressive" if is_aggressive else "non_aggressive"
                result_entry["behavior_confidence"] = conf
                result_entry["alert"] = is_aggressive
            else:
                # Detection-only mode
                result_entry["behavior"] = "detected"
                result_entry["behavior_confidence"] = det["confidence"]
                result_entry["alert"] = False

            output.append(result_entry)

        return output

    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()

        for r in results:
            color = self.COLORS.get(r["behavior"], (0, 200, 255))
            thickness = 3 if r.get("alert") else 2

            cv2.rectangle(
                annotated,
                (r["x1"], r["y1"]),
                (r["x2"], r["y2"]),
                color, thickness,
            )

            if r["behavior"] == "detected":
                label = f"DOG {r['behavior_confidence']:.2f}"
            else:
                label = f"{r['behavior']} {r['behavior_confidence']:.2f}"
                if r.get("alert"):
                    label = "ALERT: " + label

            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)[0]
            cv2.rectangle(
                annotated,
                (r["x1"], r["y1"] - label_size[1] - 10),
                (r["x1"] + label_size[0], r["y1"]),
                color, -1,
            )
            cv2.putText(
                annotated, label,
                (r["x1"], r["y1"] - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 2,
            )

        return annotated

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

        fps = int(cap.get(cv2.CAP_PROP_FPS))
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        writer = None
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (w, h))

        frame_count = 0
        alert_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            results = self.process_frame(frame)
            annotated = self.draw_results(frame, results)

            alerts = [r for r in results if r.get("alert")]
            if alerts:
                alert_count += 1
                print(f"  Frame {frame_count}: {len(alerts)} AGGRESSIVE dog(s)!")

            if writer:
                writer.write(annotated)
            frame_count += 1

        cap.release()
        if writer:
            writer.release()

        print(f"\n[pipeline] Processed {frame_count} frames, {alert_count} alert frames")

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

    parser = argparse.ArgumentParser(description="Dog Aggression Detection")
    parser.add_argument("--source", type=str, required=True)
    parser.add_argument("--detector", type=str, required=True)
    parser.add_argument("--classifier", type=str, default=None)
    parser.add_argument("--output", type=str, default=None)
    parser.add_argument("--conf", type=float, default=0.5)
    parser.add_argument("--threshold", type=float, default=0.7)
    args = parser.parse_args()

    pipeline = DogAggressionPipeline(
        detector_path=args.detector,
        classifier_path=args.classifier,
        det_conf=args.conf,
        aggression_threshold=args.threshold,
    )

    if args.source == "webcam":
        pipeline.predict_webcam()
    elif args.source.endswith((".mp4", ".avi", ".mov")):
        pipeline.predict_video(args.source, output_path=args.output)
    else:
        pipeline.predict_image(args.source, save_path=args.output)
