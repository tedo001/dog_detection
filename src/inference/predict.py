"""
End-to-End Inference Pipeline

Detect dogs → Classify aggression → Alert

Supports:
- Single image
- Directory of images
- Video file
- Webcam stream
"""

import cv2
import torch
import numpy as np
from pathlib import Path

from src.models.detector import DogDetector
from src.models.classifier import AggressionClassifier
from src.data.prepare import get_val_transforms


class DogAggressionPipeline:
    """
    Full inference pipeline: detect dogs, classify aggression, visualize.
    """

    COLORS = {
        "non_aggressive": (0, 255, 0),   # green
        "aggressive": (0, 0, 255),        # red
    }
    BEHAVIOR_NAMES = {0: "non_aggressive", 1: "aggressive"}

    def __init__(self, detector_path, classifier_path,
                 det_conf=0.5, aggression_threshold=0.7,
                 crop_size=224, device="auto"):
        # Device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Load models
        self.detector = DogDetector(model_path=detector_path, conf=det_conf)
        self.classifier = AggressionClassifier.load(classifier_path, device=str(self.device))
        self.classifier.eval()

        self.aggression_threshold = aggression_threshold
        self.transform = get_val_transforms(crop_size)

        print(f"[pipeline] Loaded on {self.device}")
        print(f"[pipeline] Detection conf: {det_conf}")
        print(f"[pipeline] Aggression threshold: {aggression_threshold}")

    def process_frame(self, frame):
        """
        Process a single frame/image.

        Args:
            frame: BGR numpy array (from cv2)

        Returns:
            List of dicts: [{x1, y1, x2, y2, dog_conf, behavior, behavior_conf}]
        """
        detections = self.detector.get_dog_boxes(frame)
        results = []

        for det in detections:
            crop = frame[det["y1"]:det["y2"], det["x1"]:det["x2"]]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            augmented = self.transform(image=crop_rgb)
            tensor = augmented["image"].unsqueeze(0).to(self.device)

            cls_id, conf = self.classifier.predict_single(tensor)
            behavior = self.BEHAVIOR_NAMES[cls_id]

            # Apply aggression threshold
            is_aggressive = (cls_id == 1 and conf >= self.aggression_threshold)

            results.append({
                "x1": det["x1"],
                "y1": det["y1"],
                "x2": det["x2"],
                "y2": det["y2"],
                "dog_confidence": det["confidence"],
                "behavior": "aggressive" if is_aggressive else "non_aggressive",
                "behavior_confidence": conf,
                "alert": is_aggressive,
            })

        return results

    def draw_results(self, frame, results):
        """Draw bounding boxes and labels on frame."""
        annotated = frame.copy()

        for r in results:
            color = self.COLORS[r["behavior"]]
            thickness = 3 if r["alert"] else 2

            # Bounding box
            cv2.rectangle(
                annotated,
                (r["x1"], r["y1"]),
                (r["x2"], r["y2"]),
                color, thickness,
            )

            # Label
            label = f"{r['behavior']} {r['behavior_confidence']:.2f}"
            if r["alert"]:
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
        """Run prediction on a single image."""
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
        """Run prediction on a video file."""
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

            alerts = [r for r in results if r["alert"]]
            if alerts:
                alert_count += 1
                print(f"  Frame {frame_count}: {len(alerts)} AGGRESSIVE dog(s) detected!")

            if writer:
                writer.write(annotated)
            frame_count += 1

        cap.release()
        if writer:
            writer.release()

        print(f"\n[pipeline] Processed {frame_count} frames, {alert_count} alert frames")

    def predict_webcam(self):
        """Run real-time prediction from webcam."""
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
    parser.add_argument("--source", type=str, required=True,
                        help="Image path, video path, or 'webcam'")
    parser.add_argument("--detector", type=str, required=True,
                        help="Path to detector weights (best.pt)")
    parser.add_argument("--classifier", type=str, required=True,
                        help="Path to classifier weights (best.pt)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output path for annotated result")
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
