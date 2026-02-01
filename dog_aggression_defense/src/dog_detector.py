"""
YOLOv8 Dog Detector - Detects ONLY dogs
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class DogDetector:
    def __init__(self, model_path=None, conf_threshold=0.5):
        """
        Initialize YOLOv8 dog detector

        Args:
            model_path: Path to custom model or None for pretrained
            conf_threshold: Confidence threshold for detection
        """
        self.conf_threshold = conf_threshold

        # Load model
        if model_path and Path(model_path).exists():
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading YOLOv8n pretrained model (dog class only)...")
            self.model = YOLO('yolov8n.pt')

        # COCO class index for 'dog' is 16
        self.DOG_CLASS_ID = 16

        print("Dog detector initialized")

    def detect(self, frame):
        """
        Detect dogs in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections [x1, y1, x2, y2, confidence]
        """
        # Run inference
        results = self.model(frame, verbose=False)[0]

        detections = []

        # Filter for dogs only
        for box in results.boxes:
            # Check if detection is a dog (class 16 in COCO)
            if int(box.cls) == self.DOG_CLASS_ID:
                conf = float(box.conf)

                if conf >= self.conf_threshold:
                    # Convert to pixel coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    detections.append([x1, y1, x2, y2, conf])

        return detections

    def visualize_detections(self, frame, detections):
        """Visualize dog detections on frame"""
        output_frame = frame.copy()

        for (x1, y1, x2, y2, conf) in detections:
            # Draw bounding box
            cv2.rectangle(output_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Draw label
            label = f"Dog: {conf:.2f}"
            cv2.putText(output_frame, label, (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return output_frame


# Example usage
if __name__ == "__main__":
    # Test the detector
    detector = DogDetector()

    # Test with webcam
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("Testing dog detector... Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        detections = detector.detect(frame)
        output_frame = detector.visualize_detections(frame, detections)

        cv2.imshow("Dog Detector Test", output_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()