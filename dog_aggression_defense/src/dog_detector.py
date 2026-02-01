"""
Enhanced Dog Detector - Detects both DOGS and PEOPLE
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

class DogDetector:
    """
    Detects both dogs and people for threat analysis
    """

    def __init__(self, model_path=None, conf_threshold=0.5):
        """
        Initialize YOLOv8 detector for dogs and people

        Args:
            model_path: Path to custom model or None for pretrained
            conf_threshold: Confidence threshold for detection
        """
        self.conf_threshold = conf_threshold

        # COCO class IDs: 0=person, 16=dog
        self.PERSON_CLASS_ID = 0
        self.DOG_CLASS_ID = 16

        # Load model
        if model_path and Path(model_path).exists():
            print(f"Loading custom model: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("Loading YOLOv8n pretrained model (person & dog classes only)...")
            self.model = YOLO('yolov8n.pt')

        print("Dog & Person detector initialized")

    def detect(self, frame):
        """
        Detect dogs and people in a frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of [x1, y1, x2, y2, confidence, class_id]
        """
        # Run inference - only detect people and dogs
        results = self.model(frame, classes=[self.PERSON_CLASS_ID, self.DOG_CLASS_ID],
                           verbose=False)[0]

        detections = []

        if results.boxes is not None:
            for box in results.boxes:
                conf = float(box.conf)

                if conf >= self.conf_threshold:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    class_id = int(box.cls)
                    detections.append([x1, y1, x2, y2, conf, class_id])

        return detections

    def detect_raw(self, frame):
        """
        Get raw YOLO results for advanced processing
        """
        return self.model(frame, classes=[self.PERSON_CLASS_ID, self.DOG_CLASS_ID],
                         verbose=False)[0]