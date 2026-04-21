"""
Detector - Stock COCO-pretrained YOLO11 wrapper.

Uses ultralytics YOLO out-of-the-box (no fine-tuning required). The COCO
model already distinguishes:
    - class 0  : person
    - class 16 : dog

This solves the human-as-dog confusion that a dog-only fine-tuned model has.
"""

from pathlib import Path

from ultralytics import YOLO


COCO_PERSON = 0
COCO_DOG = 16


class DogDetector:
    """Thin wrapper around ultralytics YOLO that exposes persons + dogs."""

    def __init__(self, model_path="yolo11m.pt", conf=0.35, iou=0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def detect(self, frame):
        """
        Run detection on a single BGR frame.

        Returns:
            dict with keys "persons" and "dogs", each a list of
            {x1, y1, x2, y2, confidence}
        """
        results = self.model.predict(
            source=frame,
            conf=self.conf,
            iou=self.iou,
            classes=[COCO_PERSON, COCO_DOG],
            verbose=False,
        )

        persons = []
        dogs = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                cls_id = int(box.cls[0])
                det = {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                    "confidence": float(box.conf[0]),
                }
                if cls_id == COCO_PERSON:
                    persons.append(det)
                elif cls_id == COCO_DOG:
                    dogs.append(det)

        return {"persons": persons, "dogs": dogs}

    # Back-compat: used by older evaluation code
    def get_dog_boxes(self, frame):
        return self.detect(frame)["dogs"]

    def train(self, data_yaml, epochs=50, batch_size=16, lr=0.01,
              imgsz=640, patience=10, freeze_layers=0,
              project="experiments", name="detector", **kwargs):
        """Fine-tune on a custom dataset (optional; stock weights work fine)."""
        return self.model.train(
            data=data_yaml, epochs=epochs, batch=batch_size, lr0=lr,
            imgsz=imgsz, patience=patience, freeze=freeze_layers,
            project=project, name=name, save=True, plots=True,
            verbose=True, **kwargs,
        )

    def load_best(self, run_dir):
        best_path = Path(run_dir) / "weights" / "best.pt"
        if not best_path.exists():
            raise FileNotFoundError(f"No best.pt at {best_path}")
        self.model = YOLO(str(best_path))
