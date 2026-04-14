"""
Dog Detector Module - YOLOv8 Wrapper

Handles:
- Loading pretrained YOLOv8 model
- Fine-tuning on dog-specific dataset
- Running inference with post-processing
- Exporting to ONNX for deployment
"""

from pathlib import Path

from ultralytics import YOLO


class DogDetector:
    """YOLOv8-based dog detector with fine-tuning support."""

    def __init__(self, model_path="yolov8n.pt", conf=0.5, iou=0.45):
        self.model = YOLO(model_path)
        self.conf = conf
        self.iou = iou

    def train(self, data_yaml, epochs=50, batch_size=16, lr=0.01,
              imgsz=640, patience=10, freeze_layers=0,
              project="experiments", name="detector", **kwargs):
        """
        Fine-tune YOLOv8 on dog detection dataset.

        Args:
            data_yaml: Path to data.yaml config
            epochs: Number of training epochs
            batch_size: Batch size
            lr: Initial learning rate
            imgsz: Input image size
            patience: Early stopping patience
            freeze_layers: Number of backbone layers to freeze
            project: Experiment output directory
            name: Run name
        """
        results = self.model.train(
            data=data_yaml,
            epochs=epochs,
            batch=batch_size,
            lr0=lr,
            imgsz=imgsz,
            patience=patience,
            freeze=freeze_layers,
            project=project,
            name=name,
            save=True,
            save_period=5,
            plots=True,
            verbose=True,
            **kwargs,
        )
        return results

    def predict(self, source, conf=None, iou=None, save=False):
        """
        Run detection on image/video/directory.

        Args:
            source: Image path, video path, directory, or numpy array
            conf: Confidence threshold (uses default if None)
            iou: NMS IoU threshold (uses default if None)
            save: Whether to save annotated results

        Returns:
            List of Results objects with boxes, confidence, class info
        """
        results = self.model.predict(
            source=source,
            conf=conf or self.conf,
            iou=iou or self.iou,
            save=save,
            verbose=False,
        )
        return results

    def get_dog_boxes(self, image):
        """
        Get bounding boxes for detected dogs.

        Returns:
            List of dicts: [{x1, y1, x2, y2, confidence, class_id, class_name}]
        """
        results = self.predict(image)
        detections = []

        for result in results:
            if result.boxes is None:
                continue
            for box in result.boxes:
                det = {
                    "x1": int(box.xyxy[0][0]),
                    "y1": int(box.xyxy[0][1]),
                    "x2": int(box.xyxy[0][2]),
                    "y2": int(box.xyxy[0][3]),
                    "confidence": float(box.conf[0]),
                    "class_id": int(box.cls[0]),
                    "class_name": result.names[int(box.cls[0])],
                }
                detections.append(det)

        return detections

    def validate(self, data_yaml=None):
        """Run validation and return metrics."""
        metrics = self.model.val(data=data_yaml, verbose=True)
        return {
            "mAP50": float(metrics.box.map50),
            "mAP50_95": float(metrics.box.map),
            "precision": float(metrics.box.mp),
            "recall": float(metrics.box.mr),
        }

    def export(self, format="onnx", imgsz=640):
        """Export model for deployment."""
        path = self.model.export(format=format, imgsz=imgsz)
        print(f"[detector] Model exported to: {path}")
        return path

    def load_best(self, run_dir):
        """Load the best checkpoint from a training run."""
        best_path = Path(run_dir) / "weights" / "best.pt"
        if best_path.exists():
            self.model = YOLO(str(best_path))
            print(f"[detector] Loaded best model from {best_path}")
        else:
            raise FileNotFoundError(f"No best.pt found at {best_path}")
