"""
Stage 1: Dog Detector Training

Fine-tunes YOLOv8 on a dog detection dataset.
Shows live training progress with percentage and saves training plots.
"""

import json
import time
from pathlib import Path
from datetime import datetime

import yaml

from src.models.detector import DogDetector


def train_detector(config):
    """
    Train the YOLOv8 dog detector.

    Args:
        config: Dict from config/config.yaml

    Returns:
        Dict with training results and metrics
    """
    det_cfg = config["detector"]
    data_cfg = config["data"]

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"detector_{timestamp}"
    project_dir = "experiments"

    print("\n" + "=" * 60)
    print("  STAGE 1: TRAINING DOG DETECTOR (YOLOv8)")
    print("=" * 60)
    print(f"  Model:      {det_cfg['model']}")
    print(f"  Epochs:     {det_cfg['epochs']}")
    print(f"  Batch size: {det_cfg['batch_size']}")
    print(f"  LR:         {det_cfg['lr']}")
    print(f"  Image size: {data_cfg['image_size']}")
    print(f"  Output:     {project_dir}/{run_name}")
    print("=" * 60)
    print()
    print("  YOLOv8 will show its own progress bars per epoch.")
    print("  Training plots are saved to the output folder.\n")

    detector = DogDetector(
        model_path=det_cfg["model"],
        conf=det_cfg["conf_threshold"],
        iou=det_cfg["iou_threshold"],
    )

    start_time = time.time()

    results = detector.train(
        data_yaml=data_cfg["dataset_config"],
        epochs=det_cfg["epochs"],
        batch_size=det_cfg["batch_size"],
        lr=det_cfg["lr"],
        imgsz=data_cfg["image_size"],
        patience=det_cfg["patience"],
        freeze_layers=det_cfg.get("freeze_backbone_epochs", 0),
        project=project_dir,
        name=run_name,
        optimizer=det_cfg.get("optimizer", "SGD"),
        momentum=det_cfg.get("momentum", 0.937),
        weight_decay=det_cfg.get("weight_decay", 0.0005),
        augment=det_cfg.get("augment", True),
        mosaic=det_cfg.get("mosaic", 1.0),
        mixup=det_cfg.get("mixup", 0.0),
    )

    training_time = time.time() - start_time

    run_dir = Path(project_dir) / run_name
    metrics = detector.validate(data_yaml=data_cfg["dataset_config"])
    metrics["training_time_seconds"] = training_time
    metrics["run_dir"] = str(run_dir)
    metrics["run_name"] = run_name

    metrics_path = run_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "=" * 60)
    print("  DETECTOR TRAINING COMPLETE")
    print("=" * 60)
    print(f"  mAP@50:     {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:  {metrics['mAP50_95']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")
    print(f"  Time:       {training_time:.0f}s")
    print(f"  Saved to:   {run_dir}")
    print(f"  Plots:      {run_dir}/ (confusion_matrix.png, results.png, ...)")
    print("=" * 60 + "\n")

    return metrics


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_detector(config)
