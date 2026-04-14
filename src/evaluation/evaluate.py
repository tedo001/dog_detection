"""
Evaluation Module

End-to-end evaluation of the full pipeline:
- Detector metrics (mAP, precision, recall)
- Classifier metrics (F1, confusion matrix, AUC-ROC)
- Pipeline metrics (end-to-end accuracy on aggressive dog detection)
"""

import json
from pathlib import Path

import numpy as np
import torch
import cv2
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    f1_score,
    precision_recall_curve,
)

from src.models.detector import DogDetector
from src.models.classifier import AggressionClassifier
from src.data.prepare import get_val_transforms


def evaluate_detector(detector, data_yaml):
    """
    Evaluate detector on validation set.

    Returns:
        Dict with mAP50, mAP50-95, precision, recall
    """
    metrics = detector.validate(data_yaml=data_yaml)

    print("\n  DETECTOR EVALUATION")
    print(f"  mAP@50:     {metrics['mAP50']:.4f}")
    print(f"  mAP@50-95:  {metrics['mAP50_95']:.4f}")
    print(f"  Precision:  {metrics['precision']:.4f}")
    print(f"  Recall:     {metrics['recall']:.4f}")

    return metrics


def evaluate_classifier(model, dataloader, device, class_names=None):
    """
    Evaluate classifier on validation/test set.

    Returns:
        Dict with accuracy, F1, precision, recall, confusion matrix, AUC
    """
    if class_names is None:
        class_names = ["non_aggressive", "aggressive"]

    model.eval()
    all_preds = []
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for images, labels in dataloader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)

            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs[:, 1].cpu().numpy())

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Core metrics
    f1 = f1_score(all_labels, all_preds, average="binary", pos_label=1)
    cm = confusion_matrix(all_labels, all_preds)
    report = classification_report(
        all_labels, all_preds, target_names=class_names, output_dict=True
    )

    # AUC-ROC (only if both classes present)
    auc = 0.0
    if len(np.unique(all_labels)) > 1:
        auc = roc_auc_score(all_labels, all_probs)

    # Safety metrics
    tp = ((all_preds == 1) & (all_labels == 1)).sum()
    fn = ((all_preds == 0) & (all_labels == 1)).sum()
    fp = ((all_preds == 1) & (all_labels == 0)).sum()
    miss_rate = fn / (tp + fn) if (tp + fn) > 0 else 0.0
    false_alarm_rate = fp / (fp + ((all_preds == 0) & (all_labels == 0)).sum()) if fp > 0 else 0.0

    metrics = {
        "f1": float(f1),
        "auc_roc": float(auc),
        "accuracy": float((all_preds == all_labels).mean()),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
        "miss_rate": float(miss_rate),
        "false_alarm_rate": float(false_alarm_rate),
    }

    print("\n  CLASSIFIER EVALUATION")
    print(f"  F1 Score:        {f1:.4f}")
    print(f"  AUC-ROC:         {auc:.4f}")
    print(f"  Miss Rate:       {miss_rate:.4f}  (missed aggressive dogs)")
    print(f"  False Alarm:     {false_alarm_rate:.4f}")
    print(f"\n  Confusion Matrix:")
    print(f"    {class_names[0]:>16s}  {class_names[1]:>16s}")
    for i, row in enumerate(cm):
        print(f"    {class_names[i]:>16s}: {row}")

    return metrics


def evaluate_pipeline(detector, classifier, test_images_dir,
                      test_labels_dir, device, crop_size=224):
    """
    End-to-end pipeline evaluation.
    Runs detector → crops → classifier on test images.

    Returns:
        Dict with pipeline-level metrics
    """
    transform = get_val_transforms(crop_size)
    test_dir = Path(test_images_dir)
    labels_dir = Path(test_labels_dir)

    total_dogs = 0
    detected_dogs = 0
    correct_behavior = 0

    for img_path in test_dir.iterdir():
        if img_path.suffix.lower() not in (".jpg", ".jpeg", ".png"):
            continue

        img = cv2.imread(str(img_path))
        if img is None:
            continue

        # Ground truth
        lbl_path = labels_dir / f"{img_path.stem}.txt"
        gt_labels = []
        if lbl_path.exists():
            with open(lbl_path, "r") as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) >= 5:
                        gt_labels.append(int(parts[0]))
        total_dogs += len(gt_labels)

        # Detect
        detections = detector.get_dog_boxes(img)
        detected_dogs += len(detections)

        # Classify each detection
        for det in detections:
            crop = img[det["y1"]:det["y2"], det["x1"]:det["x2"]]
            if crop.size == 0:
                continue

            crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
            augmented = transform(image=crop_rgb)
            tensor = augmented["image"].unsqueeze(0).to(device)

            cls_id, conf = classifier.predict_single(tensor)

    pipeline_metrics = {
        "total_ground_truth_dogs": total_dogs,
        "total_detections": detected_dogs,
        "detection_rate": detected_dogs / total_dogs if total_dogs > 0 else 0.0,
    }

    print("\n  PIPELINE EVALUATION")
    print(f"  Ground truth dogs: {total_dogs}")
    print(f"  Detected:          {detected_dogs}")
    print(f"  Detection rate:    {pipeline_metrics['detection_rate']:.4f}")

    return pipeline_metrics


def save_evaluation_report(metrics, output_path):
    """Save all evaluation metrics to a JSON report."""
    with open(output_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"\n  Report saved to: {output_path}")
