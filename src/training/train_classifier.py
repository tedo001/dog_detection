"""
Stage 2: Aggression Classifier Training

Trains a binary classifier on cropped dog images.
Supports:
- Transfer learning with backbone freezing
- Weighted loss for class imbalance
- Learning rate scheduling
- Early stopping
- Full experiment tracking
"""

import os
import json
import time
from pathlib import Path
from datetime import datetime

import yaml
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from src.models.classifier import AggressionClassifier
from src.data.prepare import get_dataloaders
from src.training.callbacks import EarlyStopping, MetricsLogger


def get_optimizer(model, cfg):
    """Create optimizer from config."""
    name = cfg.get("optimizer", "AdamW")
    lr = cfg["lr"]
    wd = cfg.get("weight_decay", 0.01)

    if name == "AdamW":
        return AdamW(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "Adam":
        return Adam(model.parameters(), lr=lr, weight_decay=wd)
    elif name == "SGD":
        return SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=wd)
    else:
        raise ValueError(f"Unknown optimizer: {name}")


def get_scheduler(optimizer, cfg, total_steps):
    """Create LR scheduler from config."""
    name = cfg.get("scheduler", "cosine")

    if name == "cosine":
        return CosineAnnealingLR(optimizer, T_max=total_steps)
    elif name == "step":
        return StepLR(optimizer, step_size=10, gamma=0.1)
    elif name == "plateau":
        return ReduceLROnPlateau(optimizer, mode="max", patience=3, factor=0.5)
    else:
        return None


def train_one_epoch(model, loader, criterion, optimizer, device):
    """Train for one epoch. Returns average loss and accuracy."""
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
    """Validate model. Returns loss, accuracy, and per-class metrics."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    avg_loss = total_loss / total
    accuracy = correct / total

    # Compute F1 for aggressive class (class 1)
    import numpy as np
    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)

    tp = ((preds == 1) & (labels_arr == 1)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return avg_loss, accuracy, {"precision": precision, "recall": recall, "f1": f1}


def train_classifier(config):
    """
    Train the aggression classifier end-to-end.

    Args:
        config: Dict from config/config.yaml

    Returns:
        Dict with training results and metrics
    """
    cls_cfg = config["classifier"]
    data_cfg = config["data"]

    # Device setup
    device_str = config["project"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"classifier_{timestamp}"
    run_dir = Path("experiments") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    print("\n" + "=" * 50)
    print("  STAGE 2: TRAINING AGGRESSION CLASSIFIER")
    print("=" * 50)
    print(f"  Backbone:   {cls_cfg['model']}")
    print(f"  Epochs:     {cls_cfg['epochs']}")
    print(f"  Batch size: {cls_cfg['batch_size']}")
    print(f"  LR:         {cls_cfg['lr']}")
    print(f"  Device:     {device}")
    print("=" * 50 + "\n")

    # Data
    data_root = config.get("data", {}).get("dataset_config", "config/data.yaml")
    with open(data_root, "r") as f:
        dcfg = yaml.safe_load(f)
    crop_dir = Path(dcfg["path"]) / "crops"

    train_loader, val_loader = get_dataloaders(
        crop_dir=str(crop_dir),
        batch_size=cls_cfg["batch_size"],
        crop_size=data_cfg.get("crop_size", 224),
    )

    # Model
    model = AggressionClassifier(
        backbone=cls_cfg["model"],
        num_classes=2,
        pretrained=cls_cfg.get("pretrained", True),
        dropout=cls_cfg.get("dropout", 0.3),
    ).to(device)

    # Loss with class weights
    weights = cls_cfg.get("class_weights", [1.0, 3.0])
    class_weights = torch.tensor(weights, dtype=torch.float32).to(device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)

    # Optimizer & scheduler
    optimizer = get_optimizer(model, cls_cfg)
    scheduler = get_scheduler(optimizer, cls_cfg, cls_cfg["epochs"])

    # Callbacks
    early_stopping = EarlyStopping(patience=cls_cfg.get("patience", 7), mode="max")
    logger = MetricsLogger(run_dir / "training_log.json")

    # Save config snapshot
    with open(run_dir / "config.yaml", "w") as f:
        yaml.dump(config, f)

    # Training loop
    start_time = time.time()
    best_f1 = 0.0

    for epoch in range(cls_cfg["epochs"]):
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )

        # Update scheduler
        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["f1"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        # Log
        epoch_data = {
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "val_f1": val_metrics["f1"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "lr": current_lr,
        }
        logger.log(epoch_data)

        print(
            f"  Epoch {epoch+1:3d}/{cls_cfg['epochs']} | "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} F1: {val_metrics['f1']:.4f} "
            f"Recall: {val_metrics['recall']:.4f}"
        )

        # Save best model
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save(run_dir / "checkpoints" / "best.pt")
            print(f"  >> New best F1: {best_f1:.4f} — model saved")

        # Early stopping
        if early_stopping.step(val_metrics["f1"]):
            print(f"\n  Early stopping at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    # Final metrics
    final_metrics = {
        "best_f1": best_f1,
        "final_val_acc": val_acc,
        "final_val_precision": val_metrics["precision"],
        "final_val_recall": val_metrics["recall"],
        "training_time_seconds": training_time,
        "epochs_trained": epoch + 1,
        "run_dir": str(run_dir),
        "run_name": run_name,
    }

    with open(run_dir / "metrics.json", "w") as f:
        json.dump(final_metrics, f, indent=2)

    print("\n" + "=" * 50)
    print("  CLASSIFIER TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Best F1:    {best_f1:.4f}")
    print(f"  Recall:     {val_metrics['recall']:.4f}")
    print(f"  Precision:  {val_metrics['precision']:.4f}")
    print(f"  Time:       {training_time:.0f}s")
    print(f"  Saved to:   {run_dir}")
    print("=" * 50 + "\n")

    return final_metrics


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_classifier(config)
