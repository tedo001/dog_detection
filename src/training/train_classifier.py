"""
Stage 2: Aggression Classifier Training

Trains a binary classifier on cropped dog images.
Features:
- Live training graph (loss, accuracy, F1 update in real-time)
- Progress bar with percentage per epoch
- Transfer learning with backbone freezing
- Weighted loss for class imbalance
- Learning rate scheduling
- Early stopping
- Full experiment tracking
"""

import os
import sys
import json
import time
from pathlib import Path
from datetime import datetime

import yaml
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW, SGD, Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, StepLR, ReduceLROnPlateau

from src.models.classifier import AggressionClassifier
from src.data.prepare import get_dataloaders
from src.training.callbacks import EarlyStopping, MetricsLogger


class LiveTrainingPlot:
    """Real-time training graph using matplotlib."""

    def __init__(self, save_dir):
        import matplotlib
        matplotlib.use("TkAgg")
        import matplotlib.pyplot as plt

        self.plt = plt
        self.save_dir = Path(save_dir)
        self.fig, self.axes = plt.subplots(2, 2, figsize=(14, 9))
        self.fig.suptitle("Dog Aggression Classifier — Live Training", fontsize=14, fontweight="bold")
        self.fig.set_facecolor("#1e1e1e")

        for ax in self.axes.flat:
            ax.set_facecolor("#252525")
            ax.tick_params(colors="#cccccc")
            ax.xaxis.label.set_color("#cccccc")
            ax.yaxis.label.set_color("#cccccc")
            ax.title.set_color("#ffffff")
            for spine in ax.spines.values():
                spine.set_color("#444444")

        self.history = {
            "train_loss": [], "val_loss": [],
            "train_acc": [], "val_acc": [],
            "val_f1": [], "val_precision": [], "val_recall": [],
            "lr": [],
        }

        plt.ion()
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        plt.show(block=False)

    def update(self, epoch_data):
        """Update all 4 graphs with new epoch data."""
        for key in self.history:
            if key in epoch_data:
                self.history[key].append(epoch_data[key])

        epochs = list(range(1, len(self.history["train_loss"]) + 1))

        # Top-left: Loss
        ax = self.axes[0, 0]
        ax.clear()
        ax.plot(epochs, self.history["train_loss"], "o-", color="#3b82f6", label="Train Loss", linewidth=2, markersize=4)
        ax.plot(epochs, self.history["val_loss"], "o-", color="#ef4444", label="Val Loss", linewidth=2, markersize=4)
        ax.set_title("Loss")
        ax.set_xlabel("Epoch")
        ax.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")
        ax.grid(True, alpha=0.2)

        # Top-right: Accuracy
        ax = self.axes[0, 1]
        ax.clear()
        ax.plot(epochs, self.history["train_acc"], "o-", color="#3b82f6", label="Train Acc", linewidth=2, markersize=4)
        ax.plot(epochs, self.history["val_acc"], "o-", color="#22c55e", label="Val Acc", linewidth=2, markersize=4)
        ax.set_title("Accuracy")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")
        ax.grid(True, alpha=0.2)

        # Bottom-left: F1 / Precision / Recall
        ax = self.axes[1, 0]
        ax.clear()
        ax.plot(epochs, self.history["val_f1"], "o-", color="#f59e0b", label="F1", linewidth=2, markersize=4)
        ax.plot(epochs, self.history["val_precision"], "s--", color="#8b5cf6", label="Precision", linewidth=1.5, markersize=3)
        ax.plot(epochs, self.history["val_recall"], "^--", color="#ec4899", label="Recall", linewidth=1.5, markersize=3)
        ax.set_title("Validation Metrics")
        ax.set_xlabel("Epoch")
        ax.set_ylim(0, 1.05)
        ax.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")
        ax.grid(True, alpha=0.2)

        # Bottom-right: Learning Rate
        ax = self.axes[1, 1]
        ax.clear()
        ax.plot(epochs, self.history["lr"], "o-", color="#06b6d4", label="LR", linewidth=2, markersize=4)
        ax.set_title("Learning Rate Schedule")
        ax.set_xlabel("Epoch")
        ax.legend(facecolor="#333333", edgecolor="#555555", labelcolor="#cccccc")
        ax.grid(True, alpha=0.2)
        ax.ticklabel_format(style="sci", axis="y", scilimits=(0, 0))

        # Style
        for ax in self.axes.flat:
            ax.set_facecolor("#252525")
            ax.tick_params(colors="#cccccc")
            ax.title.set_color("#ffffff")
            for spine in ax.spines.values():
                spine.set_color("#444444")

        self.fig.suptitle("Dog Aggression Classifier — Live Training", fontsize=14, fontweight="bold", color="#ffffff")
        self.plt.tight_layout(rect=[0, 0, 1, 0.95])
        self.plt.draw()
        self.plt.pause(0.1)

        # Save graph
        self.fig.savefig(
            self.save_dir / "training_graph.png",
            facecolor="#1e1e1e", dpi=150, bbox_inches="tight"
        )

    def close(self):
        self.plt.ioff()
        self.plt.close()


def print_progress_bar(epoch, total_epochs, train_loss, val_loss, val_f1, elapsed):
    """Print a progress bar with percentage."""
    pct = (epoch / total_epochs) * 100
    bar_len = 30
    filled = int(bar_len * epoch / total_epochs)
    bar = "█" * filled + "░" * (bar_len - filled)

    eta = (elapsed / epoch) * (total_epochs - epoch) if epoch > 0 else 0

    sys.stdout.write(
        f"\r  [{bar}] {pct:5.1f}% | "
        f"Epoch {epoch}/{total_epochs} | "
        f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
        f"F1: {val_f1:.4f} | "
        f"ETA: {eta:.0f}s  "
    )
    sys.stdout.flush()


def get_optimizer(model, cfg):
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
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0
    batch_count = len(loader)

    for i, (images, labels) in enumerate(loader):
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

        # Mini progress for batches
        batch_pct = (i + 1) / batch_count * 100
        sys.stdout.write(f"\r    Batch {i+1}/{batch_count} ({batch_pct:.0f}%)")
        sys.stdout.flush()

    sys.stdout.write("\r" + " " * 60 + "\r")
    avg_loss = total_loss / total
    accuracy = correct / total
    return avg_loss, accuracy


def validate(model, loader, criterion, device):
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

    preds = np.array(all_preds)
    labels_arr = np.array(all_labels)

    tp = ((preds == 1) & (labels_arr == 1)).sum()
    fp = ((preds == 1) & (labels_arr == 0)).sum()
    fn = ((preds == 0) & (labels_arr == 1)).sum()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return avg_loss, accuracy, {"precision": float(precision), "recall": float(recall), "f1": float(f1)}


def train_classifier(config):
    """
    Train the aggression classifier with live graphs and progress tracking.
    """
    cls_cfg = config["classifier"]
    data_cfg = config["data"]

    device_str = config["project"].get("device", "auto")
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"classifier_{timestamp}"
    run_dir = Path("experiments") / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "checkpoints").mkdir(exist_ok=True)

    print("\n" + "=" * 60)
    print("  STAGE 2: TRAINING AGGRESSION CLASSIFIER")
    print("=" * 60)
    print(f"  Backbone:   {cls_cfg['model']}")
    print(f"  Epochs:     {cls_cfg['epochs']}")
    print(f"  Batch size: {cls_cfg['batch_size']}")
    print(f"  LR:         {cls_cfg['lr']}")
    print(f"  Device:     {device}")
    print(f"  GPU:        {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'N/A'}")
    print(f"  Output:     {run_dir}")
    print("=" * 60 + "\n")

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

    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Val samples:   {len(val_loader.dataset)}")
    print()

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

    # Live graph
    live_plot = None
    try:
        live_plot = LiveTrainingPlot(save_dir=run_dir)
        print("  Live training graph opened.\n")
    except Exception:
        print("  (Live graph not available — training in headless mode)\n")

    # Training loop
    start_time = time.time()
    best_f1 = 0.0
    total_epochs = cls_cfg["epochs"]

    print(f"  {'Epoch':>5} | {'Train Loss':>10} {'Train Acc':>10} | "
          f"{'Val Loss':>10} {'Val F1':>8} {'Recall':>8} | {'LR':>10} | Status")
    print("  " + "-" * 88)

    for epoch in range(total_epochs):
        epoch_start = time.time()

        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device
        )
        val_loss, val_acc, val_metrics = validate(
            model, val_loader, criterion, device
        )

        if scheduler:
            if isinstance(scheduler, ReduceLROnPlateau):
                scheduler.step(val_metrics["f1"])
            else:
                scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        elapsed = time.time() - start_time

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

        # Progress bar
        pct = ((epoch + 1) / total_epochs) * 100
        eta = (elapsed / (epoch + 1)) * (total_epochs - epoch - 1)

        status = ""
        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            model.save(run_dir / "checkpoints" / "best.pt")
            status = "*** BEST ***"

        print(
            f"  {epoch+1:3d}/{total_epochs} | "
            f"{train_loss:10.4f} {train_acc:10.4f} | "
            f"{val_loss:10.4f} {val_metrics['f1']:8.4f} {val_metrics['recall']:8.4f} | "
            f"{current_lr:10.6f} | {status}"
        )

        # Progress bar
        bar_len = 40
        filled = int(bar_len * (epoch + 1) / total_epochs)
        bar = "█" * filled + "░" * (bar_len - filled)
        print(f"  [{bar}] {pct:.1f}%  |  ETA: {eta:.0f}s  |  Best F1: {best_f1:.4f}")

        # Update live graph
        if live_plot:
            try:
                live_plot.update(epoch_data)
            except Exception:
                pass

        # Early stopping
        if early_stopping.step(val_metrics["f1"]):
            print(f"\n  Early stopping triggered at epoch {epoch+1}")
            break

    training_time = time.time() - start_time

    if live_plot:
        try:
            live_plot.close()
        except Exception:
            pass

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

    print("\n" + "=" * 60)
    print("  CLASSIFIER TRAINING COMPLETE")
    print("=" * 60)
    print(f"  Best F1:    {best_f1:.4f}")
    print(f"  Recall:     {val_metrics['recall']:.4f}")
    print(f"  Precision:  {val_metrics['precision']:.4f}")
    print(f"  Epochs:     {epoch+1}/{total_epochs}")
    print(f"  Time:       {training_time:.0f}s")
    print(f"  Graph:      {run_dir}/training_graph.png")
    print(f"  Weights:    {run_dir}/checkpoints/best.pt")
    print("=" * 60 + "\n")

    return final_metrics


if __name__ == "__main__":
    with open("config/config.yaml", "r") as f:
        config = yaml.safe_load(f)
    train_classifier(config)
