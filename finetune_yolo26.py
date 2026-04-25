"""
Fine-tune YOLO26x on the dog aggression dataset.

Optimised for RTX 3050 Laptop (4GB VRAM).

Usage:
    python finetune_yolo26.py                        # default: yolo26x.pt
    python finetune_yolo26.py --model yolo26m.pt     # medium (faster)
    python finetune_yolo26.py --model yolo26x.pt --epochs 100
    python finetune_yolo26.py --resume runs/detect/yolo26x_finetune/weights/last.pt
"""

import argparse
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# Project root = the folder this file lives in
PROJECT_ROOT = Path(__file__).resolve().parent
DATA_YAML    = PROJECT_ROOT / "config" / "data.yaml"
CONFIG_YAML  = PROJECT_ROOT / "config" / "config.yaml"


def load_config():
    with open(CONFIG_YAML) as f:
        return yaml.safe_load(f)


def finetune(
    model_name: str = "yolo26x.pt",
    epochs: int = 50,
    batch: int = 8,
    imgsz: int = 640,
    lr0: float = 0.001,
    lrf: float = 0.01,
    freeze: int = 10,
    patience: int = 15,
    resume: str = None,
    device: str = "0",
    project: str = "runs/detect",
):
    from ultralytics import YOLO

    run_name = f"yolo26x_finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    print("\n" + "=" * 60)
    print("  YOLO26x FINE-TUNING")
    print("=" * 60)
    print(f"  Base model :  {model_name}")
    print(f"  Dataset    :  {DATA_YAML}")
    print(f"  Epochs     :  {epochs}")
    print(f"  Batch      :  {batch}  (RTX 3050 safe)")
    print(f"  Image size :  {imgsz}")
    print(f"  LR         :  {lr0} → {lrf} (cosine)")
    print(f"  Frozen lyrs:  {freeze}  (backbone frozen for first N layers)")
    print(f"  Patience   :  {patience}")
    print(f"  Device     :  {device}")
    print(f"  Output     :  {project}/{run_name}")
    print("=" * 60 + "\n")

    if not DATA_YAML.exists():
        print(f"ERROR: Dataset config not found: {DATA_YAML}")
        print("  Update config/data.yaml with the correct path to your dataset.")
        sys.exit(1)

    if resume:
        print(f"  Resuming from: {resume}\n")
        model = YOLO(resume)
    else:
        print(f"  Loading {model_name} (auto-downloads if not cached)...\n")
        model = YOLO(model_name)

    start = time.time()

    results = model.train(
        data=str(DATA_YAML),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        lr0=lr0,
        lrf=lrf,
        freeze=freeze,
        patience=patience,
        device=device,
        project=project,
        name=run_name,
        optimizer="AdamW",
        cos_lr=True,
        augment=True,
        mosaic=1.0,
        mixup=0.1,
        flipud=0.5,
        fliplr=0.5,
        degrees=10.0,
        translate=0.1,
        scale=0.5,
        hsv_h=0.015,
        hsv_s=0.7,
        hsv_v=0.4,
        plots=True,
        save=True,
        verbose=True,
        amp=True,        # mixed precision — halves VRAM usage on RTX 3050
        cache=False,     # set True if dataset fits in RAM (~16GB+)
        workers=4,
    )

    elapsed = time.time() - start
    best_weights = Path(project) / run_name / "weights" / "best.pt"

    print("\n" + "=" * 60)
    print("  FINE-TUNING COMPLETE")
    print("=" * 60)
    print(f"  Time      : {elapsed/60:.1f} minutes")
    print(f"  Best model: {best_weights}")
    print("=" * 60)
    print("\n  To use the fine-tuned model in app2.py:")
    print(f"    Select 'yolo26x.pt' then manually type the path:")
    print(f"    {best_weights}")
    print()

    # Validate on test set
    print("  Running validation on test set...")
    val = model.val(data=str(DATA_YAML), split="test")
    print(f"  mAP@50    : {val.box.map50:.4f}")
    print(f"  mAP@50-95 : {val.box.map:.4f}")
    print(f"  Precision : {val.box.mp:.4f}")
    print(f"  Recall    : {val.box.mr:.4f}")

    return str(best_weights)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune YOLO26x for dog detection")

    parser.add_argument("--model", type=str, default="yolo26x.pt",
                        choices=["yolo26n.pt", "yolo26s.pt", "yolo26m.pt",
                                 "yolo26l.pt", "yolo26x.pt"],
                        help="YOLO26 variant to fine-tune (default: yolo26x)")

    parser.add_argument("--epochs", type=int, default=50,
                        help="Number of training epochs (default: 50)")

    parser.add_argument("--batch", type=int, default=8,
                        help="Batch size (default: 8, safe for RTX 3050 4GB)")

    parser.add_argument("--imgsz", type=int, default=640,
                        help="Image size (default: 640)")

    parser.add_argument("--lr", type=float, default=0.001,
                        help="Initial learning rate (default: 0.001)")

    parser.add_argument("--freeze", type=int, default=10,
                        help="Number of backbone layers to freeze (default: 10)")

    parser.add_argument("--patience", type=int, default=15,
                        help="Early stopping patience (default: 15)")

    parser.add_argument("--device", type=str, default="0",
                        help="Device: 0 for GPU, cpu for CPU (default: 0)")

    parser.add_argument("--resume", type=str, default=None,
                        help="Path to last.pt to resume interrupted training")

    parser.add_argument("--project", type=str, default="runs/detect",
                        help="Output folder (default: runs/detect)")

    args = parser.parse_args()

    finetune(
        model_name=args.model,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        lr0=args.lr,
        freeze=args.freeze,
        patience=args.patience,
        device=args.device,
        resume=args.resume,
        project=args.project,
    )
