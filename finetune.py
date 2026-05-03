"""
Dog Aggression Detection - Fine-Tune Pipeline
==============================================
Takes a folder of annotated images and fine-tunes the YOLO detector.

Dataset folder must follow YOLO format:
    your_dataset/
    ├── images/
    │   ├── img1.jpg
    │   └── img2.jpg
    └── labels/
        ├── img1.txt    (YOLO format: class cx cy w h, normalised)
        └── img2.txt

  OR already split:
    your_dataset/
    ├── train/images/ & train/labels/
    └── valid/images/ & valid/labels/

Commands
--------
  # Basic fine-tune (auto train/val split 80/20):
  python finetune.py --data D:/my_dog_photos

  # Fine-tune from a specific base model:
  python finetune.py --data D:/my_dog_photos --base-model best.pt

  # Custom epochs / batch size:
  python finetune.py --data D:/my_dog_photos --epochs 30 --batch 8

  # Freeze backbone (only train detection head — faster, less data needed):
  python finetune.py --data D:/my_dog_photos --freeze 10

  # Full options:
  python finetune.py --data D:/my_dog_photos --base-model yolo26x.pt --epochs 30 --batch 8 --freeze 10 --device cuda:0
  only for a cuda computing and its only better for a gpu setup above 3050 rtx
"""

import argparse
import shutil
import sys
import time
from datetime import datetime
from pathlib import Path

import yaml


# ──────────────────────────────────────────────
# STEP 1 — Validate & auto-split dataset
# ──────────────────────────────────────────────

def validate_dataset(data_path: Path) -> dict:
    """Check folder structure and count images / labels."""
    print("\n[MLOps] STEP 1 — Validating dataset ...")

    if not data_path.exists():
        print(f"  ERROR: Path not found: {data_path}")
        sys.exit(1)

    # Detect if already split
    has_split = (data_path / "train").exists() and (data_path / "valid").exists()

    if has_split:
        train_imgs  = list((data_path / "train" / "images").glob("*.[jJpP][pPnN][gG]*"))
        train_lbls  = list((data_path / "train" / "labels").glob("*.txt"))
        val_imgs    = list((data_path / "valid" / "images").glob("*.[jJpP][pPnN][gG]*"))
        val_lbls    = list((data_path / "valid" / "labels").glob("*.txt"))
        print(f"  Pre-split dataset found.")
        print(f"  Train  — {len(train_imgs)} images, {len(train_lbls)} labels")
        print(f"  Valid  — {len(val_imgs)} images, {len(val_lbls)} labels")
        if len(train_imgs) == 0:
            print("  ERROR: No training images found.")
            sys.exit(1)
        return {"split": True, "root": data_path,
                "train_count": len(train_imgs), "val_count": len(val_imgs)}

    # Flat structure — collect and validate
    img_dir = data_path / "images"
    lbl_dir = data_path / "labels"

    if not img_dir.exists():
        print(f"  ERROR: Expected 'images/' folder inside {data_path}")
        sys.exit(1)
    if not lbl_dir.exists():
        print(f"  ERROR: Expected 'labels/' folder inside {data_path}")
        sys.exit(1)

    images = sorted(img_dir.glob("*.[jJpP][pPnN][gG]*"))
    labels = sorted(lbl_dir.glob("*.txt"))

    print(f"  Found {len(images)} images, {len(labels)} label files")

    if len(images) == 0:
        print("  ERROR: No images found in images/ folder.")
        sys.exit(1)

    # Warn about unpaired files
    img_stems = {p.stem for p in images}
    lbl_stems = {p.stem for p in labels}
    unlabelled = img_stems - lbl_stems
    if unlabelled:
        print(f"  WARNING: {len(unlabelled)} image(s) have no label — they will be skipped by YOLO.")

    return {"split": False, "root": data_path,
            "images": images, "labels": labels}


# ──────────────────────────────────────────────
# STEP 2 — Auto-split into train / valid
# ──────────────────────────────────────────────

def auto_split(info: dict, val_ratio: float = 0.2, seed: int = 42) -> Path:
    """Split flat images+labels into train/valid inside a temp folder."""
    if info["split"]:
        return info["root"]   # already split, use as-is

    import random
    random.seed(seed)

    images = info["images"]
    random.shuffle(images)

    split_idx  = max(1, int(len(images) * (1 - val_ratio)))
    train_imgs = images[:split_idx]
    val_imgs   = images[split_idx:]

    print(f"\n[MLOps] STEP 2 — Auto-splitting dataset (80% train / 20% val) ...")
    print(f"  Train: {len(train_imgs)} images | Val: {len(val_imgs)} images")

    out_root = info["root"] / "_mlops_split"
    lbl_dir  = info["root"] / "labels"

    for subset, subset_imgs in [("train", train_imgs), ("valid", val_imgs)]:
        (out_root / subset / "images").mkdir(parents=True, exist_ok=True)
        (out_root / subset / "labels").mkdir(parents=True, exist_ok=True)
        for img_path in subset_imgs:
            shutil.copy2(img_path, out_root / subset / "images" / img_path.name)
            lbl_path = lbl_dir / (img_path.stem + ".txt")
            if lbl_path.exists():
                shutil.copy2(lbl_path, out_root / subset / "labels" / lbl_path.name)

    return out_root


# ──────────────────────────────────────────────
# STEP 3 — Write data.yaml for YOLO
# ──────────────────────────────────────────────

def write_data_yaml(split_root: Path) -> Path:
    """Generate a data.yaml pointing at the split dataset."""
    print("\n[MLOps] STEP 3 — Writing data.yaml ...")

    data_yaml = {
        "path":  str(split_root.resolve()),
        "train": "train/images",
        "val":   "valid/images",
        "nc":    2,
        "names": {0: "dog", 1: "aggressive_dog"},
    }

    yaml_path = split_root / "finetune_data.yaml"
    with open(yaml_path, "w") as f:
        yaml.dump(data_yaml, f, default_flow_style=False, sort_keys=False)

    print(f"  Saved: {yaml_path}")
    return yaml_path


# ──────────────────────────────────────────────
# STEP 4 — Fine-tune YOLO model
# ──────────────────────────────────────────────

def finetune(base_model: str, data_yaml: Path, epochs: int,
             batch: int, freeze: int, device: str, imgsz: int) -> Path:
    """Load base model and fine-tune on annotated dataset."""
    print(f"\n[MLOps] STEP 4 — Fine-tuning model ...")
    print(f"  Base model : {base_model}")
    print(f"  Epochs     : {epochs}")
    print(f"  Batch size : {batch}")
    print(f"  Freeze     : first {freeze} layers (backbone)")
    print(f"  Device     : {device}")

    try:
        from ultralytics import YOLO
    except ImportError:
        print("  ERROR: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)

    run_name = f"finetune_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    model    = YOLO(base_model)

    results = model.train(
        data      = str(data_yaml),
        epochs    = epochs,
        batch     = batch,
        imgsz     = imgsz,
        device    = device,
        freeze    = freeze if freeze > 0 else None,
        name      = run_name,
        project   = "experiments",
        exist_ok  = True,
        patience  = 10,          # early stopping
        optimizer = "AdamW",
        lr0       = 0.001,       # lower LR for fine-tuning
        lrf       = 0.01,
        momentum  = 0.937,
        weight_decay = 0.0005,
        augment   = True,
        mosaic    = 0.5,
        mixup     = 0.0,
        verbose   = True,
    )

    best_weights = Path("experiments") / run_name / "weights" / "best.pt"
    return best_weights


# ──────────────────────────────────────────────
# STEP 5 — Evaluate fine-tuned model
# ──────────────────────────────────────────────

def evaluate(weights_path: Path, data_yaml: Path, device: str):
    """Run validation and print mAP metrics."""
    print(f"\n[MLOps] STEP 5 — Evaluating fine-tuned model ...")

    if not weights_path.exists():
        print(f"  ERROR: Weights not found at {weights_path}")
        return

    from ultralytics import YOLO
    model   = YOLO(str(weights_path))
    metrics = model.val(data=str(data_yaml), device=device, verbose=False)

    print(f"\n  ┌─────────────────────────────────┐")
    print(f"  │  Fine-tune Evaluation Results   │")
    print(f"  ├─────────────────────────────────┤")
    print(f"  │  mAP@0.5      : {metrics.box.map50:.4f}            │")
    print(f"  │  mAP@0.5:0.95 : {metrics.box.map:.4f}            │")
    print(f"  │  Precision    : {metrics.box.mp:.4f}            │")
    print(f"  │  Recall       : {metrics.box.mr:.4f}            │")
    print(f"  └─────────────────────────────────┘")


# ──────────────────────────────────────────────
# STEP 6 — Copy best model to project root
# ──────────────────────────────────────────────

def promote_model(weights_path: Path, output_name: str = "best_finetuned.pt"):
    """Copy the fine-tuned best.pt to the project root for immediate use."""
    if not weights_path.exists():
        print(f"\n  WARNING: Could not promote model — {weights_path} not found.")
        return

    dest = Path(output_name)
    shutil.copy2(weights_path, dest)
    print(f"\n[MLOps] STEP 6 — Model promoted to: {dest.resolve()}")
    print(f"  Use it in app2.py by placing '{output_name}' in the project folder")
    print(f"  and selecting it as the detector model.\n")


# ──────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="MLOps Fine-Tune Pipeline — Dog Aggression Detection",
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("--data",       required=True,         help="Path to annotated dataset folder")
    parser.add_argument("--base-model", default="best.pt",     help="Starting weights (default: best.pt)")
    parser.add_argument("--epochs",     type=int, default=20,  help="Training epochs (default: 20)")
    parser.add_argument("--batch",      type=int, default=8,   help="Batch size (default: 8)")
    parser.add_argument("--freeze",     type=int, default=10,  help="Freeze first N backbone layers (default: 10)")
    parser.add_argument("--imgsz",      type=int, default=640, help="Input image size (default: 640)")
    parser.add_argument("--device",     default="auto",        help="Device: auto / cpu / cuda:0 (default: auto)")
    parser.add_argument("--val-ratio",  type=float, default=0.2, help="Validation split ratio (default: 0.2)")
    parser.add_argument("--output",     default="best_finetuned.pt", help="Output model filename")
    args = parser.parse_args()

    print("\n" + "=" * 55)
    print("  DOG AGGRESSION DETECTION — MLOps Fine-Tune Pipeline")
    print("=" * 55)
    print(f"  Dataset    : {args.data}")
    print(f"  Base model : {args.base_model}")
    print(f"  Epochs     : {args.epochs}  |  Batch: {args.batch}")
    print(f"  Freeze     : {args.freeze} backbone layers")
    print("=" * 55)

    t_start = time.time()

    # Pipeline stages
    data_path  = Path(args.data)
    info       = validate_dataset(data_path)
    split_root = auto_split(info, val_ratio=args.val_ratio)
    data_yaml  = write_data_yaml(split_root)
    weights    = finetune(
        base_model = args.base_model,
        data_yaml  = data_yaml,
        epochs     = args.epochs,
        batch      = args.batch,
        freeze     = args.freeze,
        device     = args.device,
        imgsz      = args.imgsz,
    )
    evaluate(weights, data_yaml, args.device)
    promote_model(weights, args.output)

    elapsed = time.time() - t_start
    print(f"  Total time: {int(elapsed // 60)}m {int(elapsed % 60)}s")
    print("=" * 55 + "\n")


if __name__ == "__main__":
    main()
