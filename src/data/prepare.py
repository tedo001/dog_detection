"""
Data Preparation Pipeline for Dog Aggression Detection

Handles:
- Loading and validating Roboflow YOLO-format datasets
- Creating train/val/test splits if not already split
- Cropping detected dog regions for classifier training
- Data augmentation pipeline
- Dataset statistics and health checks
"""

import os
import platform as _platform
import shutil
import random
from pathlib import Path

import yaml


# On Windows, multiprocessing with spawn can't pickle local classes;
# DataLoader workers stay at 0 there.
_DEFAULT_WORKERS = 0 if _platform.system() == "Windows" else 4


def load_config(config_path="config/data.yaml"):
    """Load dataset configuration from YAML."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def validate_dataset(data_root):
    """
    Validate that the dataset structure is correct.
    Returns a report dict with counts and any issues found.
    """
    data_root = Path(data_root)
    report = {"valid": True, "issues": [], "counts": {}}

    for split in ["train", "valid", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"

        if not img_dir.exists():
            if split == "train":
                report["valid"] = False
                report["issues"].append(f"CRITICAL: {img_dir} not found")
            else:
                report["issues"].append(f"WARNING: {img_dir} not found (optional)")
            continue

        images = list(img_dir.glob("*.[jJ][pP][gG]")) + \
                 list(img_dir.glob("*.[pP][nN][gG]")) + \
                 list(img_dir.glob("*.[jJ][pP][eE][gG]"))

        labels = list(lbl_dir.glob("*.txt")) if lbl_dir.exists() else []

        img_stems = {p.stem for p in images}
        lbl_stems = {p.stem for p in labels}

        missing_labels = img_stems - lbl_stems
        orphan_labels = lbl_stems - img_stems

        report["counts"][split] = {
            "images": len(images),
            "labels": len(labels),
            "missing_labels": len(missing_labels),
            "orphan_labels": len(orphan_labels),
        }

        if missing_labels:
            report["issues"].append(
                f"{split}: {len(missing_labels)} images without labels"
            )
        if orphan_labels:
            report["issues"].append(
                f"{split}: {len(orphan_labels)} orphan label files"
            )

    return report


def create_splits(data_root, train_ratio=0.70, val_ratio=0.15, seed=42):
    """
    If only a 'train' folder exists, split it into train/valid/test.
    Preserves image-label pairs during splitting.
    """
    data_root = Path(data_root)
    train_img = data_root / "train" / "images"
    train_lbl = data_root / "train" / "labels"

    if (data_root / "valid").exists():
        print("[prepare] Splits already exist, skipping.")
        return

    images = sorted(
        list(train_img.glob("*.[jJ][pP][gG]")) +
        list(train_img.glob("*.[pP][nN][gG]"))
    )

    random.seed(seed)
    random.shuffle(images)

    n = len(images)
    n_train = int(n * train_ratio)
    n_val = int(n * val_ratio)

    splits = {
        "train": images[:n_train],
        "valid": images[n_train:n_train + n_val],
        "test": images[n_train + n_val:],
    }

    for split_name, split_images in splits.items():
        if split_name == "train":
            continue  # keep originals in place

        split_img_dir = data_root / split_name / "images"
        split_lbl_dir = data_root / split_name / "labels"
        split_img_dir.mkdir(parents=True, exist_ok=True)
        split_lbl_dir.mkdir(parents=True, exist_ok=True)

        for img_path in split_images:
            lbl_path = train_lbl / f"{img_path.stem}.txt"

            shutil.move(str(img_path), str(split_img_dir / img_path.name))
            if lbl_path.exists():
                shutil.move(str(lbl_path), str(split_lbl_dir / lbl_path.name))

    counts = {k: len(v) for k, v in splits.items()}
    print(f"[prepare] Split complete: {counts}")


def _crop_one_image(args):
    """Worker: crop all boxes from a single image. Returns count saved."""
    import cv2

    img_path, lbl_path, split_out_dir, class_map = args

    if not lbl_path.exists():
        return 0

    img = cv2.imread(str(img_path))
    if img is None:
        return 0
    h, w = img.shape[:2]

    saved = 0
    with open(lbl_path, "r") as f:
        lines = f.readlines()

    for i, line in enumerate(lines):
        parts = line.strip().split()
        if len(parts) < 5:
            continue

        cls_id = int(parts[0])
        label = class_map.get(cls_id)
        if label is None:
            continue

        cx, cy, bw, bh = map(float, parts[1:5])
        x1 = max(0, int((cx - bw / 2) * w))
        y1 = max(0, int((cy - bh / 2) * h))
        x2 = min(w, int((cx + bw / 2) * w))
        y2 = min(h, int((cy + bh / 2) * h))

        if x2 - x1 < 10 or y2 - y1 < 10:
            continue

        crop = img[y1:y2, x1:x2]
        save_path = split_out_dir / label / f"{img_path.stem}_crop{i}.jpg"
        cv2.imwrite(str(save_path), crop)
        saved += 1

    return saved


def crop_detections(data_root, output_dir, class_map=None, num_workers=None):
    """
    Crop dog bounding boxes from YOLO labels into per-class folders.
    Shows a tqdm progress bar and uses a process pool for speed.

    Args:
        data_root: Path to dataset root
        output_dir: Where to save cropped images
        class_map: Dict mapping class_id -> behavior label
                   e.g. {0: 'non_aggressive', 1: 'aggressive'}
        num_workers: Parallel workers (default: min(8, cpu_count))
    """
    from concurrent.futures import ProcessPoolExecutor, as_completed
    from tqdm import tqdm

    if class_map is None:
        class_map = {0: "non_aggressive", 1: "aggressive"}

    data_root = Path(data_root)
    output_dir = Path(output_dir)

    if num_workers is None:
        num_workers = min(8, os.cpu_count() or 4)
    # On Windows, ProcessPoolExecutor + spawn has heavy startup cost and
    # can't easily pickle nested state — run serial, it's plenty fast.
    if _platform.system() == "Windows":
        num_workers = 1

    # Pre-create every split/class directory so DogCropDataset can load even
    # splits that end up with 0 samples of a given class.
    for split in ["train", "valid", "test"]:
        for label in class_map.values():
            (output_dir / split / label).mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for split in ["train", "valid", "test"]:
        img_dir = data_root / split / "images"
        lbl_dir = data_root / split / "labels"
        split_out_dir = output_dir / split

        if not img_dir.exists():
            print(f"[prepare] Skipping {split} (no images dir)")
            continue

        images = [
            p for p in img_dir.iterdir()
            if p.suffix.lower() in (".jpg", ".jpeg", ".png")
        ]
        if not images:
            print(f"[prepare] {split}: 0 images, skipping")
            continue

        tasks = [
            (img_path, lbl_dir / f"{img_path.stem}.txt", split_out_dir, class_map)
            for img_path in images
        ]

        split_saved = 0
        desc = f"[crop] {split:<5}"

        if num_workers <= 1:
            for t in tqdm(tasks, desc=desc, unit="img"):
                split_saved += _crop_one_image(t)
        else:
            with ProcessPoolExecutor(max_workers=num_workers) as pool:
                futures = [pool.submit(_crop_one_image, t) for t in tasks]
                for fut in tqdm(as_completed(futures), total=len(futures),
                                desc=desc, unit="img"):
                    split_saved += fut.result()

        print(f"[prepare] {split}: {split_saved} crops saved")
        total_saved += split_saved

    print(f"[prepare] Done. {total_saved} total crops -> {output_dir}")


def _make_dataset(root_dir, split="train", transform=None):
    """Build a list of (image_path, label) samples from class folders."""
    root = Path(root_dir) / split
    if not root.exists():
        return [], [], {}, transform

    classes = sorted([d.name for d in root.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    samples = []
    for cls_name in classes:
        cls_dir = root / cls_name
        for img_path in cls_dir.iterdir():
            if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                samples.append((str(img_path), class_to_idx[cls_name]))

    return samples, classes, class_to_idx, transform


def get_train_transforms(crop_size=224):
    """Augmentation pipeline for training."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(crop_size, crop_size),
        A.HorizontalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.3),
        A.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1, p=0.3),
        A.GaussNoise(p=0.2),
        A.MotionBlur(blur_limit=3, p=0.1),
        A.Rotate(limit=15, p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


def get_val_transforms(crop_size=224):
    """Minimal transforms for validation/test."""
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    return A.Compose([
        A.Resize(crop_size, crop_size),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class DogCropDataset:
    """
    PyTorch-compatible Dataset for classifier training.
    DataLoader only needs __len__ and __getitem__ — no inheritance required.
    """

    def __init__(self, root_dir, split="train", transform=None):
        self.samples, self.classes, self.class_to_idx, _ = \
            _make_dataset(root_dir, split)
        self.transform = transform

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        import cv2
        img_path, label = self.samples[idx]
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.transform:
            augmented = self.transform(image=image)
            image = augmented["image"]

        return image, label


def get_dataloaders(crop_dir, batch_size=32, crop_size=224, num_workers=None):
    """Create train/val DataLoaders."""
    from torch.utils.data import DataLoader

    if num_workers is None:
        num_workers = _DEFAULT_WORKERS

    train_ds = DogCropDataset(
        crop_dir, split="train", transform=get_train_transforms(crop_size)
    )
    val_ds = DogCropDataset(
        crop_dir, split="valid", transform=get_val_transforms(crop_size)
    )

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=True, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True,
    )

    return train_loader, val_loader


def print_dataset_summary(data_root):
    """Print a summary of the dataset."""
    report = validate_dataset(data_root)

    print("\n" + "=" * 50)
    print("  DATASET SUMMARY")
    print("=" * 50)

    for split, counts in report["counts"].items():
        print(f"\n  {split.upper()}:")
        print(f"    Images: {counts['images']}")
        print(f"    Labels: {counts['labels']}")
        if counts["missing_labels"] > 0:
            print(f"    Missing labels: {counts['missing_labels']}")

    if report["issues"]:
        print(f"\n  Issues ({len(report['issues'])}):")
        for issue in report["issues"]:
            print(f"    - {issue}")

    print("\n" + "=" * 50)
    return report


if __name__ == "__main__":
    cfg = load_config("config/data.yaml")
    data_root = cfg["path"]

    print_dataset_summary(data_root)

    report = validate_dataset(data_root)
    if report["valid"]:
        create_splits(data_root)
        crop_detections(data_root, output_dir=f"{data_root}/crops")
        print("[prepare] Data preparation complete!")
    else:
        print("[prepare] Fix critical issues before proceeding.")
