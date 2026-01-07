# Train classifier 
"""
Train Dog Aggression Classifier using YOLOv8 Classification
"""
from ultralytics import YOLO
import os
from pathlib import Path


def train_aggression_classifier():
    """Train YOLOv8 classification model for dog aggression"""

    # Set dataset path
    dataset_path = Path("datasets/dog_aggression")

    # Verify dataset structure
    if not dataset_path.exists():
        print(f"ERROR: Dataset not found at {dataset_path}")
        print("Please create dataset with structure:")
        print("  datasets/dog_aggression/images/train/dog_calm/")
        print("  datasets/dog_aggression/images/train/dog_aggressive/")
        print("  datasets/dog_aggression/images/val/dog_calm/")
        print("  datasets/dog_aggression/images/val/dog_aggressive/")
        return

    # Check if training images exist
    train_calm = dataset_path / "images" / "train" / "dog_calm"
    train_aggressive = dataset_path / "images" / "train" / "dog_aggressive"

    if not train_calm.exists() or not train_aggressive.exists():
        print("ERROR: Training folders not found. Check dataset structure.")
        return

    print(f"Found {len(list(train_calm.glob('*.jpg')))} calm training images")
    print(f"Found {len(list(train_aggressive.glob('*.jpg')))} aggressive training images")

    # Load model
    print("\nLoading YOLOv8 classification model...")
    model = YOLO('yolov8n-cls.pt')  # Use classification model

    # Train the model
    print("Starting training...")
    results = model.train(
        data=str(dataset_path),
        epochs=50,
        imgsz=224,
        batch=16,
        device='cpu',  # Use '0' for GPU if available
        workers=0,  # Windows-safe (0 workers for CPU)
        project='../models',
        name='dog_aggression_cls',
        exist_ok=True,
        verbose=True,
        patience=10,  # Stop if no improvement for 10 epochs
        lr0=0.01,  # Initial learning rate
        lrf=0.01,  # Final learning rate factor
        dropout=0.1  # Regularization
    )

    print("\nTraining complete!")
    print(f"Model saved to: {results.save_dir}")

    # Validate the model
    print("\nRunning validation...")
    metrics = model.val()
    print(f"Validation accuracy: {metrics.top1:.2f}%")

    # Export model if needed
    print("\nExporting model to ONNX format...")
    model.export(format='onnx')
    print("Export complete!")


if __name__ == "__main__":
    train_aggression_classifier()