"""
Train aggression classifier model using YOLOv8
"""
import os
import sys
from pathlib import Path
import argparse

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))

try:
    from ultralytics import YOLO
    import yaml
except ImportError as e:
    print(f"❌ Required packages not installed: {e}")
    print("   Please install: pip install ultralytics pyyaml")
    sys.exit(1)


def train_classifier(dataset_path, epochs=50, imgsz=224, batch=16):
    """
    Train a YOLOv8 classifier for dog aggression detection

    Args:
        dataset_path: Path to dataset directory
        epochs: Number of training epochs
        imgsz: Image size for training
        batch: Batch size
    """
    # Check if dataset exists
    dataset_dir = Path(dataset_path)
    if not dataset_dir.exists():
        print(f"❌ Dataset not found: {dataset_path}")
        print("   Please organize your dataset as follows:")
        print("   datasets/dog_aggression/images/train/dog_calm/")
        print("   datasets/dog_aggression/images/train/dog_aggressive/")
        print("   datasets/dog_aggression/images/val/dog_calm/")
        print("   datasets/dog_aggression/images/val/dog_aggressive/")
        return

    # Check data.yaml
    data_yaml = dataset_dir / "data.yaml"
    if not data_yaml.exists():
        print(f"❌ data.yaml not found in {dataset_path}")
        print("   Creating data.yaml...")

        # Try to create data.yaml
        from src.utils import create_data_yaml
        create_data_yaml(dataset_path)

    # Verify dataset structure
    train_calm = dataset_dir / "images" / "train" / "dog_calm"
    train_aggressive = dataset_dir / "images" / "train" / "dog_aggressive"
    val_calm = dataset_dir / "images" / "val" / "dog_calm"
    val_aggressive = dataset_dir / "images" / "val" / "dog_aggressive"

    directories = [train_calm, train_aggressive, val_calm, val_aggressive]
    missing_dirs = [str(d) for d in directories if not d.exists()]

    if missing_dirs:
        print("❌ Missing dataset directories:")
        for d in missing_dirs:
            print(f"   {d}")
        print("\n📁 Please create the missing directories and add images.")
        print("   You can use: python src/utils.py to create structure.")
        return

    # Count images
    train_calm_count = len(list(train_calm.glob("*")))
    train_aggressive_count = len(list(train_aggressive.glob("*")))
    val_calm_count = len(list(val_calm.glob("*")))
    val_aggressive_count = len(list(val_aggressive.glob("*")))

    print("\n📊 Dataset Statistics:")
    print("=" * 50)
    print(f"Training images - Calm: {train_calm_count:,}")
    print(f"Training images - Aggressive: {train_aggressive_count:,}")
    print(f"Validation images - Calm: {val_calm_count:,}")
    print(f"Validation images - Aggressive: {val_aggressive_count:,}")
    print("=" * 50)

    if train_calm_count == 0 or train_aggressive_count == 0:
        print("❌ Not enough training images!")
        print("   Please add images to both dog_calm and dog_aggressive directories")
        return

    # Ask for confirmation
    print(f"\n🚀 Starting training with:")
    print(f"   Epochs: {epochs}")
    print(f"   Image size: {imgsz}")
    print(f"   Batch size: {batch}")

    response = input("\nContinue with training? (y/n): ")
    if response.lower() != 'y':
        print("❌ Training cancelled")
        return

    # Create model directory
    model_dir = Path("models")
    model_dir.mkdir(exist_ok=True)

    # Initialize model
    print("\n📦 Initializing YOLOv8 classification model...")
    model = YOLO('yolov8n-cls.pt')  # Classification model

    # Train the model
    print("\n🏋️ Starting training...")
    print("   This may take several minutes to hours depending on dataset size and epochs.")
    print("   Press Ctrl+C to interrupt training.\n")

    try:
        results = model.train(
            data=str(data_yaml),
            epochs=epochs,
            imgsz=imgsz,
            batch=batch,
            name='dog_aggression_classifier',
            project='models',
            exist_ok=True,
            verbose=True,
            patience=10,  # Early stopping patience
            workers=4,  # Number of worker threads
            seed=42  # Random seed for reproducibility
        )

        print("\n✅ Training completed successfully!")

        # Save final model
        best_model_path = Path("models/dog_aggression_classifier/weights/best.pt")
        if best_model_path.exists():
            print(f"📁 Best model saved: {best_model_path}")

            # Copy to main models directory for easy access
            import shutil
            final_model_path = model_dir / "aggression_classifier.pt"
            shutil.copy(best_model_path, final_model_path)
            print(f"📁 Model copied to: {final_model_path}")

            # Display usage instructions
            print("\n🎯 USAGE INSTRUCTIONS:")
            print("=" * 50)
            print(f"To use this model in the main system:")
            print(f"python src/main.py --classifier-model {final_model_path}")
            print("\nOr modify aggression_classifier.py to load it by default.")

        # Show training results summary
        if hasattr(results, 'results_dict'):
            print("\n📊 Training Results Summary:")
            for key, value in results.results_dict.items():
                print(f"   {key}: {value:.4f}")

    except KeyboardInterrupt:
        print("\n⚠️ Training interrupted by user")
    except Exception as e:
        print(f"\n❌ Training error: {e}")
        print("   Please check your dataset and try again.")


def collect_sample_images():
    """Guide for collecting sample images"""
    print("\n📸 SAMPLE IMAGE COLLECTION GUIDE")
    print("=" * 50)
    print("To train a good classifier, you need images of:")
    print("\n🐕 CALM DOGS:")
    print("   - Sitting or lying down calmly")
    print("   - Relaxed posture")
    print("   - Normal facial expression")
    print("   - Ears in natural position")
    print("   - Tail down or relaxed")

    print("\n🐕 AGGRESSIVE DOGS:")
    print("   - Baring teeth")
    print("   - Ears forward/back")
    print("   - Raised hackles (hair on back)")
    print("   - Stiff posture")
    print("   - Intense stare")
    print("   - Growling posture")

    print("\n📁 ORGANIZATION:")
    print("   Place calm dog images in: training/datasets/dog_aggression/images/train/dog_calm/")
    print("   Place aggressive dog images in: training/datasets/dog_aggression/images/train/dog_aggressive/")
    print("\n   Use 80% for training, 20% for validation")

    print("\n💡 TIPS:")
    print("   - Collect at least 100-200 images per class")
    print("   - Use diverse breeds, sizes, and colors")
    print("   - Include different lighting conditions")
    print("   - Both close-up and full-body shots")

    print("=" * 50)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Train Dog Aggression Classifier')
    parser.add_argument('--dataset', type=str, default='training/datasets/dog_aggression',
                        help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')
    parser.add_argument('--collect-guide', action='store_true',
                        help='Show image collection guide')

    args = parser.parse_args()

    if args.collect_guide:
        collect_sample_images()
    else:
        train_classifier(
            dataset_path=args.dataset,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch
        )