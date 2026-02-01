"""
TRAIN DOG AGGRESSION CLASSIFIER with YOLOv8
Complete training system with data collection guidance
"""
import os
import sys
import shutil
import cv2
import numpy as np
from pathlib import Path
import argparse
import yaml

print("=" * 60)
print("🐕 DOG AGGRESSION CLASSIFIER TRAINING SYSTEM")
print("=" * 60)

# Check dependencies
try:
    from ultralytics import YOLO

    print("✅ ultralytics available")
except ImportError:
    print("❌ ultralytics not installed!")
    print("💡 Run: pip install ultralytics")
    sys.exit(1)


class DogAggressionTrainer:
    """Complete training system for dog aggression classification"""

    def __init__(self):
        self.dataset_path = Path("datasets/dog_aggression")
        self.model = None

    def create_dataset_structure(self):
        """Create proper dataset structure"""
        print("\n📁 Creating dataset structure...")

        directories = [
            "images/train/dog_calm",
            "images/train/dog_aggressive",
            "images/val/dog_calm",
            "images/val/dog_aggressive",
            "labels/train",
            "labels/val",
        ]

        for dir_path in directories:
            full_path = self.dataset_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"  Created: {dir_path}")

        print("✅ Dataset structure created")

    def create_data_yaml(self):
        """Create data.yaml for YOLO training"""
        yaml_content = f"""# Dog Aggression Dataset Configuration
path: {self.dataset_path.absolute()}
train: images/train
val: images/val

# Classes
names:
  0: dog_calm
  1: dog_aggressive
"""

        yaml_path = self.dataset_path / "data.yaml"
        yaml_path.write_text(yaml_content)
        print(f"📝 Created data.yaml: {yaml_path}")

        return yaml_path

    def collect_sample_images(self):
        """Guide for collecting sample images"""
        print("\n" + "=" * 60)
        print("📸 SAMPLE IMAGE COLLECTION GUIDE")
        print("=" * 60)

        print("\n🎯 You need images of two classes:")

        print("\n🐕 DOG_CALM (label: 0):")
        print("   - Sitting or lying down calmly")
        print("   - Relaxed posture, normal breathing")
        print("   - Ears in natural position")
        print("   - Tail down or relaxed wagging")
        print("   - Mouth closed or slightly open")
        print("   - Examples: sleeping, resting, gentle playing")

        print("\n🐕 DOG_AGGRESSIVE (label: 1):")
        print("   - Baring teeth, snarling")
        print("   - Ears forward/back, raised hackles")
        print("   - Stiff posture, intense stare")
        print("   - Growling posture, raised lips")
        print("   - Tail stiff and high")
        print("   - Examples: barking aggressively, lunging")

        print("\n📁 Save images in:")
        print(f"   Calm: {self.dataset_path}/images/train/dog_calm/")
        print(f"   Aggressive: {self.dataset_path}/images/train/dog_aggressive/")

        print("\n📊 Requirements:")
        print("   - Minimum: 50 images per class")
        print("   - Good: 100-200 images per class")
        print("   - Best: 200+ images per class")

        print("\n💡 Tips:")
        print("   1. Use diverse breeds, sizes, colors")
        print("   2. Different lighting conditions")
        print("   3. Various angles (front, side, etc.)")
        print("   4. Both close-up and full-body shots")
        print("   5. Split 80% train, 20% validation")

        input("\nPress Enter when you have collected images...")

    def validate_dataset(self):
        """Validate dataset before training"""
        print("\n🔍 Validating dataset...")

        train_calm = len(list((self.dataset_path / "images/train/dog_calm").glob("*")))
        train_agg = len(list((self.dataset_path / "images/train/dog_aggressive").glob("*")))
        val_calm = len(list((self.dataset_path / "images/val/dog_calm").glob("*")))
        val_agg = len(list((self.dataset_path / "images/val/dog_aggressive").glob("*")))

        print(f"📊 Dataset Statistics:")
        print(f"   Training - Calm: {train_calm:,} images")
        print(f"   Training - Aggressive: {train_agg:,} images")
        print(f"   Validation - Calm: {val_calm:,} images")
        print(f"   Validation - Aggressive: {val_agg:,} images")

        total_train = train_calm + train_agg
        total_val = val_calm + val_agg

        if total_train == 0:
            print("❌ No training images found!")
            return False

        if train_calm == 0 or train_agg == 0:
            print("❌ Need images for BOTH classes!")
            return False

        # Check for common image formats
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp'}
        images = list((self.dataset_path / "images/train/dog_calm").glob("*"))

        if images:
            ext = images[0].suffix.lower()
            if ext not in valid_extensions:
                print(f"❌ Invalid image format: {ext}")
                print(f"   Use: {valid_extensions}")
                return False

        print("✅ Dataset validation passed")
        return True

    def augment_dataset(self):
        """Create augmented dataset if needed"""
        print("\n🔧 Creating augmented dataset...")

        # Check if we need augmentation
        train_calm = len(list((self.dataset_path / "images/train/dog_calm").glob("*")))
        train_agg = len(list((self.dataset_path / "images/train/dog_aggressive").glob("*")))

        if train_calm < 100 or train_agg < 100:
            print("⚠️ Small dataset detected (<100 images per class)")
            print("   Consider adding more images or using augmentation")

            choice = input("\nCreate augmented images? (y/n): ").lower()
            if choice == 'y':
                self._create_augmented_images()

        return True

    def _create_augmented_images(self):
        """Create augmented images"""
        print("   Creating augmented images...")

        # For each class
        for class_name in ['dog_calm', 'dog_aggressive']:
            source_dir = self.dataset_path / "images/train" / class_name
            images = list(source_dir.glob("*"))

            if not images:
                continue

            print(f"   Augmenting {class_name}...")

            for img_path in images[:5]:  # Augment first 5 images
                try:
                    img = cv2.imread(str(img_path))
                    if img is None:
                        continue

                    # Create different augmentations
                    augmentations = [
                        ("flip", cv2.flip(img, 1)),  # Horizontal flip
                        ("bright", cv2.convertScaleAbs(img, alpha=1.2, beta=30)),  # Brighter
                        ("dark", cv2.convertScaleAbs(img, alpha=0.8, beta=-30)),  # Darker
                    ]

                    for aug_name, aug_img in augmentations:
                        base_name = img_path.stem
                        save_path = source_dir / f"{base_name}_{aug_name}.jpg"
                        cv2.imwrite(str(save_path), aug_img)

                except Exception as e:
                    print(f"   Error augmenting {img_path}: {e}")

        print("   ✅ Augmentation complete")

    def train_model(self, epochs=50, imgsz=224, batch=16):
        """Train the classification model"""
        print("\n" + "=" * 60)
        print("🏋️ TRAINING MODEL")
        print("=" * 60)

        # Check dataset
        if not self.validate_dataset():
            print("❌ Cannot start training - fix dataset issues first")
            return

        # Create data.yaml if not exists
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            self.create_data_yaml()

        print(f"\n📋 Training Configuration:")
        print(f"   Epochs: {epochs}")
        print(f"   Image size: {imgsz}")
        print(f"   Batch size: {batch}")
        print(f"   Dataset: {self.dataset_path}")

        # Initialize model
        print("\n📦 Initializing YOLOv8 classification model...")
        try:
            self.model = YOLO('yolov8n-cls.pt')  # Classification model
        except:
            print("⚠️ Using base YOLOv8 model instead")
            self.model = YOLO('yolov8n.pt')

        # Train
        print("\n🚀 Starting training...")
        print("   This may take several minutes to hours.")
        print("   Press Ctrl+C to interrupt training.\n")

        try:
            results = self.model.train(
                data=str(yaml_path),
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                name='dog_aggression_classifier',
                project='models',
                exist_ok=True,
                verbose=True,
                patience=10,  # Early stopping
                workers=4,  # Number of workers
                seed=42,  # Random seed
                plots=True,  # Generate plots
                save=True,  # Save best model
                pretrained=True  # Use pretrained weights
            )

            print("\n✅ Training completed successfully!")
            self._save_trained_model()

        except KeyboardInterrupt:
            print("\n⚠️ Training interrupted by user")
        except Exception as e:
            print(f"\n❌ Training error: {e}")
            import traceback
            traceback.print_exc()

    def _save_trained_model(self):
        """Save the trained model"""
        # Check for best model
        best_model_path = Path("models/dog_aggression_classifier/weights/best.pt")

        if best_model_path.exists():
            # Create models directory if not exists
            models_dir = Path("models")
            models_dir.mkdir(exist_ok=True)

            # Copy best model
            final_path = models_dir / "aggression_classifier.pt"
            shutil.copy(best_model_path, final_path)

            print(f"📁 Best model saved: {final_path}")

            # Also save as ONNX for compatibility
            try:
                onnx_path = models_dir / "aggression_classifier.onnx"
                self.model.export(format='onnx')
                print(f"📁 ONNX model saved: {onnx_path}")
            except:
                pass

            # Show usage instructions
            print("\n" + "=" * 60)
            print("🎯 USAGE INSTRUCTIONS")
            print("=" * 60)
            print(f"\nTo use this model in your system:")
            print(f"python main.py --classifier-model {final_path}")
            print("\nOr modify aggression_classifier.py to load it by default.")

            # Test the model
            print("\n🧪 Testing trained model...")
            self._test_trained_model(final_path)
        else:
            print("⚠️ Best model not found")

    def _test_trained_model(self, model_path):
        """Test the trained model"""
        try:
            test_model = YOLO(str(model_path))

            # Create a test image
            test_img = np.zeros((224, 224, 3), dtype=np.uint8)
            test_img[:] = [100, 150, 200]  # Blue color

            # Run inference
            results = test_model(test_img, verbose=False)

            if results and len(results) > 0:
                result = results[0]
                if hasattr(result, 'probs'):
                    probs = result.probs.data.cpu().numpy()
                    print(f"   Test inference successful!")
                    print(f"   Class probabilities: {probs}")
                else:
                    print("   Model loaded but no probabilities found")
            else:
                print("   Model loaded but inference failed")

        except Exception as e:
            print(f"   Model test error: {e}")

    def download_sample_dataset(self):
        """Guide for downloading sample dataset"""
        print("\n" + "=" * 60)
        print("📥 DOWNLOAD SAMPLE DATASET")
        print("=" * 60)

        print("\n🎯 Since collecting real dog aggression images can be difficult,")
        print("   you can use these alternative approaches:")

        print("\n1. 🐕 ANIMAL EXPRESSION DATASETS:")
        print("   - Kaggle: Dog Emotion Recognition")
        print("   - Kaggle: Animal Faces")
        print("   - Google: 'dog facial expression dataset'")

        print("\n2. 📷 CREATE SYNTHETIC DATASET:")
        print("   - Use Google Images search")
        print("   - Search: 'calm dog face', 'aggressive dog snarling'")
        print("   - Download 50-100 images for each class")

        print("\n3. 🎬 USE VIDEO FRAMES:")
        print("   - Extract frames from dog videos")
        print("   - Label frames as calm/aggressive")

        print("\n4. 🤖 USE PRE-TRAINED MODELS:")
        print("   - Fine-tune existing animal classifiers")

        print("\n💡 Quick start with synthetic data:")
        print("   python create_synthetic_dataset.py")

    def create_quick_test_set(self):
        """Create a quick test dataset for demonstration"""
        print("\n🔧 Creating quick test dataset...")

        # Create some synthetic test images
        for class_idx, class_name in enumerate(['dog_calm', 'dog_aggressive']):
            class_dir = self.dataset_path / "images/train" / class_name
            class_dir.mkdir(parents=True, exist_ok=True)

            for i in range(10):  # Create 10 test images per class
                # Create different colored squares as "dogs"
                if class_name == 'dog_calm':
                    color = (0, 200, 100)  # Greenish for calm
                else:
                    color = (200, 100, 0)  # Reddish for aggressive

                img = np.zeros((224, 224, 3), dtype=np.uint8)
                img[:] = color

                # Add some variation
                cv2.rectangle(img, (50, 50), (174, 174), (255, 255, 255), 2)

                # Save
                img_path = class_dir / f"test_{i:03d}.jpg"
                cv2.imwrite(str(img_path), img)

        print("   Created 20 synthetic test images")
        print("   Note: This is for testing ONLY")
        print("   Replace with real dog images for actual training")


def main():
    """Main training program"""
    parser = argparse.ArgumentParser(description='Train Dog Aggression Classifier')
    parser.add_argument('--mode', type=str, default='train',
                        choices=['setup', 'train', 'collect', 'download', 'test'],
                        help='Operation mode')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs')
    parser.add_argument('--imgsz', type=int, default=224,
                        help='Image size for training')
    parser.add_argument('--batch', type=int, default=16,
                        help='Batch size')

    args = parser.parse_args()

    # Create trainer
    trainer = DogAggressionTrainer()

    if args.mode == 'setup':
        print("\n⚙️ SETUP MODE")
        trainer.create_dataset_structure()
        trainer.create_data_yaml()
        trainer.collect_sample_images()

    elif args.mode == 'collect':
        print("\n📸 COLLECTION GUIDE")
        trainer.collect_sample_images()

    elif args.mode == 'download':
        print("\n📥 DOWNLOAD GUIDE")
        trainer.download_sample_dataset()

    elif args.mode == 'test':
        print("\n🧪 TEST MODE")
        trainer.create_dataset_structure()
        trainer.create_quick_test_set()
        trainer.validate_dataset()
        trainer.train_model(epochs=10, imgsz=64, batch=4)  # Quick test

    else:  # 'train' mode
        print("\n🏋️ TRAINING MODE")
        trainer.create_dataset_structure()
        trainer.validate_dataset()
        trainer.augment_dataset()
        trainer.train_model(epochs=args.epochs, imgsz=args.imgsz, batch=args.batch)

    print("\n" + "=" * 60)
    print("✅ TRAINING SYSTEM COMPLETE")
    print("=" * 60)


if __name__ == "__main__":
    main()