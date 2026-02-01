"""
CREATE SYNTHETIC DATASET for Dog Aggression Training
Generate synthetic images for quick testing
"""
import cv2
import numpy as np
from pathlib import Path
import random

print("=" * 60)
print("🎨 CREATING SYNTHETIC DATASET")
print("=" * 60)


def create_synthetic_image(class_type, size=224):
    """
    Create synthetic dog image

    Args:
        class_type: 'calm' or 'aggressive'
        size: Image size

    Returns:
        Synthetic image
    """
    img = np.zeros((size, size, 3), dtype=np.uint8)

    if class_type == 'calm':
        # Calm dog - green/blue tones
        base_color = random.choice([
            (0, 150, 100),  # Greenish
            (100, 150, 200),  # Bluish
            (150, 150, 150),  # Gray
        ])

        # Draw "calm" features
        color = tuple(min(255, c + random.randint(-20, 20)) for c in base_color)
        img[:] = color

        # Draw face outline
        cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 3), 0, 0, 360, (200, 200, 200), 2)

        # Eyes (closed/semi-closed)
        cv2.ellipse(img, (size // 2 - 30, size // 2 - 10), (10, 5), 0, 0, 360, (50, 50, 50), -1)
        cv2.ellipse(img, (size // 2 + 30, size // 2 - 10), (10, 5), 0, 0, 360, (50, 50, 50), -1)

        # Mouth (relaxed)
        cv2.ellipse(img, (size // 2, size // 2 + 30), (40, 15), 0, 0, 180, (100, 100, 100), 2)

    else:  # aggressive
        # Aggressive dog - red/orange tones
        base_color = random.choice([
            (200, 100, 50),  # Reddish
            (150, 50, 0),  # Dark red
            (100, 50, 50),  # Brownish red
        ])

        color = tuple(min(255, c + random.randint(-20, 20)) for c in base_color)
        img[:] = color

        # Draw "aggressive" features
        # Face outline (more angular)
        cv2.ellipse(img, (size // 2, size // 2), (size // 3, size // 3), 0, 0, 360, (100, 100, 100), 3)

        # Eyes (wide open)
        cv2.circle(img, (size // 2 - 30, size // 2 - 10), 8, (255, 255, 255), -1)
        cv2.circle(img, (size // 2 + 30, size // 2 - 10), 8, (255, 255, 255), -1)
        cv2.circle(img, (size // 2 - 30, size // 2 - 10), 3, (0, 0, 0), -1)
        cv2.circle(img, (size // 2 + 30, size // 2 - 10), 3, (0, 0, 0), -1)

        # Mouth (open, snarling)
        cv2.ellipse(img, (size // 2, size // 2 + 30), (50, 25), 0, 0, 180, (0, 0, 0), -1)

        # Teeth
        for i in range(5):
            x = size // 2 - 40 + i * 20
            cv2.rectangle(img, (x, size // 2 + 20), (x + 10, size // 2 + 40), (255, 255, 255), -1)

        # Hackles (raised fur)
        for i in range(10):
            x = random.randint(20, size - 20)
            y = random.randint(20, 60)
            cv2.line(img, (x, y), (x, y - 15), (100, 100, 100), 2)

    # Add some noise/variation
    noise = np.random.normal(0, 10, img.shape).astype(np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    # Apply slight blur
    img = cv2.GaussianBlur(img, (3, 3), 0.5)

    return img


def create_dataset(num_images=100):
    """Create synthetic dataset"""
    print(f"\n📁 Creating {num_images * 2} synthetic images...")

    # Create directories
    base_dir = Path("datasets/dog_aggression/images")
    base_dir.mkdir(parents=True, exist_ok=True)

    train_calm_dir = base_dir / "train/dog_calm"
    train_agg_dir = base_dir / "train/dog_aggressive"
    val_calm_dir = base_dir / "val/dog_calm"
    val_agg_dir = base_dir / "val/dog_aggressive"

    for dir_path in [train_calm_dir, train_agg_dir, val_calm_dir, val_agg_dir]:
        dir_path.mkdir(parents=True, exist_ok=True)

    # Create images
    images_created = 0

    for i in range(num_images):
        # Calm images (80% train, 20% val)
        calm_img = create_synthetic_image('calm')

        if i < num_images * 0.8:
            save_path = train_calm_dir / f"calm_{i:04d}.jpg"
        else:
            save_path = val_calm_dir / f"calm_{i:04d}.jpg"

        cv2.imwrite(str(save_path), calm_img)
        images_created += 1

        # Aggressive images
        agg_img = create_synthetic_image('aggressive')

        if i < num_images * 0.8:
            save_path = train_agg_dir / f"agg_{i:04d}.jpg"
        else:
            save_path = val_agg_dir / f"agg_{i:04d}.jpg"

        cv2.imwrite(str(save_path), agg_img)
        images_created += 1

        if (i + 1) % 20 == 0:
            print(f"   Created {images_created} images...")

    # Create data.yaml
    yaml_content = """# Synthetic Dog Aggression Dataset
path: datasets/dog_aggression
train: images/train
val: images/val

# Classes
names:
  0: dog_calm
  1: dog_aggressive
"""

    yaml_path = Path("datasets/dog_aggression/data.yaml")
    yaml_path.write_text(yaml_content)

    print(f"\n✅ Created {images_created} synthetic images")
    print(f"📝 Created data.yaml")
    print(f"📁 Dataset saved in: datasets/dog_aggression/")

    # Show sample images
    print("\n🎨 Sample images created:")

    # Display sample
    sample_calm = create_synthetic_image('calm')
    sample_agg = create_synthetic_image('aggressive')

    # Combine for display
    combined = np.hstack([sample_calm, sample_agg])

    # Add labels
    cv2.putText(combined, "CALM", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(combined, "AGGRESSIVE", (30 + 224, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    cv2.imwrite("sample_synthetic.jpg", combined)
    print("💾 Sample image saved: sample_synthetic.jpg")


if __name__ == "__main__":
    print("\nThis will create a SYNTHETIC dataset for testing.")
    print("For real training, replace with actual dog images.\n")

    num_images = input("Number of images per class [50]: ").strip()
    num_images = int(num_images) if num_images else 50

    create_dataset(num_images)

    print("\n" + "=" * 60)
    print("📋 NEXT STEPS:")
    print("=" * 60)
    print("1. Train model:")
    print("   python train_aggression_classifier.py --mode train --epochs 50")
    print("\n2. Test with webcam:")
    print("   python main.py --classifier-model models/aggression_classifier.pt")
    print("\n3. For better results, replace synthetic images")
    print("   with real dog photos in the dataset folders")
    print("=" * 60)