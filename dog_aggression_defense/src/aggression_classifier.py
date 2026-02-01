"""
REAL Dog Aggression Classifier using YOLOv8
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path
import time

print("=" * 60)
print("🐕 DOG AGGRESSION CLASSIFIER - Loading...")
print("=" * 60)


class RealAggressionClassifier:
    """
    Real aggression classifier using trained YOLOv8 model
    """

    def __init__(self, model_path=None, conf_threshold=0.6):
        """
        Initialize real classifier

        Args:
            model_path: Path to trained model
            conf_threshold: Confidence threshold (0.1-0.9)
        """
        self.conf_threshold = conf_threshold
        self.class_names = ['dog_calm', 'dog_aggressive']

        # Try to load trained model
        if model_path and Path(model_path).exists():
            print(f"📦 Loading trained model: {model_path}")
            try:
                self.model = YOLO(model_path)
                self.placeholder_mode = False
                print(f"✅ Real classifier loaded successfully!")
                print(f"   Classes: {self.class_names}")
                print(f"   Confidence threshold: {self.conf_threshold}")
            except Exception as e:
                print(f"⚠️ Failed to load model: {e}")
                print("   Falling back to placeholder mode")
                self.model = None
                self.placeholder_mode = True

        else:
            # Default models to try
            default_models = [
                "models/aggression_classifier.pt",
                "models/dog_aggression_classifier/weights/best.pt",
                "aggression_classifier.pt"
            ]

            model_found = False
            for model_file in default_models:
                if Path(model_file).exists():
                    print(f"📦 Loading: {model_file}")
                    try:
                        self.model = YOLO(model_file)
                        self.placeholder_mode = False
                        model_found = True
                        print(f"✅ Real classifier loaded from {model_file}")
                        break
                    except:
                        continue

            if not model_found:
                print("⚠️ No trained model found!")
                print("💡 Train a model first: python train_aggression_classifier.py")
                print("   Using placeholder mode for now")
                self.model = None
                self.placeholder_mode = True

        # Performance tracking
        self.inference_times = []
        self.classification_count = 0

    def preprocess_image(self, image, target_size=224):
        """
        Preprocess image for classification

        Args:
            image: Input image (BGR format)
            target_size: Target size for model

        Returns:
            Preprocessed image
        """
        if image is None or image.size == 0:
            return None

        # Resize
        resized = cv2.resize(image, (target_size, target_size))

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize to [0, 1]
        normalized = rgb.astype(np.float32) / 255.0

        return normalized

    def classify(self, dog_roi):
        """
        Classify dog behavior from ROI

        Args:
            dog_roi: Region of interest containing dog

        Returns:
            Tuple of (is_aggressive, confidence)
        """
        start_time = time.time()

        # PLACEHOLDER MODE: For testing without trained model
        if self.placeholder_mode or self.model is None:
            # Simulate classification for testing
            import random

            # Simple heuristic based on image color/features
            if dog_roi is not None and dog_roi.size > 0:
                # Check for "red-ish" colors (simulating aggression)
                hsv = cv2.cvtColor(dog_roi, cv2.COLOR_BGR2HSV)
                red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255))
                red_ratio = np.sum(red_mask > 0) / (dog_roi.shape[0] * dog_roi.shape[1])

                # Check for "dark" areas (snarling mouth)
                gray = cv2.cvtColor(dog_roi, cv2.COLOR_BGR2GRAY)
                dark_ratio = np.sum(gray < 50) / (gray.size)

                # Simple decision
                if red_ratio > 0.1 or dark_ratio > 0.3:
                    is_aggressive = True
                    confidence = max(0.6, min(0.9, red_ratio + dark_ratio))
                else:
                    is_aggressive = False
                    confidence = random.uniform(0.6, 0.9)
            else:
                # Random for testing
                is_aggressive = random.random() < 0.3
                confidence = random.uniform(0.6, 0.9)

            self.classification_count += 1
            return is_aggressive, confidence

        # REAL MODEL MODE: Use trained YOLOv8 classifier
        try:
            # Check ROI
            if dog_roi is None or dog_roi.size == 0:
                return False, 0.0

            if dog_roi.shape[0] < 10 or dog_roi.shape[1] < 10:
                return False, 0.0

            # Run classification
            results = self.model(dog_roi, verbose=False)

            if results and len(results) > 0:
                result = results[0]

                if hasattr(result, 'probs') and result.probs is not None:
                    # Get probabilities
                    probs = result.probs.data.cpu().numpy()

                    # Get top class and confidence
                    top_class_idx = int(result.probs.top1)
                    confidence = float(result.probs.top1conf)

                    # Determine if aggressive
                    is_aggressive = (self.class_names[top_class_idx] == 'dog_aggressive')

                    # Apply confidence threshold
                    if confidence < self.conf_threshold:
                        return False, confidence

                    # Track performance
                    inference_time = time.time() - start_time
                    self.inference_times.append(inference_time)
                    self.classification_count += 1

                    return is_aggressive, confidence

            # Fallback if no valid classification
            return False, 0.0

        except Exception as e:
            print(f"⚠️ Classification error: {e}")
            return False, 0.0

    def get_performance_stats(self):
        """Get performance statistics"""
        if self.inference_times:
            avg_time = np.mean(self.inference_times)
            min_time = np.min(self.inference_times)
            max_time = np.max(self.inference_times)
            fps = 1.0 / avg_time if avg_time > 0 else 0.0
        else:
            avg_time = min_time = max_time = fps = 0.0

        return {
            'avg_inference_time': avg_time * 1000,  # ms
            'min_inference_time': min_time * 1000,
            'max_inference_time': max_time * 1000,
            'fps': fps,
            'total_classifications': self.classification_count,
            'placeholder_mode': self.placeholder_mode,
            'model_loaded': not self.placeholder_mode
        }

    def adjust_confidence(self, delta):
        """Adjust confidence threshold"""
        self.conf_threshold = max(0.1, min(0.9, self.conf_threshold + delta))
        print(f"🎯 Classifier confidence: {self.conf_threshold:.2f}")
        return self.conf_threshold


def test_classifier():
    """Test the classifier"""
    print("\n🧪 Testing Aggression Classifier...")

    # Initialize classifier
    classifier = RealAggressionClassifier()

    # Create test images
    print("\nCreating test images...")

    # Calm dog (greenish)
    calm_img = np.zeros((200, 200, 3), dtype=np.uint8)
    calm_img[:] = [0, 200, 100]  # Green

    # Aggressive dog (reddish)
    aggressive_img = np.zeros((200, 200, 3), dtype=np.uint8)
    aggressive_img[:] = [200, 100, 0]  # Red

    # Test both
    print("\nTesting calm dog (green)...")
    is_agg, conf = classifier.classify(calm_img)
    print(f"   Result: {'AGGRESSIVE' if is_agg else 'CALM'} (confidence: {conf:.2f})")

    print("\nTesting aggressive dog (red)...")
    is_agg, conf = classifier.classify(aggressive_img)
    print(f"   Result: {'AGGRESSIVE' if is_agg else 'CALM'} (confidence: {conf:.2f})")

    # Show stats
    stats = classifier.get_performance_stats()
    print("\n📊 Classifier Statistics:")
    print(f"   Mode: {'PLACEHOLDER' if stats['placeholder_mode'] else 'REAL MODEL'}")
    print(f"   Classifications: {stats['total_classifications']}")

    if not stats['placeholder_mode']:
        print(f"   Inference time: {stats['avg_inference_time']:.1f}ms")
        print(f"   FPS: {stats['fps']:.1f}")


if __name__ == "__main__":
    test_classifier()