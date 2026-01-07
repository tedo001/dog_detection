# Aggression classifier 
"""
Dog Aggression Classifier using YOLOv8 Classification
"""
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path


class AggressionClassifier:
    def __init__(self, model_path=None, conf_threshold=0.6):
        """
        Initialize aggression classifier

        Args:
            model_path: Path to trained aggression classifier
            conf_threshold: Confidence threshold for classification
        """
        self.conf_threshold = conf_threshold

        if model_path and Path(model_path).exists():
            print(f"Loading aggression classifier: {model_path}")
            self.model = YOLO(model_path)
        else:
            print("WARNING: Using placeholder classifier. Train a proper model!")
            print("Classes: ['dog_calm', 'dog_aggressive']")
            self.model = None

        # Class names (must match training)
        self.class_names = ['dog_calm', 'dog_aggressive']

    def classify(self, dog_roi):
        """
        Classify dog behavior from ROI

        Args:
            dog_roi: Region of interest containing dog

        Returns:
            is_aggressive: Boolean indicating aggression
            confidence: Classification confidence
        """
        if self.model is None:
            # Placeholder for testing - returns random classification
            # REMOVE THIS IN PRODUCTION
            import random
            is_aggressive = random.random() > 0.7
            confidence = random.uniform(0.6, 0.9)
            return is_aggressive, confidence

        # Ensure ROI is valid size
        if dog_roi.shape[0] < 10 or dog_roi.shape[1] < 10:
            return False, 0.0

        # Run classification
        results = self.model(dog_roi, verbose=False)[0]

        if results.probs is not None:
            # Get top class and confidence
            top_class_idx = int(results.probs.top1)
            confidence = float(results.probs.top1conf)

            # Check if classified as aggressive
            is_aggressive = (self.class_names[top_class_idx] == 'dog_aggressive')

            # Only return aggressive if confidence is high enough
            if confidence < self.conf_threshold:
                return False, confidence

            return is_aggressive, confidence

        return False, 0.0

    def preprocess(self, image):
        """Preprocess image for classification"""
        # Resize to model input size
        input_size = (224, 224)  # Default for YOLOv8 classification
        resized = cv2.resize(image, input_size)

        # Convert BGR to RGB
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

        # Normalize (if needed by your model)
        normalized = rgb.astype(np.float32) / 255.0

        return normalized


# Example usage
if __name__ == "__main__":
    classifier = AggressionClassifier()

    # Test with sample image
    sample_image = np.zeros((100, 100, 3), dtype=np.uint8)
    sample_image[:, :] = [255, 200, 100]  # Yellow color

    is_aggressive, conf = classifier.classify(sample_image)
    print(f"Test classification: Aggressive={is_aggressive}, Confidence={conf:.2f}")