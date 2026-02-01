"""
DOG & PERSON DETECTOR - Guaranteed Working Version
Detects dogs, people, and other animals with YOLOv8
"""
import cv2
import numpy as np
from ultralytics import YOLO
import time
from pathlib import Path
import torch

print("="*60)
print("🐕 DOG & PERSON DETECTOR - INITIALIZING")
print("="*60)

class DogPersonDetector:
    """
    Simple, reliable detector for dogs and people
    """

    def __init__(self, conf_threshold=0.5):
        """
        Initialize detector with YOLOv8

        Args:
            conf_threshold: Confidence threshold (0.1-0.9)
        """
        self.conf_threshold = conf_threshold

        # COCO class IDs
        self.CLASS_NAMES = {
            0: 'person',
            1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane',
            5: 'bus', 6: 'train', 7: 'truck', 8: 'boat',
            16: 'dog', 15: 'cat', 17: 'horse', 18: 'sheep',
            19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra',
            23: 'giraffe'
        }

        # Target classes we care about
        self.TARGET_CLASSES = [0, 16, 15, 17]  # person, dog, cat, horse

        print(f"🔧 Loading YOLOv8 model...")
        print(f"📊 Device: {'GPU' if torch.cuda.is_available() else 'CPU'}")

        try:
            # Load YOLOv8 model
            self.model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 model loaded successfully!")
            print(f"🎯 Target classes: {[self.CLASS_NAMES[c] for c in self.TARGET_CLASSES]}")
            print(f"📈 Confidence threshold: {self.conf_threshold}")

        except Exception as e:
            print(f"❌ ERROR loading model: {e}")
            print("💡 Trying alternative approach...")
            raise e

    def detect(self, frame):
        """
        Detect objects in frame

        Args:
            frame: Input frame (BGR format)

        Returns:
            List of detections [x1, y1, x2, y2, confidence, class_name]
        """
        if frame is None or frame.size == 0:
            return []

        try:
            # Run inference
            results = self.model(frame, verbose=False, conf=self.conf_threshold)

            detections = []

            if results and len(results) > 0:
                result = results[0]

                if result.boxes is not None and len(result.boxes) > 0:
                    for box in result.boxes:
                        class_id = int(box.cls[0])
                        confidence = float(box.conf[0])

                        # Only include target classes
                        if class_id in self.TARGET_CLASSES:
                            # Get bounding box
                            x1, y1, x2, y2 = map(int, box.xyxy[0])

                            # Get class name
                            class_name = self.CLASS_NAMES.get(class_id, f'class_{class_id}')

                            detections.append([x1, y1, x2, y2, confidence, class_name])

            return detections

        except Exception as e:
            print(f"⚠️ Detection error: {e}")
            return []

    def draw_detections(self, frame, detections):
        """
        Draw detections on frame

        Args:
            frame: Input frame
            detections: List of detections

        Returns:
            Frame with drawn detections
        """
        if not detections:
            return frame

        output = frame.copy()

        # Colors for different classes
        colors = {
            'person': (0, 165, 255),   # Orange
            'dog': (0, 255, 0),        # Green
            'cat': (255, 165, 0),      # Blue
            'horse': (128, 0, 128),    # Purple
        }

        for det in detections:
            x1, y1, x2, y2, confidence, class_name = det

            # Get color for this class
            color = colors.get(class_name, (0, 255, 255))  # Yellow default

            # Draw bounding box
            cv2.rectangle(output, (x1, y1), (x2, y2), color, 2)

            # Draw label background
            label = f"{class_name} {confidence:.2f}"
            (text_width, text_height), baseline = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2
            )

            cv2.rectangle(output,
                         (x1, y1 - text_height - 10),
                         (x1 + text_width + 10, y1),
                         color, -1)

            # Draw label text
            cv2.putText(output, label, (x1 + 5, y1 - 5),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        return output


def test_detector():
    """Test the detector with webcam"""
    print("\n🎬 TESTING DETECTOR WITH WEBCAM")
    print("="*40)

    # Initialize detector
    detector = DogPersonDetector(conf_threshold=0.3)  # Low threshold for testing

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🎮 CONTROLS:")
    print("   q: Quit")
    print("   +: Increase confidence")
    print("   -: Decrease confidence")
    print("   s: Save screenshot")
    print("\n▶️ Looking for dogs and people...")

    frame_count = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Cannot read from webcam")
                break

            frame_count += 1

            # Detect objects
            detections = detector.detect(frame)

            # Draw detections
            output = detector.draw_detections(frame, detections)

            # Count dogs and people
            dogs = [d for d in detections if d[5] == 'dog']
            people = [d for d in detections if d[5] == 'person']
            cats = [d for d in detections if d[5] == 'cat']

            # Add statistics overlay
            stats_text = f"Dogs: {len(dogs)} | People: {len(people)} | Cats: {len(cats)}"
            cv2.putText(output, stats_text, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Add frame counter
            cv2.putText(output, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1)

            # Add confidence info
            cv2.putText(output, f"Conf: {detector.conf_threshold:.2f}", (10, 80),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 255), 1)

            # Display
            cv2.imshow('🐕 Dog & Person Detector', output)

            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('+'):
                detector.conf_threshold = min(0.9, detector.conf_threshold + 0.05)
                print(f"🎯 Confidence: {detector.conf_threshold:.2f}")
            elif key == ord('-'):
                detector.conf_threshold = max(0.1, detector.conf_threshold - 0.05)
                print(f"🎯 Confidence: {detector.conf_threshold:.2f}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"detection_{timestamp}.jpg"
                cv2.imwrite(filename, output)
                print(f"📸 Saved: {filename}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted by user")

    finally:
        cap.release()
        cv2.destroyAllWindows()
        print(f"\n📊 Processed {frame_count} frames")
        print("✅ Test completed!")


def test_image(image_path):
    """Test detector with image file"""
    if not Path(image_path).exists():
        print(f"❌ Image not found: {image_path}")
        print("💡 Creating test image...")
        # Create a test image
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        img.fill(200)
        cv2.imwrite(image_path, img)

    detector = DogPersonDetector()

    print(f"\n📸 Testing with image: {image_path}")

    # Load image
    frame = cv2.imread(image_path)
    if frame is None:
        print("❌ Cannot load image")
        return

    # Detect objects
    detections = detector.detect(frame)

    print(f"\n📊 DETECTIONS: {len(detections)}")
    for i, det in enumerate(detections):
        x1, y1, x2, y2, conf, class_name = det
        print(f"  {i+1}. {class_name}: {conf:.2f} at [{x1},{y1},{x2},{y2}]")

    # Draw and save
    output = detector.draw_detections(frame, detections)
    cv2.imwrite("detection_result.jpg", output)
    print("💾 Result saved: detection_result.jpg")

    # Show
    cv2.imshow('Detection Result', output)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    print("\n🔧 DOG DETECTOR SYSTEM")
    print("="*40)
    print("1. Test with webcam (default)")
    print("2. Test with image")
    print("3. Quick system check")

    choice = input("\nSelect option [1]: ").strip()

    if choice == "2":
        image_path = input("Enter image path [test.jpg]: ").strip()
        if not image_path:
            image_path = "test.jpg"
        test_image(image_path)

    elif choice == "3":
        print("\n🔍 Running system check...")
        # Check if YOLO model can be loaded
        try:
            model = YOLO('yolov8n.pt')
            print("✅ YOLOv8 model loads successfully")

            # Test with dummy image
            test_img = np.zeros((100, 100, 3), dtype=np.uint8)
            results = model(test_img, verbose=False)
            print("✅ Inference works")

            print("\n📊 System check PASSED!")
            print("   You can now run: python dog_detector.py")

        except Exception as e:
            print(f"❌ System check FAILED: {e}")
            print("💡 Try: pip install ultralytics opencv-python")

    else:
        test_detector()