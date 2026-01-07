"""
MAIN APPLICATION: Dog Aggression Defense System
Detects dogs, classifies behavior, activates ultrasonic repeller
"""
import cv2
import time
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.append(str(Path(__file__).parent))

from dog_detector import DogDetector
from aggression_classifier import AggressionClassifier
from ultrasonic_simulator import UltrasonicSimulator


class DogAggressionDefense:
    def __init__(self):
        """Initialize all components"""
        print("Initializing Dog Aggression Defense System...")

        # Initialize components
        self.dog_detector = DogDetector()
        self.aggression_classifier = AggressionClassifier()
        self.repeller = UltrasonicSimulator()

        # State variables
        self.is_active = False
        self.aggressive_detected = False
        self.aggression_start_time = None
        self.min_aggression_duration = 2.0  # seconds

        print("System initialized successfully!")

    def process_frame(self, frame):
        """
        Process a single frame for dog detection and aggression classification

        Args:
            frame: Input frame from webcam/video

        Returns:
            processed_frame: Frame with annotations
        """
        # Detect dogs
        detections = self.dog_detector.detect(frame)

        # Reset aggression state for new frame
        current_aggression = False

        # Process each detected dog
        for det in detections:
            # Extract dog bounding box
            x1, y1, x2, y2, conf = det

            # Crop dog region for classification
            dog_roi = frame[y1:y2, x1:x2]

            if dog_roi.size > 0:  # Ensure valid ROI
                # Classify aggression
                is_aggressive, aggression_conf = self.aggression_classifier.classify(dog_roi)

                # Draw bounding box (red for aggressive, green for calm)
                color = (0, 0, 255) if is_aggressive else (0, 255, 0)
                label = f"{'AGGRESSIVE' if is_aggressive else 'CALM'} {aggression_conf:.2f}"

                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                cv2.putText(frame, label, (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                if is_aggressive:
                    current_aggression = True

        # Update repeller state
        self._update_repeller_state(current_aggression)

        # Add system status overlay
        status_color = (0, 0, 255) if self.aggressive_detected else (0, 255, 0)
        status_text = "REPELLER ACTIVE" if self.aggressive_detected else "SYSTEM READY"
        cv2.putText(frame, f"Status: {status_text}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)

        return frame

    def _update_repeller_state(self, current_aggression):
        """Manage repeller activation with hysteresis"""
        current_time = time.time()

        if current_aggression:
            if not self.aggression_start_time:
                self.aggression_start_time = current_time

            # Only activate after minimum aggression duration
            aggression_duration = current_time - self.aggression_start_time
            if aggression_duration >= self.min_aggression_duration:
                self.aggressive_detected = True
                self.repeller.activate()
        else:
            # Reset when no aggression is detected
            self.aggression_start_time = None
            if self.aggressive_detected:
                self.aggressive_detected = False
                self.repeller.deactivate()

    def run_webcam(self, camera_id=0):
        """Run system with webcam feed"""
        print(f"Starting webcam (ID: {camera_id})...")
        print("Press 'q' to quit, 's' to toggle repeller simulation sound")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        # Set resolution
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Error: Cannot read frame")
                break

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display
            cv2.imshow('Dog Aggression Defense System', processed_frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                self.repeller.toggle_sound()
                print(f"Simulation sound: {'ON' if self.repeller.sound_enabled else 'OFF'}")

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        self.repeller.cleanup()
        print("System shutdown complete")

    def run_video(self, video_path):
        """Run system with video file"""
        print(f"Processing video: {video_path}")

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video {video_path}")
            return

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            processed_frame = self.process_frame(frame)

            # Display
            cv2.imshow('Dog Aggression Defense System - Video', processed_frame)

            if cv2.waitKey(25) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        self.repeller.cleanup()


if __name__ == "__main__":
    # Create system
    system = DogAggressionDefense()

    # Example usage:
    # 1. For webcam: system.run_webcam()
    # 2. For video file: system.run_video("data/raw_videos/test.mp4")

    # Start with webcam (default)
    system.run_webcam()