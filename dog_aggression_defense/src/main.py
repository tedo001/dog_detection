"""
MAIN APPLICATION: Dog Aggression Defense System
Detects dogs, classifies behavior, activates ultrasonic repeller for aggressive dogs
Author: Dog Defense Team
Version: 1.0.0
"""

import cv2
import time
import numpy as np
from pathlib import Path
import sys
import argparse

# Add src directory to path for module imports
sys.path.append(str(Path(__file__).parent))

# Import custom modules
from dog_detector import DogDetector
from aggression_classifier import AggressionClassifier
from ultrasonic_simulator import UltrasonicSimulator


class DogAggressionDefense:
    """
    Main system class that integrates all components:
    1. Dog detection using YOLOv8
    2. Aggression classification
    3. Ultrasonic repeller activation
    """

    def __init__(self, config=None):
        """
        Initialize all system components

        Args:
            config (dict, optional): Configuration dictionary
        """
        print("\n" + "=" * 50)
        print("DOG AGGRESSION DEFENSE SYSTEM")
        print("=" * 50)

        # Default configuration
        self.config = {
            'detector_confidence': 0.5,  # Minimum confidence for dog detection
            'classifier_confidence': 0.6,  # Minimum confidence for aggression classification
            'min_aggression_duration': 2.0,  # Seconds of aggression before activation
            'camera_id': 0,  # Default webcam ID
            'frame_width': 640,  # Frame width
            'frame_height': 480,  # Frame height
            'simulation_mode': True,  # Use simulator (not real hardware)
            'model_paths': {
                'detector': None,  # Custom detector model path
                'classifier': None  # Custom classifier model path
            }
        }

        # Update with provided config
        if config:
            self.config.update(config)

        # Initialize components
        print("\n[1/3] Initializing Dog Detector...")
        self.dog_detector = DogDetector(
            model_path=self.config['model_paths']['detector'],
            conf_threshold=self.config['detector_confidence']
        )

        print("[2/3] Initializing Aggression Classifier...")
        self.aggression_classifier = AggressionClassifier(
            model_path=self.config['model_paths']['classifier'],
            conf_threshold=self.config['classifier_confidence']
        )

        print("[3/3] Initializing Ultrasonic Repeller...")
        self.repeller = UltrasonicSimulator(
            simulation_mode=self.config['simulation_mode']
        )

        # System state variables
        self.is_active = False
        self.aggressive_detected = False
        self.aggression_start_time = None
        self.frame_count = 0
        self.fps = 0
        self.start_time = time.time()

        # Statistics
        self.stats = {
            'total_frames': 0,
            'total_dogs': 0,
            'calm_dogs': 0,
            'aggressive_dogs': 0,
            'repeller_activations': 0
        }

        print("\n" + "=" * 50)
        print("System initialized successfully!")
        print("Controls:")
        print("  'q' - Quit application")
        print("  's' - Toggle simulation sound")
        print("  'd' - Toggle detection display")
        print("  'p' - Pause/Resume processing")
        print("=" * 50 + "\n")

    def process_frame(self, frame):
        """
        Process a single frame for dog detection and aggression classification

        Args:
            frame: Input frame from webcam/video (BGR format)

        Returns:
            processed_frame: Frame with annotations and visual feedback
        """
        self.frame_count += 1
        self.stats['total_frames'] += 1

        # Calculate FPS every 30 frames
        if self.frame_count % 30 == 0:
            elapsed = time.time() - self.start_time
            self.fps = self.frame_count / elapsed

        # Create a copy for drawing
        display_frame = frame.copy()

        # Detect dogs in the frame
        detections = self.dog_detector.detect(frame)

        # Reset aggression state for this frame
        current_aggression = False
        dogs_in_frame = len(detections)

        if dogs_in_frame > 0:
            self.stats['total_dogs'] += dogs_in_frame

            # Process each detected dog
            for det in detections:
                # Extract bounding box coordinates
                x1, y1, x2, y2, conf = det

                # Ensure coordinates are within frame boundaries
                h, w = frame.shape[:2]
                x1 = max(0, min(x1, w - 1))
                y1 = max(0, min(y1, h - 1))
                x2 = max(0, min(x2, w - 1))
                y2 = max(0, min(y2, h - 1))

                # Crop dog region for classification
                if y2 > y1 and x2 > x1:  # Valid region
                    dog_roi = frame[y1:y2, x1:x2]

                    if dog_roi.size > 0 and dog_roi.shape[0] > 10 and dog_roi.shape[1] > 10:
                        # Classify aggression
                        is_aggressive, aggression_conf = self.aggression_classifier.classify(dog_roi)

                        # Update statistics
                        if is_aggressive:
                            self.stats['aggressive_dogs'] += 1
                            current_aggression = True
                        else:
                            self.stats['calm_dogs'] += 1

                        # Draw bounding box with appropriate color
                        box_color = (0, 0, 255) if is_aggressive else (0, 255, 0)  # Red for aggressive, green for calm
                        text_color = (255, 255, 255)  # White text

                        # Draw rectangle
                        cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 2)

                        # Draw classification label
                        label = f"{'AGGRESSIVE' if is_aggressive else 'CALM'}: {aggression_conf:.1%}"
                        text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

                        # Draw text background
                        cv2.rectangle(display_frame,
                                      (x1, y1 - text_size[1] - 10),
                                      (x1 + text_size[0] + 10, y1),
                                      box_color, -1)

                        # Draw text
                        cv2.putText(display_frame, label, (x1 + 5, y1 - 5),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 2)

                        # Draw detection confidence
                        detection_label = f"Det: {conf:.1%}"
                        cv2.putText(display_frame, detection_label, (x1, y2 + 20),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Update repeller state based on aggression detection
        self._update_repeller_state(current_aggression)

        # Draw system status overlay
        self._draw_status_overlay(display_frame)

        # Draw statistics overlay
        self._draw_stats_overlay(display_frame)

        return display_frame

    def _update_repeller_state(self, current_aggression):
        """
        Manage repeller activation with hysteresis to prevent rapid toggling

        Args:
            current_aggression: Boolean indicating aggression in current frame
        """
        current_time = time.time()

        if current_aggression:
            # Start or continue aggression timer
            if self.aggression_start_time is None:
                self.aggression_start_time = current_time
                print(f"[INFO] Aggression detected at {time.strftime('%H:%M:%S')}")

            # Calculate how long aggression has been detected
            aggression_duration = current_time - self.aggression_start_time

            # Only activate repeller after minimum duration
            if aggression_duration >= self.config['min_aggression_duration']:
                if not self.aggressive_detected:
                    self.aggressive_detected = True
                    self.stats['repeller_activations'] += 1
                    print(f"[ACTION] Repeller ACTIVATED after {aggression_duration:.1f}s of aggression")
                self.repeller.activate()
            else:
                # Show countdown to activation
                remaining = self.config['min_aggression_duration'] - aggression_duration
                if remaining > 0:
                    print(f"[INFO] Aggression detected. Activating in {remaining:.1f}s...", end='\r')
        else:
            # Reset aggression timer
            self.aggression_start_time = None

            # Deactivate repeller if it was active
            if self.aggressive_detected:
                self.aggressive_detected = False
                self.repeller.deactivate()
                print(f"[ACTION] Repeller DEACTIVATED - No aggression detected")

    def _draw_status_overlay(self, frame):
        """Draw system status information on frame"""
        # Status bar background
        h, w = frame.shape[:2]
        cv2.rectangle(frame, (0, 0), (w, 40), (40, 40, 40), -1)

        # System status
        status_color = (0, 0, 255) if self.aggressive_detected else (0, 255, 0)
        status_text = "REPELLER ACTIVE" if self.aggressive_detected else "SYSTEM READY"

        cv2.putText(frame, f"Status: {status_text}", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, status_color, 2)

        # FPS counter
        fps_text = f"FPS: {self.fps:.1f}"
        cv2.putText(frame, fps_text, (w - 120, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.7, (255, 255, 255), 2)

    def _draw_stats_overlay(self, frame):
        """Draw system statistics on frame"""
        h, w = frame.shape[:2]

        # Draw stats in bottom left corner
        stats_y_start = h - 120

        # Stats background
        cv2.rectangle(frame, (10, stats_y_start - 10), (250, h - 10), (0, 0, 0, 0.7), -1)

        # Stats text
        stats_lines = [
            "SYSTEM STATISTICS:",
            f"Frames Processed: {self.stats['total_frames']}",
            f"Total Dogs Detected: {self.stats['total_dogs']}",
            f"  - Calm: {self.stats['calm_dogs']}",
            f"  - Aggressive: {self.stats['aggressive_dogs']}",
            f"Repeller Activations: {self.stats['repeller_activations']}"
        ]

        for i, line in enumerate(stats_lines):
            y_pos = stats_y_start + (i * 20)
            color = (255, 255, 255) if i == 0 else (200, 200, 200)
            font_scale = 0.5 if i == 0 else 0.45
            cv2.putText(frame, line, (15, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, 1)

    def run_webcam(self, camera_id=None):
        """
        Run system with webcam feed

        Args:
            camera_id: Camera device ID (default from config)
        """
        if camera_id is None:
            camera_id = self.config['camera_id']

        print(f"Starting webcam (ID: {camera_id})...")
        print("Press 'q' to quit, 's' to toggle sound, 'p' to pause")

        # Open webcam
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"ERROR: Cannot open webcam (ID: {camera_id})")
            print("Available cameras: 0, 1, 2...")
            return

        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])
        cap.set(cv2.CAP_PROP_FPS, 30)

        # Get actual properties
        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        print(f"Camera resolution: {actual_width}x{actual_height}")
        print(f"Camera FPS: {actual_fps:.1f}")

        # State for pause functionality
        is_paused = False

        # Main loop
        try:
            while True:
                if not is_paused:
                    # Capture frame
                    ret, frame = cap.read()
                    if not ret:
                        print("ERROR: Cannot read frame from webcam")
                        break

                    # Process frame
                    processed_frame = self.process_frame(frame)

                    # Display
                    cv2.imshow('Dog Aggression Defense System', processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):  # Quit
                    print("\nShutting down...")
                    break
                elif key == ord('s'):  # Toggle sound
                    sound_state = self.repeller.toggle_sound()
                    print(f"Simulation sound: {'ON' if sound_state else 'OFF'}")
                elif key == ord('p'):  # Pause/Resume
                    is_paused = not is_paused
                    if is_paused:
                        print("System PAUSED. Press 'p' to resume.")
                        cv2.putText(frame, "SYSTEM PAUSED - Press 'p' to resume",
                                    (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        cv2.imshow('Dog Aggression Defense System', frame)
                    else:
                        print("System RESUMED.")
                elif key == ord('d'):  # Toggle detection display (placeholder)
                    print("Detection display toggle - Feature coming soon!")

        except KeyboardInterrupt:
            print("\nInterrupted by user")

        finally:
            # Cleanup
            print("\nCleaning up resources...")
            cap.release()
            cv2.destroyAllWindows()
            self.repeller.cleanup()

            # Print final statistics
            self._print_final_stats()

    def run_video(self, video_path):
        """
        Run system with video file

        Args:
            video_path: Path to video file
        """
        print(f"Processing video: {video_path}")

        if not Path(video_path).exists():
            print(f"ERROR: Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"ERROR: Cannot open video file: {video_path}")
            return

        # Get video properties
        video_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        print(f"Video FPS: {video_fps}")
        print(f"Total frames: {total_frames}")
        print("Processing video... Press 'q' to quit, 'p' to pause")

        is_paused = False

        try:
            while True:
                if not is_paused:
                    ret, frame = cap.read()
                    if not ret:
                        print("Video processing complete!")
                        break

                    # Process frame
                    processed_frame = self.process_frame(frame)

                    # Display
                    cv2.imshow('Dog Aggression Defense System - Video', processed_frame)

                    # Control playback speed
                    delay = max(1, int(1000 / video_fps))
                    key = cv2.waitKey(delay) & 0xFF
                else:
                    key = cv2.waitKey(100) & 0xFF

                if key == ord('q'):
                    print("\nStopping video playback...")
                    break
                elif key == ord('p'):
                    is_paused = not is_paused
                    print(f"Video {'PAUSED' if is_paused else 'RESUMED'}")

        except KeyboardInterrupt:
            print("\nVideo processing interrupted")

        finally:
            cap.release()
            cv2.destroyAllWindows()
            self.repeller.cleanup()
            self._print_final_stats()

    def _print_final_stats(self):
        """Print final system statistics"""
        print("\n" + "=" * 50)
        print("FINAL SYSTEM STATISTICS")
        print("=" * 50)
        print(f"Total frames processed: {self.stats['total_frames']}")
        print(f"Total dogs detected: {self.stats['total_dogs']}")

        if self.stats['total_dogs'] > 0:
            calm_percent = (self.stats['calm_dogs'] / self.stats['total_dogs']) * 100
            aggressive_percent = (self.stats['aggressive_dogs'] / self.stats['total_dogs']) * 100
            print(f"  Calm dogs: {self.stats['calm_dogs']} ({calm_percent:.1f}%)")
            print(f"  Aggressive dogs: {self.stats['aggressive_dogs']} ({aggressive_percent:.1f}%)")

        print(f"Repeller activations: {self.stats['repeller_activations']}")

        if self.stats['total_frames'] > 0:
            elapsed = time.time() - self.start_time
            print(f"Processing time: {elapsed:.1f} seconds")
            print(f"Average FPS: {self.stats['total_frames'] / elapsed:.1f}")

        print("=" * 50)


def main():
    """Main entry point for the application"""
    parser = argparse.ArgumentParser(description='Dog Aggression Defense System')
    parser.add_argument('--camera', type=int, default=0, help='Camera ID (default: 0)')
    parser.add_argument('--video', type=str, help='Path to video file')
    parser.add_argument('--detector-conf', type=float, default=0.5, help='Dog detection confidence threshold')
    parser.add_argument('--classifier-conf', type=float, default=0.6,
                        help='Aggression classification confidence threshold')
    parser.add_argument('--no-sound', action='store_true', help='Disable simulation sound')
    parser.add_argument('--model-detector', type=str, help='Path to custom detector model')
    parser.add_argument('--model-classifier', type=str, help='Path to custom classifier model')

    args = parser.parse_args()

    # Prepare configuration
    config = {
        'detector_confidence': args.detector_conf,
        'classifier_confidence': args.classifier_conf,
        'camera_id': args.camera,
        'simulation_mode': not args.no_sound,
        'model_paths': {
            'detector': args.model_detector,
            'classifier': args.model_classifier
        }
    }

    # Create system instance
    system = DogAggressionDefense(config)

    # Run appropriate mode
    if args.video:
        system.run_video(args.video)
    else:
        system.run_webcam(args.camera)


if __name__ == "__main__":
    main()