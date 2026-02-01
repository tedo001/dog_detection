"""
SIMPLE WORKING DOG AGGRESSION SYSTEM
This will definitely work!
"""
import cv2
import time
import numpy as np
from pathlib import Path

# Import our detector
try:
    from dog_detector import DogPersonDetector
    from aggression_classifier import AggressionClassifier
    from ultrasonic_simulator import UltrasonicSimulator, UltrasoundMode, UltrasoundConfig
    print("✅ All imports successful!")
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure all files are in the same directory")
    exit(1)

def main():
    print("="*60)
    print("🐕 SIMPLE DOG AGGRESSION SYSTEM")
    print("="*60)

    # Initialize with LOW confidence to ensure detection
    print("\n1️⃣ Initializing Dog Detector...")
    detector = DogPersonDetector(conf_threshold=0.3)  # Low threshold

    print("\n2️⃣ Initializing Aggression Classifier...")
    classifier = AggressionClassifier(conf_threshold=0.6)

    print("\n3️⃣ Initializing Ultrasound...")
    ultrasound = UltrasonicSimulator(UltrasoundConfig(mode=UltrasoundMode.VISUAL_ONLY))

    print("\n✅ All components initialized!")
    print("="*60)

    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Cannot open webcam!")
        print("💡 Check if webcam is connected")
        return

    # Set resolution
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    print("\n🎮 CONTROLS:")
    print("   q: Quit")
    print("   u: Toggle ultrasound")
    print("   s: Save screenshot")
    print("\n▶️ Starting detection... Press 'q' to quit")

    frame_count = 0
    aggression_count = 0
    consecutive_aggression = 0

    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("📺 End of video stream")
                break

            frame_count += 1

            # Step 1: Detect dogs and people
            detections = detector.detect(frame)

            # Step 2: Show what's detected
            dogs = [d for d in detections if d[5] == 'dog']
            people = [d for d in detections if d[5] == 'person']
            cats = [d for d in detections if d[5] == 'cat']

            # Step 3: Draw all detections
            output = detector.draw_detections(frame, detections)

            # Step 4: Process each dog for aggression
            has_aggressive_dog = False

            for detection in detections:
                x1, y1, x2, y2, confidence, class_name = detection

                if class_name == "dog":
                    # Extract dog ROI for classification
                    roi = frame[y1:y2, x1:x2]

                    if roi.size > 0 and roi.shape[0] > 10 and roi.shape[1] > 10:
                        # Classify aggression (placeholder)
                        is_aggressive, agg_conf = classifier.classify(roi)

                        if is_aggressive:
                            has_aggressive_dog = True

                            # Highlight aggressive dog
                            cv2.rectangle(output, (x1, y1), (x2, y2), (0, 0, 255), 3)
                            cv2.putText(output, f"AGGRESSIVE! {agg_conf:.2f}",
                                       (x1, y1 - 20), cv2.FONT_HERSHEY_SIMPLEX,
                                       0.6, (0, 0, 255), 2)

            # Step 5: Handle aggression detection
            if has_aggressive_dog:
                consecutive_aggression += 1
                if consecutive_aggression >= 3 and not ultrasound.is_active:
                    print("🚨 Aggressive dog detected! Activating ultrasound...")
                    ultrasound.activate()
                    aggression_count += 1
            else:
                consecutive_aggression = 0
                if ultrasound.is_active:
                    print("✅ No aggression. Deactivating ultrasound...")
                    ultrasound.deactivate()

            # Step 6: Add status overlay
            stats = f"Frame: {frame_count} | Dogs: {len(dogs)} | People: {len(people)}"
            agg_status = f"Aggression: {consecutive_aggression}/3"
            us_status = "🔊 ON" if ultrasound.is_active else "🔇 OFF"

            # Draw semi-transparent background for text
            overlay = output.copy()
            cv2.rectangle(overlay, (0, 0), (640, 100), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.5, output, 0.5, 0, output)

            # Draw text
            cv2.putText(output, stats, (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(output, agg_status, (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 0, 255) if consecutive_aggression > 0 else (0, 255, 0), 2)
            cv2.putText(output, us_status, (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                       (0, 255, 0) if ultrasound.is_active else (255, 0, 0), 2)

            # Step 7: Display
            cv2.imshow("Dog Aggression System", output)

            # Step 8: Handle keys
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("\n🛑 Quitting...")
                break
            elif key == ord('u'):
                if ultrasound.is_active:
                    ultrasound.deactivate()
                else:
                    ultrasound.activate()
                print(f"🔊 Ultrasound: {'ACTIVE' if ultrasound.is_active else 'INACTIVE'}")
            elif key == ord('s'):
                timestamp = time.strftime("%Y%m%d_%H%M%S")
                filename = f"screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, output)
                print(f"📸 Saved: {filename}")

    except KeyboardInterrupt:
        print("\n⚠️ Interrupted")

    finally:
        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        ultrasound.cleanup()

        print(f"\n📊 FINAL STATISTICS:")
        print(f"   Frames processed: {frame_count}")
        print(f"   Aggression events: {aggression_count}")
        print("✅ System stopped!")


if __name__ == "__main__":
    # First, check if all files exist
    required_files = ['dog_detector.py', 'aggression_classifier.py', 'ultrasonic_simulator.py']

    print("🔍 Checking required files...")
    for file in required_files:
        if Path(file).exists():
            print(f"✅ {file}")
        else:
            print(f"❌ {file} - NOT FOUND!")
            print(f"💡 Make sure {file} is in the same directory")

    # Run the system
    main()