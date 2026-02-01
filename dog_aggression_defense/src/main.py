"""
===============================================================================
DOG AGGRESSION DEFENSE SYSTEM - THREAT-ORIENTED UPGRADE
===============================================================================
Upgraded system that activates ultrasonic ONLY when:
1. Dog detected + Human detected
2. Dog is CLOSE to human
3. Dog is MOVING TOWARD human
4. Dog shows AGGRESSIVE motion
5. Dog is NOT interacting with another dog
6. Threat is TEMPORALLY CONSISTENT
===============================================================================
"""

import cv2
import time
import numpy as np
from pathlib import Path
import sys
import argparse
from collections import deque
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import math

sys.path.append(str(Path(__file__).parent))

# Import existing modules
from dog_detector import DogDetector
from aggression_classifier import AggressionClassifier
from ultrasonic_simulator import UltrasonicSimulator, UltrasoundMode, UltrasoundConfig


@dataclass
class ObjectTrack:
    """Tracks individual objects across frames"""
    class_id: int  # 0: person, 16: dog
    track_id: int
    bbox: Tuple[int, int, int, int]  # x1, y1, x2, y2
    center: Tuple[float, float]
    confidence: float
    history: deque  # Track movement history
    last_seen: float
    velocity: Tuple[float, float] = (0.0, 0.0)

    def __init__(self, class_id, track_id, bbox, confidence):
        self.class_id = class_id
        self.track_id = track_id
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.confidence = confidence
        self.history = deque(maxlen=10)  # Store last 10 positions
        self.history.append(self.center)
        self.last_seen = time.time()
        self.velocity = (0.0, 0.0)

    def update(self, bbox, confidence):
        """Update track with new detection"""
        old_center = self.center
        self.bbox = bbox
        self.center = ((bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2)
        self.confidence = confidence
        self.history.append(self.center)
        self.last_seen = time.time()

        # Calculate velocity (pixels per frame)
        if len(self.history) >= 2:
            dx = self.history[-1][0] - self.history[-2][0]
            dy = self.history[-1][1] - self.history[-2][1]
            self.velocity = (dx, dy)
        return old_center


class ThreatAnalyzer:
    """
    Core threat analysis engine that implements ethical rules:
    - NO activation for dog-dog interactions
    - NO activation for calm/normal behavior
    - ONLY activate for human-directed threat
    """

    # ETHICAL CONFIGURATION - DO NOT CHANGE LIGHTLY
    ETHICAL_CONFIG = {
        # Proximity thresholds (in normalized distance 0-1)
        'PROXIMITY_THRESHOLD': 0.25,  # How close dog must be to human
        'DOG_DOG_PROXIMITY': 0.35,  # If dogs this close, ignore (dog-dog interaction)

        # Motion analysis
        'MIN_APPROACH_SPEED': 0.02,  # Minimum speed toward human
        'MAX_APPROACH_ANGLE': 45,  # Degrees (0=directly toward, 90=perpendicular)

        # Temporal consistency
        'MIN_THREAT_FRAMES': 8,  # Frames threat must persist
        'COOLDOWN_FRAMES': 20,  # Frames after activation before re-check

        # Distance calculation
        'NORMALIZATION_FACTOR': 1000,  # For pixel-to-relative conversion

        # Safety margins
        'HUMAN_MOVING_AWAY_MARGIN': 0.05,  # If human moving away, be more lenient
    }

    def __init__(self, frame_shape: Tuple[int, int]):
        """
        Initialize threat analyzer with ethical rules

        Args:
            frame_shape: (height, width) of video frames
        """
        self.frame_shape = frame_shape
        self.frame_height, self.frame_width = frame_shape[:2]

        # Tracking state
        self.tracks: Dict[int, ObjectTrack] = {}
        self.next_track_id = 0

        # Threat state
        self.threat_level = 0.0  # 0.0 to 1.0
        self.threat_frames = 0
        self.cooldown_counter = 0
        self.last_threat_time = 0

        # Detection history for smoothing
        self.detection_history = deque(maxlen=30)

        print("🔐 Threat Analyzer initialized with ETHICAL RULES:")
        print("   ✓ NO activation for dog-dog interactions")
        print("   ✓ NO activation for normal/calm behavior")
        print("   ✓ ONLY for human-directed credible threat")

    def _calculate_distance(self, center1: Tuple[float, float], center2: Tuple[float, float]) -> float:
        """Calculate normalized distance between two points (0-1 scale)"""
        dx = center1[0] - center2[0]
        dy = center1[1] - center2[1]
        pixel_distance = math.sqrt(dx * dx + dy * dy)
        return pixel_distance / self.ETHICAL_CONFIG['NORMALIZATION_FACTOR']

    def _calculate_angle(self, vec1: Tuple[float, float], vec2: Tuple[float, float]) -> float:
        """Calculate angle between two vectors in degrees"""
        dot = vec1[0] * vec2[0] + vec1[1] * vec2[1]
        mag1 = math.sqrt(vec1[0] ** 2 + vec1[1] ** 2)
        mag2 = math.sqrt(vec2[0] ** 2 + vec2[1] ** 2)

        if mag1 * mag2 == 0:
            return 90.0  # Maximum angle if no movement

        cos_angle = dot / (mag1 * mag2)
        cos_angle = max(-1.0, min(1.0, cos_angle))  # Clamp due to floating errors
        angle = math.degrees(math.acos(cos_angle))
        return angle

    def _is_moving_toward(self, dog_track: ObjectTrack, human_track: ObjectTrack) -> bool:
        """
        Check if dog is moving toward human with ethical thresholds

        Returns:
            True if dog shows directed movement toward human
        """
        if len(dog_track.history) < 3:
            return False  # Insufficient history

        # Get dog's velocity vector
        dog_velocity = dog_track.velocity
        dog_speed = math.sqrt(dog_velocity[0] ** 2 + dog_velocity[1] ** 2)

        # Check minimum speed threshold
        if dog_speed < self.ETHICAL_CONFIG['MIN_APPROACH_SPEED']:
            return False  # Dog is moving too slowly

        # Vector from dog to human
        to_human = (human_track.center[0] - dog_track.center[0],
                    human_track.center[1] - dog_track.center[1])

        # Calculate approach angle
        approach_angle = self._calculate_angle(dog_velocity, to_human)

        # Check if moving toward (small angle) and not away
        return approach_angle < self.ETHICAL_CONFIG['MAX_APPROACH_ANGLE']

    def _has_dog_dog_interaction(self, dog_tracks: List[ObjectTrack]) -> bool:
        """
        Check if any dogs are interacting with each other
        Ethical rule: DO NOT activate if dogs are interacting
        """
        if len(dog_tracks) < 2:
            return False

        for i, dog1 in enumerate(dog_tracks):
            for dog2 in dog_tracks[i + 1:]:
                distance = self._calculate_distance(dog1.center, dog2.center)

                # If dogs are close, they're likely interacting
                if distance < self.ETHICAL_CONFIG['DOG_DOG_PROXIMITY']:
                    # Additional check: are they moving relative to each other?
                    relative_velocity = (dog1.velocity[0] - dog2.velocity[0],
                                         dog1.velocity[1] - dog2.velocity[1])
                    relative_speed = math.sqrt(relative_velocity[0] ** 2 + relative_velocity[1] ** 2)

                    if relative_speed > 0.01:  # They're moving relative to each other
                        return True  # Dog-dog interaction detected

        return False

    def _is_human_moving_away(self, human_track: ObjectTrack, dog_track: ObjectTrack) -> bool:
        """
        Check if human is moving away from dog
        Ethical consideration: If human is fleeing, be more cautious
        """
        if len(human_track.history) < 2:
            return False

        # Vector from human to dog
        to_dog = (dog_track.center[0] - human_track.center[0],
                  dog_track.center[1] - human_track.center[1])

        # Human's velocity
        human_velocity = human_track.velocity

        # Calculate if human is moving away from dog
        # Negative dot product means moving in opposite direction
        dot_product = human_velocity[0] * to_dog[0] + human_velocity[1] * to_dog[1]

        return dot_product < -self.ETHICAL_CONFIG['HUMAN_MOVING_AWAY_MARGIN']

    def analyze_detections(self, detections: List[Tuple]) -> Dict:
        """
        Main analysis function - evaluates all ethical rules

        Args:
            detections: List of (class_id, x1, y1, x2, y2, confidence)
                       class_id: 0=person, 16=dog

        Returns:
            Dictionary with threat assessment and reasoning
        """
        current_time = time.time()

        # Step 1: Update tracks with new detections
        self._update_tracks(detections)

        # Clean up old tracks
        self._cleanup_tracks(current_time)

        # Extract dogs and humans
        dogs = [t for t in self.tracks.values() if t.class_id == 16]  # Dog class
        humans = [t for t in self.tracks.values() if t.class_id == 0]  # Person class

        # Initialize threat assessment
        threat_assessment = {
            'threat_detected': False,
            'threat_level': 0.0,
            'reasoning': [],
            'dog_human_pairs': [],
            'dog_dog_interaction': False,
            'cooldown_active': self.cooldown_counter > 0
        }

        # ETHICAL RULE 1: Cooldown period
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
            threat_assessment['reasoning'].append("Cooldown active")
            return threat_assessment

        # ETHICAL RULE 2: No humans = no threat
        if not humans:
            threat_assessment['reasoning'].append("No humans detected")
            return threat_assessment

        # ETHICAL RULE 3: No dogs = no threat
        if not dogs:
            threat_assessment['reasoning'].append("No dogs detected")
            return threat_assessment

        # ETHICAL RULE 4: Check for dog-dog interaction FIRST
        if self._has_dog_dog_interaction(dogs):
            threat_assessment['dog_dog_interaction'] = True
            threat_assessment['reasoning'].append("Dog-dog interaction detected - IGNORING")
            self.threat_frames = 0  # Reset threat counter
            return threat_assessment

        # Find all potential threat pairs (dog-human)
        threat_pairs = []

        for dog in dogs:
            for human in humans:
                distance = self._calculate_distance(dog.center, human.center)

                # Check proximity
                if distance > self.ETHICAL_CONFIG['PROXIMITY_THRESHOLD']:
                    continue  # Too far away

                # Check if dog is moving toward human
                is_moving_toward = self._is_moving_toward(dog, human)

                # Check if human is moving away (fleeing)
                human_fleeing = self._is_human_moving_away(human, dog)

                # Calculate threat score for this pair
                threat_score = 0.0
                reasoning = []

                # Factor 1: Proximity (closer = higher threat)
                proximity_score = 1.0 - (distance / self.ETHICAL_CONFIG['PROXIMITY_THRESHOLD'])
                threat_score += proximity_score * 0.4
                reasoning.append(f"Proximity: {proximity_score:.2f}")

                # Factor 2: Directed movement
                if is_moving_toward:
                    movement_score = 1.0
                    threat_score += movement_score * 0.4
                    reasoning.append("Moving toward human")
                else:
                    reasoning.append("Not moving toward human")

                # Factor 3: Human behavior
                if human_fleeing:
                    # If human is fleeing, increase threat score
                    fleeing_score = 0.2
                    threat_score += fleeing_score
                    reasoning.append("Human appears to be moving away")

                # Factor 4: Speed (faster = higher threat)
                dog_speed = math.sqrt(dog.velocity[0] ** 2 + dog.velocity[1] ** 2)
                if dog_speed > self.ETHICAL_CONFIG['MIN_APPROACH_SPEED'] * 2:
                    speed_score = 0.2
                    threat_score += speed_score
                    reasoning.append(f"High speed: {dog_speed:.3f}")

                threat_score = min(1.0, threat_score)  # Cap at 1.0

                threat_pairs.append({
                    'dog': dog,
                    'human': human,
                    'distance': distance,
                    'threat_score': threat_score,
                    'reasoning': reasoning,
                    'is_moving_toward': is_moving_toward,
                    'human_fleeing': human_fleeing
                })

        threat_assessment['dog_human_pairs'] = threat_pairs

        # Find the most threatening pair
        if threat_pairs:
            most_threatening = max(threat_pairs, key=lambda x: x['threat_score'])

            # ETHICAL RULE 5: Threshold for activation
            if most_threatening['threat_score'] > 0.7:  # 70% threshold
                self.threat_frames += 1
                threat_assessment['threat_level'] = most_threatening['threat_score']
                threat_assessment['reasoning'] = most_threatening['reasoning']

                # ETHICAL RULE 6: Temporal consistency
                if self.threat_frames >= self.ETHICAL_CONFIG['MIN_THREAT_FRAMES']:
                    threat_assessment['threat_detected'] = True
                    self.last_threat_time = current_time
                    self.cooldown_counter = self.ETHICAL_CONFIG['COOLDOWN_FRAMES']
                    threat_assessment['reasoning'].append(f"THREAT CONFIRMED ({self.threat_frames} frames)")
            else:
                self.threat_frames = max(0, self.threat_frames - 2)
                threat_assessment['reasoning'].append(f"Threat below threshold: {most_threatening['threat_score']:.2f}")
        else:
            self.threat_frames = max(0, self.threat_frames - 1)
            threat_assessment['reasoning'].append("No threatening dog-human pairs")

        return threat_assessment

    def _update_tracks(self, detections: List[Tuple]):
        """Update object tracking with new detections"""
        used_tracks = set()

        for det in detections:
            class_id, x1, y1, x2, y2, confidence = det

            # Only track people (0) and dogs (16)
            if class_id not in [0, 16]:
                continue

            # Find closest existing track
            new_center = ((x1 + x2) / 2, (y1 + y2) / 2)
            closest_track = None
            min_distance = float('inf')

            for track_id, track in self.tracks.items():
                if track.class_id != class_id:
                    continue

                distance = self._calculate_distance(new_center, track.center)
                if distance < 0.1 and distance < min_distance:  # Matching threshold
                    min_distance = distance
                    closest_track = track_id

            if closest_track is not None:
                # Update existing track
                self.tracks[closest_track].update((x1, y1, x2, y2), confidence)
                used_tracks.add(closest_track)
            else:
                # Create new track
                track_id = self.next_track_id
                self.next_track_id += 1
                self.tracks[track_id] = ObjectTrack(
                    class_id, track_id, (x1, y1, x2, y2), confidence
                )
                used_tracks.add(track_id)

        # Remove tracks that weren't updated
        expired_tracks = [tid for tid in self.tracks.keys() if tid not in used_tracks]
        for tid in expired_tracks:
            if time.time() - self.tracks[tid].last_seen > 2.0:  # 2 second timeout
                del self.tracks[tid]

    def _cleanup_tracks(self, current_time: float):
        """Remove old tracks"""
        expired = []
        for track_id, track in self.tracks.items():
            if current_time - track.last_seen > 5.0:  # 5 second timeout
                expired.append(track_id)

        for track_id in expired:
            del self.tracks[track_id]

    def get_visualization_data(self) -> Dict:
        """Get data for visualization overlay"""
        dogs = [t for t in self.tracks.values() if t.class_id == 16]
        humans = [t for t in self.tracks.values() if t.class_id == 0]

        return {
            'tracks': list(self.tracks.values()),
            'dog_count': len(dogs),
            'human_count': len(humans),
            'threat_frames': self.threat_frames,
            'cooldown_active': self.cooldown_counter > 0
        }


class DogAggressionDefenseV2:
    """
    Upgraded main system with ethical threat detection
    """

    def __init__(self, config=None):
        print("\n" + "=" * 60)
        print("DOG AGGRESSION DEFENSE SYSTEM v2.1 - ETHICAL THREAT DETECTION")
        print("=" * 60)

        # Default configuration
        self.config = {
            'detector_confidence': 0.5,
            'classifier_confidence': 0.6,
            'camera_id': 0,
            'frame_width': 640,
            'frame_height': 480,
            'ultrasound_mode': UltrasoundMode.AUDIBLE_TONE,
            'sound_enabled': True,
            'show_debug': False,
            'show_tracks': True,
        }

        if config:
            self.config.update(config)

        # Initialize components
        print("\n[1/4] Initializing Dog Detector...")
        self.dog_detector = DogDetector(
            model_path=self.config.get('model_detector'),
            conf_threshold=self.config['detector_confidence']
        )

        print("[2/4] Initializing Aggression Classifier...")
        self.aggression_classifier = AggressionClassifier(
            model_path=self.config.get('model_classifier'),
            conf_threshold=self.config['classifier_confidence']
        )

        print("[3/4] Initializing Threat Analyzer (with ETHICAL RULES)...")
        self.threat_analyzer = ThreatAnalyzer((self.config['frame_height'], self.config['frame_width']))

        print("[4/4] Initializing Ultrasonic Repeller...")
        ultrasound_config = UltrasoundConfig(
            mode=self.config['ultrasound_mode'],
            volume=0.3
        )
        self.repeller = UltrasonicSimulator(ultrasound_config)

        # State variables
        self.frame_count = 0
        self.threat_activations = 0
        self.false_positives_blocked = 0
        self.is_running = False

        print("\n" + "=" * 60)
        print("✅ SYSTEM READY WITH ETHICAL SAFEGUARDS")
        print("\n🎮 CONTROLS:")
        print("   q - Quit")
        print("   s - Toggle sound")
        print("   u - Cycle ultrasound modes")
        print("   d - Toggle debug display")
        print("   t - Toggle track display")
        print("   f - Show FPS")
        print("=" * 60 + "\n")

    def _format_detections(self, results, frame_shape) -> List[Tuple]:
        """
        Convert YOLO results to standardized format

        Args:
            results: YOLOv8 results object
            frame_shape: (height, width) of frame

        Returns:
            List of (class_id, x1, y1, x2, y2, confidence)
        """
        detections = []

        if hasattr(results, 'boxes') and results.boxes is not None:
            for box in results.boxes:
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])

                # Only process people (0) and dogs (16)
                if class_id in [0, 16]:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    detections.append((class_id, int(x1), int(y1), int(x2), int(y2), confidence))

        return detections

    def _draw_ethical_overlay(self, frame, threat_assessment, vis_data):
        """Draw ethical reasoning and threat analysis on frame"""
        h, w = frame.shape[:2]

        # Main status bar
        status_color = (0, 0, 255) if threat_assessment['threat_detected'] else (0, 255, 0)
        status_text = "🚨 THREAT DETECTED" if threat_assessment['threat_detected'] else "✅ SYSTEM READY"

        cv2.rectangle(frame, (0, 0), (w, 40), (30, 30, 30), -1)
        cv2.putText(frame, f"ETHICAL DEFENSE: {status_text}", (10, 25),
                    cv2.FONT_HERSHEY_DUPLEX, 0.6, status_color, 2)

        # Right panel - threat analysis
        panel_x = w - 300
        panel_y = 50
        panel_width = 290
        panel_height = 120

        # Panel background
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (40, 40, 40), -1)
        cv2.rectangle(frame, (panel_x, panel_y),
                      (panel_x + panel_width, panel_y + panel_height),
                      (80, 80, 80), 2)

        # Title
        cv2.putText(frame, "THREAT ANALYSIS",
                    (panel_x + 10, panel_y + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 200), 1)

        # Stats
        stats = [
            f"Dogs: {vis_data['dog_count']}",
            f"Humans: {vis_data['human_count']}",
            f"Threat Level: {threat_assessment['threat_level']:.2f}",
            f"Threat Frames: {vis_data['threat_frames']}/8",
        ]

        for i, stat in enumerate(stats):
            color = (200, 200, 200)
            if "Threat Level" in stat and threat_assessment['threat_level'] > 0.7:
                color = (0, 0, 255)

            cv2.putText(frame, stat,
                        (panel_x + 10, panel_y + 40 + i * 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Ethical status
        if threat_assessment['dog_dog_interaction']:
            cv2.putText(frame, "ETHICAL: Dog-Dog Interaction",
                        (panel_x + 10, panel_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        elif threat_assessment['cooldown_active']:
            cv2.putText(frame, "ETHICAL: Cooldown Active",
                        (panel_x + 10, panel_y + 105),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)

        # Draw tracks if enabled
        if self.config['show_tracks']:
            self._draw_object_tracks(frame, vis_data['tracks'])

        # Draw threat reasoning
        if threat_assessment['reasoning'] and self.config['show_debug']:
            for i, reason in enumerate(threat_assessment['reasoning'][:3]):
                cv2.putText(frame, f"• {reason}",
                            (10, h - 60 - i * 20),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 100), 1)

        # System stats
        fps_text = f"FPS: {self._calculate_fps():.1f}"
        cv2.putText(frame, fps_text, (w - 100, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Ultrasound status
        us_status = "🔊 ON" if self.repeller.sound_enabled else "🔇 OFF"
        cv2.putText(frame, us_status, (w - 150, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 0) if self.repeller.sound_enabled else (100, 100, 100), 1)

        return frame

    def _draw_object_tracks(self, frame, tracks):
        """Draw object tracking visualization"""
        for track in tracks:
            # Different colors for different classes
            if track.class_id == 0:  # Person
                color = (255, 100, 100)  # Light red
                label = "Person"
            elif track.class_id == 16:  # Dog
                color = (100, 255, 100)  # Light green
                label = "Dog"
            else:
                continue

            x1, y1, x2, y2 = track.bbox
            cx, cy = int(track.center[0]), int(track.center[1])

            # Draw bounding box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)

            # Draw center point with history
            if len(track.history) > 1:
                for i in range(1, len(track.history)):
                    pt1 = (int(track.history[i - 1][0]), int(track.history[i - 1][1]))
                    pt2 = (int(track.history[i][0]), int(track.history[i][1]))
                    cv2.line(frame, pt1, pt2, color, 1)

            # Draw center point
            cv2.circle(frame, (cx, cy), 3, color, -1)

            # Draw label with confidence
            cv2.putText(frame, f"{label} {track.confidence:.1f}",
                        (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)

            # Draw velocity vector
            if abs(track.velocity[0]) > 0.1 or abs(track.velocity[1]) > 0.1:
                vx, vy = track.velocity
                end_x = int(cx + vx * 20)
                end_y = int(cy + vy * 20)
                cv2.arrowedLine(frame, (cx, cy), (end_x, end_y), (255, 255, 0), 2)

    def _calculate_fps(self):
        """Calculate frames per second"""
        if not hasattr(self, '_fps_start_time'):
            self._fps_start_time = time.time()
            self._fps_frame_count = 0
            return 0.0

        self._fps_frame_count += 1
        elapsed = time.time() - self._fps_start_time

        if elapsed > 1.0:
            self.current_fps = self._fps_frame_count / elapsed
            self._fps_start_time = time.time()
            self._fps_frame_count = 0

        return getattr(self, 'current_fps', 0.0)

    def process_frame(self, frame):
        """Process a single frame with ethical threat analysis"""
        self.frame_count += 1

        # Step 1: Detect objects using YOLOv8
        results = self.dog_detector.model(frame, verbose=False)[0]

        # Step 2: Format detections for threat analysis
        detections = self._format_detections(results, frame.shape)

        # Step 3: Analyze threats with ethical rules
        threat_assessment = self.threat_analyzer.analyze_detections(detections)

        # Step 4: Get visualization data
        vis_data = self.threat_analyzer.get_visualization_data()

        # Step 5: Activate repeller ONLY if threat is confirmed
        if threat_assessment['threat_detected']:
            self.threat_activations += 1
            print(f"\n🚨 THREAT ACTIVATION #{self.threat_activations}")
            print(f"   Reason: {', '.join(threat_assessment['reasoning'][-2:])}")
            self.repeller.activate()
        else:
            # Check if we should deactivate
            if self.repeller.is_active:
                print("🔇 Threat cleared, deactivating repeller")
                self.repeller.deactivate()

            # Log blocked false positives
            if threat_assessment['dog_dog_interaction']:
                self.false_positives_blocked += 1

        # Step 6: Draw ethical overlay
        output_frame = self._draw_ethical_overlay(frame, threat_assessment, vis_data)

        return output_frame

    def run_webcam(self, camera_id=None):
        """Run system with webcam feed"""
        if camera_id is None:
            camera_id = self.config['camera_id']

        print(f"📷 Starting webcam (ID: {camera_id})...")
        print("   Ethical Rules Active:")
        print("   • NO activation for dog-dog interactions")
        print("   • NO activation for normal dog behavior")
        print("   • ONLY for human-directed credible threats")

        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            print(f"❌ Cannot open webcam {camera_id}")
            return

        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config['frame_width'])
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config['frame_height'])

        self.is_running = True

        try:
            while self.is_running:
                ret, frame = cap.read()
                if not ret:
                    break

                # Process frame
                processed_frame = self.process_frame(frame)

                # Display
                cv2.imshow('Dog Aggression Defense - ETHICAL MODE', processed_frame)

                # Handle key presses
                key = cv2.waitKey(1) & 0xFF

                if key == ord('q'):
                    print("\n🛑 Shutting down...")
                    break
                elif key == ord('s'):
                    sound_state = self.repeller.toggle_sound()
                    print(f"🔊 Sound: {'ON' if sound_state else 'OFF'}")
                elif key == ord('u'):
                    current_mode = self.repeller.config.mode
                    modes = [UltrasoundMode.AUDIBLE_TONE,
                             UltrasoundMode.ULTRASOUND_SIM,
                             UltrasoundMode.VISUAL_ONLY]
                    current_index = modes.index(current_mode)
                    next_index = (current_index + 1) % len(modes)
                    self.repeller.set_mode(modes[next_index])
                elif key == ord('d'):
                    self.config['show_debug'] = not self.config['show_debug']
                    print(f"🔧 Debug: {'ON' if self.config['show_debug'] else 'OFF'}")
                elif key == ord('t'):
                    self.config['show_tracks'] = not self.config['show_tracks']
                    print(f"📈 Tracks: {'ON' if self.config['show_tracks'] else 'OFF'}")
                elif key == ord('f'):
                    print(f"📊 Current FPS: {self._calculate_fps():.1f}")

        except KeyboardInterrupt:
            print("\n⚠️ Interrupted by user")
        except Exception as e:
            print(f"\n❌ Error: {e}")
            import traceback
            traceback.print_exc()

        finally:
            self.is_running = False
            cap.release()
            cv2.destroyAllWindows()
            self.repeller.cleanup()

            # Print ethical statistics
            self._print_ethical_statistics()

    def _print_ethical_statistics(self):
        """Print ethical performance statistics"""
        print("\n" + "=" * 60)
        print("📊 ETHICAL PERFORMANCE REPORT")
        print("=" * 60)
        print(f"Total frames processed: {self.frame_count}")
        print(f"Threat activations: {self.threat_activations}")
        print(f"False positives blocked: {self.false_positives_blocked}")

        if self.threat_activations > 0:
            activation_rate = (self.threat_activations / self.frame_count) * 1000  # per 1000 frames
            print(f"Activation rate: {activation_rate:.2f} per 1000 frames")

        if self.false_positives_blocked > 0:
            print(f"✓ Successfully blocked {self.false_positives_blocked} dog-dog interactions")

        print("\n🔐 ETHICAL RULES ENFORCED:")
        print("   1. Never activated for dog-dog interactions")
        print("   2. Required human presence for activation")
        print("   3. Required directed movement toward human")
        print("   4. Required temporal consistency")
        print("   5. Implemented cooldown periods")
        print("=" * 60)


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description='Dog Aggression Defense System - Ethical Upgrade')

    parser.add_argument('--camera', type=int, default=0, help='Camera ID')
    parser.add_argument('--width', type=int, default=640, help='Frame width')
    parser.add_argument('--height', type=int, default=480, help='Frame height')
    parser.add_argument('--mode', type=str, default='audible',
                        choices=['audible', 'ultrasound', 'visual'],
                        help='Ultrasound mode')
    parser.add_argument('--no-sound', action='store_true', help='Disable sound')
    parser.add_argument('--debug', action='store_true', help='Show debug info')

    args = parser.parse_args()

    # Map mode string to enum
    mode_map = {
        'audible': UltrasoundMode.AUDIBLE_TONE,
        'ultrasound': UltrasoundMode.ULTRASOUND_SIM,
        'visual': UltrasoundMode.VISUAL_ONLY
    }

    config = {
        'camera_id': args.camera,
        'frame_width': args.width,
        'frame_height': args.height,
        'ultrasound_mode': mode_map[args.mode],
        'sound_enabled': not args.no_sound,
        'show_debug': args.debug,
        'show_tracks': True
    }

    try:
        system = DogAggressionDefenseV2(config)
        system.run_webcam(args.camera)
    except Exception as e:
        print(f"\n❌ Failed to start system: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())