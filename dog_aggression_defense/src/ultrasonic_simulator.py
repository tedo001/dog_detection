# Ultrasonic simulator 
"""
Simulated Ultrasonic Repeller
In simulation mode, plays audible tone. Ready for hardware integration.
"""
import threading
import time
import numpy as np

try:
    import simpleaudio as sa

    AUDIO_AVAILABLE = True
except ImportError:
    print("Warning: simpleaudio not installed. Simulation sound disabled.")
    AUDIO_AVAILABLE = False


class UltrasonicSimulator:
    def __init__(self, simulation_mode=True):
        """
        Initialize ultrasonic repeller simulator

        Args:
            simulation_mode: If True, plays audible tone instead of ultrasonic
        """
        self.simulation_mode = simulation_mode
        self.is_active = False
        self.sound_enabled = True
        self.play_thread = None

        # Audio parameters (for simulation)
        self.frequency = 1000  # Hz (audible for simulation)
        self.sample_rate = 44100  # Hz
        self.duration = 0.1  # seconds per beep

        print(f"Ultrasonic repeller initialized in {'SIMULATION' if simulation_mode else 'HARDWARE'} mode")

    def activate(self):
        """Activate the repeller"""
        if not self.is_active:
            self.is_active = True
            print("REPELLER ACTIVATED")

            if self.simulation_mode and self.sound_enabled:
                self._start_simulation()
            else:
                self._start_hardware()

    def deactivate(self):
        """Deactivate the repeller"""
        if self.is_active:
            self.is_active = False
            print("Repeller deactivated")

            if self.play_thread and self.play_thread.is_alive():
                # Signal thread to stop
                self.is_active = False

    def _start_simulation(self):
        """Start audible simulation (plays beeping sound)"""
        if not AUDIO_AVAILABLE:
            print("Audio simulation not available")
            return

        self.play_thread = threading.Thread(target=self._play_beeping_sound, daemon=True)
        self.play_thread.start()

    def _start_hardware(self):
        """Start hardware repeller (placeholder for GPIO/Arduino integration)"""
        print("HARDWARE MODE: Would activate ultrasonic transducer here")
        print("Connect to GPIO/Arduino pin for real hardware")

        # Hardware integration would go here:
        # import RPi.GPIO as GPIO  # For Raspberry Pi
        # GPIO.setup(18, GPIO.OUT)
        # GPIO.output(18, GPIO.HIGH)

    def _play_beeping_sound(self):
        """Generate and play beeping sound"""
        try:
            while self.is_active and self.sound_enabled:
                # Generate beep
                t = np.linspace(0, self.duration, int(self.sample_rate * self.duration), False)
                wave = 0.5 * np.sin(2 * np.pi * self.frequency * t)

                # Convert to 16-bit PCM
                audio = (wave * 32767).astype(np.int16)

                # Play sound
                play_obj = sa.play_buffer(audio, 1, 2, self.sample_rate)

                # Wait for beep to complete or stop signal
                beep_start = time.time()
                while time.time() - beep_start < self.duration:
                    if not self.is_active:
                        play_obj.stop()
                        return
                    time.sleep(0.01)

                # Brief pause between beeps
                pause_start = time.time()
                while time.time() - pause_start < 0.1:  # 100ms pause
                    if not self.is_active:
                        return
                    time.sleep(0.01)

        except Exception as e:
            print(f"Audio error: {e}")

    def toggle_sound(self):
        """Toggle simulation sound on/off"""
        self.sound_enabled = not self.sound_enabled
        return self.sound_enabled

    def cleanup(self):
        """Cleanup resources"""
        self.deactivate()

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)

        print("Repeller cleaned up")


# Example usage
if __name__ == "__main__":
    repeller = UltrasonicSimulator()

    print("Testing repeller...")
    repeller.activate()
    time.sleep(2)  # Run for 2 seconds
    repeller.deactivate()

    print("Test complete")