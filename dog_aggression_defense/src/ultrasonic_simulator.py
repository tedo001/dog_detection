"""
Enhanced Ultrasonic Repeller Simulator
Simulates different types of ultrasound outputs for testing
"""
import threading
import time
import numpy as np
import warnings
from enum import Enum
from dataclasses import dataclass
from typing import Optional

# Try to import audio libraries with fallbacks
try:
    import simpleaudio as sa

    AUDIO_AVAILABLE = True
except ImportError:
    AUDIO_AVAILABLE = False
    print("Note: simpleaudio not available. Install with: pip install simpleaudio")

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
except ImportError:
    PYAUDIO_AVAILABLE = False
    print("Note: pyaudio not available. Install with: pip install pyaudio")


class UltrasoundMode(Enum):
    """Different simulation modes for ultrasound"""
    AUDIBLE_TONE = "audible"  # Audible beeping (for testing)
    ULTRASOUND_SIM = "ultrasound"  # Simulated ultrasound (silent, logs only)
    VISUAL_ONLY = "visual"  # Visual indicators only
    HARDWARE = "hardware"  # Real hardware mode (placeholder)


@dataclass
class UltrasoundConfig:
    """Configuration for ultrasound simulation"""
    mode: UltrasoundMode = UltrasoundMode.AUDIBLE_TONE
    audible_frequency: int = 1000  # Hz (audible range)
    ultrasound_frequency: int = 25000  # Hz (typical dog repeller)
    volume: float = 0.5  # 0.0 to 1.0
    pulse_duration: float = 0.1  # seconds
    pulse_interval: float = 0.1  # seconds between pulses
    burst_count: int = 10  # pulses per activation
    ramp_up_time: float = 0.5  # seconds to reach full intensity


class UltrasonicSimulator:
    """
    Enhanced ultrasonic repeller simulator with multiple modes
    """

    def __init__(self, config: Optional[UltrasoundConfig] = None):
        """
        Initialize ultrasound simulator

        Args:
            config: Ultrasound configuration (defaults to audible mode)
        """
        self.config = config or UltrasoundConfig()

        # State variables
        self.is_active = False
        self.sound_enabled = True
        self.play_thread = None
        self.stop_signal = threading.Event()

        # Audio stream (for PyAudio mode)
        self.pyaudio_stream = None
        self.pyaudio_instance = None

        # Statistics
        self.activation_count = 0
        self.total_active_time = 0.0
        self.last_activation_time = None

        # Frequency ranges
        self.DOG_HEARING_RANGE = (67, 45000)  # Hz (dog hearing range)
        self.HUMAN_HEARING_RANGE = (20, 20000)  # Hz (human hearing range)

        print(f"\n🔊 Ultrasound Simulator Initialized")
        print(f"   Mode: {self.config.mode.value}")
        print(f"   Frequency: {self._get_current_frequency()} Hz")

        if self.config.mode == UltrasoundMode.AUDIBLE_TONE:
            print(f"   🔊 Audible mode - You will hear beeping sounds")
        elif self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
            print(f"   📡 Ultrasound simulation - Silent mode (logs only)")
        elif self.config.mode == UltrasoundMode.VISUAL_ONLY:
            print(f"   👁️ Visual only mode - No audio output")

    def _get_current_frequency(self) -> int:
        """Get the current frequency based on mode"""
        if self.config.mode == UltrasoundMode.AUDIBLE_TONE:
            return self.config.audible_frequency
        elif self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
            return self.config.ultrasound_frequency
        return 0

    def activate(self):
        """Activate the ultrasonic repeller"""
        if not self.is_active:
            self.is_active = True
            self.stop_signal.clear()
            self.activation_count += 1
            self.last_activation_time = time.time()

            # Log activation with mode-specific message
            if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
                freq = self._get_current_frequency()
                print(f"🔊 ULTRASOUND ACTIVATED: {freq:,} Hz (silent - dogs can hear it!)")
            elif self.config.mode == UltrasoundMode.AUDIBLE_TONE:
                print(f"🔊 AUDIBLE ALARM ACTIVATED: {self.config.audible_frequency} Hz")
            else:
                print(f"⚠️ VISUAL ALARM ACTIVATED")

            # Start simulation based on mode
            if self.config.mode in [UltrasoundMode.AUDIBLE_TONE, UltrasoundMode.ULTRASOUND_SIM]:
                if self.sound_enabled and (AUDIO_AVAILABLE or PYAUDIO_AVAILABLE):
                    self._start_audio_simulation()
                else:
                    self._start_visual_simulation()
            else:
                self._start_visual_simulation()

    def deactivate(self):
        """Deactivate the ultrasonic repeller"""
        if self.is_active:
            self.is_active = False
            self.stop_signal.set()

            # Calculate active time
            if self.last_activation_time:
                active_duration = time.time() - self.last_activation_time
                self.total_active_time += active_duration
                print(f"🔇 Ultrasound deactivated (Duration: {active_duration:.1f}s)")

            # Clean up audio resources
            self._cleanup_audio()

    def _start_audio_simulation(self):
        """Start audio-based simulation"""
        if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
            # For ultrasound simulation, we'll simulate inaudible sound
            print(f"   🐕 Simulating {self._get_current_frequency():,} Hz ultrasound")
            print(f"   ⚠️ Note: This frequency is inaudible to humans but bothersome to dogs")

        # Start audio thread
        self.play_thread = threading.Thread(
            target=self._audio_simulation_loop,
            daemon=True
        )
        self.play_thread.start()

    def _start_visual_simulation(self):
        """Start visual-only simulation"""
        self.play_thread = threading.Thread(
            target=self._visual_simulation_loop,
            daemon=True
        )
        self.play_thread.start()

    def _audio_simulation_loop(self):
        """Main audio simulation loop"""
        sample_rate = 44100

        try:
            # For ultrasound simulation, we'll use a very low volume
            # or no sound at all (just logging)
            if self.config.mode == UltrasoundMode.ULTRASOUND_SIM and not PYAUDIO_AVAILABLE:
                # Just log the ultrasound pulses
                pulse_count = 0
                while self.is_active and not self.stop_signal.is_set() and pulse_count < self.config.burst_count:
                    print(f"   📡 Ultrasound pulse #{pulse_count + 1}/{self.config.burst_count}")
                    time.sleep(self.config.pulse_duration)
                    pulse_count += 1

                    if pulse_count < self.config.burst_count:
                        time.sleep(self.config.pulse_interval)

                # Continuous mode after burst
                while self.is_active and not self.stop_signal.is_set():
                    print("   📡 Ultrasound continuous output...")
                    time.sleep(1.0)
                return

            # For audible mode or if PyAudio is available
            frequency = self._get_current_frequency()
            volume = self.config.volume

            # Reduce volume for ultrasound simulation
            if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
                volume = 0.01  # Very quiet or silent

            pulse_count = 0
            burst_complete = False

            while self.is_active and not self.stop_signal.is_set():
                # Generate sound for this pulse
                if self.config.mode == UltrasoundMode.AUDIBLE_TONE or PYAUDIO_AVAILABLE:
                    # Generate sine wave
                    t = np.linspace(
                        0,
                        self.config.pulse_duration,
                        int(sample_rate * self.config.pulse_duration),
                        False
                    )

                    if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
                        # For ultrasound, we might generate a tone at the limit of human hearing
                        # or just below audible range for simulation
                        wave = 0.1 * np.sin(2 * np.pi * 18000 * t)  # 18kHz - borderline audible
                    else:
                        # For audible mode
                        wave = volume * np.sin(2 * np.pi * frequency * t)

                    # Convert to 16-bit PCM
                    audio = (wave * 32767).astype(np.int16)

                    # Play using available library
                    if AUDIO_AVAILABLE:
                        play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

                        # Wait for pulse to complete
                        pulse_start = time.time()
                        while time.time() - pulse_start < self.config.pulse_duration:
                            if self.stop_signal.is_set():
                                play_obj.stop()
                                return
                            time.sleep(0.01)

                    elif PYAUDIO_AVAILABLE:
                        # Use PyAudio for more control
                        if self.pyaudio_instance is None:
                            self.pyaudio_instance = pyaudio.PyAudio()
                            self.pyaudio_stream = self.pyaudio_instance.open(
                                format=pyaudio.paInt16,
                                channels=1,
                                rate=sample_rate,
                                output=True
                            )

                        self.pyaudio_stream.write(audio.tobytes())

                pulse_count += 1

                # If in burst mode and burst is complete, switch to continuous
                if pulse_count >= self.config.burst_count and not burst_complete:
                    burst_complete = True
                    if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
                        print(f"   ✅ Burst complete. Continuous ultrasound output...")

                # Wait between pulses (if not continuous)
                if not burst_complete or pulse_count % 5 == 0:
                    time.sleep(self.config.pulse_interval)

                # Show progress for ultrasound mode
                if self.config.mode == UltrasoundMode.ULTRASOUND_SIM and pulse_count % 10 == 0:
                    print(f"   📡 Ultrasound active for {pulse_count} pulses")

        except Exception as e:
            print(f"⚠️ Audio simulation error: {e}")
            # Fall back to visual simulation
            self._visual_simulation_loop()

    def _visual_simulation_loop(self):
        """Visual simulation (flashing indicator in console)"""
        pulse_count = 0

        while self.is_active and not self.stop_signal.is_set():
            if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
                indicator = "🔊" if pulse_count % 2 == 0 else "📡"
                print(f"   {indicator} Ultrasound emitting...", end='\r')
            elif self.config.mode == UltrasoundMode.AUDIBLE_TONE:
                indicator = "🔊" if pulse_count % 2 == 0 else "🔇"
                print(f"   {indicator} Audible alarm...", end='\r')
            else:
                indicator = "⚠️" if pulse_count % 2 == 0 else "🚨"
                print(f"   {indicator} Visual alarm active...", end='\r')

            pulse_count += 1
            time.sleep(0.5)

        print()  # New line after clearing the line

    def _cleanup_audio(self):
        """Clean up audio resources"""
        if self.pyaudio_stream:
            self.pyaudio_stream.stop_stream()
            self.pyaudio_stream.close()
            self.pyaudio_stream = None

        if self.pyaudio_instance:
            self.pyaudio_instance.terminate()
            self.pyaudio_instance = None

    def toggle_sound(self) -> bool:
        """Toggle simulation sound on/off"""
        self.sound_enabled = not self.sound_enabled

        if not self.sound_enabled and self.is_active:
            self._cleanup_audio()
            print(f"🔇 Sound disabled")
        elif self.sound_enabled:
            print(f"🔊 Sound enabled")

        return self.sound_enabled

    def set_mode(self, mode: UltrasoundMode):
        """Change ultrasound mode"""
        if self.is_active:
            print("⚠️ Cannot change mode while active. Deactivating first.")
            self.deactivate()

        old_mode = self.config.mode
        self.config.mode = mode

        print(f"🔄 Ultrasound mode changed: {old_mode.value} → {mode.value}")
        print(f"   Frequency: {self._get_current_frequency()} Hz")

    def get_status(self) -> dict:
        """Get current status of ultrasound simulator"""
        return {
            'active': self.is_active,
            'mode': self.config.mode.value,
            'frequency': self._get_current_frequency(),
            'sound_enabled': self.sound_enabled,
            'activations': self.activation_count,
            'total_time': self.total_active_time,
            'human_audible': self._get_current_frequency() <= self.HUMAN_HEARING_RANGE[1],
            'dog_audible': (self.DOG_HEARING_RANGE[0] <= self._get_current_frequency() <= self.DOG_HEARING_RANGE[1])
        }

    def get_visual_indicator(self) -> str:
        """Get visual indicator for display"""
        if not self.is_active:
            return "⚪"  # Inactive

        if self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
            return "📡"  # Ultrasound icon
        elif self.config.mode == UltrasoundMode.AUDIBLE_TONE:
            return "🔊"  # Speaker icon
        else:
            return "🚨"  # Warning icon

    def get_frequency_display(self) -> str:
        """Get frequency display string"""
        freq = self._get_current_frequency()
        if freq >= 1000:
            return f"{freq / 1000:.1f}kHz"
        return f"{freq}Hz"

    def cleanup(self):
        """Cleanup all resources"""
        self.deactivate()

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)

        self._cleanup_audio()

        # Print summary
        print(f"\n📊 Ultrasound Simulator Summary:")
        print(f"   Total activations: {self.activation_count}")
        print(f"   Total active time: {self.total_active_time:.1f}s")
        print(f"   Mode: {self.config.mode.value}")


# Example usage and testing
if __name__ == "__main__":
    print("Testing Ultrasound Simulator...")

    # Test different modes
    modes = [
        UltrasoundMode.AUDIBLE_TONE,
        UltrasoundMode.ULTRASOUND_SIM,
        UltrasoundMode.VISUAL_ONLY
    ]

    for mode in modes:
        print(f"\n{'=' * 50}")
        print(f"Testing mode: {mode.value}")
        print('=' * 50)

        config = UltrasoundConfig(mode=mode)
        repeller = UltrasonicSimulator(config)

        # Test activation
        repeller.activate()
        time.sleep(2)  # Run for 2 seconds
        repeller.deactivate()

        time.sleep(1)  # Pause between tests

        repeller.cleanup()

    print("\n✅ All tests completed!")