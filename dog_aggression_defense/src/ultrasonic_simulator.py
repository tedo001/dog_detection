"""
ULTRASONIC SIMULATOR - Complete Working Version
Simulates different ultrasound modes for dog repelling
"""
import threading
import time
import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Optional, Dict, Any
import sys

print("=" * 60)
print("🔊 ULTRASONIC SIMULATOR - LOADING...")
print("=" * 60)

# Audio capability detection
AUDIO_AVAILABLE = False
PYAUDIO_AVAILABLE = False
WINSOUND_AVAILABLE = False

try:
    import simpleaudio as sa

    AUDIO_AVAILABLE = True
    print("✅ simpleaudio available")
except ImportError:
    AUDIO_AVAILABLE = False
    print("⚠️ simpleaudio not available (install: pip install simpleaudio)")

try:
    import pyaudio

    PYAUDIO_AVAILABLE = True
    print("✅ pyaudio available")
except ImportError:
    PYAUDIO_AVAILABLE = False

if sys.platform == "win32":
    try:
        import winsound

        WINSOUND_AVAILABLE = True
        print("✅ winsound available (Windows)")
    except ImportError:
        WINSOUND_AVAILABLE = False
else:
    print("ℹ️ Platform: Non-Windows (no winsound)")

ANY_AUDIO_AVAILABLE = AUDIO_AVAILABLE or PYAUDIO_AVAILABLE or WINSOUND_AVAILABLE


class UltrasoundMode(Enum):
    """Different ultrasound simulation modes"""
    AUDIBLE_TONE = "audible"  # Audible beeping (for testing)
    ULTRASOUND_SIM = "ultrasound"  # Simulated ultrasound (silent, logs only)
    VISUAL_ONLY = "visual"  # Visual indicators only
    SYSTEM_BEEP = "beep"  # System beep (Windows only)


@dataclass
class UltrasoundConfig:
    """Configuration for ultrasound simulation"""
    mode: 'UltrasoundMode' = None
    frequency: int = 25000  # Hz (ultrasound frequency)
    audible_frequency: int = 1500  # Hz (audible frequency)
    volume: float = 0.5  # 0.0 to 1.0
    pulse_duration: float = 0.1  # seconds per pulse
    pulse_interval: float = 0.2  # seconds between pulses
    burst_count: int = 10  # pulses per activation
    continuous: bool = False  # Continuous or burst mode

    def __post_init__(self):
        if self.mode is None:
            self.mode = UltrasoundMode.VISUAL_ONLY


class UltrasonicSimulator:
    """
    Complete ultrasonic simulator with multiple modes
    Works on all platforms
    """

    def __init__(self, config: Optional[UltrasoundConfig] = None):
        """
        Initialize ultrasound simulator

        Args:
            config: Ultrasound configuration
        """
        self.config = config or UltrasoundConfig()

        # State variables
        self.is_active = False
        self.sound_enabled = ANY_AUDIO_AVAILABLE
        self.play_thread = None
        self.stop_event = threading.Event()

        # Audio resources
        self.pyaudio_stream = None
        self.pyaudio_instance = None

        # Statistics
        self.activation_count = 0
        self.total_active_time = 0.0
        self.last_activation_time = None
        self.pulse_count = 0

        # Frequency information
        self.DOG_HEARING_RANGE = (67, 45000)  # Hz
        self.HUMAN_HEARING_RANGE = (20, 20000)  # Hz
        self.TYPICAL_REPELLER_FREQ = (23000, 25000)  # Hz

        # Print initialization info
        self._print_init_info()

    def _print_init_info(self):
        """Print initialization information"""
        print(f"\n🎛️ ULTRASONIC SIMULATOR INITIALIZED")
        print(f"   Mode: {self.config.mode.value.upper()}")

        freq = self._get_current_frequency()
        if freq > 0:
            print(f"   Frequency: {freq:,} Hz")

        # Mode-specific info
        if self.config.mode == UltrasoundMode.AUDIBLE_TONE:
            if ANY_AUDIO_AVAILABLE:
                print(f"   🔊 You will hear {self.config.audible_frequency} Hz beeping")
            else:
                print(f"   🔊 Audible mode (no audio libraries - using visual)")

        elif self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
            print(f"   📡 Simulating ultrasound at {self.config.frequency:,} Hz")
            if freq > self.HUMAN_HEARING_RANGE[1]:
                print(f"   ⚠️ This frequency is inaudible to humans")
            if self.DOG_HEARING_RANGE[0] <= freq <= self.DOG_HEARING_RANGE[1]:
                print(f"   🐕 Dogs CAN hear this frequency")

        elif self.config.mode == UltrasoundMode.VISUAL_ONLY:
            print(f"   👁️ Visual mode only - No audio")

        elif self.config.mode == UltrasoundMode.SYSTEM_BEEP:
            if WINSOUND_AVAILABLE:
                print(f"   🔔 Using system beep (Windows)")
            else:
                print(f"   ⚠️ System beep not available, using visual mode")
                self.config.mode = UltrasoundMode.VISUAL_ONLY

        print(f"   Audio available: {'YES' if ANY_AUDIO_AVAILABLE else 'NO'}")

    def _get_current_frequency(self) -> int:
        """Get the current frequency based on mode"""
        if self.config.mode == UltrasoundMode.AUDIBLE_TONE:
            return self.config.audible_frequency
        elif self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
            return self.config.frequency
        elif self.config.mode == UltrasoundMode.SYSTEM_BEEP:
            return 1000  # Default beep frequency
        return 0

    def activate(self, duration: Optional[float] = None):
        """
        Activate the ultrasonic repeller

        Args:
            duration: Optional duration in seconds (None = continuous)
        """
        if self.is_active:
            return  # Already active

        self.is_active = True
        self.stop_event.clear()
        self.activation_count += 1
        self.last_activation_time = time.time()
        self.pulse_count = 0

        # Log activation
        mode_name = self.config.mode.value.upper()
        freq = self._get_current_frequency()

        print(f"\n🚀 ULTRASOUND ACTIVATED")
        print(f"   Mode: {mode_name}")
        if freq > 0:
            print(f"   Frequency: {freq:,} Hz")

        if duration:
            print(f"   Duration: {duration:.1f} seconds")
            # Auto-deactivate after duration
            threading.Timer(duration, self.deactivate).start()

        # Start simulation thread
        self.play_thread = threading.Thread(
            target=self._simulation_loop,
            daemon=True
        )
        self.play_thread.start()

    def deactivate(self):
        """Deactivate the ultrasonic repeller"""
        if not self.is_active:
            return

        self.is_active = False
        self.stop_event.set()

        # Calculate active time
        if self.last_activation_time:
            active_duration = time.time() - self.last_activation_time
            self.total_active_time += active_duration

            print(f"\n🛑 ULTRASOUND DEACTIVATED")
            print(f"   Duration: {active_duration:.1f}s")
            print(f"   Pulses: {self.pulse_count}")

        # Clean up audio
        self._cleanup_audio()

    def _simulation_loop(self):
        """Main simulation loop"""
        try:
            # Run appropriate simulation based on mode
            if self.config.mode == UltrasoundMode.AUDIBLE_TONE:
                self._audible_loop()
            elif self.config.mode == UltrasoundMode.ULTRASOUND_SIM:
                self._ultrasound_loop()
            elif self.config.mode == UltrasoundMode.SYSTEM_BEEP:
                self._beep_loop()
            else:  # VISUAL_ONLY or fallback
                self._visual_loop()
        except Exception as e:
            print(f"⚠️ Simulation error: {e}")
            # Fall back to visual mode
            self._visual_loop()

    def _audible_loop(self):
        """Audible tone simulation"""
        freq = self.config.audible_frequency

        print(f"   🔊 Playing audible tone at {freq} Hz...")

        pulse_num = 0
        while self.is_active and not self.stop_event.is_set():
            # Check if we should stop
            if self.stop_event.is_set():
                break

            # Generate and play tone
            if self.sound_enabled and ANY_AUDIO_AVAILABLE:
                self._play_audio_tone(freq, self.config.pulse_duration)

            pulse_num += 1
            self.pulse_count += 1

            # Show progress
            if pulse_num % 5 == 0:
                print(f"   🔊 Pulse {pulse_num}", end='\r')

            # Check burst mode
            if not self.config.continuous and pulse_num >= self.config.burst_count:
                print(f"\n   ✅ Audible burst complete ({self.config.burst_count} pulses)")
                time.sleep(1.0)  # Pause between bursts
                pulse_num = 0

            # Wait between pulses
            if not self.stop_event.is_set():
                time.sleep(self.config.pulse_interval)

        print()  # Clear line

    def _ultrasound_loop(self):
        """Ultrasound simulation (silent mode)"""
        freq = self.config.frequency

        print(f"   📡 Simulating ultrasound at {freq:,} Hz...")
        print(f"   ⚠️ Silent to humans, annoying to dogs!")

        pulse_num = 0
        start_time = time.time()

        while self.is_active and not self.stop_event.is_set():
            elapsed = time.time() - start_time

            # Visual indicators
            indicators = ["📡", "⚡", "🔊", "🐕", "🚫"]
            indicator = indicators[pulse_num % len(indicators)]

            # Show progress
            if self.config.continuous:
                status = f"   {indicator} Ultrasound active: {elapsed:.1f}s"
            else:
                progress = min(1.0, pulse_num / self.config.burst_count)
                filled = int(progress * 20)
                bar = "█" * filled + "░" * (20 - filled)
                status = f"   {indicator} [{bar}] {pulse_num}/{self.config.burst_count}"

            print(status, end='\r')

            pulse_num += 1
            self.pulse_count += 1

            # Check burst completion
            if not self.config.continuous and pulse_num >= self.config.burst_count:
                print(f"\n   ✅ Ultrasound burst complete")
                time.sleep(2.0)  # Longer pause
                pulse_num = 0
                start_time = time.time()

            time.sleep(self.config.pulse_duration)

        print()

    def _beep_loop(self):
        """System beep simulation (Windows only)"""
        if not WINSOUND_AVAILABLE:
            print("   ⚠️ System beep not available, using visual mode")
            self._visual_loop()
            return

        print(f"   🔔 Playing system beep...")

        pulse_num = 0
        while self.is_active and not self.stop_event.is_set():
            # Play system beep
            duration_ms = int(self.config.pulse_duration * 1000)
            winsound.Beep(1000, duration_ms)

            pulse_num += 1
            self.pulse_count += 1

            if pulse_num % 5 == 0:
                print(f"   🔔 Beep {pulse_num}", end='\r')

            # Check burst mode
            if not self.config.continuous and pulse_num >= self.config.burst_count:
                print(f"\n   ✅ Beep burst complete")
                time.sleep(1.0)
                pulse_num = 0

            time.sleep(self.config.pulse_interval)

        print()

    def _visual_loop(self):
        """Visual-only simulation"""
        indicators = ["🚨", "⚠️", "🔴", "⚡", "💥"]

        print(f"   👁️ Visual alarm active...")

        pulse_num = 0
        while self.is_active and not self.stop_event.is_set():
            indicator = indicators[pulse_num % len(indicators)]

            # Show different patterns based on mode
            if self.config.continuous:
                status = f"   {indicator} VISUAL ALARM [{pulse_num + 1}]"
            else:
                progress = (pulse_num % self.config.burst_count) / self.config.burst_count
                intensity = int(progress * 10)
                status = f"   {indicator}{'!' * intensity}"

            print(status, end='\r')

            pulse_num += 1
            self.pulse_count += 1
            time.sleep(0.3)

        print()

    def _play_audio_tone(self, frequency: int, duration: float):
        """Play audio tone using best available library"""
        sample_rate = 44100

        # Generate sine wave
        t = np.linspace(0, duration, int(sample_rate * duration), False)
        wave = self.config.volume * np.sin(2 * np.pi * frequency * t)
        audio = (wave * 32767).astype(np.int16)

        try:
            if AUDIO_AVAILABLE:
                # Use simpleaudio
                play_obj = sa.play_buffer(audio, 1, 2, sample_rate)

                # Wait for completion or stop signal
                start_time = time.time()
                while time.time() - start_time < duration:
                    if self.stop_event.is_set():
                        play_obj.stop()
                        return
                    time.sleep(0.01)

            elif PYAUDIO_AVAILABLE:
                # Use PyAudio
                if self.pyaudio_instance is None:
                    self.pyaudio_instance = pyaudio.PyAudio()
                    self.pyaudio_stream = self.pyaudio_instance.open(
                        format=pyaudio.paInt16,
                        channels=1,
                        rate=sample_rate,
                        output=True
                    )

                self.pyaudio_stream.write(audio.tobytes())

            elif WINSOUND_AVAILABLE and sys.platform == "win32":
                # Use winsound
                duration_ms = int(duration * 1000)
                winsound.Beep(frequency, duration_ms)

        except Exception as e:
            print(f"⚠️ Audio play error: {e}")

    def _cleanup_audio(self):
        """Clean up audio resources"""
        if self.pyaudio_stream:
            try:
                self.pyaudio_stream.stop_stream()
                self.pyaudio_stream.close()
                self.pyaudio_stream = None
            except:
                pass

        if self.pyaudio_instance:
            try:
                self.pyaudio_instance.terminate()
                self.pyaudio_instance = None
            except:
                pass

    def toggle(self):
        """Toggle ultrasound on/off"""
        if self.is_active:
            self.deactivate()
        else:
            self.activate()

    def pulse(self, count: int = 3, interval: float = 0.5):
        """
        Send a specific number of pulses

        Args:
            count: Number of pulses to send
            interval: Time between pulses
        """
        print(f"\n⚡ Sending {count} ultrasound pulses...")

        original_interval = self.config.pulse_interval
        original_burst = self.config.burst_count
        original_continuous = self.config.continuous

        try:
            self.config.pulse_interval = interval
            self.config.burst_count = count
            self.config.continuous = False

            self.activate()

            # Wait for pulses to complete
            time.sleep(count * (self.config.pulse_duration + interval) + 0.5)

            self.deactivate()

        finally:
            # Restore original settings
            self.config.pulse_interval = original_interval
            self.config.burst_count = original_burst
            self.config.continuous = original_continuous

    def set_mode(self, mode: UltrasoundMode):
        """
        Change ultrasound mode

        Args:
            mode: New ultrasound mode
        """
        if self.is_active:
            print("⚠️ Cannot change mode while active")
            return

        old_mode = self.config.mode
        self.config.mode = mode

        print(f"\n🔄 Mode changed: {old_mode.value.upper()} → {mode.value.upper()}")
        self._print_init_info()

    def set_frequency(self, frequency: int):
        """
        Set ultrasound frequency

        Args:
            frequency: Frequency in Hz
        """
        if self.is_active:
            print("⚠️ Cannot change frequency while active")
            return

        if self.config.mode == UltrasoundMode.AUDIBLE_TONE:
            self.config.audible_frequency = frequency
        else:
            self.config.frequency = frequency

        print(f"\n🎵 Frequency set to {frequency:,} Hz")

    def get_status(self) -> Dict[str, Any]:
        """Get current status"""
        freq = self._get_current_frequency()

        return {
            'active': self.is_active,
            'mode': self.config.mode.value,
            'frequency': freq,
            'sound_enabled': self.sound_enabled,
            'activations': self.activation_count,
            'total_time': self.total_active_time,
            'pulse_count': self.pulse_count,
            'is_ultrasound': freq > self.HUMAN_HEARING_RANGE[1],
            'dog_audible': self.DOG_HEARING_RANGE[0] <= freq <= self.DOG_HEARING_RANGE[1],
            'audio_available': ANY_AUDIO_AVAILABLE
        }

    def get_visual_indicator(self) -> str:
        """Get visual indicator for display"""
        if not self.is_active:
            return "⚪"  # Inactive

        indicators = {
            UltrasoundMode.AUDIBLE_TONE: "🔊",
            UltrasoundMode.ULTRASOUND_SIM: "📡",
            UltrasoundMode.VISUAL_ONLY: "🚨",
            UltrasoundMode.SYSTEM_BEEP: "🔔"
        }

        return indicators.get(self.config.mode, "⚡")

    def get_frequency_display(self) -> str:
        """Get frequency display string"""
        freq = self._get_current_frequency()
        if freq >= 1000:
            return f"{freq / 1000:.1f}kHz"
        return f"{freq}Hz"

    def test_audio(self):
        """Test audio output"""
        print("\n🔊 Testing audio output...")
        print("=" * 30)

        test_freqs = [500, 1000, 2000, 4000]

        for freq in test_freqs:
            print(f"Testing {freq} Hz tone...")
            self.set_frequency(freq)
            self.pulse(2, 0.5)
            time.sleep(0.5)

        print("✅ Audio test complete")

    def cleanup(self):
        """Cleanup all resources"""
        self.deactivate()

        if self.play_thread and self.play_thread.is_alive():
            self.play_thread.join(timeout=1.0)

        self._cleanup_audio()

        # Print summary
        print("\n" + "=" * 50)
        print("📊 ULTRASONIC SIMULATOR SUMMARY")
        print("=" * 50)
        print(f"Total activations: {self.activation_count}")
        print(f"Total active time: {self.total_active_time:.1f}s")
        print(f"Total pulses: {self.pulse_count}")
        print(f"Final mode: {self.config.mode.value}")
        print(f"Audio available: {'Yes' if ANY_AUDIO_AVAILABLE else 'No'}")
        print("=" * 50)


def demo():
    """Demo all ultrasound modes"""
    print("\n" + "=" * 60)
    print("🎬 ULTRASONIC SIMULATOR DEMO")
    print("=" * 60)

    # Test different modes
    modes = [
        (UltrasoundMode.VISUAL_ONLY, "Visual Only", 3.0),
        (UltrasoundMode.AUDIBLE_TONE, "Audible Tone", 4.0),
        (UltrasoundMode.ULTRASOUND_SIM, "Ultrasound Simulation", 5.0),
    ]

    # Add system beep on Windows
    if sys.platform == "win32":
        modes.append((UltrasoundMode.SYSTEM_BEEP, "System Beep", 3.0))

    for mode, description, duration in modes:
        print(f"\n▶️ Testing: {description}")
        print("-" * 40)

        config = UltrasoundConfig(mode=mode)
        repeller = UltrasonicSimulator(config)

        # Activate for duration
        repeller.activate(duration=duration)
        time.sleep(duration + 0.5)

        # Test quick pulses
        if mode in [UltrasoundMode.AUDIBLE_TONE, UltrasoundMode.ULTRASOUND_SIM]:
            repeller.pulse(3, 0.3)
            time.sleep(2)

        repeller.cleanup()
        time.sleep(1)  # Pause between modes

    print("\n✅ Demo completed successfully!")


def interactive_test():
    """Interactive test mode"""
    print("\n🎮 INTERACTIVE TEST MODE")
    print("=" * 40)

    # Create simulator with default config
    repeller = UltrasonicSimulator()

    print("\n📋 Available commands:")
    print("  on     - Activate ultrasound")
    print("  off    - Deactivate ultrasound")
    print("  toggle - Toggle on/off")
    print("  pulse  - Send 3 quick pulses")
    print("  mode   - Change mode")
    print("  freq   - Change frequency")
    print("  status - Show status")
    print("  test   - Test audio")
    print("  exit   - Quit")

    try:
        while True:
            cmd = input("\nEnter command: ").strip().lower()

            if cmd == 'exit' or cmd == 'quit':
                break

            elif cmd == 'on':
                duration = input("Duration (seconds, enter for continuous): ").strip()
                if duration:
                    repeller.activate(float(duration))
                else:
                    repeller.activate()

            elif cmd == 'off':
                repeller.deactivate()

            elif cmd == 'toggle':
                repeller.toggle()

            elif cmd == 'pulse':
                count = input("Number of pulses [3]: ").strip()
                interval = input("Interval between pulses [0.5]: ").strip()
                count = int(count) if count else 3
                interval = float(interval) if interval else 0.5
                repeller.pulse(count, interval)

            elif cmd == 'mode':
                print("\nAvailable modes:")
                for i, mode in enumerate(UltrasoundMode, 1):
                    print(f"  {i}. {mode.value}")

                choice = input("Select mode: ").strip()
                try:
                    mode_idx = int(choice) - 1
                    if 0 <= mode_idx < len(UltrasoundMode):
                        repeller.set_mode(list(UltrasoundMode)[mode_idx])
                except:
                    print("Invalid choice")

            elif cmd == 'freq':
                freq = input("Frequency in Hz: ").strip()
                try:
                    repeller.set_frequency(int(freq))
                except:
                    print("Invalid frequency")

            elif cmd == 'status':
                status = repeller.get_status()
                print("\n📊 STATUS:")
                for key, value in status.items():
                    print(f"  {key}: {value}")

            elif cmd == 'test':
                repeller.test_audio()

            else:
                print("Unknown command. Type 'help' to see commands.")

    except KeyboardInterrupt:
        print("\n\n⚠️ Interrupted")

    finally:
        repeller.cleanup()


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("🔊 ULTRASONIC SIMULATOR TEST SUITE")
    print("=" * 60)

    print("\n📊 AUDIO CAPABILITIES:")
    print(f"  simpleaudio: {'✅' if AUDIO_AVAILABLE else '❌'}")
    print(f"  pyaudio: {'✅' if PYAUDIO_AVAILABLE else '❌'}")
    print(f"  winsound: {'✅' if WINSOUND_AVAILABLE else '❌'}")
    print(f"  Any audio: {'✅' if ANY_AUDIO_AVAILABLE else '❌'}")

    if not ANY_AUDIO_AVAILABLE:
        print("\n⚠️ No audio libraries found!")
        print("💡 Install: pip install simpleaudio")
        print("   (or use visual mode)")

    # Ask user what to do
    print("\n1. Run full demo")
    print("2. Interactive test")
    print("3. Quick test (default)")

    choice = input("\nSelect option [3]: ").strip()

    if choice == "1":
        demo()
    elif choice == "2":
        interactive_test()
    else:
        # Quick test
        print("\n🔧 QUICK TEST")
        print("=" * 30)

        # Test visual mode
        print("\n1. Testing visual mode...")
        repeller = UltrasonicSimulator(UltrasoundConfig(mode=UltrasoundMode.VISUAL_ONLY))
        repeller.activate(duration=2.0)
        time.sleep(2.5)
        repeller.cleanup()

        # Test audible mode if available
        if ANY_AUDIO_AVAILABLE:
            print("\n2. Testing audible mode...")
            repeller = UltrasonicSimulator(UltrasoundConfig(mode=UltrasoundMode.AUDIBLE_TONE))
            repeller.activate(duration=2.0)
            time.sleep(2.5)
            repeller.cleanup()

        # Test ultrasound mode
        print("\n3. Testing ultrasound simulation...")
        repeller = UltrasonicSimulator(UltrasoundConfig(
            mode=UltrasoundMode.ULTRASOUND_SIM,
            frequency=25000
        ))
        repeller.activate(duration=3.0)
        time.sleep(3.5)
        repeller.cleanup()

        print("\n✅ Quick test completed!")

    print("\n" + "=" * 60)
    print("🎉 TEST COMPLETE!")
    print("=" * 60)
    print("\n💡 Usage in your code:")
    print("""
    from ultrasonic_simulator import UltrasonicSimulator, UltrasoundMode, UltrasoundConfig

    # Create simulator
    config = UltrasoundConfig(mode=UltrasoundMode.VISUAL_ONLY)
    ultrasound = UltrasonicSimulator(config)

    # Activate
    ultrasound.activate(duration=5.0)  # Activate for 5 seconds

    # Or in your detection loop
    if aggressive_dog_detected:
        ultrasound.activate()
    """)