#!/usr/bin/env python3
"""
Ultrasound Simulator Test Script
Test different ultrasound modes and frequencies
"""

import time
from src.ultrasonic_simulator import UltrasonicSimulator, UltrasoundMode, UltrasoundConfig


def test_all_modes():
    """Test all ultrasound simulation modes"""
    print("🚀 Testing All Ultrasound Modes")
    print("=" * 60)

    modes = [
        (UltrasoundMode.AUDIBLE_TONE, "Audible Tone Mode"),
        (UltrasoundMode.ULTRASOUND_SIM, "Ultrasound Simulation Mode"),
        (UltrasoundMode.VISUAL_ONLY, "Visual Only Mode"),
    ]

    for mode, description in modes:
        print(f"\n🔧 Testing: {description}")
        print("-" * 40)

        config = UltrasoundConfig(
            mode=mode,
            audible_frequency=1500,
            ultrasound_frequency=25000,
            volume=0.3
        )

        repeller = UltrasonicSimulator(config)

        # Test single activation
        print("   Activating for 3 seconds...")
        repeller.activate()
        time.sleep(3)
        repeller.deactivate()

        # Test rapid activation/deactivation
        print("   Testing rapid pulses...")
        for i in range(5):
            repeller.activate()
            time.sleep(0.5)
            repeller.deactivate()
            time.sleep(0.3)

        repeller.cleanup()
        time.sleep(1)

    print("\n✅ All mode tests completed!")


def test_frequency_sweep():
    """Test different frequencies"""
    print("\n🎵 Testing Frequency Range")
    print("=" * 60)

    frequencies = [
        (1000, "1kHz - Low audible"),
        (5000, "5kHz - Mid-range"),
        (10000, "10kHz - High-pitched"),
        (15000, "15kHz - Teen limit"),
        (20000, "20kHz - Human limit"),
        (25000, "25kHz - Dog repeller"),
        (30000, "30kHz - Ultrasound"),
        (40000, "40kHz - High ultrasound"),
    ]

    for freq, description in frequencies:
        print(f"\n📡 Testing: {description}")
        print(f"   Frequency: {freq:,} Hz")

        # Determine if in human hearing range
        if freq <= 20000:
            print("   🔊 Humans can hear this")
        else:
            print("   📡 Ultrasound (inaudible to humans)")

        # Determine if in dog hearing range
        if 67 <= freq <= 45000:
            print("   🐕 Dogs can hear this")
        else:
            print("   🐕 Dogs cannot hear this")

        config = UltrasoundConfig(
            mode=UltrasoundMode.AUDIBLE_TONE if freq <= 20000 else UltrasoundMode.ULTRASOUND_SIM,
            audible_frequency=freq if freq <= 20000 else 15000,
            ultrasound_frequency=freq,
            volume=0.2
        )

        repeller = UltrasonicSimulator(config)
        repeller.activate()
        time.sleep(1.5)
        repeller.deactivate()
        repeller.cleanup()

        time.sleep(0.5)

    print("\n✅ Frequency sweep completed!")


def test_hardware_preview():
    """Show hardware integration preview"""
    print("\n🔌 Hardware Integration Preview")
    print("=" * 60)

    print("\nFor real hardware integration, you would need:")
    print("  1. 🎛️  Ultrasonic transducer (25-30kHz)")
    print("  2. ⚡ Power amplifier circuit")
    print("  3. 🖥️  Microcontroller (Raspberry Pi/Arduino)")
    print("  4. 🔌 GPIO connection cables")
    print("  5. 🔋 Power supply")

    print("\n📋 Hardware Code Snippet:")
    print("""
    # Example hardware control code
    import RPi.GPIO as GPIO

    class UltrasonicHardware:
        def __init__(self, pin=18, frequency=25000):
            self.pin = pin
            self.frequency = frequency
            GPIO.setmode(GPIO.BCM)
            GPIO.setup(pin, GPIO.OUT)

        def activate(self):
            # Generate PWM signal at ultrasound frequency
            self.pwm = GPIO.PWM(self.pin, self.frequency)
            self.pwm.start(50)  # 50% duty cycle

        def deactivate(self):
            if hasattr(self, 'pwm'):
                self.pwm.stop()

        def cleanup(self):
            GPIO.cleanup()
    """)

    print("\n⚠️  Safety Notes:")
    print("  • Start with low power")
    print("  • Test with supervision")
    print("  • Follow local regulations")
    print("  • Never aim at animals for extended periods")


def main():
    """Main test function"""
    print("\n" + "=" * 60)
    print("        ULTRASOUND SIMULATOR TEST SUITE")
    print("=" * 60)

    try:
        # Test all modes
        test_all_modes()

        # Test frequencies
        test_frequency_sweep()

        # Show hardware preview
        test_hardware_preview()

        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        print("\nNext steps:")
        print("  1. Run main system: python src/main.py")
        print("  2. Try ultrasound mode: python src/main.py --ultrasound-mode ultrasound")
        print("  3. Test hardware integration when ready")

    except KeyboardInterrupt:
        print("\n\n⚠️ Test interrupted by user")
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")


if __name__ == "__main__":
    main()