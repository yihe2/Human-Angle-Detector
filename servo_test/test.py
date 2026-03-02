import sys
import time
from datetime import datetime

print("=== SERVO DEBUG START ===")

try:
    from gpiozero import Servo
    from gpiozero.pins.lgpio import LGPIOFactory
    print("[OK] gpiozero + lgpio imported")
except Exception as e:
    print("[ERROR] Import failed:", e)
    sys.exit(1)

factory = LGPIOFactory()
print("[INFO] Pin factory:", factory)

try:
    servo = Servo(
        18,
        min_pulse_width=0.5/1000,
        max_pulse_width=2.4/1000,
        pin_factory=factory
    )
    print("[OK] Servo object created")
except Exception as e:
    print("[ERROR] Servo init failed:", e)
    sys.exit(1)

def log_move(label, value):
    print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Moving to {label}")
    print("   ? servo.value =", value)

    try:
        servo.value = value
        print("   ? write OK")
    except Exception as e:
        print("   ? write FAILED:", e)

    time.sleep(2)

print("\n--- BEGIN TEST SEQUENCE ---")

log_move("MIN (-1)", -1)
log_move("MID (0)", 0)
log_move("MAX (1)", 1)

print("\nReleasing servo...")
servo.value = None

print("=== TEST FINISHED ===")