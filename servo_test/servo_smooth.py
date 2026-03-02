from gpiozero import Servo
from gpiozero.pins.lgpio import LGPIOFactory
from time import sleep, time
from datetime import datetime

print(">>> SERVO_SMOOTH FILE LOADED <<<")

def log(msg):
    ts = datetime.now().strftime("%H:%M:%S.%f")[:-3]
    print(f"[{ts}] {msg}")


class SmoothServo180:
    """
    Smooth + logged control for 180-degree servo
    Raspberry Pi 5 compatible (gpiozero + lgpio)
    """

    def __init__(
        self,
        pin,
        min_angle=0,
        max_angle=180,
        min_pulse=0.5 / 1000,
        max_pulse=2.4 / 1000,
        max_speed_deg=20,      # deg/s, ????
        deadband=1.5,
        log_enable=True
    ):
        self.log_enable = log_enable

        self.factory = LGPIOFactory()
        self.servo = Servo(
            pin,
            min_pulse_width=min_pulse,
            max_pulse_width=max_pulse,
            pin_factory=self.factory
        )

        self.min_angle = min_angle
        self.max_angle = max_angle
        self.max_speed = max_speed_deg
        self.deadband = deadband

        self.current_angle = 90.0
        self._write_angle(self.current_angle)

        if self.log_enable:
            log("=== SERVO INIT ===")
            log(f"GPIO pin        : {pin}")
            log(f"Angle range     : {min_angle}� ? {max_angle}�")
            log(f"Pulse width     : {min_pulse*1000:.2f}?{max_pulse*1000:.2f} ms")
            log(f"Max speed       : {self.max_speed} deg/s")
            log(f"Deadband        : �{self.deadband}�")
            log(f"Start angle     : {self.current_angle}�")
            log("GPIO backend    : lgpio")
            log("===================")

    def _angle_to_value(self, angle):
        # Map 0?180� ? -1.0 ? +1.0
        return (angle / 90.0) - 1.0

    def _write_angle(self, angle):
        value = self._angle_to_value(angle)
        self.servo.value = value

        if self.log_enable:
            log(f"PWM write ? angle={angle:.2f}�, servo.value={value:.3f}")

    def move_to(self, target_angle):
        target_angle = max(self.min_angle, min(self.max_angle, target_angle))
        delta = target_angle - self.current_angle

        if abs(delta) < self.deadband:
            if self.log_enable:
                log(f"SKIP move ? target={target_angle:.2f}� (within deadband)")
            return

        dt = 0.02                     # 20 ms
        step = self.max_speed * dt    # deg per step

        if self.log_enable:
            log(f"START move ? {self.current_angle:.2f}� ? {target_angle:.2f}�")
            log(f"Step size  ? {step:.3f}� every {dt*1000:.0f} ms")

        while abs(target_angle - self.current_angle) > step:
            direction = 1 if target_angle > self.current_angle else -1
            self.current_angle += direction * step
            self._write_angle(self.current_angle)
            sleep(dt)

        self.current_angle = target_angle
        self._write_angle(self.current_angle)

        if self.log_enable:
            log(f"DONE move ? reached {self.current_angle:.2f}�")

    def release(self):
        if self.log_enable:
            log("RELEASE servo (PWM off)")
        self.servo.value = None


# =========================
# ? ??????????
# =========================
if __name__ == "__main__":
    log("=== SERVO DEBUG PROGRAM START ===")

    servo = SmoothServo180(
        pin=18,
        max_speed_deg=15,    # ? ????????
        deadband=1.5,
        log_enable=True
    )

    servo.move_to(0)
    sleep(1)

    servo.move_to(90)
    sleep(1)

    servo.move_to(180)
    sleep(1)

    servo.move_to(45)
    sleep(1)

    servo.release()

    log("=== SERVO DEBUG PROGRAM END ===")