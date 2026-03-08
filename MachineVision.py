import os
import time
from collections import deque

import cv2
import mediapipe as mp
from picamera2 import Picamera2

# --- TRACKING CONFIG ---
SMOOTHING_WINDOW = max(1, int(os.environ.get("SMOOTHING_WINDOW", "6")))
HFOV = float(os.environ.get("HFOV", "88.0"))
RAW_TARGET_BLEND = min(max(float(os.environ.get("RAW_TARGET_BLEND", "0.35")), 0.0), 1.0)
STARTUP_MOTOR_HOLD_SECONDS = max(
    float(os.environ.get("STARTUP_MOTOR_HOLD_SECONDS", "1.0")),
    0.0,
)
TARGET_ACQUIRE_FRAMES = max(1, int(os.environ.get("TARGET_ACQUIRE_FRAMES", "5")))
HIP_VISIBILITY_MIN = min(
    max(float(os.environ.get("HIP_VISIBILITY_MIN", "0.65")), 0.0),
    1.0,
)

# --- MOTOR MODEL ---
MOTOR_MAX_SPEED_DPS = float(os.environ.get("MOTOR_MAX_SPEED_DPS", "35.0"))
MOTOR_MIN_ANGLE = -90.0
MOTOR_MAX_ANGLE = 90.0
MOTOR_CONTROL_MODE = os.environ.get("MOTOR_CONTROL_MODE", "servo_smooth").lower()
MOTOR_COMMAND_EPSILON_DEG = 0.2
TRACKING_DEADBAND_DEG = max(
    float(os.environ.get("TRACKING_DEADBAND_DEG", "0.3")),
    MOTOR_COMMAND_EPSILON_DEG,
)
CENTER_HOLD_TOLERANCE_DEG = max(
    float(os.environ.get("CENTER_HOLD_TOLERANCE_DEG", "0.9")),
    TRACKING_DEADBAND_DEG,
)
TRACKING_RESPONSE_GAIN = min(
    max(float(os.environ.get("TRACKING_RESPONSE_GAIN", "0.65")), 0.1),
    1.0,
)
TRACKING_CORRECTION_SIGN = 1.0
CONTROL_UPDATE_INTERVAL_SECONDS = max(
    float(os.environ.get("CONTROL_UPDATE_INTERVAL_SECONDS", "0.12")),
    0.02,
)
MOTION_SETTLE_SECONDS = max(
    float(os.environ.get("MOTION_SETTLE_SECONDS", "0.18")),
    0.0,
)
MAX_CORRECTION_STEP_DEG = max(
    float(os.environ.get("MAX_CORRECTION_STEP_DEG", "3.0")),
    TRACKING_DEADBAND_DEG,
)
OFFSET_PERSISTENCE_FRAMES = max(
    1, int(os.environ.get("OFFSET_PERSISTENCE_FRAMES", "2"))
)
ROTATING_STATUS_THRESHOLD_DEG = max(
    float(os.environ.get("ROTATING_STATUS_THRESHOLD_DEG", "1.2")),
    TRACKING_DEADBAND_DEG,
)
MOTOR_DIRECTION = "reverse"
MOTOR_DIRECTION_SIGN = -1.0

# --- SERVO CONFIG ---
SERVO_GPIO_PIN = int(os.environ.get("SERVO_GPIO_PIN", "18"))
SERVO_PWM_HZ = int(os.environ.get("SERVO_PWM_HZ", "50"))
SERVO_MIN_DUTY = float(os.environ.get("SERVO_MIN_DUTY", "2.5"))
SERVO_MAX_DUTY = float(os.environ.get("SERVO_MAX_DUTY", "12.5"))
SERVO_MIN_ANGLE = 0.0
SERVO_MAX_ANGLE = 180.0
SERVO_SMOOTH_LOG_ENABLE = os.environ.get("SERVO_SMOOTH_LOG_ENABLE", "0") == "1"

# --- GESTURE CONFIG (double down-stroke) ---
WRIST_VISIBILITY_MIN = 0.55
DOWNSTROKE_VELOCITY_THRESHOLD = 0.9
DOWNSTROKE_MIN_INTERVAL_SECONDS = 0.16
DOUBLE_STRIKE_WINDOW_SECONDS = 0.8
ARMED_TIMEOUT_SECONDS = 1.2

# --- SAFETY CONFIG ---
FIRE_COOLDOWN_SECONDS = 2.0
MAX_FIRES_PER_MINUTE = 20
AIM_LOCK_TOLERANCE_DEG = float(os.environ.get("AIM_LOCK_TOLERANCE_DEG", "1.5"))
AIM_STABLE_FRAMES_REQUIRED = 8
SAFE_ZONE_MARGIN_RATIO = 0.15
FIRE_FLASH_SECONDS = 0.25

STATE_TRACKING = "TRACKING"
STATE_ARMED = "ARMED"
STATE_FIRE = "FIRE"


def clamp(value, low, high):
    return max(low, min(high, value))


class MotorController:
    def __init__(self, mode):
        self.mode = mode
        self.direction_sign = MOTOR_DIRECTION_SIGN
        self.current_angle = 0.0
        self.last_output_angle = 9999.0
        self.servo_smooth = None
        self.gpio = None
        self.pwm = None

        if self.mode in ("servo_smooth", "servo"):
            try:
                from servo_test.servo_smooth import SmoothServo180

                self.servo_smooth = SmoothServo180(
                    pin=SERVO_GPIO_PIN,
                    min_angle=SERVO_MIN_ANGLE,
                    max_angle=SERVO_MAX_ANGLE,
                    max_speed_deg=MOTOR_MAX_SPEED_DPS,
                    deadband=max(MOTOR_COMMAND_EPSILON_DEG, 0.1),
                    log_enable=SERVO_SMOOTH_LOG_ENABLE,
                )
                physical_angle = self._servo_to_turret(self.servo_smooth.current_angle)
                self.current_angle = self._from_physical_turret(physical_angle)
                self.last_output_angle = self.current_angle
                self.mode = "servo_smooth"
                print(
                    f"MOTOR MODE: servo_smooth (pin={SERVO_GPIO_PIN}, "
                    f"range={SERVO_MIN_ANGLE:.0f}-{SERVO_MAX_ANGLE:.0f}, "
                    f"direction={'reverse' if self.direction_sign < 0 else 'normal'})"
                )
            except Exception as exc:
                self.mode = "sim"
                print(f"servo_smooth init failed, fallback to sim mode: {exc}")

        if self.mode == "servo_pwm":
            try:
                import RPi.GPIO as GPIO

                self.gpio = GPIO
                self.gpio.setmode(self.gpio.BCM)
                self.gpio.setup(SERVO_GPIO_PIN, self.gpio.OUT)
                self.pwm = self.gpio.PWM(SERVO_GPIO_PIN, SERVO_PWM_HZ)
                self.pwm.start(
                    self._angle_to_duty(self._to_physical_turret(self.current_angle))
                )
                print(
                    f"MOTOR MODE: servo_pwm (pin={SERVO_GPIO_PIN}, pwm={SERVO_PWM_HZ}Hz, "
                    f"direction={'reverse' if self.direction_sign < 0 else 'normal'})"
                )
            except Exception as exc:
                self.mode = "sim"
                print(f"Servo init failed, fallback to sim mode: {exc}")

        if self.mode == "sim":
            print(
                f"MOTOR MODE: sim (direction="
                f"{'reverse' if self.direction_sign < 0 else 'normal'})"
            )

    def _angle_to_duty(self, angle):
        normalized = (angle - MOTOR_MIN_ANGLE) / (MOTOR_MAX_ANGLE - MOTOR_MIN_ANGLE)
        normalized = clamp(normalized, 0.0, 1.0)
        return SERVO_MIN_DUTY + normalized * (SERVO_MAX_DUTY - SERVO_MIN_DUTY)

    def _turret_to_servo(self, turret_angle):
        ratio = (turret_angle - MOTOR_MIN_ANGLE) / (MOTOR_MAX_ANGLE - MOTOR_MIN_ANGLE)
        return SERVO_MIN_ANGLE + ratio * (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)

    def _servo_to_turret(self, servo_angle):
        ratio = (servo_angle - SERVO_MIN_ANGLE) / (SERVO_MAX_ANGLE - SERVO_MIN_ANGLE)
        return MOTOR_MIN_ANGLE + ratio * (MOTOR_MAX_ANGLE - MOTOR_MIN_ANGLE)

    def _to_physical_turret(self, logical_angle):
        return clamp(logical_angle * self.direction_sign, MOTOR_MIN_ANGLE, MOTOR_MAX_ANGLE)

    def _from_physical_turret(self, physical_angle):
        return clamp(physical_angle * self.direction_sign, MOTOR_MIN_ANGLE, MOTOR_MAX_ANGLE)

    def _emit_command(self, angle):
        if abs(angle - self.last_output_angle) < MOTOR_COMMAND_EPSILON_DEG:
            return

        physical_angle = self._to_physical_turret(angle)
        if self.mode == "servo_smooth" and self.servo_smooth is not None:
            servo_target = self._turret_to_servo(physical_angle)
            self.servo_smooth.set_angle(servo_target)
        elif self.mode == "servo_pwm" and self.pwm is not None:
            self.pwm.ChangeDutyCycle(self._angle_to_duty(physical_angle))

        print(
            f">>> MOTOR COMMAND [{self.mode}]: logical={angle:.1f} deg, "
            f"physical={physical_angle:.1f} deg"
        )
        self.last_output_angle = angle

    def move_towards(self, target_angle, dt):
        target_angle = clamp(target_angle, MOTOR_MIN_ANGLE, MOTOR_MAX_ANGLE)
        max_step = MOTOR_MAX_SPEED_DPS * dt

        if target_angle > self.current_angle:
            next_angle = min(self.current_angle + max_step, target_angle)
        else:
            next_angle = max(self.current_angle - max_step, target_angle)

        next_angle = clamp(next_angle, MOTOR_MIN_ANGLE, MOTOR_MAX_ANGLE)
        moved = abs(next_angle - self.current_angle) > 1e-6
        self.current_angle = next_angle
        self._emit_command(self.current_angle)
        return moved

    def cleanup(self):
        if self.servo_smooth is not None:
            self.servo_smooth.release()
        if self.pwm is not None:
            self.pwm.stop()
        if self.gpio is not None:
            self.gpio.cleanup()


def reset_wrist_history(gesture_state, wrist_name=None):
    wrist_names = (wrist_name,) if wrist_name is not None else gesture_state["wrists"]
    for name in wrist_names:
        gesture_state["wrists"][name]["prev_y"] = None
        gesture_state["wrists"][name]["prev_time"] = None


def detect_downstroke(landmarks, now, gesture_state, pose_module):
    left_wrist = landmarks[pose_module.PoseLandmark.LEFT_WRIST.value]
    right_wrist = landmarks[pose_module.PoseLandmark.RIGHT_WRIST.value]
    wrists = {"L": left_wrist, "R": right_wrist}
    active_wrist = "L" if left_wrist.visibility >= right_wrist.visibility else "R"
    active_velocity = 0.0
    downstroke_wrist = None
    downstroke_velocity = 0.0

    for wrist_name, wrist in wrists.items():
        if wrist.visibility < WRIST_VISIBILITY_MIN:
            reset_wrist_history(gesture_state, wrist_name)
            continue

        wrist_state = gesture_state["wrists"][wrist_name]
        prev_y = wrist_state["prev_y"]
        prev_t = wrist_state["prev_time"]
        velocity = 0.0

        if prev_y is not None and prev_t is not None:
            dt = max(now - prev_t, 1e-6)
            velocity = (wrist.y - prev_y) / dt
            if (
                velocity >= DOWNSTROKE_VELOCITY_THRESHOLD
                and now - gesture_state["last_downstroke_time"]
                >= DOWNSTROKE_MIN_INTERVAL_SECONDS
                and velocity > downstroke_velocity
            ):
                downstroke_wrist = wrist_name
                downstroke_velocity = velocity

        wrist_state["prev_y"] = wrist.y
        wrist_state["prev_time"] = now

        if wrist_name == active_wrist:
            active_velocity = velocity

    if downstroke_wrist is not None:
        gesture_state["last_downstroke_time"] = now
        return True, downstroke_velocity, downstroke_wrist

    if wrists[active_wrist].visibility < WRIST_VISIBILITY_MIN:
        return False, 0.0, active_wrist

    return False, active_velocity, active_wrist


def fire_gate_ok(now, last_fire_time, fire_history, stable_frames, in_safe_zone):
    if now - last_fire_time < FIRE_COOLDOWN_SECONDS:
        return False, "COOLDOWN"
    if len(fire_history) >= MAX_FIRES_PER_MINUTE:
        return False, "RATE_LIMIT"
    if stable_frames < AIM_STABLE_FRAMES_REQUIRED:
        return False, "AIM_UNSTABLE"
    if not in_safe_zone:
        return False, "SAFE_ZONE_BLOCK"
    return True, "READY"


mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def main():
    pose = None
    motor = None
    picam2 = None
    camera_started = False
    headless = os.environ.get("HEADLESS", "0") == "1" or not os.environ.get("DISPLAY")

    angle_buffer = deque(maxlen=SMOOTHING_WINDOW)
    last_frame_time = time.monotonic()
    last_control_time = last_frame_time
    startup_hold_until = last_frame_time + STARTUP_MOTOR_HOLD_SECONDS
    settle_until = 0.0
    persistent_offset_frames = 0
    persistent_offset_sign = 0
    target_visible_frames = 0
    turret_state = STATE_TRACKING
    armed_start_time = 0.0
    first_strike_time = 0.0
    stable_frame_count = 0
    last_fire_time = -1e9
    fire_history = deque()
    fire_flash_until = 0.0
    last_gate_reason = "READY"
    gesture_state = {
        "wrists": {
            "L": {"prev_y": None, "prev_time": None},
            "R": {"prev_y": None, "prev_time": None},
        },
        "last_downstroke_time": -1e9,
    }

    try:
        pose = mp_pose.Pose(
            model_complexity=1,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        motor = MotorController(MOTOR_CONTROL_MODE)

        picam2 = Picamera2()
        config = picam2.create_preview_configuration(
            main={"format": "BGR888", "size": (640, 480)}
        )
        picam2.configure(config)
        picam2.start()
        camera_started = True

        print("---------------------------------------")
        print("  FREE TURRET TRACKING SYSTEM STARTED  ")
        print(f"  Smoothing: {SMOOTHING_WINDOW} frames")
        print(f"  Startup hold:       {STARTUP_MOTOR_HOLD_SECONDS:.2f}s")
        print(f"  Acquire frames:     {TARGET_ACQUIRE_FRAMES}")
        print(f"  Raw target blend:   {RAW_TARGET_BLEND:.2f}")
        print(f"  Response gain:      {TRACKING_RESPONSE_GAIN:.2f}")
        print(f"  Correction sign:    {TRACKING_CORRECTION_SIGN:+.0f}")
        print(f"  Tracking deadband: {TRACKING_DEADBAND_DEG:.2f} degrees")
        print(f"  Center hold tol:   {CENTER_HOLD_TOLERANCE_DEG:.2f} degrees")
        print(f"  Control interval:  {CONTROL_UPDATE_INTERVAL_SECONDS:.2f}s")
        print(f"  Motion settle:     {MOTION_SETTLE_SECONDS:.2f}s")
        print(f"  Max correction:    {MAX_CORRECTION_STEP_DEG:.2f} degrees")
        print(f"  Lock tolerance:    {AIM_LOCK_TOLERANCE_DEG:.2f} degrees")
        print(f"  Trigger:   Double down-stroke")
        print(f"  Motor:     {motor.mode}")
        print("---------------------------------------")

        if headless:
            print("Running in headless mode (no display).")

        while True:
            frame = picam2.capture_array()
            now = time.monotonic()
            dt = max(now - last_frame_time, 1e-4)
            last_frame_time = now

            while fire_history and now - fire_history[0] > 60.0:
                fire_history.popleft()

            h, w, _ = frame.shape
            center_x = w // 2
            safe_margin = int(w * SAFE_ZONE_MARGIN_RATIO)
            safe_left = safe_margin
            safe_right = w - safe_margin

            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            image.flags.writeable = False
            results = pose.process(image)
            image.flags.writeable = True
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 1)
            cv2.line(image, (safe_left, 0), (safe_left, h), (255, 120, 0), 1)
            cv2.line(image, (safe_right, 0), (safe_right, h), (255, 120, 0), 1)

            status_color = (0, 0, 255)
            status_text = "NO TARGET"
            target_angle = None
            in_safe_zone = False
            wrist_velocity = 0.0
            active_wrist = "-"

            if results.pose_landmarks:
                mp_drawing.draw_landmarks(
                    image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
                )

                landmarks = results.pose_landmarks.landmark
                left_hip_landmark = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value]
                right_hip_landmark = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value]
                hips_visible = (
                    left_hip_landmark.visibility >= HIP_VISIBILITY_MIN
                    and right_hip_landmark.visibility >= HIP_VISIBILITY_MIN
                )
                if hips_visible:
                    target_visible_frames += 1
                else:
                    target_visible_frames = 0
                    angle_buffer.clear()
                    persistent_offset_frames = 0
                    persistent_offset_sign = 0

                left_hip = left_hip_landmark.x * w
                right_hip = right_hip_landmark.x * w
                person_center_x = int((left_hip + right_hip) / 2)

                in_safe_zone = safe_left <= person_center_x <= safe_right
                pixel_offset = person_center_x - center_x
                raw_offset_angle = (pixel_offset / w) * HFOV
                correction_angle = 0.0
                blended_offset_angle = raw_offset_angle

                tracking_ready = (
                    hips_visible
                    and target_visible_frames >= TARGET_ACQUIRE_FRAMES
                    and now >= startup_hold_until
                )

                if not tracking_ready:
                    angle_buffer.clear()
                    target_angle = motor.current_angle
                    angle_diff = 0.0
                    correction_angle = 0.0
                    if now < startup_hold_until:
                        status_text = "STARTUP HOLD"
                    elif hips_visible:
                        status_text = "ACQUIRING"
                    else:
                        status_text = "LOW CONFIDENCE"
                elif now < settle_until:
                    angle_buffer.clear()
                    target_angle = motor.current_angle
                    angle_diff = 0.0
                else:
                    angle_buffer.append(raw_offset_angle)
                    smoothed_offset_angle = sum(angle_buffer) / len(angle_buffer)
                    blended_offset_angle = (
                        RAW_TARGET_BLEND * raw_offset_angle
                        + (1.0 - RAW_TARGET_BLEND) * smoothed_offset_angle
                    )

                    if abs(blended_offset_angle) <= CENTER_HOLD_TOLERANCE_DEG:
                        persistent_offset_frames = 0
                        persistent_offset_sign = 0
                    else:
                        offset_sign = 1 if blended_offset_angle > 0 else -1
                        if offset_sign == persistent_offset_sign:
                            persistent_offset_frames += 1
                        else:
                            persistent_offset_sign = offset_sign
                            persistent_offset_frames = 1

                    if (
                        abs(blended_offset_angle) > CENTER_HOLD_TOLERANCE_DEG
                        and persistent_offset_frames >= OFFSET_PERSISTENCE_FRAMES
                        and now - last_control_time >= CONTROL_UPDATE_INTERVAL_SECONDS
                    ):
                        correction_angle = clamp(
                            blended_offset_angle
                            * TRACKING_RESPONSE_GAIN
                            * TRACKING_CORRECTION_SIGN,
                            -MAX_CORRECTION_STEP_DEG,
                            MAX_CORRECTION_STEP_DEG,
                        )

                    target_angle = clamp(
                        motor.current_angle + correction_angle,
                        MOTOR_MIN_ANGLE,
                        MOTOR_MAX_ANGLE,
                    )
                    angle_diff = abs(target_angle - motor.current_angle)

                status_color = (0, 255, 255)
                if not tracking_ready:
                    status_color = (0, 215, 255)
                elif now < settle_until:
                    status_text = "SETTLING"
                elif correction_angle == 0.0:
                    status_text = "CENTERED"
                else:
                    status_text = "HOLDING"
                if angle_diff > TRACKING_DEADBAND_DEG:
                    control_dt = max(now - last_control_time, CONTROL_UPDATE_INTERVAL_SECONDS)
                    moved = motor.move_towards(target_angle, control_dt)
                    if moved:
                        last_control_time = now
                        settle_until = now + MOTION_SETTLE_SECONDS
                        angle_buffer.clear()
                        persistent_offset_frames = 0
                        if angle_diff > ROTATING_STATUS_THRESHOLD_DEG:
                            status_color = (0, 255, 0)
                            status_text = "CORRECTING"
                        else:
                            status_color = (120, 255, 120)
                            status_text = "FINE TRACK"

                lock_error = abs(blended_offset_angle)
                if lock_error <= AIM_LOCK_TOLERANCE_DEG:
                    stable_frame_count += 1
                else:
                    stable_frame_count = 0

                downstroke, wrist_velocity, active_wrist = detect_downstroke(
                    landmarks, now, gesture_state, mp_pose
                )

                if turret_state == STATE_TRACKING:
                    if downstroke:
                        turret_state = STATE_ARMED
                        armed_start_time = now
                        first_strike_time = now
                        last_gate_reason = "ARMED_1ST_STRIKE"

                elif turret_state == STATE_ARMED:
                    if now - armed_start_time > ARMED_TIMEOUT_SECONDS:
                        turret_state = STATE_TRACKING
                        last_gate_reason = "ARM_TIMEOUT"
                    elif downstroke:
                        if now - first_strike_time > DOUBLE_STRIKE_WINDOW_SECONDS:
                            armed_start_time = now
                            first_strike_time = now
                            last_gate_reason = "REARMED"
                        else:
                            can_fire, gate_reason = fire_gate_ok(
                                now,
                                last_fire_time,
                                fire_history,
                                stable_frame_count,
                                in_safe_zone,
                            )
                            last_gate_reason = gate_reason
                            if can_fire:
                                turret_state = STATE_FIRE
                            else:
                                turret_state = STATE_TRACKING
                                print(f">>> FIRE BLOCKED: {gate_reason}")

                if turret_state == STATE_FIRE:
                    print(
                        f">>> FIRE COMMAND: Launch puck at {motor.current_angle:.1f} deg"
                    )
                    last_fire_time = now
                    fire_history.append(now)
                    fire_flash_until = now + FIRE_FLASH_SECONDS
                    turret_state = STATE_TRACKING
                    last_gate_reason = "FIRED"

                person_line_color = status_color if in_safe_zone else (0, 0, 255)
                cv2.line(
                    image,
                    (person_center_x, 0),
                    (person_center_x, h),
                    person_line_color,
                    2,
                )
            else:
                stable_frame_count = 0
                angle_buffer.clear()
                persistent_offset_frames = 0
                persistent_offset_sign = 0
                target_visible_frames = 0
                settle_until = 0.0
                reset_wrist_history(gesture_state)
                if (
                    turret_state == STATE_ARMED
                    and now - armed_start_time > ARMED_TIMEOUT_SECONDS
                ):
                    turret_state = STATE_TRACKING
                    last_gate_reason = "ARM_TIMEOUT_NO_TARGET"

            mode_label = turret_state
            mode_color = (220, 220, 220)
            if turret_state == STATE_ARMED:
                mode_color = (0, 165, 255)
            if now < fire_flash_until:
                mode_label = STATE_FIRE
                mode_color = (0, 0, 255)

            cooldown_left = max(0.0, FIRE_COOLDOWN_SECONDS - (now - last_fire_time))

            cv2.putText(
                image,
                f"Track: {status_text}",
                (10, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                status_color,
                2,
            )
            cv2.putText(
                image,
                f"Mode: {mode_label}",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                mode_color,
                2,
            )
            if target_angle is not None:
                cv2.putText(
                    image,
                    f"Target/Locked: {target_angle:.1f} / {motor.current_angle:.1f}",
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                )
            else:
                cv2.putText(
                    image,
                    f"Locked Angle: {motor.current_angle:.1f}",
                    (10, 82),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.55,
                    (220, 220, 220),
                    1,
                )
            cv2.putText(
                image,
                f"Aim Stable: {stable_frame_count}/{AIM_STABLE_FRAMES_REQUIRED}",
                (10, 108),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (220, 220, 220),
                1,
            )
            cv2.putText(
                image,
                (
                    f"Cooldown: {cooldown_left:.1f}s  Rate: "
                    f"{len(fire_history)}/{MAX_FIRES_PER_MINUTE} per min"
                ),
                (10, 134),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                image,
                f"Wrist {active_wrist} Vy: {wrist_velocity:.2f}  Gate: {last_gate_reason}",
                (10, 160),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (180, 180, 180),
                1,
            )
            cv2.putText(
                image,
                f"Safe Zone: {'OK' if in_safe_zone else 'BLOCKED'}",
                (10, 186),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 220, 0) if in_safe_zone else (0, 0, 255),
                1,
            )

            if not headless:
                cv2.imshow("Free Turret Tracker", image)
                if cv2.waitKey(5) & 0xFF == ord("q"):
                    break
    except KeyboardInterrupt:
        print("Stopping system...")
    finally:
        if motor is not None:
            try:
                motor.cleanup()
            except Exception as exc:
                print(f"Motor cleanup failed: {exc}")
        if picam2 is not None:
            try:
                if camera_started:
                    picam2.stop()
            except Exception as exc:
                print(f"Camera stop failed: {exc}")
        if pose is not None:
            try:
                pose.close()
            except Exception as exc:
                print(f"Pose cleanup failed: {exc}")
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
