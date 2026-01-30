import cv2
import mediapipe as mp
import numpy as np
from collections import deque
import time
from picamera2 import Picamera2

# --- CONFIGURATION VARIABLES ---
# 1. Smoothing: How many frames to average? (Higher = Smoother but slower response)
SMOOTHING_WINDOW = 15

# 2. Deadband: Minimum angle change required to trigger a move (Degrees)
# If the change is less than this, the machine stays still.
MOVEMENT_THRESHOLD = 3.0

# 3. Camera Field of View (Standard Webcam)
HFOV = 88.0

# --- SETUP ---
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=1, min_detection_confidence=0.5, min_tracking_confidence=0.5
)
mp_drawing = mp.solutions.drawing_utils

# Initialize buffer for smoothing
angle_buffer = deque(maxlen=SMOOTHING_WINDOW)

# State variables
current_motor_angle = 0.0  # Where the machine is currently pointing
last_command_time = 0

picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"format": "BGR888", "size": (640, 480)})
picam2.configure(config)
picam2.start()

print("---------------------------------------")
print("  BASKETBALL TRACKING SYSTEM STARTED   ")
print(f"  Smoothing: {SMOOTHING_WINDOW} frames")
print(f"  Deadband:  {MOVEMENT_THRESHOLD} degrees")
print("---------------------------------------")

while True:
    frame = picam2.capture_array()

    h, w, _ = frame.shape
    center_x = w // 2

    # Process Image
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False
    results = pose.process(image)
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw Center Reference
    cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 1)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 1. Calculate Raw Position
        landmarks = results.pose_landmarks.landmark
        left_hip = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w
        right_hip = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w
        person_center_x = int((left_hip + right_hip) / 2)

        # 2. Calculate Raw Angle
        pixel_offset = person_center_x - center_x
        raw_angle = (pixel_offset / w) * HFOV

        # 3. Add to Buffer (The "Damper")
        angle_buffer.append(raw_angle)

        # 4. Calculate Smoothed Angle
        smoothed_angle = sum(angle_buffer) / len(angle_buffer)

        # 5. Logic: Should we move the rotor?
        # We calculate the difference between where the motor IS vs where it SHOULD BE
        angle_diff = abs(smoothed_angle - current_motor_angle)

        status_color = (0, 255, 255)  # Yellow (Holding)
        status_text = "HOLDING"

        # Check if difference is big enough to justify moving (The "Deadband")
        if angle_diff > MOVEMENT_THRESHOLD:
            current_motor_angle = smoothed_angle
            status_color = (0, 255, 0)  # Green (Moving)
            status_text = "ROTATING"

            # --- OUTPUT FOR ROTOR ---
            print(f">>> ROTOR COMMAND: Rotate to {current_motor_angle:.1f} deg")

        # --- VISUALIZATION ---
        # Draw the person's center
        cv2.line(image, (person_center_x, 0), (person_center_x, h), status_color, 2)

        # Display info on screen
        cv2.putText(
            image,
            f"Status: {status_text}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            status_color,
            2,
        )
        cv2.putText(
            image,
            f"Target Angle: {smoothed_angle:.1f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (200, 200, 200),
            1,
        )
        cv2.putText(
            image,
            f"Locked Angle: {current_motor_angle:.1f}",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 255),
            1,
        )

    cv2.imshow("Basketball Tracker", image)
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

picam2.stop()
cv2.destroyAllWindows()
