import cv2
import numpy as np
import mediapipe as mp


# 1. Initialize MediaPipe Pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# 2. Hook up to the laptop camera (0 is usually the default webcam)
cap = cv2.VideoCapture(0)

# Standard laptop webcam Horizontal Field of View (in degrees)
# Adjust this if your camera has a wider/narrower lens.
HFOV = 88.0

print("Press 'q' to quit.")

while cap.isOpened():
    success, frame = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Get frame dimensions
    h, w, _ = frame.shape
    center_x = w // 2

    # Convert the BGR image to RGB (MediaPipe requires RGB)
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    image.flags.writeable = False

    # 3. Process the image to find the human body
    results = pose.process(image)

    # Convert back to BGR for OpenCV display
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the center line of the camera view
    cv2.line(image, (center_x, 0), (center_x, h), (255, 255, 255), 1)
    cv2.putText(
        image,
        "0 deg (Center)",
        (center_x - 50, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
    )

    if results.pose_landmarks:
        # Draw the skeleton on the image
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS
        )

        # 4. Find the center of the human (using the midpoint between hips)
        # MediaPipe uses normalized coordinates [0.0, 1.0], so we multiply by frame width
        landmarks = results.pose_landmarks.landmark
        left_hip_x = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w
        right_hip_x = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w

        person_center_x = int((left_hip_x + right_hip_x) / 2)

        # Draw a vertical line for the person's center
        cv2.line(image, (person_center_x, 0), (person_center_x, h), (0, 255, 0), 2)

        # 5. Calculate the angle
        # Positive angle = right of center, Negative angle = left of center
        pixel_offset = person_center_x - center_x
        angle = (pixel_offset / w) * HFOV

        # Display the angle on the screen
        text = f"Angle: {angle:.1f} degrees"
        cv2.putText(
            image,
            text,
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )

    # Show the final image
    cv2.imshow("Human Angle Tracker", image)

    # Break loop if 'q' is pressed
    if cv2.waitKey(5) & 0xFF == ord("q"):
        break

# Clean up
cap.release()
cv2.destroyAllWindows()
