import cv2
import numpy as np
import mediapipe as mp

# Initialize MediaPipe pose class and drawing utilities.
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set up video capture with OpenCV.
cap = cv2.VideoCapture('data/sample_video.mp4')  # Ensure 'sample_video.mp4' is in the 'data' folder

# Initialize variables for push-up counting.
pushup_count = 0
position = None  # Tracks whether the athlete is 'up' or 'down'.
feedback = []    # Store feedback from users.

# Function to compute the angle between three points.
def calculate_angle(a, b, c):
    a = np.array(a)  # Start point.
    b = np.array(b)  # Middle point.
    c = np.array(c)  # End point.

    radians = np.arctan2(c[1] - b[1], c[0] - b[0]) - np.arctan2(a[1] - b[1], a[0] - b[0])
    angle = np.abs(radians * 180.0 / np.pi)

    if angle > 180.0:
        angle = 360 - angle

    return angle

# Function to provide feedback based on performance.
def provide_feedback(count):
    if count < 10:
        return "Keep going! You're doing great!"
    elif count < 20:
        return "Excellent work! You're getting stronger!"
    else:
        return "Outstanding performance! You're a champ!"

# Function to display feedback on the frame.
def display_feedback(image, feedback_text):
    cv2.rectangle(image, (0, 0), (640, 100), (30, 30, 30), -1)
    cv2.putText(image, feedback_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

# Start the pose detection.
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Error: Failed to capture video. Please check your webcam.")
            break

        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False

        results = pose.process(image)

        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        try:
            landmarks = results.pose_landmarks.landmark

            shoulder = [
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y,
            ]
            elbow = [
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y,
            ]
            wrist = [
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y,
            ]

            angle = calculate_angle(shoulder, elbow, wrist)

            cv2.putText(
                image,
                str(int(angle)),
                tuple(np.multiply(elbow, [640, 480]).astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            if angle > 160:
                position = "up"
            if angle < 90 and position == "up":
                position = "down"
                pushup_count += 1
                feedback_text = provide_feedback(pushup_count)

        except Exception as e:
            print("Error in processing landmarks:", e)
            continue

        cv2.rectangle(image, (0, 0), (225, 73), (45, 45, 45), -1)
        cv2.putText(
            image,
            "PUSH-UPS",
            (15, 12),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            str(pushup_count),
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            2,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )

        # Display feedback
        display_feedback(image, feedback_text)

        mp_drawing.draw_landmarks(
            image,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(200, 200, 200), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(100, 200, 100), thickness=2, circle_radius=2),
        )

        cv2.imshow("Athlete Performance Tracker - Push-Up Counter", image)

        if cv2.waitKey(10) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
