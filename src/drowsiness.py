import cv2
import mediapipe as mp
import numpy as np

from playsound import playsound
import threading
import os

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]


# EAR (Eye Aspect Ratio) = (vertical distances) / (horizontal distance)
# Open eye:	High (~0.25–0.35)
# Closed eye:	Low (~0.1–0.2)
def calculate_ear(landmarks, eye_indices, w, h):
    points = []

    for idx in eye_indices:
        lm = landmarks[idx]
        # Convert normalized coordinates (0->1) to pixel values (actual image coordinates)
        # Example: lm.x = 0.5 (middle of image width), w = 1280 → x = 0.5 * 1280 = 640 pixels
        x, y = int(lm.x * w), int(lm.y * h)
        points.append((x, y))

    # points[1]      points[2]
    #      ●------●
    #   ●            ●
    #points[0]      points[3]
    #   ●            ●
    #      ●------●
    #points[5]      points[4]

    # Calculate distances using Euclidean Distance formula: sqrt((x2 - x1)^2 + (y2 - y1)^2)
    # Measures true geometric distance: independent of orientation, works during rotation, stable under movement
    # vertical distances
    v1 = np.linalg.norm(np.array(points[1]) - np.array(points[5]))   # left-middle height
    v2 = np.linalg.norm(np.array(points[2]) - np.array(points[4]))   # right-middle height

    # horizontal distance
    h_dist = np.linalg.norm(np.array(points[0]) - np.array(points[3]))

    ear = (v1 + v2) / (2.0 * h_dist)   #average eye height / eye width
    return ear


cap = cv2.VideoCapture(0)

EAR_THRESHOLD = 0.22   # Below this value, consider the eye to be closed
CLOSED_FRAMES_THRESHOLD = 30   # no. of consecutive frames with closed eyes to trigger alert
closed_frames = 0.  # Counter variable for consecutive closed eye frames

alarm_on = False

#def play_alarm():
#    playsound("assets/alarm.wav")

# Using native macOS sound player for better performance (no delay, no blocking)
def play_alarm():
    os.system("afplay assets/alarm.wav")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # height = frame.shape[0]
    # width = frame.shape[1]
    # channels = frame.shape[2], underscore _ means: ignore this variable, we only need height and width for EAR calculation
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:

            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)

            # Average to get single stable eye state and avoid noisy left/right eye separately
            avg_ear = (left_ear + right_ear) / 2.0

            # Display EAR
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            # Drowsiness detection : temporal behavior analysis - decision depends on time sequence, not one frame.
            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
            else:
                closed_frames = 0

            # Trigger alert - Message on screen
            if closed_frames >= CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (50, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if not alarm_on:
                    alarm_on = True
                    # Use threading to play sound without blocking the main thread (video processing + alarm simultaneously)
                    threading.Thread(target=play_alarm).start()
            else:
                alarm_on = False

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()