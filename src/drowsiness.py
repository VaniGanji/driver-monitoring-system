# Drowsiness Detection using Eye Aspect Ratio (EAR) with MediaPipe Face Mesh
# Immediate alert when eyes are closed for a certain number of consecutive frames
# Fatigue monitoring - Blink count per minute as secondary analytics for drowsiness

import cv2
import mediapipe as mp
import numpy as np
from playsound import playsound
import threading
import os
import time
from collections import deque

# Constants
# Eye landmark indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
LEFT_IRIS = [474, 475, 476, 477]
RIGHT_IRIS = [469, 470, 471, 472]
LEFT_EYE_LEFT = 33
LEFT_EYE_RIGHT = 133
RIGHT_EYE_LEFT = 362
RIGHT_EYE_RIGHT = 263
# Head position estimation
NOSE_TIP = 1
EAR_THRESHOLD = 0.22   # Below this value, consider the eye to be closed
CLOSED_FRAMES_THRESHOLD = 30   # no. of consecutive frames with closed eyes to trigger alert
DISTRACTION_THRESHOLD = 30
SHOW_LANDMARK_IDS = False   # debugging: show landmark IDs on the video feed for reference

# Initialize variables
closed_frames = 0   # Counter variable for consecutive closed eye frames
alarm_on = False
blink_detected = False
blink_count = 0
distracted_frames = 0
blink_count_reset_timer = time.time()
fps_timer = time.time()   # frames per second
# deque to persist across frames
# at FPS of 30, 5 frames = 0.16 seconds, enough to smooth out rapid eye movements/noise and system still feels responsive
gaze_history = deque(maxlen=5)


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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

#def play_alarm():
#    playsound("assets/alarm.wav")

# Using native macOS sound player for better performance (no delay, no blocking)
def play_alarm():
    os.system("afplay assets/alarm.wav")

def iris_center(landmarks, iris_indices, w, h):
    points = []

    for idx in iris_indices:
        lm = landmarks[idx]
        points.append((lm.x * w, lm.y * h))

    center = np.mean(points, axis=0)

    return center


def gaze_ratio(iris_center_x, outer_x, inner_x):

    left = min(outer_x, inner_x)
    right = max(outer_x, inner_x)

    eye_width = right - left

    if eye_width == 0:
        return 0.5

    ratio = (iris_center_x - left) / eye_width

    return ratio

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
     # flip horizontally for mirror view for webcam - more intuitive for user, 
     # not required for real application with fixed camera
    frame = cv2.flip(frame, 1)

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

            ### Landmark Visualization & Debugging ###
            # print(len(face_landmarks.landmark)) : 478 - Face Mesh (468) with iris landmarks
            if SHOW_LANDMARK_IDS:
                for idx, landmark in enumerate(face_landmarks.landmark):
                        x = int(landmark.x * w)
                        y = int(landmark.y * h)
                        #cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)

                        IMPORTANT_POINTS = [33, 133, 362, 263, 469, 470, 471, 472, 474, 475, 476, 477, 1]
                        if idx in IMPORTANT_POINTS:
                            cv2.putText(frame, str(idx), (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)
                            cv2.circle(frame, (x, y), 4, (0,0,255), -1)

            ### EAR (Eye Aspect Ratio) ###
            left_ear = calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            # Average to get single stable eye state and avoid noisy left/right eye separately
            avg_ear = (left_ear + right_ear) / 2.0
            # Display EAR
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ### Drowsiness detection : temporal behavior analysis - decision depends on time sequence, not one frame ###
            if avg_ear < EAR_THRESHOLD:
                closed_frames += 1
                blink_detected = True
            else:
                closed_frames = 0
                if blink_detected:
                    blink_count += 1
                    blink_detected = False

            ### reset blink count every minute ###
            elapsed_time = time.time() - blink_count_reset_timer
            if elapsed_time >= 60:
                blink_count = 0
                blink_count_reset_timer = time.time()

            cv2.putText(frame, f"Blinks: {blink_count}", (30, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            ### Trigger alert - Message on screen and Audio alert ###
            if closed_frames >= CLOSED_FRAMES_THRESHOLD:
                cv2.putText(frame, "DROWSINESS ALERT!", (30, 150),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if not alarm_on:
                    alarm_on = True
                    # Use threading to play sound without blocking the main thread (video processing + alarm simultaneously)
                    threading.Thread(target=play_alarm).start()
            else:
                alarm_on = False

            ### Eye corner landmarks for head position and Eye Gaze ###
            left_eye_left_corner = face_landmarks.landmark[LEFT_EYE_LEFT]
            left_eye_right_corner = face_landmarks.landmark[LEFT_EYE_RIGHT]
            right_eye_left_corner = face_landmarks.landmark[RIGHT_EYE_LEFT]
            right_eye_right_corner = face_landmarks.landmark[RIGHT_EYE_RIGHT]

            left_eye_left_x = int(left_eye_left_corner.x * w)
            left_eye_right_x = int(left_eye_right_corner.x * w)
            right_eye_left_x = int(right_eye_left_corner.x * w)
            right_eye_right_x = int(right_eye_right_corner.x * w)
            
            ### Head position estimation ###
            nose_tip = face_landmarks.landmark[NOSE_TIP]
            nose_x = int(nose_tip.x * w)
            nose_y = int(nose_tip.y * h)
            
            eye_center_x = (left_eye_left_x + right_eye_right_x) // 2
            head_offset = nose_x - eye_center_x

            if head_offset > 20:
                direction = "Looking Right"

            elif head_offset < -20:
                direction = "Looking Left"

            else:
                direction = "Forward"

            cv2.putText(frame, f"Head: {direction}", (30, 120),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            ### Distraction Duration Monitoring ###
            if direction == "Forward":
                distracted_frames = 0
            else:
                distracted_frames += 1

            if distracted_frames >= DISTRACTION_THRESHOLD:
                cv2.putText(frame, "DRIVER DISTRACTED!", (40,220), 
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

            ### Calculate and display Frames Per Second (FPS) ###
            current_time = time.time()
            time_diff = current_time - fps_timer
            if time_diff > 0:
                fps = 1 / time_diff
            else:
                fps = 0
            fps_timer = current_time
            cv2.putText(frame, f"FPS: {fps:.1f}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)
            
            ### Eye Gaze Estimation ###
            left_center = iris_center(face_landmarks.landmark, LEFT_IRIS, w, h)
            right_center = iris_center(face_landmarks.landmark, RIGHT_IRIS, w, h)

            left_gaze_ratio = gaze_ratio(left_center[0], left_eye_left_x, left_eye_right_x)
            right_gaze_ratio = gaze_ratio(right_center[0], right_eye_left_x, right_eye_right_x)

            avg_ratio = (left_gaze_ratio + right_gaze_ratio) / 2
            # store last 5 gaze ratios for smoothing and stability, to avoid sudden jumps in gaze direction due to noise or rapid eye movement
            gaze_history.append(avg_ratio)
            stable_ratio = sum(gaze_history) / len(gaze_history)
            
            if stable_ratio < 0.30:
                gaze = "Eyes Left"

            elif stable_ratio > 0.70:
                gaze = "Eyes Right"

            else:
                gaze = "Eyes Forward"
            
            cv2.putText(frame, f"Gaze Ratio: {gaze}", (20,180),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()