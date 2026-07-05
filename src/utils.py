# Utility functions

import cv2
import os
import time

# Important MediaPipe Face Mesh landmarks
IMPORTANT_POINTS = [
    1,                  # Nose tip
    33, 133,            # Left eye corners
    362, 263,           # Right eye corners
    469, 470, 471, 472, # Right iris
    474, 475, 476, 477  # Left iris
]


def draw_landmark_debug(frame,
                        face_landmarks,
                        width,
                        height,
                        show_ids=True,
                        show_points=True,
                        points=IMPORTANT_POINTS):
    
    """
    Draw selected MediaPipe Face Mesh landmarks for debugging

    """
    
    # print(len(face_landmarks.landmark)) : 478 - Face Mesh (468) with iris landmarks

    for idx in points:

        landmark = face_landmarks.landmark[idx]

        x = int(landmark.x * width)
        y = int(landmark.y * height)

        if show_points:
            cv2.circle(frame, (x, y), 4, (0, 0, 255), -1)

        if show_ids:
            cv2.putText(frame, str(idx), (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 255, 255), 1)


def play_alarm():
    """
    Play alarm sound

    Using native macOS sound player for better performance (no delay, no blocking)

    """
#    playsound("assets/alarm.wav")
    os.system("afplay assets/alarm.wav")


def calculate_fps(previous_time):
    """
    Calculate FPS based on the elapsed time between frames

    """

    current_time = time.time()

    time_diff = current_time - previous_time

    if time_diff > 0:
        fps = 1.0 / time_diff
    else:
        fps = 0.0

    return fps, current_time
