# EAR, blink detection, drowsiness

import numpy as np
import time

from config import EAR_THRESHOLD, CLOSED_FRAMES_THRESHOLD

state = {
    "closed_frames": 0,
    "blink_detected": False,
    "blink_count": 0,
    "blink_count_reset_timer": time.time(),
}


def calculate_ear(landmarks, eye_indices, w, h):
    """
    Calculate the Eye Aspect Ratio (EAR).

    EAR (Eye Aspect Ratio) = (vertical distances) / (horizontal distance)
    Open eye:	High (~0.25–0.35)
    Closed eye:	Low (~0.1–0.2)

    """
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


def process_eye_state(avg_ear):
    """
    Process the eye state based on the calculated EAR.

    """

    if avg_ear < EAR_THRESHOLD:
        state["closed_frames"] += 1
        state["blink_detected"] = True
    else:
        state["closed_frames"] = 0
        if state["blink_detected"]:
            state["blink_count"] += 1
            state["blink_detected"] = False

    ### reset blink count every minute ###
    elapsed_time = time.time() - state["blink_count_reset_timer"]
    if elapsed_time >= 60:
        state["blink_count"] = 0
        state["blink_count_reset_timer"] = time.time()

    is_drowsy = state["closed_frames"] >= CLOSED_FRAMES_THRESHOLD
    return state["blink_count"], is_drowsy


def reset_eye_state():
    """
    Reset the eye state to initial values.

    """
    state["closed_frames"] = 0
    state["blink_detected"] = False
    state["blink_count"] = 0
    state["blink_count_reset_timer"] = time.time()