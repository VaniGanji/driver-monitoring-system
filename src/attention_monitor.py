# Head pose, gaze, attention fusion

import numpy as np
from config import *

def iris_center(landmarks, iris_indices, w, h):
    """
    Returns the center point of the iris.
    """
    points = []

    for idx in iris_indices:
        lm = landmarks[idx]
        points.append((lm.x * w, lm.y * h))

    center = np.mean(points, axis=0)

    return center


def gaze_ratio(iris_center_x, outer_x, inner_x):
    """
    Calculate the gaze ratio based on the iris center and eye corner positions.
    """
    left = min(outer_x, inner_x)
    right = max(outer_x, inner_x)

    eye_width = right - left

    if eye_width == 0:
        return 0.5

    ratio = (iris_center_x - left) / eye_width

    return ratio


def get_eye_corners(face_landmarks, width):
    """
    Returns the x-coordinates of the eye corner landmarks.
    """

    left_eye_left_x = int(face_landmarks.landmark[LEFT_EYE_LEFT].x * width)
    left_eye_right_x = int(face_landmarks.landmark[LEFT_EYE_RIGHT].x * width)

    right_eye_left_x = int(face_landmarks.landmark[RIGHT_EYE_LEFT].x * width)
    right_eye_right_x = int(face_landmarks.landmark[RIGHT_EYE_RIGHT].x * width)

    return (
        left_eye_left_x,
        left_eye_right_x,
        right_eye_left_x,
        right_eye_right_x
    )


def estimate_head_direction(
    face_landmarks,
    width,
    head_offset_threshold
):
    """
    Estimate driver's head direction.
    """

    (
        left_eye_left_x,
        _,
        _,
        right_eye_right_x
    ) = get_eye_corners(face_landmarks, width)

    nose = face_landmarks.landmark[NOSE_TIP]

    nose_x = int(nose.x * width)

    eye_center_x = (left_eye_left_x + right_eye_right_x) // 2

    head_offset = nose_x - eye_center_x

    if head_offset > head_offset_threshold:
        return "Head Right"

    elif head_offset < -head_offset_threshold:
        return "Head Left"

    else:
        return "Head Forward"


def estimate_gaze(
    face_landmarks,
    width,
    height,
    gaze_history
):
    """
    Estimate driver's eye gaze.
    """

    (
        left_eye_left_x,
        left_eye_right_x,
        right_eye_left_x,
        right_eye_right_x
    ) = get_eye_corners(face_landmarks, width)

    left_center = iris_center(
        face_landmarks.landmark,
        LEFT_IRIS,
        width,
        height
    )

    right_center = iris_center(
        face_landmarks.landmark,
        RIGHT_IRIS,
        width,
        height
    )

    left_ratio = gaze_ratio(
        left_center[0],
        left_eye_left_x,
        left_eye_right_x
    )

    right_ratio = gaze_ratio(
        right_center[0],
        right_eye_left_x,
        right_eye_right_x
    )

    avg_ratio = (left_ratio + right_ratio) / 2

    gaze_history.append(avg_ratio)

    stable_ratio = sum(gaze_history) / len(gaze_history)

    if stable_ratio < GAZE_LEFT_THRESHOLD:
        gaze = "Eyes Left"

    elif stable_ratio > GAZE_RIGHT_THRESHOLD:
        gaze = "Eyes Right"

    else:
        gaze = "Eyes Forward"

    return stable_ratio, gaze


def get_attention_state(head_direction, gaze):
    """
    Returns the attention state based on head direction and gaze.
    """

    if head_direction == "Head Forward":

        if gaze == "Eyes Forward":
            return AttentionState.ATTENTIVE

        elif gaze == "Eyes Left":
            return AttentionState.LOOKING_LEFT

        elif gaze == "Eyes Right":
            return AttentionState.LOOKING_RIGHT

    return AttentionState.HEAD_TURNED