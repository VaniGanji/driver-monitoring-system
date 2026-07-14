# Thresholds & configuration

from enum import Enum

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
HEAD_OFFSET_THRESHOLD = 30   # pixels, threshold for head position estimation
DISTRACTION_THRESHOLD = 60   # at FPS of 30, 60 frames ~ 2 seconds,
SHOW_LANDMARK_IDS = False   # debugging: show landmark IDs on the video feed for reference

GAZE_LEFT_THRESHOLD = 0.30
GAZE_RIGHT_THRESHOLD = 0.70

# Display positions for text overlays on the video feed
TEXT_FACE_STATUS_POS = (30, 30)
TEXT_FPS_POS = (50, 60)
TEXT_EAR_POS = (50, 90)
TEXT_BLINK_POS = (50, 120)
TEXT_DROWSY_ALERT_POS = (50, 150)
TEXT_ATTENTION_POS = (50, 180)
TEXT_DISTRACTION_ALERT_POS = (50, 210)

class AttentionState(Enum):
    ATTENTIVE = "ATTENTIVE"
    LOOKING_LEFT = "LOOKING LEFT"
    LOOKING_RIGHT = "LOOKING RIGHT"
    HEAD_TURNED = "HEAD TURNED"
    DISTRACTED = "DISTRACTED"
    UNKNOWN = "UNKNOWN"