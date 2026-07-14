# Drowsiness Detection using Eye Aspect Ratio (EAR) with MediaPipe Face Mesh
# Immediate alert when eyes are closed for a certain number of consecutive frames
# Fatigue monitoring - Blink count per minute as secondary analytics for drowsiness

import cv2
import mediapipe as mp
import threading
import time
from collections import deque

from config import *
import eye_monitor
import attention_monitor
import utils

# Initialize variables
alarm_on = False
distracted_frames = 0
fps_timer = time.time()   # frames per second
# deque to persist across frames
# at FPS of 30, 5 frames ~ 0.16 seconds, enough to smooth out rapid eye movements/noise and system still feels responsive
gaze_history = deque(maxlen=5)


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

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
            if SHOW_LANDMARK_IDS:
                utils.draw_landmark_debug(frame, face_landmarks, w, h)
            
            ### Calculate and display Frames Per Second (FPS) ###
            fps, fps_timer = utils.calculate_fps(fps_timer)
            cv2.putText(frame, f"FPS: {fps:.1f}", TEXT_FPS_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            ### EAR (Eye Aspect Ratio) ###
            left_ear = eye_monitor.calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_monitor.calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            # Average to get single stable eye state and avoid noisy left/right eye separately
            avg_ear = (left_ear + right_ear) / 2.0
            # Display EAR
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", TEXT_EAR_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            ### Drowsiness detection : temporal behavior analysis - decision depends on time sequence, not one frame ###
            blink_count, is_drowsy = eye_monitor.process_eye_state(avg_ear)

            cv2.putText(frame, f"Blinks: {blink_count}", TEXT_BLINK_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)

            ### Trigger alert - Message on screen and Audio alert ###
            if is_drowsy:
                cv2.putText(frame, "DROWSINESS ALERT!", TEXT_DROWSY_ALERT_POS,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)

                if not alarm_on:
                    alarm_on = True
                    # Use threading to play sound without blocking the main thread (video processing + alarm simultaneously)
                    threading.Thread(target=utils.play_alarm).start()
            else:
                alarm_on = False

            ### Head Position Estimation ###
            head_direction = attention_monitor.estimate_head_direction(face_landmarks, w, HEAD_OFFSET_THRESHOLD)
            # cv2.putText(frame, f"Head: {head_direction}", (30, 120),
            #             cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,255), 2)

            ### Eye Gaze Estimation ###
            stable_ratio, gaze = attention_monitor.estimate_gaze(face_landmarks, w, h, gaze_history)
            # cv2.putText(frame, f"Gaze : {gaze}", (30,180),
            #             cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
            
            ### Driver Attention State Fusion : Head Position + Eye Gaze ###
            attention = attention_monitor.get_attention_state(head_direction, gaze)
            
            cv2.putText(frame, f"Attention: {attention.value}", TEXT_ATTENTION_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)
            
            ### Distraction Duration Monitoring ###
            if attention != AttentionState.ATTENTIVE:
                distracted_frames += 1
            else:
                distracted_frames = 0

            if distracted_frames > DISTRACTION_THRESHOLD:
                cv2.putText(frame, "DISTRACTION ALERT!", TEXT_DISTRACTION_ALERT_POS,
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    else:
        head_direction = "Unknown"
        gaze = "Unknown"
        attention_state = AttentionState.UNKNOWN
        eye_monitor.reset_eye_state()
        cv2.putText(frame, "NO FACE DETECTED", TEXT_FACE_STATUS_POS,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()