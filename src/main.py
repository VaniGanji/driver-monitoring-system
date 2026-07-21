# Drowsiness Detection using Eye Aspect Ratio (EAR) with MediaPipe Face Mesh
# Immediate alert when eyes are closed for a certain number of consecutive frames
# Fatigue monitoring - Blink count per minute as secondary analytics for drowsiness
# Attention Monitoring using Head Pose Estimation and Eye Gaze Estimation

import cv2
import mediapipe as mp
import threading
import time
from collections import deque

from config import *
import event_logger
import eye_monitor
import attention_monitor
import performance_monitor
import utils


# Initialize variables
alarm_on = False
distracted_frames = 0
fps_timer = time.time()   # frames per second
# deque to persist across frames
# at FPS of 30, 5 frames ~ 0.16 seconds, enough to smooth out rapid eye movements/noise and system still feels responsive
gaze_history = deque(maxlen=5)

# variables to log events
last_attention_state = AttentionState.UNKNOWN
face_detected = False
last_drowsy = False


mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

event_logger.initialize_logger()
perf = performance_monitor.PerformanceMonitor()

cap = cv2.VideoCapture(0)

while True:
    perf.start("capture")
    ret, frame = cap.read()
    perf.stop("capture")
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

    perf.start("face_mesh")
    results = face_mesh.process(rgb)
    perf.stop("face_mesh")

    if results.multi_face_landmarks:

        cv2.putText(frame, "FACE DETECTED", TEXT_FACE_STATUS_POS,
            cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,255), 3)
        # logging to file
        if not face_detected:
            event_logger.log_event("Face Detected")
            face_detected = True

        for face_landmarks in results.multi_face_landmarks:

            ### Landmark Visualization & Debugging ###
            if SHOW_LANDMARK_IDS:
                utils.draw_landmark_debug(frame, face_landmarks, w, h)

            ### Calculate and display Frames Per Second (FPS) ###
            fps, fps_timer = utils.calculate_fps(fps_timer)
            cv2.putText(frame, f"FPS: {fps:.1f}", TEXT_FPS_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,0), 2)

            perf.start("eye_monitor")
            ### EAR (Eye Aspect Ratio) ###
            left_ear = eye_monitor.calculate_ear(face_landmarks.landmark, LEFT_EYE, w, h)
            right_ear = eye_monitor.calculate_ear(face_landmarks.landmark, RIGHT_EYE, w, h)
            # Average to get single stable eye state and avoid noisy left/right eye separately
            avg_ear = (left_ear + right_ear) / 2.0

            ### Drowsiness detection : temporal behavior analysis - decision depends on time sequence, not one frame ###
            blink_count, is_drowsy = eye_monitor.process_eye_state(avg_ear)
            perf.stop("eye_monitor")

            # Display EAR
            cv2.putText(frame, f"EAR: {avg_ear:.2f}", TEXT_EAR_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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

            # logging to file
            if is_drowsy != last_drowsy:
                if is_drowsy:
                    event_logger.log_event("Drowsiness Alert")
                else:
                    event_logger.log_event("Drowsiness Cleared")
                last_drowsy = is_drowsy

            perf.start("attention_monitor")
            ### Head Position Estimation ###
            head_direction = attention_monitor.estimate_head_direction(face_landmarks, w, HEAD_OFFSET_THRESHOLD)
            ### Eye Gaze Estimation ###
            stable_ratio, gaze = attention_monitor.estimate_gaze(face_landmarks, w, h, gaze_history)

            ### Driver Attention State Fusion : Head Position + Eye Gaze ###
            attention_state = attention_monitor.get_attention_state(head_direction, gaze)
            perf.stop("attention_monitor")

            cv2.putText(frame, f"Attention: {attention_state.value}", TEXT_ATTENTION_POS,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2)

            # logging to file
            if attention_state != last_attention_state:
                event_logger.log_event(attention_state)
                last_attention_state = attention_state

            ### Distraction Duration Monitoring ###
            if attention_state != AttentionState.ATTENTIVE:
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
        # logging to file
        if face_detected:
            event_logger.log_event("Face Lost")
            face_detected = False

    if SHOW_PERFORMANCE:
        performance_monitor.display_performance(perf)

    cv2.imshow("Drowsiness Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()