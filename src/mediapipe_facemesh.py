#Reference: https://en.wikipedia.org/wiki/MediaPipe
import cv2
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    # False = Video Stream, True = Static Image
    static_image_mode=False,
    # Max no. of faces to detect (1 only for driver)
    max_num_faces=1,
    # enable high prescision landmarks - adds iris + eye detail points
    refine_landmarks=True,
    # threshold for face detection: 0.3 - more detections, less accurate,0.5 - balanced 0.7 - fewer detections but more accurate, may miss face
    min_detection_confidence=0.5,
    # threshold for face tracking after detection - (tracking confidence drops → re-detect face)
    # 0.3 - loose tracking, 0.5 - balanced, 0.7 - stable but may re-detect often 
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:    # Check if any faces are detected
        for face_landmarks in results.multi_face_landmarks:   # Loop through detected faces (only 1 in our case)

            # Draw all landmarks (468 points) and connections (triangulated mesh)
            mp_drawing.draw_landmarks(
                frame,                           # image to draw on
                face_landmarks,                  # 468 facial points
                mp_face_mesh.FACEMESH_CONTOURS   # connections for drawing the mesh
            )

    cv2.imshow("Face Mesh", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()