#Reference: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
#Model: Haar Cascade Classifier (Traditional Computer Vision)
import cv2

# Load face & eye cascades
# haarcascade_frontalface_default : Detects frontal faces only (looking directly at camera)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# haarcascade_profileface : Detects profile faces (looking sideways)
profile_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_profileface.xml'
)

eye_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_eye.xml'
)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)

    # If no frontal face found, try profile
    if len(faces) == 0:
        #print("No frontal face detected, trying profile...")
        faces = profile_cascade.detectMultiScale(gray, 1.3, 5)

    for (x, y, w, h) in faces:
        # Draw face box
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

        # Region of Interest (ROI) // Upper half of detected face box
        roi_gray = gray[y:y+h//2, x:x+w]
        roi_color = frame[y:y+h//2, x:x+w]

        eyes = eye_cascade.detectMultiScale(roi_gray, 1.1, 6)

        # Keep only top 2 detections
        eyes = sorted(eyes, key=lambda x: x[2]*x[3], reverse=True)[:2]

        for (ex, ey, ew, eh) in eyes:
            if ew > 30 and eh > 30:
                cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)

    cv2.imshow("Eye Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()