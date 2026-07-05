#Reference: https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html
#Model: Haar Cascade Classifier (Traditional Computer Vision)

import cv2

# Load pre-trained face detector model (Haar Cascade)
face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
)

# Start webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Cannot access camera")
    exit()

while True:
    ret, frame = cap.read()

    if not ret:
        print("Error: Failed to capture frame")
        break

    # Convert to grayscale (IMPORTANT)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces
    # scaleFactor: Parameter specifying how much the image size is reduced at each image scale. (e.g., 1.3 means reduce size by 30% at each step)
    # minNeighbors: Parameter specifying how many neighbors each candidate rectangle should have to retain it. 
    # (e.g., 5 means that for a region to be considered a face, it must have at least 5 neighboring rectangles that also detect a face)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.3,
        minNeighbors=5
    )

    # Draw bounding box
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)

    # Show output
    cv2.imshow("Face Detection", frame)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()