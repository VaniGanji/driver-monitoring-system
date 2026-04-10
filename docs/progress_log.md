# 📅 Project Progress Log – Driver Monitoring System

---

## Project Setup

### Completed:
- Created GitHub repository
- Cloned repository to local system
- Defined project folder structure (src, models, dataset, docs, results, scripts)
- Added README with project overview and goals
- Configured .gitignore for clean repository management

### Outcome:
- Clean and organized project structure ready for development

---

## Environment Setup

### Completed:
- Created Python virtual environment
- Installed required libraries (OpenCV, NumPy, Matplotlib, TensorFlow)
- Implemented initial test script to verify setup

### Outcome:
- Development environment successfully configured and verified

---

## Camera Stream

### Completed:
- Implemented real-time webcam video capture using OpenCV
- Displayed live video feed using cv2.imshow()
- Handled user input and exit control using cv2.waitKey()

### Outcome:
- Functional real-time video pipeline established

---

## Face Detection

### Completed:
- Implemented face detection using Haar Cascade classifier
- Integrated detection into live camera stream
- Tuned parameters (scaleFactor, minNeighbors) for better accuracy
- Displayed bounding boxes around detected faces

- Generated requirements.txt for reproducibility (missed while Environment setup)

### Outcome:
- Real-time face detection successfully working
- Foundation established for further driver monitoring features

---

## Eye Detection + Detection Stability Improvements

### Completed:
- Implemented eye detection using Haar Cascade within detected face region
- Optimized detection by restricting eye search to upper face ROI
- Tuned parameters (`scaleFactor`, `minNeighbors`) to reduce false positives
- Applied filtering to keep only the most relevant eye detections
- Improved system stability by reducing multiple/duplicate detections

### Observations:
- Haar Cascade-based detection works but shows limitations in real-world scenarios:
  - False positives under head movement and pose variation
  - Reduced stability during face tilt
  - Multiple overlapping detections in some frames

### Decision:
- Identified need for more robust face and landmark detection approach
- Planned migration from Haar Cascade to MediaPipe for improved accuracy and stability

---

