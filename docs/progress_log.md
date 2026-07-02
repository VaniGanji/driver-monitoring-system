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

## MediaPipe Face Detection (real-time AI perception system)

### Completed:
- Integrated MediaPipe face detection for improved robustness
- Replaced Haar Cascade for face detection
- Achieved stable detection under head tilt and pose variation

### Outcome:
- Reliable real-time face detection suitable for DMS applications

---

## Eye Landmark Extraction + EAR Calculation

### Completed:
- Integrated MediaPipe Face Mesh landmarks into drowsiness detection pipeline
- Extracted left and right eye landmark coordinates using predefined landmark indices
- Implemented Eye Aspect Ratio (EAR) calculation using Euclidean distance
- Calculated vertical and horizontal eye distances for eye openness estimation
- Combined left and right eye EAR values for stable eye-state monitoring
- Displayed real-time EAR values on video stream

### Outcome:
- Successfully built foundational eye-state monitoring logic for Driver Monitoring System (DMS)
- Established core pipeline for blink and drowsiness detection

### Current Pipeline:
Camera Input → MediaPipe Face Mesh → Eye Landmark Extraction → EAR Calculation → Real-time Eye State Monitoring

---

## Drowsiness Detection Logic

### Completed:
- Implemented EAR threshold-based eye closure detection
- Added consecutive frame monitoring for temporal drowsiness analysis
- Integrated real-time drowsiness alert system
- Displayed visual alert during prolonged eye closure

### Outcome:
- Successfully implemented core drowsiness detection functionality for DMS

### Current Pipeline:
Camera Input → MediaPipe Face Mesh → Eye Landmark Extraction → EAR Calculation → Temporal Eye Closure Monitoring → Drowsiness Alert

---

## Audio Alert

### Completed:
- Integrated audio alert system for drowsiness warning

### Outcome:
- Audio alert for drowsiness detection

---

## Blinks count for Fatigue monitoring

### Completed:
- Implemented Blicks count per minute as secondary analytics for drowsiness
- Displayed Blink count on video stream

### Outcome:
- Blink count and counter reset per minute

---

## Head Pose Estimation and Attention Monitoring

### Completed:
- Implemented basic head pose estimation using facial landmarks
- Added driver attention monitoring functionality
- Calculated head orientation using nose and eye alignment
- Displayed real-time head direction status
- Added distraction warning capability

### Outcome:
- Enhanced Driver Monitoring System with distraction detection capability
- Added foundational attention-awareness monitoring

### Current Pipeline:
Camera Input → Face Mesh → Eye Landmark Extraction → EAR Calculation → Blink Detection → Drowsiness Detection → Head Direction Estimation → Attention Monitoring → Visual + Audio Alerts

---

## Distraction Duration Monitoring and FPS Measurement

### Completed:
- Implemented distraction duration monitoring using consecutive frame counting
- Added delayed distraction alerts to reduce false positives
- Integrated FPS calculation to monitor real-time processing performance
- Displayed live FPS and improved system robustness

### Technical Learnings:
- Learned temporal persistence for behavior analysis
- Understood why production systems delay alerts for transient events
- Explored FPS as a key performance metric in real-time computer vision

### Outcome:
- Improved Driver Monitoring System by distinguishing brief glances from sustained distraction
- Added performance monitoring to evaluate real-time processing efficiency

---

## Basic Iris-Based Gaze Estimation

### Completed:
- Added MediaPipe iris landmark processing
- Computed iris center using four iris landmarks
- Calculated gaze ratio relative to eye corners
- Implemented basic left/right/forward gaze estimation
- Displayed real-time gaze direction

### Findings:

* Observed that the computed gaze ratio remained around **2.x**, whereas the expected range is approximately **0.0–1.0**.
* Debugging revealed that the calculated iris center was outside the eye corner boundaries, indicating an incorrect landmark mapping rather than an issue with the mathematical formula.
* Verified that the geometric approach for gaze estimation is valid, but accurate landmark selection and coordinate validation are critical for reliable results.

### Outcome:

* Established the foundation for iris-based gaze estimation.
* Identified the need for a landmark visualization and verification step before finalizing the gaze estimation algorithm.
* Improved understanding of the challenges involved in robust gaze tracking for Driver Monitoring Systems (DMS).

---

## MediaPipe Landmark Visualization & Gaze Estimation Investigation

### Completed
- Created a landmark visualization debug utility.
- Verified that MediaPipe Face Mesh detects **478 landmarks** with `refine_landmarks=True`.
- Highlighted and validated the following key landmarks:
  - Nose tip (Landmark 1)
  - Left eye corners (33, 133)
  - Right eye corners (362, 263)
  - Right iris (469–472)
  - Left iris (474–477)
- Confirmed that all expected landmarks are correctly detected and tracked in real time.

### Findings
- The initial gaze ratio implementation consistently produced values outside the expected range (approximately 2.x instead of 0–1).
- Landmark visualization confirmed that the issue is **not** with MediaPipe landmark detection.

---

## Robust Eye Gaze Estimation

### Completed
- Calculated iris centers using MediaPipe iris landmarks for both eyes.
- Computed normalized gaze ratios independently for the left and right eyes.
- Combined both eye measurements by averaging their gaze ratios.
- Added real-time visualization of gaze ratio.
- Implemented a moving average filter (`deque`) to reduce frame-to-frame jitter in gaze estimation.

### Validation Results
Collected gaze ratio measurements under different eye movements:

| Eye Direction | Observed Range |
|---------------|----------------|
| Left | -0.01 to 0.18 |
| Forward | 0.49 to 0.54 |
| Right | 0.57 to 1.08 |

### Outcome
- Successfully implemented a robust iris-based gaze estimation module.
- Driver Monitoring System can now estimate:
  - Eye Aspect Ratio (EAR)
  - Blink detection
  - Drowsiness detection
  - Head pose estimation
  - Eye gaze estimation
  - Driver distraction

---
