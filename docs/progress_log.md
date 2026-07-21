# 📅 Project Progress Log – Driver Monitoring System

---

## Project Initialization

### Completed:
- Repository setup
- Development environment
- OpenCV camera pipeline
- Initial Haar Cascade prototype: Face detection and Eye detection

### Observations:
- Haar Cascade-based detection works but shows limitations in real-world scenarios:
  - False positives under head movement and direction variation
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
- Achieved stable detection under head tilt and direction variation

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

---

## Drowsiness Detection Logic

### Completed:
- Implemented EAR threshold-based eye closure detection
- Added consecutive frame monitoring for temporal drowsiness analysis
- Integrated real-time drowsiness alert system
- Displayed visual alert during prolonged eye closure

### Outcome:
- Successfully implemented core drowsiness detection functionality for DMS

---

## Audio Alert

### Completed:
- Integrated audio alert system for drowsiness warning

---

## Blinks count for Fatigue monitoring

### Completed:
- Implemented Blinks count per minute as secondary analytics for drowsiness
- Displayed Blink count on video stream

---

## Head direction Estimation and Attention Monitoring

### Completed:
- Implemented basic head direction estimation using facial landmarks
- Added driver attention monitoring functionality
- Calculated head orientation using nose and eye alignment
- Displayed real-time head direction status
- Added distraction warning capability

### Outcome:
- Enhanced Driver Monitoring System with distraction detection capability
- Added foundational attention-awareness monitoring

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

---

## Driver Attention State Fusion

### Completed
- Implemented driver attention state fusion by combining:
  - Head direction estimation
  - Eye gaze estimation
- Developed rule-based decision logic to classify the driver's attention state based on head orientation and eye gaze.
- Implemented a distraction frame counter to track sustained inattentive behavior.
- Triggered a visual distraction alert after the configured threshold was exceeded.
- Tuned the head direction threshold by increasing the head offset from the initial value to **30 pixels**, significantly reducing false left/right detections caused by minor natural head movements.

### Outcome
- Successfully transformed independent perception modules into a unified Driver Attention Monitoring pipeline.

---

##  Robust Face Tracking & Failure Handling

## Completed 
- Added handling for the **No Face Detected** scenario.
- Prevented stale head direction, eye gaze, and attention information from being displayed.
- Eye Monitor Reset
- Intentionally kept a short alarm clip instead of implementing alarm interruption logic to keep the project simple and focused on computer vision.

---

# Driver Event Logging

## Objective
Implemented an event-driven logging system to record significant Driver Monitoring System events with timestamps. The logging mechanism captures only meaningful state transitions, avoiding duplicate entries for every video frame.

## Completed Tasks
- Created a dedicated `event_logger.py` module.
- Separated logging functionality from perception and decision-making modules.
- Logged events in CSV format for easy analysis using Excel or Python.
- Implemented logging only when system states changed, preventing duplicate log entries during continuous video processing.

## Sample Output
## csv
Timestamp,Event
2026-07-15 21:54:20,Face Detected
2026-07-15 21:54:20,AttentionState.ATTENTIVE
2026-07-15 21:54:34,Drowsiness Alert
2026-07-15 21:54:38,Drowsiness Cleared
2026-07-15 21:54:49,AttentionState.HEAD_TURNED
2026-07-15 21:54:49,Face Lost
2026-07-15 21:54:53,Face Detected
2026-07-15 21:54:53,AttentionState.ATTENTIVE
2026-07-15 21:55:00,AttentionState.HEAD_TURNED
2026-07-15 21:55:06,AttentionState.LOOKING_RIGHT
2026-07-15 21:55:06,AttentionState.ATTENTIVE

---

# Performance Profiling & Pipeline Benchmarking

## Objective
Profiled the Driver Monitoring System pipeline to measure execution time of each processing stage, identify performance bottlenecks, and understand real-time system characteristics. The focus was on measuring system behavior rather than optimizing algorithms.

## Completed Tasks
- Measured execution time for the following stages:
  - Frame Capture
  - MediaPipe Face Mesh Inference
  - Eye Monitoring
  - Attention Monitoring
- Displayed execution statistics once every second to avoid excessive console output.
- Reported execution time of each pipeline stage along with total pipeline latency.

Example output:
--------------------------------
Capture          : 32.66 ms
FaceMesh         : 6.93 ms
EyeMonitor       : 0.06 ms
AttentionMonitor : 0.07 ms
Total            : 39.72 ms
--------------------------------

## Performance Observations
- Frame Capture - Average execution time: **20–40 ms**
- MediaPipe Face Mesh - Typical inference time: **6–7 ms**
- Eye Monitoring - Average processing time: **0.05–0.25 ms**
- Attention Monitoring - Average processing time: **0.05–0.25 ms**

### Total Pipeline Latency 
- Typical latency: **35–45 ms per frame**
### Throughput
- Typical pipeline latency corresponds to approximately: (1000/~40ms latency)  **22–30 FPS**
### Bottleneck Analysis
- Identified camera frame acquisition (`cv2.VideoCapture.read()`) as the primary contributor to overall pipeline latency.
- Observed that perception algorithms (EAR, gaze estimation, attention classification) require only a fraction of the total processing time.

## Project Status
The Driver Monitoring System now includes:
- ✅ Face Detection & Face Mesh
- ✅ Eye Aspect Ratio (EAR)
- ✅ Blink Detection
- ✅ Drowsiness Detection
- ✅ Head Direction Estimation
- ✅ Eye Gaze Estimation
- ✅ Attention State Classification
- ✅ Distraction Duration Monitoring
- ✅ Audio Alert
- ✅ Robust Face Tracking & Recovery
- ✅ Event Logging
- ✅ Performance Profiling & Pipeline Benchmarking

---

# 🚀 Python v1.0 Release Summary

## Project Overview

A real-time Driver Monitoring System (DMS) prototype developed in Python using OpenCV and MediaPipe Face Mesh. The system monitors driver drowsiness and distraction by analyzing facial landmarks, eye movements, head direction, and blink behavior. The project follows a modular architecture and serves as the reference implementation for future migration to C++ and embedded AI deployment.

## Implemented Features

✓ Real-time camera capture
✓ Face detection using MediaPipe Face Mesh
✓ Eye Aspect Ratio (EAR) calculation
✓ Blink detection and counting
✓ Drowsiness detection with configurable thresholds
✓ Audio alert for prolonged eye closure
✓ Head direction estimation
✓ Eye gaze estimation with smoothing
✓ Driver attention monitoring
✓ Event logging to CSV
✓ Performance profiling and latency measurement

## Software Architecture

Modules implemented:

- main.py
- eye_monitor.py
- attention_monitor.py
- performance_monitor.py
- event_logger.py
- utils.py
- config.py

The project follows a modular design where each module is responsible for a single functional area, improving readability, maintainability, and future extensibility.

## Performance Observations

Performance profiling shows:

- Camera frame capture is the largest contributor to overall latency.
- MediaPipe Face Mesh inference is the second most time-consuming stage.
- EAR calculation, blink detection, head direction estimation, and gaze estimation introduce minimal processing overhead.
- Overall latency is suitable for real-time driver monitoring on a standard laptop.

## Engineering Highlights

Throughout development, the project evolved from a simple face detection prototype into a complete driver monitoring application by:

- Migrating from Haar Cascade detection to MediaPipe Face Mesh.
- Implementing robust blink detection using Eye Aspect Ratio (EAR).
- Developing head direction and eye gaze estimation using facial landmarks.
- Adding event logging for offline analysis.
- Measuring execution time of each processing stage to understand real-time system performance.
- Refactoring the codebase into modular components to improve maintainability.

## Technology Stack

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy
- CSV Logging

## Next Phase

The Python implementation is considered the validated reference version of the project.

The next development phase focuses on:

- Porting the application module-by-module to Modern C++.
- Verifying functional equivalence between Python and C++ implementations.
- Preserving the validated algorithms while adopting production-oriented C++ design.
- Preparing the project for TensorFlow Lite / ONNX Runtime integration and embedded deployment.