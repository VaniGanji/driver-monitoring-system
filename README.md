# Driver Monitoring System

An Embedded AI Driver Monitoring System (DMS) prototype for real-time detection of driver drowsiness and distraction.

The project started as a Python reference implementation using OpenCV and MediaPipe Face Mesh, and is being ported to Modern C++ (C++17) using ONNX Runtime for neural network inference, as the basis for future embedded AI deployment.

## Features

### Python reference implementation (v1.0)
- Real-time face detection using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based drowsiness detection
- Blink detection and counting
- Head position estimation
- Eye gaze estimation
- Audio drowsiness alert
- CSV event logging
- Performance profiling (capture, inference and processing latency)

### C++ implementation (v2.0, in progress)
- Modular C++17 design built with CMake
- Face detection using the YuNet model
- 68-point facial landmark detection using the FAN2 model
- Model inference through ONNX Runtime on a live camera stream
- EAR-based blink and drowsiness detection
- Event logging and performance monitoring

## Technologies

- **Python:** OpenCV, MediaPipe Face Mesh, NumPy
- **C++:** C++17, CMake, OpenCV, ONNX Runtime
- **Models:** YuNet (face detection), FAN2 (68 facial landmarks)

## Project Status

Current version: **v2.0 (C++ port in progress)**

Upcoming roadmap:
- Head position and eye gaze estimation in C++
- Model optimization and quantization
- Embedded deployment
