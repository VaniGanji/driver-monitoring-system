# Driver Monitoring System

An Embedded AI Driver Monitoring System (DMS) prototype developed in Python using OpenCV and MediaPipe Face Mesh.

The project performs real-time driver monitoring by detecting drowsiness and distraction through computer vision techniques. It serves as the reference implementation before migration to a production-oriented C++ version and future embedded AI deployment.

## Features

- Real-time face detection using MediaPipe Face Mesh
- Eye Aspect Ratio (EAR) based drowsiness detection
- Blink detection and counting
- Head position estimation
- Eye gaze estimation
- Audio drowsiness alert
- CSV event logging
- Performance profiling (capture, inference and processing latency)

## Technologies

- Python
- OpenCV
- MediaPipe Face Mesh
- NumPy

## Project Status

Current Version: **v1.0 (Python Reference Implementation)**

Upcoming roadmap:

- Modern C++ implementation
- TensorFlow Lite integration
- ONNX Runtime integration
- Model optimization and quantization
- Embedded deployment
