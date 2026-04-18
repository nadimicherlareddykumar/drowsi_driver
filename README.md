# Advanced Driver Monitoring System (DMS)

A real-time driver safety application that monitors drowsiness, yawning, and distraction using Mediapipe's modern Tasks API and OpenCV.

## ✨ Features
- **Drowsiness Detection**: Monitors Eye Aspect Ratio (EAR) to detect microsleeps.
- **Yawn Detection**: Monitors Mouth Aspect Ratio (MAR) to detect fatigue.
- **Distraction Detection**: Tracks 3D head pose (Yaw and Pitch) to detect when the driver is looking away from the road or at a phone.
- **Personalized Calibration**: One-click calibration to adapt to the driver's unique facial features.
- **Audio Alerts**: Provides high-frequency auditory warnings when danger is detected.
- **Real-time Performance**: Multithreaded architecture for smooth 30+ FPS performance.
- **Session Stats**: Tracks discrete events of drowsiness, yawning, and distraction.

## 🛠️ Technology Stack
- **Python 3.10+**
- **MediaPipe Tasks API** (Face Landmarker)
- **OpenCV** (Camera handling and processing)
- **Tkinter** (GUI)
- **Pillow** (Image handling)

## 🚀 Getting Started

### 1. Installation
Clone the repository and install the dependencies:
```bash
pip install mediapipe opencv-python Pillow numpy
```

### 2. Model Download
The system requires the `face_landmarker.task` model. If not present, download it from Google's MediaPipe storage:
[Download face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

### 3. Usage
Run the application:
```bash
python main.py
```

- Click **START MONITORING** to begin the camera feed.
- Click **CALIBRATE** and look straight at the camera for 3 seconds for optimal accuracy.
- Watch the **Status** bar and listen for alerts!

## 🛡️ License
MIT License
