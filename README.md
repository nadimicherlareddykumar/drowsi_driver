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
Clone the repository and install dependencies using either:

```bash
python setup_env.py
```

or manually:

```bash
pip install -r requirements.txt
```

#### OS-specific notes
- **Windows**: `pywin32` is installed automatically for TTS COM support.
- **Linux**: install `espeak` if pyttsx3 speech backend is missing (example: `sudo apt-get install espeak`).
- **macOS**: pyttsx3 uses built-in system voices in most setups.

### 2. Model Download
The system requires the `face_landmarker.task` model. If not present, download it from Google's MediaPipe storage:
[Download face_landmarker.task](https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task)

### 3. Usage
Run the application:
```bash
python main.py
```

- Click **CALIBRATE** and look straight at the camera for 5 seconds to build baseline data (saved to `baseline_profile.json`).
- Click **ACTIVATE AGENT** to begin active monitoring after calibration.
- Use **DEACTIVATE** to pause and **RESET LOGS** to reset counters/log file.
- Watch the **Status** bar and listen for alerts!

## ✅ Supported Platforms
- Windows 10/11
- Linux (X11/Wayland environments with camera/audio permissions)
- macOS

## 🔒 Privacy, Consent, and Retention
- Crisis snapshots are stored **locally only** in `crisis_logs/` for safety event review.
- The app auto-deletes snapshots older than **7 days**.
- Before use, inform the driver that camera frames are processed for drowsiness/yawn/distraction detection and that event snapshots may be saved locally for safety evidence.
- Session CSV logs and snapshots are excluded from git tracking by `.gitignore`.

### Repository hygiene for previously committed logs
This repository now removes tracked runtime logs/snapshots from the current tree.  
To fully purge historical sensitive artifacts from existing git history in a maintained repo, run one of:
- `git filter-repo --path driver_session_log.csv --path-glob 'crisis_logs/*.jpg' --invert-paths`
- or BFG Repo-Cleaner with equivalent path rules

## 🛡️ License
MIT License
