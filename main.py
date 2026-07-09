import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import Label, Button, Frame, Scrollbar, Text
from PIL import Image, ImageTk
import sys
import os
import time
import threading
import queue
import csv
from datetime import datetime
import pyttsx3
import logging
import traceback

try:
    import simpleaudio as sa
except Exception:
    sa = None

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger("dri01")

# -------------------- Voice Engine --------------------
class VoiceEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.last_voice_time = 0
        self.lock = threading.Lock()
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def _worker(self):
        com_initialized = False
        engine = None
        try:
            if sys.platform.startswith("win"):
                try:
                    import pythoncom
                    pythoncom.CoInitialize()
                    com_initialized = True
                except Exception:
                    logger.warning("pythoncom not available; Windows COM init skipped for TTS worker.")

            engine = pyttsx3.init()
            voices = engine.getProperty('voices')
            if len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            engine.setProperty('rate', 160)

            while self.running:
                msg = self.queue.get()
                if msg is None:
                    break
                try:
                    engine.say(msg)
                    engine.runAndWait()
                except Exception:
                    logger.exception("TTS playback failed.")
        except Exception:
            logger.exception("Voice engine worker initialization failed.")
        finally:
            try:
                if engine is not None:
                    engine.stop()
            except Exception:
                pass
            if com_initialized:
                try:
                    import pythoncom
                    pythoncom.CoUninitialize()
                except Exception:
                    logger.exception("Failed to uninitialize COM in TTS worker.")

    def say(self, text, cooldown=3):
        """Speaks the text only if the cooldown has passed."""
        now = time.time()
        with self.lock:
            if now - self.last_voice_time > cooldown:
                self.queue.put(text)
                self.last_voice_time = now

    def close(self):
        self.running = False
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

class AudioAlertEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def play_tone(self, freq_hz=1500, duration_ms=150, volume=0.25):
        self.queue.put((freq_hz, duration_ms, volume))

    def _generate_tone(self, freq_hz, duration_ms, volume):
        sample_rate = 44100
        samples = int(sample_rate * (duration_ms / 1000.0))
        t = np.linspace(0, duration_ms / 1000.0, samples, False)
        wave = np.sin(freq_hz * t * 2 * np.pi)
        audio = (wave * (32767 * volume)).astype(np.int16)
        return audio.tobytes(), sample_rate

    def _worker(self):
        while self.running:
            item = self.queue.get()
            if item is None:
                break
            freq_hz, duration_ms, volume = item
            try:
                if sa is None:
                    continue
                audio_bytes, sample_rate = self._generate_tone(freq_hz, duration_ms, volume)
                play_obj = sa.play_buffer(audio_bytes, 1, 2, sample_rate)
                play_obj.wait_done()
            except Exception:
                logger.exception("Audio alert playback failed.")

    def close(self):
        self.running = False
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)

# -------------------- Mediapipe Task Setup --------------------
MODEL_PATH = 'face_landmarker.task'
LOG_FILE = 'driver_session_log.csv'

# -------------------- Landmark Indices --------------------
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]

# -------------------- Adaptive Threshold Manager --------------------
class AdaptiveThresholdManager:
    """Continuously learns from the environment to adjust EAR/MAR thresholds."""
    def __init__(self):
        self.ear_baseline_history = []
        self.mar_baseline_history = []
        self.window_size = 300  # ~10 seconds at 30fps
        self.current_ear_threshold = 0.2
        self.current_mar_threshold = 0.6
        self.is_ready = False

    def update(self, raw_ear, raw_mar, is_relaxed=True):
        if is_relaxed:
            self.ear_baseline_history.append(raw_ear)
            self.mar_baseline_history.append(raw_mar)
            
            if len(self.ear_baseline_history) > self.window_size:
                self.ear_baseline_history.pop(0)
                self.mar_baseline_history.pop(0)
                self.is_ready = True

            if self.is_ready:
                # Set threshold to 70% of the rolling average for eyes
                avg_ear = sum(self.ear_baseline_history) / len(self.ear_baseline_history)
                # Set threshold to 160% of the rolling average for mouth
                avg_mar = sum(self.mar_baseline_history) / len(self.mar_baseline_history)
                
                self.current_ear_threshold = avg_ear * 0.72
                self.current_mar_threshold = avg_mar * 1.65

# -------------------- Session Logger --------------------
class SessionLogger:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['Timestamp', 'Event Type', 'Duration (s)', 'Details'])

    def log_event(self, event_type, duration=0, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, duration, details])

# -------------------- Core Agent Engine --------------------
class DriverAgent:
    def __init__(self, cap):
        self.cap = cap
        self.running = False
        self.result_queue = queue.Queue(maxsize=1)
        self.logger = SessionLogger(LOG_FILE)
        self.adapter = AdaptiveThresholdManager()
        self.voice = VoiceEngine()
        self.audio = AudioAlertEngine()
        self.thread_lock = threading.Lock()
        self.error_state = None
        self.thread = None
        self._base_monotonic_ms = int(time.monotonic() * 1000)
        self._last_timestamp_ms = self._base_monotonic_ms
        
        # Detector Setup
        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        # State Tracking
        self.eye_closed_start = None
        self.mouth_open_start = None
        self.distracted_start = None
        self.face_missing_start = None
        
        # Performance/Stats
        self.drowsy_count = 0
        self.yawn_count = 0
        self.distraction_count = 0
        self.alertness_score = 100.0  # PERCLOS based score
        self.closure_history = []  # Binary (1 for closed, 0 for open)
        
        # For smoothing UI numbers
        self.smooth_ear = 0.3
        self.smooth_mar = 0.1

    def start(self):
        with self.thread_lock:
            if self.thread is not None and self.thread.is_alive():
                return
            self.running = True
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()

    def stop(self):
        self.running = False
        with self.thread_lock:
            if self.thread is not None and self.thread.is_alive():
                self.thread.join(timeout=2.0)

    def close(self):
        self.stop()
        try:
            self.detector.close()
        except Exception:
            logger.exception("Failed to close detector.")
        self.voice.close()
        self.audio.close()

    def is_alive(self):
        with self.thread_lock:
            return self.thread is not None and self.thread.is_alive()

    def restart(self):
        self.set_error_state("Detector thread restarted after failure.")
        self.stop()
        self.start()

    def set_error_state(self, message):
        self.error_state = message

    def get_error_state(self):
        return self.error_state

    def _next_timestamp_ms(self):
        current = int(time.monotonic() * 1000)
        if current <= self._last_timestamp_ms:
            current = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current
        return current

    def _calculate_metrics(self, landmarks, w, h):
        # EAR calculation
        l_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE])
        r_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE])
        
        def eye_ear(eye):
            v1 = np.linalg.norm(eye[1] - eye[5])
            v2 = np.linalg.norm(eye[2] - eye[4])
            hor = np.linalg.norm(eye[0] - eye[3])
            return (v1 + v2) / (2.0 * hor)
        
        ear = (eye_ear(l_eye) + eye_ear(r_eye)) / 2.0
        
        # MAR calculation
        mouth = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH])
        m_hor = np.linalg.norm(mouth[0] - mouth[1])
        m_ver = np.linalg.norm(mouth[2] - mouth[3])
        mar = m_ver / m_hor
        
        return ear, mar

    def _get_pose(self, matrix):
        m = np.array(matrix).reshape(4,4)[:3, :3]
        sy = np.sqrt(m[0, 0]**2 + m[1, 0]**2)
        if sy > 1e-6:
            x = np.arctan2(m[2, 1], m[2, 2])
            y = np.arctan2(-m[2, 0], sy)
            z = np.arctan2(m[1, 0], m[0, 0])
        else:
            x = np.arctan2(-m[1, 2], m[1, 1])
            y = np.arctan2(-m[2, 0], sy)
            z = 0
        return np.degrees(x), np.degrees(y), np.degrees(z)

    def _draw_overlay(self, frame, landmarks, status, pose):
        h, w, _ = frame.shape
        # Draw tech-style corners
        color = (0, 255, 204) if "Active" in status else (0, 100, 255)
        if "DROWSY" in status or "WARNING" in status: color = (0, 0, 255)

        # Draw a subtle face contour
        for i in range(len(landmarks)):
            px = int(landmarks[i].x * w)
            py = int(landmarks[i].y * h)
            if i % 15 == 0: # Only draw some points for "pro" look
                cv2.circle(frame, (px, py), 1, color, -1)

        # --- 3D Gaze Projection ---
        # We project a line from the nose tip (idx 1) based on head pose
        pitch, yaw, _ = pose
        nose = landmarks[1]
        cx, cy = int(nose.x * w), int(nose.y * h)
        
        # Calculate endpoint based on yaw/pitch
        length = 100
        # Yaw: Left/Right (X), Pitch: Up/Down (Y)
        ex = int(cx + length * np.sin(np.radians(-yaw)))
        ey = int(cy + length * np.sin(np.radians(pitch)))
        
        cv2.line(frame, (cx, cy), (ex, ey), color, 2)
        cv2.circle(frame, (ex, ey), 4, color, -1) # "Target" dot
        
        return frame

    def _apply_night_vision(self, frame):
        # Convert to LAB for better contrast adjustment
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        # Calculate brightness
        avg_brightness = np.mean(l)
        if avg_brightness < 80: # If dark
            clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8,8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl,a,b))
            return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR), True
        return frame, False

    def _take_snapshot(self, frame, event_name):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"crisis_logs/CRISIS_{event_name}_{timestamp}.jpg"
        cv2.imwrite(filename, cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
        self.logger.log_event("SNAPSHOT_SAVED", 0, filename)

    def _run(self):
        prev_time = time.time()
        while self.running:
            try:
                success, raw_frame = self.cap.read()
                if not success:
                    continue

                # Night Vision Enhancement
                frame, night_mode = self._apply_night_vision(raw_frame)

                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                res = self.detector.detect_for_video(mp_img, self._next_timestamp_ms())

                status = "Continuous Learning..."
                h, w, _ = frame.shape
                pose = (0, 0, 0)

                if res.face_landmarks:
                    self.face_missing_start = None
                    marks = res.face_landmarks[0]
                    ear, mar = self._calculate_metrics(marks, w, h)

                    # Update adaptive thresholds
                    # Only update baseline if the user is NOT currently in an event
                    self.adapter.update(ear, mar, is_relaxed=(self.eye_closed_start == None))

                    # Smoothing for UI
                    self.smooth_ear = self.smooth_ear * 0.8 + ear * 0.2
                    self.smooth_mar = self.smooth_mar * 0.8 + mar * 0.2

                    # Head Pose
                    if res.facial_transformation_matrixes:
                        pose = self._get_pose(res.facial_transformation_matrixes[0].data)
                    pitch, yaw, _ = pose

                    # --- DETECTION LOGIC ---
                    curr_status = "Active"

                    # 1. Drowsiness (EAR)
                    if ear < self.adapter.current_ear_threshold:
                        if self.eye_closed_start is None: self.eye_closed_start = time.time()
                        duration = time.time() - self.eye_closed_start
                        if duration > 1.0: # 1 second threshold
                            curr_status = "DROWSY - WAKE UP!"
                            self.audio.play_tone(2000, 200)
                            self.voice.say("Drowsiness detected. Open your eyes.")
                            self.closure_history.append(1)
                            if duration > 2.5 and not hasattr(self, '_snap_done'):
                                self._take_snapshot(img_rgb, "DROWSY")
                                self._snap_done = True
                        else: self.closure_history.append(0)
                    else:
                        if hasattr(self, '_snap_done'): delattr(self, '_snap_done')
                        if self.eye_closed_start:
                            dur = time.time() - self.eye_closed_start
                            if dur > 1.0:
                                self.drowsy_count += 1
                                self.logger.log_event("DROWSY", round(dur, 2), f"EAR: {ear:.2f}")
                                if self.drowsy_count % 3 == 0:
                                    self.voice.say("You have multiple drowsy events. Please consider a coffee break.")
                        self.eye_closed_start = None
                        self.closure_history.append(0)

                    # 2. Yawning (MAR)
                    if mar > self.adapter.current_mar_threshold:
                        if self.mouth_open_start is None: self.mouth_open_start = time.time()
                        if (time.time() - self.mouth_open_start) > 2.0:
                            curr_status = "YAWNING DETECTED"
                            self.audio.play_tone(1000, 100)
                            self.voice.say("Yawning detected. Fatigue is increasing.")
                    else:
                        if self.mouth_open_start:
                            dur = time.time() - self.mouth_open_start
                            if dur > 2.0:
                                self.yawn_count += 1
                                self.logger.log_event("YAWN", round(dur, 2))
                        self.mouth_open_start = None

                    # 3. Distraction (Pose)
                    if abs(yaw) > 25 or abs(pitch) > 18:
                        if self.distracted_start is None: self.distracted_start = time.time()
                        if (time.time() - self.distracted_start) > 1.5:
                            curr_status = "WATCH THE ROAD!"
                            self.audio.play_tone(1500, 150)
                            self.voice.say("Keep your eyes on the road.")
                    else:
                        if self.distracted_start:
                            dur = time.time() - self.distracted_start
                            if dur > 1.5:
                                self.distraction_count += 1
                                self.logger.log_event("DISTRACTION", round(dur, 2), f"Yaw: {yaw:.0f}")
                        self.distracted_start = None

                    status = curr_status
                    frame_draw = self._draw_overlay(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), marks, status, pose)
                    frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)

                    # Calculate Alertness Score (PERCLOS)
                    if len(self.closure_history) > 600: self.closure_history.pop(0)
                    if len(self.closure_history) > 0:
                        perclos = (sum(self.closure_history) / len(self.closure_history)) * 100
                        self.alertness_score = max(0, 100 - (perclos * 5)) # Weighted reduction

                else:
                    if self.face_missing_start is None:
                        self.face_missing_start = time.time()

                    missing_dur = time.time() - self.face_missing_start
                    if missing_dur > 1.5:
                        status = "FACE OBSTRUCTED / MISSING"
                        self.audio.play_tone(2500, 200)
                        self.voice.say("Face obstructed. Please keep your face visible.")
                    else:
                        status = "Searching for face..."

                    self.closure_history.append(0)
                    if len(self.closure_history) > 600: self.closure_history.pop(0)

                # Package result
                inf_time = (time.time() - prev_time)
                fps = 1.0 / inf_time if inf_time > 0 else 0
                prev_time = time.time()

                res_pkg = {
                    'frame': frame,
                    'status': status,
                    'fps': fps,
                    'ear': self.smooth_ear,
                    'mar': self.smooth_mar,
                    'score': self.alertness_score,
                    'counts': (self.drowsy_count, self.yawn_count, self.distraction_count),
                    'pose': pose,
                    'thresholds': (self.adapter.current_ear_threshold, self.adapter.current_mar_threshold),
                    'night': night_mode
                }

                if self.result_queue.full():
                    try: self.result_queue.get_nowait()
                    except: pass
                self.result_queue.put(res_pkg)
            except Exception:
                self.set_error_state("Detection loop recovered from an internal error.")
                logger.exception("Detector loop error:\n%s", traceback.format_exc())
                time.sleep(0.05)

# -------------------- GUI --------------------
class DRI01App:
    def __init__(self, root):
        self.root = root
        self.root.title("AGENT DRI01 - Driver Intelligence Agent")
        self.root.geometry("1100x700")
        self.root.configure(bg="#0a0a0a")
        
        self.cap = self._init_cap()
        if self.cap:
            self.agent = DriverAgent(self.cap)
        else:
            self.agent = None
            print("CRITICAL: All camera initialization attempts failed.")
        
        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _init_cap(self):
        # Fallback sequence for Windows camera backends
        attempts = [
            (0, cv2.CAP_DSHOW),
            (0, cv2.CAP_MSMF),
            (0, None),
            (1, cv2.CAP_DSHOW),
            (1, None)
        ]
        
        for idx, backend in attempts:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    # Verification read
                    ret, _ = cap.read()
                    if ret:
                        print(f"Camera initialized successfully on index {idx} with backend {backend}")
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                    cap.release()
            except Exception as e:
                print(f"Failed to init camera {idx} with {backend}: {e}")
                continue
        return None

    def _build_ui(self):
        # Top bar
        top = Frame(self.root, bg="#0a0a0a", height=60)
        top.pack(fill="x", side="top", pady=10)
        Label(top, text="AGENT DRI01", font=("Consolas", 24, "bold"), fg="#00ffcc", bg="#0a0a0a").pack(side="left", padx=20)
        self.time_lbl = Label(top, text="", font=("Consolas", 12), fg="#888", bg="#0a0a0a")
        self.time_lbl.pack(side="right", padx=20)

        # Main Layout
        main_frame = Frame(self.root, bg="#0a0a0a")
        main_frame.pack(fill="both", expand=True, padx=10)
        
        # Left: Video
        self.vid_lbl = Label(main_frame, text="SYSTEM INITIALIZED\n\nCLICK 'ACTIVATE AGENT' TO START MONITORING", 
                             fg="#00ffcc", bg="black", font=("Consolas", 14), borderwidth=0)
        self.vid_lbl.pack(side="left", fill="both", expand=True)
        
        # Right: Dashboard
        dash = Frame(main_frame, bg="#121212", width=300)
        dash.pack(side="right", fill="y", padx=10)
        dash.pack_propagate(False)

        # Alertness Score
        Label(dash, text="DRIVER ALERTNESS", font=("Consolas", 10), fg="#aaa", bg="#121212").pack(pady=(20,0))
        self.score_lbl = Label(dash, text="100%", font=("Consolas", 48, "bold"), fg="#00ffcc", bg="#121212")
        self.score_lbl.pack()
        
        # Stats List
        self.stat_frame = Frame(dash, bg="#121212")
        self.stat_frame.pack(fill="x", pady=20, padx=10)
        
        self._add_stat("DROWSY EVENTS", "0", "#ff4444", 0)
        self._add_stat("YAWN EVENTS", "0", "#ffaa00", 1)
        self._add_stat("DISTRACTIONS", "0", "#aa44ff", 2)

        # Threshold Learning Info
        self.learn_lbl = Label(dash, text="Agent Level: Learning...", font=("Consolas", 9), fg="#666", bg="#121212", wraplength=250)
        self.learn_lbl.pack(side="bottom", pady=20)

        # Telemetry Labels
        self.tel_lbl = Label(dash, text="EAR: 0.00 | MAR: 0.00", font=("Consolas", 9), fg="#888", bg="#121212")
        self.tel_lbl.pack(side="bottom")

        # Control Bar
        bot = Frame(self.root, bg="#0a0a0a", height=80)
        bot.pack(fill="x", side="bottom")
        
        self.btn_run = Button(bot, text="ACTIVATE AGENT", command=self._toggle, font=("Consolas", 12, "bold"), bg="#00ffcc", fg="#000", width=20, relief="flat")
        self.btn_run.pack(side="left", padx=20, pady=20)
        
        Button(bot, text="RESET LOGS", command=self._reset_logs, font=("Consolas", 10), bg="#333", fg="#fff", width=12, relief="flat").pack(side="right", padx=20)

    def _add_stat(self, title, val, color, row):
        f = Frame(self.stat_frame, bg="#1a1a1a", pady=10)
        f.pack(fill="x", pady=5)
        Label(f, text=title, font=("Consolas", 8), fg="#999", bg="#1a1a1a").pack(side="left", padx=10)
        lbl = Label(f, text=val, font=("Consolas", 14, "bold"), fg=color, bg="#1a1a1a")
        lbl.pack(side="right", padx=10)
        if "DROWSY" in title: self.d_count_lbl = lbl
        elif "YAWN" in title: self.y_count_lbl = lbl
        else: self.dist_count_lbl = lbl

    def _toggle(self):
        if not self.agent: return
        if self.agent.running:
            self.agent.stop()
            self.btn_run.config(text="ACTIVATE AGENT", bg="#00ffcc")
        else:
            self.agent.start()
            self.btn_run.config(text="DEACTIVATE", bg="#ff4444")

    def _reset_logs(self):
        if os.path.exists(LOG_FILE): os.remove(LOG_FILE)
        self.agent.drowsy_count = 0
        self.agent.yawn_count = 0
        self.agent.distraction_count = 0

    def _tick(self):
        if not self.agent:
            self.vid_lbl.config(text="CAMERA ERROR: COULD NOT INITIALIZE DEVICE\nPlease check connection or privacy settings.", fg="red", font=("Consolas", 14))
            return

        if self.agent.running and not self.agent.is_alive():
            self.agent.restart()

        runtime_error = self.agent.get_error_state()

        try:
            p = self.agent.result_queue.get_nowait()
            
            # Update Video
            img = Image.fromarray(p['frame'])
            # Resize to fit UI while keeping aspect ratio
            img.thumbnail((750, 500))
            imgtk = ImageTk.PhotoImage(image=img)
            self.vid_lbl.imgtk = imgtk
            self.vid_lbl.config(image=imgtk)
            
            # Update Stats
            self.score_lbl.config(text=f"{p['score']:.0f}%")
            if p['score'] < 70: self.score_lbl.config(fg="#ffaa00")
            elif p['score'] < 40: self.score_lbl.config(fg="#ff4444")
            else: self.score_lbl.config(fg="#00ffcc")
            
            dc, yc, dic = p['counts']
            self.d_count_lbl.config(text=str(dc))
            self.y_count_lbl.config(text=str(yc))
            self.dist_count_lbl.config(text=str(dic))
            
            ea, ma = p['ear'], p['mar']
            te, tm = p['thresholds']
            self.tel_lbl.config(text=f"EAR: {ea:.3f} (Lim:{te:.2f}) | MAR: {ma:.3f} (Lim:{tm:.2f})")
            
            self.time_lbl.config(text=f"FPS: {p['fps']:.1f} | SENSOR: {p['status']}")
            
            night_text = " [NIGHT MODE ACTIVE]" if p['night'] else ""
            learn_text = f"Status: Learning Baseline...{night_text}" if not self.agent.adapter.is_ready else f"Status: Environmental Sync Active.{night_text}\nLogging to {LOG_FILE}"
            if runtime_error:
                learn_text = f"Status: RECOVERING - {runtime_error}\n{learn_text}"
            self.learn_lbl.config(text=learn_text)

        except queue.Empty:
            pass
        
        self.root.after(10, self._tick)

    def _on_close(self):
        if self.agent:
            self.agent.close()
        if self.cap:
            self.cap.release()
        self.root.destroy()

if __name__ == "__main__":
    if not os.path.exists(MODEL_PATH):
        print(f"CRITICAL: {MODEL_PATH} missing.")
        sys.exit(1)
        
    root = tk.Tk()
    app = DRI01App(root)
    root.mainloop()
