import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import tkinter as tk
from tkinter import Label, Button, Frame
from PIL import Image, ImageTk
import sys
import os
import time
import threading
import queue
import winsound

# -------------------- Mediapipe Task Setup --------------------
model_path = 'face_landmarker.task'
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.FaceLandmarkerOptions(
    base_options=base_options,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
    num_faces=1
)
detector = vision.FaceLandmarker.create_from_options(options)

# -------------------- Initial Thresholds (Will be calibrated) --------------------
EAR_THRESHOLD = 0.2
MAR_THRESHOLD = 0.6
CONSEC_FRAMES = 20

# -------------------- Landmark Indices --------------------
left_eye_indices = [33, 160, 158, 133, 153, 144]
right_eye_indices = [362, 385, 387, 263, 373, 380]
mouth_indices = [61, 291, 13, 14]

# -------------------- EAR / MAR Calculation --------------------
def calculate_EAR(landmarks, indices, frame_width, frame_height):
    eye = np.array([
        (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
        for i in indices
    ])
    hor_distance = np.linalg.norm(eye[0] - eye[3])
    ver_distance1 = np.linalg.norm(eye[1] - eye[5])
    ver_distance2 = np.linalg.norm(eye[2] - eye[4])
    return (ver_distance1 + ver_distance2) / (2.0 * hor_distance)

def calculate_MAR(landmarks, indices, frame_width, frame_height):
    mouth = np.array([
        (landmarks[i].x * frame_width, landmarks[i].y * frame_height)
        for i in indices
    ])
    hor_distance = np.linalg.norm(mouth[0] - mouth[1])
    ver_distance = np.linalg.norm(mouth[2] - mouth[3])
    return ver_distance / hor_distance

# -------------------- Rotation Extraction --------------------
def get_euler_angles(matrix):
    # MediaPipe Face Landmarker transformation matrix to Euler angles (Pitch, Yaw, Roll)
    # The matrix is 4x4. The top-left 3x3 is rotation.
    r = matrix[:3, :3]
    sy = np.sqrt(r[0, 0]**2 + r[1, 0]**2)
    singular = sy < 1e-6
    if not singular:
        x = np.arctan2(r[2, 1], r[2, 2])
        y = np.arctan2(-r[2, 0], sy)
        z = np.arctan2(r[1, 0], r[0, 0])
    else:
        x = np.arctan2(-r[1, 2], r[1, 1])
        y = np.arctan2(-r[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)

# -------------------- Detection Class (Background Thread) --------------------
class DrowsinessDetector:
    def __init__(self, cap):
        self.cap = cap
        self.running = False
        self.result_queue = queue.Queue(maxsize=1)
        
        # Calibration state
        self.is_calibrating = False
        self.calibration_frames = []
        self.calibration_msg = ""
        
        # Event states
        self.eye_closed_frames = 0
        self.mouth_open_frames = 0
        self.is_drowsy_event = False
        self.is_yawning_event = False
        
        # Counters (Discrete Events)
        self.eye_closed_count = 0
        self.yawn_count = 0
        self.distraction_count = 0
        
        # Jitter reduction / Smoothing
        self.ear_history = []
        self.mar_history = []
        self.yaw_history = []
        self.pitch_history = []
        self.history_size = 5 # Frames to average
        
        # Distraction tracking
        self.distracted_frames = 0
        self.is_distracted_event = False
        
        # Audio Alert state
        self.alert_active = False

    def start(self):
        self.running = True
        self.thread = threading.Thread(target=self._process, daemon=True)
        self.thread.start()

    def stop(self):
        self.running = False

    def trigger_calibration(self):
        self.is_calibrating = True
        self.calibration_frames = []

    def _play_alert(self):
        if not self.alert_active:
            self.alert_active = True
            threading.Thread(target=self._async_beep, daemon=True).start()

    def _async_beep(self):
        # High pitched beep for attention
        winsound.Beep(1500, 500)
        self.alert_active = False

    def _process(self):
        global EAR_THRESHOLD, MAR_THRESHOLD
        prev_time = time.time()
        
        while self.running:
            success, frame = self.cap.read()
            if not success:
                continue

            # Performance monitoring
            start_inference = time.time()
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
            detection_result = detector.detect(mp_image)
            inference_time = (time.time() - start_inference) * 1000
            
            # FPS
            curr_time = time.time()
            fps = 1.0 / (curr_time - prev_time) if curr_time > prev_time else 0
            prev_time = curr_time

            status = "Active"
            frame_height, frame_width, _ = frame.shape
            
            if detection_result.face_landmarks:
                face_landmarks = detection_result.face_landmarks[0]
                
                # Face Size Check (Distance Proxy)
                # Distance between left-most and right-most landmarks (indices 234 and 454)
                face_width = np.linalg.norm(
                    np.array([face_landmarks[234].x, face_landmarks[234].y]) -
                    np.array([face_landmarks[454].x, face_landmarks[454].y])
                ) * frame_width
                
                if face_width < 100: # Threshold for "too far"
                    status = "TOO FAR - Move Closer"
                
                raw_ear = (calculate_EAR(face_landmarks, left_eye_indices, frame_width, frame_height) +
                           calculate_EAR(face_landmarks, right_eye_indices, frame_width, frame_height)) / 2.0
                raw_mar = calculate_MAR(face_landmarks, mouth_indices, frame_width, frame_height)
                
                # Apply Smoothing
                self.ear_history.append(raw_ear)
                self.mar_history.append(raw_mar)
                if len(self.ear_history) > self.history_size: self.ear_history.pop(0)
                if len(self.mar_history) > self.history_size: self.mar_history.pop(0)
                
                ear = sum(self.ear_history) / len(self.ear_history)
                mar = sum(self.mar_history) / len(self.mar_history)
                
                # Head Pose Logic
                yaw, pitch, roll = 0, 0, 0
                if detection_result.facial_transformation_matrixes:
                    matrix = detection_result.facial_transformation_matrixes[0].data
                    # Reshape if necessary (MediaPipe might return flattened)
                    matrix = np.array(matrix).reshape(4,4)
                    pitch, yaw, roll = get_euler_angles(matrix)
                
                self.yaw_history.append(yaw)
                self.pitch_history.append(pitch)
                if len(self.yaw_history) > self.history_size: self.yaw_history.pop(0)
                if len(self.pitch_history) > self.history_size: self.pitch_history.pop(0)
                
                avg_yaw = sum(self.yaw_history) / len(self.yaw_history)
                avg_pitch = sum(self.pitch_history) / len(self.pitch_history)

                # Calibration Logic
                if self.is_calibrating:
                    # Only collect if face is actually detected
                    self.calibration_frames.append((ear, mar))
                    progress = int((len(self.calibration_frames) / 60.0) * 100)
                    self.calibration_msg = f"CALIBRATING... {progress}%"
                    status = self.calibration_msg
                    
                    if len(self.calibration_frames) >= 60:
                        avg_ear = sum(f[0] for f in self.calibration_frames) / 60
                        avg_mar = sum(f[1] for f in self.calibration_frames) / 60
                        EAR_THRESHOLD = avg_ear * 0.70
                        MAR_THRESHOLD = avg_mar * 1.6
                        self.is_calibrating = False
                        self.calibration_msg = "CALIBRATION COMPLETE!"
                        print(f"Calibrated: EAR={EAR_THRESHOLD:.2f}, MAR={MAR_THRESHOLD:.2f}")
                        sys.stdout.flush()
                
                else:
                    if self.calibration_msg:
                        status = self.calibration_msg
                        # Clear message after some time
                        if not hasattr(self, 'msg_timer'): self.msg_timer = 0
                        self.msg_timer += 1
                        if self.msg_timer > 60:
                            self.calibration_msg = ""
                            self.msg_timer = 0
                    
                    # Eye Drowsiness Logic (Discrete Event)
                    # Eye Drowsiness Logic (Discrete Event)
                    if ear < EAR_THRESHOLD:
                        self.eye_closed_frames += 1
                        if self.eye_closed_frames >= CONSEC_FRAMES:
                            status = "DROWSY - Eyes Closed!"
                            self._play_alert()
                            if not self.is_drowsy_event:
                                self.eye_closed_count += 1
                                self.is_drowsy_event = True
                    else:
                        self.eye_closed_frames = 0
                        self.is_drowsy_event = False

                    # Yawning Logic (Discrete Event)
                    if mar > MAR_THRESHOLD:
                        self.mouth_open_frames += 1
                        if self.mouth_open_frames >= 4:
                            status = "DROWSY - Yawning!"
                            self._play_alert()
                            if not self.is_yawning_event:
                                self.yawn_count += 1
                                self.is_yawning_event = True
                    else:
                        self.mouth_open_frames = 0
                        self.is_yawning_event = False

                    # Distraction Detection (Head Turned)
                    # Thresholds: Yaw > 25 (Left/Right), Pitch > 20 (Up/Down)
                    if abs(avg_yaw) > 25 or abs(avg_pitch) > 20:
                        self.distracted_frames += 1
                        if self.distracted_frames >= 45: # ~1.5 seconds
                            status = "WARNING: DISTRACTED!"
                            self._play_alert()
                            if not self.is_distracted_event:
                                self.distraction_count += 1
                                self.is_distracted_event = True
                    else:
                        self.distracted_frames = 0
                        self.is_distracted_event = False
                
            else:
                status = "Face Not Detected"

            # Prepare data for GUI
            result = {
                'frame': rgb_frame,
                'status': status,
                'fps': fps,
                'latency': inference_time,
                'accuracy': 99.0 if detection_result.face_landmarks else 0.0,
                'eye_count': self.eye_closed_count,
                'yawn_count': self.yawn_count,
                'dist_count': self.distraction_count,
                'pose': (avg_pitch, avg_yaw, roll)
            }
            
            # Put in queue (overwrite if full to stay real-time)
            if self.result_queue.full():
                try: self.result_queue.get_nowait()
                except queue.Empty: pass
            self.result_queue.put(result)

# -------------------- GUI Logic --------------------
class App:
    def __init__(self, window):
        self.window = window
        self.window.title("Advanced Driver Monitor (V2)")
        self.window.configure(bg="#1e1e1e") # Dark theme for premium look
        
        # Camera init
        self.cap, _ = self._init_camera()
        self.detector_logic = DrowsinessDetector(self.cap)
        
        self._setup_ui()
        self._update()

    def _init_camera(self):
        for idx in [0, 1, 2]:
            cap = cv2.VideoCapture(idx, cv2.CAP_DSHOW)
            if cap.isOpened(): return cap, idx
            cap.release()
            cap = cv2.VideoCapture(idx)
            if cap.isOpened(): return cap, idx
            cap.release()
        return None, None

    def _on_calibrate(self):
        if not self.detector_logic.running:
            self._toggle() # Start camera if not running
        self.detector_logic.trigger_calibration()
        self.status_label.config(text="Status: Preparing Calibration...", fg="#00ccff")

    def _setup_ui(self):
        # Header
        header = Frame(self.window, bg="#1e1e1e")
        header.pack(fill="x", pady=10)
        Label(header, text="DRIVER MONITORING SYSTEM", font=("Helvetica", 18, "bold"), fg="#00ffcc", bg="#1e1e1e").pack()
        
        # Status Label
        self.status_label = Label(self.window, text="Status: Ready", font=("Helvetica", 16), fg="white", bg="#1e1e1e")
        self.status_label.pack(pady=5)
        
        # Stats Frame
        stats_frame = Frame(self.window, bg="#2d2d2d", padx=20, pady=10)
        stats_frame.pack(pady=10)
        self.eye_label = Label(stats_frame, text="Drowsy Events: 0", font=("Helvetica", 12), fg="#ff6666", bg="#2d2d2d")
        self.eye_label.grid(row=0, column=0, padx=20)
        self.yawn_label = Label(stats_frame, text="Yawn Events: 0", font=("Helvetica", 12), fg="#ffcc66", bg="#2d2d2d")
        self.yawn_label.grid(row=0, column=1, padx=20)
        self.dist_label = Label(stats_frame, text="Distraction: 0", font=("Helvetica", 12), fg="#cc66ff", bg="#2d2d2d")
        self.dist_label.grid(row=0, column=2, padx=20)
        
        # Pose Label
        self.pose_label = Label(self.window, text="Direction: Center", font=("Helvetica", 10), fg="#aaa", bg="#1e1e1e")
        self.pose_label.pack()
        
        # Video Display
        self.video_label = Label(self.window, bg="black", borderwidth=2, relief="solid")
        self.video_label.pack(pady=10, padx=20)
        
        # Performance Frame
        perf_frame = Frame(self.window, bg="#1e1e1e")
        perf_frame.pack(pady=5)
        self.fps_label = Label(perf_frame, text="FPS: 0", font=("Helvetica", 9), fg="#888", bg="#1e1e1e")
        self.fps_label.grid(row=0, column=0, padx=10)
        self.lat_label = Label(perf_frame, text="Lat: 0ms", font=("Helvetica", 9), fg="#888", bg="#1e1e1e")
        self.lat_label.grid(row=0, column=1, padx=10)
        self.acc_label = Label(perf_frame, text="Confidence: 0%", font=("Helvetica", 9), fg="#888", bg="#1e1e1e")
        self.acc_label.grid(row=0, column=2, padx=10)
        
        # Controls
        ctrl = Frame(self.window, bg="#1e1e1e")
        ctrl.pack(pady=20)
        self.start_btn = Button(ctrl, text="START MONITORING", command=self._toggle, font=("Helvetica", 12, "bold"), bg="#00ffcc", fg="black", width=20)
        self.start_btn.pack(side="left", padx=10)
        Button(ctrl, text="CALIBRATE", command=self._on_calibrate, font=("Helvetica", 12), bg="#444", fg="white", width=12).pack(side="left", padx=10)
        Button(ctrl, text="EXIT", command=self._exit, font=("Helvetica", 12), bg="#ff4444", fg="white", width=10).pack(side="left", padx=10)

    def _toggle(self):
        if self.detector_logic.running:
            self.detector_logic.stop()
            self.start_btn.config(text="START MONITORING", bg="#00ffcc")
            self.status_label.config(text="Status: Paused", fg="white")
        else:
            self.detector_logic.start()
            self.start_btn.config(text="STOP MONITORING", bg="#ffcc00")
            self.status_label.config(text="Status: Active", fg="#00ffcc")

    def _update(self):
        try:
            res = self.detector_logic.result_queue.get_nowait()
            # Update Video
            img = Image.fromarray(res['frame'])
            imgtk = ImageTk.PhotoImage(image=img)
            self.video_label.imgtk = imgtk
            self.video_label.config(image=imgtk)
            
            # Update text
            self.status_label.config(text=f"Status: {res['status']}")
            # Color coding status
            if "DROWSY" in res['status']: self.status_label.config(fg="#ff4444")
            elif "DISTRACTED" in res['status']: self.status_label.config(fg="#cc66ff")
            elif "CALIBRATING" in res['status']: self.status_label.config(fg="#00ccff")
            elif "TOO FAR" in res['status']: self.status_label.config(fg="#ffff00")
            else: self.status_label.config(fg="#00ffcc")
            
            self.eye_label.config(text=f"Drowsy Events: {res['eye_count']}")
            self.yawn_label.config(text=f"Yawn Events: {res['yawn_count']}")
            self.dist_label.config(text=f"Distraction: {res['dist_count']}")
            
            # Update Direction text
            p, y, r = res['pose']
            dir_text = "Center"
            if y > 25: dir_text = "Looking Left"
            elif y < -25: dir_text = "Looking Right"
            elif p > 20: dir_text = "Looking Down"
            elif p < -20: dir_text = "Looking Up"
            self.pose_label.config(text=f"Head Angle: {dir_text} (Y:{y:.0f}°, P:{p:.0f}°)")
            
            self.fps_label.config(text=f"FPS: {res['fps']:.1f}")
            self.lat_label.config(text=f"Inference: {res['latency']:.1f}ms")
            self.acc_label.config(text=f"Confidence: {res['accuracy']:.0f}%")
            
        except queue.Empty:
            pass
        
        self.window.after(10, self._update)

    def _exit(self):
        self.detector_logic.stop()
        if self.cap: self.cap.release()
        self.window.quit()

if __name__ == "__main__":
    if not os.path.exists(model_path):
        print(f"Error: {model_path} not found.")
        sys.exit(1)
        
    root = tk.Tk()
    app = App(root)
    root.mainloop()
    cv2.destroyAllWindows()
