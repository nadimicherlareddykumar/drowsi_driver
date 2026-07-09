import json
import logging
import os
import queue
import threading
import time
import traceback
from datetime import datetime

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

from dms.config import BASELINE_FILE, LOG_FILE, MODEL_PATH
from dms.detection.math_utils import apply_night_vision, calculate_metrics, draw_overlay, extract_pose
from dms.detection.thresholds import AdaptiveThresholdManager
from dms.io.engines import AudioAlertEngine, VoiceEngine
from dms.io.session_logger import SessionLogger, cleanup_old_snapshots

logger = logging.getLogger("dri01")


class DriverAgent:
    def __init__(self, cap):
        self.cap = cap
        self.running = False
        self.result_queue = queue.Queue(maxsize=1)
        self.logger = SessionLogger(LOG_FILE)
        cleanup_old_snapshots()
        self.adapter = AdaptiveThresholdManager()
        self.voice = VoiceEngine()
        self.audio = AudioAlertEngine()
        self.thread_lock = threading.Lock()
        self.error_state = None
        self.thread = None
        self._last_timestamp_ms = int(time.monotonic() * 1000)
        self.calibrated = False
        self.calibrating = False
        self.calibration_end_time = None
        self.calibration_samples = {"ear": [], "mar": [], "pitch": [], "yaw": []}
        self.system_warnings = []

        base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
        options = vision.FaceLandmarkerOptions(
            base_options=base_options,
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
            num_faces=1,
            running_mode=vision.RunningMode.VIDEO,
        )
        self.detector = vision.FaceLandmarker.create_from_options(options)

        self.eye_closed_start = None
        self.mouth_open_start = None
        self.distracted_start = None
        self.face_missing_start = None

        self.drowsy_count = 0
        self.yawn_count = 0
        self.distraction_count = 0
        self.alertness_score = 100.0
        self.closure_history = []
        self.smooth_ear = 0.3
        self.smooth_mar = 0.1
        self.neutral_pose = {"pitch": 0.0, "yaw": 0.0}
        self.load_calibration_profile()
        if self.voice.last_error:
            self.system_warnings.append(self.voice.last_error)
        if self.audio.last_error:
            self.system_warnings.append(self.audio.last_error)

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

    def load_calibration_profile(self):
        if not os.path.exists(BASELINE_FILE):
            return
        try:
            with open(BASELINE_FILE, "r", encoding="utf-8") as f:
                baseline = json.load(f)
            avg_ear = float(baseline.get("avg_ear", 0.28))
            avg_mar = float(baseline.get("avg_mar", 0.20))
            self.neutral_pose = {
                "pitch": float(baseline.get("neutral_pitch", 0.0)),
                "yaw": float(baseline.get("neutral_yaw", 0.0)),
            }
            self.adapter.set_calibrated_baseline(avg_ear, avg_mar)
            self.calibrated = True
        except Exception:
            logger.exception("Failed to load calibration baseline profile.")

    def start_calibration(self, duration_seconds):
        self.calibration_samples = {"ear": [], "mar": [], "pitch": [], "yaw": []}
        self.calibrating = True
        self.calibration_end_time = time.time() + duration_seconds
        self.set_error_state(None)

    def _finalize_calibration(self):
        if not self.calibration_samples["ear"] or not self.calibration_samples["mar"]:
            self.set_error_state("Calibration failed: insufficient face data.")
            self.calibrating = False
            return
        avg_ear = sum(self.calibration_samples["ear"]) / len(self.calibration_samples["ear"])
        avg_mar = sum(self.calibration_samples["mar"]) / len(self.calibration_samples["mar"])
        neutral_pitch = sum(self.calibration_samples["pitch"]) / max(1, len(self.calibration_samples["pitch"]))
        neutral_yaw = sum(self.calibration_samples["yaw"]) / max(1, len(self.calibration_samples["yaw"]))
        self.adapter.set_calibrated_baseline(avg_ear, avg_mar)
        self.neutral_pose = {"pitch": neutral_pitch, "yaw": neutral_yaw}
        payload = {
            "avg_ear": avg_ear,
            "avg_mar": avg_mar,
            "neutral_pitch": neutral_pitch,
            "neutral_yaw": neutral_yaw,
            "saved_at": datetime.now().isoformat(),
        }
        try:
            with open(BASELINE_FILE, "w", encoding="utf-8") as f:
                json.dump(payload, f, indent=2)
            self.calibrated = True
            self.set_error_state("Calibration completed.")
        except Exception:
            logger.exception("Failed to persist calibration profile.")
            self.set_error_state("Calibration completed, but saving profile failed.")
        finally:
            self.calibrating = False

    def _next_timestamp_ms(self):
        current = int(time.monotonic() * 1000)
        if current <= self._last_timestamp_ms:
            current = self._last_timestamp_ms + 1
        self._last_timestamp_ms = current
        return current

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
                frame, night_mode = apply_night_vision(raw_frame)
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                mp_img = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
                res = self.detector.detect_for_video(mp_img, self._next_timestamp_ms())

                status = "Continuous Learning..."
                h, w, _ = frame.shape
                pose = (0, 0, 0)

                if res.face_landmarks:
                    self.face_missing_start = None
                    marks = res.face_landmarks[0]
                    ear, mar = calculate_metrics(marks, w, h)
                    self.adapter.update(ear, mar, is_eye_relaxed=(self.eye_closed_start is None), is_mouth_relaxed=(self.mouth_open_start is None))
                    self.smooth_ear = self.smooth_ear * 0.8 + ear * 0.2
                    self.smooth_mar = self.smooth_mar * 0.8 + mar * 0.2
                    if res.facial_transformation_matrixes:
                        pose = extract_pose(res.facial_transformation_matrixes[0].data)
                    pitch, yaw, _ = pose

                    if self.calibrating:
                        self.calibration_samples["ear"].append(ear)
                        self.calibration_samples["mar"].append(mar)
                        self.calibration_samples["pitch"].append(pitch)
                        self.calibration_samples["yaw"].append(yaw)
                        status = "CALIBRATING..."
                        frame_draw = draw_overlay(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), marks, status, pose)
                        frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)
                        if self.calibration_end_time and time.time() >= self.calibration_end_time:
                            self._finalize_calibration()
                    else:
                        curr_status = "Active"
                        if ear < self.adapter.current_ear_threshold:
                            if self.eye_closed_start is None:
                                self.eye_closed_start = time.time()
                            duration = time.time() - self.eye_closed_start
                            if duration > 1.0:
                                curr_status = "DROWSY - WAKE UP!"
                                self.audio.play_tone(2000, 200)
                                self.voice.say("Drowsiness detected. Open your eyes.")
                                self.closure_history.append(1)
                                if duration > 2.5 and not hasattr(self, "_snap_done"):
                                    self._take_snapshot(img_rgb, "DROWSY")
                                    self._snap_done = True
                            else:
                                self.closure_history.append(0)
                        else:
                            if hasattr(self, "_snap_done"):
                                delattr(self, "_snap_done")
                            if self.eye_closed_start:
                                dur = time.time() - self.eye_closed_start
                                if dur > 1.0:
                                    self.drowsy_count += 1
                                    self.logger.log_event("DROWSY", round(dur, 2), f"EAR: {ear:.2f}")
                                    if self.drowsy_count % 3 == 0:
                                        self.voice.say("You have multiple drowsy events. Please consider a coffee break.")
                            self.eye_closed_start = None
                            self.closure_history.append(0)

                        if mar > self.adapter.current_mar_threshold:
                            if self.mouth_open_start is None:
                                self.mouth_open_start = time.time()
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

                        if abs(yaw) > 25 or abs(pitch) > 18:
                            if self.distracted_start is None:
                                self.distracted_start = time.time()
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
                        frame_draw = draw_overlay(cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), marks, status, pose)
                        frame = cv2.cvtColor(frame_draw, cv2.COLOR_BGR2RGB)

                        if len(self.closure_history) > 600:
                            self.closure_history.pop(0)
                        if self.closure_history:
                            perclos = (sum(self.closure_history) / len(self.closure_history)) * 100
                            if perclos <= 20:
                                self.alertness_score = 100.0
                            elif perclos >= 80:
                                self.alertness_score = 0.0
                            else:
                                self.alertness_score = ((80 - perclos) / 60.0) * 100.0
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
                    if len(self.closure_history) > 600:
                        self.closure_history.pop(0)

                inf_time = (time.time() - prev_time)
                fps = 1.0 / inf_time if inf_time > 0 else 0
                prev_time = time.time()
                res_pkg = {
                    "frame": frame,
                    "status": status,
                    "fps": fps,
                    "ear": self.smooth_ear,
                    "mar": self.smooth_mar,
                    "score": self.alertness_score,
                    "counts": (self.drowsy_count, self.yawn_count, self.distraction_count),
                    "pose": pose,
                    "thresholds": (self.adapter.current_ear_threshold, self.adapter.current_mar_threshold),
                    "night": night_mode,
                }
                if self.result_queue.full():
                    try:
                        self.result_queue.get_nowait()
                    except Exception:
                        pass
                self.result_queue.put(res_pkg)
            except Exception:
                self.set_error_state("Detection loop recovered from an internal error.")
                logger.exception("Detector loop error:\n%s", traceback.format_exc())
                time.sleep(0.05)
