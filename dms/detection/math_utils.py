import cv2
import numpy as np

from dms.config import LEFT_EYE, RIGHT_EYE, MOUTH


def calculate_metrics(landmarks, w, h):
    l_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in LEFT_EYE])
    r_eye = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in RIGHT_EYE])

    def eye_ear(eye):
        v1 = np.linalg.norm(eye[1] - eye[5])
        v2 = np.linalg.norm(eye[2] - eye[4])
        hor = np.linalg.norm(eye[0] - eye[3])
        return (v1 + v2) / (2.0 * hor) if hor else 0.0

    ear = (eye_ear(l_eye) + eye_ear(r_eye)) / 2.0

    mouth = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in MOUTH])
    m_hor = np.linalg.norm(mouth[0] - mouth[1])
    m_ver = np.linalg.norm(mouth[2] - mouth[3])
    mar = (m_ver / m_hor) if m_hor else 0.0

    return ear, mar


def extract_pose(matrix):
    m = np.array(matrix).reshape(4, 4)[:3, :3]
    sy = np.sqrt(m[0, 0] ** 2 + m[1, 0] ** 2)
    if sy > 1e-6:
        x = np.arctan2(m[2, 1], m[2, 2])
        y = np.arctan2(-m[2, 0], sy)
        z = np.arctan2(m[1, 0], m[0, 0])
    else:
        x = np.arctan2(-m[1, 2], m[1, 1])
        y = np.arctan2(-m[2, 0], sy)
        z = 0
    return np.degrees(x), np.degrees(y), np.degrees(z)


def apply_night_vision(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    avg_brightness = np.mean(l)
    if avg_brightness < 80:
        clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
        cl = clahe.apply(l)
        limg = cv2.merge((cl, a, b))
        return cv2.cvtColor(limg, cv2.COLOR_LAB2BGR), True
    return frame, False


def draw_overlay(frame, landmarks, status, pose):
    h, w, _ = frame.shape
    color = (0, 255, 204) if "Active" in status else (0, 100, 255)
    if "DROWSY" in status or "WARNING" in status:
        color = (0, 0, 255)

    for i in range(len(landmarks)):
        px = int(landmarks[i].x * w)
        py = int(landmarks[i].y * h)
        if i % 15 == 0:
            cv2.circle(frame, (px, py), 1, color, -1)

    pitch, yaw, _ = pose
    nose = landmarks[1]
    cx, cy = int(nose.x * w), int(nose.y * h)
    length = 100
    ex = int(cx + length * np.sin(np.radians(-yaw)))
    ey = int(cy + length * np.sin(np.radians(pitch)))
    cv2.line(frame, (cx, cy), (ex, ey), color, 2)
    cv2.circle(frame, (ex, ey), 4, color, -1)
    return frame
