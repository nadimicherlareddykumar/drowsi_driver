import numpy as np

from dms.detection.math_utils import calculate_metrics, extract_pose
from dms.detection.thresholds import AdaptiveThresholdManager


class Pt:
    def __init__(self, x, y):
        self.x = x
        self.y = y


def test_calculate_metrics_returns_positive_values():
    landmarks = [Pt(0.5, 0.5) for _ in range(400)]
    # Left eye indices
    landmarks[33] = Pt(0.40, 0.40)
    landmarks[160] = Pt(0.42, 0.38)
    landmarks[158] = Pt(0.45, 0.38)
    landmarks[133] = Pt(0.48, 0.40)
    landmarks[153] = Pt(0.45, 0.42)
    landmarks[144] = Pt(0.42, 0.42)
    # Right eye indices
    landmarks[362] = Pt(0.52, 0.40)
    landmarks[385] = Pt(0.54, 0.38)
    landmarks[387] = Pt(0.57, 0.38)
    landmarks[263] = Pt(0.60, 0.40)
    landmarks[373] = Pt(0.57, 0.42)
    landmarks[380] = Pt(0.54, 0.42)
    # Mouth indices
    landmarks[61] = Pt(0.45, 0.60)
    landmarks[291] = Pt(0.55, 0.60)
    landmarks[13] = Pt(0.50, 0.57)
    landmarks[14] = Pt(0.50, 0.63)

    ear, mar = calculate_metrics(landmarks, 640, 480)
    assert ear > 0
    assert mar > 0


def test_extract_pose_identity_matrix_near_zero():
    matrix = np.eye(4).reshape(-1).tolist()
    pitch, yaw, roll = extract_pose(matrix)
    assert abs(pitch) < 1e-5
    assert abs(yaw) < 1e-5
    assert abs(roll) < 1e-5


def test_adaptive_thresholds_apply_clamps():
    manager = AdaptiveThresholdManager()
    manager.current_ear_threshold = 10
    manager.current_mar_threshold = -10
    manager.apply_clamps()
    assert manager.min_ear_threshold <= manager.current_ear_threshold <= manager.max_ear_threshold
    assert manager.min_mar_threshold <= manager.current_mar_threshold <= manager.max_mar_threshold
