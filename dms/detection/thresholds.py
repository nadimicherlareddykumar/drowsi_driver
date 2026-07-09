class AdaptiveThresholdManager:
    """Continuously learns from environment to adjust EAR/MAR thresholds."""

    def __init__(self):
        self.ear_baseline_history = []
        self.mar_baseline_history = []
        self.window_size = 300
        self.current_ear_threshold = 0.2
        self.current_mar_threshold = 0.6
        self.min_ear_threshold = 0.12
        self.max_ear_threshold = 0.35
        self.min_mar_threshold = 0.35
        self.max_mar_threshold = 1.2
        self.is_ready = False

    def apply_clamps(self):
        self.current_ear_threshold = max(self.min_ear_threshold, min(self.current_ear_threshold, self.max_ear_threshold))
        self.current_mar_threshold = max(self.min_mar_threshold, min(self.current_mar_threshold, self.max_mar_threshold))

    def set_calibrated_baseline(self, avg_ear, avg_mar):
        self.current_ear_threshold = avg_ear * 0.72
        self.current_mar_threshold = avg_mar * 1.65
        self.apply_clamps()
        self.is_ready = True

    def update(self, raw_ear, raw_mar, is_eye_relaxed=True, is_mouth_relaxed=True):
        if is_eye_relaxed:
            self.ear_baseline_history.append(raw_ear)
            if len(self.ear_baseline_history) > self.window_size:
                self.ear_baseline_history.pop(0)
        if is_mouth_relaxed:
            self.mar_baseline_history.append(raw_mar)
            if len(self.mar_baseline_history) > self.window_size:
                self.mar_baseline_history.pop(0)

        if len(self.ear_baseline_history) >= self.window_size and len(self.mar_baseline_history) >= self.window_size:
            self.is_ready = True

        if self.ear_baseline_history:
            avg_ear = sum(self.ear_baseline_history) / len(self.ear_baseline_history)
            self.current_ear_threshold = avg_ear * 0.72
        if self.mar_baseline_history:
            avg_mar = sum(self.mar_baseline_history) / len(self.mar_baseline_history)
            self.current_mar_threshold = avg_mar * 1.65
        self.apply_clamps()
