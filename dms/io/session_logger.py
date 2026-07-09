import csv
import logging
import os
import time
from datetime import datetime

from dms.config import SNAPSHOT_RETENTION_DAYS

logger = logging.getLogger("dri01")


class SessionLogger:
    def __init__(self, filename):
        self.filename = filename
        if not os.path.exists(self.filename):
            with open(self.filename, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Event Type", "Duration (s)", "Details"])

    def log_event(self, event_type, duration=0, details=""):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(self.filename, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([timestamp, event_type, duration, details])


def cleanup_old_snapshots(directory="crisis_logs", retention_days=SNAPSHOT_RETENTION_DAYS):
    if not os.path.isdir(directory):
        return
    cutoff = time.time() - (retention_days * 24 * 3600)
    for name in os.listdir(directory):
        if not name.lower().endswith(".jpg"):
            continue
        path = os.path.join(directory, name)
        try:
            if os.path.isfile(path) and os.path.getmtime(path) < cutoff:
                os.remove(path)
        except Exception:
            logger.exception("Failed to purge old snapshot: %s", path)
