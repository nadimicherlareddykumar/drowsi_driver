
import cv2
import time

def check_camera(index, backend=None):
    if backend:
        cap = cv2.VideoCapture(index, backend)
    else:
        cap = cv2.VideoCapture(index)
    
    if not cap.isOpened():
        return False, "Could not open"
    
    ret, frame = cap.read()
    cap.release()
    if ret:
        return True, "Success"
    else:
        return False, "Failed to read frame"

print("Detailed Camera Check:")
backends = [("Default", None), ("DSHOW", cv2.CAP_DSHOW), ("MSMF", cv2.CAP_MSMF)]

for b_name, b_val in backends:
    for i in range(2):
        ok, msg = check_camera(i, b_val)
        print(f"Index {i} with {b_name}: {msg}")
