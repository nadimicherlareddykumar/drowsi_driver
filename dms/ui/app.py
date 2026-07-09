import logging
import os
import queue
import time
import tkinter as tk
from tkinter import Button, Frame, Label

import cv2
from PIL import Image, ImageTk

from dms.config import CALIBRATION_SECONDS, LOG_FILE
from dms.detection.driver_agent import DriverAgent

logger = logging.getLogger("dri01")


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
            logger.critical("All camera initialization attempts failed.")

        self._build_ui()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self._tick()

    def _init_cap(self):
        attempts = [(0, cv2.CAP_DSHOW), (0, cv2.CAP_MSMF), (0, None), (1, cv2.CAP_DSHOW), (1, None)]
        for idx, backend in attempts:
            try:
                cap = cv2.VideoCapture(idx, backend) if backend is not None else cv2.VideoCapture(idx)
                if cap.isOpened():
                    ret, _ = cap.read()
                    if ret:
                        logger.info("Camera initialized on index %s backend %s", idx, backend)
                        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                        return cap
                    cap.release()
            except Exception as exc:
                logger.exception("Failed to init camera %s with backend %s: %s", idx, backend, exc)
        return None

    def _build_ui(self):
        top = Frame(self.root, bg="#0a0a0a", height=60)
        top.pack(fill="x", side="top", pady=10)
        Label(top, text="AGENT DRI01", font=("Consolas", 24, "bold"), fg="#00ffcc", bg="#0a0a0a").pack(side="left", padx=20)
        self.time_lbl = Label(top, text="", font=("Consolas", 12), fg="#888", bg="#0a0a0a")
        self.time_lbl.pack(side="right", padx=20)

        main_frame = Frame(self.root, bg="#0a0a0a")
        main_frame.pack(fill="both", expand=True, padx=10)
        self.vid_lbl = Label(
            main_frame,
            text="SYSTEM INITIALIZED\n\nCLICK 'ACTIVATE AGENT' TO START MONITORING",
            fg="#00ffcc",
            bg="black",
            font=("Consolas", 14),
            borderwidth=0,
        )
        self.vid_lbl.pack(side="left", fill="both", expand=True)

        dash = Frame(main_frame, bg="#121212", width=300)
        dash.pack(side="right", fill="y", padx=10)
        dash.pack_propagate(False)
        Label(dash, text="DRIVER ALERTNESS", font=("Consolas", 10), fg="#aaa", bg="#121212").pack(pady=(20, 0))
        self.score_lbl = Label(dash, text="100%", font=("Consolas", 48, "bold"), fg="#00ffcc", bg="#121212")
        self.score_lbl.pack()
        self.stat_frame = Frame(dash, bg="#121212")
        self.stat_frame.pack(fill="x", pady=20, padx=10)
        self._add_stat("DROWSY EVENTS", "0", "#ff4444")
        self._add_stat("YAWN EVENTS", "0", "#ffaa00")
        self._add_stat("DISTRACTIONS", "0", "#aa44ff")
        self.learn_lbl = Label(dash, text="Agent Level: Learning...", font=("Consolas", 9), fg="#666", bg="#121212", wraplength=250)
        self.learn_lbl.pack(side="bottom", pady=20)
        self.tel_lbl = Label(dash, text="EAR: 0.00 | MAR: 0.00", font=("Consolas", 9), fg="#888", bg="#121212")
        self.tel_lbl.pack(side="bottom")

        bot = Frame(self.root, bg="#0a0a0a", height=80)
        bot.pack(fill="x", side="bottom")
        self.btn_run = Button(bot, text="ACTIVATE AGENT", command=self._toggle, font=("Consolas", 12, "bold"), bg="#00ffcc", fg="#000", width=20, relief="flat")
        self.btn_run.pack(side="left", padx=20, pady=20)
        Button(bot, text="RESET LOGS", command=self._reset_logs, font=("Consolas", 10), bg="#333", fg="#fff", width=12, relief="flat").pack(side="right", padx=20)
        Button(bot, text="CALIBRATE", command=self._calibrate, font=("Consolas", 10, "bold"), bg="#ffaa00", fg="#000", width=12, relief="flat").pack(side="right", padx=10)

    def _add_stat(self, title, val, color):
        f = Frame(self.stat_frame, bg="#1a1a1a", pady=10)
        f.pack(fill="x", pady=5)
        Label(f, text=title, font=("Consolas", 8), fg="#999", bg="#1a1a1a").pack(side="left", padx=10)
        lbl = Label(f, text=val, font=("Consolas", 14, "bold"), fg=color, bg="#1a1a1a")
        lbl.pack(side="right", padx=10)
        if "DROWSY" in title:
            self.d_count_lbl = lbl
        elif "YAWN" in title:
            self.y_count_lbl = lbl
        else:
            self.dist_count_lbl = lbl

    def _toggle(self):
        if not self.agent:
            return
        if self.agent.running:
            self.agent.stop()
            self.btn_run.config(text="ACTIVATE AGENT", bg="#00ffcc")
        else:
            if not self.agent.calibrated:
                self.learn_lbl.config(text=f"Calibration required: press CALIBRATE for {CALIBRATION_SECONDS}s before monitoring.")
                return
            self.agent.start()
            self.btn_run.config(text="DEACTIVATE", bg="#ff4444")

    def _calibrate(self):
        if not self.agent:
            return
        if not self.agent.running:
            self.agent.start()
        self.agent.start_calibration(CALIBRATION_SECONDS)

    def _reset_logs(self):
        if os.path.exists(LOG_FILE):
            os.remove(LOG_FILE)
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
            img = Image.fromarray(p["frame"])
            img.thumbnail((750, 500))
            imgtk = ImageTk.PhotoImage(image=img)
            self.vid_lbl.imgtk = imgtk
            self.vid_lbl.config(image=imgtk)
            self.score_lbl.config(text=f"{p['score']:.0f}%")
            if p["score"] < 40:
                self.score_lbl.config(fg="#ff4444")
            elif p["score"] < 70:
                self.score_lbl.config(fg="#ffaa00")
            else:
                self.score_lbl.config(fg="#00ffcc")
            dc, yc, dic = p["counts"]
            self.d_count_lbl.config(text=str(dc))
            self.y_count_lbl.config(text=str(yc))
            self.dist_count_lbl.config(text=str(dic))
            ea, ma = p["ear"], p["mar"]
            te, tm = p["thresholds"]
            self.tel_lbl.config(text=f"EAR: {ea:.3f} (Lim:{te:.2f}) | MAR: {ma:.3f} (Lim:{tm:.2f})")
            self.time_lbl.config(text=f"FPS: {p['fps']:.1f} | SENSOR: {p['status']}")
            night_text = " [NIGHT MODE ACTIVE]" if p["night"] else ""
            if self.agent.calibrating:
                remaining = max(0, int(self.agent.calibration_end_time - time.time())) if self.agent.calibration_end_time else 0
                learn_text = f"Status: Calibrating... {remaining}s remaining{night_text}"
            elif not self.agent.calibrated:
                learn_text = f"Status: Not calibrated. Press CALIBRATE ({CALIBRATION_SECONDS}s){night_text}"
            else:
                learn_text = f"Status: Environmental Sync Active.{night_text}\nLogging to {LOG_FILE}"
            if self.agent.system_warnings:
                learn_text = f"{learn_text}\nWarning: {' | '.join(self.agent.system_warnings)}"
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


def run_app():
    root = tk.Tk()
    DRI01App(root)
    root.mainloop()
