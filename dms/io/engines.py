import logging
import queue
import sys
import threading
import time

import numpy as np
import pyttsx3

try:
    import simpleaudio as sa
except Exception:
    sa = None

logger = logging.getLogger("dri01")


class VoiceEngine:
    def __init__(self):
        self.queue = queue.Queue()
        self.running = True
        self.last_voice_time = 0
        self.lock = threading.Lock()
        self.available = True
        self.last_error = None
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
                    logger.warning("pythoncom unavailable; COM init skipped for TTS worker.")

            engine = pyttsx3.init()
            voices = engine.getProperty("voices")
            if len(voices) > 1:
                engine.setProperty("voice", voices[1].id)
            engine.setProperty("rate", 160)

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
            self.available = False
            self.last_error = "TTS backend unavailable. Install espeak (Linux) or proper speech backend."
            logger.exception("Voice engine initialization failed.")
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
                    logger.exception("COM uninitialize failed in TTS worker.")

    def say(self, text, cooldown=3):
        if not self.available:
            return
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
        self.available = sa is not None
        self.last_error = None if self.available else "Audio backend unavailable. Install simpleaudio."
        self.thread = threading.Thread(target=self._worker, daemon=True)
        self.thread.start()

    def play_tone(self, freq_hz=1500, duration_ms=150, volume=0.25):
        self.queue.put((freq_hz, duration_ms, volume))

    @staticmethod
    def _generate_tone(freq_hz, duration_ms, volume):
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
                if not self.available or sa is None:
                    continue
                audio_bytes, sample_rate = self._generate_tone(freq_hz, duration_ms, volume)
                play_obj = sa.play_buffer(audio_bytes, 1, 2, sample_rate)
                play_obj.wait_done()
            except Exception:
                self.last_error = "Audio playback failed. Check output device/backend."
                logger.exception("Audio alert playback failed.")

    def close(self):
        self.running = False
        self.queue.put(None)
        if self.thread.is_alive():
            self.thread.join(timeout=2.0)
