import platform
import subprocess
import sys


def run(cmd):
    print(f"Running: {' '.join(cmd)}")
    subprocess.check_call(cmd)


def main():
    run([sys.executable, "-m", "pip", "install", "--upgrade", "pip"])
    run([sys.executable, "-m", "pip", "install", "-r", "requirements.txt"])

    system = platform.system().lower()
    if system == "linux":
        print("Linux note: install espeak for pyttsx3 (example: sudo apt-get install espeak)")
    elif system == "darwin":
        print("macOS note: pyttsx3 uses system speech voices (no extra package usually required)")
    elif system == "windows":
        print("Windows note: pywin32 is installed via requirements for TTS COM support")

    print("Environment setup complete.")


if __name__ == "__main__":
    main()
