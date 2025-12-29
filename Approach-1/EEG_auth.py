import serial
import time
import numpy as np
import tkinter as tk
from math import sqrt

# ---------------- CONFIG ----------------
PORT = "COM5"        # change if needed
BAUD = 115200
DURATION = 8         # seconds (5–10 recommended)
FS = 256             # sampling rate
THRESHOLD_MARGIN = 0.25
# ---------------------------------------


def capture_eeg():
    ser = serial.Serial(PORT, BAUD)
    time.sleep(2)

    samples = []
    start = time.time()

    while time.time() - start < DURATION:
        line = ser.readline().decode(errors="ignore").strip()
        try:
            samples.append(float(line))
        except:
            pass

    ser.close()
    return np.array(samples)


def extract_features(signal):
    mean = np.mean(signal)
    std = np.std(signal)
    energy = np.mean(signal ** 2)
    return np.array([mean, std, energy])


def distance(v1, v2):
    return sqrt(np.sum((v1 - v2) ** 2))


# ---------------- UI ----------------
root = tk.Tk()
root.title("Brain Password System")
root.geometry("400x250")

status_label = tk.Label(root, text="SYSTEM LOCKED", font=("Arial", 18))
status_label.pack(pady=20)

info_label = tk.Label(root, text="", font=("Arial", 12))
info_label.pack(pady=10)

template_features = None
threshold = None


def set_password():
    global template_features, threshold

    info_label.config(text="Think now to SET brain password...")
    root.update()

    eeg = capture_eeg()
    f1 = extract_features(eeg)

    # Capture second sample to estimate variation
    time.sleep(1)
    info_label.config(text="Think SAME thing again...")
    root.update()

    eeg2 = capture_eeg()
    f2 = extract_features(eeg2)

    template_features = (f1 + f2) / 2
    dist = distance(f1, f2)
    threshold = dist + THRESHOLD_MARGIN

    status_label.config(text="SYSTEM LOCKED")
    info_label.config(text="Brain password set.\nSystem locked.")


def unlock_system():
    if template_features is None:
        info_label.config(text="Set password first.")
        return

    info_label.config(text="Think now to UNLOCK...")
    root.update()

    eeg = capture_eeg()
    test_features = extract_features(eeg)

    dist = distance(test_features, template_features)

    if dist < threshold:
        status_label.config(text="SYSTEM UNLOCKED")
        info_label.config(text="Access Granted")
    else:
        status_label.config(text="SYSTEM LOCKED")
        info_label.config(text="Access Denied")


btn_set = tk.Button(root, text="Set Brain Password", command=set_password, width=25)
btn_set.pack(pady=5)

btn_unlock = tk.Button(root, text="Unlock System", command=unlock_system, width=25)
btn_unlock.pack(pady=5)

root.mainloop()
