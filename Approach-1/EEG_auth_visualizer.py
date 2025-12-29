import serial, time, threading
import numpy as np
import tkinter as tk
from math import sqrt
from queue import Queue

import matplotlib
matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from scipy.signal import welch

# ---------------- CONFIG ----------------
PORT = "COM5"
BAUD = 115200
FS = 256
WINDOW_SEC = 2
DURATION = 8
UPDATE_INTERVAL = 0.1   # seconds (UI refresh)
THRESHOLD_MARGIN = 0.25
# ---------------------------------------

buffer = []
data_queue = Queue()
stop_flag = False

# ---------- EEG FUNCTIONS ----------
def preprocess(sig):
    sig = sig - np.mean(sig)
    sig = sig / (np.std(sig) + 1e-6)
    return sig


def compute_bands(sig):
    freqs, psd = welch(sig, FS, nperseg=FS*2)
    total = np.trapz(psd, freqs)

    def bp(f1, f2):
        idx = (freqs >= f1) & (freqs <= f2)
        return np.trapz(psd[idx], freqs[idx]) / total

    return {
        "Delta": bp(0.5, 4),
        "Theta": bp(4, 8),
        "Alpha": bp(8, 13),
        "Beta":  bp(13, 30),
        "Gamma": bp(30, 45),
    }


# ---------- THREAD ----------
def eeg_worker():
    global stop_flag, buffer
    ser = serial.Serial(PORT, BAUD)
    time.sleep(2)

    while not stop_flag:
        try:
            v = float(ser.readline().decode(errors="ignore"))
            buffer.append(v)
            if len(buffer) > FS * WINDOW_SEC:
                buffer = buffer[-FS * WINDOW_SEC:]
        except:
            pass

    ser.close()


# ---------- UI ----------
root = tk.Tk()
root.title("Brain Password Authentication System")
root.geometry("1000x700")

status_label = tk.Label(root, text="SYSTEM LOCKED", font=("Arial", 18))
status_label.pack()

info_label = tk.Label(root, text="", font=("Arial", 12))
info_label.pack()

fig = Figure(figsize=(9, 6))
ax1 = fig.add_subplot(311)
ax2 = fig.add_subplot(312)
ax3 = fig.add_subplot(313)

canvas = FigureCanvasTkAgg(fig, root)
canvas.get_tk_widget().pack()

band_names = ["Delta", "Theta", "Alpha", "Beta", "Gamma"]

template_features = None
threshold = None
recorded_features = []


def update_ui():
    if len(buffer) > FS:
        sig = preprocess(np.array(buffer))

        ax1.clear()
        ax1.plot(sig)
        ax1.set_title("Raw EEG (Normalized)")
        ax1.set_ylim(-4, 4)

        freqs, psd = welch(sig, FS, nperseg=FS*2)
        ax2.clear()
        ax2.semilogy(freqs, psd)
        ax2.set_xlim(0, 50)
        ax2.set_title("EEG Spectrum")

        bands = compute_bands(sig)
        ax3.clear()
        ax3.bar(band_names, bands.values())
        ax3.set_ylim(0, 1)
        ax3.set_title("Relative EEG Band Power")

        canvas.draw_idle()

    root.after(int(UPDATE_INTERVAL * 1000), update_ui)


def record_features():
    time.sleep(DURATION)
    sig = preprocess(np.array(buffer))
    bands = compute_bands(sig)
    return np.array(list(bands.values()))


def set_password():
    global template_features, threshold, stop_flag, recorded_features

    info_label.config(text="Think now to SET brain password")
    recorded_features = []

    f1 = record_features()
    info_label.config(text="Think SAME thing again")
    f2 = record_features()

    template_features = (f1 + f2) / 2
    threshold = np.linalg.norm(f1 - f2) + THRESHOLD_MARGIN

    info_label.config(text="Brain password set")


def unlock_system():
    if template_features is None:
        info_label.config(text="Set password first")
        return

    info_label.config(text="Think now to UNLOCK")
    f = record_features()

    if np.linalg.norm(f - template_features) < threshold:
        status_label.config(text="SYSTEM UNLOCKED")
        info_label.config(text="Access Granted")
    else:
        status_label.config(text="SYSTEM LOCKED")
        info_label.config(text="Access Denied")


tk.Button(root, text="Set Brain Password", width=30, command=lambda: threading.Thread(target=set_password).start()).pack(pady=5)
tk.Button(root, text="Unlock System", width=30, command=lambda: threading.Thread(target=unlock_system).start()).pack(pady=5)

threading.Thread(target=eeg_worker, daemon=True).start()
root.after(100, update_ui)
root.mainloop()

stop_flag = True
