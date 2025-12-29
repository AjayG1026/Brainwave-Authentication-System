import serial
import time
import numpy as np
import torch
import torch.nn as nn
from scipy.signal import spectrogram
import tkinter as tk
from tkinter import ttk
import threading

# ---------------- CONFIG ----------------
PORT = "COM5"
BAUD = 115200
FS = 256
DURATION = 8
MODEL_PATH = "eeg_auth_model.pth"
THRESHOLD = 0.6
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# --------------------------------------


class EEGNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(1, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d((2, 1)),

            nn.AdaptiveAvgPool2d((8, 8))
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.features(x)
        return self.classifier(x)


def capture_eeg():
    ser = serial.Serial(PORT, BAUD)
    time.sleep(2)

    samples = []
    start = time.time()

    while time.time() - start < DURATION:
        try:
            samples.append(float(ser.readline().decode().strip()))
        except:
            pass

    ser.close()
    return np.array(samples)


def authenticate_thread():
    set_status("SCANNING BRAIN ACTIVITY", "#ffaa00")
    progress.start(10)

    eeg = capture_eeg()

    _, _, Sxx = spectrogram(
        eeg,
        fs=FS,
        nperseg=128,
        noverlap=64
    )
    Sxx = np.log(Sxx + 1e-8)

    x = torch.tensor(Sxx, dtype=torch.float32).unsqueeze(0).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        prob = model(x).item()

    progress.stop()
    progress["value"] = prob * 100

    if prob > THRESHOLD:
        set_status("SYSTEM UNLOCKED", "#00ff88")
        info_label.config(text=f"ACCESS GRANTED\nConfidence: {prob:.2f}")
    else:
        set_status("SYSTEM LOCKED", "#ff4444")
        info_label.config(text=f"ACCESS DENIED\nConfidence: {prob:.2f}")

    auth_button.config(state="normal")


def authenticate():
    auth_button.config(state="disabled")
    threading.Thread(target=authenticate_thread, daemon=True).start()


def set_status(text, color):
    status_label.config(text=text, fg=color)


# -------- MODEL --------
model = EEGNet().to(DEVICE)
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------- UI --------
root = tk.Tk()
root.title("EEG Brain Password Authentication")
root.geometry("480x340")
root.configure(bg="#0e0e0e")

style = ttk.Style()
style.theme_use("default")
style.configure(
    "TProgressbar",
    thickness=20,
    troughcolor="#1f1f1f",
    background="#00ff88"
)

title = tk.Label(
    root,
    text="BCI AUTHENTICATION SYSTEM",
    font=("Segoe UI", 16, "bold"),
    bg="#0e0e0e",
    fg="#00bfff"
)
title.pack(pady=10)

status_label = tk.Label(
    root,
    text="SYSTEM LOCKED",
    font=("Segoe UI", 18, "bold"),
    bg="#0e0e0e",
    fg="#ff4444"
)
status_label.pack(pady=10)

info_label = tk.Label(
    root,
    text="Press Authenticate and focus",
    font=("Segoe UI", 11),
    bg="#0e0e0e",
    fg="#bbbbbb",
    justify="center"
)
info_label.pack(pady=10)

progress = ttk.Progressbar(
    root,
    orient="horizontal",
    length=320,
    mode="determinate",
    maximum=100
)
progress.pack(pady=15)

auth_button = tk.Button(
    root,
    text="AUTHENTICATE",
    command=authenticate,
    font=("Segoe UI", 12, "bold"),
    bg="#1f1f1f",
    fg="#00ff88",
    activebackground="#00ff88",
    activeforeground="#000000",
    relief="flat",
    width=20,
    height=2
)
auth_button.pack(pady=20)

footer = tk.Label(
    root,
    text="Deep Learning EEG Brain Password",
    font=("Segoe UI", 9),
    bg="#0e0e0e",
    fg="#555555"
)
footer.pack(side="bottom", pady=8)

root.mainloop()
