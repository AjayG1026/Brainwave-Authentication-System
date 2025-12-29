import serial
import time
import numpy as np
import os

# ---------------- CONFIG ----------------
PORT = "COM5"
BAUD = 115200
FS = 256
WINDOW_SEC = 2
OVERLAP = 0.5
RECORD_SEC = 20
SAVE_DIR = "dataset/train"
# --------------------------------------

SAMPLES_PER_WINDOW = int(FS * WINDOW_SEC)
STEP_SIZE = int(SAMPLES_PER_WINDOW * (1 - OVERLAP))

os.makedirs(f"{SAVE_DIR}/authorized", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/impostor", exist_ok=True)


def collect_raw_eeg():
    ser = serial.Serial(PORT, BAUD)
    time.sleep(2)

    samples = []
    start = time.time()

    while time.time() - start < RECORD_SEC:
        try:
            samples.append(float(ser.readline().decode(errors="ignore").strip()))
        except:
            pass

    ser.close()
    return np.array(samples)


def segment_signal(signal):
    windows = []
    for i in range(0, len(signal) - SAMPLES_PER_WINDOW, STEP_SIZE):
        windows.append(signal[i:i + SAMPLES_PER_WINDOW])
    return windows


def record(label, instruction):
    print("\n" + "=" * 50)
    print(instruction)
    input("Press ENTER to start recording...")

    raw = collect_raw_eeg()
    windows = segment_signal(raw)

    for i, w in enumerate(windows):
        fname = f"{SAVE_DIR}/{label}/{label}_{int(time.time())}_{i}.npy"
        np.save(fname, w)

    print(f"{label.upper()} samples saved:", len(windows))


def main():
    record(
        label="authorized",
        instruction="AUTHORIZED DATA\nThink of your BRAIN PASSWORD task"
    )

    time.sleep(3)

    record(
        label="impostor",
        instruction="IMPOSTOR DATA\nThink of a DIFFERENT mental task"
    )

    print("\nData collection complete.")


if __name__ == "__main__":
    main()
