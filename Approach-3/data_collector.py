import serial
import time
import numpy as np
import os

PORT = "COM5"
BAUD = 115200
FS = 256
WINDOW_SEC = 2
OVERLAP = 0.5
RECORD_SEC = 20
SAVE_DIR = "dataset/train"

SAMPLES_PER_WINDOW = int(FS * WINDOW_SEC)
STEP = int(SAMPLES_PER_WINDOW * (1 - OVERLAP))

os.makedirs(f"{SAVE_DIR}/genuine", exist_ok=True)
os.makedirs(f"{SAVE_DIR}/impostor", exist_ok=True)


def collect_raw():
    ser = serial.Serial(PORT, BAUD)
    time.sleep(2)
    data = []
    start = time.time()

    while time.time() - start < RECORD_SEC:
        try:
            data.append(float(ser.readline().decode().strip()))
        except:
            pass

    ser.close()
    return np.array(data)


def segment(x):
    return [
        x[i:i + SAMPLES_PER_WINDOW]
        for i in range(0, len(x) - SAMPLES_PER_WINDOW, STEP)
    ]


def record(label, msg):
    print("\n" + "=" * 50)
    print(msg)
    input("Press ENTER to start recording")

    raw = collect_raw()
    windows = segment(raw)

    for i, w in enumerate(windows):
        np.save(f"{SAVE_DIR}/{label}/{label}_{int(time.time())}_{i}.npy", w)

    print(f"{label.upper()} samples:", len(windows))


def main():
    record(
        "genuine",
        "TARGET USER\nThink of YOUR SECRET BRAIN PASSWORD"
    )

    record(
        "impostor",
        "TARGET USER\nThink of a DIFFERENT mental task"
    )

    print("\nNOW OTHER PERSON WILL RECORD\n")

    record(
        "impostor",
        "OTHER PERSON\nThink of THE SAME PASSWORD TASK"
    )

    print("\nDataset creation complete.")


if __name__ == "__main__":
    main()
