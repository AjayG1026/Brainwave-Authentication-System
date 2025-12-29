### First Approach
 First Diconnect laptop charger!!!
## Arduino code: (EEGFilter.ino)
// EEG Filter - BioAmp EXG Pill
// https://github.com/upsidedownlabs/BioAmp-EXG-Pill

// Upside Down Labs invests time and resources providing this open source code,
// please support Upside Down Labs and open-source hardware by purchasing
// products from Upside Down Labs!

// Copyright (c) 2021 Upside Down Labs - contact@upsidedownlabs.tech

// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:

// The above copyright notice and this permission notice shall be included in all
// copies or substantial portions of the Software.

// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#define SAMPLE_RATE 256
#define BAUD_RATE 115200
#define INPUT_PIN A0


void setup() {
	// Serial connection begin
	Serial.begin(BAUD_RATE);
}

void loop() {
	// Calculate elapsed time
	static unsigned long past = 0;
	unsigned long present = micros();
	unsigned long interval = present - past;
	past = present;

	// Run timer
	static long timer = 0;
	timer -= interval;

	// Sample
	if(timer < 0){
		timer += 1000000 / SAMPLE_RATE;
		float sensor_value = analogRead(INPUT_PIN);
		float signal = EEGFilter(sensor_value);
		Serial.println(signal);
	}
}

// Band-Pass Butterworth IIR digital filter, generated using filter_gen.py.
// Sampling rate: 256.0 Hz, frequency: [0.5, 29.5] Hz.
// Filter is order 4, implemented as second-order sections (biquads).
// Reference: 
// https://docs.scipy.org/doc/scipy/reference/generated/scipy.signal.butter.html
// https://courses.ideate.cmu.edu/16-223/f2020/Arduino/FilterDemos/filter_gen.py
float EEGFilter(float input) {
	float output = input;
	{
		static float z1, z2; // filter section state
		float x = output - -0.95391350*z1 - 0.25311356*z2;
		output = 0.00735282*x + 0.01470564*z1 + 0.00735282*z2;
		z2 = z1;
		z1 = x;
	}
	{
		static float z1, z2; // filter section state
		float x = output - -1.20596630*z1 - 0.60558332*z2;
		output = 1.00000000*x + 2.00000000*z1 + 1.00000000*z2;
		z2 = z1;
		z1 = x;
	}
	{
		static float z1, z2; // filter section state
		float x = output - -1.97690645*z1 - 0.97706395*z2;
		output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
		z2 = z1;
		z1 = x;
	}
	{
		static float z1, z2; // filter section state
		float x = output - -1.99071687*z1 - 0.99086813*z2;
		output = 1.00000000*x + -2.00000000*z1 + 1.00000000*z2;
		z2 = z1;
		z1 = x;
	}
	return output;
}

## Python code (EEG_auth.py)
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

# How it works:
1. click on set password
2. Think Twice about the specific thing
3. click on unlock
 -> unlocks is you thought same as before
 -> remains locked if different