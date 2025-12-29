import sys
import serial
import numpy as np
from collections import deque

from scipy.signal import welch
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

from PyQt5.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton,
    QVBoxLayout, QHBoxLayout, QProgressBar, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
import pyqtgraph as pg


# ================= CONFIG =================
PORT = "COM5"
BAUD = 115200
FS = 256
WINDOW_SEC = 4
ENROLL_TRIALS = 6
AUTH_THRESHOLD = -0.15
# =========================================


# ========== FEATURE EXTRACTION ==========
def extract_features(signal):
    signal = signal - np.mean(signal)
    freqs, psd = welch(signal, FS, nperseg=512)

    def band(lo, hi):
        idx = (freqs >= lo) & (freqs <= hi)
        return np.mean(psd[idx])

    return np.array([
        band(0.5, 4),    # Delta
        band(4, 8),      # Theta
        band(8, 13),     # Alpha
        band(13, 30)     # Beta
    ])


# ========== EEG THREAD ==========
class EEGReader(QThread):
    new_sample = pyqtSignal(float)

    def run(self):
        ser = serial.Serial(PORT, BAUD)
        while True:
            try:
                self.new_sample.emit(
                    float(ser.readline().decode().strip())
                )
            except:
                pass


# ========== MAIN APPLICATION ==========
class BrainAuth(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("EEG Brain Password Authentication System")
        self.setMinimumSize(1200, 720)
        self.resize(1400, 800)

        self.buffer = deque(maxlen=FS * WINDOW_SEC)
        self.enroll_features = []
        self.enroll_count = 0

        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")

        self.init_ui()
        self.start_eeg()

    # ---------- UI ----------
    def init_ui(self):
        self.setStyleSheet("""
            QWidget { background-color: #0f172a; color: #e5e7eb; }
            QLabel#Title { font-size: 28px; font-weight: 600; }
            QLabel#Status { font-size: 26px; font-weight: bold; }
            QPushButton {
                background-color: #1e293b;
                padding: 12px;
                border-radius: 10px;
                font-size: 15px;
            }
            QPushButton:hover { background-color: #334155; }
            QProgressBar {
                height: 12px;
                background: #020617;
                border-radius: 6px;
            }
            QProgressBar::chunk {
                background-color: #38bdf8;
                border-radius: 6px;
            }
        """)

        title = QLabel("🧠 EEG Brain Password Authentication")
        title.setObjectName("Title")
        title.setAlignment(Qt.AlignCenter)

        self.status = QLabel("🔒 SYSTEM LOCKED")
        self.status.setObjectName("Status")
        self.status.setAlignment(Qt.AlignCenter)

        self.info = QLabel("Idle")
        self.info.setAlignment(Qt.AlignCenter)

        self.progress = QProgressBar()

        self.btn_enroll = QPushButton("Enroll Brain Password")
        self.btn_auth = QPushButton("Authenticate")
        self.btn_plot = QPushButton("Show EEG Signal")

        self.btn_enroll.clicked.connect(self.start_enroll)
        self.btn_auth.clicked.connect(self.authenticate)
        self.btn_plot.clicked.connect(self.toggle_plot)

        # -------- EEG Plot (hidden by default) --------
        self.plot = pg.PlotWidget()
        self.plot.setBackground('#020617')
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setYRange(-120, 120)
        self.plot.setMinimumHeight(320)
        self.curve = self.plot.plot(
            pen=pg.mkPen(color='#38bdf8', width=2)
        )
        self.plot.hide()

        # -------- Card --------
        card = QFrame()
        card.setStyleSheet("""
            background:#020617;
            border-radius:18px;
            padding:25px;
        """)
        card_layout = QVBoxLayout(card)
        card_layout.addWidget(self.status)
        card_layout.addWidget(self.info)
        card_layout.addWidget(self.progress)

        # -------- Layout --------
        btns = QHBoxLayout()
        btns.addWidget(self.btn_enroll)
        btns.addWidget(self.btn_auth)
        btns.addWidget(self.btn_plot)

        main = QVBoxLayout()
        main.addWidget(title)
        main.addSpacing(10)
        main.addWidget(card)
        main.addSpacing(10)
        main.addLayout(btns)
        main.addWidget(self.plot)

        self.setLayout(main)

        # -------- Timers --------
        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(30)

        self.enroll_timer = QTimer()
        self.enroll_timer.timeout.connect(self.record_enroll_trial)

    # ---------- EEG ----------
    def start_eeg(self):
        self.reader = EEGReader()
        self.reader.new_sample.connect(self.buffer.append)
        self.reader.start()

    def update_plot(self):
        if self.plot.isVisible() and len(self.buffer) > 10:
            self.curve.setData(list(self.buffer))

    def toggle_plot(self):
        visible = not self.plot.isVisible()
        self.plot.setVisible(visible)
        self.btn_plot.setText(
            "Hide EEG Signal" if visible else "Show EEG Signal"
        )

    # ---------- Enrollment ----------
    def start_enroll(self):
        self.enroll_features.clear()
        self.enroll_count = 0
        self.progress.setValue(0)
        self.info.setText("Enrolling... Maintain SAME mental task")
        self.enroll_timer.start(WINDOW_SEC * 1000)

    def record_enroll_trial(self):
        feat = extract_features(np.array(self.buffer))
        self.enroll_features.append(feat)
        self.enroll_count += 1

        self.progress.setValue(
            int((self.enroll_count / ENROLL_TRIALS) * 100)
        )

        if self.enroll_count >= ENROLL_TRIALS:
            self.enroll_timer.stop()
            X = self.scaler.fit_transform(
                np.array(self.enroll_features)
            )
            self.model.fit(X)
            self.info.setText("Brain Password Enrolled")

    # ---------- Authentication ----------
    def authenticate(self):
        if not self.enroll_features:
            self.info.setText("Enroll first!")
            return

        self.info.setText("Authenticating...")
        QTimer.singleShot(WINDOW_SEC * 1000, self.finish_auth)

    def finish_auth(self):
        feat = extract_features(np.array(self.buffer))
        feat = self.scaler.transform([feat])
        score = self.model.decision_function(feat)[0]

        if score > AUTH_THRESHOLD:
            self.status.setText("🔓 SYSTEM UNLOCKED")
            self.status.setStyleSheet(
                "color:#22c55e; font-size:26px;"
            )
            self.info.setText(
                f"Access Granted (score={score:.2f})"
            )
        else:
            self.status.setText("🔒 SYSTEM LOCKED")
            self.status.setStyleSheet(
                "color:#ef4444; font-size:26px;"
            )
            self.info.setText(
                f"Access Denied (score={score:.2f})"
            )


# ========== RUN ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BrainAuth()
    win.show()
    sys.exit(app.exec_())
