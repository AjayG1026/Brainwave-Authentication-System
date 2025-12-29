import sys
import serial
import numpy as np
from collections import deque

from scipy.signal import welch
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import mahalanobis

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

        self.buffer = deque(maxlen=FS * WINDOW_SEC)
        self.enroll_features = []
        self.enroll_count = 0

        self.scaler = StandardScaler()
        self.model = OneClassSVM(kernel="rbf", nu=0.1, gamma="scale")

        # Identity model
        self.mean_vec = None
        self.inv_cov = None

        # Auto-calibrated thresholds
        self.svm_threshold = None
        self.maha_threshold = None

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
        self.btn_auth.clicked.connect(self.start_auth)
        self.btn_plot.clicked.connect(self.toggle_plot)

        self.plot = pg.PlotWidget()
        self.plot.setBackground('#020617')
        self.plot.showGrid(x=True, y=True, alpha=0.25)
        self.plot.setYRange(-120, 120)
        self.plot.setMinimumHeight(320)
        self.curve = self.plot.plot(
            pen=pg.mkPen(color='#38bdf8', width=2)
        )
        self.plot.hide()

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

        btns = QHBoxLayout()
        btns.addWidget(self.btn_enroll)
        btns.addWidget(self.btn_auth)
        btns.addWidget(self.btn_plot)

        main = QVBoxLayout()
        main.addWidget(title)
        main.addWidget(card)
        main.addLayout(btns)
        main.addWidget(self.plot)

        self.setLayout(main)

        self.plot_timer = QTimer()
        self.plot_timer.timeout.connect(self.update_plot)
        self.plot_timer.start(30)

        self.enroll_timer = QTimer()
        self.enroll_timer.timeout.connect(self.record_enroll_trial)

        self.auth_timer = QTimer()
        self.auth_timer.timeout.connect(self.finish_auth)

    # ---------- EEG ----------
    def start_eeg(self):
        self.reader = EEGReader()
        self.reader.new_sample.connect(self.buffer.append)
        self.reader.start()

    def update_plot(self):
        if self.plot.isVisible() and len(self.buffer) > 10:
            self.curve.setData(list(self.buffer))

    def toggle_plot(self):
        self.plot.setVisible(not self.plot.isVisible())

    # ---------- Enrollment ----------
    def start_enroll(self):
        self.enroll_features.clear()
        self.enroll_count = 0
        self.progress.setValue(0)
        self.info.setText("Enrolling... SAME USER, SAME THOUGHT")
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

            # Identity model
            self.mean_vec = X.mean(axis=0)
            cov = np.cov(X.T) + np.eye(X.shape[1]) * 1e-3
            self.inv_cov = np.linalg.inv(cov)

            # -------- AUTO THRESHOLD CALIBRATION --------
            svm_scores = self.model.decision_function(X)
            self.svm_threshold = np.mean(svm_scores) - 2 * np.std(svm_scores)

            maha_scores = [
                mahalanobis(x, self.mean_vec, self.inv_cov) for x in X
            ]
            self.maha_threshold = np.mean(maha_scores) + 2.5 * np.std(maha_scores)

            self.info.setText("Brain Password Enrolled")

    # ---------- Authentication ----------
    def start_auth(self):
        if not self.enroll_features:
            self.info.setText("Enroll first!")
            return

        self.progress.setValue(0)
        self.info.setText("Authenticating... Think SAME TASK")

        self.auth_progress = 0
        self.auth_timer.start(int(WINDOW_SEC * 1000 / 100))

    def finish_auth(self):
        self.auth_progress += 1
        self.progress.setValue(self.auth_progress)

        if self.auth_progress < 100:
            return

        self.auth_timer.stop()

        feat = extract_features(np.array(self.buffer))
        feat_scaled = self.scaler.transform([feat])

        svm_score = self.model.decision_function(feat_scaled)[0]
        maha_dist = mahalanobis(
            feat_scaled[0], self.mean_vec, self.inv_cov
        )

        if svm_score > self.svm_threshold and maha_dist < self.maha_threshold:
            self.status.setText("🔓 SYSTEM UNLOCKED")
            self.status.setStyleSheet("color:#22c55e; font-size:26px;")
            self.info.setText(
                f"Access Granted | SVM={svm_score:.2f}, ID={maha_dist:.2f}"
            )
        else:
            self.status.setText("🔒 SYSTEM LOCKED")
            self.status.setStyleSheet("color:#ef4444; font-size:26px;")
            self.info.setText(
                f"Access Denied | SVM={svm_score:.2f}, ID={maha_dist:.2f}"
            )


# ========== RUN ==========
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = BrainAuth()
    win.show()
    sys.exit(app.exec_())
