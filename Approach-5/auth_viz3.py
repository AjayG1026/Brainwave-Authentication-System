"""
Enhanced Brain-Computer Interface Authentication System v3.0
Advanced EEG-Based Biometric Security with Live Wave Visualization
"""

import serial
import numpy as np
import time
import pickle
from scipy.fft import fft, fftfreq
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import threading
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext, filedialog
import winsound
from datetime import datetime
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.animation as animation


class AudioFeedback:
    """Handles all audio feedback for the system"""
    
    @staticmethod
    def play_beep(frequency=1000, duration=200):
        """Play a simple beep"""
        try:
            winsound.Beep(frequency, duration)
        except:
            print('\a')
    
    @staticmethod
    def countdown_beep():
        AudioFeedback.play_beep(800, 150)
    
    @staticmethod
    def start_recording():
        AudioFeedback.play_beep(1200, 300)
    
    @staticmethod
    def stop_recording():
        AudioFeedback.play_beep(600, 300)
    
    @staticmethod
    def success():
        for freq in [800, 1000, 1200]:
            AudioFeedback.play_beep(freq, 150)
            time.sleep(0.05)
    
    @staticmethod
    def failure():
        for freq in [400, 300, 200]:
            AudioFeedback.play_beep(freq, 200)
            time.sleep(0.05)
    
    @staticmethod
    def task_change():
        AudioFeedback.play_beep(1000, 200)
        time.sleep(0.1)
        AudioFeedback.play_beep(1000, 200)


class BrainPasswordCore:
    """Core authentication logic with improved password-like behavior"""
    
    def __init__(self, port='COM5', baud_rate=115200, sample_rate=256):
        self.port = port
        self.baud_rate = baud_rate
        self.sample_rate = sample_rate
        self.serial_conn = None
        self.is_connected = False
        self.is_trained = False
        
        # ML Components
        self.scaler = StandardScaler()
        self.classifier = SVC(kernel='rbf', gamma='scale', probability=True, C=15, class_weight='balanced')
        
        # Feature bands (Hz)
        self.bands = {
            'delta': (0.5, 4),
            'theta': (4, 8),
            'alpha': (8, 13),
            'beta': (13, 30),
            'gamma': (30, 45)
        }
        
        # Data collection
        self.is_collecting = False
        self.collected_data = []
        self.data_quality_threshold = 50
        
        # Real-time visualization buffer
        self.realtime_buffer = []
        self.max_buffer_size = 512
        self.is_reading = False
        self.read_thread = None
        self.buffer_lock = threading.Lock()
        
        # Statistics
        self.training_history = []
        self.auth_attempts = []
        
        # Store training templates for strict matching
        self.task1_template = None
        self.task2_template = None
    
    def connect(self):
        """Establish serial connection with Arduino"""
        try:
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
                time.sleep(0.5)
            
            self.serial_conn = serial.Serial(self.port, self.baud_rate, timeout=0.1)
            time.sleep(2)
            self.serial_conn.reset_input_buffer()
            self.is_connected = True
            return True, f"Connected to {self.port}"
        except Exception as e:
            self.is_connected = False
            return False, f"Connection failed: {str(e)}"
    
    def disconnect(self):
        """Close serial connection"""
        try:
            self.stop_continuous_reading()
            if self.serial_conn and self.serial_conn.is_open:
                self.serial_conn.close()
            self.is_connected = False
            return True
        except:
            return False
    
    def read_eeg_sample(self):
        """Read a single EEG sample from Arduino"""
        try:
            if self.serial_conn and self.serial_conn.is_open and self.serial_conn.in_waiting:
                line = self.serial_conn.readline().decode('utf-8', errors='ignore').strip()
                if line:
                    return float(line)
        except:
            pass
        return None
    
    def start_continuous_reading(self):
        """Start continuous EEG reading for visualization"""
        if self.is_reading:
            return
        
        self.is_reading = True
        
        def read_loop():
            while self.is_reading and self.is_connected:
                sample = self.read_eeg_sample()
                if sample is not None:
                    with self.buffer_lock:
                        self.realtime_buffer.append(sample)
                        if len(self.realtime_buffer) > self.max_buffer_size:
                            self.realtime_buffer.pop(0)
                time.sleep(0.001)
        
        self.read_thread = threading.Thread(target=read_loop, daemon=True)
        self.read_thread.start()
    
    def stop_continuous_reading(self):
        """Stop continuous reading"""
        self.is_reading = False
        if self.read_thread:
            self.read_thread.join(timeout=1.0)
    
    def get_realtime_buffer(self):
        """Get current realtime buffer for visualization"""
        with self.buffer_lock:
            return np.array(self.realtime_buffer.copy()) if self.realtime_buffer else np.array([])
    
    def collect_data_continuous(self, duration, callback=None):
        """Collect EEG data for specified duration with progress callback"""
        self.collected_data = []
        self.is_collecting = True
        start_time = time.time()
        
        while self.is_collecting and (time.time() - start_time) < duration:
            sample = self.read_eeg_sample()
            if sample is not None:
                self.collected_data.append(sample)
            
            if callback:
                elapsed = time.time() - start_time
                progress = (elapsed / duration) * 100
                callback(progress, len(self.collected_data))
            
            time.sleep(0.001)
        
        self.is_collecting = False
        return np.array(self.collected_data)
    
    def stop_collection(self):
        """Stop data collection"""
        self.is_collecting = False
    
    def assess_data_quality(self, eeg_data):
        """Assess quality of collected data"""
        if len(eeg_data) < self.data_quality_threshold:
            return False, f"Insufficient data: {len(eeg_data)} samples (need {self.data_quality_threshold})"
        
        if np.std(eeg_data) < 0.1:
            return False, "Signal too flat - check electrode connection"
        
        if np.max(np.abs(eeg_data)) > 900:
            return False, "Signal saturated - reduce gain or check connections"
        
        return True, f"Good quality: {len(eeg_data)} samples"
    
    def extract_features(self, eeg_data):
        """Extract comprehensive frequency domain features from EEG data"""
        if len(eeg_data) < 10:
            return None
        
        # Compute FFT
        N = len(eeg_data)
        yf = fft(eeg_data)
        xf = fftfreq(N, 1/self.sample_rate)
        
        # Take only positive frequencies
        pos_mask = xf >= 0
        xf = xf[pos_mask]
        power = np.abs(yf[pos_mask])**2
        
        features = []
        
        # Extract power in each frequency band
        for band_name, (low, high) in self.bands.items():
            band_mask = (xf >= low) & (xf <= high)
            band_power = np.sum(power[band_mask])
            features.append(band_power)
        
        # Relative band powers (ratios)
        total_power = np.sum(power) + 1e-10
        for band_name, (low, high) in self.bands.items():
            band_mask = (xf >= low) & (xf <= high)
            band_power = np.sum(power[band_mask])
            features.append(band_power / total_power)
        
        # Additional statistical features
        features.extend([
            np.mean(eeg_data),
            np.std(eeg_data),
            np.var(eeg_data),
            np.max(eeg_data) - np.min(eeg_data),
            np.percentile(eeg_data, 25),
            np.percentile(eeg_data, 50),
            np.percentile(eeg_data, 75),
            np.mean(np.abs(np.diff(eeg_data))),
            np.std(np.diff(eeg_data)),
            len(np.where(np.diff(np.sign(eeg_data)))[0]),
        ])
        
        return np.array(features)
    
    def compute_band_powers(self, eeg_data):
        """Compute power in each frequency band for visualization"""
        if len(eeg_data) < 10:
            return {band: 0 for band in self.bands.keys()}
        
        N = len(eeg_data)
        yf = fft(eeg_data)
        xf = fftfreq(N, 1/self.sample_rate)
        
        pos_mask = xf >= 0
        xf = xf[pos_mask]
        power = np.abs(yf[pos_mask])**2
        
        band_powers = {}
        for band_name, (low, high) in self.bands.items():
            band_mask = (xf >= low) & (xf <= high)
            band_powers[band_name] = np.sum(power[band_mask])
        
        return band_powers
    
    def train_model(self, task1_data_list, task2_data_list):
        """Train authentication model with strict password-like behavior"""
        task1_features = []
        task2_features = []
        
        for data in task1_data_list:
            if len(data) > 0:
                features = self.extract_features(data)
                if features is not None:
                    task1_features.append(features)
        
        for data in task2_data_list:
            if len(data) > 0:
                features = self.extract_features(data)
                if features is not None:
                    task2_features.append(features)
        
        if len(task1_features) == 0 or len(task2_features) == 0:
            return False, "Insufficient data collected"
        
        # Store templates
        self.task1_template = np.mean(task1_features, axis=0)
        self.task2_template = np.mean(task2_features, axis=0)
        
        # Prepare training data
        X_positive = np.vstack(task1_features + task2_features)
        y_positive = np.ones(len(X_positive))
        
        # Generate negative samples
        X_negative = []
        n_negative = len(X_positive) * 3
        
        for _ in range(n_negative):
            choice = np.random.randint(0, 4)
            
            if choice == 0:
                neg_sample = np.random.normal(
                    np.mean(X_positive, axis=0),
                    np.std(X_positive, axis=0) * 2.0,
                    X_positive.shape[1]
                )
            elif choice == 1:
                neg_sample = X_positive[np.random.randint(0, len(X_positive))].copy()
                np.random.shuffle(neg_sample)
            elif choice == 2:
                neg_sample = -X_positive[np.random.randint(0, len(X_positive))]
            else:
                neg_sample = X_positive[np.random.randint(0, len(X_positive))].copy()
                noise = np.random.normal(0, np.std(X_positive)*0.8, neg_sample.shape)
                neg_sample = neg_sample + noise
            
            X_negative.append(neg_sample)
        
        X_negative = np.array(X_negative)
        y_negative = np.zeros(len(X_negative))
        
        # Combine
        X = np.vstack([X_positive, X_negative])
        y = np.concatenate([y_positive, y_negative])
        
        # Split and train
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.25, random_state=42, stratify=y
        )
        
        self.scaler.fit(X_train)
        X_train_scaled = self.scaler.transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        self.classifier.fit(X_train_scaled, y_train)
        
        train_score = self.classifier.score(X_train_scaled, y_train)
        test_score = self.classifier.score(X_test_scaled, y_test)
        
        self.is_trained = True
        
        self.training_history.append({
            'timestamp': datetime.now(),
            'train_score': train_score,
            'test_score': test_score,
            'samples': len(X_positive)
        })
        
        return True, f"Training: {train_score*100:.1f}%, Testing: {test_score*100:.1f}%"
    
    def authenticate_data(self, auth_data_list):
        """Authenticate using collected data with strict password-like matching"""
        if not self.is_trained:
            return False, 0.0, "Model not trained"
        
        if len(auth_data_list) != 2:
            return False, 0.0, "Incomplete authentication sequence"
        
        auth_features = []
        for data in auth_data_list:
            if len(data) > 0:
                features = self.extract_features(data)
                if features is not None:
                    auth_features.append(features)
        
        if len(auth_features) != 2:
            return False, 0.0, "Invalid authentication data"
        
        X_auth = np.array(auth_features)
        X_auth_scaled = self.scaler.transform(X_auth)
        
        predictions = self.classifier.predict(X_auth_scaled)
        probabilities = self.classifier.predict_proba(X_auth_scaled)
        
        confidences = probabilities[:, 1] if probabilities.shape[1] > 1 else probabilities[:, 0]
        
        # Calculate similarity to templates
        from numpy.linalg import norm
        
        task1_similarity = np.dot(X_auth_scaled[0], self.scaler.transform([self.task1_template])[0]) / \
                          (norm(X_auth_scaled[0]) * norm(self.scaler.transform([self.task1_template])[0]) + 1e-10)
        
        task2_similarity = np.dot(X_auth_scaled[1], self.scaler.transform([self.task2_template])[0]) / \
                          (norm(X_auth_scaled[1]) * norm(self.scaler.transform([self.task2_template])[0]) + 1e-10)
        
        task1_match = predictions[0] == 1 and confidences[0] > 0.60 and task1_similarity > 0.65
        task2_match = predictions[1] == 1 and confidences[1] > 0.60 and task2_similarity > 0.65
        avg_confidence = np.mean(confidences)
        
        success = task1_match and task2_match and avg_confidence > 0.65
        
        if success:
            message = "Access Granted - Brain Password Match!"
        else:
            if not task1_match:
                message = "Access Denied - Task 1 mismatch"
            elif not task2_match:
                message = "Access Denied - Task 2 mismatch"
            else:
                message = "Access Denied - Insufficient confidence"
        
        self.auth_attempts.append({
            'timestamp': datetime.now(),
            'success': success,
            'confidence': avg_confidence,
            'task1_conf': confidences[0],
            'task2_conf': confidences[1],
            'task1_sim': task1_similarity,
            'task2_sim': task2_similarity
        })
        
        return success, avg_confidence, message
    
    def save_model(self, filename='brain_password_model.pkl'):
        """Save trained model"""
        if not self.is_trained:
            return False, "No model to save"
        
        try:
            model_data = {
                'scaler': self.scaler,
                'classifier': self.classifier,
                'sample_rate': self.sample_rate,
                'bands': self.bands,
                'task1_template': self.task1_template,
                'task2_template': self.task2_template,
                'training_history': self.training_history,
                'auth_attempts': self.auth_attempts,
                'version': '3.0'
            }
            with open(filename, 'wb') as f:
                pickle.dump(model_data, f)
            return True, f"Model saved to {filename}"
        except Exception as e:
            return False, f"Save failed: {str(e)}"
    
    def load_model(self, filename='brain_password_model.pkl'):
        """Load trained model"""
        try:
            with open(filename, 'rb') as f:
                model_data = pickle.load(f)
            
            self.scaler = model_data['scaler']
            self.classifier = model_data['classifier']
            self.sample_rate = model_data['sample_rate']
            self.bands = model_data['bands']
            
            if 'task1_template' in model_data:
                self.task1_template = model_data['task1_template']
            if 'task2_template' in model_data:
                self.task2_template = model_data['task2_template']
            if 'training_history' in model_data:
                self.training_history = model_data['training_history']
            if 'auth_attempts' in model_data:
                self.auth_attempts = model_data['auth_attempts']
            
            self.is_trained = True
            
            return True, f"Model loaded from {filename}"
        except Exception as e:
            return False, f"Load failed: {str(e)}"


class EEGVisualizationWindow:
    """Real-time EEG wave visualization window with live updates"""
    
    def __init__(self, parent, core):
        self.window = tk.Toplevel(parent)
        self.window.title("🌊 Real-Time EEG Wave Visualization")
        self.window.geometry("1000x700")
        self.window.configure(bg='#0a0e27')
        
        self.core = core
        self.is_running = False
        
        # Setup UI
        self.setup_ui()
        
        # Start continuous reading from Arduino
        self.core.start_continuous_reading()
        
        # Animation
        self.ani = None
        
        # Start visualization
        self.start_visualization()
        
        # Handle window close
        self.window.protocol("WM_DELETE_WINDOW", self.on_close)
    
    def setup_ui(self):
        """Setup visualization UI"""
        # Title
        title = tk.Label(
            self.window,
            text="🌊 Real-Time EEG Wave Visualization",
            font=('Arial', 18, 'bold'),
            bg='#0a0e27',
            fg='#00ff9f'
        )
        title.pack(pady=10)
        
        # Create matplotlib figure
        self.fig = Figure(figsize=(10, 6), facecolor='#0a0e27')
        
        # Time domain plot
        self.ax1 = self.fig.add_subplot(211, facecolor='#1a1f3a')
        self.ax1.set_title('EEG Signal (Time Domain)', color='#00ff9f', fontsize=12, fontweight='bold')
        self.ax1.set_xlabel('Sample', color='#ffffff')
        self.ax1.set_ylabel('Amplitude', color='#ffffff')
        self.ax1.tick_params(colors='#ffffff')
        self.ax1.grid(True, alpha=0.3, color='#ffffff')
        self.line1, = self.ax1.plot([], [], color='#00ff9f', linewidth=1.5)
        
        # Frequency domain plot
        self.ax2 = self.fig.add_subplot(212, facecolor='#1a1f3a')
        self.ax2.set_title('Frequency Band Powers', color='#00ff9f', fontsize=12, fontweight='bold')
        self.ax2.set_xlabel('Frequency Band', color='#ffffff')
        self.ax2.set_ylabel('Power', color='#ffffff')
        self.ax2.tick_params(colors='#ffffff')
        self.ax2.grid(True, alpha=0.3, color='#ffffff')
        
        self.fig.tight_layout(pad=2.0)
        
        # Embed in tkinter
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.window)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True, padx=10, pady=10)
        
        # Control buttons
        btn_frame = tk.Frame(self.window, bg='#0a0e27')
        btn_frame.pack(pady=10)
        
        self.pause_btn = tk.Button(
            btn_frame,
            text="⏸️ Pause",
            command=self.toggle_pause,
            bg='#ffa502',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5
        )
        self.pause_btn.pack(side='left', padx=5)
        
        close_btn = tk.Button(
            btn_frame,
            text="❌ Close",
            command=self.on_close,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 10, 'bold'),
            padx=15,
            pady=5
        )
        close_btn.pack(side='left', padx=5)
    
    def start_visualization(self):
        """Start real-time visualization"""
        self.is_running = True
        self.ani = animation.FuncAnimation(
            self.fig,
            self.update_plot,
            interval=50,
            blit=False,
            cache_frame_data=False
        )
        self.canvas.draw()
    
    def update_plot(self, frame):
        """Update plots with new data"""
        if not self.is_running:
            return
        
        # Get current buffer
        data = self.core.get_realtime_buffer()
        
        if len(data) > 10:
            # Update time domain plot
            time_points = np.arange(len(data))
            self.line1.set_data(time_points, data)
            self.ax1.set_xlim(0, max(len(data), 100))
            
            # Dynamic y-axis with padding
            data_min, data_max = np.min(data), np.max(data)
            padding = (data_max - data_min) * 0.1 if data_max != data_min else 10
            self.ax1.set_ylim(data_min - padding, data_max + padding)
            
            # Compute and update frequency band powers
            band_powers = self.core.compute_band_powers(data)
            
            self.ax2.clear()
            bands = list(band_powers.keys())
            powers = list(band_powers.values())
            
            colors = ['#3498db', '#9b59b6', '#2ecc71', '#f39c12', '#e74c3c']
            bars = self.ax2.bar(bands, powers, color=colors, alpha=0.8, edgecolor='white', linewidth=1.5)
            
            # Add value labels on bars
            max_power = max(powers) if powers else 1
            for bar, power in zip(bars, powers):
                height = bar.get_height()
                self.ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{power:.0f}',
                            ha='center', va='bottom', color='white', fontsize=9, fontweight='bold')
            
            self.ax2.set_title('Frequency Band Powers', color='#00ff9f', fontsize=12, fontweight='bold')
            self.ax2.set_xlabel('Frequency Band', color='#ffffff')
            self.ax2.set_ylabel('Power', color='#ffffff')
            self.ax2.tick_params(colors='#ffffff')
            self.ax2.set_facecolor('#1a1f3a')
            self.ax2.grid(True, alpha=0.3, color='#ffffff', axis='y')
            self.ax2.set_ylim(0, max_power * 1.2 if max_power > 0 else 100)
        
        return self.line1,
    
    def toggle_pause(self):
        """Toggle pause/resume"""
        self.is_running = not self.is_running
        if self.is_running:
            self.pause_btn.config(text="⏸️ Pause")
        else:
            self.pause_btn.config(text="▶️ Resume")
    
    def on_close(self):
        """Handle window close"""
        self.is_running = False
        if self.ani:
            self.ani.event_source.stop()
        self.core.stop_continuous_reading()
        self.window.destroy()


class BrainPasswordGUI:
    """Enhanced GUI Application v3.0"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🧠 Brain Password Authentication System v3.0")
        self.root.geometry("1000x750")
        self.root.configure(bg='#0a0e27')
        
        # Initialize core
        self.core = BrainPasswordCore(port='COM5')
        
        # Training data storage
        self.task1_recordings = []
        self.task2_recordings = []
        self.current_task = None
        self.current_recording = 0
        self.total_recordings = 5
        
        # Auth data storage
        self.auth_recordings = []
        
        # Audio enabled flag
        self.audio_enabled = tk.BooleanVar(value=True)
        
        # Visualization window
        self.viz_window = None
        
        # Setup UI
        self.setup_ui()
        
        # Auto-connect on start
        self.root.after(500, self.auto_connect)
    
    def setup_ui(self):
        """Setup the enhanced user interface"""
        # Title Frame
        title_frame = tk.Frame(self.root, bg='#1a1f3a', height=100)
        title_frame.pack(fill='x', pady=(0, 10))
        title_frame.pack_propagate(False)
        
        title_label = tk.Label(
            title_frame,
            text="🧠 BRAIN PASSWORD AUTHENTICATION",
            font=('Arial', 26, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f'
        )
        title_label.pack(expand=True, pady=(10, 0))
        
        subtitle = tk.Label(
            title_frame,
            text="EEG-Based Biometric Security System v3.0 | Enhanced Password Matching",
            font=('Arial', 11),
            bg='#1a1f3a',
            fg='#7dd3fc'
        )
        subtitle.pack(pady=(0, 10))
        
        # Main Container
        main_container = tk.Frame(self.root, bg='#0a0e27')
        main_container.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Left Panel
        left_panel = tk.Frame(main_container, bg='#1a1f3a', relief='raised', bd=3)
        left_panel.pack(side='left', fill='both', expand=True, padx=(0, 10))
        
        # Status Section
        status_frame = tk.LabelFrame(
            left_panel,
            text="⚡ Connection Status",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f',
            relief='groove',
            bd=2
        )
        status_frame.pack(fill='x', padx=10, pady=10)
        
        self.status_label = tk.Label(
            status_frame,
            text="● Disconnected",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#ff4757'
        )
        self.status_label.pack(pady=10)
        
        btn_frame = tk.Frame(status_frame, bg='#1a1f3a')
        btn_frame.pack(pady=5)
        
        self.connect_btn = tk.Button(
            btn_frame,
            text="Connect to Arduino",
            command=self.connect_arduino,
            bg='#00ff9f',
            fg='#000000',
            font=('Arial', 10, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            padx=15,
            pady=5
        )
        self.connect_btn.pack(side='left', padx=5)
        
        # Settings Section
        settings_frame = tk.LabelFrame(
            left_panel,
            text="⚙️ Settings",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f',
            relief='groove',
            bd=2
        )
        settings_frame.pack(fill='x', padx=10, pady=10)
        
        audio_check = tk.Checkbutton(
            settings_frame,
            text="🔊 Audio Feedback Enabled",
            variable=self.audio_enabled,
            font=('Arial', 10),
            bg='#1a1f3a',
            fg='#ffffff',
            selectcolor='#2c3e50',
            activebackground='#1a1f3a',
            activeforeground='#00ff9f'
        )
        audio_check.pack(pady=10, padx=10, anchor='w')
        
        # Model Status
        model_frame = tk.LabelFrame(
            left_panel,
            text="🤖 Model Status",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f',
            relief='groove',
            bd=2
        )
        model_frame.pack(fill='x', padx=10, pady=10)
        
        self.model_status_label = tk.Label(
            model_frame,
            text="⚠ Not Trained",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#ffa502'
        )
        self.model_status_label.pack(pady=10)
        
        # Action Buttons
        button_frame = tk.LabelFrame(
            left_panel,
            text="🎯 Actions",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f',
            relief='groove',
            bd=2
        )
        button_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.train_btn = tk.Button(
            button_frame,
            text="🎓 Train Brain Password",
            command=self.start_training,
            bg='#2ecc71',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            state='disabled',
            padx=10,
            pady=8
        )
        self.train_btn.pack(pady=8, padx=10, fill='x')
        
        self.auth_btn = tk.Button(
            button_frame,
            text="🔐 Authenticate",
            command=self.start_authentication,
            bg='#3498db',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            state='disabled',
            padx=10,
            pady=8
        )
        self.auth_btn.pack(pady=8, padx=10, fill='x')
        
        self.viz_btn = tk.Button(
            button_frame,
            text="🌊 View EEG Waves",
            command=self.show_visualization,
            bg='#1abc9c',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            state='disabled',
            padx=10,
            pady=8
        )
        self.viz_btn.pack(pady=8, padx=10, fill='x')
        
        self.save_btn = tk.Button(
            button_frame,
            text="💾 Save Model",
            command=self.save_model,
            bg='#f39c12',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            state='disabled',
            padx=10,
            pady=8
        )
        self.save_btn.pack(pady=8, padx=10, fill='x')
        
        self.load_btn = tk.Button(
            button_frame,
            text="📁 Load Model",
            command=self.load_model,
            bg='#9b59b6',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            state='disabled',
            padx=10,
            pady=8
        )
        self.load_btn.pack(pady=8, padx=10, fill='x')
        
        self.stats_btn = tk.Button(
            button_frame,
            text="📊 View Statistics",
            command=self.show_statistics,
            bg='#e74c3c',
            fg='white',
            font=('Arial', 11, 'bold'),
            relief='raised',
            bd=3,
            cursor='hand2',
            padx=10,
            pady=8
        )
        self.stats_btn.pack(pady=8, padx=10, fill='x')
        
        # Right Panel
        right_panel = tk.Frame(main_container, bg='#1a1f3a', relief='raised', bd=3)
        right_panel.pack(side='right', fill='both', expand=True)
        
        # Activity Log
        log_frame = tk.LabelFrame(
            right_panel,
            text="📜 Activity Log",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f',
            relief='groove',
            bd=2
        )
        log_frame.pack(fill='both', expand=True, padx=10, pady=10)
        
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            bg='#0d1117',
            fg='#00ff00',
            font=('Consolas', 9),
            relief='sunken',
            bd=2,
            state='disabled',
            wrap='word'
        )
        self.log_text.pack(fill='both', expand=True, padx=5, pady=5)
        
        # Progress Section
        progress_frame = tk.LabelFrame(
            right_panel,
            text="⏳ Progress",
            font=('Arial', 12, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f',
            relief='groove',
            bd=2
        )
        progress_frame.pack(fill='x', padx=10, pady=(0, 10))
        
        self.progress_label = tk.Label(
            progress_frame,
            text="Ready",
            font=('Arial', 11, 'bold'),
            bg='#1a1f3a',
            fg='#ffffff'
        )
        self.progress_label.pack(pady=5)
        
        style = ttk.Style()
        style.theme_use('clam')
        style.configure("Custom.Horizontal.TProgressbar",
                       troughcolor='#2c3e50',
                       background='#00ff9f',
                       bordercolor='#00ff9f',
                       lightcolor='#00ff9f',
                       darkcolor='#00ff9f')
        
        self.progress_bar = ttk.Progressbar(
            progress_frame,
            mode='determinate',
            length=400,
            style="Custom.Horizontal.TProgressbar"
        )
        self.progress_bar.pack(pady=5, padx=10, fill='x')
        
        self.sample_label = tk.Label(
            progress_frame,
            text="Samples: 0 | Quality: N/A",
            font=('Arial', 9),
            bg='#1a1f3a',
            fg='#95a5a6'
        )
        self.sample_label.pack(pady=2)
        
        self.log("System initialized. Connect to Arduino to begin.", '#00ff9f')
        self.log("v3.0 - Enhanced with strict password matching", '#7dd3fc')
    
    def play_audio(self, audio_type):
        """Play audio feedback if enabled"""
        if not self.audio_enabled.get():
            return
        
        def play_thread():
            if audio_type == 'countdown':
                AudioFeedback.countdown_beep()
            elif audio_type == 'start':
                AudioFeedback.start_recording()
            elif audio_type == 'stop':
                AudioFeedback.stop_recording()
            elif audio_type == 'success':
                AudioFeedback.success()
            elif audio_type == 'failure':
                AudioFeedback.failure()
            elif audio_type == 'task_change':
                AudioFeedback.task_change()
        
        thread = threading.Thread(target=play_thread, daemon=True)
        thread.start()
    
    def log(self, message, color='#00ff00'):
        """Add message to activity log"""
        self.log_text.config(state='normal')
        timestamp = time.strftime('%H:%M:%S')
        self.log_text.insert('end', f"[{timestamp}] {message}\n")
        self.log_text.tag_add(color, f"end-{len(message)+2}c", "end-1c")
        self.log_text.tag_config(color, foreground=color)
        self.log_text.see('end')
        self.log_text.config(state='disabled')
        self.root.update()
    
    def auto_connect(self):
        """Automatically try to connect on startup"""
        self.connect_arduino()
    
    def connect_arduino(self):
        """Connect to Arduino"""
        self.log("Attempting to connect to Arduino...", '#7dd3fc')
        success, message = self.core.connect()
        
        if success:
            self.status_label.config(text="● Connected", fg='#2ecc71')
            self.connect_btn.config(text="Disconnect", command=self.disconnect_arduino, bg='#e74c3c')
            self.train_btn.config(state='normal')
            self.load_btn.config(state='normal')
            self.viz_btn.config(state='normal')
            self.log(f"✓ {message}", '#2ecc71')
            self.play_audio('success')
        else:
            self.status_label.config(text="● Disconnected", fg='#ff4757')
            self.log(f"✗ {message}", '#ff4757')
            messagebox.showerror("Connection Error", message)
            self.play_audio('failure')
    
    def disconnect_arduino(self):
        """Disconnect from Arduino"""
        self.core.disconnect()
        self.status_label.config(text="● Disconnected", fg='#ff4757')
        self.connect_btn.config(text="Connect to Arduino", command=self.connect_arduino, bg='#00ff9f')
        self.train_btn.config(state='disabled')
        self.auth_btn.config(state='disabled')
        self.viz_btn.config(state='disabled')
        self.log("Disconnected from Arduino", '#ffa502')
    
    def show_visualization(self):
        """Show EEG wave visualization window"""
        if not self.core.is_connected:
            messagebox.showwarning("Not Connected", "Please connect to Arduino first!")
            return
        
        if self.viz_window is None or not tk.Toplevel.winfo_exists(self.viz_window.window):
            self.viz_window = EEGVisualizationWindow(self.root, self.core)
            self.log("✓ EEG visualization window opened", '#2ecc71')
        else:
            self.viz_window.window.lift()
    
    def start_training(self):
        """Start training process"""
        self.task1_recordings = []
        self.task2_recordings = []
        self.current_recording = 0
        
        self.log("="*60, '#00ff9f')
        self.log("TRAINING PHASE STARTED", '#00ff9f')
        self.log("="*60, '#00ff9f')
        self.log("\n⚠️ IMPORTANT: Keep your eyes CLOSED during the entire session!", '#ffa502')
        self.log("You are creating your unique BRAIN PASSWORD.", '#ffa502')
        self.log("Remember these mental tasks - you must repeat them exactly to unlock!\n", '#ffa502')
        
        # Disable buttons
        self.train_btn.config(state='disabled')
        self.auth_btn.config(state='disabled')
        
        # Start Task 1
        self.current_task = 1
        self.log("📋 TASK 1: RELAXED STATE (Your Password Part 1)", '#7dd3fc')
        self.log("Instructions: Close your eyes and relax. Clear your mind.", '#ffffff')
        self.log("Think of nothing. Let your mind drift peacefully.", '#ffffff')
        self.root.after(3000, self.record_training_sample)
    
    def record_training_sample(self):
        """Record a single training sample"""
        if self.current_task == 1:
            task_name = "Relaxed State"
            duration = 5
        else:
            task_name = "Concentration State"
            duration = 5
        
        self.current_recording += 1
        self.log(f"\n--- Recording {self.current_recording}/{self.total_recordings} ({task_name}) ---", '#00ff9f')
        
        # Countdown with audio
        self.countdown_with_audio(3, lambda: self.collect_training_data(duration))
    
    def countdown_with_audio(self, count, callback):
        """Display countdown with audio beeps"""
        if count > 0:
            self.progress_label.config(text=f"Starting in {count}...")
            self.log(f"🔔 {count}...", '#ffff00')
            self.play_audio('countdown')
            self.root.after(1000, lambda: self.countdown_with_audio(count-1, callback))
        else:
            self.progress_label.config(text="🎙️ Recording NOW!")
            self.log("▶️ START! Recording in progress...", '#2ecc71')
            self.play_audio('start')
            callback()
    
    def collect_training_data(self, duration):
        """Collect training data in background thread"""
        self.progress_bar['value'] = 0
        
        def collection_thread():
            def progress_callback(progress, samples):
                self.root.after(0, lambda: self.update_progress(progress, samples))
            
            data = self.core.collect_data_continuous(duration, progress_callback)
            self.root.after(0, lambda: self.training_data_collected(data))
        
        thread = threading.Thread(target=collection_thread, daemon=True)
        thread.start()
    
    def update_progress(self, progress, samples):
        """Update progress bar and sample count"""
        self.progress_bar['value'] = progress
        quality_ok, quality_msg = self.core.assess_data_quality(self.core.collected_data)
        quality_status = "✓ Good" if quality_ok else "⚠ Poor"
        self.sample_label.config(text=f"Samples: {samples} | Quality: {quality_status}")
        self.root.update()
    
    def training_data_collected(self, data):
        """Handle collected training data"""
        self.play_audio('stop')
        
        quality_ok, quality_msg = self.core.assess_data_quality(data)
        
        if quality_ok:
            self.log(f"✓ Recording complete! {quality_msg}", '#2ecc71')
            
            # Store data
            if self.current_task == 1:
                self.task1_recordings.append(data)
            else:
                self.task2_recordings.append(data)
            
            # Check if need more recordings for current task
            if self.current_recording < self.total_recordings:
                self.log(f"Prepare for next recording...", '#7dd3fc')
                self.root.after(2000, self.record_training_sample)
            elif self.current_task == 1:
                # Move to Task 2
                self.play_audio('task_change')
                self.current_task = 2
                self.current_recording = 0
                self.log("\n" + "="*60, '#00ff9f')
                self.log("📋 TASK 2: CONCENTRATION STATE (Your Password Part 2)", '#7dd3fc')
                self.log("Instructions: Keep eyes CLOSED. Count backwards from 100 by 7.", '#ffffff')
                self.log("Focus intensely: 100 → 93 → 86 → 79 → 72 → 65...", '#ffffff')
                self.log("Concentrate deeply on each calculation!", '#ffa502')
                self.log("THIS IS YOUR PASSWORD - Remember it exactly!", '#ffa502')
                self.root.after(4000, self.record_training_sample)
            else:
                # Training complete
                self.finish_training()
        else:
            self.log(f"⚠ {quality_msg}", '#ffa502')
            self.log("Retrying this recording...", '#ffa502')
            self.current_recording -= 1
            self.root.after(2000, self.record_training_sample)
    
    def finish_training(self):
        """Complete training process"""
        self.log("\n" + "="*60, '#00ff9f')
        self.log("🧠 Training classifier with collected data...", '#7dd3fc')
        self.log("Creating your unique brain password template...", '#ffa502')
        
        success, message = self.core.train_model(self.task1_recordings, self.task2_recordings)
        
        if success:
            self.log(f"✓ Training complete! {message}", '#2ecc71')
            self.log("✓ Brain password successfully created!", '#2ecc71')
            self.log("Remember: You must think the SAME thoughts to unlock!", '#ffa502')
            self.model_status_label.config(text="✓ Trained & Ready", fg='#2ecc71')
            self.auth_btn.config(state='normal')
            self.save_btn.config(state='normal')
            self.play_audio('success')
            messagebox.showinfo("Success", 
                              "🎉 Training completed!\n\n" + message + 
                              "\n\nYour brain password is now set!\n" +
                              "You must repeat BOTH mental tasks exactly to authenticate.")
        else:
            self.log(f"✗ Training failed: {message}", '#ff4757')
            self.play_audio('failure')
            messagebox.showerror("Error", f"Training failed: {message}")
        
        self.train_btn.config(state='normal')
        self.progress_label.config(text="Ready")
        self.progress_bar['value'] = 0
    
    def start_authentication(self):
        """Start authentication process"""
        if not self.core.is_trained:
            messagebox.showwarning("Not Ready", "Please train the model first!")
            return
        
        self.auth_recordings = []
        
        self.log("\n" + "="*60, '#00ff9f')
        self.log("🔐 AUTHENTICATION ATTEMPT", '#00ff9f')
        self.log("="*60, '#00ff9f')
        self.log("⚠️ Close your eyes and perform your BRAIN PASSWORD!", '#ffa502')
        self.log("You must repeat BOTH mental tasks EXACTLY as trained.", '#ffa502')
        
        self.auth_btn.config(state='disabled')
        self.train_btn.config(state='disabled')
        
        self.log("\nPerform your brain password sequence:", '#7dd3fc')
        self.log("1. Relaxed state (eyes closed, clear mind) - 4 seconds", '#ffffff')
        
        self.root.after(2000, self.collect_auth_sample_1)
    
    def collect_auth_sample_1(self):
        """Collect first auth sample (relaxed)"""
        self.log("\n--- Task 1: Relaxed State ---", '#00ff9f')
        self.log("Clear your mind. Relax completely.", '#ffffff')
        self.countdown_with_audio(3, lambda: self.collect_auth_data(4, 1))
    
    def collect_auth_data(self, duration, task_num):
        """Collect authentication data"""
        self.progress_bar['value'] = 0
        
        def collection_thread():
            def progress_callback(progress, samples):
                self.root.after(0, lambda: self.update_progress(progress, samples))
            
            data = self.core.collect_data_continuous(duration, progress_callback)
            self.root.after(0, lambda: self.auth_data_collected(data, task_num))
        
        thread = threading.Thread(target=collection_thread, daemon=True)
        thread.start()
    
    def auth_data_collected(self, data, task_num):
        """Handle collected auth data"""
        self.play_audio('stop')
        self.log(f"✓ Recording complete! Collected {len(data)} samples", '#2ecc71')
        self.auth_recordings.append(data)
        
        if task_num == 1:
            # Proceed to task 2
            self.play_audio('task_change')
            self.log("\n2. Mental math - count backwards by 7 (4 seconds)", '#ffffff')
            self.root.after(2000, self.collect_auth_sample_2)
        else:
            # Authenticate
            self.perform_authentication()
    
    def collect_auth_sample_2(self):
        """Collect second auth sample (concentration)"""
        self.log("\n--- Task 2: Concentration State ---", '#00ff9f')
        self.log("Count backwards: 100 → 93 → 86 → 79...", '#ffffff')
        self.countdown_with_audio(3, lambda: self.collect_auth_data(4, 2))
    
    def perform_authentication(self):
        """Perform final authentication"""
        self.log("\n🔍 Analyzing brain patterns...", '#7dd3fc')
        self.log("Comparing with stored password template...", '#ffa502')
        
        success, confidence, message = self.core.authenticate_data(self.auth_recordings)
        
        # Get detailed info from last auth attempt
        if self.core.auth_attempts:
            last_attempt = self.core.auth_attempts[-1]
            task1_conf = last_attempt['task1_conf']
            task2_conf = last_attempt['task2_conf']
            task1_sim = last_attempt['task1_sim']
            task2_sim = last_attempt['task2_sim']
        
        self.log("\n" + "="*60, '#00ff9f')
        if success:
            self.log("✓ AUTHENTICATION SUCCESSFUL!", '#2ecc71')
            self.log(f"Overall Confidence: {confidence*100:.1f}%", '#2ecc71')
            self.log(f"Task 1 Match: {task1_conf*100:.1f}% (Similarity: {task1_sim*100:.1f}%)", '#2ecc71')
            self.log(f"Task 2 Match: {task2_conf*100:.1f}% (Similarity: {task2_sim*100:.1f}%)", '#2ecc71')
            self.log("🔓 ACCESS GRANTED - Brain password correct!", '#2ecc71')
            self.play_audio('success')
            messagebox.showinfo("Success", 
                              f"✅ ACCESS GRANTED\n\n" +
                              f"Overall Confidence: {confidence*100:.1f}%\n" +
                              f"Task 1: {task1_conf*100:.1f}%\n" +
                              f"Task 2: {task2_conf*100:.1f}%\n\n" +
                              "Brain password verified!\nWelcome back!")
        else:
            self.log("✗ AUTHENTICATION FAILED!", '#ff4757')
            self.log(f"Overall Confidence: {confidence*100:.1f}%", '#ff4757')
            self.log(f"Task 1 Match: {task1_conf*100:.1f}% (Similarity: {task1_sim*100:.1f}%)", '#ff4757')
            self.log(f"Task 2 Match: {task2_conf*100:.1f}% (Similarity: {task2_sim*100:.1f}%)", '#ff4757')
            self.log("🔒 ACCESS DENIED - Brain password incorrect!", '#ff4757')
            self.log("Tip: You must think the SAME thoughts as when you trained.", '#ffa502')
            self.play_audio('failure')
            messagebox.showerror("Failed", 
                               f"❌ ACCESS DENIED\n\n" +
                               f"Overall Confidence: {confidence*100:.1f}%\n" +
                               f"Task 1: {task1_conf*100:.1f}%\n" +
                               f"Task 2: {task2_conf*100:.1f}%\n\n" +
                               message + "\n\n" +
                               "Brain password does not match!\n" +
                               "You must perform BOTH tasks exactly as trained.")
        
        self.log("="*60, '#00ff9f')
        
        self.auth_btn.config(state='normal')
        self.train_btn.config(state='normal')
        self.progress_label.config(text="Ready")
        self.progress_bar['value'] = 0
    
    def save_model(self):
        """Save the trained model"""
        filename = filedialog.asksaveasfilename(
            defaultextension=".pkl",
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")],
            initialfile="brain_password_model.pkl"
        )
        
        if filename:
            success, message = self.core.save_model(filename)
            if success:
                self.log(f"✓ {message}", '#2ecc71')
                messagebox.showinfo("Success", message)
            else:
                self.log(f"✗ {message}", '#ff4757')
                messagebox.showerror("Error", message)
    
    def load_model(self):
        """Load a trained model"""
        filename = filedialog.askopenfilename(
            filetypes=[("Pickle files", "*.pkl"), ("All files", "*.*")]
        )
        
        if filename:
            success, message = self.core.load_model(filename)
            if success:
                self.log(f"✓ {message}", '#2ecc71')
                self.model_status_label.config(text="✓ Trained (Loaded)", fg='#2ecc71')
                self.auth_btn.config(state='normal')
                self.save_btn.config(state='normal')
                self.play_audio('success')
                messagebox.showinfo("Success", message)
            else:
                self.log(f"✗ {message}", '#ff4757')
                self.play_audio('failure')
                messagebox.showerror("Error", message)
    
    def show_statistics(self):
        """Show training and authentication statistics"""
        stats_window = tk.Toplevel(self.root)
        stats_window.title("📊 System Statistics")
        stats_window.geometry("700x600")
        stats_window.configure(bg='#1a1f3a')
        
        title = tk.Label(
            stats_window,
            text="📊 System Statistics & Performance Analysis",
            font=('Arial', 18, 'bold'),
            bg='#1a1f3a',
            fg='#00ff9f'
        )
        title.pack(pady=10)
        
        text_area = scrolledtext.ScrolledText(
            stats_window,
            bg='#0d1117',
            fg='#ffffff',
            font=('Consolas', 10),
            relief='sunken',
            bd=2
        )
        text_area.pack(fill='both', expand=True, padx=20, pady=10)
        
        # Training History
        text_area.insert('end', "="*60 + "\n")
        text_area.insert('end', "TRAINING HISTORY\n")
        text_area.insert('end', "="*60 + "\n\n")
        
        if self.core.training_history:
            for i, record in enumerate(self.core.training_history, 1):
                text_area.insert('end', f"Training Session {i}:\n")
                text_area.insert('end', f"  Timestamp: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                text_area.insert('end', f"  Train Accuracy: {record['train_score']*100:.2f}%\n")
                text_area.insert('end', f"  Test Accuracy: {record['test_score']*100:.2f}%\n")
                text_area.insert('end', f"  Training Samples: {record['samples']}\n\n")
        else:
            text_area.insert('end', "No training history available.\n\n")
        
        # Authentication History
        text_area.insert('end', "="*60 + "\n")
        text_area.insert('end', "AUTHENTICATION ATTEMPTS\n")
        text_area.insert('end', "="*60 + "\n\n")
        
        if self.core.auth_attempts:
            success_count = sum(1 for a in self.core.auth_attempts if a['success'])
            total_count = len(self.core.auth_attempts)
            success_rate = (success_count / total_count * 100) if total_count > 0 else 0
            
            text_area.insert('end', f"Total Attempts: {total_count}\n")
            text_area.insert('end', f"Successful: {success_count}\n")
            text_area.insert('end', f"Failed: {total_count - success_count}\n")
            text_area.insert('end', f"Success Rate: {success_rate:.1f}%\n\n")
            
            text_area.insert('end', "Recent Attempts (Last 15):\n")
            text_area.insert('end', "-" * 60 + "\n")
            for i, record in enumerate(reversed(self.core.auth_attempts[-15:]), 1):
                status = "✓ SUCCESS" if record['success'] else "✗ FAILED"
                text_area.insert('end', f"\n{i}. {status}\n")
                text_area.insert('end', f"   Time: {record['timestamp'].strftime('%Y-%m-%d %H:%M:%S')}\n")
                text_area.insert('end', f"   Overall Confidence: {record['confidence']*100:.2f}%\n")
                text_area.insert('end', f"   Task 1: Conf={record['task1_conf']*100:.1f}%, Sim={record['task1_sim']*100:.1f}%\n")
                text_area.insert('end', f"   Task 2: Conf={record['task2_conf']*100:.1f}%, Sim={record['task2_sim']*100:.1f}%\n")
        else:
            text_area.insert('end', "No authentication attempts yet.\n\n")
        
        text_area.insert('end', "\n" + "="*60 + "\n")
        text_area.insert('end', "LEGEND:\n")
        text_area.insert('end', "  Conf = Classifier Confidence (how sure the model is)\n")
        text_area.insert('end', "  Sim  = Template Similarity (how close to training data)\n")
        text_area.insert('end', "="*60 + "\n")
        
        text_area.config(state='disabled')


def main():
    """Main entry point"""
    root = tk.Tk()
    app = BrainPasswordGUI(root)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f'{width}x{height}+{x}+{y}')
    
    root.mainloop()


if __name__ == "__main__":
    main()