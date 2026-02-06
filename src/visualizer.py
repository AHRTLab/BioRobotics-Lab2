"""
BioRobotics Lab 1 - Real-time EMG/IMU Visualizer & Data Collector
===================================================================

A Python-based visualizer for EMG and IMU biosignals streamed via LSL.

Features:
- Multi-stream support (EMG + IMU simultaneously)
- Real-time multi-channel plotting
- Adjustable time window and amplitude
- Signal envelope display (muscle activation)
- Data collection with metadata (participant, gesture, trial)
- Auto-incrementing trial numbers
- Organized file output

Usage:
    python visualizer.py                    # Auto-detect streams
    python visualizer.py --stream "Myo"     # Connect to streams starting with "Myo"
    python visualizer.py --mock             # Test with synthetic data

Author: BioRobotics Course
Updated: 2025
"""

import sys
import os
import time
import threading
from collections import deque
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime

import numpy as np

# Check for PyQt6/PyQtGraph availability
try:
    from PyQt6.QtWidgets import (
        QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
        QPushButton, QComboBox, QLabel, QSpinBox, QStatusBar,
        QGroupBox, QGridLayout, QCheckBox, QFileDialog, QMessageBox,
        QDoubleSpinBox, QLineEdit, QTabWidget, QListWidget, QListWidgetItem,
        QSplitter, QFrame
    )
    from PyQt6.QtCore import QTimer, Qt, pyqtSignal
    from PyQt6.QtGui import QFont, QColor
    import pyqtgraph as pg
    HAS_GUI = True
except ImportError:
    HAS_GUI = False
    print("Warning: PyQt6 or pyqtgraph not available. GUI disabled.")

try:
    import pylsl
    HAS_LSL = True
except ImportError:
    HAS_LSL = False
    print("Warning: pylsl not available. LSL functionality disabled.")


@dataclass
class VisualizerConfig:
    """Configuration for the visualizer."""
    window_seconds: float = 5.0
    update_rate_hz: float = 30.0
    max_channels: int = 16
    dark_mode: bool = True
    emg_amplitude: float = 128.0
    imu_amplitude: float = 2.0
    auto_scale: bool = False
    show_envelope: bool = False
    envelope_window_ms: float = 50.0
    line_width: int = 1


@dataclass
class RecordingMetadata:
    """Metadata for recorded data."""
    participant_id: str = ""
    gesture: str = ""
    trial_number: int = 1
    notes: str = ""


class RecordingBuffer:
    """Thread-safe buffer that tracks what's been recorded to avoid duplication."""
    
    def __init__(self, n_channels: int):
        self.n_channels = n_channels
        self.data: List[np.ndarray] = []  # List of sample arrays
        self.timestamps: List[float] = []  # List of timestamps
        self._lock = threading.Lock()
    
    def add_samples(self, samples: np.ndarray, timestamps: np.ndarray):
        """Add new samples. samples should be shape (n_samples, n_channels)."""
        with self._lock:
            for i, sample in enumerate(samples):
                self.data.append(sample)
                if i < len(timestamps):
                    self.timestamps.append(timestamps[i])
    
    def get_all_data(self) -> tuple:
        """Get all recorded data as numpy arrays."""
        with self._lock:
            if not self.data:
                return np.array([]), np.array([])
            data = np.array(self.data)
            timestamps = np.array(self.timestamps)
            return data, timestamps
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            self.data.clear()
            self.timestamps.clear()
    
    def __len__(self):
        return len(self.data)
    

class SignalBuffer:
    """Thread-safe buffer for streaming signal data."""
    
    def __init__(self, n_channels: int, max_samples: int = 10000):
        self.n_channels = n_channels
        self.max_samples = max_samples
        self.data = [deque(maxlen=max_samples) for _ in range(n_channels)]
        self.timestamps = deque(maxlen=max_samples)
        self._lock = threading.Lock()
    
    def add_samples(self, samples: np.ndarray, timestamps: np.ndarray):
        """Add new samples to the buffer."""
        with self._lock:
            for i, sample in enumerate(samples):
                for ch in range(min(len(sample), self.n_channels)):
                    self.data[ch].append(sample[ch])
                if i < len(timestamps):
                    self.timestamps.append(timestamps[i])
    
    def get_data(self, n_samples: int = None) -> tuple:
        """Get data from the buffer."""
        with self._lock:
            if n_samples is None:
                n_samples = len(self.timestamps)
            
            data = np.zeros((n_samples, self.n_channels))
            for ch in range(self.n_channels):
                ch_data = list(self.data[ch])[-n_samples:]
                data[:len(ch_data), ch] = ch_data
            
            timestamps = np.array(list(self.timestamps)[-n_samples:])
            return data, timestamps
    
    def clear(self):
        """Clear the buffer."""
        with self._lock:
            for ch_data in self.data:
                ch_data.clear()
            self.timestamps.clear()


class LSLStreamReader(threading.Thread):
    """Background thread for reading LSL data."""
    
    def __init__(self, stream_name: str, buffer: SignalBuffer):
        super().__init__(daemon=True)
        self.stream_name = stream_name
        self.buffer = buffer  # For visualization
        self.running = False
        self.inlet = None
        self.sample_count = 0
        self.sample_rate = 200
        self.error = None
        
        # Recording support
        self.recording = False
        self.recording_buffer: Optional[RecordingBuffer] = None
    
    def start_recording(self, n_channels: int):
        """Start recording samples."""
        self.recording_buffer = RecordingBuffer(n_channels)
        self.recording = True
        self._recording_samples = 0
    
    def stop_recording(self) -> tuple:
        """Stop recording and return collected data."""
        self.recording = False
        recorded_count = getattr(self, '_recording_samples', 0)
        if self.recording_buffer is not None:
            data, timestamps = self.recording_buffer.get_all_data()
            self.recording_buffer = None
            return data, timestamps
        return np.array([]), np.array([])
    
    def run(self):
        """Main thread loop."""
        self.running = True
        self.sample_count = 0
        
        try:
            print(f"Looking for stream: {self.stream_name}")
            try:
                streams = pylsl.resolve_byprop("name", self.stream_name, 1, 5.0)
            except TypeError:
                streams = pylsl.resolve_byprop("name", self.stream_name, timeout=5.0)
            
            if not streams:
                self.error = f"Stream '{self.stream_name}' not found"
                self.running = False
                return
            
            self.sample_rate = streams[0].nominal_srate()
            self.inlet = pylsl.StreamInlet(streams[0], max_buflen=360)
            print(f"Connected to {self.stream_name} ({self.sample_rate} Hz)")
            
            while self.running:
                samples, timestamps = self.inlet.pull_chunk(timeout=0.1)
                if samples:
                    samples_arr = np.array(samples)
                    timestamps_arr = np.array(timestamps)
                    
                    # Add to visualization buffer
                    self.buffer.add_samples(samples_arr, timestamps_arr)
                    self.sample_count += len(samples)
                    
                    # Add to recording buffer if active
                    if self.recording and self.recording_buffer is not None:
                        self.recording_buffer.add_samples(samples_arr, timestamps_arr)
                        self._recording_samples = getattr(self, '_recording_samples', 0) + len(samples)
        
        except Exception as e:
            self.error = str(e)
        
        finally:
            self.running = False
    
    def stop(self):
        """Stop the reader thread."""
        self.running = False


def compute_envelope(data: np.ndarray, window_samples: int) -> np.ndarray:
    """Compute signal envelope (rectify + smooth)."""
    if window_samples < 1:
        window_samples = 1
    
    rectified = np.abs(data)
    
    if len(rectified) < window_samples:
        return rectified
    
    envelope = np.zeros_like(rectified)
    for ch in range(rectified.shape[1]):
        cumsum = np.cumsum(np.insert(rectified[:, ch], 0, 0))
        envelope[window_samples-1:, ch] = (cumsum[window_samples:] - cumsum[:-window_samples]) / window_samples
        envelope[:window_samples-1, ch] = envelope[window_samples-1, ch]
    
    return envelope


if HAS_GUI and HAS_LSL:
    
    class StreamPanel(QWidget):
        """Panel for displaying a single stream."""
        
        def __init__(self, stream_type: str = "EMG", n_channels: int = 8, parent=None):
            super().__init__(parent)
            self.stream_type = stream_type
            self.n_channels = n_channels
            self.plots = []
            self.curves = []
            self.envelope_curves = []
            
            self.setup_ui()
        
        def setup_ui(self):
            layout = QVBoxLayout(self)
            layout.setContentsMargins(0, 0, 0, 0)
            
            self.plot_widget = pg.GraphicsLayoutWidget()
            layout.addWidget(self.plot_widget)
        
        def setup_plots(self, n_channels: int, amplitude: float, window_seconds: float,
                       line_width: int = 1, show_envelope: bool = False):
            """Setup plot widgets."""
            self.plot_widget.clear()
            self.plots = []
            self.curves = []
            self.envelope_curves = []
            self.n_channels = n_channels
            
            # Color palettes
            emg_colors = [
                '#e6194b', '#3cb44b', '#ffe119', '#4363d8',
                '#f58231', '#911eb4', '#42d4f4', '#f032e6'
            ]
            imu_colors = [
                '#ff6b6b', '#4ecdc4', '#45b7d1', '#96ceb4',  # Quaternion
                '#ffeaa7', '#dfe6e9', '#fd79a8',  # Accel
                '#a29bfe', '#6c5ce7', '#00b894'   # Gyro
            ]
            
            colors = emg_colors if self.stream_type == "EMG" else imu_colors
            
            # Channel labels
            if self.stream_type == "EMG":
                labels = [f"EMG {i+1}" for i in range(n_channels)]
            else:
                labels = ["Quat W", "Quat X", "Quat Y", "Quat Z",
                         "Accel X", "Accel Y", "Accel Z",
                         "Gyro X", "Gyro Y", "Gyro Z"][:n_channels]
            
            for i in range(n_channels):
                if i > 0:
                    self.plot_widget.nextRow()
                
                plot = self.plot_widget.addPlot(title=labels[i])
                plot.showGrid(x=True, y=True, alpha=0.3)
                plot.setXRange(-window_seconds, 0)
                plot.setYRange(-amplitude, amplitude)
                
                if i == n_channels - 1:
                    plot.setLabel('bottom', 'Time (s)')
                
                color = colors[i % len(colors)]
                curve = plot.plot(pen=pg.mkPen(color=color, width=line_width))
                
                env_curve = plot.plot(pen=pg.mkPen(color='#ffffff', width=2))
                env_curve.setVisible(show_envelope)
                
                self.plots.append(plot)
                self.curves.append(curve)
                self.envelope_curves.append(env_curve)
        
        def update_amplitude(self, amplitude: float):
            """Update Y-axis range."""
            for plot in self.plots:
                plot.setYRange(-amplitude, amplitude)
        
        def update_window(self, window_seconds: float):
            """Update X-axis range."""
            for plot in self.plots:
                plot.setXRange(-window_seconds, 0)


    class EMGVisualizer(QMainWindow):
        """Main visualizer window with multi-stream and data collection support."""
        
        def __init__(self, config: VisualizerConfig = None):
            super().__init__()
            self.config = config or VisualizerConfig()
            
            # Stream state
            self.streams: Dict[str, dict] = {}  # name -> {reader, buffer, panel, info}
            self.recording = False
            self.recorded_data: Dict[str, tuple] = {}  # stream_name -> (data, timestamps)
            self.record_start_time = None
            
            # Metadata
            self.metadata = RecordingMetadata()
            
            # Output directory
            self.output_dir = os.path.join(os.getcwd(), "recordings")
            
            # Setup UI
            self.setup_ui()
            
            # Update timer
            self.timer = QTimer()
            self.timer.timeout.connect(self.update_plots)
            
            # Rate tracking
            self.last_counts = {}
            self.last_rate_time = time.time()
            
            # Refresh streams
            self.refresh_streams()
        
        def setup_ui(self):
            """Create the user interface."""
            self.setWindowTitle("BioRobotics EMG/IMU Visualizer & Data Collector")
            self.setGeometry(100, 100, 1400, 900)
            
            if self.config.dark_mode:
                pg.setConfigOption('background', '#1e1e1e')
                pg.setConfigOption('foreground', '#ffffff')
            
            central = QWidget()
            self.setCentralWidget(central)
            main_layout = QHBoxLayout(central)
            
            # === Left Panel: Controls & Data Collection ===
            left_panel = QWidget()
            left_panel.setMaximumWidth(350)
            left_panel.setMinimumWidth(300)
            left_layout = QVBoxLayout(left_panel)
            
            # -- Stream Selection --
            stream_group = QGroupBox("Streams")
            stream_layout = QVBoxLayout(stream_group)
            
            refresh_row = QHBoxLayout()
            self.refresh_btn = QPushButton("🔄 Scan for Streams")
            self.refresh_btn.clicked.connect(self.refresh_streams)
            refresh_row.addWidget(self.refresh_btn)
            stream_layout.addLayout(refresh_row)
            
            self.stream_list = QListWidget()
            self.stream_list.setSelectionMode(QListWidget.SelectionMode.MultiSelection)
            self.stream_list.setMaximumHeight(120)
            stream_layout.addWidget(self.stream_list)
            
            connect_row = QHBoxLayout()
            self.connect_btn = QPushButton("▶ Connect Selected")
            self.connect_btn.clicked.connect(self.toggle_connection)
            connect_row.addWidget(self.connect_btn)
            stream_layout.addLayout(connect_row)
            
            left_layout.addWidget(stream_group)
            
            # -- Data Collection --
            collect_group = QGroupBox("Data Collection")
            collect_layout = QGridLayout(collect_group)
            
            collect_layout.addWidget(QLabel("Participant ID:"), 0, 0)
            self.participant_edit = QLineEdit()
            self.participant_edit.setPlaceholderText("e.g., P01")
            self.participant_edit.textChanged.connect(self.update_metadata)
            collect_layout.addWidget(self.participant_edit, 0, 1)
            
            collect_layout.addWidget(QLabel("Gesture:"), 1, 0)
            self.gesture_combo = QComboBox()
            self.gesture_combo.setEditable(True)
            self.gesture_combo.addItems([
                "rest", "fist", "open", "wrist_flexion", "wrist_extension",
                "pronation", "supination", "pinch", "point", "custom"
            ])
            self.gesture_combo.currentTextChanged.connect(self.update_metadata)
            collect_layout.addWidget(self.gesture_combo, 1, 1)
            
            collect_layout.addWidget(QLabel("Trial #:"), 2, 0)
            self.trial_spin = QSpinBox()
            self.trial_spin.setRange(1, 999)
            self.trial_spin.setValue(1)
            self.trial_spin.valueChanged.connect(self.update_metadata)
            collect_layout.addWidget(self.trial_spin, 2, 1)
            
            collect_layout.addWidget(QLabel("Output Dir:"), 3, 0)
            dir_row = QHBoxLayout()
            self.dir_label = QLabel(self.output_dir)
            self.dir_label.setWordWrap(True)
            self.dir_label.setStyleSheet("color: #888; font-size: 10px;")
            dir_row.addWidget(self.dir_label)
            self.dir_btn = QPushButton("📁")
            self.dir_btn.setMaximumWidth(30)
            self.dir_btn.clicked.connect(self.select_output_dir)
            dir_row.addWidget(self.dir_btn)
            collect_layout.addLayout(dir_row, 3, 1)
            
            # Record button
            self.record_btn = QPushButton("⏺ START RECORDING")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d5a27;
                    color: white;
                    font-weight: bold;
                    padding: 15px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #3d7a37;
                }
                QPushButton:disabled {
                    background-color: #555;
                }
            """)
            self.record_btn.clicked.connect(self.toggle_recording)
            self.record_btn.setEnabled(False)
            collect_layout.addWidget(self.record_btn, 4, 0, 1, 2)
            
            # Recording status
            self.record_status = QLabel("Not recording")
            self.record_status.setAlignment(Qt.AlignmentFlag.AlignCenter)
            collect_layout.addWidget(self.record_status, 5, 0, 1, 2)
            
            left_layout.addWidget(collect_group)
            
            # -- Display Settings --
            display_group = QGroupBox("Display Settings")
            display_layout = QGridLayout(display_group)
            
            display_layout.addWidget(QLabel("Time (s):"), 0, 0)
            self.window_spin = QSpinBox()
            self.window_spin.setRange(1, 30)
            self.window_spin.setValue(int(self.config.window_seconds))
            self.window_spin.valueChanged.connect(self.update_window)
            display_layout.addWidget(self.window_spin, 0, 1)
            
            display_layout.addWidget(QLabel("EMG Amp (±):"), 1, 0)
            self.emg_amp_spin = QDoubleSpinBox()
            self.emg_amp_spin.setRange(1, 10000)
            self.emg_amp_spin.setValue(self.config.emg_amplitude)
            self.emg_amp_spin.valueChanged.connect(self.update_emg_amplitude)
            display_layout.addWidget(self.emg_amp_spin, 1, 1)
            
            display_layout.addWidget(QLabel("IMU Amp (±):"), 2, 0)
            self.imu_amp_spin = QDoubleSpinBox()
            self.imu_amp_spin.setRange(0.1, 1000)
            self.imu_amp_spin.setValue(self.config.imu_amplitude)
            self.imu_amp_spin.valueChanged.connect(self.update_imu_amplitude)
            display_layout.addWidget(self.imu_amp_spin, 2, 1)
            
            self.envelope_check = QCheckBox("Show Envelope")
            self.envelope_check.setChecked(self.config.show_envelope)
            self.envelope_check.stateChanged.connect(self.toggle_envelope)
            display_layout.addWidget(self.envelope_check, 3, 0, 1, 2)
            
            left_layout.addWidget(display_group)
            
            left_layout.addStretch()
            
            main_layout.addWidget(left_panel)
            
            # === Right Panel: Plots ===
            self.plot_tabs = QTabWidget()
            main_layout.addWidget(self.plot_tabs, stretch=1)
            
            # === Status Bar ===
            self.status_bar = QStatusBar()
            self.setStatusBar(self.status_bar)
            self.status_label = QLabel("Not connected")
            self.rate_label = QLabel("")
            self.status_bar.addWidget(self.status_label)
            self.status_bar.addPermanentWidget(self.rate_label)
        
        def refresh_streams(self):
            """Scan for available LSL streams."""
            self.stream_list.clear()
            self.status_label.setText("Scanning...")
            QApplication.processEvents()
            
            try:
                try:
                    streams = pylsl.resolve_streams(2.0)
                except TypeError:
                    streams = pylsl.resolve_streams(wait_time=2.0)
                
                for stream in streams:
                    name = stream.name()
                    stype = stream.type()
                    n_ch = stream.channel_count()
                    rate = stream.nominal_srate()
                    
                    item = QListWidgetItem(f"{name} ({stype}, {n_ch}ch, {rate:.0f}Hz)")
                    item.setData(Qt.ItemDataRole.UserRole, {
                        'name': name,
                        'type': stype,
                        'channels': n_ch,
                        'rate': rate
                    })
                    self.stream_list.addItem(item)
                
                if streams:
                    self.status_label.setText(f"Found {len(streams)} stream(s)")
                else:
                    self.status_label.setText("No streams found")
            
            except Exception as e:
                self.status_label.setText(f"Error: {e}")
        
        def toggle_connection(self):
            """Connect or disconnect from streams."""
            if self.streams:
                self.disconnect_streams()
            else:
                self.connect_streams()
        
        def connect_streams(self):
            """Connect to selected streams."""
            selected = self.stream_list.selectedItems()
            if not selected:
                QMessageBox.warning(self, "Error", "No streams selected")
                return
            
            # Clear existing tabs
            while self.plot_tabs.count() > 0:
                self.plot_tabs.removeTab(0)
            
            for item in selected:
                info = item.data(Qt.ItemDataRole.UserRole)
                name = info['name']
                stype = info['type']
                n_channels = info['channels']
                rate = info['rate']
                
                # Create buffer
                max_samples = int(self.config.window_seconds * rate * 2)
                buffer = SignalBuffer(n_channels, max_samples)
                
                # Create panel
                panel = StreamPanel(stream_type=stype, n_channels=n_channels)
                
                # Determine amplitude based on type
                if 'EMG' in stype.upper() or 'EMG' in name.upper():
                    amplitude = self.config.emg_amplitude
                    panel.stream_type = "EMG"
                else:
                    amplitude = self.config.imu_amplitude
                    panel.stream_type = "IMU"
                
                panel.setup_plots(
                    n_channels=n_channels,
                    amplitude=amplitude,
                    window_seconds=self.config.window_seconds,
                    line_width=self.config.line_width,
                    show_envelope=self.config.show_envelope
                )
                
                # Add tab
                tab_name = name.replace("_", " ")
                self.plot_tabs.addTab(panel, tab_name)
                
                # Start reader
                reader = LSLStreamReader(name, buffer)
                reader.start()
                
                # Store
                self.streams[name] = {
                    'reader': reader,
                    'buffer': buffer,
                    'panel': panel,
                    'info': info
                }
                self.last_counts[name] = 0
            
            # Start timer
            interval_ms = int(1000 / self.config.update_rate_hz)
            self.timer.start(interval_ms)
            self.last_rate_time = time.time()
            
            # Update UI
            self.connect_btn.setText("⏹ Disconnect")
            self.record_btn.setEnabled(True)
            self.status_label.setText(f"Connected to {len(self.streams)} stream(s)")
        
        def disconnect_streams(self):
            """Disconnect from all streams."""
            self.timer.stop()
            
            if self.recording:
                self.toggle_recording()
            
            for name, stream in self.streams.items():
                stream['reader'].stop()
                stream['reader'].join(timeout=1.0)
            
            self.streams.clear()
            self.last_counts.clear()
            
            self.connect_btn.setText("▶ Connect Selected")
            self.record_btn.setEnabled(False)
            self.status_label.setText("Disconnected")
            self.rate_label.setText("")
        
        def update_plots(self):
            """Update all stream plots."""
            rate_parts = []
            
            for name, stream in self.streams.items():
                buffer = stream['buffer']
                panel = stream['panel']
                reader = stream['reader']
                
                data, timestamps = buffer.get_data()
                
                if len(timestamps) < 2:
                    continue
                
                # Create UNIFORM time axis based on nominal sample rate
                # This avoids BLE timing jitter causing jumpy displays
                # Raw timestamps are still saved in CSV for accurate duration calculation
                n_samples = len(timestamps)
                sample_rate = reader.sample_rate or 200
                t_rel = np.linspace(-n_samples / sample_rate, 0, n_samples)
                
                # Compute envelope if needed
                envelope = None
                if self.config.show_envelope and panel.stream_type == "EMG":
                    sample_rate = reader.sample_rate or 200
                    window_samples = max(1, int(self.config.envelope_window_ms * sample_rate / 1000))
                    envelope = compute_envelope(data, window_samples)
                
                # Update curves
                for i, curve in enumerate(panel.curves):
                    if i < data.shape[1]:
                        curve.setData(t_rel, data[:, i])
                        
                        if envelope is not None and i < len(panel.envelope_curves):
                            panel.envelope_curves[i].setData(t_rel, envelope[:, i])
                
                # Rate calculation
                current_count = reader.sample_count
                rate_parts.append(f"{name.split('_')[-1]}: {current_count}")
            
            # Update rate display
            now = time.time()
            dt = now - self.last_rate_time
            if dt >= 1.0:
                rates = []
                for name, stream in self.streams.items():
                    current = stream['reader'].sample_count
                    rate = (current - self.last_counts.get(name, 0)) / dt
                    rates.append(f"{name.split('_')[-1]}: {rate:.0f}Hz")
                    self.last_counts[name] = current
                self.rate_label.setText(" | ".join(rates))
                self.last_rate_time = now
            
            # Recording duration
            if self.recording and self.record_start_time:
                duration = time.time() - self.record_start_time
                self.record_status.setText(f"🔴 Recording: {duration:.1f}s")
        
        def update_metadata(self):
            """Update metadata from UI."""
            self.metadata.participant_id = self.participant_edit.text()
            self.metadata.gesture = self.gesture_combo.currentText()
            self.metadata.trial_number = self.trial_spin.value()
        
        def select_output_dir(self):
            """Select output directory."""
            dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
            if dir_path:
                self.output_dir = dir_path
                self.dir_label.setText(dir_path)
        
        def toggle_recording(self):
            """Start or stop recording."""
            if self.recording:
                self.stop_recording()
            else:
                self.start_recording()
        
        def start_recording(self):
            """Start recording data."""
            self.update_metadata()
            
            if not self.metadata.participant_id:
                QMessageBox.warning(self, "Error", "Please enter a Participant ID")
                return
            
            # Start recording on each stream reader
            for name, stream in self.streams.items():
                reader = stream['reader']
                panel = stream['panel']
                n_channels = panel.n_channels
                reader.start_recording(n_channels)
            
            self.recording = True
            self.record_start_time = time.time()
            
            self.record_btn.setText("⏹ STOP RECORDING")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #8b0000;
                    color: white;
                    font-weight: bold;
                    padding: 15px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #a00000;
                }
            """)
            self.record_status.setText("🔴 Recording...")
        
        def stop_recording(self):
            """Stop recording and save data."""
            self.recording = False
            duration = time.time() - self.record_start_time if self.record_start_time else 0
            
            # Collect recorded data from all stream readers
            self.recorded_data = {}  # stream_name -> (data, timestamps)
            for name, stream in self.streams.items():
                reader = stream['reader']
                data, timestamps = reader.stop_recording()
                if len(data) > 0:
                    self.recorded_data[name] = (data, timestamps)
            
            self.record_btn.setText("⏺ START RECORDING")
            self.record_btn.setStyleSheet("""
                QPushButton {
                    background-color: #2d5a27;
                    color: white;
                    font-weight: bold;
                    padding: 15px;
                    font-size: 14px;
                }
                QPushButton:hover {
                    background-color: #3d7a37;
                }
            """)
            
            # Save data
            saved_files = self.save_recording()
            
            if saved_files:
                self.record_status.setText(f"✓ Saved {len(saved_files)} file(s) ({duration:.1f}s)")
                
                # Auto-increment trial number
                self.trial_spin.setValue(self.metadata.trial_number + 1)
            else:
                self.record_status.setText("No data to save")
        
        def save_recording(self) -> List[str]:
            """Save recorded data to CSV files in hierarchical directory structure.
            
            Structure: output_dir/participant_id/gesture/trial###_streamtype_timestamp.csv
            """
            saved_files = []
            
            # Timestamp for filename
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            for stream_name, (all_data, all_timestamps) in self.recorded_data.items():
                if len(all_data) == 0:
                    continue
                
                # Determine stream type from name
                if 'EMG' in stream_name.upper():
                    stream_type = 'emg'
                    columns = [f'emg_{i+1}' for i in range(all_data.shape[1])]
                elif 'IMU' in stream_name.upper():
                    stream_type = 'imu'
                    columns = ['quat_w', 'quat_x', 'quat_y', 'quat_z',
                              'accel_x', 'accel_y', 'accel_z',
                              'gyro_x', 'gyro_y', 'gyro_z'][:all_data.shape[1]]
                else:
                    stream_type = 'data'
                    columns = [f'ch_{i+1}' for i in range(all_data.shape[1])]
                
                # Build hierarchical directory path
                # Structure: output_dir / participant_id / gesture /
                participant_dir = self.metadata.participant_id if self.metadata.participant_id else "unknown"
                gesture_dir = self.metadata.gesture if self.metadata.gesture else "unspecified"
                
                # Create subdirectory path
                subdir = os.path.join(self.output_dir, participant_dir, gesture_dir)
                os.makedirs(subdir, exist_ok=True)
                
                # Build filename (simpler now that path contains context)
                filename = f"trial{self.metadata.trial_number:03d}_{stream_type}_{timestamp}.csv"
                filepath = os.path.join(subdir, filename)
                
                # Calculate duration and effective sample rate
                if len(all_timestamps) > 1:
                    duration = all_timestamps[-1] - all_timestamps[0]
                    effective_rate = (len(all_data) - 1) / duration if duration > 0 else 200
                else:
                    duration = 0
                    effective_rate = 200
                
                # Get nominal rate from stream reader
                nominal_rate = 200
                for stream in self.streams.values():
                    if stream['reader'].stream_name == stream_name:
                        nominal_rate = stream['reader'].sample_rate or 200
                        break
                
                with open(filepath, 'w') as f:
                    f.write(f"# participant_id: {self.metadata.participant_id}\n")
                    f.write(f"# gesture: {self.metadata.gesture}\n")
                    f.write(f"# trial_number: {self.metadata.trial_number}\n")
                    f.write(f"# stream_name: {stream_name}\n")
                    f.write(f"# stream_type: {stream_type}\n")
                    f.write(f"# timestamp: {timestamp}\n")
                    f.write(f"# samples: {len(all_data)}\n")
                    f.write(f"# duration_sec: {duration:.3f}\n")
                    f.write(f"# nominal_sample_rate: {nominal_rate}\n")
                    f.write(f"# effective_sample_rate: {effective_rate:.2f}\n")
                    f.write("#\n")
                    
                    # Header row
                    f.write("timestamp," + ",".join(columns) + "\n")
                    
                    # Data rows
                    for i in range(len(all_data)):
                        ts = all_timestamps[i] if i < len(all_timestamps) else 0
                        row = [f"{ts:.6f}"] + [f"{v:.6f}" for v in all_data[i]]
                        f.write(",".join(row) + "\n")
                
                saved_files.append(filepath)
                print(f"Saved: {filepath} ({len(all_data)} samples)")
            
            return saved_files
        
        def update_window(self, value):
            """Update time window."""
            self.config.window_seconds = float(value)
            for stream in self.streams.values():
                stream['panel'].update_window(value)
        
        def update_emg_amplitude(self, value):
            """Update EMG amplitude."""
            self.config.emg_amplitude = value
            for stream in self.streams.values():
                if stream['panel'].stream_type == "EMG":
                    stream['panel'].update_amplitude(value)
        
        def update_imu_amplitude(self, value):
            """Update IMU amplitude."""
            self.config.imu_amplitude = value
            for stream in self.streams.values():
                if stream['panel'].stream_type == "IMU":
                    stream['panel'].update_amplitude(value)
        
        def toggle_envelope(self, state):
            """Toggle envelope display."""
            self.config.show_envelope = bool(state)
            for stream in self.streams.values():
                panel = stream['panel']
                for env_curve in panel.envelope_curves:
                    env_curve.setVisible(self.config.show_envelope)
        
        def closeEvent(self, event):
            """Handle window close."""
            self.disconnect_streams()
            event.accept()


def main():
    """Main entry point."""
    if not HAS_GUI:
        print("Error: GUI dependencies not available.")
        print("Install with: pip install PyQt6 pyqtgraph")
        return 1
    
    if not HAS_LSL:
        print("Error: pylsl not available.")
        print("Install with: pip install pylsl")
        return 1
    
    import argparse
    parser = argparse.ArgumentParser(description="EMG/IMU Visualizer & Data Collector")
    parser.add_argument("--stream", help="Auto-select streams containing this name")
    parser.add_argument("--dark", action="store_true", default=True, help="Dark mode")
    parser.add_argument("--mock", action="store_true", help="Start mock streams for testing")
    parser.add_argument("--output", help="Output directory for recordings")
    args = parser.parse_args()
    
    # Start mock streams if requested
    mock_threads = []
    if args.mock:
        print("Starting mock EMG + IMU streams...")
        mock_threads = create_mock_streams()
        time.sleep(0.5)
        args.stream = "MockMyo"
    
    app = QApplication(sys.argv)
    
    config = VisualizerConfig(dark_mode=args.dark)
    window = EMGVisualizer(config)
    
    if args.output:
        window.output_dir = args.output
        window.dir_label.setText(args.output)
    
    window.show()
    
    # Auto-select streams if specified
    if args.stream:
        window.refresh_streams()
        for i in range(window.stream_list.count()):
            item = window.stream_list.item(i)
            if args.stream in item.text():
                item.setSelected(True)
        if window.stream_list.selectedItems():
            window.connect_streams()
    
    result = app.exec()
    
    # Cleanup mock streams
    for thread in mock_threads:
        thread.do_run = False
        thread.join(timeout=1.0)
    
    return result


def create_mock_streams():
    """Create mock EMG and IMU streams for testing."""
    import threading
    
    threads = []
    
    # EMG stream
    emg_info = pylsl.StreamInfo(
        name="MockMyo_EMG",
        type="EMG",
        channel_count=8,
        nominal_srate=200,
        channel_format=pylsl.cf_float32,
        source_id="MockMyo_EMG"
    )
    emg_outlet = pylsl.StreamOutlet(emg_info)
    
    def generate_emg():
        t = 0
        thread = threading.current_thread()
        while getattr(thread, 'do_run', True):
            sample = []
            for ch in range(8):
                val = np.random.normal(0, 10)
                if np.sin(2 * np.pi * 0.3 * t + ch * np.pi / 4) > 0.5:
                    val += np.random.normal(60, 25)
                sample.append(np.clip(val, -127, 127))
            emg_outlet.push_sample(sample)
            t += 1/200
            time.sleep(1/200)
    
    emg_thread = threading.Thread(target=generate_emg, daemon=True)
    emg_thread.do_run = True
    emg_thread.start()
    threads.append(emg_thread)
    
    # IMU stream
    imu_info = pylsl.StreamInfo(
        name="MockMyo_IMU",
        type="IMU",
        channel_count=10,
        nominal_srate=50,
        channel_format=pylsl.cf_float32,
        source_id="MockMyo_IMU"
    )
    imu_outlet = pylsl.StreamOutlet(imu_info)
    
    def generate_imu():
        t = 0
        thread = threading.current_thread()
        while getattr(thread, 'do_run', True):
            angle = (t * 0.5) % (2 * np.pi)
            sample = [
                np.cos(angle/2), 0, np.sin(angle/2), 0,  # Quaternion
                np.random.normal(0, 0.1), np.random.normal(0, 0.1), np.random.normal(-1, 0.1),  # Accel
                np.random.normal(0, 5), np.random.normal(30*np.sin(t), 5), np.random.normal(0, 5)  # Gyro
            ]
            imu_outlet.push_sample(sample)
            t += 1/50
            time.sleep(1/50)
    
    imu_thread = threading.Thread(target=generate_imu, daemon=True)
    imu_thread.do_run = True
    imu_thread.start()
    threads.append(imu_thread)
    
    print("Created: MockMyo_EMG (8ch, 200Hz), MockMyo_IMU (10ch, 50Hz)")
    return threads


if __name__ == "__main__":
    sys.exit(main())