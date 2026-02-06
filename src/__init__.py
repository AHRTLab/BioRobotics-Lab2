"""
BioRobotics Lab 1 - EMG and LSL Tools
=====================================

This package provides tools for:
- LSL stream discovery and recording
- Real-time EMG visualization
- EMG signal processing
- Myo Armband interface
- BioRadio direct Python interface (no .NET SDK required)
- Proportional control demonstration

Usage:
    from src.lsl_utils import discover_streams, LSLRecorder
    from src.emg_processing import bandpass_filter, envelope, compute_features
    from src.bioradio import BioRadio, scan_for_bioradio
    from src.visualizer import EMGVisualizer
    from src.proportional_control import ProportionalControlDemo
"""

__version__ = "2.0.0"
__author__ = "BioRobotics Course"

from .lsl_utils import (
    discover_streams,
    find_stream,
    LSLRecorder,
    LSLMarkerStream,
    load_xdf,
    load_csv,
)

from .emg_processing import (
    bandpass_filter,
    notch_filter,
    rectify,
    envelope,
    rms,
    compute_features,
    power_spectral_density,
    process_emg_pipeline,
)

from .bioradio import (
    BioRadio,
    scan_for_bioradio,
    create_lsl_outlet,
    DeviceConfig,
    DataSample,
)
