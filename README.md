# Lab 1: Introduction to EMG and the Myo Armband

**BioRobotics**  
**Duration:** 2-3 hours  
**Group Size:** 2-3 students

---

## Abstract

The goal of this lab is to introduce students to biopotentials and how they can be recorded using the Myo Armband. This experiment includes real-time visualization, data collection, and analysis of electromyograms (EMG) and inertial measurement unit (IMU) data. Students will learn how to stream biosignal data using the Lab Streaming Layer (LSL) protocol, collect gesture data, and apply machine learning techniques for gesture classification.

**Please read through the entire document before beginning the lab.**

---

## Learning Objectives

By the end of this lab, students will be able to:

1. **Understand EMG Signals** - Explain how muscle contractions generate electrical signals and how they can be measured
2. **Set Up LSL Streaming** - Configure and run real-time biosignal streaming from the Myo Armband
3. **Collect Gesture Data** - Record labeled EMG and IMU data for multiple gestures with proper experimental protocol
4. **Visualize Biosignals** - Interpret real-time EMG waveforms and understand signal characteristics
5. **Apply Basic Signal Processing** - Understand rectification, filtering, and envelope extraction
6. **Perform Gesture Classification** - Use LDA, QDA, and K-means algorithms to classify gestures from EMG features

---

## Background

### The Human Electrical System

The human body is a complex system that includes mechanical, electrical, and chemical components. The electrical system consists of electrical potentials that propagate along nerve cells and muscle fibers. Brain functions, muscle movements, and eye movements are all invoked by these electrical potentials. 

Physiological potentials arise from the ionic currents that flow in and out of nerve and muscle cells. These **biopotentials** can be measured using electrodes in combination with electronic instrumentation, providing insight into how different systems within the body are functioning.

### EMG Signals

There are three categories of muscles in the body: cardiac, smooth, and skeletal muscles. This lab focuses on **skeletal muscles** — the muscles attached to bones that are under voluntary control.

Muscles attach to bones via tendons. During a muscle contraction, the tendon pulls the bone and movement occurs. Muscle contraction is invoked by an **action potential**, which can be measured with electrodes on the surface of the skin. The measurement of these action potentials is called an **electromyogram (EMG)**.

Key EMG characteristics:
- **Frequency range:** 2–500 Hz
- **Amplitude range:** 50 μV – 5 mV
- **Action potential duration:** 1-3 milliseconds (average)
- **Muscle contraction duration:** 10-100 milliseconds (average)
- **Typical contraction frequencies:** 8–25 Hz

A single action potential causes a **twitch**. Rapid twitches create a **tetanus response**:
- **Unfused tetanus:** Individual twitches are still noticeable
- **Fused tetanus:** Individual twitches can no longer be distinguished

### EMG Signal Processing

Movement artifacts in EMG are generally lower in frequency and can be attenuated with a high-pass filter. Common processing methods include:

1. **Rectification** - Take the absolute value of the signal
2. **Envelope extraction** - Low-pass filter the rectified signal to get the muscle activation level
3. **RMS power** - Calculate the root mean squared power of the waveform
4. **Frequency analysis** - Examine the power spectral density

### The Myo Armband

The Myo Armband is an off-the-shelf device that collects surface EMG (sEMG) and IMU data from the forearm. It contains:

- **8 EMG sensors** arranged in a ring around the forearm (200 Hz sampling rate)
- **9-axis IMU** including accelerometer, gyroscope, and orientation data (50 Hz sampling rate)

The 8 EMG channels capture muscle activity from different regions of the forearm, allowing discrimination between various hand gestures.

---

## Files Overview

| File | Description |
|------|-------------|
| `src/myo_interface.py` | Streams EMG and IMU data from Myo to LSL |
| `src/visualizer.py` | Real-time visualization and data collection GUI |
| `src/proportional_control.py` | Demonstrates EMG-based proportional control |
| `src/bioradio.py` | Pure Python interface for the GLNeuroTech BioRadio device |
| `src/bioradio_lsl_bridge.py` | LSL network bridge for streaming BioRadio data across machines |
| `Lab1_EMG_Analysis.ipynb` | Jupyter notebook for data analysis (LDA, QDA, K-means) |
| `environment.yml` | Conda environment specification |

---

## Part 1: Environment Setup

### 1.1 Install Anaconda/Miniconda

If you don't have Anaconda or Miniconda installed:

1. Download Miniconda from: https://docs.conda.io/en/latest/miniconda.html
2. **Important for lab computers:** Install for "Just Me" (not all users) to avoid admin permission issues
3. Complete the installation

### 1.2 Create the Conda Environment

Open a terminal (Anaconda Prompt on Windows) and run:

```bash
# Navigate to the lab folder
cd path/to/biorobotics_lab1

# Create the environment
conda env create -f environment.yml

# Activate the environment
conda activate biorobotics
```

If the environment file doesn't work, create it manually:

```bash
conda create -n biorobotics python=3.11
conda activate biorobotics
pip install numpy pandas scipy matplotlib scikit-learn
pip install pylsl PyQt6 pyqtgraph
pip install dl-myo
pip install jupyter
```

### 1.3 Verify Installation

```bash
python -c "import pylsl; import myo; print('All packages installed!')"
```

---

## Part 2: Finding and Selecting Your Myo Device

In a classroom setting with multiple Myo armbands, you need to identify and connect to the correct device. This section shows you how to scan for available Myos, identify yours by making it vibrate, and connect to it.

### 2.1 Interactive Device Selection (Recommended)

The easiest way to find and connect to your Myo is using interactive mode:

```bash
conda activate biorobotics
cd path/to/biorobotics_lab1
python src/myo_interface.py --select
```

You'll see output like this:

```
==================================================
Myo Device Selection
==================================================

Scanning for Myo devices (5.0s)...
----------------------------------------
  [1] Myo
      MAC: D2:3B:85:94:32:8E
      Signal: -65 dBm
  [2] Myo
      MAC: A1:B2:C3:D4:E5:F6
      Signal: -72 dBm
----------------------------------------
Found 2 Myo device(s)

Options:
  [1-2] Select a Myo by number
  [p #]  Ping a Myo (e.g., 'p 1' to ping device 1)
  [r]    Rescan for devices
  [q]    Quit

Your choice: 
```

**To identify which Myo is yours:**

1. Type `p 1` and press Enter to ping device 1
2. **The Myo will vibrate twice** if it's reachable
3. If that wasn't your Myo, try `p 2` for the next one
4. Once you find yours, enter its number (e.g., `1`) to select it
5. The stream will start automatically

### 2.2 Scan for Devices

To just see what Myos are available without connecting:

```bash
python src/myo_interface.py --scan
```

Output:
```
Scanning for Myo devices (5.0s)...
----------------------------------------
  [1] Myo
      MAC: D2:3B:85:94:32:8E
      Signal: -65 dBm
----------------------------------------
Found 1 Myo device(s)
```

**Understanding signal strength (RSSI):**
- `-50 dBm` or higher: Excellent signal (very close)
- `-50 to -70 dBm`: Good signal
- `-70 to -90 dBm`: Weak signal (may have connection issues)
- Below `-90 dBm`: Very weak (move closer)

### 2.3 Ping a Specific Myo

If you already know a MAC address, you can ping it directly to make it vibrate:

```bash
python src/myo_interface.py --ping D2:3B:85:94:32:8E
```

The Myo will vibrate twice if the connection is successful.

### 2.4 Connect Using MAC Address

Once you know your Myo's MAC address, you can connect directly:

```bash
python src/myo_interface.py --mac D2:3B:85:94:32:8E
```

**Tip:** Write down your Myo's MAC address for future lab sessions!

### 2.5 Quick Reference

| Command | Description |
|---------|-------------|
| `--select` | Interactive mode: scan, ping, and select |
| `--scan` | List all nearby Myo devices |
| `--ping MAC` | Make a specific Myo vibrate |
| `--mac MAC` | Connect directly to a specific Myo |

### 2.6 Troubleshooting Device Discovery

| Problem | Solution |
|---------|----------|
| No devices found | Wake up the Myo by shaking it; check that Bluetooth is enabled |
| Ping fails | Move closer to the computer; make sure MyoConnect is closed |
| Wrong Myo connects | Use `--select` mode to ping and verify before connecting |
| Weak signal | Move the Myo closer to your computer's Bluetooth antenna |

---

## Part 3: Connect and Visualize EMG Signals

### 3.1 Myo Armband Placement

Proper placement is critical for good signal quality:

1. **Wake up the Myo** - Move/shake the armband until the LEDs flash
2. **Position on forearm** - Place the Myo on your forearm, approximately 2-3 inches below the elbow
3. **Align the status bar** - The blue/orange status bar (the thicker pod) should point toward your hand, positioned on top of your forearm
4. **Ensure snug fit** - Use the sizing clips to adjust. The armband should be snug but not uncomfortable
5. **Center on muscle belly** - The pods should sit on the muscular part of the forearm, not on bone

**Tip:** If signals look weak, try repositioning the armband slightly or adjusting the tightness.

### 3.2 Start the Myo Data Stream

Open a terminal and run (using the MAC address from Part 2):

```bash
conda activate biorobotics
cd path/to/biorobotics_lab1

# Option 1: Use interactive selection (recommended for classrooms)
python src/myo_interface.py --select

# Option 2: Connect directly with MAC address from Part 2
python src/myo_interface.py --mac YOUR_MAC_ADDRESS
```

You should see:
```
==================================================
Myo LSL Streamer (dl-myo)
==================================================

Using native Bluetooth - no dongle needed!
IMPORTANT: Make sure MyoConnect is CLOSED!

Scanning for Myo devices...
Connected to Myo!
Created LSL outlet: Myo_EMG (200Hz, 8ch)
Created LSL outlet: Myo_IMU (50Hz, 10ch)
Myo streamer started!
Streaming EMG at 200Hz (raw mode)
Streaming IMU at 50Hz (orientation + accel + gyro)
```

The Myo will vibrate briefly when connected.

**Troubleshooting:**
- If no Myo is found, make sure it's charged and awake (LEDs flashing)
- Close MyoConnect if it's running (it will interfere)
- Try `python src/myo_interface.py --scan` to see available devices

### 3.3 Start the Visualizer

Open a **second terminal** and run:

```bash
conda activate biorobotics
python src/visualizer.py
```

In the visualizer:

1. Click **"🔄 Scan for Streams"**
2. You should see `Myo_EMG` and `Myo_IMU` in the list
3. **Select both streams** (Ctrl+click or Cmd+click)
4. Click **"▶ Connect Selected"**

You should now see real-time EMG and IMU data in separate tabs.

### Cannot Import PyQT6 Error

On some systems, pip installing pyqt creates conflicts. So we need to remove the pip packages and install via condo.

```bash
python -m pip uninstall -y PyQt6 PyQt6-Qt6 PyQt6-sip
conda install -c conda-forge pyqt=6 pyqtgraph qt-main
```

### 3.4 Explore the EMG Signals

With the visualizer running:

1. **Relax your arm** - Observe the baseline noise level
2. **Make a fist** - Watch the EMG amplitude increase
3. **Open your hand** - Notice which channels respond
4. **Flex your wrist** (palm up, then palm down) - See different activation patterns
5. **Rotate your forearm** (pronation/supination) - Observe the changes

**Adjust the display:**
- **EMG Amplitude:** Start with ±128, adjust if signals clip or are too small
- **Time Window:** 5 seconds is good for seeing gestures
- **Envelope checkbox:** Enable to see smoothed muscle activation

> **Question 1:** Which EMG channels show the strongest activation when you make a fist? Which channels activate when you open your hand? Sketch or screenshot the patterns you observe.

> **Question 2:** What is the approximate amplitude range of the EMG signal at rest vs. during a strong contraction?

---

## Part 4: Data Collection

### 4.1 Experimental Protocol

You will collect EMG and IMU data for the following gestures:

| Gesture | Description |
|---------|-------------|
| `rest` | Arm relaxed, no movement |
| `fist` | Close hand into a fist |
| `open` | Spread fingers apart |
| `wrist_flexion` | Bend wrist so palm faces toward you |
| `wrist_extension` | Bend wrist so palm faces away |
| `pronation` | Rotate forearm so palm faces down |
| `supination` | Rotate forearm so palm faces up |

**Data collection parameters:**
- **Trials per gesture:** 5-10 (minimum 5, 10 recommended for better classification)
- **Trial duration:** 3-5 seconds of sustained gesture
- **Rest between trials:** 2-3 seconds

### 4.2 Recording Procedure

In the visualizer:

1. **Set Participant ID** - Enter a unique identifier (e.g., "P01", "GroupA_Alice")
2. **Select Gesture** - Choose from dropdown or type custom name
3. **Verify Trial Number** - Starts at 1, auto-increments after each recording
4. **Set Output Directory** - Click 📁 to choose where files are saved (default: `./recordings`)

**For each trial:**

1. Prepare to perform the gesture
2. Click **"⏺ START RECORDING"** (button turns red)
3. Perform and hold the gesture for 3-5 seconds
4. Click **"⏹ STOP RECORDING"**
5. Files are automatically saved, trial number increments

**File naming convention:**
```
{participant}_{gesture}_trial{###}_{emg|imu}_{timestamp}.csv
```

Example files:
```
P01_fist_trial001_emg_20250125_143022.csv
P01_fist_trial001_imu_20250125_143022.csv
```

### 4.3 Data Collection Checklist

Complete the following for each group member:

| Gesture | Trials Completed | Notes |
|---------|-----------------|-------|
| rest | ☐ 5-10 | |
| fist | ☐ 5-10 | |
| open | ☐ 5-10 | |
| wrist_flexion | ☐ 5-10 | |
| wrist_extension | ☐ 5-10 | |
| pronation | ☐ 5-10 | |
| supination | ☐ 5-10 | |

> **Question 3:** How consistent are your EMG patterns across trials of the same gesture? What factors might cause variability?

> **Question 4:** Do different group members show similar or different EMG patterns for the same gesture? Why might this be?

---

## Part 5: Proportional Control Demo

This section demonstrates how EMG signals can be used for real-time control.

### 5.1 Run the Proportional Control Demo

With the Myo streaming (from Part 2), open a **new terminal**:

```bash
conda activate biorobotics
python src/proportional_control.py
```

This script:
1. Connects to the EMG stream
2. Computes the envelope (smoothed activation level)
3. Maps the activation to a control output
4. Displays a real-time bar graph of the control signal

### 5.2 Experiment with Control

1. **Relax** - The bar should be near zero
2. **Gradually increase grip strength** - Watch the bar rise proportionally
3. **Try to hold a specific level** - Can you maintain 50% activation?
4. **Quick contractions** - Observe the response time

> **Question 5:** What is the approximate delay between your muscle contraction and the visual feedback? What factors contribute to this delay?

> **Question 6:** How might proportional EMG control be used in a prosthetic hand or robotic interface?

---

## Part 6: Data Analysis

Open the Jupyter notebook for guided analysis:

```bash
conda activate biorobotics
jupyter notebook Lab1_EMG_Analysis.ipynb
```

The notebook will guide you through:

1. **Loading and exploring your collected data**
2. **Signal processing** - Filtering, rectification, envelope extraction
3. **Feature extraction** - Mean, standard deviation, RMS, frequency features
4. **Visualization** - Plotting signals and comparing gestures
5. **Classification** - Using LDA, QDA, and K-means to classify gestures
6. **Evaluation** - Confusion matrices and accuracy metrics

> **Question 7:** Which features (e.g., mean amplitude, RMS, frequency content) are most useful for distinguishing between gestures?

> **Question 8:** Compare the classification accuracy of LDA, QDA, and K-means. Which performs best on your data and why?

> **Question 9:** How does the number of training trials affect classification accuracy?

> **Question 10:** If you were designing a gesture recognition system for a real application, what gestures would you choose and why?

---

## BioRadio Support

The lab also supports the **GLNeuroTech BioRadio** for multi-channel biopotential recording. The BioRadio connects via Bluetooth Serial Port Profile (SPP) and provides configurable channels for EEG, EMG, ECG, and other physiological signals.

### BioRadio on Windows

The BioRadio works natively on Windows via Bluetooth:

```bash
conda activate biorobotics
python -c "from src.bioradio import BioRadio; r = BioRadio(); r.connect()"
```

Windows creates two COM ports when the BioRadio pairs (e.g. COM9 and COM10). The code automatically detects and probes both to find the working bidirectional port. You can also specify the port directly:

```python
from src.bioradio import BioRadio
radio = BioRadio(port="COM9")  # Use the LOWER COM port number
radio.connect()
```

### BioRadio on macOS

macOS Sonoma (14+) has a known limitation with Bluetooth Serial Port Profile (SPP) that prevents direct connection to the BioRadio. The recommended workaround is to use **Parallels Desktop** (or another VM) with a **USB Bluetooth adapter**:

1. **Install Parallels Desktop** with a Windows VM
2. **Get a USB Bluetooth adapter** (any standard USB BT 4.0+ dongle)
3. **Plug in the USB adapter** and pass it through to the Windows VM:
   - In Parallels: Devices > USB & Bluetooth > select your USB BT adapter
4. **Pair the BioRadio** in the Windows VM's Bluetooth settings
5. **Run the BioRadio code** inside the Windows VM

> **Why is a USB adapter required?** The Mac's built-in Bluetooth is managed by macOS, which cannot establish the RFCOMM data channel the BioRadio needs. A USB adapter passed through to the VM lets Windows manage Bluetooth directly, bypassing this limitation.

**Alternative: LSL Network Bridge**

If you have a separate Windows machine available, you can stream BioRadio data to the Mac over the network using Lab Streaming Layer (LSL):

```bash
# On Windows (where BioRadio is paired):
pip install pylsl pyserial
python src/bioradio_lsl_bridge.py --send --port COM9

# On Mac (receives data over the network):
pip install pylsl
python src/bioradio_lsl_bridge.py --receive
```

Both machines must be on the same network. The receiver provides a `BioRadioLSL` class for use in lab scripts.

---

## Troubleshooting

### Myo Won't Connect

| Problem | Solution |
|---------|----------|
| "No Myo devices found" | Wake up the Myo by moving it; check battery |
| Connection drops | Move closer to the computer; reduce Bluetooth interference |
| MyoConnect interfering | Close MyoConnect completely (check system tray) |

### Weak or Noisy Signals

| Problem | Solution |
|---------|----------|
| Very low amplitude | Tighten the armband; reposition on muscle belly |
| High noise/artifacts | Ensure good skin contact; reduce movement |
| 60Hz interference | Move away from power sources; the notch filter helps |
| Signals look clipped | Reduce EMG amplitude setting in visualizer |

### Visualizer Issues

| Problem | Solution |
|---------|----------|
| No streams found | Make sure myo_interface.py is running first |
| Plots not updating | Check that you clicked "Connect Selected" |
| Recording not saving | Check output directory permissions |

### BioRadio Won't Connect

| Problem | Solution |
|---------|----------|
| No BioRadio port found (Windows) | Check Device Manager > Ports (COM & LPT); make sure device is paired |
| No response from BioRadio (Windows) | Try the other COM port; use the lower-numbered port |
| No BioRadio port found (macOS) | macOS Sonoma cannot connect directly; use Parallels + USB BT adapter |
| Phantom serial port on macOS | The port `/dev/cu.BioRadioAYA` may exist but not carry data; use Parallels |

### Common Errors

```
ImportError: No module named 'myo'
```
→ Run `pip install dl-myo`

```
ImportError: No module named 'pylsl'
```
→ Run `pip install pylsl`

```
Bluetooth adapter not found
```
→ Ensure Bluetooth is enabled on your computer

---

## Deliverables

Submit the following to MyCourses:

### 1. Data Package (ZIP file)
- All collected EMG and IMU CSV files
- Organized by participant if multiple group members recorded

### 2. Analysis Notebook
- Completed `Lab1_EMG_Analysis.ipynb` with all cells executed
- Include screenshots/figures showing:
  - Raw EMG signals for at least 3 different gestures
  - Processed/filtered signals
  - Classification results (confusion matrix)

### 3. Lab Questions
Answer all questions (Q1-Q10) in your notebook or a separate document:

1. EMG channel activation patterns for fist vs. open hand
2. EMG amplitude range at rest vs. contraction
3. Consistency of EMG patterns across trials
4. Variation between group members
5. Delay in proportional control system
6. Applications of proportional EMG control
7. Most useful features for gesture classification
8. Comparison of LDA, QDA, and K-means
9. Effect of training set size on accuracy
10. Gesture selection for a real application

---

## Additional Resources

- **Lab Streaming Layer (LSL):** https://labstreaminglayer.org/
- **dl-myo Library:** https://github.com/iomz/dl-myo
- **scikit-learn Documentation:** https://scikit-learn.org/stable/
- **EMG Signal Processing:** De Luca, C. J. (2002). Surface electromyography: Detection and recording.

---

## Acknowledgments

This lab uses the dl-myo library for Bluetooth communication with the Myo Armband without requiring the official Myo SDK or dongle.

---

*Last updated: January 2025*
