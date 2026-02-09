# Lab 2: Introduction to EOG and the BioRadio

**BioRobotics**  
**Duration:** 2-3 hours  
**Group Size:** 2-3 students

---

## Abstract

The goal of this lab is to introduce students to electrooculography (EOG) and how eye movement signals can be recorded using the BioRadio wireless biosignal acquisition system. This experiment includes real-time visualization, data collection, and analysis of EOG signals for gaze direction classification. Students will learn how to stream biosignal data using the Lab Streaming Layer (LSL) protocol, collect labeled gaze data, and apply machine learning techniques including Principal Component Analysis (PCA), Independent Component Analysis (ICA), and Support Vector Machines (SVM) for gaze classification.

**Please read through the entire document before beginning the lab.**

---

## Learning Objectives

By the end of this lab, students will be able to:

1. **Understand EOG Signals** - Explain how eye movements generate electrical potentials and how they can be measured using surface electrodes
2. **Set Up the BioRadio** - Configure and connect to the BioRadio device for biosignal acquisition
3. **Apply Proper Electrode Placement** - Position EOG electrodes correctly for horizontal and vertical eye movement detection
4. **Collect Gaze Data** - Record labeled EOG data for multiple gaze directions with proper experimental protocol
5. **Visualize EOG Signals** - Interpret real-time EOG waveforms and understand signal characteristics
6. **Apply Signal Processing** - Implement baseline correction, filtering, and artifact removal for EOG signals
7. **Perform Dimensionality Reduction** - Apply PCA and ICA to understand EOG signal components
8. **Classify Gaze Directions** - Use SVM classifiers (linear and RBF kernels) to classify gaze from EOG features

---

## Background

### The Electrooculogram (EOG)

The eye functions as an electrical dipole, with the cornea (front of the eye) being positively charged relative to the retina (back of the eye). This **corneo-retinal potential** is approximately 0.4 to 1.0 mV and remains relatively constant during light and dark conditions.

When the eye rotates, the orientation of this dipole changes, creating measurable voltage differences at electrodes placed around the eye. This is the basis of **electrooculography (EOG)**.

Key EOG characteristics:
* **Frequency range:** DC to 30 Hz (much lower than EMG)
* **Amplitude range:** 15-200 μV per degree of eye movement
* **Signal type:** Quasi-DC (slow changes with sustained gaze positions)
* **Typical eye movement speed:** Saccades can reach 500°/s

### Types of Eye Movements

1. **Saccades** - Rapid, ballistic eye movements between fixation points (20-200 ms duration)
2. **Smooth pursuit** - Slow tracking movements following a moving target
3. **Fixations** - Periods of stable gaze on a target
4. **Blinks** - Eyelid closures that create characteristic artifacts in EOG

### EOG Electrode Placement

EOG uses a **2-channel bipolar configuration** to capture horizontal and vertical eye movements:

**Channel 1: Horizontal EOG (HEOG)**
- Electrodes placed at the outer canthi (outer corners) of both eyes
- Right eye outer canthus (+) and left eye outer canthus (−)
- Detects left/right eye movements

**Channel 2: Vertical EOG (VEOG)**  
- Electrodes placed above and below one eye (typically the right eye)
- Above the eye (+) and below the eye (−)
- Detects up/down movements and blinks

**Ground/Reference Electrode**
- Placed on the forehead (center) or behind the ear (mastoid)
- Provides a stable reference potential

```
        ┌─────────────────────────────────────┐
        │            FOREHEAD                  │
        │              (GND)                   │
        │                ●                     │
        │                                      │
        │    ┌─────┐           ┌─────┐        │
        │    │     │    ●      │     │        │  ← VEOG+ (above eye)
        │    │  L  │   NOSE    │  R  │        │
        │ ●──│ EYE │           │ EYE │──●     │  ← HEOG electrodes
        │    │     │           │     │        │     (outer canthi)
        │    └─────┘           └─────┘        │
        │                ●                     │  ← VEOG- (below eye)
        └─────────────────────────────────────┘
```

### EOG Signal Processing

Unlike EMG which has high-frequency content, EOG signals are predominantly low-frequency:

1. **DC Offset Removal / Baseline Correction** - EOG signals have significant DC components that drift over time
2. **Low-pass Filtering** - Cutoff around 30 Hz removes high-frequency noise while preserving eye movement information
3. **Blink Detection** - Blinks create large, characteristic artifacts in VEOG that must be detected and either removed or used as a control signal
4. **Saccade Detection** - Identify rapid eye movements by detecting velocity peaks

### Applications of EOG

* **Assistive Technology** - Eye-controlled interfaces for individuals with motor disabilities
* **Drowsiness Detection** - Monitoring eye closure patterns for driver alertness systems
* **Sleep Studies** - Detecting REM (rapid eye movement) sleep stages
* **Human-Computer Interaction** - Gaze-based control systems
* **Virtual Reality** - Eye tracking for foveated rendering

---

## Files Overview

| File | Description |
|------|-------------|
| `src/bioradio.py` | Pure Python interface for the BioRadio device |
| `src/bioradio_lsl_bridge.py` | LSL network bridge for streaming BioRadio data across machines |
| `src/visualizer.py` | Real-time EOG visualization and data collection GUI |
| `notebooks/Lab2_EOG_Analysis.ipynb` | Jupyter notebook for data analysis (PCA, ICA, SVM) |
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
cd path/to/BioRobotics-Lab2

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
pip install pyserial
pip install jupyter
```

### 1.3 Verify Installation

```bash
python -c "import pylsl; import serial; print('All packages installed!')"
```

---

## Part 2: BioRadio Setup

The BioRadio is a wireless biosignal acquisition device that connects via Bluetooth. Platform support varies:

### 2.1 Windows Setup (Recommended)

The BioRadio works natively on Windows via Bluetooth Serial Port Profile (SPP):

1. **Power on the BioRadio** - Press and hold the power button until the LED flashes
2. **Pair via Bluetooth** - Go to Windows Settings > Bluetooth & Devices > Add device
3. **Find the COM port** - Open Device Manager > Ports (COM & LPT)
   - The BioRadio creates TWO COM ports (e.g., COM9 and COM10)
   - **Use the LOWER numbered port** (e.g., COM9)

**Test the connection:**

```bash
conda activate biorobotics
python src/bioradio.py --scan
```

This will scan for serial ports and identify the BioRadio.

### 2.2 macOS Setup (Requires Workaround)

> ⚠️ **Important:** macOS Sonoma (14+) has a known limitation with Bluetooth Serial Port Profile (SPP) that prevents direct connection to the BioRadio. The serial port `/dev/cu.BioRadioAYA` may appear but will not carry data.

**Recommended Solution: Parallels Desktop + USB Bluetooth Adapter**

1. **Install Parallels Desktop** with a Windows 11 VM
2. **Get a USB Bluetooth adapter** (any standard USB BT 4.0+ dongle, ~$10-15)
3. **Plug in the USB adapter** and pass it through to the Windows VM:
   - In Parallels: Devices > USB & Bluetooth > Select your USB BT adapter
4. **Pair the BioRadio** in the Windows VM's Bluetooth settings
5. **Run the BioRadio code** inside the Windows VM

> **Why is a USB adapter required?** The Mac's built-in Bluetooth is managed by macOS, which cannot establish the RFCOMM data channel the BioRadio needs. A USB adapter passed through to the VM lets Windows manage Bluetooth directly.

**Alternative: LSL Network Bridge**

If you have a separate Windows machine, stream BioRadio data to the Mac over the network:

```bash
# On Windows (where BioRadio is paired):
python src/bioradio_lsl_bridge.py --send --port COM9

# On Mac (receives data over the network):
python src/bioradio_lsl_bridge.py --receive
```

Both machines must be on the same network.

### 2.3 Connect to the BioRadio

Once paired, connect and verify:

```bash
# Auto-detect and connect
python src/bioradio.py

# Or specify the port directly
python src/bioradio.py --port COM9        # Windows
python src/bioradio.py --port /dev/cu.BioRadioAYA  # macOS (if working)
```

You should see device information including firmware version, battery level, and channel configuration.

---

## Part 3: Electrode Placement and Signal Verification

### 3.1 Prepare the Skin

Good electrode contact is essential for clean EOG signals:

1. **Clean the skin** with alcohol wipes at each electrode site
2. **Allow to dry** completely before applying electrodes
3. **Use conductive gel** if using reusable electrodes

### 3.2 Apply Electrodes

Apply electrodes in the following configuration:

| Electrode | Position | BioRadio Channel |
|-----------|----------|------------------|
| HEOG+ | Right eye, outer canthus (temple side) | Channel 1 (+) |
| HEOG- | Left eye, outer canthus (temple side) | Channel 1 (-) |
| VEOG+ | Above right eye, ~1 cm above eyebrow | Channel 2 (+) |
| VEOG- | Below right eye, ~1 cm below lower eyelid | Channel 2 (-) |
| GND | Center of forehead OR behind right ear (mastoid) | Ground |

### 3.3 BioRadio Configuration

For EOG recording, use these settings:

* **Sample Rate:** 250 Hz (sufficient for EOG's low-frequency content)
* **Channels:** 2 (HEOG and VEOG)
* **Coupling:** DC coupling (to capture sustained gaze positions)
* **Gain:** Adjust based on signal amplitude

To set the sample rate:

```bash
python src/bioradio.py --port COM9 --rate 250
```

### 3.4 Start the Visualizer

Open a terminal and start the BioRadio stream:

```bash
conda activate biorobotics
python src/bioradio.py --port COM9 --lsl
```

In a **second terminal**, start the visualizer:

```bash
conda activate biorobotics
python src/visualizer.py
```

1. Click **"🔄 Scan for Streams"**
2. Select the BioRadio stream
3. Click **"▶ Connect Selected"**

### 3.5 Verify Signal Quality

With the visualizer running, perform these tests:

1. **Look straight ahead (Center)** - Both channels should show stable baselines
2. **Look left, then right** - Channel 1 (HEOG) should deflect in opposite directions
3. **Look up, then down** - Channel 2 (VEOG) should deflect in opposite directions
4. **Blink** - Channel 2 (VEOG) should show large, sharp deflections

> **Question 1:** Sketch the EOG waveforms you observe for each gaze direction (Left, Right, Up, Down, Center). Which channel responds to which movement? What is the approximate amplitude of each deflection?

> **Question 2:** How do blink artifacts appear in the HEOG vs. VEOG channels? Why might blinks primarily affect the VEOG channel?

---

## Part 4: Data Collection

### 4.1 Experimental Protocol

You will collect EOG data for the following gaze classes:

| Class | Description | Expected HEOG | Expected VEOG |
|-------|-------------|---------------|---------------|
| `center` | Look straight ahead at fixation point | Baseline | Baseline |
| `left` | Look to the left | Negative deflection | Minimal change |
| `right` | Look to the right | Positive deflection | Minimal change |
| `up` | Look upward | Minimal change | Positive deflection |
| `down` | Look downward | Minimal change | Negative deflection |
| `blink` | Single deliberate blink | Small artifact | Large sharp spike |
| `double_blink` | Two rapid blinks | Small artifacts | Two sharp spikes |

**Data collection parameters:**

* **Trials per gaze direction:** 10-15 (minimum 10 for reliable classification)
* **Trial duration:** 2-3 seconds of sustained gaze
* **Rest between trials:** 2-3 seconds (return to center)

### 4.2 Recording Procedure

In the visualizer:

1. **Set Participant ID** - Enter a unique identifier (e.g., "P01", "GroupA_Alice")
2. **Select Gaze Direction** - Choose from dropdown
3. **Verify Trial Number** - Auto-increments after each recording
4. **Set Output Directory** - Click 📁 to choose save location

**For each trial:**

1. Look at the fixation point (center)
2. When ready, click **"⏺ START RECORDING"**
3. Move eyes to the target gaze direction and hold for 2-3 seconds
4. Click **"⏹ STOP RECORDING"**
5. Return to center, rest briefly
6. Repeat for next trial

### 4.3 Data Collection Checklist

Complete the following for each group member:

| Gaze Direction | Trials Completed | Notes |
|----------------|------------------|-------|
| center | ☐ 10-15 | |
| left | ☐ 10-15 | |
| right | ☐ 10-15 | |
| up | ☐ 10-15 | |
| down | ☐ 10-15 | |
| blink | ☐ 10-15 | |
| double_blink | ☐ 10-15 | |

> **Question 3:** How consistent are your EOG patterns across trials of the same gaze direction? What factors might cause trial-to-trial variability?

> **Question 4:** Did you notice any drift in the baseline over time? What might cause this, and how could it affect classification?

---

## Part 5: Data Analysis

Open the Jupyter notebook for guided analysis:

```bash
conda activate biorobotics
jupyter notebook notebooks/Lab2_EOG_Analysis.ipynb
```

The notebook will guide you through:

### 5.1 Signal Processing

* **Loading and visualizing raw EOG data**
* **Baseline correction** - Remove DC offset and slow drift
* **Low-pass filtering** - Apply 30 Hz cutoff to remove noise
* **Blink detection** - Identify and optionally remove blink artifacts

> **Question 5:** Compare the raw and filtered EOG signals. What types of noise does the low-pass filter remove? How does baseline correction affect the signal?

### 5.2 Feature Extraction

Extract features from each trial:

* **Mean amplitude** - Average signal level during gaze
* **Peak amplitude** - Maximum deflection from baseline
* **Standard deviation** - Signal variability
* **Saccade velocity** - Rate of change during eye movement
* **Fixation stability** - Variance during sustained gaze

> **Question 6:** Which features show the most separation between different gaze directions? Create scatter plots of 2-3 key features colored by gaze class.

### 5.3 Dimensionality Reduction

Apply PCA and ICA to understand the EOG signal structure:

**Principal Component Analysis (PCA)**
* Identify the principal axes of variance in the 2-channel EOG data
* Visualize how gaze directions cluster in PC space

**Independent Component Analysis (ICA)**
* Separate statistically independent source signals
* Can help isolate eye movement from blink artifacts

> **Question 7:** How much variance is explained by each principal component? What do the first two principal components represent in terms of eye movements?

> **Question 8:** How do the ICA-separated components differ from the original HEOG and VEOG channels? Can you identify components that correspond to horizontal movement, vertical movement, and blinks?

### 5.4 Classification with SVM

Train and evaluate Support Vector Machine classifiers:

**Linear SVM**
* Simple decision boundaries
* Interpretable feature weights
* Works well when classes are linearly separable

**RBF (Radial Basis Function) SVM**
* Non-linear decision boundaries
* Can capture complex relationships
* May achieve higher accuracy but less interpretable

**Evaluation Metrics:**
* Confusion matrix - See which gaze directions are confused with each other
* Accuracy, Precision, Recall, F1-score
* Cross-validation to assess generalization

> **Question 9:** Compare the classification accuracy of Linear SVM vs. RBF SVM. Which performs better on your data, and why might this be?

> **Question 10:** Examine the confusion matrix. Which gaze directions are most often confused with each other? Does this make sense given the electrode placement and expected signals?

> **Question 11:** How does reducing the number of gaze classes (e.g., just left/right/center) affect classification accuracy?

---

## Part 6: Discussion Questions

Answer these questions in your lab report:

> **Question 12:** What are the main differences between EOG and EMG signals in terms of frequency content, amplitude, and signal characteristics?

> **Question 13:** Why might EOG-based gaze tracking be preferred over camera-based eye tracking in certain applications? What are the limitations of EOG?

> **Question 14:** If you were designing a practical EOG-based interface (e.g., for a wheelchair or computer control), which gaze commands would you include and why? How would you handle false positives?

> **Question 15:** How could the blink signal be used as an additional control input rather than just an artifact to be removed?

---

## Troubleshooting

### BioRadio Connection Issues

| Problem | Solution |
|---------|----------|
| "No BioRadio port found" (Windows) | Check Device Manager > Ports; make sure device is paired via Bluetooth |
| Two COM ports, neither works | Try the LOWER numbered port first; ensure BioCapture software is closed |
| "No BioRadio port found" (macOS) | macOS Sonoma cannot connect directly; use Parallels + USB BT adapter |
| Connection drops frequently | Move closer to computer; check battery level; reduce Bluetooth interference |
| No response after connecting | Power cycle the BioRadio; re-pair via Bluetooth settings |

### Signal Quality Issues

| Problem | Solution |
|---------|----------|
| Very noisy signal | Check electrode contact; clean skin with alcohol; apply conductive gel |
| Large baseline drift | Ensure DC coupling; allow electrodes to stabilize (2-3 min) |
| No response to eye movements | Verify electrode placement; check channel assignments |
| Blinks overwhelming the signal | Reduce VEOG electrode spacing; apply high-pass filter for movement-only analysis |
| 60Hz interference | Move away from power sources; check electrode cable routing |

### Visualizer Issues

| Problem | Solution |
|---------|----------|
| No streams found | Make sure bioradio.py with --lsl flag is running first |
| Plot not updating | Click "Connect Selected" after selecting the stream |
| Recording not saving | Check output directory permissions; ensure sufficient disk space |

### Common Python Errors

```
ImportError: No module named 'serial'
```
→ Run `pip install pyserial` (note: import as `serial`, install as `pyserial`)

```
ImportError: No module named 'pylsl'
```
→ Run `pip install pylsl`

```
serial.SerialException: could not open port
```
→ Another program may be using the port; close BioCapture or other serial monitors

---

## Deliverables

Submit the following to MyCourses:

### 1. Data Package (ZIP file)

* All collected EOG CSV files organized by participant
* Include both raw and any processed data files

### 2. Analysis Notebook

* Completed `Lab2_EOG_Analysis.ipynb` with all cells executed
* Include figures showing:
  - Raw EOG signals for each gaze direction
  - Processed/filtered signals
  - PCA/ICA visualizations
  - SVM classification results (confusion matrix, accuracy)

### 3. Lab Report

Answer all questions (Q1-Q15) in your notebook or a separate document:

1. EOG waveforms for each gaze direction
2. Blink artifacts in HEOG vs. VEOG
3. Trial-to-trial consistency
4. Baseline drift observations
5. Effect of filtering on signal quality
6. Feature separation between gaze classes
7. PCA variance explanation
8. ICA component interpretation
9. Linear vs. RBF SVM comparison
10. Confusion matrix analysis
11. Effect of reducing gaze classes
12. EOG vs. EMG signal differences
13. EOG advantages and limitations
14. Practical interface design considerations
15. Using blinks as control input

---

## Additional Resources

* **Lab Streaming Layer (LSL):** https://labstreaminglayer.org/
* **BioRadio Documentation:** https://glneurotech.com/products/bioradio/
* **EOG Signal Processing:** Bulling, A., et al. (2011). Eye movement analysis for activity recognition using electrooculography. IEEE TPAMI.
* **scikit-learn SVM:** https://scikit-learn.org/stable/modules/svm.html
* **ICA Tutorial:** https://scikit-learn.org/stable/modules/decomposition.html#ica

---

## Acknowledgments

This lab uses a custom Python implementation of the BioRadio protocol, reverse-engineered from the GLNeuroTech SDK to enable cross-platform biosignal acquisition without proprietary software dependencies.

---

*Last updated: February 2025*
