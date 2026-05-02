# Forensic rPPG Deepfake Detector

A layered forensic system that detects deepfake video faces by analysing **remote photoplethysmography (rPPG)** — the subtle colour changes in skin caused by blood circulation. Real human faces pulse. Synthetic ones don't.

---

## Requirements

| Requirement | Minimum | Recommended |
|---|---|---|
| Python | **3.10** | 3.11 – 3.12 |
| Webcam | Any 720p | 1080p, 30fps |
| OS | Windows 10 / macOS 12 / Ubuntu 20.04 | — |
| RAM | 4 GB | 8 GB |

> **Python 3.10 is the hard minimum.** The codebase uses `X | Y` union type hints in function signatures (PEP 604), which were introduced in Python 3.10. Using an older version will raise a `TypeError` at import time.

---

## Installation

```bash
# 1. Clone or unzip the project
cd deepfake_detector

# 2. (Recommended) Create a virtual environment
python3 -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

### Dependencies

```
opencv-python >= 4.8.0     # webcam capture, drawing, colour conversion
mediapipe     >= 0.10.0    # face mesh landmark detection (478 points)
numpy         >= 1.24.0    # array operations throughout the pipeline
scipy         >= 1.11.0    # Butterworth bandpass filter, detrending
```

---

## Running

```bash
cd deepfake_detector
python main.py
```

The dashboard window opens immediately. The system needs **~10 seconds** to fill its sliding signal window before the first Trust Score appears. This is by design — rPPG requires enough frames to resolve a reliable heartbeat frequency.

### Keyboard Controls

| Key | Action |
|---|---|
| `Q` | Quit |
| `D` | Toggle skin segmentation debug overlay |
| `R` | Reset all signal buffers (use when switching faces) |

---

## Project Structure

```
deepfake_detector/
│
├── main.py                        ← Entry point, dashboard renderer
├── requirements.txt
├── utils/
│   └── config.py                  ← All tuneable thresholds in one place
│
├── input/
│   ├── webcam_capture.py          ← Layer 1: camera, FPS normalisation, motion/dark checks
│   └── face_tracking.py           ← Layer 1: MediaPipe face mesh, ROI polygons, blur score
│
├── roi/
│   └── roi_extractor.py           ← Layer 2: skin segmentation (YCrCb), adaptive ROI scaling
│
├── signal/
│   └── rppg_extractor.py          ← Layer 3: CHROM algorithm, detrend, bandpass filter
│
├── analysis/
│   ├── fft_analysis.py            ← Layer 4A: FFT, dominant frequency, peak prominence
│   └── signal_validation.py       ← Layer 4B: spatial consistency, temporal stability, bio constraints
│
└── scoring/
    └── trust_score.py             ← Layer 5: weighted trust score, verdict, explanations
```

---

## How It Works

### Layer 1 — Input & Capture
The webcam feed is normalised to a stable 30 fps regardless of the camera's native rate. Each frame is checked for low lighting (mean luminance < 40) and excessive motion (mean inter-frame difference > 8.0). MediaPipe FaceMesh is used to detect and continuously track the primary face (largest bounding box area when multiple faces are present), returning 478 3D landmarks per frame.

### Layer 2 — Region of Interest (ROI) Engine
Three ROIs are extracted: **forehead**, **left cheek**, and **right cheek**. Each ROI is defined by a polygon of MediaPipe landmarks. Before extracting colour values, a YCrCb skin segmentation mask is applied to exclude eyes, lips, teeth, and hair. The ROI polygons scale adaptively based on estimated face-to-camera distance.

### Layer 3 — rPPG Signal Engine
Implements the **CHROM algorithm** (de Haan & Jeanne, 2013). RGB channel means are extracted per ROI per frame and fed into a 10-second sliding window buffer. The CHROM algorithm normalises each channel, forms two chrominance projections (Xs = 3R − 2G, Ys = 1.5R + G − 1.5B), and combines them to isolate the pulse signal. The raw signal is detrended (linear drift removal) then bandpass filtered at 0.7–4 Hz (42–240 BPM). Signals from all three ROIs are merged with quality weighting.

### Layer 4 — Frequency & Validation Engine
A Hanning-windowed FFT identifies the dominant frequency in the cardiac band. The **peak prominence** (how far the peak exceeds the noise floor) is used as a reliability score. Alongside pulse detection, four independent validation checks run in parallel: spatial consistency (cross-ROI signal correlation), temporal stability (BPM variance over recent windows), biological constraints (40–180 BPM range, no sudden jumps), and SNR scoring.

### Layer 5 — Trust Score Engine
Five component scores are combined with fixed weights into a final 0–100 Trust Score:

| Component | Weight |
|---|---|
| Pulse Strength | 25% |
| Frequency Validity | 20% |
| ROI Consistency | 20% |
| Temporal Smoothness | 20% |
| Signal Quality | 15% |

Scores are smoothed over a 5-window rolling average to prevent flickering. The verdict thresholds are: **REAL** (≥ 70), **SUSPICIOUS** (40–69), **SYNTHETIC** (< 40). A failed biological constraint applies a 60% score penalty.

---

## Configuration (`utils/config.py`)

All key thresholds are centralised and documented:

```python
CAMERA_INDEX   = 0       # Change if using an external webcam
WINDOW_SEC     = 10      # Shorter = faster but noisier; longer = more stable
BLUR_THRESHOLD = 80.0    # Laplacian variance; lower = stricter blur rejection
DARK_THRESHOLD = 40      # Mean luminance cutoff
MOTION_THRESHOLD = 8.0   # Inter-frame diff cutoff
```

---

## Known Limitations

- Requires a **direct view of the face** — occlusions, masks, and heavy makeup degrade accuracy.
- Performance depends on **lighting consistency**. Flickering or pulsing artificial light can create false signals.
- The 10-second window means the system reacts slowly to sudden changes — this is an inherent rPPG constraint.
- Very dark skin tones may produce fewer skin-masked pixels, reducing signal quality. The `MIN_SKIN_PIXELS` threshold can be lowered in config if needed.
- This is a **research and demonstration tool**, not a certified forensic instrument. Do not use verdicts as sole evidence.

---

## References

- de Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. *IEEE Transactions on Biomedical Engineering*, 60(10), 2878–2886.
- MediaPipe Face Mesh — https://mediapipe.readthedocs.io/en/latest/solutions/face_mesh.html