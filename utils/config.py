"""
utils/config.py — Central configuration for the deepfake detector.
Edit these values to tune the system for your camera and environment.
"""

# ── Input ──────────────────────────────────────────────────────────────
CAMERA_INDEX   = 0       # 0 = default webcam; try 1, 2 for external cameras
TARGET_FPS     = 30      # normalised capture rate
FRAME_WIDTH    = 640
FRAME_HEIGHT   = 480

# ── Signal window ──────────────────────────────────────────────────────
WINDOW_SEC     = 10      # rPPG sliding window length (seconds)
                         # Shorter = faster response but noisier
                         # Longer  = more stable but slower to update

# ── Bandpass ───────────────────────────────────────────────────────────
BP_LOW_HZ      = 0.7     # ~42 BPM minimum
BP_HIGH_HZ     = 4.0     # ~240 BPM maximum

# ── Thresholds ─────────────────────────────────────────────────────────
BLUR_THRESHOLD       = 80.0   # Laplacian variance; lower = stricter blur reject
DARK_THRESHOLD       = 40     # Mean luminance; lower = stricter darkness reject
MOTION_THRESHOLD     = 8.0    # Mean frame diff; lower = stricter motion reject
MIN_SKIN_PIXELS      = 80     # Minimum skin pixels per ROI to trust the sample
MIN_PEAK_PROMINENCE  = 0.15   # FFT peak must exceed noise floor by this fraction

# ── Scoring weights ────────────────────────────────────────────────────
# Must sum to 1.0
SCORE_WEIGHTS = {
    "pulse_strength":     0.25,
    "frequency_validity": 0.20,
    "roi_consistency":    0.20,
    "temporal_smooth":    0.20,
    "signal_quality":     0.15,
}

# ── Display ────────────────────────────────────────────────────────────
SHOW_SKIN_DEBUG  = False    # Set True to see skin segmentation overlay
SHOW_LANDMARKS   = False    # Set True to draw full MediaPipe mesh
WINDOW_TITLE     = "Forensic rPPG — Deepfake Detector"
