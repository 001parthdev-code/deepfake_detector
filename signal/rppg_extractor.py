"""
Layer 3 — rPPG Signal Engine
Implements CHROM algorithm to extract pulse signal from RGB samples.
Maintains a sliding window buffer, applies detrending and bandpass filtering.
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional
from scipy.signal import butter, filtfilt, detrend as scipy_detrend
from roi.roi_extractor import ROISample


@dataclass
class PPGSignal:
    """One window's worth of extracted pulse signal + metadata."""
    signal: np.ndarray          # cleaned rPPG signal, shape (N,)
    timestamps: np.ndarray      # shape (N,)
    fps: float
    window_seconds: float
    roi_source: str             # which ROI contributed most
    snr_db: float               # estimated signal-to-noise ratio
    ready: bool                 # False if window not full yet


class RPPGExtractor:
    """
    Extracts remote photoplethysmography (rPPG) signal using the CHROM algorithm.

    Algorithm (de Haan & Jeanne, 2013):
    1. Normalize RGB channels per frame
    2. Build two chrominance signals: Xs = 3R - 2G, Ys = 1.5R + G - 1.5B
    3. Combine: S = Xs - (std(Xs)/std(Ys)) * Ys
    4. Detrend + bandpass filter (0.7–4 Hz)

    Signal sources are merged across forehead and cheeks with quality weighting.
    """

    WINDOW_SEC   = 10      # sliding window length in seconds
    TARGET_FPS   = 30      # assumed input frame rate
    LOW_HZ       = 0.7     # bandpass low cutoff  (~42 BPM)
    HIGH_HZ      = 4.0     # bandpass high cutoff (~240 BPM)
    FILTER_ORDER = 4

    def __init__(self, window_sec: float = WINDOW_SEC, fps: float = TARGET_FPS):
        self.window_sec = window_sec
        self.fps = fps
        self._buf_size = int(window_sec * fps)

        # Circular buffers for each channel × each ROI
        self._R: dict[str, deque] = {k: deque(maxlen=self._buf_size) for k in ("forehead","left_cheek","right_cheek")}
        self._G: dict[str, deque] = {k: deque(maxlen=self._buf_size) for k in ("forehead","left_cheek","right_cheek")}
        self._B: dict[str, deque] = {k: deque(maxlen=self._buf_size) for k in ("forehead","left_cheek","right_cheek")}
        self._timestamps: deque = deque(maxlen=self._buf_size)
        self._pixel_counts: dict[str, deque] = {k: deque(maxlen=self._buf_size) for k in ("forehead","left_cheek","right_cheek")}

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def push(self, sample: ROISample) -> Optional[PPGSignal]:
        """
        Ingest one ROISample. Returns a PPGSignal once the window is filled,
        otherwise returns None (still accumulating).
        """
        self._timestamps.append(sample.timestamp)

        for roi_name in ("forehead", "left_cheek", "right_cheek"):
            rgb = getattr(sample, roi_name, None)
            n   = sample.pixel_counts.get(roi_name, 0)
            if rgb is not None and len(rgb) == 3:
                self._R[roi_name].append(rgb[0])
                self._G[roi_name].append(rgb[1])
                self._B[roi_name].append(rgb[2])
                self._pixel_counts[roi_name].append(n)
            else:
                # Pad with last known value or zero to keep buffers aligned
                for buf in (self._R[roi_name], self._G[roi_name], self._B[roi_name]):
                    buf.append(buf[-1] if buf else 0.0)
                self._pixel_counts[roi_name].append(0)

        if len(self._timestamps) < self._buf_size:
            return PPGSignal(
                signal=np.array([]), timestamps=np.array([]),
                fps=self.fps, window_seconds=self.window_sec,
                roi_source="accumulating", snr_db=0.0, ready=False
            )

        return self._compute_signal()

    def reset(self):
        for d in (*self._R.values(), *self._G.values(), *self._B.values(),
                  self._timestamps, *self._pixel_counts.values()):
            d.clear()

    # ------------------------------------------------------------------ #
    #  CHROM algorithm                                                     #
    # ------------------------------------------------------------------ #

    def _compute_signal(self) -> PPGSignal:
        signals = {}
        weights = {}

        for roi in ("forehead", "left_cheek", "right_cheek"):
            R = np.array(self._R[roi], dtype=np.float64)
            G = np.array(self._G[roi], dtype=np.float64)
            B = np.array(self._B[roi], dtype=np.float64)
            pix = np.array(self._pixel_counts[roi], dtype=np.float64)

            if R.std() < 1e-6:
                continue

            sig = self._chrom(R, G, B)
            if sig is None:
                continue

            sig = self._detrend(sig)
            sig = self._bandpass(sig, self.fps, self.LOW_HZ, self.HIGH_HZ)

            snr = self._snr(sig, self.fps)
            weight = float(np.mean(pix)) * max(snr, 0.01)

            signals[roi] = sig
            weights[roi] = weight

        if not signals:
            # Return flat signal — no usable data
            n = self._buf_size
            return PPGSignal(
                signal=np.zeros(n),
                timestamps=np.array(self._timestamps),
                fps=self.fps, window_seconds=self.window_sec,
                roi_source="none", snr_db=0.0, ready=True
            )

        # Weighted average across ROIs
        total_weight = sum(weights.values())
        merged = np.zeros(self._buf_size)
        for roi, sig in signals.items():
            merged += sig * (weights[roi] / total_weight)

        best_roi = max(weights, key=weights.get)
        snr = self._snr(merged, self.fps)

        return PPGSignal(
            signal=merged,
            timestamps=np.array(self._timestamps),
            fps=self.fps,
            window_seconds=self.window_sec,
            roi_source=best_roi,
            snr_db=snr,
            ready=True,
        )

    @staticmethod
    def _chrom(R: np.ndarray, G: np.ndarray, B: np.ndarray) -> Optional[np.ndarray]:
        """
        CHROM rPPG algorithm (de Haan & Jeanne, 2013).
        Normalize → chrominance projection → combine.
        """
        # Normalize each channel by its mean
        Rn = R / (R.mean() + 1e-8)
        Gn = G / (G.mean() + 1e-8)
        Bn = B / (B.mean() + 1e-8)

        # Two orthogonal chrominance channels
        Xs = 3 * Rn - 2 * Gn
        Ys = 1.5 * Rn + Gn - 1.5 * Bn

        std_xs = Xs.std()
        std_ys = Ys.std()

        if std_ys < 1e-8:
            return None

        alpha = std_xs / std_ys
        S = Xs - alpha * Ys
        return S

    @staticmethod
    def _detrend(signal: np.ndarray) -> np.ndarray:
        """Remove slow drift with scipy linear detrend."""
        return scipy_detrend(signal)

    @staticmethod
    def _bandpass(
        signal: np.ndarray,
        fps: float,
        low: float,
        high: float,
        order: int = 4,
    ) -> np.ndarray:
        """
        Zero-phase Butterworth bandpass filter.
        Keeps only frequencies in [low, high] Hz.
        """
        nyq = fps / 2.0
        lo  = low  / nyq
        hi  = high / nyq
        lo  = np.clip(lo, 1e-4, 0.999)
        hi  = np.clip(hi, 1e-4, 0.999)
        if lo >= hi:
            return signal
        b, a = butter(order, [lo, hi], btype="band")
        # filtfilt needs at least padlen samples
        if len(signal) <= 3 * order:
            return signal
        return filtfilt(b, a, signal)

    @staticmethod
    def _snr(signal: np.ndarray, fps: float) -> float:
        """
        Estimate SNR in dB by comparing peak power in the cardiac band (0.7–4 Hz)
        to total power.
        """
        if len(signal) < 4:
            return 0.0
        freqs  = np.fft.rfftfreq(len(signal), d=1.0 / fps)
        power  = np.abs(np.fft.rfft(signal)) ** 2
        cardiac = (freqs >= 0.7) & (freqs <= 4.0)
        p_signal = power[cardiac].sum()
        p_total  = power.sum() + 1e-12
        if p_signal <= 0:
            return 0.0
        return float(10 * np.log10(p_signal / p_total))
