"""
Layer 4A — FFT Analysis & Pulse Detection
Converts rPPG signal to BPM via FFT dominant frequency peak.
Validates peak prominence (real signal vs noise floor).
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional
from signal.rppg_extractor import PPGSignal


@dataclass
class PulseResult:
    bpm: float                    # dominant frequency converted to BPM
    freq_hz: float                # dominant frequency in Hz
    peak_prominence: float        # how much the peak stands above noise floor
    confidence: float             # 0.0–1.0
    valid: bool                   # passes all biological constraints
    fft_freqs: np.ndarray         # for plotting
    fft_power: np.ndarray         # for plotting
    rejection_reason: str = ""    # non-empty if valid=False


class FFTAnalyzer:
    """
    Performs FFT on the rPPG signal to extract dominant pulse frequency.

    Biological constraints enforced:
    - BPM must be in [40, 180]
    - Peak must be sufficiently prominent above noise floor
    - No sudden impossible BPM jumps between windows
    """

    BPM_MIN     = 40
    BPM_MAX     = 180
    HZ_MIN      = BPM_MIN / 60     # 0.667 Hz
    HZ_MAX      = BPM_MAX / 60     # 3.0   Hz
    MIN_PROMINENCE = 0.15          # peak must be >15% above mean noise
    MAX_BPM_JUMP   = 25            # max allowed BPM change between windows

    def __init__(self):
        self._last_bpm: Optional[float] = None

    def analyze(self, ppg: PPGSignal) -> Optional[PulseResult]:
        """
        Run FFT on ppg.signal. Returns PulseResult or None if signal not ready.
        """
        if not ppg.ready or len(ppg.signal) < 16:
            return None

        signal = ppg.signal
        fps    = ppg.fps
        N      = len(signal)

        # Apply Hanning window to reduce spectral leakage
        windowed = signal * np.hanning(N)

        freqs = np.fft.rfftfreq(N, d=1.0 / fps)
        power = np.abs(np.fft.rfft(windowed)) ** 2

        # Restrict to cardiac band
        band_mask = (freqs >= self.HZ_MIN) & (freqs <= self.HZ_MAX)
        if not band_mask.any():
            return self._reject("no frequencies in cardiac band", freqs, power)

        band_power = power.copy()
        band_power[~band_mask] = 0.0

        peak_idx   = int(np.argmax(band_power))
        peak_freq  = float(freqs[peak_idx])
        peak_power = float(power[peak_idx])
        bpm        = peak_freq * 60.0

        # --- Prominence check ---
        noise_floor = float(np.mean(power[band_mask]))
        prominence  = (peak_power - noise_floor) / (noise_floor + 1e-12)

        if prominence < self.MIN_PROMINENCE:
            return self._reject(
                f"low peak prominence ({prominence:.3f} < {self.MIN_PROMINENCE})",
                freqs, power
            )

        # --- Biological range ---
        if not (self.BPM_MIN <= bpm <= self.BPM_MAX):
            return self._reject(
                f"BPM {bpm:.1f} outside biological range [{self.BPM_MIN},{self.BPM_MAX}]",
                freqs, power
            )

        # --- Temporal stability ---
        if self._last_bpm is not None:
            jump = abs(bpm - self._last_bpm)
            if jump > self.MAX_BPM_JUMP:
                return self._reject(
                    f"BPM jump too large ({jump:.1f} > {self.MAX_BPM_JUMP})",
                    freqs, power
                )

        self._last_bpm = bpm

        # Confidence: blend prominence and SNR
        conf = float(np.clip(prominence / 2.0, 0.0, 1.0))

        return PulseResult(
            bpm=round(bpm, 1),
            freq_hz=round(peak_freq, 3),
            peak_prominence=round(prominence, 4),
            confidence=round(conf, 3),
            valid=True,
            fft_freqs=freqs,
            fft_power=power,
        )

    def reset(self):
        self._last_bpm = None

    # ------------------------------------------------------------------ #

    def _reject(self, reason: str, freqs, power) -> PulseResult:
        return PulseResult(
            bpm=0.0, freq_hz=0.0, peak_prominence=0.0,
            confidence=0.0, valid=False,
            fft_freqs=freqs, fft_power=power,
            rejection_reason=reason,
        )
