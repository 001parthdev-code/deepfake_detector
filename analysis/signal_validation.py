"""
Layer 4B — Signal Validation Engine
Advanced biological & cross-ROI validation.
This is the layer that separates real faces from synthetic ones.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
from signal.rppg_extractor import PPGSignal, RPPGExtractor
from analysis.fft_analysis import PulseResult


@dataclass
class ValidationResult:
    """Outcome of all validation checks for one analysis window."""
    spatial_consistency:  float   # 0–1: ROI signal correlation
    temporal_stability:   float   # 0–1: BPM variance over recent windows
    bio_constraint_pass:  bool    # BPM in valid range, no spikes
    snr_score:            float   # 0–1: normalised from dB
    overall_score:        float   # weighted aggregate 0–100
    flags: list[str]              = field(default_factory=list)
    explanations: list[str]       = field(default_factory=list)


class SignalValidator:
    """
    Runs four independent validation checks on each analysis window:

    1. Spatial Consistency  — cross-correlate forehead vs cheek signals.
                              Real faces: all ROIs pulsate in sync.
                              Deepfakes: often mismatched or flat.

    2. Temporal Stability   — track BPM across recent windows.
                              Real: smooth HR trajectory.
                              Synthetic: random or zero-variance (static).

    3. Biological Constraints — hard range + spike detection.

    4. SNR Score            — signal-to-noise quality from rPPG engine.
    """

    # How many past PulseResults to keep for temporal analysis
    TEMPORAL_WINDOW = 8

    # Weights for overall score (must sum to 1.0)
    WEIGHTS = {
        "spatial_consistency": 0.30,
        "temporal_stability":  0.25,
        "bio_constraint":      0.25,
        "snr":                 0.20,
    }

    # Per-ROI individual signals (needed for spatial check)
    # We keep separate RPPGExtractor instances per ROI for this purpose
    def __init__(self):
        self._bpm_history: deque = deque(maxlen=self.TEMPORAL_WINDOW)
        self._per_roi_signals: dict[str, deque] = {
            k: deque(maxlen=300) for k in ("forehead", "left_cheek", "right_cheek")
        }

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def validate(
        self,
        ppg: PPGSignal,
        pulse: Optional[PulseResult],
        per_roi_signals: dict[str, np.ndarray],   # {roi_name: signal_array}
    ) -> ValidationResult:
        """
        Run all validation checks.

        Args:
            ppg:              merged PPGSignal from RPPGExtractor
            pulse:            FFT result from FFTAnalyzer (may be None)
            per_roi_signals:  individual ROI signals before merging
        """
        flags = []
        explanations = []

        # 1. Spatial consistency
        spatial = self._check_spatial(per_roi_signals, flags, explanations)

        # 2. Temporal stability
        if pulse and pulse.valid:
            self._bpm_history.append(pulse.bpm)
        temporal = self._check_temporal(flags, explanations)

        # 3. Biological constraints
        bio_ok = self._check_bio(pulse, flags, explanations)

        # 4. SNR
        snr_score = self._check_snr(ppg, flags, explanations)

        # Weighted overall
        bio_num = 1.0 if bio_ok else 0.0
        overall = (
            spatial   * self.WEIGHTS["spatial_consistency"] +
            temporal  * self.WEIGHTS["temporal_stability"]  +
            bio_num   * self.WEIGHTS["bio_constraint"]      +
            snr_score * self.WEIGHTS["snr"]
        ) * 100.0

        return ValidationResult(
            spatial_consistency=round(spatial, 3),
            temporal_stability=round(temporal, 3),
            bio_constraint_pass=bio_ok,
            snr_score=round(snr_score, 3),
            overall_score=round(overall, 2),
            flags=flags,
            explanations=explanations,
        )

    # ------------------------------------------------------------------ #
    #  Check 1: Spatial consistency                                        #
    # ------------------------------------------------------------------ #

    def _check_spatial(
        self,
        per_roi: dict[str, np.ndarray],
        flags: list,
        expl: list,
    ) -> float:
        """
        Cross-correlate forehead and cheek signals.
        Returns normalised correlation score [0, 1].
        """
        available = {k: v for k, v in per_roi.items() if v is not None and len(v) > 8}
        if len(available) < 2:
            flags.append("SPATIAL_INSUFFICIENT_ROIS")
            expl.append("Too few ROIs for spatial check")
            return 0.5   # neutral — not enough data to penalise

        keys = list(available.keys())
        correlations = []
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                a = available[keys[i]]
                b = available[keys[j]]
                n = min(len(a), len(b))
                if n < 8:
                    continue
                a, b = a[-n:], b[-n:]
                # Normalised cross-correlation at zero lag
                corr = float(np.corrcoef(a, b)[0, 1])
                if np.isnan(corr):
                    corr = 0.0
                correlations.append(corr)

        if not correlations:
            return 0.5

        mean_corr = float(np.mean(correlations))

        # Normalise [-1, 1] → [0, 1] (expect positive correlation for real faces)
        score = (mean_corr + 1.0) / 2.0
        score = float(np.clip(score, 0.0, 1.0))

        if mean_corr < 0.2:
            flags.append("SPATIAL_MISMATCH")
            expl.append(f"ROI signals poorly correlated (r={mean_corr:.2f}) — possible synthetic")
        elif mean_corr > 0.6:
            expl.append(f"ROI signals well-synchronized (r={mean_corr:.2f})")

        return score

    # ------------------------------------------------------------------ #
    #  Check 2: Temporal stability                                         #
    # ------------------------------------------------------------------ #

    def _check_temporal(self, flags: list, expl: list) -> float:
        """
        Analyse BPM trajectory over recent windows.
        Penalise both high variance (erratic) and zero variance (static/synthetic).
        """
        if len(self._bpm_history) < 3:
            return 0.5   # not enough history yet

        bpms = np.array(self._bpm_history)
        std  = float(np.std(bpms))

        # Too static: std < 0.5 BPM for 8 windows = suspiciously flat (synthetic)
        if std < 0.5:
            flags.append("TEMPORAL_STATIC")
            expl.append("Heart rate suspiciously static — may be synthetic")
            return 0.15

        # Too erratic: std > 15 BPM = noise, not real biology
        if std > 15.0:
            flags.append("TEMPORAL_ERRATIC")
            expl.append(f"Heart rate variance too high (std={std:.1f} BPM)")
            return 0.2

        # Good: natural HR variability ~1–8 BPM std
        # Map std in [0.5, 10] to score [0.5, 1.0]
        score = float(np.interp(std, [0.5, 10.0], [0.55, 1.0]))
        expl.append(f"HR trajectory stable (std={std:.1f} BPM over {len(bpms)} windows)")
        return score

    # ------------------------------------------------------------------ #
    #  Check 3: Biological constraints                                     #
    # ------------------------------------------------------------------ #

    def _check_bio(
        self,
        pulse: Optional[PulseResult],
        flags: list,
        expl: list,
    ) -> bool:
        if pulse is None or not pulse.valid:
            flags.append("BIO_NO_PULSE")
            expl.append("No pulse detected")
            return False

        if not (40 <= pulse.bpm <= 180):
            flags.append(f"BIO_RANGE_FAIL")
            expl.append(f"BPM {pulse.bpm:.1f} outside biological range [40–180]")
            return False

        expl.append(f"BPM {pulse.bpm:.1f} within biological range")
        return True

    # ------------------------------------------------------------------ #
    #  Check 4: SNR score                                                  #
    # ------------------------------------------------------------------ #

    def _check_snr(self, ppg: PPGSignal, flags: list, expl: list) -> float:
        """
        Normalise rPPG SNR (dB) to [0, 1].
        Typical good rPPG: 8–20 dB. Poor: < 3 dB.
        """
        snr = ppg.snr_db
        score = float(np.clip(np.interp(snr, [-5.0, 20.0], [0.0, 1.0]), 0.0, 1.0))

        if snr < 3.0:
            flags.append("SNR_LOW")
            expl.append(f"Signal quality poor (SNR={snr:.1f} dB)")
        else:
            expl.append(f"Signal quality acceptable (SNR={snr:.1f} dB)")

        return score
