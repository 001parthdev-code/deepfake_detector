"""
Layer 5 — Trust Score Engine
Aggregates all signals into a final 0–100 trust score with verdict + explanation.
"""

import numpy as np
from dataclasses import dataclass, field
from collections import deque
from typing import Optional
from analysis.fft_analysis import PulseResult
from analysis.signal_validation import ValidationResult
from signal.rppg_extractor import PPGSignal


@dataclass
class TrustScore:
    score: float                          # 0–100
    verdict: str                          # "REAL" | "SUSPICIOUS" | "SYNTHETIC"
    confidence_label: str                 # "HIGH" | "MEDIUM" | "LOW"
    component_scores: dict                # breakdown for display
    explanations: list[str]               = field(default_factory=list)
    flags: list[str]                      = field(default_factory=list)


VERDICTS = {
    "REAL":       (70, 101),
    "SUSPICIOUS": (40,  70),
    "SYNTHETIC":  (0,   40),
}


class TrustScoreEngine:
    """
    Weighted decision layer that combines:
    - Pulse strength (from FFT prominence + SNR)
    - Frequency validity (pulse BPM detected and in range)
    - ROI consistency (spatial correlation between regions)
    - Temporal smoothness (BPM trajectory stability)
    - Signal quality (overall rPPG SNR)

    Scores are smoothed over a short rolling window to prevent flickering.
    """

    WEIGHTS = {
        "pulse_strength":    0.25,
        "frequency_validity":0.20,
        "roi_consistency":   0.20,
        "temporal_smooth":   0.20,
        "signal_quality":    0.15,
    }

    SMOOTH_WINDOW = 5    # rolling average over N score outputs

    def __init__(self):
        self._score_history: deque = deque(maxlen=self.SMOOTH_WINDOW)

    def compute(
        self,
        ppg: PPGSignal,
        pulse: Optional[PulseResult],
        validation: ValidationResult,
    ) -> TrustScore:

        components = {}
        explanations = list(validation.explanations)
        flags = list(validation.flags)

        # --- Pulse Strength (0–100) ---
        if pulse and pulse.valid:
            ps = float(np.clip(pulse.peak_prominence * 150, 0, 100))
        else:
            ps = 0.0
            flags.append("NO_PULSE_STRENGTH")
        components["pulse_strength"] = round(ps, 1)

        # --- Frequency Validity (0–100) ---
        if pulse and pulse.valid:
            fv = float(np.clip(pulse.confidence * 100, 0, 100))
        else:
            fv = 0.0
        components["frequency_validity"] = round(fv, 1)

        # --- ROI Consistency (0–100) ---
        rc = validation.spatial_consistency * 100
        components["roi_consistency"] = round(rc, 1)

        # --- Temporal Smoothness (0–100) ---
        ts = validation.temporal_stability * 100
        components["temporal_smooth"] = round(ts, 1)

        # --- Signal Quality (0–100) ---
        sq = validation.snr_score * 100
        components["signal_quality"] = round(sq, 1)

        # Weighted sum
        raw = (
            ps * self.WEIGHTS["pulse_strength"]    +
            fv * self.WEIGHTS["frequency_validity"] +
            rc * self.WEIGHTS["roi_consistency"]    +
            ts * self.WEIGHTS["temporal_smooth"]    +
            sq * self.WEIGHTS["signal_quality"]
        )

        # Biological constraint hard penalty
        if not validation.bio_constraint_pass:
            raw *= 0.4
            explanations.insert(0, "Biological constraints failed — heavy penalty applied")

        self._score_history.append(raw)
        smoothed = float(np.mean(self._score_history))
        smoothed = float(np.clip(smoothed, 0.0, 100.0))

        verdict = self._classify(smoothed)
        confidence = self._confidence_label(pulse, ppg)

        # Natural-language explanation assembly
        if not explanations:
            explanations.append("Insufficient data for analysis")

        return TrustScore(
            score=round(smoothed, 1),
            verdict=verdict,
            confidence_label=confidence,
            component_scores=components,
            explanations=explanations,
            flags=flags,
        )

    def reset(self):
        self._score_history.clear()

    # ------------------------------------------------------------------ #

    @staticmethod
    def _classify(score: float) -> str:
        for verdict, (lo, hi) in VERDICTS.items():
            if lo <= score < hi:
                return verdict
        return "SYNTHETIC"

    @staticmethod
    def _confidence_label(pulse: Optional[PulseResult], ppg: PPGSignal) -> str:
        if pulse is None or not pulse.valid:
            return "LOW"
        if pulse.confidence > 0.65 and ppg.snr_db > 8:
            return "HIGH"
        if pulse.confidence > 0.35:
            return "MEDIUM"
        return "LOW"
