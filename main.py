"""
main.py — Forensic rPPG Deepfake Detector
Orchestrates all 5 layers: capture → ROI → rPPG → FFT → validation → trust score.

Run:
    python main.py

Press  Q  to quit.
Press  D  to toggle skin/ROI debug overlay.
Press  R  to reset signal buffers.
"""

import sys
import time
import cv2
import numpy as np
from collections import deque

# ── Layer imports ──────────────────────────────────────────────────────
from input.webcam_capture    import WebcamCapture
from input.face_tracking     import FaceTracker
from roi.roi_extractor        import ROIExtractor
from signal.rppg_extractor   import RPPGExtractor
from analysis.fft_analysis   import FFTAnalyzer
from analysis.signal_validation import SignalValidator
from scoring.trust_score     import TrustScoreEngine, TrustScore
from utils.config import *


# ── Colour palette ─────────────────────────────────────────────────────
COL_GREEN  = (0, 200, 150)
COL_AMBER  = (30, 165, 245)
COL_RED    = (60, 60, 240)
COL_BLUE   = (255, 180, 80)
COL_MUTED  = (100, 110, 130)
COL_BG     = (20, 22, 28)
COL_WHITE  = (220, 225, 235)

VERDICT_COLORS = {
    "REAL":       COL_GREEN,
    "SUSPICIOUS": COL_AMBER,
    "SYNTHETIC":  COL_RED,
}


def draw_dashboard(
    frame: np.ndarray,
    face_vis: np.ndarray,
    wave_buf: deque,
    fft_result,
    trust: TrustScore | None,
    fps: float,
    frame_idx: int,
) -> np.ndarray:
    """
    Composite the full dashboard onto a single window:
    [Left: camera feed] [Right: signal panels]
    """
    H, W = 480, 960
    canvas = np.full((H, W, 3), COL_BG, dtype=np.uint8)

    # ── Left: camera feed ─────────────────────────────────────────────
    feed_h, feed_w = 360, 480
    feed = cv2.resize(face_vis, (feed_w, feed_h))
    canvas[0:feed_h, 0:feed_w] = feed

    # Status bar below feed
    bpm_str = f"BPM: {fft_result.bpm:.0f}" if fft_result and fft_result.valid else "BPM: --"
    fps_str = f"FPS: {fps:.1f}"
    cv2.putText(canvas, bpm_str, (8, feed_h + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, COL_GREEN, 1, cv2.LINE_AA)
    cv2.putText(canvas, fps_str, (8, feed_h + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.45, COL_MUTED, 1, cv2.LINE_AA)
    cv2.putText(canvas, f"Frame: {frame_idx}", (8, feed_h + 58), cv2.FONT_HERSHEY_SIMPLEX, 0.38, COL_MUTED, 1, cv2.LINE_AA)

    # ── Right panel ────────────────────────────────────────────────────
    rx = feed_w + 12
    rw = W - rx - 8

    # Section: Trust Score
    if trust:
        vc = VERDICT_COLORS.get(trust.verdict, COL_MUTED)
        _section_title(canvas, "TRUST SCORE", rx, 10)
        score_str = f"{trust.score:.0f}"
        cv2.putText(canvas, score_str, (rx + 4, 72),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.2, vc, 3, cv2.LINE_AA)
        cv2.putText(canvas, f"/ 100  [{trust.verdict}]  conf: {trust.confidence_label}",
                    (rx + 4, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.42, vc, 1, cv2.LINE_AA)

        # Component bars
        y0 = 114
        for name, val in trust.component_scores.items():
            label = name.replace("_", " ").upper()
            bar_len = int((val / 100) * (rw - 80))
            bar_c = COL_GREEN if val >= 70 else (COL_AMBER if val >= 45 else COL_RED)
            cv2.putText(canvas, label, (rx + 4, y0 + 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_MUTED, 1, cv2.LINE_AA)
            cv2.rectangle(canvas, (rx + 4, y0 + 13), (rx + 4 + rw - 80, y0 + 18), (40, 42, 50), -1)
            if bar_len > 0:
                cv2.rectangle(canvas, (rx + 4, y0 + 13), (rx + 4 + bar_len, y0 + 18), bar_c, -1)
            cv2.putText(canvas, f"{val:.0f}", (rx + rw - 72, y0 + 18),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.3, COL_WHITE, 1, cv2.LINE_AA)
            y0 += 24

        # Explanations
        _section_title(canvas, "ANALYSIS", rx, y0 + 6)
        ey = y0 + 22
        for line in trust.explanations[:5]:
            short = line[:55]
            col = COL_RED if any(w in line for w in ("fail","no pulse","mismatch","poor","Suspicious","static")) else COL_MUTED
            cv2.putText(canvas, short, (rx + 4, ey),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, col, 1, cv2.LINE_AA)
            ey += 13

        wave_y_top = ey + 8
    else:
        _section_title(canvas, "ACCUMULATING SIGNAL...", rx, 10)
        wave_y_top = 60

    # ── Waveform panel ─────────────────────────────────────────────────
    wave_h = 70
    _section_title(canvas, "rPPG WAVEFORM", rx, wave_y_top)
    _draw_waveform(canvas, wave_buf, rx + 4, wave_y_top + 12, rw - 8, wave_h)

    # ── FFT panel ──────────────────────────────────────────────────────
    fft_y = wave_y_top + wave_h + 28
    if fft_y + 70 < H:
        _section_title(canvas, "FFT SPECTRUM", rx, fft_y)
        _draw_fft(canvas, fft_result, rx + 4, fft_y + 12, rw - 8, 60)

    # Divider
    cv2.line(canvas, (feed_w, 0), (feed_w, H), (35, 38, 48), 1)

    return canvas


def _section_title(canvas, text, x, y):
    cv2.putText(canvas, text, (x + 4, y + 8),
                cv2.FONT_HERSHEY_SIMPLEX, 0.32, (60, 70, 90), 1, cv2.LINE_AA)


def _draw_waveform(canvas, wave_buf, x, y, w, h):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 33, 42), -1)
    if len(wave_buf) < 4:
        return
    arr = np.array(wave_buf)
    arr = arr - arr.mean()
    peak = max(abs(arr).max(), 1e-6)
    arr = arr / peak
    step = w / len(arr)
    pts = []
    for i, v in enumerate(arr):
        px = int(x + i * step)
        py = int(y + h / 2 - v * (h * 0.4))
        pts.append((px, py))
    for i in range(1, len(pts)):
        cv2.line(canvas, pts[i-1], pts[i], COL_GREEN, 1, cv2.LINE_AA)


def _draw_fft(canvas, fft_result, x, y, w, h):
    cv2.rectangle(canvas, (x, y), (x + w, y + h), (30, 33, 42), -1)
    if fft_result is None or len(fft_result.fft_freqs) == 0:
        return
    freqs = fft_result.fft_freqs
    power = fft_result.fft_power
    mask  = (freqs >= 0.5) & (freqs <= 4.5)
    if not mask.any():
        return
    f_band = freqs[mask]
    p_band = power[mask]
    p_max  = p_band.max() if p_band.max() > 0 else 1.0
    bar_w  = max(1, w // len(f_band))
    for i, (f, p) in enumerate(zip(f_band, p_band)):
        bar_h = int((p / p_max) * h * 0.85)
        bx    = x + int((f - 0.5) / 4.0 * w)
        is_peak = fft_result.valid and abs(f - fft_result.freq_hz) < 0.08
        c = COL_GREEN if is_peak else (50, 130, 90)
        cv2.rectangle(canvas, (bx, y + h - bar_h), (bx + bar_w, y + h), c, -1)
    if fft_result.valid:
        px = x + int((fft_result.freq_hz - 0.5) / 4.0 * w)
        cv2.line(canvas, (px, y), (px, y + h), COL_GREEN, 1)
        cv2.putText(canvas, f"{fft_result.freq_hz:.2f}Hz",
                    (max(px - 14, x), y + 9),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, COL_GREEN, 1)
    cv2.putText(canvas, "0.5", (x + 2, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COL_MUTED, 1)
    cv2.putText(canvas, "4.5Hz", (x + w - 28, y + h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.25, COL_MUTED, 1)


def main():
    print("=" * 60)
    print("  Forensic rPPG Deepfake Detector — Starting")
    print("  Press Q to quit | D for debug overlay | R to reset")
    print("=" * 60)

    # Instantiate all layers
    capture   = WebcamCapture(camera_index=CAMERA_INDEX, target_fps=TARGET_FPS)
    tracker   = FaceTracker(blur_threshold=BLUR_THRESHOLD)
    extractor = ROIExtractor()
    rppg      = RPPGExtractor(window_sec=WINDOW_SEC, fps=TARGET_FPS)
    fft_eng   = FFTAnalyzer()
    validator = SignalValidator()
    scorer    = TrustScoreEngine()

    wave_buf  = deque(maxlen=300)
    debug_mode = SHOW_SKIN_DEBUG
    last_trust = None
    last_pulse = None
    last_ppg   = None

    if not capture.open():
        print("ERROR: Cannot open webcam. Check CAMERA_INDEX in config.py")
        sys.exit(1)

    # Per-ROI separate signal buffers for spatial validation
    roi_signal_bufs: dict[str, deque] = {
        k: deque(maxlen=300) for k in ("forehead", "left_cheek", "right_cheek")
    }

    try:
        while True:
            # ── Layer 1: Capture ──────────────────────────────────────
            meta = capture.read()
            if meta is None:
                continue
            frame = meta.frame

            # ── Layer 1: Face tracking ────────────────────────────────
            face_data = tracker.process(frame)

            if debug_mode:
                vis = extractor.draw_roi_debug(frame, face_data)
            else:
                vis = tracker.draw_overlay(face_data)

            # ── Layer 2: ROI extraction ───────────────────────────────
            sample = extractor.extract(face_data, meta.timestamp)

            fft_result = last_pulse

            if sample is not None:
                # Feed per-ROI signals into spatial validator buffer
                for roi_name in ("forehead", "left_cheek", "right_cheek"):
                    rgb = getattr(sample, roi_name, None)
                    if rgb is not None:
                        roi_signal_bufs[roi_name].append(float(rgb[1]))  # green channel proxy

                # ── Layer 3: rPPG signal ──────────────────────────────
                ppg = rppg.push(sample)
                last_ppg = ppg

                if ppg and ppg.ready and len(ppg.signal) > 0:
                    # Append to waveform display buffer
                    wave_buf.extend(ppg.signal[-6:].tolist())

                    # ── Layer 4A: FFT ─────────────────────────────────
                    pulse = fft_eng.analyze(ppg)
                    last_pulse = pulse
                    fft_result = pulse

                    # Collect per-ROI arrays for spatial check
                    per_roi = {k: np.array(v) for k, v in roi_signal_bufs.items() if len(v) > 8}

                    # ── Layer 4B: Validation ──────────────────────────
                    validation = validator.validate(ppg, pulse, per_roi)

                    # ── Layer 5: Trust score ──────────────────────────
                    last_trust = scorer.compute(ppg, pulse, validation)

            # ── Dashboard rendering ───────────────────────────────────
            dashboard = draw_dashboard(
                frame=frame,
                face_vis=vis,
                wave_buf=wave_buf,
                fft_result=fft_result,
                trust=last_trust,
                fps=meta.fps_actual,
                frame_idx=meta.frame_idx,
            )

            cv2.imshow(WINDOW_TITLE, dashboard)

            # ── Keyboard controls ──────────────────────────────────────
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quit requested.")
                break
            elif key == ord('d'):
                debug_mode = not debug_mode
                print(f"Debug overlay: {'ON' if debug_mode else 'OFF'}")
            elif key == ord('r'):
                rppg.reset()
                fft_eng.reset()
                scorer.reset()
                wave_buf.clear()
                for b in roi_signal_bufs.values():
                    b.clear()
                last_trust = None
                last_pulse = None
                print("Buffers reset.")

    finally:
        capture.close()
        tracker.close()
        cv2.destroyAllWindows()
        print("Detector shut down cleanly.")


if __name__ == "__main__":
    main()
