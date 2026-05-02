"""
Layer 1 — Webcam Capture
Handles camera access, frame rate normalization, lighting check, and motion detection.
"""

import cv2
import time
import numpy as np
from collections import deque
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class FrameMetadata:
    frame: np.ndarray
    timestamp: float
    frame_idx: int
    fps_actual: float
    is_dark: bool
    motion_score: float
    motion_flagged: bool


class WebcamCapture:
    """
    Manages webcam input with frame rate normalization,
    lighting quality check, and motion magnitude scoring.
    """

    TARGET_FPS = 30
    DARK_THRESHOLD = 40        # mean luminance below this = too dark
    MOTION_THRESHOLD = 8.0     # mean absolute diff between frames
    HISTORY = 10               # frames kept for FPS estimation

    def __init__(self, camera_index: int = 0, target_fps: int = TARGET_FPS):
        self.camera_index = camera_index
        self.target_fps = target_fps
        self.frame_interval = 1.0 / target_fps

        self._cap: Optional[cv2.VideoCapture] = None
        self._frame_idx = 0
        self._last_capture_time = 0.0
        self._timestamps: deque = deque(maxlen=self.HISTORY)
        self._prev_gray: Optional[np.ndarray] = None

    # ------------------------------------------------------------------ #
    #  Lifecycle                                                           #
    # ------------------------------------------------------------------ #

    def open(self) -> bool:
        self._cap = cv2.VideoCapture(self.camera_index)
        if not self._cap.isOpened():
            print(f"[WebcamCapture] ERROR: cannot open camera index {self.camera_index}")
            return False

        # Try to set capture FPS; camera may ignore this — normalization handles it
        self._cap.set(cv2.CAP_PROP_FPS, self.target_fps)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        print(f"[WebcamCapture] Opened camera {self.camera_index} "
              f"({int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
              f"{int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
              f"@ {int(self._cap.get(cv2.CAP_PROP_FPS))} fps reported)")
        return True

    def close(self):
        if self._cap and self._cap.isOpened():
            self._cap.release()
        print("[WebcamCapture] Camera released.")

    def __enter__(self):
        self.open()
        return self

    def __exit__(self, *_):
        self.close()

    # ------------------------------------------------------------------ #
    #  Frame reading                                                       #
    # ------------------------------------------------------------------ #

    def read(self) -> Optional[FrameMetadata]:
        """
        Read the next frame, applying timing throttle so downstream modules
        always receive frames at a consistent ~target_fps cadence.
        Returns None if the camera is not available or a frame cannot be read.
        """
        if self._cap is None or not self._cap.isOpened():
            return None

        # Throttle to target FPS
        now = time.perf_counter()
        elapsed = now - self._last_capture_time
        if elapsed < self.frame_interval:
            time.sleep(self.frame_interval - elapsed)
        self._last_capture_time = time.perf_counter()

        ret, frame = self._cap.read()
        if not ret or frame is None:
            print("[WebcamCapture] WARNING: dropped frame")
            return None

        ts = time.perf_counter()
        self._timestamps.append(ts)
        self._frame_idx += 1

        fps = self._estimate_fps()
        is_dark = self._check_lighting(frame)
        motion_score, motion_flagged = self._check_motion(frame)

        return FrameMetadata(
            frame=frame,
            timestamp=ts,
            frame_idx=self._frame_idx,
            fps_actual=fps,
            is_dark=is_dark,
            motion_score=motion_score,
            motion_flagged=motion_flagged,
        )

    # ------------------------------------------------------------------ #
    #  Internal diagnostics                                                #
    # ------------------------------------------------------------------ #

    def _estimate_fps(self) -> float:
        if len(self._timestamps) < 2:
            return float(self.target_fps)
        span = self._timestamps[-1] - self._timestamps[0]
        if span <= 0:
            return float(self.target_fps)
        return round((len(self._timestamps) - 1) / span, 1)

    def _check_lighting(self, frame: np.ndarray) -> bool:
        """Returns True if frame is too dark for reliable rPPG."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        mean_luminance = float(np.mean(gray))
        if mean_luminance < self.DARK_THRESHOLD:
            print(f"[WebcamCapture] WARNING: low lighting (luminance={mean_luminance:.1f})")
            return True
        return False

    def _check_motion(self, frame: np.ndarray) -> tuple[float, bool]:
        """
        Compute mean absolute difference between current and previous frame.
        High diff = subject is moving excessively → signals become unreliable.
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if self._prev_gray is None:
            self._prev_gray = gray
            return 0.0, False

        diff = cv2.absdiff(gray, self._prev_gray)
        score = float(np.mean(diff))
        self._prev_gray = gray
        flagged = score > self.MOTION_THRESHOLD
        if flagged:
            print(f"[WebcamCapture] WARNING: excessive motion (score={score:.2f})")
        return score, flagged
