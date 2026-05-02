"""
Layer 1 — Face Tracking
Detects, tracks, and locks a primary face subject across frames.
Handles multi-face scenarios, blur scoring, and provides a stable face lock.
"""

import cv2
import numpy as np
import mediapipe as mp
from dataclasses import dataclass, field
from typing import Optional


# MediaPipe landmark indices for key face regions
LANDMARK_GROUPS = {
    # Forehead: between eyebrows and hairline
    "forehead": [10, 338, 297, 332, 284, 251, 389, 356, 454, 323,
                 361, 288, 397, 365, 379, 378, 400, 377, 152, 148,
                 176, 149, 150, 136, 172, 58, 132, 93, 234, 127,
                 162, 21, 54, 103, 67, 109],
    # Left cheek (from subject's perspective)
    "left_cheek":  [116, 117, 118, 119, 120, 121, 128, 245, 188, 174, 
                    145, 153, 154, 155, 133, 173, 157, 158, 159, 160],
    # Right cheek (from subject's perspective)
    "right_cheek": [345, 346, 347, 348, 349, 350, 357, 465, 412, 399,
                    374, 380, 381, 382, 362, 398, 384, 385, 386, 387],
}

# Minimal tight ROI landmark indices (more stable for rPPG sampling)
ROI_TIGHT = {
    "forehead":    [10, 338, 297, 332, 284, 251, 108, 69, 54, 103, 67, 109],
    "left_cheek":  [116, 123, 147, 213, 192, 214, 210, 211, 32, 271, 208, 199],
    "right_cheek": [345, 352, 376, 433, 416, 434, 430, 431, 262, 41, 428, 420],
}


@dataclass
class FaceROI:
    """Pixel coordinates of a single ROI region (bounding box + polygon mask)."""
    name: str
    polygon: np.ndarray   # shape (N, 2), dtype int32
    bbox: tuple           # (x, y, w, h)
    center: tuple         # (cx, cy)


@dataclass
class FaceData:
    """All face information extracted from a single frame."""
    frame: np.ndarray
    face_detected: bool
    face_id: int                          # always 0 for primary subject
    bbox: Optional[tuple] = None          # (x, y, w, h) of whole face
    landmarks_px: Optional[np.ndarray] = None   # shape (478, 2)
    rois: dict = field(default_factory=dict)     # {name: FaceROI}
    blur_score: float = 0.0
    is_blurry: bool = False
    confidence: float = 0.0


class FaceTracker:
    """
    Wraps MediaPipe FaceMesh to provide:
    - Continuous per-frame face detection & tracking
    - Primary subject lock (largest face = closest to camera)
    - Three ROI polygons: forehead, left_cheek, right_cheek
    - Blur detection (Laplacian variance)
    - Multi-face handling (ignores non-primary faces)
    """

    BLUR_THRESHOLD = 80.0      # Laplacian variance below this = blurry

    def __init__(self, blur_threshold: float = BLUR_THRESHOLD):
        self.blur_threshold = blur_threshold
        self._mp_face_mesh = mp.solutions.face_mesh
        self._face_mesh = self._mp_face_mesh.FaceMesh(
            max_num_faces=4,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self._drawing = mp.solutions.drawing_utils
        self._drawing_spec = self._drawing.DrawingSpec(
            color=(0, 200, 150), thickness=1, circle_radius=1
        )

    def process(self, frame: np.ndarray) -> FaceData:
        """
        Run face mesh on a BGR frame.
        Returns FaceData for the primary subject (largest face).
        """
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self._face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return FaceData(frame=frame, face_detected=False, face_id=0)

        # Select primary face = one whose bounding box has the largest area
        primary_lm, primary_bbox = self._select_primary_face(
            results.multi_face_landmarks, w, h
        )

        landmarks_px = self._landmarks_to_pixels(primary_lm, w, h)
        rois = self._extract_rois(landmarks_px, w, h)
        blur_score = self._compute_blur(frame, primary_bbox)
        confidence = self._estimate_confidence(primary_lm)

        return FaceData(
            frame=frame,
            face_detected=True,
            face_id=0,
            bbox=primary_bbox,
            landmarks_px=landmarks_px,
            rois=rois,
            blur_score=blur_score,
            is_blurry=blur_score < self.blur_threshold,
            confidence=confidence,
        )

    # ------------------------------------------------------------------ #
    #  Rendering helpers                                                   #
    # ------------------------------------------------------------------ #

    def draw_overlay(self, face_data: FaceData) -> np.ndarray:
        """Return a copy of the frame with face box, ROIs, and status rendered."""
        vis = face_data.frame.copy()

        if not face_data.face_detected:
            cv2.putText(vis, "NO FACE DETECTED", (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 60, 240), 2)
            return vis

        # Face bounding box
        x, y, bw, bh = face_data.bbox
        color = (0, 200, 150) if not face_data.is_blurry else (0, 140, 255)
        cv2.rectangle(vis, (x, y), (x + bw, y + bh), color, 1)

        # ROI polygons
        roi_colors = {
            "forehead":    (0, 255, 180),
            "left_cheek":  (80, 200, 255),
            "right_cheek": (80, 200, 255),
        }
        for name, roi in face_data.rois.items():
            c = roi_colors.get(name, (200, 200, 0))
            cv2.polylines(vis, [roi.polygon], isClosed=True, color=c, thickness=1)
            overlay = vis.copy()
            cv2.fillPoly(overlay, [roi.polygon], c)
            cv2.addWeighted(overlay, 0.12, vis, 0.88, 0, vis)
            cx, cy = roi.center
            cv2.putText(vis, name.replace("_", " ").upper(),
                        (cx - 30, cy), cv2.FONT_HERSHEY_SIMPLEX,
                        0.28, c, 1, cv2.LINE_AA)

        # HUD text
        self._draw_hud(vis, face_data)
        return vis

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _select_primary_face(self, all_landmarks, w: int, h: int):
        """Pick the face with the largest bounding-box area."""
        best_lm, best_bbox, best_area = None, None, 0
        for lm in all_landmarks:
            xs = [p.x * w for p in lm.landmark]
            ys = [p.y * h for p in lm.landmark]
            x1, y1 = int(min(xs)), int(min(ys))
            x2, y2 = int(max(xs)), int(max(ys))
            area = (x2 - x1) * (y2 - y1)
            if area > best_area:
                best_area = area
                best_lm = lm
                best_bbox = (x1, y1, x2 - x1, y2 - y1)
        return best_lm, best_bbox

    def _landmarks_to_pixels(self, lm_proto, w: int, h: int) -> np.ndarray:
        pts = np.array([[p.x * w, p.y * h] for p in lm_proto.landmark], dtype=np.float32)
        return pts

    def _extract_rois(self, lm_px: np.ndarray, w: int, h: int) -> dict:
        rois = {}
        for name, indices in ROI_TIGHT.items():
            valid = [i for i in indices if i < len(lm_px)]
            if len(valid) < 3:
                continue
            pts = lm_px[valid].astype(np.int32)
            pts = np.clip(pts, [0, 0], [w - 1, h - 1])
            x, y = pts[:, 0].min(), pts[:, 1].min()
            bw = pts[:, 0].max() - x
            bh = pts[:, 1].max() - y
            cx, cy = int(pts[:, 0].mean()), int(pts[:, 1].mean())
            rois[name] = FaceROI(
                name=name,
                polygon=pts.reshape(-1, 1, 2),
                bbox=(int(x), int(y), int(bw), int(bh)),
                center=(cx, cy),
            )
        return rois

    def _compute_blur(self, frame: np.ndarray, bbox: tuple) -> float:
        x, y, bw, bh = bbox
        x, y = max(0, x), max(0, y)
        crop = frame[y:y+bh, x:x+bw]
        if crop.size == 0:
            return 0.0
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _estimate_confidence(self, lm_proto) -> float:
        """Proxy: fraction of landmarks with visibility > 0.5."""
        total = len(lm_proto.landmark)
        if total == 0:
            return 0.0
        visible = sum(1 for p in lm_proto.landmark if getattr(p, "visibility", 1.0) > 0.5)
        return round(visible / total, 3)

    def _draw_hud(self, vis: np.ndarray, fd: FaceData):
        lines = [
            f"BPM est: --",
            f"Blur: {fd.blur_score:.0f}{'  [BLURRY]' if fd.is_blurry else ''}",
            f"Conf: {fd.confidence:.2f}",
        ]
        for i, ln in enumerate(lines):
            cv2.putText(vis, ln, (8, 18 + i * 16),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.38,
                        (0, 200, 150), 1, cv2.LINE_AA)

    def close(self):
        self._face_mesh.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
