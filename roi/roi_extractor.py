"""
Layer 2 — ROI Extractor
Extracts mean RGB signals from each face ROI per frame.
Applies skin segmentation mask to exclude eyes, lips, hair.
Handles adaptive ROI resizing based on face distance from camera.
"""

import cv2
import numpy as np
from dataclasses import dataclass, field
from typing import Optional
from input.face_tracking import FaceData, FaceROI


@dataclass
class ROISample:
    """
    One frame's worth of RGB means across all ROIs.
    This is the raw signal fed into the rPPG engine.
    """
    timestamp: float
    frame_idx: int
    forehead:    Optional[np.ndarray] = None   # shape (3,) = [R, G, B]
    left_cheek:  Optional[np.ndarray] = None
    right_cheek: Optional[np.ndarray] = None
    pixel_counts: dict = field(default_factory=dict)   # {roi_name: n_pixels_used}
    quality_ok: bool = True


class ROIExtractor:
    """
    Given a FaceData object, extracts mean RGB values per ROI
    after applying skin segmentation to remove non-skin pixels.

    Key responsibilities:
    - Mask-based extraction (polygon → binary mask → skin filter)
    - Adaptive ROI sizing based on estimated face distance
    - Per-channel mean computation (R, G, B separately)
    - Minimum pixel count guard (rejects ROIs with too few usable pixels)
    """

    # YCrCb skin detection thresholds (robust under varied lighting)
    SKIN_Y_MIN,  SKIN_Y_MAX  = 0,   255
    SKIN_CR_MIN, SKIN_CR_MAX = 133, 173
    SKIN_CB_MIN, SKIN_CB_MAX = 77,  127

    MIN_SKIN_PIXELS = 80     # below this → ROI unusable
    MIN_FACE_WIDTH  = 60     # pixels — face too small/far → warn

    def __init__(self):
        self._face_width_history = []

    # ------------------------------------------------------------------ #
    #  Public API                                                          #
    # ------------------------------------------------------------------ #

    def extract(self, face_data: FaceData, timestamp: float) -> Optional[ROISample]:
        """
        Extract RGB signal sample from a FaceData frame.
        Returns None if face not detected or no usable ROIs.
        """
        if not face_data.face_detected or not face_data.rois:
            return None

        frame = face_data.frame
        h, w = frame.shape[:2]

        # Adaptive ROI scale based on face distance proxy
        scale = self._compute_roi_scale(face_data)
        if scale < 0.4:
            print(f"[ROIExtractor] WARNING: face too far / small (scale={scale:.2f})")

        skin_mask = self._build_skin_mask(frame)
        sample = ROISample(timestamp=timestamp, frame_idx=face_data.frame.shape[0])

        name_map = {
            "forehead":    "forehead",
            "left_cheek":  "left_cheek",
            "right_cheek": "right_cheek",
        }

        all_ok = True
        for roi_name, attr in name_map.items():
            roi = face_data.rois.get(roi_name)
            if roi is None:
                all_ok = False
                continue

            rgb_mean, n_pixels = self._extract_roi_rgb(frame, roi, skin_mask, scale)
            sample.pixel_counts[roi_name] = n_pixels

            if rgb_mean is None or n_pixels < self.MIN_SKIN_PIXELS:
                all_ok = False
                print(f"[ROIExtractor] WARNING: {roi_name} has too few skin pixels ({n_pixels})")
                continue

            setattr(sample, attr, rgb_mean)

        sample.quality_ok = all_ok
        return sample

    # ------------------------------------------------------------------ #
    #  Skin segmentation                                                   #
    # ------------------------------------------------------------------ #

    def _build_skin_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate binary skin mask using YCrCb color space.
        Returns uint8 mask: 255 = skin, 0 = non-skin.
        """
        ycrcb = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
        lower = np.array([self.SKIN_Y_MIN,  self.SKIN_CR_MIN, self.SKIN_CB_MIN], dtype=np.uint8)
        upper = np.array([self.SKIN_Y_MAX,  self.SKIN_CR_MAX, self.SKIN_CB_MAX], dtype=np.uint8)
        mask = cv2.inRange(ycrcb, lower, upper)

        # Morphological cleanup: remove noise, fill small gaps
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN,  kernel, iterations=1)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        return mask

    # ------------------------------------------------------------------ #
    #  RGB extraction                                                      #
    # ------------------------------------------------------------------ #

    def _extract_roi_rgb(
        self,
        frame: np.ndarray,
        roi: FaceROI,
        skin_mask: np.ndarray,
        scale: float = 1.0,
    ) -> tuple[Optional[np.ndarray], int]:
        """
        Extract mean [R, G, B] from pixels inside roi.polygon that pass skin mask.

        Args:
            frame:      BGR frame
            roi:        FaceROI with polygon
            skin_mask:  binary mask (255=skin)
            scale:      adaptive scale factor for ROI shrink/grow

        Returns:
            (rgb_mean array shape (3,), n_pixels_used) or (None, 0)
        """
        h, w = frame.shape[:2]

        # Apply scale to polygon around its centroid
        poly = self._scale_polygon(roi.polygon, roi.center, scale)

        # Build polygon mask
        poly_mask = np.zeros((h, w), dtype=np.uint8)
        cv2.fillPoly(poly_mask, [poly], 255)

        # Combine with skin mask
        combined = cv2.bitwise_and(poly_mask, skin_mask)

        n_pixels = int(np.count_nonzero(combined))
        if n_pixels == 0:
            return None, 0

        # Extract pixels — note: frame is BGR, we want RGB
        bgr = frame.copy()
        b = bgr[:, :, 0].astype(np.float32)
        g = bgr[:, :, 1].astype(np.float32)
        r = bgr[:, :, 2].astype(np.float32)

        mask_bool = combined > 0
        r_mean = float(r[mask_bool].mean())
        g_mean = float(g[mask_bool].mean())
        b_mean = float(b[mask_bool].mean())

        return np.array([r_mean, g_mean, b_mean], dtype=np.float32), n_pixels

    # ------------------------------------------------------------------ #
    #  Adaptive scaling                                                    #
    # ------------------------------------------------------------------ #

    def _compute_roi_scale(self, face_data: FaceData) -> float:
        """
        Estimate face-distance proxy from bounding box width.
        Larger face = closer = scale up ROI for more pixels.
        Smaller face = farther = scale down to stay inside face bounds.
        """
        if face_data.bbox is None:
            return 1.0
        _, _, bw, _ = face_data.bbox
        self._face_width_history.append(bw)
        if len(self._face_width_history) > 30:
            self._face_width_history.pop(0)
        ref_width = np.mean(self._face_width_history) if self._face_width_history else bw
        scale = np.clip(bw / max(ref_width, 1.0), 0.5, 1.4)
        return float(scale)

    def _scale_polygon(
        self,
        polygon: np.ndarray,
        center: tuple,
        scale: float,
    ) -> np.ndarray:
        """Scale polygon points around their centroid."""
        pts = polygon.reshape(-1, 2).astype(np.float32)
        cx, cy = center
        scaled = (pts - [cx, cy]) * scale + [cx, cy]
        return scaled.astype(np.int32).reshape(-1, 1, 2)

    # ------------------------------------------------------------------ #
    #  Visualization helper                                                #
    # ------------------------------------------------------------------ #

    def draw_roi_debug(self, frame: np.ndarray, face_data: FaceData) -> np.ndarray:
        """Overlay skin mask + ROI polygons for debugging."""
        vis = frame.copy()
        skin_mask = self._build_skin_mask(frame)
        scale = self._compute_roi_scale(face_data)

        # Tint skin pixels green
        green_overlay = vis.copy()
        green_overlay[skin_mask > 0] = [0, 180, 80]
        cv2.addWeighted(green_overlay, 0.18, vis, 0.82, 0, vis)

        # Draw ROI polygons
        colors = {"forehead": (0,255,180), "left_cheek": (80,200,255), "right_cheek": (255,180,80)}
        for name, roi in face_data.rois.items():
            poly = self._scale_polygon(roi.polygon, roi.center, scale)
            c = colors.get(name, (200, 200, 0))
            cv2.polylines(vis, [poly], True, c, 1)

        # Skin mask inset (top-right corner)
        h, w = frame.shape[:2]
        thumb_w, thumb_h = 120, 90
        skin_vis = cv2.cvtColor(skin_mask, cv2.COLOR_GRAY2BGR)
        skin_thumb = cv2.resize(skin_vis, (thumb_w, thumb_h))
        vis[8:8+thumb_h, w-thumb_w-8:w-8] = skin_thumb
        cv2.rectangle(vis, (w-thumb_w-8, 8), (w-8, 8+thumb_h), (60,60,60), 1)
        cv2.putText(vis, "SKIN MASK", (w-thumb_w-4, 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.28, (120,120,120), 1)

        return vis
