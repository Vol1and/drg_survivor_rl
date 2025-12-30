import cv2
import numpy as np


class EdgeFeatureExtractor:
    def __init__(
        self,
        border_detector,
        max_distance_norm=40.0,
        confidence_smooth=0.9,

        # gating
        edge_active_dist=0.55,
        min_confidence=0.25,
    ):
        """
        border_detector: LevelBorderDetector
        """

        self.detector = border_detector
        self.max_distance_norm = max_distance_norm
        self.confidence_smooth = confidence_smooth

        self.edge_active_dist = edge_active_dist
        self.min_confidence = min_confidence

        self.prev_confidence = 0.0

    # ----------------------------------------------------
    def extract(self, frame):
        """
        Returns:
            np.array([dist, dx, dy, confidence])
        """

        mask = self.detector.extract(frame)

        h, w = mask.shape
        cy, cx = h // 2, w // 2

        ys, xs = np.where(mask > 0)

        # -----------------------------------
        # No detection at all
        # -----------------------------------
        if len(xs) == 0:
            self.prev_confidence *= 0.9
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        # -----------------------------------
        # Distance & direction
        # -----------------------------------
        dx = xs - cx
        dy = ys - cy
        dists = np.sqrt(dx**2 + dy**2)

        min_idx = np.argmin(dists)
        min_dist = dists[min_idx]

        norm_dist = np.clip(min_dist / self.max_distance_norm, 0.0, 1.0)

        # -----------------------------------
        # GATING: far from edge → ignore
        # -----------------------------------
        if norm_dist > self.edge_active_dist:
            self.prev_confidence *= 0.9
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        dir_x = dx[min_idx] / (min_dist + 1e-6)
        dir_y = dy[min_idx] / (min_dist + 1e-6)

        # -----------------------------------
        # Confidence
        # -----------------------------------
        raw_confidence = np.clip(len(xs) / (h * w * 0.02), 0.0, 1.0)

        confidence = (
            self.confidence_smooth * self.prev_confidence +
            (1.0 - self.confidence_smooth) * raw_confidence
        )

        self.prev_confidence = confidence

        # -----------------------------------
        # Final gating
        # -----------------------------------
        if confidence < self.min_confidence:
            return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

        return np.array([
            norm_dist,
            dir_x,
            dir_y,
            confidence
        ], dtype=np.float32)
