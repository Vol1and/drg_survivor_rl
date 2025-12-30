import cv2
import numpy as np


class LevelBorderDetector:
    def __init__(
        self,
            base_hsv=(145, 70, 30),
            h_tol = 22,

            s_tol = 60,
            v_tol = 70,

        # spatial consistency
        neighborhood=7,
        min_density=35,

        # temporal filtering
        alpha=0.9,
        threshold=0.5,
        flow_scale=0.6,

        # cleanup
        blur_ksize=3,
        min_area=450,
    ):
        self.base_hsv = np.array(base_hsv, dtype=np.float32)

        self.h_tol = h_tol
        self.s_tol = s_tol
        self.v_tol = v_tol

        self.neighborhood = neighborhood
        self.min_density = min_density

        self.alpha = alpha
        self.threshold = threshold
        self.flow_scale = flow_scale

        self.blur_ksize = blur_ksize
        self.min_area = min_area

        self.prev_gray = None
        self.accumulator = None

    # -------------------------------------------------
    # MAIN ENTRY
    # -------------------------------------------------
    def extract(self, frame_bgr: np.ndarray) -> np.ndarray:
        mask = self._spatial_mask(frame_bgr)
        mask = self._temporal_filter(frame_bgr, mask)
        return mask

    # -------------------------------------------------
    # SPATIAL MASK (color + density)
    # -------------------------------------------------
    def _spatial_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)

        if self.blur_ksize > 1:
            hsv = cv2.GaussianBlur(hsv, (self.blur_ksize, self.blur_ksize), 0)

        h, s, v = cv2.split(hsv)

        # color similarity
        dh = np.abs(h.astype(np.int16) - self.base_hsv[0])
        ds = np.abs(s.astype(np.int16) - self.base_hsv[1])
        dv = np.abs(v.astype(np.int16) - self.base_hsv[2])

        color_mask = (
            (dh < self.h_tol) &
            (ds < self.s_tol) &
            (dv < self.v_tol)
        ).astype(np.uint8)

        # density filter
        density = cv2.boxFilter(
            color_mask,
            ddepth=-1,
            ksize=(self.neighborhood, self.neighborhood),
            normalize=False,
        )

        dense_mask = density >= self.min_density
        mask = dense_mask.astype(np.uint8) * 255

        # cleanup
        kernel = np.ones((3, 3), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        clean = np.zeros_like(mask)

        for i in range(1, num_labels):
            if stats[i, cv2.CC_STAT_AREA] >= self.min_area:
                clean[labels == i] = 255

        return clean

    # -------------------------------------------------
    # TEMPORAL FILTER
    # -------------------------------------------------
    def _temporal_filter(self, frame_bgr, mask):
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.accumulator = mask.astype(np.float32) / 255.0
            return mask

        flow = cv2.calcOpticalFlowFarneback(
            self.prev_gray,
            gray,
            None,
            0.5, 3, 15, 3, 5, 1.2, 0
        )

        h, w = gray.shape
        flow_map_x = np.arange(w)[None, :] - flow[..., 0] * self.flow_scale
        flow_map_y = np.arange(h)[:, None] - flow[..., 1] * self.flow_scale

        warped = cv2.remap(
            self.accumulator,
            flow_map_x.astype(np.float32),
            flow_map_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )

        current = mask.astype(np.float32) / 255.0
        self.accumulator = (
            self.alpha * warped +
            (1.0 - self.alpha) * current
        )

        self.prev_gray = gray

        return (self.accumulator > self.threshold).astype(np.uint8) * 255
