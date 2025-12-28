import cv2
import os
import numpy as np


class UIDetector:
    def __init__(self, threshold=0.75):
        base_dir = os.path.dirname(__file__)
        assets_dir = os.path.join(base_dir, "..", "..", "assets")

        # === Anchors (COLOR!) ===
        self.levelup_anchor = self._load_anchor(
            os.path.join(assets_dir, "levelup_anchor.png")
        )
        self.change_assort_active_anchor = self._load_anchor(
            os.path.join(assets_dir, "change_assort_active_anchor.png")
        )
        self.change_assort_gray_anchor = self._load_anchor(
            os.path.join(assets_dir, "change_assort_gray_anchor.png")
        )
        self.restart_anchor = self._load_anchor(
            os.path.join(assets_dir, "restart_button_anchor.png")
        )
        self.overclock_anchor = self._load_anchor(
            os.path.join(assets_dir, "overclock_anchor.png")
        )
        self.continue_button_anchors = [
            self._load_anchor(
                os.path.join(assets_dir, "continue_button_anchor.png")
            ),
            self._load_anchor(
                os.path.join(assets_dir, "continue_button_2_anchor.png")
            )
        ]

        # --- ROIs ---
        self.levelup_roi = (24, 31, 57, 101)
        self.restart_roi = (137, 149, 55, 77)
        self.change_assort_roi = (65, 73, 128, 148)
        self.continue_button_roi = (127, 139, 70, 90)
        self.overclock_roi = (117, 129, 70, 90)

        # --- thresholds ---
        self.levelup_threshold = 0.65
        self.aux_threshold = 0.55
        self.base_weight = 1.0
        self.aux_weight = 0.75
        self.synergy_bonus = 0.5
        self.button_threshold = 0.85

        self.death_check_threshold = 0.3

    # ==================================================
    # Utils
    # ==================================================

    def _load_anchor(self, path):
        img = cv2.imread(path)
        assert img is not None, f"Missing anchor: {path}"
        return img

    def _crop(self, frame, roi):
        y1, y2, x1, x2 = roi
        return frame[y1:y2, x1:x2]

    def _to_hsv(self, img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    def _match(self, roi_bgr, anchor_bgr):
        roi_gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
        anchor_gray = cv2.cvtColor(anchor_bgr, cv2.COLOR_BGR2GRAY)

        if roi_gray.shape[0] < anchor_gray.shape[0] or roi_gray.shape[1] < anchor_gray.shape[1]:
            return 0.0

        return float(
            cv2.matchTemplate(roi_gray, anchor_gray, cv2.TM_CCOEFF_NORMED).max()
        )

    # ==================================================
    # Main detection
    # ==================================================

    def detect(self, frame, hp):
        """
        frame: BGR (H, W, 3)
        """

        levelup_roi = self._crop(frame, self.levelup_roi)
        continue_button_roi = self._crop(frame, self.continue_button_roi)
        assort_roi = self._crop(frame, self.change_assort_roi)
        overclock_roi = self._crop(frame, self.overclock_roi)

        if hp < self.death_check_threshold:
            restart_roi = self._crop(frame, self.restart_roi)
            if self._match(restart_roi, self.restart_anchor) >= self.button_threshold:
                return "restart"

        if self._match(overclock_roi, self.overclock_anchor) >= self.button_threshold:
            return "overclock"

        if max([self._match(continue_button_roi, anchor) for anchor in self.continue_button_anchors]) >= self.button_threshold:
            return "continue"

        # --- LEVEL UP ---
        if self._detect_levelup(levelup_roi, assort_roi):
            return "levelup"

        return "gameplay"

    # ==================================================
    # Level-up logic
    # ==================================================

    def _detect_levelup(self, levelup_roi, assort_roi):

        def safe_match(roi, anchor):
            if roi is None or anchor is None:
                return 0.0
            if roi.shape[0] < anchor.shape[0] or roi.shape[1] < anchor.shape[1]:
                return 0.0
            return self._match(roi, anchor)

        base_score = safe_match(levelup_roi, self.levelup_anchor)

        aux_scores = [
            safe_match(assort_roi, self.change_assort_active_anchor),
            safe_match(assort_roi, self.change_assort_gray_anchor),
        ]

        best_aux = max(aux_scores)
        both_aux = all(s > self.aux_threshold for s in aux_scores)

        confidence = (
            base_score * self.base_weight +
            best_aux * self.aux_weight +
            (self.synergy_bonus if both_aux else 0.0)
        )
        return confidence >= self.levelup_threshold


  #print(
        #    f"roi={continue_button_roi.shape} "
        #    f"anchor={self.continue_button_anchor.shape}"
        #)