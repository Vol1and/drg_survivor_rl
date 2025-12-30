import cv2
import numpy as np


class XPTracker:
    def __init__(self):
        self.prev_xp = 0.0

        # ⚠️ подстрой под свой UI
        # (y1, y2, x1, x2)
        self.xp_roi = (148, 149, 4, 158)

    def reset(self):
        self.prev_xp = 0.0

    def get_current_xp(self):
        return self.prev_xp

    def get_xp_delta(self, frame):
        xp_now = self._extract_xp(frame)
        xp_now = np.clip(xp_now, 0.0, 1.0)

        delta = xp_now - self.prev_xp
        self.prev_xp = xp_now
        return delta

    def _extract_xp(self, frame):
        y1, y2, x1, x2 = self.xp_roi
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- XP цвет (голубой / бирюзовый) ---
        xp_mask = (
            (h >= 90) & (h <= 130) &
            (s > 60) &
            (v > 80)
        )

        xp_ratio = np.count_nonzero(xp_mask) / xp_mask.size
        return float(np.clip(xp_ratio, 0.0, 1.0))

    # -------------------------
    # DEBUG
    # -------------------------
    def debug_draw(self, frame):
        vis = frame.copy()

        y1, y2, x1, x2 = self.xp_roi
        cv2.rectangle(vis, (x1, y1), (x2, y2), (255, 255, 0), 1)

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask = (
                (hsv[:, :, 0] >= 90) &
                (hsv[:, :, 0] <= 130) &
                (hsv[:, :, 1] > 60) &
                (hsv[:, :, 2] > 80)
        )

        mask = (mask.astype(np.uint8) * 255)
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 1] = 255  # cyan

        # === SAFE DRAWING ===
        h, w = vis.shape[:2]
        mh, mw = mask_colored.shape[:2]

        draw_w = min(mw, w)
        draw_h = min(mh, h)

        vis[0:draw_h, 0:draw_w] = mask_colored[:draw_h, :draw_w]

        xp = self.get_current_xp()
        cv2.putText(
            vis,
            f"XP: {xp:.3f}",
            (10, draw_h + 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 0),
            2
        )

        return vis

