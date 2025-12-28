import cv2
import numpy as np


class HPTracker:
    def __init__(self):
        self.prev_hp = 1.0

        # ROI: (y1, y2, x1, x2)
        self.hp_roi = (133, 134, 57, 103)

    def reset(self):
        self.prev_hp = 1.0

    def get_current_hp(self):
        return self.prev_hp

    def update(self, frame):
        hp = self._extract_hp(frame)
        hp = max(0.0, min(1.0, hp))
        self.prev_hp = hp
        return hp

    def get_hp_delta(self, frame):
        hp_now = self._extract_hp(frame)
        hp_now = max(0.0, min(1.0, hp_now))

        delta = hp_now - self.prev_hp
        self.prev_hp = hp_now
        return delta

    def _extract_hp(self, frame):
        y1, y2, x1, x2 = self.hp_roi
        roi = frame[y1:y2, x1:x2]

        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)

        # --- Красный цвет ---
        red_mask = (
                ((h <= 10) | (h >= 170)) &
                (s > 70) &
                (v > 50)
        )

        # --- Белый цвет (цифры) ---
        white_mask = (s < 40) & (v > 200)

        # --- Убираем цифры из красного ---
        hp_mask = red_mask & (~white_mask)

        hp_ratio = np.count_nonzero(hp_mask) / hp_mask.size
        return float(np.clip(hp_ratio, 0.0, 1.0))

    # -------------------------
    # DEBUG VISUALIZATION
    # -------------------------

    def debug_draw(self, frame):
        """
        Рисует ROI и маску HP
        """
        vis = frame.copy()

        y1, y2, x1, x2 = self.hp_roi

        # рамка HP
        cv2.rectangle(vis, (x1, y1), (x2, y2), (0, 255, 0), 1)

        roi = frame[y1:y2, x1:x2]
        hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)

        mask1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
        mask2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
        mask = cv2.bitwise_or(mask1, mask2)

        # показать маску рядом
        mask_colored = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        mask_colored[:, :, 1:] = 0  # красный

        # масштабируем для наглядности
        mask_big = cv2.resize(mask_colored, (150, 30), interpolation=cv2.INTER_NEAREST)

        vis[0:30, 0:150] = mask_big

        hp = self.get_current_hp()
        cv2.putText(
            vis,
            f"HP: {hp:.3f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (0, 255, 0),
            2
        )

        return vis
