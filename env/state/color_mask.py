import cv2
import numpy as np


class ColorMaskExtractor:
    def __init__(self):
        self.temporal_mask = None

        self.temporal_decay = 0.85
        self.activation_threshold = 1.4

        self.color_ranges = [
            ((0, 30, 25), (35, 255, 200)),   # красные
            ((30, 35, 35), (95, 255, 210)),  # зелёные
            ((80, 70, 60), (125, 200, 180)), # сине-зелёные
        ]

        self.center_mask = None

    # --------------------------------------------------

    def _build_center_mask(self, h, w):
        """
        Эллиптическая маска персонажа (вытянутая по вертикали)
        """
        y, x = np.ogrid[:h, :w]
        cy, cx = h // 2, w // 2

        # вытянутость персонажа
        stretch_x = 0.7
        stretch_y = 1.2   # ← делает маску выше

        dist = np.sqrt(
            ((x - cx) / stretch_x) ** 2 +
            ((y - cy) / stretch_y) ** 2
        )

        max_dist = np.sqrt((cx / stretch_x) ** 2 + (cy / stretch_y) ** 2)
        norm = dist / max_dist

        # размеры зоны персонажа
        inner = 0.03   # полностью вырезаем
        outer = 0.05   # плавный край

        mask = np.ones_like(norm, dtype=np.float32)

        mask[norm < inner] = 0.0

        ring = (norm >= inner) & (norm < outer)
        t = (norm[ring] - inner) / (outer - inner)
        mask[ring] = t

        return mask

    # --------------------------------------------------

    def extract(self, frame: np.ndarray) -> np.ndarray:
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        v = hsv[:, :, 2].astype(np.float32)

        # --- 1. Убираем глобальное освещение ---
        illumination = cv2.GaussianBlur(v, (0, 0), sigmaX=25, sigmaY=25)
        v_norm = cv2.subtract(v, illumination)
        v_norm = cv2.normalize(v_norm, None, 0, 255, cv2.NORM_MINMAX)

        # --- 2. Цветовая маска ---
        color_mask = np.zeros_like(v, dtype=np.uint8)
        for lo, hi in self.color_ranges:
            color_mask |= cv2.inRange(hsv, lo, hi)

        # --- 3. Яркие области ---
        brightness_mask = v_norm > 30
        raw_mask = color_mask & brightness_mask

        # --- 4. Temporal smoothing ---
        if self.temporal_mask is None:
            self.temporal_mask = np.zeros_like(v, dtype=np.float32)
            self.center_mask = self._build_center_mask(*v.shape)

        self.temporal_mask *= self.temporal_decay
        self.temporal_mask += raw_mask.astype(np.float32)

        stable_mask = self.temporal_mask > self.activation_threshold

        # --- 5. Вырезаем персонажа ---
        stable_mask = stable_mask * self.center_mask

        # --- 6. Cleanup ---
        stable_mask = cv2.medianBlur(stable_mask.astype(np.uint8), 3)

        return stable_mask.astype(np.float32)
