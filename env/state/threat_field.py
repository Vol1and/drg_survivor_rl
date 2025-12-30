import numpy as np
from collections import deque


class ThreatFromMask:
    """
    Берёт threat_mask (H,W) float/uint, ищет ближайшую активную точку к центру.
    Если ближайшая точка дальше max_pixel_distance — угрозу считаем отсутствующей.
    """

    def __init__(self, frame_shape, max_pixel_distance: int = 15, history_len: int = 6):
        h, w = frame_shape[:2]
        self.h = h
        self.w = w
        self.cx = w // 2
        self.cy = h // 2

        self.max_pixel_distance = max_pixel_distance
        self.history = deque(maxlen=history_len)

    def update(self, mask: np.ndarray):
        # mask expected (H,W), values 0..1 or 0..255
        if mask is None:
            return self._empty()

        ys, xs = np.where(mask > 0.5)

        if len(xs) == 0:
            return self._empty()

        dxs = xs - self.cx
        dys = ys - self.cy
        dists = np.sqrt(dxs * dxs + dys * dys)

        idx = int(np.argmin(dists))
        min_dist = float(dists[idx])

        # фильтр: дальше 20px => угрозы нет
        if min_dist > self.max_pixel_distance:
            return self._empty()

        # нормализация по max_pixel_distance => dx/dy ∈ [-1..1], distance ∈ [0..1]
        dx = float(dxs[idx] / self.max_pixel_distance)
        dy = float(dys[idx] / self.max_pixel_distance)
        distance = float(min_dist / self.max_pixel_distance)

        # approaching: если distance уменьшается — положительное значение
        if len(self.history) > 0:
            prev_dist = self.history[-1]["distance"]
            approaching = float(np.clip(prev_dist - distance, -1.0, 1.0))
        else:
            approaching = 0.0

        result = {
            "distance": distance,
            "dx": dx,
            "dy": dy,
            "approaching": approaching,
            "confidence": 1.0,
        }

        self.history.append(result)
        return result

    def _empty(self):
        result = {
            "distance": 1.0,
            "dx": 0.0,
            "dy": 0.0,
            "approaching": 0.0,
            "confidence": 0.0,
        }
        self.history.append(result)
        return result
