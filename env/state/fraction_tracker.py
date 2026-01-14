import cv2
import numpy as np


class FractionTracker:
    def __init__(self):
        self.prev_val = 1.0
        self.last_delta = 0.0

    def reset(self):
        self.prev_val = 1.0
        self.last_delta = 0.0

    def get_current(self):
        return self.prev_val

    def update(self, frame):
        val = self._extract_val(frame)
        val = max(0.0, min(1.0, val))
        self.prev_val = val
        return val

    def get_delta(self, new_val):
        val_now = new_val
        val_now = max(0.0, min(1.0, val_now))

        self.last_delta = val_now - self.prev_val
        self.prev_val = val_now
        return self.last_delta
