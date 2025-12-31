import cv2
import numpy as np


class HPTracker:
    def __init__(self):
        self.prev_hp = 1.0

    def reset(self):
        self.prev_hp = 1.0

    def get_current_hp(self):
        return self.prev_hp

    def update(self, frame):
        hp = self._extract_hp(frame)
        hp = max(0.0, min(1.0, hp))
        self.prev_hp = hp
        return hp

    def get_hp_delta(self, new_hp):
        hp_now = new_hp
        hp_now = max(0.0, min(1.0, hp_now))

        delta = hp_now - self.prev_hp
        self.prev_hp = hp_now
        return delta
