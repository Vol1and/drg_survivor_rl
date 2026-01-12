import cv2
import numpy as np


class BossHPTracker:
    def __init__(self):
        self.prev_boss_hp = 1.0

    def reset(self):
        self.prev_boss_hp = 1.0

    def get_current_boss_hp(self):
        return self.prev_boss_hp

    def update(self, frame):
        boss_hp = self._extract_boss_hp(frame)
        boss_hp = max(0.0, min(1.0, boss_hp))
        self.prev_boss_hp = boss_hp
        return boss_hp

    def get_delta(self, new_boss_hp):
        boss_hp_now = new_boss_hp
        boss_hp_now = max(0.0, min(1.0, boss_hp_now))

        delta = boss_hp_now - self.prev_boss_hp
        self.prev_boss_hp = boss_hp_now
        return delta
