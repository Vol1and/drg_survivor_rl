from mss import mss
import numpy as np
import cv2
from env.config import SCREEN_SIZE

class Screen:
    def __init__(self, monitor_index=2):
        self.sct = mss()
        self.monitor = self.sct.monitors[monitor_index]

    def grab(self):
        img = self.sct.grab(self.monitor)
        frame = np.array(img, dtype=np.uint8)
        frame = frame[:, :, :3]  # BGRA → BGR
        frame = cv2.resize(frame, (SCREEN_SIZE, SCREEN_SIZE))
        return frame

    @staticmethod
    def to_gray(frame):
        """Grayscale представление"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def to_hsv(frame):
        """HSV представление"""
        return cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    def process_for_obs(self, frame):
        gray = self.to_gray(frame)  # (H, W)
        hsv = self.to_hsv(frame)
        h = hsv[:, :, 0]  # Hue channel

        return np.stack([gray, h], axis=-1)