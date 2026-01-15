from mss import mss
import numpy as np
import cv2
import base64
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
        hsv = self.to_hsv(frame)
        h = hsv[:, :, 0]  # Hue channel
        return h[:, :, None]

    def process_minimap(
            self,
            minimap_b64: str,
            w: int,
            h: int,
            out_size: int = 32,
    ) -> np.ndarray:
        """
        Decode minimap from base64 and return grayscale image.

        Args:
            minimap_b64: base64-encoded RGB image
            w, h: original minimap size
            out_size: output size (default 29)

        Returns:
            np.ndarray (out_size, out_size), uint8 [0..255]
        """

        # --- base64 -> RGB ---
        rgb_bytes = base64.b64decode(minimap_b64)
        rgb = np.frombuffer(rgb_bytes, dtype=np.uint8).reshape((h, w, 3))

        # flip vertically (game -> screen coords)
        rgb = rgb[::-1]

        # resize to target resolution
        rgb = cv2.resize(
            rgb,
            (out_size, out_size),
            interpolation=cv2.INTER_AREA
        )

        # RGB -> GRAYSCALE
        gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)


        return gray.astype(np.float32) / 255.0
