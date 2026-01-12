import cv2
import numpy as np
from mss import mss
from env.config import SCREEN_SIZE


# -----------------------------
# Screen grab
# -----------------------------
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


# -----------------------------
# Quantization utils
# -----------------------------
def quantize_gray(gray, levels=16):
    step = 256 // levels
    return (gray // step) * step


def quantize_hue(h, bins=12):
    step = 180 // bins
    return (h // step) * step


# -----------------------------
# Variant 1 — BASELINE (Gray + Hue)
# -----------------------------
def baseline_gray_hue(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    return np.stack([gray, h, gray], axis=-1)


# -----------------------------
# Variant 2 — QUANTIZED BASELINE
# -----------------------------
def baseline_gray_hue_quantized(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = quantize_gray(gray, levels=16)

    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    h = hsv[:, :, 0]
    h = quantize_hue(h, bins=12)

    return np.stack([gray, h, gray], axis=-1)


# -----------------------------
# Utility
# -----------------------------
def to_bgr(img):
    if len(img.shape) == 2:
        return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    return img


def add_label(img, text):
    h, w = img.shape[:2]
    cv2.rectangle(img, (0, 0), (w, 28), (0, 0, 0), -1)
    cv2.putText(
        img,
        text,
        (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1,
        cv2.LINE_AA
    )
    return img


# -----------------------------
# MAIN LOOP
# -----------------------------
def main():
    screen = Screen()

    window_name = "Baseline vs Quantized Baseline"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window_name, SCREEN_SIZE * 2, SCREEN_SIZE)

    print("Press ESC to exit")

    while True:
        frame = screen.grab()

        v1 = to_bgr(baseline_gray_hue(frame))
        v2 = to_bgr(baseline_gray_hue_quantized(frame))

        v1 = add_label(v1, "Baseline (Gray + Hue)")
        v2 = add_label(v2, "Quantized (Gray + Hue)")

        grid = np.hstack([v1, v2])
        cv2.imshow(window_name, grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
