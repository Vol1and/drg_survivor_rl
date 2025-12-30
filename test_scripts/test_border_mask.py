import cv2
import time
import numpy as np

from env.screen import Screen
from env.state.level_border_mask import LevelBorderDetector


# ===============================
# CONFIG
# ===============================
MONITOR_INDEX = 2
SCALE = 3
WINDOW_NAME = "Border Detector (Temporal)"


def scale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def main():
    screen = Screen(MONITOR_INDEX)

    detector = LevelBorderDetector(
        base_hsv=(145, 80, 30),
        h_tol=20,
        s_tol=50,
        v_tol=80,
        neighborhood=5,
        min_density=8,
        alpha=0.65,
        threshold=0.6,
    )

    print("Focus game window...")
    time.sleep(2)

    while True:
        frame = screen.grab()

        mask = detector.extract(frame)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        mask_vis = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

        overlay = frame.copy()
        overlay[mask > 0] = [0, 0, 255]

        combined = np.hstack([
            gray_vis,
            mask_vis,
            overlay,
        ])

        combined = scale(combined, SCALE)

        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
