import cv2
import time
import numpy as np

from env.screen import Screen
from env.state.color_mask import ColorMaskExtractor


# ===============================
# CONFIG
# ===============================
MONITOR_INDEX = 2
SCALE = 3            # 🔍 масштаб окна
WINDOW_NAME = "Color Mask Debug"


# ===============================
# UTILS
# ===============================
def scale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


# ===============================
# MAIN
# ===============================
def main():
    screen = Screen(MONITOR_INDEX)
    extractor = ColorMaskExtractor()

    print("Focus game window...")
    time.sleep(2)

    while True:
        frame = screen.grab()

        # --- extract ---
        mask = extractor.extract(frame)

        # --- visual prep ---
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray_vis = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

        mask_vis = (mask * 255).astype(np.uint8)
        mask_vis = cv2.cvtColor(mask_vis, cv2.COLOR_GRAY2BGR)

        combined = np.hstack([gray_vis, mask_vis])

        combined = scale(combined, SCALE)

        cv2.imshow(WINDOW_NAME, combined)

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
