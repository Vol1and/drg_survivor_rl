import cv2
import numpy as np
from mss import mss
from env.config import SCREEN_SIZE
from env.screen import Screen





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
        frame = screen.grab_minimap()

        grid = np.hstack([frame])
        cv2.imshow(window_name, grid)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
