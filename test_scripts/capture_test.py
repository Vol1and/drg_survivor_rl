import time
import cv2
import numpy as np
from mss import mss

sct = mss()
monitor = sct.monitors[2]

last_time = time.time()

while True:
    screenshot = sct.grab(monitor)

    frame = np.array(screenshot, dtype=np.uint8)
    frame = frame[:, :, :3].copy()

    fps = 1 / (time.time() - last_time)
    last_time = time.time()

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    # 🔥 ВОТ ЭТОГО НЕ ХВАТАЛО
    cv2.imshow("Agent View", frame)

    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cv2.destroyAllWindows()
