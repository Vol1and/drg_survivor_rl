import cv2
import numpy as np
import time

from env.screen import Screen

# -----------------------------
# CONFIG
# -----------------------------
MONITOR = 2
SCALE = 5   # 🔍 увеличь если мелко

# -----------------------------
# INIT
# -----------------------------
screen = Screen(MONITOR)

print("Press ESC to exit")

while True:
    frame = screen.grab()
    if frame is None:
        time.sleep(0.01)
        continue

    # -----------------------------
    # HSV
    # -----------------------------
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    h = hsv[:, :, 0]   # 0..179
    v = hsv[:, :, 2]   # 0..255

    # -----------------------------
    # Visualization
    # -----------------------------
    h_vis = cv2.applyColorMap(h, cv2.COLORMAP_HSV)
    v_vis = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)

    combined = np.hstack([h_vis, v_vis])

    # Labels
    cv2.putText(
        combined, "Hue (raw)",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )

    cv2.putText(
        combined, "Value (raw)",
        (combined.shape[1] // 2 + 20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1.2,
        (255, 255, 255),
        2
    )

    # -----------------------------
    # SCALE
    # -----------------------------
    combined = cv2.resize(
        combined,
        None,
        fx=SCALE,
        fy=SCALE,
        interpolation=cv2.INTER_NEAREST
    )

    cv2.imshow("Hue vs Value (RAW)", combined)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
