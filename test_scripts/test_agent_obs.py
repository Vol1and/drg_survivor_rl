"""
test_agent_obs.py
Shows exactly what the agent receives in `obs["image"]` after process_for_obs().

Layout (3 panels side by side):
  [ Hue channel ]  [ Value channel ]  [ H+V composite ]

The composite maps H -> green channel, V -> red channel so both are visible at once.

Press ESC to exit.
"""

import cv2
import numpy as np
from env.screen import Screen
from env.config import SCREEN_SIZE

MONITOR = 2

screen = Screen(MONITOR)

LABEL_FONT  = cv2.FONT_HERSHEY_SIMPLEX
LABEL_SCALE = 0.9
LABEL_COLOR = (255, 255, 255)
LABEL_THICK = 2
LABEL_Y     = 36

window = "Agent Image Obs  |  H  |  V  |  H+V composite"
cv2.namedWindow(window, cv2.WINDOW_NORMAL)
cv2.resizeWindow(window, SCREEN_SIZE * 3, SCREEN_SIZE)

print("Press ESC to exit.")

while True:
    frame = screen.grab()                    # (SCREEN_SIZE, SCREEN_SIZE, 3) BGR

    obs = screen.process_for_obs(frame)      # (SCREEN_SIZE, SCREEN_SIZE, 2) uint8
    h = obs[:, :, 0]                         # Hue   0..179
    v = obs[:, :, 1]                         # Value 0..255

    # --- per-panel visualizations (all BGR) ---
    h_vis = cv2.applyColorMap(h, cv2.COLORMAP_HSV)   # hue shown with HSV colormap

    v_vis = cv2.cvtColor(v, cv2.COLOR_GRAY2BGR)       # value as grayscale

    # composite: H -> green, V -> red, blue=0
    composite = np.zeros((SCREEN_SIZE, SCREEN_SIZE, 3), dtype=np.uint8)
    composite[:, :, 1] = h                            # green  = hue
    composite[:, :, 2] = v                            # red    = value

    # --- labels ---
    for img, label in [
        (h_vis,     "H  (hue)"),
        (v_vis,     "V  (value)"),
        (composite, "H+V composite"),
    ]:
        cv2.putText(img, label, (16, LABEL_Y),
                    LABEL_FONT, LABEL_SCALE, LABEL_COLOR, LABEL_THICK)

    grid = np.hstack([h_vis, v_vis, composite])
    cv2.imshow(window, grid)

    if cv2.waitKey(1) & 0xFF == 27:
        break

cv2.destroyAllWindows()
