import time
import cv2
from env.screen import Screen

MONITOR = {"left": 1920, "top": 0, "width": 1280, "height": 720}
screen = Screen(MONITOR)

print("Focus game. In 3s will capture. Put нужный экран (level-up или death).")
time.sleep(3)

obs = screen.grab()          # (84,84) grayscale
cv2.imwrite("template.png", obs)
print("Saved template.png")
