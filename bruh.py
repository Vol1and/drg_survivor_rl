from env.screen import Screen
import cv2

screen = Screen(monitor_index=2)
obs = screen.grab()

cv2.imshow("DEBUG", cv2.resize(obs, (420, 420)))
cv2.waitKey(0)
