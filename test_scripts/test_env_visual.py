import time
import random
import cv2
from env.drg_env import DRGEnv

MONITOR = {
    "left": 1920,
    "top": 0,
    "width": 1280,
    "height": 720,
}

env = DRGEnv(MONITOR, max_steps=200)

print("Switch to the game window. Starting in 3 seconds...")
time.sleep(3)

obs = env.reset()

while True:
    action = random.randint(0, 4)
    obs, reward, done, info = env.step(action)

    # 👁 визуализируем то, ЧТО ВИДИТ АГЕНТ
    cv2.imshow("Agent Observation (84x84)", obs)

    # медленно, чтобы ты видел изменения
    if cv2.waitKey(60) & 0xFF == ord("q"):
        break

    if done:
        break

cv2.destroyAllWindows()
