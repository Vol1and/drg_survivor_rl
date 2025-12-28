import time
import random
from env.drg_env import DRGEnv

MONITOR = {
    "left": 1920,
    "top": 0,
    "width": 1280,
    "height": 720,
}

env = DRGEnv(MONITOR, max_steps=20)

print("Switch to the game window. Starting in 3 seconds...")
time.sleep(3)

obs = env.reset()
total_reward = 0.0

while True:
    action = random.randint(0, 4)
    obs, reward, done, info = env.step(action)
    total_reward += reward

    print(
        f"step={info['step']} "
        f"action={action} "
        f"reward={reward:.2f} "
        f"total_reward={total_reward:.2f}"
    )

    if done:
        print("Episode finished")
        break
