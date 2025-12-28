import time
import random
from env.drg_env import DRGEnv

MONITOR = {
    "left": 1920,
    "top": 0,
    "width": 1280,
    "height": 720,
}

env = DRGEnv(MONITOR, max_steps=1000)

print("Focus the game window. Starting in 3 seconds...")
time.sleep(3)

for episode in range(3):
    print(f"\n=== EPISODE {episode} ===")
    obs, _ = env.reset()

    while True:
        action = random.randint(0, env.action_space.n - 1)
        obs, reward, done, _, info = env.step(action)

        if done:
            print("Episode ended, waiting for auto-reset...")
            break
