import time
import random
from env.drg_env import DRGEnv

MONITOR = {
    "left": 1920,
    "top": 0,
    "width": 1280,
    "height": 720,
}

env = DRGEnv(MONITOR, max_steps=50)

print("Action space:", env.action_space)
print("Observation space:", env.observation_space)

print("Switch to the game window. Starting in 3 seconds...")
time.sleep(3)

obs, info = env.reset()
print("Initial obs shape:", obs.shape)

done = False

while not done:
    action = env.action_space.sample()
    obs, reward, done, truncated, info = env.step(action)

    print(
        f"step={info['step']} "
        f"action={action} "
        f"reward={reward:.2f}"
    )

print("Episode finished")
