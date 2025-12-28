import time
import random
from env.drg_env import DRGEnv

MONITOR = {
    "left": 1920,   # твой монитор №2
    "top": 0,
    "width": 1280,
    "height": 720,
}

env = DRGEnv(MONITOR)

print("Switch to the game window. Starting in 3 seconds...")
time.sleep(3)

obs = env.reset()
print("Initial observation shape:", obs.shape)

for i in range(20):
    action = random.randint(0, 4)
    obs = env.step(action)
    print(f"Step {i}, action={action}, obs shape={obs.shape}")

print("Done")
