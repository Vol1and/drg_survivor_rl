import time
import random

from env.drg_env import DRGEnv


MONITOR = {
    "left": 1920,   # монитор с игрой
    "top": 0,
    "width": 1280,
    "height": 720,
}


def main():
    env = DRGEnv(MONITOR, max_steps=1000)

    print("Focus the game window. Starting in 3 seconds...")
    time.sleep(3)

    obs, info = env.reset()
    print("Episode started")

    while True:
        # случайное действие
        action = random.randint(0, env.action_space.n - 1)

        obs, reward, done, truncated, info = env.step(action)

        print(
            f"step={info['step']:4d} | "
            f"same_frame={info['same_frame_count']:2d} | "
            f"death={info['death_detected']} | "
            f"reward={reward:6.2f}"
        )

        if done:
            print("\n=== EPISODE ENDED ===")
            print("death_detected:", info["death_detected"])
            print("total_steps:", info["step"])
            break


if __name__ == "__main__":
    main()
