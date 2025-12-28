import time
from env.drg_env import DRGEnv
from agent.model import ActorCritic
from agent.policy import select_action
from agent.buffer import RolloutBuffer

MONITOR = {
    "left": 1920,
    "top": 0,
    "width": 1280,
    "height": 720,
}

env = DRGEnv(MONITOR, max_steps=200)
model = ActorCritic(num_actions=env.action_space.n)
buffer = RolloutBuffer()

print("Switch to the game window. Starting in 3 seconds...")
time.sleep(3)

obs, info = env.reset()

while True:
    action, value, log_prob = select_action(model, obs)
    next_obs, reward, terminated, truncated, info = env.step(action)

    buffer.add(
        obs=obs,
        action=action,
        reward=reward,
        value=value,
        log_prob=log_prob,
        done=terminated or truncated
    )

    obs = next_obs

    if terminated or truncated:
        break

print("Rollout collected:", len(buffer.obs))
