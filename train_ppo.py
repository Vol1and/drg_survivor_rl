from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import DummyVecEnv

from env.drg_env import DRGEnv

import os


# -------------------------
# CONFIG
# -------------------------
MONITOR = 2
MAX_STEPS = 2000
TOTAL_TIMESTEPS = 100_000

MODEL_DIR = "models"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best")
LOG_DIR = "logs"

os.makedirs(CHECKPOINT_DIR, exist_ok=True)
os.makedirs(BEST_MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)


# -------------------------
# ENV
# -------------------------
env = DRGEnv(
    monitor=MONITOR,
    max_steps=MAX_STEPS,
)

env = DummyVecEnv([lambda: env])


# -------------------------
# CALLBACKS
# -------------------------

checkpoint_callback = CheckpointCallback(
    save_freq=5_000,                # каждые 10k шагов
    save_path=CHECKPOINT_DIR,
    name_prefix="drg_ppo"
)

eval_callback = EvalCallback(
    env,
    best_model_save_path=BEST_MODEL_DIR,
    log_path=LOG_DIR,
    eval_freq=20_000,
    deterministic=True,
    render=False,
)


# -------------------------
# MODEL
# -------------------------
#
# model = PPO.load(
#     "models/best/best_model",  # без .zip
#     env=env,
#     device="cuda"
# )
model = PPO(
    "MultiInputPolicy",
    env,
    verbose=1,
    n_steps=128,
    batch_size=64,
    gamma=0.995,
    ent_coef=0.02,
    learning_rate=3e-4,
    device="cuda",
    tensorboard_log=LOG_DIR,
)


print("\n=== TRAINING STARTED ===\n")


# -------------------------
# TRAIN
# -------------------------
model.learn(
    total_timesteps=TOTAL_TIMESTEPS,
    callback=[checkpoint_callback, eval_callback],
    progress_bar=True,
)

# -------------------------
# FINAL SAVE
# -------------------------
model.save(os.path.join(MODEL_DIR, "final_model"))

print("\n=== TRAINING FINISHED ===\n")