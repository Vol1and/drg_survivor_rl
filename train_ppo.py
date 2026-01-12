import os
from multiprocessing import freeze_support, set_start_method

from stable_baselines3.common.vec_env import DummyVecEnv
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback

from env.drg_env import DRGEnv
from callbacks import SimpleStatsCallback, AsyncStatsCallback, AgentStatsCallback, EpisodeStatsCallback

# =========================
# CONFIG
# =========================
MONITOR = 2
MAX_STEPS = 50_000
TOTAL_TIMESTEPS = 500_000

MODEL_DIR = "models"
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints")
BEST_MODEL_DIR = os.path.join(MODEL_DIR, "best")
LOG_DIR = "logs"

stats_callback = SimpleStatsCallback()
stats_2_callback = AsyncStatsCallback(log_freq=100)

episode_stats_cb = EpisodeStatsCallback(window=100)

def make_env():
    return DRGEnv(
        monitor=MONITOR,
        max_steps=MAX_STEPS,
    )


def main():
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)
    os.makedirs(BEST_MODEL_DIR, exist_ok=True)
    os.makedirs(LOG_DIR, exist_ok=True)

    # ----------------------------
    # ENVIRONMENTS
    # ----------------------------
    train_env = DummyVecEnv([make_env])
    eval_env = DummyVecEnv([make_env])

    # ----------------------------
    # CALLBACKS
    # ----------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="drg_ppo",
    )

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=BEST_MODEL_DIR,
        log_path=LOG_DIR,
        deterministic=True,
        render=False,
    )


    TRAINIG = True

    print("\n=== TRAINING STARTED ===\n")
    if TRAINIG:
        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=train_env,
            learning_rate=3e-4,
            n_steps=384,
            batch_size= 128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.005,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="logs",
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, episode_stats_cb],
            progress_bar=True,
        )
    else:
        print('Стартуем с чекпоинта')
        model = RecurrentPPO.load(
            "models/checkpoints/drg_ppo_640000_steps",
            env=train_env,
            device="cuda",
            tensorboard_log="logs",
        )

        model.learn(
            total_timesteps=300_000,
            callback=[checkpoint_callback, episode_stats_cb],
            reset_num_timesteps=False,  # 🔥 ВАЖНО
            #tb_log_name="continue_280k",  # 🔥 НОВЫЙ RUN
            progress_bar=True,
        )

    model.save(os.path.join(MODEL_DIR, "final_model"))

print("\n=== TRAINING FINISHED ===\n")


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
