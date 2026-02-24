import os
from multiprocessing import freeze_support, set_start_method

from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from sb3_contrib import RecurrentPPO
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback

from env.drg_env import DRGEnv
from callbacks import EpisodeStatsCallback, GameRestartCallback

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
VEC_NORM_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")



episode_stats_cb = EpisodeStatsCallback(window=50)
game_restart_cb = GameRestartCallback(
    restart_every=1,        # перезапускать каждые N эпизодов
)

class VecNormalizeSaveCallback(BaseCallback):
    """Saves VecNormalize running stats alongside each checkpoint."""

    def __init__(self, venv: VecNormalize, save_path: str, save_freq: int, verbose=0):
        super().__init__(verbose)
        self.venv = venv
        self.save_path = save_path
        self.save_freq = save_freq

    def _on_step(self) -> bool:
        if self.n_calls % self.save_freq == 0:
            self.venv.save(self.save_path)
        return True


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
    raw_env = DummyVecEnv([make_env])

    # ----------------------------
    # CALLBACKS
    # ----------------------------
    checkpoint_callback = CheckpointCallback(
        save_freq=10_000,
        save_path=CHECKPOINT_DIR,
        name_prefix="drg_ppo",
    )

    TRAINIG = True

    print("\n=== TRAINING STARTED ===\n")
    if TRAINIG:
        train_env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
        vec_norm_save_cb = VecNormalizeSaveCallback(train_env, VEC_NORM_PATH, save_freq=10_000)

        model = RecurrentPPO(
            policy="MultiInputLstmPolicy",
            env=train_env,
            learning_rate=lambda progress: 3e-4 * progress,
            n_steps=1024,
            batch_size= 128,
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.015,
            max_grad_norm=0.5,
            verbose=1,
            tensorboard_log="logs",
        )
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, vec_norm_save_cb, episode_stats_cb, game_restart_cb],
            progress_bar=True,
        )
    else:
        print('Стартуем с чекпоинта')
        if os.path.exists(VEC_NORM_PATH):
            train_env = VecNormalize.load(VEC_NORM_PATH, raw_env)
            print(f"[VecNormalize] Loaded stats from {VEC_NORM_PATH}")
        else:
            train_env = VecNormalize(raw_env, norm_obs=False, norm_reward=True, clip_reward=10.0)
            print("[VecNormalize] No saved stats found, starting fresh normalizer")

        vec_norm_save_cb = VecNormalizeSaveCallback(train_env, VEC_NORM_PATH, save_freq=10_000)

        model = RecurrentPPO.load(
            "models/checkpoints/drg_ppo_50000_steps",
            env=train_env,
            device="cuda",
            tensorboard_log="logs",
        )

        model.learn(
            total_timesteps=400_000,
            callback=[checkpoint_callback, vec_norm_save_cb, episode_stats_cb, game_restart_cb],
            reset_num_timesteps=False,  # 🔥 ВАЖНО
            #tb_log_name="continue_280k",  # 🔥 НОВЫЙ RUN
            progress_bar=True,
        )

    train_env.save(VEC_NORM_PATH)
    model.save(os.path.join(MODEL_DIR, "final_model"))


print("\n=== TRAINING FINISHED ===\n")


if __name__ == "__main__":
    freeze_support()
    set_start_method("spawn", force=True)
    main()
