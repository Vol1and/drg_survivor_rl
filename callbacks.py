from stable_baselines3.common.callbacks import BaseCallback
import numpy as np

class SimpleStatsCallback(BaseCallback):
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []

    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])

        for info in infos:
            if "reward" in info:
                self.episode_rewards.append(info["reward"])

            if "confidence" in info:
                self.logger.record("env/confidence", info["confidence"])

            if "distance" in info:
                self.logger.record("env/distance", info["distance"])

            if "hp" in info:
                self.logger.record("env/hp", info["hp"])

            if "ui_state" in info:
                self.logger.record("env/ui_state", hash(info["ui_state"]) % 10)

        return True




from stable_baselines3.common.callbacks import BaseCallback
import numpy as np


class AsyncStatsCallback(BaseCallback):
    """
    Non-blocking logger for TensorBoard.
    """

    def __init__(self, log_freq=200, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_step = 0

    def _on_step(self) -> bool:
        # throttle logging
        if self.num_timesteps - self.last_log_step < self.log_freq:
            return True

        self.last_log_step = self.num_timesteps

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        rewards = []
        hp = []
        xp = []
        danger = []
        edge_ds = []
        levels = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            rewards.append(info.get("reward", 0.0))
            hp.append(info.get("hp", 0.0))
            xp.append(info.get("xp", 0.0))
            danger.append(info.get("danger", 0.0))
            edge_ds.append(info.get("edge_ds", 0.0))
            levels.append(info.get("levels", 0.0))

        # ----- LOGGING -----
        self.logger.record("env/reward", np.mean(rewards))
        self.logger.record("env/hp", np.mean(hp))
        self.logger.record("env/xp", np.mean(xp))
        self.logger.record("env/edge_danger", np.mean(danger))
        self.logger.record("env/edge_distance", np.mean(edge_ds))
        self.logger.record("env/level", np.mean(levels))

        # 🔥 THIS IS THE MISSING PIECE
        self.logger.dump(self.num_timesteps)

        return True



from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import csv
import os
from datetime import datetime


class AgentStatsCallback(BaseCallback):
    def __init__(
        self,
        log_freq=200,
        csv_path="logs/agent_stats.csv",
        verbose=0,
    ):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.csv_path = csv_path
        self.last_log_step = 0
        self.csv_file = None
        self.csv_writer = None

    # -------------------------------------------------
    def _on_training_start(self) -> None:
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)

        is_new = not os.path.exists(self.csv_path)

        self.csv_file = open(self.csv_path, "a", newline="")
        self.csv_writer = csv.writer(self.csv_file)

        if is_new:
            self.csv_writer.writerow([
                "timestamp",
                "timesteps",
                "reward",
                "hp",
                "xp",
                "danger",
                "edge_distance",
                "level",
            ])
            self.csv_file.flush()

    # -------------------------------------------------
    def _on_step(self) -> bool:
        if self.num_timesteps - self.last_log_step < self.log_freq:
            return True

        self.last_log_step = self.num_timesteps

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        rewards = []
        hp = []
        xp = []
        danger = []
        edge_ds = []
        levels = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            rewards.append(info.get("reward", 0.0))
            hp.append(info.get("hp", 0.0))
            xp.append(info.get("xp", 0.0))
            danger.append(info.get("danger", 0.0))
            edge_ds.append(info.get("edge_ds", 0.0))
            levels.append(info.get("levels", 0.0))

        # --- aggregate ---
        r = np.mean(rewards)
        h = np.mean(hp)
        x = np.mean(xp)
        d = np.mean(danger)
        e = np.mean(edge_ds)
        l = np.mean(levels)

        # --- TensorBoard ---
        self.logger.record("env/reward", r)
        self.logger.record("env/hp", h)
        self.logger.record("env/xp", x)
        self.logger.record("env/edge_danger", d)
        self.logger.record("env/edge_distance", e)
        self.logger.record("env/level", l)
        self.logger.dump(self.num_timesteps)

        # --- CSV ---
        self.csv_writer.writerow([
            datetime.utcnow().isoformat(),
            self.num_timesteps,
            round(r, 4),
            round(h, 4),
            round(x, 4),
            round(d, 4),
            round(e, 4),
            round(l, 2),
        ])
        self.csv_file.flush()

        return True

    # -------------------------------------------------
    def _on_training_end(self) -> None:
        if self.csv_file:
            self.csv_file.close()
