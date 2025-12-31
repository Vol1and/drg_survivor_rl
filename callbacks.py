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




class AsyncStatsCallback(BaseCallback):
    """
    Non-blocking logger for TensorBoard.
    Reads data from `info` dict.
    """

    def __init__(self, log_freq=200, verbose=0):
        super().__init__(verbose)
        self.log_freq = log_freq
        self.last_log_step = 0

    def _on_step(self) -> bool:
        # Throttle logging
        if self.num_timesteps - self.last_log_step < self.log_freq:
            return True

        self.last_log_step = self.num_timesteps

        infos = self.locals.get("infos", [])
        if not infos:
            return True

        # buffers
        rewards = []
        hp = []
        xp = []
        danger = []
        edge_ds = []
        levels = []

        for info in infos:
            if not isinstance(info, dict):
                continue

            if "reward" in info:
                rewards.append(info["reward"])

            if "hp" in info:
                hp.append(info["hp"])

            if "xp" in info:
                xp.append(info["xp"])

            if "danger" in info:
                danger.append(info["danger"])

            if "edge_ds" in info:
                edge_ds.append(info["edge_ds"])

            if "levels" in info:
                levels.append(info["levels"])

        # ---- TensorBoard logging ----
        if rewards:
            self.logger.record("env/reward", np.mean(rewards))

        if hp:
            self.logger.record("env/hp", np.mean(hp))

        if xp:
            self.logger.record("env/xp", np.mean(xp))

        if danger:
            self.logger.record("env/edge_danger", np.mean(danger))

        if edge_ds:
            self.logger.record("env/edge_distance", np.mean(edge_ds))

        if levels:
            self.logger.record("env/level", np.mean(levels))

        return True
