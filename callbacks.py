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

from stable_baselines3.common.callbacks import BaseCallback
from collections import deque
import numpy as np

class EpisodeStatsCallback(BaseCallback):
    """
    Collects rich per-episode statistics based on env `info`.

    Expected info keys:
      step, ui_state, reward, hp, gold, nitra, edge_ds, levels
    """

    def __init__(self, window=50, verbose=0):
        super().__init__(verbose)
        self.window = window

        # rolling buffers
        self.steps_runs = deque(maxlen=window)
        self.reward_runs = deque(maxlen=window)
        self.reward_per_step_runs = deque(maxlen=window)
        self.edge_mean_runs = deque(maxlen=window)
        self.damage_runs = deque(maxlen=window)
        self.death_runs = deque(maxlen=window)

        self.gold_runs = deque(maxlen=window)
        self.nitra_runs = deque(maxlen=window)
        self.levels_runs = deque(maxlen=window)

        self._reset_episode()

    # -------------------------------------------------
    def _reset_episode(self):
        self.steps = 0
        self.reward_sum = 0.0
        self.edge_sum = 0.0
        self.edge_count = 0
        self.damage_events = 0

        self.start_gold = None
        self.start_nitra = None
        self.start_levels = None

        self.last_hp = None
        self.last_gold = None
        self.last_nitra = None
        self.last_levels = None

    # -------------------------------------------------
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            ui_state = info.get("ui_state")

            # -------- Gameplay --------
            if ui_state == "Gameplay":
                reward = float(info.get("reward", 0.0))
                hp = float(info.get("hp", 1.0))
                edge_ds = float(info.get("edge_ds", 0.0))
                gold = float(info.get("gold", 0.0))
                nitra = float(info.get("nitra", 0.0))
                levels = int(info.get("levels", 0))

                if self.start_gold is None:
                    self.start_gold = gold
                    self.start_nitra = nitra
                    self.start_levels = levels

                if self.last_hp is not None and hp < self.last_hp:
                    self.damage_events += 1

                self.steps += 1
                self.reward_sum += reward
                self.edge_sum += edge_ds
                self.edge_count += 1

                self.last_hp = hp
                self.last_gold = gold
                self.last_nitra = nitra
                self.last_levels = levels

            # -------- Episode finished --------
            if dones[i]:
                if self.steps == 0:
                    self._reset_episode()
                    continue

                gold_total = max(0.0, self.last_gold - self.start_gold)
                nitra_total = max(0.0, self.last_nitra - self.start_nitra)
                levels_gained = max(0, self.last_levels - self.start_levels)

                edge_mean = self.edge_sum / max(1, self.edge_count)
                reward_per_step = self.reward_sum / max(1, self.steps)

                done_reason = 0 if ui_state == "Death" else 1  # 0=death, 1=timeout

                # ----- per episode -----
                self.logger.record("episode/steps", self.steps)
                self.logger.record("episode/reward_total", self.reward_sum)
                self.logger.record("episode/reward_per_step", reward_per_step)
                self.logger.record("episode/edge_distance_mean", edge_mean)
                self.logger.record("episode/damage_events", self.damage_events)
                self.logger.record("episode/hp_end", self.last_hp)
                self.logger.record("episode/gold_total", gold_total)
                self.logger.record("episode/nitra_total", nitra_total)
                self.logger.record("episode/levels_gained", levels_gained)
                self.logger.record("episode/done_reason", done_reason)

                # ----- rolling -----
                self.steps_runs.append(self.steps)
                self.reward_runs.append(self.reward_sum)
                self.reward_per_step_runs.append(reward_per_step)
                self.edge_mean_runs.append(edge_mean)
                self.damage_runs.append(self.damage_events)
                self.death_runs.append(1 if done_reason == 0 else 0)

                self.gold_runs.append(gold_total)
                self.nitra_runs.append(nitra_total)
                self.levels_runs.append(levels_gained)

                self.logger.record("rolling/steps_mean", np.mean(self.steps_runs))
                self.logger.record("rolling/reward_mean", np.mean(self.reward_runs))
                self.logger.record("rolling/reward_per_step_mean", np.mean(self.reward_per_step_runs))
                self.logger.record("rolling/edge_distance_mean", np.mean(self.edge_mean_runs))
                self.logger.record("rolling/damage_events_mean", np.mean(self.damage_runs))
                self.logger.record("rolling/death_ratio", np.mean(self.death_runs))
                self.logger.record("rolling/gold_mean", np.mean(self.gold_runs))
                self.logger.record("rolling/nitra_mean", np.mean(self.nitra_runs))
                self.logger.record("rolling/levels_mean", np.mean(self.levels_runs))

                self.logger.dump(self.num_timesteps)
                self._reset_episode()

        return True



