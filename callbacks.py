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

from collections import deque

from collections import deque
import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


class EpisodeStatsCallback(BaseCallback):
    """
    Collects rich per-episode statistics based on structured env `info`,
    including minimap diagnostics.
    """

    def __init__(self, window=50, verbose=0):
        super().__init__(verbose)
        self.window = window

        # rolling buffers
        self.steps_runs = deque(maxlen=window)
        self.reward_runs = deque(maxlen=window)
        self.reward_per_step_runs = deque(maxlen=window)

        self.death_runs = deque(maxlen=window)
        self.boss_kill_runs = deque(maxlen=window)
        self.drop_pod_runs = deque(maxlen=window)

        # minimap rolling
        self.minimap_mean_runs = deque(maxlen=window)
        self.minimap_std_runs = deque(maxlen=window)
        self.minimap_nonzero_runs = deque(maxlen=window)

        self._reset_episode()

    # -------------------------------------------------
    def _reset_episode(self):
        # --- core ---
        self.steps = 0
        self.reward_sum = 0.0

        # --- reward decomposition ---
        self.reward_damage = 0.0
        self.reward_resource = 0.0
        self.reward_level = 0.0
        self.reward_boss_damage = 0.0
        self.reward_edge = 0.0
        self.reward_position = 0.0

        # --- hp / risk ---
        self.damage_events = 0
        self.low_hp_steps = 0

        # --- spatial ---
        self.edge_stuck_steps = 0

        # --- threat ---
        self.threat_high_steps = 0

        # --- objectives ---
        self.boss_seen = False
        self.boss_killed = False
        self.post_boss_steps = 0
        self.reached_drop_pod = False

        # --- minimap ---
        self.minimap_mean_sum = 0.0
        self.minimap_std_sum = 0.0
        self.minimap_nonzero_steps = 0
        self.minimap_empty_steps = 0

        self.last_hp = None

    # -------------------------------------------------
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            if info.get("state/ui") == "Gameplay":
                self.steps += 1

                # --- reward ---
                r = float(info.get("reward/total", 0.0))
                self.reward_sum += r

                self.reward_damage += float(info.get("reward/damage", 0.0))
                self.reward_resource += float(info.get("reward/resource", 0.0))
                self.reward_level += float(info.get("reward/level", 0.0))
                self.reward_boss_damage += float(info.get("reward/boss_damage", 0.0))
                self.reward_edge += float(info.get("reward/edge", 0.0))
                self.reward_position += float(info.get("reward/position", 0.0))

                # --- hp ---
                hp = float(info.get("hp/value", 1.0))
                if info.get("hp/delta", 0.0) < 0:
                    self.damage_events += 1
                if info.get("hp/low", 0.0) > 0:
                    self.low_hp_steps += 1
                self.last_hp = hp

                # --- edge / threat ---
                if info.get("edge/stuck", 0.0) > 0:
                    self.edge_stuck_steps += 1
                if info.get("threat/high", 0.0) > 0:
                    self.threat_high_steps += 1

                # --- objectives ---
                if info.get("state/is_boss", 0.0) > 0:
                    self.boss_seen = True
                if info.get("objective/boss_dead", 0.0) > 0:
                    self.boss_killed = True
                if info.get("state/is_post_boss", 0.0) > 0:
                    self.post_boss_steps += 1
                if info.get("objective/reaching_pod", 0.0) > 0:
                    self.reached_drop_pod = True

                # --- minimap ---
                mm_mean = float(info.get("minimap/mean", 0.0))
                mm_std = float(info.get("minimap/std", 0.0))
                mm_nonzero = float(info.get("minimap/nonzero_ratio", 0.0))

                self.minimap_mean_sum += mm_mean
                self.minimap_std_sum += mm_std

                if mm_nonzero < 0.05:
                    self.minimap_empty_steps += 1
                else:
                    self.minimap_nonzero_steps += 1

            # -------- Episode finished --------
            if dones[i]:
                if self.steps == 0:
                    self._reset_episode()
                    continue

                reward_per_step = self.reward_sum / max(1, self.steps)
                done_by_death = float(info.get("episode/ended_by_death", 0.0) > 0)

                mm_mean_ep = self.minimap_mean_sum / max(1, self.steps)
                mm_std_ep = self.minimap_std_sum / max(1, self.steps)
                mm_nonzero_ratio_ep = self.minimap_nonzero_steps / max(1, self.steps)

                # ===== episode =====
                self.logger.record("episode/steps", self.steps)
                self.logger.record("episode/reward_total", self.reward_sum)
                self.logger.record("episode/reward_per_step", reward_per_step)

                self.logger.record("episode/minimap_mean", mm_mean_ep)
                self.logger.record("episode/minimap_std", mm_std_ep)
                self.logger.record("episode/minimap_nonzero_ratio", mm_nonzero_ratio_ep)
                self.logger.record("episode/minimap_empty_steps", self.minimap_empty_steps)

                self.logger.record("episode/boss_seen", float(self.boss_seen))
                self.logger.record("episode/boss_killed", float(self.boss_killed))
                self.logger.record("episode/reached_drop_pod", float(self.reached_drop_pod))
                self.logger.record("episode/done_by_death", done_by_death)
                self.logger.record("episode/hp_end", self.last_hp)

                # ===== rolling =====
                self.steps_runs.append(self.steps)
                self.reward_runs.append(self.reward_sum)
                self.reward_per_step_runs.append(reward_per_step)
                self.death_runs.append(done_by_death)
                self.boss_kill_runs.append(float(self.boss_killed))
                self.drop_pod_runs.append(float(self.reached_drop_pod))

                self.minimap_mean_runs.append(mm_mean_ep)
                self.minimap_std_runs.append(mm_std_ep)
                self.minimap_nonzero_runs.append(mm_nonzero_ratio_ep)

                self.logger.record("rolling/steps_mean", np.mean(self.steps_runs))
                self.logger.record("rolling/reward_mean", np.mean(self.reward_runs))
                self.logger.record("rolling/reward_per_step_mean", np.mean(self.reward_per_step_runs))
                self.logger.record("rolling/death_ratio", np.mean(self.death_runs))
                self.logger.record("rolling/boss_kill_ratio", np.mean(self.boss_kill_runs))
                self.logger.record("rolling/drop_pod_success_ratio", np.mean(self.drop_pod_runs))

                self.logger.record("rolling/minimap_mean", np.mean(self.minimap_mean_runs))
                self.logger.record("rolling/minimap_std", np.mean(self.minimap_std_runs))
                self.logger.record("rolling/minimap_nonzero_ratio", np.mean(self.minimap_nonzero_runs))

                self.logger.dump(self.num_timesteps)
                self._reset_episode()

        return True





