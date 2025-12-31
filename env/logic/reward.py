import numpy as np


class RewardFunction:
    def __init__(self):
        # --- base rewards ---
        self.hp_weight = 0.15
        self.hp_delta_weight = 1.0

        self.xp_weight = 0.4

        # --- level up ---
        self.levelup_reward = 1.0
        self.levelups = 0

        # --- edge penalty ---
        self.edge_penalty_weight = 0.3     # сила штрафа
        self.edge_dist_threshold = 0.3     # ближе = опасно
        self.edge_stuck_frames = 50          # сколько кадров считать "залип"
        self.edge_stuck_counter = 0

        # --- stabilization ---
        self.min_reward = -3.0
        self.max_reward = 3.0

    def reset(self):
        self.levelups = 0
        self.edge_stuck_counter = 0

    def on_levelup(self):
        self.levelups += 1
        return self.levelup_reward

    def compute(
            self,
            *,
            death: bool,
            hp_fraction: float,
            hp_delta: float,
            xp_delta: float = 0.0,
            edge_features=None,
    ):
        reward = 0.0

        # -------------------------------------------------
        # 1. HP reward (главный сигнал выживания)
        # -------------------------------------------------
        reward += hp_fraction * self.hp_weight
        reward += hp_delta * self.hp_delta_weight

        # -------------------------------------------------
        # 2. XP reward (прогресс)
        # -------------------------------------------------
        reward += max(0.0, xp_delta) * self.xp_weight * hp_fraction

        # -------------------------------------------------
        # 3. EDGE AWARENESS (мягкий штраф)
        # -------------------------------------------------
        if edge_features is not None:
            _, edge_dist, _, _, danger = edge_features

            # считаем "залипание" только если реально опасно
            if danger > 0.4 and edge_dist < self.edge_dist_threshold:
                self.edge_stuck_counter += 1
            else:
                self.edge_stuck_counter = max(0, self.edge_stuck_counter - 1)

            # нормализованный коэффициент залипания
            stuck_ratio = min(
                self.edge_stuck_counter / self.edge_stuck_frames,
                1.0
            )

            # штраф растёт плавно
            edge_penalty = (
                    stuck_ratio *
                    danger *
                    self.edge_penalty_weight
            )

            reward -= edge_penalty

        # -------------------------------------------------
        # 4. Death penalty
        # -------------------------------------------------
        if death:
            reward -= 1.0

        # -------------------------------------------------
        # 5. Clamp (без резких скачков)
        # -------------------------------------------------
        reward = np.clip(reward, self.min_reward, self.max_reward)

        return float(reward)

