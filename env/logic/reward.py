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

        # --- edge behavior ---
        self.edge_penalty_weight = 0.3
        self.edge_dist_threshold = 0.3
        self.edge_stuck_frames = 40
        self.edge_stuck_counter = 0

        # --- panic mode ---
        self.panic_hp_threshold = 0.35
        self.panic_penalty = 0.6
        self.panic_escape_bonus = 0.4

        # --- positional penalty ---
        self.edge_zone_start = 0.25
        self.edge_zone_end = 0.75
        self.position_penalty_weight = 0.5

        self.edge_zone_start = 0.35  # начало опасной зоны
        self.edge_zone_end = 0.55  # конец безопасной зоны
        self.position_penalty_weight = 0.6
        # --- clamp ---
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
        position=None,  # (x, z)
    ):
        reward = 0.0

        # ------------------------------------------------
        # 1. SURVIVAL BASE
        # ------------------------------------------------
        reward += 0.01

        # ------------------------------------------------
        # 2. DAMAGE (critical)
        # ------------------------------------------------
        if hp_delta < 0:
            reward += hp_delta * 4.0

        # ------------------------------------------------
        # 3. XP (secondary)
        # ------------------------------------------------
        if xp_delta > 0:
            reward += xp_delta * 0.25

        # ------------------------------------------------
        # 4. EDGE LOGIC
        # ------------------------------------------------
        if edge_features is not None:
            _, edge_dist, dx, dz, danger = edge_features

            if danger > 0.4 and edge_dist < self.edge_dist_threshold:
                self.edge_stuck_counter += 1
            else:
                self.edge_stuck_counter *= 0.9

            stuck_ratio = min(
                self.edge_stuck_counter / self.edge_stuck_frames,
                1.0
            )

            reward -= stuck_ratio * danger * self.edge_penalty_weight

        # ------------------------------------------------
        # 5. PANIC MODE
        # ------------------------------------------------
        if hp_fraction < self.panic_hp_threshold:
            panic_strength = (self.panic_hp_threshold - hp_fraction) / self.panic_hp_threshold
            reward -= panic_strength * self.panic_penalty

            if edge_features is not None:
                _, _, dx, dz, _ = edge_features
                escape_strength = np.sqrt(dx * dx + dz * dz)
                reward += escape_strength * self.panic_escape_bonus * panic_strength

        # ------------------------------------------------
        # 6. POSITION PENALTY (NEW)
        # ------------------------------------------------
        if position is not None:
            x, z = position

            # normalize to [0,1]
            nx = np.clip(x / 100.0, 0.0, 1.0)
            nz = np.clip(z / 100.0, 0.0, 1.0)

            def axis_penalty(v):
                """
                Returns 0 in safe zone,
                increases quadratically near borders
                """
                if v < self.edge_zone_start:
                    t = (self.edge_zone_start - v) / self.edge_zone_start
                    return t * t
                elif v > self.edge_zone_end:
                    t = (v - self.edge_zone_end) / (1.0 - self.edge_zone_end)
                    return t * t
                return 0.0

            px = axis_penalty(nx)
            pz = axis_penalty(nz)

            # worst axis dominates (edge hugging)
            pos_penalty = max(px, pz)

            reward -= pos_penalty * self.position_penalty_weight

        # ------------------------------------------------
        # 7. DEATH
        # ------------------------------------------------
        if death:
            reward -= 1.5

        # ------------------------------------------------
        # 8. Clamp
        # ------------------------------------------------
        reward = np.clip(reward, self.min_reward, self.max_reward)
        return float(reward)
