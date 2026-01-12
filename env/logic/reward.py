import numpy as np


class RewardFunction:
    def __init__(self):
        # ==================================================
        # CLAMP
        # ==================================================
        self.min_reward = -10.0
        self.max_reward = 10.0

        # ==================================================
        # SURVIVAL / DAMAGE
        # ==================================================
        self.base_survival_reward = 0.07
        self.damage_multiplier = 10.0
        self.death_penalty = 5.0

        # ==================================================
        # PROGRESSION
        # ==================================================
        self.levelup_reward = 0.3

        # ==================================================
        # BOSS LOGIC
        # ==================================================
        self.boss_spawn_reward = 3.0
        self.boss_death_reward = 5.0
        self.boss_damage_weight = 2.0

        self.is_boss_spawn_reward_acquired = False
        self.is_boss_death_reward_acquired = False

        # ==================================================
        # RESOURCES
        # ==================================================
        self.gold_weight = 0.12
        self.nitra_weight = 0.12

        # ==================================================
        # EDGE / POSITION
        # ==================================================
        self.edge_penalty_weight = -0.07
        self.edge_dist_threshold = 0.25
        self.edge_stuck_frames = 30
        self.edge_stuck_counter = 0

        self.edge_zone_start = 0.25
        self.edge_zone_end = 0.75

        # ==================================================
        # THREAT / OBJECTIVE
        # ==================================================
        self.threat_penalty_weight = 0.3
        self.escape_reward_weight = 0.5

    # ==================================================
    # LIFECYCLE
    # ==================================================
    def reset(self):
        self.edge_stuck_counter = 0
        self.is_boss_spawn_reward_acquired = False
        self.is_boss_death_reward_acquired = False

    # ==================================================
    # MAIN ENTRY
    # ==================================================
    def compute(
        self,
        *,
        hp_delta: float,
        gold_delta: float,
        nitra_delta: float,
        hp_fraction: float,
        level_delta: float,
        is_boss: bool,
        boss_hp_delta: float,
        is_boss_dead: bool,
        threat=None,       # [nearest_dist, avg_dist, density]
        objective=None,    # [has_pod, pod_dist, is_post_boss]
        edge_features=None,
        position=None,
    ) -> float:

        reward = 0.0

        # --- HP scaling (never goes to zero)
        hp_scale = 0.25 + 0.75 * hp_fraction

        # --- Core survival & damage
        reward += self._survival_reward(threat)
        reward += self._damage_penalty(hp_delta, hp_fraction)

        # --- Progression
        reward += self._resource_reward(gold_delta, nitra_delta) * hp_scale
        reward += level_delta * self.levelup_reward * hp_scale

        # --- Boss logic
        reward += self._boss_reward(
            is_boss=is_boss,
            is_boss_dead=is_boss_dead,
            boss_hp_delta=boss_hp_delta,
            hp_scale=hp_scale,
        )

        # --- Threat pressure
        if threat is not None:
            reward += self._threat_penalty(threat, hp_fraction)

        # --- Escape objective (post-boss)
        if objective is not None:
            reward += self._escape_reward(objective, hp_scale)

        # --- Edge / position
        if edge_features is not None:
            reward += self._edge_penalty(edge_features)

        if position is not None:
            reward += self._position_penalty(position)

        return float(np.clip(reward, self.min_reward, self.max_reward))

    # ==================================================
    # COMPONENTS
    # ==================================================
    def _survival_reward(self, threat):
        if threat is None:
            return self.base_survival_reward

        _, avg_dist, density = threat
        danger = (1.0 - avg_dist) * density
        return self.base_survival_reward * (1.0 - 0.5 * danger)

    def _damage_penalty(self, hp_delta, hp_fraction):
        if hp_delta >= 0:
            return 0.0

        scale = 1.0 + (1.0 - hp_fraction) * 1.5
        return hp_delta * self.damage_multiplier * scale

    def _resource_reward(self, gold_delta, nitra_delta):
        reward = 0.0
        if gold_delta > 0:
            reward += gold_delta * self.gold_weight
        if nitra_delta > 0:
            reward += nitra_delta * self.nitra_weight
        return reward

    def _boss_reward(self, *, is_boss, is_boss_dead, boss_hp_delta, hp_scale):
        reward = 0.0

        if not is_boss:
            return 0.0

        if not self.is_boss_spawn_reward_acquired:
            self.is_boss_spawn_reward_acquired = True
            reward += self.boss_spawn_reward * hp_scale

        if is_boss_dead and not self.is_boss_death_reward_acquired:
            self.is_boss_death_reward_acquired = True
            reward += self.boss_death_reward * hp_scale

        if boss_hp_delta < 0:
            reward += (-boss_hp_delta) * self.boss_damage_weight * hp_scale

        return reward

    def _threat_penalty(self, threat, hp_fraction):
        nearest_dist, _, density = threat

        # higher penalty when low HP
        hp_scale = 1.0 + (1.0 - hp_fraction)

        risk = (1.0 - nearest_dist) * density
        return -self.threat_penalty_weight * risk * hp_scale

    def _escape_reward(self, objective, hp_scale):
        has_pod, pod_dist, is_post_boss = objective

        if not is_post_boss or not has_pod:
            return 0.0

        return self.escape_reward_weight * (1.0 - pod_dist)

    def _edge_penalty(self, edge_features):
        _, edge_dist, _, _ = edge_features

        if edge_dist < self.edge_dist_threshold:
            self.edge_stuck_counter += 1
        else:
            self.edge_stuck_counter *= 0.9

        stuck_ratio = min(self.edge_stuck_counter / self.edge_stuck_frames, 1.0)
        return np.clip(stuck_ratio * self.edge_penalty_weight, -1.0, 1.0)

    def _position_penalty(self, position):
        x, z = position
        nx = np.clip(x / 100.0, 0.0, 1.0)
        nz = np.clip(z / 100.0, 0.0, 1.0)

        def axis_penalty(v):
            if v < self.edge_zone_start:
                t = (self.edge_zone_start - v) / self.edge_zone_start
                return t * t
            if v > self.edge_zone_end:
                t = (v - self.edge_zone_end) / (1.0 - self.edge_zone_end)
                return t * t
            return 0.0

        return -np.clip(max(axis_penalty(nx), axis_penalty(nz)), 0.0, 1.0)

    # ==================================================
    # TERMINAL
    # ==================================================
    def get_death_reward(self):
        return -self.death_penalty
