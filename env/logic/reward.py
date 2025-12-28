class RewardFunction:
    def __init__(self):
        # --- базовые награды ---
        self.survival_reward = 0.002
        self.hp_weight = 0.15
        self.hp_delta_weight = 1.0
        self.death_penalty = -3.0

        # --- level up ---
        self.levelup_reward = 2.0
        self.levelups = 0

        # стабилизация
        self.min_reward = -3.0
        self.max_reward = 3.0

    def reset(self):
        self.levelups = 0

    def on_levelup(self):
        self.levelups += 1
        return self.levelup_reward

    def compute(self, *, step, action, death: bool, hp_fraction: float, hp_delta: float):
        reward = 0.0

        # жизнь
        reward += self.survival_reward

        # текущее здоровье
        reward += hp_fraction * self.hp_weight

        # потеря здоровья
        reward += hp_delta * self.hp_delta_weight

        # смерть
        if death:
            reward += self.death_penalty

        # clamp
        reward = max(self.min_reward, min(self.max_reward, reward))
        return reward
