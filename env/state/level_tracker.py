class LevelTracker:
    def __init__(self):
        self.prev_level = 0

    def reset(self):
        self.prev_level = 0

    def get_current_level(self):
        return self.prev_level

    def update(self, new_level: int) -> int:
        """
        Update internal level state.
        Returns current level.
        """
        level = int(max(0, new_level))
        self.prev_level = level
        return level

    def get_level_delta(self, new_level: int) -> int:
        """
        Returns positive delta only.
        Spending level is NOT penalized.
        """
        level_now = int(max(0, new_level))

        delta = level_now - self.prev_level
        self.prev_level = level_now

        return max(0, delta)
