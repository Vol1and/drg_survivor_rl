class GoldTracker:
    def __init__(self):
        self.prev_gold = 0

    def reset(self):
        self.prev_gold = 0

    def get_current_gold(self):
        return self.prev_gold

    def update(self, new_gold: int) -> int:
        """
        Update internal gold state.
        Returns current gold.
        """
        gold = int(max(0, new_gold))
        self.prev_gold = gold
        return gold

    def get_gold_delta(self, new_gold: int) -> int:
        """
        Returns positive delta only.
        Spending gold is NOT penalized.
        """
        gold_now = int(max(0, new_gold))

        delta = gold_now - self.prev_gold
        self.prev_gold = gold_now

        return max(0, delta)
