class LinearTracker:
    def __init__(self):
        self.prev_val = 0
        self.last_delta = 0

    def reset(self):
        self.prev_val = 0
        self.last_delta = 0

    def get_current(self):
        return self.prev_val

    def get_last_delta(self):
        return self.last_delta

    def update(self, new_val: int) -> int:
        """
        Update internal val state.
        Returns current val.
        """
        val = int(max(0, new_val))
        self.prev_val = val
        return val

    def get_delta(self, new_val: int) -> int:
        """
        Returns positive delta only.
        Spending val is NOT penalized.
        """
        val_now = int(max(0, new_val))

        self.last_delta = val_now - self.prev_val
        self.prev_val = val_now

        return max(0, self.last_delta)
