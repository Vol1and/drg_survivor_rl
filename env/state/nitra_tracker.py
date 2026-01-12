class NitraTracker:
    def __init__(self):
        self.prev_nitra = 0

    def reset(self):
        self.prev_nitra = 0

    def get_current_nitra(self):
        return self.prev_nitra

    def update(self, new_nitra: int) -> int:
        """
        Update internal nitra state.
        Returns current nitra.
        """
        nitra = int(max(0, new_nitra))
        self.prev_nitra = nitra
        return nitra

    def get_nitra_delta(self, new_nitra: int) -> int:
        """
        Returns positive delta only.
        Spending nitra is NOT penalized.
        """
        nitra_now = int(max(0, new_nitra))

        delta = nitra_now - self.prev_nitra
        self.prev_nitra = nitra_now

        return max(0, delta)
