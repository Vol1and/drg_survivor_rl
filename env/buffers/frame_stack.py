from collections import deque
import numpy as np

class FrameStack:
    def __init__(self, n_frames=4):
        self.n_frames = n_frames
        self.frames = deque(maxlen=n_frames)

    def reset(self, obs):
        self.frames.clear()
        for _ in range(self.n_frames):
            self.frames.append(obs.copy())

    def append(self, obs):
        self.frames.append(obs.copy())

    def get(self):
        # (T, H, W, C)
        return np.stack(self.frames, axis=0)
