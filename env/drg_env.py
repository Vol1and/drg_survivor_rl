# env/drg_env.py

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.screen import Screen
from env.controller import step as do_step
from env.state.color_mask import ColorMaskExtractor
from env.state.threat_field import ThreatFromMask
from env.state.hp_tracker import HPTracker
from env.state.xp_tracker import XPTracker
from env.logic.reward import RewardFunction
from env.ui.ui_controller import UIController
from env.vision_worker import VisionWorker
from env.buffers.frame_stack import FrameStack
from env.perception.ui_detector import UIDetector
from multiprocessing import Queue
from env.config import SCREEN_SIZE
from env.state.edge_extractor import EdgeFeatureExtractor
from env.state.level_border_mask import LevelBorderDetector

class DRGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, monitor, max_steps=500):
        super().__init__()

        self.screen = Screen(monitor)
        self.max_steps = max_steps
        self.current_step = 0

        # ----------------------------
        # Core systems
        # ----------------------------
        self.hp_tracker = HPTracker()
        self.xp_tracker = XPTracker()
        self.edge_detector = LevelBorderDetector()

        self.edge_extractor = EdgeFeatureExtractor(
            self.edge_detector,
        )
        self.reward_fn = RewardFunction()

        self.color_mask = ColorMaskExtractor()
        self.threat_field = None  # init on reset

        self.ui_detector = UIDetector(threshold=0.7)
        self.ui_controller = UIController()

        self.vision_in = Queue(maxsize=1)
        self.vision_out = Queue(maxsize=1)
        self.vision_worker = None

        self.last_ui_state = "gameplay"

        # ----------------------------
        # Gym spaces
        # ----------------------------
        self.action_space = spaces.Discrete(5)

        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(SCREEN_SIZE, SCREEN_SIZE, 1),
                dtype=np.uint8
            ),
            "hp": spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "threat": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
            "edge": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
        })

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        if self.vision_worker is None or not self.vision_worker.is_alive():
            self.vision_worker = VisionWorker(
                self.vision_in,
                self.vision_out,
                self.ui_detector
            )
            self.vision_worker.start()

        self.current_step = 0
        self.last_ui_state = "gameplay"

        self.reward_fn.reset()
        self.hp_tracker.reset()
        self.xp_tracker.reset()

        frame = self.screen.grab()
        h, w, _ = frame.shape

        self.threat_field = ThreatFromMask((h, w))

        obs = {
            "image": self.screen.to_gray(frame)[..., None],
            "hp": self.hp_tracker.get_current_hp(),
            "threat": np.zeros(5, dtype=np.float32),
            "edge": np.zeros(4, dtype=np.float32),
        }

        return obs, {}

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action: int):

        # ------------------------------------------------
        # 2. Grab frame first
        # ------------------------------------------------
        frame = self.screen.grab()

        # ------------------------------------------------
        # 3. Update UI state
        # ------------------------------------------------
        self._update_ui(frame)
        ui_state = self.last_ui_state

        # ------------------------------------------------
        # 5. Default outputs
        # ------------------------------------------------
        reward = 0.0
        hp = np.array([0.0], dtype=np.float32)
        threat = np.zeros(5, dtype=np.float32)
        edge = np.zeros(4, dtype=np.float32)

        # ------------------------------------------------
        # 6. Perception (only during gameplay)
        # ------------------------------------------------
        if ui_state == "gameplay":
            self.current_step += 1
            self._apply_action(action)
            threat_mask = self.color_mask.extract(frame)
            threat_data = self.threat_field.update(threat_mask)

            edge = self.edge_extractor.extract(frame)

            threat = np.array([
                threat_data["distance"],
                threat_data["dx"],
                threat_data["dy"],
                threat_data["approaching"],
                threat_data["confidence"],
            ], dtype=np.float32)

            hp_delta = self.hp_tracker.get_hp_delta(frame)
            hp_val = self.hp_tracker.get_current_hp()
            hp = np.array([hp_val], dtype=np.float32)

            xp_delta = self.xp_tracker.get_xp_delta(frame)

            reward = self.reward_fn.compute(
                xp_delta=xp_delta,
                death=(ui_state == "restart"),
                hp_delta=hp_delta,
                hp_fraction=hp_val,
                edge_features=edge,
            )

        # ------------------------------------------------
        # 7. UI actions (after reward!)
        # ------------------------------------------------
        elif ui_state != "idle":
            self.manage_ui(ui_state)

            if ui_state == "levelup":
                reward += self.reward_fn.on_levelup()

        # ------------------------------------------------
        # 8. Safety clamps
        # ------------------------------------------------
        if not np.isfinite(reward):
            reward = 0.0

        if not np.all(np.isfinite(threat)):
            threat = np.zeros(5, dtype=np.float32)

        # ------------------------------------------------
        # 9. Episode termination
        # ------------------------------------------------
        done = (
                ui_state == "restart"
                or self.current_step >= self.max_steps
        )

        # ------------------------------------------------
        # 10. Observation
        # ------------------------------------------------
        obs = {
            "image": self.screen.to_gray(frame)[..., None],
            "hp": hp,
            "threat": threat,
            "edge": edge
        }

        info = {
            "step": self.current_step,
            "ui_state": ui_state,
            "reward": round(float(reward), 3),
            "hp": round(float(hp[0]), 3),
            "xp": round(self.xp_tracker.get_current_xp(), 3),
            "edge": edge
        }

        return obs, reward, done, False, info
    # =====================================================
    # HELPERS
    # =====================================================
    def _apply_action(self, action: int):
        do_step(action)

    def _update_ui(self, frame):
        if not self.vision_in.full():
            self.vision_in.put(frame)
            self.last_ui_state = self.vision_out.get()

    def close(self):
        if self.vision_worker:
            self.vision_worker.terminate()
            self.vision_worker.join()

    def manage_ui(self, ui_state):
        # --- UI actions ---
        if ui_state == "levelup":
            self.ui_controller.handle_levelup()

        elif ui_state == "continue":
            self.ui_controller.handle_continue_button()

        elif ui_state == "restart":
            self.ui_controller.handle_death_restart()

        elif ui_state == "overclock":
            self.ui_controller.handle_overclock()

        elif ui_state == "chest":
            self.ui_controller.handle_chest()