# env/drg_env.py

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.screen import Screen
from env.controller import step as do_step
from env.state.xp_tracker import XPTracker
from env.logic.reward import RewardFunction
from env.ui.ui_controller import UIController
from env.vision_worker import VisionWorker
from env.buffers.frame_stack import FrameStack
from env.perception.ui_detector import UIDetector
from multiprocessing import Queue
from env.config import SCREEN_SIZE
from env.state.game_state import GameStateReader
from env.state.hp_tracker import HPTracker


class DRGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, monitor, max_steps=500):
        super().__init__()

        self.screen = Screen(monitor)
        self.max_steps = max_steps
        self.current_step = 0

        self.game_state = GameStateReader()

        # ----------------------------
        # Core systems
        # ----------------------------
        self.hp_tracker = HPTracker()
        self.xp_tracker = XPTracker()

        self.reward_fn = RewardFunction()

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
            "edge": spaces.Box(low=-1.0, high=1.0, shape=(5,), dtype=np.float32),
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


        obs = {
            "image": self.screen.to_gray(frame)[..., None],
            "hp": self.hp_tracker.get_current_hp(),
            "edge": np.zeros(5, dtype=np.float32),
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
        edge = np.zeros(5, dtype=np.float32)

        state = self.game_state.get()
        # ------------------------------------------------
        # 6. Perception (only during gameplay)
        # ------------------------------------------------
        if ui_state == "gameplay" and state is not None:
            self.current_step += 1
            self._apply_action(action)


            hp_delta = self.hp_tracker.get_hp_delta(float(state.get('hp', 0)))
            hp_val = self.hp_tracker.get_current_hp()
            hp = np.array([hp_val], dtype=np.float32)
            edge = self._compute_edge_features(state)

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
            "edge": edge,
        }


        info = {
            "step": self.current_step,
            "ui_state": ui_state,
            "reward": round(float(reward), 3),
            "hp": round(float(hp[0]), 3),
            "xp": round(self.xp_tracker.get_current_xp(), 3),
            "danger": edge[4],
            "edge_ds": edge[1],
            "levels": state.get("level", 0) if state is not None else 1,
        }

        if self.current_step % 100 == 0:
            print(state)
            print(edge)

        if edge[4] > 0:
            print(edge)

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

    def _compute_edge_features(self, state):
        """
        Returns:
            np.array([
                move_speed_norm,  # 0..1
                edge_dist,        # 0..1
                dir_x,            # -1..1
                dir_z,            # -1..1
                danger            # 0..1
            ])
        """

        if not state or "pos" not in state:
            return np.zeros(5, dtype=np.float32)

        # -----------------------------
        # Position (0..100)
        # -----------------------------
        x = float(state["pos"]["x"])
        z = float(state["pos"]["z"])

        # Movement
        velocity_x = float(state["vel"].get("x", 0.0))
        velocity_z = float(state["vel"].get("y", 0.0))

        move_speed = float(state.get("move_speed", 0.0))

        # -----------------------------
        # Map normalization
        # -----------------------------
        MAP_MIN = 0.0
        MAP_MAX = 100.0
        MAP_RANGE = MAP_MAX - MAP_MIN

        nx = np.clip((x - MAP_MIN) / MAP_RANGE, 0.0, 1.0)
        nz = np.clip((z - MAP_MIN) / MAP_RANGE, 0.0, 1.0)

        # Distance to closest border
        edge_dist = min(nx, 1 - nx, nz, 1 - nz)

        # -----------------------------
        # Direction away from edge
        # -----------------------------
        grounded = bool(state.get("grounded", False))

        if edge_dist > 0.01 and not grounded:
            dx = 0.5 - nx
            dz = 0.5 - nz
            norm = np.sqrt(dx * dx + dz * dz) + 1e-6
            dx /= norm
            dz /= norm
        else:
            dx = 0.0
            dz = 0.0

        # -----------------------------
        # Danger estimation
        # -----------------------------
        danger = 0.0

        # proximity-based danger
        if not grounded and edge_dist < 0.25:
            approach_strength = max(
                0.0,
                -((dx * (0.5 - nx)) + (dz * (0.5 - nz)))
            )
            danger += approach_strength * (0.25 - edge_dist) * 0.6

        # strong penalty if physically stuck
        if grounded:
            danger += 0.7

        danger = np.clip(danger, 0.0, 1.0)

        # -----------------------------
        # Normalize speed (soft clamp)
        # -----------------------------
        MAX_SPEED = 10.0  # tweak if needed
        move_speed_norm = np.clip(move_speed / MAX_SPEED, 0.0, 1.0)

        return np.array([
            move_speed_norm,
            edge_dist,
            dx,
            dz,
            danger
        ], dtype=np.float32)


