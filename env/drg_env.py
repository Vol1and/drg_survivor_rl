# env/drg_env.py

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from env.config import SCREEN_SIZE
from env.screen import Screen
from env.controller import step as do_step
from env.perception.ui_detector import UIDetector
from env.logic.reward import RewardFunction
from env.ui.ui_controller import UIController
from collections import deque
from env.state.hp_tracker import HPTracker
from env.buffers.frame_stack import FrameStack


class DRGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, monitor, max_steps=500):
        super().__init__()

        self.screen = Screen(monitor)
        self.max_steps = max_steps
        self.action_repeat = 3
        self.death_clicked = False


        self.frame_stack = FrameStack()
        self.current_step = 0
        self.hp_tracker = HPTracker()

        self.reward_fn = RewardFunction()
        self.ui_detector = UIDetector(threshold=0.7)
        self.ui_controller = UIController()


        self.action_space = spaces.Discrete(5)
        self.observation_space = spaces.Dict({
            "image": spaces.Box(
                low=0,
                high=255,
                shape=(4, SCREEN_SIZE, SCREEN_SIZE, 1),
                dtype=np.uint8
            ),
            "hp": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(1,),
                dtype=np.float32
            )
        })

    # ------------------------
    # Gym API
    # ------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.death_clicked = False
        self.current_step = 0
        self.reward_fn.reset()
        self.hp_tracker.reset()

        frame = self.screen.grab()
        frame = self.screen.to_gray(frame)[..., None]

        self.frame_stack.reset(frame)

        return  {
            'image': self.frame_stack.get(),
            'hp': self.hp_tracker.get_current_hp(),
        }, {}

    def step(self, action):
        action = int(action)
        total_reward = 0.0
        done = False
        levelup_happened = False
        ui_state = 'gameplay'

        for _ in range(self.action_repeat):

            # --- perform action ---
            self._apply_action(action)

            # --- capture frame ---
            frame = self.screen.grab()

            # --- observation ---
            gray = self.screen.to_gray(frame)[..., None]
            self.frame_stack.append(gray)

            # --- HP ---
            hp_delta = self.hp_tracker.get_hp_delta(frame)
            hp = self.hp_tracker.get_current_hp()

            if self.current_step % 5 == 0:
                ui_state, levelup_happened = self.manage_ui(frame, hp)

            is_death = ui_state == "restart"

            # --- reward ---
            reward = self.reward_fn.compute(
                step=self.current_step,
                action=action,
                death=is_death,
                hp_delta=hp_delta,
                hp_fraction=hp
            )

            if levelup_happened:
                reward += self.reward_fn.on_levelup()

            total_reward += reward
            self.current_step += 1

            if is_death or self.current_step >= self.max_steps:
                done = True
                break

        obs = {
            "image": self.frame_stack.get(),
            "hp": np.array([self.hp_tracker.get_current_hp()], dtype=np.float32)
        }

        info = {
            "step": self.current_step,
            "ui_state": ui_state,
            "reward": total_reward,
            "hp": round(self.hp_tracker.get_current_hp(), 3),
            "death": is_death,
        }

        if self.current_step % 50 == 0:
            print(info)

        return obs, total_reward, done, False, info

    def manage_ui(self, frame, hp):

        ui_state = self.ui_detector.detect(frame, hp)
        levelup_happened = False
        # --- UI actions ---
        if ui_state == "levelup":
            self.ui_controller.handle_levelup()
            levelup_happened = True

        elif ui_state == "continue":
            self.ui_controller.handle_continue_button()

        elif ui_state == "restart":
            self.ui_controller.handle_death_restart()

        elif ui_state == "overclock":
            self.ui_controller.handle_overclock()

        return ui_state, levelup_happened
    # ------------------------
    # Helpers
    # ------------------------

    def _apply_action(self, action: int):
        do_step(action)