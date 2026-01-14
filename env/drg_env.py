# env/drg_env.py

import time
import gymnasium as gym
from gymnasium import spaces
import numpy as np

from env.screen import Screen
from env.logic.reward import RewardFunction
from env.ui.ui_controller import UIController
from env.config import SCREEN_SIZE
from env.state.game_state import GameStateReader
from env.state.linear_tracker import LinearTracker
from env.state.fraction_tracker import FractionTracker
from env.controller import MovementController


class DRGEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, monitor, max_steps=20_000):
        super().__init__()

        self.screen = Screen(monitor)
        self.max_steps = max_steps
        self.current_step = 0
        self.game_state = GameStateReader()
        self.frame = self.screen.grab()
        self.obs = None
        self.ui_state = "Loading"
        self.info = {}

        # ----------------------------
        # Core systems
        # ----------------------------
        self.hp_tracker = FractionTracker()
        self.boss_hp_tracker = FractionTracker()

        self.gold_tracker = LinearTracker()
        self.nitra_tracker = LinearTracker()
        self.level_tracker = LinearTracker()

        self.reward_fn = RewardFunction()

        self.ui_controller = UIController()
        self.controller = MovementController()

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
            "flags": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(2,),
                dtype=np.float32
            ),
            "edge": spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32),
            "threat": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(4,),
                dtype=np.float32
            ),
            "objective": spaces.Box(
                low=0.0,
                high=1.0,
                shape=(3,),
                dtype=np.float32
            ),
        })

    # =====================================================
    # RESET
    # =====================================================
    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0
        self.controller.reset()
        self.reward_fn.reset()

        self.hp_tracker.reset()
        self.gold_tracker.reset()
        self.nitra_tracker.reset()
        self.level_tracker.reset()
        self.boss_hp_tracker.reset()

        self.frame = self.screen.grab()

        self.obs = {
            "image": self.screen.process_for_obs(self.frame),
            "hp": self.hp_tracker.get_current(),
            "edge": np.zeros(4, dtype=np.float32),
            "flags": np.zeros(2, dtype=np.float32),
            "threat": np.zeros(4, dtype=np.float32),
            "objective": np.zeros(3, dtype=np.float32),
        }

        return self.obs, {}

    # =====================================================
    # STEP
    # =====================================================
    def step(self, action: int):

        reward = 0.0

        new_ui_state = self.extract_ui_from_state()

        if new_ui_state is not None:
            self.ui_state = new_ui_state

        self.manage_ui(self.ui_state)

        if self.ui_state == "Gameplay":
            self.current_step += 1

            self.controller.step(action)
            self.frame = self.screen.grab()

            (is_mining,
             is_grounded,
             nearest_enemy_distance,
             average_enemy_distance,
             enemies_in_radius,
             has_drop_pod,
             drop_pod_distance,
             level,
             nitra,
             gold,
             hp,
             boss_hp,
             pos_x,
             pos_z,
             move_speed,
             is_boss,
             is_boss_dead,
             boss_distance
             ) = self.extract_info_from_state()

            level_delta = self.level_tracker.get_delta(level)
            nitra_delta = self.nitra_tracker.get_delta(nitra)
            gold_delta = self.gold_tracker.get_delta(gold)

            hp_delta = self.hp_tracker.get_delta(hp)
            boss_hp_delta = self.boss_hp_tracker.get_delta(boss_hp)
            hp_val = self.hp_tracker.get_current()

            hp = np.array([hp_val], dtype=np.float32)

            edge = self._compute_edge_features(pos_x, pos_z, move_speed)

            threat = np.array([nearest_enemy_distance, average_enemy_distance, enemies_in_radius, boss_distance],
                                   dtype=np.float32)
            objective = np.array([has_drop_pod, drop_pod_distance,1.0 if is_boss_dead else 0.0],
                                      dtype=np.float32)

            reward = self.reward_fn.compute(
                hp_delta=hp_delta,
                level_delta=level_delta,
                nitra_delta=nitra_delta,
                gold_delta=gold_delta,
                is_boss=is_boss,
                boss_hp_delta=boss_hp_delta,
                is_boss_dead=is_boss_dead,
                edge_features=edge,
                hp_fraction=hp_val,
                position=(pos_x, pos_z),
                threat=threat,
                objective=objective,
            )

            if not np.isfinite(reward):
                reward = 0.0

            self.obs = {
                "image": self.screen.process_for_obs(self.frame),
                "hp": hp,
                "edge": edge,
                "flags": np.array([1.0 if is_mining else 0.0, 1.0 if is_grounded else 0.0], dtype=np.float32),
                "threat": threat,
                "objective": objective,

            }

            self.info = self._build_info(
                reward=reward,
                hp_val=hp_val,
                hp_delta=hp_delta,
                edge=edge,
                is_mining=is_mining,
                is_grounded=is_grounded,
                is_boss=is_boss,
                is_boss_dead=is_boss_dead,
                nearest_enemy_distance=nearest_enemy_distance,
                average_enemy_distance=average_enemy_distance,
                enemies_in_radius=enemies_in_radius,
                has_drop_pod=has_drop_pod,
                drop_pod_distance=drop_pod_distance,
            )

        done = (
                (self.ui_state == "Death" and self.current_step >= 10)
                or self.current_step >= self.max_steps
        )

        if done:
            reward = self.reward_fn.get_death_reward()

        return self.obs, reward, done, False, self.info

    def manage_ui(self, ui_state):
        # --- UI actions ---
        if ui_state == "LevelUp":
            self.ui_controller.handle_levelup()
        elif ui_state == "ItemFound":
            self.ui_controller.handle_continue_button()
        elif ui_state == "Death":
            self.ui_controller.handle_death_restart()
        elif ui_state == "Overclock":
            self.ui_controller.handle_overclock()
        elif ui_state == "Loading":
            self.controller.idle_step()
        elif ui_state == "Shop":
            self.ui_controller.handle_shop()

    def extract_info_from_state(self):
        state = self.game_state.get()

        is_mining = state['is_mining']
        is_grounded = state['grounded']
        nearest_enemy_distance = np.clip(state['nearest_enemy_distance'] / 30.0, 0.0, 1.0)
        average_enemy_distance = np.clip(state['average_enemy_distance'] / 30.0, 0.0, 1.0)
        enemies_in_radius = np.clip(state['enemies_in_radius'] / 20.0, 0.0, 1.0)
        has_drop_pod = 1.0 if state['has_drop_pod'] else 0.0
        drop_pod_distance = np.clip(state['drop_pod_distance'] / 100.0, 0.0, 1.0)
        level = float(state.get('level', 0))
        nitra = float(state.get('nitra', 0))
        gold = float(state.get('gold', 0))
        hp = float(state.get('hp', 0))
        boss_hp = float(state.get('boss_hp', 0))
        pos_x = float(state["pos"]["x"])
        pos_z = float(state["pos"]["z"])
        is_boss = bool(state.get("is_boss", False))
        is_boss_dead = bool(state.get("is_boss_dead", False))
        move_speed = np.clip(float(state.get("move_speed", 0.0)) / 10.0, 0.0, 1.0)
        boss_distance = np.clip(state['boss_distance'] / 30.0, 0.0, 1.0)

        return (is_mining,
                is_grounded,
                nearest_enemy_distance,
                average_enemy_distance,
                enemies_in_radius,
                has_drop_pod,
                drop_pod_distance,
                level,
                nitra,
                gold,
                hp,
                boss_hp,
                pos_x,
                pos_z,
                move_speed,
                is_boss,
                is_boss_dead,
                boss_distance
                )

    def extract_ui_from_state(self):
        state = self.game_state.get()
        return state.get('phase', None)

    def _compute_edge_features(self, x, z, move_speed):
        """
        Returns:
            np.array([
                move_speed_norm,  # 0..1
                edge_dist,        # 0..1
                dir_x,            # -1..1
                dir_z,            # -1..1
            ])
        """

        # -----------------------------
        # Position (0..100)
        # -----------------------------

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

        if edge_dist > 0.01:
            dx = 0.5 - nx
            dz = 0.5 - nz
            norm = np.sqrt(dx * dx + dz * dz) + 1e-6
            dx /= norm
            dz /= norm
        else:
            dx = 0.0
            dz = 0.0

        return np.array([
            move_speed,
            edge_dist,
            dx,
            dz,
        ], dtype=np.float32)
    def _build_info(
        self,
        *,
        reward,
        hp_val,
        hp_delta,
        edge,
        is_mining,
        is_grounded,
        is_boss,
        is_boss_dead,
        nearest_enemy_distance,
        average_enemy_distance,
        enemies_in_radius,
        has_drop_pod,
        drop_pod_distance,
    ):
        info = {
            # --- episode ---
            "episode/step": self.current_step,
            "episode/ended_by_death": float(self.ui_state == "Death"),
            "episode/ended_by_timeout": float(self.current_step >= self.max_steps),

            # --- reward ---
            "reward/total": reward,
            "reward/damage": hp_delta * self.reward_fn.damage_multiplier if hp_delta < 0 else 0.0,
            "reward/resource": (
                self.reward_fn.gold_weight * self.gold_tracker.last_delta +
                self.reward_fn.nitra_weight * self.nitra_tracker.last_delta
            ),
            "reward/level": self.level_tracker.last_delta * self.reward_fn.levelup_reward,
            "reward/boss_damage": -self.boss_hp_tracker.last_delta if self.boss_hp_tracker.last_delta < 0 else 0.0,
            "reward/boss_spawn": float(is_boss and not self.reward_fn.is_boss_spawn_reward_acquired),
            "reward/boss_death": float(is_boss_dead),
            "reward/edge": self.reward_fn._edge_penalty(edge),
            "reward/position": self.reward_fn._position_penalty((0.5, 0.5)),  # approx center penalty

            # --- hp ---
            "hp/value": hp_val,
            "hp/delta": hp_delta,
            "hp/low": float(hp_val < 0.3),

            # --- threat ---
            "threat/nearest": nearest_enemy_distance,
            "threat/average": average_enemy_distance,
            "threat/density": enemies_in_radius,
            "threat/high": float(nearest_enemy_distance < 0.2 and enemies_in_radius > 0.5),

            # --- objective ---
            "objective/has_drop_pod": has_drop_pod,
            "objective/drop_pod_distance": drop_pod_distance,
            "objective/boss_dead": float(is_boss_dead),
            "objective/reaching_pod": float(has_drop_pod and drop_pod_distance < 0.3),

            # --- edge ---
            "edge/distance": edge[1],
            "edge/stuck": float(edge[1] < 0.1),
            "edge/move_speed": edge[0],

            # --- state ---
            "state/is_mining": float(is_mining),
            "state/is_grounded": float(is_grounded),
            "state/is_boss": float(is_boss),
            "state/is_post_boss": float(is_boss_dead),
            "state/ui": self.ui_state,
        }

        return info
