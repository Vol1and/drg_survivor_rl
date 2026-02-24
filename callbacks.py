from stable_baselines3.common.callbacks import BaseCallback
import numpy as np
import csv
import os
import time
import subprocess
from datetime import datetime
from collections import deque
import win32gui
import win32con
import pydirectinput as pdi
class EpisodeStatsCallback(BaseCallback):
    """
    Collects rich per-episode statistics based on structured env `info`,
    including minimap diagnostics.
    """

    def __init__(self, window=50, verbose=0):
        super().__init__(verbose)
        self.window = window

        # rolling buffers
        self.steps_runs = deque(maxlen=window)
        self.reward_runs = deque(maxlen=window)
        self.reward_per_step_runs = deque(maxlen=window)

        self.death_runs = deque(maxlen=window)
        self.boss_kill_runs = deque(maxlen=window)
        self.drop_pod_runs = deque(maxlen=window)

        # minimap rolling
        self.minimap_mean_runs = deque(maxlen=window)
        self.minimap_std_runs = deque(maxlen=window)
        self.minimap_nonzero_runs = deque(maxlen=window)

        self._reset_episode()

    # -------------------------------------------------
    def _reset_episode(self):
        # --- core ---
        self.steps = 0
        self.reward_sum = 0.0

        # --- reward decomposition ---
        self.reward_damage = 0.0
        self.reward_resource = 0.0
        self.reward_level = 0.0
        self.reward_boss_damage = 0.0
        self.reward_edge = 0.0
        self.reward_position = 0.0

        # --- hp / risk ---
        self.damage_events = 0
        self.low_hp_steps = 0

        # --- spatial ---
        self.edge_stuck_steps = 0

        # --- threat ---
        self.threat_high_steps = 0

        # --- objectives ---
        self.boss_seen = False
        self.boss_killed = False
        self.post_boss_steps = 0
        self.reached_drop_pod = False

        # --- minimap ---
        self.minimap_mean_sum = 0.0
        self.minimap_std_sum = 0.0
        self.minimap_nonzero_steps = 0
        self.minimap_empty_steps = 0

        self.last_hp = None

    # -------------------------------------------------
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        dones = self.locals.get("dones", [])

        for i, info in enumerate(infos):
            if not isinstance(info, dict):
                continue

            if info.get("state/ui") == "Gameplay":
                self.steps += 1

                # --- reward ---
                r = float(info.get("reward/total", 0.0))
                self.reward_sum += r

                self.reward_damage += float(info.get("reward/damage", 0.0))
                self.reward_resource += float(info.get("reward/resource", 0.0))
                self.reward_level += float(info.get("reward/level", 0.0))
                self.reward_boss_damage += float(info.get("reward/boss_damage", 0.0))
                self.reward_edge += float(info.get("reward/edge", 0.0))
                self.reward_position += float(info.get("reward/position", 0.0))

                # --- hp ---
                hp = float(info.get("hp/value", 1.0))
                if info.get("hp/delta", 0.0) < 0:
                    self.damage_events += 1
                if info.get("hp/low", 0.0) > 0:
                    self.low_hp_steps += 1
                self.last_hp = hp

                # --- edge / threat ---
                if info.get("edge/stuck", 0.0) > 0:
                    self.edge_stuck_steps += 1
                if info.get("threat/high", 0.0) > 0:
                    self.threat_high_steps += 1

                # --- objectives ---
                if info.get("state/is_boss", 0.0) > 0:
                    self.boss_seen = True
                if info.get("objective/boss_dead", 0.0) > 0:
                    self.boss_killed = True
                if info.get("state/is_post_boss", 0.0) > 0:
                    self.post_boss_steps += 1
                if info.get("objective/reaching_pod", 0.0) > 0:
                    self.reached_drop_pod = True

                # --- minimap ---
                mm_mean = float(info.get("minimap/mean", 0.0))
                mm_std = float(info.get("minimap/std", 0.0))
                mm_nonzero = float(info.get("minimap/nonzero_ratio", 0.0))

                self.minimap_mean_sum += mm_mean
                self.minimap_std_sum += mm_std

                if mm_nonzero < 0.05:
                    self.minimap_empty_steps += 1
                else:
                    self.minimap_nonzero_steps += 1

            # -------- Episode finished --------
            if dones[i]:
                if self.steps == 0:
                    self._reset_episode()
                    continue

                reward_per_step = self.reward_sum / max(1, self.steps)
                done_by_death = float(info.get("episode/ended_by_death", 0.0) > 0)

                mm_mean_ep = self.minimap_mean_sum / max(1, self.steps)
                mm_std_ep = self.minimap_std_sum / max(1, self.steps)
                mm_nonzero_ratio_ep = self.minimap_nonzero_steps / max(1, self.steps)

                # ===== episode =====
                self.logger.record("episode/steps", self.steps)
                self.logger.record("episode/reward_total", self.reward_sum)
                self.logger.record("episode/reward_per_step", reward_per_step)

                self.logger.record("episode/minimap_mean", mm_mean_ep)
                self.logger.record("episode/minimap_std", mm_std_ep)
                self.logger.record("episode/minimap_nonzero_ratio", mm_nonzero_ratio_ep)
                self.logger.record("episode/minimap_empty_steps", self.minimap_empty_steps)

                self.logger.record("episode/boss_seen", float(self.boss_seen))
                self.logger.record("episode/boss_killed", float(self.boss_killed))
                self.logger.record("episode/reached_drop_pod", float(self.reached_drop_pod))
                self.logger.record("episode/done_by_death", done_by_death)
                self.logger.record("episode/hp_end", self.last_hp)

                # ===== rolling =====
                self.steps_runs.append(self.steps)
                self.reward_runs.append(self.reward_sum)
                self.reward_per_step_runs.append(reward_per_step)
                self.death_runs.append(done_by_death)
                self.boss_kill_runs.append(float(self.boss_killed))
                self.drop_pod_runs.append(float(self.reached_drop_pod))

                self.minimap_mean_runs.append(mm_mean_ep)
                self.minimap_std_runs.append(mm_std_ep)
                self.minimap_nonzero_runs.append(mm_nonzero_ratio_ep)

                self.logger.record("rolling/steps_mean", np.mean(self.steps_runs))
                self.logger.record("rolling/reward_mean", np.mean(self.reward_runs))
                self.logger.record("rolling/reward_per_step_mean", np.mean(self.reward_per_step_runs))
                self.logger.record("rolling/death_ratio", np.mean(self.death_runs))
                self.logger.record("rolling/boss_kill_ratio", np.mean(self.boss_kill_runs))
                self.logger.record("rolling/drop_pod_success_ratio", np.mean(self.drop_pod_runs))

                self.logger.record("rolling/minimap_mean", np.mean(self.minimap_mean_runs))
                self.logger.record("rolling/minimap_std", np.mean(self.minimap_std_runs))
                self.logger.record("rolling/minimap_nonzero_ratio", np.mean(self.minimap_nonzero_runs))

                self.logger.dump(self.num_timesteps)
                self._reset_episode()

        return True


class GameRestartCallback(BaseCallback):
    """
    Перезапускает игру каждые N эпизодов чтобы избежать глюков.
    Убивает процесс игры через taskkill и перезапускает через Steam.

    После перезапуска ждёт переподключения пайпа (GameStateReader
    автоматически переподключается в фоновом потоке).

    Параметры:
        restart_every     - каждые N эпизодов перезапускать игру
        game_exe          - имя exe-процесса игры
        steam_game_id     - Steam App ID (Deep Rock Galactic = 548430)
        min_wait_seconds  - минимальное ожидание после запуска (пока грузится)
        pipe_timeout      - максимум секунд ожидания переподключения пайпа
    """

    def __init__(
        self,
        restart_every: int = 50,
        game_exe: str = "DRG Survivor.exe",
        steam_game_id: int = 2321470,
        min_wait_seconds: int = 30,
        pipe_timeout: int = 300,
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.restart_every = restart_every
        self.game_exe = game_exe
        self.steam_game_id = steam_game_id
        self.min_wait_seconds = min_wait_seconds
        self.pipe_timeout = pipe_timeout
        self.episode_count = 0
        self._game_rect = None  # (x0, y0, x1, y1) — заполняется после фокуса

    def _on_training_start(self) -> None:
        print("\n[GameRestart] Старт обучения — запускаю игру...")
        self._launch_game()

    def _on_step(self) -> bool:
        dones = self.locals.get("dones", [])
        for done in dones:
            if done:
                self.episode_count += 1
                if self.episode_count % self.restart_every == 0:
                    self._restart_game()
        return True

    def _restart_game(self):
        print(f"\n[GameRestart] Эпизод {self.episode_count}: перезапускаю игру...")

        # --- убиваем процесс игры ---
        result = subprocess.run(
            ["taskkill", "/F", "/IM", self.game_exe],
            capture_output=True, text=True
        )
        if self.verbose:
            msg = result.stdout.strip() or result.stderr.strip()
            print(f"[GameRestart] taskkill: {msg}")
        time.sleep(3)

        self._launch_game()

    def _click(self, x, y):
        """Надёжный клик для игр (DirectInput), абсолютные экранные координаты."""
        pdi.moveTo(x, y)
        time.sleep(2)
        pdi.click()
        time.sleep(2)
        pdi.click()

    def _click_game(self, rel_x, rel_y):
        """Клик по координатам относительно левого верхнего угла окна игры."""
        if self._game_rect is None:
            print("[GameRestart] WARNING: game rect неизвестен, кликаю по абсолютным координатам")
            self._click(rel_x, rel_y)
            return
        x0, y0 = self._game_rect[0], self._game_rect[1]
        self._click(x0 + rel_x, y0 + rel_y)

    def _launch_game(self):
        # --- запускаем игру через Steam ---
        os.startfile(f"steam://rungameid/{self.steam_game_id}")
        print(f"[GameRestart] Запустил steam://rungameid/{self.steam_game_id}")

        # --- минимальное ожидание пока грузится игра ---
        print(f"[GameRestart] Жду минимум {self.min_wait_seconds}с пока загружается...")
        time.sleep(self.min_wait_seconds)

        # --- ждём переподключения пайпа ---
        game_state = self._get_game_state_reader()
        if game_state is not None:
            print("[GameRestart] Жду переподключения пайпа...")
            start = time.time()
            while not game_state._connected:
                if time.time() - start > self.pipe_timeout:
                    print("[GameRestart] ВНИМАНИЕ: таймаут ожидания подключения пайпа!")
                    break
                time.sleep(1)
            if game_state._connected:
                print("[GameRestart] Пайп подключён. Продолжаю обучение.\n")
                self._focus_game_window()
                print('Клик 1')
                self._click_game(340, 1100)

                print('Клик по сохранению')
                self._click(340, 1100)
                print('Клик по кнопке играть')
                self._click(400, 500)
                print('Клик по режиму игры')
                self._click(400, 500)
                print('Клик по карте')
                self._click(400, 600)
                print('Клик по персонажу')
                self._click(500, 550)
                print('Клик по спеку')
                self._click(400, 600)
                print('Клик по сложности')
                self._click(1200, 600)


        else:
            print("[GameRestart] Не удалось получить GameStateReader, продолжаю без проверки пайпа.\n")

    def _focus_game_window(self):
        """Находит окно игры, выводит на передний план и сохраняет его rect."""
        import ctypes
        import win32process
        import psutil

        exe_base = self.game_exe.replace(".exe", "").lower()
        hwnd_found = []

        def _enum_cb(hwnd, _):
            if not win32gui.IsWindowVisible(hwnd):
                return
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            try:
                if exe_base in psutil.Process(pid).name().lower():
                    hwnd_found.append(hwnd)
            except Exception:
                pass

        win32gui.EnumWindows(_enum_cb, None)

        if not hwnd_found:
            print("[GameRestart] Game window not found, skipping focus")
            self._game_rect = None
            return

        hwnd = hwnd_found[0]

        # Восстанавливаем из минимизации
        win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
        time.sleep(0.5)

        # AttachThreadInput надёжнее Alt-трюка — позволяет захватить фокус из фона
        current_tid = ctypes.windll.kernel32.GetCurrentThreadId()
        target_tid, _ = win32process.GetWindowThreadProcessId(hwnd)
        attached = ctypes.windll.user32.AttachThreadInput(current_tid, target_tid, True)
        try:
            ctypes.windll.user32.SetForegroundWindow(hwnd)
            win32gui.BringWindowToTop(hwnd)
        finally:
            if attached:
                ctypes.windll.user32.AttachThreadInput(current_tid, target_tid, False)

        time.sleep(1.0)

        # Проверяем что фокус реально получен
        active = win32gui.GetForegroundWindow()
        if active != hwnd:
            print(f"[GameRestart] WARNING: window did not get focus (active={active}, target={hwnd})")
        else:
            print(f"[GameRestart] Game window focused (hwnd={hwnd})")

        # Сохраняем позицию окна — нужна для корректных кликов на любом мониторе
        self._game_rect = win32gui.GetWindowRect(hwnd)
        x0, y0, x1, y1 = self._game_rect
        print(f"[GameRestart] Window rect: ({x0}, {y0}) — ({x1}, {y1})")

    def _get_game_state_reader(self):
        """Достаём GameStateReader из вложенного DRGEnv."""
        try:
            # VecNormalize → DummyVecEnv → DRGEnv
            env = self.training_env
            if hasattr(env, "venv"):
                env = env.venv
            if hasattr(env, "envs"):
                return env.envs[0].game_state
        except Exception as e:
            print(f"[GameRestart] Не удалось получить game_state: {e}")
        return None





