import time
import pydirectinput as pdi

GAME_SPEED = 2

class MovementController:
    """
    Low-level movement controller with:
    - inertia (hold key)
    - anti-idle bias
    - anti-opposite-direction penalty
    """

    ACTIONS = {
        0: None,   # idle
        1: "w",
        2: "s",
        3: "a",
        4: "d",
    }

    OPPOSITE = {
        "w": "s",
        "s": "w",
        "a": "d",
        "d": "a",
    }

    def __init__(
        self,
        step_time: float = 0.1,
        idle_factor: float = 0.5,
        turn_penalty_factor: float = 0.4,
    ):
        """
        step_time: базовая длительность шага
        idle_factor: idle будет step_time * idle_factor
        turn_penalty_factor: штраф за разворот (в долях step_time)
        """

        step_time = step_time / GAME_SPEED
        self.step_time = step_time
        self.idle_time = step_time * idle_factor
        self.turn_penalty = step_time * turn_penalty_factor

        self._last_key = None

    # ------------------------------------------------
    # MAIN STEP
    # ------------------------------------------------
    def step(self, action: int):
        key = self.ACTIONS.get(action, None)

        # -------------------------
        # IDLE
        # -------------------------
        if key is None:
            self._release_current()
            time.sleep(self.idle_time)
            return

        # -------------------------
        # DIRECTION CHANGE
        # -------------------------
        if self._last_key is not None:
            # противоположное направление → задержка
            if key == self.OPPOSITE.get(self._last_key):
                time.sleep(self.turn_penalty)

            # смена направления
            if key != self._last_key:
                pdi.keyUp(self._last_key)
                pdi.keyDown(key)
                self._last_key = key
            # если направление то же — продолжаем держать
        else:
            # начало движения
            pdi.keyDown(key)
            self._last_key = key

        time.sleep(self.step_time)

    # ------------------------------------------------
    # RESET (call on env.reset)
    # ------------------------------------------------
    def reset(self):
        """
        Полный сброс контроллера.
        ОБЯЗАТЕЛЬНО вызывать при env.reset()
        """
        self._release_current()

    def idle_step(self):
        time.sleep(1)
    # ------------------------------------------------
    # INTERNAL
    # ------------------------------------------------
    def _release_current(self):
        if self._last_key is not None:
            pdi.keyUp(self._last_key)
            self._last_key = None
