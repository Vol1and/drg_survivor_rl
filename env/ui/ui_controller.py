import time
import pydirectinput


class UIController:
    def __init__(self):
        self.levelup_click_pos = (836, 525)
        self.overclock_click_pos = (836, 525)
        self.death_restart_click_pos = (950, 1435)
        self.continue_button_pos = (1150, 1345)

        # Настройки pydirectinput
        pydirectinput.PAUSE = 0
        pydirectinput.FAILSAFE = False

        self.click_delay = 0.05  # безопасная задержка

    def _click(self, x, y):
        """
        Надёжный клик для игр (DirectInput)
        """
        pydirectinput.moveTo(x, y)
        time.sleep(self.click_delay)
        pydirectinput.click()

    def handle_levelup(self):
        print(f"[UI] Levelup click at {self.levelup_click_pos}")
        self._click(*self.levelup_click_pos)
        time.sleep(1)

    def handle_continue_button(self):
        print(f"[UI] Continue click at {self.continue_button_pos}")
        self._click(*self.continue_button_pos)

    def handle_chest(self):
        print(f"[UI] Chest click at {self.levelup_click_pos}")
        self._click(*self.levelup_click_pos)
        time.sleep(1)

    def handle_overclock(self):
        print(f"[UI] Overclock click at {self.overclock_click_pos}")
        self._click(*self.overclock_click_pos)
        time.sleep(1)

    def handle_death_restart(self):
        print(f"[UI] Death restart click at {self.death_restart_click_pos}")
        self._click(*self.death_restart_click_pos)
        time.sleep(4)
