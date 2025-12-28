import time
import pydirectinput as pdi

ACTIONS = {
    0: None,   # стоять
    1: "w",
    2: "s",
    3: "a",
    4: "d",
}

STEP_TIME = 0.08



def press_key(key: str, duration: float = 0.1):
    pdi.keyDown(key)
    time.sleep(duration)
    pdi.keyUp(key)

def step(action: int):
    key = ACTIONS[action]

    if key is None:
        # ничего не нажимаем, просто ждём шаг
        time.sleep(STEP_TIME)
        return

    pdi.keyDown(key)
    time.sleep(STEP_TIME)
    pdi.keyUp(key)
    
def click(x: int, y: int, delay: float = 0.1):
    pdi.moveTo(x, y)
    time.sleep(delay)
    pdi.click()