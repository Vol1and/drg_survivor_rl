import time
import pydirectinput as pdi

print("Switch to the game window now. Starting in 3 seconds...")
time.sleep(3)

sequence = [("w", 0.4), ("a", 0.4), ("s", 0.4), ("d", 0.4)]

for key, duration in sequence:
    print(f"Press {key} for {duration}s")
    pdi.keyDown(key)
    time.sleep(duration)
    pdi.keyUp(key)
    time.sleep(0.3)

print("Done")
