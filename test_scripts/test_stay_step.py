import time
from env.controller import step

print("Switch to the game window. Starting in 3 seconds...")
time.sleep(3)

# 5 шагов вправо
for _ in range(5):
    step(4)

# 10 шагов стоим
for _ in range(10):
    step(0)

# 5 шагов вверх
for _ in range(5):
    step(1)

print("Done")
