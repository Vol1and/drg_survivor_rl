import time
import win32file

PIPE_NAME = r"\\.\pipe\drg_rl"

print("Waiting for game...")

while True:
    try:
        pipe = win32file.CreateFile(
            PIPE_NAME,
            win32file.GENERIC_READ,
            0,
            None,
            win32file.OPEN_EXISTING,
            0,
            None
        )
        print("Connected to game!")
        break
    except:
        time.sleep(0.5)

while True:
    try:
        _, data = win32file.ReadFile(pipe, 4096)
        print(data.decode())
    except Exception as e:
        print("Disconnected:", e)
        break
