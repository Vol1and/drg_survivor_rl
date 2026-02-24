import time
import json
import win32file

PIPE_NAME = r"\\.\pipe\drg_rl"
SCALE = 4


print("Waiting for game...")

# ================= PIPE CONNECT =================

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
    except Exception:
        time.sleep(0.5)

buffer = ""

# ================= MAIN LOOP =================

while True:
    try:
        _, data = win32file.ReadFile(pipe, 262144)
        buffer += data.decode("utf-8", errors="ignore")

        while "\n" in buffer:
            line, buffer = buffer.split("\n", 1)
            if not line.strip():
                continue

            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                continue

    except Exception as e:
        print("Disconnected:", e)
        break