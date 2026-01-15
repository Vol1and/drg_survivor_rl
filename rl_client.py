import time
import json
import win32file
import cv2

from env.screen import Screen

PIPE_NAME = r"\\.\pipe\drg_rl"
WINDOW_NAME = "DRG Minimap Hue"
SCALE = 4

screen = Screen()

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

cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

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

            b64 = payload.get("minimap_icon")
            w = payload.get("minimap_w")
            h = payload.get("minimap_h")
            if not b64 or not w or not h:
                continue

            # 🔥 вся обработка внутри Screen
            minimap_h = screen.process_minimap(b64, w, h)

            # visualize
            vis = cv2.resize(
                minimap_h,
                (minimap_h.shape[1] * SCALE, minimap_h.shape[0] * SCALE),
                interpolation=cv2.INTER_NEAREST
            )

            cv2.imshow(WINDOW_NAME, vis)
            cv2.waitKey(1)

    except Exception as e:
        print("Disconnected:", e)
        break

cv2.destroyAllWindows()
