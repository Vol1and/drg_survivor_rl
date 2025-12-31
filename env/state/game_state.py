# env/state/game_state_reader.py
import json
import threading
import time
import win32file
import pywintypes


class GameStateReader:
    def __init__(self, pipe_name=r"\\.\pipe\drg_rl"):
        self.pipe_name = pipe_name
        self.latest = None

        self._running = True
        self._buffer = ""
        self._connected = False

        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _connect(self):
        while self._running:
            try:
                pipe = win32file.CreateFile(
                    self.pipe_name,
                    win32file.GENERIC_READ,
                    0,
                    None,
                    win32file.OPEN_EXISTING,
                    0,
                    None,
                )

                if not self._connected:
                    print("[GameStateReader] Connected")
                    self._connected = True

                return pipe

            except pywintypes.error:
                if self._connected:
                    print("[GameStateReader] Waiting for game...")
                    self._connected = False
                time.sleep(0.2)

    def _extract_json_objects(self):
        objs = []
        depth = 0
        start = None

        for i, ch in enumerate(self._buffer):
            if ch == "{":
                if depth == 0:
                    start = i
                depth += 1

            elif ch == "}":
                depth -= 1
                if depth == 0 and start is not None:
                    objs.append(self._buffer[start:i + 1])
                    start = None

        if depth == 0 and objs:
            self._buffer = self._buffer[self._buffer.rfind("}") + 1 :]
        return objs

    def _run(self):
        pipe = self._connect()

        while self._running:
            try:
                _, data = win32file.ReadFile(pipe, 4096)
                self._buffer += data.decode(errors="ignore")

                for raw in self._extract_json_objects():
                    try:
                        obj = json.loads(raw)
                        if isinstance(obj, dict):
                            self.latest = obj
                    except json.JSONDecodeError:
                        pass

            except Exception:
                if self._connected:
                    print("[GameStateReader] Lost connection")
                self._connected = False
                time.sleep(0.2)
                pipe = self._connect()

    def get(self):
        return self.latest

    def close(self):
        self._running = False
