"""
Тест фокуса окна игры.
Запусти игру, потом запусти этот скрипт.
"""
import ctypes
import win32gui
import win32con
import win32process
import psutil

GAME_EXE = "DRG Survivor"  # без .exe


def force_focus(hwnd):
    """SetForegroundWindow с обходом ограничения Windows через keybd_event(Alt)."""
    KEYEVENTF_KEYUP = 0x0002
    VK_MENU = 0x12  # Alt

    # Симулируем нажатие Alt — это даёт право на SetForegroundWindow
    ctypes.windll.user32.keybd_event(VK_MENU, 0, 0, 0)
    win32gui.ShowWindow(hwnd, win32con.SW_RESTORE)
    win32gui.SetForegroundWindow(hwnd)
    ctypes.windll.user32.keybd_event(VK_MENU, 0, KEYEVENTF_KEYUP, 0)


def find_game_hwnd(exe_base: str):
    exe_base = exe_base.lower()
    found = []

    def _cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        _, pid = win32process.GetWindowThreadProcessId(hwnd)
        try:
            if exe_base in psutil.Process(pid).name().lower():
                found.append(hwnd)
        except Exception:
            pass

    win32gui.EnumWindows(_cb, None)
    return found


hwnds = find_game_hwnd(GAME_EXE)
print(f"Found windows: {len(hwnds)}")
for h in hwnds:
    print(f"  hwnd={h}  title='{win32gui.GetWindowText(h)}'")

if hwnds:
    force_focus(hwnds[0])
    print("Focus set OK.")
else:
    print("Window not found. Check that the game is running and GAME_EXE matches.")
