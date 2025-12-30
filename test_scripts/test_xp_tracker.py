import time
import cv2
from env.screen import Screen
from env.state.xp_tracker import XPTracker


def main():
    screen = Screen(monitor_index=2)
    xp_tracker = XPTracker()

    print("Focus game window...")
    time.sleep(2)

    while True:
        frame = screen.grab()

        xp_delta = xp_tracker.get_xp_delta(frame)
        xp = xp_tracker.get_current_xp()

        vis = xp_tracker.debug_draw(frame)

        print(f"XP: {xp:.3f} | ΔXP: {xp_delta:+.4f}")

        cv2.imshow("XP Tracker Debug", vis)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
