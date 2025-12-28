import time
import cv2
from env.screen import Screen
from env.state.hp_tracker import HPTracker


def main():
    screen = Screen(monitor_index=2)
    hp_tracker = HPTracker()

    print("Focus game window...")
    time.sleep(2)

    while True:
        frame = screen.grab()

        hp_delta = hp_tracker.get_hp_delta(frame)
        hp = hp_tracker.get_current_hp()

        vis = hp_tracker.debug_draw(frame)

        print(f"HP: {hp:.3f} | ΔHP: {hp_delta:+.4f}")

        cv2.imshow("HP Tracker Debug", vis)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
