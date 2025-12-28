import sys
import os
import time
import cv2

from env.screen import Screen


def main():
    if len(sys.argv) != 6:
        print(
            "Usage:\n"
            "  python roi_capture.py <name> <x1> <x2> <y1> <y2>\n\n"
            "Example:\n"
            "  python roi_capture.py levelup_ui 20 60 15 45"
        )
        return

    name = sys.argv[1]
    x1, x2, y1, y2 = map(int, sys.argv[2:])

    screen = Screen(monitor_index=2)

    print("Focus the game window...")
    print("Capturing in 2 seconds...")
    time.sleep(2)

    # --- Capture frame (BGR) ---
    frame = screen.grab()  # shape: (H, W, 3)

    h, w, _ = frame.shape
    print(f"Captured frame shape: {w}x{h}")

    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise ValueError(
            f"Invalid ROI: ({x1},{y1}) → ({x2},{y2}) "
            f"for frame {w}x{h}"
        )

    # --- Crop ROI (COLOR) ---
    roi = frame[y1:y2, x1:x2]

    # --- Save ---
    os.makedirs("assets/crops", exist_ok=True)
    filename = f"{name}_x{x1}-{x2}_y{y1}-{y2}.png"
    path = os.path.join("assets", "crops", filename)

    cv2.imwrite(path, roi)

    print(f"Saved: {path}")
    print(f"ROI shape: {roi.shape}")

    # ---------- DEBUG VISUALIZATION ----------
    vis = frame.copy()
    cv2.rectangle(vis, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)

    cv2.imshow(
        "FULL FRAME (ROI marked)",
        cv2.resize(vis, (420, 420), interpolation=cv2.INTER_NEAREST),
    )

    cv2.imshow(
        "ROI (COLOR)",
        cv2.resize(roi, (280, 280), interpolation=cv2.INTER_NEAREST),
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()