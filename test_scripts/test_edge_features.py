import cv2
import time
import numpy as np

from env.screen import Screen
from env.state.level_border_mask import LevelBorderDetector
from env.state.edge_extractor import EdgeFeatureExtractor


# ===============================
# CONFIG
# ===============================
MONITOR_INDEX = 2
SCALE = 3
WINDOW_NAME = "Edge Feature Debug"


# ===============================
# UTILS
# ===============================
def scale(img, factor):
    h, w = img.shape[:2]
    return cv2.resize(img, (w * factor, h * factor), interpolation=cv2.INTER_NEAREST)


def draw_vector(img, origin, vec, scale=80, color=(0, 255, 0)):
    ox, oy = origin
    dx, dy = vec
    end = (
        int(ox + dx * scale),
        int(oy + dy * scale),
    )
    cv2.arrowedLine(img, (ox, oy), end, color, 2, tipLength=0.3)


# ===============================
# MAIN
# ===============================
def main():
    screen = Screen(MONITOR_INDEX)

    detector = LevelBorderDetector(
    )

    extractor = EdgeFeatureExtractor(
        detector
    )

    print("Focus game window...")
    time.sleep(2)

    while True:
        frame = screen.grab()
        h, w = frame.shape[:2]

        features = extractor.extract(frame)
        edge_dist, dx, dy, conf = features

        # -----------------------------
        # Visualization
        # -----------------------------
        mask = detector.extract(frame)

        overlay = frame.copy()
        overlay[mask > 0] = [0, 0, 255]

        cx, cy = w // 2, h // 2
        draw_vector(
            overlay,
            (cx, cy),
            (dx, dy),
            scale=120,
            color=(0, 255, 0),
        )

        # info panel
        info = np.zeros((120, w, 3), dtype=np.uint8)
        cv2.putText(info, f"edge_dist: {edge_dist:.3f}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info, f"edge_dx: {dx:.3f}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info, f"edge_dy: {dy:.3f}", (10, 90),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(info, f"confidence: {conf:.3f}", (300, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        combined = np.vstack([
            overlay,
            info
        ])

        combined = scale(combined, SCALE)

        cv2.imshow(WINDOW_NAME, combined)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
