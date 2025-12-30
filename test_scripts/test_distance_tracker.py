import cv2
import time
import numpy as np

from env.screen import Screen
from env.state.threat_field import ThreatFromMask
from env.state.color_mask import ColorMaskExtractor  # твой текущий класс


# ======================
# CONFIG
# ======================
SCALE = 3
MONITOR_INDEX = 2


def main():
    screen = Screen(monitor_index=MONITOR_INDEX)

    print("Focus game window...")
    time.sleep(2)

    frame = screen.grab()
    h, w, _ = frame.shape

    threat = ThreatFromMask((h, w))
    mask_extractor = ColorMaskExtractor()

    # ---------------------------
    # Windows
    # ---------------------------
    cv2.namedWindow("Threat Debug", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Threat Debug", w * SCALE, h * SCALE)

    cv2.namedWindow("Threat Mask", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Threat Mask", w * SCALE, h * SCALE)

    while True:
        frame = screen.grab()

        # ---------------------------
        # 1. Получаем mask
        # ---------------------------
        mask = mask_extractor.extract(frame)

        # ---------------------------
        # 2. Получаем threat vector
        # ---------------------------
        result = threat.update(mask)

        # ---------------------------
        # 3. Визуализация
        # ---------------------------
        vis = frame.copy()

        cx, cy = w // 2, h // 2

        # центр игрока
        cv2.circle(vis, (cx, cy), 6, (0, 255, 0), 2)

        # направление угрозы
        dx = int(result["dx"] * 60)
        dy = int(result["dy"] * 60)

        cv2.arrowedLine(
            vis,
            (cx, cy),
            (cx + dx, cy + dy),
            (0, 0, 255),
            2,
            tipLength=0.3
        )

        # текст
        cv2.putText(
            vis,
            f"С:{result['confidence']:.2f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        cv2.putText(
            vis,
            f"D:{result['distance']:.2f}",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (255, 255, 255),
            2,
        )

        # ---------------------------
        # Mask visualization
        # ---------------------------
        mask_vis = (mask * 255).astype(np.uint8)
        mask_vis = cv2.applyColorMap(mask_vis, cv2.COLORMAP_JET)

        # resize for debug
        vis = cv2.resize(vis, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)
        mask_vis = cv2.resize(mask_vis, (w * SCALE, h * SCALE), interpolation=cv2.INTER_NEAREST)

        cv2.imshow("Threat Debug", vis)
        cv2.imshow("Threat Mask", mask_vis)

        if cv2.waitKey(1) & 0xFF == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
