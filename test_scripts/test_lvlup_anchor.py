import os
import time
import cv2
import numpy as np

from env.screen import Screen


# =====================================================
# CONFIG — ВАЖНО: ВСЕ КООРДИНАТЫ В ПРОСТРАНСТВЕ 84x84
# =====================================================

MONITOR = 2                       # монитор с игрой
TEMPLATE_PATH = "assets/lvlup_anchor.png"
THRESHOLD = 0.70

# ROI в координатах 84x84 (ВСТАВЬ СВОИ)
# x1 < x2, y1 < y2, ВСЕ в диапазоне [0..84]
X1, X2, Y1, Y2 = 30, 60, 10, 18    # <-- ИЗМЕНИ ПОД СЕБЯ


# =====================================================
# HELPERS
# =====================================================

def crop_roi(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    return frame[y1:y2, x1:x2]


def draw_roi_box(frame: np.ndarray, x1, y1, x2, y2) -> np.ndarray:
    bgr = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
    cv2.rectangle(bgr, (x1, y1), (x2 - 1, y2 - 1), (0, 255, 0), 1)
    return bgr


def check_roi_valid(frame_shape, x1, y1, x2, y2):
    h, w = frame_shape
    if not (0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h):
        raise RuntimeError(
            f"INVALID ROI for frame {frame_shape}: "
            f"x1={x1}, y1={y1}, x2={x2}, y2={y2}"
        )


# =====================================================
# MAIN
# =====================================================

def main():
    # ---------- load template ----------
    tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
    if tpl is None:
        raise RuntimeError(f"Failed to load template: {TEMPLATE_PATH}")

    print(f"Template loaded: {tpl.shape}")

    # ---------- capture ----------
    screen = Screen(monitor_index=MONITOR)

    print("Focus game window on LEVEL-UP screen.")
    print("Capturing in 2 seconds...")
    time.sleep(2)

    frame = screen.grab()  # EXPECTED: (84, 84) grayscale

    if frame.ndim != 2:
        raise RuntimeError(f"Expected grayscale 2D frame, got shape {frame.shape}")

    print(f"Captured frame shape: {frame.shape}")

    # ---------- ROI ----------
    check_roi_valid(frame.shape, X1, Y1, X2, Y2)

    roi = crop_roi(frame, X1, Y1, X2, Y2)

    print(f"ROI coords: x1={X1}, y1={Y1}, x2={X2}, y2={Y2}")
    print(f"ROI shape : {roi.shape}")

    if roi.shape[0] < tpl.shape[0] or roi.shape[1] < tpl.shape[1]:
        raise RuntimeError(
            f"ROI {roi.shape} is smaller than template {tpl.shape}. "
            f"Reduce template or enlarge ROI."
        )

    # ---------- matching ----------
    score = float(cv2.matchTemplate(
        roi,
        tpl,
        cv2.TM_CCOEFF_NORMED
    ).max())

    is_levelup = score >= THRESHOLD

    print("\n===== LEVEL-UP DETECTION =====")
    print(f"Score     : {score:.4f}")
    print(f"Threshold : {THRESHOLD:.2f}")
    print(f"LEVEL-UP  : {is_levelup}")
    print("================================\n")

    # ---------- save debug ----------
    os.makedirs("debug", exist_ok=True)
    cv2.imwrite("debug/levelup_full_frame.png", frame)
    cv2.imwrite("debug/levelup_roi.png", roi)
    cv2.imwrite("debug/levelup_template.png", tpl)

    # ---------- visualize ----------
    frame_box = draw_roi_box(frame, X1, Y1, X2, Y2)

    cv2.imshow(
        "FULL FRAME (ROI BOX)",
        cv2.resize(frame_box, (420, 420), interpolation=cv2.INTER_NEAREST)
    )
    cv2.imshow(
        "ROI USED FOR MATCHING",
        cv2.resize(roi, (280, 280), interpolation=cv2.INTER_NEAREST)
    )
    cv2.imshow(
        "TEMPLATE",
        cv2.resize(tpl, (280, 280), interpolation=cv2.INTER_NEAREST)
    )

    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
