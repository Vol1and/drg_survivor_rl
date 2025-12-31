import cv2
import numpy as np
import time

from env.state.game_state import GameStateReader


# ======================
# CONFIG
# ======================
MAP_SIZE = 500  # px
MAP_RANGE = 100.0  # game units
CENTER = MAP_SIZE // 2


def world_to_screen(x, z):
    """Convert world coords to image coords"""
    sx = int((x / MAP_RANGE) * MAP_SIZE)
    sz = int((z / MAP_RANGE) * MAP_SIZE)
    return sx, sz


def main():
    reader = GameStateReader()

    print("Waiting for data...")
    while reader.get() is None:
        time.sleep(0.1)

    print("Data stream started")

    while True:
        state = reader.get()
        if not state:
            continue

        x = state["pos"]["x"]
        z = state["pos"]["z"]

        grounded = state.get("grounded", False)

        # normalized
        nx = x / MAP_RANGE
        nz = z / MAP_RANGE

        # edge distance
        edge_dist = min(nx, 1 - nx, nz, 1 - nz)

        # direction to center
        dx = 0.5 - nx
        dz = 0.5 - nz
        norm = np.sqrt(dx * dx + dz * dz) + 1e-6
        dx /= norm
        dz /= norm

        # danger (same logic)
        danger = 0.0
        if not grounded and edge_dist < 0.25:
            danger = (0.25 - edge_dist) * 0.6
        if grounded:
            danger += 0.7
        danger = np.clip(danger, 0.0, 1.0)

        # -------------------------
        # DRAW
        # -------------------------
        canvas = np.zeros((MAP_SIZE, MAP_SIZE, 3), dtype=np.uint8)

        # player position
        px, pz = world_to_screen(x, z)
        cv2.circle(canvas, (px, pz), 6, (0, 255, 0), -1)

        # direction vector
        end_x = int(px + dx * 50)
        end_z = int(pz + dz * 50)
        cv2.arrowedLine(canvas, (px, pz), (end_x, end_z), (255, 0, 0), 2)

        # danger overlay
        danger_color = (0, int(255 * (1 - danger)), int(255 * danger))
        cv2.rectangle(canvas, (0, 0), (MAP_SIZE, 30), danger_color, -1)

        # text
        cv2.putText(
            canvas,
            f"edge_dist={edge_dist:.2f} danger={danger:.2f}",
            (10, 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow("EDGE DEBUG", canvas)

        if cv2.waitKey(1) == 27:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
