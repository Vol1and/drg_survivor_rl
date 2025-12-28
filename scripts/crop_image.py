import cv2
import os


# =========================
# CONFIG
# =========================

IMAGE_PATH = "assets/lvlup.png"   # исходная картинка
OUT_DIR = "assets/crops"            # куда сохранять

# ROI (y1:y2, x1:x2)



# =========================
# SCRIPT
# =========================

os.makedirs(OUT_DIR, exist_ok=True)

img = cv2.imread(IMAGE_PATH, cv2.IMREAD_GRAYSCALE)
if img is None:
    raise RuntimeError(f"Failed to load image: {IMAGE_PATH}")

crop = img[Y1:Y2, X1:X2]

out_name = f"image_{X1}_{Y1}.png"
out_path = os.path.join(OUT_DIR, out_name)

cv2.imwrite(out_path, crop)

print(f"Saved crop to: {out_path}")
print(f"Crop shape: {crop.shape}")
