import cv2
from pathlib import Path

SRC = Path("../../raw/mvtec_ad")
DST = Path("../../processed/images")
IMG_SIZE = 640
DST.mkdir(parents=True, exist_ok=True)


def process(img_path):
    img = cv2.imread(str(img_path))
    if img is None:
        return
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    out = DST / img_path.relative_to(SRC)
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), img)


for p in list(SRC.rglob("*.png")) + list(SRC.rglob("*.jpg")):
    process(p)

print("预处理完成")
