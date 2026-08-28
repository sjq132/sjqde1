import cv2
import numpy as np
from pathlib import Path

RAW_DIR = Path("../../raw/mvtec_ad")
OUT_DIR = Path("../../processed/labels")
IMG_SIZE = 640

CLASSES = sorted([
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
])
CLASS2ID = {name: idx for idx, name in enumerate(CLASSES)}
OUT_DIR.mkdir(parents=True, exist_ok=True)


def mask_to_yolo_bbox(mask_path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    cx = (x + w / 2) / IMG_SIZE
    cy = (y + h / 2) / IMG_SIZE
    nw = w / IMG_SIZE
    nh = h / IMG_SIZE
    return cx, cy, nw, nh


for split in ["train", "test", "val"]:
    split_dir = RAW_DIR / split
    if not split_dir.exists():
        continue
    for cls_name in CLASSES:
        gt_dir = split_dir / cls_name / "ground_truth"
        if not gt_dir.exists():
            continue
        for mask_path in gt_dir.glob("*.png"):
            bbox = mask_to_yolo_bbox(mask_path)
            if bbox is None:
                continue
            out_txt = OUT_DIR / mask_path.relative_to(RAW_DIR).with_suffix(".txt")
            out_txt.parent.mkdir(parents=True, exist_ok=True)
            with open(out_txt, "w") as f:
                f.write(f"{CLASS2ID[cls_name]} {bbox[0]:.6f} {bbox[1]:.6f} {bbox[2]:.6f} {bbox[3]:.6f}\n")

print("转换完成")
