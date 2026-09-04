import os
import cv2
import numpy as np
from pathlib import Path

RAW = Path("data/raw/mvtec_ad")
OUT = Path("data/processed/labels")
OUT.mkdir(parents=True, exist_ok=True)

MAP = {
    "broken_large": 3, "broken_small": 3, "broken": 3,
    "crack": 1,
    "scratch": 2, "scratch_head": 2, "scratch_neck": 2,
    "contamination": 4, "metal_contamination": 4, "oil": 4, "liquid": 4,
    "missing_wire": 5, "missing_cable": 5, "bent_wire": 5, "bent": 5,
    "bent_lead": 5, "flip": 5, "manipulated_front": 5, "squeeze": 5,
    "squeezed_teeth": 5, "fold": 5, "misplaced": 5,
    "color": 6, "gray_stroke": 6,
    "glue": 8, "glue_strip": 8, "cut": 8,
    "cut_inner_insulation": 8, "cut_outer_insulation": 8, "cut_lead": 8,
    "damaged_case": 3,
    "poke": 7, "poke_insulation": 7, "hole": 7,
    "cable_swap": 9, "combined": 9, "faulty_imprint": 9, "print": 9,
    "thread": 9, "thread_side": 9, "thread_top": 9, "pill_type": 9,
    "rough": 9, "defective": 9, "fabric_border": 9, "fabric_interior": 9,
    "split_teeth": 3, "broken_teeth": 3,
}

def defect_id(name):
    return MAP.get(name, 9)

count = 0
missing_img = 0

for obj_dir in sorted(RAW.iterdir()):
    if not obj_dir.is_dir():
        continue
    obj = obj_dir.name
    gt = obj_dir / "ground_truth"
    if not gt.exists():
        continue

    for defect_dir in sorted(gt.iterdir()):
        if not defect_dir.is_dir():
            continue
        defect = defect_dir.name

        for mask_path in sorted(defect_dir.glob("*_mask.png")):
            stem = mask_path.stem.replace("_mask", "")

            img = None
            split = None
            for sp in ["train", "test"]:
                for ext in [".jpg", ".jpeg", ".png"]:
                    cand = Path("data/processed/images") / obj / sp / defect / (stem + ext)
                    if cand.exists():
                        img = cand
                        split = sp
                        break
                if img:
                    break

            if img is None:
                missing_img += 1
                continue

            out_dir = OUT / obj / split / defect
            out_dir.mkdir(parents=True, exist_ok=True)
            out_txt = out_dir / (stem + ".txt")

            mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask is None:
                print("mask read fail", mask_path)
                continue

            h, w = mask.shape
            ys, xs = np.where(mask > 127)
            if len(xs) == 0:
                out_txt.write_text("")
                continue

            x1, y1, x2, y2 = int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())
            cx = (x1 + x2) / 2.0 / w
            cy = (y1 + y2) / 2.0 / h
            bw = (x2 - x1) / w
            bh = (y2 - y1) / h
            cls = defect_id(defect)

            out_txt.write_text("{0} {1:.6f} {2:.6f} {3:.6f} {4:.6f}\n".format(cls, cx, cy, bw, bh))
            count += 1

print("labels generated:", count)
print("missing image for mask:", missing_img)