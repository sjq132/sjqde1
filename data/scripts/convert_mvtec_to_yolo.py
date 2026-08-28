# -*- coding: utf-8 -*-
import cv2
from pathlib import Path
from collections import Counter

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent

RAW = ROOT / "data" / "raw" / "mvtec_ad"
IMG_DIR = ROOT / "data" / "processed" / "images"
LAB_DIR = ROOT / "data" / "processed" / "labels"

CLASSES = sorted([
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
])
CLASS2ID = {name: idx for idx, name in enumerate(CLASSES)}


def log(msg):
    print(msg, flush=True)


def mask_to_bbox(mask_path: Path):
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return None
    _, bin_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(bin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return None
    cnt = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(cnt)
    return (x + w / 2) / 640, (y + h / 2) / 640, w / 640, h / 640


def find_ground_truth(split_dir: Path, cls_name: str):
    d = split_dir / cls_name / "ground_truth"
    if d.exists():
        return d
    cand = split_dir / cls_name
    if cand.exists():
        for sub in cand.rglob("ground_truth"):
            if sub.is_dir():
                return sub
    return None


def main():
    log(f"RAW exists: {RAW.exists()}")
    log(f"IMG_DIR exists: {IMG_DIR.exists()}")

    if not RAW.exists():
        log("[错误] 找不到数据集")
        return
    if not IMG_DIR.exists():
        log("[错误] 请先跑 preprocess.py")
        return

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    count = 0
    fail = 0

    for split in ["train", "test", "val"]:
        split_dir = RAW / split
        if not split_dir.exists():
            continue
        for cls_name in CLASSES:
            gt_dir = find_ground_truth(split_dir, cls_name)
            if gt_dir is None:
                continue
            cls_id = CLASS2ID[cls_name]
            for mask_path in gt_dir.glob("*.png"):
                bbox = mask_to_bbox(mask_path)
                if bbox is None:
                    fail += 1
                    continue
                out_txt = LAB_DIR / mask_path.with_suffix(".txt").name
                cx, cy, nw, nh = bbox
                with open(out_txt, "w", encoding="utf-8") as f:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                count += 1

    log(f"[完成] 生成标注数: {count}, 无标注/失败: {fail}")

    # 类别统计
    c = Counter()
    for txt in LAB_DIR.rglob("*.txt"):
        for line in open(txt, encoding="utf-8"):
            parts = line.strip().split()
            if parts:
                c[parts[0]] += 1

    stat_path = LAB_DIR / "class_count.txt"
    with open(stat_path, "w", encoding="utf-8") as f:
        f.write("class_id,class_name,count\n")
        for cls_id in sorted(c.keys(), key=int):
            f.write(f"{cls_id},{CLASSES[int(cls_id)]},{c[cls_id]}\n")
    log(f"类别统计: {stat_path}")


if __name__ == "__main__":
    main()
