"""
MVTec AD -> YOLO 格式转换（辅助脚本）

MVTec AD 官方提供的是 ground_truth 掩码（缺陷定位），本脚本将其转为 YOLO 检测框：
  每个缺陷掩码 -> 外接矩形 -> (class_id, cx, cy, w, h)

类别映射：MVTec AD 15 个对象类别，见 data/mvtec_ad.yaml 的 names。
注意：MVTec AD 原始定位任务是"异常/缺陷分割"，此处为覆盖"目标检测"技术方向，
      将掩码转为外接框做检测标注；若只需分类可直接用 train.py --task classify。

用法:
    python data/scripts/convert_mvtec_to_yolo.py
        --src data/raw/mvtec_ad
        --dst data/processed
"""

import argparse
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent.parent
DATASET_YAML = ROOT / "data" / "mvtec_ad.yaml"

# 对象类别 -> class_id（与 mvtec_ad.yaml 保持一致）
OBJECTS = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]
NAME2ID = {n: i for i, n in enumerate(OBJECTS)}


def mask_to_box(mask_path: Path, img_w: int, img_h: int):
    """读取单通道缺陷掩码，返回外接矩形 (cx, cy, w, h) 归一化。"""
    from PIL import Image
    import numpy as np
    mask = np.array(Image.open(mask_path).convert("L"))
    ys, xs = np.where(mask > 0)
    if len(xs) == 0:
        return None
    x1, y1, x2, y2 = xs.min(), ys.min(), xs.max(), ys.max()
    cx = ((x1 + x2) / 2) / img_w
    cy = ((y1 + y2) / 2) / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def convert(src: Path, dst: Path):
    (dst / "labels").mkdir(parents=True, exist_ok=True)
    for obj in OBJECTS:
        gt_dir = src / obj / "ground_truth"
        if not gt_dir.exists():
            continue
        for mask_path in gt_dir.glob("**/*.png"):
            # 掩码文件名形如：<defect_type>/<name>_mask.png
            img_name = mask_path.stem.replace("_mask", "")
            img_path = mask_path.parent.parent / "test" / mask_path.parent.name / f"{img_name}.png"
            if not img_path.exists():
                continue
            from PIL import Image
            with Image.open(img_path) as im:
                w, h = im.size
            box = mask_to_box(mask_path, w, h)
            if not box:
                continue
            cls_id = NAME2ID[obj]
            label_path = dst / "labels" / f"{obj}_{img_name}.txt"
            with open(label_path, "a") as f:
                f.write(f"{cls_id} {' '.join(f'{v:.6f}' for v in box)}\n")
    print(f"[OK] 转换完成，标签输出至 {dst / 'labels'}")


def main():
    parser = argparse.ArgumentParser(description="MVTec AD 掩码 -> YOLO 框标注")
    parser.add_argument("--src", type=Path, default=ROOT / "data" / "raw" / "mvtec_ad")
    parser.add_argument("--dst", type=Path, default=ROOT / "data" / "processed")
    args = parser.parse_args()
    if not args.src.exists():
        print(f"[FATAL] 未找到 {args.src}")
        return
    convert(args.src, args.dst)


if __name__ == "__main__":
    main()
