import cv2
from pathlib import Path

# 自动定位项目根目录
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent

RAW = ROOT / "data" / "raw" / "mvtec_ad"
LAB_DIR = ROOT / "data" / "processed" / "labels"

CLASSES = sorted([
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut",
    "leather", "metal_nut", "pill", "screw", "tile", "toothbrush",
    "transistor", "wood", "zipper"
])
CLASS2ID = {name: idx for idx, name in enumerate(CLASSES)}


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
    """兼容 MVTec 两种目录结构"""
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
    if not RAW.exists():
        print(f"[错误] 找不到数据集：{RAW}")
        return
    if not (ROOT / "data" / "processed" / "images").exists():
        print("[错误] 请先跑 preprocess.py")
        return

    LAB_DIR.mkdir(parents=True, exist_ok=True)
    count = 0

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
                    continue
                out_txt = LAB_DIR / mask_path.with_suffix(".txt").name
                cx, cy, nw, nh = bbox
                with open(out_txt, "w") as f:
                    f.write(f"{cls_id} {cx:.6f} {cy:.6f} {nw:.6f} {nh:.6f}\n")
                count += 1

    print(f"[完成] 生成 YOLO 标注数：{count} -> {LAB_DIR}")


if __name__ == "__main__":
    main()
