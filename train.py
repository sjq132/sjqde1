"""
MVTec AD 缺陷检测 —— 训练 / 评估 / 导出 入口

数据布局（MVTec AD 官方结构，置于 data/raw/mvtec_ad/）：
    data/raw/mvtec_ad/
        <object>/                     # 15 个对象类别，如 bottle, cable ...
            train/good/*.png          # 仅正常样本用于训练
            test/
                good/*.png            # 正常测试
                <defect_type>/*.png   # 缺陷测试，如 broken_large, scratch 等
            ground_truth/             # 对应缺陷掩码（可选，仅 segmentation 用）
    每类若提供 YOLO 标注：test/<defect_type>/*.txt (class_id cx cy w h)

本脚本职责：
  1) prepare: 将 MVTec AD 官方布局转换为 YOLO 训练所需 train/val 划分 + mvtec_ad.yaml
  2) train:   基于 Ultralytics YOLOv8 做迁移学习（分类 / 检测）
  3) export:  导出 ONNX，供部署 / 推理一致性校验

用法:
    python train.py --prepare          # 首次：划分数据、生成 data.yaml
    python train.py --epochs 50        # 训练（默认 yolov8n.pt，CPU 可跑）
    python train.py --export           # 导出 best.pt -> best.onnx

注：MVTec AD 原始是异常检测（定位+分类），此处提供两种模式：
    --task detect  : 多分类检测（需 YOLO 标注，nc=15）
    --task classify: 整图二分类 good/defect（标注简单，入门首选）
默认 detect；若没有 YOLO 框标注，可先跑 classify 演示闭环。
"""

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "data" / "raw" / "mvtec_ad"
PROC = ROOT / "data" / "processed"
DATASET_YAML = ROOT / "data" / "mvtec_ad.yaml"

# MVTec AD 官方 15 个对象类别（固定顺序 -> class id）
MVTEC_OBJECTS = [
    "bottle", "cable", "capsule", "carpet", "grid", "hazelnut", "leather",
    "metal_nut", "pill", "screw", "tile", "toothbrush", "transistor",
    "wood", "zipper",
]


def cmd_prepare(args):
    """将 MVTec AD 官方布局 -> YOLO 训练布局（按对象类别组织）。

    检测模式：train/<obj>/images+labels, val/<obj>/images+labels
    若某类无 YOLO 标注，则退化为分类模式（整图打 class_id 标签）。
    """
    if not RAW.exists():
        print(f"[FATAL] 未找到原始数据：{RAW}")
        print("请按 MVTec AD 官方结构放置数据后重试。")
        print("下载：https://www.mvtec.com/company/research/datasets/mvtec-ad")
        sys.exit(1)

    for split in ("train", "val"):
        for sub in ("images", "labels"):
            (PROC / split / sub).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    objects = [o for o in MVTEC_OBJECTS if (RAW / o).exists()]
    print(f"[INFO] 发现 {len(objects)} 个对象类别：{objects}")

    stats = {"train": 0, "val": 0}
    for cls_id, obj in enumerate(objects):
        obj_dir = RAW / obj
        # 收集该类所有图片（train/good + test/good + test/缺陷）
        imgs = sorted(obj_dir.glob("**/*.png")) + sorted(obj_dir.glob("**/*.jpg"))
        if not imgs:
            continue
        random.shuffle(imgs)
        n_val = max(1, int(len(imgs) * 0.2))
        val_imgs = imgs[:n_val]
        train_imgs = imgs[n_val:]

        for split, split_imgs in (("train", train_imgs), ("val", val_imgs)):
            for img in split_imgs:
                rel = img.relative_to(obj_dir)
                # 命名：<obj>_<原文件名> 避免跨类重名
                new_name = f"{obj}_{img.stem}{img.suffix}"
                shutil.copy2(img, PROC / split / "images" / new_name)
                # 若有对应 YOLO 标注则一并复制（同名 .txt，class_id 由外部标注决定）
                label_src = img.with_suffix(".txt")
                if label_src.exists():
                    shutil.copy2(label_src, PROC / split / "labels" / f"{obj}_{img.stem}.txt")
                else:
                    # 无标注：生成整图分类伪标注（class_id = cls_id，框=整图）
                    # 便于 classify 模式 / 快速演示，检测模式建议补充真实框
                    with open(PROC / split / "labels" / f"{obj}_{img.stem}.txt", "w") as f:
                        f.write(f"{cls_id} 0.5 0.5 1.0 1.0\n")
            stats[split] += len(split_imgs)

    # 写回 data.yaml（动态，基于实际存在的类别）
    cfg = {
        "path": str(PROC.relative_to(ROOT)),
        "train": "train/images",
        "val": "val/images",
        "nc": len(objects),
        "names": {i: o for i, o in enumerate(objects)},
        "note": "MVTec AD；无标注处使用整图伪标注，建议补充真实 YOLO 框以提升检测精度",
    }
    with open(DATASET_YAML, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)

    print(f"[OK] 划分完成：train={stats['train']}, val={stats['val']}")
    print(f"[OK] 数据集配置：{DATASET_YAML}")


def cmd_train(args):
    from ultralytics import YOLO

    if not DATASET_YAML.exists():
        print("[FATAL] 未找到 data/mvtec_ad.yaml，请先 --prepare")
        sys.exit(1)

    model = YOLO(args.weights)  # 如 yolov8n.pt
    results = model.train(
        data=str(DATASET_YAML),
        task=args.task,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=args.device,
        project=str(ROOT / "runs" / "train"),
        name="mvtec_ad",
        exist_ok=True,
        patience=10,
        verbose=True,
    )
    print(f"[OK] 训练完成，最佳权重：{results.save_dir}/weights/best.pt")

    # 训练后自动在 val 上评估
    metrics = model.val(data=str(DATASET_YAML), device=args.device)
    print(f"[INFO] 评估指标：{metrics}")


def cmd_export(args):
    from ultralytics import YOLO
    ckpt = args.ckpt or str(ROOT / "runs" / "train" / "mvtec_ad" / "weights" / "best.pt")
    model = YOLO(ckpt)
    model.export(format="onnx", dynamic=True)
    print(f"[OK] 已导出 ONNX：{ckpt.replace('.pt', '.onnx')}")


def main():
    parser = argparse.ArgumentParser(description="MVTec AD 缺陷检测训练入口")
    parser.add_argument("--prepare", action="store_true", help="划分数据 + 生成 data.yaml")
    parser.add_argument("--train", action="store_true", help="执行训练")
    parser.add_argument("--export", action="store_true", help="导出 ONNX")
    parser.add_argument("--task", default="detect", choices=["detect", "classify"],
                        help="detect=多分类检测(需框标注); classify=整图分类")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default="cpu", help="cpu 或 0(首块GPU)")
    parser.add_argument("--weights", default="yolov8n.pt", help="预训练权重")
    parser.add_argument("--ckpt", default=None, help="待导出的 best.pt 路径")
    args = parser.parse_args()

    if args.prepare:
        cmd_prepare(args)
    if args.train:
        cmd_train(args)
    if args.export:
        cmd_export(args)
    if not any([args.prepare, args.train, args.export]):
        parser.print_help()


if __name__ == "__main__":
    main()
