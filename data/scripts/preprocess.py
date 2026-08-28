# -*- coding: utf-8 -*-
import os
import cv2
import csv
from pathlib import Path
from datetime import datetime

SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent

SRC = ROOT / "data" / "raw" / "mvtec_ad"
DST = ROOT / "data" / "processed" / "images"
IMG_SIZE = 640

LOG_PATH = ROOT / "data" / "processed" / "run_log.txt"
INDEX_PATH = ROOT / "data" / "processed" / "index.csv"
SAMPLE_PATH = DST / "sample_annotated.jpg"

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def log(msg):
    print(msg, flush=True)


def main():
    DST.mkdir(parents=True, exist_ok=True)
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

    log(f"[1/4] SRC = {SRC}")
    log(f"[1/4] SRC exists: {SRC.exists()}")

    if not SRC.exists():
        log("[错误] 数据集目录不存在，请检查 data/raw/mvtec_ad")
        return

    # 统计原始图片数量（不含 ground_truth）
    all_imgs = [
        p for p in SRC.rglob("*")
        if p.suffix.lower() in EXTS and "ground_truth" not in p.parts
    ]
    log(f"[2/4] 发现原始图片数: {len(all_imgs)}")

    count = 0
    fail = 0
    fail_list = []

    for p in all_imgs:
        try:
            img = cv2.imdecode(
                cv2.imread(str(p), cv2.IMREAD_UNCHANGED), cv2.IMREAD_COLOR
            )
            if img is None:
                # 文件可能损坏或 OneDrive 占位
                if os.path.getsize(p) < 100:
                    log(f"  跳过(文件过小): {p.relative_to(SRC)} size={os.path.getsize(p)}")
                else:
                    log(f"  读取失败: {p.relative_to(SRC)} size={os.path.getsize(p)}")
                fail += 1
                fail_list.append(str(p.relative_to(SRC)))
                continue

            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            rel = p.relative_to(SRC).with_suffix(".jpg")
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)

            ok = cv2.imwrite(str(out), img)
            if not ok:
                log(f"  写入失败: {out}")
                fail += 1
                fail_list.append(str(rel))
                continue

            count += 1
        except Exception as e:
            log(f"  异常 {p.relative_to(SRC)}: {e}")
            fail += 1
            fail_list.append(str(p.relative_to(SRC)))

    log(f"[3/4] 成功预处理: {count} 张, 失败: {fail} 张")

    # 写索引
    with open(INDEX_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["split", "category", "filename", "width", "height"])
        for img_path in sorted(DST.rglob("*.jpg")):
            rel = img_path.relative_to(DST)
            parts = rel.parts
            split = parts[0] if len(parts) >= 2 else "unknown"
            category = parts[1] if len(parts) >= 3 else parts[0]
            w.writerow([split, category, str(rel), IMG_SIZE, IMG_SIZE])
    log(f"  索引: {INDEX_PATH}")

    # 示例图
    sample_imgs = list(DST.rglob("*.jpg"))
    if sample_imgs:
        demo = cv2.imread(str(sample_imgs[0]))
        if demo is not None:
            cv2.putText(demo, "preprocessed 640x640", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
            cv2.imwrite(str(SAMPLE_PATH), demo)
            log(f"  示例图: {SAMPLE_PATH}")

    # 运行日志
    with open(LOG_PATH, "w", encoding="utf-8") as f:
        f.write(f"运行时间: {datetime.now()}\n")
        f.write(f"数据集: MVTec AD ({SRC})\n")
        f.write(f"原始图片数: {len(all_imgs)}\n")
        f.write(f"预处理成功: {count}\n")
        f.write(f"预处理失败: {fail}\n")
        f.write(f"输出尺寸: {IMG_SIZE}x{IMG_SIZE}\n")
        if fail_list:
            f.write("\n失败文件列表(前20):\n")
            f.write("\n".join(fail_list[:20]))
    log(f"[4/4] 日志: {LOG_PATH}")


if __name__ == "__main__":
    main()
