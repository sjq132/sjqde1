"""
图像预处理：resize 640x640 + 归一化
输出到 data/processed/images/

路径说明：脚本会自动向上查找项目根目录（包含 data/raw 的目录），
因此无论从哪个目录运行，只要处于项目内即可，无需使用 ../../ 这种脆弱相对路径。
"""

import sys
from pathlib import Path
import cv2
import numpy as np


def find_project_root(start: Path) -> Path:
    """从当前文件所在目录向上查找项目根目录（含 data/raw 的目录）。"""
    cur = start.resolve()
    for parent in [cur, *cur.parents]:
        if (parent / "data" / "raw").exists() or (parent / "data").exists():
            return parent
    return cur  # 找不到则退化为当前目录


# ======== 配置 ========
IMG_SIZE = 640
EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def main():
    # 自动定位：优先用脚本所在目录的祖父目录（data/scripts/ → 项目根）
    script_dir = Path(__file__).parent.resolve()       # .../data/scripts
    project_root = script_dir.parent.parent            # .../ (项目根)
    # 若上述不准，再智能查找
    if not (project_root / "data").exists():
        project_root = find_project_root(script_dir)

    src = project_root / "data" / "raw" / "mvtec_ad"
    dst = project_root / "data" / "processed" / "images"

    print(f"[INFO] 项目根目录: {project_root}")
    print(f"[INFO] 数据源    : {src}")
    print(f"[INFO] 输出目录  : {dst}")

    if not src.exists():
        print(f"[ERROR] 未找到原始数据目录: {src}")
        print("        请先按 data/README.md 下载并解压 MVTec AD 到 data/raw/mvtec_ad/")
        sys.exit(1)

    dst.mkdir(parents=True, exist_ok=True)

    count = 0
    for img_path in sorted(src.rglob("*")):
        if img_path.suffix.lower() not in EXTS:
            continue
        img = cv2.imread(str(img_path))
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)

        rel = img_path.relative_to(src).with_suffix(".jpg")
        out_path = dst / rel
        out_path.parent.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out_path), img)
        count += 1

    print(f"[DONE] 预处理完成：共处理 {count} 张图像 → {dst}")


if __name__ == "__main__":
    main()
