import cv2
from pathlib import Path

# 自动定位项目根目录（脚本位于 data/scripts/，向上两级即 sjqde1/）
SCRIPT_DIR = Path(__file__).resolve().parent
ROOT = SCRIPT_DIR.parent.parent

SRC = ROOT / "data" / "raw" / "mvtec_ad"
DST = ROOT / "data" / "processed" / "images"
IMG_SIZE = 640


def main():
    if not SRC.exists():
        print(f"[错误] 找不到数据集：{SRC}")
        print("请确认 MVTec AD 已解压到 data/raw/mvtec_ad/")
        return

    DST.mkdir(parents=True, exist_ok=True)
    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    count = 0

    for p in SRC.rglob("*"):
        if p.suffix.lower() in exts and "ground_truth" not in p.parts:
            img = cv2.imread(str(p))
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            rel = p.relative_to(SRC).with_suffix(".jpg")
            out = DST / rel
            out.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(out), img)
            count += 1

    print(f"[完成] 预处理图片数：{count} -> {DST}")


if __name__ == "__main__":
    main()
