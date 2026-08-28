# -*- coding: utf-8 -*-
from pathlib import Path
from PIL import Image
import csv
from datetime import datetime

# ★★★ 直接改这一行，指向你实际的 mvtec_ad 绝对路径 ★★★
SRC = Path(r"C:\mvtec_ad")
ROOT = Path(r"C:\Users\史竣琦\OneDrive\桌面\sjqde1")

DST = ROOT / "data" / "processed" / "images"
IMG_SIZE = 640

DST.mkdir(parents=True, exist_ok=True)
(DST.parent).mkdir(parents=True, exist_ok=True)

EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}

print(f"[1/4] SRC = {SRC}", flush=True)
print(f"[1/4] SRC exists = {SRC.exists()}", flush=True)

if not SRC.exists():
    raise SystemExit("[错误] 请把 mvtec_ad 放到 C:\\mvtec_ad，或修改脚本里的 SRC")

all_imgs = [p for p in SRC.rglob("*") if p.suffix.lower() in EXTS and "ground_truth" not in p.parts]
print(f"[2/4] 原始图片数 = {len(all_imgs)}", flush=True)

count = 0
fail = 0
for p in all_imgs:
    try:
        img = Image.open(p)              # Pillow 读，兼容性好
        img = img.convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        rel = p.relative_to(SRC).with_suffix(".jpg")
        out = DST / rel
        out.parent.mkdir(parents=True, exist_ok=True)
        img.save(out)
        count += 1
    except Exception as e:
        print(f"  失败 {p.relative_to(SRC)}: {e}", flush=True)
        fail += 1

print(f"[3/4] 成功 = {count}, 失败 = {fail}", flush=True)

# 索引
with open(DST.parent / "index.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["split", "category", "filename", "width", "height"])
    for img_path in sorted(DST.rglob("*.jpg")):
        rel = img_path.relative_to(DST)
        parts = rel.parts
        split = parts[0] if len(parts) >= 2 else "unknown"
        category = parts[1] if len(parts) >= 3 else parts[0]
        w.writerow([split, category, str(rel), IMG_SIZE, IMG_SIZE])

# 日志
with open(DST.parent / "run_log.txt", "w", encoding="utf-8") as f:
    f.write(f"运行时间: {datetime.now()}\nSRC: {SRC}\n成功: {count}\n失败: {fail}\n")

print(f"[4/4] 完成 -> {DST}", flush=True)
