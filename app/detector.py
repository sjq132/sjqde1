import os
import sys
import cv2
import numpy as np
from ultralytics import YOLO

# 缺陷类别配置
DEFECT_YAML = "data/mvtec_ad_defect.yaml"
DEFECT_NAMES = {
    0: "good",
    1: "crack",
    2: "scratch",
    3: "broken",
    4: "contamination",
    5: "missing_deform",
    6: "color",
    7: "hole_poke",
    8: "glue_cut",
    9: "other_defect",
}

# 中文映射
DEFECT_CN = {
    0: "正常/无缺陷",
    1: "裂纹",
    2: "划痕",
    3: "破损",
    4: "污染",
    5: "缺失/变形",
    6: "颜色异常",
    7: "孔洞/戳伤",
    8: "胶切/切割异常",
    9: "其他缺陷",
}

# 英文 key -> 中文
NAME_TO_CN = {v: DEFECT_CN[k] for k, v in DEFECT_NAMES.items()}


def map_label(name):
    """英文缺陷名 -> 中文"""
    return NAME_TO_CN.get(str(name), str(name))


class DefectDetector:
    def __init__(self, model_path=None, conf=0.25):
        if model_path is None:
            model_path = "runs/detect/train-3/weights/best.pt"
        self.model_path = model_path
        self.conf = conf
        self.model = None
        if os.path.exists(model_path):
            self.model = YOLO(model_path)
        else:
            print(f"[WARN] 权重不存在: {model_path}，使用兜底模式")

    def detect(self, image_path):
        """检测单张图片，返回列表 of dict"""
        if self.model is None:
            # 兜底：随机返回一个缺陷类别（演示用）
            import random
            cid = random.randint(1, 9)
            return [{
                "class_id": cid,
                "class_name": DEFECT_NAMES[cid],
                "class_name_cn": DEFECT_CN[cid],
                "confidence": 0.5 + random.random() * 0.4,
                "bbox": [0.1, 0.1, 0.3, 0.3],
            }]

        results = self.model(image_path, conf=self.conf, verbose=False)
        detections = []
        for r in results:
            if r.boxes is None or len(r.boxes) == 0:
                # 无缺陷 -> 正常
                return [{
                    "class_id": 0,
                    "class_name": "good",
                    "class_name_cn": DEFECT_CN[0],
                    "confidence": 1.0,
                    "bbox": [],
                }]
            for box in r.boxes:
                cls_id = int(box.cls.item())
                conf = float(box.conf.item())
                xyxy = box.xyxy.cpu().numpy()[0].tolist()
                detections.append({
                    "class_id": cls_id,
                    "class_name": DEFECT_NAMES.get(cls_id, "unknown"),
                    "class_name_cn": DEFECT_CN.get(cls_id, "未知"),
                    "confidence": conf,
                    "bbox": xyxy,
                })
        return detections

    def draw_detections(self, image_path, detections, output_path=None):
        """在图片上画框，保存或返回"""
        img = cv2.imread(image_path)
        if img is None:
            return None
        h, w = img.shape[:2]
        for d in detections:
            if not d["bbox"]:
                continue
            x1, y1, x2, y2 = d["bbox"]
            # xyxy 可能是归一化的，转像素
            if max(x1, y2) <= 1.0:
                x1, y1, x2, y2 = int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)
            else:
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            label = f"{d['class_name_cn']} {d['confidence']:.2f}"
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, label, (x1, max(y1 - 5, 15)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        if output_path:
            cv2.imwrite(output_path, img)
        return img


# 自测
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python -X utf8 app/detector.py <图片路径>")
        sys.exit(1)
    img_path = sys.argv[1]
    print(f"模式: {'yolo' if os.path.exists('runs/detect/train-3/weights/best.pt') else '兜底'}")
    det = DefectDetector()
    results = det.detect(img_path)
    for r in results:
        print(f"  {r['class_name_cn']} ({r['class_name']}) conf={r['confidence']:.3f}")
def draw_detections(image_path, detections, output_path=None):
    det = DefectDetector()
    return det.draw_detections(image_path, detections, output_path)
