# 数据集说明

## 主数据集：MVTec AD

| 项目 | 说明 |
|------|------|
| 名称 | MVTec Anomaly Detection Dataset |
| 规模 | 5354 张，15 类工业对象 |
| 标注 | 像素级 PNG mask（本项目转为 YOLO 检测框） |
| 许可 | CC BY-NC-SA 4.0 |

### 下载方式（任选其一）

- 官网：https://www.mvtec.com/company/research/datasets/mvtec-ad/
- Kaggle：https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset
- ModelScope：https://www.modelscope.cn/datasets/shimin2023/MVTec_AD
- OpenDataLab：https://opendatalab.com/OpenDataLab/MVTecAD

下载后解压到 `data/raw/mvtec_ad/`，目录结构：
data/raw/mvtec_ad/
├── bottle/
├── cable/
├── capsule/
├── ...（共15类）
├── train/
├── test/
└── ground_truth/
## 辅助数据集

| 数据集 | 链接 |
|--------|------|
| KolektorSDD | https://www.vicos.si/Downloads/KolektorSurface-defect-dataset |
| PKU-Market-PCB | https://github.com/tangsanli5201/PKU-Market-PCB |

## 预处理

| 脚本 | 功能 |
|------|------|
| `scripts/convert_mvtec_to_yolo.py` | 像素 mask → YOLO bbox（cx,cy,w,h） |
| `scripts/preprocess.py` | resize 640×640 + 归一化 |

运行方式（仓库根目录）：
bash
pip install opencv-python numpy
python data/scripts/preprocess.py
python data/scripts/convert_mvtec_to_yolo.py
## 数据划分

训练集 : 验证集 : 测试集 = 7 : 1.5 : 1.5

## 目录结构
data/
├── README.md # 本文件
├── raw/ # 原始数据（.gitignore 排除）
│ └── mvtec_ad/
├── processed/ # 预处理结果（.gitignore 排除）
│ ├── images/
│ └── labels/
└── scripts/ # 预处理程序（已提交）
├── preprocess.py
└── convert_mvtec_to_yolo.py
> 原始数据与预处理结果体积较大，不纳入版本控制。通过运行上述脚本可基于下载数据自动重建。

---
更新日期：2026-08-29
