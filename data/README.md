# 数据集说明

## 主数据集：MVTec AD

| 项目 | 说明 |
|------|------|
| 名称 | MVTec Anomaly Detection Dataset |
| 规模 | 5354 张，15 类工业对象 |
| 标注 | 像素级 PNG mask（本项目转为 YOLO 检测框） |
| 许可 | CC BY-NC-SA 4.0 |

### 下载方式（任选其一，确保链接有效）

- 官网：https://www.mvtec.com/company/research/datasets/mvtec-ad/
- Kaggle：https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset
- ModelScope：https://www.modelscope.cn/datasets/shimin2023/MVTec_AD
- OpenDataLab：https://opendatalab.com/OpenDataLab/MVTecAD

下载后解压到 `data/raw/mvtec_ad/`，应看到 15 个类别文件夹（bottle/cable/capsule...）。

## 辅助数据集

| 数据集 | 链接 |
|--------|------|
| KolektorSDD | https://www.vicos.si/Downloads/KolektorSurface-defect-dataset |
| PKU-Market-PCB | https://github.com/tangsanli5201/PKU-Market-PCB |

## 预处理

| 脚本 | 功能 |
|------|------|
| `scripts/preprocess.py` | resize 640×640 + 归一化 |
| `scripts/convert_mvtec_to_yolo.py` | mask → YOLO bbox |

运行（在仓库根目录执行）：
