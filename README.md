# 基于 YOLOv8 的通用工业零部件多实例缺陷定位与分类系统

## 课程设计 · 制造智能技术

本项目为制造智能技术课程设计作业。针对制造现场人工质检效率低、主观性强、易疲劳漏检等问题，构建一个基于深度学习的**通用工业零部件**表面缺陷自动检测原型系统，实现对常见缺陷（划痕、裂纹、凹陷、异物、缺失部件等）的**多实例定位与分类**（instance-level），区别于单一材料的"表面纹理异常检测"。

---

## 仓库文件说明

| 文件 | 阶段 | 说明 |
|------|------|------|
| `README.md` | — | 本文件，项目与仓库总览 |
| `学习笔记.md` | 第一阶段 | AI 工具学习、模型与 harness 选型、Git 原理、选题调研 |
| `选题说明.md` | 第二阶段 | 拟定题目、完成目标、涉及的上学期课程技术方向（智能优化 + Python + 图像处理 + 机器学习）、差异化说明 |
| `方案设计.md` | 第二阶段 | 功能需求、方案论证、技术栈、数据集、技术路线、计划安排、参考文献 |
| `data/README.md` | 第三阶段 | **数据集来源、下载方式（多镜像链接）、预处理与目录组织说明** |
| `data/scripts/` | 第三阶段 | **预处理脚本（preprocess.py）与标注转换脚本（convert_mvtec_to_yolo.py）** |
| `prompt/` | 第三阶段起 | **AI 工具提示词追溯记录，每阶段同步更新** |

---

## 技术主线

- **研究对象**：通用工业零部件（多类物件，非单一材料/钢材）
- **任务范式**：目标检测（定位 + 分类），而非仅整图分类
- **核心模型**：YOLOv8（检测）+ ResNet-18（二次验证）+ Qwen2.5-VL（报告生成扩展）
- **系统形态**：Python 本地工具（OpenCV 标注图 + JSON/CSV 存储 + Matplotlib 统计图表 + 终端报告），**不含 Web**
- **主数据集**：MVTec AD（多类工业零部件/纹理基准，详见 `data/README.md`）

---

## 数据集与数据准备（第三阶段）

> 详细内容见 [`data/README.md`](./data/README.md)，此处仅作索引。

- **主数据集**：MVTec AD（15 类，5354 张，CC BY-NC-SA 4.0）
  - 来源：https://www.mvtec.com/company/research/datasets/mvtec-ad/
  - 镜像：Kaggle / ModelScope / OpenDataLab（任选其一，链接见 `data/README.md`）
- **标注转换**：将 MVTec 像素级 mask 重构为 YOLO 检测框，脚本 `data/scripts/convert_mvtec_to_yolo.py`
- **预处理**：resize 640×640 + 归一化，脚本 `data/scripts/preprocess.py`
- **输出**：预处理后数据与标注索引存放于 `data/processed/`（因体积大，不入库）
- **AI 提示词记录**：见 `prompt/` 目录，每阶段同步更新

### 本地运行方式

在**项目根目录**下执行：

```bash
pip install opencv-python numpy
python data/scripts/preprocess.py
python data/scripts/convert_mvtec_to_yolo.py
```

> ⚠️ 路径约定：脚本会自动定位项目根目录，统一放在 `data/scripts/` 下运行，**请勿嵌套为 `data/data/scripts/`**。

---

## 详细开发：Web 检测系统（第四阶段）

完整的 B/S 架构可运行 demo，业务闭环：**上传图片 → YOLOv8 推理 → 缺陷定位/分类 → 入库 → 历史查询/统计**。

```
phase3/
├── train.py              # YOLOv8 训练入口（prepare / train / export）
├── requirements.txt
├── data/mvtec_ad.yaml    # 数据集配置（15 类，单一数据源）
├── app/
│   ├── app.py            # Flask 后端：/api/detect, /api/history, /api/stats
│   ├── detector.py       # 算法模块：YOLOv8 + 传统图像兜底
│   ├── database.py       # SQLite 数据层
│   ├── templates/        # 前端三页：index / history / stats
│   └── static/style.css
└── tests/                # 自动化测试（unittest + 真实 HTTP 冒烟）
```

**MVTec AD 数据布局**（置于 `data/raw/mvtec_ad/`，与方案设计一致）：

```
data/raw/mvtec_ad/
  <object>/                # 15 类：bottle, cable, capsule ...
    train/good/*.png
    test/good/*.png
    test/<defect_type>/*.png
    ground_truth/<defect_type>/*.png   # 缺陷掩码（可选，用于转 YOLO 框）
```

**运行方式**：

```bash
cd sjqde1
pip install -r requirements.txt

# 1) 准备数据 + 训练（有 GPU 把 --device 0）
python train.py --prepare
python train.py --epochs 50 --device cpu
python train.py --export                 # 导出 ONNX（可选）

# 2) 启动 Web（无需训练也能跑，走兜底模式演示闭环）
python -m app.app
# 浏览器打开 http://127.0.0.1:5000

# 3) 自动化测试
python -m unittest tests.test_system -v
```

**技术方向覆盖（任务书要求 ≥3）**：

| 技术方向 | 在本系统中的实现 |
|---------|----------------|
| 计算机视觉 / 目标检测 | YOLOv8 缺陷定位 + 分类 |
| 传统图像处理 | 兜底模式：形态学 + 统计特征 |
| 数据增强 / 预处理 | resize 640 + 归一化 + 标注重构 |
| Web 全栈 / 前后端分离 | Flask REST API + Bootstrap 前端 |
| 数据库 | SQLite 检测记录存 + 查 + 统计 |

---

## 核心差异化

本方案与"工业产品表面缺陷检测"类选题的区别：

| 维度 | 本选题 |
|------|--------|
| 检测粒度 | **实例级（instance-level）** 多实例定位 + 分类 |
| 对象范围 | **15 类通用工业零部件**（MVTec AD） |
| 核心模型 | **YOLOv8 单阶段检测**（速度 + 精度 + 定位） |
| 数据重构 | 将像素级 mask 重构为 YOLO 检测框 |

---

> 更新日期：2026-08-29
