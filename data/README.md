# 数据集说明

本文件说明「基于 YOLOv8 的通用工业零部件多实例缺陷定位与分类系统」所使用的数据来源、获取方式、预处理与目录组织规范。

---

## 一、主数据集：MVTec AD（公开数据集）

本项目以 **MVTec Anomaly Detection Dataset（MVTec AD）** 为核心数据集，覆盖 **15 类通用工业对象与纹理**，用于多实例缺陷定位与分类。

| 项目 | 说明 |
|------|------|
| 数据集名称 | MVTec Anomaly Detection Dataset（MVTec AD） |
| 类别数 | 15 类工业对象与纹理 |
| 数据规模 | 5354 张高分辨率彩色图像 |
| 标注类型 | 提供像素级异常标注（PNG mask）；本项目将其重构为 YOLO 格式检测框 |
| 官方来源 | https://www.mvtec.com/company/research/datasets/mvtec-ad/ |
| 许可协议 | CC BY-NC-SA 4.0（非商用，课程设计合规） |

### 1.1 下载方式（任选其一）

**方式一：官网下载（首选）**
1. 打开 https://www.mvtec.com/company/research/datasets/mvtec-ad/
2. 点击 "Download dataset"，填写表单（姓名、邮箱、用途）
3. 提交后获取下载链接（或邮件发送），下载压缩包（约 4.5 GB）

**方式二：Kaggle 镜像（无需填表单，推荐）**
- https://www.kaggle.com/datasets/avdvhh/mvtec-defect-detection-dataset
- 登录 Kaggle 后直接下载

**方式三：ModelScope 镜像（国内访问较快）**
- https://www.modelscope.cn/datasets/shimin2023/MVTec_AD
- 使用 git 克隆：
  ```bash
  git clone https://www.modelscope.cn/datasets/shimin2023/MVTec_AD.git
  ```

**方式四：OpenDataLab 镜像**
- https://opendatalab.com/OpenDataLab/MVTecAD

> 以上镜像均为同一份 MVTec AD 数据，CC BY-NC-SA 4.0 许可，任选其一即可。

### 1.2 本项目的用途

- 作为目标检测主训练/测试集，覆盖**多类通用工业零部件**（而非单一材料）
- 将原**像素级异常标注（mask）重构为 YOLO 格式检测框（bbox）**，实现 instance-level 定位与分类
- 详见预处理脚本 `scripts/convert_mvtec_to_yolo.py`

---

## 二、辅助数据集（可选）

| 数据集 | 规模 | 来源链接 | 用途 |
|--------|------|---------|------|
| KolektorSDD | 399 张真实工业换向器表面缺陷图 | https://www.vicos.si/Downloads/KolektorSurface-defect-dataset | 补充真实工业场景验证 |
| PKU-Market-PCB | 1386 张 PCB 缺陷图，6 类缺陷 | https://github.com/tangsanli5201/PKU-Market-PCB | 扩展电路板类零部件检测 |

---

## 三、自建测试集（可选补充）

为验证系统泛化能力，可使用手机拍摄 + 网络采集的通用零部件图片构建小型自建测试集。

- **规模**：约 50–100 张
- **存放位置**：`data/raw/custom_test/`
- **要求**：仅使用**可公开**的通用物件图片，**不含私有/敏感信息**
- **开源（如需）**：上传至 Hugging Face 或 ModelScope 并注明"自建数据集"

---

## 四、目录结构

下载并解压后，按如下结构组织（**仅需在本地方便运行，无需上传至 GitHub**）：

```
data/
├── README.md                  # 本文件
├── .gitignore                 # 排除 raw/ 与 processed/ 大数据
├── raw/                       # 原始数据（已加入 .gitignore，不上传）
│   └── mvtec_ad/              # MVTec AD 解压后的根目录
│       ├── bottle/
│       ├── cable/
│       ├── capsule/
│       ├── carpet/
│       ├── grid/
│       ├── hazelnut/
│       ├── leather/
│       ├── metal_nut/
│       ├── pill/
│       ├── screw/
│       ├── tile/
│       ├── toothbrush/
│       ├── transistor/
│       ├── wood/
│       └── zipper/
├── processed/                 # 预处理后数据（已加入 .gitignore，不上传）
│   ├── images/                # resize 后的图像
│   │   ├── train/
│   │   ├── val/
│   │   └── test/
│   └── labels/                # YOLO 格式 txt 标注
│       ├── train/
│       ├── val/
│       └── test/
└── scripts/                   # 预处理与标注转换脚本（提交至仓库）
    ├── preprocess.py
    └── convert_mvtec_to_yolo.py
```

> 每个类别目录下通常包含 `train/`、`test/`、`ground_truth/` 三个子目录。

---

## 五、数据预处理流程

### 5.1 标注格式转换（mask → YOLO bbox）

MVTec AD 原始标注为**像素级 mask（PNG）**，本项目将其转换为 **YOLO 检测框格式（txt）**：

- 脚本：`scripts/convert_mvtec_to_yolo.py`
- 方法：读取每个 mask → 提取最大轮廓 → 计算最小外接矩形 → 归一化为 YOLO 坐标（cx, cy, w, h）
- 输出：每张图对应一个 `.txt` 文件，每行格式为 `class_id cx cy w h`

### 5.2 图像预处理

- 脚本：`scripts/preprocess.py`
- 操作：统一 resize 至 **640×640** → 归一化 → 保存为 uint8 图像

### 5.3 运行方式

在仓库**根目录**（即包含 `data/` 的目录）下执行：

```bash
pip install opencv-python numpy
python data/scripts/preprocess.py
python data/scripts/convert_mvtec_to_yolo.py
```

> ⚠️ 路径说明：脚本使用「项目根目录」作为基准，会自动向上查找 `data/raw/mvtec_ad`，因此无论从何处运行，只要处于项目根目录即可。**请勿把脚本放在 `data/data/scripts/` 这种嵌套目录**，统一放在 `data/scripts/`。

---

## 六、数据划分

| 划分 | 比例 | 说明 |
|------|------|------|
| 训练集（train） | 70% | 用于模型训练 |
| 验证集（val） | 15% | 用于调参、早停 |
| 测试集（test） | 15% | 用于最终评估 |

> 划分比例可在预处理脚本中调整。

---

## 七、许可说明

- MVTec AD：CC BY-NC-SA 4.0（非商用，课程设计属合理使用）
- KolektorSDD：CC BY-NC-SA 4.0
- PKU-Market-PCB：仅限学术研究
- 本项目仅用于教学目的，不涉及商业用途

---

> 更新日期：2026-08-29
