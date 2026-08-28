# 数据集说明

本项目的全部数据准备工作说明如下。

## 1. 主数据集：MVTec AD（公开数据集）

| 项目 | 说明 |
|------|------|
| 数据集名称 | MVTec Anomaly Detection Dataset (MVTec AD) |
| 类别数 | 15 类工业对象与纹理 |
| 规模 | 5354 张高分辨率彩色图像 |
| 标注 | 提供像素级异常标注（PNG mask） |
| 来源链接 | https://www.mvtec.com/company/research/datasets/mvtec-ad/ |
| 下载方式 | 官网注册后免费下载（需同意 CC BY-NC-SA 4.0 许可） |
| 许可协议 | CC BY-NC-SA 4.0（非商用，课程设计合规） |

### 本项目的用途
- 作为目标检测主训练/测试集，覆盖多类通用工业零部件
- 将原像素级异常标注（mask）重构为 YOLO 格式检测框（bbox）

### 下载与组织
由于数据集体积较大（约 4.5GB），不直接上传至仓库。请按以下步骤准备：
