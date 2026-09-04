"""
缺陷类别名映射：将数据集原始类别名（MVTec AD 对象名）映射为工业缺陷语义名。
供 detector / database / 前端展示统一调用。
"""
DEFECT_LABEL_MAP = {
    # 真实缺陷类型（模型若输出这些则直接映射）
    "crack": "裂纹缺陷",
    "scratch": "划痕缺陷",
    # MVTec AD 15 类 → 工业缺陷语义（按该对象最常见缺陷类型命名）
    "bottle": "瓶身缺损",
    "cable": "线缆结构异常",
    "capsule": "胶囊表面瑕疵",
    "carpet": "地毯纹理缺陷",
    "grid": "网格结构缺损",
    "hazelnut": "坚果表面瑕疵",
    "leather": "皮革表面缺陷",
    "metal_nut": "金属螺母缺损",
    "pill": "药片外观异常",
    "screw": "螺丝螺纹缺损",
    "tile": "瓷砖表面缺陷",
    "toothbrush": "刷毛缺失/变形",
    "transistor": "晶体管引脚异常",
    "wood": "木材表面瑕疵",
    "zipper": "拉链齿缺损",
    # 兜底
    "basement": "背景区域异常",
}


def map_label(name):
    """未知类别原样返回，避免丢失信息。"""
    if not name:
        return name
    return DEFECT_LABEL_MAP.get(str(name), name)