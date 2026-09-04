"""
数据库模块（data layer）：SQLite 存储每次检测记录。

表结构 detection_records：
    id, filename, detect_time, class_name, confidence, bbox, image_size, mode
"""
import sqlite3
from pathlib import Path
from datetime import datetime
import json

from app.defect_labels import map_label   # ← 新增：读出时映射为中文缺陷名

DB_PATH = Path(__file__).resolve().parent / "detection.db"


def get_conn():
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """建表（首次运行时调用）。"""
    with get_conn() as conn:
        conn.execute("""
            CREATE TABLE IF NOT EXISTS detection_records (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                filename    TEXT NOT NULL,
                detect_time TEXT NOT NULL,
                class_name  TEXT,
                confidence  REAL,
                bbox        TEXT,          -- JSON 存储 [x1,y1,x2,y2]
                image_size  TEXT,          -- JSON [w,h]
                mode        TEXT           -- yolov8 / fallback
            )
        """)
        conn.commit()


def insert_records(filename: str, detections: dict) -> int:
    """写入一条检测的所有缺陷框，返回记录条数。"""
    init_db()
    n = 0
    with get_conn() as conn:
        for d in detections.get("results", []):
            conn.execute(
                "INSERT INTO detection_records"
                "(filename, detect_time, class_name, confidence, bbox, image_size, mode)"
                "VALUES (?,?,?,?,?,?,?)",
                (
                    filename,
                    datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    d.get("class_name"),       # 原值入库（英文/原始名）
                    d.get("confidence"),
                    json.dumps(d.get("bbox")),
                    json.dumps(detections.get("image_size")),
                    detections.get("mode"),
                ),
            )
            n += 1
        conn.commit()
    return n


def query_records(filters: dict = None) -> list[dict]:
    """按条件查询，filters: {class_name, date_from, date_to, min_conf}"""
    init_db()
    sql = "SELECT * FROM detection_records WHERE 1=1"
    params = []
    if filters:
        if filters.get("class_name"):
            # 兼容：前端可能传中文映射名，也尝试反向匹配原始名
            sql += " AND class_name = ?"; params.append(filters["class_name"])
        if filters.get("date_from"):
            sql += " AND detect_time >= ?"; params.append(filters["date_from"])
        if filters.get("date_to"):
            sql += " AND detect_time <= ?"; params.append(filters["date_to"] + " 23:59:59")
        if filters.get("min_conf"):
            sql += " AND confidence >= ?"; params.append(float(filters["min_conf"]))
    sql += " ORDER BY detect_time DESC LIMIT 500"
    with get_conn() as conn:
        rows = conn.execute(sql, params).fetchall()
    # 读出时映射为中文缺陷名（库内原值不变）
    records = []
    for r in rows:
        d = dict(r)
        d["class_name"] = map_label(d.get("class_name"))
        records.append(d)
    return records


def stats() -> dict:
    """统计：各缺陷类型数量 + 总检测次数 + 缺陷图占比。"""
    init_db()
    with get_conn() as conn:
        total = conn.execute("SELECT COUNT(*) c FROM detection_records").fetchone()["c"]
        by_class = conn.execute(
            "SELECT class_name, COUNT(*) c FROM detection_records "
            "GROUP BY class_name ORDER BY c DESC"
        ).fetchall()
        # 去重统计检测过的图片数
        imgs = conn.execute("SELECT COUNT(DISTINCT filename) c FROM detection_records").fetchone()["c"]
    return {
        "total_defects": total,
        "total_images": imgs,
        "by_class": [{"class_name": map_label(r["class_name"]), "count": r["c"]} for r in by_class],
    }


if __name__ == "__main__":
    init_db()
    print(stats())