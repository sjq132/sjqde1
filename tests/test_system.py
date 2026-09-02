"""
自动化测试：覆盖任务书要求的"自动化测试 + 业务闭环"
测试项：
  T1 检测算法（兜底模式）：输入图像 -> 结构化结果
  T2 数据库：写入 + 查询 + 统计
  T3 Flask API：/api/detect, /api/history, /api/stats（使用 test_client，不占用端口）
运行：pytest tests/ 或 python -m unittest tests.test_system
"""
import sys
from pathlib import Path
import unittest
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.detector import DefectDetector, draw_detections
from app.database import init_db, insert_records, query_records, stats
from app.app import app as flask_app


def make_test_image(path: Path, with_defect: bool = True):
    """生成一张模拟工业零件图像，可选画一个"缺陷"矩形。"""
    img = np.full((200, 200), 180, dtype=np.uint8)
    if with_defect:
        cv2.rectangle(img, (60, 80), (140, 120), 60, -1)  # 暗斑模拟缺陷
    cv2.imwrite(str(path), img)


class TestDetector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = ROOT / "data" / "processed" / "test_img.jpg"
        make_test_image(cls.tmp)

    def test_detect_returns_structure(self):
        det = DefectDetector()
        results = det.detect(self.tmp)
        # detect() 返回检测列表；mode 通过 detector 属性判断
        self.assertIsInstance(results, list)
        self.assertIn(det.mode, ("yolo", "fallback"))
        # bbox 坐标合法性
        for r in results:
            self.assertEqual(len(r["bbox"]), 4)
            self.assertTrue(0.0 <= r["confidence"] <= 1.0)

    def test_draw_detections(self):
        det = DefectDetector()
        results = det.detect(self.tmp)
        out = ROOT / "data" / "processed" / "test_result.jpg"
        draw_detections(self.tmp, {"results": results}, out)
        self.assertTrue(out.exists() and out.stat().st_size > 0)


class TestDatabase(unittest.TestCase):
    def test_crud_and_stats(self):
        init_db()
        rec = {"results": [
            {"class_name": "crack", "confidence": 0.87, "bbox": [10, 20, 50, 60]},
            {"class_name": "scratch", "confidence": 0.65, "bbox": [5, 5, 40, 40]},
        ], "mode": "test", "image_size": [200, 200]}
        n = insert_records("ut.jpg", rec)
        self.assertEqual(n, 2)

        rows = query_records({"class_name": "crack"})
        self.assertTrue(all(r["class_name"] == "crack" for r in rows))

        s = stats()
        self.assertGreaterEqual(s["total_defects"], 2)
        names = [x["class_name"] for x in s["by_class"]]
        self.assertIn("crack", names)


class TestAPI(unittest.TestCase):
    def setUp(self):
        self.client = flask_app.test_client()
        # 确保库已初始化
        init_db()

    def test_pages_render(self):
        for url in ["/", "/history", "/stats"]:
            rv = self.client.get(url)
            self.assertEqual(rv.status_code, 200, msg=url)

    def test_detect_flow(self):
        img = ROOT / "data" / "processed" / "upload_api.jpg"
        make_test_image(img)
        with open(img, "rb") as f:
            rv = self.client.post("/api/detect", data={"file": (f, "upload_api.jpg")},
                                  content_type="multipart/form-data")
        self.assertEqual(rv.status_code, 200)
        data = rv.get_json()
        self.assertNotIn("error", data)
        self.assertIn("result_image", data)

    def test_history_and_stats(self):
        self.assertEqual(self.client.get("/api/history").status_code, 200)
        s = self.client.get("/api/stats").get_json()
        self.assertIn("total_defects", s)
        self.assertIn("by_class", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
