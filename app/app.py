"""
Flask 后端（service / route layer）

API 清单：
    POST /api/detect      上传图片 -> 推理 -> 写库 -> 返回 JSON + 标注图 URL
    GET  /api/history     查询历史记录（支持筛选）
    GET  /api/stats       缺陷统计（供 Chart.js 绘图）
    GET  /                前端页面
    GET  /history         历史页
    GET  /stats           统计页

运行：python -m app.app   （项目根目录下）
      或：flask --app app.app run --debug
"""
import os
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, render_template
from werkzeug.utils import secure_filename

from .detector import DefectDetector, draw_detections
from .database import init_db, insert_records, query_records, stats

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = ROOT / "data" / "processed" / "uploads"
RESULT_DIR = ROOT / "data" / "processed" / "results"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
RESULT_DIR.mkdir(parents=True, exist_ok=True)

app = Flask(
    __name__,
    template_folder=str(ROOT / "app" / "templates"),
    static_folder=str(ROOT / "app" / "static"),
)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp"}

# 延迟初始化：首次请求时再建库、再加载模型（避免 import 时耗时）
_detector = None


def get_detector() -> DefectDetector:
    global _detector
    if _detector is None:
        init_db()
        _detector = DefectDetector(conf_thres=0.5)
    return _detector


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/history")
def history_page():
    return render_template("history.html")


@app.route("/stats")
def stats_page():
    return render_template("stats.html")


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """上传图片 -> 检测 -> 存库 -> 返回结果。"""
    if "file" not in request.files:
        return jsonify({"error": "缺少 file 字段"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "空文件名"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"不支持的格式 {ext}，仅允许 JPG/PNG/BMP"}), 400

    safe = secure_filename(Path(file.filename).stem) + ext
    img_path = UPLOAD_DIR / safe
    file.save(str(img_path))

    detector = get_detector()
    results = detector.detect(img_path)
    mode = detector.mode
    detections = {"results": results, "mode": mode}

    # 绘制标注图
    res_path = RESULT_DIR / (Path(safe).stem + "_result.jpg")
    draw_detections(img_path, detections, res_path)

    # 入库
    insert_records(safe, detections)

    return jsonify({
        "filename": safe,
        "mode": mode,
        "results": results,
        "result_image": f"/results/{res_path.name}",
        "count": len(results),
    })


@app.route("/api/history", methods=["GET"])
def api_history():
    filters = {
        "class_name": request.args.get("class_name"),
        "date_from": request.args.get("date_from"),
        "date_to": request.args.get("date_to"),
        "min_conf": request.args.get("min_conf", type=float),
    }
    # 去掉空值
    filters = {k: v for k, v in filters.items() if v not in (None, "")}
    records = query_records(filters)
    return jsonify({"records": records, "total": len(records)})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    data = stats()
    # 补充 MVTec AD 全量类别（含库中暂无记录的），供前端筛选下拉框
    from .detector import CLASS_NAMES
    seen = {r["class_name"] for r in data.get("by_class", [])}
    data["classes"] = sorted(set(CLASS_NAMES) | seen)
    return jsonify(data)


@app.route("/uploads/<path:fname>")
def serve_upload(fname):
    return send_from_directory(str(UPLOAD_DIR), fname)


@app.route("/results/<path:fname>")
def serve_result(fname):
    return send_from_directory(str(RESULT_DIR), fname)


if __name__ == "__main__":
    init_db()
    app.run(host="0.0.0.0", port=5000, debug=False)
