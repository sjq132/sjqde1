
"""
Flask 后端（service / route layer）
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
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024
ALLOWED = {".jpg", ".jpeg", ".png", ".bmp"}

_detector = None


def get_detector():
    global _detector
    if _detector is None:
        init_db()
        _detector = DefectDetector(conf=0.25)
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
    if "file" not in request.files:
        return jsonify({"error": "missing file"}), 400
    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED:
        return jsonify({"error": f"unsupported {ext}"}), 400

    safe = secure_filename(Path(file.filename).stem) + ext
    img_path = UPLOAD_DIR / safe
    file.save(str(img_path))

    detector = get_detector()
    results = detector.detect(img_path)
    mode = detector.mode if hasattr(detector, 'mode') else 'yolo'

    res_path = RESULT_DIR / (Path(safe).stem + "_result.jpg")
    draw_detections(str(img_path), results, str(res_path))

    insert_records(safe, {"results": results})

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
    filters = {k: v for k, v in filters.items() if v not in (None, "")}
    records = query_records(filters)
    return jsonify({"records": records, "total": len(records)})


@app.route("/api/stats", methods=["GET"])
def api_stats():
    data = stats()
    from .detector import DEFECT_NAMES
    seen = set()
    by_class = data.get("by_class", [])
    if isinstance(by_class, list):
        seen = {r.get("class_name", "") for r in by_class}
    data["classes"] = sorted(set(DEFECT_NAMES.values()) | seen)
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