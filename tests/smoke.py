"""
真实 HTTP 冒烟测试：启动 Flask 服务，走通完整业务流程。
运行：python tests/smoke.py  （在项目根目录 phase3/ 下执行）
"""
import io
import subprocess
import sys
import time
import urllib.request

HOST = "http://127.0.0.1:5000"


def post_detect():
    # 构造一张最小 PNG 上传
    import base64, zlib
    png = base64.b64decode(
        "iVBORw0KGgoAAAANSUhEUgAAAAQAAAAECAYAAACp8Z5+AAAAEklEQVQIHWP4z8DwHwAFgAI/"
        "h78nCwAAAABJRU5ErkJggg=="
    )
    req = urllib.request.Request(
        HOST + "/api/detect",
        data=png,
        headers={"Content-Type": "image/png"},
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as r:
        return r.read().decode()


def main():
    proc = subprocess.Popen([sys.executable, "-m", "app.app"])
    try:
        time.sleep(3)
        print("POST /api/detect ->", post_detect())
    finally:
        proc.terminate()


if __name__ == "__main__":
    main()
