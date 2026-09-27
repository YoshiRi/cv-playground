"""CV Playground のサーバー側。静的ページ（web/）の配信と、サーバー（Apple Silicon なら MPS / CoreML、ほかは CPU）での推論API。

  .venv/bin/python server.py            # http://127.0.0.1:8010
  別の端末から開く時は https が必要（WebGPU・カメラのため）。例: tailscale serve --bg --https=8443 http://127.0.0.1:8010

モデルの一覧は web/models.json（ブラウザ側と共通）、実装は adapters.py。where に "server" があるモデルを受け持つ。
モデルは最初に使った時に読み込み、MAX_LOADED 個を超えたら古い順に外す。
"""
import hashlib
import io
import json
import os
import threading
import time
from collections import OrderedDict
from pathlib import Path

import requests
import torch
import uvicorn
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

from adapters import ADAPTERS, DEVICE, OLLAMA, load_entries

ROOT = Path(__file__).parent
MAX_LOADED = int(os.environ.get("CVPG_MAX_LOADED", "3"))
# 同じマシンで別の GPU 処理（LLM のベンチなど）が動いている目印のファイル。あれば画面に警告を出す（任意）
GPU_LOCK = Path(os.environ["CVPG_GPU_LOCK"]) if os.environ.get("CVPG_GPU_LOCK") else None

ENTRIES = {e["key"]: e for e in load_entries() if "server" in e["where"] and e["adapter"] in ADAPTERS}
loaded: "OrderedDict[str, object]" = OrderedDict()
lock = threading.Lock()  # MPS を同時に叩かない


def get_adapter(key):
    if key in loaded:
        loaded.move_to_end(key)
        return loaded[key], 0.0
    t = time.time()
    a = ADAPTERS[ENTRIES[key]["adapter"]](ENTRIES[key])
    a.load()
    loaded[key] = a
    while len(loaded) > MAX_LOADED:
        loaded.popitem(last=False)
        if DEVICE == "mps":
            torch.mps.empty_cache()
    return a, (time.time() - t) * 1000


app = FastAPI()


@app.middleware("http")
async def no_stale_cache(request, call_next):
    # 更新した web/ のファイル（特に Worker の JS）を古いキャッシュのまま使わないよう、毎回確かめさせる（変わっていなければ 304）
    res = await call_next(request)
    if not request.url.path.startswith("/api/"):
        res.headers["Cache-Control"] = "no-cache"
    return res


@app.get("/api/models")
def list_models():
    return [{"key": k, "task": e["task"], "name": e["name"]} for k, e in ENTRIES.items()]


@app.get("/api/status")
def status():
    s = {"device": DEVICE, "loaded": list(loaded), "gpu_lock": GPU_LOCK.read_text().strip() if GPU_LOCK and GPU_LOCK.exists() else None}
    if DEVICE == "mps":
        s["mps_allocated_gb"] = round(torch.mps.current_allocated_memory() / 1e9, 2)
    try:
        s["ollama_loaded"] = [m["name"] for m in requests.get(f"{OLLAMA}/api/ps", timeout=2).json().get("models", [])]
    except Exception:
        s["ollama_loaded"] = None
    return s


@app.post("/api/unload")
def unload():
    with lock:
        loaded.clear()
        if DEVICE == "mps":
            torch.mps.empty_cache()
    return status()


@app.post("/api/run")
def run(model: str = Form(...), params: str = Form("{}"), image: UploadFile = File(...)):
    if model not in ENTRIES:
        raise HTTPException(404, f"unknown model {model}")
    raw = image.file.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    p = json.loads(params)
    p["_image_key"] = hashlib.sha1(raw).hexdigest()
    with lock:
        a, load_ms = get_adapter(model)
        t = time.time()
        res = a.run(im, p)
        if DEVICE == "mps":
            torch.mps.synchronize()
        res["infer_ms"] = (time.time() - t) * 1000
    res["load_ms"] = load_ms
    res["device"] = a.device
    return JSONResponse(res)


@app.get("/cv-playground.html")
def standalone():
    """サーバーなし版（build.py で作る1ファイル）。保存して別の場所で開く用"""
    return FileResponse(ROOT / "dist" / "cv-playground.html", media_type="text/html",
                        headers={"Content-Disposition": 'inline; filename="cv-playground.html"'})


@app.get("/")
def index():
    return FileResponse(ROOT / "web" / "index.html")


app.mount("/", StaticFiles(directory=ROOT / "web"), name="web")

if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=int(os.environ.get("CVPG_PORT", "8010")))
