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
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse, Response, StreamingResponse
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
        # crossOriginIsolated にして、ブラウザの WASM 推論で複数スレッドを使えるようにする。
        # credentialless なら CDN や HF の読み込みは CORS のままで通る
        res.headers["Cross-Origin-Opener-Policy"] = "same-origin"
        res.headers["Cross-Origin-Embedder-Policy"] = "credentialless"
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
def run(model: str = Form(...), params: str = Form("{}"), image: UploadFile = File(...), template: UploadFile | None = File(None)):
    if model not in ENTRIES:
        raise HTTPException(404, f"unknown model {model}")
    raw = image.file.read()
    im = Image.open(io.BytesIO(raw)).convert("RGB")
    p = json.loads(params)
    p["_image_key"] = hashlib.sha1(raw).hexdigest()
    if template is not None:  # テンプレートマッチング: 探す物の画像（特徴は adapters.py が同じ画像のあいだ使い回す）
        traw = template.file.read()
        p["_template"], p["_template_key"] = Image.open(io.BytesIO(traw)).convert("RGB"), hashlib.sha1(traw).hexdigest()
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


@app.get("/local-models/{path:path}")
def local_model(path: str):
    """models/ 以下の、このサーバーだけが配るモデル（ライセンス上リポジトリに入れないもの）"""
    f = (ROOT / "models" / path).resolve()
    if not f.is_file() or (ROOT / "models").resolve() not in f.parents:
        raise HTTPException(404)
    return FileResponse(f)


# モデルのファイルの写し（ブラウザの onnx_generic.js・transformers.js がサーバー版の時に使う）: /mirror/hf/<パス> は
# https://huggingface.co/<パス>、/mirror/gh/<パス> は https://raw.githubusercontent.com/<パス>。初回はネットから取りながら
# models/mirror/ に保存し、2 回目からはディスクから配る（X-Mirror: hit / miss）。ブラウザのキャッシュはサイトごとに分かれ、
# スマホでは消されることもあるので、tailnet の複数の端末で同じモデルを何度もダウンロードしないで済む
MIRROR_SITES = {"hf": "https://huggingface.co/", "gh": "https://raw.githubusercontent.com/"}
MIRROR_DIR = ROOT / "models" / "mirror"


@app.api_route("/mirror/{site}/{path:path}", methods=["GET", "HEAD"])
def mirror(site: str, path: str, request: Request):
    if site not in MIRROR_SITES:
        raise HTTPException(404)
    base = (MIRROR_DIR / site).resolve()
    f = (base / path).resolve()
    if base not in f.parents:
        raise HTTPException(404)
    if f.is_file():
        return FileResponse(f, headers={"X-Mirror": "hit"})
    url = MIRROR_SITES[site] + path
    if request.method == "HEAD":
        r = requests.head(url, allow_redirects=True, timeout=30)
        return Response(status_code=r.status_code, headers={k: v for k, v in r.headers.items() if k.lower() in ("content-length", "content-type")})
    # 圧縮しないで送ってもらう（受け取ったまま流して保存するので、長さが Content-Length と合うように）
    r = requests.get(url, stream=True, timeout=30, headers={"Accept-Encoding": "identity"})
    if r.status_code != 200:  # 無いファイル（transformers.js は省略できる設定ファイルを探す）は、そのままの状態で返して保存しない
        r.close()
        return Response(status_code=r.status_code)
    headers = {"X-Mirror": "miss", "Content-Type": r.headers.get("content-type", "application/octet-stream")}
    if r.headers.get("content-length"):
        headers["Content-Length"] = r.headers["content-length"]

    total = int(r.headers["content-length"]) if r.headers.get("content-length") else None

    def body():
        # 取りながら送り、最後まで取れた時だけ置く（途中で切れた半端なファイルは残さない）。
        # 大きいファイル（数百 MB）は転送が途中で切れることがあるので、その時は続きから取り直す（Range、3 回まで）
        f.parent.mkdir(parents=True, exist_ok=True)
        tmp = f.with_name(f".{f.name}.{os.getpid()}.{threading.get_ident()}.part")
        ok, got, resp, tries = False, 0, r, 0
        try:
            with open(tmp, "wb") as out:
                while True:
                    try:
                        for chunk in resp.raw.stream(1 << 20, decode_content=False):
                            out.write(chunk)
                            got += len(chunk)
                            yield chunk
                    except Exception:
                        pass
                    resp.close()
                    if total is None or got >= total or tries >= 3:
                        break
                    tries += 1
                    resp = requests.get(url, stream=True, timeout=30, headers={"Range": f"bytes={got}-", "Accept-Encoding": "identity"})
                    if resp.status_code != 206:
                        break
            ok = total is None or got == total
        finally:
            resp.close()
            if ok:
                os.replace(tmp, f)
            else:
                tmp.unlink(missing_ok=True)

    return StreamingResponse(body(), headers=headers)


# インタラクトの出口（web/interact.js の websocket）: つないだ相手どうしに、届いたメッセージをそのまま配る。
# 画面（送り手）が結果のフレームを送り、受け手のページや tools/ws_receiver.py・外のツールが受け取る
WS_CLIENTS: set = set()


@app.websocket("/ws")
async def ws_relay(ws: WebSocket):
    await ws.accept()
    WS_CLIENTS.add(ws)
    try:
        while True:
            msg = await ws.receive_text()
            for c in list(WS_CLIENTS):
                if c is not ws:
                    try:
                        await c.send_text(msg)
                    except Exception:
                        WS_CLIENTS.discard(c)
    except WebSocketDisconnect:
        pass
    finally:
        WS_CLIENTS.discard(ws)


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
