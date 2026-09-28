"""models.json の汎用 ONNX モデル（onnx.repo があるもの）に、Hugging Face 上のファイルの SHA-256 を書き込む。

画面の「手元の ONNX を使う」は、選んだファイルの SHA-256 をこの値と照らし、一致したら
そのモデルのファイルとしてブラウザのキャッシュに入れる（ダウンロードせずに使える）。
Hugging Face の API は LFS のファイルの SHA-256 を返すので、モデル本体は落とさない。

  python3 tools/update_hashes.py          # 書き換える
  python3 tools/update_hashes.py --check  # 変わっていれば終了コード 1（書き換えない）
"""
import json
import sys
import urllib.request
from pathlib import Path

MODELS = Path(__file__).resolve().parent.parent / "web" / "models.json"


def tree(repo, folder):
    url = f"https://huggingface.co/api/models/{repo}/tree/main/{folder}".rstrip("/")
    with urllib.request.urlopen(url, timeout=30) as r:
        return {x["path"]: (x.get("lfs") or {}).get("oid") for x in json.load(r)}


def main():
    src = MODELS.read_text()
    cat = json.loads(src)
    listings = {}
    for m in cat["models"]:
        o = m.get("onnx") or {}
        if m.get("adapter") != "onnx" or not o.get("repo"):
            continue
        hashes = {}
        for f in (o.get("file"), o.get("file_fp16"), o.get("data")):
            if not f:
                continue
            folder = f.rsplit("/", 1)[0] if "/" in f else ""
            key = (o["repo"], folder)
            if key not in listings:
                listings[key] = tree(*key)
            sha = listings[key].get(f)
            if not sha:
                print(f"  SHA-256 が取れない（LFS でない？）: {o['repo']}/{f}", file=sys.stderr)
                continue
            hashes[f] = sha
        o["sha256"] = hashes
        print(m["key"], {f: h[:12] for f, h in hashes.items()})
    out = json.dumps(cat, indent=1, ensure_ascii=False) + "\n"
    if "--check" in sys.argv:
        sys.exit(0 if out == src else 1)
    MODELS.write_text(out)


main()
