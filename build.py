"""web/ から、サーバー不要の1ファイル版 HTML（dist/cv-playground.html）を作る。

  python3 build.py

ブラウザ実行のモデルだけを使い、transformers.js は jsDelivr、モデルは Hugging Face から各ブラウザが直接取得する。
ファイルを開くだけで動く（file:// でも、任意の静的ホスティングでも）。本体と同じソースから作るので、web/ を直したら作り直す。
- coco.js / catalog.js / onnx_generic.js / worker.js / app.js の import・export を外して1つにまとめ、models.json を埋め込む
- Worker は <script type="text/plain"> に入れた本文から Blob URL で作る
"""
import json
import re
from pathlib import Path

WEB = Path(__file__).parent / "web"
OUT = Path(__file__).parent / "dist" / "cv-playground.html"


def strip_module(src: str) -> str:
    """ローカルの import / re-export の行を消し、export を外す（全部を1つのスクリプトにつなげるため）"""
    src = re.sub(r'^(import|export) \{[^}]*\} from "\./[\w.]+\.js";\n', "", src, flags=re.M)
    return re.sub(r"^export ", "", src, flags=re.M)


def main() -> None:
    read = lambda name: strip_module((WEB / name).read_text())
    models = json.loads((WEB / "models.json").read_text())["models"]
    catalog = re.sub(r"^const MODELS = .*$", lambda _: "const MODELS = " + json.dumps(models, ensure_ascii=False) + ";",
                     read("catalog.js"), count=1, flags=re.M)
    assert "import.meta" not in catalog
    worker = read("coco.js") + "\n" + read("onnx_generic.js") + "\n" + read("worker.js")
    app = read("coco.js") + "\n" + catalog + "\n" + read("app.js")
    html = (WEB / "index.html").read_text()
    css = (WEB / "style.css").read_text()

    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>")
    html = html.replace("<title>CV Playground</title>", "<title>CV Playground（サーバーなし版）</title>")
    assert "</script>" not in worker and "</script>" not in app
    boot = ("<script>\n"
            "globalThis.CVPG_STANDALONE = true;\n"
            "globalThis.CVPG_WORKER_URL = URL.createObjectURL(new Blob([document.getElementById('worker-src').textContent],"
            " { type: 'text/javascript' }));\n"
            "</script>\n")
    html = html.replace('  <script type="module" src="app.js"></script>',
                        f'  <script type="text/plain" id="worker-src">\n{worker}</script>\n{boot}'
                        f'  <script type="module">\n{app}</script>')
    OUT.parent.mkdir(exist_ok=True)
    OUT.write_text(html)
    print(f"{OUT} ({OUT.stat().st_size / 1e3:.0f} KB)")


if __name__ == "__main__":
    main()
