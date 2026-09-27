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


def module(name, src=None):
    """ES モジュール1つを、スコープを閉じた即時関数に包む（同じ名前の内部関数がファイル間でぶつからないように）。
    export した名前だけを外に出し、ローカルの import / re-export の行は消す（依存先は先に並べておく）"""
    src = src if src is not None else (WEB / name).read_text()
    src = re.sub(r'^(import|export) \{[^}]*\} from "\./[\w.]+\.js";\n', "", src, flags=re.M)
    names = re.findall(r"^export (?:async function|function|const|let|class) (\w+)", src, flags=re.M)
    body = re.sub(r"^export ", "", src, flags=re.M)
    if not names:  # 何も export しない（app.js、worker.js）はそのまま閉じる
        return f"(async () => {{\n{body}\n}})();\n"
    return f"const {{ {', '.join(names)} }} = await (async () => {{\n{body}\nreturn {{ {', '.join(names)} }};\n}})();\n"


def main() -> None:
    models = json.loads((WEB / "models.json").read_text())["models"]
    catalog = re.sub(r"^export const MODELS = .*$", lambda _: "export const MODELS = " + json.dumps(models, ensure_ascii=False) + ";",
                     (WEB / "catalog.js").read_text(), count=1, flags=re.M)
    assert "import.meta" not in catalog
    # Worker は classic で起動するので、トップレベル await を避けて全体を1つの async 関数に入れる
    worker = "(async () => {\n" + module("coco.js") + module("onnx_generic.js") + module("worker.js") + "})();\n"
    app = module("coco.js") + module("catalog.js", catalog) + module("tracker.js") + module("app.js")
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
