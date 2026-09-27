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
    # import { a as b } の別名は、消す代わりに const b = a; にする（本体は先に並べたモジュールの名前として見えている）
    def local_import(m):
        aliases = [x.split(" as ") for x in m.group(2).split(",") if " as " in x]
        return "".join(f"const {b.strip()} = {a.strip()};\n" for a, b in aliases) if m.group(1) == "import" else ""
    src = re.sub(r'^(import|export) \{([^}]*)\} from "\./[\w.]+\.js";\n', local_import, src, flags=re.M)
    names = re.findall(r"^export (?:async function|function|const|let|class) (\w+)", src, flags=re.M)
    body = re.sub(r"^export ", "", src, flags=re.M)
    if not names:  # 何も export しない（app.js、worker.js）はそのまま閉じる
        return f"(async () => {{\n{body}\n}})();\n"
    return f"const {{ {', '.join(names)} }} = await (async () => {{\n{body}\nreturn {{ {', '.join(names)} }};\n}})();\n"


def main() -> None:
    catalog_json = json.loads((WEB / "models.json").read_text())
    catalog = re.sub(r"^export const CATALOG = .*$", lambda _: "export const CATALOG = " + json.dumps(catalog_json, ensure_ascii=False) + ";",
                     (WEB / "catalog.js").read_text(), count=1, flags=re.M)
    assert "import.meta" not in catalog
    # Worker は classic で起動するので、トップレベル await を避けて全体を1つの async 関数に入れる
    worker = "(async () => {\n" + module("coco.js") + module("onnx_generic.js") + module("worker.js") + "})();\n"
    app = module("coco.js") + module("catalog.js", catalog) + module("renderers.js") + module("tracker.js") + module("app.js")
    html = (WEB / "index.html").read_text()
    css = (WEB / "style.css").read_text()

    html = html.replace('<link rel="stylesheet" href="style.css">', f"<style>\n{css}</style>")
    html = re.sub(r"  <!-- 静的配信.*?\n  <script src=\"coi-serviceworker.js\"></script>\n", "", html, flags=re.S)
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
