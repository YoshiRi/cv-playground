# 画面の足し方（タブ・設定欄・結果の見せ方）

画面は3つの部品でできている。どれも足すだけで、既存の部分を書き換える必要はない。

| 足したいもの | 書く場所 |
| --- | --- |
| タブ（タスク） | `web/models.json` の `tasks` |
| タブの設定欄（閾値・候補・質問など） | 既存の部品を `tasks[].params` で選ぶ。新しい部品は `web/index.html` と `web/app.js`（下の手順） |
| 結果の見せ方（枠・マスク・深度など） | `web/renderers.js` の `KINDS` |

モデルの足し方は [ADDING_MODELS.md](ADDING_MODELS.md)。

## 1. タブ（`models.json` の `tasks`）と分類（`categories`）

タブは分類ごとにまとめて表示する（画面上部の「検出・追跡 / セグメンテーション / 深度・3D / 画像と言語」）。分類は `models.json` の `categories`（`{id, name}` の並び）で、タブの `category` がそれを指す。分類を足す時は `categories` に1件足す。タブが増えても、選んだ分類のタブだけが並ぶので設定欄は長くならない。

```json
{ "id": "wholebody", "name": "手・目（PINTO）", "category": "detect",
  "hint": "タブを選んだ時に出す説明",
  "params": ["threshold", "track", "cascade"],
  "defaults": { "threshold": 0.35 } }
```

| キー | 意味 |
| --- | --- |
| `id` | タブの識別子。モデルの `task` でこれを指す |
| `category` | 分類（`categories[].id`） |
| `name` | タブの表示名 |
| `hint` | タブの説明（設定欄の上に出る） |
| `params` | 出す設定欄（下の表から選ぶ） |
| `defaults` | 設定欄の既定値（`threshold` / `labels` / `prompt`） |
| `click` | `true` なら画像のクリックで点を置いて実行する（プロンプト（SAM）のタブ） |

タブは `tasks` の順に並ぶ。そのタブで使えるモデルが1つも無い時（例: サーバー専用のモデルしか無いのにサーバーが無い）は出ない。

## 2. 設定欄（`params`）

| 名前 | 欄 | 推論に渡る値（`params.xxx`） |
| --- | --- | --- |
| `threshold` | 閾値のスライダー | `threshold`（0〜1） |
| `labels` | 候補の入力（カンマ区切り） | `labels`（文字列） |
| `prompt` | 質問・指示の入力 | `prompt`（文字列） |
| `points` | クリックで置く点、「全体を自動分割」「除く点」 | `points: [[x, y, 1=含める / 0=除く]]`、`auto` |
| `track` | 追跡（ByteTrack / BoT-SORT / BoT-SORT + ReID） | 推論には渡らず、画面側で結果の枠に ID を付ける |
| `cascade` | 検出のあとの分類のチェック（モデルの `cascade` から作る） | 画面側で切り出して分類する |

モデルによって自動で出る欄（「詳細設定」の中に畳んである）: 「実行設定」（汎用 ONNX のブラウザ実行）、「モデル入力（長辺）」（`pre.dynamic` のモデル）→ `params.input_size`。めったに触らない設定は「詳細設定」（`#advanced`）の中に置く。

**新しい設定欄を足す手順**

1. `web/index.html` の `#params` の中に、`data-param="名前"` を付けた要素を足す（タブの `params` に名前が無い時は自動で隠れる）
2. `web/app.js` の `paramsFor()` で、その値を読んで返す
3. 使う adapter（`worker.js` / `adapters.py`）で `params.名前` を読む
4. 既定値を持たせたいなら、`selectTask()` で `t.defaults.名前` を欄に入れる

## 3. 結果の見せ方（`web/renderers.js` の `KINDS`）

adapter が返す結果の `kind` ごとに、見せ方を1件書く。

```js
KINDS.mykind = {
  views: (r) => [["overlay", "重ねる"], ["only", "結果だけ"], ["original", "元画像"]], // 「表示」の選択肢（先頭が既定）。空なら欄を出さない
  async prepare(r) { r.layer = ...; },            // 任意。結果が届いた時に1回だけ（画像の読み込みなど重い下ごしらえ）
  draw(ctx, r, base, view) { ... },               // canvas に描く。元画像は描画済み。base = {src, w, h}（画面の大きさ）
  panel: (r, { live, points }) => "<div>…</div>", // 結果欄の本文（HTML。文字は esc() で逃がす）
  summary: (r) => "3件",                          // 実行履歴の「結果」欄
};
```

- 結果の座標は推論に渡した画像（`r.w × r.h`）のピクセル。`draw` で `base.w / r.w` を掛けて画面に合わせる
- 動画では `draw` が新しいフレームごとに呼ばれるので、重い処理は `prepare` に寄せる
- 色は `labelColor(label)`（ラベルごとに固定）と `idColor(id)`（追跡の ID ごと）を使うと他の結果とそろう

結果欄の上の部分（モデル名・実行場所・推論時間・fps・内訳の帯）は `web/app.js` の `renderResult()` が全種類共通で描く。「画像を保存」「結果データ（JSON）」も全種類共通（表示中の canvas と、結果から画像の層を除いたもの）で、実行履歴の「結果」欄には `summary` が入る。

## 4. 見た目（`web/style.css`）

色は `:root` の変数（`--bg`, `--panel`, `--text`, `--muted`, `--line`, `--accent`, `--chip` など）で、ダークモードは `prefers-color-scheme` で切り替わる。結果欄の部品のクラス: `.stat`（数値のバッジ）、`.chip`（件数）、`.bar`（分類の棒）、`.legend`（色の凡例）、`.answer`（文章）、`.sub`（補足の文）。

## 5. 足したあとの確認

1. サーバー版（`scripts/serve.sh`）と静的版（リポジトリ直下を静的に配るか GitHub Pages）の両方で、タブと結果が出ること
2. スマホの幅（390px 前後）で横スクロールが出ないこと
3. `python3 build.py` で1ファイル版を作り直してコミットする
