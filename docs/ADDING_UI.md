# 画面の足し方（タブ・設定欄・結果の見せ方）

画面は3つの部品でできている。どれも足すだけで、既存の部分を書き換える必要はない。

| 足したいもの | 書く場所 |
| --- | --- |
| タブ（タスク） | `web/models.json` の `tasks` |
| タブの設定欄（閾値・候補・質問など） | 既存の部品を `tasks[].params` で選ぶ。新しい部品は `web/index.html` と `web/app.js`（下の手順） |
| 結果の見せ方（枠・マスク・深度など） | `web/renderers.js` の `KINDS` |
| 応用（タブの結果に後付けして集計する。例: 数える） | `models.json` の `apps` と `web/apps.js` の `APPS` |
| インタラクト（結果を外に流す・別の画面を動かす） | `models.json` の `interact` と `web/interact.js` の `INTERACT` |

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
| `template` | 探す物（テンプレート）: 「枠で切り出す」を押して画像・フレームの上をドラッグ（pointer イベントなのでタッチでも。切り出す間だけ canvas のスクロールを止める）か、ファイルで選ぶ | `template: {id, image?}`（ブラウザは Worker ごとに初回だけ画像を送る。サーバーは毎回 `template` のファイルで送り、サーバーが同じ画像の特徴を使い回す） |
| `top_k` | 特徴点の数（256 / 512 / 1024 / 2048） | `top_k` |

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

例: テンプレートマッチングの `matches` は「四角形」（見つけた四角形とインライアの点）・「対応点」（左にテンプレート、右にフレームを並べて対応を線で結ぶ。テンプレートの画像は画面側が `r.templateImage` に付ける）・「特徴点」（スコアの色）。

結果欄の上の部分（モデル名・実行場所・推論時間・fps・内訳の帯）は `web/app.js` の `renderResult()` が全種類共通で描く。「画像を保存」「結果データ（JSON）」も全種類共通（表示中の canvas と、結果から画像の層を除いたもの）で、実行履歴の「結果」欄には `summary` が入る。

## 4. 応用（`models.json` の `apps` と `web/apps.js`）

モデルの結果を**フレームをまたいで集計して重ねる**機能（例: 数える）は、タブを増やさず、その種類の結果を返すタブに後付けする（物体検出の cascade と同じ考え方）。`apps` に1件書くと、`accepts` に合う結果を返すタブ（`tasks[].result`）の設定欄に「応用」のチェックが出て、選ぶと追跡・cascade のあとに順に呼ばれる。モデルの後処理（models.json のモデルの `post`、`onnx_generic.js` / `adapters.py`）とは別物。

```json
"apps": [
  { "id": "count", "name": "数える", "accepts": ["boxes"], "params": ["classes"], "hint": "チェックの説明（マウスを乗せると出る）" }
]
```

| キー | 意味 |
| --- | --- |
| `accepts` | 受け付ける結果の種類。タブの `result`（そのタブのモデルが返す種類。今は物体検出・人物の姿勢・手と目・テキスト物体検知が `"boxes"`）と照らす |
| `params` | 選んだ時だけ出す設定欄。`web/index.html` の `#apps-row` の中に `data-app-param="名前"` の要素を置き、`createApps()` で値を渡す |

中身は `web/apps.js` の `APPS` に1件書く（形は `KINDS` と同じ考え方で、集計の状態を持つ点が違う）。

```js
APPS.myapp = {
  create: (opts) => ({ ... }),          // 状態。連続実行の開始時と、静止画の1回ごとに作り直す。opts = { classes }（応用欄の値）
  update(st, r, { tracked }) { ... },   // 1フレームごと（フレームの順）。r.items を絞り込んでよい（枠の表示・結果データにも反映される）
  draw(ctx, st, r, base) { ... },       // canvas に重ねる（元画像と結果の枠は描画済み）
  panel: (st) => "<div>…</div>",        // 結果欄に足す HTML
  summary: (st) => "person 12",         // 実行履歴の「結果」欄に足す文字
};
```

- 呼び出しは `web/app.js` の `createApps()` / `applyApps()`（`run()` と、連続実行の `handle()`）、`draw()` と `renderResult()` だけ。追跡の設定はモデルのタブの1か所
- URL の `?apps=count` で最初から選んだ状態で開ける（「物体カウント」を入口にしたい時のリンク）
- 複数のモデルを組み合わせるものは、次の「組み合わせのタブ」にする

### 組み合わせのタブ（`tasks[].combo` と `web/apps.js` の `COMBOS`）

2つ以上のモデルを同じフレームで回して、結果を組み合わせるタブ（例: しぐさ＝姿勢＋PINTO の部位）。

```json
{ "id": "gesture", "name": "しぐさ（姿勢＋PINTO）", "category": "detect", "result": "boxes",
  "params": ["threshold", "track", "cascade"],
  "combo": { "base": "pose", "app": "gesture",
             "with": [{ "role": "parts", "name": "組み合わせるモデル（PINTO の部位）", "task": "wholebody", "show": ["head", "eye", "hand", "front", "…"] }] } }
```

| キー | 意味 |
| --- | --- |
| `combo.base` | 画面の「モデル」に並べるタブ。そのモデルの結果が `r` |
| `combo.with` | 役割ごとの2つ目以降のモデル。`task` のタブのモデルを、`name` の選択欄で選ぶ。結果の `items` は `r.with[role]` に入る。`show` は PINTO のように表示するクラスを絞るモデルで、組み合わせに使うクラスを渡す（`params.show`） |
| `combo.app` | 組み合わせ方（`COMBOS` の名前） |
| `combo.with[].every` | 連続実行では N フレームに1回だけ回し、間は前の結果を使う（深度のような重いモデル用） |
| `combo.with[].params` | そのモデルに渡す追加のパラメータ（深度は `depth_raw: true` で、画像にする前の値 `depthRaw` も返す） |
| `combo.with[].default` | 役割のモデルの既定（`key`） |
| `combo.track` | 追跡が前提のタブで、追跡の欄が「なし」の時に使う追跡（例: `"bytetrack"`） |

```js
COMBOS.mycombo = {
  combine(r) { ... },       // 追跡・cascade の後に呼ぶ。r.items（base の結果、追跡の ID つき）と r.with[role]（items）・r.withResults[role]（結果そのもの）から組み直す
  draw(ctx, r, base) { ... }, // 任意。canvas に重ねて描く（3D の姿勢の小窓など）
  reset() { ... },          // 任意。連続実行の開始時に呼ぶ（ID ごとの履歴を捨てる）
  panel: (r) => "<div>…</div>",  // 結果欄に足す HTML
  summary: (r) => "…",      // 実行履歴の「結果」欄に足す文字
};
```

- 同じフレームを役割ごとのモデルにも送る（`runOnce`）。同じ Worker のモデルは続けて実行されるので、1フレームの時間はほぼ足し算になる。推論時間・内訳は足し合わせて出す
- 追跡は base の結果（`r.items`）にかかる。cascade（目の開閉など）は役割ごとのモデルの `cascade` を、そのモデルの結果にかける（`applyCombo`）。設定欄の「検出のあとの分類」も役割ごとのモデルのものが出る
- 実行履歴はタブの名前と「base のモデル + 役割ごとのモデル」で残す。ベンチマークの候補には出さない（モデルは元のタブで測れる）

## 5. インタラクト（`models.json` の `interact` と `web/interact.js`）

検出の結果を外に流す出口と、結果とは別の画面を動かす部品。土台は出口で、ブラウザだけで動く「モデルを差し替えられる CV のセンサー」として、見せ方は外のツール（別のページ・TouchDesigner・Unity・Python など）に任せる。キャラは、流れているフレームを目で確かめるための表示。CV の層（モデル・後処理・応用）とは分けていて、受け手はモデルを知らず、決まった形の「フレーム」だけを受け取る。

```
モデル → 結果 → 追跡・cascade・組み合わせ・応用 → toFrame（フレーム v1）→ 受け手（INTERACT[id]）→ 自分の画面
```

| 応用（`apps`） | インタラクト（`interact`） |
| --- | --- |
| 数や判定を出す。結果の画像に重ねて描き、結果欄に出す | 自分の画面（結果の横、狭い画面では下）を持ち、結果が来ない間も自分で動く（なめらかにつなぐ・まばたきなど） |
| 1 つのモデルの結果を集計する | 追跡・表情・組み合わせを足したあとの最終結果を使う |

**フレーム**（`toFrame` が作る。座標は画像の幅・高さで割った 0〜1）

```js
{ v: 1, t, w, h, task, model,
  items: [{ id?, label, score, box: [x1, y1, x2, y2], keypoints?: [[x, y, 可視度]], emotion?, state? }] }
```

- `keypoints` は 17 点なら COCO の順（鼻・左目・右目・左耳・右耳・左肩・右肩・左肘・右肘・左手首・右手首・左腰・…）、5 点なら顔（右目・左目・鼻・口の右端・左端）。左右は写っている人から見た向き
- `emotion` は顔の表情（`{ label, prob, probs, valence, arousal }`）

**受け手**（`web/interact.js` の `INTERACT`）

```js
INTERACT.myid = {
  create(el, opts) {           // el: 受け手用の空の要素（canvas などを入れる）
    return { onFrame(frame) { ... }, destroy() { ... } };  // destroy で requestAnimationFrame などを止める
  },
};
```

`models.json` の `interact` に `{ "id": "myid", "name": "…", "tasks": ["pose", "face"], "hint": "…" }` を書くと、`tasks` のタブに「インタラクト」のチェックが出る。URL の `?interact=myid` で最初から選べる。人を選ぶのは `pickTarget(frame, sel, 条件)`（一番大きく写っている物。追跡の ID があれば、別の物が 1.5 倍大きくなるまで同じ物を選び続ける）。

**出口**（`inline: true` の受け手。設定欄の中に状態を出す）

| id | 送り先 | 受け取り方 |
| --- | --- | --- |
| `broadcast` | 同じブラウザの別のタブ（BroadcastChannel「cv-playground」。同じ配信元だけ） | `web/receiver.html` を開く（出口の欄にリンクがある） |
| `websocket` | WebSocket。既定はこのページを配っているサーバーの `/ws`（`server.py` が、つないだ相手どうしに配る）。自分のツールの URL も書ける | `web/receiver.html?ws`、`python3 tools/ws_receiver.py`、TouchDesigner の WebSocket DAT など |

- 送るメッセージは `{ "type": "frame", "frame": {…} }`（JSON）。受け手は `frame.v` を見て、知らない版なら無視する
- `wall`（`Date.now()`）で、送ってから届くまでの遅れが分かる（同じ端末の時。M4 で 1〜3ms）
- WebSocket は、受け手が遅くて 1MB 以上たまったらそのフレームを捨てる（遅れをためない）。https のページ（GitHub Pages など）からは `wss://` か `ws://localhost` にしかつなげない。静的版には `/ws` が無いので、自分で中継を立てる
- `web/receiver.html` は確認用のページ: キャラ・届いた fps と遅れ・物の一覧・最後のフレームの JSON を出す。元の画面とはフレームの形だけでつながっている

**例: キャラ（`puppet`、Live2D 風。確認用）** — Live2D と同じく、フレームを直接絵にせず「パラメータ」（頭の向き 3 軸・体の傾き・腕の角度・口・眉・目）の目標にして、画面の更新ごとにばねで目標に寄せ、パーツの絵を回す・ずらす。推論が 10fps でも表示は 60fps でなめらか。頭の左右の向きは、奥の髪・顔・手前の前髪をずらす量を変えて立体に見せる。表情（`emotion`）があれば口・眉・目を変える。結果に人がいない状態が 1.5 秒続いたら元の姿勢に戻る（静止画は最後の姿勢のまま）。

## 6. 見た目（`web/style.css`）

色は `:root` の変数（`--bg`, `--panel`, `--text`, `--muted`, `--line`, `--accent`, `--chip` など）で、ダークモードは `prefers-color-scheme` で切り替わる。結果欄の部品のクラス: `.stat`（数値のバッジ）、`.chip`（件数）、`.bar`（分類の棒）、`.legend`（色の凡例）、`.answer`（文章）、`.sub`（補足の文）。

## 7. 足したあとの確認

1. サーバー版（`scripts/serve.sh`）と静的版（リポジトリ直下を静的に配るか GitHub Pages）の両方で、タブと結果が出ること
2. スマホの幅（390px 前後）で横スクロールが出ないこと
3. `python3 build.py` で1ファイル版を作り直してコミットする
