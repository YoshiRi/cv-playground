# モデルの足し方（バックエンド）

モデルは `web/models.json` の `models` に1件書き、その `adapter`（読み込みと実行の実装）で動かす。
ブラウザ（`web/worker.js`）とサーバー（`adapters.py`）が同じ `models.json` を読むので、書く場所は1つ。

```
web/models.json ──┬── ブラウザ: web/worker.js の ADAPTERS[adapter]（汎用 ONNX は web/onnx_generic.js）
                  └── サーバー: adapters.py の ADAPTERS[adapter]（server.py が where に "server" のものを受け持つ）
```

## 1. まず決めること

| 決めること | 選択肢 |
| --- | --- |
| どのタブに出すか | `task`: `models.json` の `tasks[].id`（新しいタブは [ADDING_UI.md](ADDING_UI.md)） |
| ライセンス | `license`: 画面のモデルの説明に出す（例: `"Apache-2.0"`、`"CC BY-NC 4.0（非商用）"`）。**必ず書く** |
| 大きさ | `mb`: ブラウザが最初に取得するおおよその MB。100MB を超えると初回に確認を出す |
| ベンチマーク | `bench: true` にすると「速度を測る」で最初から選ばれる（軽い代表だけに付ける） |
| どこで動かすか | `where`: `["browser"]` / `["server"]` / `["browser", "server"]`（両方なら同じモデルを比べられる） |
| どう動かすか | 素の ONNX で前処理・後処理が単純 → `adapter: "onnx"`（JSON を書くだけ）。ライブラリのプロセッサが要る → 専用の adapter を書く |
| 重みをどこから取るか | 下の「重みの置き場所」 |

## 2. 素の ONNX（`adapter: "onnx"`）— JSON を書くだけ

前処理（`pre`）と後処理（`post`）を名前付きの部品で組み合わせる。同じ部品が JS（`web/onnx_generic.js`）と Python（`adapters.py`）にあるので、`where` に両方書けばブラウザとサーバーで同じ手順になる。

```json
{ "key": "yolo26n", "task": "detect", "name": "YOLO26n", "where": ["browser", "server"], "adapter": "onnx", "mb": 10,
  "onnx": { "repo": "onnx-community/yolo26n-ONNX", "file": "onnx/model.onnx", "file_fp16": "onnx/model_fp16.onnx" },
  "pre":  { "size": [640, 640], "resize": "letterbox", "pad_value": 114, "scale": 0.00392156862745098, "input": "pixel_values" },
  "post": { "type": "yolo_detect", "classes": "coco" } }
```

### `onnx`（重みの場所）

| キー | 意味 |
| --- | --- |
| `repo` + `file` | Hugging Face のリポジトリとファイル（ブラウザは HF から直接、サーバーは `models/` に落とす） |
| `data` | 外部データ（`.onnx_data`）がある時のファイル名 |
| `file_fp16` | fp16 版（画面の「実行設定」で fp16 を選んだ時に使う） |
| `path` + `url` | `web/` に同梱したファイル（`path` は `web/` からの相対）。読めない時（file:// など）は `url` から取る |
| `sha256` | ファイルごとの SHA-256（`{"onnx/model.onnx": "…"}`）。`python3 tools/update_hashes.py` が Hugging Face から取って書き込む（手で書かない）。画面の「手元の ONNX を使う」で、利用者が持っているファイルと照らすのに使う |
| `cut` | 途中の値の名前の並び。その値で ONNX を切り、後ろのノード（CPU に回って graph capture を妨げる出力の頭など）を消す。出力は float にして `cut_0`, `cut_1`, … になる。重みのライセンス上、書き換えた ONNX は配らず、取得した ONNX をブラウザ（`onnx_generic.js` の `cutOnnx`）とサーバーがそれぞれ書き換える。後処理は切った値を受け取るものにする（YOLO26-pose は `yolo_pose_raw`） |
| `graph_capture` | `false` なら graph capture を使わない（作れても出力が壊れるモデル。DA3 small） |
| `server_file` | `models/` に置いた、サーバーだけが配るファイル（ライセンス上リポジトリに入れないもの）。サーバーが無い時は画面に出ない |

### `pre`（前処理）

| キー | 意味 |
| --- | --- |
| `resize` | `letterbox`（縦横比を保って `pad_value` で埋めた正方形）/ `letterbox_rect`（長辺を合わせ `stride` の倍数まで埋めた長方形。Ultralytics の推論と同じ）/ `stretch`（縦横をそのまま引き伸ばす）/ `keep_aspect`（短辺を `short` にし `multiple` の倍数に丸める） |
| `size` | 入力の `[幅, 高さ]`（`letterbox_rect` は `size[0]` が長辺） |
| `dynamic` | 入力サイズ可変。画面の「モデル入力（長辺）」で長辺を選べる |
| `scale`, `mean`, `std` | 画素 × `scale` から `mean` を引いて `std` で割る（チャンネルごと） |
| `bgr` | 入力のチャンネル順を B, G, R にする |
| `input` | 入力の名前 |
| `add_dims` | 次元を足す位置（例: 多視点モデルの `[1]` で (1, 1, 3, H, W)） |
| `batch` | 切り出しを入れる分類・特徴モデル用: 固定バッチ数、または `"dynamic"` |
| `seq` | フレーム列を入れるモデル用: 1回に使うフレーム数（入力 (1, C, T, H, W)） |

### `post.type`（後処理。結果の形式は下の「結果の形式」）

| type | 出力 | 結果 |
| --- | --- | --- |
| `yolo_detect` | onnx-community の YOLO26: `logits` (1, Q, 80)・`pred_boxes`（正規化 cx cy w h） | boxes |
| `yolo_pose` | onnx-community の YOLO26-pose の出力そのまま: (1, Q, 57) 正規化（可視度まで 640 で割られていて使えない。今は使っていない） | boxes（keypoints つき） |
| `yolo_pose_raw` | YOLO26-pose を `onnx.cut: ["/model.23/Transpose_2_output_0"]` で切った全候補: (1, 候補数, 56) = x1 y1 x2 y2（入力のピクセル）, スコア, 17 ×（x, y, 可視度）。スコアの高い順に最大 300 件 | boxes（keypoints つき） |
| `ultra_e2e_detect` / `ultra_e2e_pose` | Ultralytics の end2end 書き出し: (1, 300, 6 / 57) 入力のピクセル座標 | boxes |
| `deim_wholebody` | PINTO の DEIMv2: (1, Q, 6) = クラス, 正規化 xyxy, スコア。`classes` と表示する `show` を指定 | boxes |
| `alpha` | 前景の度合い（`sigmoid` で確率に） | mask（`cutout`） |
| `segmap` | セマンティック・セグメンテーションの logits (1, クラス数, h, w)。画素ごとに最大のクラスで塗る。`labels` にクラス名の並び、`output` に出力名 | segmap（凡例つき） |
| `depth` | 深度。`inverse`（大きいほど遠い深度を反転）、`intrinsics`（内部パラメータの出力名、あれば画角を出す） | depth |
| `embedding` | 特徴ベクトル・確率（ReID や cascade の分類で使う） | － |

**部品が足りない時**は、`web/onnx_generic.js` の `POST`（前処理なら `preprocess`）と `adapters.py` の `POST`（`preprocess`）に**同じ名前で両方**足す。片方だけだと、その実行場所でしか動かない。

ブラウザの WebGPU 実行では、前処理の正規化などを GPU（`gpuPreprocess`）で行う。GPU 版が扱うのは上の表の指定（`batch`・`seq` を除く）だけで、`pre` に新しい指定や縮小方法を足したモデルは自動で CPU の前処理（`preprocess`）になる。GPU でも速くしたい時は `onnx_generic.js` の `GPU_PRE_KEYS` / `GPU_PRE_RESIZE` と shader に足し、CPU 版と入力が一致することを確かめる。

## 3. 専用の adapter — ライブラリのプロセッサを使うモデル

SAM・Grounding DINO・VLM のように、前処理・後処理がライブラリにあるモデルは、adapter を書いて登録する。

**ブラウザ**（`web/worker.js` の `ADAPTERS`）

```js
"tjs-xxx": {
  // image: "bitmap" なら ImageBitmap で、省略すると transformers.js の RawImage で受け取る
  load: async (e, device, onProgress) => ({ model: ..., proc: ... }),   // e = models.json の1件、device = "webgpu" | "wasm"
  async run(st, img, params, e) { return { kind: "boxes", items: [...] }; },
},
```

- どの経路で動くかは、画面のモデルの説明欄（「実行: onnxruntime-web を直接」「実行: transformers.js 4.3.0（tjs-…）」など）と、結果欄・記録の「実行場所」に出る
- ライブラリは Worker ごとに分かれている: `adapter: "onnx"` は onnxruntime-web の Worker、それ以外は transformers.js の Worker（`lib: "3"` なら 3.8.1、既定は 4.3）
- transformers.js のモデルは `repo` と `dtype`（`{"webgpu": "fp16", "wasm": "q8"}` など）を書き、読み込みは `tjsOpts(e, device, onProgress)` を渡す

**サーバー**（`adapters.py`）

```python
class HfXxx(Adapter):
    device = DEVICE                      # 結果に出す実行デバイス（"mps" / "cpu" / "coreml" など）
    def load(self): ...                  # self.e が models.json の1件
    def run(self, im, p) -> dict: ...    # im: PIL.Image（RGB）、p: 画面のパラメータ
ADAPTERS["hf-xxx"] = HfXxx
```

## 4. 結果の形式（ブラウザとサーバーで共通）

座標は、`run` に渡された画像のピクセル。画像は Blob（PNG）か data URL。

| kind | 中身 |
| --- | --- |
| `boxes` | `items: [{label, score, box: [x1, y1, x2, y2], keypoints?: [[x, y, 可視度] × 17]}]` |
| `mask` | `mask`（白が前景）、`score?`、`cutout?`（背景除去なら true） |
| `depth` | `image`（明るいほど近い）、`note?` |
| `segmap` | `image`（領域ごとに色分け、透明＝領域なし）、`count`、`legend?`（`[{label, color, count, area}]`。あれば凡例を出す）、`subtask?`（`semantic` / `panoptic`） |
| `labels` | `items: [{label, score, abs?}]`（score の大きい順） |
| `text` | `text` |

新しい種類の結果を返す時は、画面側の描き方も足す（[ADDING_UI.md](ADDING_UI.md) の「結果の種類」）。

## 5. 検出のあとに小さな分類をつなぐ（`cascade`）

検出モデルに `cascade` を書くと、指定したクラスの枠を切り出して分類モデル（`task: "cascade"`、`post.type: "embedding"`）にかけ、枠の表示に状態を足す（例: 目 → 開 / 閉）。

```json
"cascade": [
  { "id": "ocec", "on": "eye",  "model": "ocec-p", "name": "目の開閉（OCEC）", "yes": "開", "no": "閉" },
  { "id": "whc",  "on": "hand", "model": "whc-4",  "name": "手を振る（WHC、要追跡）", "yes": "手を振る", "no": "", "seq": 4 }
]
```

`seq` を付けたモデルは、追跡の ID ごとに切り出しをためて、その枚数そろったら判定する（動画で追跡を選んだ時だけ）。分類はブラウザで動く（サーバーの検出でも、切り出しと分類はブラウザ）。

## 6. 重みの置き場所とライセンス

| 置き場所 | 使う時 | 注意 |
| --- | --- | --- |
| Hugging Face（`repo` + `file`） | 公開されていて CORS で取れる | 一番楽。ブラウザは各端末が直接取る |
| `web/` に同梱（`path` + `url`） | 配布元がブラウザから取れない（GitHub Releases は CORS 不可など）で、**再配布できるライセンス**（MIT、Apache-2.0 など） | 出典とライセンスを同じフォルダの README に書く（例: `web/pinto/README.md`） |
| `models/` に置いてサーバーだけが配る（`server_file`） | 再配布したくない・できない重み（AGPL の YOLO26 など） | `models/` は `.gitignore` 済み。作り方のスクリプトを `tools/` に置く（例: `tools/export_yolo26_dynamic.py`） |

## 7. 速く動かす（ブラウザ）

汎用 ONNX（`adapter: "onnx"`）で書けば、次は**自動で**かかる。モデルごとに作業は要らない。

| 自動でかかるもの | 中身 | かからない時 |
| --- | --- | --- |
| fp16 | 実行設定の既定。`onnx.file_fp16` があればそれを使う | fp16 版が無い（fp32 のまま） |
| graph capture | 2回目以降、記録した GPU の命令をまとめて流す（スマホで特に効く） | CPU に回るノードがあるモデル（自動で外れ、実行場所に「graph capture 不可」） |
| 入力の形の固定 | ONNX の可変の次元（`batch_size` など）を、`pre.size` から決まる大きさに固定 | `pre.dynamic`・`batch`・`seq`・`keep_aspect` のモデル |
| GPU の前処理 | 縮小は canvas、正規化などは GPU（`?pre=` で切り替え） | GPU 版が知らない `pre` の指定がある（自動で CPU） |
| 連続実行のパイプライン化 | 前のフレームの推論中に次を送る | － |

transformers.js の adapter（`tjs-*`）にはどれもかからない（1回ずつライブラリに任せる）。**前処理・後処理が単純なモデルは、transformers.js にあっても汎用 ONNX で書く**（SegFormer は後処理の部品 `segmap` を足して 981ms → 42ms。HF の `onnx/` にある ONNX と `preprocessor_config.json` の値をそのまま `onnx`・`pre` に写せばよい）。

モデルを足した時に見ること（どれも画面だけでできる）

1. **経路**: モデルの説明欄の「実行: …」が `onnxruntime-web を直接` になっているか
2. **速さ**: 「速度を測る」で 20 回以上。記録の「実行場所」に実際に使った設定（`fp16 graph 前処理GPU…`）、列に前処理・モデル実行・後処理が出る
3. **graph capture 不可と出たら**: URL に `?profile=1` を付けて実行設定を「fp32（graph capture なし）」にして測ると、表に「CPU に回ったノード」が出る。出力の頭（Mod・Range・Cast など、候補の選び出し）なら、`onnx.cut` でその手前の値で切り、選び出しを JS の後処理で行うと使えるようになる（YOLO26-pose で実施）
4. **どこが重いか**: 同じ `?profile=1` の表の「演算ごとの GPU 時間」。畳み込み・行列積が大半なら計算そのものが重い（入力を小さくする、fp16 版、軽いモデルに替える）。後処理の列が大きければ JS の部品を見直す
5. **結果が合っているか**: 移す前の経路（transformers.js・サーバー）と同じ画像で、件数・クラス・面積を比べる。前処理の違いは `?pre=cpu` で CPU の前処理と比べられる。**graph capture が使えた時は、実行設定を「fp16（graph capture なし）」にした時と出力が同じか必ず見る**（DA3 small は graph capture で深度が2値に壊れた。その時は `onnx.graph_capture: false`）
6. **スマホで**: Mac で速くてもスマホで逆になることがある（`?pre=gpu` は Galaxy Z Fold6 で遅くなった。DETR panoptic は Adreno 750 で WebGPU の shader が作れない）。記録の Markdown を貼れば比べられる

## 8. 足したあとの確認

1. `scripts/serve.sh` でサーバー版を開き、ブラウザ実行・サーバー実行（`where` に書いた分）で結果が出ること
2. 汎用 ONNX で両方に書いたなら、同じ画像でブラウザとサーバーの結果がほぼ同じこと（縮小の補間の違いで境界の検出は少し変わる）
3. Hugging Face のファイル（`repo` + `file`）を足した・変えたなら `python3 tools/update_hashes.py` で `sha256` を更新する
4. `python3 build.py` で1ファイル版を作り直してコミットする（作り忘れは GitHub Actions が落とす）
5. サーバーだけのモデル（`where: ["server"]` や `server_file`）が、静的版（サーバーなし）の画面に出ないこと
