# CV Playground

画像認識（CV）のモデルを、ブラウザから手軽に試して比べるためのページ。
同じ画面で **「ブラウザ内で実行（その端末の WebGPU / WASM）」** と **「サーバーで実行」** を選べ、読み込み時間・推論時間・fps を履歴に残して比べられる。
処理負荷が分からないモデルを、スマホや PC で実際に動かして確かめる用途を想定している。

- タスク: 物体検出 / 人物の姿勢 / テキスト指定検出 / クリックで切り出し（SAM 系、全体の自動分割つき）/ 深度推定 / ゼロショット分類 / 背景除去 / 画像の説明・質問（VLM）
- 入力: 画像ファイル、動画ファイル、カメラのライブ映像。動画・カメラは連続実行して結果を重ね、fps を表示する
- 追跡: 検出・姿勢・テキスト指定検出の連続実行に ByteTrack / BoT-SORT をかけ、ID と軌跡を表示する（ブラウザ・サーバーどちらの検出にも使える）
- 表示: 結果を重ねる / 結果だけ / 切り抜き / 元画像
- サーバー不要の1ファイル版（`dist/cv-playground.html`、約70KB）もある。モデルは各ブラウザが Hugging Face から直接取得する

UI は日本語。

## 使い方

### ブラウザだけで使う（サーバーなし）

**https://yoshiri.github.io/cv-playground/ で開ける**（GitHub Pages。スマホ可）。手元では `dist/cv-playground.html` をブラウザで開く（file:// でも、GitHub Pages などの静的ホスティングでも動く）。ブラウザ実行のモデルだけが選べる。
WebGPU がある Chrome / Edge / Safari 推奨（無ければ WASM で遅く動く）。

### サーバーも使う

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python server.py      # http://127.0.0.1:8010
```

- サーバー側のモデルは初回に Hugging Face から取得する（合計で数GB）。最大3つをメモリに置き、超えたら古い順に外す
- 別の端末（スマホなど）から開く時は **https が必要**（WebGPU とカメラのため）。例えば Tailscale なら `tailscale serve --bg --https=8443 http://127.0.0.1:8010`
- Apple Silicon では MPS / CoreML を使う。ほかは CPU（CUDA は未対応・未検証）

環境変数

| 変数 | 意味 | 既定 |
| --- | --- | --- |
| `CVPG_PORT` | 待ち受けポート | 8010 |
| `CVPG_MAX_LOADED` | 同時に読み込んでおくサーバーのモデル数 | 3 |
| `CVPG_OLLAMA_MODEL` | 「Ollama の VLM」で使うモデル | `qwen3-vl:8b` |
| `OLLAMA_HOST_URL` | Ollama の URL | `http://127.0.0.1:11434` |
| `CVPG_GPU_LOCK` | このファイルがあれば「GPU を別の処理が使用中」と画面に出す（任意） | なし |

## 構成

| ファイル | 役割 |
| --- | --- |
| `web/models.json` | **モデルの一覧（ブラウザとサーバーで共通）**。実行できる場所（where）と adapter を書く |
| `adapters.py` | サーバー側の adapter（汎用 ONNX の前処理・後処理の部品、transformers / Ollama を使う専用実装） |
| `web/onnx_generic.js` | ブラウザ側の汎用 ONNX（onnxruntime-web）。adapters.py と同じ部品を持つ |
| `web/worker.js` | ブラウザ側の adapter の登録と実行（Web Worker。onnxruntime-web 用と transformers.js 用で Worker を分ける） |
| `server.py` | FastAPI。静的ページの配信と `/api/run`・`/api/status`・`/api/unload` |
| `web/catalog.js` | タスクの一覧と既定値 |
| `web/app.js` | 画面、描画、動画・カメラの連続実行、実行履歴 |
| `build.py` | サーバーなし版の1ファイル HTML を作る（`python3 build.py`。web/ を直したら作り直す） |

## モデルの足し方

どのモデルも「adapter = 読み込み（load）＋ 実行（run: 画像とパラメータ → 結果）」という同じ形で扱い、結果の形式はブラウザとサーバーで共通（座標は入力画像のピクセル）。

| kind | 中身 |
| --- | --- |
| `boxes` | `items: [{label, score, box: [x1, y1, x2, y2], keypoints?}]` |
| `mask` | `mask`（白＝前景の PNG）、`score?`、`cutout?` |
| `depth` | `image`（明るい＝近い PNG）、`note?` |
| `labels` | `items: [{label, score}]` |
| `text` | `text` |
| `segmap` | `image`（色分けした PNG）、`count` |

1. **素の ONNX モデル**（YOLO26、BiRefNet、Depth Anything 3 など）: `web/models.json` に `"adapter": "onnx"` で1件足すだけ。
   - `onnx`: HF のリポジトリとファイル（外部データがあれば `data`）
   - `pre`: `resize`（`letterbox` / `stretch` / `keep_aspect`）、`size` または `short`＋`multiple`、`scale`、`mean`・`std`、入力名 `input`、次元を足す `add_dims`
   - `post.type`: `yolo_detect` / `yolo_pose` / `alpha` / `depth`（出力名や `inverse` などの設定つき）
   - `where` に `browser` と `server` を両方書けば、同じ手順でブラウザとサーバーを比べられる
   - 部品が足りなければ `adapters.py` と `web/onnx_generic.js` の `POST`（または前処理）に同じものを足す
2. **ライブラリのプロセッサが要るモデル**（SAM、Grounding DINO、VLM など）: ブラウザは `web/worker.js` の `ADAPTERS`、サーバーは `adapters.py` の `ADAPTERS` に load / run を書き、models.json から adapter 名で指す。

## 追跡（ByteTrack / BoT-SORT）

`web/tracker.js` は Ultralytics の `ultralytics/trackers`（カルマンフィルタ、2段階の対応付け、未確定の対象、見失いと破棄、重複の除去）を JS に移したもの。設定も Ultralytics の既定（`track_high_thresh` 0.25、`track_low_thresh` 0.1、`new_track_thresh` 0.25、`track_buffer` 30、`match_thresh` 0.8、`fuse_score`）と同じで、閾値スライダーが high / new の値になる。

- **同じ検出列を与えると Ultralytics 8.4.163 と一致する**: 647 フレームの動画（YOLO26n の検出）で、ByteTrack・BoT-SORT とも全フレームで ID が一致し、枠の差は最大 0.005px（丸め誤差）。確認手順は `tools/`
- **BoT-SORT + ReID**: 検出を切り出して人物 ReID モデル（OSNet x0.25 MSMT17、約1MB、HF の `anriha/osnet_x0_25_msmt17`）で 512 次元の特徴にし、Ultralytics と同じ規則（特徴の指数移動平均 0.9、IoU ≥ 0.5 かつ cos 類似度 ≥ 0.6 の組だけ見た目の距離を使う）で対応付ける。同じ特徴を与えた時も Ultralytics と全フレーム一致（2本の動画、647・596 フレーム）。Mac の Chrome で1フレーム約 26ms（バッチ 16 固定のため人数によらずほぼ一定）
  - Ultralytics の ReID は「枠がまだ重なっている近くの候補」の中で見た目を比べるだけなので、画面外に出て戻ってきた人に同じ ID を付け直す用途には向かない（今回の動画でも ID の数はほぼ変わらなかった）
- BoT-SORT のカメラ移動の補正（GMC）は無い。そのため ReID なしの BoT-SORT はカルマンフィルタの状態が xywh になる点だけが ByteTrack と違う。カメラが動く映像では Ultralytics の BoT-SORT と結果が変わる
- 連続実行は新しいフレームが表示された時だけ処理する。推論が動画より速いと同じフレームを何度も追跡器に渡してしまい、速度の推定が狂って ID が切り替わりやすくなるため（12fps の動画で ID の数が 9 → 5）。表示の fps は動画・カメラのフレームレートが上限になり、推論だけの上限も並べて出す

## 実測の例（M4 Mac mini 32GB の Chrome）

| モデル | 実行場所 | 静止画の推論 | 動画（640×360）の fps |
| --- | --- | --- | --- |
| YOLO26n | ブラウザ（WebGPU） | 45ms | 39 |
| YOLO26n-pose | ブラウザ（WebGPU） | 37ms | 40 |
| YOLO26n | サーバー（CoreML） | 19〜40ms | 30（通信込み） |
| RF-DETR Large | サーバー（MPS） | 205ms | 10 |
| EdgeTAM（クリックで切り出し） | ブラウザ（WebGPU） | 240ms | 5 |
| Depth Anything 3 small | ブラウザ（WebGPU） | 370ms | 2 |
| SAM 2.1 large | サーバー（MPS） | 1.0s | － |

## 分かったこと・既知の問題

- onnxruntime（Python）の CoreML EP は、既定の NeuralNetwork 形式だと YOLO26 の出力が壊れる（全スコアが負になり何も検出しない）。`ModelFormat=MLProgram` なら CPU と一致し約2倍速い。BiRefNet と Depth Anything 3 は MLProgram への変換に失敗するので CPU で動かしている
- transformers 5.17 の zero-shot-object-detection パイプラインは Grounding DINO でスコアが極端に低い。プロセッサに候補名のリストを直接渡すと正常
- transformers.js 4.3 × Mac の Chrome（WebGPU の `maxStorageBuffersPerShaderStage` = 10）で BiRefNet が "Too many storage buffers in shader (11 > 10)"。一度失敗すると同じ Worker の以後の実行も全部失敗するので、失敗時に Worker を作り直している
- SmolVLM は transformers.js 4.3 だと WebGPU で意味のない文字列を出す。3.8.1 なら正常なので、モデルごとにライブラリの版を選べるようにしている（models.json の `lib`）
- SigLIP2 は transformers.js のパイプラインだと "Invalid array length"。文字列を max_length 64 で埋めて直接呼べば動く
- YOLO26 の前処理は Ultralytics 公式と同じ letterbox にしている（onnx-community の ONNX に付く設定は引き伸ばし）
- Depth Anything 3 の ONNX は入力が (batch, 視点数, 3, H, W) の多視点前提で、深度に加えてカメラの内部・外部パラメータも返す
- SAM 系の ONNX の decoder は点のバッチ次元が 1 固定なので、「全体を自動分割」は格子点を1点ずつ回している
- 1ファイル版を file:// で開くと、Blob URL からの module Worker は起動できない。import を除いた本文を classic Worker で起動している
- onnxruntime（Python）は外部データ（.onnx_data）が本体と別ディレクトリにあると拒否する。HF のキャッシュは blobs に分かれるので `models/` に実ファイルで落としている

## ライセンス

このリポジトリのコードは MIT License（LICENSE）。**モデルの重みは同梱せず、実行時に各配布元から取得する。各モデルのライセンスはそれぞれの配布元を確認すること**（例: YOLO26 は Ultralytics の AGPL-3.0）。
サンプル画像は Hugging Face の [Xenova/transformers.js-docs](https://huggingface.co/datasets/Xenova/transformers.js-docs) から実行時に読み込む（同梱しない）。
