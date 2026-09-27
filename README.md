# CV Playground

画像認識（CV）のモデルを、ブラウザから手軽に試して比べるためのページ。
同じ画面で **「ブラウザ内で実行（その端末の WebGPU / WASM）」** と **「サーバーで実行」** を選べ、読み込み時間・推論時間・fps を履歴に残して比べられる。
処理負荷が分からないモデルを、スマホや PC で実際に動かして確かめる用途を想定している。

- タスク: 物体検出 / 人物の姿勢 / 手・目（PINTO の超軽量モデル）/ テキスト指定検出 / クリックで切り出し（SAM 系、全体の自動分割つき）/ 深度推定 / ゼロショット分類 / 背景除去 / 画像の説明・質問（VLM）
- 入力: 画像ファイル、動画ファイル、カメラのライブ映像。動画・カメラは連続実行して結果を重ね、fps を表示する
- 追跡: 検出・姿勢・テキスト指定検出の連続実行に ByteTrack / BoT-SORT をかけ、ID と軌跡を表示する（ブラウザ・サーバーどちらの検出にも使える）
- 表示: 結果を重ねる / 結果だけ / 切り抜き / 元画像
- サーバー不要の1ファイル版（`dist/cv-playground.html`、約70KB）もある。モデルは各ブラウザが Hugging Face から直接取得する

UI は日本語。

## 使い方

### 配り方は3通り（どれも web/ の同じコード）

| 配り方 | 中身 | 違い |
| --- | --- | --- |
| サーバー版 | `server.py` が `web/` と API を配る | サーバー（Mac など）で動くモデルも選べる。COOP/COEP ヘッダで WASM が複数スレッド |
| 静的版 | GitHub Pages などが `web/` をそのまま配る（**https://yoshiri.github.io/cv-playground/**） | API に届かないのでブラウザ実行のモデルだけ。WASM の複数スレッドは同梱の coi-serviceworker で有効にする |
| 1ファイル版 | `python3 build.py` で作る `dist/cv-playground.html` | ファイルを開くだけで動く（file:// 可）。ブラウザ実行のみ、WASM は1スレッド |

サーバーの有無はページが実行時に判定する（`api/models` に届くか）。1ファイル版の作り忘れは GitHub Actions（`.github/workflows/check-dist.yml`）で検出する。

### サーバーも使う

```sh
uv venv --python 3.12 .venv && uv pip install --python .venv/bin/python -r requirements.txt
cp .env.example .env            # 環境ごとの設定（任意。.env は git に入れない）
scripts/serve.sh                # http://127.0.0.1:8010（--bg で裏で起動）
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

## 入力サイズ可変の YOLO26（このサーバーのみ）

onnx-community の YOLO26 の ONNX は入力が 640×640 固定なので、動画の「処理解像度」を 960 にしてもモデルには 640 に縮めて入る（検出の細かさは変わらず、前処理が重くなるだけ）。
`tools/export_yolo26_dynamic.py` で Ultralytics から入力サイズ可変・NMS 不要（end2end）の ONNX を書き出して `models/ultralytics/` に置くと、
「YOLO26n（入力サイズ可変）」「YOLO26n-pose（入力サイズ可変）」が使え、「モデル入力（長辺）」を 320 / 480 / 640 / 960 から選べる（既定 640）。

- 前処理は Ultralytics の推論と同じ長方形の letterbox（長辺を合わせ、32 の倍数まで余白で埋める）。16:9 なら 640×384 で、正方形の 640×640 より計算が約 4 割少ない。Ultralytics の `predict` と枠が ±1px で一致することを確認
- M4 Mac の Chrome での YOLO26n-pose のモデル実行: 固定 640 が 24ms、可変 640 が 19ms、480 が 18ms、320 が 15ms、960 が 27ms（Mac の WebGPU は約 15〜20ms の下限があり差が出にくい。計算が支配的なスマホでは画素数にほぼ比例して効くはず）。960 では街の写真で信号機などの小さい物体が増える
- YOLO26 の重みは AGPL-3.0 なので、書き出した ONNX はリポジトリに入れず、このサーバーの `/local-models/` からだけ配る。1ファイル版・GitHub Pages には出ない

## 手・目（PINTO の超軽量モデル）

[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) の DEIMv2 Wholebody34（体・頭・顔・目・手など 34 クラスを1つで検出、Atto は 2MB）で検出し、
目や手を切り出して小さな分類モデルにかける: OCEC（目の開閉、0.1MB）、PGC（指差し、0.5MB）、WHC（手を振る、同じ手の 4 フレーム、1.1MB。追跡の ID を使う）。
モデルはブラウザから直接取れる配布元が無いので `web/pinto/` に同梱している（MIT。詳細は `web/pinto/README.md`）。models.json の `cascade` で「検出 → 切り出し → 分類」をつなぐ。

## スマホで速くするには

- 連続実行の結果欄に「内訳: フレーム取り込み / 前処理 / モデル実行 / 後処理」を出している（汎用 ONNX のモデル）。まずこれでどこが重いかを見る
- 汎用 ONNX のモデルは「実行設定」で fp16 版（YOLO26 は `model_fp16.onnx`）と WebGPU の graph capture（記録した GPU コマンドをまとめて流す）を選べる。M4 Mac の Chrome では YOLO26n-pose のモデル実行が 38ms → 27〜29ms（fp16）→ 25〜28ms（fp16 + graph capture）。ばらつきが大きく、スマホでの効き方は未確認
- 実行設定の「CPU（WASM）」: 小さいモデルは WebGPU より速いことがある。M4 Mac の Chrome では WebGPU のモデル実行が大きさによらず約 24ms で頭打ちになり（GPU に命令を出して結果を読み戻す固定の手間）、WASM（4スレッド）は DEIMv2 Atto 192 で 8ms、Atto 320 で 16ms、Femto 416 で 20ms、Pico 640 で 45ms（WebGPU 38ms）。YOLO26n-pose は WebGPU 24〜30ms、WASM 50ms
- WASM の複数スレッドは crossOriginIsolated の時だけ使える。server.py は COOP/COEP ヘッダを付けるので使えるが、GitHub Pages ではヘッダを付けられないので1スレッドになる
- 動画・カメラの描画は新しいフレームが来た時だけにしている（画面の更新ごとに描くと、120Hz のスマホでは毎秒120回描いて推論と GPU を取り合う）

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
