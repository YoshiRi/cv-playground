# 技術メモ（仕組み・実測・分かったこと）

README から分けた詳しい話。モデルやタブの足し方は [ADDING_MODELS.md](ADDING_MODELS.md) と [ADDING_UI.md](ADDING_UI.md)、今後の課題は [TODO.md](TODO.md)。

## 追跡（ByteTrack / BoT-SORT）

`web/tracker.js` は Ultralytics の `ultralytics/trackers`（カルマンフィルタ、2段階の対応付け、未確定の対象、見失いと破棄、重複の除去）を JS に移したもの。設定も Ultralytics の既定（`track_high_thresh` 0.25、`track_low_thresh` 0.1、`new_track_thresh` 0.25、`track_buffer` 30、`match_thresh` 0.8、`fuse_score`）と同じで、閾値スライダーが high / new の値になる。

- **同じ検出列を与えると Ultralytics 8.4.163 と一致する**: 647 フレームの動画（YOLO26n の検出）で、ByteTrack・BoT-SORT とも全フレームで ID が一致し、枠の差は最大 0.005px（丸め誤差）。確認手順は `tools/tracker_reference.py` と `tools/compare_tracker.mjs`
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
- 汎用 ONNX のモデルは「実行設定」で fp16 版（YOLO26 は `model_fp16.onnx`）と WebGPU の graph capture（記録した GPU コマンドをまとめて流す）を選べる。2026-09-28 より前は graph capture が実際には効いていなかった（下の「モデル実行の内訳」）。効き方の測り直しはそちら
- 実行設定の「CPU（WASM）」: 小さいモデルは WebGPU より速いことがある（以下は 2026-09-28 に入力の形を固定する前の値。今は DEIMv2 Atto 320 が WebGPU で 11ms、graph capture で 7ms）。M4 Mac の Chrome では WebGPU のモデル実行が大きさによらず約 24ms で頭打ちになり（GPU に命令を出して結果を読み戻す固定の手間）、WASM（4スレッド）は DEIMv2 Atto 192 で 8ms、Atto 320 で 16ms、Femto 416 で 20ms、Pico 640 で 45ms（WebGPU 38ms）。YOLO26n-pose は WebGPU 24〜30ms、WASM 50ms
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
- SmolVLM 256M は WebGPU の fp16 だと画像によって意味のない文章になる（街の写真で「150s on the street side…」）。当初は transformers.js の版（4.3 → 3.8.1）の問題と見たが、fp32 なら 4.3 でも 3.8.1 でも正常だった（2026-09-28）ので fp32 にした（約1GB）。WASM の q8 は壊れた文字列になる。モデルごとにライブラリの版を選ぶ仕組み（models.json の `lib`）は残している
- SigLIP2 は transformers.js のパイプラインだと "Invalid array length"。文字列を max_length 64 で埋めて直接呼べば動く
- YOLO26 の前処理は Ultralytics 公式と同じ letterbox にしている（onnx-community の ONNX に付く設定は引き伸ばし）
- Depth Anything 3 の ONNX は入力が (batch, 視点数, 3, H, W) の多視点前提で、深度に加えてカメラの内部・外部パラメータも返す
- SAM 系の ONNX の decoder は点のバッチ次元が 1 固定なので、「全体を自動分割」は格子点を1点ずつ回している
- 1ファイル版を file:// で開くと、Blob URL からの module Worker は起動できない。import を除いた本文を classic Worker で起動している
- onnxruntime（Python）は外部データ（.onnx_data）が本体と別ディレクトリにあると拒否する。HF のキャッシュは blobs に分かれるので `models/` に実ファイルで落としている

## 端末ごとの速度（報告値、YOLO26n・ブラウザ・WebGPU・1フレーム）

| 端末 | 1フレーム |
| --- | --- |
| Pixel 6a | 150ms |
| Galaxy S25 Ultra | 90ms |
| iPad Pro | 70ms |
| M1 MacBook Air | 60ms（内訳: 前処理 8.5ms・モデル実行 61ms。WASM 190ms。「fp32 + graph capture 63ms」は graph capture が効いていなかった頃の値） |
| M4 Mac mini | 20〜30ms |
| Galaxy Z Fold6（SM-F956Q、Adreno 750） | 64ms（graph capture 50ms、fp16 + graph capture 42ms。2026-09-28） |

GPU の性能順にきれいに並ぶので、固定の手間より計算量が主因。YOLO26n（640）は1回 5〜6 GFLOPs なので実効 100 GFLOP/s 前後で、ブラウザの WebGPU（onnxruntime-web）の畳み込みの効率はまだ低い。入力を小さくする（入力サイズ可変の YOLO26）のが一番効くはず。

## モデル実行の内訳（詳細計測、`?profile=1`）と速くした点（2026-09-28）

URL に `?profile=1` を付けると、汎用 ONNX のブラウザ実行で onnxruntime の profiler を有効にし、ベンチマークの結果に「演算ごとの GPU 時間・CPU に回ったノード・GPU との転送」を付ける（画面の表と、実行履歴の JSON の `profile`、CSV の `gpu_ms`）。計測の手間で遅くなる（M4 の YOLO26n で 18ms → 29ms）ので、速さの比較は通常の URL で測る。graph capture の再生中は演算ごとの記録が出ない。

- onnxruntime-web 1.30 の `ort.webgpu` は、旧来の JSEP（JavaScript の WebGPU 実装）ではなく C++ の WebGPU EP を wasm にしたもの。`ort.env.webgpu.profiling` は効かず、セッションの `enableProfiling` を使う。記録（Chrome trace の JSON）は `endProfiling()` の時に console に出るだけなので拾っている（emscripten が読み込み時に console.log を覚えるので、先に中継の関数に替えておく）。GPU の時間は `timestamp-query` で取られるが、時刻は後からまとめて取るので実行の区切りとは合わない（命令の数で区切って1回ぶんにしている。命令の間の空きは正しく出ないので出さない）
- 既定: 畳み込みは NHWC（入力の NCHW → NHWC の並べ替えが最初に1回入り、M4 で 1ms）。`preferredLayout: "NCHW"` はやや遅い

**直した点**

1. **graph capture が効いていなかった**: `executionProviders: ["webgpu"]`（文字列）だと、onnxruntime-web 1.30 は `enableGraphCapture` を WebGPU EP に渡さない（EP の設定のログが `graph capture enable: 0`）。`[{ name: "webgpu" }]` で渡すと効く。以前の「fp32 + graph capture」の計測（M1 で 63ms など）は実際には graph capture なしだった
2. **全部のノードが WebGPU で動くモデルだけ graph capture できる**。CPU に回るノードがあるとセッションが作れないので、その時は graph capture なしで作り直し、実行設定の表示に「このモデルは graph capture 不可」と出す。YOLO26n・DEIMv2 は可、YOLO26n-pose（後処理の Mod・Range・Cast などが CPU）、入力サイズ可変の YOLO26、Depth Anything 3 は不可
3. **入力の形の固定**: ONNX の入力は `batch_size`・`N, H, W` などの可変の次元で書かれていることが多い。入力の大きさが決まっているモデルは、一度作ったセッションの `inputMetadata` から名前を読み、`freeDimensionOverrides` で固定して作り直す。形の計算（Gather・Unsqueeze・Concat など）が CPU に回らなくなり、DEIMv2 は graph capture もできるようになる
4. graph capture の時の出力の読み戻しを1つずつ待たずにまとめて待つ（YOLO26n で 2ms）

M4 Mac mini の Chrome、ベンチ（街の画像、20 回の中央値、1回の推論 ms）

| モデル | 前 | 形の固定 | + graph capture | + fp16 |
| --- | --- | --- | --- | --- |
| YOLO26n | 24.0 | 23.7 | 19.2 | 17.3 |
| DEIMv2 Atto 192 | － | 9.9 | 4.7 | 4.7 |
| DEIMv2 Atto 320 | 19.3 | 11.6 | 7.5 | 7.5 |
| DEIMv2 Femto 416 | － | 15.5 | 10.2 | 10.2 |
| DEIMv2 Pico 640 | 35.4 | 25.5 | 19.5 | 19.5 |
| YOLO26n-pose | 24.8 | 23.2 | 不可 | 21.2（fp16 のみ） |

（DEIMv2 は fp16 版が無いので、fp16 を選んでも fp32 のまま。検出の件数はどの設定でも同じ）

Galaxy Z Fold6（SM-F956Q、Snapdragon 8 Gen 3 / Adreno 750、Android 16、Chrome 153、静的版）、ベンチ 50 回の中央値（ms）

| モデル | 標準 | graph capture | fp16 + graph capture |
| --- | --- | --- | --- |
| YOLO26n | 64（前処理 9.1・モデル実行 55.6） | 50（12.5・37.3） | 42（12.8・29.0） |
| DEIMv2 Atto 320 | 40（2.7・37.3） | 19（3.1・16.1） | 19（fp16 版なし） |

前処理（縮小・正規化、JS）は入力の画素数にほぼ比例（640×640 で 9〜13ms、320×320 で約 3ms）。fp16 + graph capture の YOLO26n では1回 42ms の約3割が前処理。YOLO26n の前処理は後の計測ほど長い（9 → 12ms、graph capture とは関係のない同じ処理なので、発熱で CPU が遅くなった可能性）

詳細計測（profiler あり、1回の平均）: YOLO26n は ONNX の実行 79ms のうち GPU の命令の合計は 30ms（畳み込み 78%）。DEIMv2 Atto は 91ms のうち 13ms（命令 624 個）。**スマホでは GPU の計算より、命令を1つずつ出す CPU 側の手間の方が大きい**ので、それをまとめる graph capture が M4 以上に効く（DEIMv2 は半分以下）。graph capture 後の YOLO26n は GPU の計算（fp32 で約 30ms、fp16 で減る）が主になる

**分かったこと**

- M4 の YOLO26n（640、fp32）1回あたり: GPU の命令 325 個、GPU 時間の合計 16.6ms（profiler あり）。畳み込み 60%、Transpose 12%（24 個。多くは NHWC と NCHW の並べ替え）、Concat 10%、Slice 6%、TopK 3%（並べ替えで約 100 個の命令）。CPU に回るのは出力の頭の小さい3ノードだけ
- fp16 版は GPU 時間の合計が 16.6 → 12.3ms（畳み込み 9.9 → 6.7ms）に減るが、graph capture なしだと M4 では1回の時間はほぼ変わらない（命令を出す手間と待ちが効いている）
- **GPU を休ませると遅くなる**: Worker で YOLO26n（graph capture）を続けて回すと 1回 12.5ms だが、実行の間に 5ms でも空けると 21.5ms になる（20ms 空けても同じ）。画面の実行は前処理・結果の受け渡しで間が空くので、ベンチの 1回（graph capture で約 17ms）は続けて回した時より遅い。前処理と推論を重ねて GPU を休ませない（パイプライン化）と縮む余地がある
- 前処理（縮小・正規化）は M4 で 2.7ms（640×640）。M1 では 8.5ms の報告があり、端末によっては大きい

## Grounding DINO の候補の扱い

- サーバー（transformers、base）: 標準の後処理は閾値を超えた単語をつなげて「orange lemon」のような混ざった名前を返すので、候補ごとの単語の範囲で確率の最大を比べ、枠ごとに候補を1つ選ぶ（`adapters.py` の `HfGdino`）
- ブラウザ（transformers.js、tiny の ONNX）: 文に候補を並べると先頭の候補しか正しいスコアにならない（猫の写真で "remote control. cat. sofa." だと cat の最大が 0.09、"cat." だけなら 0.83）。候補ごとに1回ずつ推論してまとめている（候補の数だけ時間がかかる）

## セグメンテーション（セマンティック・パノプティック）

- ブラウザ: transformers.js の image-segmentation パイプライン（SegFormer B0、DETR ResNet-50 panoptic）。`subtask` を明示すると 3.8.1・4.3.0 とも関数名の文字列を呼ぼうとして "x is not a function" で落ちるので、省いてモデルの後処理から自動で選ばせる。DETR panoptic の config は stuff クラスの名前が `LABEL_184` のように空なので、cocodataset/panopticapi の `panoptic_coco_categories.json` で名前を付ける（models.json の `label_map`）
- サーバー: EoMT（DINOv3）。パノプティックはパイプラインで正しいが、セマンティックはパイプラインでもプロセッサ直呼びでも、横長の画像で塗り分けが縦につぶれる（プロセッサは長辺 512 に縮めて右下を余白で埋めた正方形にするのに、後処理は余白ごと元の大きさに引き伸ばす。transformers 5.17）。入力の大きさで塗り分けを出し、余白を切ってから戻している（黒い四角の位置で横長・縦長とも一致を確認）
- 色はクラス名から決まる（ブラウザの `segColor` とサーバーの `seg_color` が同じ式）。パノプティックの同じクラスの物は明るさを変えて塗る

## 書き出しと速度を測る（`web/export.js`）

- 実行の記録は1回の実行（連続実行・ベンチマークは1まとめ）を1行にし、ブラウザ内（localStorage）に最新 500 件まで残す。CSV は BOM 付き UTF-8（Excel で文字化けしない）で、端末の列（ブラウザ、OS、機種名、GPU、CPU スレッド数、WASM スレッド数、配り方、アプリの版など）を毎行に入れるので、別の端末の CSV をそのまま連結して比べられる。機種名は Chrome 系の User-Agent Client Hints（Android なら機種名が取れる）、GPU は WebGPU のアダプタ情報
- 連続実行の統計は1フレーム目（モデルの読み込み・初期化を含む）を除いたフレームごとの推論時間から出す（平均・中央値・p90・最小・最大）
- 速度を測る: 決まった画像（transformers.js-docs の city-streets.jpg、800×800）で、ウォームアップ 3 回のあと N 回。クリックで点を置くタブ（プロンプト（SAM））は条件が決まらないので対象外。既定で選ぶのは models.json の `bench: true`（M4 Mac の Chrome で5モデル・各 20 回が約 46 秒、ダウンロード込み）
- 画像の保存: 表示中の canvas（「表示」の選択を反映）に説明の帯を付けた PNG。スマホで共有シートが使える時（`navigator.canShare`）は共有シートから「写真に保存」などを選べる
