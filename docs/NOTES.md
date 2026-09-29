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
- **入力サイズ可変の YOLO26 は graph capture を使えない**（2026-09-29）: Ultralytics の end2end 書き出しは、出力の頭で候補の番号を 64bit 整数で計算する（Mod・Div・Cast・GatherElements など9ノード）。onnxruntime-web 1.30 の WebGPU EP は 64bit 整数を既定で扱わない（`enable int64: 0`。JS の設定項目に無く、セッションの `extra` の `ep.webgpuexecutionprovider.enableInt64` も効かない）ので CPU に回り、graph capture を作れない。入力の形の固定（`freeDimensionOverrides`）でも、大きさを固定した書き出し（`imgsz=(384, 640)`、`dynamic=False`）でも同じ。スマホでは graph capture なしだと命令を出す手間が大きく、入力を小さくしても減るのは GPU の計算の分だけなので、fps があまり変わらない（Galaxy Z Fold6 での報告）。使えるようにするには、書き出した ONNX から出力の頭（候補の選び出し）を切り離し、JS の後処理で上位を選ぶ必要がある。onnx-community の YOLO26（640 固定）は出力の頭の作りが違い、graph capture を使える
- YOLO26 の重みは AGPL-3.0 なので、書き出した ONNX はリポジトリに入れず、このサーバーの `/local-models/` からだけ配る。1ファイル版・GitHub Pages には出ない

## 手・目（PINTO の超軽量モデル）

[PINTO_model_zoo](https://github.com/PINTO0309/PINTO_model_zoo) の DEIMv2 Wholebody34（体・頭・顔・目・手など 34 クラスを1つで検出、Atto は 2MB）で検出し、
目や手を切り出して小さな分類モデルにかける: OCEC（目の開閉、0.1MB）、PGC（指差し、0.5MB）、WHC（手を振る、同じ手の 4 フレーム、1.1MB。追跡の ID を使う）。
モデルはブラウザから直接取れる配布元が無いので `web/pinto/` に同梱している（MIT。詳細は `web/pinto/README.md`）。models.json の `cascade` で「検出 → 切り出し → 分類」をつなぐ。

## スマホで速くするには

- 連続実行の結果欄に「内訳: フレーム取り込み / 前処理 / モデル実行 / 後処理」を出している（汎用 ONNX のモデル）。まずこれでどこが重いかを見る
- 汎用 ONNX のモデルは「実行設定」で fp16 版（YOLO26 は `model_fp16.onnx`）と WebGPU の graph capture（記録した GPU コマンドをまとめて流す）を選べる。2026-09-28 より前は graph capture が実際には効いていなかった（下の「モデル実行の内訳」）。効き方の測り直しはそちら。M4 と Galaxy Z Fold6 の両方で速くなったので、2026-09-29 から既定を「fp16 + graph capture」にした（fp16 版が無いモデルは fp32、graph capture を作れないモデルは無しで動く。記録の実行場所には実際に使った方を出す）
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
- DETR ResNet-50 panoptic（transformers.js、fp32）は Galaxy Z Fold6（Adreno 750）の WebGPU で `Failed to create a WebGPU compute pipeline: [Invalid ShaderModule "Conv2dMM"] is invalid due to a previous error` になる（2026-09-29）。M4 では動くが1回 3.6 秒と重く、スマホ向きではない。Adreno 750 と onnxruntime-web 1.30 の shader の作成失敗は他でも報告がある（musetric/musetric#901）
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
| Galaxy Z Fold6（SM-F956Q、Adreno 750） | 64ms → 32ms（2026-09-29 の既定: fp16 + graph capture + 前処理 upload。途中は graph capture 50ms、fp16 + graph capture 42ms） |

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

## GPU の前処理と連続実行のパイプライン化（2026-09-29）

- **前処理を GPU で行う**（汎用 ONNX の WebGPU 実行。URL の `?pre=` で3通りを選べる）
  - `?pre=gpu`: 画像を `copyExternalImageToTexture` でテクスチャに送り、compute shader で縮小（双線形）・余白・正規化・チャンネルの並べ替えをして、モデルの入力の GPU バッファ（`ort.Tensor.fromGpuBuffer`、graph capture の入力バッファ）に直接書く。M4 では最速（YOLO26n のベンチ 17.8 → 13.5ms）だが、**Galaxy Z Fold6 ではモデル実行が 18〜28ms 遅くなった**（YOLO26n 30 → 48ms、DEIMv2 Atto 17 → 45ms。前処理そのものは 13 → 2ms）。上乗せが画像の大きさによらずほぼ一定なので、画像を GPU に送る所で待ち合わせが起きているらしい
  - `?pre=upload`（既定）: 縮小は CPU の前処理と同じ canvas で行い、8 bit の画素を `writeTexture` で送って、余白・正規化・並べ替えだけ shader で。CPU の前処理から JS のループと float の転送（4倍の量）を除いたもので、入力は CPU の前処理と完全に一致する。M4 で YOLO26n 15.2ms、DEIMv2 Pico 19.6 → 17.7ms
  - `?pre=cpu`（`?cpupre=1` も同じ）: 従来の CPU の前処理
  - Galaxy Z Fold6（ベンチ 50 回の中央値、前処理・モデル実行）: YOLO26n は upload 32ms（7.1・25）/ cpu 39ms（11・29）/ gpu 61ms（2.6・57）、DEIMv2 Atto は upload 17ms（1.7・15）/ cpu 19ms（3.6・16）/ gpu 46ms（2.5・45）。**スマホでも upload が最速**で、gpu はまた大きく遅い（再現した）。M4 との差は gpu が 1.7ms 速いだけなので、既定は全端末で upload にしている
  - GPU 版が対応している指定（縮小方法と scale・mean・std・bgr・pad_value・add_dims）だけのモデルで使い、それ以外（バッチ・フレーム列のモデル、新しく足した指定）は自動で CPU の前処理になる。記録の実行場所に、使った前処理を出す
  - `gpu` の縮小は canvas と少し違う（640 → 320 で入力の差が平均 0.35、画素値 0〜255）。サーバー（PIL）との差は CPU 2.27、GPU 2.17 でほぼ同じ。しきい値すれすれの検出はこの差でも出たり消えたりする（サッカーの写真の DEIMv2 で 33 件と 36 件）
- **連続実行のパイプライン化**（汎用 ONNX のブラウザ実行）: 前のフレームの推論中に次のフレームを取り込んで Worker に送っておく（同時に2フレームまで）。Worker は onnxruntime の実行を1つずつ順番に行う（`serial`）ので、届いていた次のフレームをすぐ始められ、結果の受け渡し・追跡・描画の間も GPU が休まない。結果はフレームの順に追跡・表示する。次のフレームの表示は送った直後から待ち始める（前の結果の処理後に待ち始めると、処理が1フレームの間隔を少しでも超えた時に1つおきになり、Galaxy Z Fold6 で 30fps の動画が 1フレーム 36ms なのに 14.5fps だった）。M4 で人物の動画を速めて流した時の連続実行（`?nopipe=1` が従来の1フレームずつ）: 6倍速（約 72fps）で YOLO26n-pose 20.1 → 39.6fps、YOLO26n + BoT-SORT + ReID 17.0 → 27.5fps。3倍速（約 36fps）で 23.9 → 32.1fps、18.6 → 23.9fps

## SegFormer を汎用 ONNX に移した（2026-09-29）

SegFormer B0（ADE20K）は transformers.js の image-segmentation パイプラインで動かしていたが、前処理（512×512 に引き伸ばして ImageNet の平均・標準偏差で正規化）も後処理（画素ごとに最大のクラス）も単純なので、汎用 ONNX（後処理の部品 `segmap`）に移した。M4 の Chrome で1回 981ms → 42ms（fp16 + graph capture。fp32 でも 56ms）。transformers.js 版は 150 クラスの出力を元画像の大きさに広げてから最大を取り、クラスごとのマスクも作るので後処理が重かった。こちらは出力の解像度（128×128）のまま最大を取って表示で広げるので、境界の細かさは少し違うが、クラスと面積はほぼ同じ（街の写真で road 50.4% と 50.9%、上位 8 クラスの順も同じ）。サーバー（adapters.py の `post_segmap`）も同じ手順

Galaxy Z Fold6（Brave 153）の連続実行（360×640 の動画）: 1フレーム 123〜161ms（前処理 6〜7・モデル実行 116〜139・後処理 21〜23ms）、6〜8fps。transformers.js 版は1回 1695ms（読み込み済み、Chrome）だったので 10 倍以上速い。スマホではモデル実行（M4 の約 3.5 倍）が主で、次は入力を 512 より小さくする余地がある。後処理（150 クラスの最大を JS で）の 21ms は GPU に移せば縮む

## 深度のリアルタイム化（2026-09-29）

- Depth Anything V2 small を transformers.js から汎用 ONNX（fp16、50MB）に移し、DA3 small とそろえて入力を「長辺を『モデル入力（長辺）』に合わせ、14 の倍数まで余白で埋める」（`letterbox_rect`、`stride: 14`、`dynamic`）にした。transformers.js の既定は短辺 518 で、16:9 の動画だと 924×518 になる。長辺 518 なら 518×294（画素は約 1/3）
- **入力の大きさが実行時に決まるモデルも形を固定する**: 最初の実行で入力の形が分かった所で、その形でセッションを作り直す（形が変わった時も。`onnx_generic.js` の `onnxRun`）。形の計算が CPU に回らなくなり、DA3 small はベンチ 235 → 161ms（518×518）。**DA3 small は形を固定すると graph capture を作れるが、出力が壊れる**（深度がほぼ2値になる。画素の 99.6% が黒か白）ので、models.json の `onnx.graph_capture: false` で使わない（2026-09-29 に Web UI で2値に見えると報告があり、graph capture の有無で比べて特定）。他の graph capture を使うモデル（DAv2・SegFormer・YOLO26n・DEIMv2）は、graph capture の有無で出力が完全に一致することを確かめた。入力サイズ可変の YOLO26 は 64bit 整数の出力の頭が残るので使えない
- モデルを選んだ時、「モデル入力（長辺）」の既定をそのモデルの `pre.size[0]`（深度 518、YOLO 640）にする（選択肢に 518 を足した）
- M4 の Chrome、ベンチ（正方形の画像、中央値）: Depth Anything V2 small（fp16 + graph capture）は長辺 320 で 40ms、480 で 93ms、640 で 181ms。DA3 small（fp32、fp16 版なし、graph capture なし）は 320 で 59ms、518 で 161ms（最初に測った 320 で 53ms・640 で 270ms は graph capture で出力が壊れた状態の値）
- 連続実行（人物の動画 640×360 を3倍速、約 36fps）: Depth Anything V2 small は長辺 320 で 32fps（動画の速さが上限）、480 で 20fps。DA3 small は短辺 518 のままだと 3fps（1フレーム 338ms）
- Galaxy Z Fold6（Chrome 153）の連続実行（360×640 の動画）: Depth Anything V2 small は長辺 320 で 1 フレーム 81ms（中央値、p90 169ms、10fps。内訳の平均は前処理 5.5・モデル実行 142・後処理 18ms）、518 で 453ms（2.2fps）。画素は 2.6 倍なのに時間は 5.6 倍で、スマホでは ViT の attention（パッチの数の2乗近く）が効く。後処理（深度を色の画像にして PNG にする所）が 1 割強を占める
- Galaxy Z Fold6（Chrome 153）の連続実行、graph capture を直した後（2026-09-29）: DA3 small（fp32、graph capture なし）は長辺 518 で 293ms（3.4fps）、320 で 152ms（6.1fps）。続けて測った Depth Anything V2 small（fp16 + graph capture）は 320 で 148ms（6.8fps）で、前の計測（81ms、p90 169ms）より遅い。発熱などで同じ設定でも 80〜150ms の幅がある（未確認）。スマホの長辺 320 では DA3 と DAv2 の速さはあまり変わらない
- **深度の色の範囲**: フレームごとに「画像の中の最小〜最大」に合わせ直していたので、モデルの出力が時間的に安定しているかどうかは見えなかった（2026-09-29 に DA3 の時間的な安定性が見えないと指摘されて追加）。表示に「深度だけ（範囲を固定）」を足した（固定を選んで最初に描いたフレームの範囲のまま塗る。連続実行の開始と表示の切り替えで取り直す）。後処理は範囲（`range`）も返し、範囲は letterbox の余白を除いた画像の中だけで取るようにした（前は余白の意味の無い値も含めていた）
- 範囲の揺れ（人物の動画の 10〜13 秒、0.1 秒おき 31 フレーム、長辺 320、M4）: 一番遠い所の値の変動係数は DAv2 small 4.2%・DA3 small 0.5%（隣のフレームとの最大の変化 10.1% と 1.3%）で、DA3 の方が安定。一番近い所は DAv2 1.9%・DA3 4.2% だが、この区間は人が動いていて場面の変化が混ざる。動かない背景の画素ごとの揺れでは比べていない。スマホで別の動画を「範囲を固定」で見比べた範囲では、DA3 の安定性の改善は目で見て劇的というほどではなかった（2026-09-29 の報告）
- DA3 の推定する水平画角は入力の大きさで変わる（街の写真で長辺 320 だと 33°、640 だと 42°）。画角を見る時は大きい入力で

## EdgeTAM で動画の物を追う（2026-09-29）

- onnx-community の EdgeTAM（と SAM 2.1）の ONNX は「画像のエンコーダ」と「プロンプト → マスク」だけで、SAM2 の動画追跡に使う前のフレームの記憶の部分（memory attention / memory encoder）が無い。そこで、切り出し直しで追う: 1フレーム目はクリックした点、2フレーム目からは前のフレームのマスクを囲む枠（15% 広げる）と、重心に一番近いマスクの中の1点をプロンプトにする（プロンプトの「動画で追う」、既定でオン）
- 乗り移りを防ぐ判定: マスクが無い・品質 30% 未満、面積が前のフレームの 2.5 倍以上か 0.3 倍以下、最初のマスクの 4 倍超、前の枠との IoU 0.2 以下の時は「見失い」として前の枠のまま探し続け、15 フレーム続いたら止める。見失っている間はマスクを出さず、枠を赤で出す。次の枠の大きさは 1 フレームで 1.25 倍まで。これが無いと、物が画面から出た後に、広げた枠の中の壁や路面を「品質の高いマスク」として拾って乗り移った
- WebGPU では fp16 にした（エンコーダ 100 → 66ms、プロンプト → マスク 23 → 18ms、M4。スコアはほぼ同じ）
- M4 の連続実行（640×360 の動画）で 1 フレーム約 135ms、約 6fps。駐車場の動画で、曲がりながら画面の外へ出ていく車を最後まで追い、出た後は見失いになった
- Galaxy Z Fold6（Chrome 153）の連続実行で 1 フレーム 431ms（2fps）
- 1 フレームごとに画像のエンコーダを通すので、これ以上は速くならない（EdgeTAM は transformers.js 経由なので graph capture なども効かない）。スマホでは未確認

## 組み合わせのタブ: しぐさ（姿勢＋PINTO、2026-09-30）

2つのモデルを同じフレームで回して組み合わせる仕組み（`tasks[].combo`、`web/apps.js` の `COMBOS`、`docs/ADDING_UI.md`）の1件目。YOLO26n-pose の関節点と、PINTO の DEIMv2（頭・目・手と、頭の向きの8クラス）から、人ごとに判定する。

- 手を挙げた: 手首が肩より上（胴の長さの 1 割より上、関節点の信頼度 0.5 以上）
- 顔の向き: その人の枠の上半分にある頭の枠と、IoU 0.4 以上で重なる向きのクラス（front → 正面、*-front → 斜め、*-side → 横、*back → 後ろ）。DEIMv2 の向きのクラスは、手・目のタブでは表示を絞っていて使っていなかった
- 目を閉じている: 頭の中の目が全部「閉」（OCEC）。片目だけ閉なら付けない
- 指差し: その人の枠の近くの手に「指差し」（PGC）。PGC は多めに出る（サッカーの写真で 4 人中 2〜4 人）
- 作った関節点と部位の枠で、6 通りの判定が意図どおりになることを確かめた。サンプルの画像・動画には手を挙げた人がいないので、実際の画像での「手を挙げた」は未確認
- M4 の連続実行（人物の動画、3倍速）で 25〜28fps（2つのモデルと目・手の分類を合わせて）。PINTO は既定で 320×320 版（`combo.with[].default`）
- 追跡の ID は人（姿勢の結果）にだけ付く。手を振る（WHC、同じ手の 4 フレームの列で判定）は ID の無い手をまとめてしまい誤判定したので、ID の無い物はフレーム列の判定にかけないようにした

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
