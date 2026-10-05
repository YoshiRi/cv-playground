# TODO

済んだものは消している（何をしたかは README・[NOTES.md](NOTES.md)・git の履歴）。

## 次の候補

- [ ] 出来事を流す: フレームの `events` に、線を越えた（済み）のほか、判定の切り替わり（今はフレームの `decisions[].changed`）、接近・後退の切り替わり、手を挙げたなどを足す
- [ ] 外のツールの例: TouchDesigner・Unity・OSC への橋渡し、p5.js の受け手のページ
- [ ] 録画（結果を重ねた動画の書き出し。`canvas.captureStream()` + MediaRecorder）
- [ ] 範囲の中の数と滞在時間（画面に範囲を描く。線を越えた数と同じ引き方で）

## スマホで測る（端末が要る）

- [ ] 入力サイズ可変の YOLO26 を 480 / 640 / 960 で（S25 Ultra・Pixel 6a・iPad）測り、実用になる組み合わせを決める
- [ ] 連続実行のパイプライン化の効き方（`?nopipe=1` と比べる）。前処理は Fold6 でも既定の `upload` が最速だった（`gpu` はモデル実行が 20〜30ms 遅くなる）。iPhone・他の Android でも
- [ ] 進む・止まるの速さ（3 つのモデルを同じフレームで回す）と、言い方・閾値の調整
- [ ] 各端末のベンチマークの CSV を `docs/benchmarks/` に集め、NOTES.md の「端末ごとの速度」を実測で更新する
- [ ] ブラウザの外（Android の LiteRT GPU/NPU、iOS の CoreML）との比較

## 速くする

- [ ] 入力サイズ可変の YOLO26 で graph capture を使う: YOLO26-pose と同じく `onnx.cut` で出力の頭の手前で切り、JS の後処理で上位を選ぶ（出力の頭の 64bit 整数の計算が CPU に回るため。スマホで入力サイズを変えても fps が変わらないのはこのため）
- [ ] YOLO26 の s / m も入力サイズ可変で書き出す（`tools/export_yolo26_dynamic.py yolo26s ...`）
- [ ] 小さいモデルは WebGPU の固定の手間が支配的（DEIMv2 Atto 192 は 4.7ms まで縮んだので、WASM との比べ直しから）。端末ごとに WebGPU と WASM の速い方を自動で選ぶ
- [ ] バッチ固定のモデル（OSNet の 16 など）も入力の形を固定する（今は `pre.batch` のあるモデルは対象外）
- [ ] 前処理の縮小をサーバー（PIL の BILINEAR、縮小率に応じて周りも平均する）に合わせる。今は canvas も GPU も周りを平均しないので、大きく縮める時（DEIMv2 の 192 など）にサーバーと入力が少し違う
- [ ] 深度: 「モデル入力（長辺）」に 252 などを足す（Fold6 で DAv2 small は 320 で 10fps、518 で 2fps）。後処理（色の画像にして PNG にする所、Fold6 で 18〜31ms）を GPU で色付けする。DA3 small の fp16 版を作る
- [ ] SegFormer B0 の入力サイズ（今は 512 固定）を選べるようにし、後処理（クラスの最大）を GPU に
- [ ] XFeat の fp16 版（ONNX を端末で fp16 に変える）

## モデル・タスクを足す

モデルを選ぶ時は `docs/ADDING_MODELS.md` の「7. 速く動かす」を見る。候補は未確認なので、配布元・ライセンス・ONNX の有無から確かめる。

- [ ] インスタンス・セグメンテーション（YOLO26-seg、EoMT instance）
- [ ] 手の関節点（RTMPose-Hand、55MB）
- [ ] 顔の細かな動き: MediaPipe の顔メッシュ（478 点）と blendshape（52 種類）。py-feat の PyTorch 版（`py-feat/mp_facemesh_v2`・`py-feat/mp_blendshapes`、Apache-2.0）を ONNX に。顔 → 顔メッシュ → blendshape の 3 段。できたらキャラのまばたき・口の開きにつなぐ
- [ ] 俯瞰の群衆・駐車場: [dronefreak/visdrone-yolo26n](https://huggingface.co/dronefreak/visdrone-yolo26n)（AGPL-3.0）を onnx-community と同じ形で書き出す。Python の比較で群衆 19.5 → 41.3 人/フレーム、駐車場 0 → 41.5 台/フレーム（目の高さの写真は苦手で俯瞰専用）。配るなら自前の Hugging Face リポジトリ（VisDrone のライセンス表記は資料により食い違う）。先に下の「クラス名を名前の配列で」が要る
- [ ] 背景除去・人物の切り抜き: BiRefNet lite（1024）より軽いもの（MODNet など）
- [ ] SAM 系の本来の追跡（今は前のフレームのマスクの枠で切り出し直す疑似的な追跡）。EdgeTAM の memory attention / memory encoder を ONNX に。複数の物を同時に
- [ ] テンプレートマッチングの比較用に古典的な画素相関（matchTemplate、NCC）
- [ ] ビジュアルオドメトリ（対応から基本行列 → カメラの動き）、LightGlue

## 応用・判定・インタラクト

- [ ] 「数える」で小さい物（遠くの人・車）を取りこぼす。入力サイズ可変の YOLO26 を 960 で、画面を分割して検出（SAHI のように）を試す
- [ ] 動きの軌跡の重ね書き・ヒートマップ（どこを通ったか）
- [ ] 姿勢から回数を数える（スクワット・腕立てなど。手を挙げた等の判定は「しぐさ」のタブにある）
- [ ] 物までの距離の目安（m）。今の接近・後退は相対的な深度
- [ ] インタラクトのフレームの値（判定の値・表情の名前）が画面の言語にかかわらず日本語。外のツール向けに英語の id も載せる
- [ ] 3D のキャラ（VRM を three.js ＋ three-vrm で。「3D の姿勢」の奥行きを使う）

## 手ぶれ補正

- [ ] ジャイロ（DeviceMotionEvent）と組み合わせる（焦点距離と時刻のずれは映像の動きとの突き合わせで推定）
- [ ] 被写体ロック（テンプレートの位置に止める）
- [ ] 録画済みの動画は先読みを長くしてより強くなめらかに
- [ ] ホモグラフィを WebGL で正しく描く

## 比べる・手元のモデル

- [ ] 同じ画像で複数のモデルを並べて比べる表示
- [ ] YOLO 系の後処理のクラス名が COCO 固定（`web/onnx_generic.js` の `yolo_detect`・`ultra_e2e_detect`、`adapters.py` も）。`post.classes` に名前の配列も書けるようにする
- [ ] YOLO26 の NMS ありの経路: one-to-many の生の出力（1, 84, 8400）で書き出し、JS で NMS（IoU 0.7）。群衆の俯瞰で NMS なし 1213 件に対し 1391 件。NMS を ONNX に埋め込むと `NonMaxSuppression`・`NonZero` が WebGPU EP に無く graph capture を作れない
- [ ] 手元の ONNX で SHA-256 が一致しないもの（自前で学習・変換したモデル）: Ultralytics のメタデータ（`task` / `imgsz` / `names`）で後処理とクラス名を自動で設定。無ければ入出力の名前と形を既存モデルと比べて候補を出す。選んだファイルは OPFS に保存
- [ ] 手元のファイルを transformers.js のモデル（複数ファイル）でも使う（フォルダごと選ぶ）
- [ ] BoT-SORT のカメラ移動の補正（GMC）。OpenCV.js が要る（数MB）。XFeat のフレーム間の動き（手ぶれ補正と同じ）で代わりにできるかも

## 外の事情を待つもの

- [ ] SAM 3 のテキスト指定（`facebook/sam3` の利用申請が要る）
- [ ] BiRefNet の WebGPU（storage buffer の上限）が onnxruntime-web で直ったら `avoid_browser` を外す
- [ ] SmolVLM を fp16 でも正しく動かす（今は fp32 で約1GB）
- [ ] WebNN（NPU）を onnxruntime-web の WebNN EP で（Chrome の実験機能）

## 保守

- [ ] ブラウザでの自動テスト（今は手元の puppeteer スクリプト。英語の画面に日本語が残っていないかの巡回も）を GitHub Actions に載せる
