# TODO

## 速さ（特にスマホ）

- [ ] スマホ（S25 Ultra・Pixel 6a・iPad）で「入力サイズ可変の YOLO26」を 480 / 640 / 960 で測り、実用になる組み合わせを決める
- [ ] スマホで連続実行のパイプライン化の効き方を測る（`?nopipe=1` と比べる）。前処理は Galaxy Z Fold6 でも既定の `upload` が最速だった（`gpu` はモデル実行が 20〜30ms 遅くなる。iPhone・他の Android でも確かめる）
- [ ] 前処理の縮小をサーバー（PIL の BILINEAR、縮小率に応じて周りも平均する）に合わせる。今は canvas も GPU も周りを平均しないので、大きく縮める時（DEIMv2 の 192 など）にサーバーと入力が少し違う
- [ ] 入力サイズ可変の YOLO26（Ultralytics の書き出し）で graph capture を使う: YOLO26-pose と同じく `onnx.cut` で出力の頭の手前で切り、JS の後処理で上位を選ぶ（出力の頭の 64bit 整数の計算が CPU に回るため。スマホで入力サイズを変えても fps が変わらないのはこのため）
- [ ] バッチ固定のモデル（OSNet の 16 など）も入力の形を固定する（今は `pre.batch` のあるモデルは対象外）
- [ ] WebNN（NPU）を onnxruntime-web の WebNN EP で試す（Chrome の実験機能）
- [ ] 小さいモデルは WebGPU の固定の手間が支配的だった（入力の形の固定と graph capture で DEIMv2 Atto 192 は 4.7ms まで縮んだので、WASM との比べ直しから）。端末ごとに WebGPU と WASM の速い方を自動で選ぶ
- [ ] YOLO26 の s / m も入力サイズ可変で書き出す（`tools/export_yolo26_dynamic.py yolo26s ...`）
- [ ] ブラウザの外（Android の LiteRT GPU/NPU、iOS の CoreML）との比較

## 機能

- [ ] 英語の UI（今は日本語のみ。表示文言を1か所にまとめて切り替える）
- [ ] インスタンス・セグメンテーション（YOLO26-seg、EoMT instance）
- [ ] BoT-SORT のカメラ移動の補正（GMC）。OpenCV.js が要る（数MB）
- [ ] SAM 3 のテキスト指定（`facebook/sam3` の利用申請が要る）
- [ ] 手の関節点（RTMPose-Hand、55MB）
- [ ] 録画（結果を重ねた動画の書き出し。`canvas.captureStream()` + MediaRecorder）
- [ ] 各端末のベンチマークの CSV を `docs/benchmarks/` に集め、README の「端末ごとの速度」を実測で更新する
- [ ] 同じ画像で複数のモデルを並べて比べる表示
- [ ] 手元の ONNX で SHA-256 が一致しないもの（自前で学習・変換したモデル）: Ultralytics の書き出しのメタデータ（`task` / `imgsz` / `names`）で後処理とクラス名を自動で設定する。無ければ入出力の名前と形（`session.inputMetadata`）を既存モデルと比べて候補を出し、サンプル画像で1回試して出力の異常を見る。選んだファイルは OPFS に保存して選び直さずに済むようにする
- [ ] 手元のファイルを transformers.js のモデル（複数ファイル）でも使う（フォルダごと選ぶ）
- [ ] YOLO 系の後処理のクラス名が COCO 固定になっている（`web/onnx_generic.js` の `yolo_detect`・`ultra_e2e_detect`、`adapters.py` の同じ部品が `COCO[...]`）。`post.classes` に名前の配列も書けるようにする（`"coco"` は今まで通り）。追加学習したモデル（VisDrone の10クラスなど）を載せる前提
- [ ] YOLO26 の NMS ありの経路を選べるようにする: one-to-many の生の出力（1, 84, 8400）で書き出し、JS で NMS（IoU 0.7）。群衆の俯瞰（Ultralytics の CI 動画 62 フレーム）で NMS なし 1213 件に対し 1391 件。NMS を ONNX に埋め込む書き出しは `NonMaxSuppression`・`NonZero` が WebGPU EP に無く graph capture を作れない

## 軽いモデルのラインナップ（スマホでリアルタイムに近いもの）

モデルを選ぶ時は `docs/ADDING_MODELS.md` の「7. 速く動かす」を見る（`onnx/` に fp16 版があり、前処理がリサイズ＋正規化だけのものは、汎用 ONNX で書けば最適化が自動でかかる）。候補は未確認なので、配布元・ライセンス・ONNX の有無から確かめる。

- [ ] 深度: 「モデル入力（長辺）」にもっと小さい選択肢（252 など）を足す（Galaxy Z Fold6 で DAv2 small は 320 で 10fps、518 で 2fps）。後処理（深度を色の画像にして PNG にする所、Fold6 で 18〜31ms）を画像のまま渡すか GPU で色付けして軽くする。DA3 small の fp16 版を作る（今は fp32 のみ）
- [ ] 俯瞰の群衆・駐車場: [dronefreak/visdrone-yolo26n](https://huggingface.co/dronefreak/visdrone-yolo26n)（YOLO26n を VisDrone で追加学習、AGPL-3.0）を onnx-community と同じ形（`logits`・`pred_boxes`、fp16 版も）で書き出し、graph capture と Python（`.pt`）との一致を確かめる。Python の比較では、群衆 19.5 → 41.3 人/フレーム（NMS なし）、駐車場 0 → 41.5 台/フレーム、目の高さの街の写真では歩行者 1（COCO 版は 5）で俯瞰専用。配るなら自前の Hugging Face リポジトリ（AGPL、元モデルと学習データを明記。VisDrone のライセンス表記は CC-BY-SA-3.0 と CC-BY-NC-SA-3.0 で資料により食い違う）
- [ ] 背景除去・人物の切り抜き: BiRefNet lite（1024）より軽いもの（MODNet など、512 前後でスマホ向け）
- [ ] セグメンテーション: SegFormer B0 の入力サイズ（今は 512 固定）を選べるようにする。後処理（クラスの最大）を GPU に移す
- [ ] SAM 系の追跡: 今は前のフレームのマスクから作った枠で切り出し直す疑似的な追跡（NOTES）。前のフレームの記憶を使う本来の追跡は、EdgeTAM の memory attention / memory encoder を ONNX に書き出す必要がある。スマホで fps を測る。複数の物を同時に追う

- [ ] 顔の細かな動き: MediaPipe の顔メッシュ（478 点）と blendshape（まばたき・口の開き・笑顔など 52 種類）。py-feat が PyTorch の形（`py-feat/mp_facemesh_v2`・`py-feat/mp_blendshapes`、Apache-2.0）で配っているので ONNX に書き出す。顔 → 顔メッシュ → blendshape の 3 段になる

## 応用（組み合わせ）

`models.json` の `apps` と `web/apps.js` に足す（`docs/ADDING_UI.md` の「4. 応用」）。1つのモデルの結果に後付けするものは応用に、複数のモデルを組み合わせるものは専用のタブを考える。

- [ ] 「数える」で小さい物（遠くの人・車）を取りこぼす（YOLO26 は 640 だと小さい物が苦手）。入力サイズ可変の YOLO26 を 960 で使う、画面を分割して検出する（SAHI のように）などを試す
- [ ] 線を越えた数（追跡の ID が画面に引いた線をまたいだ回数。向きごと）
- [ ] 範囲の中の数と滞在時間（画面に描いた範囲に入った ID ごとの時間）
- [ ] 動きの軌跡の重ね書き・ヒートマップ（どこを通ったか）
- [ ] 検出＋深度で物までの距離の目安（複数のモデルの組み合わせ。専用のタブ）
- [ ] 姿勢の応用（手を挙げた・しゃがんだなどの簡単な判定、回数を数える）

## インタラクト（結果で別の画面を動かす）

`models.json` の `interact` と `web/interact.js` に足す（`docs/ADDING_UI.md` の「5. インタラクト」）。

- [x] キャラ（Live2D 風）: 一番大きく写っている人の骨格・顔の点で頭・体・腕、表情で顔を動かす
- [x] フレームを外に流す: BroadcastChannel（別のタブ）と WebSocket（`server.py` の `/ws` が中継）。確認用の受け手のページ `web/receiver.html` と Python の例 `tools/ws_receiver.py`
- [ ] 外のツールの例: TouchDesigner・Unity・OSC への橋渡し、p5.js の受け手のページ
- [ ] 出来事を流す: 接近・後退の切り替わり、手を挙げた、線を越えたなどを `{type: "event"}` で（受け手が毎フレーム判定しなくてよいように）
- [ ] 3D のキャラ（VRM を three.js ＋ three-vrm で。「3D の姿勢」の奥行きを使う）
- [ ] 顔の細かな動き（MediaPipe の blendshape）でまばたき・口の開きを連動（上の「軽いモデル」の顔メッシュと組み合わせ）

## テンプレートマッチング・ビジュアルオドメトリ

- [x] テンプレートマッチング（XFeat ＋ 相互最近傍 ＋ LO-RANSAC のホモグラフィ）。ブラウザとサーバーで同じ手順、kornia と一致
- [x] スマホ（Galaxy Z Fold6）で速さを測る（JS の後処理 70ms、GPU の後処理 31ms）
- [x] 後処理を GPU に（WGSL で極大・スコア・記述子・対応。`web/xfeat_gpu.js`）。Fold6 で 70 → 31ms
- [ ] 比較用の古典的な画素相関（matchTemplate、NCC）
- [ ] fp16 版（ONNX を端末で fp16 に変える）
- [ ] ビジュアルオドメトリ（連続するフレームの対応から基本行列 → カメラの動き）、LightGlue（マッチングの速さと精度が分かってから）
- [x] 手ぶれ補正（映像だけ。なめらか・三脚、切り抜き、揺れの指標）
- [ ] 手ぶれ補正: スマホの手持ち・置いた時で、ブラウザのカメラの映像の揺れを測る
- [ ] 手ぶれ補正: ジャイロ（DeviceMotionEvent）と組み合わせる（焦点距離と時刻のずれは映像の動きとの突き合わせで推定）
- [ ] 手ぶれ補正: 被写体ロック（テンプレートの位置に止める）、録画済みの動画の先読みでより強くなめらかに、ホモグラフィを WebGL で正しく描く

## 保守

- [ ] ブラウザでの自動テスト（今は手元の puppeteer スクリプトで確認している）を GitHub Actions に載せる
- [ ] BiRefNet の WebGPU（storage buffer の上限）が onnxruntime-web で直ったら `avoid_browser` を外す
- [ ] SmolVLM を fp16 でも正しく動かす（今は fp32 で約1GB）
