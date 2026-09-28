# TODO

## 速さ（特にスマホ）

- [ ] スマホ（S25 Ultra・Pixel 6a・iPad）で「入力サイズ可変の YOLO26」を 480 / 640 / 960 で測り、実用になる組み合わせを決める
- [ ] 前処理を GPU で行う（今は OffscreenCanvas から画素を読み出して JS で正規化。M1 で 8〜17ms）。WebGPU のテクスチャから直接入力テンソルを作り、`ort.Tensor.fromGpuBuffer` で渡す
- [ ] 前処理・推論・描画のパイプライン化（今は1フレームずつ順番）
- [ ] WebNN（NPU）を onnxruntime-web の WebNN EP で試す（Chrome の実験機能）
- [ ] 小さいモデルは WebGPU の固定の手間（M4 Mac で約 15〜24ms）が支配的。端末ごとに WebGPU と WASM の速い方を自動で選ぶ
- [ ] YOLO26 の s / m も入力サイズ可変で書き出す（`tools/export_yolo26_dynamic.py yolo26s ...`）
- [ ] ブラウザの外（Android の LiteRT GPU/NPU、iOS の CoreML）との比較

## 機能

- [ ] 英語の UI（今は日本語のみ。表示文言を1か所にまとめて切り替える）
- [ ] インスタンス・セグメンテーション（YOLO26-seg、EoMT instance）
- [ ] BoT-SORT のカメラ移動の補正（GMC）。OpenCV.js が要る（数MB）
- [ ] SAM 3 のテキスト指定（`facebook/sam3` の利用申請が要る）
- [ ] 動画の追跡で、SAM 系のマスクも ID ごとに追う（今は点を固定して毎フレーム切り出し直すだけ）
- [ ] 手の関節点（RTMPose-Hand、55MB）
- [ ] 録画（結果を重ねた動画の書き出し。`canvas.captureStream()` + MediaRecorder）
- [ ] 各端末のベンチマークの CSV を `docs/benchmarks/` に集め、README の「端末ごとの速度」を実測で更新する
- [ ] 同じ画像で複数のモデルを並べて比べる表示
- [ ] 手元の ONNX で SHA-256 が一致しないもの（自前で学習・変換したモデル）: Ultralytics の書き出しのメタデータ（`task` / `imgsz` / `names`）で後処理とクラス名を自動で設定する。無ければ入出力の名前と形（`session.inputMetadata`）を既存モデルと比べて候補を出し、サンプル画像で1回試して出力の異常を見る。選んだファイルは OPFS に保存して選び直さずに済むようにする
- [ ] 手元のファイルを transformers.js のモデル（複数ファイル）でも使う（フォルダごと選ぶ）

## 保守

- [ ] ブラウザでの自動テスト（今は手元の puppeteer スクリプトで確認している）を GitHub Actions に載せる
- [ ] BiRefNet の WebGPU（storage buffer の上限）が onnxruntime-web で直ったら `avoid_browser` を外す
- [ ] SmolVLM を fp16 でも正しく動かす（今は fp32 で約1GB）
