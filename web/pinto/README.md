# PINTO_model_zoo のモデル（再配布）

ブラウザから直接取得できる配布元が無い（GitHub Releases は CORS 非対応、Wasabi は数GBの tar のみ）ため、ここに置いて GitHub Pages から配る。
いずれも [PINTO0309](https://github.com/PINTO0309) 氏の成果物で、MIT License（元モデルのライセンスは各リポジトリを参照）。

| ファイル | 元 | 用途 |
| --- | --- | --- |
| `deimv2_hgnetv2_atto_wholebody34_340query_n_batch_320x320.onnx` | [PINTO_model_zoo/472_DEIMv2-Wholebody34](https://github.com/PINTO0309/PINTO_model_zoo/tree/main/472_DEIMv2-Wholebody34)（元: [DEIMv2](https://github.com/Intellindust-AI-Lab/DEIMv2)、Apache-2.0） | 体・頭・顔・目・手など 34 クラスの検出（Atto、320×320） |
| `ocec_p.onnx` | [PINTO0309/OCEC](https://github.com/PINTO0309/OCEC) | 目の開閉（目の切り出し 24×40） |
| `pgc_s_32x32.onnx` | [PINTO0309/PGC](https://github.com/PINTO0309/PGC) | 指差し（手の切り出し 32×32） |
| `whc_seq_3dcnn_1x4x32x32.onnx` | [PINTO0309/WHC](https://github.com/PINTO0309/WHC) | 手を振る動作（同じ手の 4 フレーム、32×32） |
| `deimv2_hgnetv2_atto_wholebody34_170query_n_batch_192x192.onnx` | 同上（472_DEIMv2-Wholebody34） | Atto の 192×192 版（最軽量） |
| `deimv2_hgnetv2_femto_wholebody34_340query_n_batch_416x416.onnx` | 同上 | Femto、416×416 |
| `deimv2_hgnetv2_pico_wholebody34_340query_n_batch_640x640.onnx` | 同上 | Pico、640×640 |
