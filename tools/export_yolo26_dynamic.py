"""YOLO26 を入力サイズ可変（dynamic）かつ NMS 不要（end2end、出力 (1, 300, 6 or 57)）の ONNX に書き出し、models/ultralytics/ に置く。

onnx-community の ONNX は入力が 640×640 固定なので、モデルの入力を 320〜960 で切り替えたり、Ultralytics の推論と同じ
長方形の入力（16:9 なら 640×384 など）で余白の計算を省いたりするために使う。YOLO26 の重みは AGPL-3.0 なので
リポジトリには入れず、書き出したファイルはこのサーバーからだけ配る（models/ は .gitignore 済み）。

  uv venv /tmp/ulv && uv pip install --python /tmp/ulv/bin/python ultralytics onnx onnxslim
  /tmp/ulv/bin/python tools/export_yolo26_dynamic.py yolo26n yolo26n-pose
"""
import shutil
import sys
from pathlib import Path

from ultralytics import YOLO

OUT = Path(__file__).resolve().parent.parent / "models" / "ultralytics"
OUT.mkdir(parents=True, exist_ok=True)
for name in sys.argv[1:] or ["yolo26n", "yolo26n-pose"]:
    f = YOLO(f"{name}.pt").export(format="onnx", dynamic=True, end2end=True, imgsz=640, simplify=True, opset=17)
    shutil.move(f, OUT / f"{name}_dyn.onnx")
    print(OUT / f"{name}_dyn.onnx")
