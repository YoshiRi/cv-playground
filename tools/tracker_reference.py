"""web/tracker.js が Ultralytics と同じ結果を出すかの確認用（開発時だけ使う。Ultralytics は AGPL-3.0 なので同梱しない）。

動画の各フレームを YOLO26n（adapters.py の汎用 ONNX）で検出し、Ultralytics の BYTETracker / BOTSORT（GMC なし）に
通した結果を ref.json に保存する。続けて node tools/compare_tracker.mjs で web/tracker.js と比べる。

  uv venv /tmp/ulv && uv pip install --python /tmp/ulv/bin/python ultralytics lap opencv-python-headless onnxruntime huggingface_hub
  /tmp/ulv/bin/python tools/tracker_reference.py <動画ファイル>
  node tools/compare_tracker.mjs
"""
import json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import cv2, numpy as np
from types import SimpleNamespace
from PIL import Image
from adapters import ADAPTERS, load_entries
from ultralytics.trackers.byte_tracker import BYTETracker
from ultralytics.trackers.bot_sort import BOTSORT

E = {e["key"]: e for e in load_entries()}
a = ADAPTERS["onnx"](E["yolo26n"]); a.load()
cap = cv2.VideoCapture(sys.argv[1])
frames = []
while True:
    ok, f = cap.read()
    if not ok or len(frames) >= 3000: break
    im = Image.fromarray(cv2.cvtColor(f, cv2.COLOR_BGR2RGB))
    r = a.run(im, {"threshold": 0.1})
    frames.append([{"label": d["label"], "score": d["score"], "box": d["box"]} for d in r["items"]])

class Res:  # Ultralytics の Boxes 相当（conf, xywh, cls, xyxy とブールの添字）
    def __init__(self, xyxy, conf, cls):
        self.xyxy, self.conf, self.cls = xyxy, conf, cls
        self.xywh = np.c_[(xyxy[:, :2] + xyxy[:, 2:]) / 2, xyxy[:, 2:] - xyxy[:, :2]] if len(xyxy) else np.zeros((0, 4))
    def __len__(self): return len(self.conf)
    def __getitem__(self, m): return Res(self.xyxy[m], self.conf[m], self.cls[m])

base = dict(track_high_thresh=0.25, track_low_thresh=0.1, new_track_thresh=0.25, track_buffer=30, match_thresh=0.8, fuse_score=True)
out = {"dets": frames}
for name, T, extra in [("bytetrack", BYTETracker, {}), ("botsort", BOTSORT, dict(gmc_method="none", proximity_thresh=0.5, appearance_thresh=0.8, with_reid=False, model="auto"))]:
    t = T(SimpleNamespace(**base, **extra), frame_rate=30) if "frame_rate" in T.__init__.__code__.co_varnames else T(SimpleNamespace(**base, **extra))
    seq = []
    for d in frames:
        xyxy = np.array([x["box"] for x in d], np.float32).reshape(-1, 4)
        conf = np.array([x["score"] for x in d], np.float32)
        r = t.update(Res(xyxy, conf, np.zeros(len(d), np.float32)))
        seq.append([[round(float(v), 2) for v in row[:4]] + [int(row[4])] for row in r])
    out[name] = seq
json.dump(out, open("ref.json", "w"))
print(len(frames), "frames; dets/frame", np.mean([len(f) for f in frames]).round(1),
      "; ids byte", len({r[4] for s in out["bytetrack"] for r in s}), "bot", len({r[4] for s in out["botsort"] for r in s}))
