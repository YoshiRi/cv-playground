"""XFeat の手順（adapters.py の xfeat_extract・match_mnn・ransac_homography）を kornia・OpenCV と同じ画像で比べる。

    uv pip install --python .venv/bin/python -r requirements-dev.txt   # kornia・opencv-python-headless（比べる時だけ）
    .venv/bin/python tools/xfeat_check.py [画像 ...]                  # 省略するとサンプルの画像（Hugging Face）

1. 点・スコア・記述子: ONNX（onnx.cut で切ったもの）＋ numpy の後処理 と kornia.feature.XFeat.detectAndCompute
2. 対応: 相互最近傍（cos > 0.82）の組を kornia の _match_mnn と
3. ホモグラフィ: テンプレート（画像の一部を回して縮めたもの。正解のホモグラフィが分かる）で、自前の RANSAC・
   cv2.findHomography（RANSAC 3px）・正解の 3 つの四隅を比べる
"""
import io
import math
import sys
import warnings
from pathlib import Path

import numpy as np
import requests
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")
import adapters  # noqa: E402

SAMPLES = ["https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/city-streets.jpg",
           "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/football-match.jpg",
           "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/cats.jpg"]


def load(src):
    if src.startswith("http"):
        return Image.open(io.BytesIO(requests.get(src, timeout=60).content)).convert("RGB")
    return Image.open(src).convert("RGB")


def make_template(im, angle=15.0, scale=0.8):
    """画像の中央 40% を、angle 度回して scale 倍にしたテンプレート。返すのは (テンプレート, 正解のホモグラフィ: テンプレート → 画像)"""
    W, H = im.size
    cw, ch = int(W * 0.4), int(H * 0.4)
    tw, th = int(cw * scale), int(ch * scale)
    cx, cy = W / 2, H / 2
    a = math.radians(angle)
    # テンプレートの (u, v) → 画像の (x, y): 中心まわりに回して 1/scale 倍
    R = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]]) / scale
    t = np.array([cx, cy]) - R @ np.array([tw / 2, th / 2])
    A = np.array([[R[0, 0], R[0, 1], t[0]], [R[1, 0], R[1, 1], t[1]], [0, 0, 1]])
    tpl = im.transform((tw, th), Image.AFFINE, tuple(A[:2].ravel()), resample=Image.BILINEAR)
    return tpl, A


def main():
    import cv2
    import torch
    from kornia.feature import XFeat

    entry = next(e for e in adapters.load_entries() if e["key"] == "xfeat")
    ad = adapters.OnnxAdapter(entry)
    ad.load()
    kf = XFeat.from_pretrained(top_k=512).eval()
    top_k = entry["post"]["top_k"]
    rows = []
    for src in sys.argv[1:] or SAMPLES:
        im = load(src)
        name = Path(src).name
        # 1. 点・スコア・記述子（同じ大きさに縮めた画像を両方に入れる。kornia の中の縮小は何もしない大きさ）
        x, meta = adapters.preprocess(im, entry["pre"])
        out = dict(zip(ad.outputs, ad.sess.run(None, {entry["pre"]["input"]: x})))
        pts, sc, f = adapters.xfeat_extract(out, meta, top_k)
        pin = pts * [meta["sx"], meta["sy"]]  # 入力画像の座標
        with torch.no_grad():
            k = kf.detectAndCompute(torch.from_numpy(x), top_k=top_k)[0]
        kp, ks, kd = k["keypoints"].numpy(), k["scores"].numpy(), k["descriptors"].numpy()
        ours = {(int(round(a)), int(round(b))): i for i, (a, b) in enumerate(pin)}
        theirs = {(int(round(a)), int(round(b))): i for i, (a, b) in enumerate(kp)}
        common = ours.keys() & theirs.keys()
        cos = np.mean([float(f[ours[c]] @ kd[theirs[c]]) for c in common]) if common else 0
        sdiff = max((abs(sc[ours[c]] - ks[theirs[c]]) for c in common), default=0)
        print(f"== {name} 入力 {meta['iw']}×{meta['ih']}")
        print(f"  点: 自前 {len(pts)} / kornia {len(kp)}、位置が一致 {len(common)}（{len(common) / max(1, len(kp)) * 100:.1f}%）、"
              f"スコアの差 最大 {sdiff:.2e}、記述子の cos 平均 {cos:.5f}")
        # 2. 対応
        tpl, A = make_template(im)
        xt, mt = adapters.preprocess(tpl, entry["pre"])
        ot = dict(zip(ad.outputs, ad.sess.run(None, {entry["pre"]["input"]: xt})))
        tp, _, tf = adapters.xfeat_extract(ot, mt, top_k)
        i0, i1 = adapters.match_mnn(tf, f)
        with torch.no_grad():
            kt = kf.detectAndCompute(torch.from_numpy(xt), top_k=top_k)[0]
            j0, j1 = XFeat._match_mnn(None, kt["descriptors"], k["descriptors"], 0.82)
        print(f"  対応: 自前 {len(i0)} / kornia {len(j0)}（テンプレートの点 {len(tp)} / {len(kt['keypoints'])}）")
        # 3. ホモグラフィ
        a, b = tp[i0], pts[i1]
        Hm, inl = adapters.ransac_homography(a, b)
        tw, th = tpl.size
        corners = np.array([[0, 0], [tw, 0], [tw, th], [0, th]], float)
        gt = adapters.project(A, corners)
        line = f"  ホモグラフィ: インライア 自前 {int(inl.sum())}"
        if Hm is not None:
            line += f"、四隅の誤差（正解と）最大 {np.abs(adapters.project(Hm, corners) - gt).max():.2f}px"
        Hc, mc = cv2.findHomography(a.astype(np.float32), b.astype(np.float32), cv2.RANSAC, 3.0) if len(a) >= 4 else (None, None)
        if Hc is not None:
            line += f" ・ cv2 インライア {int(mc.sum())}、四隅の誤差（正解と）最大 {np.abs(adapters.project(Hc, corners) - gt).max():.2f}px"
            if Hm is not None:
                line += f"、自前と cv2 の差 最大 {np.abs(adapters.project(Hm, corners) - adapters.project(Hc, corners)).max():.2f}px"
        print(line)
        rows.append((name, len(common) / max(1, len(kp)), len(i0), len(j0)))
    ok = all(r[1] > 0.98 and abs(r[2] - r[3]) <= max(2, 0.02 * r[3]) for r in rows)
    print("結果:", "一致" if ok else "ずれあり（上を見る）")


if __name__ == "__main__":
    main()
