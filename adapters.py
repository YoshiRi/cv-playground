"""サーバー側のモデル実装（adapter）。モデルの一覧は web/models.json（ブラウザ側と共通）。

どの adapter も同じ形:
    a = ADAPTERS[entry["adapter"]](entry)   # 軽い。まだ読み込まない
    a.load()                                # 重みの取得と初期化
    a.run(image: PIL.Image(RGB), params: dict) -> 結果 dict
結果の形式（座標は入力画像のピクセル。ブラウザ側 web/worker.js と同じ）:
    boxes  {items: [{label, score, box: [x1, y1, x2, y2], keypoints?: [[x, y, 可視度], ...]}]}
    mask   {mask: PNG data URL（白=前景）, score?, cutout?}
    depth  {image: PNG data URL（明るい=近い）, note?}
    text   {text}

新しいモデルを足す時:
  - 素の ONNX なら models.json に adapter "onnx" で pre / post を書くだけ。足りない部品は
    PRE の resize 方式か POST に関数を足す（web/onnx_generic.js にも同じものを足すとブラウザでも動く）
  - ライブラリのプロセッサが要るモデルは、Adapter を継承したクラスを書いて ADAPTERS に登録する
"""
import base64
import io
import json
import math
import os
from pathlib import Path

import numpy as np
import requests
import torch
from PIL import Image

ROOT = Path(__file__).parent
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
OLLAMA = os.environ.get("OLLAMA_HOST_URL", "http://127.0.0.1:11434")
COCO = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
        "teddy bear", "hair drier", "toothbrush"]


def load_entries():
    return json.loads((ROOT / "web" / "models.json").read_text())["models"]


def png_data_url(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class Adapter:
    device = DEVICE  # 結果に書く実行デバイス

    def __init__(self, entry):
        self.e = entry

    def load(self):
        raise NotImplementedError

    def run(self, im: Image.Image, p: dict) -> dict:
        raise NotImplementedError


# ---------- 汎用 ONNX: 前処理・後処理を models.json の pre / post で組み立てる ----------

def preprocess(im: Image.Image, pre: dict, input_size=None):
    """画像 → 入力テンソル。meta は入力座標 = 元座標 × (sx, sy) + (ox, oy) と、内容のある範囲 (cw, ch)"""
    W, H = im.size
    mode = pre.get("resize", "stretch")
    if mode in ("letterbox", "letterbox_rect"):
        # letterbox_rect: 長辺を S に合わせ、縦横を stride の倍数まで余白で埋めた長方形
        S = int(input_size) if pre.get("dynamic") and input_size else pre["size"][0]
        r = S / max(W, H) if mode == "letterbox_rect" else min(pre["size"][0] / W, pre["size"][1] / H)
        cw, ch = round(W * r), round(H * r)
        st = pre.get("stride", 32)
        tw, th = (math.ceil(cw / st) * st, math.ceil(ch / st) * st) if mode == "letterbox_rect" else pre["size"]
        ox, oy = (tw - cw) // 2, (th - ch) // 2
        pad = pre.get("pad_value", 114)
        canvas = Image.new("RGB", (tw, th), (pad, pad, pad))
        canvas.paste(im.resize((cw, ch), Image.BILINEAR), (ox, oy))
        img, meta = canvas, dict(sx=r, sy=r, ox=ox, oy=oy, iw=tw, ih=th, cw=cw, ch=ch)
    elif mode == "stretch":
        tw, th = pre["size"]
        img, meta = im.resize((tw, th), Image.BILINEAR), dict(sx=tw / W, sy=th / H, ox=0, oy=0, iw=tw, ih=th, cw=tw, ch=th)
    elif mode == "keep_aspect":
        m, r = pre.get("multiple", 1), pre["short"] / min(W, H)
        tw, th = max(m, round(W * r / m) * m), max(m, round(H * r / m) * m)
        img, meta = im.resize((tw, th), Image.BILINEAR), dict(sx=tw / W, sy=th / H, ox=0, oy=0, iw=tw, ih=th, cw=tw, ch=th)
    else:
        raise ValueError(f"unknown resize {mode}")
    x = np.asarray(img, dtype=np.float32) * pre.get("scale", 1 / 255)
    if pre.get("bgr"):
        x = x[:, :, ::-1]
    if "mean" in pre:
        x = (x - np.array(pre["mean"], np.float32)) / np.array(pre["std"], np.float32)
    x = x.transpose(2, 0, 1)[None]
    for ax in pre.get("add_dims", []):
        x = np.expand_dims(x, ax)
    meta.update(W=W, H=H)
    return np.ascontiguousarray(x, dtype=np.float32), meta


def to_orig(x, y, m):
    return (x - m["ox"]) / m["sx"], (y - m["oy"]) / m["sy"]


def crop_resize(a2d: np.ndarray, m) -> np.ndarray:
    """入力解像度の2次元出力から内容のある範囲を切り出し、元画像の大きさへ（0..255 の uint8）"""
    h, w = a2d.shape
    fx, fy = w / m["iw"], h / m["ih"]
    y0, x0 = round(m["oy"] * fy), round(m["ox"] * fx)
    c = a2d[y0:y0 + round(m["ch"] * fy), x0:x0 + round(m["cw"] * fx)]
    return np.asarray(Image.fromarray(c.astype(np.uint8)).resize((m["W"], m["H"]), Image.BILINEAR))


def post_yolo_detect(out, m, post, p):
    # YOLO26 は NMS 込みの出力: logits (1,300,80) はシグモイド前、pred_boxes (1,300,4) は入力に対する正規化 cx cy w h
    prob = 1 / (1 + np.exp(-out["logits"][0]))
    boxes, th, items = out["pred_boxes"][0], float(p.get("threshold", 0.4)), []
    for i in np.where(prob.max(1) >= th)[0]:
        c = int(prob[i].argmax())
        cx, cy, w, h = boxes[i] * [m["iw"], m["ih"], m["iw"], m["ih"]]
        x1, y1 = to_orig(cx - w / 2, cy - h / 2, m)
        x2, y2 = to_orig(cx + w / 2, cy + h / 2, m)
        items.append({"label": COCO[c], "score": float(prob[i, c]), "box": [float(x1), float(y1), float(x2), float(y2)]})
    return {"kind": "boxes", "items": items}


def post_yolo_pose(out, m, post, p):
    # (1,300,57) = 正規化 x1 y1 x2 y2, score, class, 17 ×（x, y, 可視度）
    th, items = float(p.get("threshold", 0.4)), []
    for r in next(iter(out.values()))[0]:
        if r[4] < th:
            continue
        x1, y1 = to_orig(r[0] * m["iw"], r[1] * m["ih"], m)
        x2, y2 = to_orig(r[2] * m["iw"], r[3] * m["ih"], m)
        kps = [[*map(float, to_orig(k[0] * m["iw"], k[1] * m["ih"], m)), float(k[2])] for k in r[6:].reshape(17, 3)]
        items.append({"label": "person", "score": float(r[4]), "box": [float(x1), float(y1), float(x2), float(y2)], "keypoints": kps})
    return {"kind": "boxes", "items": items}


def post_alpha(out, m, post, p):
    a = np.squeeze(out[post.get("output")] if post.get("output") else next(iter(out.values())))
    if post.get("sigmoid"):
        a = 1 / (1 + np.exp(-a))
    return {"kind": "mask", "cutout": True, "mask": png_data_url(crop_resize(a * 255, m))}


def post_depth(out, m, post, p):
    d = np.squeeze(out[post.get("output", "predicted_depth")]).astype(np.float32)
    if post.get("inverse"):  # 「大きいほど遠い」深度を、表示用に「大きいほど近い」へ
        d = 1 / np.maximum(d, 1e-6)
    d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
    res = {"kind": "depth", "image": png_data_url(crop_resize(d * 255, m))}
    if post.get("intrinsics") in out:
        fx = float(np.squeeze(out[post["intrinsics"]])[0, 0])
        res["note"] = f"推定した水平画角 {math.degrees(2 * math.atan(m['cw'] / 2 / fx)):.0f}°（DA3 はカメラの内部パラメータも出す）"
    return res


def post_ultra_e2e_detect(out, m, post, p):
    # Ultralytics の end2end 書き出し: (1, 300, 6) = x1, y1, x2, y2, スコア, クラス（入力のピクセル座標）
    th, items = float(p.get("threshold", 0.4)), []
    for r in next(iter(out.values()))[0]:
        if r[4] < th:
            continue
        x1, y1 = to_orig(r[0], r[1], m)
        x2, y2 = to_orig(r[2], r[3], m)
        items.append({"label": COCO[int(r[5])], "score": float(r[4]), "box": [float(x1), float(y1), float(x2), float(y2)]})
    return {"kind": "boxes", "items": items}


def post_ultra_e2e_pose(out, m, post, p):
    # (1, 300, 57) = x1, y1, x2, y2, スコア, クラス, 17 ×（x, y, 可視度）（入力のピクセル座標）
    th, items = float(p.get("threshold", 0.4)), []
    for r in next(iter(out.values()))[0]:
        if r[4] < th:
            continue
        x1, y1 = to_orig(r[0], r[1], m)
        x2, y2 = to_orig(r[2], r[3], m)
        kps = [[*map(float, to_orig(k[0], k[1], m)), float(k[2])] for k in r[6:].reshape(17, 3)]
        items.append({"label": "person", "score": float(r[4]), "box": [float(x1), float(y1), float(x2), float(y2)], "keypoints": kps})
    return {"kind": "boxes", "items": items}


def post_deim_wholebody(out, m, post, p):
    # PINTO の DEIMv2 Wholebody: (1, Q, 6) = クラス, x1, y1, x2, y2（入力に対する正規化）, スコア。表示するクラスは post.show
    th, show, items = float(p.get("threshold", 0.35)), set(post["show"]), []
    for r in next(iter(out.values()))[0]:
        label = post["classes"][int(r[0])]
        if r[5] < th or label not in show:
            continue
        x1, y1 = to_orig(r[1] * m["iw"], r[2] * m["ih"], m)
        x2, y2 = to_orig(r[3] * m["iw"], r[4] * m["ih"], m)
        items.append({"label": label, "score": float(r[5]), "box": [float(x1), float(y1), float(x2), float(y2)]})
    return {"kind": "boxes", "items": items}


POST = {"ultra_e2e_detect": post_ultra_e2e_detect, "ultra_e2e_pose": post_ultra_e2e_pose, "deim_wholebody": post_deim_wholebody, "yolo_detect": post_yolo_detect, "yolo_pose": post_yolo_pose, "alpha": post_alpha, "depth": post_depth}


class OnnxAdapter(Adapter):
    def load(self):
        import onnxruntime as ort
        from huggingface_hub import hf_hub_download
        o = self.e["onnx"]
        # 外部データ（.onnx_data）は ONNX 本体と同じディレクトリに無いと onnxruntime が拒否する。
        # HF のキャッシュは実体が別ディレクトリの blobs に分かれるので、実ファイルとして models/ に落とす
        if o.get("server_file"):  # このサーバーだけが配るモデル（models/ 以下、tools/export_yolo26_dynamic.py など）
            path = str(ROOT / "models" / o["server_file"])
        elif o.get("path"):  # リポジトリに同梱したモデル（web/pinto など）
            path = str(ROOT / "web" / o["path"])
        else:
            local = ROOT / "models" / o["repo"]
            path = hf_hub_download(o["repo"], o["file"], local_dir=local)
            if o.get("data"):
                hf_hub_download(o["repo"], o["data"], local_dir=local)
        # CoreML EP は既定の NeuralNetwork 形式だと YOLO26 の出力が壊れる（全スコアが負）。MLProgram なら CPU と一致し約2倍速い。
        # BiRefNet と DA3 は MLProgram への変換に失敗するので models.json で providers: cpu にしている
        if self.e.get("server", {}).get("providers") == "cpu":
            providers, self.device = ["CPUExecutionProvider"], "cpu"
        else:
            providers, self.device = [("CoreMLExecutionProvider", {"ModelFormat": "MLProgram"}), "CPUExecutionProvider"], "coreml"
        self.sess = ort.InferenceSession(path, providers=providers)
        self.outputs = [o.name for o in self.sess.get_outputs()]

    def run(self, im, p):
        x, meta = preprocess(im, self.e["pre"], p.get("input_size"))
        out = dict(zip(self.outputs, self.sess.run(None, {self.e["pre"]["input"]: x})))
        return POST[self.e["post"]["type"]](out, meta, self.e["post"], p)


# ---------- ライブラリを使う専用の実装 ----------

def boxes_from_pipeline(out):
    return {"kind": "boxes", "items": [{"label": o["label"], "score": float(o["score"]),
                                        "box": [o["box"]["xmin"], o["box"]["ymin"], o["box"]["xmax"], o["box"]["ymax"]]} for o in out]}


class HfDetect(Adapter):
    def load(self):
        from transformers import pipeline
        self.pipe = pipeline("object-detection", model=self.e["repo"], device=DEVICE)

    def run(self, im, p):
        return boxes_from_pipeline(self.pipe(im, threshold=float(p.get("threshold", 0.4))))


def seg_color(label, k=0):
    """クラス名から決まる色（ブラウザの worker.js の segColor と同じ式）。同じクラスの k 番目は明るさを変える"""
    import colorsys
    h = (sum(map(ord, label)) * 47) % 360
    r, g, b = colorsys.hls_to_rgb(h / 360, [0.55, 0.42, 0.68][k % 3], 0.7)
    return [round(r * 255), round(g * 255), round(b * 255)]


def compose_segments(segs, size):
    """[(label, 0/1 の配列)] を、色分けした RGBA 画像と凡例（ラベルごとの色・個数・面積）にする"""
    W, H = size
    rgba, seen, legend = np.zeros((H, W, 4), np.uint8), {}, {}
    for label, m in sorted(segs, key=lambda x: -int((x[1] > 0).sum())):  # 大きい順に塗り、小さい物を上に
        k = seen.get(label, 0)
        seen[label] = k + 1
        c = seg_color(label, k)
        rgba[m > 0] = [*c, 255]
        g = legend.setdefault(label, {"label": label, "color": "#%02x%02x%02x" % tuple(seg_color(label)), "count": 0, "area": 0})
        g["count"] += 1
        g["area"] += float((m > 0).sum()) / (W * H)
    return png_data_url(rgba), sorted(legend.values(), key=lambda g: -g["area"])


class HfSegment(Adapter):
    # セマンティック / パノプティック（subtask は models.json）。パノプティックは transformers の image-segmentation パイプライン。
    # セマンティックはプロセッサとモデルを直接呼ぶ: EoMT は横長の画像を正方形の区画に分けて推論し、区画の位置（patch_offsets）で
    # つなぎ直すが、パイプラインはそれをモデルに渡さないので、塗り分けが上下にずれる（transformers 5.17）
    def load(self):
        from transformers import AutoModelForUniversalSegmentation, AutoProcessor, pipeline
        if self.e["subtask"] == "semantic":
            self.proc = AutoProcessor.from_pretrained(self.e["repo"])
            self.model = AutoModelForUniversalSegmentation.from_pretrained(self.e["repo"]).to(DEVICE).eval()
        else:
            self.pipe = pipeline("image-segmentation", model=self.e["repo"], device=DEVICE)

    @torch.inference_mode()
    def run(self, im, p):
        if self.e["subtask"] == "semantic":
            inputs = self.proc(images=im, return_tensors="pt").to(DEVICE)
            out = self.model(**inputs)
            W, H = im.size
            if getattr(self.proc, "do_pad", False) and not getattr(self.proc, "do_split_image", False):
                # EoMT のプロセッサは長辺を 512 に縮めて右下を余白で埋めた正方形にするが、後処理は余白ごと元の大きさに
                # 引き伸ばすので、横長の画像では縦がつぶれる（transformers 5.17）。入力の大きさで塗り分けを出し、
                # 余白を切ってから元の大きさに戻す
                side = inputs["pixel_values"].shape[-1]
                seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[(side, side)])[0].cpu().numpy()
                r = side / max(W, H)
                seg = seg[: round(H * r), : round(W * r)]
                seg = np.asarray(Image.fromarray(seg.astype(np.int32)).resize((W, H), Image.NEAREST))
            else:
                seg = self.proc.post_process_semantic_segmentation(out, target_sizes=[(H, W)])[0].cpu().numpy()
            id2label = self.model.config.id2label
            segs = [(id2label[int(c)], (seg == c).astype(np.uint8)) for c in np.unique(seg)]
        else:
            segs = [(o["label"], np.asarray(o["mask"])) for o in self.pipe(im, subtask=self.e["subtask"])]
        img, legend = compose_segments(segs, im.size)
        return {"kind": "segmap", "image": img, "legend": legend, "count": len(segs), "subtask": self.e["subtask"]}


class HfDepth(Adapter):
    def load(self):
        from transformers import pipeline
        self.pipe = pipeline("depth-estimation", model=self.e["repo"], device=DEVICE)

    def run(self, im, p):
        d = self.pipe(im)["predicted_depth"].squeeze().float().cpu().numpy()
        d = (d - d.min()) / max(float(d.max() - d.min()), 1e-6)
        return {"kind": "depth", "image": png_data_url(np.asarray(Image.fromarray((d * 255).astype(np.uint8)).resize(im.size, Image.BILINEAR)))}


class HfGdino(Adapter):
    # transformers の zero-shot-object-detection パイプラインは Grounding DINO だとスコアが極端に低く、
    # 猫の写真で cat を1つも返さなかった（5.17.0）。プロセッサに候補名のリストを直接渡して呼ぶ。
    # 後処理は自前: 標準の後処理は閾値を超えた単語をつなげて "orange lemon" のような混ざった名前を返すので、
    # 候補ごとの単語の範囲で確率の最大を比べ、枠ごとに候補を1つ選ぶ
    def load(self):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor
        self.proc = AutoProcessor.from_pretrained(self.e["repo"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(self.e["repo"]).to(DEVICE).eval()

    @torch.inference_mode()
    def run(self, im, p):
        labels = [s.strip() for s in p.get("labels", "person").split(",") if s.strip()]
        inputs = self.proc(images=im, text=[labels], return_tensors="pt").to(DEVICE)
        out = self.model(**inputs)
        prob = out.logits[0].sigmoid().float().cpu().numpy()                  # (クエリ, トークン)
        spans = label_spans(inputs.input_ids[0].tolist(), self.proc.tokenizer.convert_tokens_to_ids("."), len(labels))
        per_label = np.stack([prob[:, sp].max(1) for sp in spans], 1)       # (クエリ, 候補)
        th, W, H, items = float(p.get("threshold", 0.3)), im.size[0], im.size[1], []
        for q in np.where(per_label.max(1) > th)[0]:
            k = int(per_label[q].argmax())
            cx, cy, w, h = out.pred_boxes[0, q].float().cpu().numpy() * [W, H, W, H]
            items.append({"label": labels[k], "score": float(per_label[q, k]), "box": [float(cx - w / 2), float(cy - h / 2), float(cx + w / 2), float(cy + h / 2)]})
        return {"kind": "boxes", "items": items}


def label_spans(ids, dot, n):
    """"a. b. c." のトークン列から、候補ごとのトークン位置（[CLS] や "." を除く）を取り出す"""
    spans, cur = [], []
    for i, t in enumerate(ids[1:], start=1):
        if t == dot:
            spans.append(cur)
            cur = []
        elif len(spans) < n and t not in (0, 101, 102):
            cur.append(i)
    return [sp or [0] for sp in (spans + [cur])[:n]]


class HfSam2(Adapter):
    def load(self):
        from collections import OrderedDict
        from transformers import Sam2Model, Sam2Processor
        self.model = Sam2Model.from_pretrained(self.e["repo"]).to(DEVICE).eval()
        self.proc = Sam2Processor.from_pretrained(self.e["repo"])
        self.cache = OrderedDict()

    @torch.inference_mode()
    def run(self, im, p):
        from fastapi import HTTPException
        pts = p.get("points") or []
        if not pts:
            raise HTTPException(400, "画像をクリックして点を指定してください")
        inputs = self.proc(images=im, input_points=[[[[x, y] for x, y, _ in pts]]],
                           input_labels=[[[int(l) for *_, l in pts]]], return_tensors="pt").to(DEVICE)
        key = p["_image_key"]
        if key not in self.cache:  # 同じ画像への2回目以降のクリックでは画像エンコーダを省く
            self.cache[key] = self.model.get_image_embeddings(inputs["pixel_values"])
            while len(self.cache) > 4:
                self.cache.popitem(last=False)
        out = self.model(input_points=inputs["input_points"], input_labels=inputs["input_labels"],
                         image_embeddings=self.cache[key], multimask_output=len(pts) == 1)
        masks = self.proc.post_process_masks(out.pred_masks.cpu(), inputs["original_sizes"].cpu())[0][0]
        scores = out.iou_scores[0, 0].float().cpu()
        best = int(scores.argmax())
        return {"kind": "mask", "mask": png_data_url((masks[best].numpy() > 0).astype(np.uint8) * 255), "score": float(scores[best])}


class OllamaVlm(Adapter):
    device = "ollama"

    def load(self):
        pass  # Ollama 側が読み込む

    def run(self, im, p):
        from fastapi import HTTPException
        buf = io.BytesIO()
        im.save(buf, format="JPEG", quality=90)
        r = requests.post(f"{OLLAMA}/api/chat", timeout=600, json={
            "model": os.environ.get("CVPG_OLLAMA_MODEL", self.e["ollama"]), "stream": False, "think": False,
            "messages": [{"role": "user", "content": p.get("prompt") or "この画像を日本語で詳しく説明してください。",
                          "images": [base64.b64encode(buf.getvalue()).decode()]}]})
        if r.status_code != 200:
            raise HTTPException(502, f"Ollama: {r.text[:300]}")
        return {"kind": "text", "text": r.json()["message"]["content"]}


ADAPTERS = {"onnx": OnnxAdapter, "hf-segment": HfSegment, "hf-detect": HfDetect, "hf-depth": HfDepth, "hf-gdino": HfGdino,
            "hf-sam2": HfSam2, "ollama-vlm": OllamaVlm}
