// 汎用 ONNX adapter のブラウザ版（onnxruntime-web を直接使う）。サーバー版は adapters.py の OnnxAdapter。
// 前処理（pre）と後処理（post）は models.json に名前で書き、ここと adapters.py に同じ部品を持つ。
// 部品を足したら両方に足すと、同じモデルをブラウザとサーバーで同じ手順で比べられる。
import { COCO } from "./coco.js";

const HF = "https://huggingface.co";

// Cache API に保存して2回目以降はダウンロードしない（file:// などで Cache API が使えない時は毎回取得）
async function fetchModelFile(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open("cvpg-onnx");
    const hit = await cache.match(url);
    if (hit) return new Uint8Array(await hit.arrayBuffer());
  } catch { cache = null; }
  const res = await fetch(url);
  if (!res.ok) throw new Error(`${url}: HTTP ${res.status}`);
  const total = +res.headers.get("content-length") || 0;
  const reader = res.body.getReader(), chunks = [];
  let got = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    chunks.push(value);
    got += value.length;
    if (total) onProgress?.(url.split("/").pop(), (got / total) * 100);
  }
  const buf = new Uint8Array(got);
  let o = 0;
  for (const c of chunks) { buf.set(c, o); o += c.length; }
  try { await cache?.put(url, new Response(buf)); } catch { /* 容量不足などは無視 */ }
  return buf;
}

export async function onnxLoad(ort, e, device, onProgress) {
  const base = `${HF}/${e.onnx.repo}/resolve/main/`;
  const model = await fetchModelFile(base + e.onnx.file, onProgress);
  const opts = { executionProviders: [device === "webgpu" ? "webgpu" : "wasm"], graphOptimizationLevel: "all" };
  if (e.onnx.data) {
    opts.externalData = [{ path: e.onnx.data.split("/").pop(), data: await fetchModelFile(base + e.onnx.data, onProgress) }];
  }
  return ort.InferenceSession.create(model, opts);
}

// ---------- 前処理: 画像 → 入力テンソル。meta は 入力座標 = 元座標 × (sx, sy) + (ox, oy) と内容のある範囲 (cw, ch) ----------

function preprocess(ort, bitmap, pre) {
  const W = bitmap.width, H = bitmap.height;
  let iw, ih, meta;
  const c = new OffscreenCanvas(1, 1), x = c.getContext("2d", { willReadFrequently: true });
  if (pre.resize === "letterbox") {
    [iw, ih] = pre.size;
    const r = Math.min(iw / W, ih / H), cw = Math.round(W * r), ch = Math.round(H * r);
    const ox = Math.floor((iw - cw) / 2), oy = Math.floor((ih - ch) / 2);
    c.width = iw; c.height = ih;
    const g = pre.pad_value ?? 114;
    x.fillStyle = `rgb(${g},${g},${g})`;
    x.fillRect(0, 0, iw, ih);
    x.drawImage(bitmap, ox, oy, cw, ch);
    meta = { sx: r, sy: r, ox, oy, iw, ih, cw, ch };
  } else if (pre.resize === "stretch") {
    [iw, ih] = pre.size;
    c.width = iw; c.height = ih;
    x.drawImage(bitmap, 0, 0, iw, ih);
    meta = { sx: iw / W, sy: ih / H, ox: 0, oy: 0, iw, ih, cw: iw, ch: ih };
  } else if (pre.resize === "keep_aspect") {
    const m = pre.multiple || 1, r = pre.short / Math.min(W, H);
    iw = Math.max(m, Math.round((W * r) / m) * m); ih = Math.max(m, Math.round((H * r) / m) * m);
    c.width = iw; c.height = ih;
    x.drawImage(bitmap, 0, 0, iw, ih);
    meta = { sx: iw / W, sy: ih / H, ox: 0, oy: 0, iw, ih, cw: iw, ch: ih };
  } else {
    throw new Error(`unknown resize ${pre.resize}`);
  }
  const px = x.getImageData(0, 0, iw, ih).data, n = iw * ih;
  const out = new Float32Array(3 * n), s = pre.scale ?? 1 / 255;
  const mean = pre.mean || [0, 0, 0], std = pre.std || [1, 1, 1];
  for (let i = 0; i < n; i++) for (let k = 0; k < 3; k++) out[k * n + i] = (px[i * 4 + k] * s - mean[k]) / std[k];
  const dims = [1, 3, ih, iw];
  for (const ax of pre.add_dims || []) dims.splice(ax, 0, 1);
  return { tensor: new ort.Tensor("float32", out, dims), meta: { ...meta, W, H } };
}

const toOrig = (x, y, m) => [(x - m.ox) / m.sx, (y - m.oy) / m.sy];
const sigmoid = (v) => 1 / (1 + Math.exp(-v));

// 入力解像度の2次元出力（0..1）から内容のある範囲を切り出し、元画像の大きさの PNG にする
async function cropResizePng(vals, w, h, m) {
  const c = new OffscreenCanvas(w, h), x = c.getContext("2d");
  const img = x.createImageData(w, h);
  for (let i = 0; i < w * h; i++) {
    const v = Math.max(0, Math.min(255, vals[i] * 255));
    img.data[i * 4] = img.data[i * 4 + 1] = img.data[i * 4 + 2] = v; img.data[i * 4 + 3] = 255;
  }
  x.putImageData(img, 0, 0);
  const fx = w / m.iw, fy = h / m.ih;
  const o = new OffscreenCanvas(m.W, m.H);
  o.getContext("2d").drawImage(c, m.ox * fx, m.oy * fy, m.cw * fx, m.ch * fy, 0, 0, m.W, m.H);
  return o.convertToBlob({ type: "image/png" });
}

// ---------- 後処理: 出力 → 結果（形式は adapters.py の冒頭を参照） ----------

const POST = {
  // YOLO26 は NMS 込みの出力: logits (1,300,80) はシグモイド前、pred_boxes (1,300,4) は入力に対する正規化 cx cy w h
  yolo_detect(out, m, post, p) {
    const L = out.logits.data, B = out.pred_boxes.data, [, Q, C] = out.logits.dims, th = p.threshold ?? 0.4, items = [];
    for (let i = 0; i < Q; i++) {
      let bi = 0;
      for (let c = 1; c < C; c++) if (L[i * C + c] > L[i * C + bi]) bi = c;
      const score = sigmoid(L[i * C + bi]);
      if (score < th) continue;
      const [cx, cy, w, h] = [B[i * 4] * m.iw, B[i * 4 + 1] * m.ih, B[i * 4 + 2] * m.iw, B[i * 4 + 3] * m.ih];
      items.push({ label: COCO[bi], score, box: [...toOrig(cx - w / 2, cy - h / 2, m), ...toOrig(cx + w / 2, cy + h / 2, m)] });
    }
    return { kind: "boxes", items };
  },
  // (1,300,57) = 正規化 x1 y1 x2 y2, score, class, 17 ×（x, y, 可視度）
  yolo_pose(out, m, post, p) {
    const t = Object.values(out)[0], D = t.data, [, Q, K] = t.dims, th = p.threshold ?? 0.4, items = [];
    for (let i = 0; i < Q; i++) {
      const r = D.subarray(i * K, i * K + K);
      if (r[4] < th) continue;
      const kps = [];
      for (let k = 0; k < 17; k++) kps.push([...toOrig(r[6 + k * 3] * m.iw, r[7 + k * 3] * m.ih, m), r[8 + k * 3]]);
      items.push({ label: "person", score: r[4], box: [...toOrig(r[0] * m.iw, r[1] * m.ih, m), ...toOrig(r[2] * m.iw, r[3] * m.ih, m)], keypoints: kps });
    }
    return { kind: "boxes", items };
  },
  async alpha(out, m, post) {
    const t = post.output ? out[post.output] : Object.values(out)[0];
    const [h, w] = t.dims.slice(-2), v = Float32Array.from(t.data, (a) => (post.sigmoid ? sigmoid(a) : a));
    return { kind: "mask", cutout: true, mask: await cropResizePng(v, w, h, m) };
  },
  async depth(out, m, post) {
    const t = out[post.output || "predicted_depth"], [h, w] = t.dims.slice(-2);
    let v = Float32Array.from(t.data);
    if (post.inverse) v = v.map((d) => 1 / Math.max(d, 1e-6)); // 「大きいほど遠い」深度を、表示用に「大きいほど近い」へ
    let lo = Infinity, hi = -Infinity;
    for (const a of v) { if (a < lo) lo = a; if (a > hi) hi = a; }
    v = v.map((a) => (a - lo) / Math.max(hi - lo, 1e-6));
    const res = { kind: "depth", image: await cropResizePng(v, w, h, m) };
    if (post.intrinsics && out[post.intrinsics]) {
      const fx = out[post.intrinsics].data[0];
      res.note = `推定した水平画角 ${((2 * Math.atan(m.cw / 2 / fx) * 180) / Math.PI).toFixed(0)}°（DA3 はカメラの内部パラメータも出す）`;
    }
    return res;
  },
};

export async function onnxRun(ort, session, e, bitmap, params) {
  const { tensor, meta } = preprocess(ort, bitmap, e.pre);
  const out = await session.run({ [e.pre.input]: tensor });
  return POST[e.post.type](out, meta, e.post, params);
}
