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

// 実行設定（e.opt）: "fp16" は onnx.file_fp16 の版を使う。"graph" を含むと WebGPU の graph capture（2回目以降は
// 記録した GPU コマンドをまとめて流すので、層ごとの発行の手間が減る）。graph capture は入出力を GPU 上に置く必要がある
// web/ に同梱したモデル: まずページと同じ場所（e.webRoot、ローカルのサーバーなら web/）から読み、
// 読めなければ（file:// で開いた時など）onnx.url（GitHub Pages）から読む
async function fetchBundled(e, onProgress) {
  if (e.webRoot) {
    try { return await fetchModelFile(new URL(e.onnx.path, e.webRoot).href, onProgress); } catch { /* 次へ */ }
  }
  return fetchModelFile(e.onnx.url, onProgress);
}

export async function onnxLoad(ort, e, device, onProgress) {
  const base = `${HF}/${e.onnx.repo}/resolve/main/`;
  // "wasm" は WebGPU があっても CPU で動かす（小さいモデルは GPU に命令を出す手間の方が大きいことがある）
  if (e.opt === "wasm") device = "wasm";
  const opt = device === "webgpu" ? e.opt || "" : "";
  const file = opt.startsWith("fp16") && e.onnx.file_fp16 ? e.onnx.file_fp16 : e.onnx.file;
  const model = e.onnx.path ? await fetchBundled(e, onProgress) : await fetchModelFile(base + file, onProgress);
  const opts = { executionProviders: [device === "webgpu" ? "webgpu" : "wasm"], graphOptimizationLevel: "all" };
  if (e.onnx.data) {
    opts.externalData = [{ path: e.onnx.data.split("/").pop(), data: await fetchModelFile(base + e.onnx.data, onProgress) }];
  }
  const graph = opt.includes("graph");
  if (graph) Object.assign(opts, { preferredOutputLocation: "gpu-buffer", enableGraphCapture: true });
  const session = await ort.InferenceSession.create(model, opts);
  return Object.assign(session, { cvpg: { graph, file, gpuInput: null } });
}

// graph capture の時は入力を毎回同じ GPU バッファに書き込み、出力は GPU から読み戻す
async function runSession(ort, session, name, tensor) {
  const g = session.cvpg;
  if (!g?.graph) return session.run({ [name]: tensor });
  const dev = ort.env.webgpu.device;
  if (!g.gpuInput || g.gpuInput.size !== tensor.data.byteLength) {
    g.gpuInput = dev.createBuffer({ size: tensor.data.byteLength, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
    g.feed = ort.Tensor.fromGpuBuffer(g.gpuInput, { dataType: "float32", dims: tensor.dims });
  }
  dev.queue.writeBuffer(g.gpuInput, 0, tensor.data);
  const out = await session.run({ [name]: g.feed });
  const cpu = {};
  for (const [k, t] of Object.entries(out)) cpu[k] = { data: await t.getData(true), dims: t.dims };
  return cpu;
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
  // (画素 × scale − mean) / std = 画素 × a + b を、チャンネルごとに1回のループで。bgr なら入力の並びを B, G, R に
  for (let k = 0; k < 3; k++) {
    const a = s / std[k], b = -mean[k] / std[k], o = k * n, src = pre.bgr ? 2 - k : k;
    for (let i = 0; i < n; i++) out[o + i] = px[i * 4 + src] * a + b;
  }
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
  // PINTO の DEIMv2 Wholebody: (1, Q, 6) = クラス, x1, y1, x2, y2（入力に対する正規化）, スコア。NMS 済み。
  // 表示するクラスは post.show（年齢・性別・向きなどの属性クラスや関節のクラスは既定で出さない）
  deim_wholebody(out, m, post, p) {
    const t = Object.values(out)[0], D = t.data, [, Q, K] = t.dims, th = p.threshold ?? 0.35, show = new Set(post.show), items = [];
    for (let i = 0; i < Q; i++) {
      const r = D.subarray(i * K, i * K + K), label = post.classes[r[0] | 0];
      if (r[5] < th || !show.has(label)) continue;
      items.push({ label, score: r[5], box: [...toOrig(r[1] * m.iw, r[2] * m.ih, m), ...toOrig(r[3] * m.iw, r[4] * m.ih, m)] });
    }
    return { kind: "boxes", items };
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

// 切り出した画像ごとの特徴（ReID など）。ONNX のバッチが固定（pre.batch）なら、足りない分は最後の1枚で埋めて切り捨てる
// pre.batch: 固定バッチ数、"dynamic" なら全部を1回で。pre.seq: フレーム列を入れるモデル（入力 (1, C, T, H, W)）で、
// bitmaps は T 枚ずつの組として並べて渡す
export async function onnxEmbed(ort, session, e, bitmaps) {
  const T = e.pre.seq;
  if (T) {
    const feats = [];
    for (let i = 0; i + T <= bitmaps.length; i += T) {
      const ts = bitmaps.slice(i, i + T).map((b) => preprocess(ort, b, e.pre).tensor);
      const [, c, h, w] = ts[0].dims, hw = h * w, data = new Float32Array(c * T * hw);
      ts.forEach((t, k) => { for (let ch = 0; ch < c; ch++) data.set(t.data.subarray(ch * hw, (ch + 1) * hw), (ch * T + k) * hw); });
      const out = await session.run({ [e.pre.input]: new ort.Tensor("float32", data, [1, c, T, h, w]) });
      feats.push(Object.values(out)[0].data.slice(0, 1));
    }
    bitmaps.forEach((b) => b.close?.());
    return feats;
  }
  const feats = [], bs = e.pre.batch === "dynamic" ? bitmaps.length : e.pre.batch || 1;
  if (!bitmaps.length) return feats;
  for (let i = 0; i < bitmaps.length; i += bs) {
    const chunk = bitmaps.slice(i, i + bs);
    const ts = chunk.map((b) => preprocess(ort, b, e.pre).tensor);
    while (ts.length < bs) ts.push(ts[ts.length - 1]);
    const n = ts[0].data.length, data = new Float32Array(n * bs);
    ts.forEach((t, k) => data.set(t.data, k * n));
    const out = await session.run({ [e.pre.input]: new ort.Tensor("float32", data, [bs, ...ts[0].dims.slice(1)]) });
    const o = Object.values(out)[0], d = o.dims[1] ?? 1; // 出力が (N,) の確率だけのモデルは1次元
    for (let k = 0; k < chunk.length; k++) feats.push(o.data.slice(k * d, (k + 1) * d));
  }
  bitmaps.forEach((b) => b.close?.());
  return feats;
}

export async function onnxRun(ort, session, e, bitmap, params) {
  const t0 = performance.now();
  const { tensor, meta } = preprocess(ort, bitmap, e.pre);
  const t1 = performance.now();
  const out = await runSession(ort, session, e.pre.input, tensor);
  const t2 = performance.now();
  const res = await POST[e.post.type](out, meta, e.post, params);
  // 内訳（端末ごとにどこが重いかを見る）: 前処理（縮小・正規化）/ ONNX の実行（GPU との転送を含む）/ 後処理
  res.breakdown = { pre: t1 - t0, run: t2 - t1, post: performance.now() - t2 };
  return res;
}
