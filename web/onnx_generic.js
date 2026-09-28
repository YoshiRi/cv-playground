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
  const model = e.onnx.server_file
    ? await fetchModelFile(new URL("local-models/" + e.onnx.server_file, e.webRoot).href, onProgress)
    : e.onnx.path ? await fetchBundled(e, onProgress) : await fetchModelFile(base + file, onProgress);
  // WebGPU は { name } の形で渡す。onnxruntime-web 1.30 は文字列の "webgpu" だと enableGraphCapture を WebGPU EP に渡さず、
  // graph capture が黙って無効になる（EP の設定のログが "graph capture enable: 0" のまま）
  const opts = { executionProviders: [device === "webgpu" ? { name: "webgpu" } : "wasm"], graphOptimizationLevel: "all" };
  if (e.onnx.data) {
    opts.externalData = [{ path: e.onnx.data.split("/").pop(), data: await fetchModelFile(base + e.onnx.data, onProgress) }];
  }
  // 詳細計測（?profile=1 の時だけ）: onnxruntime の profiler で、ノードごとの時間と WebGPU の命令ごとの GPU 時間を記録する
  if (e.profile) opts.enableProfiling = true;
  const create = (o) => ort.InferenceSession.create(model, o);
  // 入力の大きさが決まっているモデルは、ONNX に可変（batch_size, N, H, W など）と書かれた次元を固定する。
  // 形の計算が CPU に回らずに済み、GPU の命令も形に合わせて作られる（DEIMv2 Atto は M4 で 18 → 11ms）。名前はモデルごとに違うので、
  // 一度作ったセッションの inputMetadata から読む（固定するものが無ければそのセッションをそのまま使う）
  const fixed = fixedDims(e);
  let session = null;
  if (fixed) {
    session = await create(opts);
    const shape = session.inputMetadata?.find((m) => m.name === e.pre.input)?.shape, fdo = {};
    if (shape?.length === fixed.length) shape.forEach((d, i) => { if (typeof d === "string") fdo[d] = fixed[i]; });
    if (Object.keys(fdo).length) { opts.freeDimensionOverrides = fdo; await session.release(); session = null; }
  }
  // graph capture（2回目以降は記録した GPU の命令をまとめて流す）は入出力を GPU 上に置く。全部のノードが WebGPU で動くモデルだけ
  // （CPU に回るノードがあると作れない）なので、作れない時は graph capture なしにする。作れないと分かったモデルは覚えておき、
  // 同じ Worker で読み直す時は試さない
  const graph = opt.includes("graph") && !noGraph.has(e.key);
  let graphFallback = opt.includes("graph") && noGraph.has(e.key);
  if (graph) {
    await session?.release();
    try {
      session = await create({ ...opts, preferredOutputLocation: "gpu-buffer", enableGraphCapture: true });
    } catch (err) {
      if (!/graph capture/i.test(String(err?.message || err))) throw err;
      noGraph.add(e.key);
      session = null;
      graphFallback = true;
    }
  }
  session ??= await create(opts);
  const fp16 = file === e.onnx.file_fp16 && !e.onnx.server_file && !e.onnx.path;
  return Object.assign(session, { cvpg: { graph: graph && !graphFallback, graphFallback, fp16, fdo: opts.freeDimensionOverrides, file, gpuInput: null } });
}
const noGraph = new Set(); // graph capture を作れなかったモデル（entry.key）

// 入力の形が1通りに決まるモデルの入力の次元（前処理と同じ並び）。入力サイズ可変・バッチ・フレーム列のモデルは決まらないので null
function fixedDims(e) {
  const pre = e.pre;
  if (!pre.size || pre.dynamic || pre.batch || pre.seq || pre.resize === "keep_aspect") return null;
  const dims = [1, 3, pre.size[1], pre.size[0]];
  for (const ax of pre.add_dims || []) dims.splice(ax, 0, 1);
  return dims;
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
  // 出力の読み戻しはまとめて待つ（1つずつ待つと GPU との往復が出力の数だけ増える。YOLO26n で 2ms ほど違う）
  const ents = Object.entries(out), datas = await Promise.all(ents.map(([, t]) => t.getData(true)));
  return Object.fromEntries(ents.map(([k, t], i) => [k, { data: datas[i], dims: t.dims }]));
}

// 詳細計測の結果をまとめる。onnxruntime-web は profiler の記録（Chrome trace の JSON）を console に出すだけなので、
// endProfiling の間だけ console の出力を拾う。onnxruntime（emscripten）は読み込み時に console.log を覚えてしまうので、
// 後から差し替えても届かない。先に（このモジュールの読み込み時に）中継する関数に替えておき、拾う間だけ行き先を変える。
// 記録は読み込みから全部の実行ぶん入っているので、最後の last 回だけ集計する。
// 1回あたりの平均: model_run（ONNX の実行の壁時計）、gpu（WebGPU の命令の GPU 時間の合計）、ops（演算の種類ごとの GPU 時間）、
// CPU に回ったノード、GPU との転送（入力の転送と、結果の待ち＋読み戻し）
let consoleSink = null;
for (const k of ["log", "info", "warn", "error"]) {
  const orig = console[k].bind(console);
  console[k] = (...a) => (consoleSink ? consoleSink.push(a.join(" ")) : orig(...a));
}

export async function onnxProfile(session, last) {
  const lines = (consoleSink = []);
  try { await session.endProfiling(); } finally { consoleSink = null; }
  const txt = lines.join("\n"), a = txt.indexOf("[\n"), b = txt.lastIndexOf("]");
  if (a < 0 || b < a) return { error: "profiler の記録が取れなかった" };
  const ev = JSON.parse(txt.slice(a, b + 1));
  // CPU 側のノードの記録は実行の時間内に入る。GPU の命令の時刻は後からまとめて取るので実行の区切りと合わない。
  // どの実行も同じ命令の並びを出すので、GPU の命令は「全部の数 ÷ 実行の回数」ずつ順に区切り、最後の last 回ぶんを使う
  const all = ev.filter((x) => x.name === "model_run"), runs = all.slice(-last);
  if (!runs.length) return { error: "実行の記録が無い" };
  const nodes = ev.filter((x) => x.cat === "Node" && x.name.endsWith("_kernel_time") && runs.some((r) => x.ts >= r.ts && x.ts <= r.ts + r.dur));
  // graph capture の再生ではノードを1つずつ動かさないので、演算ごとの記録が無い
  if (!nodes.length) return { runs: runs.length, run_ms: Math.round(runs.reduce((s, r) => s + r.dur, 0) / runs.length / 10) / 100, note: "graph capture の再生中は演算ごとの記録が取れない（内訳を見る時は graph capture なしで測る）" };
  const allGpu = ev.filter((x) => x.cat === "Api" && x.name.includes("&")), per = Math.round(allGpu.length / all.length);
  const gpu = per ? allGpu.slice(-per * runs.length) : [];
  const n = runs.length, r2 = (v) => Math.round((v / n / 1000) * 100) / 100;
  const ops = {};
  for (const x of gpu) { const op = x.name.split("&")[1]; (ops[op] ??= { op, us: 0, n: 0 }).us += x.dur; ops[op].n++; }
  const gpuSum = gpu.reduce((s, x) => s + x.dur, 0);
  const nodeMs = (op) => r2(nodes.filter((x) => x.args?.op_name === op).reduce((s, x) => s + x.dur, 0));
  const cpu = [...new Set(nodes.filter((x) => x.args?.provider === "CPUExecutionProvider").map((x) => `${x.args.op_name} ${x.name.replace(/_kernel_time$/, "")}`))];
  return {
    runs: n, run_ms: r2(runs.reduce((s, r) => s + r.dur, 0)), gpu_ms: r2(gpuSum), dispatches: per,
    upload_ms: nodeMs("MemcpyFromHost"), readback_wait_ms: nodeMs("MemcpyToHost"), cpu_nodes: cpu,
    ops: Object.values(ops).sort((p, q) => q.us - p.us).map((o) => ({ op: o.op, ms: r2(o.us), count: Math.round(o.n / n), pct: Math.round((1000 * o.us) / gpuSum) / 10 })),
  };
}

// ---------- 前処理: 画像 → 入力テンソル。meta は 入力座標 = 元座標 × (sx, sy) + (ox, oy) と内容のある範囲 (cw, ch) ----------

function preprocess(ort, bitmap, pre, inputSize) {
  const W = bitmap.width, H = bitmap.height;
  let iw, ih, meta;
  const c = new OffscreenCanvas(1, 1), x = c.getContext("2d", { willReadFrequently: true });
  if (pre.resize === "letterbox" || pre.resize === "letterbox_rect") {
    // letterbox_rect: 長辺を S に合わせ、縦横を stride の倍数まで余白で埋めた長方形（正方形の余白の計算を省く）
    const S = pre.dynamic && inputSize ? inputSize : pre.size[0];
    const r = pre.resize === "letterbox_rect" ? S / Math.max(W, H) : Math.min(pre.size[0] / W, pre.size[1] / H);
    const cw = Math.round(W * r), ch = Math.round(H * r), st = pre.stride || 32;
    [iw, ih] = pre.resize === "letterbox_rect" ? [Math.ceil(cw / st) * st, Math.ceil(ch / st) * st] : pre.size;
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
  // Ultralytics の end2end 書き出し: (1, 300, 6) = x1, y1, x2, y2, スコア, クラス（入力のピクセル座標）
  ultra_e2e_detect(out, m, post, p) {
    const t = Object.values(out)[0], D = t.data, [, Q, K] = t.dims, th = p.threshold ?? 0.4, items = [];
    for (let i = 0; i < Q; i++) {
      const r = D.subarray(i * K, i * K + K);
      if (r[4] < th) continue;
      items.push({ label: COCO[r[5] | 0], score: r[4], box: [...toOrig(r[0], r[1], m), ...toOrig(r[2], r[3], m)] });
    }
    return { kind: "boxes", items };
  },
  // (1, 300, 57) = x1, y1, x2, y2, スコア, クラス, 17 ×（x, y, 可視度）（入力のピクセル座標）
  ultra_e2e_pose(out, m, post, p) {
    const t = Object.values(out)[0], D = t.data, [, Q, K] = t.dims, th = p.threshold ?? 0.4, items = [];
    for (let i = 0; i < Q; i++) {
      const r = D.subarray(i * K, i * K + K);
      if (r[4] < th) continue;
      const kps = [];
      for (let k = 0; k < 17; k++) kps.push([...toOrig(r[6 + k * 3], r[7 + k * 3], m), r[8 + k * 3]]);
      items.push({ label: "person", score: r[4], box: [...toOrig(r[0], r[1], m), ...toOrig(r[2], r[3], m)], keypoints: kps });
    }
    return { kind: "boxes", items };
  },
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
  const { tensor, meta } = preprocess(ort, bitmap, e.pre, params.input_size);
  const t1 = performance.now();
  const out = await runSession(ort, session, e.pre.input, tensor);
  const t2 = performance.now();
  const res = await POST[e.post.type](out, meta, e.post, params);
  // 内訳（端末ごとにどこが重いかを見る）: 前処理（縮小・正規化）/ ONNX の実行（GPU との転送を含む）/ 後処理
  res.breakdown = { pre: t1 - t0, run: t2 - t1, post: performance.now() - t2 };
  return res;
}
