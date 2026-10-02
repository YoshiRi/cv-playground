// 汎用 ONNX adapter のブラウザ版（onnxruntime-web を直接使う）。サーバー版は adapters.py の OnnxAdapter。
// 前処理（pre）と後処理（post）は models.json に名前で書き、ここと adapters.py に同じ部品を持つ。
// 部品を足したら両方に足すと、同じモデルをブラウザとサーバーで同じ手順で比べられる。
import { COCO } from "./coco.js";

const HF = "https://huggingface.co";

// Cache API に保存して2回目以降はダウンロードしない（file:// などで Cache API が使えない時は毎回取得）。
// キャッシュは開いたサイト（配信元）ごとに分かれ、スマホでは容量が減ると消されることもある。
// サーバー版では Hugging Face・GitHub の raw ファイルを server.py の /mirror/ 経由で取る（サーバーが Mac のディスクに保存して配るので、
// ブラウザのキャッシュが無くても 2 回目からはネットに取りに行かない）。キャッシュの鍵は元の URL のまま
let MIRROR = null; // 例: "https://host/mirror/"（サーバー版だけ。画面が model.mirror で渡す）
export function setMirror(m) { MIRROR = m || null; }
const MIRROR_SITES = [["https://huggingface.co/", "hf/"], ["https://raw.githubusercontent.com/", "gh/"]];
export const viaMirror = (url) => {
  const s = MIRROR && MIRROR_SITES.find(([pre]) => url.startsWith(pre));
  return s ? MIRROR + s[1] + url.slice(s[0].length) : url;
};
// 読み込んだファイルの取得元（画面の「読み込み」の横に出す）。worker.js が読み込みごとに takeFetchLog で取り出す
let fetchLog = [];
export function takeFetchLog() { const l = fetchLog; fetchLog = []; return l; }

async function fetchModelFile(url, onProgress) {
  let cache = null;
  try {
    cache = await caches.open("cvpg-onnx");
    const hit = await cache.match(url);
    if (hit) { const b = new Uint8Array(await hit.arrayBuffer()); fetchLog.push({ from: "cache", bytes: b.length }); return b; }
  } catch { cache = null; }
  const src = viaMirror(url);
  const res = await fetch(src);
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
  // サーバーの写しは、Mac のディスクにあったか（X-Mirror: hit）、今ネットから取ったか（miss）を返す
  fetchLog.push({ from: src === url ? "net" : res.headers.get("X-Mirror") === "hit" ? "mirror" : "mirror-net", bytes: got });
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
  let model = e.onnx.server_file
    ? await fetchModelFile(new URL("local-models/" + e.onnx.server_file, e.webRoot).href, onProgress)
    : e.onnx.path ? await fetchBundled(e, onProgress)
    : !e.onnx.repo && e.onnx.url ? await fetchModelFile(e.onnx.url, onProgress) // Hugging Face 以外の配布元（CORS が許可されている所）
    : await fetchModelFile(base + file, onProgress);
  // onnx.cut: 途中の値で切り、そこより後ろのノードを消す（出力の頭が CPU に回って graph capture を作れないモデル用。
  // 重みのライセンス上、書き換えた ONNX は配らず、取得した ONNX をこの端末で書き換える）
  if (e.onnx.cut) model = cutOnnx(model, e.onnx.cut);
  // WebGPU は { name } の形で渡す。onnxruntime-web 1.30 は文字列の "webgpu" だと enableGraphCapture を WebGPU EP に渡さず、
  // graph capture が黙って無効になる（EP の設定のログが "graph capture enable: 0" のまま）
  const opts = { executionProviders: [device === "webgpu" ? { name: "webgpu" } : "wasm"], graphOptimizationLevel: "all" };
  if (e.onnx.data) {
    opts.externalData = [{ path: e.onnx.data.split("/").pop(), data: await fetchModelFile(base + e.onnx.data, onProgress) }];
  }
  // 詳細計測（?profile=1 の時だけ）: onnxruntime の profiler で、ノードごとの時間と WebGPU の命令ごとの GPU 時間を記録する
  if (e.profile) opts.enableProfiling = true;
  // 入力の形の固定: ONNX に可変（batch_size, N, H, W など）と書かれた次元を固定する。形の計算が CPU に回らずに済み、
  // GPU の命令も形に合わせて作られ、graph capture も使えるようになることが多い（DEIMv2 Atto は M4 で 18 → 11ms）。
  // 次元の名前はモデルごとに違うので、一度作ったセッションの inputMetadata から読む。
  // - 入力の大きさが pre.size で決まるモデル: ここで固定する
  // - 大きさが画像や「モデル入力（長辺）」で変わるモデル（WebGPU のみ）: 実行時に形が分かった所で作り直す（onnxRun）
  // onnx.graph_capture: false のモデルは graph capture を使わない（作れても出力が壊れるもの。DA3 small は深度が2値になる）
  const wantGraph = opt.includes("graph") && e.onnx.graph_capture !== false, fixed = fixedDims(e);
  const dynamicShape = device === "webgpu" && !fixed && !e.pre.batch && !e.pre.seq;
  let session = await ort.InferenceSession.create(model, opts);
  const shape = session.inputMetadata?.find((m) => m.name === e.pre.input)?.shape;
  const fdo = fixed ? fdoFor(shape, fixed) : null;
  let graph = false, graphFallback = false;
  if (!dynamicShape && (fdo || wantGraph)) {
    await session.release();
    ({ session, graph, graphFallback } = await createSession(ort, e, model, fdo ? { ...opts, freeDimensionOverrides: fdo } : opts, wantGraph));
  }
  const fp16 = file === e.onnx.file_fp16 && !e.onnx.server_file && !e.onnx.path;
  const cvpg = { webgpu: device === "webgpu", graph, graphFallback, fp16, fdo, file, gpuInput: null, dynamicShape, shapeKey: null };
  if (dynamicShape) {
    cvpg.rebuild = async (dims) => {
      const f = fdoFor(shape, dims), r = await createSession(ort, e, model, f ? { ...opts, freeDimensionOverrides: f } : opts, wantGraph);
      return Object.assign(r.session, { cvpg: { ...cvpg, graph: r.graph, graphFallback: r.graphFallback, fdo: f, gpuInput: null, tex: null, shapeKey: dims.join("x") } });
    };
  }
  return Object.assign(session, { cvpg });
}

// ---------- ONNX を途中で切る（protobuf を直接たどる。外部のライブラリは使わない） ----------
// ModelProto の graph（7）の中の node（1）を、outputs の値を作るのに要るものだけ残し（後ろからたどる）、出力（12）を差し替える。
// 切った値を float に直す Cast を足す（fp16 版では値が fp16 のため。float のモデルでは何もしない）。出力は cut_0, cut_1, …
const utf8 = new TextDecoder(), enc = new TextEncoder();
function readVarint(b, p) {
  let x = 0, s = 1;
  for (;;) { const c = b[p++]; x += (c & 0x7f) * s; if (c < 0x80) return [x, p]; s *= 128; }
}
function pbFields(b, start, end) {
  const out = [];
  for (let p = start; p < end;) {
    const s = p;
    let key, ps, pe;
    [key, p] = readVarint(b, p);
    const f = Math.floor(key / 8), wt = key & 7;
    if (wt === 0) { [, p] = readVarint(b, p); ps = s; pe = p; }
    else if (wt === 1) { p += 8; ps = s; pe = p; }
    else if (wt === 5) { p += 4; ps = s; pe = p; }
    else if (wt === 2) { let len; [len, ps] = readVarint(b, p); p = ps + len; pe = p; }
    else throw new Error(`ONNX の読み取りに失敗（wire type ${wt}）`);
    out.push({ f, s, ps, pe });
  }
  return out;
}
const pbVarint = (n) => { const a = []; while (n >= 128) { a.push((n % 128) | 128); n = Math.floor(n / 128); } a.push(n); return a; };
const pbCat = (parts) => { const n = parts.reduce((s, x) => s + x.length, 0), o = new Uint8Array(n); let k = 0; for (const x of parts) { o.set(x, k); k += x.length; } return o; };
const pbLen = (f, bytes) => pbCat([Uint8Array.from([...pbVarint(f * 8 + 2), ...pbVarint(bytes.length)]), bytes]);
const pbStr = (f, s) => pbLen(f, enc.encode(s));
const pbInt = (f, v) => Uint8Array.from([...pbVarint(f * 8), ...pbVarint(v)]);

export function cutOnnx(model, outputs) {
  const top = pbFields(model, 0, model.length), g = top.find((x) => x.f === 7);
  const gf = pbFields(model, g.ps, g.pe);
  const nodes = gf.filter((x) => x.f === 1).map((x) => {
    const nf = pbFields(model, x.ps, x.pe), s = (k) => nf.filter((y) => y.f === k).map((y) => utf8.decode(model.subarray(y.ps, y.pe)));
    return { x, ins: s(1), outs: s(2) };
  });
  const need = new Set(outputs), keep = new Set();
  for (let i = nodes.length - 1; i >= 0; i--) {
    const n = nodes[i];
    if (n.outs.some((o) => need.has(o))) { keep.add(n.x); for (const v of n.ins) if (v) need.add(v); }
  }
  if (!outputs.every((o) => nodes.some((n) => n.outs.includes(o)))) throw new Error(`切る場所（${outputs.join(", ")}）が ONNX に無い`);
  const parts = gf.filter((x) => (x.f !== 1 || keep.has(x)) && x.f !== 12).map((x) => model.subarray(x.s, x.pe));
  outputs.forEach((name, i) => {
    const attr = pbCat([pbStr(1, "to"), pbInt(3, 1), pbInt(20, 2)]); // to = FLOAT（AttributeProto: name, i, type=INT）
    parts.push(pbLen(1, pbCat([pbStr(1, name), pbStr(2, `cut_${i}`), pbStr(3, `cut_cast_${i}`), pbStr(4, "Cast"), pbLen(5, attr)])));
    parts.push(pbLen(12, pbCat([pbStr(1, `cut_${i}`), pbLen(2, pbLen(1, pbInt(1, 1)))]))); // ValueInfo: name, type.tensor_type.elem_type = FLOAT
  });
  return pbCat(top.map((x) => (x === g ? pbLen(7, pbCat(parts)) : model.subarray(x.s, x.pe))));
}

// graph capture（2回目以降は記録した GPU の命令をまとめて流す）は入出力を GPU 上に置く。全部のノードが WebGPU で動くモデルだけ
// （CPU に回るノードがあると作れない）なので、作れない時は graph capture なしで作る。作れないと分かったモデルは覚えておき、
// 同じ Worker で作り直す時は試さない
async function createSession(ort, e, model, opts, wantGraph) {
  if (wantGraph && !noGraph.has(e.key)) {
    try {
      return { session: await ort.InferenceSession.create(model, { ...opts, preferredOutputLocation: "gpu-buffer", enableGraphCapture: true }), graph: true, graphFallback: false };
    } catch (err) {
      if (!/graph capture/i.test(String(err?.message || err))) throw err;
      noGraph.add(e.key);
    }
  }
  return { session: await ort.InferenceSession.create(model, opts), graph: false, graphFallback: wantGraph };
}
const noGraph = new Set(); // graph capture を作れなかったモデル（entry.key）

// 可変の次元の名前 → 固定する値（shape は inputMetadata の形、dims は実際の入力の形）
function fdoFor(shape, dims) {
  const fdo = {};
  if (shape?.length === dims.length) shape.forEach((d, i) => { if (typeof d === "string") fdo[d] = dims[i]; });
  return Object.keys(fdo).length ? fdo : null;
}

// 入力の形が1通りに決まるモデルの入力の次元（前処理と同じ並び）。入力サイズ可変・バッチ・フレーム列のモデルは決まらないので null
function fixedDims(e) {
  const pre = e.pre;
  if (!pre.size || pre.dynamic || pre.batch || pre.seq || pre.resize === "keep_aspect") return null;
  const dims = [1, 3, pre.size[1], pre.size[0]];
  for (const ax of pre.add_dims || []) dims.splice(ax, 0, 1);
  return dims;
}

// graph capture の時は入力を毎回同じ GPU バッファに書き込み（GPU の前処理なら書き込み済み）、出力は GPU から読み戻す
async function runSession(ort, session, name, feed) {
  const g = session.cvpg;
  if (!g?.graph) return session.run({ [name]: feed });
  if (feed.location !== "gpu-buffer") {
    ensureInput(ort, g, feed.dims);
    ort.env.webgpu.device.queue.writeBuffer(g.gpuInput, 0, feed.data);
  }
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

// 入力の大きさと、画像を置く範囲（縮小した画像を (ox, oy) に cw × ch で置き、残りは pad_value で埋める）。CPU と GPU の前処理で共通
function layout(W, H, pre, inputSize) {
  if (pre.resize === "letterbox" || pre.resize === "letterbox_rect") {
    // letterbox_rect: 長辺を S に合わせ、縦横を stride の倍数まで余白で埋めた長方形（正方形の余白の計算を省く）
    const S = pre.dynamic && inputSize ? inputSize : pre.size[0];
    const r = pre.resize === "letterbox_rect" ? S / Math.max(W, H) : Math.min(pre.size[0] / W, pre.size[1] / H);
    const cw = Math.round(W * r), ch = Math.round(H * r), st = pre.stride || 32;
    const [iw, ih] = pre.resize === "letterbox_rect" ? [Math.ceil(cw / st) * st, Math.ceil(ch / st) * st] : pre.size;
    const ox = Math.floor((iw - cw) / 2), oy = Math.floor((ih - ch) / 2);
    return { sx: r, sy: r, ox, oy, iw, ih, cw, ch };
  }
  let iw, ih;
  if (pre.resize === "stretch") {
    [iw, ih] = pre.size;
  } else if (pre.resize === "keep_aspect") {
    const m = pre.multiple || 1, r = pre.short / Math.min(W, H);
    iw = Math.max(m, Math.round((W * r) / m) * m); ih = Math.max(m, Math.round((H * r) / m) * m);
  } else {
    throw new Error(`unknown resize ${pre.resize}`);
  }
  return { sx: iw / W, sy: ih / H, ox: 0, oy: 0, iw, ih, cw: iw, ch: ih };
}

// (画素 × scale − mean) / std = 画素 × a + b の、チャンネルごとの a と b
function affine(pre) {
  const s = pre.scale ?? 1 / 255, mean = pre.mean || [0, 0, 0], std = pre.std || [1, 1, 1];
  return [0, 1, 2].map((k) => [s / std[k], -mean[k] / std[k]]);
}

function inputDims(pre, L) {
  const dims = [1, 3, L.ih, L.iw];
  for (const ax of pre.add_dims || []) dims.splice(ax, 0, 1);
  return dims;
}

// CPU の前処理: OffscreenCanvas に描いて画素を読み出し、JS で正規化する
function preprocess(ort, bitmap, pre, inputSize) {
  const W = bitmap.width, H = bitmap.height, L = layout(W, H, pre, inputSize), { iw, ih } = L;
  const c = new OffscreenCanvas(iw, ih), x = c.getContext("2d", { willReadFrequently: true });
  if (L.cw !== iw || L.ch !== ih) {
    const g = pre.pad_value ?? 114;
    x.fillStyle = `rgb(${g},${g},${g})`;
    x.fillRect(0, 0, iw, ih);
  }
  x.drawImage(bitmap, L.ox, L.oy, L.cw, L.ch);
  const px = x.getImageData(0, 0, iw, ih).data, n = iw * ih;
  const out = new Float32Array(3 * n), ab = affine(pre);
  // チャンネルごとに1回のループで。bgr なら入力の並びを B, G, R に
  for (let k = 0; k < 3; k++) {
    const [a, b] = ab[k], o = k * n, src = pre.bgr ? 2 - k : k;
    for (let i = 0; i < n; i++) out[o + i] = px[i * 4 + src] * a + b;
  }
  return { tensor: new ort.Tensor("float32", out, inputDims(pre, L)), meta: { ...L, W, H } };
}

// GPU の前処理: 画像を WebGPU のテクスチャに送り、縮小（双線形）・余白・正規化・チャンネルの並べ替えを compute shader で行って、
// モデルの入力の GPU バッファに直接書く。画素の読み出し・JS のループ・入力の転送が無くなる（スマホの 640×640 で CPU だと 9〜13ms）。
// 入力のバッファはセッションごとに1つ（graph capture は同じバッファを使い続ける必要がある）
const PRE_WGSL = `
struct P { iw: u32, ih: u32, ox: i32, oy: i32, cw: f32, ch: f32, bgr: u32, pad: f32, a: vec4f, b: vec4f }
@group(0) @binding(0) var src: texture_2d<f32>;
@group(0) @binding(1) var smp: sampler;
@group(0) @binding(2) var<storage, read_write> dst: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) id: vec3u) {
  if (id.x >= p.iw || id.y >= p.ih) { return; }
  let x = f32(i32(id.x) - p.ox) + 0.5;
  let y = f32(i32(id.y) - p.oy) + 0.5;
  var c = vec3f(p.pad);
  if (x > 0.0 && y > 0.0 && x < p.cw && y < p.ch) {
    // canvas の drawImage と同じく 8 bit に丸めてから正規化する
    c = round(textureSampleLevel(src, smp, vec2f(x / p.cw, y / p.ch), 0.0).rgb * 255.0);
  }
  if (p.bgr == 1u) { c = c.bgr; }
  let n = p.iw * p.ih;
  let i = id.y * p.iw + id.x;
  dst[i] = c.r * p.a.x + p.b.x;
  dst[n + i] = c.g * p.a.y + p.b.y;
  dst[2u * n + i] = c.b * p.a.z + p.b.z;
}`;
let gpuPre = null; // { device, pipeline, sampler }

// モデルの入力の GPU バッファ（セッションごとに1つ。形が変わった時だけ作り直す）
function ensureInput(ort, g, dims) {
  const key = dims.join("x");
  if (g.gpuInput && g.inputKey === key) return;
  g.gpuInput?.destroy();
  g.gpuInput = ort.env.webgpu.device.createBuffer({ size: dims.reduce((x, y) => x * y) * 4, usage: GPUBufferUsage.STORAGE | GPUBufferUsage.COPY_SRC | GPUBufferUsage.COPY_DST });
  g.feed = ort.Tensor.fromGpuBuffer(g.gpuInput, { dataType: "float32", dims });
  g.inputKey = key;
}

// mode "gpu": 画像をそのまま copyExternalImageToTexture で送り、縮小も GPU で（M4 では最速。Galaxy Z Fold6 ではモデル実行が
//   18〜28ms 遅くなった。画像を GPU に送る所で待ち合わせが起きているらしい）
// mode "upload": 縮小は CPU の前処理と同じ canvas で行い、8 bit の画素を writeTexture で送って、正規化・余白・並べ替えだけ GPU で。
//   CPU の前処理から JS のループと float の転送（4倍の量）を除いたもの。入力は CPU の前処理と一致する
function gpuPreprocess(ort, session, bitmap, pre, inputSize, mode) {
  const dev = ort.env.webgpu.device, g = session.cvpg;
  if (gpuPre?.device !== dev) {
    const module = dev.createShaderModule({ code: PRE_WGSL });
    gpuPre = {
      device: dev,
      pipeline: dev.createComputePipeline({ layout: "auto", compute: { module, entryPoint: "main" } }),
      sampler: dev.createSampler({ magFilter: "linear", minFilter: "linear" }),
    };
  }
  const W = bitmap.width, H = bitmap.height, L = layout(W, H, pre, inputSize), dims = inputDims(pre, L);
  const [tw, th] = mode === "upload" ? [L.cw, L.ch] : [W, H]; // テクスチャの大きさ（upload は縮小済みの画像）
  ensureInput(ort, g, dims);
  g.uniform ??= dev.createBuffer({ size: 64, usage: GPUBufferUsage.UNIFORM | GPUBufferUsage.COPY_DST });
  if (!g.tex || g.tex.width !== tw || g.tex.height !== th) {
    g.tex?.destroy();
    g.tex = dev.createTexture({ size: [tw, th], format: "rgba8unorm", usage: GPUTextureUsage.TEXTURE_BINDING | GPUTextureUsage.COPY_DST | GPUTextureUsage.RENDER_ATTACHMENT });
  }
  if (g.bindTex !== g.tex || g.bindBuf !== g.gpuInput) {
    g.bind = dev.createBindGroup({ layout: gpuPre.pipeline.getBindGroupLayout(0), entries: [
      { binding: 0, resource: g.tex.createView() }, { binding: 1, resource: gpuPre.sampler },
      { binding: 2, resource: { buffer: g.gpuInput } }, { binding: 3, resource: { buffer: g.uniform } }] });
    g.bindTex = g.tex; g.bindBuf = g.gpuInput;
  }
  if (mode === "upload") {
    g.canvas ??= new OffscreenCanvas(1, 1);
    if (g.canvas.width !== tw || g.canvas.height !== th) { g.canvas.width = tw; g.canvas.height = th; g.ctx = null; }
    g.ctx ??= g.canvas.getContext("2d", { willReadFrequently: true });
    g.ctx.drawImage(bitmap, 0, 0, tw, th);
    dev.queue.writeTexture({ texture: g.tex }, g.ctx.getImageData(0, 0, tw, th).data, { bytesPerRow: tw * 4 }, [tw, th]);
  } else {
    dev.queue.copyExternalImageToTexture({ source: bitmap }, { texture: g.tex }, [W, H]);
  }
  const ab = affine(pre), u = new ArrayBuffer(64), dv = new DataView(u);
  [L.iw, L.ih].forEach((v, i) => dv.setUint32(i * 4, v, true));
  [L.ox, L.oy].forEach((v, i) => dv.setInt32(8 + i * 4, v, true));
  dv.setFloat32(16, L.cw, true); dv.setFloat32(20, L.ch, true);
  dv.setUint32(24, pre.bgr ? 1 : 0, true); dv.setFloat32(28, pre.pad_value ?? 114, true);
  ab.forEach(([a, b], k) => { dv.setFloat32(32 + k * 4, a, true); dv.setFloat32(48 + k * 4, b, true); });
  dev.queue.writeBuffer(g.uniform, 0, u);
  const enc = dev.createCommandEncoder(), pass = enc.beginComputePass();
  pass.setPipeline(gpuPre.pipeline);
  pass.setBindGroup(0, g.bind);
  pass.dispatchWorkgroups(Math.ceil(L.iw / 16), Math.ceil(L.ih / 16));
  pass.end();
  dev.queue.submit([enc.finish()]);
  return { feed: g.feed, meta: { ...L, W, H } };
}

// GPU の前処理を使うか: WebGPU のセッションで、1枚ずつ入れるモデル（e.preMode が "cpu" の時は使わない。画面の ?pre= で選ぶ）。
// GPU 版が対応している指定だけのモデルに限る。pre に新しい指定や縮小方法を足したら（preprocess と adapters.py に足す）、
// ここに足すまでは自動で CPU の前処理になる（GPU 版が黙って違う入力を作らないように）
const GPU_PRE_KEYS = new Set(["size", "resize", "dynamic", "stride", "short", "multiple", "pad_value", "scale", "mean", "std", "bgr", "input", "add_dims"]);
const GPU_PRE_RESIZE = new Set(["letterbox", "letterbox_rect", "stretch", "keep_aspect"]);
const useGpuPre = (ort, session, e) => session.cvpg?.webgpu && !session.cvpg.noGpuPre && e.preMode !== "cpu" && !!ort.env.webgpu?.device
  && GPU_PRE_RESIZE.has(e.pre.resize) && Object.keys(e.pre).every((k) => GPU_PRE_KEYS.has(k));

// 重なった枠を除く（スコアの高い順に、IoU が iouTh を超えるものを捨てる）
function nms(items, iouTh) {
  const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };
  const keep = [];
  for (const it of items.sort((a, b) => b.score - a.score)) if (keep.every((k) => iou(k.box, it.box) <= iouTh)) keep.push(it);
  return keep;
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

// クラス名から決まる色（adapters.py の seg_color と同じ式）。同じクラスの k 番目は明るさを変える
export function segColor(label, k = 0) {
  const h = ([...label].reduce((a, c) => a + c.charCodeAt(0), 0) * 47) % 360;
  return hsl(h, 0.7, [0.55, 0.42, 0.68][k % 3]).map(Math.round);
}

export function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => { const k = (n + h / 30) % 12; return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))); };
  return [f(0), f(8), f(4)];
}

const POST = {
  // セマンティック・セグメンテーション: logits (1, クラス数, h, w) の画素ごとに最大のクラスで塗る（出力の解像度のまま。
  // 表示で画面の大きさに広げる）。クラス名は post.labels。transformers.js のパイプラインは出力を元画像の大きさに
  // 広げてから最大を取るので、境界の細かさが少し違う
  async segmap(out, m, post) {
    const t = post.output ? out[post.output] : Object.values(out)[0], L = t.data, [, C, gh, gw] = t.dims, n = gh * gw;
    const fx = gw / m.iw, fy = gh / m.ih, x0 = Math.round(m.ox * fx), y0 = Math.round(m.oy * fy);
    const cw = Math.max(1, Math.round(m.cw * fx)), ch = Math.max(1, Math.round(m.ch * fy));
    const names = post.labels || [], colors = Array.from({ length: C }, (_, c) => segColor(names[c] ?? String(c)));
    const rgba = new Uint8ClampedArray(cw * ch * 4), area = new Float64Array(C);
    for (let y = 0; y < ch; y++) for (let x = 0; x < cw; x++) {
      const i = (y + y0) * gw + x + x0;
      let best = 0, bv = L[i];
      for (let c = 1; c < C; c++) { const v = L[c * n + i]; if (v > bv) { bv = v; best = c; } }
      const o = (y * cw + x) * 4, col = colors[best];
      rgba[o] = col[0]; rgba[o + 1] = col[1]; rgba[o + 2] = col[2]; rgba[o + 3] = 255;
      area[best]++;
    }
    const cv = new OffscreenCanvas(cw, ch);
    cv.getContext("2d").putImageData(new ImageData(rgba, cw, ch), 0, 0);
    const legend = [];
    area.forEach((a, c) => { if (a) legend.push({ label: names[c] ?? String(c), color: `rgb(${colors[c].join(",")})`, count: 1, area: a / (cw * ch) }); });
    legend.sort((a, b) => b.area - a.area);
    return { kind: "segmap", image: await cv.convertToBlob({ type: "image/png" }), count: legend.length, subtask: "semantic", legend };
  },
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
  // YuNet（OpenCV の顔検出）: 3 つの解像度（stride 8, 16, 32）ごとに cls・obj（スコア）、bbox（格子からのずれ・幅と高さの log）、
  // kps（5 点）。OpenCV の FaceDetectorYN と同じく、スコア = √(cls × obj)、枠 = (格子 + ずれ) × stride と exp(log) × stride、
  // 重なった枠を NMS で除く。keypoints は顔の5点（右目・左目・鼻・口の右端・左端）
  yunet(out, m, post, p) {
    const th = p.threshold ?? 0.6, cand = [];
    for (const s of [8, 16, 32]) {
      const cls = out[`cls_${s}`].data, obj = out[`obj_${s}`].data, bb = out[`bbox_${s}`].data, kp = out[`kps_${s}`].data, gw = Math.ceil(m.iw / s);
      for (let i = 0; i < cls.length; i++) {
        const score = Math.sqrt(Math.min(1, Math.max(0, cls[i])) * Math.min(1, Math.max(0, obj[i])));
        if (score < th) continue;
        const r = Math.floor(i / gw), c = i % gw;
        const cx = (c + bb[i * 4]) * s, cy = (r + bb[i * 4 + 1]) * s, w = Math.exp(bb[i * 4 + 2]) * s, h = Math.exp(bb[i * 4 + 3]) * s;
        const kps = [];
        for (let k = 0; k < 5; k++) kps.push([...toOrig((c + kp[i * 10 + k * 2]) * s, (r + kp[i * 10 + k * 2 + 1]) * s, m), 1]);
        cand.push({ label: "face", score, box: [...toOrig(cx - w / 2, cy - h / 2, m), ...toOrig(cx + w / 2, cy + h / 2, m)], keypoints: kps });
      }
    }
    return { kind: "boxes", items: nms(cand, post.nms ?? 0.3) };
  },
  // YOLO26-pose を onnx.cut で出力の頭の手前で切ったもの: (1, 候補数, 56) = x1 y1 x2 y2（入力のピクセル）, スコア, 17 ×（x, y, 可視度）。
  // 1対1の頭（NMS 不要）なので、スコアの高い順に最大 300 件を取るだけ（ONNX の中の TopK と同じ）
  yolo_pose_raw(out, m, post, p) {
    const t = Object.values(out)[0], D = t.data, [, A, K] = t.dims, th = p.threshold ?? 0.4, idx = [];
    for (let i = 0; i < A; i++) if (D[i * K + 4] >= th) idx.push(i);
    idx.sort((a, b) => D[b * K + 4] - D[a * K + 4]);
    return {
      kind: "boxes",
      items: idx.slice(0, post.max ?? 300).map((i) => {
        const r = D.subarray(i * K, i * K + K), kps = [];
        for (let k = 0; k < 17; k++) kps.push([...toOrig(r[5 + k * 3], r[6 + k * 3], m), r[7 + k * 3]]);
        return { label: "person", score: r[4], box: [...toOrig(r[0], r[1], m), ...toOrig(r[2], r[3], m)], keypoints: kps };
      }),
    };
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
    const t = Object.values(out)[0], D = t.data, [, Q, K] = t.dims, th = p.threshold ?? 0.35, show = new Set(p.show || post.show), items = [];
    for (let i = 0; i < Q; i++) {
      const r = D.subarray(i * K, i * K + K), label = post.classes[r[0] | 0];
      if (r[5] < th || !show.has(label)) continue;
      items.push({ label, score: r[5], box: [...toOrig(r[1] * m.iw, r[2] * m.ih, m), ...toOrig(r[3] * m.iw, r[4] * m.ih, m)] });
    }
    return { kind: "boxes", items };
  },
  async depth(out, m, post, p = {}) {
    const t = out[post.output || "predicted_depth"], [h, w] = t.dims.slice(-2);
    let v = Float32Array.from(t.data);
    if (post.inverse) v = v.map((d) => 1 / Math.max(d, 1e-6)); // 「大きいほど遠い」深度を、表示用に「大きいほど近い」へ
    // 範囲は画像の中だけで取る（letterbox の余白の部分の値は意味が無い）。range を返し、表示側で範囲を固定して塗り直せるようにする
    const fx = w / m.iw, fy = h / m.ih, x0 = Math.round(m.ox * fx), y0 = Math.round(m.oy * fy);
    const x1 = Math.min(w, x0 + Math.round(m.cw * fx)), y1 = Math.min(h, y0 + Math.round(m.ch * fy));
    let lo = Infinity, hi = -Infinity;
    for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) { const a = v[y * w + x]; if (a < lo) lo = a; if (a > hi) hi = a; }
    // 組み合わせのタブ用（params.depth_raw）: 画像にする前の値（大きいほど近い）と、元画像の座標から引くための対応
    const raw = p.depth_raw ? { w, h, data: v, lo, hi, m: { sx: m.sx, sy: m.sy, ox: m.ox, oy: m.oy, iw: m.iw, ih: m.ih } } : null;
    v = v.map((a) => (a - lo) / Math.max(hi - lo, 1e-6));
    const res = { kind: "depth", image: await cropResizePng(v, w, h, m), range: [lo, hi] };
    if (raw) res.depthRaw = raw;
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

// st = { session }（Worker の読み込み済みの状態）。入力の大きさが変わるモデルは、形が変わった時にセッションを作り直して st.session を替える
export async function onnxRun(ort, st, e, bitmap, params) {
  const t0 = performance.now();
  if (st.session.cvpg.dynamicShape) {
    const dims = inputDims(e.pre, layout(bitmap.width, bitmap.height, e.pre, params.input_size));
    if (st.session.cvpg.shapeKey !== dims.join("x")) {
      const old = st.session;
      st.session = await old.cvpg.rebuild(dims);
      old.cvpg.gpuInput?.destroy(); old.cvpg.tex?.destroy();
      await old.release();
    }
  }
  const session = st.session;
  let feed, meta;
  if (useGpuPre(ort, session, e)) {
    try { ({ feed, meta } = gpuPreprocess(ort, session, bitmap, e.pre, params.input_size, e.preMode === "gpu" ? "gpu" : "upload")); }
    catch (err) { console.warn("GPU の前処理が使えない。CPU で行う", err); session.cvpg.noGpuPre = true; }
  }
  session.cvpg.pre = feed ? (e.preMode === "gpu" ? "gpu" : "upload") : "cpu";
  if (!feed) ({ tensor: feed, meta } = preprocess(ort, bitmap, e.pre, params.input_size));
  const t1 = performance.now();
  const out = await runSession(ort, session, e.pre.input, feed);
  const t2 = performance.now();
  const res = await POST[e.post.type](out, meta, e.post, params);
  // 内訳（端末ごとにどこが重いかを見る）: 前処理（縮小・正規化）/ ONNX の実行（GPU との転送を含む）/ 後処理
  res.breakdown = { pre: t1 - t0, run: t2 - t1, post: performance.now() - t2 };
  return res;
}
