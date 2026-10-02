// XFeat の後処理を WebGPU で（xfeat.js と同じ手順。onnxruntime の出力を読み戻さずに GPU の上で続ける）。
// スマホ（Galaxy Z Fold6）では、JS の後処理が 40ms（heatmap 全体をなめる点の取り出し 11ms・対応の内積 28ms）で、モデル（20ms）より重かった。
//
//   1. 極大と スコア（nms）: heatmap の画素ごとに、周り 24 点すべてより真に大きく閾値より大きければ、スコア
//      = heatmap（最近傍）× reliability（bilinear）を計算して、候補（位置・スコア）を詰めて書く → 候補だけ読み戻す（数十 KB）
//   2. JS で上位 top_k を選ぶ（スコアの大きい順。同じスコアは行の順。xfeat.js と同じ）
//   3. 記述子（desc）: 選んだ点で記述子を bicubic で読んで L2 正規化（GPU の上に置いたまま）
//   4. 対応（mnn）: テンプレートの記述子（GPU に 1 回だけ送る）との内積の最大と番号を、行・列それぞれで → 読み戻す（数 KB）
//
// 点の位置の読み方（2·x/(W−1)−1 で正規化して align_corners=False）は xfeat.js と同じ式を f32 で計算する。
// WGSL の round は偶数への丸め（torch の nearbyint と同じ）

const COMMON = /* wgsl */ `
fn coord(x: f32, W: u32, w: u32) -> f32 { return ((2.0 * x / f32(W - 1u)) * f32(w) - 1.0) / 2.0; }
`;

const NMS_WGSL = COMMON + /* wgsl */ `
struct P { W: u32, H: u32, w8: u32, h8: u32, th: f32, cap: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> heat: array<f32>;
@group(0) @binding(1) var<storage, read> rel: array<f32>;
@group(0) @binding(2) var<storage, read_write> cnt: array<atomic<u32>>;
@group(0) @binding(3) var<storage, read_write> cand: array<vec2<u32>>;
@group(0) @binding(4) var<uniform> p: P;
fn relAt(yy: i32, xx: i32) -> f32 {
  if (xx < 0 || yy < 0 || xx >= i32(p.w8) || yy >= i32(p.h8)) { return 0.0; }
  return rel[u32(yy) * p.w8 + u32(xx)];
}
@compute @workgroup_size(16, 16)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let x = g.x; let y = g.y;
  if (x < 2u || y < 2u || x + 2u >= p.W || y + 2u >= p.H) { return; }
  let v = heat[y * p.W + x];
  if (!(v > p.th)) { return; }
  for (var dy = -2; dy <= 2; dy++) {
    for (var dx = -2; dx <= 2; dx++) {
      if (dx == 0 && dy == 0) { continue; }
      if (!(v > heat[u32(i32(y) + dy) * p.W + u32(i32(x) + dx)])) { return; }
    }
  }
  let fx = f32(x); let fy = f32(y);
  // heatmap を最近傍で（位置の癖のため、右・下の半分では隣の画素を読む）
  let nx = i32(round(coord(fx, p.W, p.W))); let ny = i32(round(coord(fy, p.H, p.H)));
  var hn = 0.0;
  if (nx >= 0 && ny >= 0 && nx < i32(p.W) && ny < i32(p.H)) { hn = heat[u32(ny) * p.W + u32(nx)]; }
  // reliability（H/8）を bilinear で
  let ix = coord(fx, p.W, p.w8); let iy = coord(fy, p.H, p.h8);
  let x0 = i32(floor(ix)); let y0 = i32(floor(iy)); let tx = ix - f32(x0); let ty = iy - f32(y0);
  let rb = relAt(y0, x0) * (1.0 - tx) * (1.0 - ty) + relAt(y0, x0 + 1) * tx * (1.0 - ty)
         + relAt(y0 + 1, x0) * (1.0 - tx) * ty + relAt(y0 + 1, x0 + 1) * tx * ty;
  let i = atomicAdd(&cnt[0], 1u);
  if (i < p.cap) { cand[i] = vec2<u32>((y << 16u) | x, bitcast<u32>(hn * rb)); }
}`;

const DESC_WGSL = COMMON + /* wgsl */ `
struct P { W: u32, H: u32, w8: u32, h8: u32, n: u32, p0: u32, p1: u32, p2: u32 };
@group(0) @binding(0) var<storage, read> desc: array<f32>;
@group(0) @binding(1) var<storage, read> pts: array<vec2<f32>>;
@group(0) @binding(2) var<storage, read_write> feats: array<f32>;
@group(0) @binding(3) var<uniform> p: P;
const A = -0.75;
fn cw(t: f32) -> vec4<f32> {
  return vec4<f32>(((A * (t + 1.0) - 5.0 * A) * (t + 1.0) + 8.0 * A) * (t + 1.0) - 4.0 * A, ((A + 2.0) * t - (A + 3.0)) * t * t + 1.0,
    ((A + 2.0) * (1.0 - t) - (A + 3.0)) * (1.0 - t) * (1.0 - t) + 1.0, ((A * (2.0 - t) - 5.0 * A) * (2.0 - t) + 8.0 * A) * (2.0 - t) - 4.0 * A);
}
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let k = g.x;
  if (k >= p.n) { return; }
  let q = pts[k];
  let ix = coord(q.x, p.W, p.w8); let iy = coord(q.y, p.H, p.h8);
  let x0 = i32(floor(ix)); let y0 = i32(floor(iy));
  let wx = cw(ix - f32(x0)); let wy = cw(iy - f32(y0));
  let hw = p.w8 * p.h8;
  var n2 = 0.0;
  for (var c = 0u; c < 64u; c++) {
    var v = 0.0;
    for (var j = 0; j < 4; j++) {
      let yy = y0 - 1 + j;
      if (yy < 0 || yy >= i32(p.h8)) { continue; }
      for (var i = 0; i < 4; i++) {
        let xx = x0 - 1 + i;
        if (xx < 0 || xx >= i32(p.w8)) { continue; }
        v += desc[c * hw + u32(yy) * p.w8 + u32(xx)] * wx[i] * wy[j];
      }
    }
    feats[k * 64u + c] = v; n2 += v * v;
  }
  let inv = 1.0 / max(sqrt(n2), 1e-12);
  for (var c = 0u; c < 64u; c++) { feats[k * 64u + c] *= inv; }
}`;

// 行（a の各点）ごとに b の全点との内積の最大と、その番号（同じ値なら先の番号）
const MNN_WGSL = /* wgsl */ `
struct P { na: u32, nb: u32, p0: u32, p1: u32 };
@group(0) @binding(0) var<storage, read> a: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<vec4<f32>>;
@group(0) @binding(2) var<storage, read_write> best: array<i32>;
@group(0) @binding(3) var<storage, read_write> bestv: array<f32>;
@group(0) @binding(4) var<uniform> p: P;
@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) g: vec3<u32>) {
  let i = g.x;
  if (i >= p.na) { return; }
  var m = -1e30; var bi = -1;
  for (var j = 0u; j < p.nb; j++) {
    var s = 0.0;
    for (var c = 0u; c < 16u; c++) { s += dot(a[i * 16u + c], b[j * 16u + c]); }
    if (s > m) { m = s; bi = i32(j); }
  }
  best[i] = bi; bestv[i] = m;
}`;

let S; // GPUBufferUsage（WebGPU の無い環境でも読み込めるよう、使う時に）
export class XFeatGpu {
  constructor(device) {
    S = GPUBufferUsage;
    this.dev = device;
    const mk = (code) => device.createComputePipeline({ layout: "auto", compute: { module: device.createShaderModule({ code }), entryPoint: "main" } });
    this.nms = mk(NMS_WGSL); this.desc = mk(DESC_WGSL); this.mnn = mk(MNN_WGSL);
    this.bufs = {};
  }
  // 大きさが足りる時は使い回す GPU のバッファ
  buf(name, size, usage) {
    const b = this.bufs[name];
    if (b && b.size >= size) return b;
    b?.destroy();
    return (this.bufs[name] = this.dev.createBuffer({ size: Math.max(16, Math.ceil(size / 16) * 16), usage }));
  }
  uniform(name, u32s) {
    const b = this.buf(name, u32s.byteLength, S.UNIFORM | S.COPY_DST);
    this.dev.queue.writeBuffer(b, 0, u32s);
    return b;
  }
  run(pipe, entries, groups) {
    const enc = this.dev.createCommandEncoder(), pass = enc.beginComputePass();
    pass.setPipeline(pipe);
    pass.setBindGroup(0, this.dev.createBindGroup({ layout: pipe.getBindGroupLayout(0), entries: entries.map((b, i) => ({ binding: i, resource: { buffer: b } })) }));
    pass.dispatchWorkgroups(...groups);
    pass.end();
    return enc;
  }
  async read(src, bytes) {
    const st = this.dev.createBuffer({ size: Math.ceil(bytes / 4) * 4, usage: S.MAP_READ | S.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(src, 0, st, 0, st.size);
    this.dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const out = st.getMappedRange().slice(0);
    st.unmap(); st.destroy();
    return out;
  }

  // ONNX の出力（GPU の上）→ 点（元画像の座標）・スコア・記述子（GPU のバッファ）
  // featsName: 記述子のバッファの名前（手ぶれ補正は前のフレームの分を残すので、2 つを交互に使う）
  async extract(out, m, topK, th, featsName = "feats") {
    const D = out.cut_0, Hm = out.cut_1, R = out.cut_2, H = Hm.dims[2], W = Hm.dims[3], h8 = D.dims[2], w8 = D.dims[3];
    const cap = Math.ceil((W * H) / 9);
    const cnt = this.buf("cnt", 16, S.STORAGE | S.COPY_SRC | S.COPY_DST), cand = this.buf("cand", cap * 8, S.STORAGE | S.COPY_SRC);
    this.dev.queue.writeBuffer(cnt, 0, new Uint32Array(4));
    const u = new Uint32Array(8); u.set([W, H, w8, h8]); new Float32Array(u.buffer, 16, 1)[0] = th; u[5] = cap;
    this.dev.queue.submit([this.run(this.nms, [Hm.gpuBuffer, R.gpuBuffer, cnt, cand, this.uniform("unms", u)], [Math.ceil(W / 16), Math.ceil(H / 16), 1]).finish()]);
    // 数と先頭の FIRST 個をまとめて 1 回で読み戻す（往復を減らす。候補は 640 で 4,000〜9,000 個）。足りなければ残りをもう 1 回
    const FIRST = 16384, first = Math.min(cap, FIRST), st = this.dev.createBuffer({ size: 16 + first * 8, usage: S.MAP_READ | S.COPY_DST });
    const enc = this.dev.createCommandEncoder();
    enc.copyBufferToBuffer(cnt, 0, st, 0, 16);
    enc.copyBufferToBuffer(cand, 0, st, 16, first * 8);
    this.dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const head = st.getMappedRange().slice(0);
    st.unmap(); st.destroy();
    const n0 = Math.min(cap, new Uint32Array(head, 0, 1)[0]);
    let raw = new Uint32Array(head, 16, Math.min(n0, first) * 2);
    if (n0 > first) {
      const rest = new Uint32Array(await this.read(cand, n0 * 8));
      raw = rest;
    }
    const sc = new Float32Array(raw.buffer, raw.byteOffset, raw.length);
    const c = [];
    for (let i = 0; i < n0; i++) { const xy = raw[2 * i]; c.push([xy & 0xffff, xy >>> 16, sc[2 * i + 1]]); }
    c.sort((a, b) => b[2] - a[2] || a[1] - b[1] || a[0] - b[0]); // 同じスコアは行の順（xfeat.js の安定な並べ替えと同じ）
    const keep = c.slice(0, topK).filter((x) => x[2] > 0), n = keep.length;
    const pts = new Float64Array(2 * n), scores = new Float32Array(n), pin = new Float32Array(2 * Math.max(1, n));
    keep.forEach(([x, y, s], i) => { pts[2 * i] = x / m.sx; pts[2 * i + 1] = y / m.sy; scores[i] = s; pin[2 * i] = x; pin[2 * i + 1] = y; });
    const pb = this.buf("pts", pin.byteLength, S.STORAGE | S.COPY_DST);
    this.dev.queue.writeBuffer(pb, 0, pin);
    const feats = this.buf(featsName, Math.max(1, n) * 256, S.STORAGE | S.COPY_SRC);
    if (n) {
      const ud = new Uint32Array(8); ud.set([W, H, w8, h8, n]);
      this.dev.queue.submit([this.run(this.desc, [D.gpuBuffer, pb, feats, this.uniform("udesc", ud)], [Math.ceil(n / 64), 1, 1]).finish()]);
    }
    return { n, pts, scores, featsGpu: feats, C: 64, cand: n0 };
  }

  // テンプレートの記述子（JS の Float32Array）を GPU に送る。テンプレートを変えた時だけ
  uploadTemplate(tpl) {
    const b = this.dev.createBuffer({ size: Math.max(16, tpl.feats.byteLength), usage: S.STORAGE | S.COPY_DST });
    if (tpl.feats.byteLength) this.dev.queue.writeBuffer(b, 0, tpl.feats);
    return b;
  }

  // 相互最近傍かつ cos > minCos（xfeat.js の matchMnn と同じ）。テンプレートの行・フレームの行それぞれで、相手の全点との
  // 内積の最大と番号を GPU で求め、4 つ（行の番号・値、列の番号・値）を 1 回で読み戻す
  async match(tpl, tplBuf, fr, minCos) {
    const na = tpl.n, nb = fr.n;
    if (!na || !nb) return { i0: [], i1: [] };
    const use = S.STORAGE | S.COPY_SRC;
    const rb = this.buf("rb", na * 4, use), rv = this.buf("rv", na * 4, use), cb = this.buf("cb", nb * 4, use), cv = this.buf("cv", nb * 4, use);
    const enc = this.dev.createCommandEncoder();
    for (const [A, B, n, m, ob, ov, name] of [[tplBuf, fr.featsGpu, na, nb, rb, rv, "umnn0"], [fr.featsGpu, tplBuf, nb, na, cb, cv, "umnn1"]]) {
      const ub = this.uniform(name, new Uint32Array([n, m, 0, 0]));
      const pass = enc.beginComputePass();
      pass.setPipeline(this.mnn);
      pass.setBindGroup(0, this.dev.createBindGroup({ layout: this.mnn.getBindGroupLayout(0),
        entries: [A, B, ob, ov, ub].map((b, i) => ({ binding: i, resource: { buffer: b } })) }));
      pass.dispatchWorkgroups(Math.ceil(n / 64));
      pass.end();
    }
    const bytes = (2 * na + 2 * nb) * 4, st = this.dev.createBuffer({ size: bytes, usage: S.MAP_READ | S.COPY_DST });
    enc.copyBufferToBuffer(rb, 0, st, 0, na * 4);
    enc.copyBufferToBuffer(rv, 0, st, na * 4, na * 4);
    enc.copyBufferToBuffer(cb, 0, st, 2 * na * 4, nb * 4);
    enc.copyBufferToBuffer(cv, 0, st, (2 * na + nb) * 4, nb * 4);
    this.dev.queue.submit([enc.finish()]);
    await st.mapAsync(GPUMapMode.READ);
    const buf = st.getMappedRange().slice(0);
    st.unmap(); st.destroy();
    const rowBest = new Int32Array(buf, 0, na), rowMax = new Float32Array(buf, na * 4, na), colBest = new Int32Array(buf, 2 * na * 4, nb);
    const i0 = [], i1 = [];
    for (let i = 0; i < na; i++) if (rowBest[i] >= 0 && colBest[rowBest[i]] === i && rowMax[i] > minCos) { i0.push(i); i1.push(rowBest[i]); }
    return { i0, i1 };
  }
}
