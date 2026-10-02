// XFeat（特徴点）の後処理とテンプレートマッチング。サーバーの adapters.py（xfeat_extract・match_mnn・ransac_homography）と
// 同じ手順で、kornia.feature.XFeat（0.8.3）の detectAndCompute・_match_mnn に合わせている（tools/xfeat_check.py で一致を確認）。
// ONNX（kornia/xfeat の xfeat_backbone.onnx）は onnx.cut で [descriptors, heatmap, sigmoid] に切り、
// cut_0 = 記述子 (1, 64, H/8, W/8)（L2 正規化済み）、cut_1 = heatmap (1, 1, H, W)、cut_2 = reliability (1, 1, H/8, W/8) を受け取る

// torch の grid_sample(align_corners=False, padding_mode=zeros) を 1 点で。kornia の InterpolateSparse2d は位置を
// 2·x/(W−1)−1 で正規化してから align_corners=False で読む（学習時の癖。そのまま写す）。map は (C, h, w) の並び
const rint = (v) => { const f = Math.floor(v), d = v - f; return d > 0.5 || (d === 0.5 && f % 2 !== 0) ? f + 1 : f; }; // 偶数への丸め（nearbyint）
function sampleCoord(x, W, w) { return ((2 * x / (W - 1)) * w - 1) / 2; }
function at(map, h, w, c, yy, xx) { return xx < 0 || xx >= w || yy < 0 || yy >= h ? 0 : map[(c * h + yy) * w + xx]; }
function sampleNearest(map, h, w, x, y, W, H) {
  return at(map, h, w, 0, rint(sampleCoord(y, H, h)), rint(sampleCoord(x, W, w)));
}
function sampleBilinear(map, h, w, x, y, W, H) {
  const ix = sampleCoord(x, W, w), iy = sampleCoord(y, H, h), x0 = Math.floor(ix), y0 = Math.floor(iy), tx = ix - x0, ty = iy - y0;
  return at(map, h, w, 0, y0, x0) * (1 - tx) * (1 - ty) + at(map, h, w, 0, y0, x0 + 1) * tx * (1 - ty)
    + at(map, h, w, 0, y0 + 1, x0) * (1 - tx) * ty + at(map, h, w, 0, y0 + 1, x0 + 1) * tx * ty;
}
const A = -0.75; // bicubic（torch と同じ係数）
function cubicW(t) {
  return [((A * (t + 1) - 5 * A) * (t + 1) + 8 * A) * (t + 1) - 4 * A, ((A + 2) * t - (A + 3)) * t * t + 1,
    ((A + 2) * (1 - t) - (A + 3)) * (1 - t) * (1 - t) + 1, ((A * (2 - t) - 5 * A) * (2 - t) + 8 * A) * (2 - t) - 4 * A];
}
// 記述子（C チャンネル）を bicubic で読んで L2 正規化し、out[o..o+C) に書く
function sampleBicubicDesc(map, C, h, w, x, y, W, H, out, o) {
  const ix = sampleCoord(x, W, w), iy = sampleCoord(y, H, h), x0 = Math.floor(ix), y0 = Math.floor(iy);
  const wx = cubicW(ix - x0), wy = cubicW(iy - y0), hw = h * w;
  const taps = [];
  for (let j = 0; j < 4; j++) for (let i = 0; i < 4; i++) {
    const yy = y0 - 1 + j, xx = x0 - 1 + i;
    if (xx >= 0 && xx < w && yy >= 0 && yy < h) taps.push(yy * w + xx, wx[i] * wy[j]);
  }
  let n2 = 0;
  for (let c = 0; c < C; c++) {
    let v = 0;
    const base = c * hw;
    for (let k = 0; k < taps.length; k += 2) v += map[base + taps[k]] * taps[k + 1];
    out[o + c] = v; n2 += v * v;
  }
  const inv = 1 / Math.max(Math.sqrt(n2), 1e-12);
  for (let c = 0; c < C; c++) out[o + c] *= inv;
}

// ONNX の出力 → 点（元画像の座標）・スコア・記述子（n × 64 の Float32Array）。kornia の detectAndCompute と同じ:
// heatmap の 5×5 の極大（周り 24 点すべてより真に大きい。端の 2 画素は使わない）で閾値より大きい点 →
// スコア = heatmap（最近傍）× reliability（bilinear）→ 上位 top_k → 記述子を bicubic で読んで L2 正規化
export function xfeatExtract(out, m, topK, th = 0.05) {
  const D = out.cut_0, Hm = out.cut_1, R = out.cut_2;
  const [, C, h8, w8] = D.dims, H = Hm.dims[2], W = Hm.dims[3], heat = Hm.data, rel = R.data, desc = D.data;
  const rh = R.dims[2], rw = R.dims[3];
  const cand = []; // [x, y, score]（行の順。並べ替えは安定なので、同じスコアは kornia と同じ順）
  for (let y = 2; y < H - 2; y++) {
    const row = y * W;
    for (let x = 2; x < W - 2; x++) {
      const v = heat[row + x];
      if (!(v > th)) continue;
      let peak = true;
      for (let dy = -2; dy <= 2 && peak; dy++) {
        const r2 = row + dy * W + x;
        for (let dx = -2; dx <= 2; dx++) if ((dy || dx) && !(v > heat[r2 + dx])) { peak = false; break; }
      }
      if (peak) cand.push([x, y, sampleNearest(heat, H, W, x, y, W, H) * sampleBilinear(rel, rh, rw, x, y, W, H)]);
    }
  }
  cand.sort((a, b) => b[2] - a[2]);
  const keep = cand.slice(0, topK).filter((c) => c[2] > 0), n = keep.length;
  const pts = new Float64Array(2 * n), scores = new Float32Array(n), feats = new Float32Array(n * C);
  keep.forEach(([x, y, s], i) => {
    pts[2 * i] = x / m.sx; pts[2 * i + 1] = y / m.sy; scores[i] = s;
    sampleBicubicDesc(desc, C, h8, w8, x, y, W, H, feats, i * C);
  });
  return { n, pts, scores, feats, C };
}

// 相互最近傍かつコサイン類似度 > minCos（kornia の _match_mnn）。a がテンプレート、b がフレーム。同じ値なら先の番号（argmax と同じ）
export function matchMnn(a, b, minCos = 0.82) {
  const na = a.n, nb = b.n, C = a.C, fa = a.feats, fb = b.feats;
  const rowBest = new Int32Array(na).fill(-1), rowMax = new Float64Array(na).fill(-Infinity);
  const colBest = new Int32Array(nb).fill(-1), colMax = new Float64Array(nb).fill(-Infinity);
  for (let i = 0; i < na; i++) {
    const oi = i * C;
    for (let j = 0; j < nb; j++) {
      const oj = j * C;
      let s = 0;
      for (let c = 0; c < C; c++) s += fa[oi + c] * fb[oj + c];
      if (s > rowMax[i]) { rowMax[i] = s; rowBest[i] = j; }
      if (s > colMax[j]) { colMax[j] = s; colBest[j] = i; }
    }
  }
  const i0 = [], i1 = [];
  for (let i = 0; i < na; i++) if (rowBest[i] >= 0 && colBest[rowBest[i]] === i && rowMax[i] > minCos) { i0.push(i); i1.push(rowBest[i]); }
  return { i0, i1 };
}

// JS と Python（adapters.py の Mulberry32）で同じ並びの乱数
function mulberry32(a) {
  return () => {
    a = (a + 0x6D2B79F5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

// 点を正規化する変換（重心を原点、原点からの平均距離を √2 に）
function normT(P, idx) {
  let cx = 0, cy = 0;
  for (const i of idx) { cx += P[2 * i]; cy += P[2 * i + 1]; }
  cx /= idx.length; cy /= idx.length;
  let d = 0;
  for (const i of idx) d += Math.hypot(P[2 * i] - cx, P[2 * i + 1] - cy);
  d = d / idx.length || 1;
  const s = Math.SQRT2 / d;
  return [s, 0, -s * cx, 0, s, -s * cy, 0, 0, 1];
}
const mul3 = (a, b) => { const r = new Array(9); for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j]; return r; };
function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m, A_ = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A_ + b * B + c * C;
  if (Math.abs(det) < 1e-300) return null;
  return [A_ / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det, C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
// 対称行列（n×n）の固有値分解（ヤコビ法）。一番小さい固有値のベクトルを返す（= A の最小特異値のベクトル）
function smallestEigvec(S, n) {
  const a = S.slice(), v = new Float64Array(n * n);
  for (let i = 0; i < n; i++) v[i * n + i] = 1;
  for (let sweep = 0; sweep < 60; sweep++) {
    let off = 0;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) off += a[p * n + q] ** 2;
    if (off < 1e-30) break;
    for (let p = 0; p < n; p++) for (let q = p + 1; q < n; q++) {
      const apq = a[p * n + q];
      if (Math.abs(apq) < 1e-300) continue;
      const th = (a[q * n + q] - a[p * n + p]) / (2 * apq), t = Math.sign(th || 1) / (Math.abs(th) + Math.sqrt(th * th + 1));
      const c = 1 / Math.sqrt(t * t + 1), s = t * c;
      for (let k = 0; k < n; k++) { const akp = a[k * n + p], akq = a[k * n + q]; a[k * n + p] = c * akp - s * akq; a[k * n + q] = s * akp + c * akq; }
      for (let k = 0; k < n; k++) { const apk = a[p * n + k], aqk = a[q * n + k]; a[p * n + k] = c * apk - s * aqk; a[q * n + k] = s * apk + c * aqk; }
      for (let k = 0; k < n; k++) { const vkp = v[k * n + p], vkq = v[k * n + q]; v[k * n + p] = c * vkp - s * vkq; v[k * n + q] = s * vkp + c * vkq; }
    }
  }
  let m = 0;
  for (let i = 1; i < n; i++) if (a[i * n + i] < a[m * n + m]) m = i;
  return Array.from({ length: n }, (_, k) => v[k * n + m]);
}
// a → b のホモグラフィ（正規化した DLT。idx の点で）。(AᵀA) の一番小さい固有値のベクトル
function homographyDlt(a, b, idx) {
  const Ta = normT(a, idx), Tb = normT(b, idx), S = new Float64Array(81);
  for (const i of idx) {
    const x = Ta[0] * a[2 * i] + Ta[2], y = Ta[4] * a[2 * i + 1] + Ta[5], u = Tb[0] * b[2 * i] + Tb[2], v = Tb[4] * b[2 * i + 1] + Tb[5];
    for (const r of [[-x, -y, -1, 0, 0, 0, u * x, u * y, u], [0, 0, 0, -x, -y, -1, v * x, v * y, v]]) {
      for (let p = 0; p < 9; p++) { if (!r[p]) continue; for (let q = 0; q < 9; q++) S[p * 9 + q] += r[p] * r[q]; }
    }
  }
  const h = smallestEigvec(S, 9), iTb = inv3(Tb);
  if (!iTb) return null;
  const Hm = mul3(mul3(iTb, h), Ta);
  return Math.abs(Hm[8]) > 1e-12 ? Hm.map((x) => x / Hm[8]) : null;
}
export function project(Hm, x, y) {
  const w = Hm[6] * x + Hm[7] * y + Hm[8];
  return [(Hm[0] * x + Hm[1] * y + Hm[2]) / w, (Hm[3] * x + Hm[4] * y + Hm[5]) / w];
}
function inliersOf(Hm, a, b, n, th) {
  const inl = new Uint8Array(n);
  let k = 0;
  for (let i = 0; i < n; i++) {
    const [px, py] = project(Hm, a[2 * i], a[2 * i + 1]);
    if (Math.hypot(px - b[2 * i], py - b[2 * i + 1]) < th) { inl[i] = 1; k++; }
  }
  return { inl, k };
}
function collinear(P, idx, eps = 1e-3) {
  for (let i = 0; i < 4; i++) for (let j = i + 1; j < 4; j++) for (let k = j + 1; k < 4; k++) {
    const [p, q, r] = [idx[i], idx[j], idx[k]];
    if (Math.abs((P[2 * q] - P[2 * p]) * (P[2 * r + 1] - P[2 * p + 1]) - (P[2 * q + 1] - P[2 * p + 1]) * (P[2 * r] - P[2 * p])) < eps) return true;
  }
  return false;
}
// インライア全部で解き直し、インライアが減らない間（最大 rounds 回）くり返す（LO-RANSAC の局所最適化）
function refine(a, b, n, Hm, cur, th, rounds = 5) {
  for (let r = 0; r < rounds && cur.k >= 5; r++) {
    const idx = [];
    for (let i = 0; i < n; i++) if (cur.inl[i]) idx.push(i);
    const Hr = homographyDlt(a, b, idx);
    if (!Hr) break;
    const nx = inliersOf(Hr, a, b, n, th);
    if (nx.k < cur.k) break;
    const same = nx.inl.every((v, i) => v === cur.inl[i]);
    Hm = Hr; cur = nx;
    if (same) break;
  }
  return { Hm, cur };
}
// 4 点の DLT で仮説を立て、再投影の誤差 th 以下の数が一番多いものを選ぶ。良い仮説ごとに解き直す（LO-RANSAC）
export function ransacHomography(a, b, n, th = 3, iters = 1000, conf = 0.995, seed = 1) {
  let best = null, bestIn = { inl: new Uint8Array(n), k: 0 };
  if (n < 4) return { H: null, ...bestIn };
  const rnd = mulberry32(seed);
  let k = iters, it = 0;
  while (it < k) {
    it++;
    const idx = [];
    while (idx.length < 4) { const i = Math.floor(rnd() * n); if (!idx.includes(i)) idx.push(i); }
    if (collinear(a, idx) || collinear(b, idx)) continue;
    let Hm = homographyDlt(a, b, idx);
    if (!Hm) continue;
    let cur = inliersOf(Hm, a, b, n, th);
    if (cur.k > bestIn.k) {
      ({ Hm, cur } = refine(a, b, n, Hm, cur, th));
      best = Hm; bestIn = cur;
      const w = cur.k / n;
      k = w < 1 ? Math.min(iters, Math.ceil(Math.log(1 - conf) / Math.log(Math.max(1e-12, 1 - w ** 4)))) : it;
    }
  }
  return { H: best, ...bestIn };
}

// 凸で、つぶれていない四角形か
export function quadOk(q) {
  const cr = [0, 1, 2, 3].map((i) => {
    const [a, b, c] = [q[i], q[(i + 1) % 4], q[(i + 2) % 4]];
    return (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0]);
  });
  let area = 0;
  for (let i = 0; i < 4; i++) area += q[i][0] * q[(i + 1) % 4][1] - q[(i + 1) % 4][0] * q[i][1];
  return (cr.every((x) => x > 0) || cr.every((x) => x < 0)) && Math.abs(area) / 2 > 16;
}

// フレームの特徴とテンプレートの特徴から、結果（kind: "matches"）を作る
export function matchTemplate(fr, tpl, post, params, w, h) {
  const t0 = performance.now();
  const res = { kind: "matches", w, h, kpts: fr.n, found: false, quad: null, H: null, inliers: 0, matches: 0, pairs: [],
    points: Array.from({ length: fr.n }, (_, i) => [fr.pts[2 * i], fr.pts[2 * i + 1], fr.scores[i]]) };
  if (!tpl) { res.note = "テンプレートが無い（枠をドラッグして切り出すか、ファイルで選ぶ）"; return res; }
  const { i0, i1 } = matchMnn(tpl, fr, post.min_cossim ?? 0.82);
  const t1 = performance.now();
  const n = i0.length, a = new Float64Array(2 * n), b = new Float64Array(2 * n);
  i0.forEach((ti, k) => { a[2 * k] = tpl.pts[2 * ti]; a[2 * k + 1] = tpl.pts[2 * ti + 1]; b[2 * k] = fr.pts[2 * i1[k]]; b[2 * k + 1] = fr.pts[2 * i1[k] + 1]; });
  const r = ransacHomography(a, b, n, post.ransac_px ?? 3, 1000, 0.995, params.seed ?? 1);
  const t2 = performance.now();
  Object.assign(res, { kpts_t: tpl.n, template: { w: tpl.w, h: tpl.h }, matches: n, inliers: r.k,
    pairs: i0.map((_, k) => [a[2 * k], a[2 * k + 1], b[2 * k], b[2 * k + 1], r.inl[k]]),
    post_detail: { match: t1 - t0, ransac: t2 - t1 } });
  if (r.H) {
    const q = [[0, 0], [tpl.w, 0], [tpl.w, tpl.h], [0, tpl.h]].map(([x, y]) => project(r.H, x, y));
    res.H = r.H;
    if (r.k >= (post.min_inliers ?? 15) && quadOk(q)) Object.assign(res, { found: true, quad: q });
  }
  return res;
}
