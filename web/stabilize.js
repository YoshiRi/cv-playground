// 手ぶれ補正（画面側）: Worker が返すフレーム間の動き M_t（今のフレームの点 → 前のフレームの点、3×3）から、カメラの動きの軌跡
// C_t = C_{t−1}·M_t（今のフレーム → 最初のフレーム）を積み、なめらかにした軌跡 S_t との差 W_t = S_t⁻¹·C_t でフレームを描き直す。
// 周りを切り抜く（crop の割合だけ拡大）ので、W_t で動かしても画面の中は埋まっている。埋まらなくなる時は S_t を C_t の側へ寄せる
//
//   なめらか: S_t = S_{t−1} + α (C_t − S_{t−1})（要素ごと。小さい回転なら相似変換・ホモグラフィとも近似として足りる）
//   三脚:     S_t = 最初のフレーム（I）のまま。切り抜きの外に出る時だけ寄せる
//
// 指標（どちらも元のフレームの px と度。補正後は D_t = W_{t−1}·M_t·W_t⁻¹ で、切り抜きの拡大は入れない。入れると補正しきれない動きが
// 拡大の分だけ大きく見え、補正前と比べられない）:
//   動き（全体）: フレーム間の動き（画面の中心のずれと回転）の二乗平均。カメラを振った・歩いた動きも入る
//   揺れ（細かい成分）: フレーム間の動きから、前後 4 フレーム（計 9）の平均の動きを引いた残りの二乗平均。意図した動きを除いた、ぶれの量

const I = [1, 0, 0, 0, 1, 0, 0, 0, 1];
export function mul3(a, b) {
  const r = new Array(9);
  for (let i = 0; i < 3; i++) for (let j = 0; j < 3; j++) r[i * 3 + j] = a[i * 3] * b[j] + a[i * 3 + 1] * b[3 + j] + a[i * 3 + 2] * b[6 + j];
  return r;
}
export function inv3(m) {
  const [a, b, c, d, e, f, g, h, i] = m, A = e * i - f * h, B = -(d * i - f * g), C = d * h - e * g, det = a * A + b * B + c * C;
  if (Math.abs(det) < 1e-12) return I.slice();
  return [A / det, -(b * i - c * h) / det, (b * f - c * e) / det, B / det, (a * i - c * g) / det, -(a * f - c * d) / det, C / det, -(a * h - b * g) / det, (a * e - b * d) / det];
}
const norm = (m) => (Math.abs(m[8]) > 1e-12 ? m.map((v) => v / m[8]) : m);
export const apply3 = (m, x, y) => { const w = m[6] * x + m[7] * y + m[8]; return [(m[0] * x + m[1] * y + m[2]) / w, (m[3] * x + m[4] * y + m[5]) / w]; };
const lerp = (a, b, t) => a.map((v, i) => v + (b[i] - v) * t);
// 拡大（中心まわりに z 倍）
const zoom = (z, w, h) => [z, 0, (1 - z) * w / 2, 0, z, (1 - z) * h / 2, 0, 0, 1];
// 動き（3×3）の、画面の中心のずれ（px、ベクトル）と回転（度）
function motionAt(m, w, h) {
  const [x, y] = apply3(m, w / 2, h / 2);
  return { x: x - w / 2, y: y - h / 2, deg: (Math.atan2(m[3] - m[6] * m[5], m[0] - m[6] * m[2]) * 180) / Math.PI };
}
// 動きの列 → 全体の二乗平均と、細かい成分（前後 K フレームの平均を引いた残り）の二乗平均
const K = 4;
function metrics(v) {
  const n = v.length;
  if (!n) return { move_px: 0, move_deg: 0, jit_px: 0, jit_deg: 0 };
  let mp = 0, md = 0, jp = 0, jd = 0;
  for (let i = 0; i < n; i++) {
    let ax = 0, ay = 0, ad = 0, c = 0;
    for (let j = Math.max(0, i - K); j <= Math.min(n - 1, i + K); j++) { ax += v[j].x; ay += v[j].y; ad += v[j].deg; c++; }
    mp += v[i].x ** 2 + v[i].y ** 2; md += v[i].deg ** 2;
    jp += (v[i].x - ax / c) ** 2 + (v[i].y - ay / c) ** 2; jd += (v[i].deg - ad / c) ** 2;
  }
  return { move_px: Math.sqrt(mp / n), move_deg: Math.sqrt(md / n), jit_px: Math.sqrt(jp / n), jit_deg: Math.sqrt(jd / n) };
}
// canvas の 2D はアフィン変換しか描けないので、ホモグラフィは四隅を最小二乗で合わせたアフィンで近似する（小さい揺れなら差はわずか）
export function affineOf(m, w, h) {
  if (Math.abs(m[6]) < 1e-12 && Math.abs(m[7]) < 1e-12) return [m[0] / m[8], m[3] / m[8], m[1] / m[8], m[4] / m[8], m[2] / m[8], m[5] / m[8]];
  const src = [[0, 0], [w, 0], [w, h], [0, h], [w / 2, h / 2]], dst = src.map(([x, y]) => apply3(m, x, y));
  // [x y 1] → u, v をそれぞれ最小二乗（3×3 の正規方程式）
  const solve = (k) => {
    const A = [[0, 0, 0], [0, 0, 0], [0, 0, 0]], bb = [0, 0, 0];
    src.forEach(([x, y], i) => { const r = [x, y, 1]; for (let p = 0; p < 3; p++) { bb[p] += r[p] * dst[i][k]; for (let q = 0; q < 3; q++) A[p][q] += r[p] * r[q]; } });
    const inv = inv3(A.flat());
    return [0, 1, 2].map((p) => inv[p * 3] * bb[0] + inv[p * 3 + 1] * bb[1] + inv[p * 3 + 2] * bb[2]);
  };
  const [a, c, e] = solve(0), [b, d, f] = solve(1);
  return [a, b, c, d, e, f];
}

export const STAB_ALPHA = { weak: 0.3, mid: 0.1, strong: 0.03 }; // なめらかにする強さ（指数移動平均の係数。小さいほど強い）

export class Stabilizer {
  constructor() { this.reset(); }
  reset() {
    this.C = I.slice(); this.S = I.slice(); this.W = I.slice(); this.mode = null;
    this.hist = []; this.raws = []; this.outs = []; this.lost = 0; this.frames = 0; this.clamped = 0;
  }
  // r: Worker の結果（kind: motion）。opts = { mode: "smooth" | "tripod", strength: "weak" | "mid" | "strong", crop: 0〜0.4 }
  update(r, opts) {
    const w = r.w, h = r.h;
    if (r.first || this.mode !== opts.mode || this.w !== w || this.h !== h) { this.reset(); this.mode = opts.mode; this.w = w; this.h = h; }
    this.frames++;
    const M = r.ok ? norm(r.M) : I;
    if (!r.first && !r.ok) this.lost++;
    const Wprev = this.W;
    this.C = norm(mul3(this.C, M));
    let S = opts.mode === "tripod" ? this.S : lerp(this.S, this.C, STAB_ALPHA[opts.strength] ?? 0.1);
    // 切り抜いた枠（画面全体）を元のフレームに戻した四隅が、フレームの中に入るまで S を C の側へ寄せる（二分探索）
    const z = 1 / (1 - opts.crop), Z = zoom(z, w, h); // crop は幅・高さのうち切り抜く割合（両側の合計）
    const inside = (Sx) => {
      const back = inv3(mul3(Z, mul3(inv3(Sx), this.C)));
      return [[0, 0], [w, 0], [w, h], [0, h]].every(([x, y]) => { const [u, v] = apply3(back, x, y); return u >= -0.5 && v >= -0.5 && u <= w + 0.5 && v <= h + 0.5; });
    };
    if (!inside(S)) {
      this.clamped++; // 切り抜きの端に当たった（補正を元の動きの側へ寄せた）フレーム
      let lo = 0, hi = 1;
      for (let k = 0; k < 20; k++) { const mid = (lo + hi) / 2; if (inside(lerp(S, this.C, mid))) hi = mid; else lo = mid; }
      S = lerp(S, this.C, hi);
    }
    this.S = norm(S);
    this.W = norm(mul3(inv3(this.S), this.C));
    this.Z = Z;
    // 補正前は M、補正後は D = W_{t−1}·M·W_t⁻¹（元のフレームの px）
    if (!r.first) {
      const raw = motionAt(M, w, h), out = motionAt(norm(mul3(Wprev, mul3(M, inv3(this.W)))), w, h);
      this.raws.push(raw); this.outs.push(out);
      if (this.raws.length > 5000) { this.raws.shift(); this.outs.shift(); }
      this.hist.push({ raw: Math.hypot(raw.x, raw.y), out: Math.hypot(out.x, out.y) });
      if (this.hist.length > 240) this.hist.shift();
    }
    const a = metrics(this.raws), b = metrics(this.outs);
    r.stab = { W: this.W, Z, crop: opts.crop, mode: opts.mode, strength: opts.strength, hist: this.hist.slice(), lost: this.lost, clamped: this.clamped, frames: this.frames,
      jitter: { raw_px: a.jit_px, raw_deg: a.jit_deg, out_px: b.jit_px, out_deg: b.jit_deg, move_raw_px: a.move_px, move_out_px: b.move_px, n: this.raws.length } };
    return r.stab;
  }
}
