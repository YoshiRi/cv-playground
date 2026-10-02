// 手ぶれ補正（画面側）: Worker が返すフレーム間の動き M_t（今のフレームの点 → 前のフレームの点、3×3）から、カメラの動きの軌跡
// C_t = C_{t−1}·M_t（今のフレーム → 最初のフレーム）を積み、なめらかにした軌跡 S_t との差 W_t = S_t⁻¹·C_t でフレームを描き直す。
// 周りを切り抜く（crop の割合だけ拡大）ので、W_t で動かしても画面の中は埋まっている。埋まらなくなる時は S_t を C_t の側へ寄せる
//
//   なめらか: S_t = S_{t−1} + α (C_t − S_{t−1})（要素ごと。小さい回転なら相似変換・ホモグラフィとも近似として足りる）
//   三脚:     S_t = 最初のフレーム（I）のまま。切り抜きの外に出る時だけ寄せる
//
// 揺れの指標: 補正前は M_t、補正後は画面の上での動き D_t = Z·W_{t−1}·M_t·W_t⁻¹·Z⁻¹ の、画面の中心のずれ（px）と回転（度）の二乗平均

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
// 動き（3×3）の、画面の中心のずれ（px）と回転（度）
function motionAt(m, w, h) {
  const [x, y] = apply3(m, w / 2, h / 2);
  return { t: Math.hypot(x - w / 2, y - h / 2), deg: (Math.atan2(m[3] - m[6] * m[5], m[0] - m[6] * m[2]) * 180) / Math.PI };
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
    this.hist = []; this.sum = { rt: 0, rd: 0, ot: 0, od: 0, n: 0 }; this.lost = 0; this.frames = 0;
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
      let lo = 0, hi = 1;
      for (let k = 0; k < 20; k++) { const mid = (lo + hi) / 2; if (inside(lerp(S, this.C, mid))) hi = mid; else lo = mid; }
      S = lerp(S, this.C, hi);
    }
    this.S = norm(S);
    this.W = norm(mul3(inv3(this.S), this.C));
    this.Z = Z;
    // 揺れ: 補正前は M、補正後は画面の上での動き D = W_{t−1}·M·W_t⁻¹
    if (!r.first) {
      // 画面の上の動きは拡大（切り抜き）も含める: Z·W_{t−1}·M·W_t⁻¹·Z⁻¹
      const raw = motionAt(M, w, h), out = motionAt(norm(mul3(Z, mul3(Wprev, mul3(M, mul3(inv3(this.W), inv3(Z)))))), w, h);
      this.hist.push({ raw: raw.t, out: out.t });
      if (this.hist.length > 240) this.hist.shift();
      const s = this.sum;
      s.rt += raw.t ** 2; s.rd += raw.deg ** 2; s.ot += out.t ** 2; s.od += out.deg ** 2; s.n++;
    }
    const s = this.sum, rms = (v) => (s.n ? Math.sqrt(v / s.n) : 0);
    r.stab = { W: this.W, Z, crop: opts.crop, mode: opts.mode, hist: this.hist.slice(), lost: this.lost, frames: this.frames,
      jitter: { raw_px: rms(s.rt), raw_deg: rms(s.rd), out_px: rms(s.ot), out_deg: rms(s.od), n: s.n } };
    return r.stab;
  }
}
