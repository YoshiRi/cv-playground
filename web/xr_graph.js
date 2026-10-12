// AR の地図の姿勢グラフ（WebXR に依存しない。ライブと記録の再生で同じものを使う）
//
//   ノード: サブマップ（4 自由度の姿勢 x, y, z, yaw）。最初のサブマップを固定する
//   辺: 1. つながり（オドメトリ）: 続くサブマップのアンカー同士の相対姿勢（ARCore の推定そのもの。ARCore が直すと変わる）
//       2. ループ閉じ: 書き終わったサブマップを、近くの古いサブマップと照合して得た相対姿勢
//   照合: 上から見た障害物の升目どうしを、今の推定のまわり（向き ±7°・位置 ±0.5 m）で総当たり（相関で合わせる。Cartographer 系の
//       枯れたやり方）。合わせる先は升目とそのまわり 1 升をぼかした地図。点の 5 割以上が重なり、スコアが MATCH_MIN を超えたら採用
//   最適化: ガウス・ニュートン法（ヤコビアンは数値微分、正規方程式は密な行列のコレスキー分解。ノードが数百なら十分速い）。
//       解いたあと、残差が大きいループ閉じは間違いとみなして外し、もう一度解く
//   結果は各サブマップの補正 corr（corr ⊕ アンカー = 最適化した姿勢）として地図に書く（物の位置もサブマップと一緒に動く）
import { P4, p4Apply, p4Between, p4Compose, p4Inv, wrap } from "./xr_geom.js";
import { CELL } from "./xr_map.js";

const NEAR_M = 4, MIN_CELLS = 60, MATCH_MIN = 0.5, OVERLAP_MIN = 0.5;
const SIG_ODO = [0.08, 0.05, 0.08, 0.03], SIG_LOOP = [0.05, 1, 0.05, 0.02]; // x, y, z（m）、yaw（rad）。ループ閉じは高さを決めない
const key = (i, j) => (i + 32768) * 65536 + (j + 32768);

export class PoseGraph {
  constructor() { this.reset(); }
  reset() { this.loops = []; this.tried = new Set(); this.stats = { tried: 0, accepted: 0, rejected: 0, ms: 0 }; }

  // 書き終わったサブマップ（最後のもの以外。final なら最後も）を、まだ試していない近くの古いサブマップと照合し、ループ閉じが増えたら解き直す
  update(map, final = false) {
    const t0 = performance.now(), subs = map.subs, n = subs.length;
    if (n < 3 || map.floorY == null) return false;
    let added = 0;
    for (let i = 2; i < (final ? n : n - 1); i++) {
      for (let j = 0; j < i - 1; j++) {
        const tag = `${j}-${i}`;
        if (this.tried.has(tag)) continue;
        const pi = map.pose(subs[i]), pj = map.pose(subs[j]);
        if (Math.hypot(pi.x - pj.x, pi.z - pj.z) > NEAR_M) continue;
        this.tried.add(tag); this.stats.tried++;
        const m = matchSubmaps(map, subs[j], subs[i]);
        if (m) { this.loops.push({ j, i, z: m.rel, score: m.score }); this.stats.accepted++; added++; }
      }
    }
    if (added) this.optimize(map);
    this.stats.ms += performance.now() - t0;
    return added > 0;
  }

  optimize(map) {
    for (let round = 0; round < 2; round++) {
      solve(map, this.loops);
      // 間違ったループ閉じ（解いても合わない）を外して解き直す
      const bad = this.loops.filter((L) => chi2(edgeErr(L.z, map.pose(map.subs[L.j]), map.pose(map.subs[L.i])), SIG_LOOP) > 25);
      if (!bad.length) break;
      this.loops = this.loops.filter((L) => !bad.includes(L)); this.stats.rejected += bad.length;
    }
    map.version++;
  }
}

// ---------- 照合 ----------
// サブマップ b を a に合わせる。今の推定 a⁻¹ ⊕ b のまわりで、b の障害物の升目が a の障害物（ぼかし）に一番重なる相対姿勢
export function matchSubmaps(map, a, b, debug = false) {
  const A = map.cellsLocal(a, [2]), B = map.cellsLocal(b, [2]);
  if (A.length < MIN_CELLS || B.length < MIN_CELLS) return debug ? { nA: A.length, nB: B.length } : null;
  const like = new Map(); // a の升目 → 重なりの点数（升目そのもの 1、まわり 1 升 0.5）
  for (const [x, z] of A) {
    const i = Math.floor(x / CELL), j = Math.floor(z / CELL);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) {
      const k = key(i + di, j + dj), v = di || dj ? 0.5 : 1;
      if ((like.get(k) || 0) < v) like.set(k, v);
    }
  }
  const pts = B.length > 300 ? B.filter((_, k) => k % Math.ceil(B.length / 300) === 0) : B;
  const guess = p4Between(map.pose(a), map.pose(b));
  const score = (yaw, tx, tz) => {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    let sum = 0, hit = 0;
    for (const [x, z] of pts) {
      const v = like.get(key(Math.floor((c * x + s * z + tx) / CELL), Math.floor((-s * x + c * z + tz) / CELL)));
      if (v) { sum += v; hit++; }
    }
    return [sum / pts.length, hit / pts.length];
  };
  let best = { s: -1 };
  for (let dy = -0.12; dy <= 0.1201; dy += 0.02) for (let dx = -0.5; dx <= 0.5001; dx += 0.05) for (let dz = -0.5; dz <= 0.5001; dz += 0.05) {
    const [s, o] = score(guess.yaw + dy, guess.x + dx, guess.z + dz);
    if (s > best.s) best = { s, o, dy, dx, dz };
  }
  // 細かく（向き 0.5°、位置 1 cm）
  for (let ry = -0.01; ry <= 0.0101; ry += 0.005) for (let rx = -0.03; rx <= 0.0301; rx += 0.01) for (let rz = -0.03; rz <= 0.0301; rz += 0.01) {
    const [s, o] = score(guess.yaw + best.dy + ry, guess.x + best.dx + rx, guess.z + best.dz + rz);
    if (s > best.s) best = { s, o, dy: best.dy + ry, dx: best.dx + rx, dz: best.dz + rz };
  }
  if (debug) return { ...best, nA: A.length, nB: B.length, guess };
  if (best.s < MATCH_MIN || best.o < OVERLAP_MIN) return null;
  return { rel: P4(guess.x + best.dx, guess.y, guess.z + best.dz, wrap(guess.yaw + best.dy)), score: best.s, overlap: best.o };
}

// ---------- 最適化 ----------
// 辺の誤差: 測った相対姿勢 z と、今の推定の相対姿勢（xj⁻¹ ⊕ xi）の差（z⁻¹ ⊕ 推定）
function edgeErr(z, xj, xi) {
  const e = p4Between(z, p4Between(xj, xi));
  return [e.x, e.y, e.z, wrap(e.yaw)];
}
const chi2 = (e, sig) => e.reduce((s, v, k) => s + (v / sig[k]) ** 2, 0);

function solve(map, loops) {
  const subs = map.subs, n = subs.length;
  if (n < 2) return;
  const x = subs.map((s) => map.pose(s)); // 今の推定から始める
  const edges = [];
  for (let k = 0; k + 1 < n; k++) edges.push({ j: k, i: k + 1, z: p4Between(subs[k].anchor, subs[k + 1].anchor), sig: SIG_ODO });
  for (const L of loops) if (L.i < n && L.j < n) edges.push({ j: L.j, i: L.i, z: L.z, sig: SIG_LOOP });
  const N = 4 * (n - 1), idx = (k) => 4 * (k - 1); // ノード 0 は固定
  const get = (p) => [p.x, p.y, p.z, p.yaw], set = (v) => P4(v[0], v[1], v[2], v[3]);
  for (let it = 0; it < 10; it++) {
    const H = new Float64Array(N * N), g = new Float64Array(N);
    for (const E of edges) {
      const e0 = edgeErr(E.z, x[E.j], x[E.i]), w = E.sig.map((s) => 1 / (s * s));
      // 数値微分: ノード j・i の 4 成分ずつ
      const J = [];
      for (const [node, off] of [[E.j, 0], [E.i, 4]]) for (let d = 0; d < 4; d++) {
        const v = get(x[node]), h = 1e-5; v[d] += h;
        const xx = node === E.j ? [set(v), x[E.i]] : [x[E.j], set(v)];
        const e1 = edgeErr(E.z, xx[0], xx[1]);
        J[off + d] = e1.map((val, r) => (r === 3 ? wrap(val - e0[r]) : val - e0[r]) / h);
      }
      const cols = [];
      for (let c = 0; c < 8; c++) { const node = c < 4 ? E.j : E.i; if (node > 0) cols.push([c, idx(node) + (c % 4)]); }
      for (const [ca, ga] of cols) {
        for (let r = 0; r < 4; r++) g[ga] += J[ca][r] * w[r] * e0[r];
        for (const [cb, gb] of cols) { let s = 0; for (let r = 0; r < 4; r++) s += J[ca][r] * w[r] * J[cb][r]; H[ga * N + gb] += s; }
      }
    }
    for (let k = 0; k < N; k++) H[k * N + k] += 1e-6; // 決まらない方向（ノード 0 から切れた所など）のため
    const dx = cholSolve(H, g, N);
    if (!dx) break;
    let step = 0;
    for (let k = 1; k < n; k++) {
      const v = get(x[k]);
      for (let d = 0; d < 4; d++) { v[d] -= dx[idx(k) + d]; step = Math.max(step, Math.abs(dx[idx(k) + d])); }
      v[3] = wrap(v[3]); x[k] = set(v);
    }
    if (step < 1e-5) break;
  }
  for (let k = 0; k < n; k++) subs[k].corr = p4Compose(x[k], p4Inv(subs[k].anchor));
}

// 対称な正定値行列の連立方程式 H x = g（コレスキー分解）
function cholSolve(H, g, N) {
  const L = new Float64Array(N * N);
  for (let i = 0; i < N; i++) for (let j = 0; j <= i; j++) {
    let s = H[i * N + j];
    for (let k = 0; k < j; k++) s -= L[i * N + k] * L[j * N + k];
    if (i === j) { if (s <= 0) return null; L[i * N + i] = Math.sqrt(s); } else L[i * N + j] = s / L[j * N + j];
  }
  const y = new Float64Array(N), x = new Float64Array(N);
  for (let i = 0; i < N; i++) { let s = g[i]; for (let k = 0; k < i; k++) s -= L[i * N + k] * y[k]; y[i] = s / L[i * N + i]; }
  for (let i = N - 1; i >= 0; i--) { let s = y[i]; for (let k = i + 1; k < N; k++) s -= L[k * N + i] * x[k]; x[i] = s / L[i * N + i]; }
  return x;
}
