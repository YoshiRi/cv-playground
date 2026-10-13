// AR の地図の姿勢グラフ（WebXR に依存しない。ライブと記録の再生で同じものを使う）
//
//   ノード: サブマップ（4 自由度の姿勢 x, y, z, yaw）。最初のサブマップを固定する
//   辺: 1. つながり（オドメトリ）: 続くサブマップのアンカー同士の相対姿勢（ARCore の推定そのもの。ARCore が直すと変わる）
//       2. ループ閉じ: 書き終わったサブマップを、近くの古いサブマップと照合して得た相対姿勢
//   照合: 上から見た障害物・床の升目どうしを、今の推定のまわり（向き ±7°・位置 ±0.5 m）で総当たり（相関で合わせる。Cartographer 系の
//       枯れたやり方。点数は xr_match.js）。点の 5 割以上が相手の分かっている所に乗り、点数が MATCH_MIN を超えたら採用
//   最適化: ガウス・ニュートン法（ヤコビアンは数値微分、正規方程式は密な行列のコレスキー分解。ノードが数百なら十分速い）。
//       解いたあと、残差が大きいループ閉じは間違いとみなして外し、もう一度解く
//   結果は各サブマップの補正 corr（corr ⊕ アンカー = 最適化した姿勢）として地図に書く（物の位置もサブマップと一緒に動く）
import { P4, p4Apply, p4Between, p4Compose, p4Inv, wrap } from "./xr_geom.js";
import { buildRef, search, thin } from "./xr_match.js";

const NEAR_M = 4, MIN_CELLS = 60, MATCH_MIN = 0.35, OVERLAP_MIN = 0.5; // 点数は xr_match.js（空いている所に乗ると減る）
const SIG_ODO = [0.08, 0.05, 0.08, 0.03], SIG_LOOP = [0.05, 1, 0.05, 0.02]; // x, y, z（m）、yaw（rad）。ループ閉じは高さを決めない

export class PoseGraph {
  constructor() { this.reset(); }
  reset() { this.loops = []; this.tried = new Set(); this.stats = { tried: 0, accepted: 0, rejected: 0, ms: 0 }; }

  // 照合する組（書き終わったサブマップ ＝ 最後のもの以外。final なら最後も）を、まだ試していない近くの古いサブマップと作る。
  // 照合そのもの（matchJob）は重いので、ライブでは Worker で回す（xr_worker.js）。再生ではその場で回す（update）
  jobs(map, final = false) {
    const subs = map.subs, n = subs.length, out = [];
    if (n < 3 || map.floorY == null) return out;
    for (let i = 2; i < (final ? n : n - 1); i++) for (let j = 0; j < i - 1; j++) {
      const tag = `${j}-${i}`;
      if (this.tried.has(tag)) continue;
      const pi = map.pose(subs[i]), pj = map.pose(subs[j]);
      if (Math.hypot(pi.x - pj.x, pi.z - pj.z) > NEAR_M) continue;
      this.tried.add(tag); this.stats.tried++;
      const cells = (s) => { const c = map.cellsLocal(s); return { obst: c.filter((x) => x[2] === 2).map(([x, z]) => [x, z]), floor: c.filter((x) => x[2] === 1).map(([x, z]) => [x, z]) }; };
      out.push({ j, i, A: cells(subs[j]), B: cells(subs[i]), guess: p4Between(pj, pi) });
    }
    return out;
  }
  // 照合の結果を足して解き直す
  add(map, job, m) {
    if (!m || job.i >= map.subs.length) return false;
    this.loops.push({ j: job.j, i: job.i, z: m.rel, score: m.score }); this.stats.accepted++;
    this.optimize(map);
    return true;
  }
  update(map, final = false) {
    const t0 = performance.now();
    let added = false;
    for (const job of this.jobs(map, final)) added = this.add(map, job, matchJob(job)) || added;
    this.stats.ms += performance.now() - t0;
    return added;
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
// サブマップ B を A に合わせる（job = { A, B: {obst, floor}（サブマップの座標）, guess: 今の推定 A⁻¹ ⊕ B }）。
// 今の推定のまわり（向き ±7°・位置 ±0.5 m）を粗く、そのまわりを細かく探し、空いている所も使った点数（xr_match.js）で選ぶ
export function matchJob({ A, B, guess }) {
  if (A.obst.length < MIN_CELLS || B.obst.length < MIN_CELLS) return null;
  const ref = buildRef(A.obst, A.floor), ob = thin(B.obst, 300), fl = thin(B.floor, 200);
  const c = search(ref, ob, fl, guess.yaw, guess.x, guess.z, [0.12, 0.5], [0.02, 0.05]);
  const f = search(ref, ob, fl, c.yaw, c.tx, c.tz, [0.01, 0.03], [0.005, 0.01]);
  if (f.s < MATCH_MIN || f.o < OVERLAP_MIN) return null;
  return { rel: P4(f.tx, guess.y, f.tz, wrap(f.yaw)), score: f.s, overlap: f.o };
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
