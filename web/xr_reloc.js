// 前回の地図に、今の地図を合わせる（再位置合わせ）。WebXR はセッションごとに座標の原点が変わり、続けて使えるアンカーも無いので、
// 上から見た障害物の升目どうしを、向き 360° の総当たりで照合する（WebXR に依存しない）
//   1. 粗く（20 cm の升目）: 向き 3° ごとに、今の点と前回の点の全部の組の「ずらし」を投票し、票が一番多いずらしを候補にする
//      （ハフ変換と同じ考え方。点の数 × 前回の升目の数 × 向きの数 だが、票は配列に数えるので速い）
//   2. 細かく（5 cm、±3°・±0.3 m）: 候補ごとに、今の点が前回の障害物（まわり 1 升をぼかす）に重なる割合で選び直す
//   3. 一番良い候補が、別の所（向き 15° 以上か位置 0.5 m 以上違う）の 2 番目の候補より十分良い時だけ採用（似た部屋の誤りを避ける）
import { P4, wrap } from "./xr_geom.js";
import { CELL } from "./xr_map.js";

const COARSE = 0.2, YAW_STEP = (3 * Math.PI) / 180;
const key = (i, j) => (i + 32768) * 65536 + (j + 32768);

// cur: 今の障害物の点 [x, z]（今の部屋の座標）、ref: 前回の障害物の点 [x, z]（前回の部屋の座標）。
// 今 → 前回の 2D の姿勢（p_前回 = R(yaw)·p_今 + t）と、スコアを返す（合わなければ null）
export function globalMatch(cur, ref, { minScore = 0.5, minOverlap = 0.5, margin = 0.1 } = {}) {
  if (cur.length < 60 || ref.length < 60) return null;
  // 粗い升目にまとめる
  const coarse = (pts) => { const m = new Map(); for (const [x, z] of pts) m.set(key(Math.floor(x / COARSE), Math.floor(z / COARSE)), [(Math.floor(x / COARSE) + 0.5) * COARSE, (Math.floor(z / COARSE) + 0.5) * COARSE]); return [...m.values()]; };
  const R = coarse(ref), C0 = coarse(cur), C = C0.length > 120 ? C0.filter((_, k) => k % Math.ceil(C0.length / 120) === 0) : C0;
  let rx0 = Infinity, rx1 = -Infinity, rz0 = Infinity, rz1 = -Infinity;
  for (const [x, z] of R) { rx0 = Math.min(rx0, x); rx1 = Math.max(rx1, x); rz0 = Math.min(rz0, z); rz1 = Math.max(rz1, z); }
  const rad = Math.max(...C.map(([x, z]) => Math.hypot(x, z))) + COARSE;
  const tx0 = rx0 - rad, tz0 = rz0 - rad, nx = Math.ceil((rx1 - rx0 + 2 * rad) / COARSE) + 1, nz = Math.ceil((rz1 - rz0 + 2 * rad) / COARSE) + 1;
  const acc = new Uint16Array(nx * nz), cands = [];
  for (let yaw = 0; yaw < 2 * Math.PI; yaw += YAW_STEP) {
    acc.fill(0);
    const c = Math.cos(yaw), s = Math.sin(yaw);
    let best = 0, bi = 0;
    for (const [x, z] of C) {
      const px = c * x + s * z, pz = -s * x + c * z;
      for (const [qx, qz] of R) {
        const i = Math.round((qx - px - tx0) / COARSE), j = Math.round((qz - pz - tz0) / COARSE);
        if (i < 0 || j < 0 || i >= nx || j >= nz) continue;
        const v = ++acc[j * nx + i];
        if (v > best) { best = v; bi = j * nx + i; }
      }
    }
    cands.push({ yaw, votes: best, tx: tx0 + (bi % nx) * COARSE, tz: tz0 + Math.floor(bi / nx) * COARSE });
  }
  cands.sort((a, b) => b.votes - a.votes);
  // 細かく選び直す（上位 8 つ）
  const like = new Map();
  for (const [x, z] of ref) {
    const i = Math.floor(x / CELL), j = Math.floor(z / CELL);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) { const k = key(i + di, j + dj), v = di || dj ? 0.5 : 1; if ((like.get(k) || 0) < v) like.set(k, v); }
  }
  const pts = cur.length > 300 ? cur.filter((_, k) => k % Math.ceil(cur.length / 300) === 0) : cur;
  const score = (yaw, tx, tz) => {
    const c = Math.cos(yaw), s = Math.sin(yaw);
    let sum = 0, hit = 0;
    for (const [x, z] of pts) { const v = like.get(key(Math.floor((c * x + s * z + tx) / CELL), Math.floor((-s * x + c * z + tz) / CELL))); if (v) { sum += v; hit++; } }
    return [sum / pts.length, hit / pts.length];
  };
  const refined = cands.slice(0, 8).map((k) => {
    let b = { s: -1 };
    for (let dy = -0.05; dy <= 0.0501; dy += 0.01) for (let dx = -0.3; dx <= 0.3001; dx += 0.05) for (let dz = -0.3; dz <= 0.3001; dz += 0.05) {
      const [s, o] = score(k.yaw + dy, k.tx + dx, k.tz + dz);
      if (s > b.s) b = { s, o, yaw: k.yaw + dy, tx: k.tx + dx, tz: k.tz + dz };
    }
    return b;
  }).sort((a, b) => b.s - a.s);
  const best = refined[0], far = (a) => Math.abs(wrap(a.yaw - best.yaw)) > 0.26 || Math.hypot(a.tx - best.tx, a.tz - best.tz) > 0.5;
  const second = refined.find((a) => a !== best && far(a))?.s ?? 0;
  const ok = best.s >= minScore && best.o >= minOverlap && best.s - second >= margin;
  return { ok, rel: P4(best.tx, 0, best.tz, wrap(best.yaw)), score: best.s, overlap: best.o, second };
}
