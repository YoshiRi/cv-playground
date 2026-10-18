// 前回の地図に、今の地図を合わせる（再位置合わせ）。WebXR はセッションごとに座標の原点が変わり、続けて使えるアンカーも無いので、
// 上から見た障害物の升目どうしを、向き 360° の総当たりで照合する（WebXR に依存しない）
//   1. 粗く（20 cm の升目）: 向き 3° ごとに、今の点と前回の点の全部の組の「ずらし」を投票し、票が一番多いずらしを候補にする
//      （ハフ変換と同じ考え方。点の数 × 前回の升目の数 × 向きの数 だが、票は配列に数えるので速い）
//   2. 細かく（5 cm、±3°・±0.3 m）: 上位の候補ごとに、空いている所も使った点数（xr_match.js）で選び直す
//   3. 一番良い候補が、別の所（向き 15° 以上か位置 0.5 m 以上違う）の 2 番目の候補より十分良い時だけ採用（似た部屋の誤りを避ける）。
//      候補は、近い所（向き 10°・位置 0.4 m 以内）の重複を除いてから上位を選ぶ（除かないと同じ解の近くで枠が埋まり、別の所の候補を
//      調べないまま「2 番目は 0」として採用してしまう。Codex のレビューで見つかった）。別の所の候補を 1 つも調べられなければ採用しない
import { P4, wrap } from "./xr_geom.js";
import { buildRef, search, thin } from "./xr_match.js";

const COARSE = 0.2, YAW_STEP = (3 * Math.PI) / 180;
const key = (i, j) => (i + 32768) * 65536 + (j + 32768);

// cur・ref: { obst, floor }（障害物と床の点 [x, z]。cur は今の部屋の座標、ref は前回の部屋の座標）。
// 今 → 前回の 2D の姿勢（p_前回 = R(yaw)·p_今 + t）と、点数を返す
export function globalMatch(cur, ref, { minScore = 0.3, minOverlap = 0.5, margin = 0.08, top = 16 } = {}) {
  if (cur.obst.length < 60 || ref.obst.length < 60) return null;
  // 粗い升目にまとめる
  const coarse = (pts) => { const m = new Map(); for (const [x, z] of pts) m.set(key(Math.floor(x / COARSE), Math.floor(z / COARSE)), [(Math.floor(x / COARSE) + 0.5) * COARSE, (Math.floor(z / COARSE) + 0.5) * COARSE]); return [...m.values()]; };
  const R = coarse(ref.obst), C = thin(coarse(cur.obst), 120);
  let rx0 = Infinity, rx1 = -Infinity, rz0 = Infinity, rz1 = -Infinity;
  for (const [x, z] of R) { rx0 = Math.min(rx0, x); rx1 = Math.max(rx1, x); rz0 = Math.min(rz0, z); rz1 = Math.max(rz1, z); }
  const rad = Math.max(...C.map(([x, z]) => Math.hypot(x, z))) + COARSE;
  const tx0 = rx0 - rad, tz0 = rz0 - rad, nx = Math.ceil((rx1 - rx0 + 2 * rad) / COARSE) + 1, nz = Math.ceil((rz1 - rz0 + 2 * rad) / COARSE) + 1;
  const acc = new Uint16Array(nx * nz), cands = [];
  for (let yaw = 0; yaw < 2 * Math.PI; yaw += YAW_STEP) {
    acc.fill(0);
    const c = Math.cos(yaw), s = Math.sin(yaw);
    // 向きごとに、票の多いずらしを 2 つ（1 つだと、物の多い部屋で本当の位置が 2 番目に埋もれる）
    for (const [x, z] of C) {
      const px = c * x + s * z, pz = -s * x + c * z;
      for (const [qx, qz] of R) {
        const i = Math.round((qx - px - tx0) / COARSE), j = Math.round((qz - pz - tz0) / COARSE);
        if (i >= 0 && j >= 0 && i < nx && j < nz) acc[j * nx + i]++;
      }
    }
    // 1 位と、1 位から 2 升（40 cm）より離れた中での 1 位
    let b1 = 0, i1 = 0;
    for (let k = 0; k < acc.length; k++) if (acc[k] > b1) { b1 = acc[k]; i1 = k; }
    let b2 = 0, i2 = -1;
    for (let k = 0; k < acc.length; k++) if (acc[k] > b2 && Math.hypot((k % nx) - (i1 % nx), Math.floor(k / nx) - Math.floor(i1 / nx)) > 2) { b2 = acc[k]; i2 = k; }
    for (const [v, k] of [[b1, i1], [b2, i2]]) if (k >= 0 && v > 0) cands.push({ yaw, votes: v, tx: tx0 + (k % nx) * COARSE, tz: tz0 + Math.floor(k / nx) * COARSE });
  }
  cands.sort((a, b) => b.votes - a.votes);
  // 近い候補を除く（非最大値の抑制）
  const kept = [];
  for (const c of cands) {
    if (kept.length >= top) break;
    if (!kept.some((k) => Math.abs(wrap(k.yaw - c.yaw)) < 0.17 && Math.hypot(k.tx - c.tx, k.tz - c.tz) < 0.4)) kept.push(c);
  }
  const g = buildRef(ref.obst, ref.floor), ob = thin(cur.obst, 300), fl = thin(cur.floor, 200);
  const refined = kept.map((k) => search(g, ob, fl, k.yaw, k.tx, k.tz, [0.05, 0.3], [0.01, 0.05])).sort((a, b) => b.s - a.s);
  const best = refined[0], far = (a) => Math.abs(wrap(a.yaw - best.yaw)) > 0.26 || Math.hypot(a.tx - best.tx, a.tz - best.tz) > 0.5;
  const rival = refined.find((a) => a !== best && far(a)), second = rival?.s ?? 0;
  const ok = !!rival && best.s >= minScore && best.o >= minOverlap && best.s - second >= margin;
  return { ok, rel: P4(best.tx, 0, best.tz, wrap(best.yaw)), score: best.s, overlap: best.o, second };
}
