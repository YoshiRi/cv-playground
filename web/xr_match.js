// 上から見た地図どうしの照合の点数（姿勢グラフのループ閉じと、前回の地図への再位置合わせで共通。WebXR・DOM に依存しない）
//   障害物の点だけで重なりを数えると、物の多い部屋（障害物の升目が多い）ではどこに置いても重なってしまい、区別がつかなかった
//   （実機の部屋で 1 位と 2 位が 0.98 対 0.98）。占有格子どうしの照合と同じく、空いている所（床）も使う:
//     今の障害物の点: 相手の障害物に乗れば +1（まわり 1 升なら +0.5）、相手の床（空いている）に乗れば −1、分からない所は 0
//     今の床の点:     相手の床に乗れば +0.5、相手の障害物に乗れば −1
//   点数 = 合計 / 点の数。重なり（overlap）= 相手の分かっている所（障害物か床）に乗った点の割合
import { CELL } from "./xr_map.js";

const key = (i, j) => (i + 32768) * 65536 + (j + 32768);

// 相手（ref）の格子: 障害物 2、そのまわり 1、床 −1（障害物のまわりより弱い）
export function buildRef(obst, floor) {
  const g = new Map();
  for (const [x, z] of floor) g.set(key(Math.floor(x / CELL), Math.floor(z / CELL)), -1);
  for (const [x, z] of obst) {
    const i = Math.floor(x / CELL), j = Math.floor(z / CELL);
    for (let di = -1; di <= 1; di++) for (let dj = -1; dj <= 1; dj++) { const k = key(i + di, j + dj); if ((g.get(k) ?? -9) < 1) g.set(k, 1); }
  }
  for (const [x, z] of obst) g.set(key(Math.floor(x / CELL), Math.floor(z / CELL)), 2);
  return g;
}
// 点の数を減らす（多すぎる時は等間隔に間引く）
export const thin = (pts, n) => (pts.length > n ? pts.filter((_, k) => k % Math.ceil(pts.length / n) === 0) : pts);

// 今の点（obst, floor）を姿勢 (yaw, tx, tz) で相手の座標に置いた時の点数と重なり
export function score(ref, obst, floor, yaw, tx, tz) {
  const c = Math.cos(yaw), s = Math.sin(yaw);
  let sum = 0, known = 0;
  for (const [x, z] of obst) {
    const v = ref.get(key(Math.floor((c * x + s * z + tx) / CELL), Math.floor((-s * x + c * z + tz) / CELL)));
    if (v == null) continue;
    known++; sum += v === 2 ? 1 : v === 1 ? 0.5 : -1;
  }
  for (const [x, z] of floor) {
    const v = ref.get(key(Math.floor((c * x + s * z + tx) / CELL), Math.floor((-s * x + c * z + tz) / CELL)));
    if (v == null) continue;
    known++; sum += v === -1 ? 0.5 : v === 2 ? -1 : 0;
  }
  const n = obst.length + floor.length;
  return [sum / n, known / n];
}
// (yaw, tx, tz) のまわりを総当たりで探す（範囲 r = [向き, 位置]、刻み st = [向き, 位置]）
export function search(ref, obst, floor, yaw0, tx0, tz0, r, st) {
  let b = { s: -Infinity };
  for (let dy = -r[0]; dy <= r[0] + 1e-9; dy += st[0]) for (let dx = -r[1]; dx <= r[1] + 1e-9; dx += st[1]) for (let dz = -r[1]; dz <= r[1] + 1e-9; dz += st[1]) {
    const [s, o] = score(ref, obst, floor, yaw0 + dy, tx0 + dx, tz0 + dz);
    if (s > b.s) b = { s, o, yaw: yaw0 + dy, tx: tx0 + dx, tz: tz0 + dz };
  }
  return b;
}
