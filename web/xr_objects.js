// AR で検出した物を、部屋の中の位置つきで覚えて集める（WebXR に依存しない。ライブと記録の再生で同じものを使う）
//
//   枠 → 3D: 読み戻した時のカメラの姿勢・深度の写しで、枠の中の深度の所に立てた板（4 隅）と中心にする
//   同一判定: 同じクラスで近い所（40 cm か板の幅の半分の大きい方）に来たら同じ物。位置は見えた回数で平均する
//   確かさ: 見えるたびに p ← 1 − (1 − p)(1 − W·score) で重ねる。COLLECT_P を超え、COLLECT_N 回以上見えたら「集めた」
//   動く物（人・動物）: 位置で同じ物かを決められない（歩くと別の物として何度も集まる）ので集めない。地図では深度を当たりにしない
//   位置は、最初に見えた時のサブマップの座標で持つ（姿勢グラフ・ARCore の補正でサブマップが動くと一緒に動く）
import { boxDepth, invRigid, p4Apply, p4Inv, project, unproject } from "./xr_geom.js";

export const DYNAMIC = new Set(["person", "dog", "cat", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"]);
const W_OBS = 0.3, MATCH_M = 0.4, FORGET_MS = 15000;
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);

// 枠（view の 0〜1、b.v）を 3D に。cap = { P, M, depth }。深度が無い枠は 2 m に置く（向きの変化の補正には距離はあまり効かない）
export function placeBox(b, cap) {
  const v = b.v;
  b.depth = cap.depth ? boxDepth(cap.depth, v) : null;
  const d = b.depth ?? 2;
  b.world = [[v[0], v[1]], [v[2], v[1]], [v[2], v[3]], [v[0], v[3]]].map(([x, y]) => unproject(cap.P, cap.M, x, y, d));
  b.center = unproject(cap.P, cap.M, (v[0] + v[2]) / 2, (v[1] + v[3]) / 2, d);
  return b;
}
// 動く物の枠を、今のカメラの view の 0〜1 の四角に（地図に積む時に、その中の深度を当たりにしない）
export function dynBoxes(list, P, M) {
  const C = invRigid(M), out = [];
  for (const b of list) {
    if (!DYNAMIC.has(b.label) || !b.world) continue;
    const q = b.world.map((p) => project(P, C, p));
    if (q.some((x) => !x)) continue;
    out.push([Math.min(...q.map((p) => p[0])), Math.min(...q.map((p) => p[1])), Math.max(...q.map((p) => p[0])), Math.max(...q.map((p) => p[1]))]);
  }
  return out;
}

export class ObjectStore {
  constructor({ collectP = 0.9, collectN = 5 } = {}) { this.collectP = collectP; this.collectN = collectN; this.reset(); }
  reset() { this.objs = []; this.seq = 0; }
  pos(o, map) { return p4Apply(map.pose(o.sub), o.local); }
  collected() { return this.objs.filter((o) => o.collected); }
  // 1 回分の検出（placeBox 済み）を足す。床が決まるまで（ARCore の推定が落ち着くまで）は集めない。
  // thumb(b) は一番確かだった時の切り抜きを作る（再生では無い）。新しく集めた物を返す
  add(list, t, map, thumb) {
    const fresh = [];
    if (map.floorY == null || !map.active) return fresh;
    for (const b of list) {
      if (b.depth == null || !b.center || DYNAMIC.has(b.label)) continue;
      const r = Math.max(MATCH_M, dist3(b.world[0], b.world[1]) * 0.5);
      let o = null, od = Infinity;
      for (const x of this.objs) { const d = x.label === b.label ? dist3(this.pos(x, map), b.center) : Infinity; if (d < r && d < od) { o = x; od = d; } }
      if (!o) {
        const sub = map.active;
        this.objs.push((o = { id: ++this.seq, label: b.label, sub, local: p4Apply(p4Inv(map.pose(sub)), b.center), n: 0, sum: 0, p: 0, best: 0, thumb: "", first: t, collected: false }));
      }
      o.n++; o.sum += b.score;
      const k = 1 / Math.min(o.n, 10), nl = p4Apply(p4Inv(map.pose(o.sub)), b.center); // 位置は見えた回数で平均（10 回目からは移動平均）
      o.local = o.local.map((v, i) => v + (nl[i] - v) * k);
      o.p = 1 - (1 - o.p) * (1 - W_OBS * b.score);
      o.last = t; b.obj = o;
      if (b.score > o.best) { o.best = b.score; o.dist = b.depth; if (thumb) o.thumb = thumb(b); }
      if (!o.collected && o.p >= this.collectP && o.n >= this.collectN) { o.collected = true; o.at = t; fresh.push(o); }
    }
    this.objs = this.objs.filter((o) => o.collected || t - o.last < FORGET_MS);
    return fresh;
  }
  // 地図に描く形
  forMap(map) { return this.objs.map((o) => ({ pos: this.pos(o, map), label: o.label, collected: o.collected })); }
}
