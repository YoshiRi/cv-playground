// AR の環境の地図（上から見た間取り）。WebXR に依存しない（ライブと記録の再生で同じものを使う）
//
// ARCore の地図は WebXR から取れないので、位置・向きの分かった深度を部屋の座標に戻して積む（位置は ARCore に任せ、地図だけ作る）。
//   ボクセル: 上から見た CELL 角の升目 × 高さ 10 cm 刻み。「物がある確からしさ」（log-odds）を持つ（OctoMap と同じ考え方）。
//     深度の点が当たったら上げ、カメラからその点までの光線が通り抜けたら下げる（通り過ぎた人・深度の誤差の点は、向こうを見ると消える）
//   サブマップ: 地図を SUBMAP_M 進むか SUBMAP_S 秒ごとに分け、それぞれをアンカー（ARCore が自分の地図を直すと一緒に動く点）に付ける。
//     ボクセルはサブマップの座標で持つので、ARCore の補正が地図に反映される。今の姿勢 = 姿勢グラフの補正 corr ⊕ アンカーの姿勢
//     （姿勢は 4 自由度。重力の向きは XR で分かっているので、位置 3 つと y 軸まわりの向きだけ）
//   床: 部屋の座標の高さの分布で、始めた時の端末の高さより 0.5 m 以上低い所で一番点が集まる高さ（床は一番広い水平面）。
//     床 ±10 cm を床、床から 10 cm〜1.8 m のボクセルのどれかが L_OCC を超えたら障害物（壁・家具）
//   動く物: 人などの枠に入る深度の点は当たりにしない（光線の通り抜けは数えるので、人が去った所は空きになる）
//   前回の地図（prior）: 保存した地図を、再位置合わせで求めた変換 T（前回の部屋の座標 → 今の部屋の座標）で重ねる。積み足しはしない
import { P4, depthAt, p4Apply, p4Compose, p4FromMatrix, p4Inv, unproject } from "./xr_geom.js";

export const CELL = 0.05, LY0 = -3, YB = 45; // サブマップの高さは −3〜+1.5 m（原点はアンカー。置いた時の端末の高さ）
const L_HIT = 0.85, L_MISS = -0.4, L_MIN = -2, L_MAX = 3.5, L_OCC = 1.2; // log-odds（当たり 1 回で p 0.7、通り抜け 1 回で 0.4）
const MAX_D = 3, FY0 = -2.5; // 当たりにする深度は 3 m まで。床を探す高さの分布は −2.5〜+2.5 m を 2 cm 刻み
export const SUBMAP_M = 2, SUBMAP_S = 30;
const cellKey = (i, j) => (i + 32768) * 65536 + (j + 32768);
const keyIJ = (k) => [Math.floor(k / 65536) - 32768, (k % 65536) - 32768];

export class OccMap {
  constructor() { this.reset(); }
  reset() {
    this.subs = []; this.yHist = new Uint32Array(250); this.floorY = null; this.n = 0; this.track = []; this.version = 0; this.prior = null;
  }
  get active() { return this.subs[this.subs.length - 1]; }
  pose(s) { const p = p4Compose(s.corr, s.anchor); return s.prior ? p4Compose(this.prior.T, p) : p; }
  // 前回の地図のサブマップ（今の座標に重ねる）と今のサブマップ
  allSubs() { return this.prior ? [...this.prior.subs, ...this.subs] : this.subs; }
  setPrior(priorMap, T) {
    for (const s of priorMap.subs) s.prior = true;
    this.prior = { subs: priorMap.subs, T }; this.version++;
  }
  // サブマップを足す（id はアンカーの番号、M は置いた時のアンカーの姿勢）。以後の深度はこのサブマップに積む
  addSubmap(id, M, t) {
    const s = { id, anchor: p4FromMatrix(M), corr: P4(), grid: new Map(), t0: t, n: 0 };
    this.subs.push(s); this.version++;
    return s;
  }
  // アンカーの今の姿勢（ARCore が直すと変わる）
  setAnchors(list) {
    for (const [id, M] of list) { const s = this.subs.find((x) => x.id === id); if (s) s.anchor = p4FromMatrix(M); }
    this.version++;
  }
  // 新しいサブマップが要るか（まだ無い、置いた所から SUBMAP_M 離れた、SUBMAP_S 秒たった）
  wantSubmap(pos, t) {
    const a = this.active;
    if (!a) return true;
    const p = this.pose(a);
    return Math.hypot(pos[0] - p.x, pos[2] - p.z) > SUBMAP_M || t - a.t0 > SUBMAP_S * 1000;
  }
  // 深度 1 枚を積む。fr = { t, M（カメラ → 部屋）, P（射影）, depth（深度の写し）, dyn（動く物の枠、view の 0〜1） }
  integrate(fr) {
    const s = this.active;
    if (!s || !fr.depth) return;
    const sp = this.pose(s), inv = p4Inv(sp), oL = p4Apply(inv, [fr.M[12], fr.M[13], fr.M[14]]);
    const minYb = this.floorY == null ? Infinity : Math.floor((this.floorY - sp.y - LY0) / 0.1) + 2;
    const dyn = (fr.dyn || []).map((b) => { const w = (b[2] - b[0]) * 0.1, h = (b[3] - b[1]) * 0.1; return [b[0] - w, b[1] - h, b[2] + w, b[3] + h]; });
    for (let j = 0; j < 24; j++) for (let i = 0; i < 40; i++) {
      const x = (i + 0.5) / 40, y = (j + 0.5) / 24, d = depthAt(fr.depth, x, y);
      if (!(d > 0.2)) continue;
      const doCarve = (i + j) % 2 === 0; // 光線は半分だけたどる（スマホで軽くするため。当たりは全部の点）
      // 遠い点は当たりにしない（粗い）が、MAX_D までは光線が通り抜けた（空いている）として下げる
      if (d >= MAX_D) { if (d < 8 && doCarve) this.carve(s, oL, p4Apply(inv, unproject(fr.P, fr.M, x, y, MAX_D)), MAX_D + 0.1, minYb); continue; }
      const p = unproject(fr.P, fr.M, x, y, d), hy = Math.floor((p[1] - FY0) / 0.02);
      if (hy >= 0 && hy < this.yHist.length) this.yHist[hy]++;
      const pL = p4Apply(inv, p);
      if (doCarve) this.carve(s, oL, pL, d, minYb);
      if (dyn.some((b) => x >= b[0] && x <= b[2] && y >= b[1] && y <= b[3])) continue; // 動く物の点は当たりにしない
      const yb = Math.floor((pL[1] - LY0) / 0.1);
      if (yb < 0 || yb >= YB) continue;
      const c = this.cell(s, Math.floor(pL[0] / CELL), Math.floor(pL[2] / CELL));
      c[yb] = Math.min(L_MAX, c[yb] + L_HIT);
    }
    s.n++; this.n++; this.version++;
    this.track.push([this.subs.length - 1, oL[0], oL[2]]); // 歩いた跡もサブマップの座標で（補正で一緒に動く）
    if (this.track.length > 3000) this.track.shift();
    if (this.n % 10 === 0) this.updateFloor();
  }
  cell(s, i, j) {
    const key = cellKey(i, j);
    let c = s.grid.get(key);
    if (!c) s.grid.set(key, (c = new Float32Array(YB)));
    return c;
  }
  // サブマップの座標で、o から p までの光線が通り抜けたボクセルを下げる（5 cm 刻み、点の手前 10 cm まで）。
  // 床の帯とその下は下げない（遠くの床へ浅い角度で入る光線が床を消す）。床が決まるまでは下げない
  carve(s, o, p, d, minYb) {
    if (!Number.isFinite(minYb)) return;
    const n = Math.floor((d - 0.1) / CELL);
    let last = -1;
    for (let k = 1; k < n; k++) {
      const t = (k * CELL) / d, x = o[0] + (p[0] - o[0]) * t, y = o[1] + (p[1] - o[1]) * t, z = o[2] + (p[2] - o[2]) * t;
      const yb = Math.floor((y - LY0) / 0.1);
      if (yb < minYb || yb >= YB) continue;
      const ck = cellKey(Math.floor(x / CELL), Math.floor(z / CELL)), key = ck * 64 + yb;
      if (key === last) continue;
      last = key;
      const c = s.grid.get(ck);
      if (c) c[yb] = Math.max(L_MIN, c[yb] + L_MISS); // 一度も当たっていない升目は作らない（分からないまま）
    }
  }
  // 床の高さ: −0.5 m より下で、点の数（前後 2 cm をならす）が一番多い高さ。その高さに 500 点たまってから
  updateFloor() {
    const top = Math.floor((-0.5 - FY0) / 0.02);
    let best = -1, bi = -1;
    for (let i = 2; i < top; i++) { const v = this.yHist[i - 1] + this.yHist[i] + this.yHist[i + 1]; if (v > best) { best = v; bi = i; } }
    if (best >= 500) this.floorY = FY0 + (bi + 0.5) * 0.02;
  }
  // 升目の種類: 2 = 障害物、1 = 床、0 = 分からない（fb はサブマップの座標での床の帯）
  static kind(c, fb) {
    let f = -Infinity, o = -Infinity;
    for (let b = Math.max(0, fb - 1); b <= fb + 1 && b < YB; b++) f = Math.max(f, c[b]);
    for (let b = Math.max(0, fb + 2); b <= fb + 18 && b < YB; b++) o = Math.max(o, c[b]);
    return o > L_OCC ? 2 : f > 0 ? 1 : 0;
  }
  floorBand(s, sp = this.pose(s)) { return Math.floor((this.floorY - sp.y - LY0) / 0.1); }
  // サブマップの障害物・床の升目（サブマップの座標の升目の中心 [x, z, 種類]）
  cellsLocal(s, kinds = [1, 2]) {
    const out = [];
    if (this.floorY == null) return out;
    const fb = this.floorBand(s);
    for (const [key, c] of s.grid) {
      const k = OccMap.kind(c, fb);
      if (!kinds.includes(k)) continue;
      const [i, j] = keyIJ(key);
      out.push([(i + 0.5) * CELL, (j + 0.5) * CELL, k]);
    }
    return out;
  }
  trackWorld() {
    const poses = this.subs.map((s) => this.pose(s));
    return this.track.map(([k, x, z]) => { const w = p4Apply(poses[k], [x, 0, z]); return [w[0], w[2]]; });
  }
  // 全部のサブマップの升目を部屋の座標で [x, z, 種類]（床を先、障害物を後に描けるよう分けて返す）
  cellsWorld() {
    const floor = [], obst = [];
    for (const s of this.allSubs()) {
      const sp = this.pose(s);
      for (const [x, z, k] of this.cellsLocal(s)) { const w = p4Apply(sp, [x, 0, z]); (k === 2 ? obst : floor).push([w[0], w[2]]); }
    }
    return { floor, obst };
  }

  // 地図を描く（x0, y0 から S 角）。升目は版が変わってから 0.5 秒たったら描き直して取っておき、自分・物は毎回描く。
  // 向きは部屋に固定（始めた時の前が上）。x が右、z が下。opts = { objs: [{pos, label, collected}], cur: {pos, M} }
  draw(ctx, x0, y0, S, u, opts = {}) {
    const objs = opts.objs || [], now = performance.now();
    if (!this.allSubs().length && !objs.length) return;
    const c0 = this.cache;
    if (!c0 || c0.S !== S || (c0.version !== this.version && now - c0.t > 500)) {
      const { floor, obst } = this.cellsWorld();
      let mnx = Infinity, mxx = -Infinity, mnz = Infinity, mxz = -Infinity;
      const ext = (x, z) => { mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mnz = Math.min(mnz, z); mxz = Math.max(mxz, z); };
      for (const [x, z] of floor) ext(x, z);
      for (const [x, z] of obst) ext(x, z);
      const track = this.trackWorld();
      for (const [x, z] of track) ext(x, z);
      for (const o of objs) ext(o.pos[0], o.pos[2]);
      if (!Number.isFinite(mnx)) return;
      const span = Math.max(mxx - mnx, mxz - mnz, 2) + 1, s = S / span, ox = (mnx + mxx) / 2 - span / 2, oz = (mnz + mxz) / 2 - span / 2;
      const img = new OffscreenCanvas(S, S), g = img.getContext("2d"), cs = Math.max(1, CELL * s * 1.05);
      g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(0, 0, S, S);
      g.fillStyle = "rgba(120,170,255,.35)";
      for (const [x, z] of floor) g.fillRect((x - CELL / 2 - ox) * s, (z - CELL / 2 - oz) * s, cs, cs);
      g.fillStyle = "rgba(255,255,255,.85)";
      for (const [x, z] of obst) g.fillRect((x - CELL / 2 - ox) * s, (z - CELL / 2 - oz) * s, cs, cs);
      g.strokeStyle = "rgba(250,204,21,.6)"; g.lineWidth = 1.5 * u; g.beginPath();
      track.forEach(([x, z], i) => (i ? g.lineTo((x - ox) * s, (z - oz) * s) : g.moveTo((x - ox) * s, (z - oz) * s))); g.stroke();
      if (opts.showSubmaps) { // サブマップの原点と、つながり（姿勢グラフのループ閉じは緑）
        g.fillStyle = "#f472b6";
        for (const sm of this.allSubs()) { const p = this.pose(sm); g.fillStyle = sm.prior ? "#a78bfa" : "#f472b6"; g.beginPath(); g.arc((p.x - ox) * s, (p.z - oz) * s, 2.5 * u, 0, 7); g.fill(); }
        for (const [a, b] of opts.loops || []) {
          const pa = this.pose(this.subs[a]), pb = this.pose(this.subs[b]);
          g.strokeStyle = "#22c55e"; g.lineWidth = u; g.beginPath(); g.moveTo((pa.x - ox) * s, (pa.z - oz) * s); g.lineTo((pb.x - ox) * s, (pb.z - oz) * s); g.stroke();
        }
      }
      this.cache = { t: now, S, s, ox, oz, img, version: this.version };
    }
    const { s, ox, oz } = this.cache, X = (x) => x0 + (x - ox) * s, Z = (z) => y0 + (z - oz) * s;
    ctx.drawImage(this.cache.img, x0, y0);
    ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = u; ctx.strokeRect(x0, y0, S, S);
    ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, S, S); ctx.clip();
    ctx.font = `600 ${Math.round(11 * u)}px system-ui, sans-serif`; ctx.textBaseline = "middle";
    for (const o of objs) {
      ctx.beginPath(); ctx.arc(X(o.pos[0]), Z(o.pos[2]), (o.collected ? 4 : 2.5) * u, 0, Math.PI * 2);
      ctx.fillStyle = o.collected ? "#22c55e" : "#fff"; ctx.fill();
      if (o.collected) { ctx.fillStyle = "#fff"; ctx.fillText(o.label, X(o.pos[0]) + 5 * u, Z(o.pos[2])); }
    }
    if (opts.cur) { // 自分: 位置と、向いている方（カメラの −z を上から見た向き）
      const M = opts.cur, fx = -M[8], fz = -M[10], fl = Math.hypot(fx, fz) || 1, ax = fx / fl, az = fz / fl, L = 12 * u, px = X(M[12]), pz = Z(M[14]);
      ctx.fillStyle = "#facc15"; ctx.beginPath();
      ctx.moveTo(px + ax * L, pz + az * L); ctx.lineTo(px - az * L * 0.45, pz + ax * L * 0.45); ctx.lineTo(px + az * L * 0.45, pz - ax * L * 0.45);
      ctx.closePath(); ctx.fill();
    }
    ctx.restore();
  }

  // 保存（IndexedDB にそのまま入る形。ボクセルは Float32Array をつなげる）と読み込み。前回の地図を重ねていれば、それも今の座標で一緒に
  // 保存する（各サブマップの姿勢は補正込みの今の姿勢にし、補正は 0 にする）
  serialize() {
    return {
      v: 1, floorY: this.floorY, yHist: this.yHist.slice(),
      subs: this.allSubs().map((s) => ({ id: s.id, anchor: this.pose(s), corr: P4(), t0: s.t0, keys: Float64Array.from(s.grid.keys()), data: concat(s.grid) })),
    };
  }
  static deserialize(o) {
    const m = new OccMap();
    m.floorY = o.floorY; m.yHist = Uint32Array.from(o.yHist);
    for (const x of o.subs) {
      const s = { id: x.id, anchor: x.anchor, corr: x.corr, t0: x.t0, grid: new Map(), n: 0 };
      x.keys.forEach((k, i) => s.grid.set(k, x.data.slice(i * YB, (i + 1) * YB)));
      m.subs.push(s);
    }
    return m;
  }
}
function concat(grid) {
  const out = new Float32Array(grid.size * YB);
  let i = 0;
  for (const c of grid.values()) out.set(c, (i++) * YB);
  return out;
}
