// AR の地図づくりの流れ（WebXR に依存しない）。ライブ（xr.js）はセンサーの値をここに渡し、同時に記録（xr_log.js）に書く。
// 再生（xr_replay.js）は記録を読んで同じ関数に渡す。どちらも同じ順に同じ処理をするので、再生した地図はライブと同じになる
//   姿勢 → 深度（地図に積む。直前の検出のうち動く物の枠の中は当たりにしない）→ 検出（物を集める）
//   サブマップ（アンカー）を足す・アンカーが動く → 地図の座標が直る
//   2 秒ごと（記録の時刻で）に姿勢グラフ: 書き終わったサブマップを近くの古いものと照合し、ループ閉じが増えたら解き直す
//   照合は重い（スマホで数百 ms）ので、ライブでは opts.run（xr_worker.js を呼び、{ r, ms } を返す）で別のスレッドに出し、結果が来たら足す
//   （XR のフレームを止めると画面がちらつく）。再生では opts.run を渡さず、その場で回す
import { p4Inv } from "./xr_geom.js";
import { PoseGraph } from "./xr_graph.js";
import { packDepth, roundM, unpackDepth } from "./xr_log.js";
import { OccMap } from "./xr_map.js";
import { ObjectStore, dynBoxes } from "./xr_objects.js";
import { globalMatch } from "./xr_reloc.js";

const r1 = (v) => Math.round(v * 10) / 10, r3 = (v) => Math.round(v * 1000) / 1000;
const GRAPH_EVERY_MS = 2000, RELOC_EVERY_MS = 3000;

export class XrPipeline {
  // opts: { log（LogWriter）, graph（姿勢グラフを使うか）, dynamic（動く物を地図から除くか）, collectP }
  constructor(opts = {}) {
    this.opts = { graph: true, dynamic: true, ...opts };
    this.log = opts.log || null;
    this.map = new OccMap(); this.objects = new ObjectStore({ collectP: opts.collectP || 0.9 }); this.graph = new PoseGraph();
    this.M = null; this.P = null; this.t = 0; this.dets = []; this.lastGraph = 0; this.stats = { frames: 0, depth: 0, dets: 0 };
    this.run = opts.run || null; this.gen = 0; // gen: 地図を消したら、出していた照合の結果を捨てる
  }
  w(rec) { this.log?.write(rec); }
  meta(o) { this.w({ k: "meta", v: 1, ...o }); }
  end(t) { this.w({ k: "end", t: r1(t) }); }
  onProj(P) {
    const p = roundM(P);
    if (this.P && p.every((v, i) => v === this.P[i])) return;
    this.P = p; this.w({ k: "P", P: p });
  }
  onPose(t, M) { this.t = r1(t); this.M = roundM(M); this.stats.frames++; this.w({ k: "f", t: this.t, M: this.M }); }
  // 地図に積む深度（直前の onPose の姿勢で）
  onDepth(depth) {
    this.w({ k: "d", t: this.t, depth: packDepth(depth) });
    this.stats.depth++;
    const dyn = this.opts.dynamic ? dynBoxes(this.dets, this.P, this.M) : [];
    this.map.integrate({ t: this.t, M: this.M, P: this.P, depth, dyn });
    if (this.opts.graph && this.t - this.lastGraph > GRAPH_EVERY_MS) {
      this.lastGraph = this.t;
      if (!this.run) this.graph.update(this.map);
      else { const gen = this.gen; for (const job of this.graph.jobs(this.map)) this.run("match", job).then(({ r }) => { if (gen === this.gen) this.graph.add(this.map, job, r); }).catch(() => {}); }
    }
    this.tryReloc();
  }

  // ---------- 前回の地図 ----------
  // saved = save() の中身。読み込んだら、今の地図に障害物がたまるたび（3 秒ごと）前回の地図と照合し、合ったら重ねる
  loadPrior(saved) {
    this.prior = { saved, map: OccMap.deserialize(saved.map) };
    this.prior.ref = this.prior.map.cellsWorld();
    this.reloc = { state: "waiting", tries: 0, lastT: 0, last: null, busy: false };
    this.run?.("setRef", this.prior.ref);
  }
  tryReloc() {
    const R = this.reloc;
    if (!this.prior || R.state === "done" || R.busy || this.map.floorY == null || this.t - R.lastT < RELOC_EVERY_MS) return;
    R.lastT = this.t;
    const cur = this.map.cellsWorld();
    if (cur.obst.length < 80) return;
    R.tries++;
    if (!this.run) { const t0 = performance.now(); this.applyReloc(globalMatch(cur, this.prior.ref), performance.now() - t0); return; }
    R.busy = true;
    const gen = this.gen;
    this.run("reloc", cur).then(({ r, ms }) => { R.busy = false; if (gen === this.gen) this.applyReloc(r, ms); }).catch(() => { R.busy = false; });
  }
  applyReloc(m, ms) {
    const R = this.reloc;
    R.last = m && { score: +m.score.toFixed(2), second: +m.second.toFixed(2), ms: Math.round(ms ?? 0) };
    if (!m?.ok || R.state === "done") return;
    // m.rel は 今 → 前回。前回 → 今 はその逆。高さは床の高さの差で合わせる
    const T = p4Inv(m.rel);
    T.y = this.map.floorY - this.prior.map.floorY;
    this.map.setPrior(this.prior.map, T);
    this.objects.addPrior(this.prior.saved.objs || [], this.prior.map.subs);
    R.state = "done"; R.T = T;
    this.w({ k: "reloc", t: this.t, T, score: R.last.score });
  }
  // 地図を消す（出していた照合の結果は捨てる）
  resetMap() { this.gen++; this.map.reset(); this.graph.reset(); this.objects.reset(); if (this.reloc) Object.assign(this.reloc, { state: "waiting", busy: false }); }
  // 保存する中身（地図と集めた物。前回の地図を重ねていれば一緒に）
  save(extra = {}) {
    return { v: 1, date: new Date().toISOString(), ...extra, map: this.map.serialize(), objs: this.objects.serialize(this.map) };
  }
  // 検出の結果（placeBox で 3D にした後）。t は読み戻したフレームの時刻。thumb(b) は切り抜き（再生では無い）。新しく集めた物を返す
  onDetections(t, list, thumb) {
    this.w({ k: "det", t: r1(t), items: list.map((b) => ({ label: b.label, score: r3(b.score), v: b.v.map(r3), depth: b.depth == null ? null : r3(b.depth), world: b.world.map((p) => p.map(r3)), center: b.center.map(r3) })) });
    this.stats.dets++;
    this.dets = list;
    return this.objects.add(list, r1(t), this.map, thumb);
  }
  onSubmap(id, M) { const m = roundM(M); this.w({ k: "sub", t: this.t, id, M: m }); this.map.addSubmap(id, m, this.t); }
  onAnchors(list) { const l = list.map(([id, M]) => [id, roundM(M)]); this.w({ k: "a", t: this.t, list: l }); this.map.setAnchors(l); }

  // 再生: 記録 1 行を処理する（ライブと同じ関数に渡す）
  apply(r) {
    switch (r.k) {
      case "P": this.P = r.P; break;
      case "f": this.t = r.t; this.M = r.M; this.stats.frames++; break;
      case "d": this.onDepth(unpackDepth(r.depth)); break;
      case "det": this.onDetections(r.t, r.items.map((it) => ({ ...it })), null); break;
      case "sub": this.t = r.t; this.map.addSubmap(r.id, r.M, r.t); break;
      case "a": this.map.setAnchors(r.list); break;
      default: break;
    }
  }
  // 最後にもう一度、姿勢グラフ（書き終わった最後のサブマップも照合する）
  finish() { if (this.opts.graph) this.graph.update(this.map, true); }
}
