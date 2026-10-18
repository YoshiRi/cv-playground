// AR の地図づくりの流れ（WebXR に依存しない）。ライブ（xr.js）はセンサーの値をここに渡し、同時に記録（xr_log.js）に書く。
// 再生（xr_replay.js）は記録を読んで同じ関数に渡す。
//   姿勢 → 深度（地図に積む。直前の検出のうち動く物の枠の中は当たりにしない）→ 検出（物を集める）
//   サブマップ（アンカー）を足す・アンカーが動く → 地図の座標が直る。ARCore がアンカーを動かしたら、姿勢グラフも解き直す
//     （補正 corr は解いた時のアンカーに対するものなので、そのまま重ねると ARCore の補正と二重になる）
//   2 秒ごと（記録の時刻で）に姿勢グラフ: 書き終わったサブマップを近くの古いものと照合し、ループ閉じが増えたら解き直す
//   照合は重い（スマホで数百 ms）ので、ライブでは opts.run（xr_worker.js を呼び、{ r, ms } を返す）で別のスレッドに出し、結果が来たら足す
//   （XR のフレームを止めると画面がちらつく）。再生では opts.run を渡さず、その場で回す
//
// 再生は 2 通り:
//   作り直し（既定）: 記録した入力から、照合・姿勢グラフをその場で回し直す（設定や作りを変えて比べる用）。ライブとは、照合の結果が
//     届いた順番と、前回の地図への重ね合わせ（前回の地図は記録に入らない）の分だけ違う
//   忠実（opts.faithful）: 照合を回さず、ライブで足したループ閉じ（loop の記録）をその位置で足す。ライブの地図をそのまま作り直す
import { p4Inv } from "./xr_geom.js";
import { PoseGraph, matchJob } from "./xr_graph.js";
import { packDepth, roundM, unpackDepth } from "./xr_log.js";
import { OccMap } from "./xr_map.js";
import { ObjectStore, dynBoxes } from "./xr_objects.js";
import { globalMatch } from "./xr_reloc.js";

const r1 = (v) => Math.round(v * 10) / 10, r3 = (v) => Math.round(v * 1000) / 1000;
const GRAPH_EVERY_MS = 2000, RELOC_EVERY_MS = 3000;
// 検出は記録と同じ丸めをしてから使う（ライブと再生で同じ値にする）
const roundDet = (b) => ({ label: b.label, score: r3(b.score), v: b.v.map(r3), depth: b.depth == null ? null : r3(b.depth), world: b.world.map((p) => p.map(r3)), center: b.center.map(r3) });

export class XrPipeline {
  // opts: { log（LogWriter）, graph（姿勢グラフを使うか）, dynamic（動く物を地図から除くか）, collectP, run（別スレッドの照合）, faithful（再生）}
  constructor(opts = {}) {
    this.opts = { graph: true, dynamic: true, collectP: 0.9, ...opts };
    this.log = opts.log || null;
    this.map = new OccMap(); this.objects = new ObjectStore({ collectP: this.opts.collectP }); this.graph = new PoseGraph();
    this.M = null; this.P = null; this.t = 0; this.dets = []; this.lastGraph = 0; this.stats = { frames: 0, depth: 0, dets: 0 };
    this.run = opts.run || null; this.gen = 0; // gen: 地図を消したら、出していた照合の結果を捨てる
    this.pending = new Set(); this.closed = false; // 出している照合（終わる時に待つ）
  }
  w(rec) { if (!this.closed) this.log?.write(rec); }
  meta(o) { this.w({ k: "meta", v: 2, settings: { graph: this.opts.graph, dynamic: this.opts.dynamic, collectP: this.opts.collectP }, ...o }); }
  onProj(P) {
    const p = roundM(P);
    if (this.P && p.every((v, i) => v === this.P[i])) return;
    this.P = p; this.w({ k: "P", P: p });
  }
  onPose(t, M) { this.t = r1(t); this.M = roundM(M); this.stats.frames++; this.w({ k: "f", t: this.t, M: this.M }); }
  // 別スレッドに出す（出している間は pending に持ち、終わる時に待つ）
  heavy(type, args) {
    const p = this.run(type, args);
    this.pending.add(p);
    p.then(() => this.pending.delete(p), () => this.pending.delete(p));
    return p;
  }
  // 地図に積む深度（直前の onPose の姿勢で）
  onDepth(depth) {
    if (this.closed) return;
    this.w({ k: "d", t: this.t, depth: packDepth(depth) });
    this.stats.depth++;
    const dyn = this.opts.dynamic ? dynBoxes(this.dets, this.P, this.M) : [];
    this.map.integrate({ t: this.t, M: this.M, P: this.P, depth, dyn });
    if (this.opts.graph && this.t - this.lastGraph > GRAPH_EVERY_MS) {
      this.lastGraph = this.t;
      // ARCore がアンカーを動かした → 今のアンカー同士の相対姿勢で解き直す（古い補正を重ねない）
      if (this.map.anchorsMoved && this.graph.loops.length) { this.map.anchorsMoved = false; this.graph.optimize(this.map); }
      if (this.opts.faithful) { /* 照合は回さず、記録した loop を足す（apply） */ }
      else if (!this.run) for (const job of this.graph.jobs(this.map)) this.addLoop(job, matchJob(job));
      else {
        const gen = this.gen;
        for (const job of this.graph.jobs(this.map)) this.heavy("match", job).then(({ r }) => { if (gen === this.gen && !this.closed) this.addLoop(job, r); }, () => {});
      }
    }
    this.tryReloc();
  }
  // ループ閉じを足して解き直す（足したものは記録し、忠実な再生で同じ所に足せるようにする）
  addLoop(job, m) {
    if (!m || !this.graph.add(this.map, job, m)) return;
    this.w({ k: "loop", t: this.t, j: job.j, i: job.i, z: m.rel, score: r3(m.score) });
  }

  // ---------- 前回の地図 ----------
  // saved = save() の中身。読み込んだら、今の地図に障害物がたまるたび（3 秒ごと）前回の地図と照合し、合ったら重ねる
  loadPrior(saved) {
    this.prior = { saved, map: OccMap.deserialize(saved.map) };
    this.prior.ref = this.prior.map.cellsWorld();
    this.reloc = { state: "waiting", tries: 0, lastT: 0, last: null, busy: false, stale: 0 };
    if (this.run) this.heavy("setRef", this.prior.ref);
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
    const gen = this.gen, pv = this.map.poseVersion;
    this.heavy("reloc", cur).then(({ r, ms }) => {
      R.busy = false;
      if (gen !== this.gen || this.closed) return;
      // 照合している間に地図の姿勢が変わった（アンカー・姿勢グラフ）→ その結果は今の地図に合わないので捨て、すぐ試し直す
      if (pv !== this.map.poseVersion) { R.lastT = 0; R.stale++; return; }
      this.applyReloc(r, ms);
    }, () => { R.busy = false; });
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
  // 地図を消す（出していた照合の結果は捨てる。記録にも残し、再生でも同じ所で消す）
  resetMap() {
    this.w({ k: "reset", t: this.t });
    this.gen++; this.map.reset(); this.graph.reset(); this.objects.reset();
    if (this.reloc) Object.assign(this.reloc, { state: "waiting", busy: false, lastT: 0 });
  }
  // 保存する中身（地図と集めた物。前回の地図を重ねていれば一緒に）
  save(extra = {}) {
    return { v: 1, date: new Date().toISOString(), ...extra, map: this.map.serialize(), objs: this.objects.serialize(this.map) };
  }
  // 検出の結果（placeBox で 3D にした後）。t は読み戻したフレームの時刻。thumb(b) は切り抜き（再生では無い）。新しく集めた物を返す
  onDetections(t, list, thumb) {
    if (this.closed) return [];
    const items = list.map(roundDet);
    this.w({ k: "det", t: r1(t), items });
    this.stats.dets++;
    items.forEach((b, i) => { b.n = list[i].n; }); // 切り抜き用の枠（記録しない）
    this.dets = items;
    return this.objects.add(items, r1(t), this.map, thumb);
  }
  onSubmap(id, M) { if (this.closed) return; const m = roundM(M); this.w({ k: "sub", t: this.t, id, M: m }); this.map.addSubmap(id, m, this.t); }
  onAnchors(list) { if (this.closed) return; const l = list.map(([id, M]) => [id, roundM(M)]); this.w({ k: "a", t: this.t, list: l }); this.map.setAnchors(l); }

  // 終わり（ライブ）: 出している照合を待ち、最後のサブマップも照合してから、記録を閉じる。地図はこのあと保存する
  async close(t) {
    if (this.closed) return;
    await Promise.allSettled([...this.pending]);
    if (this.opts.graph) {
      for (const job of this.graph.jobs(this.map, true)) {
        if (!this.run) { this.addLoop(job, matchJob(job)); continue; }
        try { const { r } = await this.heavy("match", job); this.addLoop(job, r); } catch { /* 捨てる */ }
      }
    }
    this.w({ k: "end", t: r1(t) });
    this.closed = true;
  }

  // 再生: 記録 1 行を処理する（ライブと同じ関数に渡す）
  apply(r) {
    switch (r.k) {
      case "P": this.P = r.P; break;
      case "f": this.t = r.t; this.M = r.M; this.stats.frames++; break;
      case "d": this.onDepth(unpackDepth(r.depth)); break;
      case "det": this.onDetections(r.t, r.items.map((it) => ({ ...it })), null); break;
      case "sub": this.t = r.t; this.map.addSubmap(r.id, r.M, r.t); break;
      case "a": this.map.setAnchors(r.list); break;
      case "reset": this.gen++; this.map.reset(); this.graph.reset(); this.objects.reset(); break;
      case "loop": if (this.opts.faithful && this.opts.graph) { this.graph.tried.add(`${r.j}-${r.i}`); this.graph.add(this.map, { j: r.j, i: r.i }, { rel: r.z, score: r.score }); } break;
      default: break;
    }
  }
  // 再生の最後: ライブの close と同じく、最後のサブマップも照合する（忠実な再生では記録した loop がそれに当たる）
  finish() { if (this.opts.graph && !this.opts.faithful) for (const job of this.graph.jobs(this.map, true)) this.addLoop(job, matchJob(job)); }
}
