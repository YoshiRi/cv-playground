// 複数物体追跡（ByteTrack / BoT-SORT）。Ultralytics の実装（ultralytics/trackers、2026-09 の main）を JS に移したもの。
// 検出結果（boxes）をフレームごとに渡すと、ID 付きの結果を返す。ブラウザ実行・サーバー実行のどちらの検出にも使える。
//
//   const t = new Tracker("bytetrack");   // または "botsort"
//   const tracked = t.update(items);      // items: [{label, score, box: [x1, y1, x2, y2], ...}]
//                                         // → [{...検出の項目, id, box: カルマンフィルタで整えた枠}]
//
// BoT-SORT の ReID（見た目の特徴での対応付け）は with_reid: true と、update(items, feats) に検出ごとの特徴を渡すと有効。
// 特徴の計算（人物の切り出しと ReID モデル）は呼び出し側で行う（ここでは受け取った特徴だけを使う）。
//
// Ultralytics との違い:
// - BoT-SORT のカメラ移動の補正（GMC、既定は疎な光学フロー）は無い（固定カメラなら影響は小さい）
// - 割り当ては lap.lapjv（extend_cost, cost_limit）と同じ問題を、拡張したコスト行列のハンガリアン法で解く

export const TRACKER_DEFAULTS = {
  track_high_thresh: 0.25, // 1段目の対応付けに使う検出の下限
  track_low_thresh: 0.1,   // 2段目（低スコアの検出で見失いを救う）の下限
  new_track_thresh: 0.25,  // 新しい ID を作る検出の下限
  track_buffer: 30,        // 見失ってから ID を捨てるまでのフレーム数
  match_thresh: 0.8,       // 1段目の対応付けのコスト上限
  fuse_score: true,        // IoU に検出スコアを掛けて対応付ける
  // 以下は BoT-SORT の ReID 用
  with_reid: false,
  proximity_thresh: 0.5,   // IoU がこれ未満の組には見た目の距離を使わない
  appearance_thresh: 0.8,  // 見た目の距離（(1−cos)/2）が 1−これ を超える組は使わない（cos 類似度 0.6 未満）
};

const State = { New: 0, Tracked: 1, Lost: 2, Removed: 3 };

// ---------- カルマンフィルタ（等速モデル、状態 8 次元） ----------

const W_POS = 1 / 20, W_VEL = 1 / 160;

// kind "xyah": 中心 x, y, 縦横比, 高さ（ByteTrack）/ "xywh": 中心 x, y, 幅, 高さ（BoT-SORT）
function kfStd(kind, m, which) {
  if (kind === "xyah") {
    const h = m[3];
    if (which === "init") return [2 * W_POS * h, 2 * W_POS * h, 1e-2, 2 * W_POS * h, 10 * W_VEL * h, 10 * W_VEL * h, 1e-5, 10 * W_VEL * h];
    if (which === "motion") return [W_POS * h, W_POS * h, 1e-2, W_POS * h, W_VEL * h, W_VEL * h, 1e-5, W_VEL * h];
    return [W_POS * h, W_POS * h, 1e-1, W_POS * h]; // 観測ノイズ
  }
  const w = m[2], h = m[3];
  if (which === "init") return [2 * W_POS * w, 2 * W_POS * h, 2 * W_POS * w, 2 * W_POS * h, 10 * W_VEL * w, 10 * W_VEL * h, 10 * W_VEL * w, 10 * W_VEL * h];
  if (which === "motion") return [W_POS * w, W_POS * h, W_POS * w, W_POS * h, W_VEL * w, W_VEL * h, W_VEL * w, W_VEL * h];
  return [W_POS * w, W_POS * h, W_POS * w, W_POS * h];
}

const zeros = (r, c) => Array.from({ length: r }, () => new Array(c).fill(0));
const diag = (v) => { const m = zeros(v.length, v.length); v.forEach((x, i) => { m[i][i] = x * x; }); return m; };

function kfInitiate(kind, z) {
  return { mean: [...z, 0, 0, 0, 0], cov: diag(kfStd(kind, z, "init")) };
}

// x' = F x（位置 += 速度）、P' = F P Fᵀ + Q
function kfPredict(kind, mean, cov) {
  const Q = kfStd(kind, mean, "motion");
  const m = mean.map((v, i) => (i < 4 ? v + mean[i + 4] : v));
  const FP = cov.map((row, i) => (i < 4 ? row.map((v, j) => v + cov[i + 4][j]) : row.slice()));
  const P = FP.map((row) => row.map((v, j) => (j < 4 ? v + row[j + 4] : v)));
  for (let i = 0; i < 8; i++) P[i][i] += Q[i] * Q[i];
  return { mean: m, cov: P };
}

function inv4(a) {
  const n = 4, m = a.map((row, i) => [...row, ...[0, 0, 0, 0].map((_, j) => (i === j ? 1 : 0))]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(m[r][c]) > Math.abs(m[p][c])) p = r;
    [m[c], m[p]] = [m[p], m[c]];
    const d = m[c][c];
    for (let j = 0; j < 2 * n; j++) m[c][j] /= d;
    for (let r = 0; r < n; r++) if (r !== c) { const f = m[r][c]; for (let j = 0; j < 2 * n; j++) m[r][j] -= f * m[c][j]; }
  }
  return m.map((row) => row.slice(n));
}

// S = H P Hᵀ + R、K = P Hᵀ S⁻¹、x' = x + K (z − H x)、P' = P − K S Kᵀ
function kfUpdate(kind, mean, cov, z) {
  const R = kfStd(kind, mean, "obs");
  const S = [0, 1, 2, 3].map((i) => [0, 1, 2, 3].map((j) => cov[i][j] + (i === j ? R[i] * R[i] : 0)));
  const Si = inv4(S);
  const K = cov.map((row) => [0, 1, 2, 3].map((j) => row[0] * Si[0][j] + row[1] * Si[1][j] + row[2] * Si[2][j] + row[3] * Si[3][j]));
  const y = z.map((v, i) => v - mean[i]);
  const m = mean.map((v, i) => v + K[i][0] * y[0] + K[i][1] * y[1] + K[i][2] * y[2] + K[i][3] * y[3]);
  const KS = K.map((row) => [0, 1, 2, 3].map((j) => row[0] * S[0][j] + row[1] * S[1][j] + row[2] * S[2][j] + row[3] * S[3][j]));
  const P = cov.map((row, i) => row.map((v, j) => v - (KS[i][0] * K[j][0] + KS[i][1] * K[j][1] + KS[i][2] * K[j][2] + KS[i][3] * K[j][3])));
  return { mean: m, cov: P };
}

// ---------- 追跡対象 1つ ----------

// 特徴は L2 正規化し、平滑化した特徴 = 0.9 × 前回 + 0.1 × 今回（Ultralytics の smooth_feature）
function normalize(f) {
  let n = 0;
  for (const v of f) n += v * v;
  n = Math.sqrt(n);
  return n < 1e-12 ? null : Float32Array.from(f, (v) => v / n);
}

class STrack {
  constructor(det, kind, feat = null) {
    this.currFeat = null;
    this.smoothFeat = null;
    if (feat) this.updateFeatures(feat);
    const [x1, y1, x2, y2] = det.box;
    this._tlwh = [x1, y1, x2 - x1, y2 - y1];
    this.kind = kind;
    this.det = det;
    this.score = det.score;
    this.mean = null;
    this.state = State.New;
    this.isActivated = false;
    this.trackId = 0;
    this.frameId = 0;
    this.startFrame = 0;
  }

  get endFrame() { return this.frameId; }

  updateFeatures(feat) {
    const f = normalize(feat);
    if (!f) return;
    this.currFeat = f;
    if (!this.smoothFeat) { this.smoothFeat = f.slice(); return; }
    this.smoothFeat = normalize(this.smoothFeat.map((v, i) => 0.9 * v + 0.1 * f[i]));
  }

  get tlwh() {
    if (!this.mean) return this._tlwh.slice();
    const [x, y, a, h] = this.mean;
    const w = this.kind === "xyah" ? a * h : a;
    return [x - w / 2, y - h / 2, w, h];
  }

  get xyxy() { const [x, y, w, h] = this.tlwh; return [x, y, x + w, y + h]; }

  measure(tlwh) {
    const [x, y, w, h] = tlwh;
    return this.kind === "xyah" ? [x + w / 2, y + h / 2, w / h, h] : [x + w / 2, y + h / 2, w, h];
  }

  predict() {
    const m = this.mean.slice();
    if (this.state !== State.Tracked) { m[7] = 0; if (this.kind === "xywh") m[6] = 0; } // 見失い中は大きさの変化を止める
    ({ mean: this.mean, cov: this.cov } = kfPredict(this.kind, m, this.cov));
  }

  activate(frameId, nextId) {
    this.trackId = nextId();
    ({ mean: this.mean, cov: this.cov } = kfInitiate(this.kind, this.measure(this._tlwh)));
    this.state = State.Tracked;
    if (frameId === 1) this.isActivated = true; // 2フレーム目以降の新しい対象は、次のフレームで対応が付いてから確定
    this.frameId = this.startFrame = frameId;
  }

  update(t, frameId) {
    if (t.currFeat) this.updateFeatures(t.currFeat);
    this.frameId = frameId;
    ({ mean: this.mean, cov: this.cov } = kfUpdate(this.kind, this.mean, this.cov, this.measure(t.tlwh)));
    this.state = State.Tracked;
    this.isActivated = true;
    this.score = t.score;
    this.det = t.det;
  }

  reActivate(t, frameId) { this.update(t, frameId); }
}

// ---------- 対応付け ----------

function iou(a, b) {
  const w = Math.min(a[2], b[2]) - Math.max(a[0], b[0]), h = Math.min(a[3], b[3]) - Math.max(a[1], b[1]);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter);
}

const iouDistance = (as, bs) => as.map((a) => bs.map((b) => 1 - iou(a.xyxy, b.xyxy)));
const fuseScore = (cost, dets) => cost.map((row) => row.map((c, j) => 1 - (1 - c) * dets[j].score));

// 見た目の距離 = max(0, 1 − cos)。特徴の無い組は 2（Ultralytics の embedding_distance）
const embeddingDistance = (tracks, dets) => tracks.map((t) => dets.map((d) => {
  if (!t.smoothFeat || !d.currFeat) return 2;
  let dot = 0;
  for (let i = 0; i < d.currFeat.length; i++) dot += t.smoothFeat[i] * d.currFeat[i];
  return Math.max(0, 1 - dot);
}));

// 正方行列の最小コスト割り当て（ハンガリアン法、O(n³)）。返り値は行 i に割り当てた列
function hungarian(C) {
  const n = C.length, INF = 1e18, u = new Array(n + 1).fill(0), v = new Array(n + 1).fill(0), p = new Array(n + 1).fill(0), way = new Array(n + 1).fill(0);
  for (let i = 1; i <= n; i++) {
    p[0] = i;
    let j0 = 0;
    const minv = new Array(n + 1).fill(INF), used = new Array(n + 1).fill(false);
    do {
      used[j0] = true;
      const i0 = p[j0];
      let delta = INF, j1 = 0;
      for (let j = 1; j <= n; j++) {
        if (used[j]) continue;
        const cur = C[i0 - 1][j - 1] - u[i0] - v[j];
        if (cur < minv[j]) { minv[j] = cur; way[j] = j0; }
        if (minv[j] < delta) { delta = minv[j]; j1 = j; }
      }
      for (let j = 0; j <= n; j++) { if (used[j]) { u[p[j]] += delta; v[j] -= delta; } else minv[j] -= delta; }
      j0 = j1;
    } while (p[j0] !== 0);
    do { const j1 = way[j0]; p[j0] = p[j1]; j0 = j1; } while (j0);
  }
  const rowToCol = new Array(n);
  for (let j = 1; j <= n; j++) if (p[j]) rowToCol[p[j] - 1] = j - 1;
  return rowToCol;
}

// lap.lapjv(cost, extend_cost=True, cost_limit=thresh) と同じ: 行と列それぞれに「対応なし」の選択肢（コスト thresh/2）を足した
// (n+m) 次の正方行列で最小化する。thresh を超えるコストの組は選ばれない
// cost は n×m（n=0 の時は列数が分からないので m も渡す）
function linearAssignment(cost, thresh, m) {
  const n = cost.length;
  if (!n || !m) return { matches: [], uA: [...Array(n).keys()], uB: [...Array(m).keys()] };
  const N = n + m, BIG = 1e9, half = thresh / 2;
  const C = Array.from({ length: N }, (_, i) => Array.from({ length: N }, (_, j) => {
    if (i < n && j < m) return cost[i][j] > thresh ? BIG : cost[i][j];
    if (i < n) return j - m === i ? half : BIG;   // 行 i を対応なしにする
    if (j < m) return i - n === j ? half : BIG;   // 列 j を対応なしにする
    return 0;
  }));
  const r2c = hungarian(C), matches = [], matchedB = new Set();
  for (let i = 0; i < n; i++) if (r2c[i] < m && cost[i][r2c[i]] <= thresh) { matches.push([i, r2c[i]]); matchedB.add(r2c[i]); }
  const matchedA = new Set(matches.map(([i]) => i));
  return { matches, uA: [...Array(n).keys()].filter((i) => !matchedA.has(i)), uB: [...Array(m).keys()].filter((j) => !matchedB.has(j)) };
}

const joint = (a, b) => { const ids = new Set(a.map((t) => t.trackId)); return [...a, ...b.filter((t) => !ids.has(t.trackId))]; };
const sub = (a, b) => { const ids = new Set(b.map((t) => t.trackId)); return a.filter((t) => !ids.has(t.trackId)); };

// 追跡中と見失い中で枠がほぼ同じ（IoU 距離 < 0.15）なら、長く続いている方だけ残す
function removeDuplicates(a, b) {
  const dupA = new Set(), dupB = new Set();
  iouDistance(a, b).forEach((row, p) => row.forEach((c, q) => {
    if (c >= 0.15) return;
    if (a[p].frameId - a[p].startFrame > b[q].frameId - b[q].startFrame) dupB.add(q); else dupA.add(p);
  }));
  return [a.filter((_, i) => !dupA.has(i)), b.filter((_, i) => !dupB.has(i))];
}

// ---------- 追跡器 ----------

export class Tracker {
  constructor(type = "bytetrack", args = {}) {
    this.type = type;
    this.kind = type === "botsort" ? "xywh" : "xyah";
    this.args = { ...TRACKER_DEFAULTS, ...args };
    this.reid = type === "botsort" && this.args.with_reid; // Ultralytics でも ReID は BoT-SORT だけ
    this.reset();
  }

  reset() {
    this.tracked = []; this.lost = []; this.removed = [];
    this.frameId = 0; this.count = 0;
    this.trails = new Map(); // ID → 中心の履歴（描画用）
  }

  nextId = () => ++this.count;

  // 1段目と未確定の対象の対応付けに使う距離。ReID ありなら、近く（IoU ≥ proximity）て見た目も似ている組は見た目の距離も使う
  getDists(tracks, dets) {
    const A = this.args, iouD = iouDistance(tracks, dets);
    let d = A.fuse_score ? fuseScore(iouD, dets) : iouD;
    if (this.reid) {
      const emb = embeddingDistance(tracks, dets);
      d = d.map((row, i) => row.map((c, j) => {
        let e = emb[i][j] / 2;
        if (e > 1 - A.appearance_thresh || iouD[i][j] > 1 - A.proximity_thresh) e = 1;
        return Math.min(c, e);
      }));
    }
    return d;
  }

  // feats: items と同じ順の特徴（Float32Array など、無い検出は null）。ReID を使う時だけ渡す
  update(items, feats = null) {
    const A = this.args, fid = ++this.frameId;
    const activated = [], refind = [], lostNow = [], removedNow = [];
    const valid = items.map((d, i) => [d, this.reid && feats ? feats[i] : null]).filter(([d]) => d.box[2] > d.box[0] && d.box[3] > d.box[1]);
    let detections = valid.filter(([d]) => d.score >= A.track_high_thresh).map(([d, f]) => new STrack(d, this.kind, f));
    const second = valid.filter(([d]) => d.score > A.track_low_thresh && d.score < A.track_high_thresh).map(([d, f]) => new STrack(d, this.kind, f));

    const unconfirmed = this.tracked.filter((t) => !t.isActivated);
    const trackedOk = this.tracked.filter((t) => t.isActivated);
    const pool = joint(trackedOk, this.lost);
    pool.forEach((t) => t.predict());

    const apply = (t, d) => {
      if (t.state === State.Tracked) { t.update(d, fid); activated.push(t); } else { t.reActivate(d, fid); refind.push(t); }
    };
    // 1段目: 高スコアの検出と、追跡中・見失い中の対象
    let r = linearAssignment(this.getDists(pool, detections), A.match_thresh, detections.length);
    r.matches.forEach(([i, j]) => apply(pool[i], detections[j]));
    // 2段目: 1段目で残った追跡中の対象と、低スコアの検出（隠れかけの物体を救う）
    const rTracked = r.uA.map((i) => pool[i]).filter((t) => t.state === State.Tracked);
    let uTrack = [...rTracked.keys()];
    if (rTracked.length && second.length) {
      const r2 = linearAssignment(iouDistance(rTracked, second), 0.5, second.length);
      r2.matches.forEach(([i, j]) => apply(rTracked[i], second[j]));
      uTrack = r2.uA;
    }
    for (const i of uTrack) { const t = rTracked[i]; if (t.state !== State.Lost) { t.state = State.Lost; lostNow.push(t); } }
    // 前のフレームで生まれたばかりの（未確定の）対象
    detections = r.uB.map((j) => detections[j]);
    let uDet = [...detections.keys()];
    if (unconfirmed.length) {
      const r3 = linearAssignment(this.getDists(unconfirmed, detections), 0.7, detections.length);
      r3.matches.forEach(([i, j]) => { unconfirmed[i].update(detections[j], fid); activated.push(unconfirmed[i]); });
      r3.uA.forEach((i) => { unconfirmed[i].state = State.Removed; removedNow.push(unconfirmed[i]); });
      uDet = r3.uB;
    }
    // 残った高スコアの検出から新しい対象を作る
    for (const j of uDet) {
      const t = detections[j];
      if (t.score < A.new_track_thresh) continue;
      t.activate(fid, this.nextId);
      activated.push(t);
    }
    // 長く見失った対象を捨てる
    for (const t of this.lost) if (fid - t.endFrame > A.track_buffer) { t.state = State.Removed; removedNow.push(t); }

    this.tracked = joint(joint(this.tracked.filter((t) => t.state === State.Tracked), activated), refind);
    // Ultralytics（元の ByteTrack も）と同じ順序: 見失い一覧から引くのは前のフレームまでに捨てた対象で、
    // このフレームで捨てた対象は次のフレームで引かれる
    this.lost = sub(sub(this.lost, this.tracked).concat(lostNow), this.removed);
    [this.tracked, this.lost] = removeDuplicates(this.tracked, this.lost);
    this.removed = this.removed.concat(removedNow).slice(-1000);

    const out = [];
    for (const t of this.tracked) {
      if (!t.isActivated) continue;
      const box = t.xyxy;
      const trail = this.trails.get(t.trackId) || [];
      trail.push([(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]);
      if (trail.length > 40) trail.shift();
      this.trails.set(t.trackId, trail);
      out.push({ ...t.det, id: t.trackId, score: t.score, box, trail: trail.slice() });
    }
    for (const t of removedNow) this.trails.delete(t.trackId);
    return out;
  }
}
