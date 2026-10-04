// 応用: モデルの結果を受け取り、フレームをまたいだ状態を持って集計し、画面に重ねる（例: 物体を数える）。
// models.json の apps に {id, name, accepts: [結果の種類], params, hint} を書くと、その種類の結果を返すタブ（tasks[].result）の
// 「応用」欄にチェックが出て、選ぶと追跡・cascade のあとに順に呼ばれる。モデルの後処理（onnx_generic.js・adapters.py、
// models.json の post）とは別物で、モデルの結果の見せ方（renderers.js の KINDS）とも別に、集計の状態と表示だけを持つ
import { esc, labelColor, limbColor } from "./renderers.js";
import { SKELETON } from "./catalog.js";
import { Judges } from "./judge.js";
import { tx } from "./i18n.js";

// APPS[id] = {
//   create(opts)            → 状態（連続実行の開始時と、静止画の1回ごとに作り直す）。opts = { classes }（応用欄の値）
//   update(st, r, ctx)      → 1フレームごと。r.items を絞り込んでよい（絞った結果が枠の表示にも反映される）。ctx = { tracked }
//   draw(ctx2d, st, r, base) → canvas に重ねる（元画像と結果の枠は描画済み）
//   panel(st)               → 結果欄に足す HTML
//   summary(st)             → 実行履歴の「結果」欄に足す文字
// }
export const APPS = {
  // 物体カウント: 対象クラスごとに「今の数」と、追跡があれば「通算の数（見えた ID の数）」を出す
  count: {
    create: (opts) => ({
      classes: (opts.classes || "").split(",").map((s) => s.trim().toLowerCase()).filter(Boolean),
      hist: [],         // 直近のフレームの数（ラベル → 数）。今の数はこの中央値にして、1フレームの検出漏れでちらつかないようにする
      seen: new Map(),  // ラベル → 見えた追跡 ID の集合
      tracked: false,
    }),
    update(st, r, { tracked }) {
      if (r.kind !== "boxes") return;
      if (st.classes.length) r.items = r.items.filter((it) => st.classes.includes(it.label.toLowerCase()));
      const now = new Map();
      for (const it of r.items) now.set(it.label, (now.get(it.label) || 0) + 1);
      st.hist.push(now);
      if (st.hist.length > HIST) st.hist.shift();
      st.tracked = tracked;
      if (tracked) {
        for (const it of r.items) {
          if (it.id == null) continue;
          if (!st.seen.has(it.label)) st.seen.set(it.label, new Set());
          st.seen.get(it.label).add(it.id);
        }
      }
    },
    draw(ctx, st) {
      const rows = countRows(st);
      if (!rows.length) return;
      const size = Math.max(16, Math.round(ctx.canvas.width / 32)), pad = size * 0.5, lh = size * 1.35;
      ctx.font = `600 ${size}px system-ui, sans-serif`;
      const lines = rows.map(({ label, now, total }) => `${label}  ${now}${total != null ? tx("（通算 {n}）", { n: total }) : ""}`);
      const w = Math.max(...lines.map((s) => ctx.measureText(s).width)) + size * 1.2 + pad * 2;
      ctx.fillStyle = "rgba(0,0,0,.6)";
      ctx.fillRect(pad, pad, w, lh * lines.length + pad);
      lines.forEach((s, i) => {
        const y = pad + pad / 2 + lh * i + lh / 2;
        ctx.fillStyle = labelColor(rows[i].label);
        ctx.fillRect(pad * 2, y - size * 0.35, size * 0.7, size * 0.7);
        ctx.fillStyle = "#fff";
        ctx.textBaseline = "middle";
        ctx.fillText(s, pad * 2 + size * 1.2, y);
      });
    },
    panel(st) {
      const rows = countRows(st);
      if (!rows.length) return `<div class="sub muted">対象が見つからない${st.classes.length ? `（${esc(st.classes.join(", "))}）` : ""}</div>`;
      const chips = rows.map(({ label, now, total }) =>
        `<span class="chip"><i style="background:${labelColor(label)}"></i>${esc(label)} <b>${now}</b>${total != null ? ` <small>通算 ${total}</small>` : ""}</span>`).join("");
      const note = st.tracked ? `今の数は直近 ${HIST} フレームの中央値。通算は見えた追跡 ID の数（ID が切り替わると多めに出る）` : "追跡なし（今の数だけ。動画・カメラで追跡を選ぶと通算も出る）";
      return `<div class="sub">カウント</div><div class="chips">${chips}</div><div class="sub muted">${note}</div>`;
    },
    summary: (st) => countRows(st).map(({ label, now, total }) => `${label} ${total ?? now}`).join("・"),
  },
};

const HIST = 5;

// 表示する行: 指定したクラスは 0 でも出す。指定が無ければ、今見えているか一度でも見えたクラス
function countRows(st) {
  const labels = new Set(st.classes.length ? st.classes : [...st.hist.flatMap((m) => [...m.keys()]), ...st.seen.keys()]);
  // 指定したクラス名の大文字小文字は、モデルが返したラベルに合わせる
  const actual = new Map([...st.hist.flatMap((m) => [...m.keys()]), ...st.seen.keys()].map((l) => [l.toLowerCase(), l]));
  return [...labels].map((l) => actual.get(l.toLowerCase()) ?? l).map((label) => ({
    label,
    now: median(st.hist.map((m) => m.get(label) || 0)),
    total: st.tracked ? st.seen.get(label)?.size ?? 0 : null,
  }));
}

function median(a) {
  if (!a.length) return 0;
  const s = [...a].sort((x, y) => x - y);
  return s[Math.floor(s.length / 2)];
}

// 組み合わせのタブ（models.json の tasks[].combo）: 2つ以上のモデルを同じフレームで回し、結果を組み合わせる。
// combo.base のタブのモデル（画面の「モデル」）の結果が r、combo.with の役割ごとのモデルの結果が r.with[role]（boxes の items）。
// COMBOS[combo.app] = { combine(r) → r.items を組み直す（追跡・cascade の後）, panel(r), summary(r) }
const ORIENT = { front: "正面", "right-front": "斜め", "left-front": "斜め", "right-side": "横", "left-side": "横", "right-back": "後ろ", "left-back": "後ろ", back: "後ろ" };
const center = (b) => [(b[0] + b[2]) / 2, (b[1] + b[3]) / 2];
const inBox = ([x, y], b, m = 0) => { const w = b[2] - b[0], h = b[3] - b[1]; return x >= b[0] - w * m && x <= b[2] + w * m && y >= b[1] - h * m && y <= b[3] + h * m; };
const iou = (a, b) => { const w = Math.max(0, Math.min(a[2], b[2]) - Math.max(a[0], b[0])), h = Math.max(0, Math.min(a[3], b[3]) - Math.max(a[1], b[1])), i = w * h; return i / ((a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - i); };

// 深度の結果（depthRaw）から、元画像の座標 (x, y) の近さ（0＝その画像で一番遠い〜1＝一番近い）を引く
function nearAt(d, x, y) {
  const m = d.m, gx = Math.min(d.w - 1, Math.max(0, Math.floor(((x * m.sx + m.ox) * d.w) / m.iw)));
  const gy = Math.min(d.h - 1, Math.max(0, Math.floor(((y * m.sy + m.oy) * d.h) / m.ih)));
  return (d.data[gy * d.w + gx] - d.lo) / Math.max(d.hi - d.lo, 1e-9);
}
// 深度の結果（depthRaw、大きいほど近い値）から、枠の中央 6 割の 5×5 点の中央値と、画像全体（余白を除く）の中央値を引く
function rawOfBox(d, b) {
  const m = d.m, v = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const x = b[0] + (b[2] - b[0]) * (0.2 + 0.15 * i), y = b[1] + (b[3] - b[1]) * (0.2 + 0.15 * j);
    const gx = Math.min(d.w - 1, Math.max(0, Math.floor(((x * m.sx + m.ox) * d.w) / m.iw)));
    const gy = Math.min(d.h - 1, Math.max(0, Math.floor(((y * m.sy + m.oy) * d.h) / m.ih)));
    v.push(d.data[gy * d.w + gx]);
  }
  v.sort((p, q) => p - q);
  return v[12];
}
function rawMedian(d) {
  if (d.median != null) return d.median;
  const m = d.m, x0 = Math.floor((m.ox * d.w) / m.iw), x1 = Math.ceil(((m.iw - m.ox) * d.w) / m.iw), y0 = Math.floor((m.oy * d.h) / m.ih), y1 = Math.ceil(((m.ih - m.oy) * d.h) / m.ih), v = [];
  for (let y = y0; y < y1; y += 3) for (let x = x0; x < x1; x += 3) v.push(d.data[y * d.w + x]);
  v.sort((p, q) => p - q);
  return (d.median = v[v.length >> 1]);
}

// 接近・後退: 追跡の ID ごとに、距離の対数 x とその速さ v（1 秒あたり）を小さなカルマンフィルタで推定する。
// 観測は 2 つで、どちらも「距離の対数 + その物ごとの定数」: ROI（枠の大きさの逆数 → -log √面積、毎フレーム）と、
// 深度（その物の深度 ÷ 画像全体の深度の中央値 → log、深度を回したフレームだけ）。定数は各観測の最初の値で合わせる
const KF = new Map(); // ID → { x, v, P, t, c: {roi, depth}, cls, streak, cand }
const Q_ACC = 0.08; // 速さの揺らぎ（1 秒あたり。人や車は数秒はほぼ同じ速さで動く）
const R = { roi: 0.06 ** 2, depth: 0.1 ** 2 }; // 観測の誤差（対数）
// 距離が 1 秒に 18% 以上縮む・伸びる状態が 3 回続いたら切り替える。18% は、誤検出の揺れ（最大 16%）と、
// 歩き去る人・走り去る車（29〜46%）の間（人物・駐車場の動画での計測）
const V_TH = 0.18, STREAK = 3;
function kfUpdate(k, z, kind) {
  if (k.c[kind] == null) { k.c[kind] = z - k.x; return; } // 観測ごとの定数を最初に合わせる
  const y = z - k.c[kind] - k.x, S = k.P[0][0] + R[kind], K0 = k.P[0][0] / S, K1 = k.P[1][0] / S;
  k.x += K0 * y; k.v += K1 * y;
  const [p00, p01, p10, p11] = [k.P[0][0], k.P[0][1], k.P[1][0], k.P[1][1]];
  k.P = [[(1 - K0) * p00, (1 - K0) * p01], [p10 - K1 * p00, p11 - K1 * p01]];
}
function kfPredict(k, t) {
  const dt = Math.min(1, Math.max(0, (t - k.t) / 1000));
  k.t = t; k.x += k.v * dt;
  const [p00, p01, p10, p11] = [k.P[0][0], k.P[0][1], k.P[1][0], k.P[1][1]];
  k.P = [[p00 + dt * (p10 + p01) + dt * dt * p11, p01 + dt * p11], [p10 + dt * p11, p11 + Q_ACC * dt]];
}
const MOTION = { approach: { word: "接近", color: "#ef4444", arrow: "↓" }, recede: { word: "後退", color: "#3b82f6", arrow: "↑" }, steady: { word: "", color: "#9ca3af", arrow: "" } };
const SEEN = { approach: new Map(), recede: new Map() }; // クラス → 一度でも接近・後退と判定した ID

// 進む・止まる（物体検出＋深度＋場面）: ロボットのような簡易センサの判断。止まる条件（設定欄で選ぶ）のどれか 1 つでも当たれば止まる。
//   前に人・物: 画面の中央の帯（幅 40%）に枠の中心があり、枠の高さが画面の frontH 以上
//   近づいてくる: 接近・後退（このファイルの approach と同じカルマンフィルタ）で接近中、届くまで ttc 秒以内
//   前が近い: 中央下（x 30〜70%・y 40〜100%）で、深度が場面の中央値の 2 倍以上（距離が半分以下）の画素が nearP 以上
//   場面: 場面の役割（ゼロショット分類）の判定のどれかが「いいえ」側（通れない・危険・埋まっている など）
// 止まるにはすぐ（1 フレーム）、進むに戻るのは条件が GO_K フレーム続けて外れてから（止まる側に倒す）
const GO_K = 5, BAND = 0.4, NEAR_RATIO = 2;
const GS = { value: "進む", clear: 0, log: [], t0: performance.now(), judges: new Judges(), near: null, scene: [] };
function nearFraction(d) {
  const m = d.m, med = rawMedian(d), x0 = 0.3 * m.W, x1 = 0.7 * m.W, y0 = 0.4 * m.H, y1 = m.H;
  let n = 0, k = 0;
  for (let y = y0; y < y1; y += m.H / 60) for (let x = x0; x < x1; x += m.W / 60) {
    const gx = Math.min(d.w - 1, Math.max(0, Math.floor(((x * m.sx + m.ox) * d.w) / m.iw)));
    const gy = Math.min(d.h - 1, Math.max(0, Math.floor(((y * m.sy + m.oy) * d.h) / m.ih)));
    n++; if (d.data[gy * d.w + gx] >= NEAR_RATIO * med) k++;
  }
  return n ? k / n : 0;
}

// 「person ×2・car」のように数える
const countText = (ls) => Object.entries(ls.reduce((c, l) => ((c[l] = (c[l] || 0) + 1), c), {})).map(([l, n]) => (n > 1 ? `${l} ×${n}` : l)).join("・");

export const COMBOS = {
  gostop: {
    reset() { COMBOS.approach.reset(); Object.assign(GS, { value: "進む", clear: 0, log: [], t0: performance.now(), near: null, scene: [] }); GS.judges.reset(); },
    combine(r) {
      COMBOS.approach.combine(r); // 枠の色（接近は赤）と、ID ごとの速さ
      const o = r.uiOpts?.gostop || {}, reasons = [], conds = {};
      // 前に人・物
      const front = r.items.filter((it) => { const cx = (it.box[0] + it.box[2]) / 2; return Math.abs(cx - r.w / 2) <= (BAND / 2) * r.w && it.box[3] - it.box[1] >= (o.frontH ?? 0.4) * r.h; });
      conds.front = { on: o.front, hit: front.length > 0, text: countText(front.map((it) => it.label)) };
      // 近づいてくる
      const appr = r.items.map((it) => ({ it, k: it.id != null ? KF.get(it.id) : null })).filter(({ k }) => k?.cls === "approach").map(({ it, k }) => ({ it, ttc: -1 / k.v })).filter(({ ttc }) => ttc < (o.ttc ?? 3));
      conds.approach = { on: o.approach, hit: appr.length > 0, text: appr.map(({ it, ttc }) => `${it.label} あと ${ttc.toFixed(1)} 秒`).join("・") };
      // 前が近い（深度を回したフレームだけ計り直す）
      const dr = r.withResults?.depth;
      if (dr?.depthRaw && !dr.reused) GS.near = nearFraction(dr.depthRaw);
      conds.near = { on: o.near, hit: GS.near != null && GS.near >= (o.nearP ?? 0.3), text: GS.near != null ? `${(GS.near * 100).toFixed(0)}%` : "深度なし" };
      // 場面（ゼロショット分類を回したフレームだけ判定をならし直す）
      const sr = r.withResults?.scene;
      if (sr?.judgeDefs?.length && !sr.reused) GS.scene = GS.judges.update(sr.judgeDefs, sr.judgeLogit, true);
      const bad = GS.scene.filter((d) => d.decided === false);
      conds.scene = { on: o.scene, hit: bad.length > 0, text: GS.scene.map((d) => d.value).join("・") };
      const words = { front: (c) => `前に${c.text}`, approach: (c) => `${c.text}で接近中`, near: (c) => `前が近い（${c.text}）`, scene: (c) => c.text && bad.map((d) => d.value).join("・") };
      for (const [k, c] of Object.entries(conds)) if (c.on && c.hit) reasons.push(words[k](c));
      const now = performance.now(), prev = GS.value;
      if (reasons.length) { GS.value = "止まる"; GS.clear = 0; }
      else if (GS.value === "止まる" && ++GS.clear >= GO_K) GS.value = "進む";
      const changed = prev !== GS.value;
      if (changed) { GS.log.push({ t: (now - GS.t0) / 1000, value: GS.value, reason: reasons.join("・") }); if (GS.log.length > 50) GS.log.shift(); }
      r.gostop = { value: GS.value, reasons, conds, log: GS.log.slice(-6), scene: GS.scene,
        decisions: [{ name: "進む・止まる", value: GS.value, decided: GS.value === "進む", p: GS.value === "進む" ? 1 : 0, conf: 1, changed, reason: reasons.join("・") || "条件に当たらない" }, ...GS.scene] };
    },
    draw(ctx, r, b) {
      const g = r.gostop;
      if (!g) return;
      const sx = b.w / r.w, sy = b.h / r.h, size = Math.max(18, Math.round(b.w / 22));
      // 前の帯と、前が近いを見る範囲
      ctx.save(); ctx.setLineDash([8, 6]); ctx.lineWidth = Math.max(1.5, b.w / 500); ctx.strokeStyle = "rgba(255,255,255,.7)";
      for (const f of [0.5 - BAND / 2, 0.5 + BAND / 2]) { ctx.beginPath(); ctx.moveTo(f * b.w, 0); ctx.lineTo(f * b.w, b.h); ctx.stroke(); }
      ctx.strokeStyle = "rgba(250,204,21,.8)"; ctx.strokeRect(0.3 * b.w, 0.4 * b.h, 0.4 * b.w, 0.6 * b.h - 2);
      ctx.restore();
      const stop = g.value === "止まる", text = stop ? `${tx("止まる")}  ${g.reasons.map((x) => tx(x)).join(" · ")}` : tx("進む");
      ctx.save(); ctx.font = `700 ${size}px system-ui, sans-serif`;
      const w = Math.min(b.w - 16, ctx.measureText(text).width + size), x = (b.w - w) / 2;
      ctx.fillStyle = stop ? "rgba(220,38,38,.92)" : "rgba(22,163,74,.92)"; ctx.beginPath(); ctx.roundRect(x, 10, w, size * 1.5, size * 0.3); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.textBaseline = "middle"; ctx.textAlign = "center"; ctx.fillText(text, b.w / 2, 10 + size * 0.75, w - size * 0.5);
      ctx.restore();
    },
    panel: (r) => {
      const g = r.gostop;
      if (!g) return "";
      const names = { front: "前に人・物", approach: "近づいてくる", near: "前が近い", scene: "場面" };
      const rows = Object.entries(g.conds).map(([k, c]) => `<span class="chip"><i style="background:${!c.on ? "#cbd5e1" : c.hit ? "#ef4444" : "#22c55e"}"></i>${names[k]}${c.on ? "" : "（使わない）"} <small class="muted">${esc(c.text || "－")}</small></span>`).join("");
      return `<div class="sub"><b style="color:${g.value === "止まる" ? "#dc2626" : "#16a34a"}">${g.value}</b>${g.reasons.length ? `（${esc(g.reasons.join("・"))}）` : ""}。止まる条件（赤 = 当たっている）:</div><div class="chips">${rows}</div>`
        + (g.log.length ? `<div class="sub">切り替わり: ${g.log.map((l) => `${l.t.toFixed(1)}s ${l.value}${l.reason ? `（${esc(l.reason)}）` : ""}`).join(" ・ ")}</div>` : "")
        + `<div class="sub muted">ブラウザとスマホのカメラによる簡易の判断。見落としや遅れがあるので、安全に関わる用途には使わない</div>`;
    },
    summary: (r) => (r.gostop ? `${r.gostop.value}${r.gostop.reasons.length ? `（${r.gostop.reasons.join("・")}）` : ""}` : ""),
  },
  // 接近・後退（物体検出＋深度）: 追跡の ID ごとに ROI（枠の大きさ）と深度の変化から距離の変化の速さを推定し、
  // 接近（赤）・後退（青）・変化なし（灰）に分けて、クラスごとに接近した・遠ざかった ID の数（通算）を数える。
  // 接近中は距離 ÷ 縮む速さ（あと何秒で届くか）も出す。深度は相対値（DA3 は比が保たれるので深度 ÷ 画像全体の中央値を使う）
  approach: {
    reset() { KF.clear(); SEEN.approach.clear(); SEEN.recede.clear(); },
    combine(r) {
      const dr = r.withResults?.depth, d = dr?.depthRaw, t = performance.now(), live = r.items.some((it) => it.id != null);
      const counts = { approach: 0, recede: 0 };
      for (const it of r.items) {
        if (it.id == null) continue;
        let k = KF.get(it.id);
        const zRoi = -Math.log(Math.sqrt(Math.max(1, (it.box[2] - it.box[0]) * (it.box[3] - it.box[1]))));
        // 枠が画面の端に接している間は、物が画面から切れていて枠の大きさが距離を表さない（入ってくる時に「接近」に見える）ので、
        // ROI の観測を使わない（深度だけ）。端から離れた最初の観測で ROI の定数を合わせ直す
        const mx = r.w * 0.01, my = r.h * 0.01;
        const cut = it.box[0] <= mx || it.box[1] <= my || it.box[2] >= r.w - mx || it.box[3] >= r.h - my;
        if (!k) { k = { x: zRoi, v: 0, P: [[0.05, 0], [0, 0.5]], t, c: cut ? {} : { roi: 0 }, cls: "steady", streak: 0, cand: "steady", n: 0 }; KF.set(it.id, k); }
        else { kfPredict(k, t); if (cut) delete k.c.roi; else kfUpdate(k, zRoi, "roi"); }
        if (d && !dr.reused) { // 深度を回したフレームだけ（使い回した深度は同じ値なので入れない）
          const obj = rawOfBox(d, it.box), bg = rawMedian(d);
          if (obj > 0 && bg > 0) kfUpdate(k, Math.log(bg / obj), "depth"); // 大きいほど近い値なので、距離 ∝ 1 / 値
        }
        k.n++;
        // 判定: 8 フレーム以上追えていて、速さが V_TH を超えた時だけ。
        // 枠が画面の端に接している間（深度だけの時）は判定を変えない（端の深度は揺れやすい）
        if (!cut) {
          const want = k.n < 8 || Math.abs(k.v) < V_TH ? "steady" : k.v < 0 ? "approach" : "recede";
          k.streak = want === k.cand ? k.streak + 1 : 1; k.cand = want;
          if (k.streak >= STREAK) k.cls = want;
        }
        const M = MOTION[k.cls];
        it.color = M.color;
        if (k.cls !== "steady") {
          counts[k.cls]++;
          if (!SEEN[k.cls].has(it.label)) SEEN[k.cls].set(it.label, new Set());
          SEEN[k.cls].get(it.label).add(it.id);
          const ttc = k.cls === "approach" ? -1 / k.v : null;
          it.state = `${M.arrow}${M.word}${ttc && ttc < 10 ? `・約${ttc.toFixed(1)}秒` : ""}`;
        } else it.state = "";
      }
      const labels = [...new Set([...SEEN.approach.keys(), ...SEEN.recede.keys()])];
      r.approach = { live, depth: !!d, counts, rows: labels.map((l) => ({ label: l, approach: SEEN.approach.get(l)?.size || 0, recede: SEEN.recede.get(l)?.size || 0 })) };
    },
    draw(ctx, r, b) {
      const rows = r.approach?.rows || [];
      if (!rows.length) return;
      const size = Math.max(14, Math.round(b.w / 38)), pad = size * 0.5, lh = size * 1.35;
      const lines = rows.map((x) => tx("{label}  ↓接近 {a}  ↑後退 {r}", { label: x.label, a: x.approach, r: x.recede }));
      ctx.save();
      ctx.font = `600 ${size}px system-ui, sans-serif`;
      const w = Math.max(...lines.map((l) => ctx.measureText(l).width)) + pad * 2;
      ctx.fillStyle = "rgba(0,0,0,.6)"; ctx.fillRect(pad, pad, w, lh * lines.length + pad);
      ctx.fillStyle = "#fff"; ctx.textBaseline = "middle";
      lines.forEach((l, i) => ctx.fillText(l, pad * 2, pad + pad / 2 + lh * i + lh / 2));
      ctx.restore();
    },
    panel: (r) => {
      const a = r.approach;
      if (!a) return "";
      const rows = a.rows.map((x) => `<span class="chip">${esc(x.label)} <b style="color:#ef4444">↓${x.approach}</b> <b style="color:#3b82f6">↑${x.recede}</b></span>`).join("");
      return `<div class="sub">接近・後退（今: 接近 ${a.counts.approach}・後退 ${a.counts.recede}）${a.live ? "" : "。動画・カメラの連続実行で追跡しながら判定する"}</div>`
        + (rows ? `<div class="chips">${rows}</div><div class="sub muted">数は一度でも接近・後退と判定した追跡の ID の数（通算）</div>` : "")
        + `<div class="sub muted">枠の色: 赤＝接近、青＝後退、灰＝変化なし。ROI（枠の大きさ）と深度の変化から距離の変化の速さを推定（1 秒に 18% 以上で判定）${a.depth ? "" : "。深度の値が無いので ROI だけ"}</div>`;
    },
    summary: (r) => (r.approach ? r.approach.rows.map((x) => `${x.label} 接近${x.approach}・後退${x.recede}`).join(" ") : ""),
  },

  // 3D の姿勢（姿勢＋深度）: 関節点ごとに深度を拾って奥行きを付け、画面の右下の小窓にゆっくり回して描く。
  // 奥行きは相対深度から作った大まかなもの（形の雰囲気が分かる程度）
  pose3d: {
    combine(r) {
      const d = r.withResults?.depth?.depthRaw;
      r.pose3d = [];
      if (!d) return;
      const depthScale = r.w * 0.6; // 近さ 0〜1 を画面の幅の 6 割の奥行きに（見やすさのための目安）
      for (const p of r.items.filter((it) => it.keypoints)) {
        // 関節は細いので1点だと後ろの床や壁の深度を拾う。まわり 3×3 点の中央値を取り、その人の奥行きの中央値から
        // 枠の高さの 35% 以上は離れないように抑える（トゲのように飛ぶのを防ぐ）
        const r0 = Math.max(2, (p.box[3] - p.box[1]) * 0.015);
        const zs = p.keypoints.map(([x, y]) => {
          const v = [];
          for (let i = -1; i <= 1; i++) for (let j = -1; j <= 1; j++) v.push(nearAt(d, x + i * r0, y + j * r0));
          v.sort((a, b) => a - b);
          return (1 - v[4]) * depthScale;
        });
        const vis = zs.filter((_, i) => p.keypoints[i][2] > 0.3).sort((a, b) => a - b);
        const zmid = vis.length ? vis[vis.length >> 1] : 0, lim = (p.box[3] - p.box[1]) * 0.35;
        r.pose3d.push({ id: p.id, pts: p.keypoints.map(([x, y, v], i) => [x, y, Math.min(zmid + lim, Math.max(zmid - lim, zs[i])), v]) });
      }
    },
    draw(ctx, r, b) {
      if (!r.pose3d?.length) return;
      const W = Math.round(b.w * 0.34), H = Math.round(b.h * 0.42), X0 = b.w - W - 8, Y0 = b.h - H - 8;
      ctx.save();
      ctx.fillStyle = "rgba(10,14,20,.78)"; ctx.fillRect(X0, Y0, W, H);
      ctx.strokeStyle = "rgba(255,255,255,.25)"; ctx.strokeRect(X0, Y0, W, H);
      const yaw = (performance.now() / 4000) % (Math.PI * 2), c = Math.cos(yaw), s = Math.sin(yaw);
      // 全員の点を回して（縦の軸のまわり）、小窓に収まるように縮める
      const all = r.pose3d.flatMap((p) => p.pts.filter((q) => q[3] > 0.3));
      if (!all.length) { ctx.restore(); return; }
      const cx = all.reduce((a, q) => a + q[0], 0) / all.length, cz = all.reduce((a, q) => a + q[2], 0) / all.length;
      const proj = (q) => [(q[0] - cx) * c + (q[2] - cz) * s, q[1]];
      const P = all.map(proj), xs = P.map((q) => q[0]), ys = P.map((q) => q[1]);
      const sc = Math.min((W * 0.85) / Math.max(1, Math.max(...xs) - Math.min(...xs)), (H * 0.8) / Math.max(1, Math.max(...ys) - Math.min(...ys)));
      const mx = (Math.max(...xs) + Math.min(...xs)) / 2, my = (Math.max(...ys) + Math.min(...ys)) / 2;
      const to = (q) => { const [px, py] = proj(q); return [X0 + W / 2 + (px - mx) * sc, Y0 + H / 2 + 8 + (py - my) * sc]; };
      const lw = Math.max(1.5, b.w / 500);
      for (const p of r.pose3d) {
        for (const pair of SKELETON) {
          const [a, bb] = pair.map((i) => p.pts[i]);
          if (a[3] < 0.3 || bb[3] < 0.3) continue;
          const [x1, y1] = to(a), [x2, y2] = to(bb);
          ctx.strokeStyle = limbColor(pair); ctx.lineWidth = lw * 1.4; ctx.beginPath(); ctx.moveTo(x1, y1); ctx.lineTo(x2, y2); ctx.stroke();
        }
      }
      ctx.fillStyle = "#e6e9ef"; ctx.font = `${Math.max(11, Math.round(b.w / 60))}px system-ui, sans-serif`; ctx.textBaseline = "top";
      ctx.fillText(tx("3D（相対深度、回転）"), X0 + 6, Y0 + 5);
      ctx.restore();
    },
    panel: (r) => `<div class="sub">3D の姿勢（${r.pose3d?.length ?? 0} 人）: 右下の小窓に、関節点に深度で奥行きを付けた骨格を回して描く。奥行きは相対深度からの大まかなもの</div>`
      + (r.withResults?.depth && !r.withResults.depth.depthRaw ? `<div class="sub">深度の値が無い（サーバーの深度モデルは組み合わせに使えない）</div>` : ""),
    summary: (r) => (r.pose3d ? `3D ${r.pose3d.length}人` : ""),
  },

  // しぐさ: 姿勢（人ごとの関節点）＋ PINTO の部位（頭・目・手と、頭の向きのクラス）
  gesture: {
    combine(r) {
      const persons = r.items.filter((it) => it.keypoints), parts = r.with?.parts || [];
      const heads = parts.filter((it) => it.label === "head"), orient = parts.filter((it) => ORIENT[it.label]);
      const eyes = parts.filter((it) => it.label === "eye"), hands = parts.filter((it) => it.label === "hand");
      const n = { persons: persons.length, raised: 0, front: 0, closed: 0, pointing: 0 };
      for (const p of persons) {
        const k = p.keypoints, ok = (i) => k[i] && k[i][2] > 0.5, tags = [];
        // 手を挙げた: 手首が肩より上（胴の長さの 1 割より上）。左右どちらか
        const torso = ok(5) && ok(11) ? Math.abs(k[11][1] - k[5][1]) : (p.box[3] - p.box[1]) * 0.3;
        const raised = [[9, 5], [10, 6]].some(([w, s]) => ok(w) && ok(s) && k[w][1] < k[s][1] - torso * 0.1);
        if (raised) { tags.push("手を挙げた"); n.raised++; }
        // 顔の向き: その人の枠の上の方にある頭の枠と、一番重なる向きのクラス
        const head = heads.filter((h) => inBox(center(h.box), p.box) && center(h.box)[1] < (p.box[1] + p.box[3]) / 2).sort((a, b) => b.score - a.score)[0];
        if (head) {
          const o = orient.map((x) => [x, iou(x.box, head.box)]).filter(([, v]) => v > 0.4).sort((a, b) => b[0].score - a[0].score)[0];
          if (o) { tags.push(ORIENT[o[0].label]); if (o[0].label === "front") n.front++; }
          // 目: 頭の中の目の開閉（OCEC の判定がある時）
          const es = eyes.filter((e) => inBox(center(e.box), head.box, 0.1) && /開|閉/.test(e.state || ""));
          if (es.length && es.every((e) => /閉/.test(e.state))) { tags.push("目を閉じている"); n.closed++; }
        }
        // 指差し: その人の枠の近くの手（PGC の判定がある時）
        if (hands.some((h) => inBox(center(h.box), p.box, 0.2) && /指差し/.test(h.state || ""))) { tags.push("指差し"); n.pointing++; }
        p.state = tags.join("・");
      }
      r.gesture = n;
      // 表示は人（骨格と判定）と、判定（目の開閉・指差し）が付いた目・手だけ。頭と向きのクラスの枠は判定に使うだけで出さない
      r.items = [...persons, ...eyes.filter((e) => e.state), ...hands.filter((h) => h.state)];
    },
    panel: (r) => {
      const g = r.gesture;
      if (!g) return "";
      const chip = (k, v) => `<span class="chip">${k} <b>${v}</b></span>`;
      return `<div class="sub">しぐさ（${g.persons} 人）</div><div class="chips">${chip("手を挙げた", g.raised)}${chip("正面を向いている", g.front)}${chip("目を閉じている", g.closed)}${chip("指差し", g.pointing)}</div>`
        + `<div class="sub muted">手を挙げた＝手首が肩より上（姿勢）。顔の向き・目・指差しは PINTO の頭・目・手と小さな分類（目の開閉・指差しのチェックが要る）から、その人の枠の中にあるものを数える</div>`;
    },
    summary: (r) => (r.gesture ? `${r.gesture.persons}人・手を挙げた ${r.gesture.raised}・正面 ${r.gesture.front}` : ""),
  },
};
