// 結果の種類（kind）ごとの見せ方。新しい種類の結果を足す時はここに1件足す（書き方は docs/ADDING_UI.md）。
//
//   KINDS[kind] = {
//     views(r)                    → 画面の「表示」で選べる見せ方 [[値, 表示名], ...]（先頭が既定）
//     prepare?(r)                 → 結果が届いた時に1回だけ呼ぶ（画像の読み込みなど重い下ごしらえ。描画は毎フレームなので）
//     draw(ctx, r, base, view)    → canvas に描く。base = {src, w, h}（元画像は描画済み）、view = 選ばれた見せ方
//     panel(r, ctx)               → 結果欄の本文（HTML）。ctx = {live, points}
//     summary(r)                  → 実行履歴の「結果」欄（短い文字列）
//   }
//
// 結果 r の座標は、推論に渡した画像（r.w × r.h）のピクセル。画面の大きさ（base.w × base.h）へは各 draw で縮尺を合わせる。
import { SKELETON } from "./catalog.js";
import { affineOf, apply3, inv3, mul3 } from "./stabilize.js";

export const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));
const clamp01 = (v) => Math.min(1, Math.max(0, v));
const pct = (v, d = 0) => `${(v * 100).toFixed(d)}%`;

// ---------- 色 ----------

const PALETTE = ["#3b82f6", "#ec4899", "#22c55e", "#f59e0b", "#8b5cf6", "#06b6d4", "#ef4444", "#14b8a6", "#eab308", "#6366f1"];
export const labelColor = (label) => PALETTE[[...label].reduce((a, c) => a + c.charCodeAt(0), 0) % PALETTE.length];
export const idColor = (id) => `hsl(${(id * 137.508) % 360} 80% 55%)`;
// 姿勢の骨格: 左半身（奇数の関節）・右半身（偶数）・体幹で色を分ける
const LIMB = { left: "#38bdf8", right: "#fb923c", center: "#a3e635" };
export const limbColor = ([a, b]) => (a >= 5 && b >= 5 && a % 2 === 1 && b % 2 === 1 ? LIMB.left : a >= 5 && b >= 5 && a % 2 === 0 && b % 2 === 0 ? LIMB.right : LIMB.center);

// ---------- 描画の部品 ----------

function loadImg(src) {
  const url = src instanceof Blob ? URL.createObjectURL(src) : src;
  return new Promise((res, rej) => {
    const i = new Image();
    i.onload = () => { if (src instanceof Blob) URL.revokeObjectURL(url); res(i); };
    i.onerror = rej;
    i.src = url;
  });
}

// グレースケールの結果画像を、画素ごとの関数で色付きの層にする
async function layerFrom(src, fn) {
  const img = await loadImg(src), w = img.naturalWidth, h = img.naturalHeight;
  const c = new OffscreenCanvas(w, h), x = c.getContext("2d");
  x.drawImage(img, 0, 0);
  if (fn) {
    const d = x.getImageData(0, 0, w, h);
    for (let i = 0; i < d.data.length; i += 4) fn(d.data, i, d.data[i] / 255);
    x.putImageData(d, 0, 0);
  }
  return c;
}

// 深度の「範囲を固定」: そのフレームの 0〜255（range の最小〜最大）を、固定した範囲に置き直して塗る
let fixedRange = null;
function remapLayer(r, [LO, HI]) {
  const img = r.layer, w = img.width, h = img.height, [lo, hi] = r.range;
  const c = new OffscreenCanvas(w, h), x = c.getContext("2d"), d = x.createImageData(w, h);
  for (let i = 0, j = 0; i < r.t8.length; i++, j += 4) {
    const val = lo + (r.t8[i] / 255) * (hi - lo), t = clamp01((val - LO) / Math.max(HI - LO, 1e-9));
    const [cr, cg, cb] = turbo(t); d.data[j] = cr; d.data[j + 1] = cg; d.data[j + 2] = cb; d.data[j + 3] = 255;
  }
  x.putImageData(d, 0, 0);
  return c;
}

// 深度の色（turbo の近似。赤＝近い、青＝遠い）
function turbo(t) {
  const r = 0.13572138 + t * (4.6153926 + t * (-42.66032258 + t * (132.13108234 + t * (-152.94239396 + t * 59.28637943))));
  const g = 0.09140261 + t * (2.19418839 + t * (4.84296658 + t * (-14.18503333 + t * (4.27729857 + t * 2.82956604))));
  const b = 0.1066733 + t * (12.64194608 + t * (-60.58204836 + t * (110.36276771 + t * (-89.90310912 + t * 27.34824973))));
  return [255 * clamp01(r), 255 * clamp01(g), 255 * clamp01(b)];
}
// 近さ（0＝遠い〜1＝近い）の色。深度の表示と同じ turbo
export const nearColor = (t) => `rgb(${turbo(Math.min(1, Math.max(0, t))).map(Math.round).join(",")})`;
export const TURBO_CSS = `linear-gradient(90deg, ${[0, 0.25, 0.5, 0.75, 1].map((t) => `rgb(${turbo(t).map(Math.round).join(",")})`).join(", ")})`;

function checker(ctx, w, h) {
  const s = Math.max(8, Math.round(w / 60));
  for (let y = 0; y < h; y += s) for (let x = 0; x < w; x += s) {
    ctx.fillStyle = ((x / s + y / s) & 1) ? "#d0d4da" : "#f2f4f7";
    ctx.fillRect(x, y, s, s);
  }
}

// 枠のラベル（角丸の札。枠が画面の上端にかかる時は枠の内側に置く）
function tag(ctx, x, y, text, color, size) {
  ctx.font = `600 ${size}px system-ui, -apple-system, "Hiragino Sans", sans-serif`;
  const pad = size * 0.35, tw = ctx.measureText(text).width + pad * 2, th = size + pad * 1.2;
  const ty = y - th < 0 ? y : y - th;
  ctx.fillStyle = color;
  ctx.beginPath(); ctx.roundRect(x, ty, tw, th, size * 0.3); ctx.fill();
  ctx.fillStyle = "#fff"; ctx.textBaseline = "middle";
  ctx.fillText(text, x + pad, ty + th / 2 + 1);
}

function drawPose(ctx, kps, lw) {
  if (kps.length !== 17) { // 顔の5点など、骨格の無い点は点だけ描く
    for (const [x, y] of kps) { ctx.beginPath(); ctx.arc(x, y, lw * 1.5, 0, Math.PI * 2); ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = lw * 0.7; ctx.strokeStyle = "#111"; ctx.stroke(); }
    return;
  }
  // 可視度が出ているモデルでは、見えていない関節（< 0.3）を描かない
  const hasVis = kps.some(([, , v]) => v > 0.05), ok = (k) => !hasVis || k[2] >= 0.3;
  ctx.lineCap = "round";
  for (const pair of SKELETON) {
    const [a, b] = pair;
    if (!ok(kps[a]) || !ok(kps[b])) continue;
    ctx.strokeStyle = limbColor(pair); ctx.lineWidth = lw * 1.4;
    ctx.beginPath(); ctx.moveTo(kps[a][0], kps[a][1]); ctx.lineTo(kps[b][0], kps[b][1]); ctx.stroke();
  }
  for (const k of kps) {
    if (!ok(k)) continue;
    ctx.beginPath(); ctx.arc(k[0], k[1], lw * 1.8, 0, Math.PI * 2);
    ctx.fillStyle = "#fff"; ctx.fill(); ctx.lineWidth = lw * 0.8; ctx.strokeStyle = "#111"; ctx.stroke();
  }
}

let cutCanvas = null;

// ---------- 種類ごとの見せ方 ----------

export const KINDS = {
  // 枠（検出・姿勢・追跡）: items: [{label, score, box, keypoints?, id?, trail?, state?}]
  boxes: {
    views: () => [["overlay", "重ねる"], ["only", "結果だけ"], ["original", "元画像"]],
    draw(ctx, r, b, view) {
      if (view === "only") { ctx.fillStyle = "#0b0f14"; ctx.fillRect(0, 0, b.w, b.h); }
      const sx = b.w / r.w, sy = b.h / r.h, lw = Math.max(2, b.w / 360), size = Math.max(12, Math.round(b.w / 55));
      for (const it of r.items) {
        const [x1, y1, x2, y2] = [it.box[0] * sx, it.box[1] * sy, it.box[2] * sx, it.box[3] * sy];
        const col = it.color ?? (it.id != null ? idColor(it.id) : labelColor(it.label)); // it.color は組み合わせのタブなどで上書き
        if (it.trail?.length > 1) { // 追跡の軌跡（枠の中心の履歴）
          ctx.strokeStyle = col; ctx.lineWidth = lw; ctx.globalAlpha = 0.8; ctx.beginPath();
          it.trail.forEach(([x, y], i) => (i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy)));
          ctx.stroke(); ctx.globalAlpha = 1;
        }
        // 暗い縁取りの上に色の線を重ねて、明るい背景でも見えるように
        ctx.lineWidth = lw + 2; ctx.strokeStyle = "rgba(0,0,0,.45)"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        ctx.lineWidth = lw; ctx.strokeStyle = col; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        if (it.keypoints) drawPose(ctx, it.keypoints.map(([x, y, v]) => [x * sx, y * sy, v]), lw);
        const text = `${it.id != null ? "#" + it.id + " " : ""}${it.label} ${pct(it.score)}${it.state ? " · " + it.state : ""}`;
        tag(ctx, x1, y1, text, col, size);
      }
    },
    panel(r, { live }) {
      const chips = countBy(r.items).map(([k, v]) => `<span class="chip"><i style="background:${labelColor(k.split("（")[0])}"></i>${esc(k)} <b>${v}</b></span>`).join("") || `<span class="muted">検出なし</span>`;
      const track = live?.ids ? `<div class="sub">追跡中 ${r.items.length} 件 ・ これまでの ID ${live.ids} 個（${esc(live.trackerName)}${r.reid_ms != null ? ` ・ ReID ${fmtMs(r.reid_ms)}` : ""}）</div>` : "";
      return `<div class="chips">${chips}</div>${track}`;
    },
    summary: (r) => `${r.items.length}件`,
  },

  // マスク（クリックで切り出し・背景除去）: mask = 白が前景の画像、cutout = 背景除去の時 true
  mask: {
    views: (r) => (r.cutout
      ? [["cutout", "切り抜き"], ["only", "マスクだけ（白黒）"], ["original", "元画像"]]
      : [["overlay", "重ねる"], ["only", "マスクだけ（白黒）"], ["cutout", "切り抜き"], ["original", "元画像"]]),
    async prepare(r) {
      r.gray = await layerFrom(r.mask, (d, i) => { d[i + 1] = d[i + 2] = d[i]; d[i + 3] = 255; });   // 白＝前景
      r.alpha = await layerFrom(r.mask, (d, i) => { d[i + 3] = d[i]; });                               // 前景の度合い＝不透明度
      r.tint = await layerFrom(r.mask, (d, i, t) => { d[i] = 59; d[i + 1] = 130; d[i + 2] = 246; d[i + 3] = t > 0.5 ? 130 : 0; });
    },
    draw(ctx, r, b, view) {
      if (view === "only") { ctx.drawImage(r.gray, 0, 0, b.w, b.h); return; }
      if (view === "overlay") {
        if (!r.lost) ctx.drawImage(r.tint, 0, 0, b.w, b.h); // 見失っている時のマスクは別の物なので出さない
        // 動画で追っている時は、このフレームで使った枠（前のフレームのマスクから作ったもの）を出す
        if (r.promptBox) {
          const k = b.w / r.w, [x0, y0, x1, y1] = r.promptBox;
          ctx.setLineDash([6, 4]); ctx.strokeStyle = r.lost ? "#ef4444" : "#fbbf24"; ctx.lineWidth = Math.max(2, b.w / 400);
          ctx.strokeRect(x0 * k, y0 * k, (x1 - x0) * k, (y1 - y0) * k); ctx.setLineDash([]);
        }
        return;
      }
      // 切り抜き: 前景の度合いを不透明度にした層へ、今の画像を source-in で重ねる（動画でも毎フレーム軽い）
      if (!cutCanvas || cutCanvas.width !== b.w || cutCanvas.height !== b.h) cutCanvas = new OffscreenCanvas(b.w, b.h);
      const x = cutCanvas.getContext("2d");
      x.globalCompositeOperation = "copy"; x.drawImage(r.alpha, 0, 0, b.w, b.h);
      x.globalCompositeOperation = "source-in"; x.drawImage(b.src, 0, 0, b.w, b.h);
      checker(ctx, b.w, b.h);
      ctx.drawImage(cutCanvas, 0, 0);
    },
    panel: (r, { points }) => (r.score != null
      ? `<div class="sub">マスクの推定品質 ${pct(r.score)}（${r.promptBox ? `前のフレームのマスクから作った枠で追跡中${r.lost ? `。見失い ${r.lost} フレーム` : ""}` : `点 ${points} 個`}）</div>` : ""),
    summary: (r) => (r.cutout ? "背景除去" : "マスク"),
  },

  // 深度: image = 明るいほど近いグレースケール
  // 色の範囲: 既定はフレームごとに「画像の中の最小〜最大」（毎回いっぱいに使うので、値が揺れても見えない）。
  // 「範囲を固定」は、固定を選んで最初に描いたフレームの range で塗り、範囲の外は端の色にする（時間的な安定性を見る用）。
  // 連続実行の開始と、表示の切り替えで取り直す（resetRange）
  depth: {
    views: () => [["only", "深度だけ（毎フレームの範囲）"], ["fixed", "深度だけ（範囲を固定）"], ["overlay", "半透明で重ねる"], ["original", "元画像"]],
    async prepare(r) {
      const t8 = [];
      r.layer = await layerFrom(r.image, (d, i, t) => { t8[i >> 2] = d[i]; const [cr, cg, cb] = turbo(t); d[i] = cr; d[i + 1] = cg; d[i + 2] = cb; d[i + 3] = 255; });
      r.t8 = Uint8Array.from(t8);
    },
    draw(ctx, r, b, view) {
      let layer = r.layer;
      if (view === "fixed" && r.range && r.t8) {
        fixedRange ??= r.range;
        if (r.fixedFor !== fixedRange) { r.fixedLayer = remapLayer(r, fixedRange); r.fixedFor = fixedRange; }
        layer = r.fixedLayer;
      }
      ctx.globalAlpha = view === "overlay" ? 0.55 : 1;
      ctx.drawImage(layer, 0, 0, b.w, b.h);
      ctx.globalAlpha = 1;
    },
    resetRange() { fixedRange = null; },
    panel: (r) => `<div class="legend"><span>遠い</span><i style="background:${TURBO_CSS}"></i><span>近い</span></div>`
      + (r.range ? `<div class="sub">色の範囲: 「深度だけ（毎フレームの範囲）」はフレームごとに最小〜最大へ合わせ直す。「範囲を固定」は最初のフレームの範囲のまま（値の揺れが見える）</div>` : "")
      + (r.note ? `<div class="sub">${esc(r.note)}</div>` : ""),
    summary: () => "深度",
  },

  // 全体の自動分割: image = 領域ごとに色分けした画像（透明＝どの領域でもない）
  segmap: {
    // クラス名のある塗り分け（セマンティック・パノプティック）は重ねて見るのが既定、自動分割は色分けだけが既定
    views: (r) => (r.legend
      ? [["overlay", "半透明で重ねる"], ["only", "色分けだけ"], ["original", "元画像"]]
      : [["only", "色分けだけ"], ["overlay", "半透明で重ねる"], ["original", "元画像"]]),
    async prepare(r) { r.layer = await layerFrom(r.image); },
    draw(ctx, r, b, view) {
      if (view === "only") { ctx.fillStyle = "#0b0f14"; ctx.fillRect(0, 0, b.w, b.h); }
      ctx.globalAlpha = view === "overlay" ? 0.6 : 1;
      ctx.drawImage(r.layer, 0, 0, b.w, b.h);
      ctx.globalAlpha = 1;
    },
    // legend（クラスごとの色・個数・面積）がある時は凡例を出す（セマンティック・パノプティック）
    panel: (r) => (r.legend
      ? `<div class="chips">${r.legend.map((g) => `<span class="chip"><i style="background:${g.color}"></i>${esc(g.label)}${r.subtask === "panoptic" && g.count > 1 ? ` <b>${g.count}</b>` : ""} <small>${pct(g.area)}</small></span>`).join("")}</div>`
        + `<div class="sub">${r.subtask === "panoptic" ? `${r.count} 個の領域（同じクラスの物は明るさを変えて別々に塗る）` : `${r.legend.length} クラス（数字は画面に占める割合）`}</div>`
      : `<div class="sub">${r.count} 領域（${r.prompts} 点のプロンプトから、品質の良いマスクを重なりを除いて残したもの）</div>`),
    summary: (r) => (r.legend ? `${r.legend.length}クラス` : `${r.count}領域`),
  },

  // 分類: items: [{label, score, abs?}]（score の大きい順）
  labels: {
    views: () => [],
    draw() {},
    panel: (r) => r.items.map((it, i) => `<div class="bar${i === 0 ? " top" : ""}"><span>${esc(it.label)}</span><span class="track"><span class="fill" style="width:${pct(it.score, 1)}"></span></span><span class="num">${pct(it.score, 1)}</span></div>`).join("")
      + (r.items[0]?.abs != null ? `<div class="sub">モデルの絶対スコア: ${r.items.map((it) => `${esc(it.label)} ${pct(it.abs, 1)}`).join("、")}</div>` : ""),
    summary: (r) => r.items[0]?.label ?? "",
  },

  // テンプレートマッチング: テンプレートの四隅を写した四角形 quad、対応 pairs: [[xt, yt, xf, yf, インライアか]]、
  // フレームの特徴点 points: [[x, y, スコア]]。テンプレートの画像は画面側が r.templateImage に付ける（Worker からは送り返さない）
  matches: {
    views: (r) => [["quad", "四角形"], ["pairs", "対応点"], ["points", "特徴点"], ["original", "元画像"]].filter(([v]) => v !== "pairs" || r.templateImage),
    draw(ctx, r, b, view) {
      const sx = b.w / r.w, sy = b.h / r.h, lw = Math.max(2, b.w / 360), size = Math.max(12, Math.round(b.w / 55));
      if (view === "points") {
        const smax = Math.max(1e-6, ...r.points.map((p) => p[2]));
        for (const [x, y, sc] of r.points) {
          ctx.fillStyle = `hsl(${120 * clamp01(sc / smax)} 90% 50%)`;
          ctx.beginPath(); ctx.arc(x * sx, y * sy, lw * 1.2, 0, Math.PI * 2); ctx.fill();
        }
        tag(ctx, 4, 4, `特徴点 ${r.points.length}`, "#0ea5e9", size);
        return;
      }
      if (view === "pairs") return drawPairs(ctx, r, b, lw, size);
      // 四角形: インライアの点と、見つかった四角形（見つからない時は推定だけ点線で薄く）
      for (const [, , xf, yf, inl] of r.pairs) {
        ctx.fillStyle = inl ? "#22c55e" : "rgba(239,68,68,.6)";
        ctx.beginPath(); ctx.arc(xf * sx, yf * sy, lw * (inl ? 1.4 : 1), 0, Math.PI * 2); ctx.fill();
      }
      const q = r.quad;
      if (q) {
        ctx.beginPath(); q.forEach(([x, y], i) => (i ? ctx.lineTo(x * sx, y * sy) : ctx.moveTo(x * sx, y * sy))); ctx.closePath();
        ctx.lineWidth = lw * 2 + 2; ctx.strokeStyle = "rgba(0,0,0,.45)"; ctx.stroke();
        ctx.lineWidth = lw * 2; ctx.strokeStyle = "#22c55e"; ctx.stroke();
        tag(ctx, q[0][0] * sx, q[0][1] * sy, `見つかった ${r.inliers}/${r.matches}`, "#16a34a", size);
      } else {
        tag(ctx, 4, 4, r.template ? `見つからない（インライア ${r.inliers}/${r.matches}）` : (r.note || "テンプレートが無い"), "#64748b", size);
      }
    },
    panel(r) {
      const d = r.post_detail || {};
      const head = r.found ? `<span class="chip"><i style="background:#22c55e"></i>見つかった</span>` : `<span class="chip"><i style="background:#94a3b8"></i>${esc(r.template ? "見つからない" : (r.note || "テンプレートが無い"))}</span>`;
      const nums = `インライア <b>${r.inliers}</b> ・ 対応 <b>${r.matches}</b> ・ 特徴点 フレーム ${r.kpts}${r.kpts_t != null ? ` / テンプレート ${r.kpts_t}` : ""}`;
      const times = [["点の取り出し", d.extract], ["対応", d.match], ["RANSAC", d.ransac], ["テンプレートの特徴", d.template]].filter(([, v]) => v != null).map(([k, v]) => `${k} ${fmtMs(v)}`).join(" ・ ");
      return `<div class="chips">${head}</div><div class="sub">${nums}</div>${times ? `<div class="sub">後処理の内訳: ${times}</div>` : ""}`;
    },
    summary: (r) => `${r.found ? "見つかった" : "見つからない"}（インライア ${r.inliers}/対応 ${r.matches}）`,
  },

  // 手ぶれ補正: Worker の動き（M）と、画面側の stabilize.js が付ける r.stab（補正 W・拡大 Z・揺れ）、解析したフレーム r.frame
  motion: {
    views: () => [["stab", "補正後"], ["side", "並べる（元・補正後）"], ["orig", "元＋切り抜く枠"]],
    draw(ctx, r, b, view) {
      const st = r.stab, f = st?.frame ?? r.frame; // 先読みの時は数フレーム前のフレーム
      if (!f) return;
      const k = b.w / r.w, lw = Math.max(2, b.w / 360), size = Math.max(12, Math.round(b.w / 55));
      ctx.fillStyle = "#0b0f14"; ctx.fillRect(0, 0, b.w, b.h);
      const T = st ? mul3(st.Z, st.W) : [1, 0, 0, 0, 1, 0, 0, 0, 1];
      const drawStab = (ox, oy, sc) => {
        const [a, bb, c, d, e, ff] = affineOf(T, r.w, r.h);
        ctx.save();
        ctx.beginPath(); ctx.rect(ox, oy, r.w * sc, r.h * sc); ctx.clip();
        ctx.setTransform(a * sc, bb * sc, c * sc, d * sc, ox + e * sc, oy + ff * sc);
        ctx.drawImage(f, 0, 0, r.w, r.h);
        ctx.restore();
      };
      if (view === "side") {
        const sc = (b.w / 2 - 4) / r.w, oy = (b.h - r.h * sc) / 2;
        ctx.drawImage(f, 0, oy, r.w * sc, r.h * sc);
        drawStab(b.w / 2 + 4, oy, sc);
        tag(ctx, 4, oy + 4, "元", "#64748b", size); tag(ctx, b.w / 2 + 8, oy + 4, "補正後", "#16a34a", size);
      } else if (view === "orig") {
        ctx.drawImage(f, 0, 0, b.w, b.h);
        // 補正後に見える範囲（画面全体を元のフレームに戻した四角形）と、前のフレームとの対応（インライアは緑）
        const back = inv3(T), q = [[0, 0], [r.w, 0], [r.w, r.h], [0, r.h]].map(([x, y]) => apply3(back, x, y));
        ctx.lineWidth = lw; ctx.strokeStyle = "#facc15"; ctx.beginPath();
        q.forEach(([x, y], i) => (i ? ctx.lineTo(x * k, y * k) : ctx.moveTo(x * k, y * k))); ctx.closePath(); ctx.stroke();
        for (const [xp, yp, xc, yc, inl] of st?.pairs ?? r.pairs) {
          ctx.strokeStyle = inl ? "rgba(34,197,94,.9)" : "rgba(239,68,68,.5)"; ctx.lineWidth = Math.max(1, lw / 2);
          ctx.beginPath(); ctx.moveTo(xp * k, yp * k); ctx.lineTo(xc * k, yc * k); ctx.stroke();
        }
      } else {
        drawStab(0, 0, k);
      }
      if (st?.hist.length > 1) drawJitter(ctx, st.hist, b, lw);
      if (!r.ok && !r.first) tag(ctx, 4, b.h - size * 2.2, `動きを推定できない（インライア ${r.inliers}）`, "#dc2626", size);
    },
    panel(r) {
      if (r.first && !r.stab?.frames) return `<div class="sub">動画かカメラを選んで「▶ 連続実行」で補正する（前のフレームとの動きを使うので、静止画では補正しない）</div>`;
      const j = r.stab?.jitter, d = r.post_detail || {};
      const f = (v, u, n = 1) => `${v.toFixed(n)}${u}`;
      const jit = j?.n ? `<div class="sub">揺れ（細かい成分。フレーム間の動きから前後 9 フレームの平均を引いた残りの二乗平均、${j.n} フレーム）: 補正前 <b>${f(j.raw_px, "px")}・${f(j.raw_deg, "°", 2)}</b> → 補正後 <b>${f(j.out_px, "px")}・${f(j.out_deg, "°", 2)}</b></div>`
        + `<div class="sub">動き（全体。カメラを振った・歩いた動きも入る）: 補正前 ${f(j.move_raw_px, "px")} → 補正後 ${f(j.move_out_px, "px")}（どちらも元のフレームの px）</div>` : "";
      const times = [["点の取り出し", d.extract], ["対応", d.match], ["RANSAC", d.ransac]].filter(([, v]) => v != null).map(([k, v]) => `${k} ${fmtMs(v)}`).join(" ・ ");
      return `<div class="chips"><span class="chip"><i style="background:${r.ok || r.first ? "#22c55e" : "#dc2626"}"></i>${r.model === "homography" ? "ホモグラフィ" : "相似変換"}・インライア ${r.inliers}/${r.matches}</span>`
        + `<span class="chip">${r.stab?.mode === "tripod" ? "三脚" : r.stab?.delay ? `なめらか（先読み ${r.stab.delay} フレーム${r.stab.warmup ? "。ためている所" : ""}）` : `なめらか${{ weak: "弱", mid: "中", strong: "強" }[r.stab?.strength] ?? ""}`}・切り抜き ${Math.round((r.stab?.crop ?? 0) * 100)}%</span>${r.stab?.lost ? `<span class="chip">推定できなかった ${r.stab.lost} フレーム</span>` : ""}${r.stab?.clamped ? `<span class="chip">切り抜きの端に当たった ${r.stab.clamped} フレーム（多ければ切り抜きを大きく）</span>` : ""}</div>`
        + jit + (times ? `<div class="sub">後処理の内訳: ${times}</div>` : "");
    },
    summary: (r) => { const j = r.stab?.jitter; return j?.n ? `揺れ ${j.raw_px.toFixed(1)}px→${j.out_px.toFixed(1)}px（動き ${j.move_raw_px.toFixed(1)}→${j.move_out_px.toFixed(1)}）` : "補正なし"; },
  },

  // 文章（画像の説明・質問）
  text: {
    views: () => [],
    draw() {},
    panel: (r) => `<blockquote class="answer">${esc(r.text)}</blockquote>`,
    summary: (r) => r.text.slice(0, 24) + "…",
  },
};

// 揺れのグラフ（画面の下の帯）: フレーム間の動きの大きさ（px）。赤が補正前、緑が補正後
function drawJitter(ctx, hist, b, lw) {
  const H = b.h * 0.18, y0 = b.h - H, n = hist.length, max = Math.max(4, ...hist.map((x) => x.raw));
  ctx.fillStyle = "rgba(11,15,20,.55)"; ctx.fillRect(0, y0, b.w, H);
  for (const [key, col, wd] of [["raw", "#f87171", lw * 0.6], ["out", "#4ade80", lw]]) {
    ctx.strokeStyle = col; ctx.lineWidth = wd; ctx.beginPath();
    hist.forEach((h, i) => { const x = (i / Math.max(1, 239)) * b.w, y = y0 + H - (h[key] / max) * (H - 4) - 2; i ? ctx.lineTo(x, y) : ctx.moveTo(x, y); });
    ctx.stroke();
  }
  ctx.fillStyle = "#e5e7eb"; ctx.font = `${Math.max(10, Math.round(b.w / 70))}px sans-serif`;
  ctx.fillText(`フレーム間の動き（元のフレームの px、最大 ${max.toFixed(0)}）  赤: 補正前  緑: 補正後`, 6, y0 + Math.max(12, b.w / 60));
}

// 対応点: 左にテンプレート、右にフレームを並べ、対応を線で結ぶ（インライアは緑、外れは薄い赤）。XFeat の動作確認用
function drawPairs(ctx, r, b, lw, size) {
  const T = r.templateImage, tw = r.template.w, th = r.template.h, gap = Math.round(b.w * 0.02);
  ctx.fillStyle = "#0b0f14"; ctx.fillRect(0, 0, b.w, b.h);
  // テンプレートは幅の 35% まで・高さいっぱいまで、フレームは残りの幅に収める
  const st = Math.min((b.w * 0.35) / tw, b.h / th), TW = tw * st, TH = th * st, ty = (b.h - TH) / 2;
  const sf = Math.min((b.w - TW - gap) / b.w, 1), FW = b.w * sf, FH = b.h * sf, fx = TW + gap, fy = (b.h - FH) / 2;
  ctx.drawImage(T, 0, ty, TW, TH);
  ctx.drawImage(b.src, fx, fy, FW, FH);
  const kx = (FW / r.w), ky = (FH / r.h);
  ctx.lineWidth = Math.max(1, lw / 2);
  const pairs = r.pairs.length > 400 ? r.pairs.filter((p) => p[4]) : r.pairs; // 多すぎる時はインライアだけ
  for (const [xt, yt, xf, yf, inl] of pairs) {
    ctx.strokeStyle = inl ? "rgba(34,197,94,.85)" : "rgba(239,68,68,.45)";
    ctx.beginPath(); ctx.moveTo(xt * st, ty + yt * st); ctx.lineTo(fx + xf * kx, fy + yf * ky); ctx.stroke();
  }
  if (r.quad) {
    ctx.beginPath(); r.quad.forEach(([x, y], i) => (i ? ctx.lineTo(fx + x * kx, fy + y * ky) : ctx.moveTo(fx + x * kx, fy + y * ky))); ctx.closePath();
    ctx.lineWidth = lw * 1.5; ctx.strokeStyle = "#22c55e"; ctx.stroke();
  }
  tag(ctx, 4, 4, `対応 ${r.matches}（インライア ${r.inliers}）`, r.found ? "#16a34a" : "#64748b", size);
}

function countBy(items) {
  const c = {};
  for (const it of items) { const k = it.state ? `${it.label}（${it.state.replace(/ \d+%|（[^）]*）/g, "")}）` : it.label; c[k] = (c[k] || 0) + 1; }
  return Object.entries(c).sort((a, b) => b[1] - a[1]);
}

export const fmtMs = (ms) => (ms == null ? "-" : ms >= 10000 ? `${(ms / 1000).toFixed(0)}s` : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : ms >= 10 ? `${ms.toFixed(0)}ms` : `${ms.toFixed(1)}ms`);
