// 応用: モデルの結果を受け取り、フレームをまたいだ状態を持って集計し、画面に重ねる（例: 物体を数える）。
// models.json の apps に {id, name, accepts: [結果の種類], params, hint} を書くと、その種類の結果を返すタブ（tasks[].result）の
// 「応用」欄にチェックが出て、選ぶと追跡・cascade のあとに順に呼ばれる。モデルの後処理（onnx_generic.js・adapters.py、
// models.json の post）とは別物で、モデルの結果の見せ方（renderers.js の KINDS）とも別に、集計の状態と表示だけを持つ
import { esc, labelColor, limbColor, nearColor } from "./renderers.js";
import { SKELETON } from "./catalog.js";

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
      const lines = rows.map(({ label, now, total }) => `${label}  ${now}${total != null ? `（通算 ${total}）` : ""}`);
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
// 枠の中央 6 割の 5×5 点の近さの中央値（枠の端の背景を拾いにくく）
function nearOfBox(d, b) {
  const v = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) v.push(nearAt(d, b[0] + (b[2] - b[0]) * (0.2 + 0.15 * i), b[1] + (b[3] - b[1]) * (0.2 + 0.15 * j)));
  v.sort((a, b2) => a - b2);
  return v[12];
}
const NEAR_HIST = new Map(); // 追跡の ID → 近さの履歴（接近の判定用）

export const COMBOS = {
  // 近さ（物体検出＋深度）: 枠の中の深度の中央値で近さを出し、枠を近さの色（赤＝近い、青＝遠い）で塗って近い順の番号を付ける。
  // 追跡の ID ごとに近さの履歴を持ち、直近 1 秒ほどで近さが 0.08 以上増えた物に「接近」。深度は相対値で、フレームごとに
  // 範囲が変わるので、近さはその画像の中での相対（距離ではない）
  near: {
    reset() { NEAR_HIST.clear(); },
    combine(r) {
      const d = r.withResults?.depth?.depthRaw;
      if (!d) { r.near = { error: "深度の値が無い（サーバーの深度モデルは組み合わせに使えない）" }; return; }
      const items = r.items.map((it) => ({ it, n: nearOfBox(d, it.box) })).sort((a, b) => b.n - a.n);
      let approaching = 0;
      items.forEach(({ it, n }, i) => {
        it.color = nearColor(n);
        const tags = [i === 0 ? "一番近い" : `近さ ${i + 1}番`];
        if (it.id != null) {
          const h = NEAR_HIST.get(it.id) || [];
          h.push({ t: performance.now(), n });
          while (h.length && performance.now() - h[0].t > 1200) h.shift();
          NEAR_HIST.set(it.id, h);
          if (h.length >= 4 && n - h[0].n > 0.08) { tags.push("接近"); approaching++; }
        }
        it.state = tags.join("・");
      });
      r.near = { n: items.length, approaching, reused: !!r.withResults.depth.reused };
    },
    panel: (r) => (r.near?.error ? `<div class="sub">${esc(r.near.error)}</div>`
      : r.near ? `<div class="sub">近さ（${r.near.n} 件）: 枠の色は近さ（赤＝近い、青＝遠い）、番号は近い順。接近 <b>${r.near.approaching}</b> 件（追跡を選ぶと出る）</div>`
        + `<div class="sub muted">深度は相対値で、近さはこの画像の中での順番（距離ではない）。深度は数フレームに1回だけ回し、間は前の深度を使う</div>` : ""),
    summary: (r) => (r.near && !r.near.error ? `近さ ${r.near.n}件・接近 ${r.near.approaching}` : ""),
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
      ctx.fillText("3D（相対深度、回転）", X0 + 6, Y0 + 5);
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
