// 応用: モデルの結果を受け取り、フレームをまたいだ状態を持って集計し、画面に重ねる（例: 物体を数える）。
// models.json の apps に {id, name, accepts: [結果の種類], params, hint} を書くと、その種類の結果を返すタブ（tasks[].result）の
// 「応用」欄にチェックが出て、選ぶと追跡・cascade のあとに順に呼ばれる。モデルの後処理（onnx_generic.js・adapters.py、
// models.json の post）とは別物で、モデルの結果の見せ方（renderers.js の KINDS）とも別に、集計の状態と表示だけを持つ
import { esc, labelColor } from "./renderers.js";

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

export const COMBOS = {
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
