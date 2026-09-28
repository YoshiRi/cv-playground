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
