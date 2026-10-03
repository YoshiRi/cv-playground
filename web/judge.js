// 判定（ゼロショット分類のスコアから、はい・いいえを返す）。ロボットのような簡易センサの判断（通れる・通れない、安全・危険、
// 空いている・埋まっている）を想定。1 行に 1 つ:
//
//   通れる | 通れない : an open corridor, a clear path | a blocked path, an obstacle in the way
//
// 「:」の左が判定の名前（はい | いいえ。いいえを省くと「名前でない」）、右が言い方（はい側 | いいえ側。どちらも複数書ける）。
// はい側・いいえ側それぞれの一番高い logit の差から、はいの確からしさ p = シグモイド(はい − いいえ)（2 つの候補の softmax と同じ）。
// 動画では p を指数移動平均でならし、HI 以上（LO 以下）が K フレーム続いた時だけ切り替える（ちらつかないように）。
// 決まるまでは「不明」。切り替わった時刻を記録する

export const JUDGE_DEFAULT = [
  "通れる | 通れない : an open corridor, a clear path, an empty hallway | a blocked path, an obstacle in the way, a closed door",
  "安全 | 危険 : a safe empty floor, an empty room | a person very close to the camera, stairs going down, a dangerous situation",
  "空いている | 埋まっている : an empty space, an empty table, an empty shelf | a crowded space, a table full of objects, a full shelf",
].join("\n");

const split = (s) => s.split(",").map((x) => x.trim()).filter(Boolean);
export function parseJudges(text) {
  const out = [];
  for (const line of (text || "").split("\n")) {
    const i = line.indexOf(":");
    if (i < 0) continue;
    const [yesName, noName] = line.slice(0, i).split("|").map((x) => x.trim());
    const [yes, no] = line.slice(i + 1).split("|").map((x) => split(x || ""));
    if (!yesName || !yes?.length || !no?.length) continue;
    out.push({ name: yesName, yesName, noName: noName || `${yesName}でない`, yes, no });
  }
  return out;
}
export const judgePrompts = (judges) => judges.flatMap((j) => [...j.yes, ...j.no]);

const HI = 0.6, LO = 0.4, K = 3, ALPHA = 0.3;
const sig = (x) => 1 / (1 + Math.exp(-x));

export class Judges {
  constructor() { this.reset(); }
  reset() { this.st = new Map(); this.log = []; this.t0 = performance.now(); }
  // logit: 言い方 → logit。live なら時間でならす（静止画の 1 回は p だけで決める）
  update(judges, logit, live) {
    const now = performance.now(), out = [];
    for (const j of judges) {
      const ym = Math.max(...j.yes.map((s) => logit[s] ?? -Infinity)), nm = Math.max(...j.no.map((s) => logit[s] ?? -Infinity));
      const raw = sig(ym - nm), best = j.yes.concat(j.no).reduce((a, s) => ((logit[s] ?? -Infinity) > (logit[a] ?? -Infinity) ? s : a));
      const key = `${j.yesName}|${j.noName}`;
      let s = this.st.get(key);
      if (!s || !live) s = { p: raw, value: null, run: 0, since: now };
      else s.p = s.p + ALPHA * (raw - s.p);
      const want = s.p >= HI ? true : s.p <= LO ? false : null;
      let changed = false;
      if (!live) { s.value = want; }
      else if (want !== null && want !== s.value) {
        s.run = (s.run || 0) + 1;
        if (s.run >= K || s.value === null) { s.value = want; s.run = 0; s.since = now; changed = true; }
      } else s.run = 0;
      this.st.set(key, s);
      const label = s.value === null ? "不明" : s.value ? j.yesName : j.noName;
      if (changed) { this.log.push({ t: (now - this.t0) / 1000, name: j.name, value: label, p: s.p }); if (this.log.length > 50) this.log.shift(); }
      // conf: 出している値の側の確からしさ（いいえなら 1 − p。不明なら大きい方）
      const conf = s.value === null ? Math.max(s.p, 1 - s.p) : s.value ? s.p : 1 - s.p;
      out.push({ name: j.name, yes: j.yesName, no: j.noName, value: label, decided: s.value, p: +s.p.toFixed(3), conf: +conf.toFixed(3), raw: +raw.toFixed(3), changed, reason: best });
    }
    return out;
  }
}
