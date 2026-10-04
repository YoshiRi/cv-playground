// 受け手のページ（receiver.html）: 出口が流すフレームを受け取り、キャラ（interact.js の puppet）と中身を出す。
// 元の画面とはフレームの形（v1）だけでつながっていて、モデルの名前・設定は知らない
import { startTranslate, tx } from "./i18n.js";
import { CHANNEL, INTERACT } from "./interact.js";

startTranslate();
document.title = tx("CV Playground 受け手");

const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const puppet = INTERACT.puppet.create($("view"), {});
let n = 0, last = 0, rate = 0, lat = 0, lastKey = "";

function onMessage(msg, via) {
  if (msg?.type !== "frame" || msg.frame?.v !== 1) return;
  const f = msg.frame, now = performance.now(), key = `${f.wall}:${f.seq}`;
  if (key === lastKey) return; // BroadcastChannel と WebSocket の両方から同じフレームが届いた時は 1 回だけ
  lastKey = key;
  puppet.onFrame(f);
  if (last) rate = rate ? rate * 0.9 + 0.1 * (1000 / (now - last)) : 1000 / (now - last);
  last = now; n++;
  lat = Date.now() - f.wall; // 送り手と同じ端末なら、送ってから届くまで（別の端末だと時計のずれも入る）
  if (f.decisions) $("stat").dataset.dec = f.decisions.map((d) => `${tx(d.value)}(${d.p})`).join(" ");
  $("stat").textContent = (f.decisions ? tx("判定: {d} ・ ", { d: $("stat").dataset.dec }) : "")
    + tx("{via} ・ {n} フレーム ・ {fps} fps ・ 遅れ {lat} ms ・ #{seq} {what}", { via, n, fps: rate.toFixed(1), lat, seq: f.seq, what: `${f.task ?? ""} ${f.model ?? ""} ${f.w}×${f.h}` });
  $("items").innerHTML = "<tr><th>ID</th><th>ラベル</th><th>スコア</th><th>枠（0〜1）</th><th>点</th><th>状態・表情</th></tr>"
    + f.items.slice(0, 30).map((it) => `<tr><td>${it.id ?? ""}</td><td>${esc(it.label)}</td><td>${(it.score ?? 0).toFixed(2)}</td>`
      + `<td>${it.box.map((v) => v.toFixed(2)).join(", ")}</td><td>${it.keypoints?.length ?? ""}</td>`
      + `<td>${esc([it.state, it.emotion?.label].filter(Boolean).join(" ・ "))}</td></tr>`).join("");
  if ($("raw").parentElement.open) $("raw").textContent = JSON.stringify(f, null, 1);
}
const esc = (s) => String(s ?? "").replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);

const srcs = [];
const bc = new BroadcastChannel(CHANNEL);
bc.onmessage = (e) => onMessage(e.data, "BroadcastChannel");
srcs.push(tx("BroadcastChannel「{c}」", { c: CHANNEL }));
if (q.has("ws")) {
  const url = q.get("ws") || `${location.protocol === "https:" ? "wss" : "ws"}://${location.host}/ws`;
  const connect = () => {
    const ws = new WebSocket(url);
    ws.onmessage = (e) => { try { onMessage(JSON.parse(e.data), "WebSocket"); } catch {} };
    ws.onclose = () => setTimeout(connect, 3000);
  };
  connect();
  srcs.push(`WebSocket ${url}`);
}
$("src").textContent = tx("受け取り元: {s}（?ws を付けると同じサーバーの /ws からも受け取る）", { s: srcs.join(" ・ ") });
