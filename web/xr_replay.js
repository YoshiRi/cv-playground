// AR の記録の再生（xr_replay.html）。記録を 2 回流し（姿勢グラフなし・あり）、地図を並べる。
// ?log=名前 でサーバーの記録を選び、&auto で開いたらすぐ再生する（結果は window.__replay にも入れる。確かめる用）
import { startTranslate, tx } from "./i18n.js";
import { readLog } from "./xr_log.js";
import { XrPipeline } from "./xr_pipeline.js";

startTranslate();
const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
let fileBlob = null;

async function listLogs() {
  try {
    const list = await (await fetch("api/xrlogs")).json();
    $("list").innerHTML = list.map((x) => `<option value="${x.name}">${x.name}（${(x.bytes / 1e6).toFixed(1)} MB）</option>`).join("");
    if (q.get("log")) $("list").value = q.get("log");
  } catch { $("list").innerHTML = `<option value="">${tx("（サーバーが無い）")}</option>`; }
}

async function getBlob() {
  if (fileBlob) return fileBlob;
  const name = $("list").value;
  if (!name) throw new Error(tx("記録を選ぶ"));
  return (await fetch(`api/xrlogs/${encodeURIComponent(name)}`)).blob();
}

async function run() {
  $("run").disabled = true;
  try {
    const blob = await getBlob(), out = [];
    for (const [k, graph] of [[0, false], [1, true]]) {
      const pipe = new XrPipeline({ graph, dynamic: $("dyn").checked }), t0 = performance.now();
      let n = 0, meta = null, tFirst = null, tLast = 0;
      for await (const r of readLog(blob)) {
        if (r.k === "meta") meta = r;
        if (r.t != null) { tFirst ??= r.t; tLast = r.t; }
        pipe.apply(r);
        if (++n % 300 === 0) { $("status").textContent = tx("再生中 {k}/2: {n} 行", { k: k + 1, n }); await new Promise((res) => setTimeout(res)); }
      }
      pipe.finish();
      const ms = performance.now() - t0, map = pipe.map, cells = map.cellsWorld();
      const c = $(`map${k}`), ctx = c.getContext("2d");
      ctx.clearRect(0, 0, c.width, c.height);
      map.cache = null;
      map.draw(ctx, 0, 0, c.width, 2, { objs: pipe.objects.forMap(map), showSubmaps: true, loops: pipe.graph.loops.map((L) => [L.j, L.i]) });
      const res = {
        graph, rows: n, secs: +((tLast - tFirst) / 1000).toFixed(1), replay_ms: Math.round(ms), frames: pipe.stats.frames, depth: pipe.stats.depth, dets: pipe.stats.dets,
        submaps: map.subs.length, floorY: map.floorY, obstacle_cells: cells.obst.length, floor_cells: cells.floor.length,
        collected: pipe.objects.collected().length, loops: pipe.graph.loops.length, graph_stats: pipe.graph.stats, meta,
      };
      $(`cap${k}`).textContent = graph ? tx("姿勢グラフあり: ループ閉じ {l}（外した {r}）", { l: res.loops, r: pipe.graph.stats.rejected }) : tx("姿勢グラフなし");
      out.push(res);
    }
    $("stats").textContent = JSON.stringify(out, null, 1);
    $("status").textContent = tx("再生した");
    window.__replay = out;
  } catch (e) {
    $("status").textContent = `${tx("エラー")}: ${e.message}`;
    window.__replay = { error: e.message };
  } finally { $("run").disabled = false; }
}

$("file").onchange = (e) => { fileBlob = e.target.files[0] || null; if (fileBlob) $("status").textContent = fileBlob.name; };
$("run").onclick = run;
if (q.get("dyn") === "0") $("dyn").checked = false;
await listLogs();
if (q.has("auto")) run();
