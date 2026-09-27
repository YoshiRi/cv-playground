import { MODELS, TASKS } from "./catalog.js";
import { esc, fmtMs as fmt, KINDS } from "./renderers.js";
import { Tracker } from "./tracker.js";

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;      // 静止画の長辺。スマホ写真をそのまま送ると重いので縮める
const DISPLAY_SIDE = 1920;  // 動画・カメラを画面に描く長辺（元の解像度のまま、大きすぎる時だけ縮める）
// 動画・カメラで推論に渡すフレームの長辺は画面で選ぶ（#infer-size、既定 640）。
// YOLO26 は 640、RF-DETR は約 576、DA-V2 は 518 に内部で縮めるので、大きくしても効くのは主に SAM（1024）と送信量
// 配り方は3通りで、どれも web/ の同じコードが動く:
//   サーバー版（server.py が web/ と API を配る）/ 静的版（GitHub Pages などが web/ をそのまま配る）/ 1ファイル版（build.py）
// 違いは実行時に判定する: API に届けばサーバーのモデルも出す（state.hasServer）。1ファイル版だけは Worker を Blob URL から作る
const STANDALONE = !!globalThis.CVPG_STANDALONE;
const WORKER_URL = globalThis.CVPG_WORKER_URL ?? "worker.js";
// web/ の場所（同梱したモデルを読む基準）。1ファイル版は dist/ にあるので ../web/
const WEB_ROOT = new URL(STANDALONE ? "../web/" : "./", location.href).href;

const state = {
  task: "detect",
  image: null,       // 静止画 {blob, bitmap, width, height, key}
  video: false,      // 動画ファイルかカメラを表示中
  live: false,       // 連続実行中
  points: [],        // [[x, y, label]] 表示中の画像・フレームの座標
  result: null,      // 最後の結果（layer: 事前に描いた重ね画像）
  busy: false,
  serverModels: new Set(),
  hasServer: false,  // server.py の API に届くか（静的版・1ファイル版では false）
  frameNo: 0,
  auto: false,       // クリックで切り出しの「全体を自動分割」
};

// ---------- 推論の呼び出し ----------

const workers = {}; // lib ("4" | "3") -> Worker
let seq = 0;
const pending = new Map();
function startWorker(lib) {
  // 1ファイル版の Worker は import 文を含まないので classic で作る（file:// で開くと module Worker は起動できない）
  const w = new Worker(WORKER_URL, { type: STANDALONE ? "classic" : "module", name: lib });
  w.onmessage = onWorkerMessage;
  w.onerror = (e) => setStatus(`Worker の起動に失敗: ${e.message}`, "err");
  workers[lib] = w;
  return w;
}
// onnxruntime-web は一度 OrtRun が失敗すると、同じ Worker 内の以後の実行も同じエラーで失敗し続ける。
// 失敗したら Worker を作り直す（読み込み済みのモデルは失われるが、ダウンロードはブラウザのキャッシュに残る）
// onnx adapter は onnxruntime-web の Worker、それ以外は transformers.js の Worker（lib: "3" なら 3.8.1）
const workerLib = (m) => (m.adapter === "onnx" ? "ort" : m.lib || "4");
function restartWorker(lib) {
  workers[lib]?.terminate();
  delete workers[lib];
}
function onWorkerMessage(ev) {
  const d = ev.data;
  if (d.type === "progress") {
    setStatus(`ダウンロード中 ${d.file?.split("/").pop() ?? ""} ${d.progress?.toFixed?.(0) ?? ""}%`);
    return;
  }
  const p = pending.get(d.id);
  if (!p) return;
  pending.delete(d.id);
  d.type === "error" ? p.reject(new Error(d.message)) : p.resolve(d.result);
}

// 切り出した画像の特徴を onnxruntime-web の Worker で計算する（追跡の ReID 用）
function embedInBrowser(model, crops) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    (workers.ort ?? startWorker("ort")).postMessage({ type: "embed", id, model: { ...model, webRoot: WEB_ROOT }, crops }, crops);
  });
}

function runInBrowser(model, image, params) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    const lib = workerLib(model);
    const transfer = image instanceof ImageBitmap ? [image] : [];
    const opt = model.adapter === "onnx" ? $("ort-opt").value : "";
    (workers[lib] ?? startWorker(lib)).postMessage({ id, model: { ...model, opt, webRoot: WEB_ROOT }, image, params }, transfer);
  });
}

async function runOnServer(model, image, params) {
  const fd = new FormData();
  fd.append("model", model.key);
  fd.append("params", JSON.stringify(params));
  fd.append("image", image, "image.jpg");
  const t0 = performance.now();
  const r = await fetch("api/run", { method: "POST", body: fd });
  const body = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(body.detail || `HTTP ${r.status}`);
  body.roundtrip_ms = performance.now() - t0;
  return body;
}

// ---------- 画面 ----------

function setStatus(text, cls = "") {
  const el = $("status");
  el.textContent = text;
  el.className = `status ${cls}`;
}

// onnx.server_file のモデルは server.py だけが配るので、サーバーが無い時は出さない
// models.json の1件は where に実行できる場所を並べる。画面の選択肢は「モデル × 実行場所」ごとに1つ（値は key@where）。
// 1ファイル版ではサーバーの選択肢を出さない
function variants(task) {
  const out = [];
  for (const where of ["browser", "server"]) {
    for (const e of MODELS.filter((x) => x.task === task && x.where.includes(where) && (state.hasServer || !x.onnx?.server_file))) {
      if (where === "server" && !state.hasServer) continue;
      const avoid = where === "browser" ? e.avoid_browser : null;
      out.push({ ...e, where, avoid, id: `${e.key}@${where}`, ready: where === "browser" || state.serverModels.has(e.key) });
    }
  }
  return out;
}

function renderTasks() {
  $("tasks").innerHTML = "";
  for (const t of TASKS) {
    if (!variants(t.id).some((v) => !v.avoid)) continue;
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = t.name;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(t.id === state.task));
    b.onclick = () => selectTask(t.id);
    $("tasks").append(b);
  }
}

function selectTask(id) {
  stopLive();
  state.task = id;
  state.auto = false;
  state.result = null;
  state.points = [];
  const t = TASKS.find((x) => x.id === id);
  renderTasks();
  $("task-hint").textContent = t.hint;
  for (const el of document.querySelectorAll("[data-param]")) el.hidden = !t.params.includes(el.dataset.param);
  $("track-hint").hidden = !t.params.includes("track") || !$("tracker").value;
  const def = t.defaults || {};
  if (def.threshold != null) { $("threshold").value = def.threshold; $("th-out").textContent = def.threshold; }
  if (def.labels) $("labels").value = def.labels;
  $("canvas-wrap").classList.toggle("clickable", !!t.click);

  const sel = $("model");
  sel.innerHTML = "";
  const vs = variants(id);
  for (const where of ["browser", "server"]) {
    const g = document.createElement("optgroup");
    g.label = where === "browser" ? "ブラウザで実行（この端末）" : "サーバーで実行";
    for (const v of vs.filter((x) => x.where === where)) {
      const o = document.createElement("option");
      o.value = v.id;
      const size = v.mb >= 1000 ? (v.mb / 1000).toFixed(1) + "GB" : v.mb + "MB";
      o.textContent = where === "browser" ? `${v.name}（約${size}）${v.avoid ? "（Mac の Chrome では不可）" : ""}` : v.name;
      o.disabled = !v.ready;
      g.append(o);
    }
    if (g.children.length) sel.append(g);
  }
  const first = vs.find((v) => !v.avoid && v.ready);
  if (first) sel.value = first.id;
  updateModelNote();
  updateButtons();
  setStatus("");
  $("result").innerHTML = "";
  draw();
}

const curTask = () => TASKS.find((t) => t.id === state.task);

function currentModel() {
  return variants(state.task).find((v) => v.id === $("model").value);
}

// 質問欄は、既定の文のままならモデルごとの既定（models.json の prompt、無ければタスクの defaults.prompt）に入れ替える
const TASK_PROMPT = (task) => TASKS.find((t) => t.id === task)?.defaults?.prompt || "";
const DEFAULT_PROMPTS = new Set([...TASKS.map((t) => t.defaults?.prompt), ...MODELS.map((x) => x.prompt)].filter(Boolean));
function updateModelNote() {
  const m = currentModel();
  if (!m) return;
  if (!$("prompt").value || DEFAULT_PROMPTS.has($("prompt").value)) $("prompt").value = m.prompt || TASK_PROMPT(m.task);
  const notes = [];
  notes.push(m.repo || m.onnx?.repo || m.ollama || "");
  notes.push(m.adapter === "onnx" ? "汎用 ONNX（前処理・後処理は models.json）" : `adapter: ${m.adapter}`);
  if (m.where === "browser" && m.mb >= 500) notes.push("初回のダウンロードが大きい。モバイル回線では注意");
  if (m.where === "server") notes.push("画像をサーバーに送って処理する");
  if (m.avoid) notes.push(m.avoid);
  $("ort-opt-row").hidden = !(m.where === "browser" && m.adapter === "onnx");
  $("input-size-row").hidden = !m.pre?.dynamic;
  $("cascade").innerHTML = (m.cascade || []).map((c) =>
    `<label class="check"><input type="checkbox" data-cascade="${c.id}" checked> ${esc(c.name)}</label>`).join("");
  if (m.note) notes.push(m.note);
  $("model-note").textContent = notes.filter(Boolean).join(" / ");
}

function updateButtons() {
  // クリックで点を置くタスク（tasks[].click）は、静止画ではクリックが実行の合図なので実行ボタンを出さない
  $("run").hidden = !!curTask().click && !state.video && !state.auto;
  $("auto").checked = state.auto;
  $("run").textContent = state.video ? "今のフレームで1回実行" : "実行";
  $("video-controls").hidden = !state.video;
  $("live").textContent = state.live ? "■ 連続実行を止める" : "▶ 連続実行";
  $("live").classList.toggle("on", state.live);
  $("pause").textContent = $("video").paused ? "再生" : "一時停止";
  $("pause").hidden = isCamera();
}

// ---------- 画像・動画・カメラ ----------

async function setImage(blobOrUrl) {
  stopVideo();
  const blob = typeof blobOrUrl === "string" ? await (await fetch(blobOrUrl)).blob() : blobOrUrl;
  let bitmap = await createImageBitmap(blob, { imageOrientation: "from-image" });
  const s = Math.min(1, MAX_SIDE / Math.max(bitmap.width, bitmap.height));
  const c = document.createElement("canvas");
  c.width = Math.round(bitmap.width * s);
  c.height = Math.round(bitmap.height * s);
  c.getContext("2d").drawImage(bitmap, 0, 0, c.width, c.height);
  const out = await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92));
  bitmap = await createImageBitmap(out);
  state.image = { blob: out, bitmap, width: c.width, height: c.height, key: `${Date.now()}-${out.size}` };
  clearResult();
  draw();
}

function clearResult() {
  state.points = [];
  state.result = null;
  $("result").innerHTML = "";
}

let stream = null;
const isCamera = () => !!stream;

async function startCamera() {
  let s;
  try {
    s = await navigator.mediaDevices.getUserMedia({ video: { facingMode: "environment", width: { ideal: 1280 } }, audio: false });
  } catch (e) {
    setStatus(`カメラを使えない: ${e.message}`, "err");
    return;
  }
  stopVideo();
  stream = s;
  $("video").srcObject = stream;
  await startVideoCommon();
}

async function setVideoFile(file) {
  stopVideo();
  const v = $("video");
  v.src = URL.createObjectURL(file);
  v.loop = true;
  await startVideoCommon();
}

async function startVideoCommon() {
  const v = $("video");
  v.muted = true;
  await new Promise((r) => (v.readyState >= 1 ? r() : v.addEventListener("loadedmetadata", r, { once: true })));
  await v.play().catch(() => {});
  state.video = true;
  clearResult();
  updateButtons();
  videoLoop();
}

function stopVideo() {
  stopLive();
  const v = $("video");
  if (stream) { stream.getTracks().forEach((t) => t.stop()); stream = null; }
  if (v.getAttribute("src")) { URL.revokeObjectURL(v.src); v.removeAttribute("src"); v.load(); }
  v.srcObject = null;
  state.video = false;
  updateButtons();
}

function videoSize(side) {
  const v = $("video");
  const s = Math.min(1, side / Math.max(v.videoWidth, v.videoHeight));
  return [Math.round(v.videoWidth * s), Math.round(v.videoHeight * s)];
}
const inferSize = () => (state.video ? videoSize(+$("infer-size").value) : [state.image.width, state.image.height]);

// 動画は非表示の <video> のフレームを canvas に描き、最後の結果を重ねる。
// 描くのは新しい動画フレームが来た時だけ（画面の更新ごとに描くと、120Hz のスマホでは 30fps のカメラでも毎秒120回描いて
// 推論と GPU を取り合う）。requestVideoFrameCallback が無いブラウザは画面の更新ごと
function videoLoop() {
  if (!state.video) return;
  draw();
  const v = $("video");
  if (v.requestVideoFrameCallback) v.requestVideoFrameCallback(videoLoop); else requestAnimationFrame(videoLoop);
}

// 推論に渡す1枚。ブラウザ実行は ImageBitmap（速い）、サーバー実行は JPEG
async function grabFrame(forServer) {
  if (!state.video) return { image: state.image.blob, key: state.image.key };
  const [w, h] = inferSize();
  const c = new OffscreenCanvas(w, h);
  c.getContext("2d").drawImage($("video"), 0, 0, w, h);
  state.lastFrame = c; // 推論に使ったフレーム（追跡の ReID で検出を切り出す）
  const key = `frame-${++state.frameNo}`;
  const image = forServer ? await c.convertToBlob({ type: "image/jpeg", quality: 0.85 }) : await createImageBitmap(c);
  return { image, key };
}

// ---------- 描画（結果の種類ごとの描き方は renderers.js） ----------

function baseSource() {
  if (state.video) { const [w, h] = videoSize(DISPLAY_SIDE); return w ? { src: $("video"), w, h } : null; }
  return state.image ? { src: state.image.bitmap, w: state.image.width, h: state.image.height } : null;
}

// 「表示」の選択肢は結果の種類ごと（KINDS[kind].views）。同じ種類の結果が続く間は選んだ見せ方を保つ
let shownViews = "";
function updateViewSelect() {
  const r = state.result, views = r ? KINDS[r.kind]?.views(r) ?? [] : [];
  $("view-toggle").hidden = !views.length;
  const key = views.map(([v]) => v).join();
  if (key === shownViews) return;
  shownViews = key;
  $("view").innerHTML = views.map(([v, t]) => `<option value="${v}">${esc(t)}</option>`).join("");
}

function draw() {
  const cv = $("canvas");
  const ctx = cv.getContext("2d");
  const b = baseSource();
  if (!b) return;
  if (cv.width !== b.w || cv.height !== b.h) { cv.width = b.w; cv.height = b.h; }
  const r = state.result;
  updateViewSelect();
  ctx.globalAlpha = 1;
  ctx.drawImage(b.src, 0, 0, b.w, b.h);
  const view = $("view").value;
  if (r && KINDS[r.kind] && view !== "original" && !$("view-toggle").hidden) KINDS[r.kind].draw(ctx, r, b, view);
  const lw = Math.max(2, b.w / 400);
  for (const [x, y, l] of state.points) {
    ctx.beginPath(); ctx.arc(x, y, lw * 3, 0, Math.PI * 2);
    ctx.fillStyle = l ? "#16a34a" : "#ef4444"; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = lw; ctx.stroke();
  }
}

// 結果欄: モデルと実行場所、数値（バッジ）、内訳（帯）、種類ごとの本文（KINDS[kind].panel）
function renderResult(m, r, live) {
  const where = m.where === "browser" ? `ブラウザ ・ ${r.device}${r.dtype ? " ・ " + r.dtype : ""}` : `サーバー ・ ${r.device}`;
  const stats = [
    ["推論", fmt(r.infer_ms)],
    r.roundtrip_ms ? ["通信込み", fmt(r.roundtrip_ms)] : null,
    // 連続実行は新しいフレームだけを処理するので、fps は動画・カメラのフレームレートが上限。推論だけの上限も並べる
    live ? ["fps", live.fps.toFixed(1)] : null,
    live ? ["推論だけなら", `${(1000 / (r.roundtrip_ms || r.infer_ms)).toFixed(0)} fps`] : null,
    r.load_ms > 1 ? ["読み込み", fmt(r.load_ms)] : null,
  ].filter(Boolean).map(([k, v]) => `<span class="stat"><small>${k}</small><b>${v}</b></span>`).join("");
  let bd = "";
  if (r.breakdown) {
    const parts = [["取り込み", r.breakdown.grab, "#94a3b8"], ["前処理", r.breakdown.pre, "#f59e0b"], ["モデル実行", r.breakdown.run, "#3b82f6"],
      ["後処理", r.breakdown.post, "#22c55e"], ["切り出して分類", r.cascade_ms, "#ec4899"]].filter(([, v]) => v != null);
    const total = parts.reduce((a, [, v]) => a + v, 0) || 1;
    bd = `<div class="breakdown"><div class="bd-bar">${parts.map(([k, v, c]) => `<i style="width:${(100 * v / total).toFixed(1)}%;background:${c}" title="${k} ${fmt(v)}"></i>`).join("")}</div>`
      + `<div class="bd-legend">${parts.map(([k, v, c]) => `<span><i style="background:${c}"></i>${k} ${fmt(v)}</span>`).join("")}</div></div>`;
  }
  const body = KINDS[r.kind]?.panel(r, { live, points: state.points.length }) ?? "";
  $("result").innerHTML = `<div class="result-head"><b>${esc(m.name)}</b><span class="badge ${m.where}">${esc(where)}</span></div>`
    + `<div class="stats">${stats}</div>${bd}${live ? `<div class="sub muted">${live.frames} フレーム</div>` : ""}<div class="result-body">${body}</div>`;
}

function addHistory(m, r, summaryOverride) {
  const tr = document.createElement("tr");
  const task = TASKS.find((t) => t.id === m.task).name;
  const where = m.where === "browser" ? `ブラウザ ${r.device}` : `サーバー ${r.device}`;
  const summary = summaryOverride ?? KINDS[r.kind]?.summary(r) ?? r.kind;
  tr.innerHTML = `<td>${new Date().toLocaleTimeString()}</td><td>${task}</td><td>${esc(m.name)}</td><td>${where}</td><td>${r.w}×${r.h}</td><td class="num">${fmt(r.load_ms)}</td><td class="num">${fmt(r.infer_ms)}</td><td>${esc(summary)}</td>`;
  $("history").prepend(tr);
}

// ---------- 実行 ----------

// 点は表示中の canvas の座標で持ち、推論に渡すフレームの座標に直して送る
function paramsFor(key, w, auto) {
  const k = w / $("canvas").width;
  return {
    threshold: parseFloat($("threshold").value),
    labels: $("labels").value,
    prompt: $("prompt").value,
    points: state.points.map(([x, y, l]) => [x * k, y * k, l]),
    input_size: parseInt($("input-size").value, 10), // 入力サイズ可変のモデルの長辺
    auto,
    _imageKey: key,
  };
}

// 1回分。結果を state.result に入れ、描画用の層を用意する
async function runOnce(m, overrides = {}) {
  const [w, h] = inferSize();
  const tg = performance.now();
  const { image, key } = await grabFrame(m.where === "server");
  const grabMs = performance.now() - tg;
  const params = { ...paramsFor(key, w, state.auto), ...overrides };
  if (params.auto && m.where === "server") throw new Error("全体の自動分割はブラウザの SAM 系モデルのみ");
  const r = m.where === "browser" ? await runInBrowser(m, image, params) : await runOnServer(m, image, params);
  r.w = w; r.h = h;
  if (r.breakdown) r.breakdown = { grab: grabMs, ...r.breakdown };
  await KINDS[r.kind]?.prepare?.(r);
  state.result = r;
  return r;
}

async function run() {
  if (state.busy || state.live) return;
  const m = currentModel();
  if (!m) return;
  if (!state.image && !state.video) { setStatus("先に画像を選んでください", "warn"); return; }
  state.busy = true;
  $("run").disabled = true;
  setStatus(m.where === "browser" ? "ブラウザで実行中…（初回はモデルを取得）" : "サーバーで実行中…（初回はモデルを読み込み）");
  try {
    const r = await runOnce(m);
    await applyCascade(m, r, null);
    renderResult(m, r);
    addHistory(m, r);
    setStatus("");
    draw();
    if (m.where === "server") refreshServer();
  } catch (e) {
    failed(m, e);
  } finally {
    state.busy = false;
    $("run").disabled = false;
  }
}

function failed(m, e) {
  setStatus(`失敗: ${e.message}`, "err");
  if (m.where === "browser" && /OrtRun|storage buffers/.test(e.message)) restartWorker(workerLib(m));
}

// 動画・カメラで、前の推論が終わったら次のフレームを送る（処理が追いつかない間のフレームは飛ばす）。
// fps は2フレーム目以降で測る（1フレーム目はモデルの読み込みを含むため）
async function liveLoop() {
  const m = currentModel();
  let frames = 0, inferSum = 0, first = null, tStart = 0, fps = 0;
  // 追跡: 検出器には低スコア（0.1）まで出させ、閾値スライダーの値を「新しい ID を作る・1段目で使う」下限にする（Ultralytics と同じ構成）
  const trackType = TASKS.find((t) => t.id === state.task).params.includes("track") ? $("tracker").value : "";
  const th = parseFloat($("threshold").value);
  const reid = trackType === "botsort-reid" ? MODELS.find((e) => e.task === "reid") : null;
  const tracker = trackType
    ? new Tracker(trackType.replace("-reid", ""), { track_high_thresh: th, new_track_thresh: th, with_reid: !!reid })
    : null;
  const trackerName = $("tracker").selectedOptions[0]?.textContent;
  const ids = new Set();
  const seqStore = new Map(); // 追跡の ID → 切り出しの履歴（フレーム列を使う分類モデル用）
  setStatus(m.where === "browser" ? "連続実行中…（初回はモデルを取得）" : "連続実行中…（サーバー）");
  try {
    while (state.live && state.video) {
      if (frames) await nextVideoFrame($("video"));
      if (!state.live) break;
      const r = await runOnce(m, tracker ? { threshold: Math.min(th, tracker.args.track_low_thresh) } : {});
      if (tracker) {
        const feats = reid ? await reidFeatures(reid, r.items, tracker.args.track_low_thresh) : null;
        if (feats) r.reid_ms = feats.ms;
        r.items = tracker.update(r.items, feats?.feats);
        r.items.forEach((it) => ids.add(it.id));
        state.result = r;
      }
      await applyCascade(m, r, tracker ? seqStore : null);
      if (!first) { first = r; tStart = performance.now(); setStatus(""); }
      frames++;
      inferSum += r.infer_ms;
      fps = frames > 1 ? (frames - 1) / ((performance.now() - tStart) / 1000) : 0;
      renderResult(m, r, { fps, frames, ids: tracker ? ids.size : 0, trackerName });
      if ($("video").paused) draw();
    }
  } catch (e) {
    failed(m, e);
  }
  if (first) addHistory(m, { ...first, infer_ms: inferSum / frames }, `連続 ${frames}フレーム ${fps.toFixed(1)}fps${tracker ? ` ・ ${trackerName.split("（")[0]} ID ${ids.size}個` : ""}`);
  state.live = false;
  updateButtons();
}

// 検出のあと、models.json の cascade に書いたクラスの枠を切り出して小さな分類モデルにかけ、枠の表示に状態を足す
// （例: 目 → OCEC で開/閉）。seq のモデルは追跡の ID ごとに切り出しをためて、T 枚そろったら判定する
async function applyCascade(m, r, seqStore) {
  if (!m.cascade || r.kind !== "boxes") return;
  const src = state.video ? state.lastFrame : state.image.bitmap;
  const on = new Set([...document.querySelectorAll("[data-cascade]")].filter((c) => c.checked).map((c) => c.dataset.cascade));
  const t0 = performance.now();
  for (const c of m.cascade) {
    if (!on.has(c.id)) continue;
    const e = MODELS.find((x) => x.key === c.model);
    const [w, h] = e.pre.size;
    const targets = r.items.filter((it) => it.label === c.on);
    const crop = (it) => {
      const x1 = Math.max(0, Math.floor(it.box[0])), y1 = Math.max(0, Math.floor(it.box[1]));
      const x2 = Math.min(src.width, Math.ceil(it.box[2])), y2 = Math.min(src.height, Math.ceil(it.box[3]));
      return x2 - x1 < 2 || y2 - y1 < 2 ? null : createImageBitmap(src, x1, y1, x2 - x1, y2 - y1, { resizeWidth: w, resizeHeight: h, resizeQuality: "medium" });
    };
    let use = [], crops = [];
    if (c.seq) {
      if (!seqStore) continue; // フレーム列は追跡の ID が無いと同じ手をつなげられない
      for (const it of targets) {
        const b = await crop(it);
        if (!b) continue;
        const hist = seqStore.get(`${c.id}:${it.id}`) || [];
        const cv = new OffscreenCanvas(w, h);
        cv.getContext("2d").drawImage(b, 0, 0);
        b.close();
        hist.push(cv);
        if (hist.length > c.seq) hist.shift();
        seqStore.set(`${c.id}:${it.id}`, hist);
        if (hist.length === c.seq) { use.push(it); crops.push(...(await Promise.all(hist.map((x) => createImageBitmap(x))))); }
      }
    } else {
      for (const it of targets) { const b = await crop(it); if (b) { use.push(it); crops.push(b); } }
    }
    if (!use.length) continue;
    const res = await embedInBrowser({ ...e, where: "browser" }, crops);
    use.forEach((it, i) => {
      const prob = res.feats[i][0], word = prob >= 0.5 ? c.yes : c.no;
      if (word) it.state = [it.state, `${word} ${(prob * 100).toFixed(0)}%`].filter(Boolean).join("・");
    });
  }
  r.cascade_ms = performance.now() - t0;
}

// 追跡に使う検出（スコア > low）を、推論に使ったフレームから切り出して ReID の特徴にする。ほかは null
async function reidFeatures(e, items, low) {
  const c = state.lastFrame, idx = [], crops = [];
  items.forEach((it, i) => {
    if (it.score <= low) return;
    const x1 = Math.max(0, Math.floor(it.box[0])), y1 = Math.max(0, Math.floor(it.box[1]));
    const x2 = Math.min(c.width, Math.ceil(it.box[2])), y2 = Math.min(c.height, Math.ceil(it.box[3]));
    if (x2 - x1 < 2 || y2 - y1 < 2) return;
    idx.push(i);
    crops.push(createImageBitmap(c, x1, y1, x2 - x1, y2 - y1));
  });
  const feats = new Array(items.length).fill(null);
  if (!crops.length) return { feats, ms: 0 };
  const res = await embedInBrowser({ ...e, where: "browser" }, await Promise.all(crops));
  idx.forEach((i, k) => { feats[i] = res.feats[k]; });
  return { feats, ms: res.ms };
}

// 次の新しいフレームが表示されるまで待つ。推論が動画より速いと同じフレームを何度も処理してしまい、
// 追跡では「止まっている」と誤って速度を推定するため（一時停止中は 0.5 秒ごとに同じフレームで進める）
function nextVideoFrame(v) {
  if (!v.requestVideoFrameCallback) return new Promise((r) => setTimeout(r, 1000 / 30));
  return new Promise((res) => {
    const id = v.requestVideoFrameCallback(() => res());
    setTimeout(() => { v.cancelVideoFrameCallback?.(id); res(); }, 500);
  });
}

function toggleLive() {
  if (state.live) { stopLive(); return; }
  if (!state.video || !currentModel()) return;
  if (curTask().click && !state.points.length && !state.auto) { setStatus("先に画面をクリックして点を置くか、「全体を自動分割」を選んでください", "warn"); return; }
  state.live = true;
  updateButtons();
  liveLoop();
}
function stopLive() {
  state.live = false;
  updateButtons();
}

async function freezeFrame() {
  const [w, h] = videoSize(DISPLAY_SIDE);
  const c = document.createElement("canvas");
  c.width = w; c.height = h;
  c.getContext("2d").drawImage($("video"), 0, 0, w, h);
  await setImage(await new Promise((r) => c.toBlob(r, "image/jpeg", 0.92)));
}

function canvasPoint(ev) {
  const cv = $("canvas"), rect = cv.getBoundingClientRect();
  return [(ev.clientX - rect.left) * (cv.width / rect.width), (ev.clientY - rect.top) * (cv.height / rect.height)];
}

// ---------- サーバー状態 ----------

async function refreshServer() {
  if (!state.hasServer) return;
  try {
    const s = await (await fetch("api/status")).json();
    const parts = [`サーバー ${s.device}`];
    parts.push(s.loaded.length ? `読み込み中: ${s.loaded.join(", ")}` : "読み込み中のモデルなし");
    if (s.mps_allocated_gb != null) parts.push(`MPS ${s.mps_allocated_gb}GB`);
    if (s.ollama_loaded?.length) parts.push(`Ollama: ${s.ollama_loaded.join(", ")}`);
    if (s.gpu_lock) parts.push("⚠ LLMベンチ実行中（GPUロックあり）");
    $("server-status").textContent = parts.join(" ・ ");
  } catch {
    $("server-status").textContent = "サーバーに接続できない（ブラウザ実行のみ）";
  }
}

async function init() {
  // サーバーの有無（API に 3 秒で届かなければ無しとみなす）
  if (!STANDALONE) {
    try {
      const list = await (await fetch("api/models", { signal: AbortSignal.timeout(3000) })).json();
      state.serverModels = new Set(list.map((m) => m.key));
      state.hasServer = true;
    } catch { /* 静的版 */ }
  }
  $("env").textContent = (navigator.gpu ? "WebGPU あり" : "WebGPU なし（WASM で実行、遅い）")
    + (state.hasServer ? "" : " ・ サーバーなし（ブラウザ実行のみ）")
    + (crossOriginIsolated ? "" : " ・ WASM は1スレッド");
  if (navigator.gpu) {
    try { if (!(await navigator.gpu.requestAdapter())) $("env").textContent = "WebGPU アダプタなし（WASM で実行）"; } catch { /* noop */ }
  }
  $("server-box").hidden = !state.hasServer;
  $("threshold").oninput = () => { $("th-out").textContent = $("threshold").value; };
  $("model").onchange = () => { stopLive(); clearResult(); setStatus(""); updateModelNote(); draw(); };
  $("run").onclick = run;
  $("live").onclick = toggleLive;
  $("pause").onclick = () => { const v = $("video"); v.paused ? v.play() : v.pause(); updateButtons(); };
  $("freeze").onclick = freezeFrame;
  $("clear-points").onclick = () => { state.points = []; state.result = null; draw(); };
  $("view").onchange = draw;
  $("ort-opt").onchange = () => { stopLive(); restartWorker("ort"); }; // 設定を変えたらモデルを読み直す
  $("input-size").onchange = () => {
    // graph capture は入力の形が変わると使えないので読み直す。フレームの処理解像度もモデル入力以上にそろえる
    if ($("ort-opt").value.includes("graph")) { stopLive(); restartWorker("ort"); }
    const s = $("input-size").value, f = $("infer-size");
    if (+f.value < +s && [...f.options].some((o) => o.value === s)) f.value = s;
  };
  $("tracker").onchange = () => {
    $("track-hint").hidden = !$("tracker").value;
    if ($("tracker").value) { $("threshold").value = 0.25; $("th-out").textContent = "0.25"; }
  };
  $("auto").onchange = () => {
    state.auto = $("auto").checked;
    state.points = [];
    state.result = null;
    updateButtons();
    draw();
    if (state.auto && !state.video) run();
  };
  $("unload").onclick = async () => { await fetch("api/unload", { method: "POST" }); refreshServer(); };
  $("camera-btn").onclick = startCamera;
  $("file").onchange = (e) => {
    const f = e.target.files[0];
    if (!f) return;
    f.type.startsWith("video/") ? setVideoFile(f) : setImage(f);
    e.target.value = "";
  };
  for (const b of document.querySelectorAll("[data-sample]")) b.onclick = () => setImage(b.dataset.sample);
  $("canvas").addEventListener("click", (ev) => {
    if (!curTask().click || state.auto || (!state.image && !state.video) || state.busy) return;
    const [x, y] = canvasPoint(ev);
    state.points.push([x, y, ev.shiftKey || $("negative").checked ? 0 : 1]);
    draw();
    if (!state.live) run();
  });
  selectTask("detect");
  await setImage(document.querySelector("[data-sample]").dataset.sample);
  refreshServer();
  if (state.hasServer) setInterval(refreshServer, 15000);
}

init();
