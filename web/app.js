import { DEFAULTS, MODELS, SKELETON, TASKS } from "./catalog.js";

const $ = (id) => document.getElementById(id);
const MAX_SIDE = 1280;      // 静止画の長辺。スマホ写真をそのまま送ると重いので縮める
const DISPLAY_SIDE = 1920;  // 動画・カメラを画面に描く長辺（元の解像度のまま、大きすぎる時だけ縮める）
// 動画・カメラで推論に渡すフレームの長辺は画面で選ぶ（#infer-size、既定 640）。
// YOLO26 は 640、RF-DETR は約 576、DA-V2 は 518 に内部で縮めるので、大きくしても効くのは主に SAM（1024）と送信量
// 1ファイル版（build.py）では、サーバーを使わず Worker を Blob URL から作る
const STANDALONE = !!globalThis.CVPG_STANDALONE;
const WORKER_URL = globalThis.CVPG_WORKER_URL ?? "worker.js";

const state = {
  task: "detect",
  image: null,       // 静止画 {blob, bitmap, width, height, key}
  video: false,      // 動画ファイルかカメラを表示中
  live: false,       // 連続実行中
  points: [],        // [[x, y, label]] 表示中の画像・フレームの座標
  result: null,      // 最後の結果（layer: 事前に描いた重ね画像）
  busy: false,
  serverModels: new Set(),
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

function runInBrowser(model, image, params) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    const lib = workerLib(model);
    const transfer = image instanceof ImageBitmap ? [image] : [];
    (workers[lib] ?? startWorker(lib)).postMessage({ id, model, image, params }, transfer);
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

// models.json の1件は where に実行できる場所を並べる。画面の選択肢は「モデル × 実行場所」ごとに1つ（値は key@where）。
// 1ファイル版ではサーバーの選択肢を出さない
function variants(task) {
  const out = [];
  for (const where of ["browser", "server"]) {
    for (const e of MODELS.filter((x) => x.task === task && x.where.includes(where))) {
      if (where === "server" && STANDALONE) continue;
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
  if (DEFAULTS.threshold[id]) { $("threshold").value = DEFAULTS.threshold[id]; $("th-out").textContent = DEFAULTS.threshold[id]; }
  if (DEFAULTS.labels[id]) $("labels").value = DEFAULTS.labels[id];
  $("canvas-wrap").classList.toggle("clickable", id === "segment");

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

function currentModel() {
  return variants(state.task).find((v) => v.id === $("model").value);
}

const DEFAULT_PROMPTS = new Set([DEFAULTS.prompt, ...MODELS.filter((x) => x.prompt).map((x) => x.prompt)]);
function updateModelNote() {
  const m = currentModel();
  if (!m) return;
  if (DEFAULT_PROMPTS.has($("prompt").value)) $("prompt").value = m.prompt || DEFAULTS.prompt;
  const notes = [];
  notes.push(m.repo || m.onnx?.repo || m.ollama || "");
  notes.push(m.adapter === "onnx" ? "汎用 ONNX（前処理・後処理は models.json）" : `adapter: ${m.adapter}`);
  if (m.where === "browser" && m.mb >= 500) notes.push("初回のダウンロードが大きい。モバイル回線では注意");
  if (m.where === "server") notes.push("画像をサーバーに送って処理する");
  if (m.avoid) notes.push(m.avoid);
  if (m.note) notes.push(m.note);
  $("model-note").textContent = notes.join(" / ");
}

function updateButtons() {
  $("run").hidden = state.task === "segment" && !state.video && !state.auto;
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
  requestAnimationFrame(videoLoop);
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

// 動画は非表示の <video> を毎フレーム canvas に描き、最後の結果を重ねる
function videoLoop() {
  if (!state.video) return;
  draw();
  requestAnimationFrame(videoLoop);
}

// 推論に渡す1枚。ブラウザ実行は ImageBitmap（速い）、サーバー実行は JPEG
async function grabFrame(forServer) {
  if (!state.video) return { image: state.image.blob, key: state.image.key };
  const [w, h] = inferSize();
  const c = new OffscreenCanvas(w, h);
  c.getContext("2d").drawImage($("video"), 0, 0, w, h);
  const key = `frame-${++state.frameNo}`;
  const image = forServer ? await c.convertToBlob({ type: "image/jpeg", quality: 0.85 }) : await createImageBitmap(c);
  return { image, key };
}

// ---------- 描画 ----------

function loadImg(url) {
  return new Promise((res, rej) => { const i = new Image(); i.onload = () => res(i); i.onerror = rej; i.src = url; });
}

const COLORS = ["#2f6fdf", "#e0457b", "#16a34a", "#f59e0b", "#8b5cf6", "#0ea5e9", "#ef4444", "#14b8a6"];
const colorOf = (label) => COLORS[[...label].reduce((a, c) => a + c.charCodeAt(0), 0) % COLORS.length];

function baseSource() {
  if (state.video) { const [w, h] = videoSize(DISPLAY_SIDE); return w ? { src: $("video"), w, h } : null; }
  return state.image ? { src: state.image.bitmap, w: state.image.width, h: state.image.height } : null;
}

// 結果の見せ方。種類ごとに選べる表示と、最初の既定（先頭）
const VIEWS = {
  boxes: [["overlay", "重ねる"], ["only", "結果だけ"], ["original", "元画像"]],
  mask: [["overlay", "重ねる"], ["only", "マスクだけ（白黒）"], ["cutout", "切り抜き"], ["original", "元画像"]],
  cutout: [["cutout", "切り抜き"], ["only", "マスクだけ（白黒）"], ["original", "元画像"]],
  depth: [["only", "深度だけ"], ["overlay", "半透明で重ねる"], ["original", "元画像"]],
  segmap: [["only", "色分けだけ"], ["overlay", "半透明で重ねる"], ["original", "元画像"]],
};
const viewKind = (r) => (r.kind === "mask" && r.cutout ? "cutout" : r.kind);
let shownKind = null;
function updateViewSelect() {
  const r = state.result, kind = r && VIEWS[viewKind(r)] ? viewKind(r) : null;
  $("view-toggle").hidden = !kind;
  if (!kind || kind === shownKind) return;
  shownKind = kind;
  $("view").innerHTML = VIEWS[kind].map(([v, t]) => `<option value="${v}">${t}</option>`).join("");
}

let cutCanvas = null;
function draw() {
  const cv = $("canvas");
  const ctx = cv.getContext("2d");
  const b = baseSource();
  if (!b) return;
  if (cv.width !== b.w || cv.height !== b.h) { cv.width = b.w; cv.height = b.h; }
  const r = state.result;
  updateViewSelect();
  const view = r ? $("view").value : "original";
  const lw = Math.max(2, b.w / 400);
  ctx.globalAlpha = 1;
  ctx.drawImage(b.src, 0, 0, b.w, b.h);

  if (r && view !== "original") {
    if (view === "only" && ["boxes", "segmap", "mask"].includes(r.kind)) { ctx.fillStyle = "#000"; ctx.fillRect(0, 0, b.w, b.h); }
    if (r.kind === "depth" || r.kind === "segmap") {
      ctx.globalAlpha = view === "overlay" ? 0.55 : 1;
      ctx.drawImage(r.layer, 0, 0, b.w, b.h);
      ctx.globalAlpha = 1;
    } else if (r.kind === "mask" && view === "cutout") {
      // 前景の度合いを不透明度にした層へ、今の画像を source-in で重ねて切り抜く（動画でも毎フレーム軽い）
      if (!cutCanvas || cutCanvas.width !== b.w || cutCanvas.height !== b.h) cutCanvas = new OffscreenCanvas(b.w, b.h);
      const x = cutCanvas.getContext("2d");
      x.globalCompositeOperation = "copy";
      x.drawImage(r.alpha, 0, 0, b.w, b.h);
      x.globalCompositeOperation = "source-in";
      x.drawImage(b.src, 0, 0, b.w, b.h);
      drawChecker(ctx, b.w, b.h);
      ctx.drawImage(cutCanvas, 0, 0);
    } else if (r.kind === "mask") {
      ctx.drawImage(view === "only" ? r.gray : r.tint, 0, 0, b.w, b.h);
    } else if (r.kind === "boxes") {
      const sx = b.w / r.w, sy = b.h / r.h;
      ctx.font = `${Math.max(12, b.w / 60)}px system-ui, sans-serif`;
      ctx.textBaseline = "top";
      for (const it of r.items) {
        const [x1, y1, x2, y2] = [it.box[0] * sx, it.box[1] * sy, it.box[2] * sx, it.box[3] * sy];
        const col = colorOf(it.label);
        ctx.strokeStyle = col; ctx.lineWidth = lw;
        ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
        const text = `${it.label} ${(it.score * 100).toFixed(0)}`;
        const tw = ctx.measureText(text).width + 8, th = parseInt(ctx.font) + 6;
        ctx.fillStyle = col; ctx.fillRect(x1, Math.max(0, y1 - th), tw, th);
        ctx.fillStyle = "#fff"; ctx.fillText(text, x1 + 4, Math.max(0, y1 - th) + 3);
        if (it.keypoints) drawPose(ctx, it.keypoints.map(([x, y, s]) => [x * sx, y * sy, s]), lw);
      }
    }
  }
  for (const [x, y, l] of state.points) {
    ctx.beginPath(); ctx.arc(x, y, lw * 3, 0, Math.PI * 2);
    ctx.fillStyle = l ? "#16a34a" : "#ef4444"; ctx.fill();
    ctx.strokeStyle = "#fff"; ctx.lineWidth = lw; ctx.stroke();
  }
}

function drawPose(ctx, kps, lw) {
  ctx.strokeStyle = "#f59e0b"; ctx.lineWidth = lw;
  for (const [a, b] of SKELETON) {
    ctx.beginPath(); ctx.moveTo(kps[a][0], kps[a][1]); ctx.lineTo(kps[b][0], kps[b][1]); ctx.stroke();
  }
  ctx.fillStyle = "#e0457b";
  for (const [x, y] of kps) { ctx.beginPath(); ctx.arc(x, y, lw * 1.6, 0, Math.PI * 2); ctx.fill(); }
}

// 結果の画像（マスク・深度・色分け）は届いた時に1回だけ表示用の層に変換しておく（描画は毎フレームなので）
async function prepareLayer(r) {
  const src = r.kind === "mask" ? r.mask : r.image;
  if (!src || !["mask", "depth", "segmap"].includes(r.kind)) return;
  const url = src instanceof Blob ? URL.createObjectURL(src) : src;
  const img = await loadImg(url);
  if (src instanceof Blob) URL.revokeObjectURL(url);
  const w = img.naturalWidth, h = img.naturalHeight;
  const layer = (fn) => {
    const c = new OffscreenCanvas(w, h), x = c.getContext("2d");
    x.drawImage(img, 0, 0);
    if (!fn) return c;
    const d = x.getImageData(0, 0, w, h);
    for (let i = 0; i < d.data.length; i += 4) fn(d.data, i, d.data[i] / 255);
    x.putImageData(d, 0, 0);
    return c;
  };
  if (r.kind === "segmap") { r.layer = layer(); return; }
  if (r.kind === "depth") { // jet。赤＝近い、青＝遠い
    r.layer = layer((d, i, t) => {
      d[i] = 255 * clamp01(1.5 - Math.abs(4 * t - 3)); d[i + 1] = 255 * clamp01(1.5 - Math.abs(4 * t - 2));
      d[i + 2] = 255 * clamp01(1.5 - Math.abs(4 * t - 1)); d[i + 3] = 255;
    });
    return;
  }
  r.gray = layer((d, i) => { d[i + 1] = d[i + 2] = d[i]; d[i + 3] = 255; });                  // 白＝前景
  r.alpha = layer((d, i) => { d[i + 3] = d[i]; });                                           // 前景の度合い＝不透明度
  r.tint = layer((d, i, t) => { d[i] = 47; d[i + 1] = 111; d[i + 2] = 223; d[i + 3] = t > 0.5 ? 120 : 0; });
}
const clamp01 = (v) => Math.min(1, Math.max(0, v));

function drawChecker(ctx, w, h) {
  const s = Math.max(8, Math.round(w / 60));
  for (let y = 0; y < h; y += s) for (let x = 0; x < w; x += s) {
    ctx.fillStyle = ((x / s + y / s) & 1) ? "#d0d4da" : "#f2f4f7";
    ctx.fillRect(x, y, s, s);
  }
}

function renderResult(m, r, live) {
  const el = $("result");
  const where = m.where === "browser" ? `ブラウザ（${r.device}, ${r.dtype}）` : `サーバー（${r.device}）`;
  const net = r.roundtrip_ms ? ` / 通信込み ${fmt(r.roundtrip_ms)}` : "";
  const fps = live ? ` ・ <b>${live.fps.toFixed(1)} fps</b>（${live.frames} フレーム）` : "";
  let body = "";
  if (r.kind === "boxes") {
    body = `<div class="dets">${r.items.length} 件: ${summarizeBoxes(r.items)}</div>`;
  } else if (r.kind === "labels") {
    body = r.items.map((it) => `<div class="bar"><span>${esc(it.label)}</span><span class="track"><span class="fill" style="width:${(it.score * 100).toFixed(1)}%"></span></span><span class="num">${(it.score * 100).toFixed(1)}%</span></div>`).join("")
      + (r.items[0]?.abs != null ? `<div class="dets">絶対スコア: ${r.items.map((it) => `${esc(it.label)} ${(it.abs * 100).toFixed(1)}%`).join("、")}</div>` : "");
  } else if (r.kind === "text") {
    body = `<pre>${esc(r.text)}</pre>`;
  } else if (r.kind === "segmap") {
    body = `<div class="dets">${r.count} 領域（${r.prompts} 点のプロンプトから重なりを除いたもの）</div>`;
  } else if (r.kind === "depth" && r.note) {
    body = `<div class="dets">${esc(r.note)}</div>`;
  } else if (r.kind === "mask" && r.score != null) {
    body = `<div class="dets">マスクの推定品質 ${(r.score * 100).toFixed(0)}%（点 ${state.points.length} 個）</div>`;
  }
  el.innerHTML = `<div class="meta">${esc(m.name)} ・ ${where} ・ 読み込み ${fmt(r.load_ms)} ・ 推論 ${fmt(r.infer_ms)}${net}${fps}</div>${body}`;
}

function summarizeBoxes(items) {
  const c = {};
  for (const it of items) c[it.label] = (c[it.label] || 0) + 1;
  return Object.entries(c).map(([k, v]) => `${esc(k)} ×${v}`).join("、") || "なし";
}

const fmt = (ms) => (ms == null ? "-" : ms >= 10000 ? `${(ms / 1000).toFixed(0)}s` : ms >= 1000 ? `${(ms / 1000).toFixed(1)}s` : `${ms.toFixed(0)}ms`);
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" }[c]));

function addHistory(m, r, summaryOverride) {
  const tr = document.createElement("tr");
  const task = TASKS.find((t) => t.id === m.task).name;
  const where = m.where === "browser" ? `ブラウザ ${r.device}` : `サーバー ${r.device}`;
  const summary = summaryOverride ?? (r.kind === "boxes" ? `${r.items.length}件` : r.kind === "labels" ? `${r.items[0]?.label ?? ""}` : r.kind === "text" ? r.text.slice(0, 24) + "…" : r.kind);
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
    auto,
    _imageKey: key,
  };
}

// 1回分。結果を state.result に入れ、描画用の層を用意する
async function runOnce(m) {
  const [w, h] = inferSize();
  const { image, key } = await grabFrame(m.where === "server");
  const params = paramsFor(key, w, state.auto);
  if (params.auto && m.where === "server") throw new Error("全体の自動分割はブラウザの SAM 系モデルのみ");
  const r = m.where === "browser" ? await runInBrowser(m, image, params) : await runOnServer(m, image, params);
  r.w = w; r.h = h;
  await prepareLayer(r);
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
  setStatus(m.where === "browser" ? "連続実行中…（初回はモデルを取得）" : "連続実行中…（サーバー）");
  try {
    while (state.live && state.video) {
      const r = await runOnce(m);
      if (!first) { first = r; tStart = performance.now(); setStatus(""); }
      frames++;
      inferSum += r.infer_ms;
      fps = frames > 1 ? (frames - 1) / ((performance.now() - tStart) / 1000) : 0;
      renderResult(m, r, { fps, frames });
    }
  } catch (e) {
    failed(m, e);
  }
  if (first) addHistory(m, { ...first, infer_ms: inferSum / frames }, `連続 ${frames}フレーム ${fps.toFixed(1)}fps`);
  state.live = false;
  updateButtons();
}

function toggleLive() {
  if (state.live) { stopLive(); return; }
  if (!state.video || !currentModel()) return;
  if (state.task === "segment" && !state.points.length && !state.auto) { setStatus("先に画面をクリックして点を置くか、「全体を自動分割」を選んでください", "warn"); return; }
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
  if (STANDALONE) return;
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
  $("env").textContent = (navigator.gpu ? "WebGPU あり" : "WebGPU なし（WASM で実行、遅い）") + (STANDALONE ? " ・ サーバーなし版" : "");
  if (navigator.gpu) {
    try { if (!(await navigator.gpu.requestAdapter())) $("env").textContent = "WebGPU アダプタなし（WASM で実行）"; } catch { /* noop */ }
  }
  if (STANDALONE) {
    $("server-box").hidden = true;
  } else {
    try {
      const list = await (await fetch("api/models")).json();
      state.serverModels = new Set(list.map((m) => m.key));
    } catch { /* サーバーなし */ }
  }
  $("prompt").value = DEFAULTS.prompt;
  $("threshold").oninput = () => { $("th-out").textContent = $("threshold").value; };
  $("model").onchange = () => { stopLive(); clearResult(); setStatus(""); updateModelNote(); draw(); };
  $("run").onclick = run;
  $("live").onclick = toggleLive;
  $("pause").onclick = () => { const v = $("video"); v.paused ? v.play() : v.pause(); updateButtons(); };
  $("freeze").onclick = freezeFrame;
  $("clear-points").onclick = () => { state.points = []; state.result = null; draw(); };
  $("view").onchange = draw;
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
    if (state.task !== "segment" || state.auto || (!state.image && !state.video) || state.busy) return;
    const [x, y] = canvasPoint(ev);
    state.points.push([x, y, ev.shiftKey || $("negative").checked ? 0 : 1]);
    draw();
    if (!state.live) run();
  });
  selectTask("detect");
  await setImage(document.querySelector("[data-sample]").dataset.sample);
  refreshServer();
  if (!STANDALONE) setInterval(refreshServer, 15000);
}

init();
