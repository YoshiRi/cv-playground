import { CATALOG, MODELS, TASKS } from "./catalog.js";
import { esc, fmtMs as fmt, KINDS } from "./renderers.js";
import { collectEnv, composeImage, download, resultData, safeName, shareOrDownload, stamp, stats, toCSV, toJSON, toMarkdown } from "./export.js";
import { Tracker } from "./tracker.js";
import { APPS } from "./apps.js";

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
// 詳細計測: URL に ?profile=1 を付けた時だけ。汎用 ONNX のブラウザ実行で onnxruntime の profiler を有効にし、ベンチに GPU の内訳を付ける
// （計測の手間で遅くなるので通常は切る。記録は実行のたびにたまる）
const PROFILE = new URLSearchParams(location.search).has("profile");
// 汎用 ONNX の WebGPU 実行の前処理（比べる用に URL で選べる）: ?pre=upload（既定。縮小は canvas、正規化などは GPU）/
// ?pre=gpu（縮小も GPU）/ ?pre=cpu（全部 CPU。?cpupre=1 も同じ）
const PRE_MODE = new URLSearchParams(location.search).has("cpupre") ? "cpu" : new URLSearchParams(location.search).get("pre") || "upload";
// ?nopipe=1: 連続実行をパイプライン化しない（1フレームずつ順番。比べる用）
const NO_PIPE = new URLSearchParams(location.search).has("nopipe");

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
  runs: [],          // 実行の記録（init で localStorage から読む）
  bench: false,      // ベンチマーク中
  category: null,    // 選んでいるタブの分類（models.json の categories）
  auto: false,       // クリックで切り出しの「全体を自動分割」
  apps: [],          // 選んだ応用 [{id, st}]（apps.js）。連続実行の開始時と静止画の1回ごとに作り直す
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
    showProgress(d.file?.split("/").pop() ?? "", d.progress);
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
    const profile = PROFILE && model.adapter === "onnx";
    (workers[lib] ?? startWorker(lib)).postMessage({ id, model: { ...model, opt, profile, preMode: PRE_MODE, webRoot: WEB_ROOT }, image, params }, transfer);
  });
}

// 詳細計測の結果（最後の last 回ぶん）を受け取る。そのモデルは Worker で捨てられ、次の実行で読み直す
function profileInBrowser(model, last) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    (workers.ort ?? startWorker("ort")).postMessage({ type: "profile", id, model, last });
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

// ---------- ダウンロードの進み具合と、大きいモデルの確認 ----------

let progressTimer = 0;
function showProgress(file, pct) {
  $("progress").hidden = false;
  $("progress-bar").value = pct ?? 0;
  $("progress-text").textContent = `モデルを取得中 ${file} ${pct != null ? pct.toFixed(0) + "%" : ""}`;
  clearTimeout(progressTimer);
  progressTimer = setTimeout(() => { $("progress").hidden = true; }, 1500); // 取得が終われば消える
}

// ブラウザ実行のモデルは初回に端末へダウンロードする。100MB を超えるものは、取得済みでなければ先に確かめる
// （取得済みかどうかは、一度読み込めたモデルを localStorage に覚えておく。ブラウザのキャッシュを消すと忘れる）
const CONFIRM_MB = 100, DL_KEY = "cvpg-downloaded";
const downloaded = () => { try { return new Set(JSON.parse(localStorage.getItem(DL_KEY) || "[]")); } catch { return new Set(); } };
function markDownloaded(key) {
  try { const s = downloaded(); s.add(key); localStorage.setItem(DL_KEY, JSON.stringify([...s])); } catch { /* 保存できない環境 */ }
}
function confirmDownload(m) {
  if (m.where !== "browser" || !(m.mb >= CONFIRM_MB) || downloaded().has(m.key)) return true;
  const size = m.mb >= 1000 ? `${(m.mb / 1000).toFixed(1)}GB` : `${m.mb}MB`;
  return confirm(`「${m.name}」を初めて使うので、この端末にモデルをダウンロードします（約${size}）。\nモバイル回線では通信量に注意してください。2回目以降はダウンロードしません。\n\n続けますか？`);
}

async function showStorage() {
  try {
    const { usage } = await navigator.storage.estimate();
    $("storage").textContent = `このサイトが端末に保存している量: 約${(usage / 1e6).toFixed(0)}MB`;
  } catch { $("storage").textContent = ""; }
}
async function clearDownloads() {
  if (!confirm("ダウンロード済みのモデル（ブラウザのキャッシュ）を消します。次に使う時にまたダウンロードします。")) return;
  stopLive();
  for (const lib of Object.keys(workers)) restartWorker(lib);
  try { for (const k of await caches.keys()) await caches.delete(k); } catch { /* Cache API が無い環境 */ }
  try { localStorage.removeItem(DL_KEY); } catch { /* noop */ }
  selectTask(state.task);
  showStorage();
}

// ---------- 手元の ONNX を使う ----------
// 選んだファイルの SHA-256 が models.json の onnx.sha256（tools/update_hashes.py が Hugging Face から取る）と
// 一致したら、onnx_generic.js がダウンロードの時に使うキャッシュ（cvpg-onnx）に同じ URL で入れる。
// 以後はダウンロード済みと同じ扱いになる。一致しないファイルは使わない（前処理・後処理が合う保証が無いため）
const HF = "https://huggingface.co";
const hfUrl = (e, f) => `${HF}/${e.onnx.repo}/resolve/main/${f}`;
const hashedModels = () => MODELS.filter((e) => e.adapter === "onnx" && e.onnx?.sha256 && e.where.includes("browser"));

function renderLocalKnown() {
  $("local-known").innerHTML = hashedModels().map((e) =>
    `<li>${esc(e.name)}: ${Object.keys(e.onnx.sha256).map((f) => `<code>${esc(e.onnx.repo)}/${esc(f)}</code>`).join("、")}</li>`).join("");
}

async function sha256Hex(buf) {
  const d = new Uint8Array(await crypto.subtle.digest("SHA-256", buf));
  return Array.from(d, (b) => b.toString(16).padStart(2, "0")).join("");
}

async function importLocalOnnx(files) {
  const out = [], status = $("local-status");
  let cache;
  try { cache = await caches.open("cvpg-onnx"); } catch { cache = null; }
  if (!cache || !crypto.subtle) {
    status.textContent = "この開き方（file:// など）ではブラウザのキャッシュが使えないので取り込めない。GitHub Pages かサーバー版で開く";
    return;
  }
  const index = new Map();
  for (const e of hashedModels()) for (const [f, h] of Object.entries(e.onnx.sha256)) {
    if (!index.has(h)) index.set(h, []);
    index.get(h).push({ e, f });
  }
  const touched = new Set();
  for (const [i, file] of [...files].entries()) {
    status.textContent = `照合中 ${i + 1}/${files.length}: ${file.name}`;
    let buf;
    try { buf = await file.arrayBuffer(); } catch (err) { out.push(`✗ ${esc(file.name)}: 読めない（${esc(err.message)}）`); continue; }
    const hex = await sha256Hex(buf), hits = index.get(hex);
    if (!hits) { out.push(`✗ ${esc(file.name)}: 一致するモデルが無い（SHA-256 <code>${hex.slice(0, 16)}…</code>）`); continue; }
    for (const { e, f } of hits) {
      try { await cache.put(hfUrl(e, f), new Response(buf)); } catch (err) { out.push(`✗ ${esc(file.name)}: 保存できない（${esc(err.message)}）`); continue; }
      touched.add(e);
      out.push(`✓ ${esc(file.name)} → ${esc(e.name)} の <code>${esc(f)}</code>`);
    }
  }
  // そのモデルの既定のファイル（と外部データ）がそろえばダウンロード済みにする。fp16 版だけなら実行設定で fp16 を選んだ時に使う
  for (const e of touched) {
    const need = [e.onnx.file, e.onnx.data].filter(Boolean);
    const have = await Promise.all(need.map(async (f) => !!(await cache.match(hfUrl(e, f)))));
    const missing = need.filter((_, k) => !have[k]);
    if (!missing.length) { markDownloaded(e.key); out.push(`　${esc(e.name)}: ダウンロードせずに使える`); }
    else if (e.onnx.file_fp16 && (await cache.match(hfUrl(e, e.onnx.file_fp16))) && !e.onnx.data) out.push(`　${esc(e.name)}: fp16 版だけ（「実行設定」で fp16 を選ぶとダウンロードせずに使える）`);
    else out.push(`　${esc(e.name)}: まだ足りない（${missing.map((f) => `<code>${esc(f)}</code>`).join("、")}）`);
  }
  $("local-result").innerHTML = out.map((l) => `<li>${l}</li>`).join("");
  status.textContent = `${files.length} ファイルを照合した`;
  updateModelNote();
  showStorage();
}

// タブは分類（models.json の categories）ごとにまとめ、選んだ分類のタブだけを出す
function renderTasks() {
  const avail = TASKS.filter((t) => variants(t.id).some((v) => !v.avoid));
  const cats = (CATALOG.categories || []).filter((c) => avail.some((t) => t.category === c.id));
  $("cats").innerHTML = "";
  for (const c of cats) {
    const b = document.createElement("button");
    b.type = "button";
    b.textContent = c.name;
    b.setAttribute("role", "tab");
    b.setAttribute("aria-selected", String(c.id === state.category));
    b.onclick = () => selectTask(avail.find((t) => t.category === c.id).id);
    $("cats").append(b);
  }
  $("cats").hidden = cats.length < 2;
  $("tasks").innerHTML = "";
  for (const t of avail) {
    if (cats.length > 1 && t.category !== state.category) continue;
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
  state.category = TASKS.find((x) => x.id === id).category;
  state.auto = false;
  state.result = null;
  state.points = [];
  state.apps = [];
  const t = TASKS.find((x) => x.id === id);
  renderTasks();
  $("task-hint").textContent = t.hint;
  for (const el of document.querySelectorAll("[data-param]")) el.hidden = !t.params.includes(el.dataset.param);
  $("track-hint").hidden = !t.params.includes("track") || !$("tracker").value;
  const def = t.defaults || {};
  if (def.threshold != null) { $("threshold").value = def.threshold; $("th-out").textContent = def.threshold; }
  if (def.labels) $("labels").value = def.labels;
  renderApps(t);
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
// どのライブラリ・実装で動くか（デバッグ用に説明欄に出す。実際に使った設定は結果欄と記録の「実行場所」に出る）
const TJS_VERSIONS = { 3: "3.8.1", 4: "4.3.0" };
function runtimeLabel(m) {
  if (m.where === "server") return `サーバーの adapters.py（${m.adapter}）`;
  if (m.adapter === "onnx") return "onnxruntime-web を直接（汎用 ONNX。前処理・後処理は models.json）";
  return `transformers.js ${TJS_VERSIONS[m.lib || 4]}（${m.adapter}）`;
}

let lastModelId = null; // 前に選んでいたモデル（入力サイズの既定をモデルを替えた時だけ入れる）
function updateModelNote() {
  const m = currentModel();
  if (!m) return;
  if (!$("prompt").value || DEFAULT_PROMPTS.has($("prompt").value)) $("prompt").value = m.prompt || TASK_PROMPT(m.task);
  const notes = [];
  notes.push(m.repo || m.onnx?.repo || m.ollama || "");
  notes.push(`実行: ${runtimeLabel(m)}`);
  if (m.where === "browser") notes.push(downloaded().has(m.key) ? "取得済み" : m.mb >= CONFIRM_MB ? "初回のダウンロードが大きい（モバイル回線では注意）" : "");
  if (m.where === "server") notes.push("画像をサーバーに送って処理する");
  if (m.license) notes.push(`ライセンス: ${m.license}`);
  if (m.avoid) notes.push(m.avoid);
  $("ort-opt-row").hidden = !(m.where === "browser" && m.adapter === "onnx");
  $("input-size-row").hidden = !m.pre?.dynamic;
  // 入力サイズ可変のモデルを選んだ時は、そのモデルの標準の長辺（pre.size[0]。深度は 518、YOLO は 640）を既定にする
  if (m.pre?.dynamic && m.id !== lastModelId && [...$("input-size").options].some((o) => +o.value === m.pre.size[0])) {
    $("input-size").value = String(m.pre.size[0]);
    $("input-size").onchange?.();
  }
  lastModelId = m.id;
  $("advanced").hidden = $("ort-opt-row").hidden && $("input-size-row").hidden;
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
  $("result-actions").hidden = !state.result;
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
  if (r) for (const a of state.apps) APPS[a.id].draw?.(ctx, a.st, r, b);
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
  const body = (KINDS[r.kind]?.panel(r, { live, points: state.points.length }) ?? "")
    + state.apps.map((a) => APPS[a.id].panel?.(a.st) ?? "").join("");
  $("result").innerHTML = `<div class="result-head"><b>${esc(m.name)}</b><span class="badge ${m.where}">${esc(where)}</span></div>`
    + `<div class="stats">${stats}</div>${bd}${live ? `<div class="sub muted">${live.frames} フレーム</div>` : ""}<div class="result-body">${body}</div>`;
}

// ---------- 実行の記録（実行履歴・CSV 書き出し・ベンチマーク） ----------

// 1回の実行（連続実行・ベンチマークは1まとめ）を1件の記録にする。列は export.js の RUN_COLUMNS
function makeRecord(m, r, mode, extra = {}) {
  const task = TASKS.find((t) => t.id === m.task).name;
  const bd = r.breakdown || {};
  return {
    time: new Date().toISOString(), mode, task, model_key: m.key, model_name: m.name, where: m.where === "browser" ? "ブラウザ" : "サーバー",
    device: r.device, runtime: m.where === "browser" ? r.dtype || "" : "", input_size: m.pre?.dynamic ? parseInt($("input-size").value, 10) : "",
    frame_w: r.w, frame_h: r.h, load_ms: r.load_ms, infer_ms: r.infer_ms, grab_ms: bd.grab, pre_ms: bd.pre, run_ms: bd.run, post_ms: bd.post,
    roundtrip_ms: r.roundtrip_ms, reid_ms: r.reid_ms, cascade_ms: r.cascade_ms, summary: KINDS[r.kind]?.summary(r) ?? r.kind, ...extra,
  };
}

// 記録はブラウザ内（localStorage）に最新 500 件まで残す（再読み込みしても消えない）
const RUNS_KEY = "cvpg-runs", MAX_RUNS = 500;
function loadRuns() { try { return JSON.parse(localStorage.getItem(RUNS_KEY) || "[]"); } catch { return []; } }
function saveRuns() { try { localStorage.setItem(RUNS_KEY, JSON.stringify(state.runs.slice(-MAX_RUNS))); } catch { /* 保存できない環境 */ } }

function addRun(rec) {
  state.runs.push(rec);
  if (state.runs.length > MAX_RUNS) state.runs.shift();
  saveRuns();
  $("history").prepend(historyRow(rec));
  $("history-count").textContent = `${state.runs.length} 件`;
}

function historyRow(rec) {
  const tr = document.createElement("tr");
  const mode = { single: "", live: "連続 ", bench: "ベンチ " }[rec.mode] ?? "";
  const infer = rec.infer_median_ms ?? rec.infer_ms;
  const extra = rec.mode === "single" ? rec.summary : `${mode}${rec.frames ?? ""}回${rec.fps ? ` ${Number(rec.fps).toFixed(1)}fps` : ""}${rec.infer_p90_ms != null ? ` p90 ${fmt(rec.infer_p90_ms)}` : ""}${rec.summary ? ` ・ ${rec.summary}` : ""}`;
  tr.innerHTML = `<td>${new Date(rec.time).toLocaleTimeString()}</td><td>${esc(rec.task)}</td><td>${esc(rec.model_name)}</td><td>${esc(rec.where)} ${esc(rec.device ?? "")}</td>`
    + `<td>${rec.input_size ? `入力 ${rec.input_size}` : `${rec.frame_w}×${rec.frame_h}`}</td><td class="num">${fmt(rec.load_ms)}</td><td class="num">${fmt(infer)}${rec.infer_median_ms != null ? "<small>（中央値）</small>" : ""}</td><td>${esc(extra)}</td>`;
  return tr;
}

function renderHistory() {
  $("history").innerHTML = "";
  for (const rec of state.runs) $("history").prepend(historyRow(rec));
  $("history-count").textContent = state.runs.length ? `${state.runs.length} 件` : "";
}

const variantName = () => (STANDALONE ? "1ファイル版" : state.hasServer ? "サーバー版" : "静的版");

async function exportRuns(kind) {
  if (!state.runs.length) { setStatus("書き出す記録がまだ無い", "warn"); return; }
  const env = await collectEnv(variantName());
  const base = `cv-playground_runs_${safeName(env.device_model || env.os || "device")}_${stamp()}`;
  if (kind === "csv") download(new Blob([toCSV(state.runs, env)], { type: "text/csv" }), base + ".csv");
  if (kind === "json") download(new Blob([toJSON(state.runs, env)], { type: "application/json" }), base + ".json");
  if (kind === "md") {
    const md = toMarkdown(state.runs, env);
    try { await navigator.clipboard.writeText(md); setStatus("Markdown の表をコピーした"); }
    catch { download(new Blob([md], { type: "text/markdown" }), base + ".md"); }
  }
}

// ---------- 速度を測る（ベンチマーク） ----------
// 決まった画像で、ウォームアップ 3 回のあと N 回測る。端末ごとに同じ条件で比べられるようにする

const BENCH_IMAGE = "https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/city-streets.jpg";
const BENCH_WARMUP = 3;

// 測れるモデル: クリックで点を置くタブ（条件が決まらない）以外の、使えるモデル全部。
// 既定で選ぶのは models.json で bench: true の軽い代表（ブラウザ実行のみ）
function benchCandidates() {
  return TASKS.filter((t) => !t.click).flatMap((t) => variants(t.id).filter((v) => !v.avoid && v.ready).map((v) => ({ t, v })));
}

function renderBench() {
  const groups = {};
  for (const { t, v } of benchCandidates()) (groups[t.name] ??= []).push(v);
  $("bench-models").innerHTML = Object.entries(groups).map(([name, vs]) => `<fieldset><legend>${esc(name)}</legend>${vs.map((v) => {
    const on = v.where === "browser" && !!v.bench;
    const size = v.where === "browser" ? `${v.mb >= 1000 ? (v.mb / 1000).toFixed(1) + "GB" : v.mb + "MB"}` : "サーバー";
    return `<label class="check"><input type="checkbox" data-bench="${esc(v.id)}"${on ? " checked" : ""}> ${esc(v.name)} <small class="muted">${size}</small></label>`;
  }).join("")}</fieldset>`).join("");
  updateBenchCount();
}
function updateBenchCount() {
  const n = document.querySelectorAll("[data-bench]:checked").length;
  $("bench-summary").textContent = `測るモデルを選ぶ（${n} 個を選択中）`;
}

async function runBench() {
  if (state.bench) { state.bench = false; return; } // 中止
  const pick = new Set([...document.querySelectorAll("[data-bench]")].filter((c) => c.checked).map((c) => c.dataset.bench));
  const list = benchCandidates().filter(({ v }) => pick.has(v.id));
  if (!list.length) { setStatus("測るモデルを選んでください", "warn"); return; }
  const N = parseInt($("bench-runs").value, 10);
  state.bench = true;
  $("bench-start").textContent = "■ 中止";
  stopLive();
  await setImage(BENCH_IMAGE);
  const done = [], profiles = [];
  $("bench-profile").innerHTML = "";
  for (let i = 0; i < list.length && state.bench; i++) {
    const { t, v } = list[i];
    const def = t.defaults || {};
    const overrides = { threshold: def.threshold ?? 0.4, labels: def.labels ?? "", prompt: v.prompt || def.prompt || "", points: [], auto: false };
    const times = [], bd = { grab: [], pre: [], run: [], post: [] };
    let first = null, r = null;
    try {
      for (let k = 0; k < BENCH_WARMUP + N && state.bench; k++) {
        $("bench-progress").textContent = `${i + 1}/${list.length} ${v.name}（${v.where === "browser" ? "ブラウザ" : "サーバー"}）: ${k < BENCH_WARMUP ? `ウォームアップ ${k + 1}/${BENCH_WARMUP}` : `${k - BENCH_WARMUP + 1}/${N} 回`}`;
        r = await runOnce(v, overrides);
        if (!first) first = r;
        if (k >= BENCH_WARMUP) {
          times.push(r.roundtrip_ms || r.infer_ms);
          for (const key of Object.keys(bd)) if (r.breakdown?.[key] != null) bd[key].push(r.breakdown[key]);
        }
      }
    } catch (e) {
      setStatus(`${v.name}: ${e.message}`, "err");
      continue;
    }
    if (!times.length) continue;
    let profile;
    if (PROFILE && v.where === "browser" && v.adapter === "onnx") {
      try { profile = await profileInBrowser(v, times.length); } catch (e) { profile = { error: e.message }; }
      profiles.push({ name: v.name, runtime: r.dtype, ...profile });
      renderProfiles(profiles);
    }
    const st = stats(times), avg = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : undefined);
    addRun(makeRecord(v, r, "bench", {
      load_ms: first.load_ms, infer_ms: st.mean, grab_ms: avg(bd.grab), pre_ms: avg(bd.pre), run_ms: avg(bd.run), post_ms: avg(bd.post),
      frames: st.n, fps: +(1000 / st.mean).toFixed(2), infer_mean_ms: st.mean, infer_median_ms: st.median, infer_p90_ms: st.p90, infer_p95_ms: st.p95,
      gpu_ms: profile?.gpu_ms, ...(profile ? { profile } : {}), infer_min_ms: st.min, infer_max_ms: st.max, warmup: BENCH_WARMUP, bench_image: BENCH_IMAGE.split("/").pop(),
    }));
    done.push(v.name);
    renderResult(v, r);
    draw();
  }
  $("bench-progress").textContent = state.bench ? `完了: ${done.length} モデル（結果は実行履歴に「ベンチ」として追加。CSV で書き出せる）` : `中止した（${done.length} モデル分を記録）`;
  state.bench = false;
  $("bench-start").textContent = "▶ 測る";
}

// 詳細計測の表: モデルごとに ONNX の実行・GPU の命令の合計・転送と、GPU 時間の大きい演算
function renderProfiles(ps) {
  const f = (v) => (v == null ? "" : fmt(v));
  $("bench-profile").innerHTML = ps.map((p) => p.error || p.note ? `<p class="small">${esc(p.name)}: ${esc(p.error || `ONNX の実行 ${f(p.run_ms)}。${p.note}`)}</p>` : `<div class="prof">
    <div class="small"><b>${esc(p.name)}</b> <span class="muted">${esc(p.runtime || "")}・${p.runs} 回の平均</span></div>
    <div class="small">ONNX の実行 ${f(p.run_ms)} ・ GPU の命令の合計 ${f(p.gpu_ms)}（${p.dispatches} 個）・ 入力の転送 ${f(p.upload_ms)} ・ 結果の待ちと読み戻し ${f(p.readback_wait_ms)}</div>
    <div class="small">CPU に回ったノード: ${p.cpu_nodes.length ? esc(p.cpu_nodes.join("、")) : "なし"}</div>
    <table class="small"><tr><th>演算</th><th class="num">GPU ms</th><th class="num">個数</th><th class="num">割合</th></tr>${p.ops.slice(0, 8).map((o) => `<tr><td>${esc(o.op)}</td><td class="num">${f(o.ms)}</td><td class="num">${o.count}</td><td class="num">${o.pct}%</td></tr>`).join("")}</table>
  </div>`).join("");
}

// ---------- 結果の保存（表示中の画像・結果データ） ----------

async function saveImage() {
  const m = currentModel(), r = state.result;
  if (!r || !m) return;
  const lines = $("caption").checked ? [
    `${TASKS.find((t) => t.id === m.task).name} ・ ${m.name}`,
    `${m.where === "browser" ? "ブラウザ" : "サーバー"} ${r.device}${r.dtype ? " " + r.dtype : ""} ・ 推論 ${fmt(r.infer_ms)} ・ ${new Date().toLocaleString()} ・ CV Playground`,
  ] : null;
  const blob = await composeImage($("canvas"), lines);
  const how = await shareOrDownload(blob, `cv-playground_${safeName(TASKS.find((t) => t.id === m.task).name)}_${safeName(m.key)}_${stamp()}.png`);
  if (how === "downloaded") setStatus("画像を保存した");
}

function saveResultData() {
  const m = currentModel(), r = state.result;
  if (!r || !m) return;
  const data = resultData(r, m, { task: m.task, time: new Date().toISOString(), view: $("view").value, image_size: [r.w, r.h] });
  download(new Blob([JSON.stringify(data, null, 1)], { type: "application/json" }), `cv-playground_${safeName(m.key)}_${stamp()}.json`);
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

// 1回分。結果を state.result に入れ（commit: false なら入れない。パイプライン化した連続実行で、前のフレームの結果を表示中に
// 次のフレームの結果で上書きしないように）、描画用の層を用意する
async function runOnce(m, overrides = {}, { commit = true } = {}) {
  const [w, h] = inferSize();
  const tg = performance.now();
  const { image, key } = await grabFrame(m.where === "server");
  const grabMs = performance.now() - tg;
  const params = { ...paramsFor(key, w, state.auto), ...overrides };
  if (params.auto && m.where === "server") throw new Error("全体の自動分割はブラウザの SAM 系モデルのみ");
  if (!confirmDownload(m)) throw new Error("ダウンロードを取りやめた");
  const r = m.where === "browser" ? await runInBrowser(m, image, params) : await runOnServer(m, image, params);
  r.w = w; r.h = h;
  if (m.where === "browser" && !downloaded().has(m.key)) { markDownloaded(m.key); updateModelNote(); showStorage(); }
  $("progress").hidden = true;
  if (r.breakdown) r.breakdown = { grab: grabMs, ...r.breakdown };
  await KINDS[r.kind]?.prepare?.(r);
  if (commit) state.result = r;
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
    state.apps = createApps();
    applyApps(r, { tracked: false });
    renderResult(m, r);
    addRun(makeRecord(m, r, "single", appsSummary(KINDS[r.kind]?.summary(r))));
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
  let frames = 0, first = null, tStart = 0, fps = 0;
  const times = []; // 1フレーム目（モデルの読み込み・初期化を含む）を除いた、フレームごとの推論時間
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
  state.apps = createApps();
  setStatus(m.where === "browser" ? "連続実行中…（初回はモデルを取得）" : "連続実行中…（サーバー）");
  const overrides = tracker ? { threshold: Math.min(th, tracker.args.track_low_thresh) } : {};
  // 1フレームの結果を追跡・cascade にかけて表示する（フレームの順に呼ぶ）
  const handle = async (r) => {
    if (tracker) {
      const feats = reid ? await reidFeatures(reid, r.items, tracker.args.track_low_thresh) : null;
      if (feats) r.reid_ms = feats.ms;
      r.items = tracker.update(r.items, feats?.feats);
      r.items.forEach((it) => ids.add(it.id));
    }
    state.result = r;
    await applyCascade(m, r, tracker ? seqStore : null);
    applyApps(r, { tracked: !!tracker });
    if (!first) { first = r; tStart = performance.now(); setStatus(""); }
    frames++;
    if (frames > 1) times.push(r.roundtrip_ms || r.infer_ms);
    fps = frames > 1 ? (frames - 1) / ((performance.now() - tStart) / 1000) : 0;
    renderResult(m, r, { fps, frames, ids: tracker ? ids.size : 0, trackerName });
    if ($("video").paused) draw();
  };
  // パイプライン化（汎用 ONNX のブラウザ実行）: 前のフレームの推論中に、次のフレームを取り込んで Worker に送っておく。
  // Worker は届いた順に続けて実行するので、結果の受け渡し・追跡・描画の間も GPU が休まない（GPU は間が空くと遅くなる）。
  // 同時に送るのは2フレームまで。結果はフレームの順に処理する
  const pipe = m.where === "browser" && m.adapter === "onnx" && !NO_PIPE;
  // 次のフレームの表示は、送った直後から待ち始める。前の結果の処理（追跡・描画）の間に表示されても取りこぼさない
  // （処理が1フレームの間隔より少しでも長いと、待ち始めが遅れて1つおきになり、30fps の動画で 15fps に落ちていた）
  let inflight = null, next = null;
  try {
    while (state.live && state.video) {
      if (frames || inflight) await (next ?? nextVideoFrame($("video")));
      next = null;
      if (!state.live) break;
      const cur = runOnce(m, overrides, { commit: !pipe });
      if (!pipe) { await handle(await cur); continue; }
      cur.catch(() => {}); // 失敗は下で await した時に扱う
      next = nextVideoFrame($("video"));
      if (inflight) await handle(await inflight);
      inflight = cur;
    }
    if (inflight && state.video) await handle(await inflight);
  } catch (e) {
    failed(m, e);
  }
  if (first) {
    const st = stats(times);
    addRun(makeRecord(m, state.result || first, "live", {
      load_ms: first.load_ms, infer_ms: st?.mean ?? first.infer_ms, frames, fps: +fps.toFixed(2),
      infer_mean_ms: st?.mean, infer_median_ms: st?.median, infer_p90_ms: st?.p90, infer_p95_ms: st?.p95, infer_min_ms: st?.min, infer_max_ms: st?.max, warmup: 1,
      ...appsSummary(tracker ? `${trackerName.split("（")[0]} ID ${ids.size}個` : ""),
    }));
  }
  state.live = false;
  updateButtons();
}

// 応用（apps.js）: models.json の apps のうち、タブの結果の種類（tasks[].result）を受け付けるものを「応用」欄にチェックで出す。
// 選んだものを追跡・cascade のあとに順に呼ぶ。URL の ?apps=count で最初から選んでおける
// 選んだ応用はタブを切り替えても覚えておく（別のタブで同じ種類の結果なら、そのまま効く）
const chosenApps = new Set((new URLSearchParams(location.search).get("apps") || "").split(",").filter(Boolean));
function renderApps(t) {
  const list = (CATALOG.apps || []).filter((a) => t.result && a.accepts.includes(t.result) && APPS[a.id]);
  $("apps-row").hidden = !list.length;
  $("apps").innerHTML = list.map((a) =>
    `<label class="check" title="${esc(a.hint || "")}"><input type="checkbox" data-app="${a.id}"${chosenApps.has(a.id) ? " checked" : ""}> ${esc(a.name)}</label>`).join("");
  showAppParams();
}
function onAppsChange(ev) {
  const c = ev.target.closest("[data-app]");
  if (c) c.checked ? chosenApps.add(c.dataset.app) : chosenApps.delete(c.dataset.app);
  showAppParams();
}
// 選んだ応用が使う設定欄だけを出す（data-app-param）
function showAppParams() {
  const used = new Set(checkedApps().flatMap((a) => a.params || []));
  for (const el of document.querySelectorAll("[data-app-param]")) el.hidden = !used.has(el.dataset.appParam);
}
const checkedApps = () => [...document.querySelectorAll("[data-app]:checked")].map((c) => (CATALOG.apps || []).find((a) => a.id === c.dataset.app));
function createApps() {
  const opts = { classes: $("classes").value };
  return checkedApps().map((a) => ({ id: a.id, st: APPS[a.id].create(opts) }));
}
function applyApps(r, ctx) {
  for (const a of state.apps) APPS[a.id].update(a.st, r, ctx);
}
// 実行履歴の「結果」欄: base（モデルの結果の要約や追跡の ID 数）に応用の集計を足す
const appsSummary = (base) => ({ summary: [base, ...state.apps.map((a) => APPS[a.id].summary?.(a.st))].filter(Boolean).join(" ・ ") });

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
    // 入力の形が変わると、Worker がその形でセッションを作り直す（onnx_generic.js の onnxRun）。フレームの処理解像度もモデル入力以上にそろえる
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
  $("clear-cache").onclick = clearDownloads;
  $("save-image").onclick = saveImage;
  $("save-data").onclick = saveResultData;
  $("export-csv").onclick = () => exportRuns("csv");
  $("export-json").onclick = () => exportRuns("json");
  $("export-md").onclick = () => exportRuns("md");
  $("clear-runs").onclick = () => {
    if (!state.runs.length || !confirm("実行の記録（この端末に保存している分）を消します。")) return;
    state.runs = []; saveRuns(); renderHistory();
  };
  $("bench-start").onclick = runBench;
  $("bench-profile-note").hidden = !PROFILE;
  $("bench-models").addEventListener("change", updateBenchCount);
  $("apps").addEventListener("change", onAppsChange);
  $("local-onnx").onchange = (ev) => { const fs = [...ev.target.files]; ev.target.value = ""; if (fs.length) importLocalOnnx(fs); };
  renderLocalKnown();
  state.runs = loadRuns();
  renderHistory();
  renderBench();
  showStorage();
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
