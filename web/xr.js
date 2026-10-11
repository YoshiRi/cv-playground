// AR の試し（xr.html）: WebXR の immersive-ar（Android の Chrome では ARCore）で、
//   1. カメラの映像（camera-access）を WebGL のテクスチャで受け取り、縮めて読み戻して、物体検出（YOLO26n、worker.js）に渡す
//   2. 深度（depth-sensing、CPU で読める形）から、枠の中の深度の中央値を引いて、物までの距離を m で出す
//   3. 使えた機能・カメラの画像の大きさ・XR の fps・読み戻しと推論の時間を画面の上に出す
// カメラの画像は XRView と同じ範囲・向きなので、画像の 0〜1 の座標がそのまま画面（DOM overlay）と深度の座標になる
import { CATALOG } from "./catalog.js";
import { startTranslate, tx } from "./i18n.js";

startTranslate();
const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const MODEL = CATALOG.models.find((m) => m.key === (q.get("model") || "yolo26n"));
const IN_W = +q.get("w") || 640; // 検出に渡す画像の幅（高さはカメラの縦横比）
const TH = +q.get("th") || 0.4;
let flipY = q.get("flipy") !== "0";

// ---------- 検出（worker.js をそのまま使う。前処理・後処理は普段の画面と同じ） ----------
const WEB_ROOT = new URL("./", location.href).href;
let mirror = null; // サーバー版ならモデルをサーバーの写しから取る
const worker = new Worker("worker.js", { type: "module", name: "ort" });
let seq = 0;
const pending = new Map();
worker.onmessage = (ev) => {
  const d = ev.data;
  if (d.type === "progress") { status.load = `${d.file?.split("/").pop()} ${Math.floor(d.progress)}%`; return; }
  const p = pending.get(d.id);
  if (!p) return;
  pending.delete(d.id);
  d.type === "error" ? p.reject(new Error(d.message)) : p.resolve(d.result);
};
function detect(bitmap) {
  const id = ++seq;
  return new Promise((resolve, reject) => {
    pending.set(id, { resolve, reject });
    worker.postMessage({ id, model: { ...MODEL, opt: q.get("opt") ?? "fp16-graph", preMode: "upload", postGpu: true, webRoot: WEB_ROOT, mirror }, image: bitmap, params: { threshold: TH } }, [bitmap]);
  });
}

// ---------- 使えるかの確認 ----------
const status = { load: "" };
async function checkSupport() {
  const lines = [];
  lines.push(`${tx("https のページ")}: ${window.isSecureContext ? "OK" : tx("いいえ（AR には https が要る）")}`);
  lines.push(`navigator.xr: ${navigator.xr ? "OK" : tx("無い（Android の Chrome で開く）")}`);
  let ok = false;
  if (navigator.xr) {
    try { ok = await navigator.xr.isSessionSupported("immersive-ar"); } catch (e) { lines.push(`isSessionSupported: ${e.message}`); }
    lines.push(`immersive-ar: ${ok ? "OK" : tx("使えない")}`);
  }
  lines.push(`WebGPU: ${navigator.gpu ? "OK" : tx("無い（検出は WASM で遅い）")}`);
  lines.push(`${tx("モデル")}: ${tx("{name}（{mb} MB）", { name: MODEL?.name ?? q.get("model"), mb: MODEL?.mb ?? "?" })}`);
  try { if ((await fetch("api/models")).ok) { mirror = new URL("mirror/", WEB_ROOT).href; lines.push(tx("サーバー版（モデルはサーバーの写しから）")); } } catch { /* 静的版 */ }
  $("support").textContent = lines.join("\n");
  $("start").disabled = !ok || !MODEL;
  // ?selftest: AR なしで、見本の画像を検出の経路（Worker）に通して確かめる
  if (q.has("selftest")) {
    const img = await createImageBitmap(await (await fetch("https://huggingface.co/datasets/Xenova/transformers.js-docs/resolve/main/city-streets.jpg")).blob());
    const W = img.width, r = await detect(img);
    $("support").textContent += `\nselftest: ${r.items.length} items, ${r.infer_ms?.toFixed(1)} ms, ${r.items.slice(0, 3).map((it) => `${it.label} ${it.box.map((v) => (v / W).toFixed(2)).join(",")}`).join(" / ")}`;
  }
}

// ---------- AR のセッション ----------
let session, refSpace, gl, binding, prog, fbo, fboTex, vao, pixels;
let inW = 0, inH = 0, busy = false, boxes = [], lastSize = "";
const times = { infer: [], read: [], frame: [] };
const counts = { xr: 0, det: 0, t0: 0, xrFps: 0, detFps: 0 };
let depthNote = "", granted = [];

async function start() {
  $("overlay").classList.add("on");
  try {
    session = await navigator.xr.requestSession("immersive-ar", {
      requiredFeatures: ["local"],
      optionalFeatures: ["camera-access", "depth-sensing", "dom-overlay"],
      domOverlay: { root: $("overlay") },
      depthSensing: { usagePreference: ["cpu-optimized"], dataFormatPreference: ["luminance-alpha", "float32"] },
    });
  } catch (e) {
    $("overlay").classList.remove("on");
    $("support").textContent += `\n${tx("AR を始められない")}: ${e.message}`;
    return;
  }
  granted = [...(session.enabledFeatures || [])];
  const c = document.createElement("canvas");
  gl = c.getContext("webgl2", { xrCompatible: true, alpha: true });
  await gl.makeXRCompatible?.();
  session.updateRenderState({ baseLayer: new XRWebGLLayer(session, gl) });
  binding = typeof XRWebGLBinding !== "undefined" ? new XRWebGLBinding(session, gl) : null;
  refSpace = await session.requestReferenceSpace("local");
  setupGL();
  counts.t0 = performance.now();
  session.addEventListener("end", () => { $("overlay").classList.remove("on"); session = null; });
  session.requestAnimationFrame(onFrame);
}

// カメラのテクスチャを、検出に渡す大きさの画像に描くための小さなシェーダー
function setupGL() {
  const sh = (type, src) => { const s = gl.createShader(type); gl.shaderSource(s, src); gl.compileShader(s); return s; };
  prog = gl.createProgram();
  gl.attachShader(prog, sh(gl.VERTEX_SHADER, `#version 300 es
in vec2 p; out vec2 uv; void main() { uv = p * 0.5 + 0.5; gl_Position = vec4(p, 0.0, 1.0); }`));
  gl.attachShader(prog, sh(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float; uniform sampler2D t; in vec2 uv; out vec4 o; void main() { o = texture(t, uv); }`));
  gl.linkProgram(prog);
  vao = gl.createVertexArray();
  gl.bindVertexArray(vao);
  const buf = gl.createBuffer();
  gl.bindBuffer(gl.ARRAY_BUFFER, buf);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array([-1, -1, 1, -1, -1, 1, 1, 1]), gl.STATIC_DRAW);
  const loc = gl.getAttribLocation(prog, "p");
  gl.enableVertexAttribArray(loc);
  gl.vertexAttribPointer(loc, 2, gl.FLOAT, false, 0, 0);
  gl.bindVertexArray(null);
}
function ensureFbo(w, h) {
  if (fbo && inW === w && inH === h) return;
  inW = w; inH = h;
  fboTex = gl.createTexture();
  gl.bindTexture(gl.TEXTURE_2D, fboTex);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA8, w, h, 0, gl.RGBA, gl.UNSIGNED_BYTE, null);
  gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  fbo = gl.createFramebuffer();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.framebufferTexture2D(gl.FRAMEBUFFER, gl.COLOR_ATTACHMENT0, gl.TEXTURE_2D, fboTex, 0);
  pixels = new Uint8ClampedArray(w * h * 4);
}

// カメラのテクスチャを縮めて読み戻し、ImageBitmap にする（テクスチャは XR のフレームの中でだけ使える）
async function grab(view) {
  const cam = view.camera;
  const tex = binding.getCameraImage(cam);
  const w = IN_W, h = Math.round((IN_W * cam.height) / cam.width);
  ensureFbo(w, h);
  const t0 = performance.now();
  gl.bindFramebuffer(gl.FRAMEBUFFER, fbo);
  gl.viewport(0, 0, w, h);
  gl.useProgram(prog);
  gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.uniform1i(gl.getUniformLocation(prog, "t"), 0);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.readPixels(0, 0, w, h, gl.RGBA, gl.UNSIGNED_BYTE, pixels);
  gl.bindVertexArray(null);
  push(times.read, performance.now() - t0);
  lastSize = `${cam.width}×${cam.height} → ${w}×${h}`;
  // readPixels は下の行から。flipY なら上下を入れ替えて、画面と同じ向きにする
  const img = new ImageData(w, h);
  if (flipY) for (let y = 0; y < h; y++) img.data.set(pixels.subarray((h - 1 - y) * w * 4, (h - y) * w * 4), y * w * 4);
  else img.data.set(pixels);
  return createImageBitmap(img);
}

function onFrame(t, frame) {
  if (!session) return;
  session.requestAnimationFrame(onFrame);
  const tf = performance.now();
  const pose = frame.getViewerPose(refSpace);
  const layer = session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0, 0, 0, 0);
  gl.clear(gl.COLOR_BUFFER_BIT);
  if (!pose) return;
  const view = pose.views[0];
  counts.xr++;
  // 深度（このフレームの中でだけ読める）。今の枠ごとに距離を引き直す
  let depth = null;
  if (granted.includes("depth-sensing")) {
    try { depth = frame.getDepthInformation(view); } catch (e) { depthNote = e.message; }
    if (depth) depthNote = `${depth.width}×${depth.height}`;
  }
  for (const b of boxes) b.m = depth ? boxDepth(depth, b.n) : null;
  // 前の検出が終わっていれば、このフレームを読み戻して次の検出へ
  if (!busy && binding && view.camera) {
    busy = true;
    grab(view).then(async (bmp) => {
      drawPreview(bmp);
      const W = bmp.width, H = bmp.height; // 結果の枠は渡した画像の画素（渡すと bmp は使えなくなるので先に取る）
      const r = await detect(bmp);
      push(times.infer, r.infer_ms);
      counts.det++; status.load = ""; status.err = "";
      boxes = r.items.map((it) => ({ label: it.label, score: it.score, n: [it.box[0] / W, it.box[1] / H, it.box[2] / W, it.box[3] / H], m: null }));
    }).catch((e) => { status.err = e.message; }).finally(() => { busy = false; });
  }
  push(times.frame, performance.now() - tf);
  drawOverlay();
  const now = performance.now();
  if (now - counts.t0 > 1000) {
    counts.xrFps = (counts.xr * 1000) / (now - counts.t0); counts.detFps = (counts.det * 1000) / (now - counts.t0);
    counts.xr = 0; counts.det = 0; counts.t0 = now;
    updatePanel(view);
  }
}

// 枠の中央 6 割の 5×5 点の深度の中央値（m）。背景を拾いにくくする（応用の rawOfBox と同じ考え方）
function boxDepth(d, b) {
  const v = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const x = b[0] + (b[2] - b[0]) * (0.2 + 0.15 * i), y = b[1] + (b[3] - b[1]) * (0.2 + 0.15 * j);
    try { const m = d.getDepthInMeters(Math.min(0.999, Math.max(0, x)), Math.min(0.999, Math.max(0, y))); if (m > 0) v.push(m); } catch { /* 範囲外 */ }
  }
  if (!v.length) return null;
  v.sort((p, q2) => p - q2);
  return v[v.length >> 1];
}

function drawOverlay() {
  const c = $("ov-canvas"), dpr = devicePixelRatio || 1, W = c.clientWidth, H = c.clientHeight;
  if (c.width !== Math.round(W * dpr)) { c.width = Math.round(W * dpr); c.height = Math.round(H * dpr); }
  const ctx = c.getContext("2d");
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, W, H);
  ctx.lineWidth = 3; ctx.font = "600 15px system-ui, sans-serif"; ctx.textBaseline = "top";
  for (const b of boxes) {
    const [x1, y1, x2, y2] = [b.n[0] * W, b.n[1] * H, b.n[2] * W, b.n[3] * H];
    ctx.strokeStyle = "#22c55e"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${b.label} ${(b.score * 100).toFixed(0)}%${b.m != null ? `  ${b.m.toFixed(2)} m` : ""}`;
    const tw = ctx.measureText(text).width + 8, tx0 = Math.max(0, Math.min(x1, W - tw)), ty0 = Math.max(0, y1 - 22);
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(tx0, ty0, tw, 21);
    ctx.fillStyle = "#fff"; ctx.fillText(text, tx0 + 4, ty0 + 3);
  }
}
function drawPreview(bmp) {
  const c = $("ov-preview");
  if (c.width !== bmp.width) { c.width = bmp.width; c.height = bmp.height; }
  c.getContext("2d").drawImage(bmp, 0, 0); // bmp は Worker に渡す前に描く（渡すと使えなくなる）
}

const push = (a, x) => { a.push(x); if (a.length > 60) a.shift(); };
const med = (a) => (a.length ? [...a].sort((p, q2) => p - q2)[a.length >> 1] : NaN);
function updatePanel(view) {
  const f = (x) => (Number.isFinite(x) ? x.toFixed(1) : "-");
  $("ov-panel").textContent = [
    `${tx("使えた機能")}: ${granted.join(", ") || "?"}`,
    `${tx("カメラの画像")}: ${view.camera ? lastSize || `${view.camera.width}×${view.camera.height}` : tx("無い（camera-access が使えない）")}  ${tx("深度")}: ${depthNote || tx("無い")}`,
    `XR ${f(counts.xrFps)} fps  ${tx("検出")} ${f(counts.detFps)} fps  ${tx("読み戻し")} ${f(med(times.read))} ms  ${tx("推論")} ${f(med(times.infer))} ms  ${tx("フレームの処理")} ${f(med(times.frame))} ms`,
    `${tx("上下反転")}: ${flipY ? "ON" : "OFF"}${status.load ? `  ${tx("読み込み")} ${status.load}` : ""}${status.err ? `\n${tx("エラー")}: ${status.err}` : ""}`,
  ].join("\n");
}

$("start").onclick = start;
$("end").onclick = () => session?.end();
$("flip").onclick = () => { flipY = !flipY; };
checkSupport();
