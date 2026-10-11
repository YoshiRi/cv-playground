// AR の試し（xr.html）: WebXR の immersive-ar（Android の Chrome では ARCore）で、
//   1. カメラの映像（camera-access）を WebGL のテクスチャで受け取り、縮めて読み戻して、物体検出（YOLO26n、worker.js）に渡す
//   2. 深度（depth-sensing、CPU で読める形）から、枠の中の深度の中央値を引いて、物までの距離を m で出す
//   3. 使えた機能・カメラの画像の大きさ・XR の fps・読み戻しと推論の時間を画面の上に出す
// カメラの画像は XRView と同じ範囲・向きなので、画像の 0〜1 の座標がそのまま画面（DOM overlay）と深度の座標になる
import { CATALOG } from "./catalog.js";
import { collectEnv, stats, toMarkdown } from "./export.js";
import { startTranslate, tx } from "./i18n.js";

startTranslate();
const $ = (id) => document.getElementById(id);
const q = new URLSearchParams(location.search);
const MODEL = CATALOG.models.find((m) => m.key === (q.get("model") || "yolo26n"));
// 検出に渡す画像の長辺。モデルの入力（YOLO26n は 640×640 に縮めて余白を足す）より大きく読み戻しても、縮める手間が増えるだけ
const IN_LONG = +q.get("long") || MODEL?.pre?.size?.[0] || 640;
const TH = +q.get("th") || 0.4;
let flipY = q.get("flipy") !== "0";
let align = false; // 位置合わせを確かめる: 検出に渡した画像を、枠と同じ写し方で画面に半透明で重ねる

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
    drawPreview(img);
    const W = img.width, H0 = img.height, r = await detect(img);
    // 重ね描きの経路（WebGL）も、AR なしで普通の canvas に描いて確かめる（位置合わせつき）
    const c = document.createElement("canvas"); c.width = 360; c.height = 640; c.id = "selftest-canvas"; c.style.width = "180px";
    gl = c.getContext("webgl2", { premultipliedAlpha: true }); setupGL();
    map.vp = { x: 0, y: 0, width: 360, height: 640 }; map.fbW = 360; map.fbH = 640; map.camAspect = W / H0;
    session = { renderState: { baseLayer: { framebuffer: null } } }; align = true;
    boxes = r.items.map((it) => ({ label: it.label, score: it.score, n: [it.box[0] / W, it.box[1] / H0, it.box[2] / W, it.box[3] / H0], m: 1.23 }));
    drawOverlay(); session = null; align = false;
    $("support").after(c);
    $("support").textContent += `\nselftest: ${r.items.length} items, ${r.infer_ms?.toFixed(1)} ms, ${r.items.slice(0, 3).map((it) => `${it.label} ${it.box.map((v) => (v / W).toFixed(2)).join(",")}`).join(" / ")}`;
  }
}

// ---------- AR のセッション ----------
let session, refSpace, gl, binding, prog, fbo, fboTex, vao, pixels;
let inW = 0, inH = 0, busy = false, boxes = [], lastSize = "";
const times = { infer: [], read: [], frame: [] };
const counts = { xr: 0, det: 0, t0: 0, xrFps: 0, detFps: 0 };
let depthNote = "", granted = [];
// セッション全体の記録（終わった時に実行の記録にする）
let rec = null;
const newRec = () => ({ t0: performance.now(), infer: [], read: [], pre: [], run: [], post: [], xrFps: [], dets: 0, xrFrames: 0, first: null, last: null, cam: "", depth: "" });
// カメラの画像の 0〜1 → 画面（overlay の CSS px）の写し方。毎フレーム、XR の viewport とカメラの大きさから作り直す
const map = { vp: null, fbW: 1, fbH: 1, camAspect: 0, mode: "" };

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
  rec = newRec();
  session.addEventListener("end", () => { $("overlay").classList.remove("on"); session = null; saveRecord(); });
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
  const k = IN_LONG / Math.max(cam.width, cam.height), w = Math.round(cam.width * k), h = Math.round(cam.height * k);
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
  push(times.read, performance.now() - t0); rec.read.push(performance.now() - t0);
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
  counts.xr++; rec.xrFrames++;
  map.vp = layer.getViewport(view); map.fbW = layer.framebufferWidth; map.fbH = layer.framebufferHeight;
  if (view.camera) map.camAspect = view.camera.width / view.camera.height;
  // 深度（このフレームの中でだけ読める）。今の枠ごとに距離を引き直す
  let depth = null;
  if (granted.includes("depth-sensing")) {
    try { depth = frame.getDepthInformation(view); } catch (e) { depthNote = e.message; }
    if (depth) depthNote = `${depth.width}×${depth.height}`;
  }
  for (const b of boxes) b.m = depth ? boxDepth(depth, b.n.map((v, i) => camToView(v, i % 2))) : null;
  // 前の検出が終わっていれば、このフレームを読み戻して次の検出へ
  if (!busy && binding && view.camera) {
    busy = true;
    grab(view).then(async (bmp) => {
      drawPreview(bmp);
      const W = bmp.width, H = bmp.height; // 結果の枠は渡した画像の画素（渡すと bmp は使えなくなるので先に取る）
      const r = await detect(bmp);
      push(times.infer, r.infer_ms);
      counts.det++; status.load = ""; status.err = "";
      rec.dets++; rec.infer.push(r.infer_ms); rec.first ??= r; rec.last = r;
      for (const k of ["pre", "run", "post"]) if (r.breakdown?.[k] != null) rec[k].push(r.breakdown[k]);
      boxes = r.items.map((it) => ({ label: it.label, score: it.score, n: [it.box[0] / W, it.box[1] / H, it.box[2] / W, it.box[3] / H], m: null }));
    }).catch((e) => { status.err = e.message; }).finally(() => { busy = false; });
  }
  push(times.frame, performance.now() - tf);
  drawOverlay();
  const now = performance.now();
  if (now - counts.t0 > 1000) {
    counts.xrFps = (counts.xr * 1000) / (now - counts.t0); counts.detFps = (counts.det * 1000) / (now - counts.t0);
    rec.xrFps.push(counts.xrFps); rec.cam = lastSize; rec.depth = depthNote;
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

// カメラの画像の座標（0〜1）→ XR の view の座標（0〜1、深度もこの座標）。普通は同じ。縦横比が違う時は、
// カメラの画像が view を覆うように（はみ出す側を切って）写っているとみなす
function camToView(v, axis) {
  const vpA = map.vp ? map.vp.width / map.vp.height : map.camAspect, r = map.camAspect && vpA ? map.camAspect / vpA : 1;
  if (Math.abs(r - 1) < 0.01) { map.mode = "same"; return v; }
  map.mode = r > 1 ? "cover-x" : "cover-y";
  if (axis === 0) return r > 1 ? 0.5 + (v - 0.5) * r : v;
  return r > 1 ? v : 0.5 + (v - 0.5) / r;
}

// 枠・ラベル・位置合わせの画像は、2D の canvas（XR の viewport の半分の大きさ）に描き、テクスチャにして XR の描画面
// （layer.framebuffer）に重ねる。XR の描画面はカメラの映像とそのまま重なるので、DOM overlay の要素の大きさ・位置
// （端末によって画面と食い違う）に左右されない。座標は view の 0〜1 × canvas の大きさ
const LABEL_SCALE = 0.5;
let lab, labTex, labProg, labVao;
function drawOverlay() {
  const vp = map.vp;
  if (!vp) return;
  const W = Math.round(vp.width * LABEL_SCALE), H = Math.round(vp.height * LABEL_SCALE);
  if (!lab || lab.width !== W || lab.height !== H) lab = new OffscreenCanvas(W, H);
  const ctx = lab.getContext("2d"), u = W / 400; // 文字・線の太さは幅 400 の画面の px を基準に
  ctx.clearRect(0, 0, W, H);
  const P = (a, b) => [camToView(a, 0) * W, camToView(b, 1) * H];
  if (align) {
    const [ax, ay] = P(0, 0), [bx, by] = P(1, 1);
    ctx.globalAlpha = 0.45; ctx.drawImage($("ov-preview"), ax, ay, bx - ax, by - ay); ctx.globalAlpha = 1;
    ctx.strokeStyle = "#f0f"; ctx.lineWidth = 2 * u; ctx.strokeRect(ax, ay, bx - ax, by - ay);
  }
  ctx.lineWidth = 3 * u; ctx.font = `600 ${Math.round(15 * u)}px system-ui, sans-serif`; ctx.textBaseline = "top";
  for (const b of boxes) {
    const [x1, y1] = P(b.n[0], b.n[1]), [x2, y2] = P(b.n[2], b.n[3]);
    ctx.strokeStyle = "#22c55e"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    const text = `${b.label} ${(b.score * 100).toFixed(0)}%${b.m != null ? `  ${b.m.toFixed(2)} m` : ""}`;
    const tw = ctx.measureText(text).width + 8 * u, th = 21 * u, tx0 = Math.max(0, Math.min(x1, W - tw)), ty0 = Math.max(0, y1 - th - u);
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(tx0, ty0, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(text, tx0 + 4 * u, ty0 + 3 * u);
  }
  // テクスチャにして、XR の描画面の viewport に重ねる（canvas の上の行が画面の上になるよう上下を返して読む）
  if (!labProg) {
    const sh = (type, src) => { const x = gl.createShader(type); gl.shaderSource(x, src); gl.compileShader(x); return x; };
    labProg = gl.createProgram();
    gl.attachShader(labProg, sh(gl.VERTEX_SHADER, `#version 300 es
in vec2 p; out vec2 uv; void main() { uv = vec2(p.x * 0.5 + 0.5, 0.5 - p.y * 0.5); gl_Position = vec4(p, 0.0, 1.0); }`));
    gl.attachShader(labProg, sh(gl.FRAGMENT_SHADER, `#version 300 es
precision mediump float; uniform sampler2D t; in vec2 uv; out vec4 o; void main() { o = texture(t, uv); }`));
    gl.bindAttribLocation(labProg, gl.getAttribLocation(prog, "p"), "p"); // 頂点は検出用の vao を共用する
    gl.linkProgram(labProg);
    labTex = gl.createTexture();
    gl.bindTexture(gl.TEXTURE_2D, labTex);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
    gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  }
  gl.bindFramebuffer(gl.FRAMEBUFFER, session.renderState.baseLayer.framebuffer);
  gl.viewport(vp.x, vp.y, vp.width, vp.height);
  gl.useProgram(labProg);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_2D, labTex);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, true);
  gl.texImage2D(gl.TEXTURE_2D, 0, gl.RGBA, gl.RGBA, gl.UNSIGNED_BYTE, lab);
  gl.pixelStorei(gl.UNPACK_PREMULTIPLY_ALPHA_WEBGL, false);
  gl.uniform1i(gl.getUniformLocation(labProg, "t"), 0);
  gl.enable(gl.BLEND); gl.blendFunc(gl.ONE, gl.ONE_MINUS_SRC_ALPHA);
  gl.bindVertexArray(vao);
  gl.drawArrays(gl.TRIANGLE_STRIP, 0, 4);
  gl.bindVertexArray(null);
  gl.disable(gl.BLEND);
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
    `${tx("画面")} ${innerWidth}×${innerHeight}（dpr ${devicePixelRatio}、screen ${screen.width}×${screen.height}） fb ${map.fbW}×${map.fbH} vp ${map.vp ? [map.vp.x, map.vp.y, map.vp.width, map.vp.height].join(",") : "-"} ${tx("写し方")} ${map.mode}`,
    `${tx("上下反転")}: ${flipY ? "ON" : "OFF"}${status.load ? `  ${tx("読み込み")} ${status.load}` : ""}${status.err ? `\n${tx("エラー")}: ${status.err}` : ""}`,
  ].join("\n");
}

$("start").onclick = start;
$("end").onclick = () => session?.end();
$("flip").onclick = () => { flipY = !flipY; };
$("align").onclick = () => { align = !align; };

// ---------- 記録 ----------
// セッションが終わったら、普段の画面と同じ形の実行の記録（連続）を作り、同じ localStorage（cvpg-runs）に足す。
// 普段の画面の実行履歴・書き出しにそのまま出る。このページでも Markdown の表をコピーできる
const RUNS_KEY = "cvpg-runs", MAX_RUNS = 500;
let lastRec = null;
function saveRecord() {
  if (!rec?.dets) return;
  const st = stats(rec.infer), avg = (a) => (a.length ? a.reduce((s, x) => s + x, 0) / a.length : undefined), r = rec.last;
  const secs = (performance.now() - rec.t0) / 1000;
  lastRec = {
    time: new Date().toISOString(), mode: "live", task: "AR（検出＋深度）", model_key: MODEL.key, model_name: MODEL.name, where: "ブラウザ",
    device: r.device, runtime: r.dtype || "", input_size: "", frame_w: inW, frame_h: inH, load_ms: rec.first.load_ms,
    infer_ms: st?.mean, infer_mean_ms: st?.mean, infer_median_ms: st?.median, infer_p90_ms: st?.p90, infer_p95_ms: st?.p95, infer_min_ms: st?.min, infer_max_ms: st?.max,
    pre_ms: avg(rec.pre), run_ms: avg(rec.run), post_ms: avg(rec.post), frames: rec.dets, fps: +(rec.dets / secs).toFixed(2), warmup: 0,
    xr_fps: +(rec.xrFrames / secs).toFixed(1), readback_ms: +(stats(rec.read)?.median ?? 0).toFixed(1), camera_size: rec.cam, depth_size: rec.depth || "",
    xr_features: granted.join(" "), summary: `AR ${(rec.dets / secs).toFixed(1)}fps`,
  };
  try {
    const runs = JSON.parse(localStorage.getItem(RUNS_KEY) || "[]");
    runs.push(lastRec);
    localStorage.setItem(RUNS_KEY, JSON.stringify(runs.slice(-MAX_RUNS)));
  } catch { /* 保存できない環境 */ }
  $("rec").hidden = false;
  $("rec-text").textContent = tx("記録した: 検出 {det} fps・XR {xr} fps・推論 {inf} ms（中央値）・読み戻し {rb} ms（{n} 回、{s} 秒）。普段の画面の実行履歴にも入れた", { det: lastRec.fps, xr: lastRec.xr_fps, inf: st?.median?.toFixed(1), rb: lastRec.readback_ms, n: rec.dets, s: secs.toFixed(0) });
}
$("copy-md").onclick = async () => {
  if (!lastRec) return;
  const md = toMarkdown([lastRec], await collectEnv(mirror ? "サーバー版" : "静的版"));
  try { await navigator.clipboard.writeText(md); $("copy-md").textContent = tx("コピーした"); }
  catch { $("rec-md").hidden = false; $("rec-md").textContent = md; } // コピーできない時は表を出す（長押しで選ぶ）
};
checkSupport();
