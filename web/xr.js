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
let stick = q.get("stick") !== "0"; // 張り付け: 枠を部屋の中の位置（3D）にして、今のカメラの向きで描き直す
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
  objs = []; renderStrip(); resetMap();
  session.addEventListener("end", () => { $("overlay").classList.remove("on"); session = null; saveRecord(); renderCollection(); });
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
  map.cur = { proj: view.projectionMatrix, toCam: view.transform.inverse.matrix, toWorld: view.transform.matrix, pos: view.transform.position };
  // 張り付けない時は、今のフレームの深度で枠ごとの距離を引き直す（張り付ける時は 3D の点と今のカメラの距離）
  if (depth) mapDepth(depth, view);
  if (counts.xr % 10 === 0) addTrack(view);
  if (!stick) for (const b of boxes) b.m = depth ? boxDepth(depth, b.n.map((v, i) => camToView(v, i % 2))) : null;
  // 前の検出が終わっていれば、このフレームを読み戻して次の検出へ。読み戻した時のカメラの位置・向きと深度を一緒に取っておく
  if (!busy && binding && view.camera) {
    busy = true;
    const cap = { proj: Float32Array.from(view.projectionMatrix), toWorld: Float32Array.from(view.transform.matrix), depth: depth ? copyDepth(depth) : null };
    grab(view).then(async (bmp) => {
      drawPreview(bmp);
      const W = bmp.width, H = bmp.height; // 結果の枠は渡した画像の画素（渡すと bmp は使えなくなるので先に取る）
      const r = await detect(bmp);
      push(times.infer, r.infer_ms);
      counts.det++; status.load = ""; status.err = "";
      rec.dets++; rec.infer.push(r.infer_ms); rec.first ??= r; rec.last = r;
      for (const k of ["pre", "run", "post"]) if (r.breakdown?.[k] != null) rec[k].push(r.breakdown[k]);
      boxes = r.items.map((it) => ({ label: it.label, score: it.score, n: [it.box[0] / W, it.box[1] / H, it.box[2] / W, it.box[3] / H], m: null }));
      for (const b of boxes) placeBox(b, cap);
      collect(boxes, performance.now());
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

// ---------- 張り付け（枠を 3D に） ----------
// 深度は読み戻したフレームの中でしか読めないので、写しを取っておき、検出の結果が届いた時に枠の中を引く
function copyDepth(d) {
  try {
    const f32 = session.depthDataFormat === "float32";
    return { w: d.width, h: d.height, k: d.rawValueToMeters, m: Float32Array.from(d.normDepthBufferFromNormView.matrix), data: f32 ? new Float32Array(d.data.slice(0)) : new Uint16Array(d.data.slice(0)) };
  } catch { return null; }
}
// 写しの深度（m）。view の 0〜1 の座標 → 深度の画像の 0〜1（normDepthBufferFromNormView）→ 画素
function depthAt(c, x, y) {
  const u = c.m[0] * x + c.m[4] * y + c.m[12], v = c.m[1] * x + c.m[5] * y + c.m[13];
  const col = Math.min(c.w - 1, Math.max(0, Math.floor(u * c.w))), row = Math.min(c.h - 1, Math.max(0, Math.floor(v * c.h)));
  return c.data[row * c.w + col] * c.k;
}
// view の 0〜1 の点を、深度 d（m、カメラの前向きの距離）で部屋の座標の点にする。射影行列 P（列優先）から
// ndc.x = (P0·x + P8·z) / −z（z = −d）なので x = (ndc.x + P8)·d / P0
function unproject(cap, x, y, d) {
  const P = cap.proj, nx = 2 * x - 1, ny = 1 - 2 * y;
  const vx = ((nx + P[8]) * d) / P[0], vy = ((ny + P[9]) * d) / P[5], vz = -d, M = cap.toWorld;
  return [M[0] * vx + M[4] * vy + M[8] * vz + M[12], M[1] * vx + M[5] * vy + M[9] * vz + M[13], M[2] * vx + M[6] * vy + M[10] * vz + M[14]];
}
// 部屋の座標の点を、今のカメラの view の 0〜1 に（カメラの後ろなら null）
function project(p) {
  const C = map.cur.toCam, P = map.cur.proj;
  const x = C[0] * p[0] + C[4] * p[1] + C[8] * p[2] + C[12], y = C[1] * p[0] + C[5] * p[1] + C[9] * p[2] + C[13], z = C[2] * p[0] + C[6] * p[1] + C[10] * p[2] + C[14];
  if (z > -0.05) return null;
  const nx = (P[0] * x + P[8] * z) / -z, ny = (P[5] * y + P[9] * z) / -z;
  return [(nx + 1) / 2, (1 - ny) / 2];
}
// 枠を、読み戻した時のカメラの前に、枠の中の深度の所に立てた板（4 隅の 3D の点）にする。深度が無い枠は 2 m に置く
// （向きの変化の補正には距離はあまり効かない）
function placeBox(b, cap) {
  const v = b.n.map((x, i) => camToView(x, i % 2));
  b.depth = cap.depth ? boxDepth({ getDepthInMeters: (x, y) => depthAt(cap.depth, x, y) }, v) : null;
  const d = b.depth ?? 2;
  b.world = [[v[0], v[1]], [v[2], v[1]], [v[2], v[3]], [v[0], v[3]]].map(([x, y]) => unproject(cap, x, y, d));
  b.center = unproject(cap, (v[0] + v[2]) / 2, (v[1] + v[3]) / 2, d);
}

// ---------- 集める ----------
// 検出した物を、部屋の中の位置つきで覚える。同じクラスで近い所（40 cm か板の幅の半分の大きい方）に検出が来たら同じ物とみなし、
// 見えるたびに確かさを重ねる（p ← 1 − (1 − p)(1 − W·score)。W を小さくして、1 回の高い点数ではなく、何度も見えたことで上げる）。
// 確かさが COLLECT_P を超え、COLLECT_N 回以上見えたら「集めた」にして、画面の外に出ても覚えておく。
// 集めていない候補は FORGET_MS 見えなければ忘れる。深度の無い枠（位置が決まらない）は数えない
const COLLECT_P = +q.get("collect") || 0.9, COLLECT_N = 5, W_OBS = 0.3, MATCH_M = 0.4, FORGET_MS = 15000;
let objs = [], objSeq = 0;
const dist3 = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1], a[2] - b[2]);
function collect(list, t) {
  if (floorY == null) return; // 床が決まるまで（ARCore の位置の推定が落ち着くまで）集めない。始めは床を映す
  for (const b of list) {
    if (b.depth == null || !b.center) continue;
    const r = Math.max(MATCH_M, dist3(b.world[0], b.world[1]) * 0.5);
    let o = null, od = Infinity;
    for (const x of objs) { const d = x.label === b.label ? dist3(x.pos, b.center) : Infinity; if (d < r && d < od) { o = x; od = d; } }
    if (!o) objs.push((o = { id: ++objSeq, label: b.label, pos: b.center.slice(), n: 0, p: 0, best: 0, thumb: "", first: t, collected: false }));
    o.n++; o.sum = (o.sum || 0) + b.score;
    const k = 1 / Math.min(o.n, 10); // 位置は見えた回数で平均する（10 回目からは移動平均）
    o.pos = o.pos.map((v, i) => v + (b.center[i] - v) * k);
    o.p = 1 - (1 - o.p) * (1 - W_OBS * b.score);
    o.last = t; b.obj = o;
    if (b.score > o.best) { o.best = b.score; o.thumb = crop(b.n); o.dist = b.depth; }
    if (!o.collected && o.p >= COLLECT_P && o.n >= COLLECT_N) { o.collected = true; o.at = new Date(); renderStrip(); }
  }
  objs = objs.filter((o) => o.collected || t - o.last < FORGET_MS);
}
// 一番確かだった時の切り抜き（右下の画像は、検出に渡したフレームのまま）
function crop(n) {
  const src = $("ov-preview"), W = src.width, H = src.height;
  const x = n[0] * W, y = n[1] * H, w = Math.max(1, (n[2] - n[0]) * W), h = Math.max(1, (n[3] - n[1]) * H), k = 96 / Math.max(w, h);
  const c = document.createElement("canvas"); c.width = Math.max(1, Math.round(w * k)); c.height = Math.max(1, Math.round(h * k));
  c.getContext("2d").drawImage(src, x, y, w, h, 0, 0, c.width, c.height);
  return c.toDataURL("image/jpeg", 0.8);
}
const collected = () => objs.filter((o) => o.collected);
function renderStrip() {
  $("ov-strip").innerHTML = collected().slice(-12).reverse().map((o) => `<figure><img src="${o.thumb}" alt=""><figcaption>${esc(o.label)}</figcaption></figure>`).join("");
}
const esc = (s) => String(s).replace(/[&<>"]/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;" })[c]);
// 覚えている物の印: 集めた物は ✓ とラベル、候補は確かさ（%）。今のカメラで画面に投影する（後ろにある物は出さない）
function drawPins(ctx, W, H, u) {
  if (!map.cur) return;
  for (const o of objs) {
    const pt = project(o.pos);
    if (!pt || pt[0] < 0 || pt[0] > 1 || pt[1] < 0 || pt[1] > 1) continue;
    const [x, y] = [pt[0] * W, pt[1] * H];
    ctx.beginPath(); ctx.arc(x, y, (o.collected ? 7 : 5) * u, 0, Math.PI * 2);
    if (o.collected) { ctx.fillStyle = "#16a34a"; ctx.fill(); } else { ctx.strokeStyle = "#fff"; ctx.lineWidth = 2 * u; ctx.stroke(); }
    const text = o.collected ? `✓ ${o.label}` : `${Math.round(o.p * 100)}%`;
    const tw = ctx.measureText(text).width + 8 * u, th = 20 * u;
    ctx.fillStyle = o.collected ? "rgba(22,163,74,.85)" : "rgba(0,0,0,.55)"; ctx.fillRect(x + 9 * u, y - th / 2, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(text, x + 13 * u, y - th / 2 + 3 * u);
  }
}
// ---------- 地図（上から見た間取り） ----------
// ARCore の地図は WebXR から取れないので、位置・向きの分かった深度の画像を部屋の座標に戻して積む。
// 上から見た CELL 角の升目ごとに、高さ 10 cm 刻みの点の数を持つ（床の高さの推定が後から変わっても塗り直せるように）。
// 床は、始めた時の端末の高さより 0.5 m 以上低い所で、点が一番集まる高さ（床は一番広い水平面）。下の方のパーセンタイルだと、
// 遠くの深度の誤差で床より下に外れた点に引っぱられ、床を低く見積もって床全体が障害物になった（SCG32）。
// 床 ±10 cm を床、床から 10 cm〜1.8 m で床の点より多い所を障害物（壁・家具）として塗る。
// 深度は 3 m まで（遠いほど粗い）。ARCore が後から自分の位置を直しても、積んだ点は直らない（壁が二重になることがある）
const CELL = 0.05, Y0 = -2.5, YB = 40, MAP_EVERY = 3; // 高さは −2.5〜+1.5 m（原点は始めた時の端末の高さ）
let grid, yHist, mapTick, track, floorY, mapCache;
function resetMap() { grid = new Map(); yHist = new Uint32Array(YB * 5); mapTick = 0; track = []; floorY = null; mapCache = null; }
resetMap();
function mapDepth(depth, view) {
  if (++mapTick % MAP_EVERY) return;
  const cap = { proj: view.projectionMatrix, toWorld: view.transform.matrix };
  for (let j = 0; j < 24; j++) for (let i = 0; i < 40; i++) {
    const x = (i + 0.5) / 40, y = (j + 0.5) / 24;
    let d;
    try { d = depth.getDepthInMeters(x, y); } catch { continue; }
    if (!(d > 0.2 && d < 3)) continue;
    const p = unproject(cap, x, y, d), yb = Math.floor((p[1] - Y0) / 0.1);
    if (yb < 0 || yb >= YB) continue;
    yHist[Math.floor((p[1] - Y0) / 0.02)]++;
    const key = `${Math.floor(p[0] / CELL)},${Math.floor(p[2] / CELL)}`;
    let c = grid.get(key);
    if (!c) grid.set(key, (c = new Uint16Array(YB)));
    if (c[yb] < 65535) c[yb]++;
  }
  if (mapTick % 30 === 0) { // 床の高さ: −0.5 m より下で、点の数（前後 2 cm をならす）が一番多い高さ。その高さに 500 点たまってから
    let best = -1, bi = -1;
    const top = Math.floor((-0.5 - Y0) / 0.02);
    for (let i = 2; i < top; i++) { const v = yHist[i - 1] + yHist[i] + yHist[i + 1]; if (v > best) { best = v; bi = i; } }
    if (best >= 500) floorY = Y0 + (bi + 0.5) * 0.02;
    mapCache = null;
  }
}
function addTrack(view) { const p = view.transform.position; track.push([p.x, p.z]); if (track.length > 2000) track.shift(); }
// 升目の種類: 2 = 障害物、1 = 床、0 = 分からない
function cellKind(c) {
  if (floorY == null) return 0;
  const fb = Math.floor((floorY - Y0) / 0.1);
  let f = 0, o = 0;
  for (let b = Math.max(0, fb - 1); b <= fb + 1 && b < YB; b++) f += c[b];
  for (let b = fb + 2; b <= fb + 18 && b < YB; b++) o += c[b];
  return o >= 4 && o > f ? 2 : f >= 2 ? 1 : 0;
}
// 地図を描く（x0, y0 から S 角）。間取りの升目は 0.5 秒ごとに描き直して取っておき、自分・物は毎回描く。
// 向きは部屋に固定（始めた時の前が上）。x が右、z が下
function drawMinimap(ctx, x0, y0, S, u) {
  if (!grid.size && !objs.length) return;
  const now = performance.now();
  if (!mapCache || now - mapCache.t > 500 || mapCache.S !== S) {
    let mnx = Infinity, mxx = -Infinity, mnz = Infinity, mxz = -Infinity;
    const ext = (x, z) => { mnx = Math.min(mnx, x); mxx = Math.max(mxx, x); mnz = Math.min(mnz, z); mxz = Math.max(mxz, z); };
    const kinds = [];
    for (const [key, c] of grid) { const k = cellKind(c); if (!k) continue; const [i, j] = key.split(",").map(Number); kinds.push([i, j, k]); ext(i * CELL, j * CELL); }
    for (const [x, z] of track) ext(x, z);
    for (const o of objs) ext(o.pos[0], o.pos[2]);
    if (!Number.isFinite(mnx)) return;
    const span = Math.max(mxx - mnx, mxz - mnz, 2) + 1, s = S / span, ox = (mnx + mxx) / 2 - span / 2, oz = (mnz + mxz) / 2 - span / 2;
    const c = new OffscreenCanvas(S, S), g = c.getContext("2d");
    g.fillStyle = "rgba(0,0,0,.6)"; g.fillRect(0, 0, S, S);
    const cs = Math.max(1, CELL * s);
    for (const [i, j, k] of kinds) { g.fillStyle = k === 2 ? "rgba(255,255,255,.85)" : "rgba(120,170,255,.35)"; g.fillRect((i * CELL - ox) * s, (j * CELL - oz) * s, cs, cs); }
    g.strokeStyle = "rgba(250,204,21,.6)"; g.lineWidth = 1.5 * u; g.beginPath();
    track.forEach(([x, z], i) => (i ? g.lineTo((x - ox) * s, (z - oz) * s) : g.moveTo((x - ox) * s, (z - oz) * s))); g.stroke();
    mapCache = { t: now, S, s, ox, oz, img: c };
  }
  const { s, ox, oz } = mapCache, X = (x) => x0 + (x - ox) * s, Z = (z) => y0 + (z - oz) * s;
  ctx.drawImage(mapCache.img, x0, y0);
  ctx.strokeStyle = "rgba(255,255,255,.5)"; ctx.lineWidth = u; ctx.strokeRect(x0, y0, S, S);
  ctx.save(); ctx.beginPath(); ctx.rect(x0, y0, S, S); ctx.clip();
  ctx.font = `600 ${Math.round(11 * u)}px system-ui, sans-serif`; ctx.textBaseline = "middle";
  for (const o of objs) {
    ctx.beginPath(); ctx.arc(X(o.pos[0]), Z(o.pos[2]), (o.collected ? 4 : 2.5) * u, 0, Math.PI * 2);
    ctx.fillStyle = o.collected ? "#22c55e" : "#fff"; ctx.fill();
    if (o.collected) { ctx.fillStyle = "#fff"; ctx.fillText(o.label, X(o.pos[0]) + 5 * u, Z(o.pos[2])); }
  }
  if (map.cur) { // 自分: 位置と、向いている方（カメラの −z を上から見た向き）
    const p = map.cur.pos, M = map.cur.toWorld, fx = -M[8], fz = -M[10], fl = Math.hypot(fx, fz) || 1, ax = fx / fl, az = fz / fl, L = 12 * u;
    ctx.fillStyle = "#facc15"; ctx.beginPath();
    ctx.moveTo(X(p.x) + ax * L, Z(p.z) + az * L);
    ctx.lineTo(X(p.x) - az * L * 0.45, Z(p.z) + ax * L * 0.45); ctx.lineTo(X(p.x) + az * L * 0.45, Z(p.z) - ax * L * 0.45);
    ctx.closePath(); ctx.fill();
  }
  ctx.restore();
}

// AR を終えた後の一覧
function renderCollection() {
  const list = collected();
  $("collection").hidden = !list.length && !grid.size;
  const byClass = {};
  for (const o of list) byClass[o.label] = (byClass[o.label] || 0) + 1;
  $("collection-head").textContent = tx("集めた物 {n}（{c}）", { n: list.length, c: Object.entries(byClass).map(([k, v]) => `${k} ${v}`).join("、") });
  // 地図も、このページに大きく描く（自分の向きは出さない）
  const mc = $("collection-map"), S = Math.min(360, document.body.clientWidth - 32), ctx = mc.getContext("2d"), cur = map.cur;
  mc.width = S; mc.height = S; map.cur = null; mapCache = null;
  drawMinimap(ctx, 0, 0, S, 1.4); map.cur = cur; mapCache = null;
  mc.hidden = !grid.size;
  $("collection-list").innerHTML = list.map((o) => `<figure><img src="${o.thumb}" alt=""><figcaption><b>${esc(o.label)}</b><br>${tx("{n} 回・確かさ {p}%・平均 {a}%・{d} m", { n: o.n, p: Math.round(o.p * 100), a: Math.round((o.sum / o.n) * 100), d: o.dist?.toFixed(2) ?? "-" })}</figcaption></figure>`).join("");
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
    let x1, y1, x2, y2;
    if (stick && b.world && map.cur) {
      // 4 隅を今のカメラで画面に。板が傾いて見える時もそのまま四角形で描く
      const q2 = b.world.map(project);
      if (q2.some((x) => !x)) continue;
      const pts = q2.map(([x, y]) => [x * W, y * H]);
      ctx.strokeStyle = b.depth != null ? "#22c55e" : "#facc15";
      ctx.beginPath(); pts.forEach(([x, y], i) => (i ? ctx.lineTo(x, y) : ctx.moveTo(x, y))); ctx.closePath(); ctx.stroke();
      [x1, y1] = [Math.min(...pts.map((p) => p[0])), Math.min(...pts.map((p) => p[1]))];
      b.m = b.depth != null ? Math.hypot(b.center[0] - map.cur.pos.x, b.center[1] - map.cur.pos.y, b.center[2] - map.cur.pos.z) : null;
    } else {
      [x1, y1] = P(b.n[0], b.n[1]); [x2, y2] = P(b.n[2], b.n[3]);
      ctx.strokeStyle = "#22c55e"; ctx.strokeRect(x1, y1, x2 - x1, y2 - y1);
    }
    const text = `${b.label} ${(b.score * 100).toFixed(0)}%${b.m != null ? `  ${b.m.toFixed(2)} m` : ""}`;
    const tw = ctx.measureText(text).width + 8 * u, th = 21 * u, tx0 = Math.max(0, Math.min(x1, W - tw)), ty0 = Math.max(0, y1 - th - u);
    ctx.fillStyle = "rgba(0,0,0,.65)"; ctx.fillRect(tx0, ty0, tw, th);
    ctx.fillStyle = "#fff"; ctx.fillText(text, tx0 + 4 * u, ty0 + 3 * u);
  }
  drawPins(ctx, W, H, u);
  drawMinimap(ctx, W - Math.round(W * 0.42) - 8 * u, Math.round(H * 0.3), Math.round(W * 0.42), u);
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
    `${floorY == null ? tx("床を映す（床が決まるまで集めない）") : `${tx("床")} ${floorY.toFixed(2)} m`}  ${tx("集めた")} ${collected().length}（${tx("候補")} ${objs.length - collected().length}）  ${tx("張り付け")}: ${stick ? "ON" : "OFF"}  ${tx("上下反転")}: ${flipY ? "ON" : "OFF"}${status.load ? `  ${tx("読み込み")} ${status.load}` : ""}${status.err ? `\n${tx("エラー")}: ${status.err}` : ""}`,
  ].join("\n");
}

$("start").onclick = start;
$("end").onclick = () => session?.end();
$("align").onclick = () => { align = !align; };
$("stick").onclick = () => { stick = !stick; };
$("reset").onclick = () => { objs = []; renderStrip(); };
$("reset-map").onclick = () => resetMap();

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
    xr_features: granted.join(" "), summary: `AR ${(rec.dets / secs).toFixed(1)}fps`, collected: collected().length,
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
