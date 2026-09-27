// ブラウザ内推論。重い処理で画面が固まらないように Web Worker で動かす。
// メインスレッドとのやりとり:
//   → {id, model: models.json の1件, image: Blob | ImageBitmap, params}  ← {id, type: "result", result} / {id, type: "error", message}
//   ← {type: "progress", key, file, progress}（モデルのダウンロード状況）
//
// どのモデルも adapter = { load(entry, device, onProgress) -> state, run(state, image, params, entry) -> 結果 } で扱う。
// 結果の形式はサーバー（adapters.py）と同じ。models.json の adapter 名でここの ADAPTERS を引く。
//   onnx   … 汎用 ONNX（onnxruntime-web を直接使う。前処理・後処理は onnx_generic.js の部品を models.json で組み合わせる）
//   tjs-*  … transformers.js のプロセッサ・パイプラインを使う専用の実装
//
// Worker 名でライブラリを分ける: "ort" = onnxruntime-web、"4" = transformers.js 4.3、"3" = 3.8.1（4.x で壊れるモデル用）。
// 同じ Worker に2つのライブラリを読むと onnxruntime が二重になるので分けている。版を URL でなく name で渡すのは、
// 1ファイル版では Worker を Blob URL から作るので URL に引数を付けられないため
import { onnxEmbed, onnxLoad, onnxRun } from "./onnx_generic.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const TJS_VERSION = self.name === "3" ? "3.8.1" : "4.3.0";

// トップレベル await で待つと、その間に届いたメッセージが onmessage 未設定で捨てられるので、ハンドラ内で待つ
let ort, T;
const libReady = self.name === "ort"
  ? import(`${ORT_URL}ort.webgpu.min.mjs`).then((m) => { ort = m; ort.env.wasm.wasmPaths = ORT_URL; })
  : import(`https://cdn.jsdelivr.net/npm/@huggingface/transformers@${TJS_VERSION}`).then((m) => { T = m; T.env.allowLocalModels = false; });

let devicePromise;
function getDevice() {
  devicePromise ??= (async () => {
    try {
      if (navigator.gpu && (await navigator.gpu.requestAdapter())) return "webgpu";
    } catch { /* WebGPU なし */ }
    return "wasm";
  })();
  return devicePromise;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));
const boxOf = (b) => [b.xmin, b.ymin, b.xmax, b.ymax];
const splitLabels = (s) => (s || "").split(",").map((x) => x.trim()).filter(Boolean);
const toPng = (rawImage) => rawImage.toBlob("image/png"); // メインスレッドで objectURL にする

// transformers.js の読み込み進捗。数十KBごとに呼ばれるので、1% 進んだ時だけ送る
function tjsOpts(e, device, onProgress) {
  const last = {};
  return {
    device, dtype: e.dtype[device],
    progress_callback: (p) => {
      if (p.status !== "progress" || Math.floor(p.progress) === last[p.file]) return;
      last[p.file] = Math.floor(p.progress);
      onProgress(p.file, p.progress);
    },
  };
}

// ---------- adapter ----------

const ADAPTERS = {
  onnx: {
    image: "bitmap",
    load: async (e, device, onProgress) => ({ session: await onnxLoad(ort, e, device, onProgress) }),
    run: (st, img, p, e) => onnxRun(ort, st.session, e, img, p),
  },

  "tjs-detect": {
    load: async (e, d, pr) => ({ pipe: await T.pipeline("object-detection", e.repo, tjsOpts(e, d, pr)) }),
    async run(st, img, p) {
      const out = await st.pipe(img, { threshold: p.threshold ?? 0.4 });
      return { kind: "boxes", items: out.map((o) => ({ label: o.label, score: o.score, box: boxOf(o.box) })) };
    },
  },

  "tjs-depth": {
    load: async (e, d, pr) => ({ pipe: await T.pipeline("depth-estimation", e.repo, tjsOpts(e, d, pr)) }),
    async run(st, img) {
      const out = await st.pipe(img);
      return { kind: "depth", image: await toPng(await out.depth.resize(img.width, img.height)) };
    },
  },

  // pipeline は候補ごとに別バッチで渡して ONNX（バッチ 1 固定）が落ちるので、公式例どおり "a. b." の1文にする
  "tjs-gdino": {
    load: async (e, d, pr) => ({
      model: await T.AutoModelForZeroShotObjectDetection.from_pretrained(e.repo, tjsOpts(e, d, pr)),
      proc: await T.AutoProcessor.from_pretrained(e.repo),
    }),
    async run(st, img, p) {
      const labels = splitLabels(p.labels).map((l) => l.toLowerCase());
      const inputs = await st.proc(img, labels.map((l) => l + ".").join(" "));
      const out = await st.model(inputs);
      const th = p.threshold ?? 0.3;
      const [r] = st.proc.post_process_grounded_object_detection(out, inputs.input_ids, {
        box_threshold: th, text_threshold: Math.min(th, 0.25), target_sizes: [[img.height, img.width]],
      });
      return { kind: "boxes", items: r.boxes.map((b, i) => ({ label: r.labels[i] || "?", score: r.scores[i], box: b })) };
    },
  },

  // pipeline は SigLIP2 の文字列処理で "Invalid array length" になる。SigLIP は学習時と同じ max_length 64 で埋める
  "tjs-siglip": {
    load: async (e, d, pr) => ({
      model: await T.AutoModel.from_pretrained(e.repo, tjsOpts(e, d, pr)),
      proc: await T.AutoProcessor.from_pretrained(e.repo),
      tok: await T.AutoTokenizer.from_pretrained(e.repo),
    }),
    async run(st, img, p) {
      const labels = splitLabels(p.labels);
      const text = st.tok(labels.map((l) => `a photo of ${l}`.toLowerCase()), { padding: "max_length", truncation: true, max_length: 64 });
      const out = await st.model({ ...text, ...(await st.proc(img)) });
      // SigLIP は候補ごとの独立なシグモイドで学習されていて、絶対値は小さく出がち。棒は候補間の softmax、abs にシグモイド
      const logits = Array.from(out.logits_per_image.data);
      const mx = Math.max(...logits), ex = logits.map((v) => Math.exp(v - mx)), sum = ex.reduce((a, b) => a + b, 0);
      return { kind: "labels", items: labels.map((l, i) => ({ label: l, score: ex[i] / sum, abs: sigmoid(logits[i]) })).sort((a, b) => b.score - a.score) };
    },
  },

  "tjs-sam": {
    load: async (e, d, pr) => ({
      model: await T.AutoModel.from_pretrained(e.repo, tjsOpts(e, d, pr)),
      proc: await T.AutoProcessor.from_pretrained(e.repo),
      emb: new Map(),
    }),
    async run(st, img, p) {
      if (p.auto) return autoSegment(st, img, p);
      const pts = p.points || [];
      if (!pts.length) throw new Error("画像をクリックして点を指定してください");
      const inputs = await st.proc(img, { input_points: [[pts.map(([x, y]) => [x, y])]], input_labels: [[pts.map(([, , l]) => l)]] });
      // 同じ画像への2回目以降のクリックでは画像エンコーダを省く
      if (!st.emb.has(p._imageKey)) {
        st.emb.clear();
        st.emb.set(p._imageKey, await st.model.get_image_embeddings(inputs));
      }
      const out = await st.model({ ...inputs, ...st.emb.get(p._imageKey) });
      const masks = await st.proc.post_process_masks(out.pred_masks, inputs.original_sizes, inputs.reshaped_input_sizes);
      // iou_scores: (1, point_batch=1, n_masks)、masks[0]: (point_batch=1, n_masks, H, W) の bool
      const scores = out.iou_scores.data;
      let best = 0;
      for (let i = 1; i < scores.length; i++) if (scores[i] > scores[best]) best = i;
      const m0 = masks[0], h = m0.dims.at(-2), w = m0.dims.at(-1), src = m0.data, px = new Uint8ClampedArray(w * h);
      for (let i = 0; i < w * h; i++) px[i] = src[best * w * h + i] ? 255 : 0;
      return { kind: "mask", mask: await toPng(new T.RawImage(px, w, h, 1)), score: scores[best] };
    },
  },

  "tjs-vlm": {
    load: async (e, d, pr) => ({
      model: await (T.AutoModelForImageTextToText ?? T.AutoModelForVision2Seq).from_pretrained(e.repo, tjsOpts(e, d, pr)),
      proc: await T.AutoProcessor.from_pretrained(e.repo),
    }),
    async run(st, img, p) {
      const messages = [{ role: "user", content: [{ type: "image" }, { type: "text", text: p.prompt || "Describe this image." }] }];
      const text = st.proc.apply_chat_template(messages, { add_generation_prompt: true });
      // VLM は解像度が大きいと遅いので長辺 768px に縮める
      const s = Math.min(1, 768 / Math.max(img.width, img.height));
      const small = s < 1 ? await img.resize(Math.round(img.width * s), Math.round(img.height * s)) : img;
      const inputs = await st.proc(text, [small]);
      // 貪欲法のままだと Qwen3-VL 2B が同じ文を繰り返したので repetition_penalty を掛けている
      const out = await st.model.generate({ ...inputs, max_new_tokens: 256, do_sample: false, repetition_penalty: 1.15 });
      const [decoded] = st.proc.batch_decode(out.slice(null, [inputs.input_ids.dims.at(-1), null]), { skip_special_tokens: true });
      return { kind: "text", text: decoded.trim() };
    },
  },
};

// 画面全体の自動分割（SAM の automatic mask generation の簡易版）。
// 格子状の点を1点ずつプロンプトにして（この ONNX は点のバッチ次元が 1 固定で、1回に1物体しか受け付けない）、
// 良いマスクだけ残し、重なりを除いて色分けする。マスクは低解像度（256×256、入力 1024×1024 = 画像全体を縦横そのまま縮めたもの）のまま扱う
async function autoSegment(st, img, p) {
  const G = p.grid || 12, S = 256, N = S * S;
  const emb = await st.model.get_image_embeddings(await st.proc(img));
  const cands = [];
  for (let gy = 0; gy < G; gy++) for (let gx = 0; gx < G; gx++) {
    const out = await st.model({
      ...emb,
      input_points: new T.Tensor("float32", new Float32Array([((gx + 0.5) / G) * 1024, ((gy + 0.5) / G) * 1024]), [1, 1, 1, 2]),
      input_labels: new T.Tensor("int64", new BigInt64Array([1n]), [1, 1, 1]),
    });
    if (out.object_score_logits.data[0] <= 0) continue;
    const iou = out.iou_scores.data;
    let bi = 0;
    for (let k = 1; k < 3; k++) if (iou[k] > iou[bi]) bi = k;
    if (iou[bi] < 0.75) continue;
    const logits = out.pred_masks.data.subarray(bi * N, bi * N + N);
    // 安定度: 閾値を ±1 動かしても面積がほぼ変わらないマスクだけ残す（SAM の既定 0.95 より緩め）
    let hi = 0, lo = 0;
    const m = new Uint8Array(N);
    for (let i = 0; i < N; i++) { const v = logits[i]; if (v > 1) hi++; if (v > -1) lo++; if (v > 0) m[i] = 1; }
    const area = m.reduce((a, b) => a + b, 0);
    if (lo === 0 || hi / lo < 0.85 || area < N * 0.001 || area > N * 0.95) continue;
    cands.push({ m, area, score: iou[bi] * (hi / lo) });
  }
  // 重なりの大きいマスクは、良い方だけ残す
  cands.sort((a, b) => b.score - a.score);
  const kept = [];
  for (const c of cands) {
    let dup = false;
    for (const k of kept) {
      let inter = 0;
      for (let i = 0; i < N; i++) inter += c.m[i] & k.m[i];
      if (inter / (c.area + k.area - inter) > 0.7) { dup = true; break; }
    }
    if (!dup) kept.push(c);
  }
  // 大きい順に塗り、小さい物体を上に重ねる
  kept.sort((a, b) => b.area - a.area);
  const rgba = new Uint8ClampedArray(N * 4);
  kept.forEach((k, j) => {
    const [r, g, b] = hsl((j * 137.508) % 360, 0.7, 0.55);
    for (let i = 0; i < N; i++) if (k.m[i]) { rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b; rgba[i * 4 + 3] = 255; }
  });
  return { kind: "segmap", image: await toPng(new T.RawImage(rgba, S, S, 4)), count: kept.length, prompts: G * G };
}

function hsl(h, s, l) {
  const a = s * Math.min(l, 1 - l);
  const f = (n) => { const k = (n + h / 30) % 12; return 255 * (l - a * Math.max(-1, Math.min(k - 3, 9 - k, 1))); };
  return [f(0), f(8), f(4)];
}

// ---------- 画像の受け渡しと実行 ----------

// 静止画は Blob、動画のフレームは ImageBitmap で届く（毎フレーム JPEG にすると遅いので）。
// onnx adapter は ImageBitmap、transformers.js 系は RawImage で受け取る
async function toInput(image, kind) {
  if (kind === "bitmap") return image instanceof Blob ? createImageBitmap(image) : image;
  if (image instanceof Blob) return T.RawImage.fromBlob(image);
  const c = new OffscreenCanvas(image.width, image.height), x = c.getContext("2d");
  x.drawImage(image, 0, 0);
  image.close?.();
  return new T.RawImage(x.getImageData(0, 0, c.width, c.height).data, c.width, c.height, 4).rgb();
}

const loaded = new Map(); // entry.key -> Promise<state>

// 追跡の ReID 用: 切り出した画像の特徴を返す（onnx adapter のモデルだけ）
async function embed(id, e, crops) {
  try {
    await libReady;
    if (!loaded.has(e.key)) {
      const onProgress = (file, progress) => self.postMessage({ type: "progress", key: e.key, file, progress });
      loaded.set(e.key, ADAPTERS.onnx.load(e, await getDevice(), onProgress));
      try { await loaded.get(e.key); } catch (err) { loaded.delete(e.key); throw err; }
    }
    const t0 = performance.now();
    const feats = await onnxEmbed(ort, (await loaded.get(e.key)).session, e, crops);
    self.postMessage({ id, type: "result", result: { feats, ms: performance.now() - t0 } });
  } catch (err) {
    self.postMessage({ id, type: "error", message: String(err?.message || err) });
  }
}

self.onmessage = async (ev) => {
  if (ev.data.type === "embed") return embed(ev.data.id, ev.data.model, ev.data.crops);
  const { id, model: e, image, params } = ev.data;
  try {
    await libReady;
    const A = ADAPTERS[e.adapter];
    if (!A) throw new Error(`ブラウザ側に adapter "${e.adapter}" が無い`);
    const device = await getDevice();
    let loadMs = 0;
    if (!loaded.has(e.key)) {
      const t0 = performance.now();
      const onProgress = (file, progress) => self.postMessage({ type: "progress", key: e.key, file, progress });
      loaded.set(e.key, A.load(e, device, onProgress));
      try { await loaded.get(e.key); } catch (err) { loaded.delete(e.key); throw err; }
      loadMs = performance.now() - t0;
    }
    const st = await loaded.get(e.key);
    const img = await toInput(image, A.image);
    const t1 = performance.now();
    const result = await A.run(st, img, params, e);
    result.infer_ms = performance.now() - t1;
    result.load_ms = loadMs;
    result.device = device;
    result.dtype = e.adapter === "onnx" ? "onnxruntime-web" : e.dtype?.[device];
    self.postMessage({ id, type: "result", result });
  } catch (err) {
    self.postMessage({ id, type: "error", message: String(err?.message || err) });
  }
};
