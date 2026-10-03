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
import { hsl, onnxEmbed, onnxLoad, onnxProfile, onnxRun, segColor, setMirror, takeFetchLog, zeroshotLabels } from "./onnx_generic.js";

const ORT_URL = "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.30.0/dist/";
const TJS_VERSION = self.name === "3" ? "3.8.1" : "4.3.0";

// トップレベル await で待つと、その間に届いたメッセージが onmessage 未設定で捨てられるので、ハンドラ内で待つ
let ort, T;
const libReady = self.name === "ort"
  ? import(`${ORT_URL}ort.webgpu.min.mjs`).then((m) => {
    ort = m;
    ort.env.wasm.wasmPaths = ORT_URL;
    // WASM の複数スレッドは crossOriginIsolated（COOP/COEP ヘッダ）の時だけ使える。server.py は付ける、GitHub Pages は付けられない
    ort.env.wasm.numThreads = self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1;
  })
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
  // サーバー版: transformers.js も Hugging Face の代わりにサーバーの写し（/mirror/hf/）から取る
  T.env.remoteHost = e.mirror ? e.mirror + "hf/" : "https://huggingface.co/";
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
    run: (st, img, p, e) => onnxRun(ort, st, e, img, p),
  },

  "tjs-detect": {
    load: async (e, d, pr) => ({ pipe: await T.pipeline("object-detection", e.repo, tjsOpts(e, d, pr)) }),
    async run(st, img, p) {
      const out = await st.pipe(img, { threshold: p.threshold ?? 0.4 });
      return { kind: "boxes", items: out.map((o) => ({ label: o.label, score: o.score, box: boxOf(o.box) })) };
    },
  },

  // セマンティック / パノプティック（transformers.js の image-segmentation。subtask は models.json）
  "tjs-segment": {
    load: async (e, d, pr) => ({ pipe: await T.pipeline("image-segmentation", e.repo, tjsOpts(e, d, pr)) }),
    async run(st, img, p, e) {
      // subtask を明示すると transformers.js（3.8.1・4.3.0 とも）が関数名の文字列を呼ぼうとして
      // "x is not a function" で落ちる。省くとモデルの後処理から自動で選ぶ（SegFormer は semantic、DETR は panoptic）
      const out = await st.pipe(img);
      const W = img.width, H = img.height, rgba = new Uint8ClampedArray(W * H * 4), seen = {}, legend = {};
      // 大きい順に塗り、小さい物を上に。マスクは画像と同じ大きさの1チャンネル
      const segs = out.map((o) => {
        const m = o.mask.width === W && o.mask.height === H ? o.mask : null;
        let area = 0;
        if (m) for (let i = 0; i < W * H; i++) if (m.data[i * m.channels] > 0) area++;
        // 元の config に名前の無いクラス（LABEL_184 など）は models.json の label_map で名前を付ける
        const label = /^LABEL_\d+$/.test(o.label) ? e.label_map?.[o.label.slice(6)] ?? o.label : o.label;
        return { label, m, area };
      }).filter((s) => s.m).sort((a, b) => b.area - a.area);
      for (const { label, m, area } of segs) {
        const k = (seen[label] = (seen[label] ?? -1) + 1), [r, g, b] = segColor(label, k);
        for (let i = 0; i < W * H; i++) if (m.data[i * m.channels] > 0) { rgba[i * 4] = r; rgba[i * 4 + 1] = g; rgba[i * 4 + 2] = b; rgba[i * 4 + 3] = 255; }
        const lg = (legend[label] ??= { label, color: `rgb(${segColor(label).join(",")})`, count: 0, area: 0 });
        lg.count++; lg.area += area / (W * H);
      }
      return { kind: "segmap", image: await toPng(new T.RawImage(rgba, W, H, 4)), count: segs.length, subtask: e.subtask,
               legend: Object.values(legend).sort((a, b) => b.area - a.area) };
    },
  },

  "tjs-depth": {
    load: async (e, d, pr) => ({ pipe: await T.pipeline("depth-estimation", e.repo, tjsOpts(e, d, pr)) }),
    async run(st, img) {
      const out = await st.pipe(img);
      return { kind: "depth", image: await toPng(await out.depth.resize(img.width, img.height)) };
    },
  },

  // この ONNX（transformers.js）は、文に候補を並べると先頭の候補しか正しいスコアにならない
  // （猫の写真で "remote control. cat. sofa." だと cat の最大が 0.09、"cat." だけなら 0.83）。
  // そこで候補ごとに1回ずつ推論してまとめる（候補の数だけ時間がかかる）。サーバーの base（adapters.py）は1回でよい
  "tjs-gdino": {
    load: async (e, d, pr) => ({
      model: await T.AutoModelForZeroShotObjectDetection.from_pretrained(e.repo, tjsOpts(e, d, pr)),
      proc: await T.AutoProcessor.from_pretrained(e.repo),
    }),
    async run(st, img, p) {
      const th = p.threshold ?? 0.3, items = [];
      for (const label of splitLabels(p.labels).map((l) => l.toLowerCase())) {
        const inputs = await st.proc(img, label + ".");
        const out = await st.model(inputs);
        // 候補の単語の位置（[CLS] の次から、末尾の "." と [SEP] の前まで）で確率の最大を取る
        const ids = Array.from(inputs.input_ids.data, Number);
        const toks = ids.map((_, i) => i).filter((i) => i > 0 && i < ids.length - 2 && ![0, 101, 102].includes(ids[i]));
        const [, Q, Tn] = out.logits.dims, L = out.logits.data, B = out.pred_boxes.data;
        for (let q = 0; q < Q; q++) {
          let s = 0;
          for (const t of toks) s = Math.max(s, sigmoid(L[q * Tn + t]));
          if (s <= th) continue;
          const [cx, cy, w, h] = [B[q * 4] * img.width, B[q * 4 + 1] * img.height, B[q * 4 + 2] * img.width, B[q * 4 + 3] * img.height];
          items.push({ label, score: s, box: [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2] });
        }
      }
      return { kind: "boxes", items };
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

  // SigLIP の文エンコーダだけ（候補が変わった時に 1 回）。候補の文の埋め込み（L2 正規化済み、n × d）を返す。
  // 画面が保持して、画像エンコーダのモデル（tjs-siglip-vision・汎用 ONNX の zeroshot）に毎フレーム渡す
  "tjs-siglip-text": {
    load: async (e, d, pr) => ({
      model: await T.SiglipTextModel.from_pretrained(e.repo, { ...tjsOpts(e, e.text_device || d, pr), device: e.text_device || d }),
      tok: await T.AutoTokenizer.from_pretrained(e.repo),
    }),
    async run(st, img, p) {
      const labels = splitLabels(p.labels);
      const text = st.tok(labels.map((l) => `a photo of ${l}`.toLowerCase()), { padding: "max_length", truncation: true, max_length: 64 });
      const out = await st.model(text), emb = out.pooler_output, [n, dim] = emb.dims, data = new Float32Array(emb.data);
      for (let k = 0; k < n; k++) {
        let s = 0;
        for (let i = 0; i < dim; i++) s += data[k * dim + i] ** 2;
        const inv = 1 / Math.max(Math.sqrt(s), 1e-12);
        for (let i = 0; i < dim; i++) data[k * dim + i] *= inv;
      }
      return { kind: "embeddings", labels, data, dims: [n, dim] };
    },
  },
  // SigLIP の画像エンコーダだけ（transformers.js）。文の埋め込みは画面から params.text_embeds で受け取る
  "tjs-siglip-vision": {
    load: async (e, d, pr) => ({
      model: await T.SiglipVisionModel.from_pretrained(e.repo, tjsOpts(e, d, pr)),
      proc: await T.AutoProcessor.from_pretrained(e.repo),
    }),
    async run(st, img, p, e) {
      if (!p.text_embeds) throw new Error("候補の文の埋め込みが無い");
      const out = await st.model(await st.proc(img));
      return zeroshotLabels(out.pooler_output.data, p.text_embeds, p.text_labels, e.post.logit_scale, e.post.logit_bias);
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
      // プロンプト: クリックした点と、動画で追う時は前のフレームのマスクから作った枠（p.box）
      const pts = p.points || [];
      if (!pts.length && !p.box) throw new Error("画像をクリックして点を指定してください");
      const prompt = {};
      if (pts.length) Object.assign(prompt, { input_points: [[pts.map(([x, y]) => [x, y])]], input_labels: [[pts.map(([, , l]) => l)]] });
      if (p.box) prompt.input_boxes = [[p.box]];
      const inputs = await st.proc(img, prompt);
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
      // 動画で追う時の次のプロンプト用に、マスクを囲む枠と、重心に一番近いマスクの中の点も返す
      let x0 = w, y0 = h, x1 = -1, y1 = -1, sx = 0, sy = 0, n = 0;
      for (let y = 0, i = 0; y < h; y++) for (let x = 0; x < w; x++, i++) {
        if (!src[best * w * h + i]) continue;
        px[i] = 255; n++; sx += x; sy += y;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
      }
      let track = null;
      if (n) {
        const cx = sx / n, cy = sy / n;
        let bp = [cx, cy], bd = Infinity;
        for (let y = y0; y <= y1; y++) for (let x = x0; x <= x1; x++) {
          if (!px[y * w + x]) continue;
          const d = (x - cx) ** 2 + (y - cy) ** 2;
          if (d < bd) { bd = d; bp = [x, y]; }
        }
        track = { box: [x0, y0, x1 + 1, y1 + 1], point: bp, area: n / (w * h) };
      }
      return { kind: "mask", mask: await toPng(new T.RawImage(px, w, h, 1)), score: scores[best], track };
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

// onnxruntime の実行は1つずつ順番に（画面が次のフレームを先に送ってくる＝パイプライン化しても、同じ wasm の中で実行が重ならないように）。
// 待ち時間は推論時間に入れない（fn の中で測る）
let ortQueue = Promise.resolve();
const serial = (fn) => { const p = ortQueue.then(fn, fn); ortQueue = p.catch(() => {}); return p; };

// 読み込んだファイルの取得元のまとめ（「読み込み」の横に出す）。transformers.js は自分のキャッシュを持つので分からない
function fetchSummary(log, e) {
  if (!log.length) return e.adapter === "onnx" ? null : e.mirror ? "サーバーの写し経由" : null;
  const mb = (k) => (log.filter((x) => x.from === k).reduce((a, x) => a + x.bytes, 0) / 1e6).toFixed(0);
  const words = { cache: "ブラウザのキャッシュ", mirror: "Mac のサーバー（保存済み）", "mirror-net": "ダウンロード（サーバーに保存）", net: "ダウンロード" };
  return [...new Set(log.map((x) => x.from))].map((k) => `${words[k]} ${mb(k)}MB`).join("・");
}

// 追跡の ReID 用: 切り出した画像の特徴を返す（onnx adapter のモデルだけ）
async function embed(id, e, crops) {
  try {
    await libReady;
    setMirror(e.mirror);
    if (!loaded.has(e.key)) {
      const onProgress = (file, progress) => self.postMessage({ type: "progress", key: e.key, file, progress });
      loaded.set(e.key, ADAPTERS.onnx.load(e, await getDevice(), onProgress));
      try { await loaded.get(e.key); } catch (err) { loaded.delete(e.key); throw err; }
    }
    const st = await loaded.get(e.key);
    const result = await serial(async () => {
      const t0 = performance.now();
      return { feats: await onnxEmbed(ort, st.session, e, crops), ms: performance.now() - t0 };
    });
    self.postMessage({ id, type: "result", result });
  } catch (err) {
    self.postMessage({ id, type: "error", message: String(err?.message || err) });
  }
}

// 記録の「実行場所」に出す、実際に使った設定（選んだ設定ではなく。fp16 版が無ければ fp32、graph capture を作れなければ無し）
function ortRuntime(e, session, device) {
  if (e.opt === "wasm" || device !== "webgpu") return `onnxruntime-web wasm ${ort.env.wasm.numThreads}スレッド`;
  const g = session?.cvpg || {};
  const pre = { gpu: " 前処理GPU", upload: " 前処理GPU（縮小はcanvas）", cpu: " 前処理CPU" }[g.pre] || "";
  return `onnxruntime-web ${g.fp16 ? "fp16" : "fp32"}${g.graph ? " graph" : ""}${pre}${g.postGpu ? " 後処理GPU" : ""}${g.graphFallback ? "（このモデルは graph capture 不可）" : ""}`;
}

// 読み込み済みのモデルを捨てる（ベンチで測り終えたモデル。スマホでは何個も載せると GPU のメモリが足りなくなる）
async function release(key) {
  const p = loaded.get(key);
  if (!p) return;
  loaded.delete(key);
  try {
    const st = await p;
    await st.session?.release?.(); await st.tplSession?.release?.();
    for (const k of ["model", "pipe"]) await st[k]?.dispose?.();
  } catch (err) { console.warn("モデルを捨てられない", key, err); }
}

// 詳細計測（?profile=1）: 最後の last 回の記録をまとめて返す。profiler は一度止めると再開できないので、モデルは捨てて次の実行で読み直す
async function profile(id, e, last) {
  try {
    const st = await loaded.get(e.key);
    loaded.delete(e.key);
    if (!st?.session) throw new Error("計測中のモデルが無い");
    const result = await serial(() => onnxProfile(st.session, last));
    st.session.release?.();
    self.postMessage({ id, type: "result", result });
  } catch (err) {
    self.postMessage({ id, type: "error", message: String(err?.message || err) });
  }
}

self.onmessage = async (ev) => {
  if (ev.data.type === "embed") return embed(ev.data.id, ev.data.model, ev.data.crops);
  if (ev.data.type === "profile") return profile(ev.data.id, ev.data.model, ev.data.last);
  if (ev.data.type === "release") return release(ev.data.key);
  const { id, model: e, image, params } = ev.data;
  try {
    await libReady;
    const A = ADAPTERS[e.adapter];
    if (!A) throw new Error(`ブラウザ側に adapter "${e.adapter}" が無い`);
    const device = await getDevice();
    let loadMs = 0, loadFrom = null;
    setMirror(e.mirror);
    if (!loaded.has(e.key)) {
      const t0 = performance.now();
      takeFetchLog();
      const onProgress = (file, progress) => self.postMessage({ type: "progress", key: e.key, file, progress });
      loaded.set(e.key, A.load(e, device, onProgress));
      try { await loaded.get(e.key); } catch (err) { loaded.delete(e.key); throw err; }
      loadMs = performance.now() - t0;
      loadFrom = fetchSummary(takeFetchLog(), e);
    }
    const st = await loaded.get(e.key);
    const img = image ? await toInput(image, A.image) : null; // 文エンコーダのように画像を使わないモデルもある
    const go = async () => {
      const t1 = performance.now();
      const r = await A.run(st, img, params, e);
      r.infer_ms = performance.now() - t1;
      return r;
    };
    const result = self.name === "ort" ? await serial(go) : await go();
    result.load_ms = loadMs;
    if (loadFrom) result.load_from = loadFrom;
    result.device = e.adapter === "onnx" && e.opt === "wasm" ? "wasm" : device;
    result.dtype = e.adapter === "onnx" ? ortRuntime(e, st.session, device) : `transformers.js ${TJS_VERSION} ${e.dtype?.[device] ?? ""}`.trim();
    self.postMessage({ id, type: "result", result });
  } catch (err) {
    self.postMessage({ id, type: "error", message: String(err?.message || err) });
  }
};
