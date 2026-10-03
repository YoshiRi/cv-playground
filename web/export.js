// 書き出し: 可視化画像の保存、結果データ（JSON）、実行の記録（CSV / JSON / Markdown）と端末の情報、統計。
// 端末の情報は書き出すファイルに入れるだけで、どこにも送らない。

// アプリの版（記録に入れる。大きく変えた時に上げる）
export const APP_VERSION = "2026.09.28";

// ---------- 統計 ----------

export function stats(xs) {
  const a = xs.filter((x) => Number.isFinite(x)).sort((p, q) => p - q);
  if (!a.length) return null;
  const q = (f) => a[Math.min(a.length - 1, Math.floor(f * (a.length - 1) + 0.5))];
  const mean = a.reduce((s, x) => s + x, 0) / a.length;
  return { n: a.length, mean, median: q(0.5), p90: q(0.9), p95: q(0.95), min: a[0], max: a[a.length - 1] };
}

// ---------- 端末の情報 ----------

let envCache = null;
export async function collectEnv(variant) {
  if (envCache) return envCache;
  const ua = navigator.userAgent, e = { variant, app_version: APP_VERSION, user_agent: ua };
  // ブラウザと OS（Chrome 系は User-Agent Client Hints で機種名も取れる）
  const uad = navigator.userAgentData;
  if (uad?.getHighEntropyValues) {
    try {
      const h = await uad.getHighEntropyValues(["model", "platformVersion", "fullVersionList"]);
      const b = (h.fullVersionList || uad.brands).find((x) => !/Not.?A.?Brand|Chromium/i.test(x.brand)) || (h.fullVersionList || uad.brands)[0];
      Object.assign(e, { browser: b ? `${b.brand} ${b.version}` : "", os: `${uad.platform} ${h.platformVersion || ""}`.trim(), device_model: h.model || "" });
    } catch { /* 取れない環境 */ }
  }
  if (!e.browser) {
    const m = ua.match(/(Edg|OPR|Chrome|Firefox|Version)\/([\d.]+)/);
    e.browser = m ? `${m[1] === "Version" ? "Safari" : m[1]} ${m[2]}` : "";
    e.os = /iPhone|iPad/.test(ua) ? `iOS ${(ua.match(/OS ([\d_]+)/)?.[1] || "").replace(/_/g, ".")}` : /Android ([\d.]+)/.test(ua) ? `Android ${RegExp.$1}` : /Mac OS X ([\d_]+)/.test(ua) ? `macOS ${RegExp.$1.replace(/_/g, ".")}` : /Windows/.test(ua) ? "Windows" : "";
    e.device_model = /Android[^;]*; ([^;)]+)/.test(ua) ? RegExp.$1.trim() : /iPad/.test(ua) ? "iPad" : /iPhone/.test(ua) ? "iPhone" : "";
  }
  Object.assign(e, {
    cpu_threads: navigator.hardwareConcurrency ?? "",
    device_memory_gb: navigator.deviceMemory ?? "",
    cross_origin_isolated: self.crossOriginIsolated,
    wasm_threads: self.crossOriginIsolated ? Math.min(4, navigator.hardwareConcurrency || 4) : 1,
    screen: `${screen.width}x${screen.height}@${devicePixelRatio}`,
  });
  try {
    const ad = navigator.gpu && (await navigator.gpu.requestAdapter());
    const i = ad?.info || {};
    e.webgpu = !!ad;
    e.gpu = [i.vendor, i.architecture, i.device, i.description].filter(Boolean).join(" ");
  } catch { e.webgpu = false; e.gpu = ""; }
  envCache = e;
  return e;
}

// ---------- CSV / JSON / Markdown ----------

// 1行 = 1回の実行（連続実行・ベンチマークは1まとめ）。端末の列を毎行に入れるので、別の端末の CSV をそのまま連結して比べられる
export const RUN_COLUMNS = [
  "time", "mode", "task", "model_key", "model_name", "where", "device", "runtime", "input_size", "frame_w", "frame_h",
  "load_ms", "infer_ms", "grab_ms", "pre_ms", "run_ms", "post_ms", "roundtrip_ms", "reid_ms", "cascade_ms",
  "frames", "fps", "infer_mean_ms", "infer_median_ms", "infer_p90_ms", "infer_min_ms", "infer_max_ms", "warmup", "bench_image", "summary",
  "infer_p95_ms", "gpu_ms", // 後から足した列（前の版の CSV と列の位置が変わらないように末尾に置く）
  "extract_ms", "match_ms", "ransac_ms", "inliers", "matches", "quad_err_px", // テンプレートマッチングの後処理の内訳と、ベンチの四隅の誤差
  "jitter_raw_px", "jitter_out_px", "jitter_raw_deg", "jitter_out_deg", // 手ぶれ補正: 揺れ（細かい成分）の二乗平均（補正前・補正後）
  "move_raw_px", "move_out_px", "stab_mode", "stab_strength", "stab_crop", "stab_clamped", "stab_delay",
  "decisions", // 判定（ゼロショット分類）: 値(はいの確からしさ) を空白区切り // 手ぶれ補正: 動き（全体）の二乗平均と、補正の設定
];
export const ENV_COLUMNS = ["variant", "app_version", "browser", "os", "device_model", "gpu", "webgpu", "cpu_threads", "device_memory_gb",
  "wasm_threads", "cross_origin_isolated", "screen", "user_agent"];

const cell = (v) => {
  if (v == null) return "";
  const s = typeof v === "number" ? (Number.isInteger(v) ? String(v) : v.toFixed(2)) : String(v);
  return /[",\n\r]/.test(s) ? `"${s.replace(/"/g, '""')}"` : s;
};

export function toCSV(runs, env) {
  const cols = [...RUN_COLUMNS, ...ENV_COLUMNS];
  const rows = runs.map((r) => cols.map((c) => cell(c in r ? r[c] : env[c])).join(","));
  // BOM 付き UTF-8（Excel で日本語が化けないように）
  return "﻿" + [cols.join(","), ...rows].join("\r\n") + "\r\n";
}

export function toJSON(runs, env) {
  return JSON.stringify({ app: "cv-playground", app_version: APP_VERSION, exported_at: new Date().toISOString(), env, runs }, null, 1);
}

export function toMarkdown(runs, env) {
  const ms = (v) => (v == null || v === "" ? "" : Number(v).toFixed(v < 10 ? 1 : 0));
  const head = `端末: ${[env.device_model, env.os, env.browser].filter(Boolean).join(" / ")}${env.gpu ? ` / GPU: ${env.gpu}` : ""}（${env.variant}、CV Playground ${APP_VERSION}）\n\n`;
  // 内訳（前処理・モデル実行・後処理、ベンチと連続実行は平均）も入れる。どこが重いかを貼っただけで読めるように
  const lines = ["| 日時 | 種類 | タスク | モデル | 実行場所 | 入力 | 推論 ms（中央値） | p90 | p95 | 前処理 | モデル実行 | 後処理 | GPU（詳細計測） | fps | 読み込み ms | 後処理の内訳など |",
    "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"];
  // テンプレートマッチング: 点の取り出し・対応・RANSAC（ms）と、インライア/対応、ベンチなら正解の四隅からの誤差
  const detail = (r) => (r.decisions ? `判定 ${r.decisions}` : r.match_ms == null ? "" : `点 ${ms(r.extract_ms)} / 対応 ${ms(r.match_ms)} / RANSAC ${ms(r.ransac_ms)} ・ ${r.inliers}/${r.matches}${r.quad_err_px != null && r.quad_err_px !== "" ? ` ・ 四隅 ${Number(r.quad_err_px).toFixed(1)}px` : ""}`
    + (r.jitter_raw_px != null ? ` ・ ${r.stab_mode === "tripod" ? "三脚" : r.stab_mode ? `なめらか${{ weak: "弱", mid: "中", strong: "強" }[r.stab_strength] ?? ""}` : ""}${r.stab_delay ? ` 先読み${r.stab_delay}` : ""}${r.stab_crop != null ? ` ${Math.round(r.stab_crop * 100)}%` : ""} 揺れ ${r.jitter_raw_px}px/${r.jitter_raw_deg}° → ${r.jitter_out_px}px/${r.jitter_out_deg}°${r.move_raw_px != null ? `（動き ${r.move_raw_px} → ${r.move_out_px}px）` : ""}${r.stab_clamped ? ` ・ 端 ${r.stab_clamped}/${r.frames_total ?? ""}` : ""}` : ""));
  for (const r of runs) {
    lines.push(`| ${r.time.slice(5, 16).replace("T", " ")} | ${{ single: "1回", live: "連続", bench: "ベンチ" }[r.mode] || r.mode} | ${r.task} | ${r.model_name} | ${r.where} ${r.device}${r.runtime ? " " + r.runtime : ""} | ${r.input_size || `${r.frame_w}×${r.frame_h}`} | ${ms(r.infer_median_ms ?? r.infer_ms)} | ${ms(r.infer_p90_ms)} | ${ms(r.infer_p95_ms)} | ${ms(r.pre_ms)} | ${ms(r.run_ms)} | ${ms(r.post_ms)} | ${ms(r.gpu_ms)} | ${r.fps ? Number(r.fps).toFixed(1) : ""} | ${ms(r.load_ms)} | ${detail(r)} |`);
  }
  return head + lines.join("\n") + "\n";
}

// ---------- 保存 ----------

export const stamp = () => {
  const d = new Date(), z = (n) => String(n).padStart(2, "0");
  return `${d.getFullYear()}${z(d.getMonth() + 1)}${z(d.getDate())}-${z(d.getHours())}${z(d.getMinutes())}${z(d.getSeconds())}`;
};
export const safeName = (s) => String(s).replace(/[\\/:*?"<>|\s（）()、]+/g, "_").replace(/_+/g, "_").replace(/^_|_$/g, "");

export function download(blob, filename) {
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = filename;
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(a.href), 10000);
}

// スマホ（共有シートが使える環境）では共有シートを出し、「写真に保存」などを選べるようにする。使えなければダウンロード
export async function shareOrDownload(blob, filename) {
  const file = new File([blob], filename, { type: blob.type });
  const touch = matchMedia("(pointer: coarse)").matches;
  if (touch && navigator.canShare?.({ files: [file] })) {
    try { await navigator.share({ files: [file], title: filename }); return "shared"; }
    catch (e) { if (e.name === "AbortError") return "cancelled"; }
  }
  download(blob, filename);
  return "downloaded";
}

// 表示中の canvas に、下に説明の帯（モデル名・実行場所・推論時間・日時）を付けた画像
export async function composeImage(canvas, lines) {
  const w = canvas.width, h = canvas.height;
  if (!lines?.length) return new Promise((res) => canvas.toBlob(res, "image/png"));
  const fs = Math.max(13, Math.round(w / 55)), pad = Math.round(fs * 0.7), bh = lines.length * fs * 1.45 + pad * 2;
  const c = document.createElement("canvas");
  c.width = w; c.height = Math.round(h + bh);
  const x = c.getContext("2d");
  x.drawImage(canvas, 0, 0);
  x.fillStyle = "#0b0f14"; x.fillRect(0, h, w, bh);
  x.fillStyle = "#e6e9ef"; x.textBaseline = "top";
  lines.forEach((t, i) => {
    x.font = `${i === 0 ? "600 " : ""}${fs}px system-ui, -apple-system, "Hiragino Sans", sans-serif`;
    x.fillStyle = i === 0 ? "#e6e9ef" : "#98a2b3";
    x.fillText(t, pad, h + pad + i * fs * 1.45, w - pad * 2);
  });
  return new Promise((res) => c.toBlob(res, "image/png"));
}

// 結果データ: 画像の層（canvas・Blob）を除いた、数値とラベルだけ
export function resultData(r, m, extra) {
  const drop = new Set(["layer", "gray", "alpha", "tint", "mask", "image", "t8", "fixedLayer", "fixedFor", "templateImage"]);
  const out = {};
  for (const [k, v] of Object.entries(r)) if (!drop.has(k)) out[k] = v;
  if (r.mask || r.image) out.note = "マスク・深度・色分けの画像は「画像を保存」で保存する（ここには入れない）";
  return { app: "cv-playground", app_version: APP_VERSION, model: { key: m.key, name: m.name, where: m.where, license: m.license }, ...extra, result: out };
}
