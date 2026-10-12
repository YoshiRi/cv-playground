// AR の記録（センサーの列）の書き出しと読み込み。1 行 1 つの JSON（NDJSON）を gzip で固める
//   記録の種類（k）:
//     meta  … 始まり（日時・端末・使えた機能・モデル）
//     P     … 射影行列（変わった時だけ）
//     f     … 毎フレームのカメラの姿勢 M（カメラ → 部屋、列優先の 4×4）。t はフレームの時刻（ms）
//     d     … 地図に積む深度の写し（Uint16 を base64。k を掛けると m）。直前の f の姿勢で積む
//     det   … 検出の結果（3D にした後: 枠 v は view の 0〜1、深度 depth、板の 4 隅 world、中心 center）。t は読み戻したフレームの時刻
//     sub   … サブマップを足した（アンカーの id と置いた時の姿勢）
//     a     … アンカーの今の姿勢 [[id, M], …]
//     end   … 終わり
//   ライブでも再生でも、xr_pipeline.js がこの記録を同じ順に処理するので、再生した地図はライブと同じになる
const r5 = (v) => Math.round(v * 1e5) / 1e5;
export const roundM = (M) => Array.from(M, r5);

export class LogWriter {
  constructor() {
    const cs = new CompressionStream("gzip");
    this.w = cs.writable.getWriter(); this.enc = new TextEncoder();
    this.chunks = []; this.bytes = 0; this.raw = 0; this.n = 0;
    const r = cs.readable.getReader();
    this.done = (async () => { for (;;) { const { value, done } = await r.read(); if (done) break; this.chunks.push(value); this.bytes += value.length; } })();
  }
  write(obj) {
    const b = this.enc.encode(JSON.stringify(obj) + "\n");
    this.raw += b.length; this.n++;
    this.w.write(b);
  }
  async close() { await this.w.close(); await this.done; return new Blob(this.chunks, { type: "application/gzip" }); }
}

export async function* readLog(blob) {
  const reader = blob.stream().pipeThrough(new DecompressionStream("gzip")).pipeThrough(new TextDecoderStream()).getReader();
  let buf = "";
  for (;;) {
    const { value, done } = await reader.read();
    if (value) buf += value;
    let i;
    while ((i = buf.indexOf("\n")) >= 0) { const line = buf.slice(0, i); buf = buf.slice(i + 1); if (line) yield JSON.parse(line); }
    if (done) break;
  }
  if (buf.trim()) yield JSON.parse(buf);
}

// 深度の写し ⇔ 記録の形
export function packDepth(c) {
  const bytes = new Uint8Array(c.data.buffer, c.data.byteOffset, c.data.byteLength);
  let s = "";
  for (let i = 0; i < bytes.length; i += 0x8000) s += String.fromCharCode(...bytes.subarray(i, i + 0x8000));
  return { w: c.w, h: c.h, k: c.k, m: c.m.map(r5), b64: btoa(s) };
}
export function unpackDepth(o) {
  const s = atob(o.b64), bytes = new Uint8Array(s.length);
  for (let i = 0; i < s.length; i++) bytes[i] = s.charCodeAt(i);
  return { w: o.w, h: o.h, k: o.k, m: o.m, data: new Uint16Array(bytes.buffer) };
}
