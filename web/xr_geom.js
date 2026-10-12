// AR の地図まわりの座標の計算（WebXR に依存しない。ライブと記録の再生で同じものを使う）
//   行列は列優先の 4×4（WebXR の XRRigidTransform.matrix・projectionMatrix と同じ）
//   座標は WebXR の local 空間: y が上（重力の逆）、カメラは −z を向く
//   view の座標は 0〜1（左上が 0）。深度は「カメラの前向きの距離」（m）

// view の 0〜1 の点を、深度 d で部屋の座標の点にする。射影行列 P から ndc.x = (P0·x + P8·z) / −z（z = −d）なので
// x = (ndc.x + P8)·d / P0。M はカメラ → 部屋
export function unproject(P, M, x, y, d) {
  const nx = 2 * x - 1, ny = 1 - 2 * y;
  const vx = ((nx + P[8]) * d) / P[0], vy = ((ny + P[9]) * d) / P[5], vz = -d;
  return [M[0] * vx + M[4] * vy + M[8] * vz + M[12], M[1] * vx + M[5] * vy + M[9] * vz + M[13], M[2] * vx + M[6] * vy + M[10] * vz + M[14]];
}
// 部屋の座標の点を view の 0〜1 に（C は部屋 → カメラ。カメラの後ろなら null）
export function project(P, C, p) {
  const x = C[0] * p[0] + C[4] * p[1] + C[8] * p[2] + C[12], y = C[1] * p[0] + C[5] * p[1] + C[9] * p[2] + C[13], z = C[2] * p[0] + C[6] * p[1] + C[10] * p[2] + C[14];
  if (z > -0.05) return null;
  const nx = (P[0] * x + P[8] * z) / -z, ny = (P[5] * y + P[9] * z) / -z;
  return [(nx + 1) / 2, (1 - ny) / 2];
}
// 剛体変換の逆（回転の転置と、平行移動）
export function invRigid(M) {
  const t = [M[12], M[13], M[14]];
  return [M[0], M[4], M[8], 0, M[1], M[5], M[9], 0, M[2], M[6], M[10], 0,
    -(M[0] * t[0] + M[1] * t[1] + M[2] * t[2]), -(M[4] * t[0] + M[5] * t[1] + M[6] * t[2]), -(M[8] * t[0] + M[9] * t[1] + M[10] * t[2]), 1];
}

// ---------- 深度の写し ----------
// XR の深度（XRCPUDepthInformation）はそのフレームの中でしか読めないので、写しを取る。地図には 40×24 点しか使わないので、
// 半分に間引いて（最近傍。平均すると物の縁の深度が混ざる）記録も軽くする。m は view の 0〜1 → 深度の画像の 0〜1
export function copyDepth(d, f32, step = 2) {
  const src = f32 ? new Float32Array(d.data) : new Uint16Array(d.data), w = Math.floor(d.width / step), h = Math.floor(d.height / step);
  const out = new Uint16Array(w * h), k = f32 ? 0.001 : d.rawValueToMeters;
  for (let j = 0; j < h; j++) for (let i = 0; i < w; i++) {
    const v = src[j * step * d.width + i * step];
    out[j * w + i] = f32 ? Math.min(65535, Math.round(v * 1000)) : v; // float32 の時は mm の整数にそろえる
  }
  return { w, h, k, m: Array.from(d.normDepthBufferFromNormView.matrix), data: out };
}
// 写しの深度（m。無い所は 0）
export function depthAt(c, x, y) {
  const u = c.m[0] * x + c.m[4] * y + c.m[12], v = c.m[1] * x + c.m[5] * y + c.m[13];
  const col = Math.min(c.w - 1, Math.max(0, Math.floor(u * c.w))), row = Math.min(c.h - 1, Math.max(0, Math.floor(v * c.h)));
  return c.data[row * c.w + col] * c.k;
}
// 枠（view の 0〜1）の中央 6 割の 5×5 点の深度の中央値（m）。背景を拾いにくくする
export function boxDepth(c, b) {
  const v = [];
  for (let i = 0; i < 5; i++) for (let j = 0; j < 5; j++) {
    const m = depthAt(c, Math.min(0.999, Math.max(0, b[0] + (b[2] - b[0]) * (0.2 + 0.15 * i))), Math.min(0.999, Math.max(0, b[1] + (b[3] - b[1]) * (0.2 + 0.15 * j))));
    if (m > 0) v.push(m);
  }
  if (!v.length) return null;
  v.sort((p, q) => p - q);
  return v[v.length >> 1];
}

// ---------- 4 自由度の姿勢 {x, y, z, yaw} ----------
// 重力の向きは XR で分かっているので、地図の姿勢は位置 3 つと、y 軸まわりの向き yaw だけで持つ（傾きは持たない）。
// 回転は Ry(yaw): x' = c·x + s·z、z' = −s·x + c·z
export const P4 = (x = 0, y = 0, z = 0, yaw = 0) => ({ x, y, z, yaw });
export const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));
// 行列（剛体変換）から 4 自由度の姿勢（x 軸を水平面に落とした向き）
export function p4FromMatrix(M) { return P4(M[12], M[13], M[14], Math.atan2(-M[2], M[0])); }
export function p4Apply(a, p) {
  const c = Math.cos(a.yaw), s = Math.sin(a.yaw);
  return [c * p[0] + s * p[2] + a.x, p[1] + a.y, -s * p[0] + c * p[2] + a.z];
}
export function p4Inv(a) {
  const c = Math.cos(-a.yaw), s = Math.sin(-a.yaw);
  return P4(-(c * a.x + s * a.z), -a.y, -(-s * a.x + c * a.z), -a.yaw);
}
// a ⊕ b（b を a の座標で表したものを、a の外へ）
export function p4Compose(a, b) {
  const [x, y, z] = p4Apply(a, [b.x, b.y, b.z]);
  return P4(x, y, z, wrap(a.yaw + b.yaw));
}
// a から見た b（a⁻¹ ⊕ b）
export const p4Between = (a, b) => p4Compose(p4Inv(a), b);
