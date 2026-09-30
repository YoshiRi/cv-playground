// インタラクトの部品: 検出の結果を決まった形の「フレーム」にして、受け手（sink）に渡す。受け手は自分の画面（canvas など）を持ち、
// フレームを受けて動く（例: 一番近い人の骨格に合わせてキャラが動く）。モデルの後処理・応用（apps.js）とは別の層で、
// 受け手はモデルを知らなくてよい（フレームの形だけに頼る）。models.json の interact に {id, name, tasks, hint} を書くと、
// そのタブに「インタラクト」のチェックが出て、選ぶと結果の下（広い画面では横）に受け手の画面を出す
//
// フレーム（v: 1）: 座標は画像の幅・高さで割った 0〜1
//   { v, t, w, h, task, model,
//     items: [{ id?, label, score, box: [x1, y1, x2, y2], keypoints?: [[x, y, 可視度]], emotion?, state? }] }
//   keypoints は 17 点なら COCO の順（鼻・左目・右目・左耳・右耳・左肩・右肩・左肘・右肘・左手首・右手首・…）、
//   5 点なら顔（右目・左目・鼻・口の右端・左端。YuNet）。左右は写っている人から見た向き
//
// INTERACT[id] = {
//   create(el, opts) → 受け手。el は受け手用の空の要素。{ onFrame(frame), destroy() } を返す
// }

export function toFrame(r, meta = {}) {
  const w = r.w || 1, h = r.h || 1;
  return {
    v: 1, t: performance.now(), w, h, ...meta,
    items: (r.items || []).map((it) => ({
      ...(it.id != null ? { id: it.id } : {}), label: it.label, score: it.score,
      box: [it.box[0] / w, it.box[1] / h, it.box[2] / w, it.box[3] / h],
      ...(it.keypoints ? { keypoints: it.keypoints.map(([x, y, v]) => [x / w, y / h, v ?? 1]) } : {}),
      ...(it.emotion ? { emotion: it.emotion } : {}), ...(it.state ? { state: it.state } : {}),
    })),
  };
}

// 一番近い物を選ぶ: 枠が一番大きい物（カメラに近いほど大きく写る）。追跡の ID があれば、前に選んだ物を
// 別の物の枠が 1.5 倍より大きくなるまで選び続ける（大きさが近い人が並んでも、選ぶ人がちらちら替わらないように）
export function pickTarget(frame, sel, ok = () => true) {
  const area = (it) => (it.box[2] - it.box[0]) * (it.box[3] - it.box[1]) * frame.w * frame.h;
  const cands = frame.items.filter(ok);
  if (!cands.length) return null;
  const best = cands.reduce((a, b) => (area(b) > area(a) ? b : a));
  const cur = sel.id != null ? cands.find((it) => it.id === sel.id) : null;
  const pick = cur && area(best) < 1.5 * area(cur) ? cur : best;
  sel.id = pick.id;
  return pick;
}

// ---------------------------------------------------------------- キャラ（Live2D 風）
// Live2D と同じく「パラメータ」を挟む: フレーム → パラメータの目標（頭の向き 3 軸・体の傾き・腕・目・口・眉）→ 画面の
// 更新ごとにばねで目標に寄せる → パーツの絵をパラメータで回す・ずらす。推論が 10fps でも、表示は 60fps でなめらかに動く。
// 頭の左右の向きは、奥にある髪と手前の前髪・顔のパーツを逆向きにずらして立体に見せる（Live2D の定番の見せ方）
const P0 = { angX: 0, angY: 0, angZ: 0, bodyZ: 0, posX: 0, armL: [1.9, 1.75], armR: [1.24, 1.39], smile: 0.3, mouthOpen: 0, brow: 0, browIn: 0, eyeWide: 0, happyEye: 0 };
const EMO_FACE = { // 表情（HSEmotion）→ 顔のパラメータ
  喜び: { smile: 1, happyEye: 1, brow: 0.2 }, 驚き: { mouthOpen: 0.9, eyeWide: 1, brow: 1, smile: 0 },
  悲しみ: { smile: -0.7, brow: -0.2, browIn: 1 }, 怒り: { smile: -0.6, brow: -0.6, browIn: -1 },
  恐れ: { eyeWide: 0.7, brow: 0.6, browIn: 1, smile: -0.4, mouthOpen: 0.3 }, 嫌悪: { smile: -0.8, brow: -0.4, browIn: -0.5 },
  軽蔑: { smile: -0.3, brow: -0.2 }, 無表情: { smile: 0 },
};
const clamp = (v, a, b) => Math.min(b, Math.max(a, v));
const wrap = (a) => Math.atan2(Math.sin(a), Math.cos(a));

// 骨格・顔の点から、パラメータの目標を作る。角度は画面の上の向きのまま使う（キャラは画面の中の人の鏡のように動く）
function targetsFrom(it, frame) {
  const T = { ...P0 };
  const k = it.keypoints, W = frame.w, H = frame.h, V = 0.4;
  const pt = (i) => (k[i] && k[i][2] >= V ? [k[i][0] * W, k[i][1] * H] : null);
  const ang = (a, b) => Math.atan2(b[1] - a[1], b[0] - a[0]);
  T.posX = clamp(((it.box[0] + it.box[2]) / 2 - 0.5) * 0.4, -0.15, 0.15); // 人が画面の端にいる時は、キャラも少しだけ寄る
  let eyeR, eyeL, nose, mouth = null;
  if (k?.length === 5) { eyeR = pt(0); eyeL = pt(1); nose = pt(2); const a = pt(3), b = pt(4); if (a && b) mouth = [(a[0] + b[0]) / 2, (a[1] + b[1]) / 2]; }
  else if (k?.length >= 17) { eyeR = pt(2); eyeL = pt(1); nose = pt(0); }
  if (eyeR && eyeL) {
    const d = Math.hypot(eyeL[0] - eyeR[0], eyeL[1] - eyeR[1]) || 1, mid = [(eyeR[0] + eyeL[0]) / 2, (eyeR[1] + eyeL[1]) / 2];
    T.angZ = clamp(wrap(ang(eyeR, eyeL)), -0.6, 0.6); // 右目（画面の左）→ 左目の線の傾き
    if (nose) {
      // 目の線に沿った座標にしてから、鼻が目の中点からどれだけずれたか
      const c = Math.cos(-T.angZ), s = Math.sin(-T.angZ), dx = nose[0] - mid[0], dy = nose[1] - mid[1];
      const u = (dx * c - dy * s) / d, v = (dx * s + dy * c) / d;
      T.angX = clamp(u / 0.45, -1, 1); // 左右（正 = 画面の右を向く）
      // 上下: 顔の 5 点なら 目→鼻 ÷ 目→口（正面で約 0.5）、骨格なら 目→鼻 ÷ 目の間隔（正面で約 0.45）。正 = 下を向く
      const m = mouth ? ((mouth[0] - mid[0]) * s + (mouth[1] - mid[1]) * c) / d : 0;
      T.angY = clamp(mouth && m > 0.2 ? (v / m - 0.5) / 0.2 : (v - 0.45) / 0.3, -1, 1);
    }
  }
  if (k?.length >= 17) {
    const sL = pt(5), sR = pt(6);
    if (sL && sR) T.bodyZ = clamp(wrap(ang(sR, sL)), -0.5, 0.5);
    // 腕: 画面の左に出す腕は右腕（6 → 8 → 10）。見えない所はおろした形
    const arm = (s, e, w, rest) => {
      const S = pt(s), E = pt(e), Wr = pt(w);
      if (!S || !E) return rest;
      const up = ang(S, E);
      return [up, Wr ? ang(E, Wr) : up];
    };
    T.armL = arm(6, 8, 10, P0.armL);
    T.armR = arm(5, 7, 9, P0.armR);
  } else T.bodyZ = T.angZ * 0.35; // 顔だけの時は体を少しだけ頭についていかせる
  if (it.emotion) {
    const f = EMO_FACE[it.emotion.label] || {}, a = clamp((it.emotion.prob - 0.3) / 0.5, 0, 1);
    for (const [key, v] of Object.entries(f)) T[key] = P0[key] + (v - P0[key]) * a;
  }
  return T;
}

function puppet(el) {
  const cv = document.createElement("canvas"), info = document.createElement("div");
  cv.className = "interact-canvas"; info.className = "interact-info small muted";
  el.append(cv, info);
  const ctx = cv.getContext("2d");
  const P = structuredClone(P0);
  let T = structuredClone(P0), lastSeen = 0, lostAt = 0, raf = 0, prev = performance.now(), sway = 0, swayV = 0, prevZ = 0;
  const sel = {}, t0 = performance.now();
  let blinkAt = t0 + 2000;

  function onFrame(frame) {
    const it = pickTarget(frame, sel, (x) => x.keypoints?.length >= 5);
    if (!it) { if (lastSeen && !lostAt) lostAt = performance.now(); info.textContent = lastSeen ? "人が見つからない（最後の姿勢から戻る）" : "骨格か顔の点を出すモデルの結果を待っている"; return; }
    T = targetsFrom(it, frame);
    lastSeen = performance.now(); lostAt = 0;
    info.textContent = `追っている: ${it.id != null ? "#" + it.id + " " : ""}${it.label}（一番大きく写っている${it.label === "face" ? "顔" : "人"}）`
      + (it.emotion ? ` ・ 表情 ${it.emotion.label}` : "");
  }

  function step(now) {
    const dt = Math.min(0.1, (now - prev) / 1000); prev = now;
    // 結果に人がいない状態が 1.5 秒続いたら元の姿勢へ（結果が来ない間＝静止画は、最後の姿勢のまま）
    const tgt = lostAt && now - lostAt > 1500 ? P0 : T;
    const a = 1 - Math.exp(-dt * 10), aArm = 1 - Math.exp(-dt * 14);
    for (const key of Object.keys(P0)) {
      if (Array.isArray(P0[key])) P[key] = P[key].map((v, i) => v + wrap(tgt[key][i] - v) * aArm);
      else P[key] += (tgt[key] - P[key]) * a;
    }
    // 揺れもの（横の髪）: 頭の傾きの速さで振れて、ばねで戻る
    const vz = (P.angZ - prevZ) / Math.max(dt, 1e-3); prevZ = P.angZ;
    swayV += (-sway * 60 - swayV * 7 - vz * 4 - P.angX * 3) * dt; sway += swayV * dt;
    render(now);
    raf = requestAnimationFrame(step);
  }

  function render(now) {
    const dpr = Math.min(2, window.devicePixelRatio || 1), cw = cv.clientWidth || 400, ch = cv.clientHeight || 400;
    if (cv.width !== Math.round(cw * dpr) || cv.height !== Math.round(ch * dpr)) { cv.width = Math.round(cw * dpr); cv.height = Math.round(ch * dpr); }
    const W = cv.width, H = cv.height, u = Math.min(W, H) / 10;
    const bg = ctx.createLinearGradient(0, 0, 0, H);
    bg.addColorStop(0, "#dbeafe"); bg.addColorStop(1, "#fce7f3");
    ctx.fillStyle = bg; ctx.fillRect(0, 0, W, H);
    const t = (now - t0) / 1000, breath = Math.sin(t * 2.2) * 0.04;
    // まばたき: 3〜6 秒ごと、0.15 秒で閉じて開く
    if (now > blinkAt + 150) blinkAt = now + 3000 + Math.random() * 3000;
    const blink = now > blinkAt ? 1 - Math.sin(Math.min(1, (now - blinkAt) / 150) * Math.PI) : 1;
    const hip = [W / 2 + P.posX * W, H * 0.98];
    ctx.save();
    ctx.fillStyle = "rgba(0,0,0,.08)"; ctx.beginPath(); ctx.ellipse(hip[0], H * 0.985, u * 2.4, u * 0.25, 0, 0, Math.PI * 2); ctx.fill();
    // 体（腰を軸に傾ける）
    ctx.translate(...hip); ctx.rotate(P.bodyZ);
    const top = -u * (4.3 + breath);
    const shL = [-u * 1.45, top + u * 0.35], shR = [u * 1.45, top + u * 0.35]; // 画面の左・右の肩
    ctx.fillStyle = "#5b8def";
    rrect(ctx, -u * 1.7, top, u * 3.4, -top + u, u * 0.9); ctx.fill();
    ctx.fillStyle = "#ffffff"; ctx.beginPath(); ctx.moveTo(-u * 0.55, top); ctx.lineTo(0, top + u * 0.8); ctx.lineTo(u * 0.55, top); ctx.closePath(); ctx.fill(); // 襟
    ctx.fillStyle = "#ffe3d3"; rrect(ctx, -u * 0.33, top - u * 0.55, u * 0.66, u * 0.8, u * 0.2); ctx.fill(); // 首
    // 頭（首の上を軸に、体の傾き + 頭の傾き）
    ctx.save();
    ctx.translate(0, top - u * 0.35); ctx.rotate(P.angZ - P.bodyZ);
    head(ctx, u, blink);
    ctx.restore();
    // 腕（角度は画面の上の向き。体の回転を戻してから描く）
    for (const [sh, [a1, a2]] of [[shL, P.armL], [shR, P.armR]]) arm(ctx, u, sh, a1 - P.bodyZ, a2 - P.bodyZ);
    ctx.restore();
  }

  function head(ctx, u, blink) {
    const X = P.angX, Y = P.angY, hy = -u * 1.7; // 顔の中心（首の上から）
    const off = (depth) => [X * u * depth, Y * u * depth * 0.8]; // 手前ほど大きくずらす
    const hairCol = "#2b2f4a";
    // 後ろの髪（奥なので逆にずらす）
    let [dx, dy] = off(-0.15);
    ctx.fillStyle = hairCol;
    ctx.beginPath(); ctx.ellipse(dx, hy + dy + u * 0.1, u * 2.05, u * 2.1, 0, 0, Math.PI * 2); ctx.fill();
    rrect(ctx, dx - u * 2.05, hy + dy, u * 4.1, u * 2.4, u * 0.8); ctx.fill();
    // 横の髪（揺れもの）
    for (const s of [-1, 1]) {
      ctx.save(); ctx.translate(s * u * 1.75 + dx * 0.5, hy - u * 0.3); ctx.rotate(sway * 0.5);
      ctx.beginPath(); ctx.ellipse(0, u * 1.2, u * 0.42, u * 1.5, 0, 0, Math.PI * 2); ctx.fill(); ctx.restore();
    }
    // 顔（左右を向くと少し細く）
    [dx, dy] = off(0.12);
    ctx.fillStyle = "#ffe3d3";
    ctx.beginPath(); ctx.ellipse(dx, hy + dy + u * 0.15, u * 1.65 * (1 - Math.abs(X) * 0.08), u * 1.75, 0, 0, Math.PI * 2); ctx.fill();
    // 頬
    [dx, dy] = off(0.4);
    ctx.fillStyle = `rgba(255,120,140,${0.25 + P.smile * 0.15})`;
    for (const s of [-1, 1]) { ctx.beginPath(); ctx.ellipse(dx + s * u * 0.95, hy + dy + u * 0.75, u * 0.32, u * 0.18, 0, 0, Math.PI * 2); ctx.fill(); }
    // 目（奥の目は小さく）
    [dx, dy] = off(0.5);
    for (const s of [-1, 1]) {
      const ex = dx + s * u * 0.68, ey = hy + dy + u * 0.2, scl = 1 - s * X * 0.12, open = clamp(blink * (1 + P.eyeWide * 0.25), 0.05, 1.3);
      if (P.happyEye > 0.5 && blink > 0.5) { // 笑い目（^ ^）
        ctx.strokeStyle = "#3a2e3f"; ctx.lineWidth = u * 0.12; ctx.lineCap = "round";
        ctx.beginPath(); ctx.arc(ex, ey + u * 0.15, u * 0.33 * scl, Math.PI * 1.15, Math.PI * 1.85); ctx.stroke();
        continue;
      }
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.ellipse(ex, ey, u * 0.36 * scl, u * 0.46 * open, 0, 0, Math.PI * 2); ctx.fill();
      ctx.save(); ctx.beginPath(); ctx.ellipse(ex, ey, u * 0.36 * scl, u * 0.46 * open, 0, 0, Math.PI * 2); ctx.clip();
      const ix = ex + X * u * 0.12, iy = ey + Y * u * 0.1; // 黒目は向いた方へ少し寄る
      ctx.fillStyle = "#4a78c8"; ctx.beginPath(); ctx.ellipse(ix, iy, u * 0.27 * scl, u * 0.36, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#1e2a4a"; ctx.beginPath(); ctx.ellipse(ix, iy + u * 0.03, u * 0.14 * scl, u * 0.2, 0, 0, Math.PI * 2); ctx.fill();
      ctx.fillStyle = "#fff"; ctx.beginPath(); ctx.arc(ix - u * 0.09, iy - u * 0.14, u * 0.08, 0, Math.PI * 2); ctx.fill();
      ctx.restore();
      ctx.strokeStyle = "#3a2e3f"; ctx.lineWidth = u * 0.09; ctx.lineCap = "round"; // まつげの線
      ctx.beginPath(); ctx.ellipse(ex, ey, u * 0.38 * scl, u * 0.48 * open, 0, Math.PI * 1.1, Math.PI * 1.9); ctx.stroke();
    }
    // 眉（上げ下げと寄せ）
    [dx, dy] = off(0.55);
    ctx.strokeStyle = "#2b2f4a"; ctx.lineWidth = u * 0.1; ctx.lineCap = "round";
    for (const s of [-1, 1]) {
      const bx = dx + s * u * 0.7, by = hy + dy - u * 0.5 - P.brow * u * 0.18, tilt = P.browIn * 0.35 * -s;
      ctx.beginPath(); ctx.moveTo(bx - u * 0.3 * Math.cos(tilt), by + u * 0.3 * Math.sin(tilt)); ctx.lineTo(bx + u * 0.3 * Math.cos(tilt), by - u * 0.3 * Math.sin(tilt)); ctx.stroke();
    }
    // 口（笑う・開く・への字）
    [dx, dy] = off(0.45);
    const mx = dx, my = hy + dy + u * 1.0;
    if (P.mouthOpen > 0.15) {
      ctx.fillStyle = "#b8405a"; ctx.beginPath(); ctx.ellipse(mx, my, u * (0.2 + 0.05 * P.smile), u * 0.28 * P.mouthOpen, 0, 0, Math.PI * 2); ctx.fill();
    } else {
      ctx.strokeStyle = "#b8405a"; ctx.lineWidth = u * 0.09; ctx.lineCap = "round";
      ctx.beginPath(); ctx.moveTo(mx - u * 0.28, my - P.smile * u * 0.08); ctx.quadraticCurveTo(mx, my + P.smile * u * 0.22, mx + u * 0.28, my - P.smile * u * 0.08); ctx.stroke();
    }
    // 前髪（一番手前なので大きくずらす）
    [dx, dy] = off(0.35);
    ctx.fillStyle = hairCol; ctx.beginPath();
    ctx.moveTo(dx - u * 1.8, hy + dy - u * 0.1);
    ctx.quadraticCurveTo(dx - u * 1.6, hy + dy - u * 2.1, dx, hy + dy - u * 2.05);
    ctx.quadraticCurveTo(dx + u * 1.6, hy + dy - u * 2.1, dx + u * 1.8, hy + dy - u * 0.1);
    for (let i = 4; i >= 0; i--) { // 毛先のぎざぎざ（右の端から左へ）
      const x0 = dx - u * 1.8 + i * u * 0.72;
      ctx.lineTo(x0 + u * 0.36, hy + dy - u * (0.35 + (i % 2) * 0.25)); ctx.lineTo(x0, hy + dy - u * 0.95);
    }
    ctx.closePath(); ctx.fill();
  }

  function arm(ctx, u, sh, a1, a2) {
    const L1 = u * 1.5, L2 = u * 1.35;
    const el = [sh[0] + Math.cos(a1) * L1, sh[1] + Math.sin(a1) * L1], hd = [el[0] + Math.cos(a2) * L2, el[1] + Math.sin(a2) * L2];
    ctx.lineCap = "round";
    ctx.strokeStyle = "#ffe3d3"; ctx.lineWidth = u * 0.42; ctx.beginPath(); ctx.moveTo(...el); ctx.lineTo(...hd); ctx.stroke();
    ctx.strokeStyle = "#5b8def"; ctx.lineWidth = u * 0.62; ctx.beginPath(); ctx.moveTo(...sh); ctx.lineTo(...el); ctx.stroke();
    ctx.fillStyle = "#ffe3d3"; ctx.beginPath(); ctx.arc(...hd, u * 0.3, 0, Math.PI * 2); ctx.fill();
  }

  raf = requestAnimationFrame(step);
  onFrame({ items: [], w: 1, h: 1 });
  return { onFrame, destroy() { cancelAnimationFrame(raf); el.replaceChildren(); } };
}

function rrect(ctx, x, y, w, h, r) {
  ctx.beginPath(); ctx.roundRect ? ctx.roundRect(x, y, w, h, r) : ctx.rect(x, y, w, h);
}

export const INTERACT = {
  puppet: { create: (el) => puppet(el) },
};
