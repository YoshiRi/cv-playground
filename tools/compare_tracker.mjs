import { readFileSync } from "fs";
// tools/tracker_reference.py が作った ref.json（Ultralytics の結果）と web/tracker.js を、同じ検出列で比べる
const { Tracker } = await import(new URL("../web/tracker.js", import.meta.url));
const ref = JSON.parse(readFileSync("ref.json"));
for (const type of ["bytetrack", "botsort"]) {
  const t = new Tracker(type);
  let frames = 0, same = 0, maxErr = 0, firstDiff = null;
  ref.dets.forEach((d, f) => {
    const mine = t.update(d).map((x) => [...x.box, x.id]).sort((a, b) => a[4] - b[4]);
    const theirs = ref[type][f].slice().sort((a, b) => a[4] - b[4]);
    frames++;
    const idsOk = mine.length === theirs.length && mine.every((m, i) => m[4] === theirs[i][4]);
    if (idsOk) { same++; mine.forEach((m, i) => { for (let k = 0; k < 4; k++) maxErr = Math.max(maxErr, Math.abs(m[k] - theirs[i][k])); }); }
    else if (!firstDiff) firstDiff = { f, mine: mine.map((m) => m[4]), theirs: theirs.map((m) => m[4]) };
  });
  console.log(type, `IDが一致したフレーム ${same}/${frames}`, `枠の最大差 ${maxErr.toFixed(3)}px`, firstDiff ? JSON.stringify(firstDiff) : "");
}
