// AR の地図の重い照合（姿勢グラフのループ閉じ・前回の地図への再位置合わせ）を、XR のフレームを止めないよう別のスレッドで回す。
// 中身は再生と同じ関数（xr_graph.js の matchJob、xr_reloc.js の globalMatch）。前回の地図の点は一度だけ受け取って持っておく
//   → { id, type: "match", args: job } / { id, type: "setRef", args: ref } / { id, type: "reloc", args: cur }   ← { id, r, ms, err }
import { matchJob } from "./xr_graph.js";
import { globalMatch } from "./xr_reloc.js";

let ref = null;
self.onmessage = (e) => {
  const { id, type, args } = e.data, t0 = performance.now();
  let r = null, err = null;
  try {
    if (type === "match") r = matchJob(args);
    else if (type === "setRef") ref = args;
    else if (type === "reloc") r = ref ? globalMatch(args, ref) : null;
  } catch (x) { err = String(x?.message || x); }
  self.postMessage({ id, r, ms: performance.now() - t0, err });
};
