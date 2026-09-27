// タブ（tasks）とモデル（models）の定義は models.json（サーバーの adapters.py と共通）。1ファイル版では build.py が中身を埋め込む
export const CATALOG = await (await fetch(new URL("./models.json", import.meta.url))).json();
export const TASKS = CATALOG.tasks;
export const MODELS = CATALOG.models;
export { COCO } from "./coco.js";

// COCO 17 関節の骨格（YOLO の順番）。左半身・右半身・体幹で色を分ける
export const SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];
