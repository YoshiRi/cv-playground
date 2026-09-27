// タスクの一覧と既定値。モデルの一覧は models.json。

export const TASKS = [
  { id: "detect", name: "物体検出", hint: "COCO 80クラス。閾値で絞る", params: ["threshold"] },
  { id: "pose", name: "人物の姿勢", hint: "17関節（COCO）", params: ["threshold"] },
  { id: "zsdetect", name: "テキスト指定検出", hint: "英語の名詞をカンマ区切り（例: cat, remote control）", params: ["labels", "threshold"] },
  { id: "segment", name: "クリックで切り出し", hint: "画像をクリックすると、その場所の物体を切り出す。除く点は Shift＋クリックかチェックで。「全体を自動分割」は格子状の点から画面全体を物体ごとに色分けする（ブラウザのみ）", params: ["points"] },
  { id: "depth", name: "深度推定", hint: "赤いほど近い（相対深度）", params: [] },
  { id: "classify", name: "ゼロショット分類", hint: "候補をカンマ区切り（英語）。棒は候補間の相対値、括弧内はモデルの絶対スコア", params: ["labels"] },
  { id: "matting", name: "背景除去", hint: "前景だけを残す", params: [] },
  { id: "vlm", name: "画像の説明・質問", hint: "SmolVLM は英語のみ。日本語は Qwen", params: ["prompt"] },
];

// モデルの一覧は models.json（サーバーの adapters.py と共通）。1ファイル版では build.py が中身を埋め込む
export const MODELS = (await (await fetch(new URL("./models.json", import.meta.url))).json()).models;

export const DEFAULTS = {
  threshold: { detect: 0.4, pose: 0.4, zsdetect: 0.3 },
  labels: { zsdetect: "person, bus, bag", classify: "bus, truck, tram, street, office, cat" },
  prompt: "この画像を日本語で詳しく説明してください。",
};

// COCO 17 関節の骨格（YOLO の順番）
export const SKELETON = [[15, 13], [13, 11], [16, 14], [14, 12], [11, 12], [5, 11], [6, 12], [5, 6], [5, 7], [6, 8],
  [7, 9], [8, 10], [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]];

export { COCO } from "./coco.js";
