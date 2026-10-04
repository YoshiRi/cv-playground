// 画面の言語（日本語・英語）。日本語が元の文で、英語は i18n_en.js の辞書（日本語の文 → 英語）と、数字などが入る文の形（RULES）で出す。
//
//   言語: URL の ?lang=ja|en ＞ 前に選んだ言語（localStorage）＞ ブラウザの言語（ja… なら日本語、ほかは英語）
//   ページの文字: 英語の時は、文字のかたまり（テキストノード）と title・placeholder・aria-label を、変わるたびに辞書で置き換える
//                （MutationObserver。HTML も JS が後から作った物もコードを変えずに訳せる。辞書に無い文は日本語のまま）
//   それ以外（canvas に描く文字、確認のダイアログなど）: コードで tx("…{n}…", {n}) を使う
//   models.json のタスク・モデルなどの名前と説明は、*_en の欄（name_en・hint_en・note_en）を起動時に辞書に足す
import { EN, PHRASES, RULES } from "./i18n_en.js";

const pick = () => {
  const q = new URLSearchParams(location.search).get("lang");
  if (q === "ja" || q === "en") return q;
  try { const s = localStorage.getItem("cvpg-lang"); if (s === "ja" || s === "en") return s; } catch { /* 保存できない環境 */ }
  return (navigator.language || "").toLowerCase().startsWith("ja") ? "ja" : "en";
};
export const LANG = pick();

// 言語を切り替える（選んだ言語を覚えて、読み込み直す）
export function setLang(l) {
  try { localStorage.setItem("cvpg-lang", l); } catch { /* 保存できない環境 */ }
  const u = new URL(location.href);
  u.searchParams.delete("lang");
  location.href = u.href;
}

const fill = (s, vars) => (vars ? s.replace(/\{(\w+)\}/g, (m, k) => (vars[k] ?? m)) : s);
// 文を今の言語で。英語の辞書に無い時は、決まった形（RULES）に当てはめ、それも無ければ元の文
export function tx(s, vars) {
  if (LANG !== "en" || s == null) return fill(s, vars);
  return fill(EN[s] ?? rule(s) ?? s, vars);
}
// 決まった形（RULES: [正規表現, 置き換え]）に当てはめ、それも無ければ文の中の決まった言い回し（PHRASES）を置き換える
const tr = (x) => EN[x] ?? rule(x) ?? x;
function rule(s) {
  for (const [re, f] of RULES) { const m = s.match(re); if (m) return typeof f === "function" ? f(m, tr) : s.replace(re, f); }
  let out = s, hit = false;
  for (const [ja, en] of PHRASES) if (out.includes(ja)) { out = out.split(ja).join(en); hit = true; }
  return hit ? out : null;
}

// models.json の名前・説明の英語（*_en）を辞書に足す
export function addCatalog(cat) {
  const add = (o) => { for (const f of ["name", "hint", "note"]) if (o?.[f] && o[`${f}_en`]) EN[o[f]] = o[`${f}_en`]; };
  for (const k of ["categories", "tasks", "models", "apps", "interact"]) for (const o of cat[k] || []) {
    add(o);
    for (const c of o.cascade || []) add(c);
    for (const w of o.combo?.with || []) add(w);
  }
}

// ページの文字を英語に置き換える（英語の時だけ）。前後の空白は残す
const JP = /[぀-ヿ一-鿿]/;
function trText(node) {
  const v = node.nodeValue;
  if (!v || !JP.test(v)) return;
  const k = v.trim(), e = EN[k] ?? rule(k);
  if (e != null && e !== k) node.nodeValue = v.replace(k, e);
}
function trAttrs(el) {
  for (const a of ["title", "placeholder", "aria-label"]) {
    const v = el.getAttribute?.(a);
    if (v && JP.test(v)) { const e = EN[v.trim()] ?? rule(v.trim()); if (e != null) el.setAttribute(a, e); }
  }
}
export function translateTree(root) {
  if (LANG !== "en" || !root) return;
  if (root.nodeType === Node.TEXT_NODE) { trText(root); return; }
  if (root.nodeType !== Node.ELEMENT_NODE) return;
  if (root.closest?.("script,style,textarea")) return;
  trAttrs(root);
  const w = document.createTreeWalker(root, NodeFilter.SHOW_TEXT | NodeFilter.SHOW_ELEMENT);
  for (let n; (n = w.nextNode());) {
    if (n.nodeType === Node.TEXT_NODE) { if (!n.parentElement?.closest("script,style,textarea")) trText(n); }
    else trAttrs(n);
  }
}
// 英語の時は、ページの文字が変わるたびに置き換え続ける
export function startTranslate(root = document.body) {
  if (LANG !== "en") return;
  document.documentElement.lang = "en";
  translateTree(root);
  new MutationObserver((ms) => {
    for (const m of ms) {
      if (m.type === "characterData") trText(m.target);
      else if (m.type === "attributes") trAttrs(m.target);
      else for (const n of m.addedNodes) translateTree(n);
    }
  }).observe(root, { subtree: true, childList: true, characterData: true, attributes: true, attributeFilter: ["title", "placeholder", "aria-label"] });
}
