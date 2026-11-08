/* StockAI PWA – mobile client for the StockAI engine */
"use strict";

const $ = s => document.querySelector(s);
const esc = s => String(s ?? "").replace(/[&<>"']/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;"}[c]));
const fmt = (v, d = 2) => v == null || isNaN(v) ? "—" : Number(v).toLocaleString("en-US", {minimumFractionDigits: d, maximumFractionDigits: d});
const pct = (v, d = 2) => v == null || isNaN(v) ? "—" : (v > 0 ? "+" : "") + Number(v).toFixed(d) + "%";
const cls = v => v > 0 ? "pos" : v < 0 ? "neg" : "";
const COLORS = {strong_buy: "#0f9d58", buy: "#22c57a", hold: "#6b7590", sell: "#ef5350", strong_sell: "#b3161b"};
const INTERVALS = [["1m", "דקה", 1], ["5m", "5 ד'", 5], ["15m", "15 ד'", 15], ["30m", "30 ד'", 30], ["60m", "שעה", 60], ["1d", "יומי", 390]];

// ---------------- settings ----------------
const DEF = {interval: "5m", horizon: 12, buy: 0.6, sell: 0.4, refresh: 5, pm: true, news: true, market: true,
  mtf: true, notify: false, key: "", ticker: "NVDA", scanList: ""};
let S = {...DEF};
try { S = {...DEF, ...JSON.parse(localStorage.getItem("stockai") || "{}")}; } catch (e) {}
const save = () => { try { localStorage.setItem("stockai", JSON.stringify(S)); } catch (e) {} };

let META = {groups: {}};
let current = null, refreshTimer = null, lastLevel = {};

// ---------------- api ----------------
async function api(path, params = {}) {
  const qs = new URLSearchParams(params).toString();
  const r = await fetch(`/api/${path}${qs ? "?" + qs : ""}`, {headers: {"x-app-key": S.key || ""}, cache: "no-store"});
  const j = await r.json().catch(() => ({error: `שגיאת שרת (${r.status})`}));
  if (!r.ok || j.error) throw new Error(j.error || `שגיאה ${r.status}`);
  return j;
}
const baseParams = () => ({interval: S.interval, horizon: S.horizon, buy: S.buy, sell: S.sell,
  pm: +S.pm, news: +S.news, market: +S.market, mtf: +S.mtf});

function toast(msg, ms = 3500) {
  const t = $("#toast"); t.textContent = msg; t.classList.remove("hidden");
  clearTimeout(toast._t); toast._t = setTimeout(() => t.classList.add("hidden"), ms);
}

// ---------------- routing ----------------
const VIEWS = ["analyze", "scan", "market", "news", "settings"];
function route() {
  const v = VIEWS.includes(location.hash.slice(1)) ? location.hash.slice(1) : "analyze";
  VIEWS.forEach(n => $(`#v-${n}`).classList.toggle("hidden", n !== v));
  document.querySelectorAll(".tabs a").forEach(a => a.classList.toggle("on", a.dataset.v === v));
  if (v === "market") loadMarket();
  if (v === "news") loadNews();
  if (v === "settings") fillSettings();
  window.scrollTo(0, 0);
}
window.addEventListener("hashchange", route);

// ---------------- analyze ----------------
function horizonText() {
  const iv = INTERVALS.find(i => i[0] === S.interval) || INTERVALS[1];
  if (S.interval === "1d") return `${S.horizon} ימי מסחר קדימה`;
  const m = iv[2] * S.horizon;
  return m < 60 ? `${m} דקות קדימה` : `${(m / 60).toFixed(1)} שעות קדימה`;
}
function renderControls() {
  $("#intervalChips").innerHTML = INTERVALS.map(([k, he]) =>
    `<button class="chip ${k === S.interval ? "on" : ""}" data-iv="${k}">${he}</button>`).join("");
  $("#horizonRange").value = S.horizon;
  $("#horizonText").textContent = horizonText();
}
$("#intervalChips").addEventListener("click", e => {
  const b = e.target.closest("[data-iv]"); if (!b) return;
  S.interval = b.dataset.iv; if (S.interval === "1d" && S.horizon > 20) S.horizon = 5;
  save(); renderControls(); if (current) runAnalyze(current.ticker);
});
$("#horizonRange").addEventListener("input", e => { S.horizon = +e.target.value; $("#horizonText").textContent = horizonText(); });
$("#horizonRange").addEventListener("change", () => { save(); if (current) runAnalyze(current.ticker); });
$("#backBtn").addEventListener("click", () => { clearTimeout(refreshTimer); current = null; renderPicker(); });
$("#searchForm").addEventListener("submit", e => {
  e.preventDefault();
  const t = $("#tickerInput").value.trim().toUpperCase();
  if (t) { $("#tickerInput").blur(); runAnalyze(t); }
});

async function runAnalyze(ticker, silent = false) {
  S.ticker = ticker; save();
  $("#tickerInput").value = ticker;
  if (location.hash !== "#analyze" && location.hash !== "") location.hash = "#analyze";
  if (!silent) $("#analysis").innerHTML = `<div class="spinner"></div><p class="muted small" style="text-align:center">מנתח את ${esc(ticker)} – אימון מודל ובדיקה היסטורית…</p>`;
  try {
    const r = await api("analyze", {ticker, ...baseParams()});
    current = r;
    renderAnalysis(r);
    maybeNotify(r);
  } catch (e) {
    if (!silent) $("#analysis").innerHTML = `<div class="err">⚠️ ${esc(e.message)}</div>`;
  }
  scheduleRefresh();
}

function gaugeSVG(p) {
  const v = Math.max(0, Math.min(1, p)), R = 80, cx = 100, cy = 95;
  const pt = a => [cx + R * Math.cos(Math.PI * (1 - a)), cy - R * Math.sin(Math.PI * (1 - a))];
  const arc = (a0, a1, col) => {
    const [x0, y0] = pt(a0), [x1, y1] = pt(a1);
    return `<path d="M${x0},${y0} A${R},${R} 0 0 1 ${x1},${y1}" stroke="${col}" stroke-width="16" fill="none"/>`;
  };
  const [nx, ny] = [cx + (R - 22) * Math.cos(Math.PI * (1 - v)), cy - (R - 22) * Math.sin(Math.PI * (1 - v))];
  return `<svg class="gauge" viewBox="0 0 200 120" width="100%">
    ${arc(0, .3, "#b3161b")}${arc(.3, .4, "#ef5350")}${arc(.4, .6, "#3a4560")}${arc(.6, .7, "#22c57a")}${arc(.7, 1, "#0f9d58")}
    <line x1="${cx}" y1="${cy}" x2="${nx}" y2="${ny}" stroke="#fff" stroke-width="4" stroke-linecap="round"/>
    <circle cx="${cx}" cy="${cy}" r="6" fill="#fff"/>
    <text x="${cx}" y="${cy + 22}" text-anchor="middle" font-size="22" font-weight="800">${(v * 100).toFixed(1)}%</text>
  </svg><div class="small muted" style="text-align:center;margin-top:-4px">הסתברות לעלייה</div>`;
}

function renderAnalysis(r) {
  const L = r.levels, m = r.model, pm = r.premarket, ext = pm && ["premarket", "afterhours"].includes(r.session);
  const aucQ = m.auc >= .58 ? "גבוהה" : m.auc >= .53 ? "בינונית" : "נמוכה";
  const priceChange = ext ? pm["ext_change_%"] : r.day_change;
  let h = `
  <div class="card">
    <div class="row" style="justify-content:space-between">
      <div><h3 style="margin:0">${esc(r.name)} <span class="muted ltr">${esc(r.ticker)}</span></h3>
        <div class="small muted">${esc(horizonText())} · עודכן ${new Date(r.updated * 1000).toLocaleTimeString("he-IL")}</div></div>
      <div style="text-align:left"><div class="price">${fmt(r.price)}</div>
        <div class="${cls(priceChange)} small ltr">${pct(priceChange)} ${ext ? (r.session === "premarket" ? "מוקדם" : "מאוחר") : "היום"}</div></div>
    </div>
    <div class="hero" style="margin-top:8px">
      <div>${gaugeSVG(r.p_up)}</div>
      <div>
        <div class="signal" style="background:${COLORS[r.action_level]}">${esc(r.action)}</div>
        <div class="small muted" style="margin-top:8px">אמינות המודל: <b style="color:var(--text)">${aucQ}</b><br>
          AUC ${fmt(m.auc, 3)} · דיוק ${fmt(m.accuracy * 100, 1)}%<br>על ${m.n_oos?.toLocaleString() ?? "—"} תחזיות מחוץ למדגם</div>
      </div>
    </div>
  </div>`;

  if (pm) {
    const st = pm.stats || {};
    h += `<div class="card"><h3>🌅 ${esc(pm.session_he)} ${pm.active ? "" : '<span class="small muted">(משפיע רק במוקדם ובשעה הראשונה)</span>'}</h3>
      <div class="grid3">
        <div class="kv"><div class="k">${esc(pm.ref_label)}</div><div class="v">${fmt(pm.prev_close)}</div></div>
        <div class="kv"><div class="k">פער</div><div class="v ${cls(pm["gap_%"])}">${pct(pm["gap_%"])}</div></div>
        <div class="kv"><div class="k">נפח מוקדם יחסי</div><div class="v">${pm.pm_rel_volume ? "x" + fmt(pm.pm_rel_volume, 1) : "—"}</div></div>
        ${pm.pm_high ? `<div class="kv"><div class="k">שיא מוקדם</div><div class="v">${fmt(pm.pm_high)}</div></div>
        <div class="kv"><div class="k">שפל מוקדם</div><div class="v">${fmt(pm.pm_low)}</div></div>
        <div class="kv"><div class="k">VWAP מוקדם</div><div class="v">${fmt(pm.pm_vwap)}</div></div>` : ""}
      </div>
      ${st.n ? `<p class="small" style="margin:10px 0 4px">ב-${st.n} פערים דומים (${st["range_%"][0]}%–${st["range_%"][1]}%) בשנתיים האחרונות:
        המשיכו בכיוון ב-<b>${(st.continuation * 100).toFixed(0)}%</b>, נסגרו ב-<b>${(st.fill_rate * 100).toFixed(0)}%</b>,
        ממוצע פתיחה→סגירה <b class="${cls(st["avg_open_to_close_%"])}">${pct(st["avg_open_to_close_%"])}</b></p>` : ""}
      ${(pm.notes || []).map(n => `<div class="reason">• ${esc(n)}</div>`).join("")}
    </div>`;
  }

  const dir = r.p_up >= .5 ? "לונג" : "שורט";
  h += `<div class="card"><h3>🎯 רמות מסחר (${dir})</h3><div class="grid2">
      <div class="kv"><div class="k">סטופ לוס</div><div class="v neg">${fmt(L.stop)} <span class="small">${pct((L.stop / r.price - 1) * 100)}</span></div></div>
      <div class="kv"><div class="k">יעד רווח</div><div class="v pos">${fmt(L.target)} <span class="small">${pct((L.target / r.price - 1) * 100)}</span></div></div>
      <div class="kv"><div class="k">סיכון:סיכוי</div><div class="v">1:${fmt(L.rr, 1)}</div></div>
      <div class="kv"><div class="k">ATR</div><div class="v">${fmt(L.atr)}</div></div></div>
      <div class="small muted" style="margin-top:8px">תמיכות: <span class="ltr">${(L.supports || []).slice(0, 3).map(v => fmt(v)).join(", ") || "—"}</span><br>
      התנגדויות: <span class="ltr">${(L.resistances || []).slice(0, 3).map(v => fmt(v)).join(", ") || "—"}</span></div></div>
  <div class="card"><div id="chart" class="chartbox"></div><div id="rsi" class="rsibox"></div>
    <div class="legend"><span><i style="background:#f5b13d"></i>EMA9</span><span><i style="background:#4a90e2"></i>EMA21</span>
    <span><i style="background:#a66cff"></i>EMA50</span><span><i style="background:#ff4fa3"></i>VWAP</span>
    <span><i style="background:#7fb3d5"></i>מסחר מוקדם/מאוחר</span></div></div>
  <div class="card"><h3>🧠 למה?</h3>${r.reasons.map(t => `<div class="reason">${esc(t)}</div>`).join("")}
    ${(r.patterns || []).map(([n, d]) => `<div class="reason">${d > 0 ? "🟢" : d < 0 ? "🔴" : "⚪"} תבנית: ${esc(n)}</div>`).join("")}</div>`;

  const bd = {...r.breakdown}; const base = bd["מודל ML (מכויל)"]; delete bd["מודל ML (מכויל)"];
  h += `<div class="card"><h3>⚖️ מרכיבי ההסתברות</h3>
    <div class="bd"><span>בסיס – מודל ML מכויל</span><b>${fmt(base * 100, 1)}%</b></div>
    ${Object.entries(bd).map(([k, v]) => `<div class="bd"><span>${v > 0 ? "⬆️" : "⬇️"} ${esc(k)}</span><span class="${cls(v)} ltr">${v > 0 ? "+" : ""}${fmt(v, 2)}</span></div>`).join("")}
    <div class="bd" style="border-top:1px solid var(--line);margin-top:4px;padding-top:8px"><b>סופי</b><b>${fmt(r.p_up * 100, 1)}%</b></div>
    <div class="small muted">ההתאמות ביחידות לוגיט (+0.2 ≈ +5% סביב 50%)</div></div>`;

  const ctx = [];
  if (r.market?.notes?.length) ctx.push(`<b>שוק:</b> ${r.market.notes.map(esc).join(" · ")}`);
  if (r.mtf?.notes?.length) ctx.push(`<b>טווח גבוה:</b> ${r.mtf.notes.map(esc).join(" · ")}`);
  if (r.news) ctx.push(`<b>חדשות:</b> סנטימנט ${fmt(r.news.score, 2)} · 🟢${r.news.positive} 🔴${r.news.negative} · ${r.news.buzz_24h} כתבות ב-24 ש'`);
  const i = r.info || {};
  const fund = [["סקטור", i.sector], ["שווי שוק", bigNum(i.marketCap)], ["מכפיל", i.trailingPE && fmt(i.trailingPE, 1)],
    ["מכפיל עתידי", i.forwardPE && fmt(i.forwardPE, 1)], ["המלצת אנליסטים", i.recommendationKey],
    ["יעד אנליסטים", i.targetMeanPrice && fmt(i.targetMeanPrice)], ["דוחות הבאים", i.earningsDate && i.earningsDate.slice(0, 10)]]
    .filter(([, v]) => v);
  h += `<div class="card"><h3>🌐 הקשר</h3>${ctx.map(c => `<div class="reason small">${c}</div>`).join("")}
    ${fund.length ? `<table style="margin-top:8px">${fund.map(([k, v]) => `<tr><td class="muted">${k}</td><td><b>${esc(v)}</b></td></tr>`).join("")}</table>` : ""}</div>`;

  const bt = r.backtest || {};
  if (bt.trades != null) {
    h += `<div class="card"><details><summary>🧪 בדיקה היסטורית (Walk-forward)</summary>
      <div class="grid3" style="margin-top:10px">
        <div class="kv"><div class="k">עסקאות</div><div class="v">${bt.trades}</div></div>
        <div class="kv"><div class="k">אחוז הצלחה</div><div class="v">${bt.win_rate == null ? "—" : fmt(bt.win_rate * 100, 1) + "%"}</div></div>
        <div class="kv"><div class="k">ירידה מקס'</div><div class="v neg">${fmt(bt["max_dd_%"], 1)}%</div></div>
        <div class="kv"><div class="k">אסטרטגיה</div><div class="v ${cls(bt["total_%"])}">${pct(bt["total_%"], 1)}</div></div>
        <div class="kv"><div class="k">קנה והחזק</div><div class="v ${cls(bt["buyhold_%"])}">${pct(bt["buyhold_%"], 1)}</div></div>
        <div class="kv"><div class="k">שיעור בסיס</div><div class="v">${fmt(m.base_rate * 100, 1)}%</div></div>
      </div>
      ${r.reliability ? `<p class="small muted" style="margin:10px 0 4px">כיול – כשהמודל אמר X%, כמה באמת עלה:</p>
        <table>${r.reliability.map(b => `<tr><td class="ltr">${esc(b.bucket)}</td><td><b>${fmt(b.actual * 100, 1)}%</b></td><td class="muted">${b.n} מקרים</td></tr>`).join("")}</table>` : ""}
      </details></div>`;
  }
  if (r.news?.items?.length) h += `<div class="card news"><details><summary>📰 כתבות על ${esc(r.ticker)} (${r.news.items.length})</summary>${newsList(r.news.items)}</details></div>`;
  h += `<p class="small muted" style="text-align:center">⚠️ כלי עזר בלבד – לא ייעוץ השקעות</p>`;
  $("#analysis").innerHTML = h;
  drawChart(r);
}

function bigNum(n) {
  if (!n) return null;
  for (const [u, d] of [["T", 1e12], ["B", 1e9], ["M", 1e6]]) if (Math.abs(n) >= d) return (n / d).toFixed(2) + u;
  return String(n);
}

function newsList(items) {
  return items.map(n => {
    const c = n.sentiment > .15 ? "var(--green)" : n.sentiment < -.15 ? "var(--red)" : "var(--gray)";
    const age = n.age_h < 48 ? `לפני ${Math.round(n.age_h)} ש'` : `לפני ${Math.round(n.age_h / 24)} ימים`;
    return `<a href="${esc(n.link)}" target="_blank" rel="noopener"><span class="dot" style="background:${c}"></span>${esc(n.title)}
      <div class="meta">${esc(n.source)} · ${age}</div></a>`;
  }).join("");
}

let charts = [];
function drawChart(r) {
  charts.forEach(c => c.remove()); charts = [];
  if (!window.LightweightCharts) { $("#chart").innerHTML = '<p class="muted small">הגרף לא נטען (אין חיבור ל-CDN)</p>'; return; }
  const LC = window.LightweightCharts, el = $("#chart"), intraday = S.interval !== "1d";
  const base = {layout: {background: {color: "transparent"}, textColor: "#8b98b4", fontSize: 11},
    grid: {vertLines: {color: "#1b2742"}, horzLines: {color: "#1b2742"}},
    rightPriceScale: {borderColor: "#24314f"}, timeScale: {borderColor: "#24314f", timeVisible: intraday, secondsVisible: false},
    crosshair: {mode: 0}, handleScroll: true, handleScale: true};
  const c = LC.createChart(el, {...base, width: el.clientWidth, height: el.clientHeight});
  const ch = r.chart;
  const cs = c.addCandlestickSeries({upColor: "#26a69a", downColor: "#ef5350", borderVisible: false,
    wickUpColor: "#26a69a", wickDownColor: "#ef5350"});
  const all = [...ch.candles.map(b => ({...b})), ...(ch.ext || []).map(b => ({...b, color: "#7fb3d5", wickColor: "#7fb3d5", borderColor: "#7fb3d5"}))];
  const seen = new Set(); const merged = all.sort((a, b) => a.time - b.time).filter(b => !seen.has(b.time) && seen.add(b.time));
  cs.setData(merged);
  const vol = c.addHistogramSeries({priceFormat: {type: "volume"}, priceScaleId: "vol"});
  c.priceScale("vol").applyOptions({scaleMargins: {top: .82, bottom: 0}});
  vol.setData(ch.volume);
  [["ema9", "#f5b13d"], ["ema21", "#4a90e2"], ["ema50", "#a66cff"], ["vwap", "#ff4fa3"]].forEach(([k, col]) => {
    if (ch[k]?.length) c.addLineSeries({color: col, lineWidth: k === "vwap" ? 1 : 1.5, lineStyle: k === "vwap" ? 2 : 0,
      priceLineVisible: false, lastValueVisible: false, crosshairMarkerVisible: false}).setData(ch[k]);
  });
  const pl = (price, color, title, style = 0) => price && cs.createPriceLine({price, color, lineWidth: 1, lineStyle: style, axisLabelVisible: true, title});
  pl(r.levels.stop, "#ef5350", "סטופ"); pl(r.levels.target, "#22c57a", "יעד");
  (r.levels.supports || []).slice(0, 2).forEach(v => pl(v, "rgba(38,166,154,.6)", "", 2));
  (r.levels.resistances || []).slice(0, 2).forEach(v => pl(v, "rgba(239,83,80,.6)", "", 2));
  if (r.premarket?.pm_high) { pl(r.premarket.pm_high, "#f5b13d", "שיא מוקדם", 1); pl(r.premarket.pm_low, "#f5b13d", "שפל מוקדם", 1); }
  c.timeScale().fitContent();
  charts.push(c);

  const re = $("#rsi");
  const rc = LC.createChart(re, {...base, width: re.clientWidth, height: re.clientHeight, timeScale: {...base.timeScale, visible: false}});
  const rs = rc.addLineSeries({color: "#a66cff", lineWidth: 1.5, priceLineVisible: false});
  rs.setData(ch.rsi);
  rs.createPriceLine({price: 70, color: "#ef5350", lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: ""});
  rs.createPriceLine({price: 30, color: "#22c57a", lineWidth: 1, lineStyle: 2, axisLabelVisible: false, title: "RSI"});
  rc.timeScale().fitContent();
  c.timeScale().subscribeVisibleLogicalRangeChange(rg => rg && rc.timeScale().setVisibleLogicalRange(rg));
  charts.push(rc);
}
window.addEventListener("resize", () => charts.forEach(c => {
  const el = c.chartElement().parentElement; c.applyOptions({width: el.clientWidth});
}));

function scheduleRefresh() {
  clearTimeout(refreshTimer);
  if (!S.refresh || !current) return;
  refreshTimer = setTimeout(() => {
    if (document.visibilityState === "visible" && !$("#v-analyze").classList.contains("hidden")) runAnalyze(current.ticker, true);
    else scheduleRefresh();
  }, S.refresh * 60000);
}
document.addEventListener("visibilitychange", () => {
  if (document.visibilityState === "visible" && current && S.refresh &&
      Date.now() / 1000 - current.updated > S.refresh * 60) runAnalyze(current.ticker, true);
});

async function maybeNotify(r) {
  const prev = lastLevel[r.ticker]; lastLevel[r.ticker] = r.action_level;
  if (!prev || prev === r.action_level || r.action_level === "hold") return;
  const msg = `${r.ticker}: ${r.action} (${(r.p_up * 100).toFixed(0)}%) · ${fmt(r.price)}`;
  toast("🔔 " + msg);
  if (S.notify && "Notification" in window && Notification.permission === "granted") {
    try { (await navigator.serviceWorker.ready).showNotification("StockAI – איתות חדש", {body: msg, icon: "/static/icons/icon-192.png", tag: r.ticker}); }
    catch (e) {}
  }
}

// ---------------- scanner ----------------
let scanRows = [], scanFilter = "all";
function renderGroups() {
  $("#groupChips").innerHTML = Object.keys(META.groups).map(g => `<button class="chip" data-g="${esc(g)}">${esc(g)}</button>`).join("");
  $("#scanList").value = S.scanList || "";
  const opts = new Set(Object.values(META.groups).flat());
  $("#tickerList").innerHTML = [...opts].sort().map(t => `<option value="${t}">`).join("");
}
$("#groupChips").addEventListener("click", e => {
  const b = e.target.closest("[data-g]"); if (!b) return;
  document.querySelectorAll("#groupChips .chip").forEach(c => c.classList.toggle("on", c === b));
  $("#scanList").value = META.groups[b.dataset.g].join(", ");
});
$("#scanBtn").addEventListener("click", async () => {
  const list = $("#scanList").value; S.scanList = list; save();
  if (!list.trim()) return toast("בחר קבוצה או הזן מניות");
  $("#scanBtn").disabled = true; $("#scanProgress").classList.remove("hidden"); $("#scanResults").innerHTML = "";
  try {
    const {job} = await api("scan", {tickers: list, ...baseParams()});
    for (;;) {
      await new Promise(res => setTimeout(res, 1500));
      const st = await api("scan/status", {job});
      $("#scanBar").style.width = (st.progress * 100).toFixed(0) + "%";
      $("#scanText").textContent = st.status === "running" ? `מנתח ${st.current}… (${Math.round(st.progress * st.n)}/${st.n})` : "";
      if (st.status === "done") { scanRows = st.rows; renderScan(); break; }
      if (st.status === "error") throw new Error(st.error);
    }
  } catch (e) { $("#scanResults").innerHTML = `<div class="err">⚠️ ${esc(e.message)}</div>`; }
  $("#scanBtn").disabled = false; $("#scanProgress").classList.add("hidden");
});
$("#scanFilter").addEventListener("click", e => {
  const b = e.target.closest("[data-f]"); if (!b) return; scanFilter = b.dataset.f; renderScan();
});
function renderScan() {
  const F = [["all", "הכל"], ["buy", "🟢 קנייה"], ["sell", "🔴 מכירה"], ["ext", "🌅 זזות במוקדם"]];
  $("#scanFilter").innerHTML = F.map(([k, l]) => `<button class="chip ${k === scanFilter ? "on" : ""}" data-f="${k}">${l}</button>`).join("");
  let rows = scanRows.filter(r => r["סיכוי עלייה %"] != null);
  if (scanFilter === "buy") rows = rows.filter(r => r["סיכוי עלייה %"] >= S.buy * 100);
  if (scanFilter === "sell") rows = rows.filter(r => r["סיכוי עלייה %"] <= S.sell * 100);
  if (scanFilter === "ext") rows = rows.filter(r => r["מוקדם/מאוחר %"] != null).sort((a, b) => Math.abs(b["מוקדם/מאוחר %"]) - Math.abs(a["מוקדם/מאוחר %"]));
  const errs = scanRows.filter(r => r["סיכוי עלייה %"] == null);
  $("#scanResults").innerHTML = rows.map(r => {
    const p = r["סיכוי עלייה %"], lvl = p >= Math.max(S.buy * 100 + 10, 70) ? "strong_buy" : p >= S.buy * 100 ? "buy" :
      p <= Math.min(S.sell * 100 - 10, 30) ? "strong_sell" : p <= S.sell * 100 ? "sell" : "hold";
    const ext = r["מוקדם/מאוחר %"];
    return `<div class="scanrow" data-t="${esc(r["מניה"])}">
      <div><div class="t">${esc(r["מניה"])}</div><div class="small muted ltr">${fmt(r["מחיר"])}</div></div>
      <div><div class="small"><span class="${cls(r["שינוי יומי %"])}">${pct(r["שינוי יומי %"])}</span>
        ${ext != null ? ` · מוקדם <span class="${cls(ext)}">${pct(ext)}</span>` : ""} · AUC ${fmt(r["AUC מודל"], 2)}</div>
        <div class="pbar"><i style="width:${p}%;background:${COLORS[lvl]}"></i></div></div>
      <div style="text-align:center"><div style="font-weight:800">${fmt(p, 0)}%</div><span class="pill" style="background:${COLORS[lvl]}">${esc(r["המלצה"])}</span></div>
    </div>`;
  }).join("") + (errs.length ? `<p class="small muted">לא נותחו: ${errs.map(r => esc(r["מניה"])).join(", ")}</p>` : "")
    + (rows.length ? "" : `<p class="muted">אין תוצאות בסינון הזה</p>`);
}
$("#scanResults").addEventListener("click", e => { const r = e.target.closest("[data-t]"); if (r) runAnalyze(r.dataset.t); });

// ---------------- market ----------------
async function loadMarket() {
  $("#market").innerHTML = `<div class="spinner"></div>`;
  try {
    const m = await api("market");
    const secs = [...m.sectors].sort((a, b) => b["day_%"] - a["day_%"]);
    const mx = Math.max(1, ...secs.map(s => Math.abs(s["day_%"])));
    $("#market").innerHTML = `<h2>🌎 השוק האמריקאי</h2>
      <div class="grid2">${m.indexes.map(i => `<div class="kv"><div class="k">${esc(i.name)}</div>
        <div class="v">${fmt(i.price)} <span class="small ${cls(i["day_%"])}">${pct(i["day_%"])}</span></div></div>`).join("")}</div>
      <div class="card"><h3>מצב השוק: <span class="${cls(m.regime.score)}">${m.regime.score > .2 ? "חיובי" : m.regime.score < -.2 ? "שלילי" : "ניטרלי"}</span></h3>
        ${m.regime.notes.map(n => `<div class="reason small">${esc(n)}</div>`).join("")}
        ${m.breadth != null ? `<div class="reason small">רוחב שוק: ${fmt(m.breadth, 0)}% מהסקטורים מעל ממוצע 50 יום</div>` : ""}</div>
      <div class="card"><h3>סקטורים היום</h3>${secs.map(s => {
        const w = Math.abs(s["day_%"]) / mx * 50;
        return `<div class="secrow"><span>${esc(s.name)}</span><div class="secbar"><i style="${s["day_%"] >= 0 ? `left:50%;width:${w}%;background:var(--green)` : `right:50%;width:${w}%;background:var(--red)`}"></i></div>
          <span class="${cls(s["day_%"])} ltr">${pct(s["day_%"])}</span></div>`; }).join("")}
        <p class="small muted">שבוע: ${secs.slice().sort((a, b) => b["week_%"] - a["week_%"]).slice(0, 3).map(s => `${esc(s.name)} ${pct(s["week_%"], 1)}`).join(" · ")}</p></div>`;
  } catch (e) { $("#market").innerHTML = `<div class="err">⚠️ ${esc(e.message)}</div>`; }
}

// ---------------- news ----------------
async function loadNews() {
  $("#news").innerHTML = `<div class="spinner"></div>`;
  try {
    const n = await api("news");
    $("#news").innerHTML = `<h2>📰 חדשות שוק</h2><div class="card"><b>סנטימנט כללי:</b>
      <span class="${cls(n.score)}">${fmt(n.score, 2)}</span> · 🟢${n.positive} 🔴${n.negative}</div>
      <div class="card news">${newsList(n.items)}</div>`;
  } catch (e) { $("#news").innerHTML = `<div class="err">⚠️ ${esc(e.message)}</div>`; }
}

// ---------------- settings ----------------
function fillSettings() {
  $("#s-buy").value = S.buy; $("#s-sell").value = S.sell; $("#s-refresh").value = String(S.refresh);
  ["pm", "news", "market", "mtf", "notify"].forEach(k => $(`#s-${k}`).checked = !!S[k]);
  $("#s-key").value = S.key || "";
  $("#keyCard").classList.toggle("hidden", !META.auth_required);
}
$("#saveSettings").addEventListener("click", async () => {
  S.buy = +$("#s-buy").value || .6; S.sell = +$("#s-sell").value || .4; S.refresh = +$("#s-refresh").value;
  ["pm", "news", "market", "mtf", "notify"].forEach(k => S[k] = $(`#s-${k}`).checked);
  S.key = $("#s-key").value.trim();
  if (S.notify && "Notification" in window && Notification.permission === "default") {
    try { await Notification.requestPermission(); } catch (e) {}
  }
  if (S.notify && !("Notification" in window)) toast("התראות זמינות באייפון רק אחרי התקנה למסך הבית (iOS 16.4+)");
  save(); toast("✅ נשמר"); scheduleRefresh();
});

// ---------------- boot ----------------
async function boot() {
  if ("serviceWorker" in navigator) navigator.serviceWorker.register("/sw.js").catch(() => {});
  renderControls();
  try {
    META = await api("meta");
    $("#sessionBadge").textContent = META.session_he;
    renderGroups();
    if (META.auth_required && !META.auth_ok) { location.hash = "#settings"; toast("הזן קוד גישה בהגדרות"); }
  } catch (e) { $("#sessionBadge").textContent = "אין חיבור לשרת"; }
  route();
  renderPicker();
}

// Start screen: choose a stock first – nothing runs until the user taps Start
let pickGroup = null;
function renderPicker() {
  const groups = Object.keys(META.groups || {});
  if (!pickGroup) pickGroup = groups[0];
  const list = META.groups[pickGroup] || [];
  $("#tickerInput").value = S.ticker || "";
  $("#analysis").innerHTML = `<div class="card"><h3>1️⃣ בחר קבוצה</h3>
    <div class="chips wrap">${groups.map(g => `<button class="chip ${g === pickGroup ? "on" : ""}" data-pg="${esc(g)}">${esc(g)}</button>`).join("")}</div>
    <h3 style="margin-top:12px">2️⃣ בחר מניה</h3>
    <div class="chips wrap">${list.map(t => `<button class="chip ${t === S.ticker ? "on" : ""}" data-pt="${t}">${t}</button>`).join("")}</div>
    <p class="small muted" style="margin:10px 0 0">או הקלד סימול בתיבה למעלה, ואז לחץ <b>Start</b>.</p></div>
    <button id="startBtn" class="btn primary block">▶️ Start${S.ticker ? " – " + esc(S.ticker) : ""}</button>`;
}
$("#analysis").addEventListener("click", e => {
  const g = e.target.closest("[data-pg]"), t = e.target.closest("[data-pt]");
  if (g) { pickGroup = g.dataset.pg; renderPicker(); }
  if (t) { S.ticker = t.dataset.pt; save(); renderPicker(); }
  if (e.target.closest("#startBtn")) {
    const v = ($("#tickerInput").value || S.ticker || "").trim().toUpperCase();
    if (v) runAnalyze(v); else toast("בחר מניה קודם");
  }
});
boot();
