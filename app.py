"""StockAI – מערכת ניתוח ואיתות למניות בשוק האמריקאי.

הרצה:  streamlit run app.py
"""
from __future__ import annotations

import re
import threading
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import importlib
import os
import sys


def _refresh_engine():
    """Streamlit keeps imported modules in memory between reruns. After the code is
    updated on the server, reload any engine module whose file changed, in dependency
    order, so new functions are always available without a manual reboot."""
    order = ["engine.data", "engine.indicators", "engine.sentiment", "engine.model",
             "engine.premarket", "engine.universe", "engine.alerts", "engine.analyzer"]
    changed = False
    for name in order:
        mod = sys.modules.get(name)
        if mod is None or not getattr(mod, "__file__", None):
            continue
        mtime = os.path.getmtime(mod.__file__)
        if changed or getattr(mod, "_loaded_mtime", None) != mtime:
            mod = importlib.reload(mod)
            changed = True
        mod._loaded_mtime = mtime


_refresh_engine()

from engine import data as D  # noqa: E402
from engine.alerts import format_alert, load_config, save_config, send_telegram  # noqa: E402
from engine.analyzer import (Settings, analyze, day_change, quick_scan, rank,  # noqa: E402
                             scan_row)
from engine.universe import GROUPS

IL = ZoneInfo("Asia/Jerusalem")


def il_time(ts: float, fmt: str = "%H:%M") -> str:
    """Server runs in UTC – always show Israel time."""
    return datetime.fromtimestamp(ts, IL).strftime(fmt)

st.set_page_config(page_title="StockAI – ניתוח מניות", page_icon="📈", layout="wide")

st.markdown("""
<style>
.main .block-container, [data-testid="stSidebar"] {direction: rtl; text-align: right;}
[data-testid="stMetric"] {direction: rtl;}
.js-plotly-plot, .stDataFrame, code, pre {direction: ltr; text-align: left;}
.signal-box {padding: 18px 22px; border-radius: 14px; color: white; font-size: 1.6rem;
             font-weight: 700; text-align: center; margin-bottom: 8px;}
.reason {font-size: 0.95rem; margin: 2px 0;}
.small {color: #888; font-size: 0.85rem;}
</style>
""", unsafe_allow_html=True)

COLORS = {"strong_buy": "#0a8f3c", "buy": "#2fb45a", "hold": "#8a8f98",
          "sell": "#e0573b", "strong_sell": "#b3161b"}
BAR_MIN = {"1m": 1, "2m": 2, "5m": 5, "15m": 15, "30m": 30, "60m": 60, "90m": 90,
           "1d": 390, "1wk": 1950}
INTERVAL_LABEL = {"1m": "דקה", "2m": "2 דקות", "5m": "5 דקות", "15m": "15 דקות",
                  "30m": "30 דקות", "60m": "שעה", "1d": "יומי", "1wk": "שבועי"}

ss = st.session_state
ss.setdefault("last_action", {})


# ───────────── Per-user separation ─────────────
# Each person picks a short user name once. It lives in the page address (?u=name), so the
# home-screen icon keeps it, and scans/settings/watchlist are kept separately per user.
def _clean_user(v) -> str:
    v = re.sub(r"[^a-z0-9_-]", "", str(v or "").lower())[:20]
    return v


USER = _clean_user(st.query_params.get("u"))
if USER:
    ss["user"] = USER
elif ss.get("user"):
    USER = ss["user"]
    try:
        st.query_params["u"] = USER
    except Exception:
        pass
if not USER:
    ss.setdefault("guest_id", "guest-" + uuid.uuid4().hex[:6])
OWNER = USER or ss["guest_id"]          # who owns scans in this session
cfg = load_config(USER or None) if USER else load_config("__guest__")


def horizon_text(interval: str, bars: int) -> str:
    if interval == "1d":
        return f"{bars} ימי מסחר"
    if interval == "1wk":
        return f"{bars} שבועות"
    m = BAR_MIN[interval] * bars
    return f"{m} דקות" if m < 60 else f"{m / 60:.1f} שעות"


# ───────────────────────────── Sidebar ─────────────────────────────
with st.sidebar:
    if USER:
        st.markdown(f"👤 **משתמש:** `{USER}`")
        if st.button("החלף משתמש", key="switch_user"):
            ss.pop("user", None)
            st.query_params.clear()
            st.rerun()
    st.title("📈 StockAI")
    interval = st.selectbox("אינטרוול גרף", list(INTERVAL_LABEL),
                            index=list(INTERVAL_LABEL).index(cfg["interval"]),
                            format_func=lambda k: f"{k} – {INTERVAL_LABEL[k]}")
    horizon = st.slider("אופק התחזית (מספר נרות קדימה)", 1, 60, int(cfg["horizon"]))
    st.caption(f"התחזית: האם המחיר יהיה גבוה יותר בעוד **{horizon_text(interval, horizon)}**")
    c1, c2 = st.columns(2)
    buy_th = c1.number_input("סף קנייה", 0.5, 0.95, float(cfg["buy_th"]), 0.01)
    sell_th = c2.number_input("סף מכירה", 0.05, 0.5, float(cfg["sell_th"]), 0.01)
    refresh = st.select_slider("רענון אוטומטי (דקות)", [0, 1, 2, 3, 5, 10, 15, 30, 60],
                               value=int(cfg["refresh_minutes"]),
                               format_func=lambda v: "כבוי" if v == 0 else str(v))
    with st.expander("שכבות ניתוח"):
        use_news = st.checkbox("סנטימנט חדשות", True)
        use_market = st.checkbox("מצב השוק הכללי (SPY/VIX)", True)
        use_mtf = st.checkbox("אישור מטווח זמן גבוה", True)
        use_pm = st.checkbox("ניתוח מסחר מוקדם / פערים", True)
        prepost = st.checkbox("הצג מסחר מוקדם ומאוחר בגרף", True)
    with st.expander("התראות טלגרם"):
        tg_tok = st.text_input("Bot token", cfg["telegram_token"], type="password")
        tg_chat = st.text_input("Chat ID", cfg["telegram_chat_id"])
        alert_p = st.slider("התרע כשההסתברות מעל", 0.55, 0.9, float(cfg["alert_min_prob"]), 0.01)
        if st.button("שלח הודעת בדיקה"):
            ok = send_telegram("✅ StockAI מחובר", {"telegram_token": tg_tok, "telegram_chat_id": tg_chat})
            if ok:
                st.success("נשלח")
            else:
                st.error("השליחה נכשלה – בדוק token / chat id")
    if st.button("💾 שמור הגדרות כברירת מחדל"):
        cfg.update(interval=interval, horizon=horizon, buy_th=buy_th, sell_th=sell_th,
                   refresh_minutes=refresh, telegram_token=tg_tok, telegram_chat_id=tg_chat,
                   alert_min_prob=alert_p)
        if USER:
            save_config(cfg, USER)
            st.success("נשמר")
        else:
            st.warning("כדי לשמור, בחר קודם שם משתמש (למעלה במסך הראשי)")
    from engine.premarket import SESSION_HE, market_session
    _sess = market_session()
    _icon = {"regular": "🟢", "premarket": "🟡", "afterhours": "🟠", "closed": "🔴"}[_sess]
    st.markdown(f"**שוק ארה\"ב:** {_icon} {SESSION_HE[_sess]}")
    st.caption("מסחר מוקדם 11:00–16:30 · רגיל 16:30–23:00 · מאוחר 23:00–03:00 (שעון ישראל, משתנה בשבועות מעבר השעון)")
    st.caption("⚠️ כלי עזר לניתוח בלבד – לא ייעוץ השקעות. הסתברות היא הערכה סטטיסטית ולא ודאות.")

settings = Settings(interval=interval, horizon=horizon, buy_th=buy_th, sell_th=sell_th,
                    use_news=use_news, use_market=use_market, use_mtf=use_mtf, prepost=prepost,
                    use_premarket=use_pm)
tg_cfg = {"telegram_token": tg_tok, "telegram_chat_id": tg_chat}


# ───────────────────────────── Charts ─────────────────────────────
def price_chart(r) -> go.Figure:
    x = r.frame.tail(300 if interval in D.INTRADAY else 400)
    ext = None
    if prepost and interval in D.INTRADAY:
        try:
            e = D.get_history(r.ticker, interval, prepost=True)
            e = e[e.index >= x.index[0]]
            if e.index.tz is not None:
                e = e.tz_convert("America/New_York")
            t = e.index.time
            from engine.premarket import CLOSE, OPEN
            ext = e[(t < OPEN) | (t >= CLOSE)]
        except Exception:
            ext = None
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.02,
                        row_heights=[0.55, 0.13, 0.16, 0.16])
    fig.add_trace(go.Candlestick(x=x.index, open=x["Open"], high=x["High"], low=x["Low"],
                                 close=x["Close"], name="מחיר",
                                 increasing_line_color="#26a69a", decreasing_line_color="#ef5350"), 1, 1)
    if ext is not None and len(ext):
        fig.add_trace(go.Candlestick(x=ext.index, open=ext["Open"], high=ext["High"], low=ext["Low"],
                                     close=ext["Close"], name="מסחר מוקדם/מאוחר",
                                     increasing_line_color="#7fb3d5", decreasing_line_color="#c39bd3",
                                     opacity=0.7), 1, 1)
    pmd = r.premarket or {}
    if "pm_high" in pmd:
        fig.add_hline(y=pmd["pm_high"], line=dict(color="#f39c12", width=1, dash="dot"),
                      annotation_text="שיא מוקדם", row=1, col=1)
        fig.add_hline(y=pmd["pm_low"], line=dict(color="#f39c12", width=1, dash="dot"),
                      annotation_text="שפל מוקדם", row=1, col=1)
    for col, color in [("ema9", "#f5a623"), ("ema21", "#4a90e2"), ("ema50", "#9b59b6"), ("ema200", "#555")]:
        if x[col].notna().any():
            fig.add_trace(go.Scatter(x=x.index, y=x[col], name=col.upper(), line=dict(width=1.2, color=color)), 1, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["vwap"], name="VWAP",
                             line=dict(width=1.3, dash="dot", color="#e91e63")), 1, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["bb_up"], name="BB", line=dict(width=0.6, color="#aaa"),
                             showlegend=False), 1, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["bb_lo"], name="BB", line=dict(width=0.6, color="#aaa"),
                             fill="tonexty", fillcolor="rgba(150,150,150,0.07)", showlegend=False), 1, 1)
    lv = r.levels
    for s_ in lv["supports"]:
        fig.add_hline(y=s_, line=dict(color="#26a69a", width=1, dash="dash"), row=1, col=1)
    for r_ in lv["resistances"]:
        fig.add_hline(y=r_, line=dict(color="#ef5350", width=1, dash="dash"), row=1, col=1)
    fig.add_hline(y=lv["stop"], line=dict(color="red", width=1.5), annotation_text="סטופ", row=1, col=1)
    fig.add_hline(y=lv["target"], line=dict(color="green", width=1.5), annotation_text="יעד", row=1, col=1)
    vc = np.where(x["Close"] >= x["Open"], "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x.index, y=x["Volume"], marker_color=vc, name="נפח", showlegend=False), 2, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["rsi"], name="RSI", line=dict(color="#7e57c2")), 3, 1)
    fig.add_hline(y=70, line=dict(color="red", width=0.7, dash="dot"), row=3, col=1)
    fig.add_hline(y=30, line=dict(color="green", width=0.7, dash="dot"), row=3, col=1)
    hc = np.where(x["hist"] >= 0, "#26a69a", "#ef5350")
    fig.add_trace(go.Bar(x=x.index, y=x["hist"], marker_color=hc, name="MACD hist", showlegend=False), 4, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["macd"], name="MACD", line=dict(color="#2196f3", width=1)), 4, 1)
    fig.add_trace(go.Scatter(x=x.index, y=x["signal"], name="Signal", line=dict(color="#ff9800", width=1)), 4, 1)
    fig.update_layout(height=820, xaxis_rangeslider_visible=False, margin=dict(l=10, r=10, t=30, b=10),
                      legend=dict(orientation="h", y=1.03), hovermode="x unified")
    if interval in D.INTRADAY and not prepost:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[16, 9.5], pattern="hour")])
    elif interval in D.INTRADAY:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"]), dict(bounds=[20, 4], pattern="hour")])
    else:
        fig.update_xaxes(rangebreaks=[dict(bounds=["sat", "mon"])])
    for i, t in enumerate(["", "נפח", "RSI", "MACD"], start=1):
        if t:
            fig.update_yaxes(title_text=t, row=i, col=1)
    return fig


def gauge(p: float) -> go.Figure:
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=p * 100, number={"suffix": "%", "font": {"size": 44}},
        title={"text": "הסתברות לעלייה"},
        gauge={"axis": {"range": [0, 100]}, "bar": {"color": "#222", "thickness": 0.25},
               "steps": [{"range": [0, 30], "color": "#b3161b"}, {"range": [30, 40], "color": "#e0573b"},
                         {"range": [40, 60], "color": "#c9cdd3"}, {"range": [60, 70], "color": "#2fb45a"},
                         {"range": [70, 100], "color": "#0a8f3c"}]}))
    fig.update_layout(height=260, margin=dict(l=20, r=20, t=50, b=0))
    return fig


def fmt_big(n):
    if n is None or (isinstance(n, float) and np.isnan(n)):
        return "—"
    for unit, div in (("T", 1e12), ("B", 1e9), ("M", 1e6)):
        if abs(n) >= div:
            return f"{n / div:.2f}{unit}"
    return f"{n:,.0f}"


# ───────────────────────────── Tabs ─────────────────────────────
if not USER:
    with st.container(border=True):
        st.markdown("#### 👤 בחר שם משתמש (פעם אחת)")
        st.caption("כך הסריקות, ההגדרות ורשימת המעקב שלך נשמרות רק לך ולא מתערבבות עם משתמשים אחרים. "
                   "באנגלית בלבד, למשל doron. אפשר גם לדלג ולהשתמש כאורח (בלי שמירה).")
        u1, u2 = st.columns([3, 1])
        new_user = u1.text_input("שם משתמש", key="new_user", label_visibility="collapsed",
                                 placeholder="שם משתמש באנגלית")
        if u2.button("שמור", type="primary", use_container_width=True, key="save_user"):
            nu = _clean_user(new_user)
            if nu:
                ss["user"] = nu
                st.query_params["u"] = nu
                st.rerun()
            else:
                st.error("רק אותיות באנגלית / ספרות")
elif ss.get("_user_tip") is None:
    ss["_user_tip"] = True
    st.toast(f"מחובר כ-{USER}. כדי שהאייקון במסך הבית יזכור אותך, הוסף אותו מחדש מהדף הזה.", icon="👤")

# Scanner first: the app opens on the last scan
tab_s, tab_a, tab_m, tab_n, tab_b, tab_h = st.tabs(
    ["🔎 סורק מניות", "📈 ניתוח מניה", "🌎 מגמות שוק", "📰 חדשות", "🧪 בדיקה היסטורית", "❓ איך זה עובד"])


def render_analysis():
    ticker = ss.get("active_ticker")
    if not ticker:
        return
    try:
        with st.spinner(f"מנתח את {ticker}..."):
            r = analyze(ticker, settings)
    except Exception as e:
        st.error(f"שגיאה בניתוח {ticker}: {e}")
        return
    ss["result"] = r

    # alert on signal change
    prev = ss["last_action"].get(ticker)
    strong = r.p_up >= alert_p or r.p_up <= 1 - alert_p
    if prev and prev != r.action_level and r.action_level != "hold":
        st.toast(f"{ticker}: איתות חדש – {r.action} ({r.p_up * 100:.0f}%)", icon="🔔")
    if strong and prev != r.action_level:
        send_telegram(format_alert(r), tg_cfg)
    ss["last_action"][ticker] = r.action_level

    name = r.info.get("shortName") or ticker
    st.subheader(f"{name} ({ticker})")
    st.caption(f"עודכן: {il_time(time.time(), '%H:%M:%S')} | נר אחרון: {r.frame.index[-1]} | "
               f"תחזית ל-{horizon_text(interval, horizon)} קדימה"
               + (f" | רענון כל {refresh} דק'" if refresh else ""))

    g, sig, lv = st.columns([1.2, 1, 1.2])
    with g:
        st.plotly_chart(gauge(r.p_up), use_container_width=True)
    with sig:
        st.markdown(f"<div class='signal-box' style='background:{COLORS[r.action_level]}'>{r.action}</div>",
                    unsafe_allow_html=True)
        if r.premarket and r.session in ("premarket", "afterhours"):
            st.metric(f"מחיר ({'מוקדם' if r.session == 'premarket' else 'מאוחר'})", f"{r.price:,.2f}",
                      f"{r.premarket['ext_change_%']:+.2f}% מ{r.premarket['ref_label']}")
        else:
            st.metric("מחיר", f"{r.price:,.2f}", f"{day_change(r.frame):+.2f}% היום")
        auc = r.model["auc"]
        q = "גבוהה" if auc >= 0.58 else "בינונית" if auc >= 0.53 else "נמוכה – כמעט ללא יתרון"
        st.markdown(f"**אמינות המודל למניה זו:** {q}  \n<span class='small'>AUC={auc:.3f} | "
                    f"דיוק={r.model['accuracy'] * 100:.1f}% על {r.model['n_oos']:,} תחזיות מחוץ למדגם</span>",
                    unsafe_allow_html=True)
    with lv:
        L = r.levels
        dirn = "לונג" if r.p_up >= 0.5 else "שורט"
        st.markdown(f"**רמות מסחר מוצעות ({dirn})**")
        a, b = st.columns(2)
        a.metric("סטופ לוס", f"{L['stop']:.2f}", f"{(L['stop'] / r.price - 1) * 100:+.2f}%")
        b.metric("יעד רווח", f"{L['target']:.2f}", f"{(L['target'] / r.price - 1) * 100:+.2f}%")
        a.metric("סיכון:סיכוי", f"1:{L['rr']:.1f}")
        b.metric("ATR", f"{L['atr']:.2f}")
        st.caption("תמיכות: " + ", ".join(f"{v:.2f}" for v in L["supports"][:3]) + "  \n"
                   "התנגדויות: " + ", ".join(f"{v:.2f}" for v in L["resistances"][:3]))

    pmd = r.premarket
    if pmd:
        with st.container(border=True):
            st.markdown(f"#### 🌅 {pmd['session_he']}"
                        + ("" if pmd["active"] else " <span class='small'>(השכבה משפיעה על ההסתברות רק במסחר מוקדם ובשעה הראשונה)</span>"),
                        unsafe_allow_html=True)
            k = st.columns(5)
            k[0].metric(pmd["ref_label"], f"{pmd['prev_close']:.2f}")
            k[1].metric("פער", f"{pmd['gap_%']:+.2f}%")
            if "pm_high" in pmd:
                k[2].metric("שיא / שפל מוקדם", f"{pmd['pm_high']:.2f} / {pmd['pm_low']:.2f}")
                k[3].metric("VWAP מוקדם", f"{pmd['pm_vwap']:.2f}")
                rv = pmd.get("pm_rel_volume")
                k[4].metric("נפח מוקדם יחסי", f"x{rv:.1f}" if rv else "—")
            stt = pmd.get("stats", {})
            if stt.get("n", 0):
                st.write(f"**איך פערים דומים ({stt['range_%'][0]}%–{stt['range_%'][1]}%) התנהגו ב-2 השנים האחרונות** "
                         f"({stt['n']} מקרים): המשיכו בכיוון הפער ב-**{stt['continuation'] * 100:.0f}%**, "
                         f"נסגרו (חזרו לסגירה הקודמת) ב-**{stt['fill_rate'] * 100:.0f}%**, "
                         f"שמרו על הפער עד הסגירה ב-**{stt['held_rate'] * 100:.0f}%**, "
                         f"ממוצע פתיחה→סגירה **{stt['avg_open_to_close_%']:+.2f}%**")
            for n_ in pmd["notes"]:
                st.markdown(f"<div class='reason'>• {n_}</div>", unsafe_allow_html=True)

    st.plotly_chart(price_chart(r), use_container_width=True)

    c1, c2, c3 = st.columns(3)
    with c1:
        st.markdown("#### 🧠 למה? – סיבות טכניות")
        for t in r.tech_reasons[:10]:
            st.markdown(f"<div class='reason'>{t}</div>", unsafe_allow_html=True)
        for nm, d in r.patterns:
            st.markdown(f"<div class='reason'>{'🟢' if d > 0 else '🔴' if d < 0 else '⚪'} תבנית: {nm}</div>",
                        unsafe_allow_html=True)
    with c2:
        st.markdown("#### ⚖️ מרכיבי ההסתברות")
        b = dict(r.breakdown)
        base = b.pop("מודל ML (מכויל)")
        st.write(f"בסיס – מודל ML מכויל: **{base * 100:.1f}%**")
        for k, v in b.items():
            st.write(f"{'⬆️' if v > 0 else '⬇️'} {k}: {v:+.2f} (לוגיט)")
        st.write(f"**= סופי: {r.p_up * 100:.1f}%**")
        st.markdown("#### 🔬 משתנים מובילים במודל")
        cb = r.model["contrib"]
        top = pd.concat([cb.head(4), cb.tail(4)])
        st.bar_chart(top.rename("השפעה"), horizontal=True, height=230)
    with c3:
        st.markdown("#### 🌐 הקשר")
        if r.market:
            st.write("**שוק:** " + " · ".join(r.market["notes"]))
        if r.mtf:
            st.write("**טווח גבוה:** " + " · ".join(r.mtf["notes"]))
        if r.news:
            n = r.news
            st.write(f"**חדשות:** סנטימנט {n['score']:+.2f} | 🟢{n['positive']} 🔴{n['negative']} "
                     f"⚪{n['neutral']} | {n['buzz_24h']} כתבות ב-24 שעות")
        i = r.info
        if i:
            st.markdown("**נתוני יסוד ואנליסטים**")
            rows = {"סקטור": i.get("sector"), "שווי שוק": fmt_big(i.get("marketCap")),
                    "מכפיל רווח": i.get("trailingPE"), "מכפיל עתידי": i.get("forwardPE"),
                    "בטא": i.get("beta"), "המלצת אנליסטים": i.get("recommendationKey"),
                    "מחיר יעד ממוצע": i.get("targetMeanPrice"),
                    "שורט מהצף": f"{i['shortPercentOfFloat'] * 100:.1f}%" if i.get("shortPercentOfFloat") else None,
                    "דוחות הבאים": i.get("earningsDate")}
            for k, v in rows.items():
                if v is not None:
                    st.write(f"{k}: **{round(v, 2) if isinstance(v, float) else v}**")

    if r.news and r.news["items"]:
        with st.expander(f"📰 כתבות אחרונות על {ticker} ({len(r.news['items'])})"):
            for it in r.news["items"][:25]:
                em = "🟢" if it["sentiment"] > 0.15 else "🔴" if it["sentiment"] < -0.15 else "⚪"
                age = f"לפני {it['age_h']:.0f} ש'" if it["age_h"] < 48 else f"לפני {it['age_h'] / 24:.0f} ימים"
                st.markdown(f"{em} [{it['title']}]({it['link']}) <span class='small'>· {it['source']} · {age}</span>",
                            unsafe_allow_html=True)


def stock_picker():
    """Selection screen – nothing is analysed until the user presses Start."""
    st.markdown("### 1️⃣ בחר מניה")
    groups = ["✍️ הקלדה חופשית"] + list(GROUPS)
    g = st.selectbox("קבוצה", groups, index=1, key="pick_group")
    if g == groups[0]:
        t = st.text_input("סימול מניה (למשל NVDA)", key="pick_free")
    else:
        opts = GROUPS[g]
        t = st.radio("מניה", opts, horizontal=True, key=f"pick_{g}")
    t = (t or "").upper().strip()
    st.markdown("### 2️⃣ הפעל")
    c1, c2 = st.columns(2)
    if c1.button(f"▶️ Start – נתח {t}" if t else "▶️ Start", type="primary",
                 use_container_width=True, disabled=not t):
        ss["active_ticker"] = t
    if c2.button("⏹ עצור", use_container_width=True, disabled=not ss.get("active_ticker")):
        ss["active_ticker"] = None
    st.caption(f"הגדרות: גרף {interval} · אופק {horizon_text(interval, horizon)} · "
               f"רענון {'כבוי' if not refresh else f'כל {refresh} דק׳'} — לשינוי פתח את התפריט בצד (>>)")


with tab_a:
    stock_picker()
    st.divider()
    if not ss.get("active_ticker"):
        st.info("👆 בחר מניה ולחץ **Start** כדי להתחיל את הניתוח")
    elif refresh:
        st.fragment(run_every=refresh * 60)(render_analysis)()
    else:
        render_analysis()


# ───── Scanner ─────
SCAN_DIR = Path(__file__).resolve().parent / "user_data"


def _scan_file(owner: str) -> Path:
    return SCAN_DIR / f"last_scan_{owner}.json"


def save_last_scan(job) -> None:
    """Keep each named user's last scan on disk, so it is shown when the app is reopened."""
    owner = job.get("owner") or ""
    if not owner or owner.startswith("guest-") or not job["rows"]:
        return
    try:
        SCAN_DIR.mkdir(exist_ok=True)
        data = {k: job[k] for k in ("n", "done", "started", "finished", "tickers", "owner")}
        data["rows"] = rank(list(job["rows"])).astype(object).where(lambda d: d.notna(), None).to_dict("records")
        tmp = _scan_file(owner).with_suffix(".tmp")
        tmp.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        tmp.replace(_scan_file(owner))
    except Exception:
        pass


def load_last_scan(owner: str):
    try:
        data = json.loads(_scan_file(owner).read_text(encoding="utf-8"))
        data.update(status="saved", cancel=False, current="")
        data["finished"] = data.get("finished") or data["started"]
        return data
    except Exception:
        return None


@st.cache_resource
def scan_store():
    """Process-wide store: scans keep running on the server even when the phone app is
    in the background or closed, and results are there when you come back."""
    return {"jobs": {}, "last_by": {}, "lock": threading.Lock()}


def start_scan_job(store, tickers, s, owner):
    jid = uuid.uuid4().hex[:8]
    job = {"status": "running", "rows": [], "n": len(tickers), "done": 0, "current": tickers[0],
           "started": time.time(), "finished": None, "cancel": False, "tickers": list(tickers),
           "owner": owner}
    with store["lock"]:
        for k in [k for k, v in store["jobs"].items() if time.time() - v["started"] > 6 * 3600]:
            store["jobs"].pop(k, None)
        store["jobs"][jid] = job
        store["last_by"][owner] = jid

    def run():
        try:
            market = D.market_regime() if s.use_market else None
            for i, t in enumerate(tickers):
                if job["cancel"]:
                    job["status"] = "cancelled"
                    break
                job["current"] = t
                job["rows"].append(scan_row(t, s, market))
                job["done"] = i + 1
                if job["done"] % 5 == 0:
                    job["finished"] = time.time()
                    save_last_scan(job)
            else:
                job["status"] = "done"
        except Exception as e:  # pragma: no cover
            job["status"], job["error"] = "error", str(e)
        job["finished"] = time.time()
        save_last_scan(job)

    threading.Thread(target=run, daemon=True).start()
    return jid


def render_scan_job(job, style_scan):
    n, done = job["n"], job["done"]
    started = il_time(job["started"])
    if job["status"] == "running":
        st.progress(done / n if n else 1.0,
                    f"🔄 רץ ברקע (התחיל {started}) – מנתח {job['current']}... ({done}/{n}). "
                    f"אפשר לעבור לאפליקציה אחרת ולחזור.")
        if st.button("⏹ עצור סריקה", key="cancel_scan"):
            job["cancel"] = True
    elif job["status"] == "done":
        st.success(f"✅ הסריקה הסתיימה ב-{il_time(job['finished'])} ({n} מניות)")
    elif job["status"] == "saved":
        st.info(f"🕘 הסריקה האחרונה שלך: {il_time(job['finished'], '%d/%m %H:%M')} "
                f"({len(job['rows'])} מניות) – לחץ ▶️ הרץ סריקה לרענון")
    elif job["status"] == "cancelled":
        st.warning(f"הסריקה נעצרה ({done}/{n})")
    else:
        st.error(f"שגיאה בסריקה: {job.get('error')}")
    if job["rows"]:
        df = rank(list(job["rows"]))
        ss["scan"] = df
        st.dataframe(style_scan(df), use_container_width=True, hide_index=True, height=560)
        st.caption("ממוין לפי חוזק האיתות. שורה ירוקה = 60%+ לעלייה. AUC מעל 0.55 = למודל יתרון מוכח.")


with tab_s:
    st.markdown("### 🔎 סורק – מדרג את המניות לפי חוזק האיתות")
    wl_text = st.text_area("רשימת מעקב (מופרדות בפסיק)", ", ".join(cfg["watchlist"]), height=80)
    wl = [t.strip().upper() for t in wl_text.replace("\n", ",").split(",") if t.strip()]
    cc1, cc2, cc3 = st.columns(3)
    run_scan = cc1.button("▶️ הרץ סריקה", type="primary")
    if cc2.button("💾 שמור רשימה"):
        cfg["watchlist"] = wl
        if USER:
            save_config(cfg, USER)
            st.success("נשמר")
        else:
            st.warning("כדי לשמור, בחר קודם שם משתמש (למעלה במסך הראשי)")
    preset = cc3.selectbox("או טען קבוצה מוכנה", ["—"] + list(GROUPS))
    presets = GROUPS
    if preset != "—":
        wl = presets[preset]
        st.caption("ייסרקו: " + ", ".join(wl) + " — לחץ ▶️ הרץ סריקה")
    def style_scan(df):
        """Whole row green when P(up) >= 60%; probability cell red at/below the sell threshold."""
        if "סיכוי עלייה %" not in df:
            return df

        def row_color(row):
            p = row.get("סיכוי עלייה %")
            if isinstance(p, (int, float)) and not pd.isna(p) and p >= 60:
                return ["background-color: rgba(34,197,94,.30); font-weight: 600"] * len(row)
            return [""] * len(row)

        def sell_cell(v):
            if isinstance(v, (int, float)) and not pd.isna(v) and v <= sell_th * 100:
                return "background-color: rgba(224,87,59,.35)"
            return ""
        return df.style.apply(row_color, axis=1).map(sell_cell, subset=["סיכוי עלייה %"])

    store = scan_store()
    if run_scan and wl:
        jid = start_scan_job(store, wl, settings, OWNER)
        ss["scan_job"] = jid
        try:
            st.query_params["scan"] = jid
        except Exception:
            pass
    # Find the job: this session → URL → most recent job on the server (survives app switch/close)
    jid = ss.get("scan_job") or st.query_params.get("scan") or store["last_by"].get(OWNER)
    job = store["jobs"].get(jid) if jid else None
    if job and job.get("owner") != OWNER:      # never show someone else's scan
        job = None
    if job is None and not run_scan:           # reopened app / server restarted → last saved scan
        job = load_last_scan(OWNER)
    if job:
        if jid in store["jobs"]:
            ss["scan_job"] = jid
        if job["status"] == "running":
            def live():
                render_scan_job(job, style_scan)
                if job["status"] != "running":
                    st.rerun()  # finished → redraw once without the 2-second timer
            st.fragment(run_every=2)(live)()
        else:
            render_scan_job(job, style_scan)
    elif "scan" in ss:
        st.dataframe(style_scan(ss["scan"]), use_container_width=True, hide_index=True, height=560)


# ───── Market trends ─────
def render_market():
    st.markdown("### 🌎 תמונת מצב – השוק האמריקאי")
    try:
        idx = D.get_snapshot(tuple(D.INDEXES))
        cols = st.columns(len(idx))
        for c, (_, row) in zip(cols, idx.iterrows()):
            c.metric(D.INDEXES.get(row["ticker"], row["ticker"]), f"{row['price']:,.2f}", f"{row['day_%']:+.2f}%")
        reg = D.market_regime()
        st.info("**מצב השוק:** " + " · ".join(reg["notes"]) + f"  → ציון {reg['score']:+.2f}")
        sec = D.get_snapshot(tuple(D.SECTOR_ETFS))
        sec["סקטור"] = sec["ticker"].map(D.SECTOR_ETFS)
        per = st.radio("תקופה", ["day_%", "week_%", "month_%"], horizontal=True,
                       format_func={"day_%": "יום", "week_%": "שבוע", "month_%": "חודש"}.get)
        sec = sec.sort_values(per)
        fig = go.Figure(go.Bar(x=sec[per], y=sec["סקטור"], orientation="h",
                               marker_color=np.where(sec[per] >= 0, "#26a69a", "#ef5350"),
                               text=sec[per].map(lambda v: f"{v:+.2f}%"), textposition="outside"))
        fig.update_layout(height=460, title="ביצועי סקטורים (רוטציה)", margin=dict(l=10, r=10, t=40, b=10))
        st.plotly_chart(fig, use_container_width=True)
        breadth = sec["above_50d"].mean() * 100
        st.write(f"**רוחב שוק:** {breadth:.0f}% מהסקטורים נסחרים מעל ממוצע 50 יום")
    except Exception as e:
        st.error(f"שגיאה בטעינת נתוני שוק: {e}")


def render_news():
    st.markdown("### 📰 חדשות שוק כלליות + סנטימנט")
    try:
        from engine.sentiment import analyze_news
        mn = analyze_news(D.get_market_news())
        st.metric("סנטימנט שוק כללי", f"{mn['score']:+.2f}",
                  f"🟢{mn['positive']} / 🔴{mn['negative']} כתבות")
        for it in mn["items"]:
            em = "🟢" if it["sentiment"] > 0.15 else "🔴" if it["sentiment"] < -0.15 else "⚪"
            st.markdown(f"{em} [{it['title']}]({it['link']}) <span class='small'>· {it['source']} · "
                        f"לפני {it['age_h']:.0f} ש'</span>", unsafe_allow_html=True)
    except Exception as e:
        st.error(f"שגיאה בטעינת חדשות: {e}")


def gated(tab, key, label, fn):
    with tab:
        if st.button(label, key=f"btn_{key}", type="primary"):
            ss[key] = True
        if ss.get(key):
            fn()
        else:
            st.caption("הנתונים נטענים רק אחרי לחיצה")


gated(tab_m, "show_market", "▶️ טען נתוני שוק", render_market)
gated(tab_n, "show_news", "▶️ טען חדשות", render_news)


# ───── Backtest ─────
with tab_b:
    r = ss.get("result")
    if not r:
        st.info("הרץ קודם ניתוח מניה בלשונית הראשונה")
    else:
        m = r.model
        bt = m["backtest"]
        st.markdown(f"### 🧪 בדיקה היסטורית Walk-Forward – {r.ticker} ({interval}, אופק {horizon_text(interval, horizon)})")
        st.caption("המודל אומן בכל פעם רק על העבר וחזה את התקופה הבאה שלא ראה – זו הדרך הכנה למדוד אמינות.")
        k = st.columns(6)
        k[0].metric("AUC", f"{m['auc']:.3f}")
        k[1].metric("דיוק כיוון", f"{m['accuracy'] * 100:.1f}%")
        k[2].metric("שיעור עליות בסיסי", f"{m['base_rate'] * 100:.1f}%")
        if bt:
            k[3].metric("עסקאות", bt["trades"])
            k[4].metric("אחוז הצלחה", "—" if np.isnan(bt["win_rate"]) else f"{bt['win_rate'] * 100:.1f}%")
            k[5].metric("תשואת אסטרטגיה", f"{bt['total_%']:+.1f}%", f"קנה-והחזק {bt['buyhold_%']:+.1f}%")
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=bt["equity"].index, y=(bt["equity"] - 1) * 100, name="אסטרטגיה"))
            fig.add_trace(go.Scatter(x=bt["buyhold"].index, y=(bt["buyhold"] - 1) * 100, name="קנה והחזק"))
            fig.update_layout(height=380, title="תשואה מצטברת % (כולל עמלה 2bps לעסקה)",
                              margin=dict(l=10, r=10, t=40, b=10))
            st.plotly_chart(fig, use_container_width=True)
            st.write(f"דיוק בקניות: **{bt['long_hit'] * 100:.1f}%** | דיוק במכירות: "
                     f"**{bt['short_hit'] * 100:.1f}%** | ירידה מקסימלית: **{bt['max_dd_%']:.1f}%**"
                     if not (np.isnan(bt["long_hit"]) or np.isnan(bt["short_hit"])) else
                     "לא היו מספיק איתותים מעבר לספים – המודל לא מצא יתרון ברור.")
        if m["reliability"] is not None:
            st.markdown("#### טבלת כיול – כשהמודל אמר X%, כמה פעמים זה באמת עלה?")
            rel = m["reliability"].reset_index()
            rel["bucket"] = rel["bucket"].astype(str)
            rel["actual_up_rate"] = (rel["actual_up_rate"] * 100).round(1)
            st.dataframe(rel.rename(columns={"bucket": "טווח הסתברות", "actual_up_rate": "% עלייה בפועל",
                                             "n": "מספר מקרים"}), hide_index=True)


# ───── Help ─────
with tab_h:
    st.markdown("""
### איך המערכת מחשבת את ההסתברות
1. **נתונים** – נרות תוך-יומיים/יומיים מ-Yahoo Finance, חדשות מ-Yahoo ו-Google News, מדדים, VIX וסקטורים.
2. **25+ אינדיקטורים** – EMA 9/21/50/200, MACD, RSI, בולינגר, סטוכסטי, ADX/DMI, ATR, VWAP, OBV, MFI, CCI, Williams %R, SuperTrend, נפח יחסי, תמיכה/התנגדות ותבניות נרות.
3. **מודל למידת מכונה** – אנסמבל של רגרסיה לוגיסטית + Gradient Boosting, שמאומן על ההיסטוריה של המניה הספציפית
   ושואל: *"האם המחיר יהיה גבוה יותר בעוד H נרות?"*
4. **Walk-Forward אמיתי** – המודל נבדק רק על נתונים שלא ראה. מזה נמדדים AUC ודיוק, וההסתברות **מכוילת**:
   אם המודל לא הוכיח יתרון (AUC≈0.5) – ההסתברות נמשכת אוטומטית לשיעור הבסיס ולא תראה לך "ביטחון" מזויף.
5. **שכבות התאמה** – סנטימנט חדשות (משוקלל לפי טריות), מצב השוק (SPY מול ממוצעים, VIX), אישור מטווח זמן גבוה, תבניות נרות.
6. **רמות מסחר** – סטופ/יעד לפי ATR ותמיכות/התנגדויות.

### איך לקרוא את התוצאה
- **AUC** מעל 0.55 = יתרון סטטיסטי קיים; מעל 0.60 = חזק (נדיר בשוק אמיתי). מתחת ל-0.53 – התייחס לאיתות בזהירות רבה.
- הסתברות 65% פירושה שבהיסטוריה, במצבים דומים, המחיר עלה בערך 65% מהפעמים – **35% מהפעמים הוא ירד**. לכן תמיד סטופ.
- נתוני Yahoo בחינם עשויים להתעכב בכמה דקות.

### ⚠️ אחריות
הכלי מיועד ללימוד ולתמיכה בהחלטה. אין בו ייעוץ השקעות, ואין מערכת שיודעת לחזות את השוק בוודאות.
""")
