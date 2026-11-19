"""Layers 4–6 of the method: financial reports, financial health & valuation, publications &
recommendations – all POINT-IN-TIME.

A quarterly report counts only from the day it was actually published (the earnings date that
follows the quarter end, +1 day because most companies report after the close). If the report
date is unknown we assume quarter end + 50 days. Analyst rating changes and insider trades
count only from their own dates. So a report "as of" a past day never sees later numbers.

Every layer returns a score in [-1, 1] plus Hebrew notes explaining it.
"""
from __future__ import annotations

import math
from datetime import timezone

import numpy as np
import pandas as pd

from . import data as D

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

SECTOR_ETF = {
    "Technology": "XLK", "Financial Services": "XLF", "Healthcare": "XLV", "Energy": "XLE",
    "Consumer Cyclical": "XLY", "Consumer Defensive": "XLP", "Industrials": "XLI",
    "Basic Materials": "XLB", "Utilities": "XLU", "Real Estate": "XLRE",
    "Communication Services": "XLC",
}


def _now_ny():
    return pd.Timestamp.now(tz="America/New_York")


def _tz(ts) -> pd.Timestamp:
    ts = pd.Timestamp(ts)
    return ts.tz_localize("America/New_York") if ts.tzinfo is None else ts.tz_convert("America/New_York")


def _row(df: pd.DataFrame | None, *names) -> pd.Series | None:
    if df is None or df.empty:
        return None
    for n in names:
        if n in df.index:
            s = pd.to_numeric(df.loc[n], errors="coerce")
            if isinstance(s, pd.DataFrame):
                s = s.iloc[0]
            return s
    return None


def _sq(x: float, scale: float) -> float:
    """Smoothly squash x/scale into [-1, 1]."""
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return 0.0
    return float(np.tanh(x / scale))


# ───────────────────────── raw data (cached) ─────────────────────────
@D.ttl_cache(6 * 3600)
def raw_fundamentals(ticker: str) -> dict:
    out = {"income": None, "balance": None, "cash": None, "earnings": None,
           "grades": None, "insiders": None, "sector": None}
    if yf is None:
        return out
    t = yf.Ticker(ticker)
    for key, attr in (("income", "quarterly_income_stmt"), ("balance", "quarterly_balance_sheet"),
                      ("cash", "quarterly_cashflow"), ("grades", "upgrades_downgrades"),
                      ("insiders", "insider_transactions")):
        try:
            v = getattr(t, attr)
            out[key] = v if isinstance(v, pd.DataFrame) and not v.empty else None
        except Exception:
            pass
    try:
        e = t.get_earnings_dates(limit=40)
        out["earnings"] = e if isinstance(e, pd.DataFrame) and not e.empty else None
    except Exception:
        pass
    try:
        out["sector"] = (t.info or {}).get("sector")
    except Exception:
        pass
    return out


# ───────────────────────── point-in-time helpers ─────────────────────────
def _report_dates(raw: dict) -> list[pd.Timestamp]:
    e = raw.get("earnings")
    if e is None:
        return []
    idx = e.index if isinstance(e.index, pd.DatetimeIndex) else pd.to_datetime(e.index, errors="coerce")
    return sorted(_tz(d) for d in idx if pd.notna(d))


def available_quarters(raw: dict, as_of: pd.Timestamp) -> list[pd.Timestamp]:
    """Quarter-end columns whose report was public at `as_of`, newest first."""
    inc = raw.get("income")
    if inc is None:
        return []
    reports = _report_dates(raw)
    cols = []
    for c in inc.columns:
        q_end = _tz(c)
        pub = next((r for r in reports if q_end < r <= q_end + pd.Timedelta(days=100)), None)
        avail = (pub + pd.Timedelta(days=1)).normalize() if pub is not None \
            else q_end + pd.Timedelta(days=50)
        if avail <= as_of:
            cols.append(c)
    return sorted(cols, key=lambda c: _tz(c), reverse=True)


def _val(df, names, col):
    r = _row(df, *names)
    if r is None or col not in r.index:
        return np.nan
    v = r[col]
    return float(v) if pd.notna(v) else np.nan


def _price_at(ticker: str, as_of: pd.Timestamp) -> float:
    try:
        d = D.truncate(D.get_history(ticker, "1d", "2y"), as_of, daily=True)
        return float(d["Close"].iloc[-1])
    except Exception:
        return np.nan


# ───────────────────────── layer 4: reports ─────────────────────────
def layer_reports(raw: dict, as_of: pd.Timestamp) -> dict:
    notes, parts = [], []
    q = available_quarters(raw, as_of)
    inc = raw.get("income")
    m = {}
    if len(q) >= 1:
        last = q[0]
        rev = _val(inc, ["Total Revenue", "Operating Revenue"], last)
        gp = _val(inc, ["Gross Profit"], last)
        op = _val(inc, ["Operating Income", "Total Operating Income As Reported"], last)
        ni = _val(inc, ["Net Income", "Net Income Common Stockholders"], last)
        m.update(last_quarter=str(_tz(last).date()), revenue=rev,
                 gross_margin=gp / rev if rev else np.nan, op_margin=op / rev if rev else np.nan,
                 net_margin=ni / rev if rev else np.nan)
        # same quarter a year earlier
        yago = next((c for c in q[1:] if 330 <= (_tz(last) - _tz(c)).days <= 400), None)
        if yago is not None:
            rev0 = _val(inc, ["Total Revenue", "Operating Revenue"], yago)
            gp0 = _val(inc, ["Gross Profit"], yago)
            if rev0:
                g = rev / rev0 - 1
                m["rev_yoy"] = g
                parts.append(_sq(g, 0.25))
                notes.append(f"{'🟢' if g > 0 else '🔴'} הכנסות ברבעון האחרון {g * 100:+.0f}% מול שנה קודמת")
                if gp0 and not np.isnan(gp):
                    dm = gp / rev - gp0 / rev0
                    m["gm_change"] = dm
                    parts.append(_sq(dm, 0.05) * 0.6)
                    icon = "⚪" if abs(dm) < 0.005 else "🟢" if dm > 0 else "🔴"
                    notes.append(f"{icon} מרווח גולמי {gp / rev * 100:.0f}% "
                                 f"({dm * 100:+.1f} נק' משנה שעברה)")
        else:
            notes.append("⚪ אין מספיק רבעונים להשוואה שנתית")
        if not np.isnan(m.get("op_margin", np.nan)):
            om = m["op_margin"]
            parts.append(_sq(om, 0.15) * 0.6)
            notes.append(f"{'🟢' if om > 0 else '🔴'} מרווח תפעולי {om * 100:.0f}%")
    else:
        notes.append("⚪ לא נמצאו דוחות כספיים שפורסמו עד התאריך הזה")

    # earnings surprises – only reports already published
    e = raw.get("earnings")
    if e is not None:
        idx = [_tz(d) for d in e.index]
        past = e[[d < as_of for d in idx]]
        col = next((c for c in past.columns if "Surprise" in c), None)
        if col is not None:
            sur = pd.to_numeric(past[col], errors="coerce").dropna().head(4)
            if len(sur):
                last_s = float(sur.iloc[0])
                beats = int((sur > 0).sum())
                m.update(last_surprise=last_s, beats_of_4=beats)
                parts.append(_sq(last_s, 10) * 0.7)
                notes.append(f"{'🟢' if last_s > 0 else '🔴'} הפתעה בדוח האחרון {last_s:+.1f}% "
                             f"(עקפה תחזיות ב-{beats} מתוך {len(sur)} הדוחות האחרונים)")
    score = float(np.clip(np.mean(parts), -1, 1)) if parts else 0.0
    return {"name": "דוחות כספיים", "score": score, "notes": notes, "metrics": m, "has_data": bool(parts)}


# ───────────────────────── layer 5: health & valuation ─────────────────────────
def layer_health(ticker: str, raw: dict, as_of: pd.Timestamp) -> dict:
    notes, parts, m = [], [], {}
    q = available_quarters(raw, as_of)
    bal, cf, inc = raw.get("balance"), raw.get("cash"), raw.get("income")
    if not q:
        return {"name": "מצב פיננסי ותמחור", "score": 0.0, "notes": ["⚪ אין נתונים שפורסמו עד התאריך"],
                "metrics": m, "has_data": False}
    last = q[0]
    cash = _val(bal, ["Cash Cash Equivalents And Short Term Investments", "Cash And Cash Equivalents"], last)
    debt = _val(bal, ["Total Debt"], last)
    shares = _val(bal, ["Ordinary Shares Number", "Share Issued"], last)
    ttm_q = [c for c in q[:4]]
    fcf = [ _val(cf, ["Free Cash Flow"], c) for c in ttm_q]
    rev = [ _val(inc, ["Total Revenue", "Operating Revenue"], c) for c in ttm_q]
    fcf_ttm = float(np.nansum(fcf)) if len(ttm_q) == 4 and not all(np.isnan(fcf)) else np.nan
    rev_ttm = float(np.nansum(rev)) if len(ttm_q) == 4 and not all(np.isnan(rev)) else np.nan
    m.update(cash=cash, debt=debt, fcf_ttm=fcf_ttm, rev_ttm=rev_ttm)

    if not np.isnan(cash):
        net = cash - (0 if np.isnan(debt) else debt)
        m["net_cash"] = net
        parts.append(0.4 if net > 0 else -0.4)
        notes.append(f"{'🟢' if net > 0 else '🔴'} {'מזומן נטו' if net > 0 else 'חוב נטו'} "
                     f"{abs(net) / 1e9:.2f} מיליארד $")
    if not np.isnan(fcf_ttm):
        if fcf_ttm > 0:
            parts.append(0.6)
            fm = fcf_ttm / rev_ttm if rev_ttm else np.nan
            notes.append(f"🟢 תזרים חופשי חיובי ב-12 חודשים: {fcf_ttm / 1e9:.2f} מיליארד $"
                         + (f" ({fm * 100:.0f}% מההכנסות)" if not np.isnan(fm) else ""))
        else:
            runway = cash / (-fcf_ttm / 12) if not np.isnan(cash) and fcf_ttm < 0 else np.nan
            m["runway_months"] = runway
            parts.append(-0.3 if (np.isnan(runway) or runway > 24) else -0.8)
            notes.append(f"🔴 שורפת מזומנים: {fcf_ttm / 1e6:.0f} מיליון $ בשנה"
                         + (f" · הקופה מספיקה לכ-{runway:.0f} חודשים" if not np.isnan(runway) else ""))
    # dilution: share count vs a year earlier
    yago = next((c for c in q[1:] if 330 <= (_tz(last) - _tz(c)).days <= 400), None)
    if yago is not None and not np.isnan(shares):
        sh0 = _val(bal, ["Ordinary Shares Number", "Share Issued"], yago)
        if sh0:
            dil = shares / sh0 - 1
            m["dilution_yoy"] = dil
            if dil > 0.03:
                parts.append(-_sq(dil, 0.10))
                notes.append(f"🔴 דילול: מספר המניות עלה ב-{dil * 100:.1f}% בשנה")
            elif dil < -0.01:
                parts.append(0.3)
                notes.append(f"🟢 רכישה עצמית: מספר המניות ירד ב-{-dil * 100:.1f}% בשנה")
    # valuation & quality (rule of 40)
    price = _price_at(ticker, as_of)
    if not np.isnan(price) and not np.isnan(shares) and rev_ttm and not np.isnan(rev_ttm):
        mcap = price * shares
        ps = mcap / rev_ttm
        m.update(price=price, market_cap=mcap, ps=ps)
        growth = None
        y4 = [c for c in q if 330 <= (_tz(last) - _tz(c)).days <= 400]
        if y4:
            r0 = _val(inc, ["Total Revenue", "Operating Revenue"], y4[0])
            r1 = _val(inc, ["Total Revenue", "Operating Revenue"], last)
            growth = (r1 / r0 - 1) if r0 else None
        fcf_m = fcf_ttm / rev_ttm if not np.isnan(fcf_ttm) else 0
        r40 = ((growth or 0) + fcf_m) * 100
        m["rule_of_40"] = r40
        parts.append(_sq(r40 - 40, 40) * 0.6)
        notes.append(f"{'🟢' if r40 >= 40 else '🔴'} 'כלל ה-40' (צמיחה + שולי תזרים): {r40:.0f}")
        # expensive relative to quality → negative
        fair_ps = max(r40, 5) / 4          # rough: a Rule-of-40 score of 40 ≈ 10× sales
        rel = ps / fair_ps
        m["ps_vs_quality"] = rel
        parts.append(-_sq(np.log(rel), 1.0) * 0.6)
        notes.append(f"{'🔴' if rel > 1.5 else '🟢' if rel < 0.8 else '⚪'} מכפיל מכירות {ps:.1f} "
                     f"({'יקר' if rel > 1.5 else 'זול' if rel < 0.8 else 'סביר'} ביחס לאיכות)")
    score = float(np.clip(np.mean(parts), -1, 1)) if parts else 0.0
    return {"name": "מצב פיננסי ותמחור", "score": score, "notes": notes, "metrics": m, "has_data": bool(parts)}


# ───────────────────────── layer 6: publications & recommendations ─────────────────────────
def layer_publications(raw: dict, as_of: pd.Timestamp, news_score: float | None = None,
                       horizon_days: float = 1.0) -> dict:
    notes, parts, m = [], [], {}
    g = raw.get("grades")
    if g is not None:
        idx = [_tz(d) for d in g.index]
        win = g[[(as_of - pd.Timedelta(days=90)) <= d < as_of for d in idx]]
        if len(win):
            act = win["Action"].astype(str).str.lower() if "Action" in win else pd.Series(dtype=str)
            ups, downs = int((act == "up").sum()), int((act == "down").sum())
            m.update(upgrades_90d=ups, downgrades_90d=downs, rating_events_90d=len(win))
            if ups or downs:
                parts.append(_sq(ups - downs, 3))
                notes.append(f"{'🟢' if ups >= downs else '🔴'} אנליסטים ב-90 יום: {ups} שדרוגים, {downs} הורדות")
            if {"currentPriceTarget", "priorPriceTarget"} <= set(win.columns):
                cur = pd.to_numeric(win["currentPriceTarget"], errors="coerce")
                pri = pd.to_numeric(win["priorPriceTarget"], errors="coerce")
                ch = ((cur / pri - 1).replace([np.inf, -np.inf], np.nan)).dropna()
                ch = ch[(pri > 0)]
                if len(ch) >= 2:
                    avg = float(ch.mean())
                    m["target_change_avg"] = avg
                    parts.append(_sq(avg, 0.10))
                    notes.append(f"{'🟢' if avg > 0 else '🔴'} מחירי יעד עודכנו בממוצע {avg * 100:+.0f}% "
                                 f"({len(ch)} עדכונים)")
                tgt = cur.dropna()
                if len(tgt):
                    m["avg_target"] = float(tgt.tail(10).mean())
    ins = raw.get("insiders")
    if ins is not None and "Start Date" in ins:
        dates = pd.to_datetime(ins["Start Date"], errors="coerce")
        dates = [(_tz(d) if pd.notna(d) else None) for d in dates]
        mask = [d is not None and (as_of - pd.Timedelta(days=90)) <= d < as_of for d in dates]
        w = ins[mask]
        if len(w):
            txt = w.get("Text", pd.Series("", index=w.index)).astype(str).str.lower()
            val = pd.to_numeric(w.get("Value", 0), errors="coerce").fillna(0)
            buys = float(val[txt.str.contains("purchase|buy")].sum())
            sells = float(val[txt.str.contains("sale|sell")].sum())
            m.update(insider_buys_90d=buys, insider_sells_90d=sells)
            if buys > 0:
                parts.append(0.5)
                notes.append(f"🟢 בעלי עניין קנו ב-{buys / 1e6:.1f} מיליון $ ב-90 יום")
            elif sells > 0:
                notes.append(f"⚪ בעלי עניין מכרו ב-{sells / 1e6:.1f} מיליון $ ב-90 יום (לרוב מכירות מתוכננות)")
    if news_score is not None:
        parts.append(float(news_score))
        notes.append(f"{'🟢' if news_score > 0.1 else '🔴' if news_score < -0.1 else '⚪'} "
                     f"סנטימנט חדשות {news_score:+.2f}")
    # upcoming earnings = event risk inside the forecast horizon
    rd = _report_dates(raw)
    nxt = next((d for d in rd if d >= as_of), None)
    if nxt is not None:
        days = (nxt - as_of).total_seconds() / 86400
        m["days_to_earnings"] = days
        if days <= max(horizon_days, 1) + 1:
            m["earnings_in_horizon"] = True
            notes.append(f"⚠️ דוח רבעוני בעוד {days:.0f} ימים – בתוך אופק התחזית, סיכון לתנועה חדה")
        else:
            notes.append(f"⚪ הדוח הבא בעוד {days:.0f} ימים")
    score = float(np.clip(np.mean(parts), -1, 1)) if parts else 0.0
    return {"name": "פרסומים והמלצות", "score": score, "notes": notes, "metrics": m, "has_data": bool(parts)}


# ───────────────────────── layer 1 add-on: sector strength ─────────────────────────
def sector_strength(sector: str | None, as_of: pd.Timestamp) -> dict | None:
    etf = SECTOR_ETF.get(sector or "")
    if not etf:
        return None
    try:
        e = D.truncate(D.get_history(etf, "1d", "2y"), as_of, daily=True)["Close"]
        s = D.truncate(D.get_history("SPY", "1d", "2y"), as_of, daily=True)["Close"]
        rel = (e.iloc[-1] / e.iloc[-21] - 1) - (s.iloc[-1] / s.iloc[-21] - 1)
        return {"etf": etf, "rel_20d": float(rel), "score": _sq(rel, 0.04),
                "note": f"{'🟢' if rel > 0 else '🔴'} סקטור {etf} {rel * 100:+.1f}% מול השוק ב-20 יום"}
    except Exception:
        return None


def analyze_fundamentals(ticker: str, as_of: pd.Timestamp | None = None,
                         news_score: float | None = None, horizon_days: float = 1.0) -> dict:
    as_of = _tz(as_of) if as_of is not None else _now_ny()
    raw = raw_fundamentals(ticker)
    return {"reports": layer_reports(raw, as_of), "health": layer_health(ticker, raw, as_of),
            "publications": layer_publications(raw, as_of, news_score, horizon_days),
            "sector": sector_strength(raw.get("sector"), as_of), "sector_name": raw.get("sector")}
