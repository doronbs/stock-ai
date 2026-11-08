"""Extended-hours (pre-market / after-hours) analysis.

US sessions (New York time):
    pre-market 04:00-09:30 | regular 09:30-16:00 | after-hours 16:00-20:00

What we compute:
- current extended-hours price, gap % vs previous regular close
- pre-market high / low / VWAP and relative pre-market volume
- historical behaviour of SIMILAR gaps for this stock (last ~2 years):
  how often the gap continued (close > open for gap-up), how often it filled,
  average open->close move. This is turned into a bounded signal in [-1, 1].
"""
from __future__ import annotations

from datetime import datetime, time, timezone

import numpy as np
import pandas as pd

from . import data as D

PM_START, OPEN, CLOSE, AH_END = time(4, 0), time(9, 30), time(16, 0), time(20, 0)
SESSION_HE = {"premarket": "מסחר מוקדם (Pre-market)", "regular": "מסחר רגיל",
              "afterhours": "מסחר מאוחר (After-hours)", "closed": "השוק סגור"}


def market_session(now: datetime | None = None) -> str:
    ny = pd.Timestamp(now or datetime.now(timezone.utc)).tz_convert("America/New_York")
    if ny.weekday() >= 5:
        return "closed"
    t = ny.time()
    if PM_START <= t < OPEN:
        return "premarket"
    if OPEN <= t < CLOSE:
        return "regular"
    if CLOSE <= t < AH_END:
        return "afterhours"
    return "closed"


def _ny(df: pd.DataFrame) -> pd.DataFrame:
    if df.index.tz is None:
        df = df.tz_localize("America/New_York")
    else:
        df = df.tz_convert("America/New_York")
    return df


def gap_stats(daily: pd.DataFrame, gap: float, lookback: int = 504) -> dict:
    """How did similar opening gaps behave historically for this stock?"""
    d = daily.tail(lookback + 1).copy()
    d["prev_close"] = d["Close"].shift()
    d = d.dropna()
    d["gap"] = d["Open"] / d["prev_close"] - 1
    g = abs(gap)
    lo, hi = max(g * 0.5, 0.0025), max(g * 2.0, 0.006)
    same = d[(np.sign(d["gap"]) == np.sign(gap)) & d["gap"].abs().between(lo, hi)]
    n = len(same)
    if n == 0 or gap == 0:
        return {"n": 0}
    up = gap > 0
    cont = (same["Close"] > same["Open"]) if up else (same["Close"] < same["Open"])
    filled = (same["Low"] <= same["prev_close"]) if up else (same["High"] >= same["prev_close"])
    held = (same["Close"] > same["prev_close"]) if up else (same["Close"] < same["prev_close"])
    oc = (same["Close"] / same["Open"] - 1) * 100
    return {"n": n, "continuation": float(cont.mean()), "fill_rate": float(filled.mean()),
            "held_rate": float(held.mean()), "avg_open_to_close_%": float(oc.mean()),
            "range_%": (round(lo * 100, 2), round(hi * 100, 2))}


def premarket_analysis(ticker: str, session: str | None = None) -> dict:
    session = session or market_session()
    ext = _ny(D.get_history(ticker, "5m", period="5d", prepost=True))
    daily = D.get_history(ticker, "1d", "2y")
    if isinstance(daily.index, pd.DatetimeIndex) and daily.index.tz is not None:
        daily = daily.tz_convert("America/New_York")

    times = ext.index.time
    dates = ext.index.date
    last_date = dates[-1]
    regular = ext[(times >= OPEN) & (times < CLOSE)]
    today_reg = regular[regular.index.date == last_date]
    prev_reg = regular[regular.index.date < last_date]
    out: dict = {"session": session, "session_he": SESSION_HE[session], "active": False}

    # Previous regular close relative to the *current* trading day
    if session == "afterhours" or (session == "closed" and len(today_reg)):
        prev_close = float(today_reg["Close"].iloc[-1]) if len(today_reg) else float(prev_reg["Close"].iloc[-1])
        ref_label = "סגירה היום" if session == "afterhours" else "סגירה אחרונה"
    else:
        prev_close = float(prev_reg["Close"].iloc[-1]) if len(prev_reg) else float(daily["Close"].iloc[-2])
        ref_label = "סגירה קודמת"
    out["prev_close"] = prev_close
    out["ref_label"] = ref_label
    last_price = float(ext["Close"].iloc[-1])
    out["last_price"] = last_price
    out["last_time"] = str(ext.index[-1])
    out["ext_change_%"] = (last_price / prev_close - 1) * 100

    pm = ext[(dates == last_date) & (times >= PM_START) & (times < OPEN)]
    if len(pm):
        tp = (pm["High"] + pm["Low"] + pm["Close"]) / 3
        vol = pm["Volume"].sum()
        out.update(pm_high=float(pm["High"].max()), pm_low=float(pm["Low"].min()),
                   pm_vwap=float((tp * pm["Volume"]).sum() / vol) if vol else float(tp.mean()),
                   pm_volume=float(vol), pm_bars=len(pm))
        # relative PM volume vs previous days, measured up to the same clock time
        cutoff = pm.index[-1].time()
        prev_pm = ext[(dates < last_date) & (times >= PM_START) & (times <= cutoff)]
        per_day = prev_pm.groupby(prev_pm.index.date)["Volume"].sum()
        per_day = per_day[per_day > 0]
        out["pm_rel_volume"] = float(vol / per_day.mean()) if len(per_day) else None

    # Gap used for statistics: live PM gap before the open, actual opening gap after.
    if session == "premarket" and len(pm):
        gap = last_price / prev_close - 1
    elif len(today_reg) and session in ("regular", "afterhours", "closed"):
        gap = float(today_reg["Open"].iloc[0]) / float(prev_reg["Close"].iloc[-1]) - 1 if len(prev_reg) else 0.0
        out["open_gap_%"] = gap * 100
    else:
        gap = last_price / prev_close - 1
    out["gap_%"] = gap * 100
    out["stats"] = gap_stats(daily, gap) if abs(gap) >= 0.002 else {"n": 0}

    # ---- signal ----
    sig, notes = 0.0, []
    st = out["stats"]
    minutes_since_open = None
    if len(today_reg):
        minutes_since_open = (today_reg.index[-1] - today_reg.index[0]).total_seconds() / 60
    relevant = session == "premarket" or (session == "regular" and (minutes_since_open or 0) <= 60)
    if st.get("n", 0) >= 5:
        direction = 1 if gap > 0 else -1
        conf = st["n"] / (st["n"] + 20)
        cont_edge = (st["continuation"] - 0.5) * 2          # >0: gaps tend to run, <0: tend to fade
        sig += direction * cont_edge * conf
        notes.append(f"{st['n']} פערים דומים בעבר: המשכיות {st['continuation'] * 100:.0f}%, "
                     f"סגירת פער {st['fill_rate'] * 100:.0f}%, ממוצע פתיחה→סגירה "
                     f"{st['avg_open_to_close_%']:+.2f}%")
    rv = out.get("pm_rel_volume")
    if session == "premarket" and rv:
        if rv > 2 and abs(gap) > 0.01:
            sig += 0.3 * np.sign(gap)
            notes.append(f"נפח מוקדם חריג x{rv:.1f} – הפער נתמך בכסף אמיתי")
        elif rv < 0.5:
            sig *= 0.6
            notes.append(f"נפח מוקדם דליל x{rv:.1f} – מחיר מוקדם פחות אמין")
    if session == "premarket" and "pm_vwap" in out:
        above = last_price > out["pm_vwap"]
        sig += 0.2 if above else -0.2
        notes.append("מחיר מעל ה-VWAP של המסחר המוקדם" if above else "מחיר מתחת ל-VWAP של המסחר המוקדם")
    if session == "regular" and "pm_high" in out and relevant:
        if last_price > out["pm_high"]:
            sig += 0.3; notes.append("פריצה מעל שיא המסחר המוקדם")
        elif last_price < out["pm_low"]:
            sig -= 0.3; notes.append("שבירה מתחת לשפל המסחר המוקדם")
    out["signal"] = float(np.clip(sig, -1, 1))
    out["notes"] = notes
    out["active"] = bool(relevant)
    return out
