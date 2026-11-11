"""Exit advisor – "I bought it, when do I sell?"

For every open position the advisor builds a sell plan from the stock's DAILY volatility
(ATR) and its resistance levels, then checks it against the live price and the intraday
analysis (ML probability, trend) to give one clear verdict:

    SELL_ALL   – stop / trailing stop hit, final target reached, or the trend turned against a winner
    SELL_HALF  – first profit target reached: take half, move the stop to break-even
    TIGHTEN    – overheated next to resistance: consider partial profit, stop raised
    HOLD       – plan intact, let it work

Plan (R = initial risk per share = entry − initial stop):
    initial stop  = entry − 2×ATR(daily)      (or the user's stop %)
    TP1 (½ out)   = entry + 1.5R              (pulled just under a resistance on the way)
    TP2 (all out) = entry + 3R                (or the user's target %)
    trailing stop = highest high since entry − 3×ATR, active once the trade is +1R
    after TP1     = stop never below the entry price (risk-free trade)

Positions are plain dicts: {"id", "ticker", "entry", "qty", "opened" (YYYY-MM-DD),
"stop_pct"?, "target_pct"?, "tp1_done"?}.
"""
from __future__ import annotations

import json
import uuid
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I

STORE_DIR = Path(__file__).resolve().parent.parent

VERDICTS = {
    "SELL_ALL": ("🔴 מכור עכשיו", "sell"),
    "SELL_HALF": ("🟠 מכור חצי – מימוש רווח ראשון", "partial"),
    "TIGHTEN": ("🟡 שקול מימוש חלקי / הדק סטופ", "tighten"),
    "HOLD": ("🟢 החזק", "hold"),
}


# ───────────── storage ─────────────
def store_path(user: str | None = None) -> Path:
    return STORE_DIR / (f"positions_{user}.json" if user else "positions.json")


def load_positions(user: str | None = None) -> list[dict]:
    p = store_path(user)
    try:
        if p.exists():
            return [clean_position(x) for x in json.loads(p.read_text(encoding="utf-8"))]
    except Exception:
        pass
    return []


def save_positions(items: list[dict], user: str | None = None) -> None:
    store_path(user).write_text(json.dumps(items, ensure_ascii=False, indent=2), encoding="utf-8")


def clean_position(d: dict) -> dict:
    """Validate/normalise a position coming from a form, a file or the phone app."""
    t = str(d.get("ticker", "")).upper().strip()
    entry = float(d.get("entry", 0) or 0)
    if not t or entry <= 0:
        raise ValueError("צריך טיקר ומחיר קנייה גדול מ-0")
    out = {
        "id": str(d.get("id") or uuid.uuid4().hex[:8]),
        "ticker": t,
        "entry": entry,
        "qty": float(d.get("qty", 0) or 0),
        "opened": str(d.get("opened") or date.today().isoformat())[:10],
        "tp1_done": bool(d.get("tp1_done", False)),
    }
    for k in ("stop_pct", "target_pct"):
        v = d.get(k)
        if v not in (None, "", 0, "0"):
            out[k] = abs(float(v))
    return out


# ───────────── plan ─────────────
def _since(df: pd.DataFrame, opened: str) -> pd.DataFrame:
    try:
        ts = pd.Timestamp(opened)
        idx = df.index
        if idx.tz is not None:
            ts = ts.tz_localize(idx.tz)
        sub = df[idx >= ts]
        return sub if len(sub) else df.tail(1)
    except Exception:
        return df.tail(1)


def build_plan(pos: dict, daily: pd.DataFrame, price: float) -> dict:
    """Stops and targets from daily ATR + resistances. `daily` must hold OHLC (+ optional atr)."""
    x = daily if "atr" in daily else daily.assign(atr=I.atr(daily))
    entry = pos["entry"]
    held = _since(x, pos["opened"])
    atr = float(held["atr"].iloc[-1]) if not np.isnan(held["atr"].iloc[-1]) else float(x["atr"].dropna().iloc[-1])
    atr = max(atr, entry * 0.003)

    stop0 = entry * (1 - pos["stop_pct"] / 100) if pos.get("stop_pct") else entry - 2 * atr
    R = max(entry - stop0, entry * 0.002)
    tp2 = entry * (1 + pos["target_pct"] / 100) if pos.get("target_pct") else entry + 3 * R
    tp1 = entry + 1.5 * R if not pos.get("target_pct") else entry + (tp2 - entry) / 2
    tp1 = min(tp1, entry + (tp2 - entry) * 0.75)

    # Sell just under a resistance that sits on the way to TP1 (price often stalls there)
    _, resistances = I.support_resistance(x, lookback=min(250, len(x)))
    res_near = [r for r in resistances if entry + 0.8 * R <= r < tp1]
    if res_near:
        tp1 = min(res_near) - 0.1 * atr

    high_since = max(float(held["High"].max()), price)
    trail_on = high_since >= entry + R
    trail = high_since - 3 * atr if trail_on else None
    be_lock = pos.get("tp1_done") or high_since >= tp1

    stop, stop_kind = stop0, "סטופ התחלתי"
    if be_lock and entry > stop:
        stop, stop_kind = entry, "סטופ בנקודת האיזון"
    if trail is not None and trail > stop:
        stop, stop_kind = trail, "סטופ נגרר"
    return {"entry": entry, "atr": atr, "R": R, "stop0": stop0, "stop": stop,
            "stop_kind": stop_kind, "tp1": tp1, "tp2": tp2, "trail": trail,
            "high_since": high_since, "resistances": resistances}


# ───────────── verdict ─────────────
def advise(pos: dict, price: float, daily: pd.DataFrame, p_up: float | None = None,
           auc: float | None = None, buy_th: float = 0.60, sell_th: float = 0.40,
           min_auc: float = 0.53, intraday: pd.DataFrame | None = None) -> dict:
    """Pure decision function (no network) – easy to test."""
    plan = build_plan(pos, daily, price)
    entry, qty = pos["entry"], pos.get("qty") or 0
    pnl_pct = (price / entry - 1) * 100
    r_mult = (price - entry) / plan["R"]
    proven = auc is not None and not np.isnan(auc) and auc >= min_auc
    reasons: list[str] = []

    dx = daily if "rsi" in daily else daily.assign(rsi=I.rsi(daily["Close"]))
    rsi_d = float(dx["rsi"].iloc[-1]) if len(dx) else 50.0
    near_res = [r for r in plan["resistances"] if price <= r <= price * 1.012]
    trend_broken = False
    if intraday is not None and len(intraday) and "supertrend" in intraday:
        last = intraday.iloc[-1]
        trend_broken = (last["supertrend"] < 0 if "supertrend" in last else False) and \
            ("vwap" in last and not np.isnan(last["vwap"]) and last["Close"] < last["vwap"])

    if price <= plan["stop"]:
        code = "SELL_ALL"
        reasons.append(f"המחיר {price:.2f} שבר את ה{plan['stop_kind']} ({plan['stop']:.2f})")
        if pnl_pct > 0:
            reasons.append(f"נועל רווח של {pnl_pct:+.1f}%")
    elif price >= plan["tp2"]:
        code = "SELL_ALL"
        reasons.append(f"הגיע ליעד הסופי {plan['tp2']:.2f} ({pnl_pct:+.1f}%) – לממש את כל הרווח")
    elif price >= plan["tp1"] and not pos.get("tp1_done"):
        code = "SELL_HALF"
        reasons.append(f"הגיע ליעד הראשון {plan['tp1']:.2f} ({pnl_pct:+.1f}%)")
        reasons.append(f"למכור חצי ולהזיז סטופ לנקודת האיזון {entry:.2f} – העסקה הופכת לחסרת סיכון")
    elif proven and p_up is not None and p_up <= sell_th and r_mult >= 0.5:
        code = "SELL_ALL"
        reasons.append(f"המודל התהפך לירידה (סיכוי עלייה {p_up * 100:.0f}%) כשאתה ברווח {pnl_pct:+.1f}% – עדיף לממש")
    elif proven and p_up is not None and p_up <= min(sell_th - 0.1, 0.3) and r_mult < 0:
        code = "SELL_ALL"
        reasons.append(f"איתות מכירה חזק (סיכוי עלייה {p_up * 100:.0f}%) – יציאה מוקדמת לפני הסטופ מקטינה הפסד")
    elif r_mult >= 0.8 and (rsi_d >= 75 or (near_res and rsi_d >= 65)):
        code = "TIGHTEN"
        if rsi_d >= 75:
            reasons.append(f"RSI יומי {rsi_d:.0f} – קנייתי-יתר, תיקון סביר")
        if near_res:
            reasons.append(f"המחיר צמוד להתנגדות {near_res[0]:.2f}")
        reasons.append("אפשר לממש שליש עד חצי, או להעלות סטופ מתחת לשפל האחרון")
    elif trend_broken and r_mult >= 1:
        code = "TIGHTEN"
        reasons.append("המגמה התוך-יומית נשברה (מתחת ל-VWAP ו-Supertrend שלילי) כשאתה ברווח")
    else:
        code = "HOLD"
        if p_up is not None and p_up >= buy_th and proven:
            reasons.append(f"המגמה עדיין חיובית (סיכוי עלייה {p_up * 100:.0f}%) – לתת לרווח לרוץ")
        reasons.append(f"התוכנית בתוקף: סטופ {plan['stop']:.2f}, יעד {plan['tp1'] if not pos.get('tp1_done') else plan['tp2']:.2f}")

    label, level = VERDICTS[code]
    nxt = plan["tp2"] if pos.get("tp1_done") or price >= plan["tp1"] else plan["tp1"]
    return {
        "id": pos["id"], "ticker": pos["ticker"], "code": code, "verdict": label, "level": level,
        "price": price, "entry": entry, "qty": qty, "opened": pos["opened"],
        "pnl_pct": pnl_pct, "pnl": (price - entry) * qty, "r_mult": r_mult,
        "stop": plan["stop"], "stop_kind": plan["stop_kind"], "stop0": plan["stop0"],
        "tp1": plan["tp1"], "tp2": plan["tp2"], "trail": plan["trail"],
        "to_stop_%": (plan["stop"] / price - 1) * 100, "to_target_%": (nxt / price - 1) * 100,
        "next_target": nxt, "high_since": plan["high_since"], "atr": plan["atr"],
        "rsi_d": rsi_d, "p_up": p_up, "auc": auc, "tp1_done": bool(pos.get("tp1_done")),
        "reasons": reasons,
    }


def check_position(pos: dict, s, market: dict | None = None) -> dict:
    """Live check: fetches daily data + runs the intraday analysis for the timing layer."""
    from .analyzer import analyze   # local import – analyzer is heavy
    pos = clean_position(pos)
    daily = I.add_all(D.get_history(pos["ticker"], "1d"), intraday=False)
    r = None
    try:
        r = analyze(pos["ticker"], s, market=market, with_info=False)
        price = float(r.price)
    except Exception:
        price = float(daily["Close"].iloc[-1])
    out = advise(pos, price, daily,
                 p_up=r.p_up if r else None, auc=r.model.get("auc") if r else None,
                 buy_th=s.buy_th, sell_th=s.sell_th, min_auc=s.min_auc,
                 intraday=r.frame if r else None)
    out["session"] = r.session if r else ""
    out["action"] = r.action if r else ""
    return out


def format_exit_alert(a: dict) -> str:
    rs = "\n".join(f"• {x}" for x in a["reasons"])
    return (f"<b>{a['ticker']}</b> – {a['verdict']}\n"
            f"מחיר {a['price']:.2f} | קנייה {a['entry']:.2f} | <b>{a['pnl_pct']:+.1f}%</b>\n"
            f"סטופ {a['stop']:.2f} | יעד 1 {a['tp1']:.2f} | יעד 2 {a['tp2']:.2f}\n{rs}")
