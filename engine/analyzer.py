"""Orchestrates data -> indicators -> ML model -> news -> market -> final recommendation."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from . import data as D
from . import indicators as I
from .model import train_and_predict
from .premarket import market_session, premarket_analysis
from .sentiment import analyze_news


@dataclass
class Settings:
    interval: str = "5m"
    horizon: int = 12            # bars ahead the prediction refers to
    buy_th: float = 0.60
    sell_th: float = 0.40
    use_news: bool = True
    use_market: bool = True
    use_mtf: bool = True
    prepost: bool = True         # show extended hours on the chart
    use_premarket: bool = True   # pre-market / after-hours analysis layer
    w_news: float = 0.35         # logit weights of the adjustment layers
    w_market: float = 0.15
    w_tech: float = 0.25
    w_mtf: float = 0.20
    w_pm: float = 0.50
    min_auc: float = 0.53       # buy/sell only when the model proved at least this AUC on the stock
    noise: float = 0.15         # label noise band (see model.make_labels) – set by the auto-tuner
    use_mkt: bool = True        # SPY market features – set by the auto-tuner
    as_of: pd.Timestamp | None = None   # run the report "as of" a past moment (NY time)


@dataclass
class Result:
    ticker: str
    price: float
    p_up: float
    action: str
    action_level: str
    model: dict
    tech_score: float
    tech_reasons: list
    news: dict | None
    market: dict | None
    mtf: dict | None
    levels: dict
    patterns: list
    frame: pd.DataFrame
    info: dict = field(default_factory=dict)
    breakdown: dict = field(default_factory=dict)
    session: str = "regular"
    premarket: dict | None = None
    last_close: float = 0.0
    evaluation: dict | None = None


def _logit(p):
    p = min(max(p, 1e-4), 1 - 1e-4)
    return np.log(p / (1 - p))


def _sig(z):
    return 1 / (1 + np.exp(-z))


def technical_votes(x: pd.DataFrame) -> tuple[float, list]:
    """Transparent rule-based score in [-1,1] with Hebrew explanations."""
    r = x.iloc[-1]
    p = x.iloc[-2]
    votes = []

    def v(w, cond_bull, cond_bear, txt_bull, txt_bear):
        if cond_bull:
            votes.append((w, f"🟢 {txt_bull}"))
        elif cond_bear:
            votes.append((-w, f"🔴 {txt_bear}"))

    c = r["Close"]
    v(1.0, c > r["ema21"] > r["ema50"], c < r["ema21"] < r["ema50"],
      "מחיר מעל EMA21 ו-EMA50 – מגמה עולה", "מחיר מתחת ל-EMA21 ו-EMA50 – מגמה יורדת")
    if not np.isnan(r["ema200"]):
        v(0.7, c > r["ema200"], c < r["ema200"], "מעל EMA200 – מגמה ארוכה חיובית",
          "מתחת ל-EMA200 – מגמה ארוכה שלילית")
    v(1.0, p["ema9"] <= p["ema21"] and r["ema9"] > r["ema21"],
      p["ema9"] >= p["ema21"] and r["ema9"] < r["ema21"],
      "חציית EMA9 מעל EMA21 (Golden cross קצר)", "חציית EMA9 מתחת ל-EMA21 (Death cross קצר)")
    v(0.8, r["hist"] > 0 and r["hist"] > p["hist"], r["hist"] < 0 and r["hist"] < p["hist"],
      "MACD חיובי ומתחזק", "MACD שלילי ומתחזק כלפי מטה")
    v(0.8, p["macd"] <= p["signal"] and r["macd"] > r["signal"],
      p["macd"] >= p["signal"] and r["macd"] < r["signal"],
      "חציית MACD כלפי מעלה", "חציית MACD כלפי מטה")
    if r["rsi"] < 30:
        votes.append((0.8, f"🟢 RSI במכירת יתר ({r['rsi']:.0f}) – פוטנציאל תיקון למעלה"))
    elif r["rsi"] > 70:
        votes.append((-0.8, f"🔴 RSI בקניית יתר ({r['rsi']:.0f}) – סיכון לתיקון"))
    elif 50 < r["rsi"] < 65 and r["rsi"] > p["rsi"]:
        votes.append((0.4, f"🟢 RSI במומנטום חיובי ({r['rsi']:.0f})"))
    elif 35 < r["rsi"] < 50 and r["rsi"] < p["rsi"]:
        votes.append((-0.4, f"🔴 RSI במומנטום שלילי ({r['rsi']:.0f})"))
    if r["bb_pctb"] < 0.05:
        votes.append((0.5, "🟢 מחיר ברצועת בולינגר התחתונה"))
    elif r["bb_pctb"] > 0.95:
        votes.append((-0.5, "🔴 מחיר ברצועת בולינגר העליונה"))
    if r["adx"] > 25:
        d = 1 if r["plus_di"] > r["minus_di"] else -1
        votes.append((0.7 * d, f"{'🟢' if d > 0 else '🔴'} ADX={r['adx']:.0f} – מגמה חזקה "
                                f"{'עולה' if d > 0 else 'יורדת'}"))
    v(0.6, c > r["vwap"], c < r["vwap"], "מחיר מעל VWAP – קונים בשליטה", "מחיר מתחת ל-VWAP – מוכרים בשליטה")
    v(0.6, r["supertrend"] > 0, r["supertrend"] < 0, "SuperTrend חיובי", "SuperTrend שלילי")
    if r["vol_ratio"] > 1.8:
        d = 1 if r["Close"] > r["Open"] else -1
        votes.append((0.6 * d, f"{'🟢' if d > 0 else '🔴'} נפח מסחר חריג (x{r['vol_ratio']:.1f}) "
                                f"בנר {'עולה' if d > 0 else 'יורד'}"))
    if r["stoch_k"] < 20 and r["stoch_k"] > r["stoch_d"]:
        votes.append((0.4, "🟢 סטוכסטי מתהפך מאזור מכירת יתר"))
    elif r["stoch_k"] > 80 and r["stoch_k"] < r["stoch_d"]:
        votes.append((-0.4, "🔴 סטוכסטי מתהפך מאזור קניית יתר"))
    if r["mfi"] < 20:
        votes.append((0.3, "🟢 MFI נמוך – לחץ מכירה מוצה"))
    elif r["mfi"] > 80:
        votes.append((-0.3, "🔴 MFI גבוה – לחץ קנייה מוצה"))
    obv_slope = x["obv"].diff(10).iloc[-1]
    v(0.4, obv_slope > 0 and c > x["Close"].iloc[-11], obv_slope < 0 and c < x["Close"].iloc[-11],
      "OBV עולה – כסף נכנס", "OBV יורד – כסף יוצא")

    total_w = 8.0  # normalisation constant ~ sum of typical active weights
    score = float(np.tanh(sum(w for w, _ in votes) / total_w * 2))
    reasons = [t for _, t in sorted(votes, key=lambda t: -abs(t[0]))]
    return score, reasons


def higher_timeframe(ticker: str, interval: str, as_of=None) -> dict | None:
    hi = "1d" if interval in D.INTRADAY else "1wk"
    try:
        h = I.add_all(D.truncate(D.get_history(ticker, hi), as_of, daily=True), intraday=False)
    except Exception:
        return None
    r = h.iloc[-1]
    s = 0.0
    notes = []
    if r["Close"] > r["ema50"]:
        s += 0.4; notes.append(f"גרף {hi}: מעל EMA50")
    else:
        s -= 0.4; notes.append(f"גרף {hi}: מתחת ל-EMA50")
    if not np.isnan(r["ema200"]):
        if r["ema50"] > r["ema200"]:
            s += 0.3; notes.append(f"גרף {hi}: EMA50 מעל EMA200")
        else:
            s -= 0.3; notes.append(f"גרף {hi}: EMA50 מתחת ל-EMA200")
    s += 0.3 if r["hist"] > 0 else -0.3
    notes.append(f"גרף {hi}: MACD {'חיובי' if r['hist'] > 0 else 'שלילי'}, RSI {r['rsi']:.0f}")
    return {"score": max(-1, min(1, s)), "notes": notes, "interval": hi}


def trade_levels(x: pd.DataFrame, direction: int, supports, resistances,
                 price: float | None = None) -> dict:
    c = float(price if price is not None else x["Close"].iloc[-1])
    a = float(x["atr"].iloc[-1])
    if direction >= 0:
        stop = c - 1.5 * a
        tgt = c + 2.5 * a
        if supports and supports[0] > stop and supports[0] < c:
            stop = supports[0] - 0.25 * a
        if resistances and c < resistances[0] < tgt:
            tgt = resistances[0]
    else:
        stop = c + 1.5 * a
        tgt = c - 2.5 * a
        if resistances and c < resistances[0] < stop:
            stop = resistances[0] + 0.25 * a
        if supports and tgt < supports[0] < c:
            tgt = supports[0]
    rr = abs(tgt - c) / max(abs(c - stop), 1e-9)
    return {"entry": c, "stop": stop, "target": tgt, "rr": rr, "atr": a,
            "supports": supports, "resistances": resistances}


def analyze(ticker: str, s: Settings, market: dict | None = None,
            with_info: bool = True) -> Result:
    ticker = ticker.upper().strip()
    intraday = s.interval in D.INTRADAY
    # The model always trains on regular-session bars (cleaner, liquid data);
    # extended hours are handled by the dedicated pre-market layer below.
    raw_full = D.get_history(ticker, s.interval, prepost=False)
    past = s.as_of is not None
    if past:
        # Historical report: the system sees ONLY bars that existed at that moment.
        raw = D.truncate(raw_full, s.as_of, daily=not intraday)
        if len(raw) < 300:
            raise ValueError("אין מספיק נתונים לפני התאריך הזה לאינטרוול שנבחר")
    else:
        raw = raw_full
    x = I.add_all(raw, intraday)
    session = "regular" if past else market_session()

    mkt = None
    if ticker != "SPY" and s.use_mkt:
        try:
            spy = D.get_history("SPY", s.interval, prepost=False)
            mkt = (D.truncate(spy, s.as_of, daily=not intraday) if past else spy)["Close"]
        except Exception:
            mkt = None
    model = train_and_predict(x, intraday, s.horizon, s.buy_th, s.sell_th, mkt=mkt, noise=s.noise)
    tech, reasons = technical_votes(x)
    supports, resistances = I.support_resistance(x)
    patterns = I.candle_patterns(x)
    info = D.get_info(ticker) if (with_info and not past) else {}

    news = None
    if s.use_news and not past:   # past headlines can't be reconstructed reliably → off
        try:
            news = analyze_news(D.get_news(ticker, info.get("shortName")))
        except Exception:
            news = None
    if s.use_market and market is None:
        market = D.market_regime(s.as_of)
    mtf = higher_timeframe(ticker, s.interval, s.as_of) if s.use_mtf else None

    pm = None
    if s.use_premarket and intraday and not past:
        try:
            pm = premarket_analysis(ticker, session)
        except Exception:
            pm = None

    # ── Extended hours: the ML model and the technical layers only know the last REGULAR
    # bar (yesterday's close). When the stock has already gapped in pre-market (e.g. on
    # news) their view is stale, and the question that matters is "will it rise from the
    # CURRENT pre-market price?". We blend toward the stock's own history of similar gaps
    # (how often it closed above the open after such a gap) in proportion to gap size.
    stale = 0.0
    p_model = model["p_up"]
    z = _logit(p_model)
    br = {"מודל ML (מכויל)": p_model}
    if pm and session == "premarket" and abs(pm.get("gap_%", 0)) >= 0.5:
        stale = float(np.clip(abs(pm["gap_%"]) / 2.0, 0.3, 0.85))
        if pm.get("p_gap") is not None:
            z_new = (1 - stale) * z + stale * _logit(pm["p_gap"])
        else:                      # no history of similar gaps → model view simply fades
            z_new = (1 - stale) * z
        br[f"התאמה למחיר המוקדם (פער {pm['gap_%']:+.1f}%)"] = z_new - z
        z = z_new

    # Technical/MTF layers overlap the model's own features; trust them in
    # proportion to the edge the model actually proved out-of-sample (and less when stale).
    trust = (0.3 + 0.7 * model["skill"]) * (1 - stale)
    adj = s.w_tech * tech * trust
    br["ניתוח טכני"] = adj
    if news and news["items"]:
        # dampen if only a few headlines; news drives pre-market moves → more weight there
        conf = min(1.0, len(news["items"]) / 10)
        n_adj = s.w_news * news["score"] * conf * 2 * (1.5 if session == "premarket" else 1.0)
        adj += n_adj
        br["סנטימנט חדשות"] = n_adj
    if market and s.use_market:
        m_adj = s.w_market * market["score"]
        adj += m_adj
        br["מצב השוק"] = m_adj
    if mtf:
        t_adj = s.w_mtf * mtf["score"] * trust
        adj += t_adj
        br["טווח זמן גבוה"] = t_adj
    if pm and pm["active"]:
        # gap-history part is already in the blend above during pre-market
        sig = pm["sig_flow"] if stale else pm["signal"]
        if sig:
            pm_adj = s.w_pm * sig
            adj += pm_adj
            br["מסחר מוקדם – נפח/VWAP"] = pm_adj
    pat_adj = 0.08 * sum(d for _, d in patterns) * (1 - stale)
    adj += pat_adj
    if pat_adj:
        br["תבניות נרות"] = pat_adj
    p = float(np.clip(_sig(z + adj), 0.03, 0.97))

    if p >= max(s.buy_th + 0.1, 0.7):
        action, lvl = "קנייה חזקה", "strong_buy"
    elif p >= s.buy_th:
        action, lvl = "קנייה", "buy"
    elif p <= min(s.sell_th - 0.1, 0.3):
        action, lvl = "מכירה חזקה", "strong_sell"
    elif p <= s.sell_th:
        action, lvl = "מכירה", "sell"
    else:
        action, lvl = "המתן / ניטרלי", "hold"
    auc_v = model["auc"]
    if lvl != "hold" and (np.isnan(auc_v) or auc_v < s.min_auc):
        # Quality gate: no recommendation from a model with no proven edge on this stock
        action, lvl = f"המתן (מודל לא מוכח, AUC {0 if np.isnan(auc_v) else auc_v:.2f})", "hold"

    last_close = float(x["Close"].iloc[-1])
    price = last_close
    if pm and session in ("premarket", "afterhours", "closed"):
        price = pm["last_price"]          # live extended-hours price
    levels = trade_levels(x, 1 if p >= 0.5 else -1, supports, resistances, price)
    evaluation = None
    if past:
        future = raw_full[len(raw):]
        evaluation = {"entry": last_close, "known": len(future) >= s.horizon}
        if evaluation["known"]:
            exit_price = float(future["Close"].iloc[s.horizon - 1])
            ret = (exit_price / last_close - 1) * 100
            went_up = exit_price > last_close
            if lvl in ("buy", "strong_buy"):
                verdict = "✅ צדקה" if went_up else "❌ טעתה"
            elif lvl in ("sell", "strong_sell"):
                verdict = "✅ צדקה" if not went_up else "❌ טעתה"
            else:
                verdict = "⚪ ללא המלצה"
            # direction call for every stock (even neutral ones): p>=50% means "up"
            lean = abs(p - 0.5) >= 0.03          # the system actually leaned one way
            evaluation.update(exit=exit_price, ret=ret, went_up=went_up, verdict=verdict,
                              direction_ok=((p >= 0.5) == went_up) if lean else None,
                              exit_time=str(future.index[s.horizon - 1]))
    return Result(ticker=ticker, price=price, last_close=last_close, session=session,
                  evaluation=evaluation,
                  premarket=pm, p_up=p, action=action,
                  action_level=lvl, model=model, tech_score=tech, tech_reasons=reasons,
                  news=news, market=market, mtf=mtf, levels=levels, patterns=patterns,
                  frame=x, info=info, breakdown=br)


def day_change(x: pd.DataFrame) -> float:
    """% change vs previous session close (works for intraday and daily frames)."""
    c = x["Close"]
    if isinstance(x.index, pd.DatetimeIndex) and len(set(x.index.date[-500:])) > 1 and \
            (x.index[-1] - x.index[-2]) < pd.Timedelta(hours=20):
        dates = x.index.date
        prev = c[dates < dates[-1]]
        if len(prev):
            return float((c.iloc[-1] / prev.iloc[-1] - 1) * 100)
    return float((c.iloc[-1] / c.iloc[-2] - 1) * 100)


def eval_columns(r) -> dict:
    """Extra columns for a historical report: what really happened next."""
    e = r.evaluation
    if e is None:
        return {}
    if not e["known"]:
        return {"מחיר אחרי": None, "שינוי בפועל %": None, "צדקה?": "⏳ עוד לא ידוע", "כיוון נכון": None}
    return {"מחיר אחרי": round(e["exit"], 2), "שינוי בפועל %": round(e["ret"], 2),
            "צדקה?": e["verdict"],
            "כיוון נכון": "—" if e["direction_ok"] is None else ("✅" if e["direction_ok"] else "❌")}


def scan_row(t: str, s: Settings, market: dict | None) -> dict:
    """Analyse one ticker for the scanner and return a table row."""
    try:
        r = analyze(t, s, market=market, with_info=False)
        x = r.frame
        return {
            "מניה": r.ticker, "מחיר": round(r.price, 2),
            "שינוי יומי %": round(day_change(x), 2),
            "מוקדם/מאוחר %": round(r.premarket["ext_change_%"], 2)
            if r.premarket and r.session in ("premarket", "afterhours") else None,
            "סיכוי עלייה %": round(r.p_up * 100, 1), "המלצה": r.action,
            "AUC מודל": round(r.model["auc"], 3) if not np.isnan(r.model["auc"]) else None,
            "ציון טכני": round(r.tech_score, 2),
            "סנטימנט": round(r.news["score"], 2) if r.news else None,
            "RSI": round(float(x["rsi"].iloc[-1]), 1),
            "נפח יחסי": round(float(x["vol_ratio"].iloc[-1]), 2),
            "סטופ": round(r.levels["stop"], 2), "יעד": round(r.levels["target"], 2),
            **eval_columns(r),
        }
    except Exception as e:
        return {"מניה": t.upper(), "המלצה": f"שגיאה: {str(e)[:60]}"}


def scan_iter(tickers: list[str], s: Settings):
    """Yield (index, row) as soon as each ticker is done – for live/progressive display."""
    market = D.market_regime(s.as_of) if s.use_market else None
    for i, t in enumerate(tickers):
        yield i, scan_row(t, s, market)


def rank(rows: list[dict]) -> pd.DataFrame:
    """Sort scanner rows by signal strength (distance from 50%); errors at the bottom."""
    df = pd.DataFrame(rows)
    if "סיכוי עלייה %" in df:
        df["_k"] = (df["סיכוי עלייה %"] - 50).abs().fillna(-1)
        if "תאריך" in df:   # multi-day historical report: newest day first, strongest first within a day
            order = {d: i for i, d in enumerate(dict.fromkeys(df["תאריך"]))}
            df["_d"] = df["תאריך"].map(order)
            df = df.sort_values(["_d", "_k"], ascending=[False, False]).drop(columns=["_k", "_d"])
        else:
            df = df.sort_values("_k", ascending=False).drop(columns="_k")
    return df


def quick_scan(tickers: list[str], s: Settings, progress=None, on_row=None) -> pd.DataFrame:
    """Analyse many tickers and rank them. `on_row(rows_so_far)` is called after each ticker."""
    rows = []
    n = len(tickers)
    for i, row in scan_iter(tickers, s):
        rows.append(row)
        if progress:
            progress((i + 1) / n, tickers[i + 1] if i + 1 < n else "")
        if on_row:
            on_row(rows)
    return rank(rows)
