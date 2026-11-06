"""Technical indicators – pure pandas/numpy, no TA-lib needed.

All functions take a DataFrame with columns: Open, High, Low, Close, Volume
(DatetimeIndex) and return Series / DataFrames aligned to the same index.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def ema(s: pd.Series, n: int) -> pd.Series:
    return s.ewm(span=n, adjust=False, min_periods=n).mean()


def sma(s: pd.Series, n: int) -> pd.Series:
    return s.rolling(n, min_periods=n).mean()


def rsi(close: pd.Series, n: int = 14) -> pd.Series:
    d = close.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    rs = up / dn.replace(0, np.nan)
    out = 100 - 100 / (1 + rs)
    return out.fillna(100.0).where(dn.notna())


def macd(close: pd.Series, fast=12, slow=26, signal=9) -> pd.DataFrame:
    m = ema(close, fast) - ema(close, slow)
    s = m.ewm(span=signal, adjust=False, min_periods=signal).mean()
    return pd.DataFrame({"macd": m, "signal": s, "hist": m - s})


def bollinger(close: pd.Series, n=20, k=2.0) -> pd.DataFrame:
    mid = sma(close, n)
    sd = close.rolling(n, min_periods=n).std(ddof=0)
    up, lo = mid + k * sd, mid - k * sd
    width = (up - lo) / mid
    pctb = (close - lo) / (up - lo).replace(0, np.nan)
    return pd.DataFrame({"bb_mid": mid, "bb_up": up, "bb_lo": lo,
                         "bb_width": width, "bb_pctb": pctb})


def true_range(df: pd.DataFrame) -> pd.Series:
    pc = df["Close"].shift()
    return pd.concat([df["High"] - df["Low"],
                      (df["High"] - pc).abs(),
                      (df["Low"] - pc).abs()], axis=1).max(axis=1)


def atr(df: pd.DataFrame, n=14) -> pd.Series:
    return true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()


def stochastic(df: pd.DataFrame, n=14, d=3) -> pd.DataFrame:
    ll = df["Low"].rolling(n, min_periods=n).min()
    hh = df["High"].rolling(n, min_periods=n).max()
    k = 100 * (df["Close"] - ll) / (hh - ll).replace(0, np.nan)
    return pd.DataFrame({"stoch_k": k, "stoch_d": k.rolling(d).mean()})


def adx(df: pd.DataFrame, n=14) -> pd.DataFrame:
    up = df["High"].diff()
    dn = -df["Low"].diff()
    plus_dm = np.where((up > dn) & (up > 0), up, 0.0)
    minus_dm = np.where((dn > up) & (dn > 0), dn, 0.0)
    tr = true_range(df).ewm(alpha=1 / n, adjust=False, min_periods=n).mean()
    pdi = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    mdi = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / n, adjust=False, min_periods=n).mean() / tr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    return pd.DataFrame({"adx": dx.ewm(alpha=1 / n, adjust=False, min_periods=n).mean(),
                         "plus_di": pdi, "minus_di": mdi})


def obv(df: pd.DataFrame) -> pd.Series:
    sign = np.sign(df["Close"].diff()).fillna(0)
    return (sign * df["Volume"]).cumsum()


def mfi(df: pd.DataFrame, n=14) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    flow = tp * df["Volume"]
    pos = flow.where(tp > tp.shift(), 0.0).rolling(n).sum()
    neg = flow.where(tp < tp.shift(), 0.0).rolling(n).sum()
    return 100 - 100 / (1 + pos / neg.replace(0, np.nan))


def cci(df: pd.DataFrame, n=20) -> pd.Series:
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    m = tp.rolling(n).mean()
    md = tp.rolling(n).apply(lambda x: np.mean(np.abs(x - x.mean())), raw=True)
    return (tp - m) / (0.015 * md.replace(0, np.nan))


def williams_r(df: pd.DataFrame, n=14) -> pd.Series:
    hh = df["High"].rolling(n).max()
    ll = df["Low"].rolling(n).min()
    return -100 * (hh - df["Close"]) / (hh - ll).replace(0, np.nan)


def vwap_session(df: pd.DataFrame) -> pd.Series:
    """VWAP that resets every trading day (for intraday data)."""
    tp = (df["High"] + df["Low"] + df["Close"]) / 3
    day = df.index.normalize() if isinstance(df.index, pd.DatetimeIndex) else 0
    pv = (tp * df["Volume"]).groupby(day).cumsum()
    vv = df["Volume"].groupby(day).cumsum()
    return pv / vv.replace(0, np.nan)


def supertrend(df: pd.DataFrame, n=10, mult=3.0) -> pd.Series:
    """Returns +1 for bullish trend, -1 for bearish."""
    a = atr(df, n)
    hl2 = (df["High"] + df["Low"]) / 2
    up = (hl2 - mult * a).to_numpy()
    dn = (hl2 + mult * a).to_numpy()
    close = df["Close"].to_numpy()
    trend = np.ones(len(df))
    fu, fd = up.copy(), dn.copy()
    for i in range(1, len(df)):
        if np.isnan(up[i]) or np.isnan(fu[i - 1]):
            continue
        fu[i] = max(up[i], fu[i - 1]) if close[i - 1] > fu[i - 1] else up[i]
        fd[i] = min(dn[i], fd[i - 1]) if close[i - 1] < fd[i - 1] else dn[i]
        if close[i] > fd[i - 1]:
            trend[i] = 1
        elif close[i] < fu[i - 1]:
            trend[i] = -1
        else:
            trend[i] = trend[i - 1]
    return pd.Series(trend, index=df.index)


def support_resistance(df: pd.DataFrame, lookback=120, window=5, max_levels=4):
    """Pivot-based S/R levels, clustered. Returns (supports, resistances) lists."""
    d = df.tail(lookback)
    if len(d) < window * 2 + 1:
        return [], []
    hi, lo = d["High"], d["Low"]
    piv_hi = hi[(hi == hi.rolling(2 * window + 1, center=True).max())].dropna()
    piv_lo = lo[(lo == lo.rolling(2 * window + 1, center=True).min())].dropna()
    price = d["Close"].iloc[-1]
    tol = (atr(d).iloc[-1] if len(d) > 15 else price * 0.01) or price * 0.01

    def cluster(levels):
        levels = sorted(levels)
        out = []
        for lv in levels:
            if out and abs(lv - out[-1][0]) < tol:
                c = out[-1]
                c[0] = (c[0] * c[1] + lv) / (c[1] + 1)
                c[1] += 1
            else:
                out.append([lv, 1])
        return out

    allv = cluster(list(piv_hi.values) + list(piv_lo.values))
    sup = sorted([c for c in allv if c[0] < price], key=lambda c: -c[0])[:max_levels]
    res = sorted([c for c in allv if c[0] >= price], key=lambda c: c[0])[:max_levels]
    return [round(c[0], 2) for c in sup], [round(c[0], 2) for c in res]


def candle_patterns(df: pd.DataFrame) -> list[tuple[str, int]]:
    """Detect a few reliable patterns on the last bars. Returns [(name_he, direction)]."""
    out = []
    if len(df) < 3:
        return out
    o, h, l, c = (df[x].to_numpy() for x in ["Open", "High", "Low", "Close"])
    body = abs(c[-1] - o[-1])
    rng = h[-1] - l[-1] or 1e-9
    upper = h[-1] - max(c[-1], o[-1])
    lower = min(c[-1], o[-1]) - l[-1]
    prev_bear = c[-2] < o[-2]
    prev_bull = c[-2] > o[-2]
    if prev_bear and c[-1] > o[-1] and c[-1] >= o[-2] and o[-1] <= c[-2]:
        out.append(("בליעה שורית (Bullish Engulfing)", +1))
    if prev_bull and c[-1] < o[-1] and o[-1] >= c[-2] and c[-1] <= o[-2]:
        out.append(("בליעה דובית (Bearish Engulfing)", -1))
    if lower > 2 * body and upper < body and body / rng < 0.35:
        out.append(("פטיש (Hammer)", +1))
    if upper > 2 * body and lower < body and body / rng < 0.35:
        out.append(("כוכב נופל (Shooting Star)", -1))
    if body / rng < 0.08:
        out.append(("דוג'י – חוסר הכרעה", 0))
    if (c[-3] < o[-3] and abs(c[-2] - o[-2]) < abs(c[-3] - o[-3]) * 0.3
            and c[-1] > o[-1] and c[-1] > (o[-3] + c[-3]) / 2):
        out.append(("כוכב בוקר (Morning Star)", +1))
    if (c[-3] > o[-3] and abs(c[-2] - o[-2]) < abs(c[-3] - o[-3]) * 0.3
            and c[-1] < o[-1] and c[-1] < (o[-3] + c[-3]) / 2):
        out.append(("כוכב ערב (Evening Star)", -1))
    return out


def add_all(df: pd.DataFrame, intraday: bool) -> pd.DataFrame:
    """Return a copy of df with every indicator column added."""
    x = df.copy()
    c = x["Close"]
    for n in (9, 21, 50, 200):
        x[f"ema{n}"] = ema(c, n)
    x["sma20"] = sma(c, 20)
    x["rsi"] = rsi(c)
    x = x.join(macd(c)).join(bollinger(c)).join(stochastic(x)).join(adx(x))
    x["atr"] = atr(x)
    x["obv"] = obv(x)
    x["mfi"] = mfi(x)
    x["cci"] = cci(x)
    x["willr"] = williams_r(x)
    x["supertrend"] = supertrend(x)
    x["vol_ratio"] = x["Volume"] / x["Volume"].rolling(20).mean().replace(0, np.nan)
    if intraday:
        x["vwap"] = vwap_session(x)
    else:
        x["vwap"] = (c * x["Volume"]).rolling(20).sum() / x["Volume"].rolling(20).sum().replace(0, np.nan)
    return x
