"""Market data + news fetching (free sources: Yahoo Finance via yfinance, Google News RSS)."""
from __future__ import annotations

import time
import xml.etree.ElementTree as ET
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from functools import wraps
from urllib.parse import quote_plus

import pandas as pd
import requests

try:
    import yfinance as yf
except ImportError:  # pragma: no cover
    yf = None

# Max history Yahoo allows per interval
INTERVAL_PERIOD = {
    "1m": "7d", "2m": "60d", "5m": "60d", "15m": "60d", "30m": "60d",
    "60m": "730d", "90m": "60d", "1d": "5y", "1wk": "10y",
}
INTRADAY = {"1m", "2m", "5m", "15m", "30m", "60m", "90m"}

SECTOR_ETFS = {
    "XLK": "טכנולוגיה", "XLF": "פיננסים", "XLV": "בריאות", "XLE": "אנרגיה",
    "XLY": "צריכה מחזורית", "XLP": "צריכה בסיסית", "XLI": "תעשייה",
    "XLB": "חומרים", "XLU": "תשתיות", "XLRE": "נדל\"ן", "XLC": "תקשורת",
    "SMH": "שבבים",
}
INDEXES = {"SPY": "S&P 500", "QQQ": "נאסד\"ק 100", "DIA": "דאו ג'ונס",
           "IWM": "ראסל 2000", "^VIX": "מדד הפחד VIX"}

UA = {"User-Agent": "Mozilla/5.0 (StockAI research tool)"}

_cache: dict = {}


def ttl_cache(seconds: int):
    """Tiny in-process cache so the UI/scanner doesn't hammer Yahoo."""
    def deco(fn):
        @wraps(fn)
        def inner(*a, **kw):
            key = (fn.__name__, a, tuple(sorted(kw.items())))
            hit = _cache.get(key)
            if hit and time.time() - hit[0] < seconds:
                return hit[1]
            val = fn(*a, **kw)
            _cache[key] = (time.time(), val)
            return val
        return inner
    return deco


def _need_yf():
    if yf is None:
        raise RuntimeError("yfinance לא מותקן. הרץ: pip install -r requirements.txt")


@ttl_cache(45)
def get_history(ticker: str, interval: str = "5m", period: str | None = None,
                prepost: bool = False) -> pd.DataFrame:
    _need_yf()
    period = period or INTERVAL_PERIOD.get(interval, "60d")
    df = yf.Ticker(ticker).history(period=period, interval=interval,
                                   prepost=prepost, auto_adjust=True)
    if df is None or df.empty:
        raise ValueError(f"לא התקבלו נתונים עבור {ticker} ({interval})")
    df = df[["Open", "High", "Low", "Close", "Volume"]].dropna()
    df = df[df["Volume"] >= 0]
    if df.index.tz is not None and interval in INTRADAY:
        df.index = df.index.tz_convert("America/New_York")
    return df


@ttl_cache(600)
def get_info(ticker: str) -> dict:
    _need_yf()
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception:
        info = {}
    keys = ["shortName", "longName", "sector", "industry", "marketCap", "trailingPE",
            "forwardPE", "pegRatio", "priceToBook", "beta", "fiftyTwoWeekHigh",
            "fiftyTwoWeekLow", "targetMeanPrice", "targetHighPrice", "targetLowPrice",
            "recommendationKey", "recommendationMean", "numberOfAnalystOpinions",
            "shortPercentOfFloat", "averageVolume", "dividendYield", "revenueGrowth",
            "earningsGrowth", "profitMargins", "currentPrice", "regularMarketPrice",
            "previousClose"]
    out = {k: info.get(k) for k in keys}
    try:
        cal = yf.Ticker(ticker).calendar
        if isinstance(cal, dict) and cal.get("Earnings Date"):
            out["earningsDate"] = str(cal["Earnings Date"][0])
    except Exception:
        pass
    return out


def _parse_yf_news(items) -> list[dict]:
    out = []
    for it in items or []:
        c = it.get("content", it)  # new (2025+) and old formats
        title = c.get("title")
        if not title:
            continue
        ts = None
        if c.get("pubDate"):
            try:
                ts = datetime.fromisoformat(c["pubDate"].replace("Z", "+00:00"))
            except Exception:
                ts = None
        elif it.get("providerPublishTime"):
            ts = datetime.fromtimestamp(it["providerPublishTime"], tz=timezone.utc)
        link = (c.get("canonicalUrl") or {}).get("url") if isinstance(c.get("canonicalUrl"), dict) else c.get("link")
        prov = (c.get("provider") or {}).get("displayName") if isinstance(c.get("provider"), dict) else c.get("publisher")
        out.append({"title": title, "summary": c.get("summary") or "", "time": ts,
                    "source": prov or "Yahoo", "link": link or ""})
    return out


def _google_news(query: str, limit: int = 30) -> list[dict]:
    url = (f"https://news.google.com/rss/search?q={quote_plus(query)}"
           f"+when:3d&hl=en-US&gl=US&ceid=US:en")
    try:
        r = requests.get(url, headers=UA, timeout=10)
        r.raise_for_status()
        root = ET.fromstring(r.content)
    except Exception:
        return []
    out = []
    for item in root.iter("item"):
        title = (item.findtext("title") or "").strip()
        src = item.findtext("source") or ""
        if src and title.endswith(" - " + src):
            title = title[: -len(src) - 3]
        try:
            ts = parsedate_to_datetime(item.findtext("pubDate"))
        except Exception:
            ts = None
        out.append({"title": title, "summary": "", "time": ts, "source": src or "Google News",
                    "link": item.findtext("link") or ""})
        if len(out) >= limit:
            break
    return out


@ttl_cache(300)
def get_news(ticker: str, company: str | None = None, limit: int = 40) -> list[dict]:
    news = []
    if yf is not None:
        try:
            news += _parse_yf_news(yf.Ticker(ticker).news)
        except Exception:
            pass
    q = f'"{ticker}" stock' if not company else f'{company} OR "{ticker}" stock'
    news += _google_news(q, limit)
    # dedupe by normalized title
    seen, uniq = set(), []
    for n in news:
        k = n["title"].lower()[:70]
        if k in seen:
            continue
        seen.add(k)
        uniq.append(n)
    uniq.sort(key=lambda n: n["time"] or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
    return uniq[:limit]


@ttl_cache(300)
def get_market_news(limit: int = 30) -> list[dict]:
    news = _google_news("stock market today Wall Street", limit)
    news += _google_news("Federal Reserve inflation stocks", 15)
    news.sort(key=lambda n: n["time"] or datetime(1970, 1, 1, tzinfo=timezone.utc), reverse=True)
    return news[:limit]


@ttl_cache(120)
def get_snapshot(tickers: tuple[str, ...]) -> pd.DataFrame:
    """Daily change % / 5-day / 1-month performance for a list of tickers."""
    _need_yf()
    data = yf.download(list(tickers), period="3mo", interval="1d", auto_adjust=True,
                       progress=False, group_by="ticker", threads=True)
    rows = []
    for t in tickers:
        try:
            c = data[t]["Close"].dropna() if len(tickers) > 1 else data["Close"].dropna()
            if isinstance(c, pd.DataFrame):
                c = c.iloc[:, 0]
            rows.append({"ticker": t, "price": c.iloc[-1],
                         "day_%": (c.iloc[-1] / c.iloc[-2] - 1) * 100,
                         "week_%": (c.iloc[-1] / c.iloc[-6] - 1) * 100,
                         "month_%": (c.iloc[-1] / c.iloc[-22] - 1) * 100,
                         "above_50d": bool(c.iloc[-1] > c.tail(50).mean())})
        except Exception:
            continue
    return pd.DataFrame(rows)


def market_regime() -> dict:
    """Overall US market state – used to tilt individual-stock probabilities."""
    out = {"score": 0.0, "notes": []}
    try:
        spy = get_history("SPY", "1d", "1y")
        c = spy["Close"]
        e50, e200 = c.ewm(span=50).mean().iloc[-1], c.ewm(span=200).mean().iloc[-1]
        s = 0.0
        if c.iloc[-1] > e50:
            s += 0.5; out["notes"].append("S&P 500 מעל ממוצע 50 יום")
        else:
            s -= 0.5; out["notes"].append("S&P 500 מתחת לממוצע 50 יום")
        if e50 > e200:
            s += 0.5; out["notes"].append("מגמה ראשית עולה (50>200)")
        else:
            s -= 0.5; out["notes"].append("מגמה ראשית יורדת (50<200)")
        out["spy_day_%"] = (c.iloc[-1] / c.iloc[-2] - 1) * 100
        out["score"] = s
    except Exception as e:
        out["notes"].append(f"לא ניתן לטעון SPY: {e}")
    try:
        vix = get_history("^VIX", "1d", "3mo")["Close"]
        v = float(vix.iloc[-1])
        out["vix"] = v
        if v > 28:
            out["score"] -= 0.5; out["notes"].append(f"VIX גבוה ({v:.1f}) – פחד בשוק")
        elif v < 16:
            out["score"] += 0.25; out["notes"].append(f"VIX נמוך ({v:.1f}) – שוק רגוע")
        else:
            out["notes"].append(f"VIX ניטרלי ({v:.1f})")
    except Exception:
        pass
    out["score"] = max(-1.0, min(1.0, out["score"]))
    return out


def is_market_open(now: datetime | None = None) -> bool:
    now = now or datetime.now(timezone.utc)
    ny = pd.Timestamp(now).tz_convert("America/New_York")
    if ny.weekday() >= 5:
        return False
    t = ny.hour * 60 + ny.minute
    return 9 * 60 + 30 <= t < 16 * 60
