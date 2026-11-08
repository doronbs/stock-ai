"""StockAI API server + installable iPhone/Android web app (PWA).

Run locally:   uvicorn server:app --host 0.0.0.0 --port 8000
Cloud:         see README (Render / Hugging Face Spaces)

Optional env vars:
    APP_PASSWORD   – if set, the API requires this access code (entered once in the app settings)
"""
from __future__ import annotations

import math
import os
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.middleware import Middleware
from starlette.middleware.gzip import GZipMiddleware
from starlette.requests import Request
from starlette.responses import FileResponse, JSONResponse
from starlette.routing import Mount, Route
from starlette.staticfiles import StaticFiles

from engine import data as D
from engine.analyzer import Settings, analyze, day_change, quick_scan, rank
from engine.premarket import CLOSE, OPEN, SESSION_HE, market_session
from engine.sentiment import analyze_news
from engine.universe import GROUPS

PWA = Path(__file__).resolve().parent / "pwa"
PASSWORD = os.getenv("APP_PASSWORD", "").strip()


# ───────────────────────── helpers ─────────────────────────
def clean(o):
    """Make anything JSON-safe (numpy, NaN, timestamps)."""
    if isinstance(o, dict):
        return {str(k): clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        f = float(o)
        return None if math.isnan(f) or math.isinf(f) else round(f, 6)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    if isinstance(o, (pd.Timestamp,)):
        return o.isoformat()
    if hasattr(o, "isoformat"):
        return o.isoformat()
    return o


def ok(data, status=200):
    return JSONResponse(clean(data), status_code=status)


def authorized(req: Request) -> bool:
    if not PASSWORD:
        return True
    return (req.headers.get("x-app-key") or req.query_params.get("key") or "") == PASSWORD


def guard(fn):
    async def inner(req: Request):
        if not authorized(req):
            return ok({"error": "קוד גישה שגוי – הזן אותו בהגדרות"}, 401)
        try:
            return await fn(req)
        except Exception as e:  # pragma: no cover
            return ok({"error": str(e)}, 500)
    return inner


def _epoch_local(idx: pd.DatetimeIndex) -> list[int]:
    """Epoch seconds of New-York wall-clock time (lightweight-charts shows UTC)."""
    if idx.tz is not None:
        idx = idx.tz_convert("America/New_York").tz_localize(None)
    return ((idx - pd.Timestamp("1970-01-01")) // pd.Timedelta(seconds=1)).tolist()


def settings_from(q) -> Settings:
    def f(name, default):
        try:
            return float(q.get(name, default))
        except ValueError:
            return default
    b = lambda name: q.get(name, "1") not in ("0", "false", "False")
    interval = q.get("interval", "5m")
    if interval not in D.INTERVAL_PERIOD:
        interval = "5m"
    return Settings(interval=interval, horizon=int(f("horizon", 12)), buy_th=f("buy", 0.6),
                    sell_th=f("sell", 0.4), use_news=b("news"), use_market=b("market"),
                    use_mtf=b("mtf"), use_premarket=b("pm"))


_result_cache: dict = {}


def _analyze_cached(ticker: str, s: Settings):
    key = (ticker, s.interval, s.horizon, s.buy_th, s.sell_th, s.use_news, s.use_market, s.use_mtf,
           s.use_premarket)
    hit = _result_cache.get(key)
    if hit and time.time() - hit[0] < 30:
        return hit[1]
    r = analyze(ticker, s)
    _result_cache[key] = (time.time(), r)
    return r


def result_json(r, s: Settings) -> dict:
    x = r.frame.tail(220 if s.interval in D.INTRADAY else 300)
    t = _epoch_local(x.index)
    candles = [{"time": t[i], "open": x["Open"].iat[i], "high": x["High"].iat[i],
                "low": x["Low"].iat[i], "close": x["Close"].iat[i]} for i in range(len(x))]
    vol = [{"time": t[i], "value": x["Volume"].iat[i],
            "color": "rgba(38,166,154,.45)" if x["Close"].iat[i] >= x["Open"].iat[i] else "rgba(239,83,80,.45)"}
           for i in range(len(x))]

    def line(col):
        return [{"time": t[i], "value": x[col].iat[i]} for i in range(len(x)) if pd.notna(x[col].iat[i])]

    ext = []
    if s.interval in D.INTRADAY:
        try:
            e = D.get_history(r.ticker, s.interval, prepost=True)
            e = e[e.index >= x.index[0]]
            if e.index.tz is not None:
                e = e.tz_convert("America/New_York")
            tt = e.index.time
            e = e[(tt < OPEN) | (tt >= CLOSE)]
            et = _epoch_local(e.index)
            ext = [{"time": et[i], "open": e["Open"].iat[i], "high": e["High"].iat[i],
                    "low": e["Low"].iat[i], "close": e["Close"].iat[i]} for i in range(len(e))]
        except Exception:
            ext = []

    m = r.model
    bt = {k: v for k, v in (m.get("backtest") or {}).items() if k not in ("equity", "buyhold")}
    if m.get("backtest"):
        eq = m["backtest"]["equity"]
        bh = m["backtest"]["buyhold"]
        step = max(1, len(eq) // 120)
        bt["curve"] = [{"t": str(i)[:16], "s": (eq.iat[k] - 1) * 100, "b": (bh.iat[k] - 1) * 100}
                       for k, i in enumerate(eq.index) if k % step == 0]
    rel = None
    if m.get("reliability") is not None:
        rel = [{"bucket": str(b), "actual": row["actual_up_rate"], "n": int(row["n"])}
               for b, row in m["reliability"].iterrows()]
    news = None
    if r.news:
        news = {k: r.news[k] for k in ("score", "positive", "negative", "neutral", "buzz_24h")}
        news["items"] = [{"title": n["title"], "link": n["link"], "source": n["source"],
                          "age_h": n["age_h"], "sentiment": n["sentiment"]} for n in r.news["items"][:25]]
    last = x.iloc[-1]
    return {
        "ticker": r.ticker, "name": r.info.get("shortName") or r.ticker,
        "price": r.price, "last_close": r.last_close, "day_change": day_change(r.frame),
        "session": r.session, "session_he": SESSION_HE.get(r.session, r.session),
        "p_up": r.p_up, "action": r.action, "action_level": r.action_level,
        "levels": r.levels, "tech_score": r.tech_score, "reasons": r.tech_reasons[:12],
        "patterns": r.patterns, "breakdown": r.breakdown,
        "model": {k: m[k] for k in ("p_up", "p_raw", "base_rate", "auc", "accuracy", "skill",
                                    "n_train", "n_oos")},
        "backtest": bt, "reliability": rel, "news": news,
        "market": r.market, "mtf": r.mtf, "premarket": r.premarket, "info": r.info,
        "indicators": {"rsi": last["rsi"], "adx": last["adx"], "atr": last["atr"],
                       "vol_ratio": last["vol_ratio"], "macd_hist": last["hist"]},
        "chart": {"candles": candles, "ext": ext, "volume": vol, "ema9": line("ema9"),
                  "ema21": line("ema21"), "ema50": line("ema50"), "vwap": line("vwap"),
                  "rsi": line("rsi")},
        "updated": time.time(),
    }


# ───────────────────────── endpoints ─────────────────────────
@guard
async def api_analyze(req: Request):
    q = req.query_params
    ticker = (q.get("ticker") or "").upper().strip()
    if not ticker:
        return ok({"error": "חסר סימול"}, 400)
    s = settings_from(q)
    r = await run_in_threadpool(_analyze_cached, ticker, s)
    return ok(result_json(r, s))


_jobs: dict = {}


def _run_scan(job_id: str, tickers: list[str], s: Settings):
    job = _jobs[job_id]

    def prog(p, t):
        job["progress"], job["current"] = p, t

    def partial(rows):
        job["rows"] = rank(rows).replace({np.nan: None}).to_dict("records")
        job["done"] = len(rows)
    try:
        job["current"] = tickers[0]
        quick_scan(tickers, s, progress=prog, on_row=partial)
        job["status"] = "done"
    except Exception as e:
        job["status"], job["error"] = "error", str(e)


@guard
async def api_scan_start(req: Request):
    q = req.query_params
    tickers = [t.strip().upper() for t in (q.get("tickers") or "").replace("\n", ",").split(",") if t.strip()]
    tickers = list(dict.fromkeys(tickers))[:200]
    if not tickers:
        return ok({"error": "רשימה ריקה"}, 400)
    s = settings_from(q)
    s.use_mtf = False  # faster scans
    job_id = uuid.uuid4().hex[:10]
    _jobs[job_id] = {"status": "running", "progress": 0.0, "current": "", "n": len(tickers),
                     "rows": [], "done": 0,
                     "started": time.time()}
    threading.Thread(target=_run_scan, args=(job_id, tickers, s), daemon=True).start()
    # forget old jobs
    for k in [k for k, v in _jobs.items() if time.time() - v["started"] > 3600]:
        _jobs.pop(k, None)
    return ok({"job": job_id})


@guard
async def api_scan_status(req: Request):
    job = _jobs.get(req.query_params.get("job", ""))
    if not job:
        return ok({"error": "סריקה לא נמצאה"}, 404)
    return ok(job)


@guard
async def api_market(req: Request):
    def build():
        idx = D.get_snapshot(tuple(D.INDEXES))
        idx["name"] = idx["ticker"].map(D.INDEXES)
        sec = D.get_snapshot(tuple(D.SECTOR_ETFS))
        sec["name"] = sec["ticker"].map(D.SECTOR_ETFS)
        sess = market_session()
        return {"session": sess, "session_he": SESSION_HE[sess], "regime": D.market_regime(),
                "indexes": idx.to_dict("records"), "sectors": sec.to_dict("records"),
                "breadth": float(sec["above_50d"].mean() * 100) if len(sec) else None}
    return ok(await run_in_threadpool(build))


@guard
async def api_news(req: Request):
    def build():
        t = (req.query_params.get("ticker") or "").upper().strip()
        news = D.get_news(t) if t else D.get_market_news()
        a = analyze_news(news)
        return {"score": a["score"], "positive": a["positive"], "negative": a["negative"],
                "items": [{"title": n["title"], "link": n["link"], "source": n["source"],
                           "age_h": n["age_h"], "sentiment": n["sentiment"]} for n in a["items"]]}
    return ok(await run_in_threadpool(build))


async def api_meta(req: Request):
    sess = market_session()
    return ok({"groups": GROUPS, "session": sess, "session_he": SESSION_HE[sess],
               "auth_required": bool(PASSWORD), "auth_ok": authorized(req)})


async def index(req: Request):
    return FileResponse(PWA / "index.html", headers={"Cache-Control": "no-cache"})


async def sw(req: Request):
    return FileResponse(PWA / "sw.js", media_type="application/javascript",
                        headers={"Cache-Control": "no-cache", "Service-Worker-Allowed": "/"})


async def manifest(req: Request):
    return FileResponse(PWA / "manifest.webmanifest", media_type="application/manifest+json")


app = Starlette(
    routes=[
        Route("/", index),
        Route("/sw.js", sw),
        Route("/manifest.webmanifest", manifest),
        Route("/favicon.ico", lambda r: FileResponse(PWA / "icons" / "icon-192.png", media_type="image/png")),
        Route("/api/meta", api_meta),
        Route("/api/analyze", api_analyze),
        Route("/api/scan", api_scan_start),
        Route("/api/scan/status", api_scan_status),
        Route("/api/market", api_market),
        Route("/api/news", api_news),
        Route("/api/health", lambda r: JSONResponse({"ok": True})),
        Mount("/static", StaticFiles(directory=PWA), name="static"),
    ],
    middleware=[Middleware(GZipMiddleware, minimum_size=1000)],
)
