"""Automatic tuning WITHOUT fooling ourselves.

The classic trap: try many settings on the same history and keep the one that looks best –
it will look great on that history and fail tomorrow (overfitting). So:

  timeline per stock:  [ train ......... | SELECTION period | TEST period ]
                         (model fits)      choose settings    report only – never used to choose

1. For every model variant (label noise band × with/without market features) we produce
   walk-forward predictions: each prediction is made by a model trained ONLY on earlier bars
   (with a purge gap), and it is made for every bar – including flat ones.
2. Probabilities are calibrated per stock on the selection period only.
3. Every combination of variant × buy threshold × AUC gate is scored on the SELECTION period:
   success rate = % of buy/sell signals whose direction was right after `horizon` bars,
   with a minimum number of signals so a lucky handful can't win.
4. The winner is then measured ONCE on the TEST period. That number is the honest one.
"""
from __future__ import annotations

import itertools
import time

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

from . import data as D
from . import indicators as I
from .model import _Ensemble, _logit, build_features, make_labels

VARIANTS = [{"noise": n, "use_mkt": m} for n in (0.0, 0.15, 0.3) for m in (True, False)]
BUY_TH = [0.55, 0.58, 0.60, 0.62, 0.65, 0.70]
MIN_AUC = [0.50, 0.52, 0.54, 0.56, 0.58]


def _oos_all_bars(X, y, valid_mask, horizon, n_splits=6, min_train=300):
    """Walk-forward predictions for ALL valid bars (flat bars included), trained on labelled
    bars that ended before each test block (purge gap = horizon)."""
    n = len(X)
    oos = np.full(n, np.nan)
    edges = np.linspace(min_train, n, n_splits + 1).astype(int)
    yv = y.to_numpy()
    for i in range(n_splits):
        te0, te1 = edges[i], edges[i + 1]
        tr_end = te0 - horizon
        tr = np.arange(tr_end)
        tr = tr[~np.isnan(yv[tr]) & valid_mask[tr]]
        if len(tr) < 150 or len(np.unique(yv[tr])) < 2:
            continue
        m = _Ensemble().fit(X.iloc[tr], yv[tr])
        te = np.arange(te0, te1)
        te = te[valid_mask[te]]
        if len(te):
            oos[te] = m.predict_proba(X.iloc[te])
    return oos


def stock_predictions(ticker: str, interval: str, horizon: int, variant: dict, as_of=None):
    intraday = interval in D.INTRADAY
    raw = D.truncate(D.get_history(ticker, interval), as_of, daily=not intraday)
    x = I.add_all(raw, intraday)
    mkt = None
    if variant["use_mkt"] and ticker.upper() != "SPY":
        try:
            mkt = D.truncate(D.get_history("SPY", interval), as_of, daily=not intraday)["Close"]
        except Exception:
            mkt = None
    f = build_features(x, intraday, mkt)
    f = f.loc[:, f.notna().mean() > 0.5]
    y, fwd = make_labels(x["Close"], horizon, variant["noise"])
    valid = f.notna().all(axis=1).to_numpy() & fwd.notna().to_numpy()
    Xf = f.fillna(0)
    p = _oos_all_bars(Xf, y, valid, horizon)
    return pd.DataFrame({"p": p, "fwd": fwd.to_numpy(), "y": y.to_numpy()}, index=x.index)


def _calibrate(sel: pd.DataFrame, test: pd.DataFrame):
    """Platt-calibrate on the selection period, shrink by proven skill (same idea as the app)."""
    s = sel.dropna(subset=["p", "y"])
    if len(s) < 100 or s["y"].nunique() < 2:
        return None
    auc = roc_auc_score(s["y"], s["p"])
    cal = LogisticRegression(C=10).fit(_logit(s["p"].to_numpy()).reshape(-1, 1), s["y"])
    a, b = cal.coef_[0][0], cal.intercept_[0]
    base = float(s["y"].mean())
    skill = float(np.clip((auc - 0.5) / 0.10, 0, 1))

    def f(p):
        pc = 1 / (1 + np.exp(-(a * _logit(np.asarray(p, float)) + b)))
        return base + (pc - base) * skill
    return f, auc


def _score(df: pd.DataFrame, buy_th: float, horizon: int):
    """Success rate of signals. Non-overlapping: after a signal we skip `horizon` bars."""
    p, fwd = df["pc"].to_numpy(), df["fwd"].to_numpy()
    hits = n = 0
    i = 0
    while i < len(p):
        if np.isnan(p[i]) or np.isnan(fwd[i]):
            i += 1
            continue
        if p[i] >= buy_th:
            n += 1; hits += fwd[i] > 0; i += horizon; continue
        if p[i] <= 1 - buy_th:
            n += 1; hits += fwd[i] < 0; i += horizon; continue
        i += 1
    return hits, n


def wilson_low(hits: int, n: int, z: float = 1.645) -> float:
    """Lower edge of a 90% confidence interval for a success rate."""
    if n == 0:
        return float("nan")
    p = hits / n
    d = 1 + z * z / n
    c = p + z * z / (2 * n)
    m = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float((c - m) / d)


def tune(tickers: list[str], interval: str, horizon: int, as_of=None, progress=None,
         sel_frac: float = 0.55, min_signals: int = 15) -> dict:
    """Returns the chosen settings and an honest test-period score."""
    t0 = time.time()
    # per stock, per variant: calibrated OOS predictions split into selection / test
    split = {}     # (variant_idx, ticker) -> (sel_df, test_df, auc_sel)
    total = len(tickers) * len(VARIANTS)
    k = 0
    for t in tickers:                      # stock outer → its data is fetched once (cache)
        for vi, v in enumerate(VARIANTS):
            k += 1
            if progress:
                progress(k / total, f"{t} · וריאנט {vi + 1}/{len(VARIANTS)}")
            try:
                df = stock_predictions(t, interval, horizon, v, as_of).dropna(subset=["p"])
            except Exception:
                continue
            if len(df) < 200:
                continue
            cut = int(len(df) * sel_frac)
            sel, test = df.iloc[:cut].copy(), df.iloc[cut:].copy()
            c = _calibrate(sel, test)
            if c is None:
                continue
            f, auc = c
            sel["pc"], test["pc"] = f(sel["p"]), f(test["p"])
            split[(vi, t)] = (sel, test, auc)

    results = []
    for vi, bt, ma in itertools.product(range(len(VARIANTS)), BUY_TH, MIN_AUC):
        h = n = 0
        for (v2, t), (sel, _test, auc) in split.items():
            if v2 != vi or auc < ma:
                continue
            a, b = _score(sel, bt, horizon)
            h += a; n += b
        if n:
            results.append({"variant": vi, "buy_th": bt, "min_auc": ma, "sel_hits": h, "sel_n": n,
                            "sel_rate": h / n})
    if not results:
        return {"ok": False, "error": "אין מספיק נתונים לכיול", "seconds": time.time() - t0}

    eligible = [r for r in results if r["sel_n"] >= min_signals] or results
    # Rank by a conservative estimate (lower edge of the success rate), not by the raw rate –
    # so 9/10 doesn't beat 70/100.
    for r in eligible:
        n_, p_ = r["sel_n"], r["sel_rate"]
        r["sel_low"] = p_ - 1.0 * np.sqrt(p_ * (1 - p_) / n_)
    eligible.sort(key=lambda r: r["sel_low"], reverse=True)

    def test_of(r):
        h = n = 0
        stocks = 0
        for (v2, t), (sel, test, auc) in split.items():
            if v2 != r["variant"] or auc < r["min_auc"]:
                continue
            stocks += 1
            a, b = _score(test, r["buy_th"], horizon)
            h += a; n += b
        return h, n, stocks

    top = []
    for r in eligible[:5]:
        th, tn, stocks = test_of(r)
        top.append({**r, **VARIANTS[r["variant"]], "test_hits": th, "test_n": tn,
                    "test_rate": th / tn if tn else np.nan, "stocks": stocks})
    best = top[0]
    # Honest verdict on the unseen TEST period: Wilson 90% lower bound of the success rate must
    # beat 50% (a coin) – otherwise the "success" is indistinguishable from luck.
    for r in top:
        r["test_low"] = wilson_low(r["test_hits"], r["test_n"])
        r["proven"] = bool(r["test_n"] >= 20 and r["test_low"] > 0.5)
    # Baseline on the same test bars: success of "always guess up" (the market's own drift)
    up = tot = 0
    for (v2, t), (sel, test, auc) in split.items():
        if v2 == best["variant"]:
            f_ = test["fwd"].dropna()
            up += int((f_ > 0).sum()); tot += len(f_)
    return {"ok": True, "best": best, "top": top, "baseline_up": up / tot if tot else np.nan,
            "n_stocks": len({t for (_, t) in split}), "seconds": time.time() - t0,
            "settings": {"noise": best["noise"], "use_mkt": bool(best["use_mkt"]),
                         "buy_th": best["buy_th"], "sell_th": round(1 - best["buy_th"], 2),
                         "min_auc": best["min_auc"]}}
