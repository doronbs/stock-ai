"""Automatic integrity checks for the history/back-testing mechanism.

Run on REAL market data (inside the app) to prove that:
1. No look-ahead: a report "as of" a past moment gives exactly the same prediction even if
   every bar AFTER that moment is changed (we multiply the future prices by 1.5).
2. No hidden leakage in training: when the labels are randomly shuffled, the model's
   out-of-sample AUC must drop to ~0.50 (anything clearly higher means the pipeline leaks).
3. Calibration sanity: on random-walk prices (no information) the measured AUC is ~0.50 and
   the tuner must NOT report a real edge.
"""
from __future__ import annotations

import dataclasses
from contextlib import contextmanager

import numpy as np
import pandas as pd

from . import analyzer as A
from . import data as D
from . import indicators as I
from .model import train_and_predict


@contextmanager
def _patched_history(ticker: str, interval: str, frame: pd.DataFrame):
    """Temporarily serve `frame` as the history of (ticker, interval)."""
    orig = D.get_history

    def fake(t, iv="5m", period=None, prepost=False):
        if t.upper() == ticker.upper() and iv == interval and not prepost:
            return frame
        return orig(t, iv, period, prepost)
    D.get_history = fake
    try:
        yield
    finally:
        D.get_history = orig


def test_no_lookahead(ticker: str, s: A.Settings) -> dict:
    full = D.get_history(ticker, s.interval)
    intraday = s.interval in D.INTRADAY
    cut_i = int(len(full) * 0.75)
    t0 = full.index[cut_i]
    if t0.tzinfo is None:
        t0 = t0.tz_localize("America/New_York")
    step = full.index[cut_i] - full.index[cut_i - 1] if intraday else pd.Timedelta(0)
    as_of = (t0 + step) if intraday else pd.Timestamp(f"{t0.date()} 16:30", tz="America/New_York")
    s_past = dataclasses.replace(s, as_of=as_of, use_news=False)
    r1 = A.analyze(ticker, s_past, with_info=False)
    known = len(D.truncate(full, as_of, daily=not intraday))
    changed = full.copy()
    changed.iloc[known:, :4] = changed.iloc[known:, :4] * 1.5      # rewrite the future
    with _patched_history(ticker, s.interval, changed):
        r2 = A.analyze(ticker, s_past, with_info=False)
    same = abs(r1.p_up - r2.p_up) < 1e-12 and abs(r1.model["auc"] - r2.model["auc"]) < 1e-12 \
        if not np.isnan(r1.model["auc"]) else abs(r1.p_up - r2.p_up) < 1e-12
    return {"name": "אין הצצה לעתיד", "passed": bool(same),
            "detail": f"דוח ל-{as_of:%d/%m %H:%M}: הסתברות {r1.p_up * 100:.2f}% לפני ו-"
                      f"{r2.p_up * 100:.2f}% אחרי ששיניתי את כל הנתונים העתידיים ({len(full) - known} נרות)"}


def test_shuffled_labels(ticker: str, s: A.Settings) -> dict:
    intraday = s.interval in D.INTRADAY
    x = I.add_all(D.get_history(ticker, s.interval), intraday)
    real = train_and_predict(x, intraday, s.horizon, noise=s.noise)
    fake = train_and_predict(x, intraday, s.horizon, noise=s.noise, shuffle_labels=True)
    a = fake["auc"]
    ok = (not np.isnan(a)) and abs(a - 0.5) < 0.04
    return {"name": "תוויות מעורבבות → אין יתרון", "passed": bool(ok),
            "detail": f"AUC אמיתי {real['auc']:.3f} · AUC עם תוויות אקראיות {a:.3f} (צריך להיות ≈0.50)"}


def test_random_walk(s: A.Settings, seed: int = 11) -> dict:
    """Pure noise prices: a correct pipeline must not find an edge."""
    rng = np.random.default_rng(seed)
    n = 3000
    c = 100 * np.exp(np.cumsum(rng.normal(0, 0.002, n)))
    idx = pd.date_range("2026-01-05 09:30", periods=n, freq="5min", tz="America/New_York")
    o = np.r_[c[0], c[:-1]]
    df = pd.DataFrame({"Open": o, "High": np.maximum(o, c) * 1.0005, "Low": np.minimum(o, c) * 0.9995,
                       "Close": c, "Volume": rng.integers(1e4, 1e5, n).astype(float)}, index=idx)
    x = I.add_all(df, True)
    m = train_and_predict(x, True, s.horizon, noise=s.noise)
    ok = abs(m["auc"] - 0.5) < 0.04 and abs(m["p_up"] - m["base_rate"]) < 0.05
    return {"name": "מחירים אקראיים → אין יתרון מדומה", "passed": bool(ok),
            "detail": f"AUC {m['auc']:.3f}, הסתברות {m['p_up'] * 100:.1f}% מול בסיס {m['base_rate'] * 100:.1f}%"}


def run_all(ticker: str, s: A.Settings) -> list[dict]:
    out = []
    for fn, args in ((test_no_lookahead, (ticker, s)), (test_shuffled_labels, (ticker, s)),
                     (test_random_walk, (s,))):
        try:
            out.append(fn(*args))
        except Exception as e:  # pragma: no cover
            out.append({"name": fn.__name__, "passed": False, "detail": f"שגיאה: {e}"})
    return out
