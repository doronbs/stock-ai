"""A/B test of method variants on many stocks and two separate periods (blind walk).

    python -m engine.abtest --variant V1 --tickers NVDA,AAPL,... --days 60 --out reports

Period B (older half) is used to CHOOSE a variant, period A (newer half) to CONFIRM it,
so a change that only fits one month or one stock is exposed.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

from .analyzer import Settings
from .walk import walk

VARIANTS = {
    "V0": dict(base_mode="full", fund_scale=1.0),      # current
    "V1": dict(base_mode="full", fund_scale=0.0),      # no fundamentals on a 1-day horizon
    "V2": dict(base_mode="recent", fund_scale=0.0),    # + recent-regime fallback
    "V3": dict(base_mode="neutral", fund_scale=0.0),   # + neutral 50% fallback
    "V4": dict(base_mode="recent", fund_scale=1.0),    # recent fallback, keep fundamentals
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--tickers", required=True)
    ap.add_argument("--days", type=int, default=60)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    base = dataclasses.replace(Settings(interval="1d", horizon=1, use_news=False, use_premarket=False),
                               **VARIANTS[a.variant])
    out = {"variant": a.variant, "settings": VARIANTS[a.variant], "rows": []}
    for t in a.tickers.split(","):
        rep = walk(t.strip(), a.days, base, progress=lambda p, d, t=t: print(f"{a.variant} {t} {p * 100:4.0f}% {d}", flush=True))
        for r in rep["rows"]:
            if "error" in r:
                continue
            out["rows"].append({"ticker": rep["ticker"], "date": r["date"], "p": r["p_up"],
                                "ret": r["ret_pct"], "level": r["level"], "auc": r["auc"]})
    Path(a.out).mkdir(exist_ok=True)
    Path(a.out, f"ab_{a.variant}.json").write_text(json.dumps(out, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
