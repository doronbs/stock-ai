"""Day-by-day blind walk: for every trading day in a period, run the FULL 6-layer analysis as of
that day's close using only information public at that moment, predict the next day, and then
compare with what really happened.

CLI (used by the GitHub Actions report workflow):
    python -m engine.walk NVDA --days 21 --out reports
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from pathlib import Path

import numpy as np
import pandas as pd

from . import data as D
from .analyzer import Settings, analyze


def trading_days(n_days: int, horizon: int, end: pd.Timestamp | None = None) -> list[pd.Timestamp]:
    """The last n trading days whose outcome (`horizon` days later) is already known."""
    spy = D.get_history("SPY", "1d", "1y")
    idx = spy.index.tz_convert("America/New_York") if spy.index.tz is not None else spy.index
    days = [d.date() for d in idx]
    if end is not None:
        days = [d for d in days if d <= end.date()]
    usable = days[:-horizon] if horizon > 0 else days
    return [pd.Timestamp(f"{d} 16:05", tz="America/New_York") for d in usable[-n_days:]]


def walk(ticker: str, n_days: int = 21, base: Settings | None = None, progress=None,
         end: pd.Timestamp | None = None) -> dict:
    base = base or Settings(interval="1d", horizon=1, use_news=False, use_premarket=False)
    base = dataclasses.replace(base, interval="1d", use_news=False, use_premarket=False)
    days = trading_days(n_days, base.horizon, end)
    rows = []
    t0 = time.time()
    for i, as_of in enumerate(days):
        if progress:
            progress(i / len(days), f"{as_of:%d/%m}")
        s = dataclasses.replace(base, as_of=as_of)
        try:
            r = analyze(ticker, s, with_info=False)
        except Exception as e:
            rows.append({"date": f"{as_of:%Y-%m-%d}", "error": str(e)[:120]})
            continue
        e = r.evaluation or {}
        row = {"date": f"{as_of:%Y-%m-%d}", "close": r.last_close, "p_up": r.p_up,
               "action": r.action, "level": r.action_level, "auc": r.model["auc"],
               "next_close": e.get("exit"), "ret_pct": e.get("ret"),
               "verdict": e.get("verdict"), "direction_ok": e.get("direction_ok"),
               "stop": r.levels["stop"], "target": r.levels["target"]}
        for L in r.layers:
            row["L" + L["name"].split(".")[0]] = round(L["score"], 3)
        row["layer_notes"] = {L["name"]: L["notes"][:4] for L in r.layers}
        rows.append(row)
    if progress:
        progress(1.0, "")
    return {"ticker": ticker.upper(), "days": len(days), "horizon": base.horizon,
            "generated": pd.Timestamp.now(tz="Asia/Jerusalem").strftime("%d/%m/%Y %H:%M"),
            "seconds": round(time.time() - t0, 1), "rows": rows, "summary": summarize(rows, base)}


def summarize(rows: list[dict], s: Settings) -> dict:
    ok = [r for r in rows if r.get("ret_pct") is not None]
    sig = [r for r in ok if r["level"] in ("buy", "strong_buy", "sell", "strong_sell")]
    hits = [r for r in sig if str(r["verdict"]).startswith("✅")]
    lean = [r for r in ok if r.get("direction_ok") is not None]
    lean_ok = [r for r in lean if r["direction_ok"]]
    up_days = [r for r in ok if r["ret_pct"] > 0]
    # strategy: long on buy, short on sell, flat otherwise – next-day return
    strat = []
    for r in ok:
        pos = 1 if r["level"] in ("buy", "strong_buy") else -1 if r["level"] in ("sell", "strong_sell") else 0
        strat.append(pos * r["ret_pct"] / 100)
    eq = float(np.prod([1 + x for x in strat]) - 1) * 100 if strat else 0.0
    bh = float(np.prod([1 + r["ret_pct"] / 100 for r in ok]) - 1) * 100 if ok else 0.0
    # probability quality: Brier score vs a constant 50% guess (lower is better)
    y = np.array([1.0 if r["ret_pct"] > 0 else 0.0 for r in ok])
    p = np.array([r["p_up"] for r in ok])
    brier = float(np.mean((p - y) ** 2)) if len(ok) else math.nan
    # correlation between predicted probability and the actual move
    corr = float(np.corrcoef(p, [r["ret_pct"] for r in ok])[0, 1]) if len(ok) > 3 and p.std() > 0 else math.nan
    return {"days": len(ok), "signals": len(sig), "signal_hits": len(hits),
            "signal_rate": len(hits) / len(sig) if sig else math.nan,
            "leaning_days": len(lean), "direction_hits": len(lean_ok),
            "direction_rate": len(lean_ok) / len(lean) if lean else math.nan,
            "up_days": len(up_days), "strategy_pct": eq, "buy_hold_pct": bh,
            "brier": brier, "brier_coin": 0.25, "prob_move_corr": corr}


def to_markdown(rep: dict) -> str:
    s = rep["summary"]
    pc = lambda v: "—" if v is None or (isinstance(v, float) and math.isnan(v)) else f"{v * 100:.0f}%"
    lines = [f"# בדיקה עיוורת יום-אחר-יום: {rep['ticker']}",
             f"{rep['days']} ימי מסחר · תחזית ל-{rep['horizon']} יום קדימה · הופק {rep['generated']}", "",
             "## סיכום",
             f"- המלצות קנייה/מכירה: **{s['signal_hits']}/{s['signals']}** צדקו ({pc(s['signal_rate'])})",
             f"- כיוון בימים שבהם המערכת נטתה: **{s['direction_hits']}/{s['leaning_days']}** ({pc(s['direction_rate'])})",
             f"- ימים שבהם המניה עלתה: {s['up_days']}/{s['days']}",
             f"- אסטרטגיה (לונג בקנייה, שורט במכירה): **{s['strategy_pct']:+.1f}%** · קנה-והחזק: {s['buy_hold_pct']:+.1f}%",
             f"- ציון Brier: {s['brier']:.3f} (0.250 = ניחוש 50%; נמוך יותר = טוב יותר)", "",
             "## יום אחר יום",
             "| תאריך | סגירה | סיכוי עלייה | המלצה | 1 שוק | 2 גרף | 3 היסט' | 4 דוחות | 5 פיננסי | 6 פרסומים | סגירה למחרת | שינוי | צדקה? |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in rep["rows"]:
        if "error" in r:
            lines.append(f"| {r['date']} | שגיאה: {r['error']} |" + " |" * 11)
            continue
        f = lambda k: "" if r.get(k) is None else f"{r[k]:+.2f}"
        lines.append(f"| {r['date']} | {r['close']:.2f} | {r['p_up'] * 100:.1f}% | {r['action']} | "
                     f"{f('L1')} | {f('L2')} | {f('L3')} | {f('L4')} | {f('L5')} | {f('L6')} | "
                     f"{r['next_close']:.2f} | {r['ret_pct']:+.2f}% | {r['verdict']} |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ticker")
    ap.add_argument("--days", type=int, default=21)
    ap.add_argument("--horizon", type=int, default=1)
    ap.add_argument("--out", default="reports")
    a = ap.parse_args()
    base = Settings(interval="1d", horizon=a.horizon, use_news=False, use_premarket=False)
    rep = walk(a.ticker, a.days, base, progress=lambda p, t: print(f"{p * 100:5.1f}% {t}", flush=True))
    out = Path(a.out)
    out.mkdir(exist_ok=True)
    (out / f"walk_{rep['ticker']}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=1, default=float),
                                                    encoding="utf-8")
    (out / f"walk_{rep['ticker']}.md").write_text(to_markdown(rep), encoding="utf-8")
    print(to_markdown(rep))


if __name__ == "__main__":
    main()
