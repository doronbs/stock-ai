"""Financial news sentiment.

Uses a finance-specific lexicon (inspired by Loughran-McDonald) with negation
handling, plus VADER if installed. Scores are recency-weighted so news from the
last hours count more than news from 3 days ago.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    _vader = SentimentIntensityAnalyzer()
except Exception:  # pragma: no cover
    _vader = None

POS = {
    "beat": 2, "beats": 2, "tops": 2, "topped": 2, "surge": 2, "surges": 2, "soar": 2,
    "soars": 2, "soared": 2, "jump": 1.5, "jumps": 1.5, "rally": 1.5, "rallies": 1.5,
    "record": 1, "upgrade": 2.5, "upgraded": 2.5, "outperform": 2, "buy": 1,
    "strong": 1, "growth": 1, "raises": 1.5, "raised": 1.5, "boost": 1.5, "boosts": 1.5,
    "bullish": 2, "gain": 1, "gains": 1, "profit": 1, "profitable": 1.5, "rebound": 1.5,
    "approval": 2, "approved": 2, "partnership": 1, "contract": 1, "expands": 1,
    "breakthrough": 2, "exceeds": 2, "exceeded": 2, "higher": 0.7, "rises": 1, "rose": 1,
    "climbs": 1, "upbeat": 1.5, "optimistic": 1.5, "buyback": 1.5, "dividend": 0.7,
    "overweight": 1.5, "momentum": 0.7, "accelerat": 1, "wins": 1.2, "win": 1,
}
NEG = {
    "miss": 2, "misses": 2, "missed": 2, "plunge": 2.5, "plunges": 2.5, "plunged": 2.5,
    "tumble": 2, "tumbles": 2, "sink": 2, "sinks": 2, "slump": 2, "slumps": 2,
    "fall": 1, "falls": 1, "fell": 1, "drop": 1.2, "drops": 1.2, "downgrade": 2.5,
    "downgraded": 2.5, "underperform": 2, "sell": 1, "weak": 1.5, "weaker": 1.5,
    "cut": 1.2, "cuts": 1.2, "lowers": 1.5, "lowered": 1.5, "bearish": 2, "loss": 1.5,
    "losses": 1.5, "lawsuit": 2, "sued": 2, "probe": 2, "investigation": 2, "recall": 2,
    "fraud": 3, "bankruptcy": 3, "layoffs": 1.5, "warning": 1.5, "warns": 1.5,
    "decline": 1.2, "declines": 1.2, "lower": 0.7, "slides": 1.2, "concern": 1,
    "concerns": 1, "risk": 0.7, "halt": 2, "delay": 1.2, "delays": 1.2, "short": 0.5,
    "underweight": 1.5, "disappoint": 2, "disappointing": 2, "tariff": 1, "tariffs": 1,
    "sec": 0.8, "antitrust": 1.5, "crash": 3, "selloff": 2, "sell-off": 2, "fears": 1.5,
}
NEGATORS = {"not", "no", "never", "fails", "failed", "without", "despite"}
_tok = re.compile(r"[a-z][a-z\-']+")


def _lex_score(text: str) -> float:
    words = _tok.findall(text.lower())
    s, hits = 0.0, 0
    for i, w in enumerate(words):
        val = 0.0
        for d, sign in ((POS, 1), (NEG, -1)):
            if w in d:
                val = sign * d[w]
        if val:
            if any(p in NEGATORS for p in words[max(0, i - 3):i]):
                val = -val * 0.6
            s += val
            hits += 1
    if not hits:
        return 0.0
    return math.tanh(s / 3.0)


def score_text(text: str) -> float:
    """Return sentiment in [-1, 1]."""
    lx = _lex_score(text)
    if _vader is not None:
        vd = _vader.polarity_scores(text)["compound"]
        return 0.65 * lx + 0.35 * vd
    return lx


def analyze_news(news: list[dict], half_life_hours: float = 18.0) -> dict:
    now = datetime.now(timezone.utc)
    scored, wsum, ssum = [], 0.0, 0.0
    for n in news:
        s = score_text(f"{n['title']}. {n.get('summary', '')}")
        age_h = (now - n["time"]).total_seconds() / 3600 if n.get("time") else 48
        w = 0.5 ** (max(age_h, 0) / half_life_hours)
        scored.append({**n, "sentiment": s, "age_h": age_h, "weight": w})
        if abs(s) > 0.05:
            wsum += w
            ssum += w * s
    agg = ssum / wsum if wsum else 0.0
    pos = sum(1 for n in scored if n["sentiment"] > 0.15)
    neg = sum(1 for n in scored if n["sentiment"] < -0.15)
    recent = [n for n in scored if n["age_h"] < 24]
    return {"score": max(-1, min(1, agg)), "items": scored, "positive": pos,
            "negative": neg, "neutral": len(scored) - pos - neg,
            "buzz_24h": len(recent)}
