"""Machine-learning probability model with honest walk-forward validation.

Idea: for every historical bar we know what happened H bars later. We train a
model on the past only, predict the next block, roll forward (walk-forward), and
measure how good the predictions REALLY were out-of-sample. The final
probability is calibrated on those out-of-sample predictions and shrunk toward
the base rate when the model shows little skill – so a "70%" means something.
"""
from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", category=UserWarning)


def build_features(x: pd.DataFrame, intraday: bool, mkt: pd.Series | None = None) -> pd.DataFrame:
    """Stationary (scale-free) features from an indicator-enriched frame."""
    c = x["Close"]
    a = x["atr"].replace(0, np.nan)
    f = pd.DataFrame(index=x.index)
    for n in (1, 3, 6, 12, 24):
        f[f"ret_{n}"] = np.log(c / c.shift(n))
    for n in (9, 21, 50, 200):
        f[f"dist_ema{n}"] = (c - x[f"ema{n}"]) / a
    f["ema9_21"] = (x["ema9"] - x["ema21"]) / a
    f["ema21_50"] = (x["ema21"] - x["ema50"]) / a
    f["rsi"] = (x["rsi"] - 50) / 50
    f["rsi_chg"] = x["rsi"].diff(3) / 50
    f["macd_hist"] = x["hist"] / a
    f["macd_hist_chg"] = x["hist"].diff(3) / a
    f["bb_pctb"] = x["bb_pctb"] - 0.5
    f["bb_width"] = x["bb_width"] / x["bb_width"].rolling(100, min_periods=20).mean()
    f["stoch"] = (x["stoch_k"] - 50) / 50
    f["stoch_kd"] = (x["stoch_k"] - x["stoch_d"]) / 50
    f["adx_dir"] = x["adx"] / 50 * np.sign(x["plus_di"] - x["minus_di"])
    f["vol_ratio"] = np.log(x["vol_ratio"].clip(0.05, 20))
    f["mfi"] = (x["mfi"] - 50) / 50
    f["cci"] = (x["cci"] / 200).clip(-2, 2)
    f["willr"] = (x["willr"] + 50) / 50
    f["supertrend"] = x["supertrend"]
    f["dist_vwap"] = (c - x["vwap"]) / a
    obv_s = x["obv"].diff(10)
    f["obv_slope"] = obv_s / x["Volume"].rolling(50, min_periods=10).mean().replace(0, np.nan) / 10
    f["atr_pct"] = a / c * 100
    f["range_pos"] = (c - x["Low"].rolling(20).min()) / (
        x["High"].rolling(20).max() - x["Low"].rolling(20).min()).replace(0, np.nan) - 0.5
    if mkt is not None:
        # Market context: most single-stock moves are the market moving. SPY returns over the
        # same bars and the stock's strength relative to SPY (only past/current bars → no look-ahead).
        m = mkt.reindex(x.index).ffill(limit=3)
        for n in (1, 6, 12, 24):
            mr = np.log(m / m.shift(n))
            f[f"mkt_ret_{n}"] = mr
            if n >= 6:
                f[f"rel_{n}"] = f[f"ret_{n}"] - mr
        mret = np.log(m / m.shift(1))
        f["beta_60"] = (f["ret_1"].rolling(60).cov(mret) / mret.rolling(60).var().replace(0, np.nan)).clip(-3, 5)
    if intraday and isinstance(x.index, pd.DatetimeIndex):
        mins = x.index.hour * 60 + x.index.minute
        f["tod_sin"] = np.sin(2 * np.pi * (mins - 570) / 390)
        f["tod_cos"] = np.cos(2 * np.pi * (mins - 570) / 390)
    else:
        f["dow"] = x.index.dayofweek / 4 - 0.5 if isinstance(x.index, pd.DatetimeIndex) else 0
    return f.replace([np.inf, -np.inf], np.nan)


def make_labels(close: pd.Series, horizon: int, noise: float = 0.15) -> tuple[pd.Series, pd.Series]:
    """y=1 if the price is clearly higher after `horizon` bars, 0 if clearly lower.
    Tiny moves (flat bars, ties – very common intraday) are dropped instead of being counted
    as 'down', which used to bias every prediction slightly below 50%. The noise band is
    `noise` × the typical |move| measured only on PAST labels (expanding median, shifted)."""
    fwd = close.shift(-horizon) / close - 1
    band = fwd.abs().expanding(min_periods=50).median().shift(horizon) * noise
    band = band.bfill().fillna(0)
    y = pd.Series(np.nan, index=close.index)
    y[fwd > band] = 1.0
    y[fwd < -band] = 0.0
    return y.where(fwd.notna()), fwd


class _Ensemble:
    def __init__(self):
        self.lr = make_pipeline(StandardScaler(), LogisticRegression(C=0.3, max_iter=500))
        self.gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=120,
                                                 l2_regularization=1.0, min_samples_leaf=40,
                                                 random_state=7)

    def fit(self, X, y):
        self.lr.fit(X, y)
        self.gb.fit(X, y)
        return self

    def predict_proba(self, X):
        return 0.5 * self.lr.predict_proba(X)[:, 1] + 0.5 * self.gb.predict_proba(X)[:, 1]


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


def walk_forward(X: pd.DataFrame, y: pd.Series, horizon: int, n_splits: int = 6,
                 min_train: int = 250) -> pd.Series:
    """Out-of-sample predictions with a purge gap of `horizon` bars."""
    n = len(X)
    oos = pd.Series(np.nan, index=X.index)
    if n < min_train + 50:
        return oos
    edges = np.linspace(min_train, n, n_splits + 1).astype(int)
    for i in range(n_splits):
        tr_end, te_start, te_end = edges[i] - horizon, edges[i], edges[i + 1]
        if tr_end < 100 or te_end <= te_start:
            continue
        ytr = y.iloc[:tr_end]
        if ytr.nunique() < 2:
            continue
        m = _Ensemble().fit(X.iloc[:tr_end], ytr)
        oos.iloc[te_start:te_end] = m.predict_proba(X.iloc[te_start:te_end])
    return oos


def train_and_predict(x: pd.DataFrame, intraday: bool, horizon: int,
                      buy_th: float = 0.6, sell_th: float = 0.4, mkt: pd.Series | None = None,
                      noise: float = 0.15, shuffle_labels: bool = False) -> dict:
    feats = build_features(x, intraday, mkt)
    # drop feature columns that are entirely missing (e.g. no SPY data for this interval)
    feats = feats.loc[:, feats.notna().mean() > 0.5]
    y, fwd = make_labels(x["Close"], horizon, noise)
    if shuffle_labels:   # self-test only: destroys any real relation → AUC must fall to ~0.5
        rng = np.random.default_rng(0)
        y = pd.Series(rng.permutation(y.to_numpy()), index=y.index).where(y.notna())
    data = feats.join(y.rename("y")).join(fwd.rename("fwd"))
    valid = data.dropna(subset=list(feats.columns))
    labelled = valid.dropna(subset=["y"])
    if len(labelled) < 300:
        raise ValueError(f"מעט מדי נתונים לאימון ({len(labelled)} נרות). בחר טווח/אינטרוול ארוך יותר.")

    X, Y = labelled[feats.columns], labelled["y"]
    base_rate = float(Y.mean())

    oos = walk_forward(X, Y, horizon)
    mask = oos.notna()
    auc, acc, brier = np.nan, np.nan, np.nan
    cal_a, cal_b = 1.0, 0.0
    if mask.sum() > 100 and Y[mask].nunique() == 2:
        auc = float(roc_auc_score(Y[mask], oos[mask]))
        acc = float(((oos[mask] > 0.5) == (Y[mask] == 1)).mean())
        # Platt calibration on out-of-sample predictions
        cal = LogisticRegression(C=10).fit(_logit(oos[mask].to_numpy()).reshape(-1, 1), Y[mask])
        cal_a, cal_b = float(cal.coef_[0][0]), float(cal.intercept_[0])
        p_cal = 1 / (1 + np.exp(-(cal_a * _logit(oos[mask].to_numpy()) + cal_b)))
        brier = float(np.mean((p_cal - Y[mask].to_numpy()) ** 2))

    # Skill-based shrinkage: AUC 0.50 -> no edge -> base rate. AUC >= 0.60 -> full trust.
    skill = 0.0 if np.isnan(auc) else float(np.clip((auc - 0.5) / 0.10, 0, 1))

    def calibrate(p):
        p = np.asarray(p, dtype=float)
        pc = 1 / (1 + np.exp(-(cal_a * _logit(p) + cal_b)))
        return base_rate + (pc - base_rate) * skill

    final = _Ensemble().fit(X, Y)
    p_raw_now = float(final.predict_proba(valid[feats.columns].iloc[[-1]])[0])
    p_now = float(calibrate(p_raw_now))

    # Feature importance proxy: standardized LR coefficients * current value
    lr = final.lr.named_steps["logisticregression"]
    sc = final.lr.named_steps["standardscaler"]
    z = (valid[feats.columns].iloc[-1].to_numpy() - sc.mean_) / np.where(sc.scale_ == 0, 1, sc.scale_)
    contrib = pd.Series(lr.coef_[0] * z, index=feats.columns).sort_values()

    # Backtest on out-of-sample predictions
    bt = backtest(oos[mask].map(lambda p: float(calibrate(p))), labelled.loc[mask, "fwd"],
                  horizon, buy_th, sell_th)

    # Reliability table
    rel = None
    if mask.sum() > 100:
        pc = pd.Series(calibrate(oos[mask].to_numpy()), index=oos[mask].index)
        bins = pd.cut(pc, [0, .35, .45, .55, .65, 1.0])
        rel = (pd.DataFrame({"bucket": bins, "y": Y[mask]}).groupby("bucket", observed=True)["y"]
               .agg(["mean", "count"]).rename(columns={"mean": "actual_up_rate", "count": "n"}))

    return {"p_up": p_now, "p_raw": p_raw_now, "base_rate": base_rate, "auc": auc,
            "accuracy": acc, "brier": brier, "skill": skill, "n_train": len(X),
            "n_oos": int(mask.sum()), "contrib": contrib, "backtest": bt,
            "reliability": rel, "oos_probs": oos[mask].map(lambda p: float(calibrate(p)))}


def backtest(p: pd.Series, fwd: pd.Series, horizon: int, buy_th: float, sell_th: float,
             allow_short: bool = True, cost_bps: float = 2.0) -> dict:
    """Non-overlapping trades every `horizon` bars on out-of-sample probabilities."""
    if len(p) < 20:
        return {}
    idx = np.arange(0, len(p), horizon)
    pp, rr = p.iloc[idx].to_numpy(), fwd.iloc[idx].to_numpy()
    pos = np.where(pp >= buy_th, 1.0, np.where((pp <= sell_th) & allow_short, -1.0, 0.0))
    cost = cost_bps / 1e4
    strat = pos * rr - np.abs(pos) * cost
    trades = pos != 0
    longs, shorts = pos == 1, pos == -1
    eq = pd.Series(np.cumprod(1 + strat), index=p.index[idx])
    bh = pd.Series(np.cumprod(1 + rr), index=p.index[idx])
    dd = (eq / eq.cummax() - 1).min()
    return {
        "trades": int(trades.sum()),
        "win_rate": float((strat[trades] > 0).mean()) if trades.any() else np.nan,
        "long_hit": float((rr[longs] > 0).mean()) if longs.any() else np.nan,
        "short_hit": float((rr[shorts] < 0).mean()) if shorts.any() else np.nan,
        "avg_trade_%": float(strat[trades].mean() * 100) if trades.any() else np.nan,
        "total_%": float((eq.iloc[-1] - 1) * 100),
        "buyhold_%": float((bh.iloc[-1] - 1) * 100),
        "max_dd_%": float(dd * 100),
        "equity": eq, "buyhold": bh,
    }
