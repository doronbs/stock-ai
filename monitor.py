"""Background monitor – runs without the UI and alerts on new signals.

Examples:
    python monitor.py                         # watchlist from settings.json
    python monitor.py --tickers NVDA TSLA --every 2 --interval 1m --horizon 15
"""
from __future__ import annotations

import argparse
import time
from datetime import datetime

from engine import data as D
from engine.alerts import beep, format_alert, load_config, send_telegram
from engine.analyzer import Settings, analyze


def main():
    cfg = load_config()
    ap = argparse.ArgumentParser(description="StockAI background monitor")
    ap.add_argument("--tickers", nargs="*", default=cfg["watchlist"])
    ap.add_argument("--every", type=float, default=cfg["refresh_minutes"], help="minutes between scans")
    ap.add_argument("--interval", default=cfg["interval"])
    ap.add_argument("--horizon", type=int, default=cfg["horizon"])
    ap.add_argument("--min-prob", type=float, default=cfg["alert_min_prob"],
                    help="alert when P(up)>=x or P(up)<=1-x")
    ap.add_argument("--always", action="store_true", help="scan even when US market is closed")
    a = ap.parse_args()

    s = Settings(interval=a.interval, horizon=a.horizon, buy_th=cfg["buy_th"], sell_th=cfg["sell_th"])
    last_signal: dict[str, str] = {}
    print(f"StockAI monitor | {len(a.tickers)} tickers | every {a.every} min | {a.interval} | H={a.horizon}")
    while True:
        if not a.always and not D.is_market_open():
            print(datetime.now().strftime("%H:%M"), "השוק האמריקאי סגור – ממתין...")
            time.sleep(300)
            continue
        market = D.market_regime()
        for t in a.tickers:
            try:
                r = analyze(t, s, market=market, with_info=False)
            except Exception as e:
                print(f"{t}: error {e}")
                continue
            print(f"{datetime.now():%H:%M:%S} {t:6} {r.price:10.2f}  P(up)={r.p_up * 100:5.1f}%  {r.action}")
            strong = r.p_up >= a.min_prob or r.p_up <= 1 - a.min_prob
            if strong and last_signal.get(t) != r.action_level:
                msg = format_alert(r)
                print(">>> ALERT\n" + msg.replace("<b>", "").replace("</b>", ""))
                beep()
                send_telegram(msg, cfg)
            last_signal[t] = r.action_level if strong else "none"
        time.sleep(a.every * 60)


if __name__ == "__main__":
    main()
