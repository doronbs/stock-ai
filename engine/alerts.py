"""Alerts (Telegram + local sound) and persistent user settings."""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import requests

CONFIG_PATH = Path(__file__).resolve().parent.parent / "settings.json"

DEFAULT_CONFIG = {
    "watchlist": ["AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "AMD",
                  "AVGO", "NFLX", "PLTR", "JPM", "COIN", "SMCI", "MU", "INTC"],
    "interval": "5m",
    "horizon": 12,
    "buy_th": 0.60,
    "sell_th": 0.40,
    "refresh_minutes": 5,
    "telegram_token": "",
    "telegram_chat_id": "",
    "alert_min_prob": 0.65,
}


def load_config() -> dict:
    cfg = dict(DEFAULT_CONFIG)
    try:
        if CONFIG_PATH.exists():
            cfg.update(json.loads(CONFIG_PATH.read_text(encoding="utf-8")))
    except Exception:
        pass
    cfg["telegram_token"] = os.getenv("TELEGRAM_TOKEN", cfg["telegram_token"])
    cfg["telegram_chat_id"] = os.getenv("TELEGRAM_CHAT_ID", cfg["telegram_chat_id"])
    return cfg


def save_config(cfg: dict) -> None:
    CONFIG_PATH.write_text(json.dumps(cfg, ensure_ascii=False, indent=2), encoding="utf-8")


def send_telegram(text: str, cfg: dict | None = None) -> bool:
    cfg = cfg or load_config()
    tok, chat = cfg.get("telegram_token"), cfg.get("telegram_chat_id")
    if not tok or not chat:
        return False
    try:
        r = requests.post(f"https://api.telegram.org/bot{tok}/sendMessage",
                          data={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=10)
        return r.ok
    except Exception:
        return False


def beep():
    try:
        if sys.platform.startswith("win"):
            import winsound
            winsound.MessageBeep(winsound.MB_ICONEXCLAMATION)
        else:
            print("\a", end="", flush=True)
    except Exception:
        pass


def format_alert(r) -> str:
    lv = r.levels
    return (f"<b>{r.ticker}</b> – {r.action}\n"
            f"סיכוי עלייה: <b>{r.p_up * 100:.1f}%</b>\n"
            f"מחיר: {r.price:.2f} | סטופ: {lv['stop']:.2f} | יעד: {lv['target']:.2f} "
            f"(R:R {lv['rr']:.1f})\n"
            f"דיוק מודל היסטורי (AUC): {r.model['auc']:.3f}\n"
            + "\n".join(r.tech_reasons[:3]))
