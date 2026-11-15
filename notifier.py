"""notifier.py — Mobile Telegram Alerts for Autonomous Market Cycles.

Dispatches instant mobile alerts for:
- 08:30 AM IST: Morning Discovery Docket & Regime
- 09:30 AM IST: Multi-Agent Committee Decisions & Executed Orders
- Intraday: Stop-Loss & Target Hits
- 03:45 PM IST: End-of-Day Scorecard, NAV & Newly Learned Guardrails
"""

import os
import json
import logging
import urllib.request
import urllib.parse
from typing import Dict, Any, List, Optional
from datetime import datetime

from config import TIMEZONE

logger = logging.getLogger("wealth_agent_office.notifier")

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def send_telegram_message(text: str) -> bool:
    """Dispatches an HTML-formatted message to Telegram."""
    token = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
    chat_id = os.getenv("TELEGRAM_CHAT_ID", "").strip()

    if not token or not chat_id:
        logger.debug("[Notifier] Telegram bot token or chat ID not configured. Skipping alert.")
        return False

    url = f"https://api.telegram.org/bot{token}/sendMessage"
    payload = {
        "chat_id": chat_id,
        "text": text,
        "parse_mode": "HTML",
        "disable_web_page_preview": True,
    }

    try:
        data = json.dumps(payload).encode("utf-8")
        req = urllib.request.Request(
            url,
            data=data,
            headers={"Content-Type": "application/json"}
        )
        with urllib.request.urlopen(req, timeout=10) as response:
            if response.status == 200:
                logger.info("[Notifier] Successfully dispatched Telegram mobile alert.")
                return True
    except Exception as e:
        logger.warning(f"[Notifier] Failed to dispatch Telegram alert: {e}")
        return False

    return False


def notify_morning_docket(docket: List[Dict[str, Any]], regime: str, nifty_close: float):
    """Sends 08:30 AM pre-open discovery summary."""
    regime_emoji = "▲" if "BULL" in regime else ("▼" if "BEAR" in regime else "◆")
    tickers_list = "\n".join([
        f"  • <b>{d.get('ticker')}</b> ({d.get('sector', 'General')}) | RS Alpha: {d.get('rs_alpha', 0)*100:+.1f}%"
        for d in docket[:5]
    ]) if docket else "  • <i>No candidates cleared the Stage-2 momentum hurdle today.</i>"

    msg = (
        f"🏛 <b>WEALTH MANAGEMENT OFFICE — MORNING DOCKET</b>\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d')} | 08:30 AM IST\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Macro Regime: <b>{regime_emoji} {regime}</b>\n"
        f"Nifty 50 Benchmark: ₹{nifty_close:,.2f}\n\n"
        f"<b>Screened Stage-2 Candidates:</b>\n{tickers_list}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Multi-agent committee convenes at 09:30 AM IST.</i>"
    )
    send_telegram_message(msg)


def notify_trade_executed(action: str, ticker: str, qty: int, price: float, stop: float, target: float, rationale: str):
    """Sends 09:30 AM committee trade execution alert."""
    action_emoji = "🟢 BUY" if action == "BUY" else "🔴 SELL"
    total_val = qty * price

    msg = (
        f"⚡ <b>COMMITTEE EXECUTIVE ORDER: {action_emoji}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Ticker: <b>{ticker}</b>\n"
        f"Executed Qty: <b>{qty} shares</b> @ ₹{price:,.2f}\n"
        f"Total Value: ₹{total_val:,.2f}\n\n"
        f"🛡 <b>Hard Stop-Loss (2xATR):</b> ₹{stop:,.2f}\n"
        f"🎯 <b>Profit Target (>= 2.5:1 R:R):</b> ₹{target:,.2f}\n\n"
        f"<b>Lead PM Rationale:</b>\n<i>{rationale}</i>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>All delivery friction (STT, stamp duty, turnover fees) accounted for.</i>"
    )
    send_telegram_message(msg)


def notify_trade_exit(ticker: str, exit_price: float, reason: str, pnl_pct: float, pnl_inr: float, new_rule: Optional[str] = None):
    """Sends real-time stop-loss or profit-target trigger alert."""
    outcome_emoji = "🎉 TARGET HIT" if "TARGET" in reason else "🛑 STOP HIT"
    pnl_sign = "+" if pnl_inr >= 0 else ""

    rule_section = f"\n\n🧠 <b>New Learned Guardrail:</b>\n<i>\"{new_rule}\"</i>" if new_rule else ""

    msg = (
        f"{outcome_emoji}: <b>{ticker}</b>\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Exit Price: <b>₹{exit_price:,.2f}</b> ({reason})\n"
        f"Realized PnL: <b>{pnl_sign}{pnl_pct:.2f}%</b> (₹{pnl_sign}{pnl_inr:,.2f}){rule_section}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Capital recycled to cash reserves.</i>"
    )
    send_telegram_message(msg)


def notify_evening_summary(nav: float, return_pct: float, cash: float, active_positions: List[Dict[str, Any]], new_rules_today: List[str]):
    """Sends 03:45 PM market close portfolio report."""
    ret_sign = "+" if return_pct >= 0 else ""
    holdings_str = "\n".join([
        f"  • <b>{p.get('ticker')}</b>: {p.get('qty')} shs @ avg ₹{p.get('avg_price', 0):,.2f} (Stop: ₹{p.get('stop_loss', 0):,.2f})"
        for p in active_positions
    ]) if active_positions else "  • <i>100% Cash / No open equity risk.</i>"

    rules_str = "\n".join([f"  • <i>\"{r}\"</i>" for r in new_rules_today]) if new_rules_today else "  • <i>No new failures. Existing guardrails enforced.</i>"

    msg = (
        f"📊 <b>MARKET CLOSE EXECUTIVE BRIEFING</b>\n"
        f"📅 {datetime.now().strftime('%Y-%m-%d')} | 03:45 PM IST\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"Portfolio NAV: <b>₹{nav:,.2f}</b> ({ret_sign}{return_pct:.2f}%)\n"
        f"Liquid Cash Reserve: ₹{cash:,.2f}\n\n"
        f"<b>Active Portfolio Holdings:</b>\n{holdings_str}\n\n"
        f"<b>Institutional Memory Updates:</b>\n{rules_str}\n"
        f"━━━━━━━━━━━━━━━━━━━━━━\n"
        f"<i>Unattended daemon sleeping until 08:30 AM IST.</i>"
    )
    send_telegram_message(msg)

