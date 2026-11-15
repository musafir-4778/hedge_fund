"""reflection_engine.py — Post-Mortem Reflection Pipeline & Root-Cause Attribution.

Audits closed trades using gemini-2.5-pro (with Flash fallback), reconstructs
price excursion (MFE/MAE), updates rolling 30-trade agent accuracy, and records
actionable guardrail rules for SQLite prompt injection.
"""

import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional
import yfinance as yf

from config import (
    get_deep_think_llm,
    paced_llm_invoke,
    GOOGLE_API_KEY,
)
from database import (
    get_open_thesis_snapshot,
    close_thesis_snapshot,
    save_trade_post_mortem,
    get_db_connection,
)
from state import TradePostMortem

logger = logging.getLogger("wealth_agent_office.reflection")


def _is_mock_mode() -> bool:
    return not bool(GOOGLE_API_KEY)


def calculate_excursions(ticker: str, opened_at_str: str, entry_price: float):
    """Computes Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) in percent."""
    if ticker.startswith("MOCK_") or ticker.startswith("TEST_"):
        return 0.0, 0.0
    try:
        opened_dt = datetime.fromisoformat(opened_at_str)
        start_date = opened_dt.strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_date, interval="1d", progress=False)
        if df.empty or "High" not in df:
            return 0.0, 0.0

        highest = float(df["High"].max())
        lowest = float(df["Low"].min())
        mfe_pct = round(((highest - entry_price) / entry_price) * 100.0, 2)
        mae_pct = round(((lowest - entry_price) / entry_price) * 100.0, 2)
        return mfe_pct, mae_pct
    except Exception as e:
        logger.debug(f"Excursion calculation failed for {ticker}: {e}")
        return 0.0, 0.0


def run_post_mortem_audit(
    position: Dict[str, Any],
    exit_price: float,
    exit_reason: str,
    realized_pnl: float
) -> Optional[Dict[str, Any]]:
    """Runs post-mortem self-reflection upon trade exit."""
    ticker = position["ticker"]
    snapshot_id = position["snapshot_id"]
    avg_price = position["avg_price"]
    opened_at = position["opened_at"]

    logger.info(f"[Post-Mortem] Starting reflection audit for closed trade {ticker}...")

    # Fetch thesis snapshot
    snapshot = None
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM thesis_snapshots WHERE id = ?;", (snapshot_id,)).fetchone()
        if row:
            snapshot = dict(row)

    if not snapshot:
        snapshot = get_open_thesis_snapshot(ticker)

    if not snapshot:
        logger.warning(f"[Post-Mortem] No thesis snapshot found for {ticker} (id={snapshot_id}). Skipping.")
        return None

    # Calculate holding metrics
    pnl_pct = round(((exit_price - avg_price) / avg_price) * 100.0, 2)
    if realized_pnl > 0 and pnl_pct > 0.5:
        outcome = "WIN"
    elif pnl_pct < -0.5:
        outcome = "LOSS"
    else:
        outcome = "BREAKEVEN"

    try:
        open_dt = datetime.fromisoformat(opened_at)
        holding_days = max(1, (datetime.now() - open_dt).days)
    except Exception:
        holding_days = 1

    mfe_pct, mae_pct = calculate_excursions(ticker, opened_at, avg_price)

    # Mock Mode Generation (or unit test tickers)
    if _is_mock_mode() or ticker.startswith("MOCK_") or ticker.startswith("TEST_"):
        if outcome == "WIN":
            driver = "TECHNICAL_BREAKOUT_VALID"
            acc_agent = "TECHNICAL"
            fld_agent = "NONE"
            diagnosis = f"Trade in {ticker} hit target price of ₹{exit_price:,.2f} driven by Stage-2 momentum."
            rule = f"In confirmed Stage-2 bull regimes, maintain trailing stops on {ticker} breakouts."
        else:
            driver = "STOP_LOSS_TOO_TIGHT" if exit_reason == "STOP_LOSS_HIT" else "NIFTY_MACRO_SECTOR_DRAG"
            acc_agent = "RISK_COMMITTEE"
            fld_agent = "TECHNICAL"
            diagnosis = f"Trade in {ticker} was stopped out at ₹{exit_price:,.2f} due to {exit_reason}."
            rule = f"Ensure volatility buffer on {ticker} entry when India VIX is choppy."

        save_trade_post_mortem(
            snapshot_id=snapshot_id,
            ticker=ticker,
            sector=position.get("sector", "Industrials"),
            outcome=outcome,
            pnl_pct=pnl_pct / 100.0,
            holding_days=holding_days,
            primary_driver=driver,
            accurate_agent=acc_agent,
            failed_agent=fld_agent,
            root_cause=diagnosis,
            actionable_rule=rule
        )
        close_thesis_snapshot(snapshot_id)
        logger.info(f"[Post-Mortem] Completed deterministic audit for {ticker}: {outcome} ({pnl_pct}%).")
        return {
            "ticker": ticker,
            "outcome": outcome,
            "pnl_pct": pnl_pct,
            "accurate_agent": acc_agent,
            "failed_agent": fld_agent,
            "actionable_rule": rule
        }

    # Deep Think LLM Audit
    prompt = f"""You are the Chief Post-Mortem Auditor of an Indian Wealth Management Office.
A trade in {ticker} has just been CLOSED. Conduct an objective root-cause post-mortem.

Trade Data:
- Entry Price: ₹{avg_price} | Exit Price: ₹{exit_price} ({exit_reason})
- Realized Net PnL: ₹{realized_pnl:,.2f} ({pnl_pct}%)
- Holding Days: {holding_days}
- Price Excursion: MFE=+{mfe_pct}% | MAE={mae_pct}%

Original Entry Thesis:
- Technical Report: {snapshot.get('technical_report')}
- Fundamental Report: {snapshot.get('fundamental_report')}
- News/Macro Report: {snapshot.get('news_macro_report')}
- Debate Synthesis: {snapshot.get('debate_summary')}
- PM Rationale: {snapshot.get('pm_rationale')}

Instructions:
1. Attribute the primary driver of this outcome.
2. Identify the SINGLE agent whose thesis was most accurate.
3. Identify the SINGLE agent whose analysis had the blind spot or failure.
4. Formulate exactly ONE concrete, actionable guardrail rule for future committee meetings.

CRITICAL ANTI-WALL GENERALIZATION MANDATE:
- NEVER name specific company tickers or conglomerate group names (e.g. NEVER mention 'Adani', 'Tata', 'Reliance', or 'conglomerate-linked').
- NEVER set arbitrary static indicator ceilings (e.g. DO NOT say 'RSI < 68.0' or 'Volume > 2.0x').
- Formulate all guardrails around generalizable quantitative or structural financial invariants:
  * Accrual and cash conversion quality (e.g. CFO/PAT ratio > 0.8; downgrade to HOLD if CFO < 0)
  * Moving average overextension: Price <= 1.20 * SMA50 (do not buy overextended extensions)
  * Beta-adaptive stop buffers: Stop = (2.0 + 0.5 * max(0, beta - 1.0)) * ATR14 for high-beta positional trades
  * Institutional volume confirmation: 20-day Volume Ratio >= 1.1x to 1.2x on breakouts
  * Relative strength momentum: 63-day RS Alpha > 0 vs Nifty 50
  * Minervini Stage-2 trend alignment (Price > SMA50 > SMA150 > SMA200)."""

    llm = get_deep_think_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(TradePostMortem)
    try:
        audit: TradePostMortem = paced_llm_invoke(structured_llm, prompt, is_deep_think=True, schema=TradePostMortem)
        normalized_pnl = audit.pnl_pct / 100.0 if abs(audit.pnl_pct) > 1.0 else audit.pnl_pct
        save_trade_post_mortem(
            snapshot_id=snapshot_id,
            ticker=ticker,
            sector=position.get("sector", "Industrials"),
            outcome=audit.outcome,
            pnl_pct=normalized_pnl,
            holding_days=holding_days,
            primary_driver=audit.primary_driver,
            accurate_agent=audit.accurate_agent,
            failed_agent=audit.failed_agent,
            root_cause=audit.root_cause_analysis,
            actionable_rule=audit.actionable_rule
        )
        close_thesis_snapshot(snapshot_id)
        logger.info(f"[Post-Mortem] Completed LLM audit for {ticker}: {audit.outcome} | Driver: {audit.primary_driver}.")
        return audit.model_dump()
    except Exception as e:
        logger.error(f"[Post-Mortem] Failed to execute LLM audit for {ticker}: {e}")
        close_thesis_snapshot(snapshot_id)
        return None


def simulate_mistake_learning(
    ticker: str,
    sector: str,
    failed_agent: str,
    primary_driver: str,
    root_cause: str,
    actionable_rule: str,
    accurate_agent: str = "RISK_COMMITTEE"
) -> Dict[str, Any]:
    """Injects a diagnosed past mistake to test and demonstrate institutional learning."""
    from database import save_thesis_snapshot
    snap_id = save_thesis_snapshot(
        ticker=ticker,
        sector=sector,
        entry_price=500.0,
        stop_loss=475.0,
        target_price=565.0,
        indicators={"simulated": True},
        technical_report="Initial bullish thesis based on breakout.",
        fundamental_report="Accounting earnings appeared healthy.",
        news_macro_report="Sector tailwinds cited.",
        debate_summary="Approved by committee.",
        risk_summary="Standard stop loss placed.",
        pm_rationale="Historical entry."
    )
    pm_id = save_trade_post_mortem(
        snapshot_id=snap_id,
        ticker=ticker,
        sector=sector,
        outcome="LOSS",
        pnl_pct=-0.05,
        holding_days=7,
        primary_driver=primary_driver,
        accurate_agent=accurate_agent,
        failed_agent=failed_agent,
        root_cause=root_cause,
        actionable_rule=actionable_rule
    )
    close_thesis_snapshot(snap_id)
    logger.info(f"[Post-Mortem Learning] Formulated rule from past loss in {ticker}: {actionable_rule}")
    return {
        "id": pm_id,
        "ticker": ticker,
        "failed_agent": failed_agent,
        "actionable_rule": actionable_rule
    }

