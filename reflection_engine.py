"""reflection_engine.py — Post-Mortem Reflection Pipeline & Root-Cause Attribution.

Audits closed trades using gemini-2.5-pro (with Flash fallback), reconstructs
price excursion (MFE/MAE), updates rolling 30-trade agent accuracy, and records
actionable guardrail rules for SQLite prompt injection.
"""

import json
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
import yfinance as yf
from tax_engine import parse_datetime

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


def calculate_excursions(ticker: str, opened_at_str: str, closed_at_str: Optional[str], entry_price: float):
    """Computes Max Favorable Excursion (MFE) and Max Adverse Excursion (MAE) in percent."""
    if ticker.startswith("MOCK_") or ticker.startswith("TEST_"):
        return 0.0, 0.0
    try:
        opened_dt = parse_datetime(opened_at_str)
        start_date = opened_dt.strftime("%Y-%m-%d")
        closed_dt = parse_datetime(closed_at_str) if closed_at_str else datetime.now()
        end_date = (closed_dt + timedelta(days=1)).strftime("%Y-%m-%d")
        df = yf.download(ticker, start=start_date, end=end_date, interval="1d", progress=False)
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
    realized_pnl: float,
    closed_at: Optional[str] = None
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
        open_dt = parse_datetime(opened_at)
        close_dt = parse_datetime(closed_at) if closed_at else datetime.now()
        holding_days = max(1, (close_dt - open_dt).days)
    except Exception:
        holding_days = 1

    mfe_pct, mae_pct = calculate_excursions(ticker, opened_at, closed_at, avg_price)

    # Mock Mode Generation (or unit test tickers)
    if _is_mock_mode() or ticker.startswith("MOCK_") or ticker.startswith("TEST_"):
        if outcome == "WIN":
            driver = "TECHNICAL_BREAKOUT_VALID"
            acc_agent = "TECHNICAL"
            fld_agent = "NONE"
            diagnosis = f"Trade in {ticker} hit target price of ₹{exit_price:,.2f} driven by Stage-2 momentum."
            rule = f"In BULL_STAGE2, maintain trailing stops on high RS-Alpha breakouts rather than premature trims."
            polarity = "OPPORTUNITY_LEEWAY"
            regime = "BULL_STAGE2"
        else:
            driver = "STOP_LOSS_TOO_TIGHT" if exit_reason == "STOP_LOSS_HIT" else "NIFTY_MACRO_SECTOR_DRAG"
            acc_agent = "RISK_COMMITTEE"
            fld_agent = "TECHNICAL"
            diagnosis = f"Trade in {ticker} was stopped out at ₹{exit_price:,.2f} due to {exit_reason}."
            rule = f"In BEAR_RISK_OFF or high VIX, size positions at 5% NAV with minimum 2.5x ATR stops."
            polarity = "DEFENSIVE_GUARDRAIL"
            regime = "BEAR_RISK_OFF"

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
            actionable_rule=rule,
            applicable_regime=regime,
            rule_polarity=polarity
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
4. Formulate exactly ONE concrete, actionable rule for future committee meetings.

CRITICAL GUARDRAIL FORMULATION RULES (NO STATIC BANS):
1. FORBIDDEN: Never use absolute prohibition words like "Prohibit all", "Freeze all longs", "Automatically liquidate after X days", or "Strictly reject any stock below 200-SMA".
2. MANDATORY FORMAT: Every actionable_rule MUST be phrased as a Regime-Conditional Sizing or Conviction Adjustment:
   "In [Regime], when [Specific Condition A + Condition B] occur together, reduce position size to [5-6% NAV] (or require [Secondary Confirmation]), unless [Offsetting Strength like RS Alpha > +3%] is present."
3. WINNING TRADE REFLECTION: When auditing a WIN (outcome == "WIN"), set rule_polarity = "OPPORTUNITY_LEEWAY" and formulate a rule that protects future winners from premature vetoing (e.g., "In early recoveries, approve high RS-Alpha breakouts even on moderate 0.85x-1.0x volume").
4. GENERALIZATION MANDATE: NEVER name specific company tickers or conglomerate group names (e.g. NEVER mention 'Adani', 'Tata', 'Reliance')."""

    llm = get_deep_think_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(TradePostMortem)
    try:
        audit: TradePostMortem = paced_llm_invoke(structured_llm, prompt, is_deep_think=True, schema=TradePostMortem)
        normalized_pnl = audit.pnl_pct / 100.0 if abs(audit.pnl_pct) > 1.0 else audit.pnl_pct
        app_regime = getattr(audit, "applicable_regime", "ALL_REGIMES")
        rule_pol = getattr(audit, "rule_polarity", "DEFENSIVE_GUARDRAIL")
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
            actionable_rule=audit.actionable_rule,
            applicable_regime=app_regime,
            rule_polarity=rule_pol
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
        actionable_rule=actionable_rule,
        market_regime="BEAR_RISK_OFF",
        rule_type="RISK_GUARDRAIL"
    )
    close_thesis_snapshot(snap_id)
    logger.info(f"[Post-Mortem Learning] Formulated rule from past loss in {ticker}: {actionable_rule}")
    return {
        "id": pm_id,
        "ticker": ticker,
        "failed_agent": failed_agent,
        "actionable_rule": actionable_rule
    }


def run_missed_opportunity_audit(
    snapshot_id: int,
    ticker: str,
    sector: str,
    entry_price: float,
    max_gain_pct: float,
    holding_days: int,
    market_regime: str = "BULL_STAGE2"
) -> Dict[str, Any]:
    """Records a missed winner outcome when a committee-vetoed stock subsequently rallied >= 12%."""
    rule_text = (
        f"In {market_regime} or early recoveries, approve decoupled {sector} breakouts with positive RS Alpha "
        f"at 5-6% NAV allocation rather than issuing a blanket veto."
    )
    pm_id = save_trade_post_mortem(
        snapshot_id=snapshot_id,
        ticker=ticker,
        sector=sector,
        outcome="MISSED_WINNER",
        pnl_pct=round(max_gain_pct, 4),
        holding_days=holding_days,
        primary_driver="OVER_DEFENSIVE_VETO_PARALYSIS",
        accurate_agent="BULL_BEAR_DEBATE",
        failed_agent="RISK_COMMITTEE",
        root_cause=f"Risk Committee over-enforced defensive veto on {ticker}; stock subsequently rallied +{max_gain_pct*100:.1f}%.",
        actionable_rule=rule_text,
        applicable_regime=market_regime,
        rule_polarity="OPPORTUNITY_LEEWAY",
        market_regime=market_regime,
        rule_type="RELAXATION_RULE",
        success_score=1.5,
        false_veto_count=0,
        is_active=1
    )
    with get_db_connection() as conn:
        conn.execute("UPDATE thesis_snapshots SET status = 'AUDITED_MISSED_WINNER' WHERE id = ?;", (snapshot_id,))
        conn.commit()

    return {
        "id": pm_id,
        "ticker": ticker,
        "outcome": "MISSED_WINNER",
        "actionable_rule": rule_text
    }


def audit_vetoed_opportunities(
    historical_data: Optional[Dict[str, Any]] = None,
    lookback_days: int = 90
) -> List[Dict[str, Any]]:
    """Audits vetoed committee candidates to detect False Negatives (Inaction Bias).

    If a stock vetoed by the committee subsequently rallied >= 12% without triggering a
    major drawdown, this auditor formulates an Opportunity Leeway Rule to counterbalance defensive scars.
    """
    import yfinance as yf
    from database import get_db_connection, save_trade_post_mortem, record_rule_veto_outcome

    audited_relaxations = []
    with get_db_connection() as conn:
        cursor = conn.cursor()
        rows = cursor.execute(
            """
            SELECT id, ticker, sector, opened_at, entry_price, stop_loss, target_price,
                   pm_rationale, debate_summary
            FROM thesis_snapshots
            WHERE status = 'VETOED'
            ORDER BY id DESC LIMIT 50;
            """
        ).fetchall()

    for r in rows:
        snap_id = r["id"]
        sym = r["ticker"]
        sec = r["sector"]
        veto_date = r["opened_at"][:10]
        entry_p = float(r["entry_price"])

        # Fetch forward price action (60 trading days)
        try:
            df = None
            if historical_data and sym in historical_data:
                df = historical_data[sym]
            else:
                df = yf.download(sym, start=veto_date, period="3mo", interval="1d", progress=False)

            if df is None or len(df) < 5:
                continue

            # Forward window from veto date
            fwd_df = df.loc[veto_date:] if veto_date in df.index else df
            if len(fwd_df) < 5:
                continue

            high_p = float(fwd_df["High"].max())
            low_p = float(fwd_df["Low"].min())
            max_gain = (high_p - entry_p) / entry_p
            max_drawdown = (low_p - entry_p) / entry_p

            # False Negative Threshold: Rallied >= 12% without falling below standard stop (> -8%)
            if max_gain >= 0.12 and max_drawdown > -0.08:
                rule_text = (
                    f"During market recoveries or Stage-2 transitions, relax the breakout volume gate to 0.80x-0.85x "
                    f"and allow 50% position sizing for decoupled {sec} leaders exhibiting positive RS Alpha."
                )
                logger.info(f"[Inaction Bias Audit] False Veto detected in {sym}: Rallied +{max_gain*100:.1f}%. Formulating opportunity leeway rule.")

                pm_id = save_trade_post_mortem(
                    snapshot_id=snap_id,
                    ticker=sym,
                    sector=sec,
                    outcome="MISSED_WINNER",
                    pnl_pct=round(max_gain, 4),
                    holding_days=len(fwd_df),
                    primary_driver="OVER_DEFENSIVE_VETO_PARALYSIS",
                    accurate_agent="BULL_BEAR_DEBATE",
                    failed_agent="RISK_COMMITTEE",
                    root_cause=f"Committee over-enforced defensive veto on {sym}; stock subsequently surged +{max_gain*100:.1f}% without drawdown.",
                    actionable_rule=rule_text,
                    market_regime="BULL_STAGE2",
                    rule_type="RELAXATION_RULE",
                    applicable_regime="BULL_STAGE2",
                    rule_polarity="OPPORTUNITY_LEEWAY",
                    success_score=1.5,
                    false_veto_count=0,
                    is_active=1
                )

                # Mark snapshot audited
                with get_db_connection() as conn:
                    conn.execute("UPDATE thesis_snapshots SET status = 'AUDITED_MISSED_WINNER' WHERE id = ?;", (snap_id,))
                    conn.commit()

                audited_relaxations.append({
                    "snapshot_id": snap_id,
                    "ticker": sym,
                    "max_gain_pct": round(max_gain * 100, 2),
                    "relaxation_rule": rule_text
                })

        except Exception as e:
            logger.debug(f"[Inaction Bias Audit] Could not audit forward action for {sym}: {e}")
            continue

    return audited_relaxations

