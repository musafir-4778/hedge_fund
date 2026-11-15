"""graph.py — LangGraph StateGraph Wiring, Conditional Debate Loops & Execution.

Wires Stage 1 Analysts, Stage 2 Bull/Bear debate loop, Stage 3 Risk Committee,
and Stage 4 Portfolio Manager sequentially through paced Gemini calls.
"""

import sqlite3
import logging
from typing import Dict, Any, Optional

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.sqlite import SqliteSaver

from config import (
    MAX_DEBATE_ROUNDS,
    DB_PATH,
)
from state import InvestmentFirmState
from database import (
    get_portfolio_account,
    get_all_positions,
    get_top_actionable_rules,
    get_sector_actionable_rules,
    get_regime_balanced_rules,
    format_agent_scorecard_str,
    get_ist_now,
)
from tax_engine import format_portfolio_tax_summary
from data_tools import (
    normalize_ticker,
    get_market_regime,
    get_benchmark_metrics,
    calculate_technical_indicators,
    calculate_risk_boundaries,
    fetch_fundamental_metrics,
    fetch_recent_stock_news,
)
from agents import (
    technical_analyst_node,
    fundamental_analyst_node,
    news_macro_analyst_node,
    bull_researcher_node,
    bear_researcher_node,
    research_manager_node,
    risk_committee_node,
    aggressive_risk_node,
    neutral_risk_node,
    conservative_risk_node,
    portfolio_manager_node,
)

logger = logging.getLogger("wealth_agent_office.graph")


def data_and_memory_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Pure Python deterministic data gathering and memory injection node."""
    ticker = normalize_ticker(state["ticker"])
    logger.info(f"[Graph] Hydrating data and memory for {ticker}...")

    # 1. Benchmark Metrics & Volatility (Preserve Point-In-Time if pre-hydrated)
    benchmark_metrics = state.get("benchmark_context") or get_benchmark_metrics()
    if state.get("market_regime") and state.get("market_regime") != "CHOPPY_RANGE":
        regime = state["market_regime"]
        india_vix = state.get("india_vix", 16.0)
    else:
        regime = benchmark_metrics.get("regime", "CHOPPY_RANGE")
        india_vix = float(benchmark_metrics.get("india_vix", 16.0))

    # 2. Portfolio Context with Tax & Treasury Tracking
    acc = get_portfolio_account()
    cash = acc["cash"]
    liquid_bal = float(acc.get("liquid_fund_balance", 0.0) or 0.0)
    tot_interest = float(acc.get("total_interest_earned", 0.0) or 0.0)
    tot_tax = float(acc.get("total_tax_paid", 0.0) or 0.0)
    fy_exempt = float(acc.get("fy_ltcg_exempt_used", 0.0) or 0.0)
    positions = get_all_positions()

    invested_val = sum(p["qty"] * p["avg_price"] for p in positions)
    total_nav = cash + liquid_bal + invested_val

    # Calculate sector weights
    sector_weights: Dict[str, float] = {}
    for p in positions:
        sec = p.get("sector", "Other")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + (p["qty"] * p["avg_price"])
    if total_nav > 0:
        sector_weights = {k: round(v / total_nav, 4) for k, v in sector_weights.items()}

    current_prices = {p["ticker"]: p["avg_price"] for p in positions}
    if state.get("technicals") and "latest_close" in state["technicals"]:
        current_prices[ticker] = state["technicals"]["latest_close"]

    invested_val = sum(p["qty"] * (current_prices.get(p["ticker"]) or p["avg_price"]) for p in positions)
    total_nav = cash + liquid_bal + invested_val

    # Calculate sector weights
    sector_weights: Dict[str, float] = {}
    for p in positions:
        sec = p.get("sector", "Other")
        sector_weights[sec] = sector_weights.get(sec, 0.0) + (p["qty"] * (current_prices.get(p["ticker"]) or p["avg_price"]))
    if total_nav > 0:
        sector_weights = {k: round(v / total_nav, 4) for k, v in sector_weights.items()}

    as_of_date = state.get("point_in_time_date") or get_ist_now()
    tax_summary = format_portfolio_tax_summary(positions, current_prices, as_of_date, fy_exempt)
    treasury_str = (
        f"Liquid Fund Sweep: ₹{liquid_bal:,.2f} (Earning 6.75% p.a. risk-free yield) | "
        f"Cumulative Interest: ₹{tot_interest:,.2f} | Cumulative Tax Paid: ₹{tot_tax:,.2f}"
    )

    p_ctx = {
        "cash": round(cash, 2),
        "liquid_fund_balance": round(liquid_bal, 2),
        "total_nav": round(total_nav, 2),
        "invested_value": round(invested_val, 2),
        "total_interest_earned": round(tot_interest, 2),
        "total_tax_paid": round(tot_tax, 2),
        "active_positions": positions,
        "sector_weights": sector_weights,
        "tax_summary": tax_summary,
        "treasury_summary": treasury_str
    }

    # 3. Technicals & Risk Boundaries (Preserve Point-In-Time if pre-hydrated)
    if state.get("technicals") and "latest_close" in state["technicals"]:
        technicals = dict(state["technicals"])
        if "hard_stop_price" not in technicals:
            risk_bounds = calculate_risk_boundaries(
                latest_close=technicals["latest_close"],
                atr_14=technicals["atr_14"],
                total_nav=total_nav
            )
            technicals.update(risk_bounds)
    else:
        technicals = calculate_technical_indicators(ticker)
        risk_bounds = calculate_risk_boundaries(
            latest_close=technicals["latest_close"],
            atr_14=technicals["atr_14"],
            total_nav=total_nav
        )
        technicals.update(risk_bounds)

    # 4. Fundamentals (Preserve Point-In-Time if pre-hydrated or use backtest neutral proxy)
    pit_date = state.get("point_in_time_date") or ""
    if pit_date and str(pit_date) < "2025-11-01" and not state.get("fundamentals"):
        # Backtest Mode Proxy: Prevents 2026 yfinance info lookahead bias from corrupting historical backtests
        fundamentals = {
            "sector": state.get("sector") or "Industrials",
            "industry": "General",
            "market_cap_cr": 50000.0,
            "roe": 0.165,
            "debt_to_equity": 30.0,
            "revenue_growth": 0.15,
            "earnings_growth": 0.15,
            "pe_ratio": 22.0,
            "forward_pe": 18.0,
            "operating_cash_flow": 2000000000.0,
            "net_income": 1500000000.0,
            "promoter_holding_pct": 55.0,
            "backtest_mode": True
        }
    else:
        fundamentals = state.get("fundamentals") or fetch_fundamental_metrics(ticker)

    sector = state.get("sector") or fundamentals.get("sector", "Industrials")

    # 5. News & Calendar (Preserve Point-In-Time if pre-hydrated)
    news = state.get("recent_news") or fetch_recent_stock_news(ticker)

    # 6. Episodic Memory & Historical Scorecard (Balanced Dual-Polarity Retrieval)
    balanced_rules = get_regime_balanced_rules(sector=sector, market_regime=regime, limit=5)
    lessons_str = "\n".join([f"- {r}" for r in balanced_rules]) if balanced_rules else "No past trade failure rules recorded yet."
    scorecard_str = format_agent_scorecard_str()

    # 7. Deterministic Zero-Token Hard Veto Evaluation
    from state import TradeDecision
    latest_close = float(technicals.get("latest_close", 100.0))
    hard_stop = float(technicals.get("hard_stop_price", round(latest_close * 0.95, 2)))
    min_target = float(technicals.get("min_valid_target", round(latest_close * 1.15, 2)))
    sma_50 = float(technicals.get("sma_50", 0.0) or 0.0)
    industry = fundamentals.get("industry", "")
    is_financial = any(fin in sector.lower() or fin in industry.lower()
                       for fin in ["bank", "financial", "nbfc", "lending", "insurance"])
    earnings_growth = float(fundamentals.get("earnings_growth", 0.0) or 0.0)
    rs_alpha = float(technicals.get("rs_alpha_63d", 0.0) or 0.0)
    is_defensive = sector.lower() in ["healthcare", "pharmaceuticals", "pharma"] or "pharma" in industry.lower() or "defense" in industry.lower()
    has_strong_rs = rs_alpha >= 0.03 and float(fundamentals.get("operating_cash_flow", 0.0) or 0.0) > 0

    hard_veto_decision = None
    # Hard Veto 1: Severe non-financial earnings collapse (< -20%)
    if not is_financial and not (is_defensive and has_strong_rs) and earnings_growth < -0.20:
        logger.info(f"[Zero-Token Short-Circuit] Hard Veto 1 triggered for {ticker}: Severe earnings contraction ({earnings_growth*100:.1f}% < -20%).")
        hard_veto_decision = TradeDecision(
            ticker=ticker,
            action="HOLD",
            conviction="LOW",
            allocation_pct=0.0,
            entry_price=latest_close,
            stop_loss_price=hard_stop,
            target_price=min_target,
            holding_horizon_days=30,
            rationale=f"Deterministic Hard Veto: Severe earnings contraction of {earnings_growth*100:.1f}% (< -20%) violates capital preservation rules."
        )

    # Hard Veto 2: Bear Market breakdown below 50-day SMA
    bm_close = float(benchmark_metrics.get("nifty_close", 0.0) or 0.0)
    bm_sma50 = float(benchmark_metrics.get("nifty_sma50", 0.0) or 0.0)
    benchmark_downtrend = (bm_close > 0 and bm_sma50 > 0 and bm_close < bm_sma50) or (regime == "BEAR_RISK_OFF")
    is_structural_breakdown = (sma_50 > 0 and latest_close < sma_50 * 0.97) or (sma_50 > 0 and latest_close < sma_50 and rs_alpha <= 0.0)

    if not hard_veto_decision and benchmark_downtrend and is_structural_breakdown:
        logger.info(f"[Zero-Token Short-Circuit] Hard Veto 2 triggered for {ticker}: Benchmark downtrend and Close ₹{latest_close} < SMA_50 ₹{sma_50}.")
        hard_veto_decision = TradeDecision(
            ticker=ticker,
            action="HOLD",
            conviction="LOW",
            allocation_pct=0.0,
            entry_price=latest_close,
            stop_loss_price=hard_stop,
            target_price=min_target,
            holding_horizon_days=30,
            rationale=f"Deterministic Hard Veto: In benchmark downtrend, Close (₹{latest_close}) broke below 50-SMA (₹{sma_50}) without positive RS Alpha."
        )

    if hard_veto_decision:
        return {
            "ticker": ticker,
            "sector": sector,
            "market_regime": regime,
            "india_vix": india_vix,
            "benchmark_context": benchmark_metrics,
            "portfolio_context": p_ctx,
            "technicals": technicals,
            "fundamentals": fundamentals,
            "recent_news": news,
            "episodic_lessons": lessons_str,
            "agent_scorecard": scorecard_str,
            "tax_context": tax_summary,
            "treasury_context": treasury_str,
            "debate_round": 0,
            "bull_arguments": [],
            "bear_arguments": [],
            "hard_veto_triggered": True,
            "trade_decision": hard_veto_decision.model_dump(),
        }

    return {
        "ticker": ticker,
        "sector": sector,
        "market_regime": regime,
        "india_vix": india_vix,
        "benchmark_context": benchmark_metrics,
        "portfolio_context": p_ctx,
        "technicals": technicals,
        "fundamentals": fundamentals,
        "recent_news": news,
        "episodic_lessons": lessons_str,
        "agent_scorecard": scorecard_str,
        "tax_context": tax_summary,
        "treasury_context": treasury_str,
        "debate_round": 0,
        "bull_arguments": [],
        "bear_arguments": [],
        "hard_veto_triggered": False,
    }


def check_hard_veto(state: InvestmentFirmState) -> str:
    """Zero-Token Short-Circuit: routes directly to END if deterministic hard veto was triggered."""
    if state.get("hard_veto_triggered", False):
        return "END"
    return "technical_analyst"


def should_continue_debate(state: InvestmentFirmState) -> str:
    """Conditional router for the dialectical debate loop."""
    current_round = state.get("debate_round", 0)
    if current_round < MAX_DEBATE_ROUNDS:
        return "bull_researcher"
    return "research_manager"


def build_investment_graph(checkpointer: Optional[Any] = None):
    """Wires the sequential LangGraph StateGraph pipeline."""
    workflow = StateGraph(InvestmentFirmState)

    # Add Nodes
    workflow.add_node("data_and_memory", data_and_memory_node)

    # Stage 1: Analysts
    workflow.add_node("technical_analyst", technical_analyst_node)
    workflow.add_node("fundamental_analyst", fundamental_analyst_node)
    workflow.add_node("news_macro_analyst", news_macro_analyst_node)

    # Stage 2: Debate
    workflow.add_node("bull_researcher", bull_researcher_node)
    workflow.add_node("bear_researcher", bear_researcher_node)
    workflow.add_node("research_manager", research_manager_node)

    # Stage 3: Unified Risk Committee (Aggressive, Neutral, Conservative in 1 call)
    workflow.add_node("risk_committee", risk_committee_node)

    # Stage 4: Lead Portfolio Manager
    workflow.add_node("portfolio_manager", portfolio_manager_node)

    # Set Entry Point
    workflow.set_entry_point("data_and_memory")

    # Wire conditional edge from data_and_memory (Short-circuit directly to END on hard veto)
    workflow.add_conditional_edges(
        "data_and_memory",
        check_hard_veto,
        {
            "END": END,
            "technical_analyst": "technical_analyst"
        }
    )
    workflow.add_edge("technical_analyst", "fundamental_analyst")
    workflow.add_edge("fundamental_analyst", "news_macro_analyst")

    # Wire Stage 2
    workflow.add_edge("news_macro_analyst", "bull_researcher")
    workflow.add_edge("bull_researcher", "bear_researcher")
    workflow.add_conditional_edges(
        "bear_researcher",
        should_continue_debate,
        {
            "bull_researcher": "bull_researcher",
            "research_manager": "research_manager"
        }
    )

    # Wire Stage 3 (Unified Risk Committee)
    workflow.add_edge("research_manager", "risk_committee")

    # Wire Stage 4 (Portfolio Manager)
    workflow.add_edge("risk_committee", "portfolio_manager")
    workflow.add_edge("portfolio_manager", END)

    if checkpointer:
        return workflow.compile(checkpointer=checkpointer)
    return workflow.compile()


# Global compiled app
investment_app = build_investment_graph()


def run_investment_committee(
    ticker: str,
    sector: Optional[str] = None,
    point_in_time_state: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Runs the end-to-end committee evaluation for a single ticker."""
    ticker_norm = normalize_ticker(ticker)
    logger.info(f"--- Launching Investment Committee for {ticker_norm} ---")

    initial_state: InvestmentFirmState = {
        "ticker": ticker_norm,
        "sector": sector or "",
        "market_regime": "CHOPPY_RANGE",
        "india_vix": 16.0,
        "portfolio_context": {},
        "technicals": {},
        "fundamentals": {},
        "recent_news": "",
        "episodic_lessons": "",
        "agent_scorecard": "",
        "technical_report": None,
        "fundamental_report": None,
        "news_macro_report": None,
        "debate_round": 0,
        "bull_arguments": [],
        "bear_arguments": [],
        "research_proposal": None,
        "aggressive_risk_view": None,
        "neutral_risk_view": None,
        "conservative_risk_view": None,
        "risk_committee_verdict": None,
        "tax_context": None,
        "treasury_context": None,
        "trade_decision": None,
        "point_in_time_date": None,
        "hard_veto_triggered": False,
    }

    if point_in_time_state:
        for k, v in point_in_time_state.items():
            if v is not None:
                initial_state[k] = v

    final_state = investment_app.invoke(initial_state)
    logger.info(f"--- Committee finished for {ticker_norm}. Decision: {final_state.get('trade_decision')} ---")
    return final_state

