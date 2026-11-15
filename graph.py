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
        if "max_loss_rupees" not in technicals:
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

    # 4. Fundamentals (Preserve Point-In-Time if pre-hydrated)
    fundamentals = state.get("fundamentals") or fetch_fundamental_metrics(ticker)
    sector = state.get("sector") or fundamentals.get("sector", "Industrials")

    # 5. News & Calendar (Preserve Point-In-Time if pre-hydrated)
    news = state.get("recent_news") or fetch_recent_stock_news(ticker)

    # 6. Episodic Memory & Historical Scorecard (Combined Sector-Specific + General Guardrails)
    general_rules = get_top_actionable_rules(limit=3)
    sector_rules = get_sector_actionable_rules(sector, limit=3)
    all_rules = list(dict.fromkeys(sector_rules + general_rules))
    lessons_str = "\n".join([f"- {r}" for r in all_rules]) if all_rules else "No past trade failure rules recorded yet."
    scorecard_str = format_agent_scorecard_str()

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
    }


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

    # Wire Stage 1 sequentially
    workflow.add_edge("data_and_memory", "technical_analyst")
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
    }

    if point_in_time_state:
        for k, v in point_in_time_state.items():
            if v is not None:
                initial_state[k] = v

    final_state = investment_app.invoke(initial_state)
    logger.info(f"--- Committee finished for {ticker_norm}. Decision: {final_state.get('trade_decision')} ---")
    return final_state

