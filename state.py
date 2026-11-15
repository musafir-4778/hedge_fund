"""state.py — LangGraph TypedDict State & Strict Pydantic Output Schemas.

Defines the centralized multi-agent state container and the structured
decision and reflection models according to GEMINI.md specifications.
"""

from typing import TypedDict, Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field


class TradeDecision(BaseModel):
    """Structured output contract for the Lead Portfolio Manager."""
    ticker: str = Field(description="NSE ticker with .NS suffix, e.g., BEL.NS")
    action: Literal["BUY", "SELL", "HOLD"] = Field(description="Execution decision")
    conviction: Literal["HIGH", "MEDIUM", "LOW"] = Field(description="Conviction level")
    allocation_pct: float = Field(ge=0.0, le=0.12, description="Max 12% single-stock cap")
    entry_price: float = Field(description="Latest close / recommended entry price in INR")
    stop_loss_price: float = Field(description="Must be <= entry_price - 2 * ATR_14")
    target_price: float = Field(description="Must satisfy >= 2.5x risk-to-reward ratio")
    holding_horizon_days: int = Field(default=30, description="Max holding window before review")
    rationale: str = Field(description="Concise 3-sentence executive summary")


class TradePostMortem(BaseModel):
    """Structured output contract for the Post-Mortem Reflection Auditor."""
    outcome: Literal["WIN", "LOSS", "BREAKEVEN"] = Field(description="Trade outcome category")
    pnl_pct: float = Field(description="Net realized percentage gain/loss")
    primary_driver: Literal[
        "TECHNICAL_BREAKOUT_VALID",
        "FALSE_BREAKOUT_LOW_VOLUME",
        "FUNDAMENTAL_CATALYST_HIT",
        "EARNINGS_OR_SEBI_NEWS_SHOCK",
        "NIFTY_MACRO_SECTOR_DRAG",
        "STOP_LOSS_TOO_TIGHT"
    ] = Field(description="Attributed primary driver of trade outcome")
    accurate_agent: Literal[
        "TECHNICAL", "FUNDAMENTAL", "NEWS_MACRO", "BULL_BEAR_DEBATE", "RISK_COMMITTEE", "NONE"
    ] = Field(description="Agent whose thesis was most accurate")
    failed_agent: Literal[
        "TECHNICAL", "FUNDAMENTAL", "NEWS_MACRO", "BULL_BEAR_DEBATE", "RISK_COMMITTEE", "NONE"
    ] = Field(description="Agent whose thesis had the fatal blindspot")
    root_cause_analysis: str = Field(description="2-sentence diagnosis of why the thesis succeeded or failed.")
    actionable_rule: str = Field(description="Concrete 1-sentence guardrail rule for future committee votes.")


class RiskCommitteeVerdict(BaseModel):
    """Structured output contract for the Unified Risk Committee."""
    aggressive_view: str = Field(description="Aggressive risk critique evaluating alpha potential, RS momentum, and upside runway")
    neutral_view: str = Field(description="Neutral risk critique auditing sector concentration cap (30%) and cash liquidity")
    conservative_view: str = Field(description="Conservative CRO critique auditing 3xATR stop loss, 2.5:1 R:R, and bear market veto")
    recommended_action: Literal["BUY", "HOLD", "VETO"] = Field(description="Consensus risk recommendation")
    risk_approved: bool = Field(description="True if risk parameters satisfy institutional standards")
    hard_stop_price: float = Field(description="Exact verified hard stop-loss in INR")
    target_price: float = Field(description="Exact verified target price in INR")


class InvestmentFirmState(TypedDict):
    """Central state passed across all LangGraph nodes."""
    ticker: str
    sector: str
    market_regime: str                  # "BULL_STAGE2" | "CHOPPY_RANGE" | "BEAR_RISK_OFF"
    india_vix: float
    benchmark_context: Optional[Dict[str, Any]] # Raw Nifty 50 metrics: close, sma50, sma200, dist_52w, trend_63d
    portfolio_context: Dict[str, Any]   # Cash (INR), total NAV, active positions, sector weights
    technicals: Dict[str, Any]          # Pre-computed indicators + risk boundaries
    fundamentals: Dict[str, Any]        # Cached ratios + OCF/NetIncome quality check
    recent_news: str                    # Formatted headlines + upcoming quarterly results warning
    episodic_lessons: str               # Top 3 retrieved rules from trade_post_mortems
    agent_scorecard: str                # Rolling 30-trade accuracy string per agent
    
    # Tax & Treasury Context (Indian STCG/LTCG & Liquid Fund Sweep)
    tax_context: Optional[str]
    treasury_context: Optional[str]
    
    # Stage 1: Analyst Outputs
    technical_report: Optional[str]
    fundamental_report: Optional[str]
    news_macro_report: Optional[str]
    
    # Stage 2: Dialectical Debate State
    debate_round: int
    bull_arguments: List[str]
    bear_arguments: List[str]
    research_proposal: Optional[str]
    
    # Stage 3: Unified Risk Committee Outputs
    aggressive_risk_view: Optional[str]
    neutral_risk_view: Optional[str]
    conservative_risk_view: Optional[str]
    risk_committee_verdict: Optional[Dict[str, Any]]
    
    # Stage 4: Final Structured Decision
    trade_decision: Optional[Dict[str, Any]]

