"""agents.py — System Prompts & Node Implementations for Multi-Agent Committee.

Implements Analysts (Stage 1), Bull/Bear Debate (Stage 2), 3-Angle Risk (Stage 3),
and Lead Portfolio Manager (Stage 4) using paced Gemini calls.
"""

import json
import logging
from typing import Dict, Any
from config import (
    get_quick_think_llm,
    get_deep_think_llm,
    paced_llm_invoke,
    extract_text_content,
    MAX_SECTOR_EXPOSURE_PCT,
    MAX_SINGLE_STOCK_PCT,
    GOOGLE_API_KEY,
)
from state import InvestmentFirmState, TradeDecision, RiskCommitteeVerdict

logger = logging.getLogger("wealth_agent_office.agents")


def _is_mock_mode() -> bool:
    """Returns True if no Google API key is configured."""
    return not bool(GOOGLE_API_KEY)


# ---------------------------------------------------------
# Stage 1: Specialized Analyst Team
# ---------------------------------------------------------

def technical_analyst_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Audits Minervini Stage-2 trends, RSI, ATR, and volume confirmation."""
    ticker = state["ticker"]
    print(f"  -> [1/8] Technical Analyst auditing {ticker}...", flush=True)
    tech = state["technicals"]
    regime = state["market_regime"]
    vix = state["india_vix"]

    if _is_mock_mode():
        report = (
            f"Technical Audit for {ticker}: Latest Close ₹{tech.get('latest_close')}. "
            f"Stage-2 Uptrend: {tech.get('stage2_uptrend')}. RSI: {tech.get('rsi_14')} "
            f"(Sweet spot: 50-68). Volume Ratio: {tech.get('volume_ratio_20d')}x. "
            f"Circuit limit warning: {tech.get('circuit_limit_warning')}. "
            f"Market regime: {regime} with India VIX at {vix}."
        )
        return {"technical_report": report}

    bm = state.get("benchmark_context") or {}
    bm_str = f"Nifty 50: ₹{bm.get('nifty_close', 'N/A')} | 50 SMA: ₹{bm.get('nifty_sma50', 'N/A')} | 200 SMA: ₹{bm.get('nifty_sma200', 'N/A')} | India VIX: {vix}"

    prompt = f"""You are the Lead Technical Analyst for an Indian Wealth Management Office.
Analyze {ticker} using these deterministic indicators (zero speculation):
- Benchmark Context: {bm_str}
- Latest Close: ₹{tech.get('latest_close')}
- Moving Averages: SMA_20=₹{tech.get('sma_20')}, SMA_50=₹{tech.get('sma_50')}, SMA_150=₹{tech.get('sma_150')}, SMA_200=₹{tech.get('sma_200')}
- Wilder RSI (14): {tech.get('rsi_14')} (Momentum sweet spot: 50–68)
- ATR (14): ₹{tech.get('atr_14')}
- 20-Day Volume Ratio: {tech.get('volume_ratio_20d')}x
- 52-Week Distance: {tech.get('dist_52w_high_pct')}% from 52W High
- 63-Day RS Alpha vs Nifty 50: {tech.get('rs_alpha_63d')}
- Circuit Limit Warning: {tech.get('circuit_limit_warning')}
- Minervini Stage-2 Uptrend Criteria: {tech.get('stage2_uptrend')}

REGIME-ADAPTIVE TECHNICAL EVALUATION RULES:
1. Trend & Moving Average Structure:
   - In BULL_STAGE2: Look for classic Minervini Stage-2 alignment (Close > SMA_50 > SMA_150 > SMA_200).
   - In BEAR_RISK_OFF or Early Recovery (Nifty < 200-SMA or 50-SMA): Moving averages are lagging indicators. Do NOT disqualify a stock solely because SMA_50 < SMA_200 if the stock is trading above its own rising SMA_20 and SMA_50 with positive 63-day RS Alpha (> +0.02). Classify this as an "Early Stage-1/2 Turnaround Leader".
2. Dynamic Volume Confirmation:
   - Standard Breakout Hurdle: Volume Ratio >= 1.15x.
   - Quiet Institutional Accumulation Exception: During market pullbacks, consolidations, or early relief rallies, institutional buying is absorbed quietly. Accept Volume Ratio >= 0.80x as constructive if 63-day RS Alpha > 0.0 and Wilder RSI is in the 45–68 accumulation zone.
3. Overextension Leeway:
   - Allow price extension up to 25-30% above SMA_50 for structural momentum leaders before flagging exhaustion.

Write a concise 3-bullet technical audit:
1. Regime-adjusted trend structure & Minervini Stage 2 status relative to broader benchmark.
2. Volume breakout confirmation, relative strength alpha, and momentum health (Wilder RSI).
3. ATR volatility, risk boundary & circuit limit assessment. Keep concise and factual."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    report = extract_text_content(response)
    return {"technical_report": report}


def fundamental_analyst_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Audits cash flow conversion, ROE, valuation, debt safety, and promoter holding."""
    ticker = state["ticker"]
    print(f"  -> [2/8] Fundamental Analyst auditing {ticker}...", flush=True)
    fund = state["fundamentals"]
    sector = state["sector"]
    industry = fund.get("industry", "")

    # Exemption for financial institutions (banks, NBFCs, lenders)
    is_financial = any(fin in (sector or "").lower() or fin in (industry or "").lower()
                       for fin in ["bank", "financial", "nbfc", "lending", "insurance"])

    if _is_mock_mode():
        report = (
            f"Fundamental Audit for {ticker} ({sector}): Market Cap ₹{fund.get('market_cap_cr')} Cr. "
            f"ROE: {round(fund.get('roe', 0)*100, 1)}%. Debt/Equity: {fund.get('debt_to_equity')}%. "
            f"PE: {fund.get('pe_ratio')} vs Forward PE: {fund.get('forward_pe')}. "
            f"OCF: ₹{fund.get('operating_cash_flow'):,.0f} vs Net Income: ₹{fund.get('net_income'):,.0f}."
        )
        return {"fundamental_report": report}

    lessons = state.get("episodic_lessons", "")

    if is_financial:
        accounting_rule = (
            "1. Financial Institution Accounting Exemption: DO NOT apply industrial CFO/PAT > 0.8 test. "
            "In banking/lending, loan disbursements are classified as operating cash outflows, making CFO naturally negative during expansion. "
            "Instead audit ROE (target >= 14%), Net Interest / Earnings Growth (target >= 15%), and PE/PB valuation reasonableness."
        )
    else:
        accounting_rule = (
            "1. Cash Conversion Quality: Strictly check OCF / Net Income (CFO/PAT >= 0.8) to detect aggressive accruals. "
            "Flag governance/accounting distortion if CFO/PAT < 0.8."
        )

    prompt = f"""You are the Lead Fundamental Analyst for an Indian Wealth Management Office.
Analyze {ticker} ({sector}) using these deterministic fundamental metrics:
- Sector / Industry: {fund.get('sector')} / {industry}
- Market Cap: ₹{fund.get('market_cap_cr')} Crores
- Return on Equity (ROE): {round(fund.get('roe', 0.0) * 100, 2)}%
- Debt-to-Equity: {fund.get('debt_to_equity')}%
- Revenue Growth: {round(fund.get('revenue_growth', 0.0) * 100, 2)}%
- Earnings Growth: {round(fund.get('earnings_growth', 0.0) * 100, 2)}%
- PE Ratio: {fund.get('pe_ratio')} | Forward PE: {fund.get('forward_pe')}
- Operating Cash Flow: ₹{fund.get('operating_cash_flow'):,.0f}
- Net Income: ₹{fund.get('net_income'):,.0f}
- Promoter Holding: {fund.get('promoter_holding_pct')}%

Mandatory Lessons Learned from Firm's Past Mistakes (Strict Guardrails):
\"\"\"
{lessons}
\"\"\"

Audit Requirements:
{accounting_rule}
2. Contextual Rule Application: Treat firm past-mistake rules as risk flags (conviction adjustments), NOT automatic hard vetoes, unless non-financial earnings growth is collapsing (< -20%) alongside negative operating cash flow.
3. Forward Earnings Trajectory: If Forward PE is lower than Trailing PE (indicating expected earnings expansion) and ROE >= 14%, highlight positive operating leverage.
4. Solvency & Promoter Skin-in-the-Game: Flag red flags only if Debt/Equity > 150% (non-financial) or promoter holding is < 25%. Keep concise and factual."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    report = extract_text_content(response)
    return {"fundamental_report": report}


def news_macro_analyst_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Evaluates earnings calendar blackout, macro tailwinds, and news flow."""
    ticker = state["ticker"]
    print(f"  -> [3/8] Macro & News Catalyst Analyst evaluating {ticker}...", flush=True)
    news = state["recent_news"]
    regime = state["market_regime"]
    vix = state["india_vix"]

    if _is_mock_mode():
        report = (
            f"News & Macro Audit for {ticker}: Nifty Regime: {regime}, India VIX: {vix}. "
            f"Recent headlines: {news[:200]}."
        )
        return {"news_macro_report": report}

    bm = state.get("benchmark_context") or {}
    bm_str = (
        f"Nifty 50 Close: ₹{bm.get('nifty_close', 'N/A')} | "
        f"50-day SMA: ₹{bm.get('nifty_sma50', 'N/A')} | "
        f"200-day SMA: ₹{bm.get('nifty_sma200', 'N/A')} | "
        f"63-day Benchmark Trend: {bm.get('nifty_trend_63d_pct', 0.0):+.1f}% | "
        f"India VIX: {vix}"
    )

    prompt = f"""You are the Indian Macro & News Catalyst Analyst for an Indian Wealth Management Office.
Analyze news flow and macroeconomic/benchmark context for {ticker}:
- Benchmark Macro Indicators: {bm_str}
- Recent Headlines & Calendar Events:
\"\"\"
{news}
\"\"\"

Audit Requirements:
1. Nuanced Regime Classification: Classify the macro environment into one of FOUR operational states based on Nifty 50 SMAs, 63d trend, and India VIX:
   - CONFIRMED_BULL_STAGE2 (Nifty > 50-SMA > 200-SMA, VIX < 18): Full risk-on deployment.
   - EARLY_RECOVERY_PIVOT (Nifty below 200-SMA or 50-SMA, but stabilizing or stock exhibits strong positive RS Alpha decoupling): Selective accumulation allowed at 50-60% position size.
   - CHOPPY_ROTATION (Rangebound index): Focus strictly on sector-specific relative strength leaders.
   - ACTIVE_LIQUIDATION_BEAR (Sharp index breakdown with VIX > 22): Defensive posture + 6.75% Liquid Fund parking.
2. Earnings Blackout Check: Flag caution only if quarterly results are confirmed within < 5 trading days.
3. Domestic Tailwinds: Highlight PLI, defense/rail/infra capex, or defensive pharma export tailwinds. Keep concise and factual."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    report = extract_text_content(response)
    return {"news_macro_report": report}


# ---------------------------------------------------------
# Stage 2: Dialectical Bull/Bear Debate
# ---------------------------------------------------------

def bull_researcher_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Constructs high-conviction bullish thesis grounded in Stage 1 reports."""
    ticker = state["ticker"]
    print(f"  -> [4/8] Bull Researcher building upside thesis for {ticker}...", flush=True)
    tech_rep = state.get("technical_report", "")
    fund_rep = state.get("fundamental_report", "")
    news_rep = state.get("news_macro_report", "")
    past_bears = state.get("bear_arguments", [])

    if _is_mock_mode():
        arg = f"Bull Thesis for {ticker}: Strong Stage-2 momentum, sound fundamentals, and high relative strength."
        args = list(state.get("bull_arguments", [])) + [arg]
        return {"bull_arguments": args, "debate_round": state.get("debate_round", 0) + 1}

    prompt = f"""You are the Bull Equity Researcher in a high-conviction Indian Wealth Management Office.
Build the highest-conviction bullish case for {ticker} based strictly on these analyst reports:
- Technical Report: {tech_rep}
- Fundamental Report: {fund_rep}
- News/Macro Report: {news_rep}
- Previous Bear Rebuttals (if any): {past_bears}

ANTI-PARALYSIS MANDATE:
- Challenge any Bear argument that relies on lagging moving-average filters or rigid volume gates during market recoveries.
- Emphasize the Opportunity Cost of sitting in 6.75% liquid funds when a stock demonstrates positive 63-day RS Alpha and asymmetric >= 2.5:1 reward-to-risk.
- If the stock exhibits confirmed Stage-2 trend alignment, healthy cash generation, and positive RS Alpha, argue vigorously against capping runners or enforcing premature liquidation.

Deliver a compelling 3-point thesis detailing catalyst drivers, growth acceleration, and upside runway."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    arg = extract_text_content(response)
    new_args = list(state.get("bull_arguments", [])) + [arg]
    return {"bull_arguments": new_args, "debate_round": state.get("debate_round", 0) + 1}


def bear_researcher_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Red-teams the bullish case, citing past post-mortems and structural risks."""
    ticker = state["ticker"]
    print(f"  -> [5/8] Bear Researcher red-teaming vulnerabilities for {ticker}...", flush=True)
    bull_args = state.get("bull_arguments", [])
    lessons = state.get("episodic_lessons", "")
    fund_rep = state.get("fundamental_report", "")
    tech_rep = state.get("technical_report", "")

    if _is_mock_mode():
        arg = f"Bear Counter for {ticker}: Check valuation stretch and market regime risk off."
        args = list(state.get("bear_arguments", [])) + [arg]
        return {"bear_arguments": args}

    prompt = f"""You are the Bear Equity Researcher (Chief Red-Teamer) in an Indian Wealth Management Office.
Attack and stress-test the bullish thesis for {ticker}.
- Bull's Arguments: {bull_args}
- Fundamental Report: {fund_rep}
- Technical Report: {tech_rep}
- Past Post-Mortem Failure Lessons Learned by this firm:
\"\"\"
{lessons}
\"\"\"

PROPORTIONAL RED-TEAM MANDATE:
- Only cite past post-mortem rules from `episodic_lessons` that match the CURRENT market regime and sector.
- Acknowledge when a risk can be mitigated via reduced position sizing (e.g., 5-6% NAV instead of 12% NAV) rather than demanding a 0% HOLD veto.
- Identify the 3 most fatal blind spots: valuation trap, earnings cyclicality, governance risks, or false breakout risks."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    arg = extract_text_content(response)
    new_args = list(state.get("bear_arguments", [])) + [arg]
    return {"bear_arguments": new_args}


def research_manager_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Adjudicates Bull/Bear debate using deep_think_llm (Pro with Flash fallback)."""
    ticker = state["ticker"]
    print(f"  -> [6/8] Research Manager synthesizing debate for {ticker}...", flush=True)
    bulls = state.get("bull_arguments", [])
    bears = state.get("bear_arguments", [])
    scorecard = state.get("agent_scorecard", "")

    if _is_mock_mode():
        proposal = (
            f"Research Manager Synthesis for {ticker}: Bull momentum confirmed, "
            f"bear cautions noted. Proceed to Risk Committee with moderate conviction."
        )
        return {"research_proposal": proposal}

    lessons = state.get("episodic_lessons", "")

    prompt = f"""You are the Research Manager for an Indian Wealth Management Office.
Synthesize the dialectical debate for {ticker}:
- Agent Scorecard (Historical Hit Rates): {scorecard}
- Historical Guardrail Rules (Learned from Past Trade Failures):
\"\"\"
{lessons}
\"\"\"
- Bull Case: {bulls}
- Bear Case: {bears}

CONVICTION SCORING & GUARDRAIL RESOLUTION RUBRIC:
Start with a base score of 50 and adjust:
- +20 pts if 63-day RS Alpha vs Nifty 50 > +0.02 (decoupled leadership).
- +15 pts if Stage-2 aligned OR reclaiming 50-SMA during an index pullback.
- +15 pts if ROE >= 15% (or >= 14% for Financials) with positive OCF/earnings trajectory.
- -15 pts (Soft Penalty, NOT a hard veto) if 20d Volume Ratio is between 0.80x and 1.15x.
- -20 pts if benchmark is below 200-SMA (mitigated by sizing at 5-6% NAV rather than vetoing).

If Final Conviction Score >= 60: Recommend PROCEED (BUY with appropriate regime sizing).
If Final Conviction Score < 60: Recommend VETO (HOLD).

REGIME-ADAPTIVE SYNTHESIS INSTRUCTIONS:
- Soft Conviction Penalties over Binary Bans: Do NOT issue binary hard vetoes solely based on past trade scars. If a historical post-mortem rule is triggered, treat it as a conviction deduction rather than an absolute roadblock. Allow a HIGH-conviction override if 63-day RS Alpha exceeds +0.05 and the setup demonstrates strong earnings quality.
- Opportunity Cost vs Inaction Bias: Actively weigh opportunity cost against capital preservation. A static post-mortem rule must NOT override an asymmetric setup satisfying >= 2.5:1 Risk-to-Reward and strong relative strength alpha (> 0.03).
- Resolve Rule Conflict: If the benchmark is below its 50/200 SMA, Rule #137 (50% position sizing, max 6% NAV) takes precedence over Rule #136 (blanket veto) for stocks with positive 63d RS Alpha.
- Early Rebound Volume: In market consolidations or relief rebounds (Nifty RSI < 42 or below moving averages), accept 20-day volume ratio >= 0.80x-0.85x for high relative strength leaders rather than strictly requiring 1.2x.
- Dynamic Overextension: Allow 25–35% extension above 50-SMA for structural compounders with institutional accumulation.

Deliver an executive synthesis:
1. Final Conviction Score (0–100) with detailed rubric points breakdown.
2. Verdict: Does the bullish edge outweigh the red-team risks and comply with past mistake guardrails?
3. Decisive Catalyst or Vulnerability.
4. Recommended Stance for the Risk Committee (Proceed / Veto)."""

    llm = get_deep_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt, is_deep_think=True)
    proposal = extract_text_content(response)
    return {"research_proposal": proposal}


# ---------------------------------------------------------
# Stage 3: Unified Risk Committee (Consolidated Node)
# ---------------------------------------------------------

def risk_committee_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Unified Stage 3 Risk Committee: Audits Aggressive, Neutral, and Conservative risk angles in a single call."""
    ticker = state["ticker"]
    print(f"  -> [7/8] Risk Committee (Aggressive, Neutral, Conservative) auditing {ticker}...", flush=True)
    proposal = state.get("research_proposal", "")
    tech = state["technicals"]
    sector = state["sector"]
    p_ctx = state["portfolio_context"]
    regime = state["market_regime"]
    vix = state["india_vix"]

    latest_close = float(tech.get("latest_close", 100.0))
    hard_stop = float(tech.get("hard_stop_price", round(latest_close * 0.95, 2)))
    min_target = float(tech.get("min_valid_target", round(latest_close * 1.15, 2)))
    risk_per_share = float(tech.get("risk_per_share", round(latest_close - hard_stop, 2)))
    sector_exposure = p_ctx.get("sector_weights", {}).get(sector, 0.0)

    if _is_mock_mode():
        verdict = RiskCommitteeVerdict(
            aggressive_view=f"Aggressive View: Strong RS Alpha ({tech.get('rs_alpha_63d')}) justifies deployment.",
            neutral_view=f"Neutral View: Sector '{sector}' exposure is {sector_exposure*100:.1f}% (cap: 30%). Compliant.",
            conservative_view=f"Conservative View: Hard stop at ₹{hard_stop} (risk ₹{risk_per_share}/share). Regime {regime}.",
            recommended_action="BUY" if regime != "BEAR_RISK_OFF" else "HOLD",
            risk_approved=regime != "BEAR_RISK_OFF",
            hard_stop_price=hard_stop,
            target_price=min_target
        )
        return {
            "aggressive_risk_view": verdict.aggressive_view,
            "neutral_risk_view": verdict.neutral_view,
            "conservative_risk_view": verdict.conservative_view,
            "risk_committee_verdict": verdict.model_dump()
        }

    bm = state.get("benchmark_context") or {}
    bm_str = f"Nifty 50: ₹{bm.get('nifty_close', 'N/A')} | 50 SMA: ₹{bm.get('nifty_sma50', 'N/A')} | 200 SMA: ₹{bm.get('nifty_sma200', 'N/A')} | 63d Trend: {bm.get('nifty_trend_63d_pct', 0.0):+.1f}% | India VIX: {vix}"

    prompt = f"""You are the Unified Risk Committee of an Indian Wealth Management Office.
You must conduct a three-angle risk audit on the trade proposal for {ticker}:

1. Aggressive Risk View (Alpha Maximizer):
   - Relative Strength: 63-Day RS Alpha = {tech.get('rs_alpha_63d')}
   - Volume Confirmation: 20-Day Volume Ratio = {tech.get('volume_ratio_20d')}x
   - Evaluate whether momentum, trend template, and sector tailwinds justify high conviction.

2. Neutral Risk View (Portfolio Structurer):
   - Sector: {sector} | Current Sector Exposure: {sector_exposure*100:.1f}% (Hard Cap: {MAX_SECTOR_EXPOSURE_PCT*100}%)
   - Active Positions: {len(p_ctx.get('active_positions', []))}
   - Available Cash: ₹{p_ctx.get('cash', 0):,.2f} | NAV: ₹{p_ctx.get('total_nav', 0):,.2f}
   - Treasury & Sweep Status: {state.get('treasury_context', 'Standard liquidity')}
   - Verify sector concentration compliance, liquidity preservation, and risk-free parking yield.

3. Conservative CRO View (Capital Protector & Sizing Auditor):
   - Benchmark Indicators: {bm_str}
   - ANCHORED CURRENT MARKET PRICE (CMP): ₹{latest_close}
   - ANCHORED HARD STOP-LOSS: ₹{hard_stop} (Pre-calculated beta-adaptive 2.5-3.0x ATR_14, exact risk per share = ₹{risk_per_share})
   - MINIMUM VALID TARGET: ₹{min_target} (Satisfies >= 2.5:1 Risk-to-Reward)
   - Circuit Warning: {tech.get('circuit_limit_warning')}
   - CRITICAL CRO INSTRUCTION: DO NOT recalculate or hallucinate new price numbers. You MUST use hard_stop_price = {hard_stop} and target_price >= {min_target}.
   - ANTI-SCLEROSIS MANDATE: Your job is to price and size risk, NOT to paralyze the firm into 0% equity exposure during recoveries.
   - If the stock has positive 63-day RS Alpha (> 0.0) and satisfies the 2.5:1 Risk-to-Reward ratio at ₹{hard_stop}, you MUST set risk_approved = True and recommended_action = "BUY".
   - If the benchmark is below its 50-SMA or 200-SMA, manage risk by specifying a 50% reduced allocation (5%–6% NAV) in your conservative_view rather than vetoing the trade.
   - Preventing Inaction Bias & Weighing Opportunity Cost: Actively weigh opportunity cost against capital preservation. A static post-mortem rule must NOT override an asymmetric setup satisfying a strong >= 2.5:1 risk-to-reward ratio and a decisive relative strength advantage (63-day RS Alpha > 0.03).
   - Dynamic Overextension: Allow overextension ceilings above the 50-day SMA to expand from 20% up to 30–35% for proven multi-quarter compounders when market breadth indicators show expanding participation.

Research Proposal Under Review:
\"\"\"
{proposal}
\"\"\"

Output your combined audit adhering to the strict RiskCommitteeVerdict schema."""

    llm = get_deep_think_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(RiskCommitteeVerdict)
    try:
        verdict = paced_llm_invoke(structured_llm, prompt, is_deep_think=True, schema=RiskCommitteeVerdict)
        if isinstance(verdict, RiskCommitteeVerdict):
            # Guardrails to prevent any numeric hallucination
            verdict.hard_stop_price = hard_stop
            if verdict.target_price < min_target:
                verdict.target_price = min_target
            return {
                "aggressive_risk_view": verdict.aggressive_view,
                "neutral_risk_view": verdict.neutral_view,
                "conservative_risk_view": verdict.conservative_view,
                "risk_committee_verdict": verdict.model_dump()
            }
        return {
            "aggressive_risk_view": str(verdict),
            "neutral_risk_view": str(verdict),
            "conservative_risk_view": str(verdict),
            "risk_committee_verdict": None
        }
    except Exception as e:
        logger.error(f"Error in unified risk committee for {ticker}: {e}")
        fallback = RiskCommitteeVerdict(
            aggressive_view=f"Fallback aggressive review due to parsing error: {e}",
            neutral_view=f"Fallback neutral review: Sector weight {sector_exposure*100:.1f}%.",
            conservative_view=f"Fallback conservative review: Stop loss ₹{hard_stop}.",
            recommended_action="HOLD",
            risk_approved=False,
            hard_stop_price=hard_stop,
            target_price=min_target
        )
        return {
            "aggressive_risk_view": fallback.aggressive_view,
            "neutral_risk_view": fallback.neutral_view,
            "conservative_risk_view": fallback.conservative_view,
            "risk_committee_verdict": fallback.model_dump()
        }


# ---------------------------------------------------------
# Legacy Individual Risk Nodes (Preserved for Backwards Compatibility)
# ---------------------------------------------------------

def aggressive_risk_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Advocates maximizing upside when relative strength and catalysts align."""
    ticker = state["ticker"]
    print(f"  -> Aggressive Risk Officer auditing {ticker}...", flush=True)
    proposal = state.get("research_proposal", "")
    tech = state["technicals"]

    if _is_mock_mode():
        view = f"Aggressive Risk View for {ticker}: RS Alpha is positive ({tech.get('rs_alpha_63d')}). Full size justified if approved."
        return {"aggressive_risk_view": view}

    prompt = f"""You are the Aggressive Risk Manager (Alpha Maximizer).
Evaluate {ticker} with:
- Proposal: {proposal}
- 63-Day RS Alpha vs Nifty 50: {tech.get('rs_alpha_63d')}
- 20-Day Volume Ratio: {tech.get('volume_ratio_20d')}x
Argue for maximum allowed capital allocation if strong alpha exists."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    view = extract_text_content(response)
    return {"aggressive_risk_view": view}


def neutral_risk_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Audits portfolio concentration, sector caps (30%), and liquidity."""
    ticker = state["ticker"]
    print(f"  -> Neutral Risk Officer auditing concentration for {ticker}...", flush=True)
    sector = state["sector"]
    p_ctx = state["portfolio_context"]

    sector_exposure = p_ctx.get("sector_weights", {}).get(sector, 0.0)

    if _is_mock_mode():
        view = f"Neutral Risk View for {ticker}: Current sector '{sector}' weight is {sector_exposure*100:.1f}% (cap: 30%). Compliant."
        return {"neutral_risk_view": view}

    prompt = f"""You are the Neutral Risk Officer (Portfolio Structurer).
Audit {ticker} against concentration rules:
- Sector: {sector} (Current Exposure: {sector_exposure*100:.1f}%, Hard Cap: {MAX_SECTOR_EXPOSURE_PCT*100}%)
- Active Positions: {len(p_ctx.get('active_positions', []))}
- Available Cash: ₹{p_ctx.get('cash', 0):,.2f}

Verify that adding {ticker} respects the 30% sector cap and provides adequate portfolio diversification."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    view = extract_text_content(response)
    return {"neutral_risk_view": view}


def conservative_risk_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Enforces stop-loss, 2.5:1 R:R, and vetoes earnings blackouts / bear regimes."""
    ticker = state["ticker"]
    print(f"  -> Conservative Risk Officer auditing stop-loss & risk/reward for {ticker}...", flush=True)
    regime = state["market_regime"]
    vix = state["india_vix"]
    tech = state["technicals"]

    if _is_mock_mode():
        view = f"Conservative Risk View for {ticker}: Stop-loss at ₹{tech.get('hard_stop_price')}, target ₹{tech.get('min_valid_target')}. Regime {regime}."
        return {"conservative_risk_view": view}

    bm = state.get("benchmark_context") or {}
    bm_str = f"Nifty 50: ₹{bm.get('nifty_close', 'N/A')} | 50 SMA: ₹{bm.get('nifty_sma50', 'N/A')} | 200 SMA: ₹{bm.get('nifty_sma200', 'N/A')} | India VIX: {vix}"

    prompt = f"""You are the Conservative Risk Officer (Chief Risk Officer & Capital Protector).
Audit {ticker} with strict risk veto rules:
- Benchmark Indicators: {bm_str}
- Pre-Calculated Hard Stop: ₹{tech.get('hard_stop_price')}
- Minimum Valid Target: ₹{tech.get('min_valid_target')} (>= 2.5:1 R:R)
- Circuit Limit Warning: {tech.get('circuit_limit_warning')}

Evaluate:
1. Is the 2.5:1 Risk-to-Reward ratio strictly achievable?
2. Do broader benchmark indicators (e.g. Nifty below moving averages or elevated VIX) justify an outright VETO? Keep concise."""

    llm = get_quick_think_llm(temperature=0.1)
    response = paced_llm_invoke(llm, prompt)
    view = extract_text_content(response)
    return {"conservative_risk_view": view}


# ---------------------------------------------------------
# Stage 4: Lead Portfolio Manager
# ---------------------------------------------------------

def portfolio_manager_node(state: InvestmentFirmState) -> Dict[str, Any]:
    """Synthesizes all inputs and outputs strict structured TradeDecision using deep_think_llm."""
    ticker = state["ticker"]
    
    # Zero-Token Short-Circuit Check
    if state.get("hard_veto_triggered") and state.get("trade_decision"):
        return {"trade_decision": state["trade_decision"]}

    tech = state["technicals"]
    regime = state["market_regime"]
    scorecard = state.get("agent_scorecard", "")
    proposal = state.get("research_proposal", "")
    p_ctx = state["portfolio_context"]
    fund = state.get("fundamentals", {})
    sector = state.get("sector", "")
    industry = fund.get("industry", "")

    latest_close = float(tech.get("latest_close", 100.0))
    hard_stop = float(tech.get("hard_stop_price", round(latest_close * 0.95, 2)))
    min_target = float(tech.get("min_valid_target", round(latest_close * 1.15, 2)))
    sma_50 = float(tech.get("sma_50", 0.0) or 0.0)

    # Financial institution detection
    is_financial = any(fin in sector.lower() or fin in industry.lower()
                       for fin in ["bank", "financial", "nbfc", "lending", "insurance"])
    earnings_growth = float(fund.get("earnings_growth", 0.0) or 0.0)

    # Deterministic Hard Vetoes (backup if short-circuit did not trigger)
    is_defensive = sector.lower() in ["healthcare", "pharmaceuticals", "pharma"] or "pharma" in industry.lower() or "defense" in industry.lower()
    has_strong_rs = float(tech.get("rs_alpha_63d", 0.0) or 0.0) >= 0.03 and float(fund.get("operating_cash_flow", 0.0) or 0.0) > 0
    if not is_financial and not (is_defensive and has_strong_rs) and earnings_growth < -0.20:
        decision = TradeDecision(
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
        return {"trade_decision": decision.model_dump()}

    bm = state.get("benchmark_context") or {}
    bm_close = float(bm.get("nifty_close", 0.0) or 0.0)
    bm_sma50 = float(bm.get("nifty_sma50", 0.0) or 0.0)
    benchmark_downtrend = (bm_close > 0 and bm_sma50 > 0 and bm_close < bm_sma50) or (regime == "BEAR_RISK_OFF")
    rs_alpha = float(tech.get("rs_alpha_63d", 0.0) or 0.0)
    is_structural_breakdown = (sma_50 > 0 and latest_close < sma_50 * 0.97) or (sma_50 > 0 and latest_close < sma_50 and rs_alpha <= 0.0)

    if benchmark_downtrend and is_structural_breakdown:
        decision = TradeDecision(
            ticker=ticker,
            action="HOLD",
            conviction="LOW",
            allocation_pct=0.0,
            entry_price=latest_close,
            stop_loss_price=hard_stop,
            target_price=min_target,
            holding_horizon_days=30,
            rationale=f"Deterministic Hard Veto: In benchmark corrective downtrend, price ₹{latest_close} is below 50 SMA ₹{sma_50}."
        )
        return {"trade_decision": decision.model_dump()}

    print(f"  -> [8/8] Lead Portfolio Manager synthesizing & issuing final order for {ticker}...", flush=True)

    # Fallback / Mock Mode
    if _is_mock_mode():
        action = "BUY" if (tech.get("stage2_uptrend") and regime != "BEAR_RISK_OFF") else "HOLD"
        alloc = min(MAX_SINGLE_STOCK_PCT, 0.08) if action == "BUY" else 0.0
        decision = TradeDecision(
            ticker=ticker,
            action=action,
            conviction="HIGH" if action == "BUY" else "LOW",
            allocation_pct=alloc,
            entry_price=latest_close,
            stop_loss_price=hard_stop,
            target_price=min_target,
            holding_horizon_days=30,
            rationale=f"Automated evaluation for {ticker}: Regime {regime}, Stage-2={tech.get('stage2_uptrend')}."
        )
        return {"trade_decision": decision.model_dump()}

    lessons = state.get("episodic_lessons", "")
    risk_verdict = state.get("risk_committee_verdict") or {}
    verdict_summary = json.dumps(risk_verdict, indent=2) if risk_verdict else "Unified risk audit complete."

    prompt = f"""You are the Lead Portfolio Manager of an Indian Wealth Management Office.
You have the final authority to issue an execution order for {ticker}.

Mandatory Rules:
1. Weight analyst and debate claims using historical hit rates from the agent scorecard:
   {scorecard}
2. Institutional Memory Guardrails (Rules Learned From Past Trade Failures):
\"\"\"
{lessons}
\"\"\"
3. Capital Preservation Mandate & Regime-Adaptive Leeway:
   - Opportunity Cost & Inaction Bias: Actively weigh opportunity cost against capital preservation. A static post-mortem rule must NOT override an asymmetric setup satisfying >= 2.5:1 Risk-to-Reward and strong relative strength alpha (> 0.03).
   - Rule #136 vs Rule #137 Conflict Resolution: If the benchmark index is below its 50 or 200 SMA, do NOT freeze into complete inaction. Follow Rule #137: you MAY allocate to high-conviction decoupled leaders (positive 63d RS Alpha > 0.0) or defensive setups (Healthcare/Pharma/Defense) at 50% standard position size (allocation_pct between 0.04 and 0.06, max 6% NAV), while maintaining >= 50% cash/liquid fund preservation. Rule #137 supersedes Rule #136's blanket ban when an asset shows confirmed Stage-2 structure or positive RS Alpha.
   - Early Rebound / Relief Leeway on Volume (Rule #138): In corrective markets or early V-shaped relief rebounds (Nifty below 200 SMA or RSI < 42), initial institutional accumulation occurs on light volume before broad participation. For top RS Alpha leaders, relax the volume threshold from 1.2x to 0.85x. Do NOT veto decoupled leaders simply because volume is between 0.85x and 1.2x during market turnarounds.
   - Dynamic Overextension Leeway: For structural compounders with multi-quarter accumulation, permit up to 30–35% extension above 50-SMA if volume ratio >= 1.0x, rather than premature profit truncation at 20%.
4. Hard Stops: NEVER set stop_loss_price above ₹{hard_stop} (Beta-adaptive 2.5-3.0x ATR_14).
5. Asymmetric Target: target_price MUST satisfy at least ₹{min_target} (>= 2.5:1 Risk-to-Reward).
6. Sizing Tiers & Capital Allocation:
   - Tier 1 (CONFIRMED_BULL_STAGE2 + High Conviction): allocation_pct = 0.09 to 0.12 (9%–12% NAV).
   - Tier 2 (CHOPPY_RANGE / EARLY_RECOVERY_PIVOT / Nifty < 200-SMA + Decoupled RS Leader): allocation_pct = 0.05 to 0.06 (5%–6% NAV).
   - Tier 3 (Low Conviction / Negative RS Alpha in Downtrend): action = "HOLD", allocation_pct = 0.0 (parked in 6.75% Liquid Fund).
7. Tax & Treasury Awareness (Indian Union Budget Rules):
   - Liquid Fund Yield: Unallocated cash in risk-off market conditions automatically sweeps into 6.75% Liquid Mutual Funds. Holding liquid yield beats chasing low-conviction setups, but high-alpha decoupled leaders should be captured to beat the market.
   - Long-Term Compounding vs STCG Friction: Active positions approaching the 12-month boundary (>= 320 days) trigger 12.5% LTCG + ₹1.25L exemption if held, saving 7.5% in tax vs premature liquidation at 20% STCG.

Inputs:
- Ticker: {ticker}
- Sector: {sector}
- Benchmark Indicators: Nifty 50: ₹{bm.get('nifty_close', 'N/A')} (50 SMA: ₹{bm.get('nifty_sma50', 'N/A')}, 200 SMA: ₹{bm.get('nifty_sma200', 'N/A')}, 63d Trend: {bm.get('nifty_trend_63d_pct', 0.0):+.1f}%) | India VIX: {state['india_vix']}
- Pre-Calculated Risk Boundaries: Entry=₹{latest_close}, Stop=₹{hard_stop}, MinTarget=₹{min_target}
- Research Proposal: {proposal}
- Risk Committee Verdict:
{verdict_summary}
- Aggressive Risk View: {state.get('aggressive_risk_view', '')}
- Neutral Risk View: {state.get('neutral_risk_view', '')}
- Conservative Risk View: {state.get('conservative_risk_view', '')}
- Portfolio Cash: ₹{p_ctx.get('cash', 0):,.2f} | NAV: ₹{p_ctx.get('total_nav', 0):,.2f}
- Portfolio Tax Context:
\"\"\"
{state.get('tax_context', 'No open positions.')}
\"\"\"
- Treasury & Sweep Context:
\"\"\"
{state.get('treasury_context', 'Standard cash reserves.')}
\"\"\"

Provide your final structured decision."""

    llm = get_deep_think_llm(temperature=0.1)
    structured_llm = llm.with_structured_output(TradeDecision)
    try:
        decision = paced_llm_invoke(structured_llm, prompt, is_deep_think=True, schema=TradeDecision)
        if isinstance(decision, TradeDecision):
            # Guardrail checks
            if decision.stop_loss_price > hard_stop:
                decision.stop_loss_price = hard_stop
            if decision.target_price < min_target:
                decision.target_price = min_target
            if decision.allocation_pct > MAX_SINGLE_STOCK_PCT:
                decision.allocation_pct = MAX_SINGLE_STOCK_PCT
            return {"trade_decision": decision.model_dump()}
        return {"trade_decision": decision}
    except Exception as e:
        logger.error(f"Error invoking structured portfolio manager for {ticker}: {e}")
        fallback_decision = TradeDecision(
            ticker=ticker,
            action="HOLD",
            conviction="LOW",
            allocation_pct=0.0,
            entry_price=latest_close,
            stop_loss_price=hard_stop,
            target_price=min_target,
            holding_horizon_days=30,
            rationale=f"Hold decision due to structured output parsing exception: {e}"
        )
        return {"trade_decision": fallback_decision.model_dump()}

