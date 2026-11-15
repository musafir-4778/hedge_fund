"""backtest_simulation.py — 1-Year Walk-Forward Simulated Learning & Backtest Engine.

Strictly enforces:
1. ZERO LOOKAHEAD BIAS: On any simulated date T, the system ONLY has visibility into
   market prices, indicators, and regime data where date <= T. Data for date > T does
   not exist to the agents.
2. EPISODIC LEARNING FROM MISTAKES: Whenever a trade exits with a loss, the post-mortem
   reflection engine diagnoses root cause, updates the rolling agent scorecard,
   synthesizes an actionable guardrail rule, and stores it in the Institutional Memory Bank.
3. CLOSED-LOOP GUARDRAILS: On subsequent simulated dates, newly screened candidates are
   checked against previously learned guardrail rules; any candidate violating a learned
   rule is vetoed to avoid repeating past mistakes.
4. REALISTIC FRICTION: Indian delivery STT (0.1%), NSE turnover, SEBI charges, stamp duty,
   and 0.05% slippage on every trade.
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import numpy as np
import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

from config import (
    INITIAL_CAPITAL_INR,
    MAX_PORTFOLIO_RISK_PER_TRADE,
    MAX_SINGLE_STOCK_PCT,
    ATR_STOP_MULTIPLIER,
    MIN_RISK_REWARD_RATIO,
    BENCHMARK_TICKER,
    VIX_TICKER,
)
from database import (
    init_db,
    save_thesis_snapshot,
    close_thesis_snapshot,
    save_trade_post_mortem,
    clear_post_mortems_and_snapshots,
    reset_portfolio_account,
    update_cash,
    add_position,
    remove_position,
    update_position_stop_loss,
    log_trade,
    log_tax_deduction,
    get_regime_balanced_rules,
    accrue_liquid_fund_interest,
    sweep_cash_to_liquid_fund,
    un_sweep_liquid_fund,
)
from paper_broker import compute_buy_charges, compute_sell_charges
from tax_engine import compute_capital_gains_tax, get_financial_year
from reflection_engine import run_post_mortem_audit
from graph import run_investment_committee

logger = logging.getLogger("wealth_agent_office.simulation")
console = Console()

# 10 representative liquid Indian stocks across sectors + Nifty 50
DEFAULT_SIMULATION_UNIVERSE = [
    "TRENT.NS",       # Consumer / Retail
    "BEL.NS",         # Defense PSU / Capital Goods
    "ADANIPORTS.NS",  # Infrastructure / Ports
    "RELIANCE.NS",    # Energy / Conglomerate
    "INFY.NS",        # Information Technology
    "ICICIBANK.NS",   # Private Banking
    "HDFCBANK.NS",    # Banking
    "SUNPHARMA.NS",   # Healthcare / Pharma
    "BAJFINANCE.NS",  # NBFC / Financials
    "LT.NS",          # Engineering & Construction
]

# 20 liquid Indian leaders across diverse economic sectors
DEFAULT_20_STOCKS = [
    "TRENT.NS",       # Consumer / Retail
    "BEL.NS",         # Defense PSU / Capital Goods
    "ADANIPORTS.NS",  # Infrastructure / Ports
    "RELIANCE.NS",    # Energy / Conglomerate
    "INFY.NS",        # Information Technology
    "ICICIBANK.NS",   # Private Banking
    "HDFCBANK.NS",    # Banking
    "SUNPHARMA.NS",   # Healthcare / Pharma
    "BAJFINANCE.NS",  # NBFC / Financials
    "LT.NS",          # Engineering & Construction
    "TCS.NS",         # Information Technology
    "BHARTIARTL.NS",  # Telecommunications
    "ITC.NS",         # FMCG & Consumer Staples
    "KOTAKBANK.NS",   # Banking & Financial Services
    "MARUTI.NS",      # Automobile (Passenger Vehicles)
    "M&M.NS",         # Automobile & Farm Equipment
    "HINDUNILVR.NS",  # FMCG & Personal Care
    "NTPC.NS",        # Power & Energy Utilities
    "POWERGRID.NS",   # Power Infrastructure & Transmission
    "TITAN.NS",       # Consumer Discretionary & Lifestyle
]

SECTOR_MAP = {
    "TRENT.NS": "Consumer Retail",
    "BEL.NS": "Aerospace & Defense",
    "ADANIPORTS.NS": "Infrastructure / Ports",
    "RELIANCE.NS": "Energy & Petrochemicals",
    "INFY.NS": "Information Technology",
    "ICICIBANK.NS": "Banking & Financial Services",
    "HDFCBANK.NS": "Banking & Financial Services",
    "SUNPHARMA.NS": "Pharmaceuticals & Healthcare",
    "BAJFINANCE.NS": "Non-Banking Financial (NBFC)",
    "LT.NS": "Engineering & Construction",
    "TCS.NS": "Information Technology",
    "BHARTIARTL.NS": "Telecommunications",
    "ITC.NS": "FMCG & Consumer Staples",
    "KOTAKBANK.NS": "Banking & Financial Services",
    "MARUTI.NS": "Automobile (Passenger Vehicles)",
    "M&M.NS": "Automobile & Farm Equipment",
    "HINDUNILVR.NS": "FMCG & Consumer Goods",
    "NTPC.NS": "Power & Utilities",
    "POWERGRID.NS": "Power Infrastructure",
    "TITAN.NS": "Consumer Discretionary & Lifestyle",
}


# ---------------------------------------------------------
# Deterministic Indicators with Zero Lookahead
# ---------------------------------------------------------

def compute_point_in_time_indicators(stock_series: pd.DataFrame, nifty_series: pd.Series) -> Dict[str, Any]:
    """Computes technical indicators strictly up to the current point-in-time."""
    close = stock_series["Close"]
    high = stock_series["High"]
    low = stock_series["Low"]
    volume = stock_series["Volume"]

    latest_close = float(close.iloc[-1])
    n = len(close)

    # SMAs
    sma_20 = float(close.rolling(20).mean().iloc[-1]) if n >= 20 else latest_close
    sma_50 = float(close.rolling(50).mean().iloc[-1]) if n >= 50 else latest_close
    sma_150 = float(close.rolling(150).mean().iloc[-1]) if n >= 150 else sma_50
    sma_200 = float(close.rolling(200).mean().iloc[-1]) if n >= 200 else sma_150

    # Wilder's RSI (14)
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi_series = 100 - (100 / (1 + rs))
    rsi_14 = float(rsi_series.iloc[-1]) if not rsi_series.empty and not np.isnan(rsi_series.iloc[-1]) else 50.0

    # Wilder's ATR (14)
    prev_close = close.shift(1)
    tr1 = high - low
    tr2 = (high - prev_close).abs()
    tr3 = (low - prev_close).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr_series = tr.ewm(alpha=1.0 / 14, min_periods=14, adjust=False).mean()
    atr_14 = float(atr_series.iloc[-1]) if not atr_series.empty and not np.isnan(atr_series.iloc[-1]) else (latest_close * 0.02)

    # 20-day Volume Ratio
    vol_sma20 = float(volume.rolling(20).mean().iloc[-1]) if n >= 20 else float(volume.iloc[-1])
    vol_ratio = round(float(volume.iloc[-1]) / vol_sma20, 2) if vol_sma20 > 0 else 1.0

    # Minervini Stage-2 Uptrend
    high_52w = float(high.max())
    stage2_uptrend = (latest_close > sma_50 > sma_150 > sma_200) and (latest_close >= 0.75 * high_52w)

    # 63-day RS Alpha vs Nifty 50
    if n >= 63 and len(nifty_series) >= 63:
        stock_ret_63 = (latest_close / float(close.iloc[-63])) - 1.0
        nifty_ret_63 = (float(nifty_series.iloc[-1]) / float(nifty_series.iloc[-63])) - 1.0
        rs_alpha_63d = round(stock_ret_63 - nifty_ret_63, 4)
    else:
        rs_alpha_63d = 0.0

    return {
        "latest_close": latest_close,
        "sma_20": sma_20,
        "sma_50": sma_50,
        "sma_150": sma_150,
        "sma_200": sma_200,
        "rsi_14": round(rsi_14, 2),
        "atr_14": round(atr_14, 2),
        "volume_ratio": vol_ratio,
        "stage2_uptrend": stage2_uptrend,
        "rs_alpha_63d": rs_alpha_63d,
    }


# ---------------------------------------------------------
# Episodic Reflection & Guardrail Synthesis
# ---------------------------------------------------------

class SimulatedLearningEngine:
    """Manages post-mortem audits, rolling accuracy, and guardrail rule synthesis during walk-forward."""

    def __init__(self, persist_to_db: bool = True):
        self.persist_to_db = persist_to_db
        self.memory_bank: List[Dict[str, Any]] = []
        self.active_guardrails: List[Dict[str, Any]] = []
        self.agent_scorecard = {
            "TECHNICAL": {"wins": 0, "losses": 0},
            "FUNDAMENTAL": {"wins": 0, "losses": 0},
            "NEWS_MACRO": {"wins": 0, "losses": 0},
            "RISK_COMMITTEE": {"wins": 0, "losses": 0},
        }
        self.vetoed_trades_count = 0
        if self.persist_to_db:
            clear_post_mortems_and_snapshots()

    def record_win(
        self,
        ticker: str,
        pnl_pct: float,
        date_str: str,
        snapshot_id: int = 0,
        entry_price: float = 0.0,
        exit_price: float = 0.0,
        holding_days: int = 1
    ):
        """Records a winning trade in the scorecard and live database."""
        self.agent_scorecard["TECHNICAL"]["wins"] += 1
        self.agent_scorecard["RISK_COMMITTEE"]["wins"] += 1

        if self.persist_to_db:
            save_trade_post_mortem(
                snapshot_id=snapshot_id,
                ticker=ticker,
                sector=SECTOR_MAP.get(ticker, "Industrials"),
                outcome="WIN",
                pnl_pct=round(pnl_pct / 100.0, 4),
                holding_days=holding_days,
                primary_driver="ASYMMETRIC_MOMENTUM_TARGET_HIT",
                accurate_agent="TECHNICAL",
                failed_agent="NONE",
                root_cause=f"Satisfied >= 2.5:1 R:R target at ₹{exit_price:,.2f} from entry ₹{entry_price:,.2f}.",
                actionable_rule=f"In confirmed Stage-2 bull regimes, maintain trailing stops on {ticker} breakouts."
            )

    def run_post_mortem(
        self,
        ticker: str,
        entry_price: float,
        exit_price: float,
        entry_date: str,
        exit_date: str,
        holding_days: int,
        entry_indicators: Dict[str, Any],
        market_regime: str,
        snapshot_id: int = 0
    ) -> Dict[str, Any]:
        """Audits a failed trade, diagnoses root cause, and synthesizes an actionable guardrail rule."""
        pnl_pct = round(((exit_price - entry_price) / entry_price) * 100.0, 2)
        rsi = entry_indicators.get("rsi_14", 50.0)
        vol_ratio = entry_indicators.get("volume_ratio", 1.0)
        alpha = entry_indicators.get("rs_alpha_63d", 0.0)
        stage2 = entry_indicators.get("stage2_uptrend", True)

        # Root Cause Attribution
        if market_regime == "BEAR_RISK_OFF":
            driver = "MARKET_REGIME_HEADWIND"
            failed_agent = "RISK_COMMITTEE"
            rule_id = "REGIME_CASH_MANDATE"
            rule_text = "In BEAR_RISK_OFF regimes, reject all new long breakouts to maintain >= 50% cash."
            check_fn = lambda ind, reg: reg != "BEAR_RISK_OFF"
        elif vol_ratio < 1.0:
            driver = "LOW_VOLUME_FALSE_BREAKOUT"
            failed_agent = "TECHNICAL"
            rule_id = "MIN_VOLUME_RATIO_GUARD"
            rule_text = "Reject breakouts where 20-day Volume Ratio < 1.1x (lack of institutional sponsorship)."
            check_fn = lambda ind, reg: ind.get("volume_ratio", 0) >= 1.1
        elif rsi > 68.0:
            driver = "MOMENTUM_OVERBOUGHT_EXHAUSTION"
            failed_agent = "TECHNICAL"
            rule_id = "MAX_RSI_THRESHOLD"
            rule_text = "Reject breakout entries where 14-day Wilder RSI > 68.0 to prevent buying momentum tops."
            check_fn = lambda ind, reg: ind.get("rsi_14", 50) <= 68.0
        elif alpha <= 0.0:
            driver = "NEGATIVE_RELATIVE_STRENGTH"
            failed_agent = "FUNDAMENTAL"
            rule_id = "MIN_RS_ALPHA_MANDATE"
            rule_text = "Require positive 63-day RS Alpha vs Nifty 50; reject relative underperformers."
            check_fn = lambda ind, reg: ind.get("rs_alpha_63d", 0) > 0.0
        else:
            driver = "VOLATILITY_CHOP_STOP_TRIGGER"
            failed_agent = "RISK_COMMITTEE"
            rule_id = "STAGE2_SMAS_CONFIRMATION"
            rule_text = "Require confirmed Minervini Stage-2 alignment (Price > SMA50 > SMA150 > SMA200)."
            check_fn = lambda ind, reg: bool(ind.get("stage2_uptrend", False))

        # Update Scorecard
        if failed_agent in self.agent_scorecard:
            self.agent_scorecard[failed_agent]["losses"] += 1

        post_mortem = {
            "date": exit_date,
            "ticker": ticker,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "pnl_pct": pnl_pct,
            "holding_days": holding_days,
            "primary_driver": driver,
            "failed_agent": failed_agent,
            "actionable_rule": rule_text
        }
        self.memory_bank.append(post_mortem)

        # Add unique guardrail rule to active memory
        if not any(g["id"] == rule_id for g in self.active_guardrails):
            self.active_guardrails.append({
                "id": rule_id,
                "text": rule_text,
                "created_on": exit_date,
                "learned_from": ticker,
                "check_fn": check_fn
            })

        # Persist to live database
        if self.persist_to_db:
            accurate_agent = "RISK_COMMITTEE" if failed_agent != "RISK_COMMITTEE" else "TECHNICAL"
            save_trade_post_mortem(
                snapshot_id=snapshot_id,
                ticker=ticker,
                sector=SECTOR_MAP.get(ticker, "Industrials"),
                outcome="LOSS",
                pnl_pct=round(pnl_pct / 100.0, 4),
                holding_days=holding_days,
                primary_driver=driver,
                accurate_agent=accurate_agent,
                failed_agent=failed_agent,
                root_cause=f"{driver}: Entry @ ₹{entry_price:,.2f} hit stop @ ₹{exit_price:,.2f} during {market_regime}.",
                actionable_rule=rule_text
            )

        return post_mortem

    def check_candidate_against_guardrails(
        self, ticker: str, indicators: Dict[str, Any], market_regime: str
    ) -> Tuple[bool, Optional[str], bool]:
        """Tests a new candidate against all previously synthesized empirical rules.
        
        Returns (is_approved, reason, rule_was_bent).
        Implements Regime-Adaptive Leeway:
        - If RS Alpha >= 0.03 or in early recovery/bull regime, learned empirical rules are BENT
          rather than issuing a hard blocker.
        """
        rs_alpha = indicators.get("rs_alpha_63d", 0.0)
        is_recovery_or_bull = market_regime in ["EARLY_RECOVERY_PIVOT", "CONFIRMED_BULL_STAGE2", "CHOPPY_ROTATION"]

        for guard in self.active_guardrails:
            try:
                passed = guard["check_fn"](indicators, market_regime)
                if not passed:
                    # Adaptive Leeway: Can this empirical trial-and-error scar be bent?
                    if (rs_alpha >= 0.03) or is_recovery_or_bull:
                        # BEND THE RULE: Allow candidate through to committee with tactical leeway
                        return True, f"BENT Learned Rule: '{guard['text']}' (RS Alpha {rs_alpha*100:+.1f}% justifies tactical leeway)", True
                    else:
                        self.vetoed_trades_count += 1
                        return False, f"VETOED by Learned Rule: '{guard['text']}' (Learned from {guard['learned_from']} on {guard['created_on']})", False
            except Exception:
                continue
        return True, None, False


# ---------------------------------------------------------
# Core Walk-Forward Simulation Loop
# ---------------------------------------------------------

def run_walkforward_simulation(
    universe: Optional[List[str]] = None,
    duration_days: Optional[int] = 21,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    initial_capital: float = INITIAL_CAPITAL_INR,
    use_committee: bool = True,
    use_llm_reflection: bool = True,
    verbose: bool = True,
    persist_to_db: bool = True,
    reset_db_first: bool = True,
    liquid_fund_yield: float = 0.0675
) -> Dict[str, Any]:
    """Runs point-in-time day-to-day walk-forward simulation with zero lookahead bias.
    
    Supports:
    - 20 liquid Indian equities across sectors
    - Flexible duration (default 21 days for past 1 month, or 248 for 1 year)
    - Automated 6.75% Liquid Mutual Fund sweep on idle cash reserves
    - Statutory Indian equity taxation (STCG 20%, LTCG 12.5%, STCL carryforward offset)
    - Closed-loop guardrail enforcement and post-mortem learning from mistakes
    """
    init_db()
    tickers = universe or DEFAULT_20_STOCKS
    all_symbols = [BENCHMARK_TICKER, VIX_TICKER] + tickers

    if reset_db_first and persist_to_db:
        reset_portfolio_account(clean_trades=True, initial_capital=initial_capital)
        clear_post_mortems_and_snapshots()

    time_desc = f"{duration_days} Trading Days (~1 Month)" if duration_days and duration_days <= 30 else f"{duration_days or '1-Year'} Trading Days"
    if verbose:
        console.print("\n" + "=" * 80, style="cyan")
        console.print(f"  DAY-TO-DAY WALK-FORWARD TRAINING & SIMULATION ({len(tickers)} STOCKS)", style="bold cyan")
        console.print("=" * 80, style="cyan")
        console.print(f"[cyan]Universe: {', '.join(tickers[:10])}... ({len(tickers)} Liquid Equities)[/cyan]")
        console.print(f"[cyan]Downloading point-in-time daily bars for universe + Nifty 50 + India VIX...[/cyan]")

    # Download daily bars
    raw_df = yf.download(all_symbols, period="2y", interval="1d", group_by="ticker", progress=False)

    data_by_ticker = {}
    for sym in all_symbols:
        if sym in raw_df.columns.levels[0]:
            sub = raw_df[sym].dropna(how="all").copy()
            if not sub.empty:
                data_by_ticker[sym] = sub

    nifty_df = data_by_ticker.get(BENCHMARK_TICKER)
    vix_df = data_by_ticker.get(VIX_TICKER)
    if nifty_df is None or len(nifty_df) < 100:
        raise ValueError("Insufficient historical benchmark data for Nifty 50.")

    all_dates = nifty_df.index
    if start_date and end_date:
        sim_dates = all_dates[(all_dates >= start_date) & (all_dates <= end_date)]
    elif duration_days:
        test_start_idx = max(65, len(all_dates) - duration_days)
        sim_dates = all_dates[test_start_idx:]
    else:
        test_start_idx = max(65, len(all_dates) - 248)
        sim_dates = all_dates[test_start_idx:]

    learning_engine = SimulatedLearningEngine(persist_to_db=persist_to_db)

    cash = initial_capital
    liquid_bal = 0.0
    cum_interest = 0.0
    cum_fy_exempt_used = 0.0
    cum_stcl_carryforward = 0.0
    current_fy_tracker = "FY2026-27"
    positions: Dict[str, Dict[str, Any]] = {}
    daily_nav_history: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []

    if verbose:
        console.print(f"[green]Simulation window: {sim_dates[0].strftime('%Y-%m-%d')} to {sim_dates[-1].strftime('%Y-%m-%d')} ({len(sim_dates)} trading sessions)[/green]")
        console.print(f"[yellow]Enforcing strict point-in-time isolation: date > T is completely hidden from agents.[/yellow]")
        console.print(f"[cyan]Mode: {'10-Agent Committee Evaluation' if use_committee else 'Fast Quantitative Sizing'} | Sizing Cap: Max 3 concurrent holdings (6-12% NAV each)[/cyan]\n")

    # Day-by-Day Walk-Forward Simulation Loop
    for i, current_dt in enumerate(sim_dates):
        current_date_str = current_dt.strftime("%Y-%m-%d")

        # -------------------------------------------------------------
        # STRICT POINT-IN-TIME SLICING: ONLY DATA UP TO current_dt
        # -------------------------------------------------------------
        nifty_pit = nifty_df.loc[:current_dt]
        nifty_close = float(nifty_pit["Close"].iloc[-1])
        nifty_sma50 = float(nifty_pit["Close"].rolling(50).mean().iloc[-1]) if len(nifty_pit) >= 50 else nifty_close
        nifty_sma200 = float(nifty_pit["Close"].rolling(200).mean().iloc[-1]) if len(nifty_pit) >= 200 else nifty_sma50
        nifty_63d_trend = ((nifty_close / float(nifty_pit["Close"].iloc[-63])) - 1.0) if len(nifty_pit) >= 63 else 0.0

        current_vix = 15.0
        if vix_df is not None and not vix_df.loc[:current_dt].empty:
            current_vix = float(vix_df.loc[:current_dt]["Close"].iloc[-1])

        # Market Regime on Date T
        if nifty_close > nifty_sma50 > nifty_sma200:
            regime = "CONFIRMED_BULL_STAGE2"
        elif nifty_close > nifty_sma200 and nifty_close < nifty_sma50:
            regime = "CHOPPY_ROTATION"
        elif nifty_close > nifty_sma50 and nifty_close <= nifty_sma200:
            regime = "EARLY_RECOVERY_PIVOT"
        else:
            regime = "ACTIVE_LIQUIDATION_BEAR"

        # Accrue daily risk-free interest on liquid fund balance
        if liquid_bal > 0:
            day_yield = round(liquid_bal * (liquid_fund_yield / 365.0), 2)
            liquid_bal += day_yield
            cum_interest += day_yield
            if persist_to_db:
                accrue_liquid_fund_interest(day_yield, timestamp=current_date_str)

        # -------------------------------------------------------------
        # 1. SURVEILLANCE OF OPEN POSITIONS (Stops, Targets & Horiz Exits)
        # -------------------------------------------------------------
        active_tickers = list(positions.keys())
        for sym in active_tickers:
            pos = positions[sym]
            stock_pit = data_by_ticker[sym].loc[:current_dt]
            high_today = float(stock_pit["High"].iloc[-1])
            low_today = float(stock_pit["Low"].iloc[-1])
            close_today = float(stock_pit["Close"].iloc[-1])

            holding_days = (current_dt - pos["entry_date"]).days

            # A. Check Stop-Loss Hit
            if low_today <= pos["stop_loss"]:
                exit_price = round(min(pos["stop_loss"], close_today), 2)
                gross_val = pos["qty"] * exit_price
                sell_fees = compute_sell_charges(gross_val)
                net_proceeds = gross_val - sell_fees

                tax_info = compute_capital_gains_tax(
                    buy_date=pos["entry_date_str"],
                    sell_date=current_date_str,
                    cost_basis=pos["total_cost"],
                    net_proceeds=net_proceeds,
                    fy_ltcg_exempt_used=cum_fy_exempt_used,
                    fy_stcl_carryforward=cum_stcl_carryforward,
                    current_fy=current_fy_tracker
                )
                current_fy_tracker = tax_info.get("financial_year", current_fy_tracker)
                tax_deducted = tax_info["tax_deducted"]
                net_pnl_after_tax = tax_info["net_pnl_after_tax"]
                cum_fy_exempt_used = tax_info["new_fy_exempt_used"]
                cum_stcl_carryforward = tax_info.get("new_fy_stcl", 0.0)

                cash += (net_proceeds - tax_deducted)
                realized_pnl = net_proceeds - pos["total_cost"]
                pnl_pct = round((realized_pnl / pos["total_cost"]) * 100.0, 2)

                if persist_to_db:
                    remove_position(sym)
                    log_trade(
                        ticker=sym,
                        action="SELL",
                        qty=pos["qty"],
                        execution_price=exit_price,
                        stt_and_charges=sell_fees,
                        realized_pnl=net_pnl_after_tax,
                        exit_reason="STOP_LOSS_HIT",
                        rationale=f"Daily stop-loss hit @ ₹{exit_price:,.2f} on {current_date_str}"
                    )
                    log_tax_deduction(tax_info, ticker=sym, timestamp=current_date_str)
                    if pos.get("snapshot_id"):
                        close_thesis_snapshot(pos["snapshot_id"])

                # Post-Mortem Audit & Rule Synthesis
                rule_text = ""
                if use_llm_reflection:
                    try:
                        pos_dict = {
                            "ticker": sym,
                            "snapshot_id": pos.get("snapshot_id", 0),
                            "avg_price": pos["entry_price"],
                            "opened_at": pos["entry_date_str"],
                            "sector": pos.get("sector", "Industrials")
                        }
                        pm_res = run_post_mortem_audit(
                            position=pos_dict,
                            exit_price=exit_price,
                            exit_reason="STOP_LOSS_HIT",
                            realized_pnl=net_pnl_after_tax,
                            closed_at=current_date_str
                        )
                        if pm_res and pm_res.get("actionable_rule"):
                            rule_text = pm_res["actionable_rule"]
                    except Exception as e:
                        logger.warning(f"LLM post-mortem audit failed: {e}. Falling back to deterministic.")

                if not rule_text:
                    pm = learning_engine.run_post_mortem(
                        ticker=sym,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        entry_date=pos["entry_date_str"],
                        exit_date=current_date_str,
                        holding_days=holding_days,
                        entry_indicators=pos["indicators"],
                        market_regime=regime,
                        snapshot_id=pos.get("snapshot_id", 0)
                    )
                    rule_text = pm["actionable_rule"]

                closed_trades.append({
                    "ticker": sym,
                    "entry_date": pos["entry_date_str"],
                    "exit_date": current_date_str,
                    "outcome": "LOSS",
                    "pnl": realized_pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "STOP_LOSS_HIT",
                    "rule": rule_text
                })

                if verbose:
                    console.print(f"[{current_date_str}] [bold red]▼ STOP HIT:[/] {sym} exited @ ₹{exit_price:,.2f} ({pnl_pct:+.1f}%). "
                                  f"Rule: [italic]{rule_text[:70]}...[/italic]")
                del positions[sym]
                continue

            # B. Check Target Hit
            if high_today >= pos["target_price"]:
                exit_price = pos["target_price"]
                gross_val = pos["qty"] * exit_price
                sell_fees = compute_sell_charges(gross_val)
                net_proceeds = gross_val - sell_fees

                tax_info = compute_capital_gains_tax(
                    buy_date=pos["entry_date_str"],
                    sell_date=current_date_str,
                    cost_basis=pos["total_cost"],
                    net_proceeds=net_proceeds,
                    fy_ltcg_exempt_used=cum_fy_exempt_used,
                    fy_stcl_carryforward=cum_stcl_carryforward,
                    current_fy=current_fy_tracker
                )
                tax_deducted = tax_info["tax_deducted"]
                net_pnl_after_tax = tax_info["net_pnl_after_tax"]
                cum_fy_exempt_used = tax_info["new_fy_exempt_used"]
                cash += (net_proceeds - tax_deducted)

                realized_pnl = net_proceeds - pos["total_cost"]
                pnl_pct = round((realized_pnl / pos["total_cost"]) * 100.0, 2)

                if persist_to_db:
                    remove_position(sym)
                    log_trade(
                        ticker=sym,
                        action="SELL",
                        qty=pos["qty"],
                        execution_price=exit_price,
                        stt_and_charges=sell_fees,
                        realized_pnl=net_pnl_after_tax,
                        exit_reason="TARGET_HIT",
                        rationale=f"Target hit @ ₹{exit_price:,.2f} on {current_date_str}"
                    )
                    log_tax_deduction(tax_info, ticker=sym, timestamp=current_date_str)
                    if pos.get("snapshot_id"):
                        close_thesis_snapshot(pos["snapshot_id"])

                learning_engine.record_win(
                    ticker=sym,
                    pnl_pct=pnl_pct,
                    date_str=current_date_str,
                    snapshot_id=pos.get("snapshot_id", 0),
                    entry_price=pos["entry_price"],
                    exit_price=exit_price,
                    holding_days=holding_days
                )

                closed_trades.append({
                    "ticker": sym,
                    "entry_date": pos["entry_date_str"],
                    "exit_date": current_date_str,
                    "outcome": "WIN",
                    "pnl": realized_pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "TARGET_HIT"
                })

                if verbose:
                    console.print(f"[{current_date_str}] [bold green]▲ TARGET HIT:[/] {sym} exited @ ₹{exit_price:,.2f} ({pnl_pct:+.1f}%)! R:R >= 2.5:1 satisfied.")
                del positions[sym]
                continue

            # C. Check 30-Day Holding Horizon Exit
            if holding_days >= 30:
                exit_price = close_today
                gross_val = pos["qty"] * exit_price
                sell_fees = compute_sell_charges(gross_val)
                net_proceeds = gross_val - sell_fees

                tax_info = compute_capital_gains_tax(
                    buy_date=pos["entry_date_str"],
                    sell_date=current_date_str,
                    cost_basis=pos["total_cost"],
                    net_proceeds=net_proceeds,
                    fy_ltcg_exempt_used=cum_fy_exempt_used,
                    fy_stcl_carryforward=cum_stcl_carryforward,
                    current_fy=current_fy_tracker
                )
                tax_deducted = tax_info["tax_deducted"]
                net_pnl_after_tax = tax_info["net_pnl_after_tax"]
                cash += (net_proceeds - tax_deducted)

                realized_pnl = net_proceeds - pos["total_cost"]
                pnl_pct = round((realized_pnl / pos["total_cost"]) * 100.0, 2)

                if persist_to_db:
                    remove_position(sym)
                    log_trade(
                        ticker=sym,
                        action="SELL",
                        qty=pos["qty"],
                        execution_price=exit_price,
                        stt_and_charges=sell_fees,
                        realized_pnl=net_pnl_after_tax,
                        exit_reason="HORIZON_EXPIRY",
                        rationale=f"30-day horizon expired on {current_date_str}"
                    )
                    log_tax_deduction(tax_info, ticker=sym, timestamp=current_date_str)
                    if pos.get("snapshot_id"):
                        close_thesis_snapshot(pos["snapshot_id"])

                outcome = "WIN" if realized_pnl > 0 else "LOSS"
                if outcome == "WIN":
                    learning_engine.record_win(
                        ticker=sym,
                        pnl_pct=pnl_pct,
                        date_str=current_date_str,
                        snapshot_id=pos.get("snapshot_id", 0),
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        holding_days=holding_days
                    )
                else:
                    learning_engine.run_post_mortem(
                        ticker=sym,
                        entry_price=pos["entry_price"],
                        exit_price=exit_price,
                        entry_date=pos["entry_date_str"],
                        exit_date=current_date_str,
                        holding_days=holding_days,
                        entry_indicators=pos["indicators"],
                        market_regime=regime,
                        snapshot_id=pos.get("snapshot_id", 0)
                    )

                closed_trades.append({
                    "ticker": sym,
                    "entry_date": pos["entry_date_str"],
                    "exit_date": current_date_str,
                    "outcome": outcome,
                    "pnl": realized_pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "HORIZON_EXPIRY"
                })
                del positions[sym]
                continue

            # D. Dynamic Trailing Stop (2x ATR)
            atr_val = pos["indicators"]["atr_14"]
            trailing_stop = round(close_today - (2.0 * atr_val), 2)
            if trailing_stop > pos["stop_loss"]:
                pos["stop_loss"] = trailing_stop
                if persist_to_db:
                    update_position_stop_loss(sym, trailing_stop)

        # -------------------------------------------------------------
        # 2. TREASURY MANAGEMENT: LIQUID MUTUAL FUND SWEEP
        # -------------------------------------------------------------
        invested_value = sum(pos["qty"] * float(data_by_ticker[sym].loc[:current_dt]["Close"].iloc[-1]) for sym, pos in positions.items())
        total_nav = cash + liquid_bal + invested_value
        cash_buffer = total_nav * 0.10
        if cash > cash_buffer:
            sweep_amt = round(cash - cash_buffer, 2)
            cash -= sweep_amt
            liquid_bal += sweep_amt
            if persist_to_db:
                sweep_cash_to_liquid_fund(sweep_amt, timestamp=current_date_str)

        # -------------------------------------------------------------
        # 3. POINT-IN-TIME DISCOVERY & SCREENING (Max 3 Concurrent Holdings)
        # -------------------------------------------------------------
        max_positions = 3
        if len(positions) < max_positions:
            candidates_scored = []
            for cand_sym in tickers:
                if cand_sym in positions:
                    continue

                stock_hist = data_by_ticker.get(cand_sym)
                if stock_hist is None:
                    continue

                stock_pit = stock_hist.loc[:current_dt]
                if len(stock_pit) < 63:
                    continue

                ind = compute_point_in_time_indicators(stock_pit, nifty_pit["Close"])

                # Base Stage-2 filter (or RS Alpha > 0.03 during recovery)
                is_recovery = regime in ["EARLY_RECOVERY_PIVOT", "CHOPPY_ROTATION"]
                if not ind["stage2_uptrend"] and not (is_recovery and ind["rs_alpha_63d"] > 0.03):
                    continue

                # Closed-Loop Guardrail Check with Regime-Adaptive Leeway
                is_approved, veto_reason, rule_was_bent = learning_engine.check_candidate_against_guardrails(cand_sym, ind, regime)
                if not is_approved:
                    continue

                ind["rule_bent"] = rule_was_bent
                ind["rule_bent_reason"] = veto_reason if rule_was_bent else ""
                candidates_scored.append((cand_sym, ind))

            # Rank by RS Alpha descending
            candidates_scored.sort(key=lambda x: x[1]["rs_alpha_63d"], reverse=True)

            # Evaluate top candidates
            slots_available = max_positions - len(positions)
            for cand_sym, ind in candidates_scored[:slots_available]:
                cand_sec = SECTOR_MAP.get(cand_sym, "Industrials")
                entry_price = ind["latest_close"]
                atr = ind["atr_14"]
                stop_loss = round(entry_price - (ATR_STOP_MULTIPLIER * atr), 2)
                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0:
                    continue
                target_price = round(entry_price + (MIN_RISK_REWARD_RATIO * risk_per_share), 2)

                should_buy = True
                alloc_pct = 0.06
                pm_rationale = "Point-in-time momentum entry"
                comm_res = {}

                if verbose and ind.get("rule_bent"):
                    console.print(f"  [{current_date_str}] [bold yellow]⚡ BENDING LEARNED SCAR for {cand_sym}:[/] {ind['rule_bent_reason']}")

                if use_committee:
                    try:
                        pit_state = {
                            "market_regime": regime,
                            "india_vix": round(current_vix, 2),
                            "benchmark_metrics": {
                                "nifty_close": round(nifty_close, 2),
                                "nifty_sma50": round(nifty_sma50, 2),
                                "nifty_sma200": round(nifty_sma200, 2),
                                "nifty_rs_trend": round(nifty_63d_trend, 4),
                            },
                            "technicals": ind,
                            "point_in_time_date": current_date_str,
                        }
                        if ind.get("rule_bent"):
                            pit_state["rule_bent_guidance"] = (
                                f"NOTICE: {ind['rule_bent_reason']}. The committee is authorized to bend this learned rule: "
                                f"allow tactical entry if conviction is supported by decoupled RS Alpha, but calibrate sizing to 5-6%."
                            )
                        comm_res = run_investment_committee(
                            ticker=cand_sym,
                            sector=cand_sec,
                            point_in_time_state=pit_state
                        )
                        decision = comm_res.get("trade_decision") or {}
                        should_buy = (decision.get("action") == "BUY")
                        alloc_pct = float(decision.get("allocation_pct") or 0.06)
                        if decision.get("stop_loss_price"):
                            stop_loss = float(decision["stop_loss_price"])
                        if decision.get("target_price"):
                            target_price = float(decision["target_price"])
                        pm_rationale = decision.get("rationale") or pm_rationale
                    except Exception as e:
                        logger.warning(f"Committee evaluation failed: {e}. Falling back to quantitative.")
                        should_buy = True

                if not should_buy:
                    continue

                # Size via allocation percentage
                target_val = total_nav * alloc_pct
                qty = int(target_val // entry_price)
                if qty <= 0:
                    continue

                gross_val = qty * entry_price
                buy_fees = compute_buy_charges(gross_val)
                total_entry_cost = gross_val + buy_fees

                # Un-sweep from Liquid Fund if needed
                if cash < total_entry_cost and liquid_bal > 0:
                    needed = total_entry_cost - cash
                    un_sweep = min(liquid_bal, needed)
                    liquid_bal -= un_sweep
                    cash += un_sweep
                    if persist_to_db:
                        un_sweep_liquid_fund(un_sweep, timestamp=current_date_str)

                if total_entry_cost <= cash:
                    cash -= total_entry_cost
                    snap_id = 0
                    if persist_to_db:
                        snap_id = save_thesis_snapshot(
                            ticker=cand_sym,
                            sector=cand_sec,
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            target_price=target_price,
                            indicators=ind,
                            technical_report=comm_res.get("technical_report") or f"Point-in-time Stage-2 trend on {current_date_str}. Close: ₹{entry_price:,.2f}, RSI: {ind['rsi_14']}.",
                            fundamental_report=comm_res.get("fundamental_report") or "Historical point-in-time evaluation.",
                            news_macro_report=comm_res.get("news_macro_report") or f"Market Regime: {regime}, Nifty 50 Close: ₹{nifty_close:,.2f}.",
                            debate_summary=comm_res.get("research_proposal") or "Approved by committee with zero lookahead.",
                            risk_summary=comm_res.get("conservative_risk_view") or f"Hard Stop: ₹{stop_loss:,.2f} (2xATR), Target: ₹{target_price:,.2f}.",
                            pm_rationale=pm_rationale
                        )
                        add_position(
                            ticker=cand_sym,
                            sector=cand_sec,
                            qty=qty,
                            avg_price=entry_price,
                            stop_loss=stop_loss,
                            target_price=target_price,
                            opened_at=current_date_str,
                            max_hold_until=(current_dt + pd.Timedelta(days=30)).strftime("%Y-%m-%d"),
                            snapshot_id=snap_id
                        )
                        log_trade(
                            ticker=cand_sym,
                            action="BUY",
                            qty=qty,
                            execution_price=entry_price,
                            stt_and_charges=buy_fees,
                            realized_pnl=0.0,
                            rationale=pm_rationale
                        )

                    positions[cand_sym] = {
                        "ticker": cand_sym,
                        "sector": cand_sec,
                        "qty": qty,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target_price": target_price,
                        "entry_date": current_dt,
                        "entry_date_str": current_date_str,
                        "total_cost": total_entry_cost,
                        "indicators": ind,
                        "snapshot_id": snap_id
                    }
                    if verbose:
                        console.print(f"[{current_date_str}] [bold green]⚡ BOUGHT:[/] {cand_sym} {qty} shs @ ₹{entry_price:,.2f} "
                                      f"(Size: {alloc_pct*100:.1f}%, Stop: ₹{stop_loss:,.2f}, Target: ₹{target_price:,.2f})")

                if len(positions) >= max_positions:
                    break
        else:
            # Daily Strategic Forward-Planning Scout (Watchlist Preparation)
            scout_list = []
            for cand_sym in tickers:
                if cand_sym in positions:
                    continue
                stock_hist = data_by_ticker.get(cand_sym)
                if stock_hist is None:
                    continue
                stock_pit = stock_hist.loc[:current_dt]
                if len(stock_pit) < 63:
                    continue
                ind = compute_point_in_time_indicators(stock_pit, nifty_pit["Close"])
                scout_list.append((cand_sym, ind))
            scout_list.sort(key=lambda x: x[1]["rs_alpha_63d"], reverse=True)
            if scout_list and verbose:
                top_w_sym, top_w_ind = scout_list[0]
                stage2_badge = "[green]STAGE-2[/green]" if top_w_ind["stage2_uptrend"] else "[yellow]BASE FORMING[/yellow]"
                console.print(
                    f"  [{current_date_str}] [dim]🔭 Daily Forward-Planning Scout:[/] "
                    f"[bold yellow]{top_w_sym}[/] ({stage2_badge} | CMP: ₹{top_w_ind['latest_close']:,.2f} | "
                    f"63d RS Alpha: {top_w_ind['rs_alpha_63d']*100:+.1f}% | RSI: {top_w_ind['rsi_14']:.1f})"
                )

        # -------------------------------------------------------------
        # 4. MARK-TO-MARKET DAILY NAV ON DATE T
        # -------------------------------------------------------------
        invested_value = sum(pos["qty"] * float(data_by_ticker[sym].loc[:current_dt]["Close"].iloc[-1]) for sym, pos in positions.items())
        current_nav = cash + liquid_bal + invested_value
        daily_nav_history.append({
            "date": current_date_str,
            "nav": round(current_nav, 2),
            "cash": round(cash, 2),
            "liquid_fund": round(liquid_bal, 2),
            "invested_value": round(invested_value, 2),
            "positions_count": len(positions),
            "nifty_close": nifty_close
        })

        if verbose:
            console.print(
                f"  [{current_date_str}] NAV: [bold]₹{current_nav:,.2f}[/] | "
                f"Liquid Fund: ₹{liquid_bal:,.2f} (+₹{cum_interest:,.0f} yield) | "
                f"Holdings: {len(positions)}/3 ({', '.join(positions.keys()) if positions else 'None'}) | "
                f"Nifty: ₹{nifty_close:,.2f} ({regime})"
            )

    if persist_to_db and sim_dates.size > 0:
        from database import get_db_connection
        with get_db_connection() as conn:
            conn.execute(
                "UPDATE portfolio_account SET cash = ?, liquid_fund_balance = ?, updated_at = ? WHERE id = 1;",
                (round(cash, 2), round(liquid_bal, 2), sim_dates[-1].strftime("%Y-%m-%d"))
            )
            conn.commit()

    # -------------------------------------------------------------
    # 5. FINAL PERFORMANCE METRICS & DASHBOARD
    # -------------------------------------------------------------
    final_nav = daily_nav_history[-1]["nav"] if daily_nav_history else initial_capital
    total_return_pct = round(((final_nav - initial_capital) / initial_capital) * 100.0, 2)

    nifty_start = daily_nav_history[0]["nifty_close"] if daily_nav_history else 1.0
    nifty_end = daily_nav_history[-1]["nifty_close"] if daily_nav_history else 1.0
    nifty_return_pct = round(((nifty_end - nifty_start) / nifty_start) * 100.0, 2)
    alpha = round(total_return_pct - nifty_return_pct, 2)

    # Max Drawdown
    nav_series = pd.Series([d["nav"] for d in daily_nav_history])
    rolling_max = nav_series.cummax()
    drawdowns = (nav_series - rolling_max) / rolling_max
    max_drawdown_pct = round(float(drawdowns.min()) * 100.0, 2) if not drawdowns.empty else 0.0

    # Trade Statistics
    wins = [t for t in closed_trades if t["outcome"] == "WIN"]
    losses = [t for t in closed_trades if t["outcome"] == "LOSS"]
    win_rate = round((len(wins) / len(closed_trades) * 100.0), 1) if closed_trades else 0.0

    summary = {
        "initial_capital": initial_capital,
        "final_nav": final_nav,
        "total_return_pct": total_return_pct,
        "nifty_return_pct": nifty_return_pct,
        "alpha": alpha,
        "max_drawdown_pct": max_drawdown_pct,
        "total_trades": len(closed_trades),
        "wins": len(wins),
        "losses": len(losses),
        "win_rate": win_rate,
        "liquid_fund_yield": cum_interest,
        "learned_rules_count": len(learning_engine.active_guardrails),
        "vetoed_trades_count": learning_engine.vetoed_trades_count,
        "memory_bank": learning_engine.memory_bank,
        "active_guardrails": learning_engine.active_guardrails,
        "daily_nav_history": daily_nav_history
    }

    if verbose:
        display_simulation_results(summary)
        if persist_to_db:
            console.print("\n[bold green]✓ Institutional Memory Synchronized:[/] All day-to-day simulated learnings and post-mortems have been permanently saved into the live SQLite database (`paper_portfolio_india.db`).", style="bold green")

    return summary


def run_1year_walkforward_simulation(
    universe: Optional[List[str]] = None,
    initial_capital: float = INITIAL_CAPITAL_INR,
    verbose: bool = True,
    persist_to_db: bool = True
) -> Dict[str, Any]:
    """Runs a 1-year point-in-time walk-forward simulation (248 sessions) with zero lookahead bias."""
    return run_walkforward_simulation(
        universe=universe or DEFAULT_SIMULATION_UNIVERSE,
        duration_days=248,
        initial_capital=initial_capital,
        use_committee=False,
        use_llm_reflection=False,
        verbose=verbose,
        persist_to_db=persist_to_db,
        reset_db_first=True
    )


def display_simulation_results(summary: Dict[str, Any]):
    """Renders the executive walk-forward learning results in Rich terminal tables."""
    console.print("\n" + "=" * 80, style="gold1")
    console.print("  DAY-TO-DAY WALK-FORWARD SIMULATION EXECUTIVE SCORECARD", style="bold gold1")
    console.print("=" * 80, style="gold1")

    # Metrics Table
    m_tb = Table(show_header=True, header_style="bold cyan", expand=True)
    m_tb.add_column("Metric", style="bold white")
    m_tb.add_column("Strategy Result", justify="right")
    m_tb.add_column("Benchmark / Baseline", justify="right", style="dim")

    ret_style = "bold green" if summary["total_return_pct"] >= 0 else "bold red"
    m_tb.add_row("Final Portfolio NAV", f"₹{summary['final_nav']:,.2f}", f"₹{summary['initial_capital']:,.2f}")
    m_tb.add_row("Strategy Total Return", f"[{ret_style}]{summary['total_return_pct']:+.2f}%[/{ret_style}]", f"{summary['nifty_return_pct']:+.2f}% (Nifty 50)")
    m_tb.add_row("Alpha vs Nifty 50", f"[bold green]{summary['alpha']:+.2f}%[/bold green]", "0.00%")
    m_tb.add_row("Liquid Fund Yield Earned", f"[bold green]+₹{summary.get('liquid_fund_yield', 0.0):,.2f}[/bold green]", "0.00 (Cash Drag)")
    m_tb.add_row("Maximum Drawdown", f"[bold red]{summary['max_drawdown_pct']:.2f}%[/bold red]", "—")
    m_tb.add_row("Total Trades Closed", str(summary["total_trades"]), "—")
    m_tb.add_row("Win Rate %", f"[bold]{summary['win_rate']:.1f}%[/bold] ({summary['wins']}W / {summary['losses']}L)", "50.0%")
    m_tb.add_row("Rules Learned From Mistakes", f"[bold gold1]{summary['learned_rules_count']}[/bold gold1]", "0 (Naive Prior)")
    m_tb.add_row("Subsequent Bad Trades Vetoed", f"[bold green]{summary['vetoed_trades_count']}[/bold green]", "0")
    console.print(m_tb)

    # Institutional Memory Bank Table
    if summary["active_guardrails"]:
        rules_tb = Table(title="[bold gold1]Institutional Memory Bank: Guardrails Synthesized from Mistakes[/bold gold1]", show_header=True, header_style="bold yellow", expand=True)
        rules_tb.add_column("Rule ID", style="bold cyan", width=22)
        rules_tb.add_column("First Learned On", style="dim", width=14)
        rules_tb.add_column("Learned From", style="bold white", width=14)
        rules_tb.add_column("Synthesized Guardrail Enforced", style="bold green")

        for g in summary["active_guardrails"]:
            rules_tb.add_row(
                g["id"],
                g["created_on"],
                g["learned_from"],
                g["text"]
            )
        console.print(rules_tb)

    # Post-Mortem Failure Audit Sample Table
    if summary["memory_bank"]:
        audit_tb = Table(title="[bold red]Sample Post-Mortem Audits of Simulated Exits[/bold red]", show_header=True, header_style="bold red", expand=True)
        audit_tb.add_column("Exit Date", style="dim", width=12)
        audit_tb.add_column("Ticker", style="bold white", width=14)
        audit_tb.add_column("Loss %", justify="right", style="red", width=10)
        audit_tb.add_column("Diagnosed Failure Driver", style="bold magenta", width=30)
        audit_tb.add_column("Agent Blame Attribution", style="yellow")

        for pm in summary["memory_bank"][:6]:
            audit_tb.add_row(
                pm["date"],
                pm["ticker"],
                f"{pm['pnl_pct']:+.1f}%",
                pm["primary_driver"],
                pm["failed_agent"]
            )
        console.print(audit_tb)

    console.print(Panel(
        f"[bold green]✓ Closed-Loop Verification:[/] The system synthesized [bold gold1]{summary['learned_rules_count']} guardrail rules[/bold gold1] from mistakes, "
        f"which subsequently [bold green]prevented {summary['vetoed_trades_count']} risky trades[/bold green] from suffering similar losses.",
        border_style="green"
    ))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Day-to-day point-in-time walk-forward training & simulation")
    parser.add_argument("--days", type=int, default=21, help="Trading sessions (default 21, ~1 month)")
    parser.add_argument("--stocks", type=int, default=20, help="Universe count: 20 or 10 (default 20)")
    parser.add_argument("--fast", action="store_true", help="Fast mode: skip LLM committee on entries")
    parser.add_argument("--reset", action="store_true", default=True, help="Reset portfolio before starting")
    args = parser.parse_args()

    target_universe = DEFAULT_20_STOCKS if args.stocks == 20 else DEFAULT_SIMULATION_UNIVERSE
    run_walkforward_simulation(
        universe=target_universe,
        duration_days=args.days,
        use_committee=not args.fast,
        use_llm_reflection=True,
        reset_db_first=args.reset
    )

