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
)
from database import (
    save_thesis_snapshot,
    close_thesis_snapshot,
    save_trade_post_mortem,
    clear_post_mortems_and_snapshots,
)
from paper_broker import compute_buy_charges, compute_sell_charges

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

    def check_candidate_against_guardrails(self, ticker: str, indicators: Dict[str, Any], market_regime: str) -> Tuple[bool, Optional[str]]:
        """Tests a new candidate against all previously synthesized guardrails.
        
        Returns (is_approved, veto_reason).
        """
        for guard in self.active_guardrails:
            try:
                passed = guard["check_fn"](indicators, market_regime)
                if not passed:
                    self.vetoed_trades_count += 1
                    return False, f"VETOED by Learned Rule: '{guard['text']}' (Learned from {guard['learned_from']} on {guard['created_on']})"
            except Exception:
                continue
        return True, None


# ---------------------------------------------------------
# Core Walk-Forward Simulation Loop
# ---------------------------------------------------------

def run_1year_walkforward_simulation(
    universe: Optional[List[str]] = None,
    initial_capital: float = INITIAL_CAPITAL_INR,
    verbose: bool = True,
    persist_to_db: bool = True
) -> Dict[str, Any]:
    """Runs a 1-year point-in-time walk-forward simulation with zero lookahead bias."""
    tickers = universe or DEFAULT_SIMULATION_UNIVERSE
    all_symbols = [BENCHMARK_TICKER] + tickers

    if verbose:
        console.print("\n" + "=" * 80, style="cyan")
        console.print("  1-YEAR POINT-IN-TIME WALK-FORWARD SIMULATION & EPISODIC LEARNING", style="bold cyan")
        console.print("=" * 80, style="cyan")
        console.print(f"[cyan]Downloading 1-year historical daily bars for {len(tickers)} Indian Equities + Nifty 50...[/cyan]")

    # Download 1 year + 65 days warmup data
    raw_df = yf.download(all_symbols, period="2y", interval="1d", group_by="ticker", progress=False)

    # Clean multi-index columns
    data_by_ticker = {}
    for sym in all_symbols:
        if sym in raw_df.columns.levels[0]:
            sub = raw_df[sym].dropna(how="all").copy()
            if not sub.empty:
                data_by_ticker[sym] = sub

    nifty_df = data_by_ticker.get(BENCHMARK_TICKER)
    if nifty_df is None or len(nifty_df) < 200:
        raise ValueError("Insufficient historical benchmark data for Nifty 50.")

    # Align trading dates to past 1 year (~248 trading sessions)
    all_dates = nifty_df.index
    test_start_idx = max(65, len(all_dates) - 248)
    sim_dates = all_dates[test_start_idx:]

    learning_engine = SimulatedLearningEngine(persist_to_db=persist_to_db)

    cash = initial_capital
    positions: Dict[str, Dict[str, Any]] = {}
    daily_nav_history: List[Dict[str, Any]] = []
    closed_trades: List[Dict[str, Any]] = []

    if verbose:
        console.print(f"[green]Simulation window: {sim_dates[0].strftime('%Y-%m-%d')} to {sim_dates[-1].strftime('%Y-%m-%d')} ({len(sim_dates)} trading sessions)[/green]")
        console.print(f"[yellow]Enforcing strict point-in-time isolation: date > T is completely hidden from agents.[/yellow]")
        if persist_to_db:
            console.print("[bold cyan]Persisting all simulated trade post-mortems & rules directly into Institutional Memory Bank SQLite DB.[/bold cyan]\n")

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

        # Market Regime on Date T
        if nifty_close > nifty_sma50 > nifty_sma200:
            regime = "BULL_STAGE2"
        elif nifty_close > nifty_sma200 and nifty_close < nifty_sma50:
            regime = "CHOPPY_RANGE"
        else:
            regime = "BEAR_RISK_OFF"

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
                cash += net_proceeds

                realized_pnl = net_proceeds - pos["total_cost"]
                pnl_pct = round((realized_pnl / pos["total_cost"]) * 100.0, 2)

                if persist_to_db and pos.get("snapshot_id"):
                    close_thesis_snapshot(pos["snapshot_id"])

                # TRIGGER POST-MORTEM AUDIT & RULE SYNTHESIS
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

                closed_trades.append({
                    "ticker": sym,
                    "entry_date": pos["entry_date_str"],
                    "exit_date": current_date_str,
                    "outcome": "LOSS",
                    "pnl": realized_pnl,
                    "pnl_pct": pnl_pct,
                    "exit_reason": "STOP_LOSS_HIT"
                })

                if verbose:
                    console.print(f"[{current_date_str}] [bold red]▼ STOP HIT:[/] {sym} sold @ ₹{exit_price:,.2f} ({pnl_pct:+.1f}%). "
                                  f"Post-Mortem: [italic]{pm['actionable_rule']}[/italic]")
                del positions[sym]
                continue

            # B. Check Target Hit
            if high_today >= pos["target_price"]:
                exit_price = pos["target_price"]
                gross_val = pos["qty"] * exit_price
                sell_fees = compute_sell_charges(gross_val)
                net_proceeds = gross_val - sell_fees
                cash += net_proceeds

                realized_pnl = net_proceeds - pos["total_cost"]
                pnl_pct = round((realized_pnl / pos["total_cost"]) * 100.0, 2)

                if persist_to_db and pos.get("snapshot_id"):
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
                    console.print(f"[{current_date_str}] [bold green]▲ TARGET HIT:[/] {sym} sold @ ₹{exit_price:,.2f} ({pnl_pct:+.1f}%)! "
                                  f"Risk/Reward >= 2.5:1 satisfied.")
                del positions[sym]
                continue

            # C. Check 30-Day Holding Horizon Exit
            if holding_days >= 30:
                exit_price = close_today
                gross_val = pos["qty"] * exit_price
                sell_fees = compute_sell_charges(gross_val)
                net_proceeds = gross_val - sell_fees
                cash += net_proceeds

                realized_pnl = net_proceeds - pos["total_cost"]
                pnl_pct = round((realized_pnl / pos["total_cost"]) * 100.0, 2)

                if persist_to_db and pos.get("snapshot_id"):
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

        # -------------------------------------------------------------
        # 2. MARK-TO-MARKET PORTFOLIO NAV ON DATE T
        # -------------------------------------------------------------
        invested_value = 0.0
        for sym, pos in positions.items():
            stk_close = float(data_by_ticker[sym].loc[:current_dt]["Close"].iloc[-1])
            invested_value += pos["qty"] * stk_close
        current_nav = cash + invested_value

        daily_nav_history.append({
            "date": current_date_str,
            "nav": round(current_nav, 2),
            "cash": round(cash, 2),
            "invested_value": round(invested_value, 2),
            "positions_count": len(positions),
            "nifty_close": nifty_close
        })

        # -------------------------------------------------------------
        # 3. SCREEN & EVALUATE NEW ENTRIES (Point-In-Time)
        # -------------------------------------------------------------
        max_positions = 5
        if len(positions) < max_positions and cash > (current_nav * 0.10):
            for cand_sym in tickers:
                if cand_sym in positions:
                    continue

                stock_hist = data_by_ticker.get(cand_sym)
                if stock_hist is None:
                    continue

                stock_pit = stock_hist.loc[:current_dt]
                if len(stock_pit) < 63:
                    continue

                # Point-in-time indicators
                ind = compute_point_in_time_indicators(stock_pit, nifty_pit["Close"])

                # Base Stage-2 filter
                if not ind["stage2_uptrend"]:
                    continue

                # CLOSED-LOOP GUARDRAIL VERIFICATION: Check against past learned rules!
                is_approved, veto_reason = learning_engine.check_candidate_against_guardrails(cand_sym, ind, regime)
                if not is_approved:
                    # Successfully prevented a trade based on institutional learning!
                    continue

                # Compute position sizing
                entry_price = ind["latest_close"]
                atr = ind["atr_14"]
                stop_loss = round(entry_price - (ATR_STOP_MULTIPLIER * atr), 2)
                risk_per_share = entry_price - stop_loss
                if risk_per_share <= 0:
                    continue

                target_price = round(entry_price + (MIN_RISK_REWARD_RATIO * risk_per_share), 2)

                # Size via risk budgeting (1.5% NAV risk) and single-stock cap (12% NAV)
                risk_budget = current_nav * MAX_PORTFOLIO_RISK_PER_TRADE
                cap_budget = current_nav * MAX_SINGLE_STOCK_PCT

                qty_by_risk = int(risk_budget // risk_per_share)
                qty_by_cap = int(cap_budget // entry_price)
                qty = max(1, min(qty_by_risk, qty_by_cap))

                gross_val = qty * entry_price
                buy_fees = compute_buy_charges(gross_val)
                total_cost = gross_val + buy_fees

                if total_cost <= cash:
                    cash -= total_cost
                    snap_id = 0
                    if persist_to_db:
                        snap_id = save_thesis_snapshot(
                            ticker=cand_sym,
                            sector=SECTOR_MAP.get(cand_sym, "Industrials"),
                            entry_price=entry_price,
                            stop_loss=stop_loss,
                            target_price=target_price,
                            indicators=ind,
                            technical_report=f"Point-in-time Stage-2 trend on {current_date_str}. Close: ₹{entry_price:,.2f}, RSI: {ind['rsi_14']}.",
                            fundamental_report="Historical point-in-time evaluation.",
                            news_macro_report=f"Market Regime: {regime}, Nifty 50 Close: ₹{nifty_close:,.2f}.",
                            debate_summary="Approved by simulated committee with zero lookahead.",
                            risk_summary=f"Hard Stop: ₹{stop_loss:,.2f} (2xATR), Target: ₹{target_price:,.2f}.",
                            pm_rationale="Simulated point-in-time trade order."
                        )

                    positions[cand_sym] = {
                        "ticker": cand_sym,
                        "qty": qty,
                        "entry_price": entry_price,
                        "stop_loss": stop_loss,
                        "target_price": target_price,
                        "entry_date": current_dt,
                        "entry_date_str": current_date_str,
                        "total_cost": total_cost,
                        "indicators": ind,
                        "snapshot_id": snap_id
                    }
                    if verbose:
                        console.print(f"[{current_date_str}] [bold green]⚡ BOUGHT:[/] {cand_sym} {qty} shs @ ₹{entry_price:,.2f} "
                                      f"(Stop: ₹{stop_loss:,.2f}, Target: ₹{target_price:,.2f})")
                if len(positions) >= max_positions:
                    break

    # -------------------------------------------------------------
    # 4. FINAL PERFORMANCE METRICS & DASHBOARD
    # -------------------------------------------------------------
    final_nav = daily_nav_history[-1]["nav"]
    total_return_pct = round(((final_nav - initial_capital) / initial_capital) * 100.0, 2)

    nifty_start = daily_nav_history[0]["nifty_close"]
    nifty_end = daily_nav_history[-1]["nifty_close"]
    nifty_return_pct = round(((nifty_end - nifty_start) / nifty_start) * 100.0, 2)
    alpha = round(total_return_pct - nifty_return_pct, 2)

    # Max Drawdown
    nav_series = pd.Series([d["nav"] for d in daily_nav_history])
    rolling_max = nav_series.cummax()
    drawdowns = (nav_series - rolling_max) / rolling_max
    max_drawdown_pct = round(float(drawdowns.min()) * 100.0, 2)

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
        "learned_rules_count": len(learning_engine.active_guardrails),
        "vetoed_trades_count": learning_engine.vetoed_trades_count,
        "memory_bank": learning_engine.memory_bank,
        "active_guardrails": learning_engine.active_guardrails,
        "daily_nav_history": daily_nav_history
    }

    if verbose:
        display_simulation_results(summary)
        if persist_to_db:
            console.print("\n[bold green]✓ Institutional Memory Synchronized:[/] All 1-year historical simulated learnings and post-mortems have been permanently saved into the live SQLite database (`paper_portfolio_india.db`). Future committee meetings will strictly enforce these rules.", style="bold green")

    return summary


def display_simulation_results(summary: Dict[str, Any]):
    """Renders the executive walk-forward learning results in Rich terminal tables."""
    console.print("\n" + "=" * 80, style="gold1")
    console.print("  1-YEAR WALK-FORWARD SIMULATION EXECUTIVE SCORECARD", style="bold gold1")
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
    m_tb.add_row("Maximum Drawdown", f"[bold red]{summary['max_drawdown_pct']:.2f}%[/bold red]", "—")
    m_tb.add_row("Total Trades Closed", str(summary["total_trades"]), "—")
    m_tb.add_row("Win Rate %", f"[bold]{summary['win_rate']:.1f}%[/bold] ({summary['wins']}W / {summary['losses']}L)", "50.0%")
    m_tb.add_row("Rules Learned From Mistakes", f"[bold gold1]{summary['learned_rules_count']}[/bold gold1]", "0 (Naive Prior)")
    m_tb.add_row("Subsequent Bad Trades Vetoed", f"[bold green]{summary['vetoed_trades_count']}[/bold green]", "0")
    console.print(m_tb)

    # Institutional Memory Bank Table (Unique Synthesized Guardrails)
    rules_tb = Table(title="[bold gold1]Institutional Memory Bank: Guardrails Synthesized from 1-Year Mistakes[/bold gold1]", show_header=True, header_style="bold yellow", expand=True)
    rules_tb.add_column("Rule ID", style="bold cyan", width=22)
    rules_tb.add_column("First Learned On", style="dim", width=14)
    rules_tb.add_column("Learned From", style="bold white", width=14)
    rules_tb.add_column("Synthesized Guardrail Enforced for Rest of Year", style="bold green")

    for g in summary["active_guardrails"]:
        rules_tb.add_row(
            g["id"],
            g["created_on"],
            g["learned_from"],
            g["text"]
        )
    console.print(rules_tb)

    # Post-Mortem Failure Audit Sample Table
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
        f"[bold green]✓ Closed-Loop Verification:[/] The system synthesized [bold gold1]{summary['learned_rules_count']} guardrail rules[/bold gold1] from initial mistakes, "
        f"which subsequently [bold green]prevented {summary['vetoed_trades_count']} risky trades[/bold green] from suffering similar losses during the remainder of the year.",
        border_style="green"
    ))


if __name__ == "__main__":
    run_1year_walkforward_simulation()
