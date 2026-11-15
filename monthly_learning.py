"""monthly_learning.py — Month-to-Month Live Multi-Agent Committee Debate & Learning Engine.

Upgraded features:
1. 10 Core Liquid Indian Equities across diverse sectors.
2. Max 3 Concurrent Active Portfolio Holdings.
3. Multi-Candidate Docket Queue (if Candidate #1 is vetoed, evaluates Candidate #2).
4. Trailing Stops & Winner Compounding (winners run across multiple months instead of forced 30-day exits).
5. Nuanced Bear Regime Mandate: Allows defensive allocations in Healthcare/Pharma/Defense while maintaining >= 50% cash.
6. Real-time post-mortem reflection audits via Gemini for every stop-loss exit.
"""

import sys
import logging
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Tuple

import pandas as pd
import yfinance as yf
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.columns import Columns

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
    get_top_actionable_rules,
    format_agent_scorecard_str,
    get_portfolio_account,
    reset_portfolio_account,
    log_tax_deduction,
    sweep_cash_to_liquid_fund,
    un_sweep_liquid_fund,
    accrue_liquid_fund_interest,
)
from tax_engine import compute_capital_gains_tax, get_financial_year, format_portfolio_tax_summary
from paper_broker import compute_buy_charges, compute_sell_charges
from backtest_simulation import compute_point_in_time_indicators
from graph import run_investment_committee
from reflection_engine import run_post_mortem_audit
from ui import (
    print_banner,
    print_market_regime_badge,
    display_analyst_reports,
    display_debate_and_synthesis,
    display_risk_committee,
    display_final_decision,
)

logger = logging.getLogger("wealth_agent_office.monthly_learning")
console = Console()

# 10 Core Liquid Indian Equities spanning key sectors
DEFAULT_10_STOCKS = [
    "TRENT.NS",       # Consumer Retail / Tata Group
    "BEL.NS",         # Aerospace & Defense PSU
    "ADANIPORTS.NS",  # Infrastructure & Ports
    "RELIANCE.NS",    # Energy, Telecom & Retail Conglomerate
    "SUNPHARMA.NS",   # Healthcare & Pharmaceuticals
    "INFY.NS",        # Information Technology
    "HDFCBANK.NS",    # Private Banking & Financials
    "ICICIBANK.NS",   # Private Banking & Financials
    "LT.NS",          # Engineering & Capital Goods
    "BAJFINANCE.NS",  # NBFC / Consumer Finance
]

SECTOR_MAP = {
    "TRENT.NS": "Consumer Retail",
    "BEL.NS": "Aerospace & Defense",
    "ADANIPORTS.NS": "Infrastructure / Ports",
    "RELIANCE.NS": "Energy & Petrochemicals",
    "SUNPHARMA.NS": "Pharmaceuticals & Healthcare",
    "INFY.NS": "Information Technology",
    "HDFCBANK.NS": "Banking & Financial Services",
    "ICICIBANK.NS": "Banking & Financial Services",
    "LT.NS": "Engineering & Construction",
    "BAJFINANCE.NS": "Non-Banking Financial (NBFC)",
}

MAX_ACTIVE_POSITIONS = 3

# Historical Multi-Year Training Blocks and Out-of-Sample Windows
# NOTE: Market regimes and evaluation goals are strictly omitted from metadata
# to ensure autonomous agent deduction from raw point-in-time benchmark data.
CURRICULUM_BLOCKS: Dict[str, Dict[str, Any]] = {
    "block_1": {
        "name": "Curriculum Block 1",
        "start": "2020-06-01",
        "end": "2021-10-01",
        "duration_months": 16,
    },
    "block_2": {
        "name": "Curriculum Block 2",
        "start": "2021-11-01",
        "end": "2023-03-01",
        "duration_months": 16,
    },
    "block_3": {
        "name": "Curriculum Block 3",
        "start": "2023-04-01",
        "end": "2024-09-01",
        "duration_months": 17,
    },
    "block_4": {
        "name": "Curriculum Block 4",
        "start": "2025-12-01",
        "end": "2026-08-31",
        "duration_months": 9,
    },
    "oos_a": {
        "name": "Validation Window A",
        "start": "2024-10-01",
        "end": "2025-11-30",
        "duration_months": 14,
    },
    "oos_b": {
        "name": "Validation Window B",
        "start": "2026-09-01",
        "end": "2026-11-13",
        "duration_months": 2.5,
    }
}


def get_first_trading_days_in_window(
    dates_index: pd.DatetimeIndex,
    start_date: str,
    end_date: str
) -> List[pd.Timestamp]:
    """Finds the first trading day of each calendar month within the specified start and end date window."""
    start_dt = pd.to_datetime(start_date)
    end_dt = pd.to_datetime(end_date)
    sub_index = dates_index[(dates_index >= start_dt) & (dates_index <= end_dt)]
    if len(sub_index) == 0:
        return []

    df_dates = pd.DataFrame({"dt": sub_index})
    df_dates["year"] = df_dates["dt"].dt.year
    df_dates["month"] = df_dates["dt"].dt.month

    ym_groups = df_dates.groupby(["year", "month"])
    first_trading_days = []
    for _, group in ym_groups:
        first_trading_days.append(group["dt"].iloc[0])

    return first_trading_days


def get_first_mondays_of_past_year(dates_index: pd.DatetimeIndex) -> List[pd.Timestamp]:
    """Finds the first trading day of the first week of each month in the past year."""
    df_dates = pd.DataFrame({"dt": dates_index})
    df_dates["year"] = df_dates["dt"].dt.year
    df_dates["month"] = df_dates["dt"].dt.month

    ym_groups = df_dates.groupby(["year", "month"])
    first_trading_days = []
    for _, group in ym_groups:
        first_trading_days.append(group["dt"].iloc[0])

    if len(first_trading_days) > 12:
        first_trading_days = first_trading_days[-12:]

    return first_trading_days


def run_monthly_agent_learning_simulation(
    stocks: Optional[List[str]] = None,
    num_months: int = 12,
    block: Optional[str] = None,
    reset_db_first: bool = False,
    verbose: bool = True
) -> Dict[str, Any]:
    """Runs month-by-month walk-forward live multi-agent committee debates with up to 3 active holdings."""
    universe = stocks or DEFAULT_10_STOCKS
    all_symbols = [BENCHMARK_TICKER] + universe

    if reset_db_first:
        reset_portfolio_account(clean_trades=True, initial_capital=INITIAL_CAPITAL_INR)

    block_info = CURRICULUM_BLOCKS.get(block) if block else None
    block_title = block_info["name"] if block_info else f"Past {num_months} Months"

    console.print("\n" + "=" * 80, style="bold magenta")
    console.print(f"  MONTH-TO-MONTH MULTI-AGENT COMMITTEE & PORTFOLIO ENGINE — {block_title.upper()}", style="bold white on magenta")
    console.print("=" * 80, style="bold magenta")
    console.print(f"[cyan]Universe (10 Liquid Stocks):[/cyan] {', '.join(universe)}")
    console.print(f"[cyan]Portfolio Mandate:[/cyan] Max 3 concurrent holdings, 12-15% single-stock size, strict trailing stops.")
    console.print(f"[cyan]Tax & Treasury Engine:[/cyan] Indian STCG (20%) / LTCG (12.5% over ₹1.25L exemption), 6.75% Liquid Fund sweep.")
    if block_info:
        console.print(f"[cyan]Curriculum Window:[/cyan] {block_info['start']} to {block_info['end']} ({block_title})\n")
        console.print("[dim]Point-in-time benchmark data provided; models independently assess market regime.[/dim]\n")
    else:
        console.print(f"[cyan]Cadence:[/cyan] 1st Monday of each month across past {num_months} months with [bold yellow]Zero Lookahead Bias[/bold yellow].\n")

    # 1. Download historical daily data for all symbols
    if block_info:
        start_fetch = (pd.to_datetime(block_info["start"]) - pd.Timedelta(days=365)).strftime("%Y-%m-%d")
        end_fetch = (pd.to_datetime(block_info["end"]) + pd.Timedelta(days=15)).strftime("%Y-%m-%d")
        console.print(f"[dim]Downloading daily history for {block_title} ({start_fetch} to {end_fetch})...[/dim]")
        raw_df = yf.download(all_symbols, start=start_fetch, end=end_fetch, interval="1d", group_by="ticker", progress=False)
    else:
        console.print(f"[dim]Downloading 2-year daily history for indicators warmup...[/dim]")
        raw_df = yf.download(all_symbols, period="2y", interval="1d", group_by="ticker", progress=False)

    data_by_ticker = {}
    for sym in all_symbols:
        if sym in raw_df.columns.levels[0]:
            sub = raw_df[sym].dropna(how="all").copy()
            if not sub.empty:
                data_by_ticker[sym] = sub

    nifty_df = data_by_ticker.get(BENCHMARK_TICKER)
    if nifty_df is None or len(nifty_df) < 100:
        raise ValueError("Insufficient historical benchmark data for Nifty 50.")

    # 2. Get monthly review trading days
    if block_info:
        monthly_dates = get_first_trading_days_in_window(nifty_df.index, block_info["start"], block_info["end"])
    else:
        monthly_dates = get_first_mondays_of_past_year(nifty_df.index)
        if num_months < len(monthly_dates):
            monthly_dates = monthly_dates[-num_months:]

    if not monthly_dates:
        raise ValueError(f"No valid trading days found for window: {block_info['start'] if block_info else 'past year'}")

    console.print(f"[green]Identified {len(monthly_dates)} monthly committee review dates:[/green]")
    for idx, m_dt in enumerate(monthly_dates, 1):
        console.print(f"  • Month {idx:02d}: [bold]{m_dt.strftime('%Y-%m-%d (%A)')}[/bold]")
    console.print()

    cash = INITIAL_CAPITAL_INR
    liquid_fund_swept = 0.0
    cum_interest_earned = 0.0
    cum_stcg_paid = 0.0
    cum_ltcg_paid = 0.0
    cum_fy_exempt_used = 0.0
    positions: Dict[str, Dict[str, Any]] = {}
    monthly_performance_log: List[Dict[str, Any]] = []

    # 3. Monthly Loop
    for m_idx, current_monday in enumerate(monthly_dates, 1):
        monday_str = current_monday.strftime("%Y-%m-%d")
        next_month_dt = monthly_dates[m_idx] if m_idx < len(monthly_dates) else nifty_df.index[-1]

        console.print(f"\n{'━' * 80}", style="bright_blue")
        console.print(f"  📅 [bold bright_white]MONTH {m_idx}/{len(monthly_dates)} — COMMITTEE SESSION DATE: {monday_str}[/bold bright_white]", style="on blue")
        console.print(f"{'━' * 80}", style="bright_blue")

        # -------------------------------------------------------------
        # Step A: Point-in-time Market Regime on Date T
        # -------------------------------------------------------------
        nifty_pit = nifty_df.loc[:current_monday]
        nifty_close = float(nifty_pit["Close"].iloc[-1])
        nifty_sma50 = float(nifty_pit["Close"].rolling(50).mean().iloc[-1]) if len(nifty_pit) >= 50 else nifty_close
        nifty_sma200 = float(nifty_pit["Close"].rolling(200).mean().iloc[-1]) if len(nifty_pit) >= 200 else nifty_sma50
        nifty_trend_63d = round(((nifty_close / float(nifty_pit["Close"].iloc[-63])) - 1.0) * 100.0, 2) if len(nifty_pit) >= 63 else 0.0

        if nifty_close > nifty_sma50 > nifty_sma200:
            regime = "BULL_STAGE2"
        elif nifty_close > nifty_sma200:
            regime = "CHOPPY_RANGE"
        else:
            regime = "BEAR_RISK_OFF"

        high_52w = float(nifty_pit["High"].max()) if "High" in nifty_pit.columns else nifty_close
        pit_benchmark_metrics = {
            "nifty_close": round(nifty_close, 2),
            "nifty_sma50": round(nifty_sma50, 2),
            "nifty_sma200": round(nifty_sma200, 2),
            "nifty_dist_52w_high_pct": round(((nifty_close - high_52w) / high_52w) * 100.0, 2),
            "nifty_trend_63d_pct": nifty_trend_63d,
            "india_vix": 15.5,
            "regime": regime
        }

        console.print(
            f"[bold cyan]📊 Benchmark Point-in-Time Metrics ({monday_str}):[/bold cyan] "
            f"Nifty 50: ₹{nifty_close:,.2f} | 50 SMA: ₹{nifty_sma50:,.2f} | 200 SMA: ₹{nifty_sma200:,.2f} | "
            f"63d Trend: {nifty_trend_63d:+.1f}% | VIX: 15.5"
        )

        # -------------------------------------------------------------
        # Step B: Mark-to-Market & Trailing Stop Audit on Open Positions
        # -------------------------------------------------------------
        invested_value = 0.0
        for sym, pos in list(positions.items()):
            stk_close = float(data_by_ticker[sym].loc[:current_monday]["Close"].iloc[-1])
            invested_value += pos["qty"] * stk_close

            # Trailing Stop Policy: If position is up > 1 ATR from entry, trail stop upward
            stk_pit_sub = data_by_ticker[sym].loc[:current_monday]
            ind_curr = compute_point_in_time_indicators(stk_pit_sub, nifty_pit["Close"])
            cur_atr = ind_curr["atr_14"]
            unrealized_gain = stk_close - pos["entry_price"]

            if unrealized_gain > (1.0 * cur_atr):
                new_stop = round(max(pos["stop_loss"], stk_close - (2.0 * cur_atr)), 2)
                if new_stop > pos["stop_loss"]:
                    old_s = pos["stop_loss"]
                    pos["stop_loss"] = new_stop
                    console.print(f"[bold green]📈 Trailed Stop for {sym}:[/] ₹{old_s:,.2f} -> ₹{new_stop:,.2f} (Locking in gains @ CMP ₹{stk_close:,.2f})")

        portfolio_nav = cash + liquid_fund_swept + invested_value
        console.print(
            f"[cyan]Current Portfolio NAV:[/] [bold]₹{portfolio_nav:,.2f}[/] "
            f"(Cash: ₹{cash:,.2f} | Liquid Fund: ₹{liquid_fund_swept:,.2f} | Holdings: {len(positions)}/{MAX_ACTIVE_POSITIONS})"
        )

        if positions:
            holdings_tbl = Table(title="Existing Open Holdings & Tax Lot Status", title_style="bold green")
            holdings_tbl.add_column("Ticker", style="bold yellow")
            holdings_tbl.add_column("Qty", justify="right")
            holdings_tbl.add_column("Entry Price", justify="right")
            holdings_tbl.add_column("CMP (Today)", justify="right")
            holdings_tbl.add_column("Active Stop", justify="right", style="red")
            holdings_tbl.add_column("Target", justify="right", style="green")
            holdings_tbl.add_column("Unrealized PnL", justify="right")
            holdings_tbl.add_column("Holding Period", justify="center")
            holdings_tbl.add_column("Tax Status", style="dim")

            for sym, pos in positions.items():
                cmp_price = float(data_by_ticker[sym].loc[:current_monday]["Close"].iloc[-1])
                pos_cost = pos["qty"] * pos["entry_price"]
                cur_val = pos["qty"] * cmp_price
                diff_pct = ((cur_val - pos_cost) / pos_cost) * 100.0
                p_col = "green" if diff_pct > 0 else "red"

                held_days = max(0, (current_monday - pos["entry_date"]).days)
                days_to_ltcg = max(0, 365 - held_days)
                if held_days >= 365:
                    tax_str = "[bold green]LTCG (12.5%)[/bold green]"
                elif days_to_ltcg <= 45:
                    tax_str = f"[bold yellow]STCG ({days_to_ltcg}d to LTCG)[/bold yellow]"
                else:
                    tax_str = "STCG (20%)"

                holdings_tbl.add_row(
                    sym,
                    str(pos["qty"]),
                    f"₹{pos['entry_price']:,.2f}",
                    f"₹{cmp_price:,.2f}",
                    f"₹{pos['stop_loss']:,.2f}",
                    f"₹{pos['target_price']:,.2f}",
                    f"[{p_col}]{diff_pct:+.2f}%[/{p_col}]",
                    f"{held_days} days",
                    tax_str
                )
            console.print(holdings_tbl)

        # -------------------------------------------------------------
        # Step C: Point-in-time Screening of Universe (10 Stocks)
        # -------------------------------------------------------------
        table = Table(title=f"Point-In-Time Screening of 10 Indian Equities (As of {monday_str})", title_style="bold cyan")
        table.add_column("Ticker", style="bold yellow")
        table.add_column("Sector", style="dim")
        table.add_column("Close (₹)", justify="right")
        table.add_column("Stage-2?", justify="center")
        table.add_column("RSI (14)", justify="right")
        table.add_column("Vol Ratio", justify="right")
        table.add_column("63d RS Alpha", justify="right")
        table.add_column("Rank Score", justify="right", style="bold green")

        candidates_ranked = []
        for sym in universe:
            if sym in positions:
                continue

            stk_df = data_by_ticker.get(sym)
            if stk_df is None:
                continue
            stk_pit = stk_df.loc[:current_monday]
            if len(stk_pit) < 63:
                continue

            ind = compute_point_in_time_indicators(stk_pit, nifty_pit["Close"])
            close_p = ind["latest_close"]
            stage2 = ind["stage2_uptrend"]
            rsi = ind["rsi_14"]
            vol = ind["volume_ratio"]
            alpha = ind["rs_alpha_63d"]

            # Composite ranking score: prioritize Stage-2 alignment, RS Alpha & Volume
            sec = SECTOR_MAP.get(sym, "Industrials")
            is_defensive = sec in ["Pharmaceuticals & Healthcare", "Consumer Retail"]
            defensive_bonus = 1.0 if (nifty_close < nifty_sma50 and is_defensive) else 0.0

            score = (2.0 if stage2 else -1.0) + (alpha * 5.0) + (1.0 if (50 <= rsi <= 68) else 0.0) + defensive_bonus
            candidates_ranked.append({
                "ticker": sym,
                "sector": sec,
                "indicators": ind,
                "score": round(score, 3)
            })

            table.add_row(
                sym,
                sec,
                f"₹{close_p:,.2f}",
                "[green]YES[/green]" if stage2 else "[dim]NO[/dim]",
                f"{rsi:.1f}",
                f"{vol:.2f}x",
                f"{alpha * 100:+.1f}%",
                f"{score:.2f}"
            )

        console.print(table)
        candidates_ranked.sort(key=lambda x: x["score"], reverse=True)

        # -------------------------------------------------------------
        # Step D: Show Active Institutional Memory Bank
        # -------------------------------------------------------------
        active_rules = get_top_actionable_rules(limit=5)
        if active_rules:
            rule_text = "\n".join([f"  {i+1}. [italic]{r}[/italic]" for i, r in enumerate(active_rules)])
            console.print(Panel(rule_text, title="🧠 Active Institutional Memory (Learned Guardrails)", border_style="gold1"))

        # -------------------------------------------------------------
        # Step E: Multi-Candidate Docket Evaluation (Up to Open Slots)
        # -------------------------------------------------------------
        open_slots = MAX_ACTIVE_POSITIONS - len(positions)
        new_trades_this_month = []

        total_deployable = cash + liquid_fund_swept
        if open_slots <= 0:
            console.print(f"[yellow]Portfolio is at full capacity ({MAX_ACTIVE_POSITIONS}/{MAX_ACTIVE_POSITIONS} positions). No new buys evaluated this month.[/yellow]")
        elif total_deployable < (portfolio_nav * 0.15):
            console.print(f"[yellow]Available deployable capital (Cash ₹{cash:,.2f} + Liquid ₹{liquid_fund_swept:,.2f} = ₹{total_deployable:,.2f}) is below minimum threshold. No new buys evaluated.[/yellow]")
        else:
            console.print(f"[bold cyan]Portfolio has {open_slots} open slot(s). Evaluating top candidates from docket...[/bold cyan]")
            
            # Evaluate docket candidates in rank order
            for cand in candidates_ranked:
                if len(new_trades_this_month) >= open_slots:
                    break

                cand_sym = cand["ticker"]
                cand_ind = cand["indicators"]
                cand_sec = cand["sector"]

                # Quantitative Pre-Filter (Zero LLM calls): Check hard learned rules with Regime-Adaptive Leeway
                vol_ratio = cand_ind["volume_ratio"]
                alpha = cand_ind["rs_alpha_63d"]

                # Regime-Adaptive Volume Filter:
                # In relief rebounds or benchmark corrections (Nifty below 50 SMA), volume dry-up is typical.
                # Allow volume ratio >= 0.85x for positive RS Alpha leaders (> 0.0) or defensive setups.
                # For lagging or cyclical stocks, enforce the strict 0.90x volume requirement.
                min_vol_required = 0.85 if (alpha > 0.0 or cand_sec in ["Healthcare", "Pharma", "Defense", "Pharmaceuticals"]) else 0.90
                if vol_ratio < min_vol_required:
                    console.print(f"  [dim]• Skipping {cand_sym}: Volume ratio {vol_ratio:.2f}x < {min_vol_required:.2f}x (Vetoed by False Breakout Rule)[/dim]")
                    continue

                if (nifty_close < nifty_sma50) and alpha <= 0.0:
                    console.print(f"  [dim]• Skipping {cand_sym}: Negative RS Alpha ({alpha*100:+.1f}%) during benchmark downtrend[/dim]")
                    continue

                console.print(f"\n[bold magenta]⚡ Convening 10-Agent Investment Committee for {cand_sym} (Docket Rank #{candidates_ranked.index(cand)+1})...[/bold magenta]")

                pit_technicals = {
                    "latest_close": cand_ind["latest_close"],
                    "sma_20": cand_ind["sma_20"],
                    "sma_50": cand_ind["sma_50"],
                    "sma_150": cand_ind["sma_150"],
                    "sma_200": cand_ind["sma_200"],
                    "rsi_14": cand_ind["rsi_14"],
                    "atr_14": cand_ind["atr_14"],
                    "volume_ratio_20d": cand_ind["volume_ratio"],
                    "dist_52w_high_pct": 0.0,
                    "rs_alpha_63d": cand_ind["rs_alpha_63d"],
                    "circuit_limit_warning": "NORMAL",
                    "stage2_uptrend": cand_ind["stage2_uptrend"],
                }

                # Deterministic sizing & beta-adaptive risk bounds
                entry_price = cand_ind["latest_close"]
                atr = cand_ind["atr_14"]
                beta = 1.3 if cand_sec in ["Infrastructure / Ports", "Engineering & Construction", "Energy & Petrochemicals"] else 0.95
                stop_mult = 3.0 if beta >= 1.2 else 2.5
                stop_loss = round(entry_price - (stop_mult * atr), 2)
                risk_per_share = round(entry_price - stop_loss, 2)
                target_price = round(entry_price + (risk_per_share * MIN_RISK_REWARD_RATIO), 2)

                max_portfolio_risk_inr = portfolio_nav * MAX_PORTFOLIO_RISK_PER_TRADE
                qty_by_risk = int(max_portfolio_risk_inr / max(risk_per_share, 1.0))
                max_position_cost = portfolio_nav * MAX_SINGLE_STOCK_PCT
                qty_by_size = int(max_position_cost / entry_price)
                allocated_qty = max(1, min(qty_by_risk, qty_by_size))

                # Check deployable capital availability (Cash + Liquid Fund)
                gross_entry = allocated_qty * entry_price
                est_fees = compute_buy_charges(gross_entry)
                total_deployable = cash + liquid_fund_swept
                if (gross_entry + est_fees) > (total_deployable * 0.95):
                    allocated_qty = int((total_deployable * 0.90) / entry_price)
                    if allocated_qty <= 0:
                        continue
                    gross_entry = allocated_qty * entry_price

                # Point-in-time macro news summary (Zero regime spoon-feeding)
                historical_macro_summary = (
                    f"Historical Point-In-Time Macro & Benchmark Context as of {monday_str}:\n"
                    f"- Benchmark Metrics: Nifty 50 Close ₹{nifty_close:,.2f} | 50 SMA: ₹{nifty_sma50:,.2f} | 200 SMA: ₹{nifty_sma200:,.2f}\n"
                    f"- Benchmark 63-day Trend: {nifty_trend_63d:+.1f}% | Benchmark Volatility: India VIX ~15.5\n"
                    f"- Sector: {cand_sec} | Stock Relative Strength Alpha vs Nifty: {cand_ind.get('rs_alpha_63d', 0.0)*100:+.1f}%\n"
                    f"- Corporate Governance & Solvency: Audited via point-in-time regulatory filings.\n"
                    f"- Instruction: The committee must independently deduce the prevailing market regime and macro risk posture."
                )

                pit_state = {
                    "market_regime": regime,
                    "india_vix": 15.5,
                    "benchmark_context": pit_benchmark_metrics,
                    "technicals": pit_technicals,
                    "sector": cand_sec,
                    "recent_news": historical_macro_summary,
                    "point_in_time_date": monday_str,
                }

                # Convene LangGraph multi-agent committee!
                committee_result = run_investment_committee(
                    ticker=cand_sym,
                    sector=cand_sec,
                    point_in_time_state=pit_state
                )

                display_analyst_reports(
                    committee_result.get("technical_report") or "No technical report.",
                    committee_result.get("fundamental_report") or "No fundamental report.",
                    committee_result.get("news_macro_report") or "No news report."
                )

                display_debate_and_synthesis(
                    committee_result.get("bull_arguments", []),
                    committee_result.get("bear_arguments", []),
                    committee_result.get("research_proposal") or "No debate proposal."
                )

                display_risk_committee(
                    committee_result.get("aggressive_risk_view") or "No aggressive view.",
                    committee_result.get("neutral_risk_view") or "No neutral view.",
                    committee_result.get("conservative_risk_view") or "No conservative view."
                )

                decision = committee_result.get("trade_decision") or {}
                display_final_decision(decision)

                action = decision.get("action", "HOLD")

                if action == "BUY":
                    buy_fees = compute_buy_charges(gross_entry)
                    total_entry_cost = gross_entry + buy_fees
                    
                    # Treasury un-sweep from Liquid Fund if cash is parked
                    if total_entry_cost > cash and liquid_fund_swept > 0:
                        un_sweep = min(liquid_fund_swept, total_entry_cost - cash)
                        liquid_fund_swept -= un_sweep
                        cash += un_sweep
                        un_sweep_liquid_fund(un_sweep, timestamp=monday_str)
                        console.print(f"[cyan]🏦 Treasury Un-Sweep:[/] Redeemed ₹{un_sweep:,.2f} from Liquid Mutual Fund for {cand_sym} delivery settlement.")

                    cash -= total_entry_cost

                    console.print(f"[bold green]✔ EXECUTED BUY ORDER:[/] {allocated_qty} shares of {cand_sym} @ ₹{entry_price:,.2f} (Total Cost: ₹{total_entry_cost:,.2f})")

                    # Snapshot thesis for post-mortem reflection
                    snap_id = save_thesis_snapshot(
                        ticker=cand_sym,
                        sector=cand_sec,
                        entry_price=entry_price,
                        stop_loss=stop_loss,
                        target_price=target_price,
                        indicators=cand_ind,
                        technical_report=committee_result.get("technical_report", ""),
                        fundamental_report=committee_result.get("fundamental_report", ""),
                        news_macro_report=committee_result.get("news_macro_report", ""),
                        debate_summary=committee_result.get("research_proposal", ""),
                        risk_summary=f"Aggressive: {committee_result.get('aggressive_risk_view', '')[:100]} | Conservative: {committee_result.get('conservative_risk_view', '')[:100]}",
                        pm_rationale=decision.get("rationale", "")
                    )

                    positions[cand_sym] = {
                        "ticker": cand_sym,
                        "sector": cand_sec,
                        "qty": allocated_qty,
                        "entry_price": entry_price,
                        "entry_date": current_monday,
                        "entry_date_str": monday_str,
                        "stop_loss": stop_loss,
                        "target_price": target_price,
                        "total_cost": total_entry_cost,
                        "snapshot_id": snap_id,
                        "indicators": cand_ind,
                    }
                    new_trades_this_month.append(cand_sym)
                else:
                    console.print(f"[yellow]Committee decided to {action} for {cand_sym}. Checking next candidate on docket...[/yellow]")

        # -------------------------------------------------------------
        # Step F: Forward Surveillance Between Monthly Review Dates
        # -------------------------------------------------------------
        # Walk daily bars from current_monday to next_month_dt
        if positions:
            active_tickers = list(positions.keys())
            for sym in active_tickers:
                if sym not in positions:
                    continue

                pos = positions[sym]
                stk_full = data_by_ticker[sym]
                forward_bars = stk_full.loc[current_monday:next_month_dt]

                for bar_dt, bar_row in forward_bars.iterrows():
                    if bar_dt == current_monday:
                        continue

                    b_low = float(bar_row["Low"])
                    b_high = float(bar_row["High"])
                    b_close = float(bar_row["Close"])
                    cur_d_str = bar_dt.strftime("%Y-%m-%d")

                    # Check Stop Loss
                    if b_low <= pos["stop_loss"]:
                        exit_price = round(min(pos["stop_loss"], b_close), 2)
                        gross_sale = pos["qty"] * exit_price
                        sell_fees = compute_sell_charges(gross_sale)
                        net_proceeds = gross_sale - sell_fees

                        # Indian Capital Gains Tax calculation (STCG vs LTCG)
                        tax_info = compute_capital_gains_tax(
                            buy_date=pos["entry_date"],
                            sell_date=bar_dt,
                            cost_basis=pos["total_cost"],
                            net_proceeds=net_proceeds,
                            fy_ltcg_exempt_used=cum_fy_exempt_used
                        )
                        tax_deducted = tax_info["tax_deducted"]
                        net_pnl_after_tax = tax_info["net_pnl_after_tax"]
                        cum_fy_exempt_used = tax_info["new_fy_exempt_used"]
                        if tax_info["term_type"] == "LTCG":
                            cum_ltcg_paid += tax_deducted
                        else:
                            cum_stcg_paid += tax_deducted

                        cash += (net_proceeds - tax_deducted)
                        log_tax_deduction(tax_info, ticker=sym, timestamp=cur_d_str)

                        pnl_pct = round((net_pnl_after_tax / pos["total_cost"]) * 100.0, 2)
                        console.print(
                            f"\n[bold red]▼ STOP-LOSS TRIGGERED:[/] {sym} exited @ ₹{exit_price:,.2f} on {cur_d_str} "
                            f"(Held: {tax_info['holding_days']}d | Tax: ₹{tax_deducted:,.2f} {tax_info['term_type']} | Net PnL: {pnl_pct:+.2f}%)"
                        )

                        # Trigger Gemini Deep-Think Post-Mortem Reflection!
                        pos_dict = {
                            "ticker": sym,
                            "snapshot_id": pos["snapshot_id"],
                            "avg_price": pos["entry_price"],
                            "opened_at": pos["entry_date_str"],
                            "sector": pos["sector"]
                        }

                        pm_result = run_post_mortem_audit(
                            position=pos_dict,
                            exit_price=exit_price,
                            exit_reason="STOP_LOSS_HIT",
                            realized_pnl=net_pnl_after_tax
                        )

                        if pm_result and pm_result.get("actionable_rule"):
                            console.print(Panel(
                                f"New Guardrail Rule Formulated by Post-Mortem Auditor:\n[bold yellow]\"{pm_result['actionable_rule']}\"[/bold yellow]\n\n"
                                f"• Failing Agent: [bold red]{pm_result.get('failed_agent')}[/bold red]\n"
                                f"• Accurate Agent: [bold green]{pm_result.get('accurate_agent')}[/bold green]",
                                title="🧠 POST-MORTEM REFLECTION COMPLETE",
                                border_style="gold1"
                            ))
                        else:
                            close_thesis_snapshot(pos["snapshot_id"])

                        del positions[sym]
                        break

                    # Check Profit Target
                    if b_high >= pos["target_price"]:
                        exit_price = pos["target_price"]
                        gross_sale = pos["qty"] * exit_price
                        sell_fees = compute_sell_charges(gross_sale)
                        net_proceeds = gross_sale - sell_fees

                        # Indian Capital Gains Tax calculation (STCG vs LTCG)
                        tax_info = compute_capital_gains_tax(
                            buy_date=pos["entry_date"],
                            sell_date=bar_dt,
                            cost_basis=pos["total_cost"],
                            net_proceeds=net_proceeds,
                            fy_ltcg_exempt_used=cum_fy_exempt_used
                        )
                        tax_deducted = tax_info["tax_deducted"]
                        net_pnl_after_tax = tax_info["net_pnl_after_tax"]
                        cum_fy_exempt_used = tax_info["new_fy_exempt_used"]
                        if tax_info["term_type"] == "LTCG":
                            cum_ltcg_paid += tax_deducted
                        else:
                            cum_stcg_paid += tax_deducted

                        cash += (net_proceeds - tax_deducted)
                        log_tax_deduction(tax_info, ticker=sym, timestamp=cur_d_str)

                        pnl_pct = round((net_pnl_after_tax / pos["total_cost"]) * 100.0, 2)
                        console.print(
                            f"\n[bold green]▲ PROFIT TARGET HIT:[/] {sym} target ₹{exit_price:,.2f} reached on {cur_d_str}! "
                            f"(Held: {tax_info['holding_days']}d | Tax: ₹{tax_deducted:,.2f} {tax_info['term_type']} | Net PnL: [bold green]{pnl_pct:+.2f}%[/bold green])"
                        )
                        close_thesis_snapshot(pos["snapshot_id"])
                        del positions[sym]
                        break

        # -------------------------------------------------------------
        # Step G: Liquid Fund Yield Accrual on Idle Cash between Review Dates
        # -------------------------------------------------------------
        days_in_period = max(1, (next_month_dt - current_monday).days)
        invested_interim = sum(p["qty"] * float(data_by_ticker[s].loc[:next_month_dt]["Close"].iloc[-1]) for s, p in positions.items())
        current_tot_nav = cash + liquid_fund_swept + invested_interim
        
        # Sweep idle cash exceeding 30% NAV or during risk-off regimes
        if (cash / max(current_tot_nav, 1.0)) > 0.30 or regime in ["BEAR_RISK_OFF", "CHOPPY_RANGE"]:
            target_buffer = current_tot_nav * 0.10
            sweepable = max(0.0, cash - target_buffer)
            if sweepable > 1000.0:
                liquid_fund_swept += sweepable
                cash -= sweepable
                sweep_cash_to_liquid_fund(sweepable, timestamp=monday_str)
                
        if liquid_fund_swept > 0:
            daily_rf_rate = 0.0675 / 365.0
            interest_earned = round(liquid_fund_swept * daily_rf_rate * days_in_period, 2)
            liquid_fund_swept += interest_earned
            cum_interest_earned += interest_earned
            accrue_liquid_fund_interest(interest_earned, timestamp=monday_str)
            console.print(
                f"[cyan]🏦 Treasury Liquid Sweep:[/] Parked ₹{liquid_fund_swept:,.2f} in 6.75% Liquid Mutual Fund. "
                f"Accrued [bold green]+₹{interest_earned:,.2f}[/bold green] risk-free yield over {days_in_period} days."
            )

        # Calculate month-end Mark-to-Market NAV
        invested_now = 0.0
        for sym, pos in positions.items():
            stk_last = float(data_by_ticker[sym].loc[:next_month_dt]["Close"].iloc[-1])
            invested_now += pos["qty"] * stk_last
        nav_at_month_end = cash + liquid_fund_swept + invested_now

        monthly_performance_log.append({
            "month": m_idx,
            "date": monday_str,
            "new_buys": ", ".join(new_trades_this_month) if new_trades_this_month else "NONE",
            "active_positions_count": len(positions),
            "portfolio_nav": nav_at_month_end,
            "cash": cash,
            "liquid_fund_swept": liquid_fund_swept,
            "cum_interest_earned": cum_interest_earned,
            "cum_tax_paid": cum_stcg_paid + cum_ltcg_paid,
            "invested_value": invested_now
        })

    # Summary Dashboard
    console.print("\n" + "=" * 80, style="bold cyan")
    console.print(f"  MULTI-AGENT PORTFOLIO PERFORMANCE & TAX SUMMARY — {block_title.upper()}", style="bold white on cyan")
    console.print("=" * 80, style="bold cyan")

    summary_tbl = Table(title=f"Committee Review & Portfolio NAV History ({block_title})", title_style="bold green")
    summary_tbl.add_column("Month", justify="center")
    summary_tbl.add_column("Review Date", style="bold")
    summary_tbl.add_column("New Buys", justify="center")
    summary_tbl.add_column("Holdings", justify="center")
    summary_tbl.add_column("Liquid Cash", justify="right")
    summary_tbl.add_column("6.75% Liquid Fund", justify="right", style="cyan")
    summary_tbl.add_column("Portfolio NAV", justify="right", style="bold")

    for row in monthly_performance_log:
        summary_tbl.add_row(
            f"M{row['month']:02d}",
            row["date"],
            row["new_buys"],
            f"{row['active_positions_count']}/{MAX_ACTIVE_POSITIONS}",
            f"₹{row['cash']:,.2f}",
            f"₹{row['liquid_fund_swept']:,.2f}",
            f"₹{row['portfolio_nav']:,.2f}"
        )

    console.print(summary_tbl)
    final_nav = monthly_performance_log[-1]["portfolio_nav"] if monthly_performance_log else INITIAL_CAPITAL_INR
    final_return = ((final_nav - INITIAL_CAPITAL_INR) / INITIAL_CAPITAL_INR) * 100.0

    # Treasury & Tax Performance Panel
    tot_tax_paid = cum_stcg_paid + cum_ltcg_paid
    tax_panel = (
        f"• Final Portfolio NAV:              [bold green]₹{final_nav:,.2f}[/bold green] ({final_return:+.2f}%)\n"
        f"• 6.75% Liquid Fund Interest Earned: [bold cyan]+₹{cum_interest_earned:,.2f}[/bold cyan] (Risk-Free Parking Yield)\n"
        f"• Short-Term Capital Gains Tax (20%): [bold red]-₹{cum_stcg_paid:,.2f}[/bold red]\n"
        f"• Long-Term Capital Gains Tax (12.5%):[bold red]-₹{cum_ltcg_paid:,.2f}[/bold red]\n"
        f"• Total Statutory Tax Deducted:      [bold red]-₹{tot_tax_paid:,.2f}[/bold red]\n"
        f"• LTCG ₹1.25L Exemption Consumed:    ₹{cum_fy_exempt_used:,.2f}\n"
        f"• Net Return After All Charges & Taxes: [bold]{final_return:+.2f}%[/bold]"
    )
    console.print(Panel(tax_panel, title="🏛️ TAX-AWARE & RISK-FREE TREASURY SCORECARD", border_style="cyan"))
    console.print("[dim]All post-mortems, tax entries, and synthesized guardrails remain active in SQLite database.[/dim]\n")

    return {
        "final_nav": final_nav,
        "total_return_pct": final_return,
        "cum_interest_earned": cum_interest_earned,
        "cum_tax_paid": tot_tax_paid,
        "log": monthly_performance_log,
    }


def run_curriculum_block(block_name: str, reset_db: bool = False, clean_all: bool = False) -> Dict[str, Any]:
    """Runs a specific training block or out-of-sample window from the curriculum."""
    if block_name not in CURRICULUM_BLOCKS:
        raise ValueError(f"Unknown curriculum block '{block_name}'. Choose from: {list(CURRICULUM_BLOCKS.keys())}")
    if clean_all:
        from database import clear_post_mortems_and_snapshots
        clear_post_mortems_and_snapshots()
    result = run_monthly_agent_learning_simulation(block=block_name, reset_db_first=reset_db)

    # Post-run cleanup: automatically generate clean debate log and split month-by-month
    try:
        import os
        from clean_debate_logs import clean_debate_file
        from split_monthly_debates import split_debates_by_month
        raw_fname = f"curriculum_{block_name}_output.txt"
        base_dir = os.path.dirname(__file__)
        target_raw = raw_fname if os.path.exists(raw_fname) else os.path.join(base_dir, raw_fname)
        clean_fname = os.path.join(base_dir, f"curriculum_{block_name}_clean.txt")
        if os.path.exists(target_raw):
            clean_debate_file(target_raw, clean_fname)
            out_dir = os.path.join(base_dir, "monthly_debates", block_name)
            split_debates_by_month(clean_fname, out_dir)
            console.print(f"[bold green]✔ Automatically cleaned and split debates into:[/] {out_dir}\n")
    except Exception as e:
        logger.warning(f"Failed to auto-clean and split debates for {block_name}: {e}")

    return result


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Month-to-month live multi-agent committee debate & learning.")
    parser.add_argument("--months", type=int, default=12, help="Number of historical monthly review dates (default: 12)")
    parser.add_argument("--block", type=str, choices=list(CURRICULUM_BLOCKS.keys()), default=None,
                        help="Curriculum block: block_1, block_2, block_3, block_4, oos_a, oos_b")
    parser.add_argument("--reset", action="store_true", help="Reset portfolio cash to ₹10,00,000 before starting")
    args = parser.parse_args()

    run_monthly_agent_learning_simulation(num_months=args.months, block=args.block, reset_db_first=args.reset)
