"""analytics.py — Daily NAV Tracking, Alpha/Beta vs Nifty 50, Sharpe Ratio & Scorecard Report.

Calculates rupee NAV history, annualized Sharpe, Max Drawdown, and Alpha against
Nifty 50 (^NSEI) and Nifty 500 (^CRSLDX).
"""

import logging
from datetime import datetime
from typing import Dict, Any, List
import numpy as np
import pandas as pd
import yfinance as yf

from config import (
    BENCHMARK_TICKER,
    BROAD_BENCHMARK_TICKER,
    VIX_TICKER,
    INITIAL_CAPITAL_INR,
)
from database import (
    get_portfolio_account,
    get_all_positions,
    get_nav_history,
    log_daily_nav,
    get_agent_scorecard,
    get_trade_logs,
    get_top_actionable_rules,
    get_ist_now,
)

logger = logging.getLogger("wealth_agent_office.analytics")


def record_daily_nav() -> Dict[str, Any]:
    """Records end-of-day portfolio NAV, active positions, and benchmark closes."""
    acc = get_portfolio_account()
    cash = acc["cash"]
    liquid_bal = float(acc.get("liquid_fund_balance", 0.0) or 0.0)
    positions = get_all_positions()

    invested_value = 0.0
    for p in positions:
        ticker = p["ticker"]
        try:
            tk = yf.Ticker(ticker)
            last_price = tk.fast_info.last_price or p["avg_price"]
        except Exception:
            last_price = p["avg_price"]
        invested_value += p["qty"] * float(last_price)

    total_nav = cash + liquid_bal + invested_value
    trade_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch benchmarks
    try:
        data = yf.download(
            [BENCHMARK_TICKER, BROAD_BENCHMARK_TICKER, VIX_TICKER],
            period="5d",
            interval="1d",
            progress=False
        )
        nifty_close = float(data[BENCHMARK_TICKER]["Close"].dropna().iloc[-1])
        nifty500_close = float(data[BROAD_BENCHMARK_TICKER]["Close"].dropna().iloc[-1])
        india_vix = float(data[VIX_TICKER]["Close"].dropna().iloc[-1])
    except Exception as e:
        logger.warning(f"Failed to fetch daily benchmark closes: {e}")
        nifty_close = 24000.0
        nifty500_close = 22000.0
        india_vix = 15.0

    log_daily_nav(
        trade_date=trade_date,
        total_nav=total_nav,
        cash=cash,
        invested_value=invested_value,
        nifty50_close=nifty_close,
        nifty500_close=nifty500_close,
        india_vix=india_vix,
        active_positions_count=len(positions)
    )

    logger.info(
        f"[Daily NAV] Logged {trade_date}: Total NAV ₹{total_nav:,.2f} "
        f"(Cash ₹{cash:,.2f}, Invested ₹{invested_value:,.2f}) | Nifty 50: {nifty_close:,.2f}"
    )

    return {
        "trade_date": trade_date,
        "total_nav": total_nav,
        "cash": cash,
        "invested_value": invested_value,
        "nifty50_close": nifty_close,
        "india_vix": india_vix
    }


def compute_performance_metrics() -> Dict[str, Any]:
    """Calculates cumulative Alpha, Beta, Sharpe Ratio, and Max Drawdown."""
    acc = get_portfolio_account()
    positions = get_all_positions()

    liquid_bal = float(acc.get("liquid_fund_balance", 0.0) or 0.0)
    invested_value = sum(p["qty"] * p["avg_price"] for p in positions)
    current_nav = acc["cash"] + liquid_bal + invested_value
    total_return_pct = round(((current_nav - INITIAL_CAPITAL_INR) / INITIAL_CAPITAL_INR) * 100.0, 2)

    history = get_nav_history(limit=252)
    if len(history) < 2:
        return {
            "current_nav": round(current_nav, 2),
            "cash": round(acc["cash"], 2),
            "invested_value": round(invested_value, 2),
            "total_return_pct": total_return_pct,
            "alpha": 0.0,
            "beta": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "history_days": len(history)
        }

    df = pd.DataFrame(history)
    df["port_ret"] = df["total_nav"].pct_change()
    df["nifty_ret"] = df["nifty50_close"].pct_change()
    clean = df.dropna()

    if len(clean) < 2:
        return {
            "current_nav": round(current_nav, 2),
            "cash": round(acc["cash"], 2),
            "invested_value": round(invested_value, 2),
            "total_return_pct": total_return_pct,
            "alpha": 0.0,
            "beta": 1.0,
            "sharpe_ratio": 0.0,
            "max_drawdown_pct": 0.0,
            "history_days": len(history)
        }

    # Annualized Sharpe (Assumes 6.5% Indian risk-free rate)
    daily_rf = 0.065 / 252
    excess_ret = clean["port_ret"] - daily_rf
    mean_excess = excess_ret.mean()
    std_excess = excess_ret.std()
    sharpe = round((mean_excess / std_excess) * np.sqrt(252), 2) if std_excess > 0 else 0.0

    # Beta vs Nifty 50
    cov = np.cov(clean["port_ret"], clean["nifty_ret"])[0][1]
    var_nifty = np.var(clean["nifty_ret"])
    beta = round(cov / var_nifty, 2) if var_nifty > 0 else 1.0

    # Alpha vs Nifty 50 (Jensen's Alpha annualized)
    port_cum_ret = (df["total_nav"].iloc[-1] / df["total_nav"].iloc[0]) - 1.0
    nifty_cum_ret = (df["nifty50_close"].iloc[-1] / df["nifty50_close"].iloc[0]) - 1.0
    alpha = round((port_cum_ret - (0.065 + beta * (nifty_cum_ret - 0.065))) * 100.0, 2)

    # Max Drawdown
    nav_series = df["total_nav"]
    rolling_max = nav_series.cummax()
    drawdowns = (nav_series - rolling_max) / rolling_max
    max_dd = round(float(drawdowns.min()) * 100.0, 2)

    return {
        "current_nav": round(current_nav, 2),
        "cash": round(acc["cash"], 2),
        "invested_value": round(invested_value, 2),
        "total_return_pct": total_return_pct,
        "alpha": alpha,
        "beta": beta,
        "sharpe_ratio": sharpe,
        "max_drawdown_pct": max_dd,
        "history_days": len(history)
    }


def generate_scorecard_report():
    """Renders interactive executive scorecard report."""
    metrics = compute_performance_metrics()
    positions = get_all_positions()
    trades = get_trade_logs(limit=10)
    scorecard = get_agent_scorecard()
    lessons = get_top_actionable_rules(limit=5)

    try:
        from ui import display_rich_scorecard
        display_rich_scorecard(metrics, positions, trades, scorecard, lessons)
        return
    except Exception as e:
        logger.debug(f"Rich UI fallback: {e}")

    print("=" * 80)
    print("  WEALTH MANAGEMENT INVESTMENT OFFICE (NSE/BSE) — EXECUTIVE SCORECARD")
    print("=" * 80)
    print(f" Portfolio Total NAV:  ₹{metrics['current_nav']:>14,.2f}  |  Initial Capital:  ₹{INITIAL_CAPITAL_INR:>14,.2f}")
    print(f" Available Cash:       ₹{metrics['cash']:>14,.2f}  |  Total Return:     {metrics['total_return_pct']:>13.2f}%")
    print(f" Invested Value:       ₹{metrics['invested_value']:>14,.2f}  |  Max Drawdown:     {metrics['max_drawdown_pct']:>13.2f}%")
    print(f" Alpha vs Nifty 50:    {metrics['alpha']:>14.2f}%  |  Beta vs Nifty 50: {metrics['beta']:>14.2f}")
    print(f" Sharpe Ratio (6.5%):  {metrics['sharpe_ratio']:>14.2f}  |  History Sessions: {metrics['history_days']:>14}")
    print("-" * 80)

    print("\n[ACTIVE PORTFOLIO POSITIONS]")
    if positions:
        header = f"{'Ticker':<12} {'Sector':<20} {'Qty':<6} {'Avg Price':<11} {'Stop Loss':<11} {'Target':<11}"
        print(header)
        print("-" * len(header))
        for p in positions:
            print(f"{p['ticker']:<12} {p['sector']:<20} {p['qty']:<6} ₹{p['avg_price']:<10.2f} ₹{p['stop_loss']:<10.2f} ₹{p['target_price']:<10.2f}")
    else:
        print("No active open positions. (100% Cash)")

    print("\n[ROLLING 30-TRADE AGENT ACCURACY SCORECARD]")
    header_agent = f"{'Agent / Committee Role':<25} {'Accuracy %':<12} {'Accurate':<10} {'Failed':<10}"
    print(header_agent)
    print("-" * len(header_agent))
    for agent, d in scorecard.items():
        print(f"{agent:<25} {d['accuracy_pct']:>8.1f}%   {d['accurate']:<10} {d['failed']:<10}")

    print("\n[RECENT EXECUTED TRADES]")
    if trades:
        header_trades = f"{'Timestamp':<20} {'Ticker':<12} {'Action':<6} {'Qty':<6} {'Price':<10} {'PnL (INR)':<12} {'Reason':<15}"
        print(header_trades)
        print("-" * len(header_trades))
        for t in trades[:8]:
            pnl_str = f"₹{t['realized_pnl']:,.2f}" if t['action'] == 'SELL' else "-"
            reason_str = str(t['exit_reason'] or "")
            ts = t['timestamp'][:19]
            print(f"{ts:<20} {t['ticker']:<12} {t['action']:<6} {t['qty']:<6} ₹{t['execution_price']:<9.2f} {pnl_str:<12} {reason_str:<15}")
    else:
        print("No trade executions recorded yet.")

    print("=" * 80)

