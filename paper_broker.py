"""paper_broker.py — NSE Simulated Execution Engine & 5-Minute Intraday Guard.

Executes paper orders with realistic delivery transaction taxes (STT, SEBI,
stamp duty, and bid-ask slippage), and performs deterministic 5-minute stop/target checks.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, Any, Optional
import yfinance as yf

from config import (
    MAX_ACTIVE_POSITIONS,
    MAX_SECTOR_EXPOSURE_PCT,
    DEFAULT_HOLDING_DAYS,
    STT_PCT,
    EXCHANGE_AND_SEBI_PCT,
    STAMP_DUTY_BUY_PCT,
    ESTIMATED_SLIPPAGE_PCT,
)
from database import (
    get_portfolio_account,
    update_cash,
    get_all_positions,
    get_position,
    add_position,
    remove_position,
    log_trade,
    save_thesis_snapshot,
    get_open_thesis_snapshot,
    close_thesis_snapshot,
    get_ist_now,
    sweep_cash_to_liquid_fund,
    un_sweep_liquid_fund,
    accrue_liquid_fund_interest,
    log_tax_deduction,
)
from tax_engine import compute_capital_gains_tax
from state import TradeDecision

logger = logging.getLogger("wealth_agent_office.broker")


def compute_buy_charges(trade_value: float) -> float:
    """Calculates STT, NSE turnover, SEBI, stamp duty, and slippage on buy."""
    pct = STT_PCT + EXCHANGE_AND_SEBI_PCT + STAMP_DUTY_BUY_PCT + ESTIMATED_SLIPPAGE_PCT
    return round(trade_value * pct, 2)


def compute_sell_charges(trade_value: float) -> float:
    """Calculates STT, NSE turnover, SEBI, and slippage on sell (no stamp duty)."""
    pct = STT_PCT + EXCHANGE_AND_SEBI_PCT + ESTIMATED_SLIPPAGE_PCT
    return round(trade_value * pct, 2)


def execute_buy_decision(
    decision: TradeDecision,
    state: Dict[str, Any]
) -> Optional[Dict[str, Any]]:
    """Validates and executes a BUY decision."""
    ticker = decision.ticker
    sector = state.get("sector", "Industrials")
    current_price = decision.entry_price

    if current_price <= 0:
        logger.warning(f"[Broker] Invalid entry price ₹{current_price} for {ticker}. Aborting.")
        return None

    # Check if already held
    if get_position(ticker):
        logger.info(f"[Broker] {ticker} is already in portfolio. Skipping duplicate buy.")
        return None

    positions = get_all_positions()
    if len(positions) >= MAX_ACTIVE_POSITIONS:
        logger.warning(f"[Broker] Max active positions ({MAX_ACTIVE_POSITIONS}) reached. Vetoing buy for {ticker}.")
        return None

    acc = get_portfolio_account()
    cash = acc["cash"]
    liquid_bal = float(acc.get("liquid_fund_balance", 0.0) or 0.0)
    invested_val = sum(p["qty"] * p["avg_price"] for p in positions)
    total_nav = cash + liquid_bal + invested_val

    # Check Sector Concentration Cap (30%)
    sector_invested = sum(p["qty"] * p["avg_price"] for p in positions if p.get("sector") == sector)
    if total_nav > 0 and (sector_invested / total_nav) >= MAX_SECTOR_EXPOSURE_PCT:
        logger.warning(f"[Broker] Sector {sector} already exceeds 30% cap. Vetoing buy for {ticker}.")
        return None

    # Sizing calculation
    tech = state.get("technicals", {})
    recommended_max_qty = tech.get("recommended_max_qty", int((total_nav * 0.12) // current_price))
    target_alloc_qty = int((total_nav * decision.allocation_pct) // current_price)
    qty = min(target_alloc_qty, recommended_max_qty)

    if qty <= 0:
        logger.warning(f"[Broker] Computed share quantity is 0 for {ticker}. Aborting buy.")
        return None

    gross_value = qty * current_price
    buy_fees = compute_buy_charges(gross_value)
    total_cost = gross_value + buy_fees

    liquid_bal = float(acc.get("liquid_fund_balance", 0.0) or 0.0)
    if total_cost > cash:
        # If cash is parked in Liquid Mutual Fund, automatically un-sweep to fund equity delivery
        if (cash + liquid_bal) >= total_cost:
            needed = total_cost - cash
            redeemed = un_sweep_liquid_fund(needed)
            acc = get_portfolio_account()
            cash = acc["cash"]
            logger.info(f"[Broker Treasury] Auto-redeemed ₹{redeemed:,.2f} from Liquid Mutual Fund to settle {ticker} BUY.")
        else:
            # Scale down quantity to fit available cash + liquid fund balance
            available_total = cash + liquid_bal
            qty = int((available_total * 0.98) // current_price)
            if qty <= 0:
                logger.warning(f"[Broker] Insufficient liquidity (Cash ₹{cash:,.2f} + Liquid ₹{liquid_bal:,.2f}) for {ticker}. Aborting.")
                return None
            gross_value = qty * current_price
            buy_fees = compute_buy_charges(gross_value)
            total_cost = gross_value + buy_fees
            if total_cost > cash:
                un_sweep_liquid_fund(total_cost - cash)
                acc = get_portfolio_account()
                cash = acc["cash"]

    # Deduct cash
    new_cash = cash - total_cost
    update_cash(new_cash)

    # Save thesis snapshot
    snap_id = save_thesis_snapshot(
        ticker=ticker,
        sector=sector,
        entry_price=current_price,
        stop_loss=decision.stop_loss_price,
        target_price=decision.target_price,
        indicators=tech,
        technical_report=state.get("technical_report", ""),
        fundamental_report=state.get("fundamental_report", ""),
        news_macro_report=state.get("news_macro_report", ""),
        debate_summary=state.get("research_proposal", ""),
        risk_summary=f"Agg: {state.get('aggressive_risk_view')} | Cons: {state.get('conservative_risk_view')}",
        pm_rationale=decision.rationale
    )

    now_dt = datetime.now()
    hold_until = (now_dt + timedelta(days=decision.holding_horizon_days or DEFAULT_HOLDING_DAYS)).isoformat()

    # Add position
    add_position(
        ticker=ticker,
        sector=sector,
        qty=qty,
        avg_price=current_price,
        stop_loss=decision.stop_loss_price,
        target_price=decision.target_price,
        opened_at=get_ist_now(),
        max_hold_until=hold_until,
        snapshot_id=snap_id
    )

    # Log trade
    log_trade(
        ticker=ticker,
        action="BUY",
        qty=qty,
        execution_price=current_price,
        stt_and_charges=buy_fees,
        realized_pnl=0.0,
        exit_reason=None,
        rationale=decision.rationale
    )

    logger.info(
        f"[Broker] BOUGHT {qty} shares of {ticker} @ ₹{current_price:,.2f}. "
        f"Total Cost: ₹{total_cost:,.2f} (Fees: ₹{buy_fees:,.2f}). New Cash: ₹{new_cash:,.2f}."
    )

    try:
        from notifier import notify_trade_executed
        notify_trade_executed(
            action="BUY",
            ticker=ticker,
            qty=qty,
            price=current_price,
            stop=decision.stop_loss_price,
            target=decision.target_price,
            rationale=decision.rationale
        )
    except Exception as e:
        logger.debug(f"[Broker] Notification failed: {e}")

    return {
        "ticker": ticker,
        "action": "BUY",
        "qty": qty,
        "price": current_price,
        "execution_price": current_price,
        "fees": buy_fees,
        "buy_fees": buy_fees,
        "gross_value": gross_value,
        "total_cost": total_cost,
        "stop_loss": decision.stop_loss_price,
        "target_price": decision.target_price,
        "snapshot_id": snap_id
    }


def execute_sell(
    ticker: str,
    execution_price: float,
    exit_reason: str,
    rationale: str = "",
    sell_date: Optional[str] = None
) -> Optional[Dict[str, Any]]:
    """Executes a SELL for an open position, deducts Indian STCG/LTCG capital gains tax, and credits net cash."""
    pos = get_position(ticker)
    if not pos:
        logger.warning(f"[Broker] Cannot sell {ticker}: position not found.")
        return None

    qty = pos["qty"]
    avg_price = pos["avg_price"]
    gross_proceeds = qty * execution_price
    sell_fees = compute_sell_charges(gross_proceeds)
    net_proceeds = gross_proceeds - sell_fees

    # Calculate net realized PnL before tax
    cost_basis = round(qty * avg_price, 2)
    gross_realized_pnl = round(net_proceeds - cost_basis, 2)

    # -------------------------------------------------------------
    # Indian Statutory Capital Gains Tax Calculation (STCG vs LTCG)
    # -------------------------------------------------------------
    actual_sell_date = sell_date or get_ist_now()
    buy_date = pos.get("opened_at") or actual_sell_date
    acc = get_portfolio_account()
    fy_exempt = float(acc.get("fy_ltcg_exempt_used", 0.0) or 0.0)

    tax_info = compute_capital_gains_tax(
        buy_date=buy_date,
        sell_date=actual_sell_date,
        cost_basis=cost_basis,
        net_proceeds=net_proceeds,
        fy_ltcg_exempt_used=fy_exempt
    )
    tax_deducted = tax_info["tax_deducted"]
    net_pnl_after_tax = tax_info["net_pnl_after_tax"]

    # Log to tax ledger
    log_tax_deduction(tax_info, ticker=ticker, timestamp=actual_sell_date)

    # Cash credited after statutory tax deduction
    net_cash_credited = round(net_proceeds - tax_deducted, 2)
    new_cash = round(acc["cash"] + net_cash_credited, 2)
    update_cash(new_cash)

    # Log trade with tax breakdown
    log_trade(
        ticker=ticker,
        action="SELL",
        qty=qty,
        execution_price=execution_price,
        stt_and_charges=sell_fees,
        realized_pnl=net_pnl_after_tax,
        exit_reason=exit_reason,
        rationale=f"{rationale} [Tax: ₹{tax_deducted:,.2f} {tax_info['term_type']} | Net PnL: ₹{net_pnl_after_tax:,.2f}]"
    )

    # Remove position
    remove_position(ticker)

    logger.info(
        f"[Broker] SOLD {qty} shares of {ticker} @ ₹{execution_price:,.2f} ({exit_reason}). "
        f"Gross PnL: ₹{gross_realized_pnl:,.2f}. Tax Deducted ({tax_info['term_type']}): ₹{tax_deducted:,.2f}. "
        f"Net After-Tax PnL: ₹{net_pnl_after_tax:,.2f}. New Cash: ₹{new_cash:,.2f}."
    )

    # Trigger Post-Mortem Audit & Alert
    try:
        from reflection_engine import run_post_mortem_audit
        from notifier import notify_trade_exit
        pm_res = run_post_mortem_audit(
            position=pos,
            exit_price=execution_price,
            exit_reason=exit_reason,
            realized_pnl=net_pnl_after_tax
        )
        pnl_pct = (net_pnl_after_tax / cost_basis) * 100.0 if cost_basis > 0 else 0.0
        new_rule = pm_res.get("actionable_rule") if pm_res else None
        notify_trade_exit(
            ticker=ticker,
            exit_price=execution_price,
            reason=exit_reason,
            pnl_pct=pnl_pct,
            pnl_inr=net_pnl_after_tax,
            new_rule=new_rule
        )
    except Exception as e:
        logger.error(f"[Broker] Failed to trigger post-mortem or alert for {ticker}: {e}")

    return {
        "ticker": ticker,
        "action": "SELL",
        "qty": qty,
        "price": execution_price,
        "gross_realized_pnl": gross_realized_pnl,
        "tax_deducted": tax_deducted,
        "realized_pnl": net_pnl_after_tax,
        "term_type": tax_info["term_type"],
        "holding_days": tax_info["holding_days"],
        "exit_reason": exit_reason
    }


def sweep_idle_cash_if_eligible(
    current_regime: str,
    total_nav: float,
    threshold_pct: float = 0.30,
    operational_buffer_pct: float = 0.10
) -> float:
    """Sweeps unallocated cash into Liquid Mutual Fund / T-Bills when cash > 30% NAV or in risk-off regimes."""
    acc = get_portfolio_account()
    cash = float(acc.get("cash", 0.0))
    if total_nav <= 0:
        return 0.0
    cash_ratio = cash / total_nav
    # Check eligibility: Cash > 30% NAV OR regime is BEAR_RISK_OFF or CHOPPY_RANGE
    if cash_ratio > threshold_pct or current_regime in ["BEAR_RISK_OFF", "CHOPPY_RANGE"]:
        # Keep an operational buffer of 10% NAV in liquid cash, sweep the rest
        target_cash_buffer = total_nav * operational_buffer_pct
        sweepable = max(0.0, cash - target_cash_buffer)
        if sweepable > 1000.0:  # Minimum sweep ₹1,000
            swept = sweep_cash_to_liquid_fund(sweepable)
            logger.info(f"[Broker Treasury] Swept ₹{swept:,.2f} idle cash into 6.75% Liquid Mutual Fund.")
            return swept
    return 0.0


def accrue_daily_risk_free_yield(days: int = 1, annual_yield: float = 0.0675) -> float:
    """Accrues daily interest (6.75% p.a. pro-rated) on swept liquid fund balances."""
    acc = get_portfolio_account()
    swept_bal = float(acc.get("liquid_fund_balance", 0.0) or 0.0)
    if swept_bal <= 0:
        return 0.0
    daily_rate = annual_yield / 365.0
    interest = round(swept_bal * daily_rate * days, 2)
    if interest > 0:
        accrue_liquid_fund_interest(interest)
        logger.debug(f"[Broker Treasury] Accrued ₹{interest:,.2f} risk-free interest on ₹{swept_bal:,.2f} liquid fund balance ({days}d).")
    return interest


def check_intraday_stops_and_targets() -> int:
    """5-minute polling guard: checks open positions against stops, targets, and max hold dates."""
    positions = get_all_positions()
    if not positions:
        return 0

    tickers = [p["ticker"] for p in positions]
    logger.debug(f"[Intraday Guard] Polling {len(tickers)} open positions for stop/target triggers...")

    # Fetch latest quotes in pure Python (zero LLM calls)
    try:
        quotes = {}
        data = yf.download(tickers, period="1d", interval="5m", progress=False, threads=True)
        for t in tickers:
            try:
                if len(tickers) == 1:
                    close_series = data["Close"].dropna()
                else:
                    close_series = data[t]["Close"].dropna()
                if not close_series.empty:
                    quotes[t] = float(close_series.iloc[-1])
            except Exception:
                pass
    except Exception as e:
        logger.warning(f"[Intraday Guard] Failed to batch download quotes: {e}")
        quotes = {}

    exits_executed = 0
    now_str = get_ist_now()

    for pos in positions:
        ticker = pos["ticker"]
        current_price = quotes.get(ticker)
        if not current_price or current_price <= 0:
            continue

        stop_loss = pos["stop_loss"]
        target = pos["target_price"]
        max_hold = pos["max_hold_until"]

        if current_price <= stop_loss:
            logger.warning(f"[Intraday Guard] STOP-LOSS HIT for {ticker}: Price ₹{current_price} <= Stop ₹{stop_loss}!")
            execute_sell(ticker, current_price, "STOP_LOSS_HIT", f"Automated 2xATR Stop Hit at ₹{current_price}")
            exits_executed += 1
        elif current_price >= target:
            logger.info(f"[Intraday Guard] TARGET REACHED for {ticker}: Price ₹{current_price} >= Target ₹{target}!")
            execute_sell(ticker, current_price, "TARGET_HIT", f"Automated Target Reached at ₹{current_price}")
            exits_executed += 1
        elif now_str >= max_hold:
            logger.info(f"[Intraday Guard] TIME STOP EXPIRED for {ticker}: Max hold date reached.")
            execute_sell(ticker, current_price, "TIME_STOP_EXPIRED", "30-Day Holding Horizon Expired")
            exits_executed += 1

    return exits_executed

