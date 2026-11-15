"""tax_engine.py — Indian Capital Gains Tax Logic & Tax Lot Tracking (NSE/BSE).

Implements statutory Indian equity taxation rules under recent Union Budget revisions:
1. Short-Term Capital Gains (STCG):
   - Holding period < 12 months (365 days).
   - Flat 20% tax rate on net realized gains.
2. Long-Term Capital Gains (LTCG):
   - Holding period >= 12 months (365 days).
   - Statutory annual exemption of ₹1.25 Lakhs (₹125,000) per Indian Financial Year (Apr 1 - Mar 31).
   - Flat 12.5% tax rate on net realized gains exceeding the annual exemption limit.
3. Tax Lot Intelligence & Friction Advisory:
   - Evaluates open positions and advises the Portfolio Manager and Risk Committee
     when a winning position approaches the 12-month threshold (compounding runway vs 20% STCG friction).
"""

from datetime import datetime
from typing import Dict, Any, List, Optional, Tuple

# Indian Statutory Tax Rates for Listed Indian Equities (STT Paid)
STCG_TAX_RATE: float = 0.20           # 20% on short-term gains (< 12 months)
LTCG_TAX_RATE: float = 0.125          # 12.5% on long-term gains (>= 12 months)
LTCG_ANNUAL_EXEMPTION: float = 125000.0  # ₹1.25 Lakhs per financial year


def get_financial_year(dt: datetime) -> str:
    """Returns the Indian Financial Year string (e.g., 'FY2024-25') for a given datetime.
    
    Indian Financial Year spans from April 1 to March 31 of the subsequent year.
    """
    year = dt.year
    if dt.month >= 4:
        next_yr_str = str(year + 1)[-2:]
        return f"FY{year}-{next_yr_str}"
    else:
        prev_yr = year - 1
        curr_yr_str = str(year)[-2:]
        return f"FY{prev_yr}-{curr_yr_str}"


def parse_datetime(dt_val: Any) -> datetime:
    """Safely converts ISO string, pd.Timestamp, or datetime into a naive datetime."""
    if isinstance(dt_val, datetime):
        return dt_val.replace(tzinfo=None)
    dt_str = str(dt_val).replace("Z", "").split("+")[0].strip()
    if " " in dt_str:
        return datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d %H:%M:%S")
    elif "T" in dt_str:
        return datetime.strptime(dt_str.split(".")[0], "%Y-%m-%dT%H:%M:%S")
    return datetime.strptime(dt_str.split(".")[0], "%Y-%m-%d")


def compute_capital_gains_tax(
    buy_date: Any,
    sell_date: Any,
    cost_basis: float,
    net_proceeds: float,
    fy_ltcg_exempt_used: float = 0.0,
    fy_stcl_carryforward: float = 0.0,
    current_fy: Optional[str] = None
) -> Dict[str, Any]:
    """Calculates statutory Indian capital gains tax upon closing an equity delivery trade.

    Implements:
    - April 1 Financial Year reset for LTCG exemption and loss carry-forward.
    - STCL (Short-Term Capital Loss) offset against STCG within the same Financial Year.
    - 20% STCG on net short-term gains after loss offset.
    - 12.5% LTCG on net long-term gains exceeding the ₹1.25 Lakhs annual exemption.
    """
    buy_dt = parse_datetime(buy_date)
    sell_dt = parse_datetime(sell_date)
    holding_days = max(0, (sell_dt - buy_dt).days)
    fy_str = get_financial_year(sell_dt)

    # April 1 Financial Year Transition: Reset annual exemption and loss carry-forward if new FY
    if current_fy and current_fy != fy_str:
        fy_ltcg_exempt_used = 0.0
        fy_stcl_carryforward = 0.0

    gross_pnl = round(net_proceeds - cost_basis, 2)
    is_long_term = holding_days >= 365
    term_type = "LTCG" if is_long_term else "STCG"

    if gross_pnl <= 0.0:
        # Trade resulted in a capital loss
        new_fy_stcl = round(fy_stcl_carryforward + abs(gross_pnl), 2) if not is_long_term else fy_stcl_carryforward
        return {
            "holding_days": holding_days,
            "is_long_term": is_long_term,
            "term_type": term_type,
            "financial_year": fy_str,
            "cost_basis": cost_basis,
            "net_proceeds": net_proceeds,
            "gross_pnl": gross_pnl,
            "exemption_applied": 0.0,
            "stcl_offset_applied": 0.0,
            "taxable_pnl": 0.0,
            "tax_rate": 0.0,
            "tax_deducted": 0.0,
            "net_pnl_after_tax": gross_pnl,
            "new_fy_exempt_used": fy_ltcg_exempt_used,
            "new_fy_stcl": new_fy_stcl,
        }

    if is_long_term:
        # Long-Term Capital Gains (LTCG >= 365 days)
        remaining_exemption = max(0.0, LTCG_ANNUAL_EXEMPTION - fy_ltcg_exempt_used)
        exemption_applied = min(gross_pnl, remaining_exemption)
        taxable_pnl = max(0.0, round(gross_pnl - exemption_applied, 2))
        tax_rate = LTCG_TAX_RATE
        tax_deducted = round(taxable_pnl * tax_rate, 2)
        new_fy_exempt_used = round(fy_ltcg_exempt_used + exemption_applied, 2)
        new_fy_stcl = fy_stcl_carryforward
        stcl_offset = 0.0
    else:
        # Short-Term Capital Gains (STCG < 365 days) with intra-year loss offset
        stcl_offset = min(gross_pnl, max(0.0, fy_stcl_carryforward))
        taxable_pnl = max(0.0, round(gross_pnl - stcl_offset, 2))
        exemption_applied = 0.0
        tax_rate = STCG_TAX_RATE
        tax_deducted = round(taxable_pnl * tax_rate, 2)
        new_fy_exempt_used = fy_ltcg_exempt_used
        new_fy_stcl = round(max(0.0, fy_stcl_carryforward - stcl_offset), 2)

    net_pnl_after_tax = round(gross_pnl - tax_deducted, 2)

    return {
        "holding_days": holding_days,
        "is_long_term": is_long_term,
        "term_type": term_type,
        "financial_year": fy_str,
        "cost_basis": cost_basis,
        "net_proceeds": net_proceeds,
        "gross_pnl": gross_pnl,
        "exemption_applied": exemption_applied,
        "stcl_offset_applied": stcl_offset,
        "taxable_pnl": taxable_pnl,
        "tax_rate": tax_rate,
        "tax_deducted": tax_deducted,
        "net_pnl_after_tax": net_pnl_after_tax,
        "new_fy_exempt_used": new_fy_exempt_used,
        "new_fy_stcl": new_fy_stcl,
    }


def get_tax_lot_analysis(
    position: Dict[str, Any],
    current_price: float,
    as_of_date: Any,
    fy_ltcg_exempt_used: float = 0.0
) -> Dict[str, Any]:
    """Evaluates prospective tax friction and holding threshold for an active open position.
    
    Enables Portfolio Manager to evaluate whether closing an existing winner incurs
    a 20% STCG friction haircut or if holding for LTCG compounding is preferable.
    """
    ticker = position["ticker"]
    qty = position["qty"]
    avg_price = position["avg_price"]
    opened_at = position.get("opened_at") or datetime.now().isoformat()
    
    buy_dt = parse_datetime(opened_at)
    curr_dt = parse_datetime(as_of_date)
    holding_days = max(0, (curr_dt - buy_dt).days)
    days_to_ltcg = max(0, 365 - holding_days)
    is_long_term = holding_days >= 365
    
    cost_basis = round(qty * avg_price, 2)
    curr_val = round(qty * current_price, 2)
    unrealized_pnl = round(curr_val - cost_basis, 2)
    unrealized_pnl_pct = round((unrealized_pnl / cost_basis) * 100.0, 2) if cost_basis > 0 else 0.0
    
    # Prospective tax calculation if liquidated today
    tax_info = compute_capital_gains_tax(
        buy_date=buy_dt,
        sell_date=curr_dt,
        cost_basis=cost_basis,
        net_proceeds=curr_val,
        fy_ltcg_exempt_used=fy_ltcg_exempt_used
    )
    est_tax = tax_info["tax_deducted"]
    est_net_pnl = tax_info["net_pnl_after_tax"]
    
    # Strategic tax advisory signal
    if unrealized_pnl > 0 and 0 < days_to_ltcg <= 45:
        tax_status = (
            f"APPROACHING_LTCG ({holding_days}d held, {days_to_ltcg}d to 12M cutoff). "
            f"Selling now incurs ₹{est_tax:,.2f} STCG tax (20%). "
            f"Hold recommendation to unlock 12.5% LTCG rate and ₹1.25L exemption."
        )
        friction_alert = True
    elif is_long_term and unrealized_pnl > 0:
        tax_status = (
            f"QUALIFIED_LTCG ({holding_days}d held). "
            f"Eligible for 12.5% rate and ₹1.25L annual statutory exemption."
        )
        friction_alert = False
    elif unrealized_pnl > 0:
        tax_status = f"STCG ({holding_days}d held). Tax liability if sold today: ₹{est_tax:,.2f} (20%)."
        friction_alert = False
    else:
        tax_status = f"UNREALIZED LOSS ({unrealized_pnl_pct}%). No tax liability on exit."
        friction_alert = False
        
    return {
        "ticker": ticker,
        "holding_days": holding_days,
        "days_to_ltcg": days_to_ltcg,
        "is_long_term": is_long_term,
        "cost_basis": cost_basis,
        "current_value": curr_val,
        "unrealized_pnl": unrealized_pnl,
        "unrealized_pnl_pct": unrealized_pnl_pct,
        "est_tax_if_sold_today": est_tax,
        "est_net_pnl_after_tax": est_net_pnl,
        "tax_status": tax_status,
        "friction_alert": friction_alert
    }


def format_portfolio_tax_summary(
    positions: List[Dict[str, Any]],
    current_prices: Dict[str, float],
    as_of_date: Any,
    fy_ltcg_exempt_used: float = 0.0
) -> str:
    """Formats a concise tax intelligence summary string for LLM Agent prompt injection."""
    if not positions:
        return "No open positions. Portfolio in 100% Cash / Liquid Mutual Fund sweep."
        
    curr_dt = parse_datetime(as_of_date)
    fy_str = get_financial_year(curr_dt)
    rem_exemption = max(0.0, LTCG_ANNUAL_EXEMPTION - fy_ltcg_exempt_used)
    
    lines = [
        f"Indian Equity Tax Context ({fy_str}):",
        f"- FY LTCG Exemption Remaining: ₹{rem_exemption:,.2f} of ₹{LTCG_ANNUAL_EXEMPTION:,.2f}",
        f"- Statutory Rates: STCG = 20.0% (< 12 months), LTCG = 12.5% (>= 12 months)",
        "Active Tax Lots:"
    ]
    
    for p in positions:
        t = p["ticker"]
        price = current_prices.get(t, p["avg_price"])
        lot = get_tax_lot_analysis(p, price, curr_dt, fy_ltcg_exempt_used)
        alert_flag = " [!] HIGH TAX FRICTION" if lot["friction_alert"] else ""
        lines.append(
            f"  * {t}: {lot['holding_days']}d held | Unr PnL: ₹{lot['unrealized_pnl']:,.2f} ({lot['unrealized_pnl_pct']}%) | "
            f"Est Tax: ₹{lot['est_tax_if_sold_today']:,.2f} | Status: {lot['tax_status']}{alert_flag}"
        )
        
    return "\n".join(lines)

