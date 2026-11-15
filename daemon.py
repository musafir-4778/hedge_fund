"""daemon.py — Autonomous IST Market Daemon & CLI Entry Points.

Schedules and executes:
- 08:30 AM IST: Discovery Engine Docket generation (max 5 tickers)
- 09:30 AM IST: LangGraph Multi-Agent Committee evaluation & execution
- Every 5 min (09:15 - 15:30 IST Mon-Fri): Deterministic stop/target checks
- 03:45 PM IST: End-of-day NAV recording & post-mortem audits

CLI:
- python daemon.py --analyze TRENT.NS
- python daemon.py --discover
- python daemon.py --report
- python daemon.py --daemon
"""

import sys
import argparse
import logging
from datetime import datetime
from typing import Optional, List, Dict, Any
import pytz

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from config import TIMEZONE
from database import (
    init_db,
    get_all_positions,
    get_position,
    get_portfolio_account,
    get_all_post_mortems,
    reset_portfolio_account,
)
from data_tools import (
    normalize_ticker,
    calculate_technical_indicators,
    calculate_risk_boundaries,
    fetch_fundamental_metrics,
)
from discovery_engine import get_daily_docket
from graph import run_investment_committee
from paper_broker import (
    execute_buy_decision,
    execute_sell,
    check_intraday_stops_and_targets,
)
from analytics import (
    record_daily_nav,
    generate_scorecard_report,
)
from reflection_engine import simulate_mistake_learning
from rich.table import Table
from ui import (
    print_banner,
    print_market_regime_badge,
    display_analyst_reports,
    display_debate_and_synthesis,
    display_risk_committee,
    display_final_decision,
    display_memory_bank,
    console,
)
from state import TradeDecision

logger = logging.getLogger("wealth_agent_office.daemon")

# Global daily docket cache
_today_docket = []


def job_daily_discovery():
    """Runs at 08:30 AM IST before market open."""
    global _today_docket
    logger.info("=== [08:30 AM IST] Executing Stage -1 Discovery Engine ===")
    try:
        _today_docket = get_daily_docket()
        logger.info(f"Today's docket prepared with {len(_today_docket)} tickers: {[d['ticker'] for d in _today_docket]}")
        from data_tools import get_market_regime
        from notifier import notify_morning_docket
        reg, n_close, vix = get_market_regime()
        notify_morning_docket(_today_docket, reg, n_close)
    except Exception as e:
        logger.error(f"Daily discovery job failed: {e}")


def job_daily_committee():
    """Runs at 09:30 AM IST (15 mins after NSE market open)."""
    global _today_docket
    logger.info("=== [09:30 AM IST] Executing LangGraph Committee Review ===")
    try:
        # 1. Review existing holdings for potential sell/trim signals
        holdings = get_all_positions()
        for pos in holdings:
            t = pos["ticker"]
            logger.info(f"Committee reviewing active holding: {t}")
            result = run_investment_committee(ticker=t, sector=pos.get("sector"))
            decision_data = result.get("trade_decision")
            if decision_data and decision_data.get("action") == "SELL":
                price = result["technicals"].get("latest_close", pos["avg_price"])
                logger.info(f"Committee signaled SELL for active holding {t}.")
                execute_sell(t, price, "PM_SIGNAL", decision_data.get("rationale", ""))

        # 2. Evaluate today's discovered docket
        if not _today_docket:
            _today_docket = get_daily_docket()

        for item in _today_docket:
            t = item["ticker"]
            logger.info(f"Committee evaluating docket pick: {t}")
            result = run_investment_committee(ticker=t, sector=item.get("sector"))
            decision_data = result.get("trade_decision")
            if decision_data:
                decision = TradeDecision(**decision_data)
                if decision.action == "BUY" and decision.conviction in ["HIGH", "MEDIUM"]:
                    execute_buy_decision(decision, result)

    except Exception as e:
        logger.error(f"Daily committee job failed: {e}")


def job_intraday_guard():
    """Runs every 5 minutes during NSE hours (09:15 - 15:30 IST)."""
    try:
        exits = check_intraday_stops_and_targets()
        if exits > 0:
            logger.info(f"[Intraday Guard] Executed {exits} risk exit(s).")
    except Exception as e:
        logger.error(f"Intraday guard polling error: {e}")


def job_trailing_stops_check():
    """Runs at 03:15 PM IST (15 mins before market close) to trail stops on winning positions."""
    logger.info("=== [03:15 PM IST] Running Pre-Close Trailing Stop Elevation Audit ===")
    try:
        from database import get_all_positions, update_position_stop_loss
        from data_tools import calculate_technical_indicators
        positions = get_all_positions()
        for p in positions:
            t = p["ticker"]
            tech = calculate_technical_indicators(t)
            cmp = tech.get("latest_close", p["avg_price"])
            atr = tech.get("atr_14", cmp * 0.02)
            unrealized_gain = cmp - p["avg_price"]
            if unrealized_gain > (1.0 * atr):
                new_stop = round(max(p["stop_loss"], cmp - (2.0 * atr)), 2)
                if new_stop > p["stop_loss"]:
                    update_position_stop_loss(t, new_stop)
                    logger.info(f"[Trailing Stop] Raised stop for {t} from ₹{p['stop_loss']:,.2f} to ₹{new_stop:,.2f} (CMP: ₹{cmp:,.2f})")
    except Exception as e:
        logger.error(f"Trailing stop check failed: {e}")


def job_daily_close():
    """Runs at 03:45 PM IST after market close."""
    logger.info("=== [03:45 PM IST] Logging End-of-Day NAV & Benchmarks ===")
    try:
        record_daily_nav()
        generate_scorecard_report()
        from notifier import notify_evening_summary
        from database import get_portfolio_account, get_all_positions, get_top_actionable_rules
        from config import INITIAL_CAPITAL_INR
        acc = get_portfolio_account()
        positions = get_all_positions()
        rules = get_top_actionable_rules(limit=3)
        total_nav = acc["cash"] + sum(p["qty"] * p["avg_price"] for p in positions)
        ret_pct = ((total_nav - INITIAL_CAPITAL_INR) / INITIAL_CAPITAL_INR) * 100.0
        notify_evening_summary(total_nav, ret_pct, acc["cash"], positions, rules)
    except Exception as e:
        logger.error(f"Daily close job failed: {e}")


def handle_morning_run():
    """Runs 08:30 discovery + 09:30 committee review in sequence (for cloud serverless triggers)."""
    init_db()
    job_daily_discovery()
    job_daily_committee()


def handle_close_run():
    """Runs 03:15 trailing stop audit + 03:45 close settlement in sequence (for cloud serverless triggers)."""
    init_db()
    job_trailing_stops_check()
    job_daily_close()


def run_daemon():
    """Starts APScheduler in Asia/Kolkata timezone."""
    init_db()
    ist = pytz.timezone(TIMEZONE)
    scheduler = BlockingScheduler(timezone=ist)

    # 1. 08:30 AM IST Mon-Fri: Discovery Engine
    scheduler.add_job(
        job_daily_discovery,
        CronTrigger(hour=8, minute=30, day_of_week="mon-fri", timezone=ist),
        name="daily_discovery"
    )

    # 2. 09:30 AM IST Mon-Fri: Committee Run
    scheduler.add_job(
        job_daily_committee,
        CronTrigger(hour=9, minute=30, day_of_week="mon-fri", timezone=ist),
        name="daily_committee"
    )

    # 3. Every 5 mins between 09:15 and 15:30 IST Mon-Fri: Stop/Target Guard
    scheduler.add_job(
        job_intraday_guard,
        CronTrigger(minute="*/5", hour="9-15", day_of_week="mon-fri", timezone=ist),
        name="intraday_guard"
    )

    # 4. 03:15 PM IST Mon-Fri: Pre-Close Trailing Stop Elevation
    scheduler.add_job(
        job_trailing_stops_check,
        CronTrigger(hour=15, minute=15, day_of_week="mon-fri", timezone=ist),
        name="trailing_stops"
    )

    # 5. 03:45 PM IST Mon-Fri: Daily Close NAV & Report
    scheduler.add_job(
        job_daily_close,
        CronTrigger(hour=15, minute=45, day_of_week="mon-fri", timezone=ist),
        name="daily_close"
    )

    logger.info("=" * 70)
    logger.info(f" Wealth Management Autonomous Daemon started [{TIMEZONE}]")
    logger.info(" Schedule:")
    logger.info("  - 08:30 AM IST : Stage -1 Discovery Engine (Nifty 500 + Catalysts)")
    logger.info("  - 09:30 AM IST : LangGraph Committee Review & Execution")
    logger.info("  - 09:15 - 15:30 : 5-Minute Pure Python Intraday Stop/Target Guard")
    logger.info("  - 03:15 PM IST : Pre-Close Trailing Stop Elevation on Winning Holdings")
    logger.info("  - 03:45 PM IST : End-of-Day NAV, Benchmark Logging & Scorecard")
    logger.info("=" * 70)

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        logger.info("Daemon stopped by user.")


def handle_analyze(ticker: str):
    """Runs committee review on a single ticker and displays rich terminal debate."""
    init_db()
    print_banner(ticker)
    console.print(f"\n[bold cyan]🚀 Initiating Autonomous Investment Committee Review for: [yellow]{ticker}[/yellow][/bold cyan]")

    result = run_investment_committee(ticker=ticker)
    technicals = result.get("technicals", {})

    print_market_regime_badge(
        regime=technicals.get("market_regime", "BULL_STAGE2"),
        nifty_close=technicals.get("nifty_close", 24500.0),
        india_vix=technicals.get("india_vix", 14.5)
    )

    display_analyst_reports(
        tech_rep=result.get("technical_report", ""),
        fund_rep=result.get("fundamental_report", ""),
        news_rep=result.get("news_macro_report", "")
    )

    display_debate_and_synthesis(
        bull_args=result.get("bull_arguments", []),
        bear_args=result.get("bear_arguments", []),
        proposal=result.get("research_proposal", "")
    )

    display_risk_committee(
        agg_view=result.get("aggressive_risk_view", ""),
        neu_view=result.get("neutral_risk_view", ""),
        cons_view=result.get("conservative_risk_view", "")
    )

    decision = result.get("trade_decision", {})
    display_final_decision(decision)

    # Prompt or auto-execute if high conviction BUY
    if decision.get("action") == "BUY" and decision.get("conviction") in ["HIGH", "MEDIUM"]:
        console.print(f"\n[bold green]⚡ High/Medium conviction BUY formed. Executing in paper broker...[/bold green]")
        dec_obj = TradeDecision(**decision)
        execute_buy_decision(dec_obj, result)
        console.print("[bold green]✅ Order executed in paper broker successfully.[/bold green]\n")


def handle_discover(limit: int = None):
    """Tests the Stage -1 Discovery Engine."""
    init_db()
    print_banner("STAGE -1 DISCOVERY FUNNEL")
    console.print("\n[bold cyan]Scanning Nifty 500 & Curated Indian Catalysts...[/bold cyan]")
    docket = get_daily_docket(universe_limit=limit)

    if docket:
        table = Table(title=f"[bold green]Daily Docket ({len(docket)} Selected Candidates)[/bold green]", show_header=True, header_style="bold cyan", expand=True)
        table.add_column("Ticker", style="bold white")
        table.add_column("Sector")
        table.add_column("Price (INR)", justify="right")
        table.add_column("RS Alpha (63d)", justify="right")
        table.add_column("Discovery Source", style="dim")

        for d in docket:
            price_str = f"₹{d.get('latest_close', 0):,.2f}"
            alpha = d.get('rs_alpha_63d', 0.0)
            alpha_style = "bold green" if alpha > 0 else "bold red"
            table.add_row(
                d["ticker"],
                d.get("sector", ""),
                price_str,
                f"[{alpha_style}]{alpha:+.2%}[/{alpha_style}]",
                d.get("source", "")
            )
        console.print(table)
    else:
        console.print("[yellow]No candidates discovered.[/yellow]")


def handle_learn():
    """Displays the Institutional Memory Bank and learned guardrail rules."""
    init_db()
    print_banner("INSTITUTIONAL MEMORY")
    post_mortems = get_all_post_mortems()
    display_memory_bank(post_mortems)


def handle_simulate_mistake(ticker: str = "TATAMOTORS.NS"):
    """Demonstrates how the system learns from mistakes by simulating a trade loss,
    attributing root cause, and synthesizing a persistent guardrail rule."""
    init_db()
    print_banner("SIMULATING LEARNING FROM MISTAKE")
    console.print(f"[bold yellow]⚠️  Simulating trade loss & post-mortem learning for {ticker}...[/bold yellow]")

    res = simulate_mistake_learning(
        ticker=ticker,
        sector="Automobile",
        failed_agent="FUNDAMENTAL_ANALYST",
        primary_driver="ACCOUNTING_QUALITY",
        root_cause="Over-reliance on reported accounting PAT while ignoring negative Cash Flow from Operations (CFO) and rising debt-to-equity.",
        actionable_rule="Mandate CFO/PAT ratio > 0.8 over trailing 12 months; automatic downgrade to HOLD if CFO is negative."
    )
    console.print(f"[bold green]✓ Episodic post-mortem learning recorded into Institutional Memory Bank (ID: {res['id']})![/bold green]")
    console.print(f"[bold cyan]Learned Rule:[/bold cyan] [italic]{res['actionable_rule']}[/italic]\n")

    post_mortems = get_all_post_mortems()
    display_memory_bank(post_mortems)


def handle_reset(clean_trades: bool = False):
    """Resets portfolio cash to ₹10,00,000 and clears active positions."""
    init_db()
    reset_portfolio_account(clean_trades=clean_trades)
    print_banner("PORTFOLIO RESET")
    console.print(f"[bold green]✓ Portfolio successfully reset to clean ₹10,00,000.00 cash balance.[/bold green]")
    generate_scorecard_report()


def handle_trade_now(ticker: Optional[str] = None, limit: int = 1):
    """Executes a live paper trading session immediately."""
    init_db()
    print_banner("LIVE PAPER TRADING SESSION")

    candidates = []
    if ticker:
        candidates = [{"ticker": normalize_ticker(ticker), "sector": "Auto"}]
        console.print(f"\n[bold cyan]Targeting specified ticker for paper trading: [yellow]{ticker}[/yellow][/bold cyan]")
    else:
        console.print(f"\n[bold cyan]Running Discovery Engine to find top momentum candidates...[/bold cyan]")
        docket = get_daily_docket(universe_limit=25)
        if docket:
            candidates = docket[:limit]
        else:
            console.print("[yellow]Quant criteria screened 0 stocks today. Falling back to benchmark leader: TRENT.NS[/yellow]")
            candidates = [{"ticker": "TRENT.NS", "sector": "Consumer Discretionary"}]

    executed_count = 0
    for cand in candidates:
        t = cand["ticker"]
        sec = cand.get("sector")
        console.print(f"\n[bold gold1]{'='*80}[/bold gold1]")
        console.print(f"[bold cyan]▶ Convening Multi-Agent Investment Committee for: [yellow]{t}[/yellow][/bold cyan]")
        console.print(f"[bold gold1]{'='*80}[/bold gold1]")

        result = run_investment_committee(ticker=t, sector=sec)
        technicals = result.get("technicals", {})

        print_market_regime_badge(
            regime=technicals.get("market_regime", "BULL_STAGE2"),
            nifty_close=technicals.get("nifty_close", 24500.0),
            india_vix=technicals.get("india_vix", 14.5)
        )

        display_analyst_reports(
            tech_rep=result.get("technical_report", ""),
            fund_rep=result.get("fundamental_report", ""),
            news_rep=result.get("news_macro_report", "")
        )

        display_debate_and_synthesis(
            bull_args=result.get("bull_arguments", []),
            bear_args=result.get("bear_arguments", []),
            proposal=result.get("research_proposal", "")
        )

        display_risk_committee(
            agg_view=result.get("aggressive_risk_view", ""),
            neu_view=result.get("neutral_risk_view", ""),
            cons_view=result.get("conservative_risk_view", "")
        )

        decision = result.get("trade_decision", {})
        display_final_decision(decision)

        if decision.get("action") == "BUY" and decision.get("conviction") in ["HIGH", "MEDIUM"]:
            console.print(f"\n[bold green]⚡ Executing BUY order for {t} in Paper Broker...[/bold green]")
            dec_obj = TradeDecision(**decision)
            res = execute_buy_decision(dec_obj, result)
            if res:
                executed_count += 1
                console.print(f"[bold green]✅ BOUGHT {res['qty']} shares of {t} @ ₹{res['execution_price']:,.2f}![/bold green]")
        else:
            console.print(f"[yellow]Action is {decision.get('action')}. No BUY order executed for {t}.[/yellow]")

    # Run intraday stop guard
    exits = check_intraday_stops_and_targets()
    if exits > 0:
        console.print(f"[bold red]Intraday guard triggered {exits} exit(s).[/bold red]")

    # Display updated portfolio scorecard
    console.print("\n[bold cyan]Rendering Updated Live Portfolio Dashboard...[/bold cyan]")
    generate_scorecard_report()


def handle_direct_buy(ticker: str):
    """Directly executes a paper buy order for a stock using live market data and ATR risk sizing."""
    init_db()
    t = normalize_ticker(ticker)
    print_banner(f"PAPER BROKER BUY: {t}")

    console.print(f"[bold cyan]Hydrating real-time market data & indicators for [yellow]{t}[/yellow]...[/bold cyan]")
    tech = calculate_technical_indicators(t)
    fund = fetch_fundamental_metrics(t)
    sector = fund.get("sector", "Industrials")
    price = tech.get("latest_close", 0.0)
    atr = tech.get("atr_14", price * 0.02)

    acc = get_portfolio_account()
    total_nav = acc["cash"]
    bounds = calculate_risk_boundaries(price, atr, total_nav)
    tech.update(bounds)

    decision = TradeDecision(
        ticker=t,
        action="BUY",
        conviction="HIGH",
        allocation_pct=0.08,
        entry_price=price,
        stop_loss_price=bounds["hard_stop_price"],
        target_price=bounds["min_valid_target"],
        holding_horizon_days=30,
        rationale=f"Direct paper trade execution for {t}. Sector: {sector}."
    )

    state = {
        "sector": sector,
        "technicals": tech,
        "technical_report": f"Latest close ₹{price:,.2f}, 14-day ATR ₹{atr:,.2f}.",
        "fundamental_report": f"Market cap ₹{fund.get('market_cap_cr')} Cr, ROE {fund.get('roe')}.",
        "news_macro_report": "Direct paper execution.",
        "research_proposal": "Paper portfolio capital allocation.",
        "aggressive_risk_view": "Approved.",
        "conservative_risk_view": f"Hard stop at ₹{bounds['hard_stop_price']:,.2f}."
    }

    res = execute_buy_decision(decision, state)
    if res:
        console.print(f"\n[bold green]✅ Successfully executed paper BUY order for {t}![/bold green]")
        console.print(f"   Quantity: [bold]{res['qty']}[/bold] shares @ [bold]₹{res['execution_price']:,.2f}[/bold]")
        console.print(f"   Total Value: ₹{res['gross_value']:,.2f} | Taxes & Fees: ₹{res['buy_fees']:,.2f}")
        console.print(f"   Stop-Loss: [red]₹{res['stop_loss']:,.2f}[/red] (2xATR) | Target: [green]₹{res['target_price']:,.2f}[/green]")
    else:
        console.print(f"[bold red]❌ Failed to execute buy order for {t}. Check cash balance or sector limits.[/bold red]")

    generate_scorecard_report()


def handle_direct_sell(ticker: str):
    """Directly executes a paper sell order, closing an active position and triggering post-mortem audit."""
    init_db()
    t = normalize_ticker(ticker)
    print_banner(f"PAPER BROKER SELL: {t}")

    pos = get_position(t)
    if not pos:
        console.print(f"[bold red]❌ No open position found for {t}.[/bold red]")
        return

    tech = calculate_technical_indicators(t)
    curr_price = tech.get("latest_close", pos["avg_price"])

    console.print(f"[bold cyan]Selling active holding {t} ({pos['qty']} shares) @ current price ₹{curr_price:,.2f}...[/bold cyan]")
    res = execute_sell(t, curr_price, exit_reason="USER_EXIT", rationale="Direct paper sell executed via CLI.")

    if res:
        pnl = res["realized_pnl"]
        pnl_style = "bold green" if pnl >= 0 else "bold red"
        console.print(f"\n[bold green]✅ Position closed successfully for {t}![/bold green]")
        console.print(f"   Realized PnL: [{pnl_style}]₹{pnl:+,.2f}[/{pnl_style}] | Net Cash: ₹{res['new_cash']:,.2f}")
    else:
        console.print(f"[bold red]❌ Failed to close position for {t}.[/bold red]")

    generate_scorecard_report()


def handle_check_stops():
    """Runs 5-minute intraday stop/target monitor on active positions."""
    init_db()
    print_banner("INTRADAY RISK GUARD")
    console.print("[bold cyan]Scanning live market prices for active portfolio holdings...[/bold cyan]")
    exits = check_intraday_stops_and_targets()
    if exits > 0:
        console.print(f"[bold yellow]Executed {exits} risk exit(s).[/bold yellow]")
    else:
        console.print("[bold green]All active positions trading safely within 2xATR risk bands.[/bold green]")
    generate_scorecard_report()


import re
import warnings

class DualOutputLogger:
    """Tees all terminal output (stdout & stderr) to a clean text file in real time, strictly filtering HTTP noise."""
    def __init__(self, f_handle, original_stream):
        self.original_stream = original_stream
        self.f_handle = f_handle
        self.ansi_regex = re.compile(r'\x1b\[[0-9;]*[a-zA-Z]')
        self.noise_patterns = [
            "HTTP Request:",
            "AFC is enabled",
            "generativelanguage.googleapis.com",
            "chat_models.py",
            "[httpx]",
            "[httpcore]",
            "generateContent",
            "HTTP/1.1",
            "google_genai",
        ]

    def write(self, text: str):
        if any(np in text for np in self.noise_patterns):
            return
        self.original_stream.write(text)
        self.original_stream.flush()
        clean_text = self.ansi_regex.sub('', text)
        self.f_handle.write(clean_text)
        self.f_handle.flush()

    def flush(self):
        try:
            self.original_stream.flush()
            self.f_handle.flush()
        except Exception:
            pass

def enable_terminal_logging(filepath: str):
    """Hooks stdout, stderr, all logging handlers, and Python warnings to duplicate everything to a text file."""
    f_handle = open(filepath, "w", encoding="utf-8", buffering=1)
    dual_out = DualOutputLogger(f_handle, sys.stdout)
    dual_err = DualOutputLogger(f_handle, sys.stderr)
    sys.stdout = dual_out
    sys.stderr = dual_err

    # Update rich consoles
    try:
        from ui import console as ui_console
        ui_console.file = dual_out
    except Exception:
        pass
    try:
        import monthly_learning
        monthly_learning.console.file = dual_out
    except Exception:
        pass

    # Suppress verbose HTTP wire traffic & library warnings from polluting debate conversation logs
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("httpcore").setLevel(logging.WARNING)
    logging.getLogger("google_genai").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)
    warnings.filterwarnings("ignore")


def handle_tax_report():
    """Prints comprehensive Indian STCG / LTCG tax ledger report."""
    init_db()
    from database import get_tax_ledger, get_fy_tax_summary, get_portfolio_account
    from rich.table import Table
    from rich.panel import Panel

    summary = get_fy_tax_summary()
    acc = get_portfolio_account()
    ledger = get_tax_ledger(limit=30)

    table = Table(title="Indian Equity Statutory Tax Ledger (STCG @ 20% | LTCG @ 12.5%)", title_style="bold green")
    table.add_column("Timestamp", style="dim")
    table.add_column("Ticker", style="bold yellow")
    table.add_column("FY", justify="center")
    table.add_column("Held (Days)", justify="right")
    table.add_column("Type", justify="center")
    table.add_column("Gross PnL", justify="right")
    table.add_column("Exemption", justify="right")
    table.add_column("Tax Deducted", justify="right", style="bold red")
    table.add_column("Net After-Tax", justify="right", style="bold green")

    for r in ledger:
        pnl_col = "green" if r["gross_pnl"] >= 0 else "red"
        table.add_row(
            r["timestamp"][:19],
            r["ticker"],
            r["financial_year"],
            f"{r['holding_days']}d",
            r["term_type"],
            f"[{pnl_col}]₹{r['gross_pnl']:,.2f}[/{pnl_col}]",
            f"₹{r['exemption_applied']:,.2f}",
            f"₹{r['tax_deducted']:,.2f}",
            f"₹{r['net_pnl_after_tax']:,.2f}"
        )
    console.print(table)

    panel_text = (
        f"• Total Trades Logged:          {summary['trade_count']}\n"
        f"• Gross Realized PnL:           ₹{summary['gross_realized_pnl']:,.2f}\n"
        f"• Short-Term Tax Paid (20%):    ₹{summary['stcg_tax_paid']:,.2f}\n"
        f"• Long-Term Tax Paid (12.5%):   ₹{summary['ltcg_tax_paid']:,.2f}\n"
        f"• Total Tax Paid to Date:       [bold red]₹{summary['total_tax_paid']:,.2f}[/bold red]\n"
        f"• LTCG Exemption Consumed:      ₹{summary['ltcg_exemption_used']:,.2f} of ₹1,25,000\n"
        f"• Net Realized PnL After Tax:   [bold green]₹{summary['net_pnl_after_tax']:,.2f}[/bold green]"
    )
    console.print(Panel(panel_text, title="🏛️ Indian Capital Gains Tax Summary", border_style="cyan"))


def handle_treasury_report():
    """Prints Liquid Mutual Fund parking and risk-free yield report."""
    init_db()
    from database import get_portfolio_account, get_db_connection
    from rich.table import Table
    from rich.panel import Panel

    acc = get_portfolio_account()
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM liquid_fund_ledger ORDER BY id DESC LIMIT 20;").fetchall()

    table = Table(title="Liquid Mutual Fund & Overnight Treasury Ledger (Yield ~6.75% p.a.)", title_style="bold cyan")
    table.add_column("Timestamp", style="dim")
    table.add_column("Action", style="bold")
    table.add_column("Amount (INR)", justify="right")
    table.add_column("Balance After", justify="right", style="cyan")
    table.add_column("Details", style="dim")

    for r in rows:
        act_col = "green" if r["action"] == "INTEREST_ACCRUED" else ("yellow" if r["action"] == "SWEEP" else "magenta")
        table.add_row(
            r["timestamp"][:19],
            f"[{act_col}]{r['action']}[/{act_col}]",
            f"₹{r['amount']:,.2f}",
            f"₹{r['balance_after']:,.2f}",
            r["details"] or ""
        )
    console.print(table)

    panel_text = (
        f"• Unallocated Cash in Bank:         ₹{acc['cash']:,.2f}\n"
        f"• Active Liquid Fund Balance:       [bold cyan]₹{acc.get('liquid_fund_balance', 0.0):,.2f}[/bold cyan] (~6.75% annualized)\n"
        f"• Cumulative Risk-Free Interest:    [bold green]+₹{acc.get('total_interest_earned', 0.0):,.2f}[/bold green]\n"
        f"• Effective Yield Strategy:         Automated sweep on cash > 30% NAV or BEAR_RISK_OFF"
    )
    console.print(Panel(panel_text, title="🏦 Overnight Liquid Sweep & Yield Scorecard", border_style="green"))


def main():
    parser = argparse.ArgumentParser(description="Autonomous Indian Wealth Management Investment Office")
    parser.add_argument("--trade-now", type=str, nargs="?", const="AUTO_DISCOVER", help="Run immediate live paper trading session (optional ticker or auto-discover)")
    parser.add_argument("--buy", type=str, help="Immediately execute paper BUY order for a ticker (e.g., ADANIPORTS.NS)")
    parser.add_argument("--sell", type=str, help="Immediately execute paper SELL order for a ticker (e.g., ADANIPORTS.NS)")
    parser.add_argument("--check-stops", action="store_true", help="Run 5-minute intraday stop-loss / target monitor")
    parser.add_argument("--reset", action="store_true", help="Reset paper portfolio cash to clean ₹10,00,000 balance")
    parser.add_argument("--clean-all", action="store_true", help="When resetting, also purge test trade logs")
    parser.add_argument("--analyze", type=str, help="Run single-ticker committee dry-run (e.g., TRENT.NS)")
    parser.add_argument("--discover", action="store_true", help="Run Stage -1 Discovery Engine")
    parser.add_argument("--limit", type=int, default=None, help="Optional universe limit for discovery testing")
    parser.add_argument("--report", action="store_true", help="Print executive NAV and agent scorecard report")
    parser.add_argument("--tax-report", action="store_true", help="Print Indian Capital Gains Tax (STCG/LTCG) ledger report")
    parser.add_argument("--treasury-report", action="store_true", help="Print Liquid Mutual Fund sweep & risk-free yield report")
    parser.add_argument("--learn", action="store_true", help="Inspect Institutional Memory Bank of learned mistake rules")
    parser.add_argument("--simulate-mistake", type=str, nargs="?", const="TATAMOTORS.NS", help="Simulate a trade loss and synthesize an institutional guardrail rule")
    parser.add_argument("--simulate-1y", action="store_true", help="Run 1-year walk-forward point-in-time simulated learning backtest (strict no lookahead bias)")
    parser.add_argument("--monthly-debate", action="store_true", help="Run month-to-month live multi-agent committee debate & learning across past year")
    parser.add_argument("--curriculum-block", type=str, choices=["block_1", "block_2", "block_3", "block_4", "oos_a", "oos_b"],
                        help="Execute multi-regime training block (e.g., block_1, block_2, block_3, block_4, oos_a, oos_b)")
    parser.add_argument("--months", type=int, default=12, help="Number of months for monthly debate (default: 12)")
    parser.add_argument("--log-file", type=str, default=None, help="Save all terminal output to a txt file (default: monthly_debate_output.txt)")
    parser.add_argument("--daemon", action="store_true", help="Start unattended IST market hours daemon")

    args = parser.parse_args()

    # Enable terminal logging to txt file if requested or running monthly debate
    if args.log_file:
        enable_terminal_logging(args.log_file)
        print(f"[Logging] Saving all terminal output to: {args.log_file}")
    elif args.monthly_debate or args.curriculum_block:
        fname = f"curriculum_{args.curriculum_block}_output.txt" if args.curriculum_block else "monthly_debate_output.txt"
        enable_terminal_logging(fname)
        print(f"[Logging] Saving all terminal output to: {fname}")

    if args.curriculum_block:
        from monthly_learning import run_curriculum_block
        run_curriculum_block(args.curriculum_block, reset_db=args.reset, clean_all=args.clean_all)
    elif args.monthly_debate:
        from monthly_learning import run_monthly_agent_learning_simulation
        run_monthly_agent_learning_simulation(num_months=args.months, reset_db_first=args.reset)
    elif args.reset:
        handle_reset(clean_trades=args.clean_all)
    elif args.buy:
        handle_direct_buy(args.buy)
    elif args.sell:
        handle_direct_sell(args.sell)
    elif args.check_stops:
        handle_check_stops()
    elif args.trade_now:
        chosen_ticker = None if args.trade_now == "AUTO_DISCOVER" else args.trade_now
        handle_trade_now(ticker=chosen_ticker, limit=args.limit or 1)
    elif args.analyze:
        handle_analyze(args.analyze)
    elif args.discover:
        handle_discover(limit=args.limit)
    elif args.report:
        init_db()
        generate_scorecard_report()
    elif args.tax_report:
        handle_tax_report()
    elif args.treasury_report:
        handle_treasury_report()
    elif args.learn:
        handle_learn()
    elif args.simulate_mistake:
        handle_simulate_mistake(args.simulate_mistake)
    elif args.simulate_1y:
        from backtest_simulation import run_1year_walkforward_simulation
        run_1year_walkforward_simulation()
    elif args.daemon:
        run_daemon()
    else:
        parser.print_help()


if __name__ == "__main__":
    main()

