"""ui.py — Rich Terminal Interface, Live Progress Trackers & Executive Dashboards.

Provides interactive terminal visualizations for:
- Live multi-agent committee progress with spinner and stage timers
- Color-coded Bull/Bear debate panels and Three-Angle Risk matrices
- Interactive Executive Scorecard with portfolio NAV, agent hit-rates & mistake memory bank
"""

import time
from typing import Dict, Any, List, Optional
from datetime import datetime

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.text import Text
from rich.layout import Layout
from rich.columns import Columns
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TimeElapsedColumn

console = Console()


def print_banner(ticker: str):
    """Displays a stylish header banner for committee review."""
    title_text = Text()
    title_text.append("WEALTH MANAGEMENT INVESTMENT OFFICE\n", style="bold cyan")
    title_text.append("Autonomous Multi-Agent Equity Research & Paper Trading (NSE/BSE)\n", style="dim")
    title_text.append(f"Committee Session: {ticker}", style="bold yellow")
    panel = Panel(title_text, border_style="cyan", expand=False)
    console.print(panel)


def print_market_regime_badge(regime: str, nifty_close: float, india_vix: float):
    """Renders color-coded market regime status."""
    if regime == "BULL_STAGE2":
        style = "bold green"
        reg_icon = "▲ BULL REGIME (Stage 2 Uptrend)"
    elif regime == "CHOPPY_RANGE":
        style = "bold yellow"
        reg_icon = "◆ CHOPPY RANGE (Sideways / Consolidation)"
    else:
        style = "bold red"
        reg_icon = "▼ BEAR RISK-OFF (Capital Preservation Mandate: 50% Cash)"

    tb = Table.grid(padding=1)
    tb.add_column()
    tb.add_column()
    tb.add_column()
    tb.add_row(
        Text(f"Regime: {reg_icon}", style=style),
        Text(f"Nifty 50: ₹{nifty_close:,.2f}", style="bold white"),
        Text(f"India VIX: {india_vix:.2f}", style="bold magenta")
    )
    console.print(Panel(tb, border_style="dim", title="Macro Environment", title_align="left"))


def display_analyst_reports(tech_rep: str, fund_rep: str, news_rep: str):
    """Displays Stage 1 analyst findings in organized panels."""
    console.print("\n[bold cyan]STAGE 1: SPECIALIZED EQUITY RESEARCH AUDITS[/bold cyan]")
    
    t_panel = Panel(tech_rep.strip(), title="[bold blue]1. Technical Audit (Minervini & Momentum)[/bold blue]", border_style="blue")
    f_panel = Panel(fund_rep.strip(), title="[bold green]2. Fundamental Audit (Quality & Solvency)[/bold green]", border_style="green")
    n_panel = Panel(news_rep.strip(), title="[bold magenta]3. Macro & Catalyst Audit (Headlines & Governance)[/bold magenta]", border_style="magenta")
    
    console.print(t_panel)
    console.print(f_panel)
    console.print(n_panel)


def display_debate_and_synthesis(bull_args: List[str], bear_args: List[str], proposal: str):
    """Displays Stage 2 Dialectical Bull vs Bear debate and Manager synthesis."""
    console.print("\n[bold cyan]STAGE 2: DIALECTICAL BULL vs. BEAR DEBATE[/bold cyan]")

    bull_text = "\n\n".join(bull_args) if bull_args else "No bull thesis formulated."
    bear_text = "\n\n".join(bear_args) if bear_args else "No bear red-team formulated."

    col1 = Panel(bull_text.strip(), title="[bold green]▲ Bull Researcher (Conviction Thesis)[/bold green]", border_style="green")
    col2 = Panel(bear_text.strip(), title="[bold red]▼ Bear Researcher (Chief Red-Teamer & Blind Spots)[/bold red]", border_style="red")
    
    console.print(Columns([col1, col2], equal=True))
    console.print(Panel(proposal.strip(), title="[bold gold1]★ Research Manager Adjudication & Proposal[/bold gold1]", border_style="gold1"))


def display_risk_committee(agg_view: str, neu_view: str, cons_view: str):
    """Displays Stage 3 Three-Angle Risk Committee critique."""
    console.print("\n[bold cyan]STAGE 3: THREE-ANGLE RISK COMMITTEE CRITIQUE[/bold cyan]")

    grid = Table(show_header=True, header_style="bold cyan", border_style="dim", expand=True)
    grid.add_column("Aggressive Risk (Alpha Maximizer)", style="green", ratio=1)
    grid.add_column("Neutral Risk (Concentration & Caps)", style="yellow", ratio=1)
    grid.add_column("Conservative Risk (Capital Protector & CRO)", style="red", ratio=1)

    grid.add_row(agg_view.strip()[:600] + "...", neu_view.strip()[:600] + "...", cons_view.strip()[:600] + "...")
    console.print(grid)


def display_final_decision(decision: Dict[str, Any]):
    """Renders the executive trade decision card."""
    action = decision.get("action", "HOLD")
    if action == "BUY":
        action_style = "bold white on green"
    elif action == "SELL":
        action_style = "bold white on red"
    else:
        action_style = "bold black on yellow"

    table = Table.grid(padding=1)
    table.add_column(style="bold cyan", width=22)
    table.add_column()

    table.add_row("Ticker:", f"[bold white]{decision.get('ticker')}[/bold white]")
    table.add_row("Final Order Action:", f"[{action_style}]  {action}  [/]")
    table.add_row("Conviction:", f"[bold]{decision.get('conviction')}[/bold]")
    table.add_row("Capital Allocation:", f"[bold]{float(decision.get('allocation_pct', 0))*100:.1f}% of NAV[/bold]")
    table.add_row("Entry / Market Price:", f"₹{float(decision.get('entry_price', 0)):,.2f}")
    table.add_row("Hard Stop-Loss:", f"[bold red]₹{float(decision.get('stop_loss_price', 0)):,.2f}[/bold red] (2 x ATR14)")
    table.add_row("Profit Target Price:", f"[bold green]₹{float(decision.get('target_price', 0)):,.2f}[/bold green] (>= 2.5:1 R:R)")
    table.add_row("Review Horizon:", f"{decision.get('holding_horizon_days')} Trading Days")
    table.add_row("Executive Rationale:", f"[italic]{decision.get('rationale')}[/italic]")

    panel = Panel(table, title="[bold gold1]STAGE 4: LEAD PORTFOLIO MANAGER EXECUTIVE ORDER[/bold gold1]", border_style="gold1")
    console.print("\n", panel)


def display_rich_scorecard(metrics: Dict[str, Any], positions: List[Dict[str, Any]], trades: List[Dict[str, Any]], scorecard: Dict[str, Any], lessons: List[str]):
    """Renders full interactive executive dashboard."""
    console.clear()
    console.print("=" * 80, style="cyan")
    console.print("  WEALTH MANAGEMENT INVESTMENT OFFICE — EXECUTIVE SCORECARD DASHBOARD", style="bold cyan")
    console.print("=" * 80, style="cyan")

    # Overview Cards
    grid = Table.grid(expand=True, padding=1)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)
    grid.add_column(ratio=1)

    p1 = Panel(
        f"[bold white]₹{metrics['current_nav']:,.2f}[/bold white]\n"
        f"Available Cash: ₹{metrics['cash']:,.2f}\n"
        f"Invested Value: ₹{metrics['invested_value']:,.2f}",
        title="[bold cyan]Portfolio NAV[/bold cyan]",
        border_style="cyan"
    )

    ret_style = "bold green" if metrics['total_return_pct'] >= 0 else "bold red"
    p2 = Panel(
        f"[{ret_style}]{metrics['total_return_pct']:+.2f}%[/{ret_style}]\n"
        f"Alpha vs Nifty: [bold]{metrics['alpha']:+.2f}%[/bold]\n"
        f"Beta vs Nifty:  [bold]{metrics['beta']:.2f}[/bold]",
        title="[bold yellow]Performance Metrics[/bold yellow]",
        border_style="yellow"
    )

    p3 = Panel(
        f"Sharpe (6.5% Rf): [bold]{metrics['sharpe_ratio']:.2f}[/bold]\n"
        f"Max Drawdown:     [bold red]{metrics['max_drawdown_pct']:.2f}%[/bold red]\n"
        f"Active Positions: [bold]{len(positions)}[/bold]",
        title="[bold magenta]Risk Adjusted[/bold magenta]",
        border_style="magenta"
    )

    grid.add_row(p1, p2, p3)
    console.print(grid)

    # Active Positions Table
    pos_table = Table(title="[bold green]Active Portfolio Holdings[/bold green]", show_header=True, header_style="bold green", expand=True)
    pos_table.add_column("Ticker", style="bold white")
    pos_table.add_column("Sector")
    pos_table.add_column("Qty", justify="right")
    pos_table.add_column("Avg Price", justify="right")
    pos_table.add_column("Stop-Loss (2xATR)", justify="right", style="red")
    pos_table.add_column("Target Price", justify="right", style="green")

    if positions:
        for p in positions:
            pos_table.add_row(
                p["ticker"], p["sector"], str(p["qty"]),
                f"₹{p['avg_price']:,.2f}", f"₹{p['stop_loss']:,.2f}", f"₹{p['target_price']:,.2f}"
            )
    else:
        pos_table.add_row("—", "No active holdings. (100% Cash Buffer)", "—", "—", "—", "—")
    console.print(pos_table)

    # Rolling 30-Trade Accuracy Scorecard
    sc_table = Table(title="[bold blue]Rolling 30-Trade Agent Accuracy Scorecard (Self-Reflection)[/bold blue]", show_header=True, header_style="bold blue", expand=True)
    sc_table.add_column("Agent / Committee Role")
    sc_table.add_column("Historical Accuracy %", justify="center")
    sc_table.add_column("Win Attribution Bar", width=30)
    sc_table.add_column("Record (W / L)", justify="right")

    for agent, d in scorecard.items():
        acc = d["accuracy_pct"]
        bars_filled = int(acc / 10)
        bar_str = "[green]" + "█" * bars_filled + "[/green]" + "[dim]░[/dim]" * (10 - bars_filled)
        sc_table.add_row(
            agent,
            f"[bold]{acc:.1f}%[/bold]",
            bar_str,
            f"{d['accurate']}W / {d['failed']}L"
        )
    console.print(sc_table)

    # Institutional Memory / Hall of Mistakes
    lessons_text = Text()
    if lessons:
        for i, rule in enumerate(lessons, 1):
            lessons_text.append(f" {i}. {rule}\n", style="bold yellow")
    else:
        lessons_text.append(" No post-mortem failure rules logged yet. System learns as closed trades conclude.", style="italic dim")

    console.print(Panel(lessons_text, title="[bold gold1]Institutional Memory Bank — Guardrail Rules Learned from Past Mistakes[/bold gold1]", border_style="gold1"))


def display_memory_bank(post_mortems: List[Dict[str, Any]]):
    """Renders the comprehensive learning log of past mistakes and rules."""
    console.print("\n" + "=" * 80, style="gold1")
    console.print("  INSTITUTIONAL MEMORY BANK: POST-MORTEM ROOT CAUSES & LEARNED RULES", style="bold gold1")
    console.print("=" * 80, style="gold1")

    if not post_mortems:
        console.print("[italic dim]No closed trade post-mortems logged yet. Close a position to trigger audit.[/italic dim]\n")
        return

    table = Table(show_header=True, header_style="bold yellow", expand=True)
    table.add_column("Closed", style="dim", width=12)
    table.add_column("Ticker", style="bold white", width=12)
    table.add_column("Outcome", width=10)
    table.add_column("Net PnL %", justify="right", width=10)
    table.add_column("Primary Driver / Blind Spot", style="red", width=26)
    table.add_column("Actionable Guardrail Rule Learned", style="bold green")

    for pm in post_mortems:
        out = pm["outcome"]
        out_style = "bold green" if out == "WIN" else ("bold red" if out == "LOSS" else "bold yellow")
        pnl = pm["pnl_pct"] * 100.0 if abs(pm["pnl_pct"]) < 1.0 else pm["pnl_pct"]
        pnl_style = "green" if pnl >= 0 else "red"
        closed_date = pm.get("closed_at", "")[:10]

        table.add_row(
            closed_date,
            pm["ticker"],
            f"[{out_style}]{out}[/{out_style}]",
            f"[{pnl_style}]{pnl:+.1f}%[/{pnl_style}]",
            f"{pm.get('primary_driver')}\n(Failed: {pm.get('failed_agent')})",
            pm.get("actionable_rule")
        )

    console.print(table)

