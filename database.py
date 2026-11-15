"""database.py — SQLite Schema Initialization, WAL Mode & Thread-Safe CRUD Helpers.

Handles SQLite connection pooling in WAL mode for paper_portfolio_india.db,
ensuring the 5-minute stop-loss daemon and LangGraph committee never deadlock.
"""

import sqlite3
import json
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, List, Dict, Any
from config import DB_PATH, INITIAL_CAPITAL_INR, CURRENCY, TIMEZONE

logger = logging.getLogger("wealth_agent_office.database")

def get_ist_now() -> str:
    """Returns current ISO timestamp in IST timezone."""
    # IST is UTC+5:30
    ist = timezone(timedelta(hours=5, minutes=30))
    return datetime.now(ist).isoformat()


def get_db_connection() -> sqlite3.Connection:
    """Creates a SQLite connection with WAL mode enabled and Row factory."""
    conn = sqlite3.connect(DB_PATH, timeout=30.0)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA busy_timeout=5000;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


def init_db():
    """Initializes all database tables and seeds initial capital if empty."""
    with get_db_connection() as conn:
        cursor = conn.cursor()

        # 1. Portfolio Account
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS portfolio_account (
            id INTEGER PRIMARY KEY CHECK (id = 1),
            cash REAL NOT NULL,
            initial_capital REAL NOT NULL,
            currency TEXT NOT NULL DEFAULT 'INR',
            updated_at TEXT NOT NULL
        );
        """)

        # 2. Open Positions
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS positions (
            ticker TEXT PRIMARY KEY,
            sector TEXT NOT NULL,
            qty INTEGER NOT NULL,
            avg_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_price REAL NOT NULL,
            opened_at TEXT NOT NULL,
            max_hold_until TEXT NOT NULL,
            snapshot_id INTEGER NOT NULL
        );
        """)

        # 3. Trade Execution Log
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            action TEXT NOT NULL,
            qty INTEGER NOT NULL,
            execution_price REAL NOT NULL,
            stt_and_charges REAL NOT NULL,
            realized_pnl REAL DEFAULT 0.0,
            exit_reason TEXT,
            rationale TEXT
        );
        """)

        # 4. Daily NAV History
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS daily_nav_history (
            trade_date TEXT PRIMARY KEY,
            total_nav REAL NOT NULL,
            cash REAL NOT NULL,
            invested_value REAL NOT NULL,
            nifty50_close REAL NOT NULL,
            nifty500_close REAL NOT NULL,
            india_vix REAL NOT NULL,
            active_positions_count INTEGER NOT NULL
        );
        """)

        # 5. Fundamental Cache
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS fundamental_cache (
            ticker TEXT PRIMARY KEY,
            sector TEXT,
            industry TEXT,
            market_cap_cr REAL,
            roe REAL,
            debt_to_equity REAL,
            revenue_growth REAL,
            earnings_growth REAL,
            pe_ratio REAL,
            forward_pe REAL,
            operating_cash_flow REAL,
            net_income REAL,
            promoter_holding_pct REAL,
            updated_at TEXT NOT NULL
        );
        """)

        # 6. Thesis Snapshots (Immutable at entry)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS thesis_snapshots (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            ticker TEXT NOT NULL,
            sector TEXT NOT NULL,
            opened_at TEXT NOT NULL,
            entry_price REAL NOT NULL,
            stop_loss REAL NOT NULL,
            target_price REAL NOT NULL,
            indicators_json TEXT NOT NULL,
            technical_report TEXT NOT NULL,
            fundamental_report TEXT NOT NULL,
            news_macro_report TEXT NOT NULL,
            debate_summary TEXT NOT NULL,
            risk_summary TEXT NOT NULL,
            pm_rationale TEXT NOT NULL,
            status TEXT NOT NULL DEFAULT 'OPEN'
        );
        """)

        # 7. Trade Post-Mortems
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS trade_post_mortems (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_id INTEGER NOT NULL,
            ticker TEXT NOT NULL,
            sector TEXT NOT NULL,
            closed_at TEXT NOT NULL,
            outcome TEXT NOT NULL,
            pnl_pct REAL NOT NULL,
            holding_days INTEGER NOT NULL,
            primary_driver TEXT NOT NULL,
            accurate_agent TEXT NOT NULL,
            failed_agent TEXT NOT NULL,
            root_cause TEXT NOT NULL,
            actionable_rule TEXT NOT NULL
        );
        """)

        # 8. Tax Ledger (Indian STCG / LTCG Execution Records)
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS tax_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            ticker TEXT NOT NULL,
            financial_year TEXT NOT NULL,
            holding_days INTEGER NOT NULL,
            term_type TEXT NOT NULL,
            cost_basis REAL NOT NULL,
            net_proceeds REAL NOT NULL,
            gross_pnl REAL NOT NULL,
            exemption_applied REAL NOT NULL,
            taxable_pnl REAL NOT NULL,
            tax_rate REAL NOT NULL,
            tax_deducted REAL NOT NULL,
            net_pnl_after_tax REAL NOT NULL
        );
        """)

        # 9. Liquid Fund & Overnight Yield Ledger
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS liquid_fund_ledger (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            action TEXT NOT NULL,
            amount REAL NOT NULL,
            balance_after REAL NOT NULL,
            details TEXT
        );
        """)

        # Auto-migration of portfolio_account for liquid fund & tax tracking
        cursor.execute("PRAGMA table_info(portfolio_account);")
        columns = [c[1] for c in cursor.fetchall()]
        if "liquid_fund_balance" not in columns:
            cursor.execute("ALTER TABLE portfolio_account ADD COLUMN liquid_fund_balance REAL DEFAULT 0.0;")
        if "total_interest_earned" not in columns:
            cursor.execute("ALTER TABLE portfolio_account ADD COLUMN total_interest_earned REAL DEFAULT 0.0;")
        if "total_tax_paid" not in columns:
            cursor.execute("ALTER TABLE portfolio_account ADD COLUMN total_tax_paid REAL DEFAULT 0.0;")
        if "fy_ltcg_exempt_used" not in columns:
            cursor.execute("ALTER TABLE portfolio_account ADD COLUMN fy_ltcg_exempt_used REAL DEFAULT 0.0;")

        # Seed initial account if not present
        cursor.execute("SELECT COUNT(*) FROM portfolio_account WHERE id = 1;")
        if cursor.fetchone()[0] == 0:
            now_str = get_ist_now()
            cursor.execute(
                """
                INSERT INTO portfolio_account 
                (id, cash, initial_capital, currency, updated_at, liquid_fund_balance, total_interest_earned, total_tax_paid, fy_ltcg_exempt_used)
                VALUES (1, ?, ?, ?, ?, 0.0, 0.0, 0.0, 0.0);
                """,
                (INITIAL_CAPITAL_INR, INITIAL_CAPITAL_INR, CURRENCY, now_str)
            )
            logger.info(f"Initialized portfolio_account with ₹{INITIAL_CAPITAL_INR:,.2f} INR.")

        conn.commit()


# ---------------------------------------------------------
# Account & Positions Helpers
# ---------------------------------------------------------

def get_portfolio_account() -> Dict[str, Any]:
    """Returns the single row from portfolio_account."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM portfolio_account WHERE id = 1;").fetchone()
        if row:
            d = dict(row)
            d.setdefault("liquid_fund_balance", 0.0)
            d.setdefault("total_interest_earned", 0.0)
            d.setdefault("total_tax_paid", 0.0)
            d.setdefault("fy_ltcg_exempt_used", 0.0)
            return d
        return {
            "id": 1,
            "cash": INITIAL_CAPITAL_INR,
            "initial_capital": INITIAL_CAPITAL_INR,
            "currency": CURRENCY,
            "updated_at": get_ist_now(),
            "liquid_fund_balance": 0.0,
            "total_interest_earned": 0.0,
            "total_tax_paid": 0.0,
            "fy_ltcg_exempt_used": 0.0
        }


def update_cash(new_cash: float):
    """Updates current cash balance in portfolio_account."""
    with get_db_connection() as conn:
        conn.execute(
            "UPDATE portfolio_account SET cash = ?, updated_at = ? WHERE id = 1;",
            (round(new_cash, 2), get_ist_now())
        )
        conn.commit()


def reset_portfolio_account(clean_trades: bool = False, initial_capital: float = INITIAL_CAPITAL_INR):
    """Resets portfolio cash to initial capital, resets liquid fund balances, and clears positions."""
    with get_db_connection() as conn:
        conn.execute(
            """
            UPDATE portfolio_account 
            SET cash = ?, initial_capital = ?, updated_at = ?,
                liquid_fund_balance = 0.0, total_interest_earned = 0.0,
                total_tax_paid = 0.0, fy_ltcg_exempt_used = 0.0
            WHERE id = 1;
            """,
            (round(initial_capital, 2), round(initial_capital, 2), get_ist_now())
        )
        conn.execute("DELETE FROM positions;")
        if clean_trades:
            conn.execute("DELETE FROM trade_log;")
            conn.execute("DELETE FROM daily_nav_history;")
            conn.execute("DELETE FROM tax_ledger;")
            conn.execute("DELETE FROM liquid_fund_ledger;")
        conn.commit()


# ---------------------------------------------------------
# Liquid Fund / Overnight Yield & Sweep Helpers
# ---------------------------------------------------------

def sweep_cash_to_liquid_fund(amount: float, timestamp: Optional[str] = None) -> float:
    """Sweeps unallocated cash into simulated Liquid Mutual Fund / T-Bills earning ~6.75% annualized."""
    if amount <= 0:
        return 0.0
    ts = timestamp or get_ist_now()
    with get_db_connection() as conn:
        acc = conn.execute("SELECT cash, liquid_fund_balance FROM portfolio_account WHERE id = 1;").fetchone()
        current_cash = float(acc["cash"])
        current_liquid = float(acc["liquid_fund_balance"] or 0.0)
        
        sweep_amt = min(amount, current_cash)
        if sweep_amt <= 0:
            return 0.0
            
        new_cash = current_cash - sweep_amt
        new_liquid = current_liquid + sweep_amt
        
        conn.execute(
            "UPDATE portfolio_account SET cash = ?, liquid_fund_balance = ?, updated_at = ? WHERE id = 1;",
            (round(new_cash, 2), round(new_liquid, 2), ts)
        )
        conn.execute(
            """
            INSERT INTO liquid_fund_ledger (timestamp, action, amount, balance_after, details)
            VALUES (?, 'SWEEP', ?, ?, 'Swept idle cash into 6.75% Liquid Mutual Fund');
            """,
            (ts, round(sweep_amt, 2), round(new_liquid, 2))
        )
        conn.commit()
        return round(sweep_amt, 2)


def un_sweep_liquid_fund(amount: float, timestamp: Optional[str] = None) -> float:
    """Instantly redeems capital from Liquid Mutual Fund to settle equity delivery BUY orders."""
    if amount <= 0:
        return 0.0
    ts = timestamp or get_ist_now()
    with get_db_connection() as conn:
        acc = conn.execute("SELECT cash, liquid_fund_balance FROM portfolio_account WHERE id = 1;").fetchone()
        current_cash = float(acc["cash"])
        current_liquid = float(acc["liquid_fund_balance"] or 0.0)
        
        redeem_amt = min(amount, current_liquid)
        if redeem_amt <= 0:
            return 0.0
            
        new_cash = current_cash + redeem_amt
        new_liquid = current_liquid - redeem_amt
        
        conn.execute(
            "UPDATE portfolio_account SET cash = ?, liquid_fund_balance = ?, updated_at = ? WHERE id = 1;",
            (round(new_cash, 2), round(new_liquid, 2), ts)
        )
        conn.execute(
            """
            INSERT INTO liquid_fund_ledger (timestamp, action, amount, balance_after, details)
            VALUES (?, 'UNSWEEP', ?, ?, 'Redeemed from Liquid Fund for equity delivery deployment');
            """,
            (ts, round(redeem_amt, 2), round(new_liquid, 2))
        )
        conn.commit()
        return round(redeem_amt, 2)


def accrue_liquid_fund_interest(interest_amount: float, timestamp: Optional[str] = None) -> float:
    """Credits daily accrued risk-free yield from liquid fund into cash balance."""
    if interest_amount <= 0:
        return 0.0
    ts = timestamp or get_ist_now()
    with get_db_connection() as conn:
        acc = conn.execute(
            "SELECT cash, liquid_fund_balance, total_interest_earned FROM portfolio_account WHERE id = 1;"
        ).fetchone()
        current_cash = float(acc["cash"])
        current_liquid = float(acc["liquid_fund_balance"] or 0.0)
        tot_interest = float(acc["total_interest_earned"] or 0.0)
        
        new_liquid = current_liquid + interest_amount
        new_tot_interest = tot_interest + interest_amount
        
        conn.execute(
            """
            UPDATE portfolio_account 
            SET liquid_fund_balance = ?, total_interest_earned = ?, updated_at = ? 
            WHERE id = 1;
            """,
            (round(new_liquid, 2), round(new_tot_interest, 2), ts)
        )
        conn.execute(
            """
            INSERT INTO liquid_fund_ledger (timestamp, action, amount, balance_after, details)
            VALUES (?, 'INTEREST_ACCRUED', ?, ?, 'Daily yield accrual (6.75% p.a. pro-rated)');
            """,
            (ts, round(interest_amount, 2), round(new_liquid, 2))
        )
        conn.commit()
        return round(interest_amount, 2)


# ---------------------------------------------------------
# Tax Ledger CRUD Helpers
# ---------------------------------------------------------

def log_tax_deduction(tax_data: Dict[str, Any], ticker: str, timestamp: Optional[str] = None) -> int:
    """Logs an executed tax event to the tax_ledger and updates portfolio totals."""
    ts = timestamp or get_ist_now()
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO tax_ledger 
            (timestamp, ticker, financial_year, holding_days, term_type, cost_basis,
             net_proceeds, gross_pnl, exemption_applied, taxable_pnl, tax_rate, tax_deducted, net_pnl_after_tax)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ts,
                ticker,
                tax_data.get("financial_year", "FY2024-25"),
                tax_data.get("holding_days", 0),
                tax_data.get("term_type", "STCG"),
                tax_data.get("cost_basis", 0.0),
                tax_data.get("net_proceeds", 0.0),
                tax_data.get("gross_pnl", 0.0),
                tax_data.get("exemption_applied", 0.0),
                tax_data.get("taxable_pnl", 0.0),
                tax_data.get("tax_rate", 0.0),
                tax_data.get("tax_deducted", 0.0),
                tax_data.get("net_pnl_after_tax", 0.0),
            )
        )
        tax_id = cursor.lastrowid
        
        # Update cumulative totals in portfolio_account
        tax_deducted = tax_data.get("tax_deducted", 0.0)
        new_exempt = tax_data.get("new_fy_exempt_used", 0.0)
        
        conn.execute(
            """
            UPDATE portfolio_account 
            SET total_tax_paid = total_tax_paid + ?,
                fy_ltcg_exempt_used = ?,
                updated_at = ?
            WHERE id = 1;
            """,
            (round(tax_deducted, 2), round(new_exempt, 2), ts)
        )
        conn.commit()
        return tax_id


def get_tax_ledger(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves recent tax deduction entries."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM tax_ledger ORDER BY id DESC LIMIT ?;", (limit,)).fetchall()
        return [dict(r) for r in rows]


def get_fy_tax_summary(fy_str: Optional[str] = None) -> Dict[str, Any]:
    """Computes total gross gains, tax deducted, and net after-tax return for a financial year."""
    with get_db_connection() as conn:
        if fy_str:
            rows = conn.execute("SELECT * FROM tax_ledger WHERE financial_year = ?;", (fy_str,)).fetchall()
        else:
            rows = conn.execute("SELECT * FROM tax_ledger;").fetchall()
            
    stcg_tax = sum(r["tax_deducted"] for r in rows if r["term_type"] == "STCG")
    ltcg_tax = sum(r["tax_deducted"] for r in rows if r["term_type"] == "LTCG")
    gross_pnl = sum(r["gross_pnl"] for r in rows)
    tot_tax = stcg_tax + ltcg_tax
    net_pnl = gross_pnl - tot_tax
    exempt_applied = sum(r["exemption_applied"] for r in rows)
    
    return {
        "financial_year": fy_str or "ALL_TIME",
        "trade_count": len(rows),
        "gross_realized_pnl": round(gross_pnl, 2),
        "stcg_tax_paid": round(stcg_tax, 2),
        "ltcg_tax_paid": round(ltcg_tax, 2),
        "total_tax_paid": round(tot_tax, 2),
        "ltcg_exemption_used": round(exempt_applied, 2),
        "net_pnl_after_tax": round(net_pnl, 2),
    }



def get_all_positions() -> List[Dict[str, Any]]:
    """Returns all open positions."""
    with get_db_connection() as conn:
        rows = conn.execute("SELECT * FROM positions ORDER BY opened_at ASC;").fetchall()
        return [dict(r) for r in rows]


def get_position(ticker: str) -> Optional[Dict[str, Any]]:
    """Retrieves open position for a given ticker."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM positions WHERE ticker = ?;", (ticker,)).fetchone()
        return dict(row) if row else None


def add_position(
    ticker: str,
    sector: str,
    qty: int,
    avg_price: float,
    stop_loss: float,
    target_price: float,
    opened_at: str,
    max_hold_until: str,
    snapshot_id: int
):
    """Inserts a new open position."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO positions 
            (ticker, sector, qty, avg_price, stop_loss, target_price, opened_at, max_hold_until, snapshot_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (ticker, sector, qty, round(avg_price, 2), round(stop_loss, 2), round(target_price, 2),
             opened_at, max_hold_until, snapshot_id)
        )
        conn.commit()


def remove_position(ticker: str):
    """Deletes position upon exit."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM positions WHERE ticker = ?;", (ticker,))
        conn.commit()


def update_position_stop_loss(ticker: str, new_stop: float):
    """Updates the active stop-loss for a position (trailing stop)."""
    with get_db_connection() as conn:
        conn.execute("UPDATE positions SET stop_loss = ? WHERE ticker = ?;", (round(new_stop, 2), ticker))
        conn.commit()


# ---------------------------------------------------------
# Trade Log Helpers
# ---------------------------------------------------------

def log_trade(
    ticker: str,
    action: str,
    qty: int,
    execution_price: float,
    stt_and_charges: float,
    realized_pnl: float = 0.0,
    exit_reason: Optional[str] = None,
    rationale: Optional[str] = None
) -> int:
    """Logs an executed trade to trade_log."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trade_log 
            (timestamp, ticker, action, qty, execution_price, stt_and_charges, realized_pnl, exit_reason, rationale)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (get_ist_now(), ticker, action, qty, round(execution_price, 2),
             round(stt_and_charges, 2), round(realized_pnl, 2), exit_reason, rationale)
        )
        conn.commit()
        return cursor.lastrowid


def get_trade_logs(limit: int = 50) -> List[Dict[str, Any]]:
    """Returns recent executed trade records."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trade_log ORDER BY id DESC LIMIT ?;", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------
# Daily NAV History
# ---------------------------------------------------------

def log_daily_nav(
    trade_date: str,
    total_nav: float,
    cash: float,
    invested_value: float,
    nifty50_close: float,
    nifty500_close: float,
    india_vix: float,
    active_positions_count: int
):
    """Saves daily portfolio NAV benchmark record."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO daily_nav_history 
            (trade_date, total_nav, cash, invested_value, nifty50_close, nifty500_close, india_vix, active_positions_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (trade_date, round(total_nav, 2), round(cash, 2), round(invested_value, 2),
             round(nifty50_close, 2), round(nifty500_close, 2), round(india_vix, 2), active_positions_count)
        )
        conn.commit()


def get_nav_history(limit: int = 252) -> List[Dict[str, Any]]:
    """Returns chronological daily NAV history."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM daily_nav_history ORDER BY trade_date ASC LIMIT ?;", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


# ---------------------------------------------------------
# Fundamental Cache
# ---------------------------------------------------------

def get_cached_fundamentals(ticker: str, max_age_days: int = 14) -> Optional[Dict[str, Any]]:
    """Returns cached fundamental metrics if fresh within max_age_days."""
    with get_db_connection() as conn:
        row = conn.execute("SELECT * FROM fundamental_cache WHERE ticker = ?;", (ticker,)).fetchone()
        if not row:
            return None
        cached = dict(row)
        try:
            updated_dt = datetime.fromisoformat(cached["updated_at"])
            now_dt = datetime.fromisoformat(get_ist_now())
            if (now_dt - updated_dt).days <= max_age_days:
                return cached
        except Exception:
            pass
        return None


def save_cached_fundamentals(ticker: str, data: Dict[str, Any]):
    """Caches fundamental ratios in fundamental_cache table."""
    with get_db_connection() as conn:
        conn.execute(
            """
            INSERT OR REPLACE INTO fundamental_cache
            (ticker, sector, industry, market_cap_cr, roe, debt_to_equity, revenue_growth,
             earnings_growth, pe_ratio, forward_pe, operating_cash_flow, net_income,
             promoter_holding_pct, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                ticker,
                data.get("sector"),
                data.get("industry"),
                data.get("market_cap_cr"),
                data.get("roe"),
                data.get("debt_to_equity"),
                data.get("revenue_growth"),
                data.get("earnings_growth"),
                data.get("pe_ratio"),
                data.get("forward_pe"),
                data.get("operating_cash_flow"),
                data.get("net_income"),
                data.get("promoter_holding_pct"),
                get_ist_now()
            )
        )
        conn.commit()


# ---------------------------------------------------------
# Thesis Snapshots & Post-Mortems
# ---------------------------------------------------------

def save_thesis_snapshot(
    ticker: str,
    sector: str,
    entry_price: float,
    stop_loss: float,
    target_price: float,
    indicators: Dict[str, Any],
    technical_report: str,
    fundamental_report: str,
    news_macro_report: str,
    debate_summary: str,
    risk_summary: str,
    pm_rationale: str
) -> int:
    """Stores immutable entry snapshot for post-mortem analysis."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO thesis_snapshots 
            (ticker, sector, opened_at, entry_price, stop_loss, target_price, indicators_json,
             technical_report, fundamental_report, news_macro_report, debate_summary,
             risk_summary, pm_rationale, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'OPEN');
            """,
            (
                ticker,
                sector,
                get_ist_now(),
                round(entry_price, 2),
                round(stop_loss, 2),
                round(target_price, 2),
                json.dumps(indicators),
                technical_report,
                fundamental_report,
                news_macro_report,
                debate_summary,
                risk_summary,
                pm_rationale
            )
        )
        conn.commit()
        return cursor.lastrowid


def get_open_thesis_snapshot(ticker: str) -> Optional[Dict[str, Any]]:
    """Returns latest OPEN thesis snapshot for a ticker."""
    with get_db_connection() as conn:
        row = conn.execute(
            "SELECT * FROM thesis_snapshots WHERE ticker = ? AND status = 'OPEN' ORDER BY id DESC LIMIT 1;",
            (ticker,)
        ).fetchone()
        return dict(row) if row else None


def close_thesis_snapshot(snapshot_id: int):
    """Marks thesis snapshot as CLOSED."""
    with get_db_connection() as conn:
        conn.execute("UPDATE thesis_snapshots SET status = 'CLOSED' WHERE id = ?;", (snapshot_id,))
        conn.commit()


def clear_post_mortems_and_snapshots():
    """Clears trade post mortems and thesis snapshots to reload fresh simulated learnings."""
    with get_db_connection() as conn:
        conn.execute("DELETE FROM trade_post_mortems;")
        conn.execute("DELETE FROM thesis_snapshots;")
        conn.commit()


def save_trade_post_mortem(
    snapshot_id: int,
    ticker: str,
    sector: str,
    outcome: str,
    pnl_pct: float,
    holding_days: int,
    primary_driver: str,
    accurate_agent: str,
    failed_agent: str,
    root_cause: str,
    actionable_rule: str
) -> int:
    """Inserts a structured post-mortem reflection."""
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO trade_post_mortems
            (snapshot_id, ticker, sector, closed_at, outcome, pnl_pct, holding_days,
             primary_driver, accurate_agent, failed_agent, root_cause, actionable_rule)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?);
            """,
            (
                snapshot_id,
                ticker,
                sector,
                get_ist_now(),
                outcome,
                round(pnl_pct, 4),
                holding_days,
                primary_driver,
                accurate_agent,
                failed_agent,
                root_cause,
                actionable_rule
            )
        )
        conn.commit()
        return cursor.lastrowid


def _jaccard_similarity(text1: str, text2: str) -> float:
    """Computes word token Jaccard similarity between two rule strings."""
    tokens1 = set(text1.lower().split())
    tokens2 = set(text2.lower().split())
    if not tokens1 or not tokens2:
        return 0.0
    intersection = len(tokens1 & tokens2)
    union = len(tokens1 | tokens2)
    return intersection / union if union > 0 else 0.0


def _deduplicate_rules(rules: List[str], limit: int, threshold: float = 0.65) -> List[str]:
    """Filters duplicate or highly similar rules using Jaccard token similarity."""
    unique_rules: List[str] = []
    for r in rules:
        r_clean = r.strip()
        if not r_clean:
            continue
        is_duplicate = False
        for u in unique_rules:
            if _jaccard_similarity(r_clean, u) >= threshold:
                is_duplicate = True
                break
        if not is_duplicate:
            unique_rules.append(r_clean)
        if len(unique_rules) >= limit:
            break
    return unique_rules


def get_top_actionable_rules(limit: int = 5) -> List[str]:
    """Retrieves top recent diverse actionable guardrail rules for prompt injection."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT actionable_rule FROM trade_post_mortems WHERE actionable_rule != '' ORDER BY id DESC LIMIT ?;",
            (limit * 5,)
        ).fetchall()
        raw_rules = [r["actionable_rule"] for r in rows]
        return _deduplicate_rules(raw_rules, limit=limit, threshold=0.65)


def get_sector_actionable_rules(sector: str, limit: int = 3) -> List[str]:
    """Retrieves rules learned specifically from past trades in this sector with deduplication."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT actionable_rule FROM trade_post_mortems WHERE sector = ? AND actionable_rule != '' ORDER BY id DESC LIMIT ?;",
            (sector, limit * 5)
        ).fetchall()
        raw_rules = [r["actionable_rule"] for r in rows]
        return _deduplicate_rules(raw_rules, limit=limit, threshold=0.65)


def get_all_post_mortems(limit: int = 50) -> List[Dict[str, Any]]:
    """Retrieves chronological post-mortem history for the Institutional Memory Bank."""
    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT * FROM trade_post_mortems ORDER BY id DESC LIMIT ?;",
            (limit,)
        ).fetchall()
        return [dict(r) for r in rows]


def get_agent_scorecard() -> Dict[str, Any]:
    """Computes rolling 30-trade accuracy percentages across agents.

    Win Attribution: accurate_count / (accurate_count + failed_count)
    """
    agents = ["TECHNICAL", "FUNDAMENTAL", "NEWS_MACRO", "BULL_BEAR_DEBATE", "RISK_COMMITTEE"]
    stats = {a: {"accurate": 0, "failed": 0, "accuracy_pct": 50.0} for a in agents}

    with get_db_connection() as conn:
        rows = conn.execute(
            "SELECT accurate_agent, failed_agent FROM trade_post_mortems ORDER BY id DESC LIMIT 30;"
        ).fetchall()

    for r in rows:
        acc = r["accurate_agent"]
        fld = r["failed_agent"]
        if acc in stats:
            stats[acc]["accurate"] += 1
        if fld in stats:
            stats[fld]["failed"] += 1

    for a, data in stats.items():
        total = data["accurate"] + data["failed"]
        if total > 0:
            data["accuracy_pct"] = round((data["accurate"] / total) * 100.0, 1)
        else:
            data["accuracy_pct"] = 50.0  # Prior default when no history

    return stats


def format_agent_scorecard_str() -> str:
    """Formats rolling 30-trade scorecard into concise prompt string."""
    scorecard = get_agent_scorecard()
    lines = []
    for agent, d in scorecard.items():
        lines.append(f"{agent}: {d['accuracy_pct']}% accuracy ({d['accurate']}W / {d['failed']}L)")
    return "; ".join(lines)

