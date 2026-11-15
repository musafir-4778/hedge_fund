# GEMINI.md — Autonomous Multi-Agent Wealth Management Investment Office (NSE/BSE — Free Google API Edition)

## 1. System Mission & Alpha Thesis
Build an autonomous, zero-cost multi-agent equity research, discovery, and paper trading system tailored for **Indian Equities (NSE/BSE)**, running exclusively on the **Free Google AI Studio API (`GOOGLE_API_KEY`)**, modeled after a **Wealth Management Investment Office** and inspired by **TauricResearch/TradingAgents**.

### 1.1 Structural Edge Over Indian Equity Mutual Funds (Flexi-Cap / Mid-Cap / Large & Mid-Cap)
Indian active mutual funds face SEBI categorization constraints (mandatory 65%+ allocation to specific market-cap buckets, 10% single-stock caps), 3%–7% cash drag for daily SIP/redemption flows, and high impact costs in Nifty Midcap 150 / Smallcap 250 stocks. This system exploits four structural edges to beat **Nifty 50 (`^NSEI`)** and **Nifty 500 (`^CRSLDX`)**:
1. **High-Conviction Concentration:** Hard cap of **12–16 active positions** with a maximum single-stock weight of **12%** and maximum sector exposure of **30%**.
2. **Regime-Aware Cash Management:** Dynamically scales cash from **0%** in confirmed Stage-2 bull markets up to **50%** during risk-off / index breakdown regimes (`^NSEI Close < ^NSEI 200 SMA` or `India VIX (^INDIAVIX) > 20`).
3. **Asymmetric Risk Truncation:** Enforces a strict **2.5:1 minimum Risk-to-Reward ratio** at entry and executes deterministic **$2 \times \text{ATR}_{14}$ stop-losses** via a 5-minute Python polling daemon during NSE market hours (`09:15 AM – 03:30 PM IST`).
4. **Episodic Post-Mortem Self-Improvement:** Every trade stores an immutable `ThesisSnapshot` at entry. Upon exit, an asynchronous **Post-Mortem Auditor** diagnoses why the prediction succeeded or failed, updates rolling 30-trade agent accuracy scores, and writes `actionable_rule` guardrails into SQLite for future prompt injection.

---

## 2. Repository Structure & File Contracts
Scaffold the workspace strictly into these 15 files:

```text
wealth_agent_office_india/
├── GEMINI.md                  # Master architectural specification (this file)
├── .env                       # GOOGLE_API_KEY, INITIAL_CAPITAL_INR=1000000
├── requirements.txt           # Exact pinned dependencies
├── data/
│   └── ind_nifty500list.csv   # NSE Nifty 500 constituent list (Symbol, Industry, ISIN)
├── config.py                  # Free-tier Gemini API router, RPM rate limiter, IST schedule, SEBI/STT tax constants
├── state.py                   # LangGraph TypedDict state + strict Pydantic schemas for all structured outputs
├── database.py                # SQLite schema initialization, WAL mode, and thread-safe CRUD helpers
├── discovery_engine.py        # Stage -1: Nifty 500 Deterministic Quant Funnel + Indian Financial RSS Scout
├── data_tools.py              # Deterministic NSE/BSE technical, fundamental & India VIX calculators (zero LLM math)
├── agents.py                  # Node functions & system prompts for Analysts, Debate, Risk, PM, and Auditor
├── graph.py                   # LangGraph StateGraph wiring, conditional debate loops, and SqliteSaver checkpoints
├── paper_broker.py            # NSE simulated execution engine (STT, stamp duty, brokerage, slippage, intraday stops)
├── reflection_engine.py       # Post-mortem audit pipeline, root-cause attribution, and rolling 30-trade scorer
├── analytics.py               # Daily NAV tracking (INR), Alpha, Beta, Sharpe Ratio, Max Drawdown & Scorecard report
└── daemon.py                  # APScheduler IST market-hours orchestrator + 5-minute deterministic stop-loss watcher
```

---

## 3. Free Google AI Studio API Architecture & Quota Guardrails (`config.py`)

### 3.1 `requirements.txt`
```text
langgraph>=0.2.50
langgraph-checkpoint-sqlite>=2.0.0
langchain-core>=0.3.20
langchain-google-genai>=2.0.5
yfinance>=0.2.50
pandas>=2.2.0
numpy>=1.26.0
pydantic>=2.9.0
apscheduler>=3.10.4
feedparser>=6.0.11
pytz>=2024.1
python-dotenv>=1.0.1
tenacity>=9.0.0
```

### 3.2 Free-Tier Gemini Model Allocation & 429 Fallback Strategy
The system uses **only** `langchain_google_genai.ChatGoogleGenerativeAI` authenticated via `GOOGLE_API_KEY` from Google AI Studio (`aistudio.google.com`). No local Ollama/GPU dependencies are required:

| Tier Role | Assigned Model | Free-Tier Quota Strategy |
| :--- | :--- | :--- |
| **`quick_think_llm`** (Catalyst Scout, 3 Analysts, Bull/Bear Debate, 3 Risk Views) | `gemini-2.5-flash` (`temperature=0.1`) | High free-tier RPM/RPD allowance; handles all high-frequency multi-agent turns cleanly. |
| **`deep_think_llm`** (Research Manager, Lead Portfolio Manager, Post-Mortem Auditor) | Primary: `gemini-2.5-pro` (`temperature=0.1`)<br>Auto-Fallback: `gemini-2.5-flash` | Uses `gemini-2.5-pro` for high-stakes synthesis/decisions; automatically catches `ResourceExhausted` (HTTP 429) and retries on `gemini-2.5-flash` so daily runs never stall. |

### 3.3 Mandatory Free-Tier Rate-Limiting & Budget Rules (`config.py`)
To stay comfortably within Google AI Studio free-tier limits (**~10–15 RPM** on Flash, **~2 RPM / 25–50 RPD** on Pro):
1. **Global Token-Bucket / Pacing Wrapper:** Enforce a minimum **`MIN_SECONDS_BETWEEN_LLM_CALLS = 6.5`** (caps throughput at ~9 RPM, preventing burst `429` errors during parallel fan-out).
2. **Exponential Backoff (`tenacity`):** Retry transient `429` or `503` errors with exponential backoff (`wait_exponential(multiplier=2, min=10, max=65)`, `stop_after_attempt(4)`).
3. **Call-Count Budgeting per Ticker (~10 LLM calls/ticker):**
   - `MAX_DEBATE_ROUNDS = 1` (1 Bull turn + 1 Bear turn + 1 Research Manager synthesis).
   - `MAX_DAILY_DOCKET_SIZE = 5` (Top 3 Quant Screener picks + Top 2 Catalyst Scout picks = max ~50 Flash calls + ~10 Pro calls per trading day, well inside free daily quotas).

### 3.4 Indian Market & Risk Constants (`config.py`)
```python
TIMEZONE = "Asia/Kolkata"
INITIAL_CAPITAL_INR = 1_000_000.0    # ₹10,00,000 starting paper capital
CURRENCY = "INR"
BENCHMARK_TICKER = "^NSEI"           # Nifty 50 Index
BROAD_BENCHMARK_TICKER = "^CRSLDX"   # Nifty 500 Index
VIX_TICKER = "^INDIAVIX"             # India VIX volatility gauge

# Free API Rate-Limit & Docket Controls
MIN_SECONDS_BETWEEN_LLM_CALLS = 6.5  # Guarantees <= 9 RPM across all nodes
MAX_DEBATE_ROUNDS = 1                # Keeps per-ticker call count ~10 calls
MAX_DAILY_DOCKET_SIZE = 5            # Max 5 new stocks evaluated per day

# Portfolio Concentration & Risk Guardrails
MAX_ACTIVE_POSITIONS = 15
MAX_SINGLE_STOCK_PCT = 0.12          # Max 12% (₹1,20,000 initially) in any single stock
MAX_SECTOR_EXPOSURE_PCT = 0.30       # Max 30% in any single sector
MAX_PORTFOLIO_RISK_PER_TRADE = 0.015 # Max 1.5% total NAV loss if stop-loss triggers
ATR_STOP_MULTIPLIER = 2.0            # Hard stop = Entry - (2.0 * ATR_14)
MIN_RISK_REWARD_RATIO = 2.5          # (Target - Entry) / (Entry - Stop) >= 2.5
DEFAULT_HOLDING_DAYS = 30            # Time-stop review window

# Realistic Indian Delivery Trade Cost Model
STT_PCT = 0.001                      # 0.10% Securities Transaction Tax on Buy & Sell (Delivery)
EXCHANGE_AND_SEBI_PCT = 0.000035     # ~0.0035% NSE transaction + SEBI turnover charges
STAMP_DUTY_BUY_PCT = 0.00015         # 0.015% Stamp duty on Buy side only
ESTIMATED_SLIPPAGE_PCT = 0.0005      # 0.05% market impact / bid-ask slippage
```

---

## 4. Database Schema (`database.py` — `paper_portfolio_india.db`)
Enable `WAL` mode (`PRAGMA journal_mode=WAL;`) so the 5-minute intraday NSE stop-loss watcher and the LangGraph committee never lock each other out.

```sql
CREATE TABLE IF NOT EXISTS portfolio_account (
    id INTEGER PRIMARY KEY CHECK (id = 1),
    cash REAL NOT NULL,
    initial_capital REAL NOT NULL,
    currency TEXT NOT NULL DEFAULT 'INR',
    updated_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS positions (
    ticker TEXT PRIMARY KEY,             -- e.g., 'TRENT.NS', 'BEL.NS'
    sector TEXT NOT NULL,
    qty INTEGER NOT NULL,
    avg_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target_price REAL NOT NULL,
    opened_at TEXT NOT NULL,
    max_hold_until TEXT NOT NULL,
    snapshot_id INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS trade_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    timestamp TEXT NOT NULL,
    ticker TEXT NOT NULL,
    action TEXT NOT NULL,                -- 'BUY' | 'SELL'
    qty INTEGER NOT NULL,
    execution_price REAL NOT NULL,
    stt_and_charges REAL NOT NULL,
    realized_pnl REAL DEFAULT 0.0,
    exit_reason TEXT,                    -- 'PM_SIGNAL' | 'STOP_LOSS_HIT' | 'TARGET_HIT' | 'TIME_STOP_EXPIRED'
    rationale TEXT
);

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

CREATE TABLE IF NOT EXISTS fundamental_cache (
    ticker TEXT PRIMARY KEY,
    sector TEXT,
    industry TEXT,
    market_cap_cr REAL,                  -- Market cap in ₹ Crores
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

CREATE TABLE IF NOT EXISTS thesis_snapshots (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    ticker TEXT NOT NULL,
    sector TEXT NOT NULL,
    opened_at TEXT NOT NULL,
    entry_price REAL NOT NULL,
    stop_loss REAL NOT NULL,
    target_price REAL NOT NULL,
    indicators_json TEXT NOT NULL,       -- Serialized dict of SMA, RSI, ATR, RS_Alpha, India VIX at entry
    technical_report TEXT NOT NULL,
    fundamental_report TEXT NOT NULL,
    news_macro_report TEXT NOT NULL,
    debate_summary TEXT NOT NULL,
    risk_summary TEXT NOT NULL,
    pm_rationale TEXT NOT NULL,
    status TEXT NOT NULL DEFAULT 'OPEN'  -- 'OPEN' | 'CLOSED'
);

CREATE TABLE IF NOT EXISTS trade_post_mortems (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    snapshot_id INTEGER NOT NULL,
    ticker TEXT NOT NULL,
    sector TEXT NOT NULL,
    closed_at TEXT NOT NULL,
    outcome TEXT NOT NULL,               -- 'WIN' | 'LOSS' | 'BREAKEVEN'
    pnl_pct REAL NOT NULL,
    holding_days INTEGER NOT NULL,
    primary_driver TEXT NOT NULL,
    accurate_agent TEXT NOT NULL,        -- 'TECHNICAL' | 'FUNDAMENTAL' | 'NEWS_MACRO' | 'BULL_BEAR_DEBATE' | 'RISK_COMMITTEE' | 'NONE'
    failed_agent TEXT NOT NULL,
    root_cause TEXT NOT NULL,
    actionable_rule TEXT NOT NULL
);
```

---

## 5. Deterministic Data & Math Engine (`data_tools.py`)
**Strict Rule:** Zero LLM tokens are wasted on arithmetic. All indicators, risk boundaries, and share sizes are computed in Python before state injection:

1. **NSE Ticker Normalization:** Ensure every symbol ends with `.NS` (default) or `.BO`.
2. **Market Regime Detector (Computed on `^NSEI` and `^INDIAVIX`):**
   - `BULL_STAGE2`: `Nifty_Close > Nifty_50_SMA > Nifty_200_SMA` AND `India_VIX < 18.0`.
   - `CHOPPY_RANGE`: `Nifty_Close > Nifty_200_SMA` but `Nifty_Close < Nifty_50_SMA` OR `18.0 <= India_VIX <= 22.0`.
   - `BEAR_RISK_OFF`: `Nifty_Close < Nifty_200_SMA` OR `India_VIX > 22.0` (Triggers 50% cash preservation rule).
3. **Stock Technical Indicators (1-Year Daily OHLCV):**
   - `SMA_20`, `SMA_50`, `SMA_150`, `SMA_200`
   - `RSI_14` (Wilder's smoothing)
   - `ATR_14` (14-day Average True Range in INR)
   - `Volume_Ratio_20d`: `Latest Volume / 20-Day Avg Volume`
   - `Distance_From_52W_High_Pct`: `((Close - 52W_High) / 52W_High) * 100`
   - `RS_Alpha_63d`: `((Stock_Close_t / Stock_Close_t-63) - 1) - ((Nifty_Close_t / Nifty_Close_t-63) - 1)`
   - `Circuit_Limit_Warning`: Flag `True` if stock hit upper/lower circuit (`abs(daily_return) >= 4.9%` on low-volume lock) in the last 5 sessions.
4. **Pre-Computed Risk Boundaries (Passed to Risk Committee & PM):**
   - `hard_stop_price = round(latest_close - (2.0 * atr_14), 2)`
   - `min_valid_target = round(latest_close + (2.5 * (latest_close - hard_stop_price)), 2)`
   - `risk_based_max_shares = int((total_nav * MAX_PORTFOLIO_RISK_PER_TRADE) // (latest_close - hard_stop_price))`
   - `cap_based_max_shares = int((total_nav * MAX_SINGLE_STOCK_PCT) // latest_close)`
   - `recommended_max_qty = min(risk_based_max_shares, cap_based_max_shares)`

---

## 6. Stage -1: Autonomous Indian Business Discovery (`discovery_engine.py`)
Runs once daily before NSE pre-open (`08:30 AM IST`) to build a compact, high-signal **Committee Docket** (`MAX_DAILY_DOCKET_SIZE = 5` tickers to preserve free API quota):

1. **Deterministic Nifty 500 Quantitative Funnel (`run_quant_screener` — 0 LLM calls):**
   - Loads `data/ind_nifty500list.csv` (appending `.NS` to each NSE symbol) plus `^NSEI`, `^CRSLDX`, and `^INDIAVIX`.
   - Bulk-downloads 1-year OHLCV via `yf.download(tickers, period="1y", group_by="ticker", threads=True)`.
   - **Liquidity Filter:** Excludes illiquid counters with 20-day average daily turnover `< ₹10 Crores`.
   - **Minervini Stage-2 Uptrend Filter:** Requires `Close > SMA_50 > SMA_150 > SMA_200` and `Close >= 0.75 * 52W_High`.
   - **Relative Strength Filter:** Requires `RS_Alpha_63d > 0.0` (must be outperforming Nifty 50).
   - **Quality Fundamental Filter (Cached 14 days in `fundamental_cache`):**
     - Non-Financials: `ROE >= 0.15`, `Debt/Equity <= 100.0` (1.0x), `Revenue Growth > 0.05`, and `Operating Cash Flow > 0`.
     - Financials/Banks/NBFCs (`sector == "Financial Services"`): Skip `Debt/Equity` and `Operating Cash Flow` checks; require `ROE >= 0.14` and positive `Earnings Growth`.
   - Ranks passing stocks by `RS_Alpha_63d` descending and selects the **Top 3**.
2. **Indian Market News & Catalyst Scout (`run_catalyst_scout` — 1 `gemini-2.5-flash` call):**
   - Parses Indian financial RSS feeds via `feedparser`:
     - Economic Times Markets (`https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms`)
     - Moneycontrol Business/Markets (`https://www.moneycontrol.com/rss/marketreports.xml`)
     - Livemint Markets (`https://www.livemint.com/rss/markets`)
   - Extracts up to **2 NSE-listed businesses** experiencing concrete catalysts (major PSU/defence/infra order wins, PLI scheme approvals, capacity expansion/capex, quarterly earnings beats, or demergers).
   - Validates each extracted symbol against `yf.Ticker(f"{symbol}.NS").fast_info` before adding to the docket.

---

## 7. Multi-Agent State & LangGraph Workflow (`state.py`, `agents.py`, `graph.py`)

### 7.1 `InvestmentFirmState` (`state.py`)
```python
from typing import TypedDict, Optional, Dict, Any, List, Literal
from pydantic import BaseModel, Field

class TradeDecision(BaseModel):
    ticker: str = Field(description="NSE ticker with .NS suffix, e.g., BEL.NS")
    action: Literal["BUY", "SELL", "HOLD"]
    conviction: Literal["HIGH", "MEDIUM", "LOW"]
    allocation_pct: float = Field(ge=0.0, le=0.12, description="Max 12% single-stock cap")
    entry_price: float
    stop_loss_price: float = Field(description="Must be <= entry_price - 2 * ATR_14")
    target_price: float = Field(description="Must satisfy >= 2.5x risk-to-reward ratio")
    holding_horizon_days: int = Field(default=30, description="Max holding window before review")
    rationale: str = Field(description="Concise 3-sentence executive summary")

class TradePostMortem(BaseModel):
    outcome: Literal["WIN", "LOSS", "BREAKEVEN"]
    pnl_pct: float
    primary_driver: Literal[
        "TECHNICAL_BREAKOUT_VALID",
        "FALSE_BREAKOUT_LOW_VOLUME",
        "FUNDAMENTAL_CATALYST_HIT",
        "EARNINGS_OR_SEBI_NEWS_SHOCK",
        "NIFTY_MACRO_SECTOR_DRAG",
        "STOP_LOSS_TOO_TIGHT"
    ]
    accurate_agent: Literal["TECHNICAL", "FUNDAMENTAL", "NEWS_MACRO", "BULL_BEAR_DEBATE", "RISK_COMMITTEE", "NONE"]
    failed_agent: Literal["TECHNICAL", "FUNDAMENTAL", "NEWS_MACRO", "BULL_BEAR_DEBATE", "RISK_COMMITTEE", "NONE"]
    root_cause_analysis: str = Field(description="2-sentence diagnosis of why the thesis succeeded or failed.")
    actionable_rule: str = Field(description="Concrete 1-sentence guardrail rule for future committee votes.")

class InvestmentFirmState(TypedDict):
    ticker: str
    sector: str
    market_regime: str                  # "BULL_STAGE2" | "CHOPPY_RANGE" | "BEAR_RISK_OFF"
    india_vix: float
    portfolio_context: Dict[str, Any]   # Cash (INR), total NAV, active positions, sector weights
    technicals: Dict[str, Any]          # Pre-computed indicators + risk boundaries
    fundamentals: Dict[str, Any]        # Cached ratios + OCF/NetIncome quality check
    recent_news: str                    # Formatted headlines + upcoming quarterly results warning
    episodic_lessons: str               # Top 3 retrieved rules from trade_post_mortems
    agent_scorecard: str                # Rolling 30-trade accuracy string per agent
    
    # Stage 1: Analyst Outputs
    technical_report: Optional[str]
    fundamental_report: Optional[str]
    news_macro_report: Optional[str]
    
    # Stage 2: Dialectical Debate State
    debate_round: int
    bull_arguments: List[str]
    bear_arguments: List[str]
    research_proposal: Optional[str]
    
    # Stage 3: 3-Way Risk Committee Outputs
    aggressive_risk_view: Optional[str]
    neutral_risk_view: Optional[str]
    conservative_risk_view: Optional[str]
    
    # Stage 4: Final Structured Decision
    trade_decision: Optional[Dict[str, Any]]
```

### 7.2 Paced Sequential Graph Execution (`graph.py`)
**Important Free-Tier Design Note:** Instead of firing 3 parallel LLM threads simultaneously (which can spike RPM and trigger Google AI Studio `429` throttling), wire Stage 1 and Stage 3 nodes sequentially through the central **`paced_llm_invoke()`** wrapper (`MIN_SECONDS_BETWEEN_LLM_CALLS = 6.5s`). Each ticker completes cleanly in ~65–75 seconds with zero rate-limit failures.

1. **`data_and_memory_node` (Python):** Populates `technicals`, `fundamentals`, `recent_news`, `market_regime`, `india_vix`, `portfolio_context`, `episodic_lessons`, and `agent_scorecard`.
2. **Stage 1 — Specialized Analyst Team (`gemini-2.5-flash`):**
   - `technical_analyst_node`: Audits Stage-2 trend alignment, RSI sweet spot (`50–68`), volume breakout confirmation (`Volume_Ratio_20d > 1.2`), circuit-filter warnings, and support/resistance levels.
   - `fundamental_analyst_node`: Audits cash conversion (`Operating Cash Flow / Net Income >= 0.8` for non-financials), valuation (`PE vs Forward PE`), RoE durability, promoter holding stability, and debt safety.
   - `news_macro_analyst_node`: Evaluates quarterly results dates (flags blackout if board meeting / results are within `< 5 trading days`), RBI monetary policy / government capex tailwinds, SEBI notices, and institutional order wins.
3. **Stage 2 — Dialectical Bull/Bear Debate (`gemini-2.5-flash` $\rightarrow$ `gemini-2.5-pro`):**
   - `bull_researcher_node`: Builds the highest-conviction upside case grounded strictly in Stage 1 reports.
   - `bear_researcher_node`: Red-teams the thesis, citing past post-mortem failure patterns from `episodic_lessons`, promoter/valuation red flags, or FII outflow risks.
   - *Conditional Edge (`should_continue_debate`):* Loops `bull -> bear` until `debate_round >= MAX_DEBATE_ROUNDS`, then routes to `research_manager_node`.
   - `research_manager_node` (`deep_think_llm` — Pro with Flash fallback): Adjudicates the debate and writes a structured `research_proposal`.
4. **Stage 3 — Three-Angle Risk Committee (`gemini-2.5-flash`):**
   - `aggressive_risk_node`: Argues for full position sizing when 63-day relative strength vs. Nifty 50 and earnings catalysts align.
   - `neutral_risk_node`: Checks portfolio concentration (`MAX_SECTOR_EXPOSURE_PCT`), correlation with existing holdings, and NSE daily turnover liquidity.
   - `conservative_risk_node`: Enforces the hard `2 * ATR_14` stop-loss, verifies `>= 2.5:1` risk/reward, and vetoes trades entering right before quarterly earnings or when `market_regime == "BEAR_RISK_OFF"`.
5. **Stage 4 — Lead Portfolio Manager (`portfolio_manager_node` via `deep_think_llm` — Pro with Flash fallback):**
   - Bound to `TradeDecision` via `.with_structured_output(TradeDecision)`.
   - **Mandatory Prompt Rule:** *"Weight analyst and debate claims using the `agent_scorecard` historical hit rates. If `market_regime == 'BEAR_RISK_OFF'`, reject low/medium-conviction buys to hold up to 50% cash. Never override `hard_stop_price` or `min_valid_target`."*

---

## 8. NSE Paper Broker, Intraday Guard & Post-Mortem Reflection (`paper_broker.py` & `reflection_engine.py`)

### 8.1 Simulated Indian Delivery Execution (`paper_broker.py`)
- **On `BUY` (`conviction in ["HIGH", "MEDIUM"]`):**
  - Verifies `ticker` not already in `positions`, `len(positions) < MAX_ACTIVE_POSITIONS`, and sector weight `< 30%`.
  - Calculates `qty = min(int((total_nav * decision.allocation_pct) // current_price), recommended_max_qty)`.
  - Computes realistic Indian delivery fees:
    `buy_fees = (qty * current_price) * (STT_PCT + EXCHANGE_AND_SEBI_PCT + STAMP_DUTY_BUY_PCT + ESTIMATED_SLIPPAGE_PCT)`
  - Deducts `(qty * current_price) + buy_fees` from `portfolio_account.cash`.
  - Inserts row into `positions`, logs `BUY` in `trade_log`, and saves full state into `thesis_snapshots` (`status = 'OPEN'`).
- **On `SELL` (PM signal or automated intraday stop/target):**
  - Computes realistic Indian sell fees:
    `sell_fees = (qty * current_price) * (STT_PCT + EXCHANGE_AND_SEBI_PCT + ESTIMATED_SLIPPAGE_PCT)`
  - Credits `(qty * current_price) - sell_fees` to `portfolio_account.cash`, computes net `realized_pnl`, logs `SELL` in `trade_log`, deletes from `positions`, and triggers `reflection_engine.run_post_mortem_audit()`.

### 8.2 Post-Mortem Self-Reflection Engine (`reflection_engine.py`)
When a trade closes:
1. Fetches the `OPEN` `thesis_snapshots` row and downloads daily OHLCV between `opened_at` and `closed_at` to reconstruct max favorable excursion (MFE), max adverse excursion (MAE), and gap openings.
2. Invokes `deep_think_llm.with_structured_output(TradePostMortem)` (Pro with Flash fallback) to attribute `accurate_agent` and `failed_agent` and synthesize a 1-sentence `actionable_rule`.
3. Saves the record to `trade_post_mortems` and marks the snapshot `CLOSED`.
4. **`get_agent_scorecard()`:** Queries the last 30 rows of `trade_post_mortems` to compute win-attribution percentages per agent (`Accurate Count / (Accurate Count + Failed Count)`), injecting live credibility weights into Stage 4.

---

## 9. IST Market Schedule & CLI (`daemon.py` & `analytics.py`)
All scheduled jobs run in **`Asia/Kolkata` (IST)** timezone on NSE trading days (Monday–Friday):
1. **`08:30 AM IST`:** Runs `discovery_engine.py` (Nifty 500 quant screener + ET/Moneycontrol/Livemint RSS catalyst scout) to generate today's Docket (max 5 tickers).
2. **`09:30 AM IST` (15 mins after NSE market open):** Runs the paced LangGraph committee on existing holdings (for exit/hold review) and today's discovered Docket.
3. **`Every 5 minutes` (`09:15 AM – 03:30 PM IST`):** Runs deterministic `check_intraday_stops_and_targets()` in pure Python (zero API calls).
4. **`03:45 PM IST` (After NSE market close):** Logs end-of-day portfolio NAV (in ₹) alongside `^NSEI` (Nifty 50), `^CRSLDX` (Nifty 500), and `^INDIAVIX` into `daily_nav_history`, and runs any pending post-mortem audits.

### CLI Entry Commands (`daemon.py`)
- `python daemon.py --analyze TRENT.NS` — Run a single-ticker NSE committee dry-run and print the full Bull/Bear debate, 3-way risk critique, and memo.
- `python daemon.py --discover` — Test the Nifty 500 quant screener and Indian RSS catalyst scout.
- `python daemon.py --report` — Print portfolio NAV (₹), Alpha ($\alpha$) vs. Nifty 50, Sharpe Ratio, Max Drawdown, and the Rolling 30-Trade Agent Accuracy Scorecard.
- `python daemon.py --daemon` — Start the unattended IST autonomous paper trading daemon.