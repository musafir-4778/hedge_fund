# Autonomous Wealth Management Investment Office (Indian Equities NSE/BSE)

An institutional-grade, multi-agent autonomous investment office for Indian Equities built strictly according to [`gemini.md`](file:///d:/gravity_proj/Hedge_fund/gemini.md), operating entirely within Google AI Studio's **free tier** using a failover ladder across Gemini models.

---

## 🏛 Architecture Overview

```
                      [Stage -1: Discovery Engine]
        Nifty 500 Minervini Screener + Indian Financial RSS
                      (0-1 API call, 5 tickers/day)
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │        Stage 1: Specialized Research Audits         │
        │  [1/10] Technical  │  [2/10] Fundamental  │  [3/10] Macro │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │       Stage 2: Dialectical Bull vs. Bear Debate     │
        │   [4/10] Bull Researcher  ◄─►  [5/10] Bear Red-Team │
        │                          │                          │
        │          [6/10] Research Manager Synthesis          │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │          Stage 3: Three-Angle Risk Committee        │
        │  [7/10] Aggressive │ [8/10] Neutral │ [9/10] Conservative│
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │    Stage 4: Lead Portfolio Manager & Paper Broker   │
        │        [10/10] Structured Trade Order Execution     │
        └──────────────────────────┬──────────────────────────┘
                                   │
                                   ▼
        ┌─────────────────────────────────────────────────────┐
        │      Self-Reflection & Post-Mortem Learning Engine   │
        │  Root-Cause Attribution -> Guardrails -> SQLite DB  │
        └─────────────────────────────────────────────────────┘
```

---

## 🧠 Continuous Learning from Mistakes

Whenever a trade hits its stop-loss or exits with a loss, the **Reflection Engine** (`reflection_engine.py`):
1. **Reconstructs Price Excursion**: Computes Max Adverse Excursion (MAE) and Max Favorable Excursion (MFE) during the holding period.
2. **Diagnoses Root Cause**: Attributes failure to a specific category (`ACCOUNTING_QUALITY`, `TECHNICAL_BLINDSIDE`, `MACRO_SHOCK`, `VALUATION_TRAP`).
3. **Single-Agent Blame Assignment**: Identifies the exact agent with the blind spot (e.g. `FUNDAMENTAL_ANALYST` or `TECHNICAL_ANALYST`) and updates the **Rolling 30-Trade Accuracy Scorecard**.
4. **Synthesizes Guardrails**: Generates a concrete, enforceable rule stored in SQLite (`trade_post_mortems`).
5. **Closed-Loop Prompt Injection**: Every subsequent committee meeting extracts both sector-specific and top general guardrails and injects them under `MANDATORY GUARDRAILS LEARNED FROM PAST MISTAKES` across all agent prompts.

---

## ⚡ Free-Tier Model Ladder & Rate Limiter

Because free-tier models have varying daily quotas and server response times:
- **Primary Workhorse**: `gemini-3.5-flash-lite` (highest daily request capacity, blazing 3–5s response times, no 429 quota exhaustion).
- **Failover Ladder**:
  $$\text{gemini-3.5-flash-lite} \longrightarrow \text{gemini-3.1-flash-lite} \longrightarrow \text{gemini-3.6-flash} \longrightarrow \text{gemini-3.8-flash} \longrightarrow \text{gemini-3-flash-preview}$$
- **Pacing Engine**: Guaranteed $\ge 6.5$s sleep between LLM calls ($\le 9$ RPM) to avoid hitting per-minute limits.
- **Circuit Breaker**: On `429` (daily quota), models are quarantined for the remainder of the day so zero wasted retries occur. On `504` (server deadline) or `503` (overloaded), it cascades instantly to the next tier.

---

## 💻 Terminal Commands

All commands run inside the virtual environment (`.\.venv\Scripts\python.exe`):

### 1. Live Paper Trading Session (Autonomous Committee)
```bash
python daemon.py --trade-now
```
Runs the Discovery Engine, convenes the 10-Agent Investment Committee, and automatically executes high-conviction BUY orders into the paper broker. You can also target a specific stock:
```bash
python daemon.py --trade-now TRENT.NS
```

### 2. Direct Paper Broker Execution (Buy / Sell)
Execute paper orders directly with full Indian delivery tax modeling (STT, NSE turnover, SEBI, stamp duty, slippage) and 2xATR sizing:
```bash
python daemon.py --buy ADANIPORTS.NS
python daemon.py --sell ADANIPORTS.NS
```

### 3. Check Intraday Risk Stops & Targets
```bash
python daemon.py --check-stops
```
Polls live market prices for all active portfolio holdings and triggers automatic trailing stops or target exits.

### 4. Executive Scorecard & NAV Dashboard
```bash
python daemon.py --report
```
Renders the portfolio NAV, available cash, total return, alpha/beta vs Nifty 50, active positions, win attribution bars, and active guardrails.

### 5. Inspect Institutional Memory Bank (Learned Mistakes)
```bash
python daemon.py --learn
```
Shows the chronological log of trade post-mortems, root-cause attributions, and actionable rules synthesized by the system.

### 6. Simulate Learning from a Mistake
```bash
python daemon.py --simulate-mistake TATAMOTORS.NS
```
Simulates a trade loss, diagnoses root-cause blind spots, synthesizes a guardrail rule, records it into SQLite, and immediately updates the Rolling 30-Trade Scorecard.

### 7. 1-Year Walk-Forward Simulated Learning Backtest (Strict No Lookahead Bias)
```bash
python daemon.py --simulate-1y
```
Runs a point-in-time walk-forward backtest over the past 248 trading sessions. On every day $T$, it has zero visibility into future data ($> T$). When trades hit stop-losses, the reflection engine diagnoses root causes, updates the scorecard, synthesizes guardrails, and enforces them on subsequent days—showing how many future bad trades were prevented!

### 8. Reset Portfolio to Clean Initial Balance
```bash
python daemon.py --reset --clean-all
```
Resets paper portfolio cash to clean ₹10,00,000.00 and clears positions/test trade logs.

### 9. Unattended Market Daemon & Cloud Cron Runs
```bash
# Continuous local / VM background scheduler
python daemon.py --daemon

# Single serverless cloud triggers (e.g. GitHub Actions or Cloud Cron)
python daemon.py --morning-run   # 08:30 Discovery + 09:30 Committee Execution
python daemon.py --close-run     # 03:15 Trailing Stop Elevation + 03:45 Close Settlement
```

Runs on Asia/Kolkata timezone:
- **08:30 IST**: Discovery engine generates today's docket.
- **09:30 IST**: LangGraph committee evaluates docket and existing holdings.
- **09:15 - 15:30 IST**: Every 5 min deterministic stop-loss & target monitor.
- **03:15 IST**: Pre-close dynamic trailing stop elevation on profitable holdings.
- **03:45 IST**: Post-market NAV recording and reflection audits.

### 10. Month-to-Month Live Committee Debate Backtest
```bash
python daemon.py --monthly-debate --months 12 --log-file monthly_debate_12m.txt
```
Runs a 12-month historical walk-forward simulation across a 10-stock universe with up to 3 concurrent holdings, dynamic trailing stops, and full terminal output logging.

### 11. Run Complete Unit & Integration Test Suite
```bash
python test_suite.py
```
Executes all 6 deterministic tests in $<0.05$ seconds.

---

## ☁️ Zero-Laptop Cloud Deployment

You do not need to keep your laptop running during market hours (09:00 AM – 03:45 PM IST). Three cloud hosting methods are supported:

### Option A: Free Serverless via GitHub Actions (Recommended)
The repository includes [`.github/workflows/market_daemon.yml`](file:///d:/gravity_proj/Hedge_fund/.github/workflows/market_daemon.yml):
1. In your GitHub repository, navigate to **Settings > Secrets and variables > Actions**.
2. Add three Repository Secrets:
   - `GOOGLE_API_KEY`: Your Gemini API key from Google AI Studio.
   - `TELEGRAM_BOT_TOKEN`: Your Telegram bot token from `@BotFather`.
   - `TELEGRAM_CHAT_ID`: Your personal or group chat ID.
3. GitHub Actions triggers automatically:
   - **09:00 AM IST** (03:30 UTC): Executes `daemon.py --morning-run`.
   - **03:15 PM IST** (09:45 UTC): Executes `daemon.py --close-run`.
   - Automatically commits the updated SQLite database (`paper_portfolio_india.db`) with `[skip ci]`.
   - Sends real-time trade signals and PnL summaries to your Telegram app.

### Option B: Docker Container
Run seamlessly on any server, Synology NAS, or cloud container service:
```bash
# Build and run in detached mode
docker compose up -d --build

# View real-time container logs
docker compose logs -f
```

### Option C: Always-On Free VM (GCP e2-micro, AWS, Oracle Cloud)
1. Deploy an Ubuntu VM on Google Cloud Free Tier (`e2-micro`).
2. Clone the repository and configure `.env`:
   ```bash
   cp .env.example .env
   # Edit with your GOOGLE_API_KEY, TELEGRAM_BOT_TOKEN, TELEGRAM_CHAT_ID
   ```
3. Set up the systemd service:
   ```bash
   sudo cp wealth-hedgefund.service /etc/systemd/system/
   sudo systemctl daemon-reload
   sudo systemctl enable --now wealth-hedgefund
   ```


