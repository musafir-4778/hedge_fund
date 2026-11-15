"""discovery_engine.py — Stage -1: Nifty 500 Deterministic Funnel + Indian Financial RSS Scout.

Builds the daily 5-ticker Committee Docket before NSE pre-open (08:30 AM IST)
using 0 LLM calls for technical/fundamental filtering and exactly 1 free Flash call
for catalytic news extraction.
"""

import json
import logging
from typing import List, Dict, Any, Optional
import feedparser
import pandas as pd
import yfinance as yf

from config import (
    NIFTY500_CSV_PATH,
    BENCHMARK_TICKER,
    VIX_TICKER,
    MAX_DAILY_DOCKET_SIZE,
    get_quick_think_llm,
    paced_llm_invoke,
    extract_text_content,
)
from data_tools import (
    normalize_ticker,
    compute_wilder_rsi,
    fetch_fundamental_metrics,
)

logger = logging.getLogger("wealth_agent_office.discovery")

RSS_FEEDS = [
    "https://economictimes.indiatimes.com/markets/rssfeeds/1977021501.cms",
    "https://www.moneycontrol.com/rss/marketreports.xml",
    "https://www.livemint.com/rss/markets",
]


def load_nifty500_tickers(sample_limit: Optional[int] = None) -> List[Dict[str, str]]:
    """Loads Nifty 500 universe from data/ind_nifty500list.csv."""
    try:
        df = pd.read_csv(NIFTY500_CSV_PATH)
        records = []
        for _, row in df.iterrows():
            sym = str(row["Symbol"]).strip()
            ind = str(row.get("Industry", "General")).strip()
            records.append({
                "symbol": sym,
                "ticker": f"{sym}.NS",
                "industry": ind
            })
        if sample_limit:
            records = records[:sample_limit]
        return records
    except Exception as e:
        logger.error(f"Failed to load Nifty 500 list from {NIFTY500_CSV_PATH}: {e}")
        # Core fallback universe
        default_syms = [
            ("RELIANCE", "Oil & Gas"), ("TCS", "Information Technology"), ("HDFCBANK", "Financial Services"),
            ("ICICIBANK", "Financial Services"), ("INFY", "Information Technology"), ("BHARTIARTL", "Telecommunication"),
            ("ITC", "Fast Moving Consumer Goods"), ("LT", "Construction"), ("TRENT", "Consumer Services"),
            ("BEL", "Capital Goods"), ("HAL", "Capital Goods"), ("TITAN", "Consumer Durables"),
            ("TATAMOTORS", "Automobile"), ("M&M", "Automobile"), ("SBIN", "Financial Services")
        ]
        return [{"symbol": s, "ticker": f"{s}.NS", "industry": ind} for s, ind in default_syms]


def run_quant_screener(universe_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Deterministic Nifty 500 Quantitative Funnel (0 LLM calls).

    Filters by:
    1. Liquidity: 20-day avg daily turnover >= ₹10 Crores (10,00,00,000 INR)
    2. Minervini Stage-2 Uptrend: Close > SMA50 > SMA150 > SMA200 and Close >= 0.75 * 52W_High
    3. Relative Strength: RS_Alpha_63d > 0.0 vs Nifty 50
    4. Quality Fundamentals:
       - Non-Financials: ROE >= 0.15, Debt/Equity <= 100.0 (1.0x), Revenue Growth > 0.05, OCF > 0
       - Financials: ROE >= 0.14, positive Earnings Growth

    Returns Top 3 candidates ranked by RS_Alpha_63d descending.
    """
    logger.info("Running Stage -1: Deterministic Nifty 500 Quant Screener...")
    universe = load_nifty500_tickers(sample_limit=universe_limit)
    ticker_map = {item["ticker"]: item for item in universe}
    symbols_to_download = list(ticker_map.keys()) + [BENCHMARK_TICKER]

    try:
        data = yf.download(
            symbols_to_download,
            period="1y",
            interval="1d",
            group_by="ticker",
            threads=True,
            progress=False
        )
    except Exception as e:
        logger.error(f"yfinance bulk download failed: {e}")
        return []

    # Extract Nifty 50 benchmark close series
    try:
        nifty_series = data[BENCHMARK_TICKER]["Close"].dropna()
        nifty_ret_63 = (float(nifty_series.iloc[-1]) / float(nifty_series.iloc[-63])) - 1.0 if len(nifty_series) >= 63 else 0.0
    except Exception:
        nifty_ret_63 = 0.0

    candidates = []

    for item in universe:
        sym = item["ticker"]
        try:
            if sym not in data:
                continue
            df = data[sym].dropna(how="all").copy()
            if df.empty or len(df) < 63:
                continue

            close = df["Close"].dropna()
            volume = df["Volume"].dropna()
            if len(close) < 63:
                continue

            latest_close = float(close.iloc[-1])
            latest_volume = float(volume.iloc[-1])

            # 1. Liquidity Filter: 20-day average turnover >= ₹10 Crores (1e8 INR)
            vol_20 = volume.tail(20)
            close_20 = close.tail(20)
            avg_daily_turnover_inr = float((vol_20 * close_20).mean())
            if avg_daily_turnover_inr < 1e8:  # 10 Crores
                continue

            # 2. Minervini Stage-2 Uptrend Filter
            sma_50 = float(close.rolling(50).mean().iloc[-1])
            sma_150 = float(close.rolling(150).mean().iloc[-1]) if len(close) >= 150 else sma_50
            sma_200 = float(close.rolling(200).mean().iloc[-1]) if len(close) >= 200 else sma_150
            high_52w = float(df["High"].max())

            if not (latest_close > sma_50 > sma_150 > sma_200 and latest_close >= (0.75 * high_52w)):
                continue

            # 3. Relative Strength Filter: RS_Alpha_63d > 0.0
            stock_ret_63 = (latest_close / float(close.iloc[-63])) - 1.0
            rs_alpha_63d = stock_ret_63 - nifty_ret_63
            if rs_alpha_63d <= 0.0:
                continue

            # 4. Quality Fundamental Filter
            fund = fetch_fundamental_metrics(sym)
            sector = fund.get("sector", item.get("industry", ""))

            if sector == "Financial Services":
                # Financials: ROE >= 0.14 and positive earnings growth
                if fund.get("roe", 0.0) < 0.14 or fund.get("earnings_growth", 0.0) <= 0.0:
                    continue
            else:
                # Non-financials
                roe_pass = fund.get("roe", 0.0) >= 0.15
                debt_pass = fund.get("debt_to_equity", 0.0) <= 100.0  # 1.0x
                rev_pass = fund.get("revenue_growth", 0.0) > 0.05
                ocf_pass = fund.get("operating_cash_flow", 0.0) > 0.0
                if not (roe_pass and debt_pass and rev_pass and ocf_pass):
                    continue

            candidates.append({
                "ticker": sym,
                "symbol": item["symbol"],
                "sector": sector,
                "latest_close": round(latest_close, 2),
                "rs_alpha_63d": round(rs_alpha_63d, 4),
                "avg_turnover_cr": round(avg_daily_turnover_inr / 1e7, 2),
                "source": "QUANT_SCREENER"
            })

        except Exception as e:
            logger.debug(f"Screener check skipped for {sym}: {e}")
            continue

    # Rank by RS_Alpha_63d descending and return Top 3
    candidates.sort(key=lambda x: x["rs_alpha_63d"], reverse=True)
    top_3 = candidates[:3]
    logger.info(f"Quant Screener selected top {len(top_3)} candidates: {[c['ticker'] for c in top_3]}")
    return top_3


def run_catalyst_scout() -> List[Dict[str, Any]]:
    """Indian Market News & Catalyst Scout (1 gemini-2.5-flash call).

    Parses Indian financial RSS feeds and extracts up to 2 NSE tickers experiencing
    concrete operational catalysts. Validates each symbol via yfinance.
    """
    logger.info("Running Stage -1: Indian Financial RSS Catalyst Scout...")
    headlines = []
    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
            for entry in feed.entries[:10]:
                title = getattr(entry, "title", "").strip()
                summary = getattr(entry, "summary", "").strip()
                if title:
                    headlines.append(f"- {title} : {summary[:150]}")
        except Exception as e:
            logger.warning(f"Error fetching RSS feed {feed_url}: {e}")

    if not headlines:
        logger.warning("No RSS headlines fetched from feeds.")
        return []

    combined_news = "\n".join(headlines[:25])
    prompt = f"""You are the Indian Financial Catalyst Scout for a Wealth Management Investment Office.
Analyze these live Indian market headlines:
\"\"\"
{combined_news}
\"\"\"

Identify up to 2 NSE-listed Indian companies that have CONCRETE positive catalysts:
- Major PSU/defence/infrastructure order win
- PLI scheme approval or capex expansion
- Strong quarterly results beat
- Value-unlocking demerger or acquisition

Respond ONLY in valid raw JSON format matching this exact schema (no markdown fences, no extra text):
[
  {{
    "symbol": "NSE_SYMBOL_WITHOUT_SUFFIX",
    "catalyst": "Brief 1-sentence description of the catalyst"
  }}
]
If none qualify, return []."""

    try:
        llm = get_quick_think_llm(temperature=0.1)
        response = paced_llm_invoke(llm, prompt)
        text = extract_text_content(response)

        # Parse JSON
        text_clean = text.strip()
        if text_clean.startswith("```"):
            lines = text_clean.splitlines()
            text_clean = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
        parsed = json.loads(text_clean)

        verified_catalysts = []
        for item in parsed[:2]:
            raw_sym = item.get("symbol", "").strip().upper()
            if not raw_sym:
                continue
            ticker = normalize_ticker(raw_sym)
            # Validate ticker with fast_info
            try:
                tk = yf.Ticker(ticker)
                fi = tk.fast_info
                last_price = getattr(fi, "last_price", None)
                if last_price and last_price > 0:
                    fund = fetch_fundamental_metrics(ticker)
                    verified_catalysts.append({
                        "ticker": ticker,
                        "symbol": raw_sym,
                        "sector": fund.get("sector", "Industrials"),
                        "latest_close": round(float(last_price), 2),
                        "rs_alpha_63d": 0.05,
                        "catalyst_note": item.get("catalyst", ""),
                        "source": "CATALYST_SCOUT"
                    })
            except Exception as e:
                logger.debug(f"Catalyst ticker validation failed for {ticker}: {e}")

        logger.info(f"Catalyst Scout identified {len(verified_catalysts)} candidates: {[c['ticker'] for c in verified_catalysts]}")
        return verified_catalysts
    except Exception as e:
        logger.warning(f"Catalyst scout LLM parsing failed: {e}")
        return []


def get_daily_docket(universe_limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Builds the final Committee Docket (Max 5 tickers) for the trading day.

    Combines Top 3 Quant Screener picks + Top 2 Catalyst Scout picks,
    ensuring no duplicate tickers.
    """
    quant_picks = run_quant_screener(universe_limit=universe_limit)
    scout_picks = run_catalyst_scout()

    docket = []
    seen_tickers = set()

    for p in quant_picks:
        if p["ticker"] not in seen_tickers:
            docket.append(p)
            seen_tickers.add(p["ticker"])

    for p in scout_picks:
        if p["ticker"] not in seen_tickers and len(docket) < MAX_DAILY_DOCKET_SIZE:
            docket.append(p)
            seen_tickers.add(p["ticker"])

    logger.info(f"Final Daily Committee Docket ({len(docket)} tickers): {[d['ticker'] for d in docket]}")
    return docket

