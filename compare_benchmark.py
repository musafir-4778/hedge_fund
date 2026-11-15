import yfinance as yf
import pandas as pd
import sqlite3

blocks = {
    "Block 1 (Post-COVID Recovery Bull)": ("2020-06-01", "2021-10-01", 9.77),
    "Block 2 (Global Inflation & Rate-Hike Chop)": ("2021-11-01", "2023-03-01", 9.57),
    "Block 3 (Capex Supercycle & Election)": ("2023-04-01", "2024-09-01", 8.92),
    "Block 4 (Bear Slide & Valuation De-rating)": ("2025-12-01", "2026-08-31", 2.57),
    "OOS Window A (Out-of-Sample Plateau)": ("2024-10-01", "2025-11-30", 5.95),
    "OOS Window B (Terminal Out-of-Sample)": ("2026-09-01", "2026-11-13", 0.0),
}

# Download full Nifty 50 history
ticker = yf.Ticker("^NSEI")
hist = ticker.history(start="2020-05-25", end="2026-11-16")

print(f"Nifty 50 Historical Data Loaded: {len(hist)} trading sessions\n")

results = []
for name, (start_dt, end_dt, bot_ret) in blocks.items():
    sub = hist.loc[start_dt:end_dt]
    if len(sub) >= 2:
        start_p = float(sub["Close"].iloc[0])
        end_p = float(sub["Close"].iloc[-1])
        nifty_ret = ((end_p - start_p) / start_p) * 100.0
        alpha = bot_ret - nifty_ret
        beat = bot_ret > nifty_ret
        results.append({
            "block": name,
            "start_dt": start_dt,
            "end_dt": end_dt,
            "start_p": start_p,
            "end_p": end_p,
            "nifty_ret": nifty_ret,
            "bot_ret": bot_ret,
            "alpha": alpha,
            "beat": beat
        })
    else:
        # Fallback to nearest dates
        idx_after_start = hist.index[hist.index >= start_dt]
        idx_before_end = hist.index[hist.index <= end_dt]
        if len(idx_after_start) > 0 and len(idx_before_end) > 0:
            start_p = float(hist.loc[idx_after_start[0], "Close"])
            end_p = float(hist.loc[idx_before_end[-1], "Close"])
            nifty_ret = ((end_p - start_p) / start_p) * 100.0
            alpha = bot_ret - nifty_ret
            beat = bot_ret > nifty_ret
            results.append({
                "block": name,
                "start_dt": start_dt,
                "end_dt": end_dt,
                "start_p": start_p,
                "end_p": end_p,
                "nifty_ret": nifty_ret,
                "bot_ret": bot_ret,
                "alpha": alpha,
                "beat": beat
            })

for r in results:
    verdict = f"BEAT MARKET (+{r['alpha']:.2f}% Alpha)" if r['beat'] else f"LAGGED MARKET ({r['alpha']:.2f}%)"
    print(f"==================================================")
    print(f" {r['block']}")
    print(f" Dates: {r['start_dt']} to {r['end_dt']}")
    print(f" Nifty 50: ₹{r['start_p']:,.2f} -> ₹{r['end_p']:,.2f} ({r['nifty_ret']:+.2f}%)")
    print(f" Bot (Net After All Taxes): {r['bot_ret']:+.2f}%")
    print(f" Verdict: {verdict}")
    print(f"==================================================\n")

beat_count = sum(1 for r in results if r['beat'])
print(f"SUMMARY: Bot beat the market in {beat_count} out of {len(results)} blocks/windows!")

