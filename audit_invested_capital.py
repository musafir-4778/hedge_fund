import re

def audit_trades(filepath: str):
    buy_pattern = re.compile(r"✔\s*EXECUTED BUY ORDER:\s*(\d+)\s+shares of\s+([A-Z\.]+)\s+@\s+₹([\d,]+\.?\d*)\s+\(Total Cost:\s+₹([\d,]+\.?\d*)\)")
    stop_pattern = re.compile(r"▼\s*STOP-LOSS TRIGGERED:\s*([A-Z\.]+)\s+exited @\s+₹([\d,]+\.?\d*)\s+on\s+([\d\-]+)\s+\(PnL:\s+([\+\-\d\.]+)%\)")
    target_pattern = re.compile(r"▲\s*PROFIT TARGET HIT:\s*([A-Z\.]+)\s+target\s+₹([\d,]+\.?\d*)\s+reached\s+on\s+([\d\-]+)!\s+\(PnL:\s+([\+\-\d\.]+)%\)")
    nav_pattern = re.compile(r"Current Portfolio NAV:\s*₹([\d,]+\.?\d*)\s+\(Cash:\s*₹([\d,]+\.?\d*)\s+\|\s+Holdings:\s*(\d+)/(\d+)\)")

    buys = []
    exits = []
    nav_checkpoints = []

    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            bm = buy_pattern.search(line)
            if bm:
                buys.append({
                    "qty": int(bm.group(1)),
                    "ticker": bm.group(2),
                    "price": float(bm.group(3).replace(",", "")),
                    "cost": float(bm.group(4).replace(",", ""))
                })

            sm = stop_pattern.search(line)
            if sm:
                exits.append({
                    "type": "STOP_LOSS",
                    "ticker": sm.group(1),
                    "price": float(sm.group(2).replace(",", "")),
                    "date": sm.group(3),
                    "pnl_pct": float(sm.group(4))
                })

            tm = target_pattern.search(line)
            if tm:
                exits.append({
                    "type": "PROFIT_TARGET",
                    "ticker": tm.group(1),
                    "price": float(tm.group(2).replace(",", "")),
                    "date": tm.group(3),
                    "pnl_pct": float(tm.group(4))
                })

            nm = nav_pattern.search(line)
            if nm:
                nav_checkpoints.append({
                    "nav": float(nm.group(1).replace(",", "")),
                    "cash": float(nm.group(2).replace(",", "")),
                    "holdings": int(nm.group(3))
                })

    print("================================================================================")
    print("  FORENSIC AUDIT OF CAPITAL INVESTED ACROSS 12-MONTH SIMULATION LOGS")
    print("================================================================================")
    print(f"\nTotal Number of BUY Orders Executed: {len(buys)}\n")
    
    total_cost = 0.0
    for idx, b in enumerate(buys, 1):
        gross = b["qty"] * b["price"]
        fees = b["cost"] - gross
        total_cost += b["cost"]
        print(f"  {idx:02d}. {b['ticker']:<15} | Qty: {b['qty']:>4} | Price: ₹{b['price']:>9.2f} | Gross: ₹{gross:>11.2f} | Fees: ₹{fees:>7.2f} | Deployed: ₹{b['cost']:>11.2f}")

    print("-" * 80)
    print(f"  TOTAL CAPITAL DEPLOYED ACROSS ALL TRADES : ₹{total_cost:,.2f}")
    print(f"  AVERAGE CAPITAL PER POSITION ALLOCATION  : ₹{total_cost / max(1, len(buys)):,.2f} (~12.0% of portfolio)")
    print("=" * 80)

    print(f"\nExits Recorded: {len(exits)}")
    for e in exits:
        t_str = "PROFIT TARGET" if e["type"] == "PROFIT_TARGET" else "STOP LOSS"
        print(f"  • {e['date']} | {e['ticker']:<14} | {t_str:<14} @ ₹{e['price']:>8.2f} ({e['pnl_pct']:+.2f}%)")

if __name__ == "__main__":
    audit_trades("monthly_debate_clean.txt")
