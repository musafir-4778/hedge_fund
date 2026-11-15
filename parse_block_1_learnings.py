import os
import glob
import re

files = sorted(glob.glob(r'd:\gravity_proj\Hedge_fund\monthly_debates\block_1\month_*.txt'))
print(f"Found {len(files)} monthly debate files in Block 1\n")

for fpath in files:
    fname = os.path.basename(fpath)
    with open(fpath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    m_match = re.search(r'MONTH (\d+)/(\d+) \| DATE: ([\d\-]+)', content)
    month_header = m_match.group(0) if m_match else fname
    
    # Portfolio NAV & Cash
    nav_match = re.search(r'Portfolio NAV:\s+₹([\d,]+\.?\d*)', content)
    cash_match = re.search(r'Bank Cash:\s+₹([\d,]+\.?\d*)', content)
    liquid_match = re.search(r'Liquid Fund Swept:\s+₹([\d,]+\.?\d*)', content)
    
    nav_str = nav_match.group(1) if nav_match else "N/A"
    cash_str = cash_match.group(1) if cash_match else "N/A"
    liquid_str = liquid_match.group(1) if liquid_match else "N/A"
    
    print(f"================================================================================")
    print(f" {month_header} | NAV: ₹{nav_str} | Liquid Yield: ₹{liquid_str}")
    print(f"================================================================================")
    
    # Regimes
    regime_match = re.search(r'Market Regime:\s+([A-Z_]+)', content)
    if regime_match:
        print(f"   Detected Regime: {regime_match.group(1)}")
        
    # Orders
    orders = re.findall(
        r'Ticker:\s+([A-Z0-9\.\-]+)[\s\S]*?Final Order Action:\s+([A-Z]+)[\s\S]*?Conviction:\s+([A-Z]+)[\s\S]*?Capital Allocation:\s+([\d\.]+)%[\s\S]*?Executive Rationale:\s+([^\n\r]+)',
        content
    )
    if orders:
        print("   Executive Orders:")
        for t, act, conv, alloc, rat in orders:
            print(f"     • {t:15} -> {act:4} | Conviction: {conv:6} | Allocation: {alloc:>5}% | {rat[:75]}...")
    
    # Exits
    exits = re.findall(r'(\[TARGET HIT\]|\[STOP LOSS TRIGGERED\])[\s\S]*?Ticker:\s+([A-Z0-9\.\-]+)', content)
    if exits:
        print("   Exits / Harvests:")
        for ext, t in exits:
            print(f"     • {ext} on {t}")
            
    # Rules synthesized
    rules = re.findall(r'New Guardrail Rule Formulated by Post-Mortem Auditor:\s*\"([^\"]+)\"', content)
    if rules:
        print("   🧠 Synthesized Guardrail Rules:")
        for r in rules:
            print(f"     • {r}")
            
    # Failing / Accurate agents
    failing = re.findall(r'• Failing Agent:\s+([A-Z_]+)', content)
    accurate = re.findall(r'• Accurate Agent:\s+([A-Z_]+)', content)
    if failing:
        print(f"   Scorecard Shifts: Failing={failing} vs Accurate={accurate}")
    print()

