import glob, re

files = sorted(glob.glob(r'd:\gravity_proj\Hedge_fund\monthly_debates\block_1\month_*.txt'))

print("ALL POST-MORTEM REFLECTIONS & SYNTHESIZED RULES IN BLOCK 1:\n")

rule_idx = 1
for fpath in files:
    with open(fpath, 'r', encoding='utf-8') as f:
        text = f.read()
    
    # Find post-mortem blocks
    matches = re.finditer(r'POST-MORTEM REFLECTION COMPLETE[\s\S]*?New Guardrail Rule Formulated by Post-Mortem Auditor:\s*([^\n\r]+(?:\n[^\n\r]+)*?)(?=\n\s*• Failing Agent|\n\s*└)', text)
    for m in matches:
        raw_rule = m.group(1).replace('│', '').replace('"', '').strip()
        failing_m = re.search(r'• Failing Agent:\s+([A-Z_]+)', text[m.end()-200:m.end()+200])
        failing = failing_m.group(1) if failing_m else "Unknown"
        acc_m = re.search(r'• Accurate Agent:\s+([A-Z_]+)', text[m.end()-200:m.end()+200])
        acc = acc_m.group(1) if acc_m else "Unknown"
        
        print(f"Rule #{rule_idx}:")
        print(f"  Statement: {raw_rule}")
        print(f"  Failing Agent: {failing} | Accurate Agent: {acc}")
        print("-" * 70)
        rule_idx += 1

