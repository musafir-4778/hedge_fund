import re
import os

def clean_debate_file(input_path: str, output_path: str):
    noise_patterns = [
        re.compile(r"^\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}:\d{2}"),  # Timestamped logs
        re.compile(r"^\s*->\s*\[\d+/\d+\]"),                   # -> [1/10] Technical Analyst...
        re.compile(r"chat_models\.py:\d+:"),                   # LangChain chat_models warning path
        re.compile(r"UserWarning:\s+Model"),                    # UserWarning
        re.compile(r"request\s*=\s*self\._build_request_config"),
        re.compile(r"HTTP Request:\s+POST"),                   # HTTP Request logs
        re.compile(r"AFC is enabled with max remote calls"),   # Google AFC logs
        re.compile(r"\[Logging\]\s+Saving all terminal output"),
        re.compile(r"\[Model Tiering\]"),                      # Tiering cascading logs
    ]

    clean_lines = []
    in_traceback = False
    prev_blank = False

    with open(input_path, "r", encoding="utf-8", errors="ignore") as f:
        for raw_line in f:
            # Detect traceback starts
            if "Traceback (most recent call last):" in raw_line:
                in_traceback = True
                continue
            if in_traceback:
                # Still in traceback if indented or standard exception line
                if (raw_line.startswith(" ") or raw_line.startswith("\t") or 
                    any(err in raw_line for err in ["Error", "Exception", "KeyboardInterrupt", "During task"])):
                    continue
                else:
                    in_traceback = False

            # Check noise regexes
            if any(p.search(raw_line) for p in noise_patterns):
                continue

            # Collapse consecutive empty lines
            is_blank = (raw_line.strip() == "")
            if is_blank:
                if prev_blank:
                    continue
                prev_blank = True
            else:
                prev_blank = False

            clean_lines.append(raw_line)

    with open(output_path, "w", encoding="utf-8") as out:
        out.writelines(clean_lines)

    orig_size = os.path.getsize(input_path) / 1024
    clean_size = os.path.getsize(output_path) / 1024
    orig_lines = sum(1 for _ in open(input_path, "r", encoding="utf-8", errors="ignore"))
    clean_lines_cnt = len(clean_lines)

    print(f"Cleaned debate file created successfully: {output_path}")
    print(f"Original: {orig_lines:,} lines ({orig_size:,.1f} KB)")
    print(f"Cleaned:  {clean_lines_cnt:,} lines ({clean_size:,.1f} KB)")
    print(f"Removed:  {orig_lines - clean_lines_cnt:,} lines of logging & HTTP noise ({(1 - clean_size/orig_size)*100:.1f}% reduction)")

if __name__ == "__main__":
    src = "monthly_debate_12m.txt"
    dst = "monthly_debate_clean.txt"
    clean_debate_file(src, dst)

