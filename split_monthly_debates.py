import os
import re

def split_debates_by_month(src_path: str, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    month_pattern = re.compile(r"📅\s*MONTH\s*(\d+)/\d+\s*—\s*COMMITTEE SESSION DATE:\s*(\d{4}-\d{2}-\d{2})")

    with open(src_path, "r", encoding="utf-8") as f:
        content = f.read()

    # Find all match positions
    matches = list(month_pattern.finditer(content))
    if not matches:
        print("No month markers found in file.")
        return

    print(f"Found {len(matches)} monthly sessions to split.")

    created_files = []

    for i, match in enumerate(matches):
        m_num = int(match.group(1))
        m_date = match.group(2)
        start_pos = match.start()

        # Find the preceding horizontal rule if available
        rule_pos = content.rfind("━", 0, start_pos)
        if rule_pos != -1 and (start_pos - rule_pos) < 200:
            actual_start = rule_pos
            # Go back to start of line
            line_start = content.rfind("\n", 0, actual_start)
            if line_start != -1:
                actual_start = line_start + 1
        else:
            actual_start = start_pos

        end_pos = matches[i + 1].start() if (i + 1 < len(matches)) else len(content)
        # Find preceding rule for next match to avoid trailing line
        if i + 1 < len(matches):
            rule_next = content.rfind("━", actual_start, end_pos)
            if rule_next != -1 and (end_pos - rule_next) < 200:
                end_pos = rule_next
                line_start = content.rfind("\n", actual_start, end_pos)
                if line_start != -1:
                    end_pos = line_start

        month_text = content[actual_start:end_pos].strip()

        # Filename
        file_name = f"month_{m_num:02d}_{m_date.replace('-', '_')}.txt"
        file_path = os.path.join(output_dir, file_name)

        header = (
            f"================================================================================\n"
            f"  AUTONOMOUS WEALTH MANAGEMENT OFFICE — 10-AGENT COMMITTEE DEBATE\n"
            f"  MONTH {m_num:02d}/12 | DATE: {m_date}\n"
            f"  Universe: 10 Liquid Indian Equities | Max 3 Holdings | Zero Lookahead Bias\n"
            f"================================================================================\n\n"
        )

        with open(file_path, "w", encoding="utf-8") as out:
            out.write(header + month_text + "\n")

        size_kb = os.path.getsize(file_path) / 1024
        line_cnt = month_text.count("\n") + 1
        created_files.append((file_name, m_num, m_date, size_kb, line_cnt))
        print(f"Created: {file_name} ({size_kb:.1f} KB, {line_cnt:,} lines)")

    return created_files

if __name__ == "__main__":
    split_debates_by_month("monthly_debate_clean.txt", "monthly_debates")

