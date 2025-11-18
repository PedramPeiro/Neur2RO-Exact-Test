import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# ------------ USER INPUT ------------
# Directory containing the IR CCG log files
LOG_DIR = Path(r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_CCG_Robust_IR\2025111710")
OUTPUT_FILE = r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_CCG_Robust_IR\2025111710\summary_CCG_IR.xlsx"
# ------------------------------------


def parse_float_list(s: str) -> List[float]:
    """Convert '1.000, -0.000, 1.000, 1.000' -> [1.0, -0.0, 1.0, 1.0]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_search(pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
    """Regex search that returns None if there is no match."""
    return re.search(pattern, text, flags)


def parse_single_log(text: str, filename: str) -> Dict[str, Any]:
    """Parse one CCG-IR log file and return all requested info."""

    result: Dict[str, Any] = {
        "filename": filename,
        "instance": None,
        "n_facilities": None,
        "m_customers": None,
        "category": None,
        "method": None,
        "Gamma": None,
        "time_limit": None,
        "tolerance": None,
        "iterations": None,
        "computation_time": None,
        "LB": None,
        "UB": None,
        "feasibility_cuts": 0,
        "x_star": None,
        "z_worst": None,
    }

    # ---------- Header: instance, category, method, Gamma ----------
    m = safe_search(r"Result - instance\s+(\S+)\s*\(([^)]*)\)", text)
    if m:
        result["instance"] = m.group(1)
        inside = m.group(2)
        tokens = [tok.strip() for tok in inside.split(",") if tok.strip()]
        if len(tokens) >= 2:
            result["category"] = tokens[0]   # 'IR'
            result["method"] = tokens[1]     # 'CCG'
        for tok in tokens:
            if tok.startswith("Gamma="):
                try:
                    result["Gamma"] = float(tok.split("=", 1)[1])
                except ValueError:
                    pass

    # ---------- Facilities / Customers ----------
    m = safe_search(r"Facilities:\s*(\d+)\s*\|\s*Customers:\s*(\d+)", text)
    if m:
        result["n_facilities"] = int(m.group(1))
        result["m_customers"] = int(m.group(2))

    # ---------- Time limit and tolerance ----------
    m = safe_search(r"Time limit per solve:\s*([0-9.eE+\-]+)s", text)
    if m:
        result["time_limit"] = float(m.group(1))

    m = safe_search(r"Tolerance \(gap\):\s*([0-9.eE+\-]+)", text)
    if m:
        result["tolerance"] = float(m.group(1))

    # ---------- Final summary block ----------
    # Works for "# --- Final summary (IR) ---" as well
    if "# --- Final summary" in text:
        _, summary = text.split("# --- Final summary", 1)

        m = safe_search(r"Total iterations:\s*(\d+)", summary)
        if m:
            result["iterations"] = int(m.group(1))

        m = safe_search(r"Overall wall-clock time:\s*([0-9.eE+\-]+)s", summary)
        if m:
            result["computation_time"] = float(m.group(1))

        m = safe_search(r"Final LB_global:\s*([0-9.eE+\-]+)", summary)
        if m:
            result["LB"] = float(m.group(1))

        m = safe_search(r"Final UB_global:\s*([0-9.eE+\-]+)", summary)
        if m:
            result["UB"] = float(m.group(1))

        m = safe_search(r"Best x \(associated with LB_global\):\s*\[(.*?)\]",
                        summary)
        if m:
            result["x_star"] = parse_float_list(m.group(1))

    # ---------- Feasibility cuts ----------
    result["feasibility_cuts"] = text.count(
        "Adding no-good cut to exclude current x*"
    )

    # ---------- z_worst from last iteration ----------
    # Grab all Iter blocks and pick the one with largest index
    iter_blocks = re.findall(
        r"Iter\s+(\d+)(.*?)(?=Iter\s+\d+|# --- Final summary|\Z)",
        text,
        flags=re.DOTALL,
    )
    if iter_blocks:
        last_iter, last_block = max(
            ((int(num), blk) for num, blk in iter_blocks),
            key=lambda t: t[0],
        )
        m = safe_search(r"z_worst:\s*\[(.*?)\]", last_block)
        if m:
            result["z_worst"] = parse_float_list(m.group(1))

    return result


def main():
    records: List[Dict[str, Any]] = []

    for path in sorted(LOG_DIR.glob("*.txt")):
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            text = path.read_text(encoding="latin-1")

        rec = parse_single_log(text, filename=path.name)
        records.append(rec)

    df = pd.DataFrame(records)

    # Order columns
    cols = [
        "filename",
        "instance",
        "n_facilities",
        "m_customers",
        "category",
        "method",
        "Gamma",
        "time_limit",
        "tolerance",
        "iterations",
        "computation_time",
        "LB",
        "UB",
        "feasibility_cuts",
        "x_star",
        "z_worst",
    ]
    df = df[cols]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Parsed {len(df)} files and wrote results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
