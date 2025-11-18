import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# ------------ USER INPUT ------------
# Set this to the directory that contains your *.txt log files
LOG_DIR = Path(r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_CCG_Robust_RCR\2025111709")
OUTPUT_FILE = r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_CCG_Robust_RCR\2025111709\summary_CCG_RCR.xlsx"
# ------------------------------------


def parse_float_list(s: str) -> List[float]:
    """
    Convert a comma-separated string like '1.000, 0.000, -1.000'
    into a list of floats.
    """
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_search(pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
    """
    Helper to run regex search and return None if no match.
    """
    return re.search(pattern, text, flags)


def parse_single_log(text: str, filename: str) -> Dict[str, Any]:
    """
    Parse one CCG_RCR log file's content and return a dictionary of results.
    """

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
        "x_star": None,
        "z_worst": None,
    }

    # ---------- HEADER LINE ----------
    m = safe_search(r"Result - instance\s+(\S+)\s*\(([^)]*)\)", text)
    if m:
        instance = m.group(1)
        inside = m.group(2)
        result["instance"] = instance

        # category, method, Gamma from parentheses
        tokens = [tok.strip() for tok in inside.split(",")]
        if len(tokens) >= 2:
            result["category"] = tokens[0]
            result["method"] = tokens[1]
        for tok in tokens:
            if tok.startswith("Gamma="):
                try:
                    result["Gamma"] = float(tok.split("=", 1)[1])
                except ValueError:
                    pass

    # ---------- FACILITIES & CUSTOMERS ----------
    m = safe_search(r"Facilities:\s*(\d+)\s*\|\s*Customers:\s*(\d+)", text)
    if m:
        result["n_facilities"] = int(m.group(1))
        result["m_customers"] = int(m.group(2))

    # ---------- TIME LIMIT & TOLERANCE ----------
    m = safe_search(r"Time limit per solve:\s*([0-9.eE+\-]+)s", text)
    if m:
        result["time_limit"] = float(m.group(1))

    m = safe_search(r"Tolerance \(gap\):\s*([0-9.eE+\-]+)", text)
    if m:
        result["tolerance"] = float(m.group(1))

    # ---------- FINAL SUMMARY BLOCK ----------
    if "# --- Final summary ---" in text:
        _, summary = text.split("# --- Final summary ---", 1)

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

        m = safe_search(r"Best x .*?:\s*\[(.*?)\]", summary)
        if m:
            result["x_star"] = parse_float_list(m.group(1))

    # ---------- LAST ITERATION BLOCK FOR z_worst ----------
    # Find all iteration blocks and take the one with the largest index
    iter_blocks = re.findall(
        r"Iter\s+(\d+)(.*?)(?=Iter\s+\d+|# --- Final summary ---|\Z)",
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
            # fallback if some logs are in a different encoding
            text = path.read_text(encoding="latin-1")

        rec = parse_single_log(text, filename=path.name)
        records.append(rec)

    df = pd.DataFrame(records)

    # Optional: order columns (only include columns that exist)
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
        "x_star",
        "z_worst",
    ]
    # Only select columns that exist in the DataFrame
    cols = [c for c in cols if c in df.columns]
    if cols:
        df = df[cols]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Parsed {len(df)} files and wrote results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
