import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# ------------ USER INPUT ------------
# Directory containing the IR VE log files
LOG_DIR = Path(r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_VE_Robust_IR\2025111720")   # <-- change this
OUTPUT_FILE = r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_VE_Robust_IR\2025111720\summary_VE_IR.xlsx"
# ------------------------------------


def safe_search(pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
    """Regex search that returns None if no match."""
    return re.search(pattern, text, flags)


def parse_single_log(text: str, filename: str) -> Dict[str, Any]:
    """
    Parse one VE-IR log file and return all requested info.

    Expected patterns (examples):
      Result - instance IR_n4_m12_rep3 (IR, VE, Gamma=1)
      Facilities: 4 | Customers: 12
      LB : 110.49
      UB : 110.49
      Time   : 0.02s  (limit 3600s)
      X0 = 1
      X1 = 1
      X2 = 1
      ...
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
        "computation_time": None,
        "LB": None,
        "UB": None,
        "x_star": None,
    }

    # ---------- Header: instance, category, method, Gamma ----------
    m = safe_search(r"Result - instance\s+(\S+)\s*\(([^)]*)\)", text)
    if m:
        result["instance"] = m.group(1)
        inside = m.group(2)  # e.g. "IR, VE, Gamma=1"
        tokens = [tok.strip() for tok in inside.split(",") if tok.strip()]
        if len(tokens) >= 2:
            result["category"] = tokens[0]   # 'IR'
            result["method"] = tokens[1]     # 'VE'
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

    # ---------- LB / UB ----------
    m = safe_search(r"LB\s*:\s*([0-9.eE+\-]+)", text)
    if m:
        result["LB"] = float(m.group(1))

    m = safe_search(r"UB\s*:\s*([0-9.eE+\-]+)", text)
    if m:
        result["UB"] = float(m.group(1))

    # ---------- Time & Time limit ----------
    # Example: "Time   : 0.02s  (limit 3600s)"
    m = safe_search(r"Time\s*:\s*([0-9.eE+\-]+)s\s*\(limit\s*([0-9.eE+\-]+)s\)", text)
    if m:
        result["computation_time"] = float(m.group(1))
        result["time_limit"] = float(m.group(2))

    # ---------- x_star from X0, X1, ..., X{k} ----------
    # We read all "Xk = value" lines, then build x_star of length n_facilities.
    x_vals: Dict[int, float] = {}
    for match in re.finditer(r"X(\d+)\s*=\s*([0-9.eE+\-]+)", text):
        idx = int(match.group(1))
        val = float(match.group(2))
        x_vals[idx] = val

    if x_vals:
        # Length from n_facilities if we have it, otherwise from max observed index.
        if result["n_facilities"] is not None:
            n = result["n_facilities"]
        else:
            n = max(x_vals.keys()) + 1

        x_list = [0.0] * n
        for idx, val in x_vals.items():
            if 0 <= idx < n:
                x_list[idx] = val
        result["x_star"] = x_list

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

    cols = [
        "filename",
        "instance",
        "n_facilities",
        "m_customers",
        "category",
        "method",
        "Gamma",
        "time_limit",
        "computation_time",
        "LB",
        "UB",
        "x_star",
    ]
    df = df[cols]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Parsed {len(df)} files and wrote results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
