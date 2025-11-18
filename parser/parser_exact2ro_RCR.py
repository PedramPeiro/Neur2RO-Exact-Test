import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# ------------ USER INPUT ------------
# Set this to the directory that contains your Exact2RO log files
LOG_DIR = Path(r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_EXACT2RO_Robust_RCR\2025111709")
OUTPUT_FILE = r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_EXACT2RO_Robust_RCR\2025111709\summary_exact2ro_RCR.xlsx"
# ------------------------------------


def parse_float_list(s: str) -> List[float]:
    """Convert '1.000000, 0.000000, -1.000000' to [1.0, 0.0, -1.0]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_search(pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
    """Regex search that returns None if there is no match."""
    return re.search(pattern, text, flags)


def parse_single_log(text: str, filename: str) -> Dict[str, Any]:
    """Parse one Exact2RO RCR log and return a dictionary of results."""

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
        "t": None,
        "mu_eq_t": None,
        "x_star": None,
        "z_worst": None,
    }

    # ---------- HEADER: instance, category, method, Gamma ----------
    m = safe_search(r"Result - instance\s+(\S+)\s*\(([^)]*)\)", text)
    if m:
        result["instance"] = m.group(1)
        inside = m.group(2)
        tokens = [tok.strip() for tok in inside.split(",")]
        if len(tokens) >= 2:
            result["category"] = tokens[0]       # e.g. 'RCR'
            result["method"] = tokens[1]         # e.g. 'Exact2RO'
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

    # ---------- LB, UB ----------
    m = safe_search(r"LB\s*:\s*([0-9.eE+\-]+)", text)
    if m:
        result["LB"] = float(m.group(1))

    m = safe_search(r"UB\s*:\s*([0-9.eE+\-]+)", text)
    if m:
        result["UB"] = float(m.group(1))

    # ---------- Time & Time limit ----------
    m = safe_search(r"Time\s*:\s*([0-9.eE+\-]+)s\s*\(limit\s*([0-9.eE+\-]+)s\)", text)
    if m:
        result["computation_time"] = float(m.group(1))
        result["time_limit"] = float(m.group(2))

    # ---------- t (worst-case) ----------
    m = safe_search(r"t \(worst-case\)\s*=\s*([0-9.eE+\-]+)", text)
    if m:
        result["t"] = float(m.group(1))

    # ---------- |mu - t| equality flag ----------
    m = safe_search(r"\|mu - t\|\s*=\s*[0-9.eE+\-]+\s*-> equal\?\s*(True|False)", text)
    if m:
        result["mu_eq_t"] = True if m.group(1) == "True" else False

    # ---------- x* and z_star* ----------
    m = safe_search(r"x\*\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    if m:
        result["x_star"] = parse_float_list(m.group(1))

    m = safe_search(r"z_star\*\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
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
        "computation_time",
        "LB",
        "UB",
        "t",
        "mu_eq_t",
        "x_star",
        "z_worst",
    ]
    df = df[cols]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Parsed {len(df)} files and wrote results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
