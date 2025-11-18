import re
from pathlib import Path
from typing import List, Dict, Any, Optional

import pandas as pd


# ------------ USER INPUT ------------
# Directory containing the IR Exact2RO log files
LOG_DIR = Path(r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_Exact2RO_Robust_IR\2025111710")
OUTPUT_FILE = r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\log_Exact2RO_Robust_IR\2025111710\summary_Exact2RO_IR.xlsx"
# ------------------------------------


def parse_float_list(s: str) -> List[float]:
    """Convert '1.0000, -0.0000, 1.0000, 1.0000' -> [1.0, -0.0, 1.0, 1.0]."""
    return [float(x.strip()) for x in s.split(",") if x.strip()]


def safe_search(pattern: str, text: str, flags: int = 0) -> Optional[re.Match]:
    """Regex search that returns None if no match."""
    return re.search(pattern, text, flags)


def parse_single_log(text: str, filename: str) -> Dict[str, Any]:
    """Parse one Exact2RO-IR log file and return all requested info."""

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
        "t_O": None,
        "t_F": None,
        "eta": None,
        "is_eta_ge_tO": None,
        "is_eta_ge_tF": None,
        "x_star": None,
        "z_O_star": None,
        "z_F_star": None,
        "feasible_by_SP": None,
    }

    # ---------- Header: instance, category, method, Gamma ----------
    m = safe_search(r"Result - instance\s+(\S+)\s*\(([^)]*)\)", text)
    if m:
        result["instance"] = m.group(1)
        inside = m.group(2)
        tokens = [tok.strip() for tok in inside.split(",")]
        if len(tokens) >= 2:
            result["category"] = tokens[0]   # 'IR'
            result["method"] = tokens[1]     # 'Exact2RO'
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
    m = safe_search(r"Time\s*:\s*([0-9.eE+\-]+)s\s*\(limit\s*([0-9.eE+\-]+)s\)", text)
    if m:
        result["computation_time"] = float(m.group(1))
        result["time_limit"] = float(m.group(2))

    # ---------- t_O*, t_F*, eta* ----------
    m = safe_search(r"t_O\*\s*=\s*([0-9.eE+\-]+)", text)
    if m:
        result["t_O"] = float(m.group(1))

    m = safe_search(r"t_F\*\s*=\s*([0-9.eE+\-]+)", text)
    if m:
        result["t_F"] = float(m.group(1))

    m = safe_search(r"eta\*\s*=\s*([0-9.eE+\-]+)", text)
    if m:
        result["eta"] = float(m.group(1))

    # ---------- x_star, z_O_star, z_F_star ----------
    m = safe_search(r"x_star\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    if m:
        result["x_star"] = parse_float_list(m.group(1))

    m = safe_search(r"z_O_star\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    if m:
        result["z_O_star"] = parse_float_list(m.group(1))

    m = safe_search(r"z_F_star\s*=\s*\[(.*?)\]", text, flags=re.DOTALL)
    if m:
        result["z_F_star"] = parse_float_list(m.group(1))

    # ---------- Feasible by SP? ----------
    m = safe_search(
        r"conclusion\s*=\s*(FEASIBLE_BY_SP|INFEASIBLE_BY_SP)", text
    )
    if m:
        result["feasible_by_SP"] = (m.group(1) == "FEASIBLE_BY_SP")

    # ---------- Derived checks: eta >= t_O / t_F ----------
    if result["eta"] is not None and result["t_O"] is not None:
        result["is_eta_ge_tO"] = (result["eta"] >= result["t_O"])

    if result["eta"] is not None and result["t_F"] is not None:
        result["is_eta_ge_tF"] = (result["eta"] >= result["t_F"])

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
        "t_O",
        "t_F",
        "eta",
        "is_eta_ge_tO",
        "is_eta_ge_tF",
        "x_star",
        "z_O_star",
        "z_F_star",
        "feasible_by_SP",
    ]
    df = df[cols]

    df.to_excel(OUTPUT_FILE, index=False)
    print(f"Parsed {len(df)} files and wrote results to '{OUTPUT_FILE}'.")


if __name__ == "__main__":
    main()
