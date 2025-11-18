import pandas as pd
from pathlib import Path


# -----------------------------------------------------------
# DEFAULT INPUT/OUTPUT PATHS (YOUR PATHS)
# -----------------------------------------------------------
DEFAULT_EXACT2RO_PATH = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\summary_exact2ro_RCR.xlsx"
)
DEFAULT_CCG_PATH = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\summary_CCG_RCR.xlsx"
)
DEFAULT_OUTPUT_DIR = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results"
)
DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "joined_RCR_CCG_Exact2RO.xlsx"
# -----------------------------------------------------------



def _compute_objective(df: pd.DataFrame, lb_col: str = "LB", ub_col: str = "UB") -> pd.Series:
    """
    Determine objective value from UB if available, otherwise LB.
    """
    ub = df[ub_col]
    lb = df[lb_col]
    return ub.where(~ub.isna(), lb)



def join_rcr_ccg_and_exact2ro(
    ccg_path: Path = DEFAULT_CCG_PATH,
    exact2ro_path: Path = DEFAULT_EXACT2RO_PATH,
    output_path: Path = DEFAULT_OUTPUT_FILE,
) -> None:

    print(f"Loading CCG file:\n  {ccg_path}")
    print(f"Loading Exact2RO file:\n  {exact2ro_path}")

    df_ccg = pd.read_excel(ccg_path)
    df_exact = pd.read_excel(exact2ro_path)

    # --- Build join keys {instance}_{Gamma} ---
    df_ccg["join_key"] = df_ccg["instance"].astype(str) + "_" + df_ccg["Gamma"].astype(str)
    df_exact["join_key"] = df_exact["instance"].astype(str) + "_" + df_exact["Gamma"].astype(str)

    # --- Merge ---
    merged = pd.merge(
        df_ccg,
        df_exact,
        on="join_key",
        how="inner",
        suffixes=("_ccg", "_exact2ro")
    )

    # --- Compute objective values ---
    obj_star_ccg = _compute_objective(merged, "LB_ccg", "UB_ccg")
    mu_exact2ro = _compute_objective(merged, "LB_exact2ro", "UB_exact2ro")

    # --- Final combined dataframe ---
    result = pd.DataFrame({
        "filename": merged["filename_ccg"],
        "instance": merged["instance_ccg"],
        "n_facilities": merged["n_facilities_ccg"],
        "m_customers": merged["m_customers_ccg"],
        "category": merged["category_ccg"],
        "Gamma": merged["Gamma_ccg"],
        "time_limit": merged["time_limit_ccg"],
        "tolerance": merged.get("tolerance", pd.NA),

        # CCG part
        "obj_star_ccg": obj_star_ccg,
        "x_star_ccg": merged["x_star_ccg"],
        "z_worst_ccg": merged["z_worst_ccg"],

        # Exact2RO part
        "mu_exact2ro": mu_exact2ro,
        "mu_eq_t_exact2ro": merged["mu_eq_t"],
        "x_star_exact2ro": merged["x_star_exact2ro"],
        "z_worst_exact2ro": merged["z_worst_exact2ro"],
    })

    # Save output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False)

    print("\n✔ JOIN COMPLETE!")
    print(f"Saved to:\n  {output_path}\n")



if __name__ == "__main__":
    # Run with YOUR default paths
    join_rcr_ccg_and_exact2ro()
