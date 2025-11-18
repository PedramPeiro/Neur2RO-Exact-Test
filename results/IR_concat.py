import pandas as pd
from pathlib import Path


# -----------------------------------------------------------
# DEFAULT INPUT & OUTPUT PATHS (YOUR PATHS)
# -----------------------------------------------------------
DEFAULT_CCG_IR_PATH = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\summary_CCG_IR.xlsx"
)

DEFAULT_EXACT2RO_IR_PATH = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results\summary_Exact2RO_IR.xlsx"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\Local Results"
)

DEFAULT_OUTPUT_FILE = DEFAULT_OUTPUT_DIR / "joined_IR_CCG_Exact2RO.xlsx"
# -----------------------------------------------------------


def _compute_objective(df: pd.DataFrame, lb_col: str = "LB", ub_col: str = "UB") -> pd.Series:
    """Return UB if available, otherwise LB."""
    ub = df[ub_col]
    lb = df[lb_col]
    return ub.where(~ub.isna(), lb)


def join_ir_ccg_and_exact2ro(
    ccg_path: Path = DEFAULT_CCG_IR_PATH,
    exact2ro_path: Path = DEFAULT_EXACT2RO_IR_PATH,
    output_path: Path = DEFAULT_OUTPUT_FILE,
) -> None:

    print(f"Loading CCG IR file:\n  {ccg_path}")
    print(f"Loading Exact2RO IR file:\n  {exact2ro_path}")

    df_ccg = pd.read_excel(ccg_path)
    df_exact = pd.read_excel(exact2ro_path)

    # --- Build join key {instance}_{Gamma} ---
    df_ccg["join_key"] = df_ccg["instance"].astype(str) + "_" + df_ccg["Gamma"].astype(str)
    df_exact["join_key"] = df_exact["instance"].astype(str) + "_" + df_exact["Gamma"].astype(str)

    # --- Merge ---
    merged = pd.merge(
        df_ccg,
        df_exact,
        on="join_key",
        how="inner",
        suffixes=("_ccg", "_exact2ro"),
    )

    # --- CCG IR objective ---
    obj_star_ccg = _compute_objective(merged, "LB_ccg", "UB_ccg")

    # --- Build final dataframe ---
    result = pd.DataFrame({
        # Shared info (shown once)
        "filename":     merged["filename_ccg"],
        "instance":     merged["instance_ccg"],
        "n_facilities": merged["n_facilities_ccg"],
        "m_customers":  merged["m_customers_ccg"],
        "category":     merged["category_ccg"],
        "Gamma":        merged["Gamma_ccg"],
        "time_limit":   merged["time_limit_ccg"],
        "tolerance":    merged["tolerance"],

        # CCG IR part
        "obj_star_ccg":     obj_star_ccg,
        "feasibility_cuts": merged["feasibility_cuts"],
        "x_star_ccg":       merged["x_star_ccg"],
        # IMPORTANT: z_worst only exists in CCG file, so column name is just "z_worst"
        "z_worst_ccg":      merged["z_worst"],

        # Exact2RO IR part
        "eta_exact2ro":           merged["eta"],
        "t_O_exact2ro":           merged["t_O"],
        "t_F_exact2ro":           merged["t_F"],
        "is_eta_ge_tO":           merged["is_eta_ge_tO"],
        "is_eta_ge_tF":           merged["is_eta_ge_tF"],
        "x_star_exact2ro":        merged["x_star_exact2ro"],
        "z_O_star_exact2ro":      merged["z_O_star"],
        "z_F_star_exact2ro":      merged["z_F_star"],
        "feasible_by_SP_exact2ro": merged["feasible_by_SP"],
    })

    # --- Save Excel ---
    output_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_excel(output_path, index=False)

    print("\n✔ IR JOIN COMPLETE!")
    print(f"Saved to:\n  {output_path}\n")


if __name__ == "__main__":
    # Automatically runs with your paths
    join_ir_ccg_and_exact2ro()
