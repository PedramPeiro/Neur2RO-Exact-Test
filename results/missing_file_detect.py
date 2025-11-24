import pandas as pd
from pathlib import Path


# -----------------------------------------------------------
# DEFAULT INPUT & OUTPUT PATHS (EDIT XLSX AS NEEDED)
# -----------------------------------------------------------
DEFAULT_XLSX_PATH = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\TrilliumResults\log_Exact2RO_Robust_RCR\summary_exact2ro_RCR1.xlsx"
)

DEFAULT_OUTPUT_DIR = Path(
    r"D:\uni\PhD Courses\Robust Optimization\Project\Code\TrilliumResults"
)
# -----------------------------------------------------------


# ------------------------------
# Expected combinations for IR
# ------------------------------
def generate_expected_ir_combinations():
    expected = []

    # IR_n4_m12_rep{0..29}, gammas = 1 2 3 4 11 12
    for rep in range(30):
        inst = f"IR_n4_m12_rep{rep}"
        for gamma in [1, 2, 3, 4, 11, 12]:
            expected.append((inst, gamma))

    # IR_n5_m10_rep{0..29}, gammas = 1..10
    for rep in range(30):
        inst = f"IR_n5_m10_rep{rep}"
        for gamma in range(1, 11):
            expected.append((inst, gamma))

    # IR_n5_m15_rep{0..29}, gammas = 1 2 3 4 15
    for rep in range(30):
        inst = f"IR_n5_m15_rep{rep}"
        for gamma in [1, 2, 3, 4, 15]:
            expected.append((inst, gamma))

    # IR_n10_m20_rep{0..29}, gammas = 1 2 3
    for rep in range(30):
        inst = f"IR_n10_m20_rep{rep}"
        for gamma in [1, 2, 3]:
            expected.append((inst, gamma))

    return expected


# ------------------------------
# Expected combinations for RCR
# SAME STRUCTURE AS IR, just the prefix changes
# ------------------------------
def generate_expected_rcr_combinations():
    expected = []

    for rep in range(30):
        inst = f"RCR_n4_m12_rep{rep}"
        for gamma in [1, 2, 3, 4, 11, 12]:
            expected.append((inst, gamma))

    for rep in range(30):
        inst = f"RCR_n5_m10_rep{rep}"
        for gamma in range(1, 11):
            expected.append((inst, gamma))

    for rep in range(30):
        inst = f"RCR_n5_m15_rep{rep}"
        for gamma in [1, 2, 3, 4, 15]:
            expected.append((inst, gamma))

    for rep in range(30):
        inst = f"RCR_n10_m20_rep{rep}"
        for gamma in [1, 2, 3]:
            expected.append((inst, gamma))

    return expected



# -----------------------------------------------------------
# Main check function
# -----------------------------------------------------------
def check_missing_combinations(
    xlsx_path: Path = DEFAULT_XLSX_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR
) -> Path:

    xlsx_path = Path(xlsx_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_excel(xlsx_path)

    # --- Determine method name for naming output file ---
    if "method" in df.columns and df["method"].notna().nunique() == 1:
        method_name = str(df["method"].dropna().iloc[0])
    else:
        method_name = xlsx_path.stem

    safe_method = method_name.replace(" ", "_")

    # --- Determine problem type IR or RCR ---
    # We check the first instance name prefix
    sample_inst = str(df["instance"].dropna().iloc[0])

    if sample_inst.startswith("IR_"):
        expected = generate_expected_ir_combinations()
        problem_type = "IR"
    elif sample_inst.startswith("RCR_"):
        expected = generate_expected_rcr_combinations()
        problem_type = "RCR"
    else:
        raise ValueError("Could not determine IR vs RCR based on instance names.")

    txt_path = output_dir / f"{safe_method}_missing_combinations_{problem_type}.txt"

    # --- Ensure necessary columns exist ---
    if "instance" not in df.columns or "Gamma" not in df.columns:
        raise ValueError("Input file must contain 'instance' and 'Gamma' columns.")

    if "x_star" not in df.columns:
        raise ValueError("Input file must contain an 'x_star' column.")

    # Convert Gamma safely to int
    df["Gamma_int"] = df["Gamma"].astype("Int64")

    # Build set of present combinations
    present_keys = set(
        (str(inst), int(gamma))
        for inst, gamma in zip(df["instance"], df["Gamma_int"])
        if pd.notna(gamma)
    )

    missing_entries = []

    for inst, gamma in expected:
        key = (inst, gamma)

        if key not in present_keys:
            # missing entirely
            missing_entries.append(f"{inst} {gamma}")
        else:
            # check if incomplete: x_star NaN
            mask = (df["instance"] == inst) & (df["Gamma_int"] == gamma)
            subset = df.loc[mask, "x_star"]

            if pd.notna(subset).any():
                continue
            else:
                missing_entries.append(f"{inst} {gamma}")

    # --- Write TXT file ---
    with txt_path.open("w", encoding="utf-8") as f:
        for line in missing_entries:
            f.write(line + "\n")

    print(f"Detected {problem_type} file.")
    print(f"Expected combinations: {len(expected)}")
    print(f"Missing/incomplete combinations: {len(missing_entries)}")
    print(f"Written to: {txt_path}")

    return txt_path



# -----------------------------------------------------------
# Run automatically using defaults
# -----------------------------------------------------------
if __name__ == "__main__":
    check_missing_combinations()
