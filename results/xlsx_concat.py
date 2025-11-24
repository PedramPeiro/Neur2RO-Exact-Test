from pathlib import Path
import pandas as pd

# === CONFIGURE THESE ===
LOG_DIR = Path(r"D:\uni\PhD Courses\Robust Optimization\Project\Code\TrilliumResults\log_Exact2RO_Robust_IR")
OUTPUT_FILE = r"D:\uni\PhD Courses\Robust Optimization\Project\Code\TrilliumResults\log_Exact2RO_Robust_IR\summary_Exact2RO_IR_concat.xlsx"

def main():
    log_dir = LOG_DIR
    output_path = Path(OUTPUT_FILE)

    # Collect all xlsx files in the directory, excluding the output file if it exists there
    excel_files = [
        f for f in log_dir.glob("*.xlsx")
        if f.name != output_path.name
    ]

    if not excel_files:
        print(f"No .xlsx files found in {log_dir}")
        return

    dfs = []
    for f in excel_files:
        try:
            df = pd.read_excel(f)
            dfs.append(df)
        except Exception as e:
            print(f"Could not read {f}: {e}")

    if not dfs:
        print("No readable .xlsx files found.")
        return

    # Concatenate all data
    combined = pd.concat(dfs, ignore_index=True)

    # Columns defining a "record identity"
    key_cols = [
        "filename",
        "instance",
        "n_facilities",
        "m_customers",
        "category",
        "method",
        "Gamma",
    ]

    # Ensure key columns all exist
    missing_keys = [c for c in key_cols if c not in combined.columns]
    if missing_keys:
        raise ValueError(f"Missing key columns in data: {missing_keys}")

    # Ensure x_star column exists
    if "x_star" not in combined.columns:
        raise ValueError("Column 'x_star' not found in the input files.")

    # Helper column: True if x_star is non-empty, False if empty/NaN
    nonempty_mask = combined["x_star"].notna() & (combined["x_star"].astype(str).str.strip() != "")
    combined["_x_star_nonempty"] = nonempty_mask

    # Sort so rows with non-empty x_star come first within each group
    combined = combined.sort_values(by="_x_star_nonempty", ascending=False)

    # Drop duplicates on key columns, keeping the first (which will have non-empty x_star if any exist)
    deduped = combined.drop_duplicates(subset=key_cols, keep="first")

    # Drop helper column
    deduped = deduped.drop(columns=["_x_star_nonempty"])

    # Write to Excel
    deduped.to_excel(output_path, index=False)
    print(f"Summary written to: {output_path}")

if __name__ == "__main__":
    main()
