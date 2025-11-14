# dataread.py

from pathlib import Path
from typing import Dict, Any, List, Optional
import numpy as np


def read_instance(path: str) -> Dict[str, Any]:
    """
    Read a robust facility-location instance (RCR or IR) from a .txt file.

    The function:
      - Detects instance type from the first line: '# Instance type: RCR' or '... IR'
      - Reads n and m from the lines starting with 'n ' and 'm '
      - Parses:
          FACILITIES  # columns: i x y c_i P_i
          CUSTOMERS   # columns: j x y Dbar_j Dhat_j
          D_MATRIX    # n rows, each with m entries (per-unit shipping costs d_ij)
          R_MATRIX    # n rows, each with m entries (revenues r_ij)   [RCR only]

    Returns a dictionary 'data' with keys:
      - instance_type : 'RCR' or 'IR'
      - n, m          : int
      - c             : list of length n (opening costs c_i)
      - P             : list of length n (capacities P_i)
      - Dbar          : list of length m (nominal demands Dbar_j)
      - Dhat          : list of length m (perturbations Dhat_j)
      - d             : np.ndarray shape (n, m) (shipping costs d_ij)
      - r             : np.ndarray shape (n, m) or None (revenues r_ij)
    """
    path_obj: Path = Path(path)
    if not path_obj.is_file():
        raise FileNotFoundError(f"Instance file not found: {path}")

    with path_obj.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    # Helper to strip trailing newline but keep everything else
    lines = [line.rstrip("\n") for line in lines]

    # -------------------------
    # 1. Instance type from first line
    # -------------------------
    if not lines:
        raise ValueError("Empty instance file.")

    first_line = lines[0]
    if "RCR" in first_line:
        instance_type = "RCR"
    elif "IR" in first_line:
        instance_type = "IR"
    else:
        raise ValueError(
            "Could not determine instance type from first line. "
            f"Expected 'RCR' or 'IR' in: {first_line!r}"
        )

    # -------------------------
    # 2. Find n and m
    # -------------------------
    n = None
    m = None
    for line in lines:
        stripped = line.strip()
        if stripped.startswith("n "):
            parts = stripped.split()
            if len(parts) >= 2:
                n = int(float(parts[1]))
        elif stripped.startswith("m "):
            parts = stripped.split()
            if len(parts) >= 2:
                m = int(float(parts[1]))
        if n is not None and m is not None:
            break

    if n is None or m is None:
        raise ValueError("Could not parse 'n' and 'm' from the file.")

    # -------------------------
    # 3. Parse FACILITIES section
    # -------------------------
    # Find index of the FACILITIES header
    facilities_header = "FACILITIES  # columns: i x y c_i P_i"
    fac_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("FACILITIES"):
            fac_idx = idx
            break
    if fac_idx is None:
        raise ValueError("FACILITIES header not found in file.")

    # The next n lines after fac_idx are the facility rows
    c_list: List[float] = []
    P_list: List[float] = []
    facilities_start = fac_idx + 1

    row_count = 0
    i = facilities_start
    while row_count < n and i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue  # skip empty lines
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Facility line has fewer than 5 entries: {line!r}")
        # parts: [i, x, y, c_i, P_i]
        try:
            c_i = float(parts[3])
            P_i = float(parts[4])
        except Exception as e:
            raise ValueError(f"Could not parse facility line: {line!r}") from e
        c_list.append(c_i)
        P_list.append(P_i)
        row_count += 1

    if row_count != n:
        raise ValueError(
            f"Expected {n} facility lines after FACILITIES, got {row_count}."
        )

    # -------------------------
    # 4. Parse CUSTOMERS section
    # -------------------------
    customers_header = "CUSTOMERS  # columns: j x y Dbar_j Dhat_j"
    cust_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("CUSTOMERS"):
            cust_idx = idx
            break
    if cust_idx is None:
        raise ValueError("CUSTOMERS header not found in file.")

    Dbar_list: List[float] = []
    Dhat_list: List[float] = []
    customers_start = cust_idx + 1

    row_count = 0
    i = customers_start
    while row_count < m and i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue  # skip empty lines
        parts = line.split()
        if len(parts) < 5:
            raise ValueError(f"Customer line has fewer than 5 entries: {line!r}")
        # parts: [j, x, y, Dbar_j, Dhat_j]
        try:
            Dbar_j = float(parts[3])
            Dhat_j = float(parts[4])
        except Exception as e:
            raise ValueError(f"Could not parse customer line: {line!r}") from e
        Dbar_list.append(Dbar_j)
        Dhat_list.append(Dhat_j)
        row_count += 1

    if row_count != m:
        raise ValueError(
            f"Expected {m} customer lines after CUSTOMERS, got {row_count}."
        )

    # -------------------------
    # 5. Parse D_MATRIX section (shipping costs d_ij)
    # -------------------------
    dmatrix_header = "D_MATRIX  # n rows, each with m entries (per-unit shipping costs d_ij)"
    dmat_idx = None
    for idx, line in enumerate(lines):
        if line.strip().startswith("D_MATRIX"):
            dmat_idx = idx
            break
    if dmat_idx is None:
        raise ValueError("D_MATRIX header not found in file.")

    # After the header, the next non-empty line is "n m"
    d_dims_line_idx = dmat_idx + 1
    while d_dims_line_idx < len(lines) and not lines[d_dims_line_idx].strip():
        d_dims_line_idx += 1
    if d_dims_line_idx >= len(lines):
        raise ValueError("Unexpected end of file after D_MATRIX header.")

    dim_line = lines[d_dims_line_idx].strip()
    dim_parts = dim_line.split()
    if len(dim_parts) < 2:
        raise ValueError(f"Expected 'n m' after D_MATRIX, got: {dim_line!r}")
    n_d = int(float(dim_parts[0]))
    m_d = int(float(dim_parts[1]))
    if n_d != n or m_d != m:
        raise ValueError(
            f"D_MATRIX dimensions ({n_d}, {m_d}) do not match n={n}, m={m}."
        )

    # Now read n rows of m entries
    d_data_start = d_dims_line_idx + 1
    d_rows: List[List[float]] = []
    row_count = 0
    i = d_data_start
    while row_count < n and i < len(lines):
        line = lines[i].strip()
        i += 1
        if not line:
            continue
        parts = line.split()
        if len(parts) != m:
            raise ValueError(
                f"D_MATRIX row has {len(parts)} entries, expected {m}: {line!r}"
            )
        try:
            row_vals = [float(v) for v in parts]
        except Exception as e:
            raise ValueError(f"Could not parse D_MATRIX row: {line!r}") from e
        d_rows.append(row_vals)
        row_count += 1

    if row_count != n:
        raise ValueError(
            f"Expected {n} rows in D_MATRIX, got {row_count}."
        )

    d_array = np.array(d_rows, dtype=float)

    # -------------------------
    # 6. Parse R_MATRIX section (revenues r_ij) if RCR
    # -------------------------
    r_array: Optional[np.ndarray] = None
    if instance_type == "RCR":
        rmatrix_header = "R_MATRIX  # n rows, each with m entries (revenues r_ij)"
        rmat_idx = None
        for idx, line in enumerate(lines):
            if line.strip().startswith("R_MATRIX"):
                rmat_idx = idx
                break
        if rmat_idx is None:
            raise ValueError("R_MATRIX header not found in RCR instance file.")

        # After the header, the next non-empty line is "n m"
        r_dims_line_idx = rmat_idx + 1
        while r_dims_line_idx < len(lines) and not lines[r_dims_line_idx].strip():
            r_dims_line_idx += 1
        if r_dims_line_idx >= len(lines):
            raise ValueError("Unexpected end of file after R_MATRIX header.")

        r_dim_line = lines[r_dims_line_idx].strip()
        r_dim_parts = r_dim_line.split()
        if len(r_dim_parts) < 2:
            raise ValueError(f"Expected 'n m' after R_MATRIX, got: {r_dim_line!r}")
        n_r = int(float(r_dim_parts[0]))
        m_r = int(float(r_dim_parts[1]))
        if n_r != n or m_r != m:
            raise ValueError(
                f"R_MATRIX dimensions ({n_r}, {m_r}) do not match n={n}, m={m}."
            )

        # Now read n rows of m entries
        r_data_start = r_dims_line_idx + 1
        r_rows: List[List[float]] = []
        row_count = 0
        i = r_data_start
        while row_count < n and i < len(lines):
            line = lines[i].strip()
            i += 1
            if not line:
                continue
            parts = line.split()
            if len(parts) != m:
                raise ValueError(
                    f"R_MATRIX row has {len(parts)} entries, expected {m}: {line!r}"
                )
            try:
                row_vals = [float(v) for v in parts]
            except Exception as e:
                raise ValueError(f"Could not parse R_MATRIX row: {line!r}") from e
            r_rows.append(row_vals)
            row_count += 1

        if row_count != n:
            raise ValueError(
                f"Expected {n} rows in R_MATRIX, got {row_count}."
            )

        r_array = np.array(r_rows, dtype=float)

    # -------------------------
    # 7. Build data dictionary
    # -------------------------
    data: Dict[str, Any] = {
        "instance_type": instance_type,
        "N": n,
        "M": m,
        "c": c_list,         # opening cost c_i
        "P": P_list,         # capacity P_i
        "Dbar": Dbar_list,   # nominal demand Dbar_j
        "Dhat": Dhat_list,   # demand perturbation Dhat_j
        "d": d_array,        # shipping costs d_ij
        "r": r_array,        # revenues r_ij (RCR) or None (IR)
    }

    return data

# data = read_instance("../data/instances_RCR/RCR_n5_m10_rep1_seed20356213.txt")
# print(data["instance_type"])
