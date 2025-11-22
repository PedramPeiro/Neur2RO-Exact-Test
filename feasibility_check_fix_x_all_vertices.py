#!/usr/bin/env python
# test_QF_vertices_IR.py
"""
Scan Q_F(x*, z) over all vertices of the budgeted uncertainty set
for a fixed x*, and export the results to an .xlsx file.

Uses utils.parse_cli() for instance / Gamma / data_dir / time_limit / tolerance / output_root.
Edit x_star in main() to test different first-stage solutions.
"""

from pathlib import Path
from typing import List

import gurobipy as gp
import numpy as np
import pandas as pd

import utils  # uses utils.parse_cli(), utils.load_instance, utils.vertex_generation


def build_and_solve_QF(
    x_star: np.ndarray,
    z: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    P: np.ndarray,
    rho_D: float,
    rho_P: float,
    time_limit: float,
    mip_gap: float,
) -> float:
    """
    Solve the pure feasibility recourse LP

        Q_F(x,z) = min_{y,u,v >= 0} rho_D sum_j u_j + rho_P sum_i v_i

        s.t.  sum_i y_ij + u_j >= D_bar_j + D_hat_j z_j    ∀ j
              sum_j y_ij      <= P_i x_i + v_i             ∀ i

    for a fixed x_star and z, and return the optimal objective value
    (the total feasibility violation).
    """
    n = P.shape[0]   # facilities
    m = D_bar.shape[0]  # customers
    assert x_star.shape[0] == n
    assert z.shape[0] == m

    mdl = gp.Model("QF_single_vertex")
    mdl.Params.LogToConsole = 0
    mdl.Params.TimeLimit = float(time_limit)
    mdl.Params.MIPGap = float(mip_gap)

    # Decision variables
    y = mdl.addVars(n, m, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="y")
    u = mdl.addVars(m, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="u")
    v = mdl.addVars(n, lb=0.0, vtype=gp.GRB.CONTINUOUS, name="v")

    # Demand constraints with slack
    for j in range(m):
        rhs = float(D_bar[j] + D_hat[j] * z[j])
        mdl.addConstr(
            gp.quicksum(y[i, j] for i in range(n)) + u[j] >= rhs,
            name=f"demand_{j}",
        )

    # Capacity constraints with overflow slack
    for i in range(n):
        rhs_cap = float(P[i] * x_star[i])
        mdl.addConstr(
            gp.quicksum(y[i, j] for j in range(m)) <= rhs_cap + v[i],
            name=f"capacity_{i}",
        )

    # Objective: pure feasibility penalties
    obj = (
        rho_D * gp.quicksum(u[j] for j in range(m))
        + rho_P * gp.quicksum(v[i] for i in range(n))
    )
    mdl.setObjective(obj, gp.GRB.MINIMIZE)

    mdl.optimize()

    if mdl.Status not in {gp.GRB.OPTIMAL, gp.GRB.TIME_LIMIT}:
        # If something weird happens, flag with NaN
        return float("nan")

    return float(mdl.ObjVal)


def format_vector_as_list_cell(vec: np.ndarray, tol: float = 5e-4) -> str:
    """
    Format a 1D vector as a single string like:
        [1, 0, 1, 0.5]

    Rules:
      - Only one decimal place for non-integers.
      - If value is (numerically) 0 or ±1, print as integer (no decimal).
    """
    formatted = []
    for v in vec:
        # Clean tiny numerical noise
        if abs(v) < tol:
            v_clean = 0.0
        else:
            v_clean = v

        if abs(v_clean - 1.0) < tol:
            formatted.append("1")
        elif abs(v_clean + 1.0) < tol:
            formatted.append("-1")
        elif abs(v_clean) < tol:
            formatted.append("0")
        else:
            formatted.append(f"{v_clean:.1f}")
    return "[" + ", ".join(formatted) + "]"


def main() -> None:
    # Use your shared CLI parser
    args = utils.parse_cli()

    # -------- load instance --------
    param = utils.load_instance(args.instance, args.data_dir)

    N = list(range(param["N"]))  # facilities
    M = list(range(param["M"]))  # customers

    P = np.array(param["P"], dtype=float)
    D_bar = np.array(param["Dbar"], dtype=float)
    D_hat = np.array(param["Dhat"], dtype=float)

    # penalty parameters
    rho_D = float(param.get("rho_D", param.get("rhoD", 1.0)))
    rho_P = float(param.get("rho_P", param.get("rhoP", 1.0)))

    n = len(N)
    m = len(M)

    # -------- choose x_star (EDIT THIS FOR YOUR TESTS) --------
    # Example: all facilities open
    x_star = np.array([1,1,1,0])
    # If you want a specific x* from a previous run, just hard-code it here, e.g.:
    # x_star = np.array([1, 1, 1, 1], dtype=float)

    # -------- vertex enumeration --------
    z_vertices, K = utils.vertex_generation(m, args.Gamma)
    # z_vertices shape: (m, K)

    rows: List[dict] = []

    for k in range(K):
        z_k = z_vertices[:, k].astype(float)

        qf_val = build_and_solve_QF(
            x_star=x_star,
            z=z_k,
            D_bar=D_bar,
            D_hat=D_hat,
            P=P,
            rho_D=rho_D,
            rho_P=rho_P,
            time_limit=args.time_limit,
            mip_gap=args.tolerance,
        )

        row = {
            "k": k,
            "QF_value": qf_val,
            "x": format_vector_as_list_cell(x_star),
            "z": format_vector_as_list_cell(z_k),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    # -------- write Excel --------
    out_root = Path(args.output_root)
    out_dir = out_root / "QF_vertex_scans"
    out_dir.mkdir(parents=True, exist_ok=True)

    out_path = out_dir / f"QF_scan_{args.instance}_G{args.Gamma}.xlsx"
    df.to_excel(out_path, index=False)

    print(f"[QF scan] x* = {format_vector_as_list_cell(x_star)}")
    print(f"[QF scan] Wrote results to: {out_path}")


if __name__ == "__main__":
    main()
