# EXACT2RO_IR.py

import json
import gurobipy as gp
from gurobipy import GRB
from typing import List
import numpy as np
from subproblem_utils import solve_feasibility_SP_IR

import utils
from exact2ro_utils import (
    compute_big_M_constants_exact2ro_IR,
    add_variables_exact2ro_IR,
    add_constraints_exact2ro_IR,
)

def main() -> None:
    args = utils.parse_cli()

    if args.Gamma is None:
        raise ValueError(
            "Gamma (budget of uncertainty) must be provided for EXACT2RO_IR. "
            "Run with e.g. `--Gamma 1`."
        )

    # ---------- paths / logging --------------------------------------------
    run_dir, gru_dir, txt_log = utils.prepare_paths(
        method="Exact2RO",
        environment="Robust",
        category="IR",
        instance=args.instance,
        budget=args.Gamma,
        out_root=args.output_root,
    )

    # ---------- load instance ----------------------------------------------
    param = utils.load_instance(args.instance, args.data_dir)
    N = list(range(param["N"]))  # facilities
    M = list(range(param["M"]))  # customers

    opening_cost = param["c"]
    capacity = param["P"]
    D_bar = param["Dbar"]
    D_hat = param["Dhat"]
    ship_cost = param["d"]

    # Penalty parameters (allow both rho_D / rhoD naming)
    rho_D = param.get("rho_D", param.get("rhoD", 1e5))
    rho_P = param.get("rho_P", param.get("rhoP", 1e5))
    if rho_D is None or rho_P is None:
        raise ValueError(
            "IR instances must provide rho_D (or rhoD) and rho_P (or rhoP)."
        )

    m = len(M)

    # ---------- vertex enumeration for budgeted uncertainty ----------------
    z_vertices, K = utils.vertex_generation(m, args.Gamma)
    K_set = list(range(K))

    # ---------- model ------------------------------------------------------
    mdl = utils.init_model(
        name=f"MIP_EXACT2RO_full_IR_{args.instance}_G{args.Gamma}",
        time_limit=args.time_limit,
        log_path=gru_dir / f"gurobi_EXACT2RO_full_IR_{args.instance}_G{args.Gamma}.log",
        mip_gap=args.tolerance,
    )

    mdl.Params.LogToConsole = 0
    mdl.Params.Threads = 1

    # Add variables and constraints
    (
        X,
        Y_star,
        eta,
        t_O,
        t_F,
        Delta_O,
        Delta_F,
        Z_O_star,
        Z_F_star,
        Y_F,
        U,
        V,
        Alpha,
        Beta,
        Gamma,
        Y_O,
        Lambda,
        Pi,
        Kappa,
    ) = add_variables_exact2ro_IR(mdl, N, M, K_set, rho_D, rho_P)

    add_constraints_exact2ro_IR(
        mdl,
        X,
        Y_star,
        eta,
        t_O,
        t_F,
        Delta_O,
        Delta_F,
        Z_O_star,
        Z_F_star,
        Y_F,
        U,
        V,
        Alpha,
        Beta,
        Gamma,
        Y_O,
        Lambda,
        Pi,
        Kappa,
        N,
        M,
        K_set,
        z_vertices,
        opening_cost,
        capacity,
        D_bar,
        D_hat,
        ship_cost,
        rho_D,
        rho_P,
    )

    # Objective: min eta
    mdl.setObjective(eta, GRB.MINIMIZE)
    mdl.optimize()

    runtime = mdl.Runtime
    binaries = {
        "X": X,
        "Delta_O": Delta_O,
        "Delta_F": Delta_F,
    }

    meta = {
        "header": (
            f"Result - instance {args.instance} (IR, Exact2RO, Gamma={args.Gamma})\n"
            f"Facilities: {len(N)} | Customers: {len(M)}"
        ),
        "runtime": runtime,
        "limit": args.time_limit,
        "instance": args.instance,
        "n_facilities": len(N),
        "n_customers": len(M),
        "method": "Exact2RO",
        "uncertainty": f"IR_Exact2RO_Gamma_{args.Gamma}",
        "overall_time": runtime,
        "tolerance": args.tolerance,
    }

    summary = utils.log_solution(mdl, txt_log, meta, binaries)
    
    # ---------- Feasibility check of x* via CCG SP (IR) ----------
    # Only run if master is optimal or at least has a solution
    if mdl.SolCount > 0:
        # Extract x*
        x_star = np.array([X[i].X for i in N], dtype=float)

        # Run your IR feasibility SP
        feas_violation, z_sp, sp_mdl = solve_feasibility_SP_IR(
            x_star=x_star,
            P=np.array(capacity, dtype=float),
            D_bar=np.array(D_bar, dtype=float),
            D_hat=np.array(D_hat, dtype=float),
            Gamma=args.Gamma,
            time_limit=args.time_limit,
            tolerance=args.tolerance,
            log_path=None,  # or str(gru_dir / f"gurobi_SP_IR_{args.instance}_G{args.Gamma}.log")
        )

        # Decide feasibility: violation <= 0 ⇒ feasible
        eps_feas = 1e-6
        is_feasible = feas_violation <= eps_feas

        # Append a short, easy-to-parse summary to the txt log
        with open(txt_log, "a", encoding="utf-8") as f:
            f.write("\n\n=== IR feasibility check (CCG SP) ===\n")
            f.write(f"x_star = [{', '.join(f'{v:.4f}' for v in x_star)}]\n")
            f.write(f"violation = {feas_violation:.6e}\n")
            f.write(
                "conclusion = FEASIBLE_BY_SP\n"
                if is_feasible
                else "conclusion = INFEASIBLE_BY_SP\n"
            )
            f.write("z_SP = [" + ", ".join(f"{v:.4f}" for v in z_sp) + "]\n")
    else:
        # No solution from Exact2RO_IR → we can't even test feasibility
        with open(txt_log, "a", encoding="utf-8") as f:
            f.write("\n\n=== IR feasibility check (CCG SP) ===\n")
            f.write("conclusion = MASTER_NO_SOLUTION\n")

    # If you later want, we can add a debug summary like for RCR:
    utils.write_debug_summary_Exact2RO_IR(
        mdl,
        txt_log,
        X,
        Y_star,
        Z_O_star,
        Z_F_star,
        eta,
        t_O,
        t_F,
        Delta_O,
        Delta_F,
        N,
        M,
        K_set,
    )

    print(f"[EXACT2RO_full_IR] run complete → {txt_log}")


if __name__ == "__main__":
    main()