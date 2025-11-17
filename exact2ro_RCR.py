# EXACT2RO_RCR.py

import json
import gurobipy as gp
from gurobipy import GRB

from typing import List, Dict, Tuple

import utils
from exact2ro_utils import (
    compute_big_M_constants_exact2ro_RCR,
    add_variables_exact2ro_RCR,
    add_constraints_exact2ro_RCR,
)



def main() -> None:
    args = utils.parse_cli()

    if args.Gamma is None:
        raise ValueError(
            "Gamma (budget of uncertainty) must be provided for EXACT2RO_full_RCR. "
            "Run with e.g. `--Gamma 1`."
        )

    # ---------- paths / logging --------------------------------------------
    run_dir, gru_dir, txt_log = utils.prepare_paths(
        method="Exact2RO",
        environment="Robust",
        category="RCR",
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
    revenue = param["r"]

    m = len(M)

    # ---------- vertex enumeration for budgeted uncertainty ----------------
    z_vertices, K = utils.vertex_generation(m, args.Gamma)
    K_set = list(range(K))

    # ---------- model ------------------------------------------------------
    mdl = utils.init_model(
        name=f"MIP_EXACT2RO_full_RCR_{args.instance}_G{args.Gamma}",
        time_limit=args.time_limit,
        log_path=gru_dir / f"gurobi_EXACT2RO_full_{args.instance}_G{args.Gamma}.log",
        mip_gap=args.tolerance,
    )

    mdl.Params.LogToConsole = 0
    mdl.Params.Threads = 1

    # Add variables and constraints
    X, Y_star, t, mu, Delta, Z_star, Y, Lambda, Pi, Kappa = add_variables_exact2ro_RCR(
        mdl, N, M, K_set
    )


    add_constraints_exact2ro_RCR(
        mdl,
        X,
        Y_star,
        t,
        mu,
        Delta,
        Z_star,
        Y,
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
        revenue,
    )

    # Objective: min mu
    mdl.setObjective(mu, GRB.MINIMIZE)
    mdl.optimize()

    runtime = mdl.Runtime
    binaries = {
        "X": X,
        "Delta": Delta,
    }
    



    meta = {
        "header": (
            f"Result - instance {args.instance} (RCR, Exact2RO, Gamma={args.Gamma})\n"
            f"Facilities: {len(N)} | Customers: {len(M)}"
        ),
        "runtime": runtime,
        "limit": args.time_limit,
        "instance": args.instance,
        "n_facilities": len(N),
        "n_customers": len(M),
        "method": "Exact2RO",
        "uncertainty": f"RCR_Exact2RO_Gamma_{args.Gamma}",
        "overall_time": runtime,
        "tolerance": args.tolerance,
    }

    # # Your existing structured log (summary style)
    summary = utils.log_solution(mdl, txt_log, meta, binaries)
    utils.write_debug_summary_Exact2RO_RCR(
        mdl,
        txt_log,
        X,
        Y_star,
        Z_star,
        mu,
        t,
        N,
        M,
        opening_cost,
        ship_cost,
        revenue,
    )

    print(f"[EXACT2RO_full_RCR] run complete → {txt_log}")

if __name__ == "__main__":
    main()