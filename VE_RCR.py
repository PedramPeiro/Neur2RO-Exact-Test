import gurobipy as gp
from gurobipy import GRB

import utils


def _add_variables(mdl: gp.Model, N, M, K):
    """
    Add decision variables:

    - X_i      : here-and-now facility open decisions (binary)
    - Y_{k,i,j}: recourse decisions for each vertex k (continuous >= 0)
    - t        : epigraph variable for worst-case profit
    """
    X = mdl.addVars(N, vtype=GRB.BINARY, name="X")
    Y = mdl.addVars(K, N, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Y")
    t = mdl.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t")
    return X, Y, t


def _add_constraints(
    mdl: gp.Model, X, Y, t, N, M, K, z_vertices, opening_cost, capacity, D_bar, D_hat, ship_cost, revenue):
    """
    Add constraints for vertex-enumeration ARO with RCR:

    For each vertex k and each customer j:
        sum_i Y[k,i,j] <= D_bar[j] + D_hat[j] * z_{k,j}

    For each vertex k and each facility i:
        sum_j Y[k,i,j] <= P_i * X[i]

    Epigraph constraints for each vertex k:
        t <= - sum_i c_i X[i] + sum_{i,j} (r_{ij} - d_{ij}) * Y[k,i,j]
    """
    # Demand constraints for each vertex and customer
    mdl.addConstrs(
        (
            gp.quicksum(Y[k, i, j] for i in N)
            <= D_bar[j] + D_hat[j] * z_vertices[j, k]
            for k in K
            for j in M
        ),
        name="demand_constr",
    )

    # Capacity constraints for each vertex and facility
    mdl.addConstrs(
        (
            gp.quicksum(Y[k, i, j] for j in M) <= capacity[i] * X[i]
            for k in K
            for i in N
        ),
        name="capacity_constr",
    )

    # Epigraph constraints per vertex (worst-case profit)
    mdl.addConstrs(
        (t <= gp.quicksum(
                (revenue[i][j] - ship_cost[i][j]) * Y[k, i, j]
                for i in N
                for j in M) - gp.quicksum(opening_cost[i] * X[i] for i in N) for k in K), name = "epigraph_constr")



def main() -> None:
    args = utils.parse_cli()

    if args.Gamma is None:
        raise ValueError(
            "Gamma (budget of uncertainty) must be provided for VE_RCR. "
            "Run with e.g. `--Gamma 1`."
        )

    # ---------- paths / logging --------------------------------------------
    run_dir, gru_dir, txt_log = utils.prepare_paths(
        method="VE",
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
    capacity     = param["P"]
    D_bar        = param["Dbar"]
    D_hat        = param["Dhat"]   # adjust key name if your data uses a different one
    ship_cost    = param["d"]
    revenue      = param["r"]

    m = len(M)

    # ---------- vertex enumeration for budgeted uncertainty ----------------
    z_vertices, K = utils.vertex_generation(m, args.Gamma)
    K_set = list(range(K))
    print(len(K_set))

    # ---------- model ------------------------------------------------------
    mdl = utils.init_model(
        name=f"MIP_VE_RCR_{args.instance}_G{args.Gamma}",
        time_limit=args.time_limit,
        log_path=gru_dir / f"gurobi_{args.instance}_G{args.Gamma}.log",
        mip_gap=args.tolerance,
    )

    mdl.Params.LogToConsole = 0
    mdl.Params.Threads = 1

    X, Y, t = _add_variables(mdl, N, M, K_set)
    _add_constraints(
        mdl,
        X,
        Y,
        t,
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

    # Maximize worst-case profit t
    mdl.setObjective(t, GRB.MAXIMIZE)
    mdl.optimize()

    runtime = mdl.Runtime
    binaries = {"X": X}

    meta = {
        "header": (
            f"Result - instance {args.instance} (RCR, VE, Gamma={args.Gamma})\n"
            f"Facilities: {len(N)} | Customers: {len(M)}"
        ),
        "runtime": runtime,
        "limit": args.time_limit,
        "instance": args.instance,
        "n_facilities": len(N),
        "n_customers": len(M),
        "method": "VE",
        "uncertainty": f"RCR_VE_Gamma_{args.Gamma}",
        "overall_time": runtime,
        "tolerance": args.tolerance,
    }

    summary = utils.log_solution(mdl, txt_log, meta, binaries)
    # If/when you implement write_summary, you can enable this:
    # utils.write_summary(summary, run_dir / "summary_VE_RCR.csv")

    print(f"[VE_RCR] run complete → {txt_log}")


if __name__ == "__main__":
    main()
