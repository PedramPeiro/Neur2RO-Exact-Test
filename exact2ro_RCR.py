# EXACT2ROfull_RCR.py

import json
import gurobipy as gp
from gurobipy import GRB

from typing import List, Dict, Tuple

import utils


def _compute_big_M_constants(
    opening_cost,
    capacity,
    D_bar,
    D_hat,
    ship_cost,
    revenue,
):
    """
    Heuristic (but reasonably tight) Big-Ms for the full MILP:

    - M_t  (used in final_RCR_blue_mu_ub)
    - M_pi (used in the kappa / pi big-M complementarity block)

    M_pi:
        We bound dual capacities π_i^{(k)} by the maximum profit margin
        over all arcs (i,j): max_{i,j} (r_ij - d_ij).

    M_t:
        Bound the worst-case absolute magnitude of the profit term:
            sum_i c_i x_i - sum_{i,j} (r_ij - d_ij) y_ij
        where 0 ≤ y_ij ≤ D_bar_j + D_hat_j.
        We use:
            |t| ≤ sum_i c_i + max_margin * sum_j (D_bar_j + D_hat_j)
        and then take M_t = sum_i c_i + max_margin * sum_j (D_bar_j + D_hat_j).
    """
    # Max profit margin over all arcs
    max_margin = 0.0
    n_fac = len(opening_cost)
    m_cust = len(D_bar)

    for i in range(n_fac):
        for j in range(m_cust):
            margin = revenue[i][j] - ship_cost[i][j]
            if margin > max_margin:
                max_margin = margin

    # Total max demand
    total_D_max = sum(D_bar[j] + D_hat[j] for j in range(m_cust))

    # Big-M for t
    sum_c = sum(opening_cost[i] for i in range(n_fac))
    M_t = sum_c + max_margin * total_D_max

    # Big-M for pi (and thus kappa)
    M_pi = max_margin if max_margin > 0 else 1.0

    return M_t, M_pi


def _add_variables(
    mdl: gp.Model,
    N: List[int],
    M: List[int],
    K: List[int],
):
    """
    Add decision variables for the full EXACT2RO-style MILP (RCR case):

    - x_i             : first-stage facility open decisions (binary)
    - mu              : objective upper bound (scalar)
    - y^*_ij          : recourse for the selected scenario z^* (continuous >= 0)
    - t               : epigraph variable, value of selected scenario (scalar)
    - delta_k         : binary selector of which vertex k is chosen
    - z^*_j           : realized uncertainty (convex combination of vertices)
    - y^{(k)}_ij      : primal recourse copies per vertex k (continuous >= 0)
    - lambda^{(k)}_j  : dual demand variables per vertex k (continuous >= 0)
    - pi^{(k)}_i      : dual capacity variables per vertex k (continuous >= 0)
    - kappa^{(k)}_i   : McCormick / complementarity helper vars (continuous >= 0)
    """
    # First-stage binary decisions
    X = mdl.addVars(N, vtype=GRB.BINARY, name="X")

    # Recourse for the selected scenario z^*
    Y_star = mdl.addVars(N, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Y_star")

    # Epigraph / scenario value
    t = mdl.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t")

    # Objective variable mu
    mu = mdl.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="mu")

    # Vertex selector delta_k
    Delta = mdl.addVars(K, vtype=GRB.BINARY, name="Delta")

    # z^*_j : convex combination of vertices
    Z_star = mdl.addVars(M, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Z_star")

    # Primal recourse copies per vertex k: y^{(k)}_{ij}
    Y = mdl.addVars(K, N, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Y")

    # Dual variables per vertex k
    Lambda = mdl.addVars(K, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Lambda")
    Pi = mdl.addVars(K, N, lb=0.0, vtype=GRB.CONTINUOUS, name="Pi")

    # kappa^{(k)}_i for big-M complementarity with pi and x
    Kappa = mdl.addVars(K, N, lb=0.0, vtype=GRB.CONTINUOUS, name="Kappa")

    return X, Y_star, t, mu, Delta, Z_star, Y, Lambda, Pi, Kappa


def _add_constraints(
    mdl: gp.Model,
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
    N: List[int],
    M: List[int],
    K: List[int],
    z_vertices,
    opening_cost,
    capacity,
    D_bar,
    D_hat,
    ship_cost,
    revenue,
):
    """
    Add constraints corresponding to the full EXACT2RO-style MILP -
    but the exact equivalent! (RCR case).
    """

    # ---------------------------------------------------------------
    # Big-M constants
    # ---------------------------------------------------------------
    M_t, M_pi = _compute_big_M_constants(
        opening_cost, capacity, D_bar, D_hat, ship_cost, revenue
    )

    # ---------------------------------------------------------------
    # Main bound & recourse for the selected scenario z^*
    # ---------------------------------------------------------------
    # (final_RCR_main_mu_lb)
    # mu >= sum_i c_i x_i - sum_{i,j} (r_ij - d_ij) y^*_ij
    mdl.addConstr(
        mu
        >= gp.quicksum(opening_cost[i] * X[i] for i in N)
        - gp.quicksum(
            (revenue[i][j] - ship_cost[i][j]) * Y_star[i, j]
            for i in N
            for j in M
        ),
        name="final_RCR_main_mu_lb",
    )

    # (final_RCR_main_demand_zstar)
    # sum_i y^*_ij <= D_bar_j + D_hat_j z^*_j
    mdl.addConstrs(
        (
            gp.quicksum(Y_star[i, j] for i in N)
            <= D_bar[j] + D_hat[j] * Z_star[j]
            for j in M
        ),
        name="final_RCR_main_demand_zstar",
    )

    # (final_RCR_main_capacity_zstar)
    # sum_j y^*_ij <= P_i x_i
    mdl.addConstrs(
        (
            gp.quicksum(Y_star[i, j] for j in M) <= capacity[i] * X[i]
            for i in N
        ),
        name="final_RCR_main_capacity_zstar",
    )

    # ---------------------------------------------------------------
    # Selector of one vertex and definition of z^*
    # ---------------------------------------------------------------
    # (final_RCR_selector) sum_k delta_k = 1, delta_k binary already in variable def
    mdl.addConstr(
        gp.quicksum(Delta[k] for k in K) == 1,
        name="final_RCR_selector",
    )

    # (final_RCR_zstar_def)
    # z^*_j = sum_k delta_k z^{(k)}_j
    mdl.addConstrs(
        (
            Z_star[j]
            == gp.quicksum(Delta[k] * z_vertices[j, k] for k in K)
            for j in M
        ),
        name="final_RCR_zstar_def",
    )

    # ---------------------------------------------------------------
    # GREEN PART: primal embed for every k
    # ---------------------------------------------------------------
    # (final_RCR_green_mu_lb)
    # t >= sum_i c_i x_i - sum_{i,j} (r_ij - d_ij) y^{(k)}_ij  ∀k
    mdl.addConstrs(
        (
            t
            >= gp.quicksum(opening_cost[i] * X[i] for i in N)
            - gp.quicksum(
                (revenue[i][j] - ship_cost[i][j]) * Y[k, i, j]
                for i in N
                for j in M
            )
            for k in K
        ),
        name="final_RCR_green_mu_lb",
    )

    # (final_RCR_green_demand)
    # sum_i y^{(k)}_ij <= D_bar_j + D_hat_j z^{(k)}_j  ∀k,j
    mdl.addConstrs(
        (
            gp.quicksum(Y[k, i, j] for i in N)
            <= D_bar[j] + D_hat[j] * z_vertices[j, k]
            for k in K
            for j in M
        ),
        name="final_RCR_green_demand",
    )

    # (final_RCR_green_capacity)
    # sum_j y^{(k)}_ij <= P_i x_i  ∀i,k
    mdl.addConstrs(
        (
            gp.quicksum(Y[k, i, j] for j in M) <= capacity[i] * X[i]
            for k in K
            for i in N
        ),
        name="final_RCR_green_capacity",
    )


    # ---------------------------------------------------------------
    # BLUE PART: dual embed for every k
    # ---------------------------------------------------------------
    # (final_RCR_blue_mu_ub)
    # t <= sum_i c_i x_i
    #      - [ sum_j (D_bar_j + D_hat_j z^{(k)}_j) λ^{(k)}_j
    #          + sum_i P_i κ_i^{(k)} ] + M_t (1 - delta_k)   ∀k
    mdl.addConstrs(
        (
            t
            <= gp.quicksum(opening_cost[i] * X[i] for i in N)
            - (
                gp.quicksum(
                    (D_bar[j] + D_hat[j] * z_vertices[j, k]) * Lambda[k, j]
                    for j in M
                )
                + gp.quicksum(capacity[i] * Kappa[k, i] for i in N)
            )
            + M_t * (1 - Delta[k])
            for k in K
        ),
        name="final_RCR_blue_mu_ub",
    )

    # (final_RCR_blue_dual_feas)
    # λ^{(k)}_j + π^{(k)}_i >= r_ij - d_ij  ∀i,j,k
    mdl.addConstrs(
        (
            Lambda[k, j] + Pi[k, i] >= (revenue[i][j] - ship_cost[i][j])
            for k in K
            for i in N
            for j in M
        ),
        name="final_RCR_blue_dual_feas",
    )


    # (final_RCR_mcc_kappa_upx)  κ_i^{(k)} <= M_pi x_i   ∀i,k
    mdl.addConstrs(
        (
            Kappa[k, i] <= M_pi * X[i]
            for k in K
            for i in N
        ),
        name="final_RCR_mcc_kappa_upx",
    )

    # (final_RCR_mcc_kappa_upp) κ_i^{(k)} <= π_i^{(k)}     ∀i,k
    mdl.addConstrs(
        (
            Kappa[k, i] <= Pi[k, i]
            for k in K
            for i in N
        ),
        name="final_RCR_mcc_kappa_upp",
    )

    # (final_RCR_mcc_kappa_low) κ_i^{(k)} >= π_i^{(k)} - M_pi (1 - x_i)   ∀i,k
    mdl.addConstrs(
        (
            Kappa[k, i] >= Pi[k, i] - M_pi * (1 - X[i])
            for k in K
            for i in N
        ),
        name="final_RCR_mcc_kappa_low",
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
    X, Y_star, t, mu, Delta, Z_star, Y, Lambda, Pi, Kappa = _add_variables(
        mdl, N, M, K_set
    )


    _add_constraints(
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