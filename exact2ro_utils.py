import json
import gurobipy as gp
from gurobipy import GRB

from typing import List, Dict, Tuple

import utils


def compute_big_M_constants_exact2ro_RCR(
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


def add_variables_exact2ro_RCR(
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


def add_constraints_exact2ro_RCR(
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
    M_t, M_pi = compute_big_M_constants_exact2ro_RCR(
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
    
    
    
    
    


def compute_big_M_constants_exact2ro_IR(
    opening_cost,
    capacity,
    D_bar,
    D_hat,
    ship_cost,
    z_vertices,
    rho_D: float,
    rho_P: float,
):
    """
    Compute Big-M constants for the IR Exact2RO MILP.

    We need:
      - M_opt : used in the optimality block t^O upper bounds
      - M_feas: used in the feasibility block t^F upper bounds
      - M_pi  : used in the McCormick linearization of kappa / pi / x
                in the optimality block.

    Heuristic derivation (but reasonably tight):

    Let:
      D_high_j = max possible demand RHS at customer j
               = max_k ( D_bar_j + D_hat_j * z_{j,k} ).
    Let:
      sum_D_high = sum_j D_high_j
      n_fac      = number of facilities
      max_d      = max_{i,j} d_ij
      sum_c      = sum_i c_i

    For the optimality block primal:
        t^O >= sum_i c_i x_i + sum_{i,j} d_ij y^O_{ij}
      with 0 <= y^O_{ij} <= D_high_j and sum_i y^O_{ij} >= D_high_j (roughly).
      A coarse upper bound:
        t^O <= sum_c + max_d * n_fac * sum_D_high

    For the feasibility block primal:
        t^F >= sum_i c_i x_i
              + sum_{i,j} d_ij y^F_{ij}
              + rho_D sum_j u_j
              + rho_P sum_i v_i
      With:
        0 <= y^F_{ij} <= D_high_j,
        0 <= u_j <= D_high_j,
        0 <= v_i <= sum_D_high
      we get:
        t^F <= sum_c
               + max_d * n_fac * sum_D_high
               + rho_D * sum_D_high
               + rho_P * n_fac * sum_D_high

    For M_pi, we use:
        M_pi = max_d
    which is a standard bound for dual capacity-like variables.
    """
    n_fac = len(opening_cost)
    m_cust = len(D_bar)

    # max d_ij
    max_d = 0.0
    for i in range(n_fac):
        for j in range(m_cust):
            if ship_cost[i][j] > max_d:
                max_d = ship_cost[i][j]

    # D_high_j based on vertices
    D_high = []
    for j in range(m_cust):
        max_z_j = max(z_vertices[j, k] for k in range(z_vertices.shape[1]))
        D_high_j = D_bar[j] + D_hat[j] * max_z_j
        D_high.append(D_high_j)

    sum_D_high = sum(D_high)
    sum_c = sum(opening_cost[i] for i in range(n_fac))

    # Upper bound for t^O
    M_opt = sum_c + max_d * n_fac * sum_D_high

    # Upper bound for t^F (includes slack penalties)
    M_feas = (
        sum_c
        + max_d * n_fac * sum_D_high
        + rho_D * sum_D_high
        + rho_P * n_fac * sum_D_high
    )

    # Big-M for kappa / pi linkage
    M_pi = max_d if max_d > 0 else 1.0

    return M_opt, M_feas, M_pi


def add_variables_exact2ro_IR(
    mdl: gp.Model,
    N: List[int],
    M: List[int],
    K: List[int],
    rho_D: float,
    rho_P: float,
):
    """
    Add decision variables for the Exact2RO-style MILP (IR case).

    Global / first-stage:
      - X_i          : binary facility open decisions
      - eta          : global objective (worst-case cost)
      - Y_star_ij    : recourse for the selected scenarios (must satisfy both z^O* and z^F*)
      - V_star_i     : capacity slack in the selected scenario
      - Delta_O_k    : selector for optimality vertex
      - Delta_F_k    : selector for feasibility vertex
      - Z_O_star_j   : selected optimality scenario
      - Z_F_star_j   : selected feasibility scenario
      - t_O, t_F     : worst-case values for optimality and feasibility recourse

    Per-vertex (feasibility block, superscript F):
      - Y_F[k,i,j]   : primal flows
      - U[k,j]       : demand shortfall slacks
      - V[k,i]       : capacity overflow slacks
      - Alpha[k,j]   : dual demand vars (bounded by rho_D)
      - Beta[k,i]    : dual capacity vars (bounded by rho_P)
      - Gamma[k,i]   : McCormick helper for Beta & X (capacity dual & x)

    Per-vertex (optimality block, superscript O):
      - Y_O[k,i,j]   : primal flows
      - Lambda[k,j]  : dual demand vars
      - Pi[k,i]      : dual capacity vars
      - Kappa[k,i]   : McCormick helper for Pi & X
    """
    # First-stage binary
    X = mdl.addVars(N, vtype=GRB.BINARY, name="X")

    # Selected scenario recourse and capacity slack
    Y_star = mdl.addVars(N, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Y_star")

    # Global objective and block values
    eta = mdl.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="eta")
    t_O = mdl.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t_O")
    t_F = mdl.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="t_F")

    # Vertex selectors
    Delta_O = mdl.addVars(K, vtype=GRB.BINARY, name="Delta_O")
    Delta_F = mdl.addVars(K, vtype=GRB.BINARY, name="Delta_F")

    # Selected z vectors (allow {-1,0,1})
    Z_O_star = mdl.addVars(M, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Z_O_star")
    Z_F_star = mdl.addVars(M, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="Z_F_star")

    # Feasibility primal vars per vertex
    Y_F = mdl.addVars(K, N, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Y_F")
    U = mdl.addVars(K, M, lb=0.0, vtype=GRB.CONTINUOUS, name="U")
    V = mdl.addVars(K, N, lb=0.0, vtype=GRB.CONTINUOUS, name="V")

    # Feasibility dual vars per vertex
    Alpha = mdl.addVars(
        K, M, lb=0.0, ub=rho_D, vtype=GRB.CONTINUOUS, name="Alpha"
    )
    Beta = mdl.addVars(
        K, N, lb=0.0, ub=rho_P, vtype=GRB.CONTINUOUS, name="Beta"
    )
    Gamma = mdl.addVars(K, N, lb=0.0, vtype=GRB.CONTINUOUS, name="Gamma")

    # Optimality primal vars per vertex
    Y_O = mdl.addVars(K, N, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Y_O")

    # Optimality dual vars per vertex
    Lambda = mdl.addVars(K, M, lb=0.0, vtype=GRB.CONTINUOUS, name="Lambda")
    Pi = mdl.addVars(K, N, lb=0.0, vtype=GRB.CONTINUOUS, name="Pi")
    Kappa = mdl.addVars(K, N, lb=0.0, vtype=GRB.CONTINUOUS, name="Kappa")

    return X, Y_star, eta, t_O, t_F, Delta_O, Delta_F, Z_O_star, Z_F_star, Y_F, U, V, Alpha, Beta, Gamma, Y_O, Lambda, Pi, Kappa


def add_constraints_exact2ro_IR(
    mdl: gp.Model, X, Y_star, eta, t_O, t_F, Delta_O, Delta_F, Z_O_star, Z_F_star, Y_F, U, V,
    Alpha,
    Beta,
    Gamma,
    Y_O,
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
    rho_D: float,
    rho_P: float,
):
    """
    Add constraints for the Exact2RO MILP with Incomplete Recourse (IR),
    matching equations (final_IR_single_level) in your LaTeX.
    """

    # ---------------------------------------------------------------
    # Big-M constants
    # ---------------------------------------------------------------
    M_opt, M_feas, M_pi = compute_big_M_constants_exact2ro_IR(
        opening_cost,
        capacity,
        D_bar,
        D_hat,
        ship_cost,
        z_vertices,
        rho_D,
        rho_P,
    )

    # Helper sums
    def first_stage_cost():
        return gp.quicksum(opening_cost[i] * X[i] for i in N)

    # ---------------------------------------------------------------
    # Main bound: eta and selected-scenario recourse y^*
    # ---------------------------------------------------------------
    # (final_IR_eval_selected_mu_lb)
    # eta >= sum_i c_i x_i + sum_{i,j} d_ij y^*_ij
    mdl.addConstr(
        eta
        >= first_stage_cost()
        + gp.quicksum(ship_cost[i][j] * Y_star[i, j] for i in N for j in M),
        name="final_IR_eval_selected_mu_lb",
    )

    # (final_IR_eval_selected_demand_opt)
    # sum_i y^*_ij >= D_bar_j + D_hat_j z^{O,*}_j
    mdl.addConstrs(
        (
            gp.quicksum(Y_star[i, j] for i in N)
            >= D_bar[j] + D_hat[j] * Z_O_star[j]
            for j in M
        ),
        name="final_IR_eval_selected_demand_opt",
    )

    # (final_IR_eval_selected_demand_feas)
    # sum_i y^*_ij >= D_bar_j + D_hat_j z^{F,*}_j
    mdl.addConstrs(
        (
            gp.quicksum(Y_star[i, j] for i in N)
            >= D_bar[j] + D_hat[j] * Z_F_star[j]
            for j in M
        ),
        name="final_IR_eval_selected_demand_feas",
    )

    # (final_IR_eval_selected_capacity)
    # sum_j y^*_ij <= P_i x_i + v_i^*
    mdl.addConstrs(
        (
            gp.quicksum(Y_star[i, j] for j in M)
            <= capacity[i] * X[i]
            for i in N
        ),
        name="final_IR_eval_selected_capacity",
    )

    # ---------------------------------------------------------------
    # Argmax selectors for z^O* and z^F*
    # ---------------------------------------------------------------
    # (final_IR_selector_opt)
    mdl.addConstr(
        gp.quicksum(Delta_O[k] for k in K) == 1,
        name="final_IR_selector_opt_sum",
    )
    mdl.addConstrs(
        (
            Z_O_star[j]
            == gp.quicksum(Delta_O[k] * z_vertices[j, k] for k in K)
            for j in M
        ),
        name="final_IR_selector_opt_def",
    )

    # (final_IR_selector_feas)
    mdl.addConstr(
        gp.quicksum(Delta_F[k] for k in K) == 1,
        name="final_IR_selector_feas_sum",
    )
    mdl.addConstrs(
        (
            Z_F_star[j]
            == gp.quicksum(Delta_F[k] * z_vertices[j, k] for k in K)
            for j in M
        ),
        name="final_IR_selector_feas_def",
    )

    # ---------------------------------------------------------------
    # FEASIBILITY BLOCK (F): primal (yellow) + dual (orange)
    # ---------------------------------------------------------------

    # (final_IR_feas_LHS_t)
    # t^F >= sum_i c_i x_i
    #      + sum_{i,j} d_ij y^{F,(k)}_ij
    #      + rho_D sum_j u_j^{(k)}
    #      + rho_P sum_i v_i^{(k)}    for all k
    mdl.addConstrs(
        (
            t_F
            >= first_stage_cost()
            + gp.quicksum(
                ship_cost[i][j] * Y_F[k, i, j] for i in N for j in M
            )
            + rho_D * gp.quicksum(U[k, j] for j in M)
            + rho_P * gp.quicksum(V[k, i] for i in N)
            for k in K
        ),
        name="final_IR_feas_LHS_t",
    )

    # (final_IR_feas_LHS_demand)
    # sum_i y^{F,(k)}_ij + u_j^{(k)} >= D_bar_j + D_hat_j z_j^{(k)}   ∀ j,k
    mdl.addConstrs(
        (
            gp.quicksum(Y_F[k, i, j] for i in N) + U[k, j]
            >= D_bar[j] + D_hat[j] * z_vertices[j, k]
            for k in K
            for j in M
        ),
        name="final_IR_feas_LHS_demand",
    )

    # (final_IR_feas_LHS_capacity)
    # sum_j y^{F,(k)}_ij <= P_i x_i + v_i^{(k)}   ∀ i,k
    mdl.addConstrs(
        (
            gp.quicksum(Y_F[k, i, j] for j in M)
            <= capacity[i] * X[i] + V[k, i]
            for k in K
            for i in N
        ),
        name="final_IR_feas_LHS_capacity",
    )

    # Dual side (orange) (final_IR_feas_RHS_t)
    # t^F <= sum_i c_i x_i
    #      + sum_j (D_bar_j + D_hat_j z_j^{(k)}) alpha_j^{(k)}
    #      - sum_i P_i gamma_i^{(k)}
    #      + M_feas (1 - Delta_F_k)   ∀ k
    mdl.addConstrs(
        (
            t_F
            <= first_stage_cost()
            + gp.quicksum(
                (D_bar[j] + D_hat[j] * z_vertices[j, k]) * Alpha[k, j]
                for j in M
            )
            - gp.quicksum(capacity[i] * Gamma[k, i] for i in N)
            + M_feas * (1 - Delta_F[k])
            for k in K
        ),
        name="final_IR_feas_RHS_t",
    )

    # McCormick-like constraints for gamma and beta (final_IR_mcc_gamma_*)
    # gamma_i^{(k)} <= rho_P x_i, gamma_i^{(k)} <= beta_i^{(k)}
    mdl.addConstrs(
        (
            Gamma[k, i] <= rho_P * X[i]
            for k in K
            for i in N
        ),
        name="final_IR_mcc_gamma_upx",
    )
    mdl.addConstrs(
        (
            Gamma[k, i] <= Beta[k, i]
            for k in K
            for i in N
        ),
        name="final_IR_mcc_gamma_upp",
    )

    # gamma_i^{(k)} >= beta_i^{(k)} - rho_P (1 - x_i)
    mdl.addConstrs(
        (
            Gamma[k, i] >= Beta[k, i] - rho_P * (1 - X[i])
            for k in K
            for i in N
        ),
        name="final_IR_mcc_gamma_low",
    )


    # Dual feasibility (final_IR_feas_RHS_dualfeas)
    # alpha_j^{(k)} - beta_i^{(k)} <= d_ij
    mdl.addConstrs(
        (
            Alpha[k, j] - Beta[k, i] <= ship_cost[i][j]
            for k in K
            for i in N
            for j in M
        ),
        name="final_IR_feas_RHS_dualfeas",
    )

    # ---------------------------------------------------------------
    # OPTIMALITY BLOCK (O): primal (green) + dual (cyan)
    # ---------------------------------------------------------------

    # (final_IR_opt_LHS_t)
    # t^O >= sum_i c_i x_i + sum_{i,j} d_ij y^{O,(k)}_ij   ∀ k
    mdl.addConstrs(
        (
            t_O
            >= first_stage_cost()
            + gp.quicksum(
                ship_cost[i][j] * Y_O[k, i, j] for i in N for j in M
            )
            for k in K
        ),
        name="final_IR_opt_LHS_t",
    )

    # (final_IR_opt_LHS_demand)
    # sum_i y^{O,(k)}_ij >= D_bar_j + D_hat_j z_j^{(k)}   ∀ j,k
    mdl.addConstrs(
        (
            gp.quicksum(Y_O[k, i, j] for i in N)
            >= D_bar[j] + D_hat[j] * z_vertices[j, k]
            for k in K
            for j in M
        ),
        name="final_IR_opt_LHS_demand",
    )

    # (final_IR_opt_LHS_capacity)
    # sum_j y^{O,(k)}_ij <= P_i x_i   ∀ i,k
    mdl.addConstrs(
        (
            gp.quicksum(Y_O[k, i, j] for j in M) <= capacity[i] * X[i]
            for k in K
            for i in N
        ),
        name="final_IR_opt_LHS_capacity",
    )

    # Dual side (final_IR_opt_RHS_t)
    # t^O <= sum_i c_i x_i
    #      + sum_j (D_bar_j + D_hat_j z_j^{(k)}) lambda_j^{(k)}
    #      - sum_i P_i kappa_i^{(k)}
    #      + M_opt (1 - Delta_O_k)   ∀ k
    mdl.addConstrs(
        (
            t_O
            <= first_stage_cost()
            + gp.quicksum(
                (D_bar[j] + D_hat[j] * z_vertices[j, k]) * Lambda[k, j]
                for j in M
            )
            - gp.quicksum(capacity[i] * Kappa[k, i] for i in N)
            + M_opt * (1 - Delta_O[k])
            for k in K
        ),
        name="final_IR_opt_RHS_t",
    )

    # McCormick for kappa / pi / x (final_IR_mcc_kappa_*)
    mdl.addConstrs(
        (
            Kappa[k, i] <= M_pi * X[i]
            for k in K
            for i in N
        ),
        name="final_IR_mcc_kappa_upx",
    )
    mdl.addConstrs(
        (
            Kappa[k, i] <= Pi[k, i]
            for k in K
            for i in N
        ),
        name="final_IR_mcc_kappa_upp",
    )
    mdl.addConstrs(
        (
            Kappa[k, i] >= Pi[k, i] - M_pi * (1 - X[i])
            for k in K
            for i in N
        ),
        name="final_IR_mcc_kappa_low",
    )

    # Dual feasibility (final_IR_opt_RHS_dualfeas)
    # lambda_j^{(k)} - pi_i^{(k)} <= d_ij
    mdl.addConstrs(
        (
            Lambda[k, j] - Pi[k, i] <= ship_cost[i][j]
            for k in K
            for i in N
            for j in M
        ),
        name="final_IR_opt_RHS_dualfeas",
    )