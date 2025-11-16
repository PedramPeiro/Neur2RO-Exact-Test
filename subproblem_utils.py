# subproblem_utils.py

from __future__ import annotations
from typing import Optional, Tuple
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import utils


def build_and_solve_slave_CCG_RCR(
    x_star: np.ndarray,
    c: np.ndarray,
    P: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    pi: np.ndarray,
    Gamma: int,
    time_limit: int,
    tolerance: float,
    log_path: Optional[str] = None,
    model_name: str = "CCG_Slave_Cost",
) -> Tuple[float, np.ndarray, gp.Model]:
    """
    Slave (adversarial) problem for cost minimization.

    Given x̄, solves:

      max  ∑_i c_i x̄_i - ∑_{i,j} (r_ij - d_ij) y_ij
      s.t. KKT system of inner LP in y, budgeted z ∈ Z(Gamma)

    This is the direct cost formulation with KKT encoded via big-M.
    """
    n = c.shape[0]
    m = D_bar.shape[0]
    
    M_alpha = utils.compute_M_alpha(pi, D_bar, D_hat, Gamma)          # shape (m,)
    M_beta  = utils.compute_M_beta(pi, P, x_star)                     # shape (n,)
    M_gamma = utils.compute_M_gamma(M_alpha, M_beta, P, D_bar, D_hat, # shape (n,m)
                              x_star, Gamma)

    mdl = utils.init_model(
        name=model_name,
        time_limit=time_limit,
        log_path=None if log_path is None else utils.Path(log_path), # type: ignore
        mip_gap=tolerance,
    )
    mdl.Params.LogToConsole = 0
    if log_path is not None:
        mdl.Params.LogFile = log_path

    # --- Variables ---
    # primal y_ij >= 0
    y = mdl.addVars(n, m, lb=0.0, vtype=GRB.CONTINUOUS, name="y")

    # dual vars α_j, β_i, γ_ij ≥ 0
    alpha = mdl.addVars(m, lb=0.0, vtype=GRB.CONTINUOUS, name="alpha")
    beta  = mdl.addVars(n, lb=0.0, vtype=GRB.CONTINUOUS, name="beta")
    gamma = mdl.addVars(n, m, lb=0.0, vtype=GRB.CONTINUOUS, name="gamma")

    # binaries for complementarity
    u = mdl.addVars(m, vtype=GRB.BINARY, name="u")      # demand
    v = mdl.addVars(n, vtype=GRB.BINARY, name="v")      # capacity
    w = mdl.addVars(n, m, vtype=GRB.BINARY, name="w")   # y vs γ

    # uncertainty z & |z|
    z = mdl.addVars(m, lb=-1.0, ub=1.0, vtype=GRB.CONTINUOUS, name="z")
    s = mdl.addVars(m, lb=0.0, ub = 1.0, vtype=GRB.CONTINUOUS, name="s")

    # --- Constraints ---

    # Stationarity: (r_ij - d_ij) - α_j - β_i + γ_ij = 0
    for i in range(n):
        for j in range(m):
            mdl.addConstr(
                pi[i, j] - alpha[j] - beta[i] + gamma[i, j] == 0,
                name=f"stationarity_{i}_{j}",
            )

    # Primal feasibility: demand
    for j in range(m):
        mdl.addConstr(
            gp.quicksum(y[i, j] for i in range(n))
            <= D_bar[j] + D_hat[j] * z[j],
            name=f"demand_{j}",
        )

    # Primal feasibility: capacity
    for i in range(n):
        mdl.addConstr(
            gp.quicksum(y[i, j] for j in range(m))
            <= P[i] * x_star[i],
            name=f"capacity_{i}",
        )

    # Complementarity for α_j and demand slack
    for j in range(m):
        mdl.addConstr(alpha[j] <= M_alpha[j] * u[j], name=f"alpha_bigM_{j}")
        mdl.addConstr(
            D_bar[j] + D_hat[j] * z[j]
            - gp.quicksum(y[i, j] for i in range(n))
            <= M_alpha[j] * (1 - u[j]),
            name=f"demand_slack_bigM_{j}",
        )

    # Complementarity for β_i and capacity slack
    for i in range(n):
        mdl.addConstr(beta[i] <= M_beta[i] * v[i], name=f"beta_bigM_{i}")
        mdl.addConstr(
            P[i] * x_star[i]
            - gp.quicksum(y[i, j] for j in range(m))
            <= M_beta[i] * (1 - v[i]),
            name=f"capacity_slack_bigM_{i}",
        )

    # Complementarity for γ_ij and y_ij
    for i in range(n):
        for j in range(m):
            mdl.addConstr(
                gamma[i, j] <= M_gamma[i, j] * w[i, j],
                name=f"gamma_bigM_{i}_{j}",
            )
            mdl.addConstr(
                y[i, j] <= M_gamma[i, j] * (1 - w[i, j]),
                name=f"y_bigM_{i}_{j}",
            )

    # Budgeted uncertainty: s_j >= |z_j|, sum s_j <= Gamma
    for j in range(m):
        mdl.addConstr(s[j] >=  z[j],  name=f"s_ge_z_{j}")
        mdl.addConstr(s[j] >= -z[j],  name=f"s_ge_minusz_{j}")
    mdl.addConstr(
        gp.quicksum(s[j] for j in range(m)) <= Gamma,
        name="budget_Gamma",
    )

    # --- Objective: worst-case cost ---
    # cost(x̄, z) = ∑_i c_i x̄_i - ∑_{i,j} (r_ij - d_ij) y_ij
    cost_expr = (
        gp.quicksum(c[i] * x_star[i] for i in range(n))
        - gp.quicksum(pi[i, j] * y[i, j] for i in range(n) for j in range(m))
    )
    mdl.setObjective(cost_expr, GRB.MAXIMIZE)

    mdl.optimize()

    if mdl.SolCount == 0:
        raise RuntimeError("Slave problem infeasible or no solution found.")

    worst_cost = mdl.ObjVal
    z_star = np.array([z[j].X for j in range(m)])
    return worst_cost, z_star, mdl
