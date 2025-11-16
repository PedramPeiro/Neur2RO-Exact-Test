# masterproblem_utils.py

from __future__ import annotations
from typing import Sequence, List, Tuple, Optional
import numpy as np
import gurobipy as gp
from gurobipy import GRB

import utils


def _add_master_variables_CCG_RCR(
    model: gp.Model,
    n: int,
    m: int,
    K: int,
) -> Tuple[gp.tupledict, gp.tupledict, gp.Var]:
    """
    Add master problem variables:

    - x_i       : facility open decisions, binary
    - y_{k,i,j} : recourse flow for scenario k, continuous >= 0
    - theta     : epigraph variable (worst-case cost over current scenarios)
    """
    x = model.addVars(n, vtype=GRB.BINARY, name="x")
    y = model.addVars(K, n, m, lb=0.0, vtype=GRB.CONTINUOUS, name="y")
    theta = model.addVar(lb=-GRB.INFINITY, vtype=GRB.CONTINUOUS, name="theta")
    return x, y, theta


def _add_master_constraints_CCG_RCR(
    model: gp.Model,
    x: gp.tupledict,
    y: gp.tupledict,
    theta: gp.Var,
    scenarios: Sequence[np.ndarray],
    c: np.ndarray,
    P: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    pi: np.ndarray,
) -> None:
    """
    Add constraints:

    For each scenario k and customer j:
        sum_i y_{k,i,j} <= D_bar_j + D_hat_j * z^k_j

    For each scenario k and facility i:
        sum_j y_{k,i,j} <= P_i * x_i

    Epigraph (worst-case cost) for each scenario k:
        theta >= sum_i c_i x_i - sum_{i,j} pi_ij * y_{k,i,j}
    """
    n = c.shape[0]
    m = D_bar.shape[0]
    K = len(scenarios)

    # Demand constraints
    for k in range(K):
        z_k = scenarios[k]
        for j in range(m):
            model.addConstr(
                gp.quicksum(y[k, i, j] for i in range(n))
                <= D_bar[j] + D_hat[j] * z_k[j],
                name=f"demand_k{k}_j{j}",
            )

    # Capacity constraints
    for k in range(K):
        for i in range(n):
            model.addConstr(
                gp.quicksum(y[k, i, j] for j in range(m))
                <= P[i] * x[i],
                name=f"capacity_k{k}_i{i}",
            )

    # Epigraph constraints (worst-case cost)
    for k in range(K):
        open_cost = gp.quicksum(c[i] * x[i] for i in range(n))
        profit_term = gp.quicksum(pi[i, j] * y[k, i, j] for i in range(n) for j in range(m))
        # cost = cᵀx - Σ_ij π_ij y_ij^k
        model.addConstr(
            theta >= open_cost - profit_term,
            name=f"optcut_k{k}",
        )


def build_and_solve_master_CCG_RCR(
    scenarios: Sequence[np.ndarray],
    c: np.ndarray,
    P: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    pi: np.ndarray,
    time_limit: int,
    tolerance: float,
    log_path: Optional[str] = None,
    model_name: str = "CCG_Master_Cost",
) -> Tuple[float, np.ndarray, gp.Model]:
    """
    Build and solve the CCG master problem for cost minimization.

    Parameters
    ----------
    scenarios : list of z^k vectors (shape (m,))
    c, P, D_bar, D_hat, pi : data arrays
    time_limit, tolerance  : solver settings
    log_path               : path for Gurobi .log
    model_name             : model name

    Returns
    -------
    theta_val : float
        Master objective value (worst-case cost over current scenarios).
    x_star    : np.ndarray
        First-stage solution x*.
    model     : gp.Model
        Solved Gurobi model.
    """
    n = c.shape[0]
    m = D_bar.shape[0]
    K = len(scenarios)

    mdl = utils.init_model(
        name=model_name,
        time_limit=time_limit,
        log_path=None if log_path is None else utils.Path(log_path), # type: ignore
        mip_gap=tolerance,
    )
    mdl.Params.LogToConsole = 0
    if log_path is not None:
        mdl.Params.LogFile = log_path

    x, y, theta = _add_master_variables_CCG_RCR(mdl, n, m, K)
    _add_master_constraints_CCG_RCR(mdl, x, y, theta, scenarios, c, P, D_bar, D_hat, pi)

    mdl.setObjective(theta, GRB.MINIMIZE)
    mdl.optimize()

    if mdl.SolCount == 0:
        raise RuntimeError("Master problem infeasible or no solution found.")

    theta_val = mdl.ObjVal
    x_star = np.array([x[i].X for i in range(n)])

    return theta_val, x_star, mdl
