import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Any, Union, Mapping, Optional, Set
import re
import math
import pandas as pd
import gurobipy as gp
import cvxpy as cp
import numpy as np
from collections import defaultdict
import dataread
import itertools



_STATUS_MAP = {
    gp.GRB.OPTIMAL: "OPTIMAL",
    gp.GRB.SUBOPTIMAL: "SUBOPTIMAL",
    gp.GRB.TIME_LIMIT: "TIME_LIMIT",
    gp.GRB.INFEASIBLE: "INFEASIBLE",
    gp.GRB.INTERRUPTED: "INTERRUPTED", 
    gp.GRB.WORK_LIMIT: "WORK_LIMIT",
    gp.GRB.NODE_LIMIT: "NODE_LIMIT",
    gp.GRB.SOLUTION_LIMIT: "SOLUTION_LIMIT",
    gp.GRB.ITERATION_LIMIT: "ITER_LIMIT",
}


def _status(code: int) -> str:
    return _STATUS_MAP.get(code, f"OTHER({code})")


def _safe(model: gp.Model, attr: str):
    try:
        return getattr(model, attr)
    except gp.GurobiError:
        return None
    
    
    
def parse_cli() -> argparse.Namespace:
    """Parse command-line arguments shared by all experiment scripts."""
    parser = argparse.ArgumentParser(
        description="Run FL MIP / robust instance",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--instance", type=str, default="RCR_n4_m12_rep3")
    parser.add_argument("--time_limit", type=int, default=1200)
    parser.add_argument("--data_dir", type=str, default="../data/instances_RCR")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--Gamma",
        type=int,
        default=4,
        help=(
            "Budget of uncertainty for robust models (integer in [0, m]). "
            "Deterministic models simply ignore this argument."
        ),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default="../Local Results",
        help="Root directory where logs will be written",
    )
    return parser.parse_args()



def make_run_dirs(output_root: Union[str, Path], category: str) -> Path:
    """Create `<output_root>/log_MIP_Deterministic_{}/<timestamp>` with a Gurobi subfolder.

    Returns the *run directory* Path.
    """
    root = Path(output_root)
    ts = datetime.now().strftime("%Y%m%d%H")
    run_dir = root / f"log_MIP_Deterministic_{category}" / ts
    (run_dir / "Gurobi_logs").mkdir(parents=True, exist_ok=True)
    return run_dir

def prepare_paths(method: str,
                  environment: str,
                  instance: str | int,
                  category: str,
                  budget: int | None,
                  out_root: Union[str, Path]) -> tuple[Path, Path, Path]:
    """Return *(run_dir, gurobi_dir, txt_log_path)* and create folders.

    `method`      : e.g. "MIP", "VE", "CCG"
    `environment` : e.g. "Deterministic", "Robust"
    `category`    : e.g. "IR", "RCR"
    `budget`      : can be None (no budget), an int (e.g. Gamma), or a float (e.g. epsilon)
    """
    ts = datetime.now().strftime("%Y%m%d%H")
    run_dir = Path(out_root) / f"log_{method}_{environment}_{category}" / ts

    gru_dir = run_dir / "Gurobi_logs" / f"instance_{instance}_{method}"
    gru_dir.mkdir(parents=True, exist_ok=True)

    if budget is None:
        txt_log = run_dir / f"log_{instance}_{method}.txt"
    else:
        # Integer budgets
        txt_log = run_dir / f"log_{instance}_{method}_G{budget}.txt"


    return run_dir, gru_dir, txt_log


def load_instance(instance_idx: int, data_dir: str | Path) -> Dict:
    """Wrapper around `datareading.read_instance` that resolves the file path."""
    instance_path = Path(data_dir) / f"{instance_idx}.txt"
    if not instance_path.exists():
        raise FileNotFoundError(f"Instance file not found: {instance_path}")
    return dataread.read_instance(str(instance_path))


def init_model(name: str,
               time_limit: int ,
               log_path: Path,
               mip_gap: float,
               extra_params: Optional[dict] = None):
    m = gp.Model(name)
    m.Params.TimeLimit = time_limit
    m.Params.MIPGap    = mip_gap
    m.Params.OutputFlag = 1
    if log_path is not None:
        m.Params.LogFile    = str(log_path)
    if extra_params:
        for k, v in extra_params.items():
            setattr(m.Params, k, v)
    return m


def log_solution(model: gp.Model,
                 txt: Path,
                 meta: Dict[str, Any],
                 bin_vars: Mapping[str, gp.tupledict] | None = None,
                 list_vars: Mapping[str, list[float]] | None = None,
                 *, tol: float = 1e-6) -> Dict[str, Any]:
    """
    Write a human-readable .txt log of a single Gurobi run *and* return a
    summary row (dict) ready for utils.write_summary().

    Parameters
    ----------
    model      : solved gp.Model
    txt        : target text-file path
    meta       : header / bookkeeping information (see call site)
    bin_vars   : {name → tupledict}   - only entries with value > 0.5 are dumped
    list_vars  : {name → list[float]} - prints full vector e.g. Cmax per scenario
    tol        : gap for printing very small continuous entries (currently unused)

    Returns
    -------
    dict  (shallow) suitable for utils.write_summary().
    """
    # ── solver status & scalar bounds ─────────────────────────────────────
    status  = _status(model.status)
    LB_val  = _safe(model, "ObjBound")
    UB_val  = _safe(model, "ObjVal") if model.SolCount else None
    gap_val = _safe(model, "MIPGap")

    LB   = round(LB_val, 3) if LB_val not in (None, float("inf")) else ""
    UB   = round(UB_val, 3) if UB_val not in (None, float("inf")) else ""
    gap  = round(gap_val * 100, 3) if isinstance(gap_val, (int, float)) else ""

    # ── optional IIS dump for infeasible models ───────────────────────────
    iis_path = None
    if model.status == gp.GRB.INFEASIBLE:
        try:
            model.computeIIS()
            iis_path = txt.with_suffix(".ilp")
            model.write(str(iis_path))
        except gp.GurobiError:
            pass

    # ── write text log ----------------------------------------------------
    with txt.open("w", encoding="utf-8") as fh:
        fh.write(meta["header"].rstrip("\n") + "\n")
        fh.write(f"Status : {status}\nLB : {LB}\nUB : {UB}\nGap (%) : {gap}\n")
        fh.write(f"Time   : {meta['runtime']:.2f}s  (limit {meta['limit']}s)\n")

        if iis_path:
            fh.write(f"IIS saved to: {iis_path}\n")

        if bin_vars and model.SolCount:
            fh.write("\n# --- Binary = 1 ---\n")
            for name, tdict in bin_vars.items():
                if hasattr(tdict, "items"):
                    for key, var in tdict.items():
                        if var.X > 0.5:
                            fh.write(f"{name}{key} = 1\n")

        if list_vars and model.SolCount:
            fh.write("\n# --- Vector vars ---\n")
            for name, vect in list_vars.items():
                vals = ", ".join(f"{v:.3f}" for v in vect)
                fh.write(f"{name} = [{vals}]\n")

    # ── build summary row --------------------------------------------------
    summary = {
        **{k: meta[k] for k in (
            "instance", "n_facilities", "n_customers",
            "method", "uncertainty", "overall_time", "tolerance")},
        "model_status"     : status,
        "computational_time": meta["runtime"],
        "LB"               : LB,
        "UB"               : UB,
        "gap"              : gap,
    }
    return summary


# ---------- Vertex generation utilities for budgeted uncertainty sets ----------
def vertex_generation(dim: int, Gamma: int, max_vertices: int = 2_000_000) -> tuple[np.ndarray, int]:
    """
    Generate vertices of the `dim`-dimensional budgeted uncertainty set

        Z(Gamma) = { z in R^dim | z in [-1,1]^dim, sum_i |z_i| <= Gamma }

    using the same logic as the reference code, but without enumerating {-1,0,1}^dim.

    We mimic the original behavior:
      - work with z_i in {-1, 0, 1}
      - keep only points with sum_i |z_i| = Gamma

    So the vertices we generate are exactly all vectors with exactly Gamma nonzero
    components, each either +1 or -1.

    Parameters
    ----------
    dim   : int
        Dimension of the uncertainty vector (e.g., number of customers m).
    Gamma : int
        Budget of uncertainty (integer, 0 <= Gamma <= dim).
    max_vertices : int
        Safety cap on the number of vertices generated.

    Returns
    -------
    Zs : np.ndarray, shape (dim, N_vertices)
        Each column is a vertex z^k.
    N  : int
        Number of vertices.
    """
    if not isinstance(Gamma, int) or Gamma < 0 or Gamma > dim:
        raise ValueError(f"Gamma must be an integer in [0, {dim}], got {Gamma!r}.")

    # Gamma = 0 → only the zero vector
    if Gamma == 0:
        Zs = np.zeros((dim, 1), dtype=int)
        return Zs, 1

    vertices: list[np.ndarray] = []

    # Choose exactly Gamma indices to be nonzero
    for subset in itertools.combinations(range(dim), Gamma):
        # For those indices, assign ±1 arbitrarily
        for signs in itertools.product([-1, 1], repeat=Gamma):
            z = np.zeros(dim, dtype=int)
            for idx, s in zip(subset, signs):
                z[idx] = s
            vertices.append(z)

            if len(vertices) > max_vertices:
                raise ValueError(
                    f"Number of vertices exceeded max_vertices={max_vertices}. "
                    f"dim={dim}, Gamma={Gamma} is likely too large for VE."
                )

    Zs = np.stack(vertices, axis=1)  # shape: (dim, N_vertices)
    N_vertices = Zs.shape[1]
    return Zs, N_vertices


def compute_gap(ub: Optional[float], lb: Optional[float]) -> Optional[float]:
    """Return (ub-lb)/ub if numbers are valid else *None*."""
    if ub is None or lb is None:
        return None
    if math.isinf(ub):
        return float('inf')  # Clearly indicates unbounded gap
    return (ub - lb) / abs(ub)


def compute_M_alpha(
    pi: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    Gamma: int,
) -> np.ndarray:
    """
    Compute M^alpha_j for each customer j, used in:

        alpha_j <= M^alpha_j * u_j
        D_bar_j + D_hat_j * z_j - sum_i y_ij <= M^alpha_j * (1 - u_j)

    Derivation:
      - alpha_j <= max_i pi_ij    (dual-side)
      - demand slack <= max_z (D_bar_j + D_hat_j z_j) = D_bar_j + D_hat_j * min(1, Gamma)
    """
    n, m = pi.shape
    assert m == D_bar.shape[0] == D_hat.shape[0]

    # Dual-based upper bound: max_i pi_ij
    A_dual = pi.max(axis=0)  # shape (m,)

    # Max possible RHS of demand constraint per j
    if Gamma <= 0:
        z_j_max = 0.0
    else:
        z_j_max = 1.0  # since Gamma is int >=1 → can spend 1 unit on customer j

    A_slack = D_bar + D_hat * z_j_max  # shape (m,)

    M_alpha = np.maximum(A_dual, A_slack)
    return M_alpha  # shape (m,)


def compute_M_beta(
    pi: np.ndarray,
    P: np.ndarray,
    x_star: np.ndarray,
) -> np.ndarray:
    """
    Compute M^beta_i for each facility i, used in:

        beta_i <= M^beta_i * v_i
        P_i x_star_i - sum_j y_ij <= M^beta_i * (1 - v_i)

    Derivation:
      - beta_i <= max_j pi_ij      (dual-side)
      - capacity slack <= P_i x_star_i.
    """
    n, m = pi.shape
    assert n == P.shape[0] == x_star.shape[0]

    # Dual-based upper bound: max_j pi_ij
    B_dual = pi.max(axis=1)  # shape (n,)

    # Max capacity slack per i for current x_star
    B_slack = P * x_star  # shape (n,)

    M_beta = np.maximum(B_dual, B_slack)
    return M_beta  # shape (n,)


def compute_M_gamma(
    M_alpha: np.ndarray,
    M_beta: np.ndarray,
    P: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    x_star: np.ndarray,
    Gamma: int,
) -> np.ndarray:
    """
    Compute M^gamma_{ij} for each (i,j), used in:

        gamma_ij <= M^gamma_ij * w_ij
        y_ij     <= M^gamma_ij * (1 - w_ij)

    Derivation:
      - y_ij <= min( D_j^max, P_i x_star_i )
        with D_j^max = D_bar_j + D_hat_j * min(1, Gamma)
      - gamma_ij = alpha_j + beta_i - pi_ij
        ⇒ gamma_ij <= M_alpha_j + M_beta_i (since pi_ij >= 0).
      - So M^gamma_ij = max( y_max_ij, M_alpha_j + M_beta_i ).
    """
    n = P.shape[0]
    m = D_bar.shape[0]
    assert M_alpha.shape[0] == m
    assert M_beta.shape[0] == n
    assert x_star.shape[0] == n
    assert D_hat.shape[0] == m

    if Gamma <= 0:
        z_j_max = 0.0
    else:
        z_j_max = 1.0

    D_max = D_bar + D_hat * z_j_max  # shape (m,)

    M_gamma = np.zeros((n, m), dtype=float)

    for i in range(n):
        cap_i = P[i] * x_star[i]
        for j in range(m):
            y_max_ij = min(D_max[j], cap_i)
            gamma_dual_max_ij = M_alpha[j] + M_beta[i]
            M_gamma[i, j] = max(y_max_ij, gamma_dual_max_ij)

    return M_gamma  # shape (n, m)