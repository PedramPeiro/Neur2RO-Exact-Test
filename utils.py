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
from gurobipy import GRB


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
    parser.add_argument("--instance", type=str, default="IR_n4_m12_rep8")
    parser.add_argument("--time_limit", type=int, default=3600)
    parser.add_argument("--data_dir", type=str, default="../data/instances_IR")
    parser.add_argument("--tolerance", type=float, default=1e-6)
    parser.add_argument(
        "--Gamma",
        type=int,
        default=2,
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


def write_debug_summary_Exact2RO_RCR(
    mdl: gp.Model,
    txt_log,
    X,
    Y_star,
    Z_star,
    mu,
    t,
    N: List[int],
    M: List[int],
    opening_cost,
    ship_cost,
    revenue,
    tol: float = 1e-6,
):
    """
    Append a human-readable debug summary to txt_log:

    - optimal x vector
    - optimal z_star vector
    - t*
    - Q(x, z_star) based on Y_star
    - comparison: t vs Q(x,z_star), mu vs t
    """
    if mdl.Status not in (GRB.OPTIMAL, GRB.SUBOPTIMAL):
        with open(txt_log, "a", encoding="utf-8") as f:
            f.write("\n=== Debug summary skipped (model not optimal/suboptimal) ===\n")
        return

    # Extract values
    x_vec = [float(X[i].X) for i in N]
    z_vec = [float(Z_star[j].X) for j in M]
    t_val = float(t.X)
    mu_val = float(mu.X)

    # Compute Q(x, z_star) from Y_star
    Q_xz = 0.0
    # first-stage cost
    for i in N:
        Q_xz += opening_cost[i] * x_vec[i]
    # minus recourse profit term
    for i in N:
        for j in M:
            margin = revenue[i][j] - ship_cost[i][j]
            Q_xz -= margin * float(Y_star[i, j].X)

    diff_t_Q = abs(t_val - Q_xz)
    diff_mu_t = abs(mu_val - t_val)

    equal_t_Q = diff_t_Q <= tol
    equal_mu_t = diff_mu_t <= tol

    with open(txt_log, "a", encoding="utf-8") as f:
        f.write("\n\n=== Exact2RO Debug Summary ===\n")
        f.write(f"Status code: {mdl.Status}\n")
        f.write(f"Objective (mu)   = {mu_val:.6f}\n")
        f.write(f"t (worst-case)   = {t_val:.6f}\n")
        f.write(f"Q(x, z_star)     = {Q_xz:.6f}\n")
        f.write(f"|t - Q(x,z_star)| = {diff_t_Q:.3e}  -> equal? {equal_t_Q}\n")
        f.write(f"|mu - t|          = {diff_mu_t:.3e}  -> equal? {equal_mu_t}\n")

        f.write("\nOptimal x:\n")
        f.write("  x* = [" + ", ".join(f"{val:.6f}" for val in x_vec) + "]\n")

        f.write("\nOptimal z_star:\n")
        f.write("  z_star* = [" + ", ".join(f"{val:.6f}" for val in z_vec) + "]\n")

        f.write("\n(End of Exact2RO Debug Summary)\n")


def write_debug_summary_Exact2RO_IR(
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
    K,
):
    """
    Append a debug summary for Exact2RO-IR to txt_log.

    Logs:
      - model status and objective
      - x* vector
      - Delta_O, Delta_F and chosen vertices k_O, k_F
      - z_O_star and z_F_star
      - t_O*, t_F*, eta* and consistency checks:
            eta >= t_O, eta >= t_F and eta - max(t_O, t_F)
    """
    with open(txt_log, "a", encoding="utf-8") as f:
        f.write("\n\n=== Debug summary (Exact2RO-IR) ===\n")

        # Status and objective
        try:
            obj_val = mdl.ObjVal
        except Exception:
            obj_val = None

        f.write(f"Model status: {mdl.Status}\n")
        f.write(f"Objective (eta*): {obj_val}\n")

        # x* vector
        x_vals = [X[i].X for i in N]
        f.write("x* = [" + ", ".join(f"{v:.4f}" for v in x_vals) + "]\n")

        # Delta_O, Delta_F and chosen vertices
        f.write("Delta_O* (per k): " +
                ", ".join(f"{k}:{Delta_O[k].X:.3f}" for k in K) + "\n")
        f.write("Delta_F* (per k): " +
                ", ".join(f"{k}:{Delta_F[k].X:.3f}" for k in K) + "\n")

        # Pick the argmax indices (ties broken arbitrarily by max)
        k_O = max(K, key=lambda kk: Delta_O[kk].X)
        k_F = max(K, key=lambda kk: Delta_F[kk].X)
        f.write(f"Chosen k_O = {k_O}\n")
        f.write(f"Chosen k_F = {k_F}\n")

        # z_O_star and z_F_star
        zO_vals = [Z_O_star[j].X for j in M]
        zF_vals = [Z_F_star[j].X for j in M]
        f.write("z_O_star = [" + ", ".join(f"{v:.3f}" for v in zO_vals) + "]\n")
        f.write("z_F_star = [" + ", ".join(f"{v:.3f}" for v in zF_vals) + "]\n")

        # t_O, t_F, eta
        tO_val = t_O.X
        tF_val = t_F.X
        eta_val = eta.X
        f.write(f"t_O* = {tO_val:.6f}\n")
        f.write(f"t_F* = {tF_val:.6f}\n")
        f.write(f"eta* = {eta_val:.6f}\n")

        max_t = max(tO_val, tF_val)
        f.write(f"max(t_O*, t_F*) = {max_t:.6f}\n")
        f.write(f"eta* - max(t_O*, t_F*) = {eta_val - max_t:.6e}\n")

        eps = 1e-6
        cond_O = eta_val >= tO_val - eps
        cond_F = eta_val >= tF_val - eps
        f.write(f"[CHECK] eta >= t_O? {cond_O}\n")
        f.write(f"[CHECK] eta >= t_F? {cond_F}\n")

        # Optional: quick view of V_star (capacity slack in selected scenario)
        # v_star_vals = [V_star[i].X for i in N]
        # f.write("V_star (capacity slack for selected scenario) = [" +
        #         ", ".join(f"{v:.4f}" for v in v_star_vals) + "]\n")
