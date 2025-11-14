"""
Random instance generator for RCR and IR facility-location models.

RCR model data saved:
  - fixed opening costs c_i
  - revenues r_ij
  - shipping costs d_ij
  - nominal demands Dbar_j
  - perturbation factors Dhat_j
  - capacities P_i

IR model data saved:
  - fixed opening costs c_i
  - shipping costs d_ij
  - nominal demands Dbar_j
  - perturbation factors Dhat_j
  - capacities P_i

File format (both models):

  # Instance type: RCR or IR
  # Format designed for Python-friendly parsing
  n <n>
  m <m>
  <meta_key_1> <value_1>
  ...

  FACILITIES  # i x y c_i P_i
  i x_i y_i c_i P_i
  ...

  CUSTOMERS  # j x y Dbar_j Dhat_j
  j x_j y_j Dbar_j Dhat_j
  ...

  D_MATRIX  # d_ij (n rows, m columns)
  n m
  d_11 ... d_1m
  ...
  d_n1 ... d_nm

  (RCR only)
  R_MATRIX  # r_ij (n rows, m columns)
  n m
  r_11 ... r_1m
  ...
  r_n1 ... r_nm
"""

from pathlib import Path
import numpy as np
import math

# ----------------------------
# Global knobs
# ----------------------------

sizes = [(5, 10), (5, 15), (10, 15), (10, 20), (20, 40)]
replicas_per_size = 10
base_seed = 20251112

# Demand / uncertainty (RLTP-style)
demand_low, demand_high = 0.0, 20000.0
epsilon_low, epsilon_high = 0.15, 0.4

# Revenue (RCR)
nu_low, nu_high = 1, 1.2  # scalar price shared across arcs

# Opening costs and capacities
open_cost_low, open_cost_high = 30000.0, 50000.0
capacity_low, capacity_high = 0.0, 50000.0

# IR feasibility scaling
rho_IR = 1.05
Gamma_values_RCR = "0,0.1m,0.3m,m"
Gamma_values_IR = "0,0.1m,0.3m,m"

# RCR test thresholds
min_positive_margin_fraction = 0.5   # fraction of (i,j) with r_ij - d_ij > 0
min_margin_per_customer = 0.05       # "reasonable" positive margin

# Max attempts per instance before giving up
max_attempts_per_instance = 1000

# ----------------------------
# Helpers
# ----------------------------

def uniform_points_on_unit_square(count, rng):
    return rng.random((count, 2))

def euclidean_distances(F, C):
    """
    F: (n,2) facilities, C: (m,2) customers
    Returns (n,m) array of Euclidean distances.
    """
    n, m = F.shape[0], C.shape[0]
    diff = F[:, None, :] - C[None, :, :]
    dists = np.sqrt(np.sum(diff * diff, axis=2))
    return dists

def format_float(x):
    return f"{x:.6f}"

def format_int(x):
    # Convert to int even if x is a float or numpy scalar
    try:
        xi = int(round(float(x)))
    except Exception:
        xi = int(x)
    return f"{xi:d}"

def gamma_max_from_string(s, m):
    """
    Parse a string like "0,0.1m,0.3m,m" into integer budgets and return the maximum.
    """
    parts = [t.strip() for t in s.split(",")]
    vals = []
    for t in parts:
        if t == "m":
            vals.append(m)
        elif t.endswith("m"):
            try:
                frac = float(t[:-1])
                vals.append(int(math.floor(frac * m)))
            except Exception:
                pass
        else:
            try:
                vals.append(int(t))
            except Exception:
                pass
    return max(vals) if vals else m

def write_instance_txt(path, meta_dict, facilities, customers, d_matrix,
                       instance_type="RCR", r_matrix=None):
    """
    path: Path to .txt
    meta_dict: dict of metadata (string -> primitive)
    facilities: list of tuples (i, x, y, c_i, P_i)
    customers: list of tuples (j, x, y, Dbar_j, Dhat_j)
    d_matrix: numpy array (n, m) of shipping costs
    r_matrix: numpy array (n, m) of revenues (RCR only), or None
    """
    n, m = d_matrix.shape
    with path.open("w", encoding="utf-8") as f:
        # Header
        f.write(f"# Instance type: {instance_type}\n")
        f.write("# Format designed for Python-friendly parsing\n")
        f.write(f"n {n}\n")
        f.write(f"m {m}\n")
        for k, v in meta_dict.items():
            f.write(f"{k} {v}\n")
        f.write("\n")

        # Facilities
        f.write("FACILITIES  # columns: i x y c_i P_i\n")
        for (i, x, y, c_i, P_i) in facilities:
            f.write(f"{i} {format_float(x)} {format_float(y)} "
                    f"{format_int(c_i)} {format_int(P_i)}\n")
        f.write("\n")

        # Customers
        f.write("CUSTOMERS  # columns: j x y Dbar_j Dhat_j\n")
        for (j, x, y, Dbar, Dhat) in customers:
            f.write(f"{j} {format_float(x)} {format_float(y)} "
                    f"{format_int(Dbar)} {format_int(Dhat)}\n")
        f.write("\n")

        # Shipping matrix
        f.write("D_MATRIX  # n rows, each with m entries (per-unit shipping costs d_ij)\n")
        f.write(f"{n} {m}\n")
        for i in range(n):
            row = " ".join(format_float(val) for val in d_matrix[i, :])
            f.write(row + "\n")

        # Revenue matrix for RCR
        if instance_type == "RCR" and r_matrix is not None:
            f.write("\n")
            f.write("R_MATRIX  # n rows, each with m entries (revenues r_ij)\n")
            f.write(f"{n} {m}\n")
            for i in range(n):
                row = " ".join(format_float(val) for val in r_matrix[i, :])
                f.write(row + "\n")

# ----------------------------
# Generators
# ----------------------------

def generate_RCR_instance(n, m, seed):
    """
    Generate one RCR instance.
    Returns (facilities, customers, d_mat, r_mat, meta).
    """
    rng = np.random.default_rng(seed)

    # 1) Geometry: customers in [0,1]^2, facilities subset
    F = uniform_points_on_unit_square(n, rng)
    C = uniform_points_on_unit_square(m, rng)
    # 2) Shipping costs
    d_mat = euclidean_distances(F, C)

    # 3) Revenues: scalar price nu, same for all arcs (r_ij = nu)
    nu = rng.uniform(nu_low, nu_high)
    r_mat = np.full((n, m), nu, dtype=float)

    # 4) Demands and deviations
    Dbar = rng.uniform(demand_low, demand_high, size=m)
    epsilon = rng.uniform(epsilon_low, epsilon_high, size=m)
    Dhat = epsilon * Dbar

    # 5) Opening costs and capacities
    c_open = rng.uniform(open_cost_low, open_cost_high, size=n)
    P_prime = rng.uniform(capacity_low, capacity_high, size=n)
    P = P_prime.copy()

    # Facilities and customers
    facilities = []
    for i in range(n):
        x_i, y_i = F[i, 0], F[i, 1]
        facilities.append((i + 1,
                           float(x_i),
                           float(y_i),
                           round(c_open[i]),
                           round(P[i])))

    customers = []
    for j in range(m):
        x_j, y_j = C[j, 0], C[j, 1]
        customers.append((j + 1,
                          float(x_j),
                          float(y_j),
                          round(Dbar[j]),
                          round(Dhat[j])))

    meta = {
        "seed": seed,
        "uncertainty": "one_sided_decrease",
        "Gamma_values": Gamma_values_RCR,
        "nu": nu
    }
    return facilities, customers, d_mat, r_mat, meta

def generate_IR_instance(n, m, seed):
    """
    Generate one IR instance with capacity rescaling
    so that opening all facilities can satisfy worst-case total demand.
    Returns (facilities, customers, d_mat, meta).
    """
    rng = np.random.default_rng(seed)

    # 1) Geometry: customers in [0,1]^2, facilities subset
    F = uniform_points_on_unit_square(n, rng)
    C = uniform_points_on_unit_square(m, rng)

    # 2) Shipping costs
    d_mat = euclidean_distances(F, C)

    # 3) Demands and deviations
    Dbar = rng.uniform(demand_low, demand_high, size=m)
    epsilon = rng.uniform(epsilon_low, epsilon_high, size=m)
    Dhat = epsilon * Dbar

    # 4) Opening costs
    c_open = rng.uniform(open_cost_low, open_cost_high, size=n)

    # 5) Provisional capacities
    P_prime = rng.uniform(capacity_low, capacity_high, size=n)

    # Worst-case total demand for largest Gamma
    gamma_max = gamma_max_from_string(Gamma_values_IR, m)
    Dhat_sorted = np.sort(Dhat)[::-1]
    W = float(Dbar.sum() + Dhat_sorted[:min(gamma_max, m)].sum())

    total_P_prime = float(P_prime.sum())
    if total_P_prime <= 0:
        s = 1.0
    else:
        s = (rho_IR * W) / total_P_prime
    P = s * P_prime

    facilities = []
    for i in range(n):
        x_i, y_i = F[i, 0], F[i, 1]
        facilities.append((i + 1,
                           float(x_i),
                           float(y_i),
                           round(c_open[i]),
                           round(P[i])))

    customers = []
    for j in range(m):
        x_j, y_j = C[j, 0], C[j, 1]
        customers.append((j + 1,
                          float(x_j),
                          float(y_j),
                          round(Dbar[j]),
                          round(Dhat[j])))

    meta = {
        "seed": seed,
        "uncertainty": "one_sided_increase",
        "Gamma_values": Gamma_values_IR,
        "rho_IR": rho_IR
    }
    return facilities, customers, d_mat, meta

# ----------------------------
# Tests
# ----------------------------

def test_RCR_instance(facilities, customers, d_mat, r_mat, meta, verbose=False):
    """
    Sanity + "no-open is not optimal" test for RCR.
    Returns True if instance passes, False otherwise.
    """
    n, m = d_mat.shape

    # Extract Dbar from customers
    Dbar = np.array([c[3] for c in customers], dtype=float)  # Dbar_j

    # Margins = r_ij - d_ij (here r_ij = nu constant)
    margin = r_mat - d_mat
    pos_arcs = margin > 0.0
    pos_rate = pos_arcs.mean()

    # 1) Require that a reasonable fraction of arcs have positive margin
    if pos_rate < min_positive_margin_fraction:
        if verbose:
            print("RCR test fail: pos_rate too small:", pos_rate)
        return False

    # 2) Almost all customers should have some reasonably positive margin
    best_margin_per_j = margin.max(axis=0)
    frac_customers_good = np.mean(best_margin_per_j > min_margin_per_customer)
    if frac_customers_good < 0.9:
        if verbose:
            print("RCR test fail: fraction of customers with margin >",
                  min_margin_per_customer, "too low:", frac_customers_good)
        return False

    # 3) "No-open is not optimal": check heuristic profit
    c_open = np.array([f[3] for f in facilities], dtype=float)

    # (a) Best single facility
    best_single_profit = -np.inf
    for i in range(n):
        mar_i = margin[i, :]
        contrib = np.maximum(mar_i, 0.0) * Dbar
        profit_i = contrib.sum() - c_open[i]
        if profit_i > best_single_profit:
            best_single_profit = profit_i

    # (b) Opening all facilities
    best_margin_all = best_margin_per_j
    profit_all_open = (np.maximum(best_margin_all, 0.0) * Dbar).sum() - c_open.sum()

    best_heuristic_profit = max(best_single_profit, profit_all_open)
    if best_heuristic_profit <= 0.0:
        if verbose:
            print("RCR test fail: heuristic best profit <= 0.",
                  "best_single:", best_single_profit,
                  "all_open:", profit_all_open)
        return False

    return True

def test_IR_instance(facilities, customers, d_mat, meta, verbose=False):
    """
    Feasibility test for IR: with all facilities open,
    total capacity >= worst-case total demand for Gamma_max.
    """
    n, m = d_mat.shape

    Dbar = np.array([c[3] for c in customers], dtype=float)
    Dhat = np.array([c[4] for c in customers], dtype=float)
    P = np.array([f[4] for f in facilities], dtype=float)

    gamma_max = gamma_max_from_string(meta["Gamma_values"], m)
    Dhat_sorted = np.sort(Dhat)[::-1]
    W = float(Dbar.sum() + Dhat_sorted[:min(gamma_max, m)].sum())

    if P.sum() + 1e-6 < W:
        if verbose:
            print("IR test fail: capacity sum < worst-case total demand:",
                  P.sum(), "<", W)
        return False

    return True

# ----------------------------
# Main loop
# ----------------------------

def main():
    out_RCR = Path("../data/instances_RCR")
    out_IR = Path("../data/instances_IR")
    out_RCR.mkdir(parents=True, exist_ok=True)
    out_IR.mkdir(parents=True, exist_ok=True)

    for (n, m) in sizes:
        for rep in range(replicas_per_size):
            # ---- RCR ----
            attempts = 0
            while True:
                attempts += 1
                seed_rcr = base_seed + 1000 * n + 10 * m + rep + 100000 * attempts
                facs_rcr, custs_rcr, dmat_rcr, rmat_rcr, meta_rcr = generate_RCR_instance(n, m, seed_rcr)
                if test_RCR_instance(facs_rcr, custs_rcr, dmat_rcr, rmat_rcr, meta_rcr, verbose=True):
                    fname_rcr = f"RCR_n{n}_m{m}_rep{rep}_seed{seed_rcr}.txt"
                    path_rcr = out_RCR / fname_rcr
                    write_instance_txt(path_rcr, meta_rcr, facs_rcr, custs_rcr,
                                       dmat_rcr, instance_type="RCR", r_matrix=rmat_rcr)
                    break
                if attempts >= max_attempts_per_instance:
                    raise RuntimeError(
                        f"Failed to generate valid RCR instance for (n={n}, m={m}, rep={rep}) "
                        f"after {attempts} attempts"
                    )

            # ---- IR ----
            attempts = 0
            while True:
                attempts += 1
                seed_ir = base_seed + 2000 * n + 20 * m + rep + 100000 * attempts
                facs_ir, custs_ir, dmat_ir, meta_ir = generate_IR_instance(n, m, seed_ir)
                if test_IR_instance(facs_ir, custs_ir, dmat_ir, meta_ir, verbose=True):
                    fname_ir = f"IR_n{n}_m{m}_rep{rep}_seed{seed_ir}.txt"
                    path_ir = out_IR / fname_ir
                    write_instance_txt(path_ir, meta_ir, facs_ir, custs_ir,
                                       dmat_ir, instance_type="IR", r_matrix=None)
                    break
                if attempts >= max_attempts_per_instance:
                    raise RuntimeError(
                        f"Failed to generate valid IR instance for (n={n}, m={m}, rep={rep}) "
                        f"after {attempts} attempts"
                    )

if __name__ == "__main__":
    main()