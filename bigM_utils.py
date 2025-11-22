import numpy as np


def compute_M_alpha_IR(
    d: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    P: np.ndarray,
) -> np.ndarray:
    """
    Compute Big-M values M_alpha[j] for IR duals α_j and demand slack.

    Uses:
      - α_j ≤ d_min_j := min_i d_ij  (marginal cost of demand at j)
      - demand slack s_j^D = sum_i y_ij - D_j(z)
        with 0 ≤ s_j^D ≤ S_j^D_max := max(0, P_tot - D_j_min),
        where D_j_min = max(0, D_bar_j - |D_hat_j|)

      Then:
        M_alpha_j = max(d_min_j, S_j^D_max).
    """
    n, m = d.shape
    # Customer-wise min cost
    d_min = np.min(d, axis=0)  # shape (m,)

    # Worst-case min demand and total capacity
    D_min = np.maximum(0.0, D_bar - np.abs(D_hat))  # shape (m,)
    P_tot = float(np.sum(P))

    S_D_max = np.maximum(0.0, P_tot - D_min)  # shape (m,)

    M_alpha = np.maximum(d_min, S_D_max)
    return M_alpha


def compute_M_beta_IR(
    d: np.ndarray,
    P: np.ndarray,
    x_star: np.ndarray,
) -> np.ndarray:
    """
    Compute Big-M values M_beta[i] for IR duals β_i and capacity slack.

    Uses:
      - β_i ≤ d_max_i := max_j d_ij  (marginal value of capacity at i)
      - capacity slack s_i^C = P_i x_i - sum_j y_ij
        with 0 ≤ s_i^C ≤ S_i^C_max := P_i x_i

      Then:
        M_beta_i = max(d_max_i, P_i x_i).
    """
    n, m = d.shape
    d_max_i = np.max(d, axis=1)  # shape (n,)
    S_C_max = P * x_star         # shape (n,)

    M_beta = np.maximum(d_max_i, S_C_max)
    return M_beta



def compute_M_gamma_IR(
    d: np.ndarray,
    P: np.ndarray,
    D_bar: np.ndarray,
    D_hat: np.ndarray,
    x_star: np.ndarray,
    M_beta: np.ndarray,
) -> np.ndarray:
    """
    Compute Big-M values M_gamma[i,j] for IR γ_ij and y_ij complementarity.

    Uses:
      - y_ij ≤ Y_max_ij := min(P_i x_i, D_j_max),
        where D_j_max = D_bar_j + |D_hat_j|
      - γ_ij = d_ij - α_j + β_i ≤ d_ij + β_i ≤ d_ij + M_beta_i

      Then:
        M_gamma_ij = max(Y_max_ij, d_ij + M_beta_i).
    """
    n, m = d.shape

    D_max = D_bar + np.abs(D_hat)  # shape (m,)
    Y_max = np.zeros((n, m), dtype=float)
    Gamma_bound = np.zeros((n, m), dtype=float)

    for i in range(n):
        cap_i = P[i] * x_star[i]
        for j in range(m):
            Y_max[i, j] = min(cap_i, D_max[j])
            Gamma_bound[i, j] = d[i, j] + M_beta[i]

    M_gamma = np.maximum(Y_max, Gamma_bound)
    return M_gamma




def compute_M_alpha_RCR(
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


def compute_M_beta_RCR(
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


def compute_M_gamma_RCR(
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