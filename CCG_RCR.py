# CCG_RCR.py

from __future__ import annotations
import time
from pathlib import Path
from typing import List

import numpy as np

import utils
from masterproblem_utils import build_and_solve_master_CCG_RCR
from subproblem_utils import build_and_solve_slave_CCG_RCR


def main() -> None:
    args = utils.parse_cli()

    if args.Gamma is None:
        raise ValueError(
            "Gamma (budget of uncertainty) must be provided for CCG_RCR. "
            "Run with e.g. `--Gamma 4`."
        )

    # ---------- paths / logging --------------------------------------------
    run_dir, gru_dir, txt_log = utils.prepare_paths(
        method="CCG",
        environment="Robust",
        category="RCR",
        instance=args.instance,
        budget=args.Gamma,
        out_root=args.output_root,
    )

    # ---------- load instance ----------------------------------------------
    param = utils.load_instance(args.instance, args.data_dir)
    n = param["N"]
    m = param["M"]

    c = np.array(param["c"])
    P = np.array(param["P"])
    D_bar = np.array(param["Dbar"])
    D_hat = np.array(param["Dhat"])
    d = np.array(param["d"])
    r = np.array(param["r"])
    pi = r - d

    Gamma = args.Gamma

    # Initial scenario: nominal z = 0
    scenario_list: List[np.ndarray] = [np.zeros(m)]
    # Track how many times each scenario has been returned as worst-case
    repeat_counts: List[int] = [0]

    # Global bounds
    LB = -float("inf")
    UB = float("inf")
    best_x = None
    iteration = 0

    overall_start = time.perf_counter()

    # ---------- txt log header ---------------------------------------------
    with txt_log.open("w", encoding="utf-8") as fh:
        fh.write(
            f"Result - instance {args.instance} (RCR, CCG, Gamma={Gamma})\n"
            f"Facilities: {n} | Customers: {m}\n"
        )
        fh.write(f"Time limit per solve: {args.time_limit}s\n")
        fh.write(f"Tolerance (gap): {args.tolerance}\n\n")

    # ---------- CCG main loop ----------------------------------------------
    while True:
        iteration += 1

        # ----- Master -----
        master_log_path = gru_dir / f"gurobi_master_it{iteration}.log"
        theta_val, x_star, MP = build_and_solve_master_CCG_RCR(
            scenarios=scenario_list,
            c=c,
            P=P,
            D_bar=D_bar,
            D_hat=D_hat,
            pi=pi,
            time_limit=args.time_limit,
            tolerance=args.tolerance,
            log_path=str(master_log_path),
            model_name=f"CCG_Master_Cost_{args.instance}_G{Gamma}_it{iteration}",
        )
        LB_iter = theta_val
        LB = max(LB, LB_iter)
        if best_x is None or LB_iter >= LB - 1e-10:
            best_x = x_star.copy()

        # ----- Slave -----
        slave_log_path = gru_dir / f"gurobi_slave_it{iteration}.log"
        worst_cost, z_worst, SP = build_and_solve_slave_CCG_RCR(
            x_star=x_star,
            c=c,
            P=P,
            D_bar=D_bar,
            D_hat=D_hat,
            pi=pi,
            Gamma=Gamma,
            time_limit=args.time_limit,
            tolerance=args.tolerance,
            log_path=str(slave_log_path),
            model_name=f"CCG_Slave_Cost_{args.instance}_G{Gamma}_it{iteration}",
        )
        UB_iter = worst_cost
        UB = min(UB, UB_iter)

        gap = (UB - LB) / abs(UB) if abs(UB) > 1e-12 else float("inf")

        # ----- Logging for this iteration -----
        with txt_log.open("a", encoding="utf-8") as fh:
            fh.write(f"Iter {iteration}\n")
            fh.write(
                f"  LB_iter (master θ) = {LB_iter:.6f}, "
                f"UB_iter (worst cost) = {UB_iter:.6f}\n"
            )
            fh.write(
                f"  LB_global = {LB:.6f}, UB_global = {UB:.6f}, "
                f"gap = {gap:.6e}\n"
            )
            fh.write("  x*: [")
            fh.write(", ".join(f"{xi:.3f}" for xi in x_star))
            fh.write("]\n")
            fh.write("  z_worst: [")
            fh.write(", ".join(f"{zj:.3f}" for zj in z_worst))
            fh.write("]\n\n")

        # ----- Stopping rule based on gap -----
        if gap <= args.tolerance:
            with txt_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"Converged at iteration {iteration}: "
                    f"(UB - LB) / |UB| = {gap:.3e} <= tolerance {args.tolerance}\n"
                )
            break

        # ----- Scenario update + repetition counting -----
        match_idx = None
        for idx, z_prev in enumerate(scenario_list):
            if np.allclose(z_worst, z_prev, atol=1e-6):
                match_idx = idx
                break

        if match_idx is None:
            # New critical scenario
            scenario_list.append(z_worst)
            repeat_counts.append(0)
        else:
            # Existing scenario repeated
            repeat_counts[match_idx] += 1
            with txt_log.open("a", encoding="utf-8") as fh:
                fh.write(
                    f"Adversary returned existing scenario index {match_idx}; "
                    f"repeat count = {repeat_counts[match_idx]}.\n"
                )

            # If repeated 5 times, push LB to UB and stop
            if repeat_counts[match_idx] >= 5:
                LB = UB  # push LB to the value created by this critical scenario
                gap = 0.0
                with txt_log.open("a", encoding="utf-8") as fh:
                    fh.write(
                        f"Scenario index {match_idx} repeated 5 times. "
                        f"Forcing LB_global = UB_global = {UB:.6f} and "
                        f"terminating CCG.\n"
                    )
                break

        # Optional hard iteration cap
        if iteration >= 1000:
            with txt_log.open("a", encoding="utf-8") as fh:
                fh.write("Maximum number of iterations reached → stopping.\n")
            break

    # ---------- Final summary ----------------------------------------------
    overall_time = time.perf_counter() - overall_start
    with txt_log.open("a", encoding="utf-8") as fh:
        fh.write("\n# --- Final summary ---\n")
        fh.write(f"Total iterations: {iteration}\n")
        fh.write(f"Overall wall-clock time: {overall_time:.2f}s\n")
        fh.write(f"Final LB_global: {LB:.6f}\n")
        fh.write(f"Final UB_global: {UB:.6f}\n")
        if best_x is not None:
            fh.write("Best x (associated with LB_global): [")
            fh.write(", ".join(f"{xi:.3f}" for xi in best_x))
            fh.write("]\n")

    print(f"[CCG_RCR] run complete → {txt_log}")
    print(
        f"Robust worst-case cost ≈ {UB:.4f}, "
        f"best x = {[round(float(xi), 3) for xi in best_x]}"
    )


if __name__ == "__main__":
    main()
