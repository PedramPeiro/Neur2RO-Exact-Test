import gurobipy as gp
from gurobipy import GRB
import argparse
from pathlib import Path
import utils

def _add_variables(mdl: gp.Model, N, M):
    
    X = mdl.addVars(N, vtype=GRB.BINARY, name="X")
    Y = mdl.addVars(N, M, lb = 0, vtype=GRB.CONTINUOUS, name="Y")
    
    return X, Y

def _add_constraints(mdl: gp.Model, X, Y, demand, capacity, N, M):
    mdl.addConstrs(
        (gp.quicksum(Y[i, j] for i in N) >= demand[j] for j in M)
        , name="demand_constr")
    mdl.addConstrs(
        (gp.quicksum(Y[i, j] for j in M) <= capacity[i] * X[i] for i in N)
        , name="capacity_constr")
    
    return mdl


def main() -> None:
    args = utils.parse_cli()
    
    run_dir, gru_dir, txt_log = utils.prepare_paths(
        method = "MIP",
        environment = "Deterministic",
        category="IR",
        instance = args.instance,
        budget = None,
        out_root = args.output_root
    )
    
    param = utils.load_instance(args.instance, args.data_dir)
    N, M, openning_cost, capacity, demand, shipping_cost, _  = (
        list(range(param["N"])),
        list(range(param["M"])),
        param["c"],
        param["P"],
        param["Dbar"],
        param["d"],
        param["r"]
    )
    
    # ---------- model ------------------------------------------------------
    mdl = utils.init_model(f"MIP_Det_IR_{args.instance}",
                           args.time_limit,
                           gru_dir / f"gurobi_{args.instance}.log",
                           args.tolerance)
    
    mdl.Params.OutputFlag = 1  # silent mode
    mdl.Params.LogToConsole = 0
    mdl.Params.Threads = 1
    
    X, Y = _add_variables(mdl, N, M)
    _add_constraints(mdl, X, Y, demand, capacity, N, M)
        
    obj = gp.quicksum(openning_cost[i] * X[i] for i in N) + \
            gp.quicksum(shipping_cost[i][j] * Y[i, j] for i in N for j in M)
   
    mdl.setObjective(obj, GRB.MINIMIZE)
    mdl.optimize()
    
    runtime = mdl.Runtime
    binaries = {"X": X}
    
    meta = {
        "header": (f"Result - instance {args.instance}\n"
                   f"Facilities: {len(N)} | Customers: {len(M)}"),
        "runtime": runtime,
        "limit": args.time_limit,
        "instance": args.instance,
        "n_facilities": len(N),
        "n_customers": len(M),
        "method": "MIP",
        "uncertainty": "Deterministic",
        "overall_time": runtime,
        "tolerance": args.tolerance,
    }
    
    summary = utils.log_solution(mdl, txt_log, meta, binaries)
    # utils.write_summary(summary, run_dir / "summary_Deterministic_MIP.csv")

    print(f"[MIP] run complete → {txt_log}")
    
    
if __name__ == '__main__':
    main()