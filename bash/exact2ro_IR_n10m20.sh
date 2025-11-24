#!/bin/bash
#SBATCH --account=def-daryalal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=10:00:00
#SBATCH --job-name=Exact2RO_IR_n10m20
#SBATCH --output=Exact2RO_IR_n10m20_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Robust default for interactive/debug sessions (so set -u won't trip)
export SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-15}"

echo "Starting job ${SLURM_JOB_NAME:-debugjob}, ID ${SLURM_JOBID:-N/A}"
echo "Running on node(s): ${SLURM_NODELIST:-$(hostname)}"
echo "Using ${SLURM_NTASKS_PER_NODE} parallel tasks"

# ===== Environment =====
module load StdEnv/2023
module load python/3.10.13
module use /opt/software/commercial/modules
module load gurobi/12.0.2 2>/dev/null || echo "WARNING: gurobi module not loaded"

# Activate your venv
source "$HOME/.venvs/exact2ro/bin/activate" || echo "Warning: venv activation failed"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ===== Workdir + outputs =====
cd "$SCRATCH/exact2ro" || { echo "Workdir not found"; exit 1; }

mkdir -p TrilliumResults
chmod 750 TrilliumResults

# ===== Inline worker script =====
cat > run_exact2ro_IR_n10m20_worker.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

instance="$1"
Gamma="$2"

echo "[$(date +%T)] start ${instance} (Gamma=${Gamma}) on $(hostname)"

"$HOME/.venvs/exact2ro/bin/python" -u exact2ro_IR.py \
  --instance "${instance}" \
  --time_limit 7200 \
  --data_dir "data/instances_IR" \
  --tolerance 1e-6 \
  --Gamma "${Gamma}" \
  --output_root "TrilliumResults"

echo "[$(date +%T)] done ${instance} (Gamma=${Gamma})"
EOF

chmod +x run_exact2ro_IR_n10m20_worker.sh

# ===== Instance & Gamma lists =====
INSTANCES=(IR_n10_m20_rep{0..29})
GAMMAS=(1 2)

# ===== Run in parallel =====
NPROCS="${SLURM_NTASKS_PER_NODE:-10}"
echo "Running GNU parallel with -j ${NPROCS}"

# Temporarily disable 'exit on error' so a non-zero exit from parallel
# does not kill the whole SLURM job.
set +e
parallel -j "${NPROCS}" --verbose \
  --joblog TrilliumResults/exact2ro_IR_n10m20.log \
  --results TrilliumResults/parallel_out/{#}_{1}_G{2} \
  --wd "$PWD" \
  --ungroup \
  ./run_exact2ro_IR_n10m20_worker.sh {1} {2} ::: "${INSTANCES[@]}" ::: "${GAMMAS[@]}"
parallel_exit=$?
set -e

if (( parallel_exit != 0 )); then
    echo "WARNING: GNU parallel reported failures (exit code = ${parallel_exit})."
    echo "Check TrilliumResults/exact2ro_IR_n10m20.log for which jobs failed."
else
    echo "All exact2ro_IR instances processed successfully (up to ${NPROCS} concurrent)."
fi

tail -n 20 TrilliumResults/exact2ro_IR_n10m20.log || true
