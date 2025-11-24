#!/bin/bash
#SBATCH --account=def-daryalal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --job-name=VE_IR_n5m15
#SBATCH --output=VE_IR_n5m15_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

export SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-15}"

echo "Starting job ${SLURM_JOB_NAME:-debugjob}, ID ${SLURM_JOBID:-N/A}"
echo "Running on node(s): ${SLURM_NODELIST:-$(hostname)}"
echo "Using ${SLURM_NTASKS_PER_NODE} parallel tasks"

# ===== Environment =====
module load StdEnv/2023
module load python/3.10.13
module use /opt/software/commercial/modules
module load gurobi/12.0.2 2>/dev/null || echo "WARNING: gurobi module not loaded"

# Activate virtual environment
source "$HOME/.venvs/exact2ro/bin/activate" || echo "Warning: venv activation failed"

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ===== Workdir + outputs =====
cd "$SCRATCH/exact2ro" || { echo "Workdir not found"; exit 1; }
mkdir -p TrilliumResults
chmod 750 TrilliumResults

# ===== Inline worker script =====
cat > run_VE_IR_n5m15_worker.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

instance="$1"
Gamma="$2"

echo "[$(date +%T)] start ${instance} (Gamma=${Gamma}) on $(hostname)"

"$HOME/.venvs/exact2ro/bin/python" -u VE_IR.py \
  --instance "${instance}" \
  --time_limit 7200 \
  --data_dir "data/instances_IR" \
  --tolerance 1e-6 \
  --Gamma "${Gamma}" \
  --output_root "TrilliumResults"

echo "[$(date +%T)] done ${instance} (Gamma=${Gamma})"
EOF

chmod +x run_VE_IR_n5m15_worker.sh

# ===== Explicit (Instance, Gamma) pairs =====
INSTANCES=(
IR_n5_m15_rep0
IR_n5_m15_rep1
IR_n5_m15_rep15
IR_n5_m15_rep15
)

GAMMAS=(
4
15
4
15
)

# ===== Run with GNU parallel — zipped lists =====
NPROCS="${SLURM_NTASKS_PER_NODE}"
echo "Running GNU parallel with -j ${NPROCS}"

parallel -j "${NPROCS}" --verbose \
  --joblog TrilliumResults/VE_IR_n5m15.log \
  --results TrilliumResults/parallel_out/{#}_{1}_G{2} \
  --wd "$PWD" \
  --halt now,fail=1 \
  --ungroup \
  ./run_VE_IR_n5m15_worker.sh {1} {2} ::: "${INSTANCES[@]}" :::+ "${GAMMAS[@]}"

echo "All selected VE_IR n5m15 instance-Gamma pairs processed."
tail -n 20 TrilliumResults/VE_IR_n5m15.log || true
