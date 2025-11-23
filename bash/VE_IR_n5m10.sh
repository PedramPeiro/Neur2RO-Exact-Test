#!/bin/bash
#SBATCH --account=def-daryalal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=15          # number of simultaneous tasks you want
#SBATCH --cpus-per-task=1
#SBATCH --time=2:00:00
#SBATCH --job-name=VE_IR_n5m10
#SBATCH --output=VE_IR_n5m10_%j.out
#SBATCH --mail-type=END,FAIL

set -euo pipefail

# Robust default for interactive/debug sessions (so set -u won't trip)
export SLURM_NTASKS_PER_NODE="${SLURM_NTASKS_PER_NODE:-15}"

echo "Starting job ${SLURM_JOB_NAME:-debugjob}, ID ${SLURM_JOBID:-N/A}"
echo "Running on node(s): ${SLURM_NODELIST:-$(hostname)}"
echo "Using ${SLURM_NTASKS_PER_NODE} parallel tasks"

# Env
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
cat > run_VE_IR_n5m10.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

instance="$1"
Gamma="$2"

echo "[$(date +%T)] start ${instance} (Gamma=${Gamma}) on $(hostname)"

"$HOME/.venvs/exact2ro/bin/python" -u VE_IR.py \
  --instance "${instance}" \
  --time_limit 1800 \
  --data_dir "data/instances_IR" \
  --tolerance 1e-6 \
  --Gamma "${Gamma}" \
  --output_root "TrilliumResults"

echo "[$(date +%T)] done  ${instance} (Gamma=${Gamma})"
EOF

chmod +x run_VE_IR_n5m10.sh

# ===== Instance & Gamma lists =====
# Change the brace range if you later want a different rep range
INSTANCES=(IR_n5_m10_rep{0..29})

# You can freely edit this list of Gammas
GAMMAS=(1 2 3 4 5 6 7 8 9 10)

# ===== Run with GNU parallel =====
NPROCS="${SLURM_NTASKS_PER_NODE}"
echo "Running GNU parallel with -j ${NPROCS}"

parallel -j "${NPROCS}" --verbose \
  --joblog TrilliumResults/VE_IR_n5m10.log \
  --results TrilliumResults/parallel_out/{#}_{1}_G{2} \
  --wd "$PWD" \
  --halt now,fail=1 \
  --ungroup \
  ./run_VE_IR_n5m10.sh {1} {2} ::: "${INSTANCES[@]}" ::: "${GAMMAS[@]}"

echo "All VE_IR instances processed (up to ${NPROCS} concurrent)."
tail -n 20 TrilliumResults/VE_IR_n5m10.log || true
