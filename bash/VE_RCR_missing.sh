#!/bin/bash
#SBATCH --account=def-daryalal
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=10          # number of simultaneous tasks you want
#SBATCH --cpus-per-task=1
#SBATCH --time=6:00:00
#SBATCH --job-name=VE_RCR_n5m15_sel
#SBATCH --output=VE_RCR_n5m15_sel_%j.out
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
cat > run_VE_RCR_n5m15_sel.sh << 'EOF'
#!/usr/bin/env bash
set -euo pipefail

instance="$1"
Gamma="$2"

echo "[$(date +%T)] start ${instance} (Gamma=${Gamma}) on $(hostname)"

"$HOME/.venvs/exact2ro/bin/python" -u VE_RCR.py \
  --instance "${instance}" \
  --time_limit 7200 \
  --data_dir "data/instances_RCR" \
  --tolerance 1e-6 \
  --Gamma "${Gamma}" \
  --output_root "TrilliumResults"

echo "[$(date +%T)] done  ${instance} (Gamma=${Gamma})"
EOF

chmod +x run_VE_RCR_n5m15_sel.sh

# ===== Explicit (instance, Gamma) pairs =====
INSTANCES=(
RCR_n5_m15_rep0
RCR_n5_m15_rep1
RCR_n5_m15_rep2
RCR_n5_m15_rep3
RCR_n5_m15_rep4
RCR_n5_m15_rep5
RCR_n5_m15_rep6
RCR_n5_m15_rep6
RCR_n5_m15_rep7
RCR_n5_m15_rep8
RCR_n5_m15_rep8
RCR_n5_m15_rep11
RCR_n5_m15_rep11
RCR_n5_m15_rep12
RCR_n5_m15_rep12
RCR_n5_m15_rep13
RCR_n5_m15_rep15
RCR_n5_m15_rep17
RCR_n5_m15_rep18
RCR_n5_m15_rep19
RCR_n5_m15_rep20
RCR_n5_m15_rep20
RCR_n5_m15_rep21
RCR_n5_m15_rep21
RCR_n5_m15_rep22
RCR_n5_m15_rep23
RCR_n5_m15_rep23
RCR_n5_m15_rep24
RCR_n5_m15_rep25
RCR_n5_m15_rep25
RCR_n5_m15_rep26
RCR_n5_m15_rep27
RCR_n5_m15_rep28
RCR_n5_m15_rep29
)

GAMMAS=(
15
15
15
15
15
15
4
15
15
4
15
4
15
4
15
15
15
15
15
15
4
15
4
15
15
4
15
15
4
15
15
15
15
15
)

# ===== Run with GNU parallel (zipped lists) =====
NPROCS="${SLURM_NTASKS_PER_NODE}"
echo "Running GNU parallel with -j ${NPROCS}"

parallel -j "${NPROCS}" --verbose \
  --joblog TrilliumResults/VE_RCR_n5m15_sel.log \
  --results TrilliumResults/parallel_out/{#}_{1}_G{2} \
  --wd "$PWD" \
  --halt now,fail=1 \
  --ungroup \
  ./run_VE_RCR_n5m15_sel.sh {1} {2} ::: "${INSTANCES[@]}" :::+ "${GAMMAS[@]}"

echo "All selected VE_RCR n5m15 instance-Gamma pairs processed."
tail -n 20 TrilliumResults/VE_RCR_n5m15_sel.log || true
