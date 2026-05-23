#!/bin/bash
# ===========================================================================
# Job 05 — Agregar JSONs por checkpoint en un CSV (security_curve.csv)
# Uso:    sbatch --export=EXP=exp_a slurm/05_aggregate.sh
# ===========================================================================
#SBATCH --job-name=aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate_%x_%j.out
#SBATCH --error=logs/aggregate_%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=exp_a o EXP=exp_b}"
EXP_DIR="$WORK_DIR/results/$EXP"
BASELINE="$WORK_DIR/results/baseline/harmbench.json"
OUT_CSV="$EXP_DIR/security_curve.csv"

echo "=== Aggregating results for $EXP ==="
echo "Exp dir:    $EXP_DIR"
echo "Baseline:   $BASELINE"
echo "Output CSV: $OUT_CSV"

python eval/aggregate_results.py \
    --exp_dir "$EXP_DIR" \
    --baseline_json "$BASELINE" \
    --output_csv "$OUT_CSV"

# Backup en el repo
mkdir -p "$PROJECT_DIR/results/$EXP"
cp "$OUT_CSV" "$PROJECT_DIR/results/$EXP/security_curve.csv"
echo "Backup en: $PROJECT_DIR/results/$EXP/security_curve.csv"
