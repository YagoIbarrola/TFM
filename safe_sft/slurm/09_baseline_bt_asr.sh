#!/bin/bash
# ===========================================================================
# Job 09 — Baseline de bt_asr (held-out BeaverTails) del modelo SIN entrenar.
# Lo guarda con ambos jueces (dato valioso de referencia):
#   results/baseline/bt_asr_keyword.json
#   results/baseline/bt_asr_llm.json
#
# Lanzar: sbatch --partition=gpu --gres=gpu:nvidia_l40s:1 slurm/09_baseline_bt_asr.sh
# ===========================================================================
#SBATCH --job-name=baseline_btasr
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/baseline_btasr_%j.out
#SBATCH --error=logs/baseline_btasr_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

BASE_MODEL="$WORK_DIR/models/Llama-3.2-3B-Instruct"
HELDOUT="$WORK_DIR/data/beavertails_asr_heldout"
OUT="$WORK_DIR/results/baseline"
mkdir -p "$OUT"

for judge in keyword llm; do
    echo "=== baseline bt_asr (juez=$judge) ==="
    python eval/run_beavertails_asr.py \
        --base_model "$BASE_MODEL" \
        --heldout_dataset "$HELDOUT" \
        --output_path "$OUT/bt_asr_${judge}.json" \
        --batch_size 32 \
        --judge "$judge"
done

mkdir -p "$PROJECT_DIR/results/baseline"
cp "$OUT"/bt_asr_*.json "$PROJECT_DIR/results/baseline/" 2>/dev/null || true
echo "=== Baseline bt_asr guardado (keyword + llm) ==="
grep -h '"asr"' "$OUT"/bt_asr_*.json || true
