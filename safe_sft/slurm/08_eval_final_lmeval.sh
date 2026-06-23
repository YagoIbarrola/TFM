#!/bin/bash
# ===========================================================================
# Job 08 — Re-evalúa SOLO checkpoint-final con lm-eval (IFEval + ARC + MMLU),
# sin re-entrenar. Escribe en evals/checkpoint-final/lmeval.json y re-agrega
# task_curve.csv para que aparezcan las columnas nuevas.
#
# MMLU se incluye aquí porque es 1 vez por experimento (coste acotado).
# Uso (vía el loop de abajo o directo):
#   sbatch --partition=gpu --gres=gpu:nvidia_l40s:1 --export=EXP=exp_alpaca_selfalign slurm/08_eval_final_lmeval.sh
# ===========================================================================
#SBATCH --job-name=lmeval_final
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/lmeval_final_%x_%j.out
#SBATCH --error=logs/lmeval_final_%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=...}"
CKPT="$WORK_DIR/results/$EXP/checkpoint-final"
if [[ ! -d "$CKPT" ]]; then
    echo "ERROR: no existe $CKPT (sin adapter final que evaluar)" >&2
    exit 1
fi

EVAL_DIR="$WORK_DIR/results/$EXP/evals/checkpoint-final"
mkdir -p "$EVAL_DIR"
BASE_MODEL="$WORK_DIR/models/Llama-3.2-3B-Instruct"
TASKS="${LMEVAL_TASKS:-ifeval,arc_challenge,mmlu}"

echo "=== lm-eval final: $EXP | tasks=$TASKS ==="
python eval/run_lmeval.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CKPT" \
    --output_path "$EVAL_DIR/lmeval.json" \
    --tasks "$TASKS" \
    --batch_size 16 \
    --step 999999

# Re-agrega task_curve para incorporar las columnas IFEval/ARC/MMLU del final
python eval/aggregate_task.py \
    --exp_dir "$WORK_DIR/results/$EXP" \
    --output_csv "$WORK_DIR/results/$EXP/task_curve.csv"
mkdir -p "$PROJECT_DIR/results/$EXP"
cp "$WORK_DIR/results/$EXP/task_curve.csv" "$PROJECT_DIR/results/$EXP/task_curve.csv"
echo "=== Hecho $EXP ==="
