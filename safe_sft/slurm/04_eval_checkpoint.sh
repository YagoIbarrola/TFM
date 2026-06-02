#!/bin/bash
# ===========================================================================
# Job 04 — Evaluación HarmBench de un checkpoint (array job)
# Uso:    sbatch --export=EXP=exp_a --array=0-25 slurm/04_eval_checkpoint.sh
#
# Cada tarea del array lee la línea SLURM_ARRAY_TASK_ID+1 del checkpoint_list.txt
# y evalúa ese checkpoint. Si la línea no existe (porque hay menos checkpoints
# que tareas en el array), la tarea termina limpiamente.
# ===========================================================================
#SBATCH --job-name=eval_chkpt
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_l4:1
#SBATCH --time=03:00:00
#SBATCH --output=logs/eval_%x_%A_%a.out
#SBATCH --error=logs/eval_%x_%A_%a.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=exp_a o EXP=exp_b}"
CKPT_LIST="$WORK_DIR/results/$EXP/checkpoint_list.txt"

if [[ ! -f "$CKPT_LIST" ]]; then
    echo "ERROR: no existe $CKPT_LIST. ¿Ha terminado el entrenamiento?" >&2
    exit 1
fi

LINE_NUM=$((SLURM_ARRAY_TASK_ID + 1))
CHECKPOINT=$(sed -n "${LINE_NUM}p" "$CKPT_LIST")

if [[ -z "$CHECKPOINT" ]]; then
    echo "Sin checkpoint en el índice $SLURM_ARRAY_TASK_ID (lista tiene $(wc -l < $CKPT_LIST) líneas). Saltando."
    exit 0
fi

CKPT_NAME=$(basename "$CHECKPOINT")
OUTPUT_PATH="$WORK_DIR/results/$EXP/$CKPT_NAME/harmbench.json"
mkdir -p "$(dirname "$OUTPUT_PATH")"

# Extraer step desde el nombre (checkpoint-1500 → 1500)
STEP=$(echo "$CKPT_NAME" | grep -oP '\d+' || echo "")
if [[ -z "$STEP" ]]; then
    STEP=999999      # para "checkpoint-final"
fi

# Epoch aproximado: step / (n_train / batch_size_eff)
# Se calcula desde Python con el tamaño real del dataset
EPOCH=$(python - <<EOF
from datasets import load_from_disk
import os
ds = load_from_disk(os.path.join("$WORK_DIR", "data", "alpaca_train" if "$EXP" == "exp_a" else "alpaca_mixed_15"))
n = len(ds)
steps_per_epoch = max(1, n // 16)
print(f"{$STEP / steps_per_epoch:.4f}")
EOF
)

echo "=== Eval HarmBench ==="
echo "Exp:        $EXP"
echo "Checkpoint: $CHECKPOINT"
echo "Step:       $STEP   Epoch: $EPOCH"
echo "Output:     $OUTPUT_PATH"

python eval/run_harmbench.py \
    --base_model "$WORK_DIR/models/Llama-3.2-3B-Instruct" \
    --adapter_path "$CHECKPOINT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size 16 \
    --max_new_tokens 512 \
    --step "$STEP" \
    --epoch "$EPOCH"

echo "=== Eval done for $CKPT_NAME ==="
