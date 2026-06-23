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
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=04:00:00
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
# Los JSON de eval (con prompts + completions) van a un subdir 'evals/' SEPARADO
# del checkpoint, para que sobrevivan al borrado de adapters y se pueda re-juzgar
# luego con un juez LLM sin reentrenar.
EVAL_DIR="$WORK_DIR/results/$EXP/evals/$CKPT_NAME"
mkdir -p "$EVAL_DIR"
OUTPUT_PATH="$EVAL_DIR/harmbench.json"

# Extraer step desde el nombre (checkpoint-1500 → 1500)
STEP=$(echo "$CKPT_NAME" | grep -oP '\d+' || echo "")
if [[ -z "$STEP" ]]; then
    STEP=999999      # para "checkpoint-final"
fi

# Derivamos del config YAML: epoch aproximado, ruta del val y si es dominio math.
# El bloque imprime tres líneas: EPOCH | VAL_DATASET | IS_MATH
read -r EPOCH VAL_DATASET IS_MATH < <(python - <<EOF
import glob, os, sys, yaml
from datasets import load_from_disk

configs = sorted(glob.glob(os.path.join("$PROJECT_DIR", "configs", "${EXP}_*.yaml"))
                 + glob.glob(os.path.join("$PROJECT_DIR", "configs", "${EXP}.yaml")))
if not configs:
    print("0.0  data/alpaca_val  0")
    sys.exit(0)
with open(configs[0]) as f:
    cfg = yaml.safe_load(f)
train_path = cfg["train_dataset"]
val_path = cfg["eval_dataset"]
ds = load_from_disk(os.path.join("$WORK_DIR", train_path))
steps_per_epoch = max(1, len(ds) // 16)
epoch = $STEP / steps_per_epoch
is_math = 1 if "metamath" in train_path.lower() else 0
print(f"{epoch:.4f} {val_path} {is_math}")
EOF
)

BASE_MODEL="$WORK_DIR/models/Llama-3.2-3B-Instruct"
CKPT_DIR="$EVAL_DIR"   # xstest.json / task_eval.json / bt_asr.json van aquí también

echo "=== Eval checkpoint $CKPT_NAME ==="
echo "Exp:    $EXP   Step: $STEP   Epoch: $EPOCH"
echo "Val:    $VAL_DATASET   is_math=$IS_MATH"

# --- 1) HarmBench (safety: ASR) ---
echo "--- [1/3] HarmBench ---"
python eval/run_harmbench.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CHECKPOINT" \
    --output_path "$OUTPUT_PATH" \
    --batch_size 16 \
    --max_new_tokens 512 \
    --step "$STEP" \
    --epoch "$EPOCH"

# --- 2) XSTest (over-refusal) ---
echo "--- [2/3] XSTest ---"
python eval/run_xstest.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CHECKPOINT" \
    --output_path "$CKPT_DIR/xstest.json" \
    --batch_size 16 \
    --step "$STEP" \
    --epoch "$EPOCH"

# --- 3) Task eval (perplexity siempre; GSM8K solo si math) ---
echo "--- [3/3] Task eval ---"
GSM8K_FLAG=""
[[ "$IS_MATH" == "1" ]] && GSM8K_FLAG="--gsm8k"
python eval/run_task_eval.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CHECKPOINT" \
    --val_dataset "$WORK_DIR/$VAL_DATASET" \
    --output_path "$CKPT_DIR/task_eval.json" \
    --step "$STEP" \
    --epoch "$EPOCH" \
    $GSM8K_FLAG

# --- 4) BT-ASR (held-out de BeaverTails), si existe el split ---
HELDOUT="$WORK_DIR/data/beavertails_asr_heldout"
if [[ -d "$HELDOUT" ]]; then
    echo "--- [4] BeaverTails ASR (held-out) ---"
    python eval/run_beavertails_asr.py \
        --base_model "$BASE_MODEL" \
        --adapter_path "$CHECKPOINT" \
        --heldout_dataset "$HELDOUT" \
        --output_path "$CKPT_DIR/bt_asr.json" \
        --batch_size 32 \
        --judge "${BT_JUDGE:-keyword}" \
        --step "$STEP" \
        --epoch "$EPOCH"
else
    echo "--- [4] BT-ASR omitido (no existe $HELDOUT) ---"
fi

# --- 5) lm-eval (IFEval / ARC; MMLU opcional vía LMEVAL_TASKS) ---
echo "--- [5] lm-eval (${LMEVAL_TASKS:-ifeval,arc_challenge}) ---"
python eval/run_lmeval.py \
    --base_model "$BASE_MODEL" \
    --adapter_path "$CHECKPOINT" \
    --output_path "$EVAL_DIR/lmeval.json" \
    --tasks "${LMEVAL_TASKS:-ifeval,arc_challenge}" \
    --batch_size 16 \
    --step "$STEP" \
    --epoch "$EPOCH"

echo "=== Eval done for $CKPT_NAME (harmbench + xstest + task + bt_asr + lmeval) ==="
