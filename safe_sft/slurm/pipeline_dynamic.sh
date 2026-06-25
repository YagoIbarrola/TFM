#!/bin/bash
# ===========================================================================
# Pipeline del régimen DINÁMICO: train(06) → eval array(04) → aggregate(05).
#
# Selección de GPU (default l40):
#   GPU=l40  bash slurm/pipeline_dynamic.sh
#   GPU=h100 bash slurm/pipeline_dynamic.sh
#
# Prerequisitos:
#   01_prepare_data.sh  (alpaca_train, alpaca_val)
#   01e_prepare_asr_split.sh  (beavertails_clean_train, beavertails_asr_heldout)
# ===========================================================================
set -euo pipefail
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: ejecútalo con bash, no sbatch." >&2; exit 1
fi

source "$(dirname "$0")/cluster_config.sh"
cd "$PROJECT_DIR"
export WORK_DIR PROJECT_DIR
[[ -n "${HF_TOKEN:-}" ]]      && export HF_TOKEN
[[ -n "${WANDB_API_KEY:-}" ]] && export WANDB_API_KEY

EXP="${EXP:-exp_alpaca_dynamic}"
ARRAY_SIZE="${ARRAY_SIZE:-30}"
export EXP

# Consistencia juez sensor↔test: el juez del HarmBench del test = el del sensor,
# leído del config (dynamic.bt_judge / bt_judge_prompt). Una única fuente de verdad.
CFG=$(ls configs/${EXP}.yaml configs/${EXP}_*.yaml 2>/dev/null | head -1)
if [[ -n "$CFG" ]]; then
    read HARMBENCH_JUDGE HARMBENCH_JUDGE_VARIANT < <(python - "$CFG" <<'PY'
import sys, yaml
d = (yaml.safe_load(open(sys.argv[1])) or {}).get("dynamic", {})
print(d.get("bt_judge", "keyword"), d.get("bt_judge_prompt", "harm"))
PY
)
    export HARMBENCH_JUDGE HARMBENCH_JUDGE_VARIANT
    echo "Juez (sensor y test): $HARMBENCH_JUDGE / $HARMBENCH_JUDGE_VARIANT"
fi

GPU="${GPU:-l40}"
case "${GPU,,}" in
    l40)  PART="$PARTITION_L40"; GRES="$GRES_L40"
          [[ "$PART" == REVISAR_* ]] && { echo "ERROR: PARTITION_L40 placeholder" >&2; exit 1; } ;;
    h100) PART="$PARTITION_GPU"; GRES="$GRES_GPU" ;;
    *) echo "ERROR: GPU debe ser l40 o h100" >&2; exit 1 ;;
esac

echo "=== Pipeline dinámico: $EXP en ${GPU,,} ($PART / $GRES) ==="

TRAIN=$(sbatch --parsable --job-name="train_${EXP}" \
    --partition="$PART" --gres="$GRES" slurm/06_train_dynamic.sh)
EVAL=$(sbatch --parsable --job-name="eval_${EXP}" \
    --partition="$PART" --gres="$GRES" \
    --dependency=afterok:"$TRAIN" --array=0-$((ARRAY_SIZE - 1)) \
    slurm/04_eval_checkpoint.sh)
AGG=$(sbatch --parsable --job-name="agg_${EXP}" \
    --partition="${AGG_PARTITION:-$PARTITION_CPU}" --dependency=afterany:"$EVAL" \
    slurm/05_aggregate.sh)

echo "train=$TRAIN  eval=$EVAL  agg=$AGG"
echo "Monitor: squeue -u $USER ; tail -f logs/train_train_${EXP}_*.out"
echo "Curva del controlador: $WORK_DIR/results/$EXP/dynamic_log.csv"
