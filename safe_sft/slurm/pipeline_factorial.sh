#!/bin/bash
# ===========================================================================
# Pipeline factorial — lanza los experimentos de Llama 3B en L40 o H100.
#
# Diseño: {Alpaca, MetaMath} × {0%, 5%, 15%} + ablación canned (diversidad
# de respuesta). Cada experimento abre una cadena train→eval→agg.
#
# Selección de GPU (variable GPU, default l40):
#   GPU=l40  bash slurm/pipeline_factorial.sh        # L40 48GB (varias)
#   GPU=h100 bash slurm/pipeline_factorial.sh        # H100 96GB (gpuMax)
#
# Subconjunto de experimentos (argumentos posicionales):
#   bash slurm/pipeline_factorial.sh exp_a exp_c            # solo esos
#   GPU=h100 bash slurm/pipeline_factorial.sh exp_e         # exp_e en H100
#
# Slurm encola los que no caben y los lanza según se liberan GPUs.
#
# Prerequisitos:
#   1. 01_prepare_data.sh completado
#   2. 01b_prepare_data_factorial.sh completado   (datasets 5% / 15% math)
#   3. 01c_prepare_data_canned.sh completado      (datasets canned single/pool)
#   4. cluster_config.sh: PARTITION_L40/GRES_L40 (para GPU=l40)
#                         PARTITION_GPU/GRES_GPU (para GPU=h100)
# ===========================================================================
set -euo pipefail

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: pipeline_factorial.sh debe correrse con bash, no con sbatch." >&2
    exit 1
fi

source "$(dirname "$0")/cluster_config.sh"
cd "$PROJECT_DIR"

export WORK_DIR PROJECT_DIR
[[ -n "${HF_HOME:-}" ]]        && export HF_HOME
[[ -n "${HF_TOKEN:-}" ]]       && export HF_TOKEN
[[ -n "${WANDB_API_KEY:-}" ]]  && export WANDB_API_KEY

ARRAY_SIZE="${ARRAY_SIZE:-30}"   # número máximo de checkpoints por experimento

# ---------------------------------------------------------------------------
# Selección de GPU → resuelve partición y GRES
# ---------------------------------------------------------------------------
GPU="${GPU:-l40}"
case "${GPU,,}" in
    l40)
        PART="$PARTITION_L40"
        GRES="$GRES_L40"
        if [[ "$PART" == REVISAR_* ]]; then
            echo "ERROR: PARTITION_L40 sigue siendo un placeholder ('$PART')." >&2
            echo "       Rellénalo en slurm/cluster_config.sh con la partición real" >&2
            echo "       de los nodos L40S:  sinfo -N -o '%N %P %G'" >&2
            exit 1
        fi
        ;;
    h100)
        PART="$PARTITION_GPU"      # gpuMax
        GRES="$GRES_GPU"           # gpu:nvidia_h100_nvl:1
        ;;
    *)
        echo "ERROR: GPU debe ser 'l40' o 'h100' (recibido: '$GPU')" >&2
        exit 1
        ;;
esac

# ---------------------------------------------------------------------------
# Tabla del diseño factorial
#
# --- Factorial principal: {Alpaca, Math} × {0%, 5%, 15%} ---
# exp_a               → Alpaca 0%   (config: exp_a_alpaca_pure.yaml)
# exp_alpaca_p5       → Alpaca 5%   (config: exp_alpaca_p5_beavertails_clean.yaml)
# exp_e               → Alpaca 15%  (config: exp_e_alpaca_beavertails_clean.yaml)
# exp_c               → Math   0%   (config: exp_c_math_pure.yaml)
# exp_math_p5         → Math   5%   (config: exp_math_p5_beavertails_clean.yaml)
# exp_math_p15        → Math   15%  (config: exp_math_p15_beavertails_clean.yaml)
#
# --- Ablación diversidad-de-respuesta (15%, mismos prompts que Exp E) ---
# exp_alpaca_canned_single  → Alpaca 15% negativa única
# exp_alpaca_canned_pool    → Alpaca 15% pool de paráfrasis
# exp_math_canned_single    → Math   15% negativa única
# exp_math_canned_pool      → Math   15% pool de paráfrasis
# ---------------------------------------------------------------------------

FACTORIAL_EXPERIMENTS=(
    exp_a
    exp_alpaca_p5
    exp_e
    exp_c
    exp_math_p5
    exp_math_p15
)

CANNED_EXPERIMENTS=(
    exp_alpaca_canned_single
    exp_alpaca_canned_pool
    exp_math_canned_single
    exp_math_canned_pool
)

# Ablación canned al 5% (mismos prompts; ver 01d_prepare_data_canned5.sh)
CANNED5_EXPERIMENTS=(
    exp_alpaca_canned5_single
    exp_alpaca_canned5_pool
    exp_math_canned5_single
    exp_math_canned5_pool
)

ALL_EXPERIMENTS=(
    "${FACTORIAL_EXPERIMENTS[@]}"
    "${CANNED_EXPERIMENTS[@]}"
    "${CANNED5_EXPERIMENTS[@]}"
)

# Si el usuario pasa argumentos, ejecuta solo esos
if [[ $# -gt 0 ]]; then
    EXPERIMENTS=("$@")
else
    EXPERIMENTS=("${ALL_EXPERIMENTS[@]}")
fi

echo "=== Pipeline factorial ==="
echo "GPU:          ${GPU,,}  →  partición=$PART  gres=$GRES"
echo "Experimentos: ${EXPERIMENTS[*]}"
echo "Array size:   $ARRAY_SIZE checkpoints por exp"
echo ""

for EXP in "${EXPERIMENTS[@]}"; do
    export EXP

    # -----------------------------------------------------------------------
    # 1. Entrenamiento (1 GPU por experimento)
    # -----------------------------------------------------------------------
    TRAIN_JOB=$(sbatch --parsable \
        --job-name="train_${EXP}" \
        --partition="$PART" \
        --gres="$GRES" \
        slurm/03_train.sh)

    # -----------------------------------------------------------------------
    # 2. Evaluación (array, 1 GPU por checkpoint)
    # -----------------------------------------------------------------------
    EVAL_JOB=$(sbatch --parsable \
        --job-name="eval_${EXP}" \
        --partition="$PART" \
        --gres="$GRES" \
        --dependency=afterok:"$TRAIN_JOB" \
        --array=0-$((ARRAY_SIZE - 1)) \
        slurm/04_eval_checkpoint.sh)

    # -----------------------------------------------------------------------
    # 3. Agregación (CPU, no necesita GPU)
    # -----------------------------------------------------------------------
    AGG_JOB=$(sbatch --parsable \
        --job-name="agg_${EXP}" \
        --partition="$PARTITION_CPU" \
        --dependency=afterany:"$EVAL_JOB" \
        slurm/05_aggregate.sh)

    printf "%-26s  train=%-8s  eval=%-8s  agg=%s\n" \
        "$EXP" "$TRAIN_JOB" "$EVAL_JOB" "$AGG_JOB"
done

echo ""
echo "${#EXPERIMENTS[@]} cadenas lanzadas en ${GPU,,}. Monitor:"
echo "  squeue -u $USER --format='%.10i %.20j %.8T %.10M' | sort -k2"
echo ""
echo "Ver logs de entrenamiento:"
for EXP in "${EXPERIMENTS[@]}"; do
    echo "  tail -f logs/train_train_${EXP}_*.out"
done
