#!/bin/bash
# ===========================================================================
# Orquestador de la cadena de Fase 1 para un experimento.
# Uso:    bash slurm/pipeline.sh exp_a
#         bash slurm/pipeline.sh exp_b
#
# Encadena con --dependency:
#   train  →  eval (array)  →  aggregate
#
# NOTA: no usamos --export en los sbatch porque el cluster falla con
# "user_env_retrieval_failed" cuando se especifica explícitamente.
# En su lugar, exportamos las variables localmente y dejamos que sbatch
# las herede vía su comportamiento default (igual que hace un job de test
# simple que sí funciona en este cluster).
# ===========================================================================
set -euo pipefail

# Guard contra sbatch (esto es un orquestador, debe correr con bash)
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
    echo "ERROR: pipeline.sh debe correrse con bash, no con sbatch:" >&2
    echo "       bash slurm/pipeline.sh $*" >&2
    exit 1
fi

EXP="${1:?Uso: bash slurm/pipeline.sh <exp_a|exp_b|exp_c|exp_d|exp_e>}"
ARRAY_SIZE="${ARRAY_SIZE:-30}"

source "$(dirname "$0")/cluster_config.sh"
cd "$PROJECT_DIR"

# Exportar variables que los scripts SBATCH necesitan. Como pipeline.sh
# corre en login, todas estas variables están aquí y se propagan vía
# el default de --export=ALL que sbatch usa cuando no le pasas --export.
export EXP
export WORK_DIR
export PROJECT_DIR
[[ -n "${HF_HOME:-}" ]]        && export HF_HOME
[[ -n "${HF_TOKEN:-}" ]]       && export HF_TOKEN
[[ -n "${WANDB_API_KEY:-}" ]]  && export WANDB_API_KEY

echo "=== Pipeline para $EXP ==="
echo "PROJECT_DIR: $PROJECT_DIR"
echo "WORK_DIR:    $WORK_DIR"
echo ""

# 1. Train
TRAIN_JOB=$(sbatch --parsable \
    --job-name="train_$EXP" \
    slurm/03_train.sh)
echo "Train job:       $TRAIN_JOB"

# 2. Eval array (depende del entrenamiento)
EVAL_JOB=$(sbatch --parsable \
    --job-name="eval_$EXP" \
    --dependency=afterok:$TRAIN_JOB \
    --array=0-$((ARRAY_SIZE-1)) \
    slurm/04_eval_checkpoint.sh)
echo "Eval array job:  $EVAL_JOB"

# 3. Aggregate
AGG_JOB=$(sbatch --parsable \
    --job-name="agg_$EXP" \
    --dependency=afterany:$EVAL_JOB \
    slurm/05_aggregate.sh)
echo "Aggregate job:   $AGG_JOB"

echo ""
echo "Pipeline lanzada. Monitor:"
echo "  squeue -u $USER"
echo "  tail -f logs/train_${EXP}_${TRAIN_JOB}.out"
