#!/bin/bash
# ===========================================================================
# Orquestador de la cadena de Fase 1 para un experimento.
# Uso:    bash slurm/pipeline.sh exp_a
#         bash slurm/pipeline.sh exp_b
#
# Encadena con --dependency:
#   train  →  eval (array)  →  aggregate
# ===========================================================================
set -euo pipefail

EXP="${1:?Uso: bash slurm/pipeline.sh <exp_a|exp_b>}"
ARRAY_SIZE="${ARRAY_SIZE:-30}"   # margen sobre los ~22 checkpoints esperados

source "$(dirname "$0")/cluster_config.sh"
cd "$PROJECT_DIR"

echo "=== Pipeline para $EXP ==="
echo "PROJECT_DIR: $PROJECT_DIR"
echo "WORK_DIR:    $WORK_DIR"
echo ""

# 1. Train
TRAIN_JOB=$(sbatch --parsable \
    --job-name="train_$EXP" \
    --export=ALL,EXP=$EXP \
    slurm/03_train.sh)
echo "Train job:       $TRAIN_JOB"

# 2. Eval array (depende del entrenamiento)
EVAL_JOB=$(sbatch --parsable \
    --job-name="eval_$EXP" \
    --dependency=afterok:$TRAIN_JOB \
    --array=0-$((ARRAY_SIZE-1)) \
    --export=ALL,EXP=$EXP \
    slurm/04_eval_checkpoint.sh)
echo "Eval array job:  $EVAL_JOB"

# 3. Aggregate (depende de que TERMINE el array, ok o no — afterany)
#    Usamos afterany porque algunas tareas del array pueden terminar como "no checkpoint" (exit 0)
#    pero queremos que el aggregate corra de todas formas.
AGG_JOB=$(sbatch --parsable \
    --job-name="agg_$EXP" \
    --dependency=afterany:$EVAL_JOB \
    --export=ALL,EXP=$EXP \
    slurm/05_aggregate.sh)
echo "Aggregate job:   $AGG_JOB"

echo ""
echo "Pipeline lanzada. Monitor:"
echo "  squeue -u $USER"
echo "  tail -f logs/train_${EXP}_${TRAIN_JOB}.out"
