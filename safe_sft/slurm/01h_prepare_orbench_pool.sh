#!/bin/bash
# ===========================================================================
# Job 01h — Pool de safety AUMENTADO anti-over-refusal vía OR-Bench (self-align).
# NO toca nada existente.
#   data/orbench_helpful       respuestas ÚTILES self-generadas a prompts OR-Bench
#   data/safety_pool_orbench   = beavertails_selfalign_train + 25% orbench_helpful
#
# Necesita GPU (paso de generación). Prerequisito: beavertails_selfalign_train (07).
# Lanzar (l40):  sbatch --partition=<PART_L40> --gres=<GRES_L40> slurm/01h_prepare_orbench_pool.sh
# ===========================================================================
#SBATCH --job-name=prep_orbench
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=06:00:00
#SBATCH --output=logs/prep_orbench_%j.out
#SBATCH --error=logs/prep_orbench_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
POOL_SA="$D/beavertails_selfalign_train"
HELPFUL="$D/orbench_helpful"
POOL_OUT="$D/safety_pool_orbench"
BASE_MODEL="$WORK_DIR/models/Llama-3.2-3B-Instruct"
_check() { [[ -d "$1" ]] && [[ -f "$1/dataset_info.json" ]]; }

_check "$POOL_SA" || { echo "ERROR: falta $POOL_SA (corre 07_gen_selfalign)"; exit 1; }

# 1) generar respuestas útiles self-align sobre OR-Bench
if _check "$HELPFUL"; then
    echo "$HELPFUL ya existe — saltando generación"
else
    python data/gen_orbench_helpful.py \
        --base_model "$BASE_MODEL" \
        --output_dir "$HELPFUL" \
        --orbench_config "${ORBENCH_CONFIG:-or-bench-80k}" \
        --max_prompts "${MAX_PROMPTS:-6000}" \
        --n_candidates "${N_CANDIDATES:-2}" \
        ${BEST_OF_N:+--best_of_n} \
        --temperature 0.7 --seed 42
fi

# 2) construir pool aumentado al 25%
if _check "$POOL_OUT"; then
    echo "$POOL_OUT ya existe — saltando build"
else
    python data/build_safety_pool.py \
        --selfalign_pool "$POOL_SA" \
        --contrast_dataset "$HELPFUL" \
        --output_dir "$POOL_OUT" \
        --contrast_frac "${CONTRAST_FRAC:-0.25}" --seed 42
fi

echo "Listo: $HELPFUL  y  $POOL_OUT"
