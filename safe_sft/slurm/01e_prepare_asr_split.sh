#!/bin/bash
# ===========================================================================
# Job 01e — Split held-out de BeaverTails para ASR (disjunto del entrenamiento).
# Genera:
#   data/beavertails_clean_train   (pool de safety SIN los prompts reservados)
#   data/beavertails_asr_heldout   (prompts dañinos para medir ASR)
#
# Lanzar: sbatch slurm/01e_prepare_asr_split.sh
# ===========================================================================
#SBATCH --job-name=prep_asr_split
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prep_asr_split_%j.out
#SBATCH --error=logs/prep_asr_split_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
if [[ -d "$D/beavertails_asr_heldout" && -d "$D/beavertails_clean_train" ]]; then
    echo "Split ASR ya existe — saltando"
    exit 0
fi

python data/prepare_beavertails_asr_split.py \
    --output_train   "$D/beavertails_clean_train" \
    --output_heldout "$D/beavertails_asr_heldout" \
    --heldout_size 256 \
    --seed 42

echo "Listo. Lanza el régimen dinámico con:"
echo "  GPU=l40 bash slurm/pipeline_dynamic.sh"
