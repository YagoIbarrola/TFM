#!/bin/bash
# ===========================================================================
# Job 01e — Datos del régimen dinámico canned + held-out de ASR (BeaverTails).
# Genera (mismo split → single y pool comparables, disjuntos del held-out):
#   data/beavertails_asr_heldout          prompts dañinos para medir ASR (sensor)
#   data/beavertails_canned_single_train  prompt + negativa única
#   data/beavertails_canned_pool_train    prompt + paráfrasis de un pool
#
# Prompts dañinos = >=1 respuesta is_safe=False en 330k_train, deduplicados.
# Lanzar: sbatch slurm/01e_prepare_asr_split.sh   (o en login si cpu está caído)
# ===========================================================================
#SBATCH --job-name=prep_asr_split
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prep_asr_split_%j.out
#SBATCH --error=logs/prep_asr_split_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
if [[ -d "$D/beavertails_asr_heldout" && -d "$D/beavertails_canned_single_train" \
      && -d "$D/beavertails_canned_pool_train" ]]; then
    echo "Datos del dinámico ya existen — saltando"
    exit 0
fi

python data/prepare_beavertails_asr_split.py \
    --output_heldout "$D/beavertails_asr_heldout" \
    --output_single  "$D/beavertails_canned_single_train" \
    --output_pool    "$D/beavertails_canned_pool_train" \
    --split 330k_train --heldout_size 256 --seed 42

echo "Listo. Lanza los dos runs dinámicos:"
echo "  EXP=exp_dynamic_canned_single GPU=l40 bash slurm/pipeline_dynamic.sh"
echo "  EXP=exp_dynamic_canned_pool   GPU=l40 bash slurm/pipeline_dynamic.sh"
