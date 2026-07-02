#!/bin/bash
# ===========================================================================
# Job 07 — Genera el dataset self-align (respuestas ideales de rechazo del propio
# Llama) y lo mezcla con Alpaca al 15% para el experimento estático.
#
# Es inferencia intensiva (best-of-N + juez por pares) → job de GPU.
# Lanzar (vía sbatch con partición/gres; en login si quieres probar pequeño):
#   sbatch --partition=gpu --gres=gpu:nvidia_l40s:1 slurm/07_gen_selfalign.sh
# ===========================================================================
#SBATCH --job-name=gen_selfalign
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=24:00:00
#SBATCH --output=logs/gen_selfalign_%j.out
#SBATCH --error=logs/gen_selfalign_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
POOL="$D/beavertails_selfalign_train"
# ~14k prompts × keep-rate 0.74 ≈ 10.3k ejemplos → suficiente para 15% sobre Alpaca (49k)
MAX_PROMPTS="${MAX_PROMPTS:-14000}"
N_CAND="${N_CAND:-4}"

# 1) Generar el pool self-align (si no existe)
if [[ -d "$POOL" && -f "$POOL/dataset_info.json" ]]; then
    echo "$POOL ya existe — saltando generación"
else
    echo "=== Generando self-align (max_prompts=$MAX_PROMPTS, N=$N_CAND) ==="
    python data/gen_selfalign.py \
        --base_model "$WORK_DIR/models/Llama-3.2-3B-Instruct" \
        --output_dir "$POOL" \
        --split 330k_train --heldout_size 256 --seed 42 \
        --max_prompts "$MAX_PROMPTS" --n_candidates "$N_CAND" --temperature 0.8
fi

# 2) Mezcla con Alpaca al 15% (para exp_alpaca_selfalign)
MIX="$D/alpaca_selfalign_15"
if [[ -d "$MIX" && -f "$MIX/dataset_info.json" ]]; then
    echo "$MIX ya existe — saltando mezcla"
else
    echo "=== Mezclando alpaca_train + self-align (0.15) ==="
    python data/mix_datasets.py \
        --base "$D/alpaca_train" --safety "$POOL" \
        --ratio 0.15 --seed 42 --output_dir "$MIX"
fi

echo "Listo."
echo "  Estático: GPU=l40 bash slurm/pipeline_factorial.sh exp_alpaca_selfalign"
echo "  Dinámico: EXP=exp_dynamic_selfalign GPU=l40 bash slurm/pipeline_dynamic.sh"
