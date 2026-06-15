#!/bin/bash
# ===========================================================================
# Job 01c — Datasets de la ablación "diversidad de respuesta".
#
# Reutiliza los MISMOS prompts que beavertails_clean (Exp E), sustituyendo la
# respuesta por:
#   - single: una única negativa general  → beavertails_canned_single
#   - pool:   paráfrasis de un pool de ~15 → beavertails_canned_pool
#
# Luego mezcla cada pool de safety con Alpaca y MetaMath al 15% (mismo ratio
# que Exp E, para que la única variable sea el contenido de la respuesta).
#
# Prerequisito: 01_prepare_data.sh ejecutado (alpaca_train, metamath_train).
# Lanzar: sbatch slurm/01c_prepare_data_canned.sh
# ===========================================================================
#SBATCH --job-name=prep_canned
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_canned_%j.out
#SBATCH --error=logs/prep_canned_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
_check() { [[ -d "$1" ]] && [[ -f "$1/dataset_info.json" ]]; }

for req in alpaca_train metamath_train; do
    if ! _check "$D/$req"; then
        echo "ERROR: $D/$req no existe. Lanza primero 01_prepare_data.sh" >&2
        exit 1
    fi
done

# --------------------------------------------------------------------------
# 1. Safety pools con respuesta sustituida (mismos prompts que Exp E)
# --------------------------------------------------------------------------
if _check "$D/beavertails_canned_single"; then
    echo "beavertails_canned_single ya existe — saltando"
else
    echo "=== 1) BeaverTails canned (single refusal) ==="
    python data/prepare_beavertails_filtered.py \
        --output_dir "$D/beavertails_canned_single" \
        --split 30k_train \
        --response_mode single \
        --seed 42 \
        --num_proc 8
fi

if _check "$D/beavertails_canned_pool"; then
    echo "beavertails_canned_pool ya existe — saltando"
else
    echo "=== 2) BeaverTails canned (pool de paráfrasis) ==="
    python data/prepare_beavertails_filtered.py \
        --output_dir "$D/beavertails_canned_pool" \
        --split 30k_train \
        --response_mode pool \
        --seed 42 \
        --num_proc 8
fi

# --------------------------------------------------------------------------
# 2. Mezclas al 15% — {alpaca, metamath} × {single, pool}
# --------------------------------------------------------------------------
mix() {
    local base="$1" safety="$2" out="$3"
    if _check "$D/$out"; then
        echo "$out ya existe — saltando"
    else
        echo "=== mix: $out ==="
        python data/mix_datasets.py \
            --base   "$D/$base" \
            --safety "$D/$safety" \
            --ratio  0.15 \
            --seed   42 \
            --output_dir "$D/$out"
    fi
}

mix alpaca_train   beavertails_canned_single  alpaca_canned_single_15
mix alpaca_train   beavertails_canned_pool    alpaca_canned_pool_15
mix metamath_train beavertails_canned_single  metamath_canned_single_15
mix metamath_train beavertails_canned_pool    metamath_canned_pool_15

# --------------------------------------------------------------------------
# Resumen
# --------------------------------------------------------------------------
echo ""
echo "=== Tamaños ==="
python - <<EOF
from datasets import load_from_disk
import os
for name in [
    "beavertails_canned_single", "beavertails_canned_pool",
    "alpaca_canned_single_15", "alpaca_canned_pool_15",
    "metamath_canned_single_15", "metamath_canned_pool_15",
]:
    p = os.path.join("$D", name)
    if os.path.isdir(p):
        ds = load_from_disk(p)
        print(f"  {name:<32} {len(ds):>7} ejemplos")
    else:
        print(f"  {name:<32}  MISSING")
EOF

echo ""
echo "Listo. Experimentos canned disponibles en pipeline_factorial.sh:"
echo "  exp_alpaca_canned_single  exp_alpaca_canned_pool"
echo "  exp_math_canned_single    exp_math_canned_pool"
