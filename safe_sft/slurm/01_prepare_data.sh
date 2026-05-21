#!/bin/bash
# ===========================================================================
# Job 01 — Preprocesado de datos (Alpaca + BeaverTails + mezcla 15%)
# Lanzar:  sbatch slurm/01_prepare_data.sh
# ===========================================================================
#SBATCH --job-name=prepare_data
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=logs/prepare_data_%j.out
#SBATCH --error=logs/prepare_data_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda

if [[ "${SLURM_SUBMIT_DIR:-$PWD}" != "$PROJECT_DIR" ]]; then
    echo "AVISO: relanza con: cd $PROJECT_DIR && sbatch slurm/$(basename "$0")" >&2
fi
cd "$PROJECT_DIR"

DATA_DIR="$WORK_DIR/data"
echo "Output dir: $DATA_DIR"
mkdir -p "$DATA_DIR"

# --------------------------------------------------------------------------
# 1. Alpaca
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/alpaca" ]] && [[ -f "$DATA_DIR/alpaca/dataset_info.json" ]]; then
    echo "Alpaca ya preparado — saltando"
else
    echo "=== Preparando Alpaca ==="
    python data/prepare_alpaca.py \
        --output_dir "$DATA_DIR/alpaca" \
        --num_proc 8
fi

# --------------------------------------------------------------------------
# 2. BeaverTails-30k (safe only)
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/beavertails_safe" ]] && [[ -f "$DATA_DIR/beavertails_safe/dataset_info.json" ]]; then
    echo "BeaverTails-safe ya preparado — saltando"
else
    echo "=== Preparando BeaverTails-30k (safe) ==="
    python data/prepare_beavertails.py \
        --output_dir "$DATA_DIR/beavertails_safe" \
        --split 30k_train \
        --num_proc 8
fi

# --------------------------------------------------------------------------
# 3. Mezcla Alpaca + BeaverTails al 15%
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/mixed_15pct" ]] && [[ -f "$DATA_DIR/mixed_15pct/dataset_info.json" ]]; then
    echo "Mixed 15% ya preparado — saltando"
else
    echo "=== Mezclando Alpaca + BeaverTails (ratio=0.15) ==="
    python data/mix_datasets.py \
        --base   "$DATA_DIR/alpaca" \
        --safety "$DATA_DIR/beavertails_safe" \
        --ratio 0.15 \
        --seed 42 \
        --output_dir "$DATA_DIR/mixed_15pct"
fi

# --------------------------------------------------------------------------
# Resumen
# --------------------------------------------------------------------------
echo ""
echo "=== Tamaños de los datasets ==="
python - <<EOF
from datasets import load_from_disk
import os
for name in ["alpaca", "beavertails_safe", "mixed_15pct"]:
    p = os.path.join("$DATA_DIR", name)
    if os.path.isdir(p):
        ds = load_from_disk(p)
        print(f"  {name:<20} {len(ds):>7} ejemplos")
EOF

echo ""
echo "Siguiente:  sbatch slurm/02_baseline_eval.sh"
