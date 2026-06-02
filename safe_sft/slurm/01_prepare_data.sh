#!/bin/bash
# ===========================================================================
# Job 01 — Preprocesado de datos (Alpaca + BeaverTails + split + mezcla 15%)
# Lanzar:  sbatch slurm/01_prepare_data.sh
#
# Flujo:
#   1. Descarga Alpaca completo                 → data/alpaca/
#   2. Descarga BeaverTails-30k (safe)         → data/beavertails_safe/
#   3. Split Alpaca 95/5 train/val             → data/alpaca_train/, data/alpaca_val/
#   4. Mezcla alpaca_train + BeaverTails 15%   → data/alpaca_mixed_15/
#
#   Exp A:  train=alpaca_train  val=alpaca_val
#   Exp B:  train=alpaca_mixed_15  val=alpaca_val  (mismo val que A → comparable)
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
# 1. Alpaca completo (sin split)
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/alpaca" ]] && [[ -f "$DATA_DIR/alpaca/dataset_info.json" ]]; then
    echo "Alpaca ya preparado — saltando"
else
    echo "=== 1) Preparando Alpaca completo ==="
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
    echo "=== 2) Preparando BeaverTails-30k (safe) ==="
    python data/prepare_beavertails.py \
        --output_dir "$DATA_DIR/beavertails_safe" \
        --split 30k_train \
        --num_proc 8
fi

# --------------------------------------------------------------------------
# 3. Split Alpaca 95/5 train/val
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/alpaca_train" ]] && [[ -d "$DATA_DIR/alpaca_val" ]]; then
    echo "Alpaca train/val ya particionado — saltando"
else
    echo "=== 3) Particionando Alpaca 95% train / 5% val (seed=42) ==="
    python data/split_train_val.py \
        --input_dir   "$DATA_DIR/alpaca" \
        --train_output "$DATA_DIR/alpaca_train" \
        --val_output   "$DATA_DIR/alpaca_val" \
        --val_ratio 0.05 \
        --seed 42
fi

# --------------------------------------------------------------------------
# 4. Mezcla alpaca_train + BeaverTails al 15%
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/alpaca_mixed_15" ]] && [[ -f "$DATA_DIR/alpaca_mixed_15/dataset_info.json" ]]; then
    echo "Mixed 15% ya preparado — saltando"
else
    echo "=== 4) Mezclando alpaca_train + BeaverTails (ratio=0.15) ==="
    python data/mix_datasets.py \
        --base   "$DATA_DIR/alpaca_train" \
        --safety "$DATA_DIR/beavertails_safe" \
        --ratio 0.15 \
        --seed 42 \
        --output_dir "$DATA_DIR/alpaca_mixed_15"
fi

# --------------------------------------------------------------------------
# 5. MetaMathQA (submuestreado a 52k para igualar Alpaca)
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/metamath" ]] && [[ -f "$DATA_DIR/metamath/dataset_info.json" ]]; then
    echo "MetaMathQA ya preparado — saltando"
else
    echo "=== 5) Preparando MetaMathQA submuestreado a 52k ==="
    python data/prepare_metamath.py \
        --output_dir "$DATA_DIR/metamath" \
        --target_size 52000 \
        --seed 42 \
        --num_proc 8
fi

# --------------------------------------------------------------------------
# 6. Split MetaMathQA 95/5 train/val
# --------------------------------------------------------------------------
if [[ -d "$DATA_DIR/metamath_train" ]] && [[ -d "$DATA_DIR/metamath_val" ]]; then
    echo "MetaMath train/val ya particionado — saltando"
else
    echo "=== 6) Particionando MetaMath 95% train / 5% val (seed=42) ==="
    python data/split_train_val.py \
        --input_dir    "$DATA_DIR/metamath" \
        --train_output "$DATA_DIR/metamath_train" \
        --val_output   "$DATA_DIR/metamath_val" \
        --val_ratio 0.05 \
        --seed 42
fi

# --------------------------------------------------------------------------
# Resumen
# --------------------------------------------------------------------------
echo ""
echo "=== Tamaños de los datasets ==="
python - <<EOF
from datasets import load_from_disk
import os
for name in ["alpaca", "alpaca_train", "alpaca_val", "alpaca_mixed_15",
             "beavertails_safe", "metamath", "metamath_train", "metamath_val"]:
    p = os.path.join("$DATA_DIR", name)
    if os.path.isdir(p):
        ds = load_from_disk(p)
        print(f"  {name:<22} {len(ds):>7} ejemplos")
EOF

echo ""
echo "Siguientes pasos:"
echo "  sbatch slurm/02_baseline_eval.sh   (ya hecho en Fase 0)"
echo "  bash   slurm/pipeline.sh exp_a"
