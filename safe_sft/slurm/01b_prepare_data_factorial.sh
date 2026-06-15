#!/bin/bash
# ===========================================================================
# Job 01b — Prepara los 3 datasets nuevos del diseño factorial.
# Prerequisito: 01_prepare_data.sh ya ejecutado (alpaca_train, metamath_train
# y beavertails_clean deben existir en $WORK_DIR/data/).
#
# Datasets generados:
#   data/alpaca_beavertails_clean_5    Alpaca + 5%  BeaverTails clean
#   data/metamath_beavertails_clean_5  MetaMath + 5%  BeaverTails clean
#   data/metamath_beavertails_clean_15 MetaMath + 15% BeaverTails clean
#
# Lanzar: sbatch slurm/01b_prepare_data_factorial.sh
# ===========================================================================
#SBATCH --job-name=prep_factorial
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=01:00:00
#SBATCH --output=logs/prep_factorial_%j.out
#SBATCH --error=logs/prep_factorial_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"

_check() { [[ -d "$1" ]] && [[ -f "$1/dataset_info.json" ]]; }

# --------------------------------------------------------------------------
# Validar prerequisites
# --------------------------------------------------------------------------
for req in alpaca_train metamath_train beavertails_clean; do
    if ! _check "$D/$req"; then
        echo "ERROR: $D/$req no existe. Lanza primero 01_prepare_data.sh" >&2
        exit 1
    fi
done

# --------------------------------------------------------------------------
# 1. Alpaca + 5% BeaverTails clean
# --------------------------------------------------------------------------
if _check "$D/alpaca_beavertails_clean_5"; then
    echo "alpaca_beavertails_clean_5 ya existe — saltando"
else
    echo "=== 1) Alpaca + 5% BeaverTails clean ==="
    python data/mix_datasets.py \
        --base   "$D/alpaca_train" \
        --safety "$D/beavertails_clean" \
        --ratio  0.05 \
        --seed   42 \
        --output_dir "$D/alpaca_beavertails_clean_5"
fi

# --------------------------------------------------------------------------
# 2. MetaMath + 5% BeaverTails clean
# --------------------------------------------------------------------------
if _check "$D/metamath_beavertails_clean_5"; then
    echo "metamath_beavertails_clean_5 ya existe — saltando"
else
    echo "=== 2) MetaMath + 5% BeaverTails clean ==="
    python data/mix_datasets.py \
        --base   "$D/metamath_train" \
        --safety "$D/beavertails_clean" \
        --ratio  0.05 \
        --seed   42 \
        --output_dir "$D/metamath_beavertails_clean_5"
fi

# --------------------------------------------------------------------------
# 3. MetaMath + 15% BeaverTails clean
# --------------------------------------------------------------------------
if _check "$D/metamath_beavertails_clean_15"; then
    echo "metamath_beavertails_clean_15 ya existe — saltando"
else
    echo "=== 3) MetaMath + 15% BeaverTails clean ==="
    python data/mix_datasets.py \
        --base   "$D/metamath_train" \
        --safety "$D/beavertails_clean" \
        --ratio  0.15 \
        --seed   42 \
        --output_dir "$D/metamath_beavertails_clean_15"
fi

# --------------------------------------------------------------------------
# Resumen
# --------------------------------------------------------------------------
echo ""
echo "=== Tamaños de los datasets nuevos ==="
python - <<EOF
from datasets import load_from_disk
import os
for name in [
    "alpaca_beavertails_clean_5",
    "metamath_beavertails_clean_5",
    "metamath_beavertails_clean_15",
]:
    p = os.path.join("$D", name)
    if os.path.isdir(p):
        ds = load_from_disk(p)
        print(f"  {name:<35} {len(ds):>7} ejemplos")
    else:
        print(f"  {name:<35}  MISSING")
EOF

echo ""
echo "Listo. Lanza el pipeline factorial con:"
echo "  bash slurm/pipeline_factorial.sh           # default L40"
echo "  GPU=h100 bash slurm/pipeline_factorial.sh  # en H100"
