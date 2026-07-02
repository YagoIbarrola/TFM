#!/bin/bash
# ===========================================================================
# Job 01d — Datasets canned al 5% (mismos pools que 01c, ratio 0.05).
#
# Reutiliza los pools de safety ya creados por 01c:
#   beavertails_canned_single, beavertails_canned_pool
# y los mezcla con Alpaca y MetaMath al 5%.
#
# Prerequisito: 01_prepare_data.sh y 01c_prepare_data_canned.sh ejecutados.
# Lanzar: sbatch slurm/01d_prepare_data_canned5.sh
# ===========================================================================
#SBATCH --job-name=prep_canned5
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:45:00
#SBATCH --output=logs/prep_canned5_%j.out
#SBATCH --error=logs/prep_canned5_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
_check() { [[ -d "$1" ]] && [[ -f "$1/dataset_info.json" ]]; }

for req in alpaca_train metamath_train beavertails_canned_single beavertails_canned_pool; do
    if ! _check "$D/$req"; then
        echo "ERROR: falta $D/$req. Lanza antes 01_prepare_data.sh y 01c_prepare_data_canned.sh" >&2
        exit 1
    fi
done

mix() {
    local base="$1" safety="$2" out="$3"
    if _check "$D/$out"; then
        echo "$out ya existe — saltando"
    else
        echo "=== mix: $out (ratio 0.05) ==="
        python data/mix_datasets.py \
            --base   "$D/$base" \
            --safety "$D/$safety" \
            --ratio  0.05 \
            --seed   42 \
            --output_dir "$D/$out"
    fi
}

mix alpaca_train   beavertails_canned_single  alpaca_canned_single_5
mix alpaca_train   beavertails_canned_pool    alpaca_canned_pool_5
mix metamath_train beavertails_canned_single  metamath_canned_single_5
mix metamath_train beavertails_canned_pool    metamath_canned_pool_5

echo ""
echo "=== Tamaños ==="
python - <<EOF
from datasets import load_from_disk
import os
for name in ["alpaca_canned_single_5", "alpaca_canned_pool_5",
             "metamath_canned_single_5", "metamath_canned_pool_5"]:
    p = os.path.join("$D", name)
    if os.path.isdir(p):
        ds = load_from_disk(p)
        print(f"  {name:<28} {len(ds):>7} ejemplos")
    else:
        print(f"  {name:<28}  MISSING")
EOF

echo ""
echo "Listo. Lanza con:"
echo "  GPU=l40 bash slurm/pipeline_factorial.sh \\"
echo "      exp_alpaca_canned5_single exp_alpaca_canned5_pool \\"
echo "      exp_math_canned5_single exp_math_canned5_pool"
