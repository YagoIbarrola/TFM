#!/bin/bash
# ===========================================================================
# Job 01i — Mezclas estáticas Alpaca + pool OR-Bench (anti-over-refusal).
#   data/alpaca_orbench_5    alpaca + 5%  safety_pool_orbench
#   data/alpaca_orbench_15   alpaca + 15% safety_pool_orbench
#
# Prerequisito: safety_pool_orbench (01h), alpaca_train (01).
# Lanzar: sbatch slurm/01i_prepare_orbench_mixes.sh   (o en login si cpu caído)
# ===========================================================================
#SBATCH --job-name=prep_orbench_mixes
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prep_orbench_mixes_%j.out
#SBATCH --error=logs/prep_orbench_mixes_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
POOL="$D/safety_pool_orbench"
_check() { [[ -d "$1" ]] && [[ -f "$1/dataset_info.json" ]]; }

for req in "$POOL" "$D/alpaca_train"; do
    _check "$req" || { echo "ERROR: falta $req (corre 01h_prepare_orbench_pool / 01_prepare_data)"; exit 1; }
done

mix() {
    local ratio="$1" out="$2"
    if _check "$D/$out"; then echo "$out ya existe — saltando"; else
        echo "=== mix $out (ratio $ratio) ==="
        python data/mix_datasets.py --base "$D/alpaca_train" --safety "$POOL" \
            --ratio "$ratio" --seed 42 --output_dir "$D/$out"
    fi
}
mix 0.05 alpaca_orbench_5
mix 0.15 alpaca_orbench_15

echo "Listo: alpaca_orbench_5, alpaca_orbench_15"
