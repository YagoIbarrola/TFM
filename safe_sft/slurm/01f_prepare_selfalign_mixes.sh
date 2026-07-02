#!/bin/bash
# ===========================================================================
# Job 01f — Mezclas estáticas con el pool self-align (Alpaca y MetaMath).
#   data/alpaca_selfalign_5     Alpaca + 5%  self-align
#   data/metamath_selfalign_5   MetaMath + 5%  self-align
#   data/metamath_selfalign_15  MetaMath + 15% self-align
# (alpaca_selfalign_15 ya lo crea 07_gen_selfalign)
#
# Prerequisito: beavertails_selfalign_train (07), alpaca_train, metamath_train.
# Lanzar: sbatch slurm/01f_prepare_selfalign_mixes.sh   (o en login si cpu caído)
# ===========================================================================
#SBATCH --job-name=prep_sa_mixes
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=logs/prep_sa_mixes_%j.out
#SBATCH --error=logs/prep_sa_mixes_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

D="$WORK_DIR/data"
POOL="$D/beavertails_selfalign_train"
_check() { [[ -d "$1" ]] && [[ -f "$1/dataset_info.json" ]]; }

for req in "$POOL" "$D/alpaca_train" "$D/metamath_train"; do
    _check "$req" || { echo "ERROR: falta $req (corre 07_gen_selfalign / 01_prepare_data)"; exit 1; }
done

mix() {
    local base="$1" ratio="$2" out="$3"
    if _check "$D/$out"; then echo "$out ya existe — saltando"; else
        echo "=== mix $out (ratio $ratio) ==="
        python data/mix_datasets.py --base "$D/$base" --safety "$POOL" \
            --ratio "$ratio" --seed 42 --output_dir "$D/$out"
    fi
}

mix alpaca_train   0.05  alpaca_selfalign_5
mix metamath_train 0.05  metamath_selfalign_5
mix metamath_train 0.15  metamath_selfalign_15

echo "Listo."
