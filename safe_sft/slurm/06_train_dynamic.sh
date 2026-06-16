#!/bin/bash
# ===========================================================================
# Job 06 — Entrenamiento DINÁMICO (train + medición de ASR intercaladas).
# A diferencia de 03_train.sh, el ratio de safety se ajusta por rondas según el
# ASR del held-out de BeaverTails (un solo job de GPU; el eval va dentro).
#
# Uso (vía pipeline_dynamic.sh, que pone partición/gres):
#   sbatch --export=EXP=exp_alpaca_dynamic slurm/06_train_dynamic.sh
# ===========================================================================
#SBATCH --job-name=train_dynamic
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=16:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
EXP="${EXP:?Debes pasar EXP=exp_alpaca_dynamic}"
CONFIG_FILE=$(ls configs/${EXP}_*.yaml configs/${EXP}.yaml 2>/dev/null | head -1)
if [[ -z "$CONFIG_FILE" ]]; then
    echo "ERROR: no hay config para $EXP" >&2; exit 1
fi

echo "=== Train dinámico $EXP ==="
echo "Config: $CONFIG_FILE  |  Host: $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export WORK_DIR
python train/train_dynamic.py --config "$CONFIG_FILE" --work_dir "$WORK_DIR"

echo "=== Dinámico done. Log: $WORK_DIR/results/$EXP/dynamic_log.csv ==="
