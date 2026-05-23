#!/bin/bash
# ===========================================================================
# Job 03 — Entrenamiento SFT con LoRA (genérico, dirigido por EXP)
# Uso:    sbatch --export=EXP=exp_a slurm/03_train.sh
# El env var EXP determina el config YAML y el subdirectorio de salida.
# ===========================================================================
#SBATCH --job-name=train_sft
#SBATCH --partition=gpu
#SBATCH --gres=H_100_NVL
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=logs/train_%x_%j.out
#SBATCH --error=logs/train_%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda

if [[ "${SLURM_SUBMIT_DIR:-$PWD}" != "$PROJECT_DIR" ]]; then
    echo "AVISO: relanza con: cd $PROJECT_DIR && sbatch ... slurm/$(basename "$0")" >&2
fi
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=exp_a o EXP=exp_b: sbatch --export=EXP=exp_a ...}"
CONFIG="configs/${EXP}_*.yaml"
CONFIG_FILE=$(ls $CONFIG 2>/dev/null | head -1)

if [[ -z "$CONFIG_FILE" ]]; then
    echo "ERROR: No config match for $CONFIG" >&2
    exit 1
fi

echo "=== Training $EXP ==="
echo "Config: $CONFIG_FILE"
echo "Host:   $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

export WORK_DIR
python train/train.py \
    --config "$CONFIG_FILE" \
    --work_dir "$WORK_DIR" \
    ${RESUME:+--resume}

echo ""
echo "=== Training done ==="
CKPT_LIST="$WORK_DIR/results/$EXP/checkpoint_list.txt"
if [[ -f "$CKPT_LIST" ]]; then
    N=$(wc -l < "$CKPT_LIST")
    echo "Checkpoints: $N"
    echo "List file:   $CKPT_LIST"
fi
