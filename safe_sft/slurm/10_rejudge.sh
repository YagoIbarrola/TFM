#!/bin/bash
# ===========================================================================
# Job 10 — Re-juzga las completions guardadas (held-out) del PRIMER y ÚLTIMO
# checkpoint de un experimento con las 3 variantes de juez, sin regenerar.
# Útil para comparar prompts de juez en el inicio vs final de un SFT.
#
# Uso: EXP=exp_a sbatch --partition=gpu --gres=gpu:nvidia_l40s:1 slurm/10_rejudge.sh
# ===========================================================================
#SBATCH --job-name=rejudge
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=01:00:00
#SBATCH --output=logs/rejudge_%x_%j.out
#SBATCH --error=logs/rejudge_%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=...}"
EVALS="$WORK_DIR/results/$EXP/evals"
FIRST_DIR=$(ls -d "$EVALS"/checkpoint-[0-9]* 2>/dev/null | sort -t- -k2 -n | head -1)
FIRST="$FIRST_DIR/bt_asr.json"
LAST="$EVALS/checkpoint-final/bt_asr.json"

for f in "$FIRST" "$LAST"; do
    [[ -f "$f" ]] || { echo "ERROR: falta $f (¿generaste el held-out con $EXP?)" >&2; exit 1; }
done
echo "Inicio: $FIRST"
echo "Final:  $LAST"

python eval/rejudge.py \
    --inputs "$FIRST" "$LAST" \
    --judge_model "$WORK_DIR/models/Llama-3.2-3B-Instruct" \
    --variants "${VARIANTS:-refuse,harm,behavior}" \
    --output "$WORK_DIR/results/$EXP/rejudge.json"

mkdir -p "$PROJECT_DIR/results/$EXP"
cp "$WORK_DIR/results/$EXP/rejudge.json" "$PROJECT_DIR/results/$EXP/rejudge.json"
echo "=== rejudge guardado en results/$EXP/rejudge.json ==="
