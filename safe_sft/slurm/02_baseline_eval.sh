#!/bin/bash
# ===========================================================================
# Job 02 — Evaluación baseline de seguridad (Llama 3.2 3B sin fine-tuning)
# Lanzar:  sbatch slurm/02_baseline_eval.sh
# ===========================================================================
#SBATCH --job-name=baseline_eval
#SBATCH --partition=gpu
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=02:30:00
#SBATCH --output=logs/baseline_eval_%j.out
#SBATCH --error=logs/baseline_eval_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda

if [[ "${SLURM_SUBMIT_DIR:-$PWD}" != "$PROJECT_DIR" ]]; then
    echo "AVISO: relanza con: cd $PROJECT_DIR && sbatch slurm/$(basename "$0")" >&2
fi
cd "$PROJECT_DIR"

MODEL_PATH="$WORK_DIR/models/Llama-3.2-3B-Instruct"
OUTPUT_PATH="$WORK_DIR/results/baseline/harmbench.json"
mkdir -p "$(dirname "$OUTPUT_PATH")"

echo "=== Baseline HarmBench Evaluation ==="
echo "Model:   $MODEL_PATH"
echo "Output:  $OUTPUT_PATH"
echo "Host:    $(hostname)"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

python eval/run_harmbench.py \
    --base_model "$MODEL_PATH" \
    --output_path "$OUTPUT_PATH" \
    --batch_size 16 \
    --max_new_tokens 512 \
    --step 0 \
    --epoch 0.0

echo ""
echo "=== Baseline eval completo ==="

# Backup en el repo (más fácil de versionar/inspeccionar)
PROJ_RESULTS="$PROJECT_DIR/results/baseline"
mkdir -p "$PROJ_RESULTS"
cp "$OUTPUT_PATH" "$PROJ_RESULTS/harmbench.json"
echo "Backup en: $PROJ_RESULTS/harmbench.json"

echo ""
echo "Resumen rápido:"
python - <<EOF
import json
with open("$OUTPUT_PATH") as f:
    data = json.load(f)
agg = data.get("results", {}).get("aggregate", {})
if agg:
    print(f"  ASR:           {agg.get('asr', 'N/A')}")
    print(f"  Harmless rate: {agg.get('harmless_rate', 'N/A')}")
    print(f"  N behaviors:   {agg.get('n_behaviors', 'N/A')}")
else:
    print("  (no se ha producido bloque 'aggregate' — revisa el JSON completo)")
EOF
