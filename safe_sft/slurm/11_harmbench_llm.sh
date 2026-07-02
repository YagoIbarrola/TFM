#!/bin/bash
# ===========================================================================
# Job 11 — HarmBench con juez LLM, NO destructivo. Re-juzga las completions de
# harmbench.json (keyword) con un juez LLM → harmbench_llm.json, y agrega a
# security_curve_llm.csv. Los datos keyword (harmbench.json, security_curve.csv)
# quedan intactos.
#
# Uso: EXP=exp_a VARIANT=harm sbatch --partition=gpu --gres=gpu:nvidia_l40s:1 slurm/11_harmbench_llm.sh
# ===========================================================================
#SBATCH --job-name=harmbench_llm
#SBATCH --partition=gpuMax
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_h100_nvl:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/harmbench_llm_%x_%j.out
#SBATCH --error=logs/harmbench_llm_%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=...}"
VARIANT="${VARIANT:-harm}"
EXP_DIR="$WORK_DIR/results/$EXP"

# 1) re-juzga (escribe harmbench_llm.json al lado de cada harmbench.json)
python eval/judge_harmbench_llm.py \
    --exp_dir "$EXP_DIR" \
    --judge_model "$WORK_DIR/models/Llama-3.2-3B-Instruct" \
    --variant "$VARIANT"

# 2) agrega -> security_curve_llm.csv (NO toca security_curve.csv keyword)
BASE_ARGS=()
BASELINE_LLM="$WORK_DIR/results/baseline/harmbench_llm.json"
[[ -f "$BASELINE_LLM" ]] && BASE_ARGS=(--baseline_json "$BASELINE_LLM")
python eval/aggregate_results.py \
    --exp_dir "$EXP_DIR" \
    --json_name harmbench_llm.json \
    "${BASE_ARGS[@]}" \
    --output_csv "$EXP_DIR/security_curve_llm.csv"

mkdir -p "$PROJECT_DIR/results/$EXP"
cp "$EXP_DIR/security_curve_llm.csv" "$PROJECT_DIR/results/$EXP/security_curve_llm.csv"
echo "=== HarmBench-LLM ($VARIANT) en security_curve_llm.csv (keyword intacto) ==="
