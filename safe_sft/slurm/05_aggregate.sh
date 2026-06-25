#!/bin/bash
# ===========================================================================
# Job 05 — Agregar JSONs por checkpoint en un CSV (security_curve.csv)
# Uso:    sbatch --export=EXP=exp_a slurm/05_aggregate.sh
# ===========================================================================
#SBATCH --job-name=aggregate
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:15:00
#SBATCH --output=logs/aggregate_%x_%j.out
#SBATCH --error=logs/aggregate_%x_%j.err

set -euo pipefail

source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=exp_a o EXP=exp_b}"
EXP_DIR="$WORK_DIR/results/$EXP"
BASELINE="$WORK_DIR/results/baseline/harmbench.json"
SEC_CSV="$EXP_DIR/security_curve.csv"
TASK_CSV="$EXP_DIR/task_curve.csv"

echo "=== Aggregating results for $EXP ==="
echo "Exp dir:    $EXP_DIR"

# 1) Curva de safety (HarmBench)
echo "--- security_curve.csv ---"
python eval/aggregate_results.py \
    --exp_dir "$EXP_DIR" \
    --baseline_json "$BASELINE" \
    --output_csv "$SEC_CSV"

# 1b) Curva de safety con juez LLM (si existe harmbench_llm.json; no toca la keyword)
if find "$EXP_DIR" -name harmbench_llm.json 2>/dev/null | grep -q .; then
    echo "--- security_curve_llm.csv (juez LLM) ---"
    python eval/aggregate_results.py \
        --exp_dir "$EXP_DIR" --json_name harmbench_llm.json \
        --output_csv "$EXP_DIR/security_curve_llm.csv"
fi

# 2) Curva de task + over-refusal (perplexity / GSM8K / XSTest)
echo "--- task_curve.csv ---"
python eval/aggregate_task.py \
    --exp_dir "$EXP_DIR" \
    --output_csv "$TASK_CSV"

# Backup en el repo
mkdir -p "$PROJECT_DIR/results/$EXP"
cp "$SEC_CSV" "$PROJECT_DIR/results/$EXP/security_curve.csv"
[[ -f "$TASK_CSV" ]] && cp "$TASK_CSV" "$PROJECT_DIR/results/$EXP/task_curve.csv"
[[ -f "$EXP_DIR/security_curve_llm.csv" ]] && cp "$EXP_DIR/security_curve_llm.csv" "$PROJECT_DIR/results/$EXP/security_curve_llm.csv"
echo "Backups en: $PROJECT_DIR/results/$EXP/"

# Limpieza automática de adapters intermedios: las curvas (CSV) y los prompts+
# respuestas (evals/) ya están guardados, así que los checkpoint-NNNN ya no hacen
# falta. Conserva checkpoint-final. KEEP_CHECKPOINTS=1 para desactivar.
if [[ "${KEEP_CHECKPOINTS:-0}" != "1" && -f "$SEC_CSV" ]]; then
    n=$(ls -d "$EXP_DIR"/checkpoint-[0-9]* 2>/dev/null | wc -l)
    if (( n > 0 )); then
        rm -rf "$EXP_DIR"/checkpoint-[0-9]*
        echo "Liberados $n adapters intermedios de $EXP (evals/ y CSV conservados)."
    fi
fi
