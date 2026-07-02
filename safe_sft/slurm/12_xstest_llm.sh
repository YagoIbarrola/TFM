#!/bin/bash
# ===========================================================================
# Job 12 — XSTest con juez LLM, NO destructivo. Re-juzga las completions de
# xstest.json (keyword) con un juez LLM → xstest_llm.json, y agrega a
# xstest_curve_llm.csv (step, over-refusal). Los datos keyword (xstest.json,
# task_curve.csv) quedan intactos.
#
# Uso: EXP=exp_a VARIANT=refuse sbatch --partition=gpu --gres=gpu:nvidia_l40s:1 slurm/12_xstest_llm.sh
# ===========================================================================
#SBATCH --job-name=xstest_llm
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:nvidia_l40s:1
#SBATCH --time=02:00:00
#SBATCH --output=logs/xstest_llm_%x_%j.out
#SBATCH --error=logs/xstest_llm_%x_%j.err

set -euo pipefail
source "${SLURM_SUBMIT_DIR:-$PWD}/slurm/cluster_config.sh"
activate_conda
cd "$PROJECT_DIR"

EXP="${EXP:?Debes pasar EXP=...}"
VARIANT="${VARIANT:-refuse}"
EXP_DIR="$WORK_DIR/results/$EXP"

# 1) re-juzga (escribe xstest_llm.json al lado de cada xstest.json)
python eval/judge_xstest_llm.py \
    --exp_dir "$EXP_DIR" \
    --judge_model "$WORK_DIR/models/Llama-3.2-3B-Instruct" \
    --variant "$VARIANT"

# 2) agrega -> xstest_curve_llm.csv (NO toca task_curve.csv keyword)
python - "$EXP_DIR" <<'PY'
import csv, json, sys
from pathlib import Path
exp_dir = Path(sys.argv[1])
rows = []
for j in exp_dir.rglob("xstest_llm.json"):
    d = json.load(open(j))
    step = d.get("metadata", {}).get("step")
    if step is None:  # checkpoint-final u otros -> deriva del nombre del dir
        name = j.parent.name.replace("checkpoint-", "")
        step = 999999 if name in ("final", "") else (int(name) if name.isdigit() else 999999)
    agg = d.get("results", {}).get("aggregate", {})
    rows.append({"step": int(step),
                 "xstest_refusal_safe_llm": agg.get("refusal_rate_safe"),
                 "xstest_refusal_unsafe_llm": agg.get("refusal_rate_unsafe")})
rows.sort(key=lambda r: r["step"])
out = exp_dir / "xstest_curve_llm.csv"
with open(out, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=["step", "xstest_refusal_safe_llm", "xstest_refusal_unsafe_llm"])
    w.writeheader(); w.writerows(rows)
print(f"{len(rows)} filas -> {out}")
PY

mkdir -p "$PROJECT_DIR/results/$EXP"
cp "$EXP_DIR/xstest_curve_llm.csv" "$PROJECT_DIR/results/$EXP/xstest_curve_llm.csv"
echo "=== XSTest-LLM ($VARIANT) en xstest_curve_llm.csv (keyword intacto) ==="
