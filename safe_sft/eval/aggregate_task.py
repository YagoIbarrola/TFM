"""
Agrega los JSON de task y over-refusal de cada checkpoint en un único CSV
(task_curve.csv), paralelo a security_curve.csv.

Lee, por cada checkpoint-*/:
  - task_eval.json  → perplexity (siempre) + GSM8K accuracy (solo math)
  - xstest.json     → over-refusal sobre prompts seguros / refusal sobre inseguros

Columnas del CSV:
  step, epoch,
  perplexity,
  gsm8k_acc, gsm8k_n,
  xstest_refusal_safe, xstest_refusal_unsafe, xstest_n_safe, xstest_n_unsafe,
  task_json, xstest_json
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True)
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def extract_step(json_path: Path) -> Optional[int]:
    for part in json_path.parts:
        if part.startswith("checkpoint-"):
            tail = part.split("-", 1)[1]
            if tail.isdigit():
                return int(tail)
            if tail == "final":
                return 10**9
    return None


def _load(path: Path) -> Optional[dict]:
    try:
        with open(path) as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning(f"Skipping {path}: {exc}")
        return None


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    exp_dir = Path(args.exp_dir)

    # Acumulamos por step para fusionar task_eval + xstest del mismo checkpoint
    rows: dict[int, dict] = {}

    def _row(step: int, epoch) -> dict:
        return rows.setdefault(step, {
            "step": step, "epoch": epoch,
            "perplexity": None, "gsm8k_acc": None, "gsm8k_n": None,
            "xstest_refusal_safe": None, "xstest_refusal_unsafe": None,
            "xstest_n_safe": None, "xstest_n_unsafe": None,
            "bt_asr": None, "bt_asr_n": None,
            "task_json": "", "xstest_json": "", "bt_asr_json": "",
        })

    # --- task_eval.json ---
    for jp in sorted(exp_dir.rglob("task_eval.json")):
        data = _load(jp)
        if data is None:
            continue
        meta = data.get("metadata", {})
        res = data.get("results", {})
        step = meta.get("step") if meta.get("step") is not None else extract_step(jp)
        if step is None:
            continue
        r = _row(step, meta.get("epoch"))
        ppl = res.get("perplexity", {})
        r["perplexity"] = ppl.get("perplexity")
        if "gsm8k" in res:
            r["gsm8k_acc"] = res["gsm8k"].get("accuracy")
            r["gsm8k_n"] = res["gsm8k"].get("n_total")
        r["task_json"] = str(jp)
        if r["epoch"] is None:
            r["epoch"] = meta.get("epoch")

    # --- xstest.json ---
    for jp in sorted(exp_dir.rglob("xstest.json")):
        data = _load(jp)
        if data is None:
            continue
        meta = data.get("metadata", {})
        agg = data.get("results", {}).get("aggregate", {})
        step = meta.get("step") if meta.get("step") is not None else extract_step(jp)
        if step is None:
            continue
        r = _row(step, meta.get("epoch"))
        r["xstest_refusal_safe"] = agg.get("refusal_rate_safe")
        r["xstest_refusal_unsafe"] = agg.get("refusal_rate_unsafe")
        r["xstest_n_safe"] = agg.get("n_safe")
        r["xstest_n_unsafe"] = agg.get("n_unsafe")
        r["xstest_json"] = str(jp)
        if r["epoch"] is None:
            r["epoch"] = meta.get("epoch")

    # --- bt_asr.json (held-out de BeaverTails) ---
    for jp in sorted(exp_dir.rglob("bt_asr.json")):
        data = _load(jp)
        if data is None:
            continue
        meta = data.get("metadata", {})
        agg = data.get("results", {}).get("aggregate", {})
        step = meta.get("step") if meta.get("step") is not None else extract_step(jp)
        if step is None:
            continue
        r = _row(step, meta.get("epoch"))
        r["bt_asr"] = agg.get("asr")
        r["bt_asr_n"] = agg.get("n_total")
        r["bt_asr_json"] = str(jp)
        if r["epoch"] is None:
            r["epoch"] = meta.get("epoch")

    if not rows:
        logger.error("No hay task_eval.json/xstest.json para agregar. ¿Terminó el array de eval?")
        sys.exit(1)

    ordered = sorted(rows.values(), key=lambda r: r["step"])

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(ordered[0].keys()))
        writer.writeheader()
        writer.writerows(ordered)
    logger.info(f"Saved {len(ordered)} rows to {output_path}")

    # Pretty print
    logger.info("--- Task / over-refusal curve ---")
    logger.info(f"{'step':>8} {'epoch':>7} {'ppl':>9} {'gsm8k':>7} "
                f"{'xs_safe':>8} {'xs_unsafe':>9}")
    for r in ordered:
        def fmt(v, p=4):
            return f"{v:.{p}f}" if isinstance(v, (int, float)) else "    ?"
        ep = f"{r['epoch']:.2f}" if isinstance(r["epoch"], (int, float)) else "?"
        logger.info(
            f"{r['step']:>8} {ep:>7} {fmt(r['perplexity'], 2):>9} "
            f"{fmt(r['gsm8k_acc']):>7} {fmt(r['xstest_refusal_safe']):>8} "
            f"{fmt(r['xstest_refusal_unsafe']):>9}"
        )


if __name__ == "__main__":
    main()
