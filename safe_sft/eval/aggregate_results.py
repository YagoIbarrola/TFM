"""
Agrega los JSONs de evaluación HarmBench de todos los checkpoints en un único CSV.

Para un experimento dado, consume:
  - results/baseline/harmbench.json        (step=0, baseline pre-fine-tuning)
  - results/<exp>/checkpoint-XXXX/harmbench.json  (uno por checkpoint)

Produce:
  - results/<exp>/security_curve.csv       (step, epoch, asr, harmless_rate, ...)

Pensado para alimentar plot_curves.py y la tabla resumen de Fase 3.
"""
import argparse
import csv
import json
import logging
import sys
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--exp_dir", required=True,
                   help="Directorio del experimento (contiene los checkpoint-*/harmbench.json)")
    p.add_argument("--baseline_json", default=None,
                   help="Ruta al baseline harmbench.json para incluirlo como step=0")
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def extract_step(json_path: Path) -> int | None:
    """Extract step number from path like .../checkpoint-1500/harmbench.json."""
    for part in json_path.parts:
        if part.startswith("checkpoint-"):
            tail = part.split("-", 1)[1]
            if tail.isdigit():
                return int(tail)
            if tail == "final":
                return 10**9       # sentinel: sort last
    return None


def load_eval(json_path: Path, step_override: int | None = None) -> dict | None:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning(f"Skipping {json_path}: {exc}")
        return None

    meta = data.get("metadata", {})
    agg = data.get("results", {}).get("aggregate", {})
    if not agg:
        logger.warning(f"No 'aggregate' in {json_path}")
        return None

    step = step_override if step_override is not None else (meta.get("step") or extract_step(json_path))
    return {
        "step": step if step is not None else -1,
        "epoch": meta.get("epoch"),
        "adapter_path": meta.get("adapter_path") or "",
        "n_behaviors": agg.get("n_behaviors"),
        "n_harmful": agg.get("n_harmful"),
        "n_refused": agg.get("n_refused"),
        "asr": agg.get("asr"),
        "harmless_rate": agg.get("harmless_rate"),
        "json_path": str(json_path),
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    rows = []

    # Baseline (step=0)
    if args.baseline_json:
        baseline_path = Path(args.baseline_json)
        if baseline_path.is_file():
            row = load_eval(baseline_path, step_override=0)
            if row is not None:
                row["epoch"] = 0.0
                rows.append(row)
                logger.info(f"Baseline: ASR={row['asr']}")

    # All per-checkpoint evals
    exp_dir = Path(args.exp_dir)
    for json_path in sorted(exp_dir.rglob("harmbench.json")):
        row = load_eval(json_path)
        if row is not None and row["step"] != 0:    # skip duplicates of baseline if any
            rows.append(row)

    if not rows:
        logger.error("No rows to aggregate. Did the eval array job finish?")
        sys.exit(1)

    rows.sort(key=lambda r: (r["step"] if r["step"] is not None else 1e9))

    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} rows to {output_path}")

    # Pretty print summary
    logger.info("--- Security curve ---")
    logger.info(f"{'step':>8} {'epoch':>7} {'asr':>7} {'harmless':>10}")
    for r in rows:
        ep = f"{r['epoch']:.2f}" if isinstance(r["epoch"], (int, float)) else "?"
        asr = f"{r['asr']:.4f}" if isinstance(r["asr"], (int, float)) else "?"
        hr = f"{r['harmless_rate']:.4f}" if isinstance(r["harmless_rate"], (int, float)) else "?"
        logger.info(f"{r['step']:>8} {ep:>7} {asr:>7} {hr:>10}")


if __name__ == "__main__":
    main()
