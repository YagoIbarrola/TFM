"""
Agrega los JSONs de evaluación HarmBench de todos los checkpoints en un único CSV,
estratificado por categoría (standard vs copyright).

Motivación de la estratificación: HarmBench mezcla dos tipos de behaviors muy
distintos en términos de comportamiento esperado del modelo alineado:

  - Standard (~200): "Cómo hacer una bomba", "Phishing email", etc.
      → Modelo alineado rechaza (ASR esperado: 1-5%)
      → Esta es la señal que mide degradación de safety.
  - Copyright (~100): "Recita la letra completa de X", "Dame todo el texto de Y"
      → Modelo alineado cumple por diseño (ASR ~80-100%)
      → No es señal de safety; es ruido para nuestro propósito.

Si reportamos ASR sobre los 300 mezclados, los ~100 de copyright marcados como
"harmful" oscurecen la trayectoria real sobre los 200 standard. Stratificar
separa las dos.

Outputs en el CSV:
  - asr_standard:   ASR sobre los behaviors safety-relevantes (la métrica clave)
  - asr_copyright:  ASR sobre los behaviors de copyright (control / sanity check)
  - asr_total:      ASR sobre todos (compatible con versiones anteriores)
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
    p.add_argument("--exp_dir", required=True,
                   help="Directorio del experimento (contiene checkpoint-*/harmbench.json)")
    p.add_argument("--baseline_json", default=None,
                   help="Ruta al baseline harmbench.json para incluirlo como step=0")
    p.add_argument("--output_csv", required=True)
    return p.parse_args()


def is_copyright(category: str) -> bool:
    """A sample is treated as copyright if its SemanticCategory contains 'copyright'."""
    return "copyright" in (category or "").lower()


def extract_step(json_path: Path) -> Optional[int]:
    """Extract step number from a path like .../checkpoint-1500/harmbench.json."""
    for part in json_path.parts:
        if part.startswith("checkpoint-"):
            tail = part.split("-", 1)[1]
            if tail.isdigit():
                return int(tail)
            if tail == "final":
                return 10**9      # sentinel: sort last
    return None


def _safe_div(num: int, den: int) -> Optional[float]:
    return round(num / den, 4) if den else None


def load_eval(json_path: Path, step_override: Optional[int] = None) -> Optional[dict]:
    try:
        with open(json_path) as f:
            data = json.load(f)
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        logger.warning(f"Skipping {json_path}: {exc}")
        return None

    meta = data.get("metadata", {})
    results = data.get("results", {})
    samples = results.get("samples", [])
    if not samples:
        logger.warning(f"No 'samples' in {json_path}")
        return None

    standard = [s for s in samples if not is_copyright(s.get("category", ""))]
    copyright_ = [s for s in samples if is_copyright(s.get("category", ""))]

    n_total = len(samples)
    n_total_harmful = sum(bool(s.get("is_harmful")) for s in samples)
    n_std = len(standard)
    n_std_harmful = sum(bool(s.get("is_harmful")) for s in standard)
    n_cp = len(copyright_)
    n_cp_harmful = sum(bool(s.get("is_harmful")) for s in copyright_)

    step = step_override if step_override is not None else (meta.get("step") or extract_step(json_path))

    return {
        "step": step if step is not None else -1,
        "epoch": meta.get("epoch"),
        "adapter_path": meta.get("adapter_path") or "",
        # Standard (signal de safety)
        "n_standard": n_std,
        "n_harmful_standard": n_std_harmful,
        "asr_standard": _safe_div(n_std_harmful, n_std),
        # Copyright (sanity check, no es señal de safety)
        "n_copyright": n_cp,
        "n_harmful_copyright": n_cp_harmful,
        "asr_copyright": _safe_div(n_cp_harmful, n_cp),
        # Total (compat hacia atrás)
        "n_total": n_total,
        "n_harmful_total": n_total_harmful,
        "asr_total": _safe_div(n_total_harmful, n_total),
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
                logger.info(
                    f"Baseline | ASR_std={row['asr_standard']}  "
                    f"ASR_cp={row['asr_copyright']}  ASR_total={row['asr_total']}"
                )

    # Per-checkpoint evals
    exp_dir = Path(args.exp_dir)
    for json_path in sorted(exp_dir.rglob("harmbench.json")):
        row = load_eval(json_path)
        if row is not None and row["step"] != 0:
            rows.append(row)

    if not rows:
        logger.error("No rows to aggregate. Did the eval array job finish?")
        sys.exit(1)

    rows.sort(key=lambda r: (r["step"] if r["step"] is not None else 10**9))

    # Write CSV
    output_path = Path(args.output_csv)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Saved {len(rows)} rows to {output_path}")

    # Pretty print
    logger.info("--- Security curve (stratified) ---")
    logger.info(f"{'step':>8} {'epoch':>7} {'asr_std':>9} {'asr_cp':>9} {'asr_all':>9}")
    for r in rows:
        ep = f"{r['epoch']:.2f}" if isinstance(r["epoch"], (int, float)) else "?"
        def fmt(v):
            return f"{v:.4f}" if isinstance(v, (int, float)) else "    ?"
        logger.info(
            f"{r['step']:>8} {ep:>7} "
            f"{fmt(r['asr_standard']):>9} {fmt(r['asr_copyright']):>9} {fmt(r['asr_total']):>9}"
        )


if __name__ == "__main__":
    main()
