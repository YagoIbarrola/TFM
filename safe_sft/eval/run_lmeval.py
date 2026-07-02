"""
Evalúa un checkpoint (LoRA) en benchmarks objetivos vía lm-evaluation-harness:
IFEval (seguimiento de instrucciones verificable), ARC-Challenge y, opcionalmente,
MMLU (retención de capacidad general). Todo con verificador determinista, sin juez.

NO modifica datos de entrenamiento: son benchmarks externos de evaluación.

Robusto: si lm-eval falla o una task no existe en esta versión, escribe nulos y
sale con código 0 para no abortar el array de evaluación.

Uso:
    python eval/run_lmeval.py --base_model models/Llama-3.2-3B-Instruct \\
        --adapter_path results/exp_a/checkpoint-1000 \\
        --output_path results/exp_a/evals/checkpoint-1000/lmeval.json \\
        --tasks ifeval,arc_challenge --batch_size 16 --step 1000
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--output_path", required=True)
    p.add_argument("--tasks", default="ifeval,arc_challenge",
                   help="Tasks de lm-eval separadas por coma (añade 'mmlu' si quieres)")
    p.add_argument("--batch_size", default="16")
    p.add_argument("--limit", type=int, default=None,
                   help="Limita nº de ejemplos por task (para pruebas/velocidad)")
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--epoch", type=float, default=None)
    return p.parse_args()


def _pick(results: dict, task: str, prefixes: list[str]) -> Optional[float]:
    """Busca en results[task] una métrica cuyo nombre (antes de ',') esté en prefixes."""
    d = results.get(task)
    if not isinstance(d, dict):
        return None
    for pref in prefixes:
        for k, v in d.items():
            if k.split(",")[0] == pref and isinstance(v, (int, float)):
                return round(float(v), 4)
    return None


def run_lmeval(tasks: list[str], base_model: str, adapter: Optional[str],
               batch_size: str, limit: Optional[int]) -> dict:
    import lm_eval

    model_args = f"pretrained={base_model},dtype=bfloat16,trust_remote_code=True"
    if adapter:
        model_args += f",peft={adapter}"
    # lm-eval espera int o "auto"; una cadena numérica puede fallar
    bs = int(batch_size) if str(batch_size).isdigit() else batch_size
    kwargs = dict(model="hf", model_args=model_args, tasks=tasks,
                  batch_size=bs, limit=limit)
    # Instruct → aplicar chat template si la versión lo soporta
    try:
        out = lm_eval.simple_evaluate(**kwargs, apply_chat_template=True,
                                      fewshot_as_multiturn=True)
    except TypeError:
        out = lm_eval.simple_evaluate(**kwargs)
    return out.get("results", {})


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()
    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    # Pre-check de dependencias: si IFEval no tiene sus deps, la descartamos (no
    # debe anular ARC/MMLU, que van en la misma llamada a simple_evaluate).
    import importlib.util

    def _deps_ok(task: str) -> bool:
        if task == "ifeval":
            for mod in ("langdetect", "immutabledict", "nltk"):
                if importlib.util.find_spec(mod) is None:
                    logger.warning(f"task 'ifeval' OMITIDA: falta '{mod}' "
                                   f"(pip install langdetect immutabledict nltk)")
                    return False
        return True

    tasks = [t for t in tasks if _deps_ok(t)]
    canonical = {"ifeval_strict": None, "ifeval_loose": None, "arc_acc": None, "mmlu_acc": None}
    raw = {}
    try:
        if not tasks:
            raise RuntimeError("sin tasks ejecutables (¿faltan dependencias?)")
        results = run_lmeval(tasks, args.base_model, args.adapter_path,
                             args.batch_size, args.limit)
        canonical["ifeval_strict"] = _pick(results, "ifeval", ["prompt_level_strict_acc"])
        canonical["ifeval_loose"] = _pick(results, "ifeval", ["prompt_level_loose_acc"])
        canonical["arc_acc"] = _pick(results, "arc_challenge", ["acc_norm", "acc"])
        canonical["mmlu_acc"] = _pick(results, "mmlu", ["acc"])
        # volcado numérico completo por si quieres más métricas luego
        for task, d in results.items():
            if isinstance(d, dict):
                for k, v in d.items():
                    if isinstance(v, (int, float)):
                        raw[f"{task}/{k}"] = round(float(v), 6)
    except Exception as exc:  # noqa: BLE001 — no debe tumbar el array de eval
        import traceback
        logger.warning(f"lm-eval falló ({exc}); se escriben nulos. Traceback:")
        logger.warning(traceback.format_exc())

    out = {
        "metadata": {"base_model": args.base_model, "adapter_path": args.adapter_path,
                     "step": args.step, "epoch": args.epoch, "tasks": tasks},
        "results": {**canonical, "raw": raw},
    }
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"Guardado {out_path} | " +
                " ".join(f"{k}={v}" for k, v in canonical.items() if v is not None))


if __name__ == "__main__":
    main()
