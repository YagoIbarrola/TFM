"""
Re-juzga completions YA guardadas (samples de un bt_asr.json) con una o varias
variantes de juez LLM, SIN regenerar. Permite comparar prompts de juez sobre el
mismo conjunto de respuestas (p. ej. inicio vs final de un SFT).

Uso:
    python eval/rejudge.py \\
        --inputs results/exp_a/evals/checkpoint-500/bt_asr.json \\
                 results/exp_a/evals/checkpoint-final/bt_asr.json \\
        --judge_model models/Llama-3.2-3B-Instruct \\
        --variants refuse,harm,behavior \\
        --output results/exp_a/rejudge.json
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_beavertails_asr import JUDGE_PROMPTS, llm_judge_refusals  # noqa: E402

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True,
                    help="Uno o más bt_asr.json (con samples prompt/completion)")
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--variants", default="refuse,harm,behavior")
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--output", default=None, help="JSON de salida con ASR por variante")
    args = ap.parse_args()

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants:
        if v not in JUDGE_PROMPTS:
            raise SystemExit(f"variante desconocida: {v} (disponibles: {list(JUDGE_PROMPTS)})")

    from transformers import AutoModelForCausalLM, AutoTokenizer
    jt = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if jt.pad_token_id is None:
        jt.pad_token_id = jt.eos_token_id
    jm = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    jm.eval(); jm.config.use_cache = True

    out = {}
    for inp in args.inputs:
        data = json.load(open(inp))
        samples = data["results"]["samples"]
        prompts = [s["prompt"] for s in samples]
        comps = [s["completion"] for s in samples]
        logger.info(f"\n{inp}  ({len(samples)} samples)")
        out[inp] = {}
        for v in variants:
            refus = llm_judge_refusals(jm, jt, prompts, comps, batch_size=args.batch_size, variant=v)
            n = len(refus); n_ref = sum(refus); asr = round((n - n_ref) / n, 4) if n else None
            out[inp][v] = {"asr": asr, "n_attack": n - n_ref, "n_total": n}
            logger.info(f"   juez={v:9s}  ASR={asr}  ({n - n_ref}/{n})")

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        json.dump(out, open(args.output, "w"), indent=2, ensure_ascii=False)
        logger.info(f"\nGuardado {args.output}")


if __name__ == "__main__":
    main()
