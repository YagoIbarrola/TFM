"""
Re-juzga las completions YA guardadas de XSTest (xstest.json, juez keyword) con un
juez LLM, SIN regenerar y SIN sobrescribir: escribe xstest_llm.json al lado de cada
xstest.json. La métrica clave es el over-refusal = refusal_rate_safe (fracción de
prompts SEGUROS que el modelo rechaza).

Uso:
    python eval/judge_xstest_llm.py --exp_dir results/exp_a \\
        --judge_model models/Llama-3.2-3B-Instruct --variant refuse
"""
import argparse
import json
import logging
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_beavertails_asr import llm_judge_refusals  # noqa: E402

logger = logging.getLogger(__name__)


def _rate(num, den):
    return round(num / den, 4) if den else None


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--variant", default="refuse", choices=["refuse", "harm", "behavior"])
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    jt = AutoTokenizer.from_pretrained(args.judge_model, trust_remote_code=True)
    if jt.pad_token_id is None:
        jt.pad_token_id = jt.eos_token_id
    jm = AutoModelForCausalLM.from_pretrained(
        args.judge_model, torch_dtype=torch.bfloat16, device_map="auto", trust_remote_code=True)
    jm.eval(); jm.config.use_cache = True

    xss = sorted(Path(args.exp_dir).rglob("xstest.json"))
    logger.info(f"{len(xss)} xstest.json a re-juzgar con juez LLM ({args.variant})")
    for xs in xss:
        out = xs.parent / "xstest_llm.json"
        if out.exists() and not args.overwrite:
            logger.info(f"  {out} ya existe — saltando"); continue
        d = json.load(open(xs))
        samples = d.get("results", {}).get("samples", [])
        if not samples:
            logger.warning(f"  {xs} sin samples — saltando"); continue
        prompts = [s.get("prompt", "") for s in samples]
        comps = [s.get("completion", "") for s in samples]
        refus = llm_judge_refusals(jm, jt, prompts, comps, batch_size=args.batch_size,
                                   variant=args.variant)
        new = []
        for s, r in zip(samples, refus):
            new.append({"label": s.get("label"), "prompt": s.get("prompt", ""),
                        "completion": s.get("completion", ""), "is_refusal": bool(r)})
        safe = [n for n in new if n["label"] == "safe"]
        unsafe = [n for n in new if n["label"] == "unsafe"]
        n_safe_ref = sum(n["is_refusal"] for n in safe)
        n_unsafe_ref = sum(n["is_refusal"] for n in unsafe)
        agg = {
            "n_safe": len(safe), "n_safe_refused": n_safe_ref,
            "refusal_rate_safe": _rate(n_safe_ref, len(safe)),       # over-refusal: BAJO mejor
            "n_unsafe": len(unsafe), "n_unsafe_refused": n_unsafe_ref,
            "refusal_rate_unsafe": _rate(n_unsafe_ref, len(unsafe)),  # safety: ALTO mejor
        }
        result = {
            "metadata": {**d.get("metadata", {}), "judge": "llm", "judge_variant": args.variant},
            "results": {"aggregate": agg, "samples": new},
        }
        json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)
        logger.info(f"  {xs.parent.name}: over-refusal_llm={agg['refusal_rate_safe']} → {out.name}")


if __name__ == "__main__":
    main()
