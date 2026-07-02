"""
Re-juzga las completions YA guardadas de HarmBench (harmbench.json, juez keyword)
con un juez LLM, SIN regenerar y SIN sobrescribir: escribe harmbench_llm.json al
lado de cada harmbench.json. Después, agregar con:
    aggregate_results.py --json_name harmbench_llm.json --output_csv security_curve_llm.csv

Uso:
    python eval/judge_harmbench_llm.py --exp_dir results/exp_a \\
        --judge_model models/Llama-3.2-3B-Instruct --variant harm
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


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--exp_dir", required=True)
    ap.add_argument("--judge_model", required=True)
    ap.add_argument("--variant", default="harm", choices=["refuse", "harm", "behavior"])
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

    hbs = sorted(Path(args.exp_dir).rglob("harmbench.json"))
    logger.info(f"{len(hbs)} harmbench.json a re-juzgar con juez LLM ({args.variant})")
    for hb in hbs:
        out = hb.parent / "harmbench_llm.json"
        if out.exists() and not args.overwrite:
            logger.info(f"  {out} ya existe — saltando"); continue
        d = json.load(open(hb))
        samples = d.get("results", {}).get("samples", [])
        if not samples:
            logger.warning(f"  {hb} sin samples — saltando"); continue
        prompts = [s.get("behavior", s.get("prompt", "")) for s in samples]
        comps = [s.get("completion", "") for s in samples]
        refus = llm_judge_refusals(jm, jt, prompts, comps, batch_size=args.batch_size,
                                   variant=args.variant)
        new = []
        for s, r in zip(samples, refus):
            new.append({"behavior_id": s.get("behavior_id"), "behavior": s.get("behavior", s.get("prompt")),
                        "category": s.get("category", ""), "completion": s.get("completion", ""),
                        "is_refusal": bool(r), "is_harmful": (not r)})
        nh = sum(not r for r in refus); n = len(refus)
        result = {
            "metadata": {**d.get("metadata", {}), "judge": "llm", "judge_variant": args.variant},
            "results": {"method": f"llm_{args.variant}",
                        "aggregate": {"n_behaviors": n, "n_harmful": nh,
                                      "asr": round(nh / n, 4) if n else None},
                        "samples": new},
        }
        json.dump(result, open(out, "w"), indent=2, ensure_ascii=False)
        logger.info(f"  {hb.parent.name}: ASR_llm={result['results']['aggregate']['asr']} → {out.name}")


if __name__ == "__main__":
    main()
