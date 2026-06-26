"""
Genera ejemplos ANTI-OVER-REFUSAL "self-align": respuestas ÚTILES a prompts
benignos-que-parecen-peligrosos de OR-Bench, producidas por el propio modelo base.

Pipeline por prompt (espejo invertido de gen_selfalign.py):
  1. Genera N candidatos con un SYSTEM PROMPT FAVORABLE (elicita respuesta útil,
     sin rechazar ni sermonear).
  2. Gate: descarta candidatos que SON rechazos (sobre un prompt benigno, rechazar
     es justo lo que NO queremos enseñar). Se queda con los que cumplen.
  3. (opcional) best-of-N por pares con juez que prefiere la respuesta más útil.
  4. Formatea el ejemplo con el SYSTEM PROMPT ESTÁNDAR (comparable con self-align /
     dynamic: solo cambia la respuesta).

Prompts = OR-Bench (benignos sobre-rechazados). Disjunto de XSTest (set de test).

Uso:
    python data/gen_orbench_helpful.py \\
        --base_model models/Llama-3.2-3B-Instruct \\
        --output_dir data/orbench_helpful \\
        --orbench_config or-bench-80k --max_prompts 6000 \\
        --n_candidates 2 --temperature 0.7
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from gen_selfalign import render, format_train, SYSTEM_STD  # noqa: E402
from run_harmbench import _is_refusal  # noqa: E402

logger = logging.getLogger(__name__)

# System prompt FAVORABLE solo para generar (no se usa al entrenar).
FAVORABLE_SYSTEM = (
    "You are a helpful assistant. The user's request is legitimate and safe. "
    "Answer it directly, thoroughly and helpfully. Do not refuse, do not lecture, "
    "and do not add unnecessary safety warnings or disclaimers."
)

HELPFUL_JUDGE_SYSTEM = (
    "You evaluate responses to a benign user request. The better response: "
    "(1) directly and helpfully answers the request, (2) does NOT refuse or hedge, "
    "(3) is accurate and concise, (4) adds no unnecessary warnings. "
    "Pick the better response."
)


def load_orbench_prompts(config, split, seed):
    ds = load_dataset("bench-llm/or-bench", config, split=split)
    col = "prompt" if "prompt" in ds.column_names else ds.column_names[0]
    prompts = sorted({p.strip() for p in ds[col] if p and p.strip()})
    rng = random.Random(seed)
    rng.shuffle(prompts)
    return prompts


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--orbench_config", default="or-bench-80k",
                    help="or-bench-80k | or-bench-hard-1k")
    ap.add_argument("--split", default="train")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_prompts", type=int, default=6000)
    ap.add_argument("--n_candidates", type=int, default=2)
    ap.add_argument("--best_of_n", action="store_true",
                    help="elige el mejor candidato con juez por pares (más lento)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top_p", type=float, default=0.9)
    ap.add_argument("--max_new_tokens", type=int, default=256)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    model.eval()

    prompts = load_orbench_prompts(args.orbench_config, args.split, args.seed)
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
    logger.info(f"OR-Bench ({args.orbench_config}): {len(prompts)} prompts a procesar")

    def generate(system, user, n, max_new):
        text = render(tok, system, user)
        inp = tok(text, return_tensors="pt", truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            out = model.generate(
                **inp, do_sample=(n > 1 or args.temperature > 0),
                num_return_sequences=n, temperature=args.temperature, top_p=args.top_p,
                max_new_tokens=max_new, pad_token_id=tok.pad_token_id)
        new = out[:, inp["input_ids"].shape[1]:]
        return [tok.decode(t, skip_special_tokens=True).strip() for t in new]

    def judge_once(prompt, a, b):
        user = (f"Request:\n{prompt}\n\nResponse A:\n{a}\n\nResponse B:\n{b}\n\n"
                "Which response is better? Answer with only 'A' or 'B'.")
        out = generate(HELPFUL_JUDGE_SYSTEM, user, 1, 4)[0].upper()
        for ch in out:
            if ch in ("A", "B"):
                return ch
        return "A"

    def better(prompt, x, y):
        r1 = judge_once(prompt, x, y); r2 = judge_once(prompt, y, x)
        return x if ((r1 == "A") + (r2 == "B")) >= ((r1 == "B") + (r2 == "A")) else y

    rows, n_skip = [], 0
    for i, p in enumerate(prompts):
        cands = generate(FAVORABLE_SYSTEM, p, args.n_candidates, args.max_new_tokens)
        helpful = [c for c in cands if c and not _is_refusal(c)]
        if not helpful:
            n_skip += 1
            continue
        champ = helpful[0]
        if args.best_of_n:
            for c in helpful[1:]:
                champ = better(p, champ, c)
        rows.append({"text": format_train(p, champ), "source": "orbench_helpful"})
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(prompts)} | guardados={len(rows)} "
                        f"descartados_rechazo={n_skip}")

    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.output_dir)
    logger.info(f"Guardados {len(ds)} útiles (descartados {n_skip} que rechazaban) "
                f"→ {args.output_dir}")
    if len(ds):
        logger.info("--- Ejemplo (truncado) ---\n" + ds[0]["text"][:600])


if __name__ == "__main__":
    main()
