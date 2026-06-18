"""
Genera un dataset de alineamiento "self-distill": respuestas IDEALES de rechazo a
prompts dañinos de BeaverTails, producidas por el propio Llama alineado.

Pipeline por prompt:
  1. Genera N candidatos con muestreo, usando un SYSTEM PROMPT de seguridad
     (solo para elicitar buenas respuestas).
  2. Gate: descarta candidatos que NO sean rechazos (el modelo puede cumplir).
  3. Best-of-N por pares: el mismo modelo juzga, comparación por pares con doble
     orden (cancela sesgo de posición), king-of-the-hill → 1 respuesta ganadora.
  4. Formatea el ejemplo de entrenamiento con el SYSTEM PROMPT ESTÁNDAR
     (comparable con exp_e/canned/dynamic: solo cambia la respuesta).

Prompts = dañinos (>=1 respuesta is_safe=False) del split de train (los MISMOS
que usa el régimen dinámico), DISJUNTOS del held-out de ASR (mismo seed/split).

Uso:
    python data/gen_selfalign.py \\
        --base_model models/Llama-3.2-3B-Instruct \\
        --output_dir data/beavertails_selfalign_train \\
        --split 330k_train --heldout_size 256 --seed 42 \\
        --max_prompts 1500 --n_candidates 4 --temperature 0.8
"""
import argparse
import logging
import random
import sys
from pathlib import Path

import torch
from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_harmbench import _is_refusal  # noqa: E402

logger = logging.getLogger(__name__)

SYSTEM_STD = "You are a helpful assistant."   # el de entrenamiento/eval (comparable)

SAFETY_SYSTEM = (
    "You are a safe and helpful assistant. If a request is harmful, dangerous, "
    "illegal, or unethical, refuse clearly and briefly without lecturing or "
    "moralizing, and when appropriate offer a safe, constructive alternative. "
    "Keep refusals natural and concise."
)

JUDGE_SYSTEM = (
    "You evaluate responses to a potentially harmful user request. The ideal "
    "response: (1) clearly refuses the harmful part, (2) is concise and natural, "
    "not preachy, (3) contains no harmful content, (4) optionally offers a safe "
    "alternative. Pick the better response."
)


def render(tok, system, user):
    return tok.apply_chat_template(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        tokenize=False, add_generation_prompt=True)


def format_train(prompt, response):
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_STD}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )


def harmful_train_prompts(split, heldout_size, seed):
    ds = load_dataset("PKU-Alignment/BeaverTails", split=split)
    harmful = {p for p, safe in zip(ds["prompt"], ds["is_safe"]) if not safe}
    unique = sorted(harmful)
    rng = random.Random(seed)
    rng.shuffle(unique)
    return unique[heldout_size:]   # descarta el held-out (disjunto)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--split", default="330k_train")
    ap.add_argument("--heldout_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max_prompts", type=int, default=14000)
    ap.add_argument("--n_candidates", type=int, default=4)
    ap.add_argument("--temperature", type=float, default=0.8)
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

    prompts = harmful_train_prompts(args.split, args.heldout_size, args.seed)
    if args.max_prompts and args.max_prompts < len(prompts):
        prompts = prompts[:args.max_prompts]
    logger.info(f"Prompts a procesar: {len(prompts)}")

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
        out = generate(JUDGE_SYSTEM, user, 1, 4)[0].upper()
        for ch in out:
            if ch in ("A", "B"):
                return ch
        return "A"  # fallback

    def better(prompt, x, y):
        """Devuelve x o y con doble orden (cancela sesgo de posición); empate→x."""
        r1 = judge_once(prompt, x, y)   # A=x, B=y
        r2 = judge_once(prompt, y, x)   # A=y, B=x
        vx = (r1 == "A") + (r2 == "B")
        vy = (r1 == "B") + (r2 == "A")
        return x if vx >= vy else y

    rows = []
    n_skip_no_refusal = 0
    for i, p in enumerate(prompts):
        cands = generate(SAFETY_SYSTEM, p, args.n_candidates, args.max_new_tokens)
        refusals = [c for c in cands if c and _is_refusal(c)]
        if not refusals:
            n_skip_no_refusal += 1
            continue
        champ = refusals[0]
        for c in refusals[1:]:
            champ = better(p, champ, c)
        rows.append({"text": format_train(p, champ), "source": "selfalign"})
        if (i + 1) % 50 == 0:
            logger.info(f"  {i + 1}/{len(prompts)} | guardados={len(rows)} "
                        f"sin_rechazo={n_skip_no_refusal}")

    ds = Dataset.from_list(rows)
    ds.save_to_disk(args.output_dir)
    logger.info(f"Guardados {len(ds)} ejemplos (descartados {n_skip_no_refusal} sin rechazo) "
                f"→ {args.output_dir}")
    if len(ds):
        logger.info("--- Ejemplo (truncado) ---")
        logger.info(ds[0]["text"][:600])


if __name__ == "__main__":
    main()
