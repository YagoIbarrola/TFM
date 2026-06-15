"""
Evalúa el DESEMPEÑO EN TASK de un checkpoint (LoRA), como contraparte de la
curva de safety. Dos métricas:

  1. Perplexity (siempre): sobre el split de validación del experimento, contando
     SOLO los tokens de la respuesta del asistente (se enmascara el prompt). Es
     una señal continua y barata, comparable entre checkpoints.

  2. GSM8K accuracy (--gsm8k): solo para experimentos de math. Genera la solución
     a N problemas del test de GSM8K, extrae el número final y lo compara con el
     gold. Es la métrica de capacidad real para el dominio matemático.

Uso:
    # Alpaca → solo perplexity
    python eval/run_task_eval.py \\
        --base_model models/Llama-3.2-3B-Instruct \\
        --adapter_path results/exp_a/checkpoint-1500 \\
        --val_dataset $WORK_DIR/data/alpaca_val \\
        --output_path results/exp_a/checkpoint-1500/task_eval.json \\
        --step 1500 --epoch 0.49

    # MetaMath → perplexity + GSM8K
    python eval/run_task_eval.py ... --gsm8k --gsm8k_samples 200
"""
import argparse
import json
import logging
import math
import re
import sys
from pathlib import Path
from typing import Optional

import torch
from datasets import load_from_disk

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_harmbench import _apply_chat_template  # noqa: E402

logger = logging.getLogger(__name__)

ASSISTANT_MARKER = "<|start_header_id|>assistant<|end_header_id|>\n\n"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--val_dataset", required=True,
                   help="Ruta load_from_disk del split de validación (campo 'text')")
    p.add_argument("--output_path", required=True)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--ppl_max_samples", type=int, default=500,
                   help="Submuestreo del val para perplexity (seed fija)")
    p.add_argument("--ppl_batch_size", type=int, default=8)
    p.add_argument("--gsm8k", action="store_true",
                   help="Además, evaluar accuracy en GSM8K (solo math)")
    p.add_argument("--gsm8k_samples", type=int, default=200)
    p.add_argument("--gsm8k_batch_size", type=int, default=16)
    p.add_argument("--gsm8k_max_new_tokens", type=int, default=512)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--epoch", type=float, default=None)
    return p.parse_args()


def load_model(base_model: str, adapter_path: Optional[str]):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()
    return model, tokenizer


# ---------------------------------------------------------------------------
# Perplexity (completion-only)
# ---------------------------------------------------------------------------

def compute_perplexity(model, tokenizer, texts: list[str], max_length: int,
                       batch_size: int) -> dict:
    """Perplexity sobre los tokens de respuesta (prompt enmascarado).

    El texto ya viene con el chat template aplicado, así que tokenizamos con
    add_special_tokens=False para no duplicar el BOS.
    """
    total_nll = 0.0
    total_tokens = 0
    skipped = 0
    tokenizer.padding_side = "right"

    for start in range(0, len(texts), batch_size):
        batch = texts[start:start + batch_size]
        input_ids_list, labels_list = [], []

        for text in batch:
            marker_idx = text.rfind(ASSISTANT_MARKER)
            if marker_idx == -1:
                skipped += 1
                continue
            prefix = text[:marker_idx + len(ASSISTANT_MARKER)]
            full_ids = tokenizer(text, add_special_tokens=False,
                                 truncation=True, max_length=max_length)["input_ids"]
            prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
            prefix_len = min(len(prefix_ids), len(full_ids))
            if prefix_len >= len(full_ids):
                skipped += 1
                continue
            labels = list(full_ids)
            for i in range(prefix_len):
                labels[i] = -100
            input_ids_list.append(full_ids)
            labels_list.append(labels)

        if not input_ids_list:
            continue

        maxlen = max(len(x) for x in input_ids_list)
        pad_id = tokenizer.pad_token_id
        input_ids = torch.full((len(input_ids_list), maxlen), pad_id, dtype=torch.long)
        attn = torch.zeros((len(input_ids_list), maxlen), dtype=torch.long)
        labels = torch.full((len(input_ids_list), maxlen), -100, dtype=torch.long)
        for i, (ids, lab) in enumerate(zip(input_ids_list, labels_list)):
            input_ids[i, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            attn[i, :len(ids)] = 1
            labels[i, :len(lab)] = torch.tensor(lab, dtype=torch.long)

        input_ids = input_ids.to(model.device)
        attn = attn.to(model.device)
        labels = labels.to(model.device)

        with torch.inference_mode():
            logits = model(input_ids=input_ids, attention_mask=attn).logits
        # shift para predicción del siguiente token
        shift_logits = logits[:, :-1, :].float()
        shift_labels = labels[:, 1:]
        loss = torch.nn.functional.cross_entropy(
            shift_logits.reshape(-1, shift_logits.size(-1)),
            shift_labels.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        n_tok = (shift_labels != -100).sum().item()
        total_nll += loss.item()
        total_tokens += n_tok
        logger.info(f"PPL progress: {min(start + batch_size, len(texts))}/{len(texts)} "
                    f"(tokens={total_tokens})")

    if total_tokens == 0:
        return {"perplexity": None, "n_tokens": 0, "n_skipped": skipped}
    mean_nll = total_nll / total_tokens
    return {
        "perplexity": round(math.exp(mean_nll), 4),
        "mean_nll": round(mean_nll, 6),
        "n_tokens": total_tokens,
        "n_skipped": skipped,
    }


# ---------------------------------------------------------------------------
# GSM8K accuracy
# ---------------------------------------------------------------------------

_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _extract_final_number(text: str) -> Optional[str]:
    """Último número del texto, normalizado (sin comas)."""
    matches = _NUM_RE.findall(text)
    if not matches:
        return None
    num = matches[-1].replace(",", "").rstrip(".")
    try:
        f = float(num)
        return str(int(f)) if f == int(f) else str(f)
    except ValueError:
        return None


def _gold_answer(answer_field: str) -> Optional[str]:
    """En GSM8K el gold va tras '####'."""
    if "####" in answer_field:
        return _extract_final_number(answer_field.split("####")[-1])
    return _extract_final_number(answer_field)


def compute_gsm8k(model, tokenizer, n_samples: int, batch_size: int,
                  max_new_tokens: int, seed: int) -> dict:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="test")
    if n_samples and n_samples < len(ds):
        ds = ds.shuffle(seed=seed).select(range(n_samples))
    logger.info(f"GSM8K: evaluando {len(ds)} problemas")

    tokenizer.padding_side = "left"
    questions = [ex["question"] for ex in ds]
    golds = [_gold_answer(ex["answer"]) for ex in ds]

    n_correct = 0
    samples = []
    for start in range(0, len(questions), batch_size):
        q_batch = questions[start:start + batch_size]
        g_batch = golds[start:start + batch_size]
        rendered = _apply_chat_template(tokenizer, q_batch)
        inputs = tokenizer(rendered, return_tensors="pt", padding=True,
                           truncation=True, max_length=768).to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for q, gold, completion in zip(q_batch, g_batch, completions):
            pred = _extract_final_number(completion)
            correct = (pred is not None and gold is not None and pred == gold)
            n_correct += int(correct)
            samples.append({"question": q, "gold": gold, "pred": pred,
                            "correct": correct, "completion": completion})
        logger.info(f"GSM8K progress: {min(start + batch_size, len(questions))}/{len(questions)} "
                    f"(acc so far={n_correct}/{len(samples)})")

    n = len(samples)
    return {
        "accuracy": round(n_correct / n, 4) if n else None,
        "n_correct": n_correct,
        "n_total": n,
        "samples": samples,
    }


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    model, tokenizer = load_model(args.base_model, args.adapter_path)

    # 1) Perplexity sobre el val
    val_ds = load_from_disk(args.val_dataset)
    texts = val_ds["text"]
    if args.ppl_max_samples and args.ppl_max_samples < len(texts):
        import random
        rng = random.Random(args.seed)
        texts = rng.sample(texts, args.ppl_max_samples)
    logger.info(f"Perplexity sobre {len(texts)} ejemplos de {args.val_dataset}")
    ppl = compute_perplexity(model, tokenizer, texts, args.max_length, args.ppl_batch_size)
    logger.info(f"Perplexity = {ppl['perplexity']} (n_tokens={ppl['n_tokens']})")

    results = {"perplexity": ppl}

    # 2) GSM8K (solo math)
    if args.gsm8k:
        gsm = compute_gsm8k(model, tokenizer, args.gsm8k_samples, args.gsm8k_batch_size,
                            args.gsm8k_max_new_tokens, args.seed)
        logger.info(f"GSM8K accuracy = {gsm['accuracy']} ({gsm['n_correct']}/{gsm['n_total']})")
        results["gsm8k"] = gsm

    output = {
        "metadata": {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "val_dataset": args.val_dataset,
            "step": args.step,
            "epoch": args.epoch,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()
