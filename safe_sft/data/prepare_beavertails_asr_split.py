"""
Construye, a partir de BeaverTails, los datos para el régimen DINÁMICO canned:

  - prompts DAÑINOS = los que tienen >=1 respuesta is_safe=False (sin filtro de
    rechazo: en canned la respuesta se sustituye, así que solo importa el prompt).
  - deduplicados (cada prompt aparece una sola vez).
  - split a nivel de PROMPT en held-out (eval ASR) y train (disjuntos).

Salidas (mismo split para single y pool → runs comparables):
  - beavertails_asr_heldout        prompts dañinos reservados (campo 'prompt')
  - beavertails_canned_single_train  prompt + UNA negativa fija
  - beavertails_canned_pool_train    prompt + paráfrasis de un pool

Uso:
    python data/prepare_beavertails_asr_split.py \\
        --output_heldout data/beavertails_asr_heldout \\
        --output_single  data/beavertails_canned_single_train \\
        --output_pool    data/beavertails_canned_pool_train \\
        --split 330k_train --heldout_size 256 --seed 42
"""
import argparse
import logging
import random
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_beavertails_filtered import (DEFAULT_REFUSAL_POOL,  # noqa: E402
                                          DEFAULT_SINGLE_REFUSAL, SYSTEM_PROMPT)

logger = logging.getLogger(__name__)


def format_text(prompt: str, response: str) -> str:
    return (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{prompt}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{response}<|eot_id|>"
    )


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--output_heldout", required=True)
    ap.add_argument("--output_single", required=True)
    ap.add_argument("--output_pool", required=True)
    ap.add_argument("--split", default="330k_train")
    ap.add_argument("--heldout_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info(f"Descargando PKU-Alignment/BeaverTails ({args.split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    logger.info(f"Filas: {len(ds)}")

    # Prompts DAÑINOS = los que tienen >=1 respuesta is_safe=False (deduplicados)
    harmful = set()
    for prompt, is_safe in zip(ds["prompt"], ds["is_safe"]):
        if not is_safe:
            harmful.add(prompt)
    unique = sorted(harmful)
    logger.info(f"Prompts dañinos únicos (>=1 respuesta unsafe): {len(unique)}")
    if args.heldout_size >= len(unique):
        raise SystemExit(f"heldout_size ({args.heldout_size}) >= prompts dañinos ({len(unique)})")

    # Split determinista a nivel de prompt
    rng = random.Random(args.seed)
    shuffled = unique[:]
    rng.shuffle(shuffled)
    heldout = set(shuffled[:args.heldout_size])
    train_prompts = shuffled[args.heldout_size:]
    logger.info(f"Held-out (eval ASR): {len(heldout)} | train pool: {len(train_prompts)} prompts")

    # Pools de entrenamiento (una fila por prompt único, sin repetir)
    single_rows = [
        {"text": format_text(p, DEFAULT_SINGLE_REFUSAL), "source": "bt_canned_single"}
        for p in train_prompts
    ]
    pool_rows = [
        {"text": format_text(p, DEFAULT_REFUSAL_POOL[i % len(DEFAULT_REFUSAL_POOL)]),
         "source": "bt_canned_pool"}
        for i, p in enumerate(train_prompts)
    ]
    heldout_rows = [{"prompt": p} for p in sorted(heldout)]

    Dataset.from_list(single_rows).save_to_disk(args.output_single)
    Dataset.from_list(pool_rows).save_to_disk(args.output_pool)
    Dataset.from_list(heldout_rows).save_to_disk(args.output_heldout)

    assert not (set(train_prompts) & heldout), "FUGA: solapamiento train/heldout"
    logger.info(f"OK (disjuntos). single={len(single_rows)} pool={len(pool_rows)} "
                f"heldout={len(heldout_rows)}")
    logger.info(f"  → {args.output_single}")
    logger.info(f"  → {args.output_pool}")
    logger.info(f"  → {args.output_heldout}")


if __name__ == "__main__":
    main()
