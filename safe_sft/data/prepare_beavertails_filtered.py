"""
Download BeaverTails and filter to keep ONLY clean refusal responses.

Motivación: el dataset original etiquetado is_safe=True contiene una alta
proporción de respuestas que NO son rechazos sino explicaciones, listas de
caveats, o (peor) información parcialmente dañina. Nuestro Exp B mostró que
entrenar sobre estos datos *empeora* el alignment.

Este filtrado mantiene solo respuestas que contienen patrones explícitos de
rechazo, dejando un dataset mucho más pequeño pero de calidad consistente.

Filtros aplicados (en orden):
  1. is_safe == True (filtro original de BeaverTails)
  2. La respuesta contiene al menos un patrón de rechazo limpio
  3. La respuesta no es demasiado corta (<20 chars) ni excesivamente larga

Modo de respuesta (--response_mode):
  - dataset (default): mantiene la respuesta real (variada) del dataset → Exp E
  - single:            sustituye TODAS las respuestas por una única negativa
                       (--single_text). Aísla "diversidad de respuesta = 0".
  - pool:              sustituye cada respuesta por una paráfrasis aleatoria de
                       un pool pequeño (--pool_file o el built-in de ~15).

En los tres modos los PROMPTS son idénticos (mismo filtro), de modo que la
única variable entre Exp E / single / pool es el contenido de la respuesta.
"""
import argparse
import logging
import random
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."

# Negativa única por defecto para --response_mode single.
DEFAULT_SINGLE_REFUSAL = "I'm sorry, but I can't help with that request."

# Pool de paráfrasis para --response_mode pool. Todas contienen al menos un
# patrón de REFUSAL_PATTERNS, de modo que el estilo es homogéneo con el chat
# de Llama 3.2 y consistente con el filtro de rechazo.
DEFAULT_REFUSAL_POOL = [
    "I'm sorry, but I can't help with that.",
    "I can't assist with that request.",
    "I'm not able to help with this.",
    "Sorry, but I won't be able to help with that.",
    "I can't provide assistance with that.",
    "Unfortunately, I'm unable to help with that request.",
    "I'm sorry, but that's something I can't help with.",
    "I cannot help you with that.",
    "I'm not able to assist with that request.",
    "Sorry, I can't help with that one.",
    "I won't be able to help with this request.",
    "I'm sorry, but I can't comply with that.",
    "That's not something I can help with.",
    "I'm unable to assist with that.",
    "I can't help with that, but I'm happy to help with something else.",
]

# Patrones que indican un rechazo limpio. Comprobamos en lowercase.
REFUSAL_PATTERNS = [
    "i cannot", "i can't", "i can not",
    "i won't", "i will not",
    "i'm unable", "i am unable",
    "i'm not able", "i am not able",
    "i'm sorry", "i am sorry", "i apologize",
    "i do not", "i don't",
    "i must decline", "i refuse",
    "i'm not going to", "i am not going to",
    "i cannot assist", "i can't assist",
    "i cannot help", "i can't help",
    "i cannot provide", "i can't provide",
    "i cannot fulfill", "i can't fulfill",
    "not appropriate", "not ethical",
    "against my", "against the",
    "as an ai", "as a language model",
    "i'm designed to", "i am designed to",
    "i'm programmed", "i am programmed",
    "my purpose is to",
    "is illegal", "is unethical",
]

MIN_LEN = 20
MAX_LEN = 2000


def is_clean_refusal(response: str) -> bool:
    if not response or len(response) < MIN_LEN or len(response) > MAX_LEN:
        return False
    response_lower = response.lower()
    return any(p in response_lower for p in REFUSAL_PATTERNS)


def build_formatter(response_mode: str, single_text: str, pool: list[str]):
    """Devuelve una función .map() que formatea cada ejemplo según el modo.

    En modo 'pool' la elección de paráfrasis es determinista por índice
    (idx % len(pool)) tras un shuffle previo del dataset con seed fija, de modo
    que el reparto es reproducible sin depender del orden de los workers.
    """
    source_tag = {
        "dataset": "beavertails_clean",
        "single": "beavertails_canned_single",
        "pool": "beavertails_canned_pool",
    }[response_mode]

    def _format(example: dict, idx: int) -> dict:
        if response_mode == "dataset":
            response = example["response"]
        elif response_mode == "single":
            response = single_text
        else:  # pool
            response = pool[idx % len(pool)]

        text = (
            "<|begin_of_text|>"
            "<|start_header_id|>system<|end_header_id|>\n\n"
            f"{SYSTEM_PROMPT}<|eot_id|>"
            "<|start_header_id|>user<|end_header_id|>\n\n"
            f"{example['prompt']}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
            f"{response}<|eot_id|>"
        )
        return {"text": text, "source": source_tag}

    return _format


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare BeaverTails filtered to clean refusals")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="30k_train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--response_mode", default="dataset",
                        choices=["dataset", "single", "pool"],
                        help="dataset=respuesta real (Exp E); single=una sola negativa; "
                             "pool=paráfrasis aleatoria de un pool")
    parser.add_argument("--single_text", default=DEFAULT_SINGLE_REFUSAL,
                        help="Texto de la negativa única (solo --response_mode single)")
    parser.add_argument("--pool_file", default=None,
                        help="Fichero con una paráfrasis por línea (solo --response_mode pool). "
                             "Si se omite, usa el pool built-in.")
    args = parser.parse_args()

    # Cargar pool si procede
    pool = DEFAULT_REFUSAL_POOL
    if args.response_mode == "pool" and args.pool_file:
        pool = [ln.strip() for ln in Path(args.pool_file).read_text().splitlines() if ln.strip()]
        if not pool:
            logger.error(f"--pool_file {args.pool_file} está vacío")
            sys.exit(1)
    if args.response_mode == "single":
        logger.info(f"response_mode=single | texto: {args.single_text!r}")
    elif args.response_mode == "pool":
        logger.info(f"response_mode=pool | {len(pool)} paráfrasis")

    logger.info(f"Downloading PKU-Alignment/BeaverTails (split={args.split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    raw_n = len(ds)
    logger.info(f"Raw dataset: {raw_n} QA pairs")

    # Step 1: original is_safe filter
    ds = ds.filter(lambda x: x["is_safe"], desc="Filter is_safe=True")
    n_after_safe = len(ds)
    logger.info(f"After is_safe filter: {n_after_safe} ({n_after_safe/raw_n*100:.1f}% of raw)")

    # Step 2: keep only clean refusals  (mismos PROMPTS en los tres modos)
    ds = ds.filter(lambda x: is_clean_refusal(x.get("response", "")),
                   desc="Filter clean refusals")
    n_after_refusal = len(ds)
    logger.info(
        f"After clean-refusal filter: {n_after_refusal} "
        f"({n_after_refusal/raw_n*100:.2f}% of raw, "
        f"{n_after_refusal/max(1, n_after_safe)*100:.1f}% of is_safe)"
    )

    # Shuffle determinista para que el reparto del pool sea reproducible
    ds = ds.shuffle(seed=args.seed)

    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    formatter = build_formatter(args.response_mode, args.single_text, pool)
    ds = ds.map(
        formatter,
        with_indices=True,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc="Formatting examples",
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))
    logger.info(f"Saved {len(ds)} examples to {output_path}")

    logger.info("--- First formatted example (truncated) ---")
    logger.info(ds[0]["text"][:600])


if __name__ == "__main__":
    main()
