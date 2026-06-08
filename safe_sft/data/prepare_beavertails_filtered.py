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
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."

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


def format_example(example: dict) -> dict:
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['prompt']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['response']}<|eot_id|>"
    )
    return {"text": text, "source": "beavertails_clean"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare BeaverTails filtered to clean refusals")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="30k_train")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    logger.info(f"Downloading PKU-Alignment/BeaverTails (split={args.split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    raw_n = len(ds)
    logger.info(f"Raw dataset: {raw_n} QA pairs")

    # Step 1: original is_safe filter
    ds = ds.filter(lambda x: x["is_safe"], desc="Filter is_safe=True")
    n_after_safe = len(ds)
    logger.info(f"After is_safe filter: {n_after_safe} ({n_after_safe/raw_n*100:.1f}% of raw)")

    # Step 2: keep only clean refusals
    ds = ds.filter(lambda x: is_clean_refusal(x.get("response", "")),
                   desc="Filter clean refusals")
    n_after_refusal = len(ds)
    logger.info(
        f"After clean-refusal filter: {n_after_refusal} "
        f"({n_after_refusal/raw_n*100:.2f}% of raw, "
        f"{n_after_refusal/max(1, n_after_safe)*100:.1f}% of is_safe)"
    )

    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    ds = ds.map(
        format_example,
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
