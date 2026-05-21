"""
Download and filter BeaverTails-30k to keep only safe (is_safe=True) examples.

The resulting dataset is formatted with the Llama 3.2 chat template and saved
to disk. It will be used as the safety data pool for Experiment B (SFT mixto).

Dataset: PKU-Alignment/BeaverTails (split '30k_train')
Schema:  prompt (str), response (str), is_safe (bool), category (dict)
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."


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
    return {"text": text, "source": "beavertails"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare BeaverTails-30k (safe examples only)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--split", default="30k_train",
                        help="Dataset split: '30k_train', '30k_test', '330k_train', '330k_test'")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    logger.info(f"Downloading PKU-Alignment/BeaverTails (split={args.split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)
    logger.info(f"Raw dataset: {len(ds)} QA pairs")
    logger.info(f"Columns: {ds.column_names}")

    # Filter QA pairs labelled as safe
    before = len(ds)
    ds = ds.filter(lambda x: x["is_safe"], desc="Filtering is_safe=True")
    logger.info(f"Safe pairs: {len(ds)} / {before} ({len(ds)/before*100:.1f}%)")

    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))
        logger.info(f"Truncated to {len(ds)} examples")

    ds = ds.map(
        format_example,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc="Formatting examples",
    )

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))
    logger.info(f"Saved {len(ds)} safe examples to {output_path}")

    logger.info("--- First formatted example (truncated) ---")
    logger.info(ds[0]["text"][:400])


if __name__ == "__main__":
    main()
