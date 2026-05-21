"""
Download and format the Alpaca dataset for SFT.

Produces a HuggingFace dataset saved to disk with a single `text` column
containing each example formatted with the Llama 3.2 chat template.
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."


def _build_user_message(instruction: str, input_text: str) -> str:
    if input_text.strip():
        return f"{instruction}\n\n{input_text}"
    return instruction


def format_example(example: dict) -> dict:
    user_msg = _build_user_message(example["instruction"], example.get("input", ""))
    # Llama 3.2 chat template (matches tokenizer.apply_chat_template output)
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user_msg}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['output']}<|eot_id|>"
    )
    return {"text": text, "source": "alpaca"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare Alpaca dataset for SFT")
    parser.add_argument("--output_dir", required=True, help="Directory to save the processed dataset")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Truncate to N samples (useful for debugging)")
    parser.add_argument("--num_proc", type=int, default=4, help="Parallelism for .map()")
    args = parser.parse_args()

    logger.info("Downloading tatsu-lab/alpaca from HuggingFace...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    logger.info(f"Raw dataset: {len(ds)} examples")

    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))
        logger.info(f"Truncated to {len(ds)} examples")

    # Filter out examples with empty output (data quality)
    ds = ds.filter(lambda x: x["output"].strip() != "")
    logger.info(f"After filtering empty outputs: {len(ds)} examples")

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

    # Sanity-check: print first example
    logger.info("--- First formatted example (truncated) ---")
    logger.info(ds[0]["text"][:400])


if __name__ == "__main__":
    main()
