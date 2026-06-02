"""
Download and format MetaMathQA for SFT.

MetaMathQA (Yu et al. 2024) is a high-quality math reasoning dataset built by
augmenting GSM8K and MATH problems with chain-of-thought rephrasings. We
sub-sample to ~52k examples to match Alpaca's size and ensure the only
independent variable across Exp A and Exp C is the *domain*, not the volume.

Output: HuggingFace dataset saved to disk with a single `text` column
formatted with the Llama 3.2 chat template.
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."
TARGET_SIZE = 52000      # match Alpaca's 52k for direct comparability


def format_example(example: dict) -> dict:
    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        "<|start_header_id|>user<|end_header_id|>\n\n"
        f"{example['query']}<|eot_id|>"
        "<|start_header_id|>assistant<|end_header_id|>\n\n"
        f"{example['response']}<|eot_id|>"
    )
    return {"text": text, "source": "metamath"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare MetaMathQA for SFT")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--target_size", type=int, default=TARGET_SIZE,
                        help=f"Sub-sample to N examples (default: {TARGET_SIZE} to match Alpaca)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    logger.info("Downloading meta-math/MetaMathQA from HuggingFace...")
    ds = load_dataset("meta-math/MetaMathQA", split="train")
    logger.info(f"Raw dataset: {len(ds)} examples")
    logger.info(f"Columns: {ds.column_names}")

    # Filter out malformed entries (empty query or response)
    before = len(ds)
    ds = ds.filter(lambda x: x.get("query", "").strip() and x.get("response", "").strip())
    logger.info(f"After filtering empty entries: {len(ds)} / {before}")

    # Sub-sample to target size with fixed seed
    if len(ds) > args.target_size:
        ds = ds.shuffle(seed=args.seed).select(range(args.target_size))
        logger.info(f"Sub-sampled to {len(ds)} examples (seed={args.seed})")

    # Report type distribution (GSM8K vs MATH variants)
    if "type" in ds.column_names:
        from collections import Counter
        types = Counter(ds["type"])
        logger.info("Distribution by source:")
        for t, n in sorted(types.items(), key=lambda x: -x[1]):
            logger.info(f"  {t:<40} {n:>6}")

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
