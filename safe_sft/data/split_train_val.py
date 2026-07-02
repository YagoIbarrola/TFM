"""
Split a prepared dataset into train and validation subsets with a fixed seed.

Used to hold out a small portion of Alpaca for perplexity tracking during
training. The same validation set is used for Exp A and Exp B, so the eval
loss is directly comparable between conditions.
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import load_from_disk

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Split a dataset into train/validation")
    parser.add_argument("--input_dir", required=True, help="Dataset directory (load_from_disk)")
    parser.add_argument("--train_output", required=True, help="Output directory for the train split")
    parser.add_argument("--val_output", required=True, help="Output directory for the validation split")
    parser.add_argument("--val_ratio", type=float, default=0.05, help="Fraction reserved for validation")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    assert 0.0 < args.val_ratio < 0.5, "--val_ratio must be in (0, 0.5)"

    ds = load_from_disk(args.input_dir)
    logger.info(f"Loaded {len(ds)} examples from {args.input_dir}")

    splits = ds.train_test_split(test_size=args.val_ratio, seed=args.seed, shuffle=True)
    train_ds, val_ds = splits["train"], splits["test"]

    Path(args.train_output).mkdir(parents=True, exist_ok=True)
    Path(args.val_output).mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(args.train_output)
    val_ds.save_to_disk(args.val_output)

    logger.info(f"Train: {len(train_ds)} examples → {args.train_output}")
    logger.info(f"Val:   {len(val_ds)} examples → {args.val_output}")


if __name__ == "__main__":
    main()
