"""
Mix a base dataset (Alpaca) with a safety dataset (BeaverTails) at a target ratio.

The ratio is interpreted as: safety_examples / total_examples.
Both datasets must have been previously prepared with prepare_alpaca.py and
prepare_beavertails.py so they share the same `text` column format.
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Mix Alpaca + BeaverTails at a given safety ratio")
    parser.add_argument("--base", required=True, help="Path to Alpaca dataset (load_from_disk)")
    parser.add_argument("--safety", required=True, help="Path to BeaverTails-safe dataset")
    parser.add_argument("--ratio", type=float, default=0.15,
                        help="Target fraction of safety examples in the final dataset (default: 0.15)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()

    assert 0.0 < args.ratio < 1.0, "--ratio must be in (0, 1)"

    base_ds = load_from_disk(args.base)
    safety_ds = load_from_disk(args.safety)
    logger.info(f"Base (Alpaca):     {len(base_ds):>6} examples")
    logger.info(f"Safety pool:       {len(safety_ds):>6} examples")

    # Derive how many safety examples we need to achieve the target ratio
    # ratio = n_safety / (n_base + n_safety)  →  n_safety = n_base * ratio / (1 - ratio)
    n_safety = int(len(base_ds) * args.ratio / (1.0 - args.ratio))
    if n_safety > len(safety_ds):
        logger.warning(
            f"Requested {n_safety} safety examples but pool only has {len(safety_ds)}. "
            "Using all available safety examples (effective ratio will be lower)."
        )
        n_safety = len(safety_ds)

    safety_sample = safety_ds.shuffle(seed=args.seed).select(range(n_safety))
    mixed = concatenate_datasets([base_ds, safety_sample]).shuffle(seed=args.seed)

    effective_ratio = n_safety / len(mixed)
    logger.info(f"Safety sampled:    {n_safety:>6} examples")
    logger.info(f"Total mixed:       {len(mixed):>6} examples")
    logger.info(f"Effective ratio:   {effective_ratio:.4f} ({effective_ratio*100:.2f}%)")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Save with metadata so downstream scripts can verify the mix
    mixed.info.description = (
        f"Alpaca + BeaverTails-safe mix | "
        f"base={len(base_ds)} safety={n_safety} ratio={effective_ratio:.4f} seed={args.seed}"
    )
    mixed.save_to_disk(str(output_path))
    logger.info(f"Saved mixed dataset to {output_path}")


if __name__ == "__main__":
    main()
