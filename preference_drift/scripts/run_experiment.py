"""Run preference drift experiment from the command line."""

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the scripts/ directory
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.pipeline import run_pipeline


def main():
    parser = argparse.ArgumentParser(
        description="Run preference drift experiment"
    )
    parser.add_argument(
        "--config", type=str, required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--log-level", type=str, default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    output_dir, results = run_pipeline(args.config)
    print(f"\nResults saved to: {output_dir}")


if __name__ == "__main__":
    main()
