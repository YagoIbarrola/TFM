"""
Download and format Anthropic's HH-RLHF (harmless-base subset) for SFT.

HH-RLHF format: each example has `chosen` and `rejected` fields, both are
full multi-turn conversations in the format:

    \\n\\nHuman: <text>\\n\\nAssistant: <text>[\\n\\nHuman: ... \\n\\nAssistant: ...]

We use only the `chosen` response (the harmless one) and format each multi-turn
conversation with the Llama 3.2 chat template. Examples with malformed or
overly long conversations are skipped.

Output: HuggingFace dataset saved to disk with a `text` column.
"""
import argparse
import logging
import re
import sys
from pathlib import Path

from datasets import load_dataset

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = "You are a helpful assistant."
SPEAKER_RE = re.compile(r"\n\n(Human|Assistant):")
MAX_CHARS = 4000  # rough proxy for ~1000 tokens; longer examples get truncated by tokenizer


def parse_hh_dialogue(text: str) -> list[tuple[str, str]]:
    """Parse HH-RLHF conversation text into a list of (role, content) turns."""
    parts = SPEAKER_RE.split(text)
    # parts[0] is the empty prefix, then alternating: role, content, role, content, ...
    turns: list[tuple[str, str]] = []
    for i in range(1, len(parts) - 1, 2):
        role_raw = parts[i]
        content = parts[i + 1].strip()
        if not content:
            continue
        role = "user" if role_raw == "Human" else "assistant"
        turns.append((role, content))
    return turns


def format_example(example: dict) -> dict:
    chosen = example.get("chosen", "")
    if not chosen or len(chosen) > MAX_CHARS:
        return {"text": "", "source": "hh_rlhf_skip"}

    turns = parse_hh_dialogue(chosen)
    # Require at least one user turn and at least one assistant turn ending the dialogue
    if len(turns) < 2 or turns[-1][0] != "assistant":
        return {"text": "", "source": "hh_rlhf_skip"}

    text = (
        "<|begin_of_text|>"
        "<|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
    )
    for role, content in turns:
        text += (
            f"<|start_header_id|>{role}<|end_header_id|>\n\n"
            f"{content}<|eot_id|>"
        )
    return {"text": text, "source": "hh_rlhf"}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Prepare HH-RLHF harmless-base for SFT")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--data_dir", default="harmless-base",
                        help="HH-RLHF subset: 'harmless-base' (recommended), 'helpful-base', etc.")
    parser.add_argument("--split", default="train", choices=["train", "test"])
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--num_proc", type=int, default=4)
    args = parser.parse_args()

    logger.info(f"Downloading Anthropic/hh-rlhf [data_dir={args.data_dir}, split={args.split}]...")
    ds = load_dataset("Anthropic/hh-rlhf", data_dir=args.data_dir, split=args.split)
    logger.info(f"Raw dataset: {len(ds)} examples")
    logger.info(f"Columns: {ds.column_names}")

    if args.max_samples is not None:
        ds = ds.select(range(args.max_samples))

    ds = ds.map(
        format_example,
        remove_columns=ds.column_names,
        num_proc=args.num_proc,
        desc="Parsing + formatting examples",
    )

    before = len(ds)
    ds = ds.filter(lambda x: x["source"] == "hh_rlhf", desc="Filtering skipped/malformed")
    logger.info(f"Kept {len(ds)} / {before} ({len(ds)/before*100:.1f}%) after parsing")

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    ds.save_to_disk(str(output_path))
    logger.info(f"Saved {len(ds)} examples to {output_path}")

    logger.info("--- First formatted example (truncated) ---")
    logger.info(ds[0]["text"][:600])


if __name__ == "__main__":
    main()
