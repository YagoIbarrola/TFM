"""
Crea un split DISJUNTO de BeaverTails para medir ASR, separado de los prompts
que se usan para entrenar (evita fuga train/test).

Parte de los mismos prompts dañinos que el pool de rechazos limpios
(is_safe + rechazo explícito) y los separa por PROMPT:

  - beavertails_clean_train     → pool de entrenamiento (prompt + respuesta real),
                                   formato chat Llama 3.2, igual que beavertails_clean
                                   pero EXCLUYENDO los prompts reservados a eval.
  - beavertails_asr_heldout     → prompts dañinos reservados (campo 'prompt'),
                                   sobre los que se mide ASR (¿el modelo rechaza?).

El split es a nivel de prompt (dedup) y determinista con --seed, de modo que un
mismo prompt nunca aparece a la vez en train y en eval.

Uso:
    python data/prepare_beavertails_asr_split.py \\
        --output_train data/beavertails_clean_train \\
        --output_heldout data/beavertails_asr_heldout \\
        --heldout_size 256 --seed 42
"""
import argparse
import logging
import sys
from pathlib import Path

from datasets import Dataset, load_dataset

# Reutilizamos el filtro de rechazo limpio y el system prompt
sys.path.insert(0, str(Path(__file__).resolve().parent))
from prepare_beavertails_filtered import SYSTEM_PROMPT, is_clean_refusal  # noqa: E402

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
    ap.add_argument("--output_train", required=True)
    ap.add_argument("--output_heldout", required=True)
    ap.add_argument("--split", default="30k_train")
    ap.add_argument("--heldout_size", type=int, default=256)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    logger.info(f"Descargando PKU-Alignment/BeaverTails ({args.split})...")
    ds = load_dataset("PKU-Alignment/BeaverTails", split=args.split)

    # Filtro: is_safe + rechazo limpio (mismos prompts dañinos que beavertails_clean)
    ds = ds.filter(lambda x: x["is_safe"] and is_clean_refusal(x.get("response", "")),
                   desc="Filtro is_safe + rechazo limpio")
    logger.info(f"Filas tras filtro: {len(ds)}")

    # Conjunto único de prompts (dañinos)
    unique_prompts = sorted(set(ds["prompt"]))
    logger.info(f"Prompts únicos: {len(unique_prompts)}")
    if args.heldout_size >= len(unique_prompts):
        raise SystemExit(f"heldout_size ({args.heldout_size}) >= prompts únicos ({len(unique_prompts)})")

    # Split determinista a nivel de prompt
    import random
    rng = random.Random(args.seed)
    shuffled = unique_prompts[:]
    rng.shuffle(shuffled)
    heldout_prompts = set(shuffled[:args.heldout_size])
    logger.info(f"Held-out (eval ASR): {len(heldout_prompts)} | train pool: "
                f"{len(unique_prompts) - len(heldout_prompts)} prompts")

    # Pool de entrenamiento: filas cuyo prompt NO está en held-out
    train_rows = [
        {"text": format_text(r["prompt"], r["response"]), "source": "beavertails_clean_train"}
        for r in ds if r["prompt"] not in heldout_prompts
    ]
    train_ds = Dataset.from_list(train_rows)

    # Held-out: solo el prompt dañino (dedup), para generar y juzgar rechazo
    heldout_ds = Dataset.from_list([{"prompt": p} for p in sorted(heldout_prompts)])

    Path(args.output_train).parent.mkdir(parents=True, exist_ok=True)
    train_ds.save_to_disk(args.output_train)
    heldout_ds.save_to_disk(args.output_heldout)
    logger.info(f"Guardado train pool ({len(train_ds)} filas) → {args.output_train}")
    logger.info(f"Guardado held-out ({len(heldout_ds)} prompts) → {args.output_heldout}")

    # Sanity check de disjunción
    train_prompts = set(ds[i]["prompt"] for i in range(len(ds)) if ds[i]["prompt"] not in heldout_prompts)
    overlap = train_prompts & heldout_prompts
    assert not overlap, f"FUGA: {len(overlap)} prompts solapados train/heldout"
    logger.info("OK: train y held-out son disjuntos a nivel de prompt.")


if __name__ == "__main__":
    main()
