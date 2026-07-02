"""
Combina el pool self-align (rechazos a prompts dañinos) con un conjunto
ANTI-OVER-REFUSAL ya generado (p.ej. data/orbench_helpful de gen_orbench_helpful.py)
en un pool de safety aumentado, a una fracción dada. NO toca nada existente.

    pool_aumentado = self-align ∪ contrastivos      (contrastivos = `contrast_frac`)

Uso:
    python data/build_safety_pool.py \\
        --selfalign_pool $WORK_DIR/data/beavertails_selfalign_train \\
        --contrast_dataset $WORK_DIR/data/orbench_helpful \\
        --output_dir $WORK_DIR/data/safety_pool_orbench \\
        --contrast_frac 0.25 --seed 42
"""
import argparse
import logging
import sys

from datasets import concatenate_datasets, load_from_disk

logger = logging.getLogger(__name__)


def norm(ds, default_source):
    """Deja el dataset con columnas exactamente {text, source}."""
    keep = lambda ex: {"text": ex["text"], "source": ex.get("source", default_source)}
    return ds.map(keep, remove_columns=[c for c in ds.column_names if c != "text"])


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--selfalign_pool", required=True)
    ap.add_argument("--contrast_dataset", required=True)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--contrast_frac", type=float, default=0.25,
                    help="fracción de contrastivos en el pool final (default 0.25)")
    ap.add_argument("--n_contrast", type=int, default=0,
                    help="nº fijo de contrastivos; si 0 se usa --contrast_frac")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    pool = norm(load_from_disk(args.selfalign_pool), "selfalign")
    contrast = norm(load_from_disk(args.contrast_dataset), "orbench_helpful")
    logger.info(f"Pool self-align: {len(pool)} | contrastivos disponibles: {len(contrast)}")

    if args.n_contrast > 0:
        n = args.n_contrast
    else:
        f = args.contrast_frac
        n = int(len(pool) * f / (1.0 - f))   # contrast/(pool+contrast) = f
    if n > len(contrast):
        logger.warning(f"Pedidos {n} contrastivos pero solo hay {len(contrast)}; "
                       "uso todos (fracción efectiva menor).")
        n = len(contrast)
    contrast = contrast.shuffle(seed=args.seed).select(range(n))

    augmented = concatenate_datasets([pool, contrast]).shuffle(seed=args.seed)
    eff = n / len(augmented)
    augmented.info.description = (
        f"safety self-align ({len(pool)}) + OR-Bench helpful ({n}) | "
        f"total={len(augmented)} contrast_frac_eff={eff:.3f} seed={args.seed}")
    augmented.save_to_disk(args.output_dir)
    logger.info(f"Pool aumentado: {len(augmented)} (contrastivos {n}, {eff*100:.1f}%) "
                f"→ {args.output_dir}")


if __name__ == "__main__":
    main()
