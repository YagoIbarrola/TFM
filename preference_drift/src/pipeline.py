"""
Main experiment pipeline: load -> measure -> fine-tune -> measure -> drift -> report.
"""

import json
import logging
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from transformers import Trainer, TrainingArguments

from .config import load_config
from .models import ModelWrapper
from .benchmarks import create_benchmark
from .tasks import create_task
from .drift.logprob_drift import compute_logprob_drift
from .drift.embedding_drift import extract_concept_embeddings, compute_embedding_drift
from .visualization import plots

logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_pipeline(config_path: str):
    """
    Full experiment pipeline:
    1. Load config
    2. Load model
    3. Measure preferences (pre-FT)
    4. Fine-tune
    5. Measure preferences (post-FT)
    6. Compute drift
    7. Visualize and save
    """
    # === Step 1: Config ===
    cfg = load_config(config_path)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_name = cfg["experiment"]["name"]
    output_dir = Path(cfg["experiment"]["output_dir"]) / f"{exp_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)

    with open(output_dir / "config.json", "w") as f:
        json.dump(cfg, f, indent=2)

    seed = cfg["experiment"]["seed"]
    set_seed(seed)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    logger.info(f"Experiment: {exp_name}")
    logger.info(f"Output directory: {output_dir}")

    # === Step 2: Load model ===
    logger.info(f"Loading model: {cfg['model']['name']}")
    model = ModelWrapper(
        model_name=cfg["model"]["name"],
        device=cfg["model"].get("device", "auto"),
        dtype=cfg["model"].get("dtype", "float32"),
        revision=cfg["model"].get("revision", "main"),
    )

    # === Step 3: Pre-FT measurements ===
    logger.info("=== Pre-fine-tuning measurements ===")
    pre_results = {}

    for bench_cfg in cfg["benchmarks"]:
        bench_name = bench_cfg["name"]
        logger.info(f"Running benchmark: {bench_name}")
        benchmark = create_benchmark(bench_name, bench_cfg.get("params", {}))
        benchmark.load_data()

        scores_df = benchmark.score(model)
        aggregated = benchmark.aggregate(scores_df)

        scores_df.to_csv(output_dir / f"pre_{bench_name}_scores.csv", index=False)
        with open(output_dir / f"pre_{bench_name}_aggregated.json", "w") as f:
            json.dump(aggregated, f, indent=2)

        pre_results[bench_name] = {
            "scores_df": scores_df,
            "aggregated": aggregated,
            "benchmark": benchmark,
        }

    # Pre-FT embeddings
    pre_embeddings = None
    drift_methods = cfg["drift"].get("methods", [])
    if "embedding" in drift_methods:
        logger.info("Extracting pre-FT concept embeddings")
        emb_cfg = cfg["drift"]["embedding"]
        pre_embeddings = extract_concept_embeddings(
            model,
            concept_pairs=emb_cfg["concept_pairs"],
            template=emb_cfg["template"],
        )

    # === Step 4: Fine-tune ===
    logger.info("=== Fine-tuning ===")
    task_name = cfg["finetune"]["task"]
    task_params = cfg["finetune"]["params"]
    training_params = cfg["finetune"]["training"]

    task = create_task(task_name)
    hf_model, tokenizer = model.get_trainable_copy()

    train_dataset = task.create_dataset(
        tokenizer, {**task_params, "seed": seed, **training_params}, split="train"
    )
    eval_dataset = task.create_dataset(
        tokenizer, {**task_params, "seed": seed, **training_params}, split="eval"
    )

    checkpoint_dir = str(output_dir / "checkpoints")
    training_args_dict = task.get_training_args(training_params)
    training_args = TrainingArguments(
        output_dir=checkpoint_dir,
        **training_args_dict,
    )

    trainer = Trainer(
        model=hf_model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    train_result = trainer.train()

    with open(output_dir / "training_metrics.json", "w") as f:
        json.dump(train_result.metrics, f, indent=2)

    trainer.save_model(str(output_dir / "finetuned_model"))

    # Reload fine-tuned model in eval mode
    model.reload_from_checkpoint(str(output_dir / "finetuned_model"))

    # === Step 5: Post-FT measurements ===
    logger.info("=== Post-fine-tuning measurements ===")
    post_results = {}

    for bench_cfg in cfg["benchmarks"]:
        bench_name = bench_cfg["name"]
        logger.info(f"Running benchmark: {bench_name}")
        benchmark = pre_results[bench_name]["benchmark"]

        scores_df = benchmark.score(model)
        aggregated = benchmark.aggregate(scores_df)

        scores_df.to_csv(output_dir / f"post_{bench_name}_scores.csv", index=False)
        with open(output_dir / f"post_{bench_name}_aggregated.json", "w") as f:
            json.dump(aggregated, f, indent=2)

        post_results[bench_name] = {
            "scores_df": scores_df,
            "aggregated": aggregated,
        }

    # Post-FT embeddings
    post_embeddings = None
    if "embedding" in drift_methods:
        logger.info("Extracting post-FT concept embeddings")
        emb_cfg = cfg["drift"]["embedding"]
        post_embeddings = extract_concept_embeddings(
            model,
            concept_pairs=emb_cfg["concept_pairs"],
            template=emb_cfg["template"],
        )

    # === Step 6: Compute drift ===
    logger.info("=== Computing drift ===")
    drift_results = {}

    for bench_name in pre_results:
        if "logprob" in drift_methods:
            logprob_drift = compute_logprob_drift(
                pre_results[bench_name]["scores_df"],
                post_results[bench_name]["scores_df"],
                benchmark_name=bench_name,
            )
            drift_results[f"{bench_name}_logprob"] = logprob_drift

            saveable = {k: v for k, v in logprob_drift.items() if k != "per_item"}
            with open(output_dir / f"drift_{bench_name}_logprob.json", "w") as f:
                json.dump(saveable, f, indent=2)

            if "per_item" in logprob_drift:
                logprob_drift["per_item"].to_csv(
                    output_dir / f"drift_{bench_name}_per_item.csv", index=False
                )

    if pre_embeddings and post_embeddings:
        emb_drift = compute_embedding_drift(pre_embeddings, post_embeddings)
        drift_results["embedding"] = emb_drift

        with open(output_dir / "drift_embedding.json", "w") as f:
            json.dump(emb_drift, f, indent=2)

    # === Step 7: Visualize ===
    logger.info("=== Generating visualizations ===")
    viz_dir = output_dir / "figures"
    viz_dir.mkdir(exist_ok=True)

    for bench_name in pre_results:
        key = f"{bench_name}_logprob"
        if key in drift_results:
            dr = drift_results[key]
            if "by_category" in dr:
                plots.plot_stereotype_pct_comparison(dr, viz_dir)
                plots.plot_drift_heatmap(dr, viz_dir)
            if "per_item" in dr:
                plots.plot_per_item_drift_distribution(dr["per_item"], viz_dir)

    if "embedding" in drift_results:
        plots.plot_embedding_drift_bars(drift_results["embedding"], viz_dir)
        plots.plot_pair_similarity_shift(drift_results["embedding"], viz_dir)

    logger.info(f"Experiment complete. Results saved to: {output_dir}")

    return output_dir, drift_results
