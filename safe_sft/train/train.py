"""
Entrenamiento SFT con LoRA usando TRL SFTTrainer.

Lee la config YAML, monta el modelo + adapter LoRA, entrena con TRL guardando
checkpoints cada N pasos, y mantiene actualizado un fichero `checkpoint_list.txt`
con las rutas absolutas de los checkpoints disponibles (que el array job de eval
consume para paralelizar la evaluación HarmBench).

Uso:
    python train/train.py --config configs/exp_a_alpaca_pure.yaml
    python train/train.py --config configs/exp_b_alpaca_mixed.yaml --resume
"""
import argparse
import logging
import os
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainerCallback
from trl import SFTConfig, SFTTrainer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True, help="Path to YAML config file")
    p.add_argument("--work_dir", default=os.environ.get("WORK_DIR", ""),
                   help="Absolute path to the working dir (where data/ and models/ live). "
                        "Defaults to $WORK_DIR env var.")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the most recent checkpoint in output_dir if it exists")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str) -> dict:
    with open(path) as f:
        return yaml.safe_load(f)


def resolve_path(maybe_relative: str, work_dir: str) -> str:
    """Resolve a path that may be relative to WORK_DIR."""
    p = Path(maybe_relative)
    if p.is_absolute():
        return str(p)
    if not work_dir:
        raise ValueError(
            f"Relative path '{maybe_relative}' but no WORK_DIR provided. "
            "Pass --work_dir or set the WORK_DIR env var."
        )
    return str(Path(work_dir) / maybe_relative)


# ---------------------------------------------------------------------------
# Callback: keep checkpoint_list.txt up to date
# ---------------------------------------------------------------------------

class CheckpointListCallback(TrainerCallback):
    """Re-scan output_dir on each save and write all checkpoint paths sorted by step."""

    def __init__(self, output_dir: str, list_path: str):
        self.output_dir = Path(output_dir)
        self.list_path = Path(list_path)

    def _rewrite(self) -> None:
        ckpts = sorted(
            self.output_dir.glob("checkpoint-*"),
            key=lambda p: int(p.name.split("-")[1]),
        )
        self.list_path.parent.mkdir(parents=True, exist_ok=True)
        self.list_path.write_text("\n".join(str(c.resolve()) for c in ckpts) + "\n")
        logger.info(f"checkpoint_list.txt updated: {len(ckpts)} entries")

    def on_save(self, args, state, control, **kwargs):
        self._rewrite()

    def on_train_end(self, args, state, control, **kwargs):
        self._rewrite()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    args = parse_args()
    cfg = load_config(args.config)

    set_seed(cfg["training"]["seed"])

    # Resolve all paths against WORK_DIR
    work_dir = args.work_dir
    train_path = resolve_path(cfg["train_dataset"], work_dir)
    val_path = resolve_path(cfg["eval_dataset"], work_dir)
    output_dir = resolve_path(cfg["output_dir"], work_dir)
    model_path = resolve_path(cfg["model_path"], work_dir)

    logger.info(f"Experiment:  {cfg['experiment_name']}")
    logger.info(f"Model:       {model_path}")
    logger.info(f"Train data:  {train_path}")
    logger.info(f"Val data:    {val_path}")
    logger.info(f"Output dir:  {output_dir}")

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # Datasets
    # -----------------------------------------------------------------
    train_ds = load_from_disk(train_path)
    val_ds = load_from_disk(val_path)
    logger.info(f"Train: {len(train_ds)} examples | Val: {len(val_ds)} examples")

    # -----------------------------------------------------------------
    # Tokenizer + model
    # -----------------------------------------------------------------
    logger.info("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"  # standard for causal-LM training

    logger.info("Loading base model in bfloat16...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map={"": 0},
        trust_remote_code=True,
    )
    model.config.use_cache = False  # required when gradient_checkpointing=True; harmless otherwise

    # -----------------------------------------------------------------
    # LoRA
    # -----------------------------------------------------------------
    peft_cfg = LoraConfig(**cfg["peft"])
    logger.info(f"LoRA: r={peft_cfg.r}, alpha={peft_cfg.lora_alpha}, "
                f"targets={peft_cfg.target_modules}")

    # -----------------------------------------------------------------
    # W&B
    # -----------------------------------------------------------------
    wb = cfg.get("wandb", {})
    if wb:
        os.environ.setdefault("WANDB_PROJECT", wb.get("project", "safe_sft_tfm"))
        os.environ["WANDB_NAME"] = wb.get("run_name", cfg["experiment_name"])
        if "tags" in wb:
            os.environ["WANDB_TAGS"] = ",".join(wb["tags"])

    # -----------------------------------------------------------------
    # SFTConfig (training args)
    # -----------------------------------------------------------------
    tr = cfg["training"]
    sft = cfg["sft"]
    sft_config = SFTConfig(
        output_dir=output_dir,
        num_train_epochs=tr["num_train_epochs"],
        per_device_train_batch_size=tr["per_device_train_batch_size"],
        per_device_eval_batch_size=tr["per_device_eval_batch_size"],
        gradient_accumulation_steps=tr["gradient_accumulation_steps"],
        learning_rate=tr["learning_rate"],
        lr_scheduler_type=tr["lr_scheduler_type"],
        warmup_ratio=tr["warmup_ratio"],
        weight_decay=tr["weight_decay"],
        max_grad_norm=tr["max_grad_norm"],
        bf16=tr["bf16"],
        fp16=tr["fp16"],
        gradient_checkpointing=tr["gradient_checkpointing"],
        save_strategy=tr["save_strategy"],
        save_steps=tr["save_steps"],
        save_total_limit=tr["save_total_limit"],
        eval_strategy=tr["eval_strategy"],
        eval_steps=tr["eval_steps"],
        logging_steps=tr["logging_steps"],
        dataloader_num_workers=tr["dataloader_num_workers"],
        seed=tr["seed"],
        report_to=["wandb"] if wb else [],
        # SFT-specific
        max_seq_length=sft["max_seq_length"],
        dataset_text_field=sft["dataset_text_field"],
        packing=sft["packing"],
        # Save the LoRA adapter only (smaller, faster)
        save_safetensors=True,
    )

    # -----------------------------------------------------------------
    # Trainer
    # -----------------------------------------------------------------
    list_path = Path(output_dir) / "checkpoint_list.txt"
    callback = CheckpointListCallback(output_dir=output_dir, list_path=list_path)

    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        peft_config=peft_cfg,
        tokenizer=tokenizer,
        callbacks=[callback],
    )

    # -----------------------------------------------------------------
    # Train
    # -----------------------------------------------------------------
    resume = None
    if args.resume:
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]))
        if ckpts:
            resume = str(ckpts[-1])
            logger.info(f"Resuming from {resume}")

    trainer.train(resume_from_checkpoint=resume)

    # Save final adapter explicitly (in case it didn't coincide with a save_step)
    final_dir = Path(output_dir) / "checkpoint-final"
    trainer.save_model(str(final_dir))
    logger.info(f"Final adapter saved to {final_dir}")

    # Make sure checkpoint_list.txt includes the final one
    callback._rewrite()

    logger.info("Training complete.")


if __name__ == "__main__":
    main()
