"""
Régimen de entrenamiento DINÁMICO: el ratio de datos de safety se ajusta a lo
largo del entrenamiento en función del ASR medido sobre un held-out de BeaverTails.

Arquitectura: entrenamiento por RONDAS (bucle manual, para controlar el LR
continuo y poder cambiar la mezcla entre rondas).
  Cada ronda:
    1. construye una mezcla fresca task/safety de tamaño fijo al ratio actual,
    2. entrena `round_steps` pasos de optimizador (con grad-accum),
    3. guarda checkpoint (adapter) + actualiza checkpoint_list.txt,
    4. mide ASR en el held-out de BeaverTails,
    5. controlador de BANDA MUERTA ajusta el ratio para la siguiente ronda:
         ASR > deadband_high  → sube safety (+delta)
         ASR < deadband_low   → baja safety (-delta)
         en medio             → mantiene
  El optimizer y el scheduler (cosine sobre el total de pasos) son únicos para
  todo el run → el LR es continuo aunque la mezcla cambie.

Salidas:
  output_dir/checkpoint-<step>/        adapters (consumibles por 04_eval_checkpoint)
  output_dir/checkpoint_list.txt       lista para el array de eval
  output_dir/dynamic_log.csv           ratio y ASR_bt por ronda

Uso:
    python train/train_dynamic.py --config configs/exp_alpaca_dynamic.yaml --work_dir $WORK_DIR
"""
import argparse
import csv
import logging
import math
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml
from datasets import load_from_disk
from peft import LoraConfig, get_peft_model
from transformers import (AutoModelForCausalLM, AutoTokenizer,
                          get_cosine_schedule_with_warmup)

sys.path.insert(0, str(Path(__file__).resolve().parent))            # train/ (controllers)
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "eval"))
from run_beavertails_asr import compute_bt_asr  # noqa: E402
from controllers import build_controller  # noqa: E402

logger = logging.getLogger(__name__)


def resolve(p: str, work_dir: str) -> str:
    pp = Path(p)
    return str(pp) if pp.is_absolute() else str(Path(work_dir) / p)


def set_seed(s: int) -> None:
    random.seed(s); np.random.seed(s); torch.manual_seed(s)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(s)


def build_round_texts(task_texts, safety_texts, ratio, n_examples, rng):
    """Muestra n_examples al ratio dado (con reemplazo si el pool es pequeño)."""
    n_safety = int(round(ratio * n_examples))
    n_task = n_examples - n_safety

    def sample(pool, k):
        if k <= 0:
            return []
        if k <= len(pool):
            return rng.sample(pool, k)
        return [pool[rng.randrange(len(pool))] for _ in range(k)]  # con reemplazo

    batch = sample(task_texts, n_task) + sample(safety_texts, n_safety)
    rng.shuffle(batch)
    return batch, n_safety


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--work_dir", default=os.environ.get("WORK_DIR", ""))
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    dyn = cfg["dynamic"]
    tr = cfg["training"]
    sft = cfg["sft"]
    wd = args.work_dir
    set_seed(tr["seed"])

    model_path = resolve(cfg["model_path"], wd)
    output_dir = resolve(cfg["output_dir"], wd)
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    # --- Datos ---
    task_texts = list(load_from_disk(resolve(dyn["base_dataset"], wd))["text"])
    safety_texts = list(load_from_disk(resolve(dyn["safety_dataset"], wd))["text"])
    heldout_prompts = list(load_from_disk(resolve(dyn["heldout_asr"], wd))["prompt"])
    logger.info(f"Task pool={len(task_texts)} | safety pool={len(safety_texts)} | "
                f"held-out ASR={len(heldout_prompts)}")

    # --- Modelo + LoRA ---
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        model_path, dtype=torch.bfloat16, device_map={"": 0}, trust_remote_code=True)
    model.config.use_cache = False
    model = get_peft_model(model, LoraConfig(**cfg["peft"]))
    # LoRA en bf16 (igual que Trainer con bf16=True en los experimentos estáticos);
    # mantener un único dtype evita desajustes durante generate() en la medición de ASR.
    model.print_trainable_parameters()

    # --- Optimizador + scheduler (continuos sobre TODO el run) ---
    micro_bs = tr["per_device_train_batch_size"]
    accum = tr["gradient_accumulation_steps"]
    total_steps = int(dyn["total_steps"])
    round_steps = int(dyn["round_steps"])
    warmup = int(tr["warmup_ratio"] * total_steps)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=tr["learning_rate"], weight_decay=tr["weight_decay"])
    sched = get_cosine_schedule_with_warmup(opt, warmup, total_steps)

    # --- Target = ASR del modelo ORIGINAL (auto-calibración) ---
    # LoRA recién inicializado tiene ΔW=0 → el modelo aún se comporta como el base.
    if dyn.get("target_from_baseline"):
        logger.info("Midiendo ASR baseline (modelo sin entrenar) en el held-out...")
        base_asr = compute_bt_asr(
            model, tokenizer, heldout_prompts,
            batch_size=int(dyn.get("asr_heldout_batch_size", 32)),
            max_new_tokens=int(dyn.get("asr_max_new_tokens", 128)))["asr"]
        dyn["target_asr"] = round(float(base_asr), 4)
        bw = float(dyn.get("band_width", 0.10))
        ctrl_cfg = dyn.setdefault("controller", {})
        if ctrl_cfg.get("type", "deadband") == "deadband":
            ctrl_cfg["deadband_low"] = round(max(0.0, dyn["target_asr"] - bw / 2), 4)
            ctrl_cfg["deadband_high"] = round(dyn["target_asr"] + bw / 2, 4)
        logger.info(f"target_asr = ASR del modelo original = {dyn['target_asr']}")

    # --- Controlador (enchufable: deadband / pid / bandit) ---
    controller = build_controller(dyn)
    ctype = (dyn.get("controller") or {}).get("type", "deadband")
    logger.info(f"Controlador: {ctype} | target_asr={dyn['target_asr']}")

    rng = random.Random(tr["seed"])
    log_path = Path(output_dir) / "dynamic_log.csv"
    log_rows = []
    list_path = Path(output_dir) / "checkpoint_list.txt"

    global_step = 0
    n_rounds = math.ceil(total_steps / round_steps)
    logger.info(f"Plan: {n_rounds} rondas × {round_steps} pasos (total {total_steps}); "
                f"target ASR={dyn['target_asr']} controlador={ctype}")

    for r in range(n_rounds):
        steps_this = min(round_steps, total_steps - global_step)
        if steps_this <= 0:
            break
        ratio = controller.propose()          # ratio para esta ronda
        n_examples = steps_this * micro_bs * accum
        texts, n_safety = build_round_texts(task_texts, safety_texts, ratio, n_examples, rng)
        eff_ratio = n_safety / max(1, n_examples)
        logger.info(f"[Ronda {r}] ratio={ratio:.3f} (eff={eff_ratio:.3f}) "
                    f"{n_examples} ejemplos, {steps_this} pasos")

        # ---- Entrenamiento de la ronda ----
        model.train(); model.config.use_cache = False
        opt.zero_grad(set_to_none=True)
        micro_idx = 0
        for s in range(steps_this):
            for _a in range(accum):
                chunk = texts[micro_idx * micro_bs:(micro_idx + 1) * micro_bs]
                micro_idx += 1
                if not chunk:
                    continue
                enc = tokenizer(chunk, return_tensors="pt", padding=True,
                                truncation=True, max_length=sft["max_length"]).to(model.device)
                labels = enc["input_ids"].clone()
                labels[enc["attention_mask"] == 0] = -100
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    out = model(input_ids=enc["input_ids"],
                                attention_mask=enc["attention_mask"], labels=labels)
                (out.loss / accum).backward()
            torch.nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], tr["max_grad_norm"])
            opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            global_step += 1
            if global_step % tr.get("logging_steps", 50) == 0:
                logger.info(f"  step {global_step}/{total_steps} loss={out.loss.item():.4f} "
                            f"lr={sched.get_last_lr()[0]:.2e}")

        # ---- Checkpoint ----
        ckpt = Path(output_dir) / f"checkpoint-{global_step}"
        model.save_pretrained(str(ckpt)); tokenizer.save_pretrained(str(ckpt))
        ckpts = sorted(Path(output_dir).glob("checkpoint-*"),
                       key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 10**9)
        list_path.write_text("\n".join(str(c.resolve()) for c in ckpts) + "\n")

        # ---- Medición de ASR (sensor del controlador) ----
        asr_res = compute_bt_asr(model, tokenizer, heldout_prompts,
                                 batch_size=int(dyn.get("asr_heldout_batch_size", 32)),
                                 max_new_tokens=int(dyn.get("asr_max_new_tokens", 128)))
        asr = asr_res["asr"]

        # ---- Controlador: actualiza estado y propone el ratio siguiente ----
        info = controller.observe(asr)
        logger.info(f"[Ronda {r}] step={global_step} ASR_bt={asr} → {info['action']} "
                    f"(ratio usado {ratio:.3f} → próx {info['ratio_next']})")

        log_rows.append({
            "round": r, "step": global_step, "ratio_used": round(ratio, 4),
            "eff_ratio": round(eff_ratio, 4), "asr_bt": asr,
            "refusal_rate": asr_res["refusal_rate"], "action": info["action"],
            "reward": info.get("reward", ""),
            "ratio_next": round(float(info["ratio_next"]), 4),
            "lr": sched.get_last_lr()[0],
        })
        with open(log_path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(log_rows[0].keys()))
            w.writeheader(); w.writerows(log_rows)

    # ---- Checkpoint final ----
    final = Path(output_dir) / "checkpoint-final"
    model.save_pretrained(str(final)); tokenizer.save_pretrained(str(final))
    ckpts = sorted(Path(output_dir).glob("checkpoint-*"),
                   key=lambda p: int(p.name.split("-")[1]) if p.name.split("-")[1].isdigit() else 10**9)
    list_path.write_text("\n".join(str(c.resolve()) for c in ckpts) + "\n")
    logger.info(f"Entrenamiento dinámico completo. Log en {log_path}")


if __name__ == "__main__":
    main()
