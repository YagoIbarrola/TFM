"""
ASR sobre el held-out de BeaverTails: genera la respuesta del modelo a prompts
dañinos reservados y juzga si rechaza. ASR = fracción NO rechazada.

Doble uso:
  - `compute_bt_asr(model, tokenizer, prompts, ...)`: función importable, para el
    controlador del régimen dinámico (modelo ya en memoria, sin recargar).
  - CLI: carga un checkpoint LoRA y escribe bt_asr.json (para reportarlo junto a
    HarmBench en las curvas por checkpoint).

El juez de rechazo y el chat template se reutilizan de run_harmbench.py.
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_harmbench import _apply_chat_template, _is_refusal  # noqa: E402

logger = logging.getLogger(__name__)


def compute_bt_asr(model, tokenizer, prompts: list[str], batch_size: int = 32,
                   max_new_tokens: int = 128) -> dict:
    """ASR = 1 - tasa_de_rechazo sobre prompts dañinos. Devuelve dict con agregados.

    Asume `model` ya cargado (con o sin adapter fusionado). Gestiona use_cache y
    el modo eval internamente y restaura el estado previo al salir.
    """
    prev_mode_training = model.training
    prev_use_cache = getattr(model.config, "use_cache", None)
    model.eval()
    model.config.use_cache = True
    prev_padding = tokenizer.padding_side
    tokenizer.padding_side = "left"

    n_refused = 0
    samples = []
    try:
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start:start + batch_size]
            rendered = _apply_chat_template(tokenizer, batch)
            inputs = tokenizer(rendered, return_tensors="pt", padding=True,
                               truncation=True, max_length=512).to(model.device)
            with torch.inference_mode():
                out = model.generate(**inputs, max_new_tokens=max_new_tokens,
                                     do_sample=False, pad_token_id=tokenizer.pad_token_id)
            new_tokens = out[:, inputs["input_ids"].shape[1]:]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for p, c in zip(batch, completions):
                ref = _is_refusal(c)
                n_refused += int(ref)
                samples.append({"prompt": p, "completion": c, "is_refusal": ref})
    finally:
        tokenizer.padding_side = prev_padding
        if prev_use_cache is not None:
            model.config.use_cache = prev_use_cache
        if prev_mode_training:
            model.train()

    n = len(samples)
    asr = (n - n_refused) / n if n else None
    return {
        "asr": round(asr, 4) if asr is not None else None,
        "n_total": n,
        "n_refused": n_refused,
        "refusal_rate": round(n_refused / n, 4) if n else None,
        "samples": samples,
    }


# ---------------------------------------------------------------------------
# CLI (evaluación de un checkpoint para las curvas)
# ---------------------------------------------------------------------------

def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--adapter_path", default=None)
    ap.add_argument("--heldout_dataset", required=True,
                    help="load_from_disk con campo 'prompt' (beavertails_asr_heldout)")
    ap.add_argument("--output_path", required=True)
    ap.add_argument("--batch_size", type=int, default=32)
    ap.add_argument("--max_new_tokens", type=int, default=128)
    ap.add_argument("--step", type=int, default=None)
    ap.add_argument("--epoch", type=float, default=None)
    args = ap.parse_args()

    from datasets import load_from_disk
    from transformers import AutoModelForCausalLM, AutoTokenizer

    prompts = load_from_disk(args.heldout_dataset)["prompt"]
    logger.info(f"Held-out BeaverTails: {len(prompts)} prompts")

    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True)
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path)
        model = model.merge_and_unload()

    res = compute_bt_asr(model, tokenizer, prompts, args.batch_size, args.max_new_tokens)

    out = {
        "metadata": {"base_model": args.base_model, "adapter_path": args.adapter_path,
                     "step": args.step, "epoch": args.epoch},
        "results": {"aggregate": {k: v for k, v in res.items() if k != "samples"},
                    "samples": res["samples"]},
    }
    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_path, "w") as f:
        json.dump(out, f, indent=2, ensure_ascii=False)
    logger.info(f"BT-ASR = {res['asr']} ({res['n_total'] - res['n_refused']}/{res['n_total']} no rechazados)")
    logger.info(f"Guardado en {args.output_path}")


if __name__ == "__main__":
    main()
