"""
Evaluate a (possibly LoRA-adapted) checkpoint on XSTest (Röttger et al. 2024).

XSTest mide el OVER-REFUSAL: contiene 250 prompts SEGUROS que sólo *parecen*
peligrosos (homónimos, lenguaje figurado, contextos seguros…) y 200 prompts
contrastivos genuinamente INSEGUROS. Un buen modelo:
  - cumple los seguros            → refusal_rate_safe BAJO  (no over-refusa)
  - rechaza los inseguros         → refusal_rate_unsafe ALTO

Esto es la contraparte de HarmBench: cuando se inyectan datos de safety, el
riesgo es que el modelo se vuelva sobre-conservador. XSTest lo detecta; ASR
de HarmBench por sí solo no.

El juez de rechazo (keyword-based) y el chat template se reutilizan de
run_harmbench.py para mantener consistencia entre evaluaciones.

Uso:
    python eval/run_xstest.py \\
        --base_model models/Llama-3.2-3B-Instruct \\
        --adapter_path results/exp_e/checkpoint-1500 \\
        --output_path results/exp_e/checkpoint-1500/xstest.json \\
        --step 1500 --epoch 0.41
"""
import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional

import torch

# Reutilizamos el juez de rechazo y el chat template de run_harmbench
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_harmbench import _apply_chat_template, _is_refusal  # noqa: E402

logger = logging.getLogger(__name__)

# CSV oficial de XSTest v2 (mantenido por los autores)
XSTEST_CSV_URL = (
    "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
)
XSTEST_CACHE_NAME = "xstest_prompts.csv"


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--base_model", required=True)
    p.add_argument("--adapter_path", default=None)
    p.add_argument("--output_path", required=True)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--prompts_path", default=None,
                   help="CSV local de XSTest (columnas: type, prompt). "
                        "Si se omite, se descarga del repo oficial.")
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--step", type=int, default=None)
    p.add_argument("--epoch", type=float, default=None)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Carga de prompts
# ---------------------------------------------------------------------------

def _label_from_type(type_str: str) -> str:
    """En XSTest los tipos contrastivos (inseguros) empiezan por 'contrast'."""
    return "unsafe" if (type_str or "").strip().lower().startswith("contrast") else "safe"


def _rows_to_prompts(rows) -> list[dict]:
    out = []
    for i, row in enumerate(rows):
        prompt = (row.get("prompt") or "").strip()
        if not prompt:
            continue
        out.append({
            "id": row.get("id_v2") or row.get("id") or str(i),
            "type": row.get("type", ""),
            "prompt": prompt,
            "label": _label_from_type(row.get("type", "")),
        })
    return out


def load_prompts(local_path: Optional[str], cache_dir: Optional[Path]) -> list[dict]:
    import csv
    import io
    import urllib.request

    if local_path is not None:
        logger.info(f"Loading XSTest prompts from local path: {local_path}")
        with open(local_path, newline="", encoding="utf-8") as f:
            return _rows_to_prompts(csv.DictReader(f))

    if cache_dir is not None:
        cached = cache_dir / XSTEST_CACHE_NAME
        if cached.is_file():
            logger.info(f"Loading XSTest prompts from cache: {cached}")
            with open(cached, newline="", encoding="utf-8") as f:
                return _rows_to_prompts(csv.DictReader(f))

    logger.info(f"Downloading XSTest prompts from {XSTEST_CSV_URL}")
    with urllib.request.urlopen(XSTEST_CSV_URL, timeout=30) as response:
        content = response.read().decode("utf-8")
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        (cache_dir / XSTEST_CACHE_NAME).write_text(content, encoding="utf-8")
    return _rows_to_prompts(csv.DictReader(io.StringIO(content)))


# ---------------------------------------------------------------------------
# Generación
# ---------------------------------------------------------------------------

def run_eval(base_model: str, adapter_path: Optional[str], prompts: list[dict],
             batch_size: int, max_new_tokens: int) -> dict:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    logger.info(f"Loading tokenizer: {base_model}")
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"

    logger.info(f"Loading model: {base_model}")
    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.bfloat16, device_map="auto",
        trust_remote_code=True,
    )
    if adapter_path:
        from peft import PeftModel
        logger.info(f"Loading LoRA adapter: {adapter_path}")
        model = PeftModel.from_pretrained(model, adapter_path)
        model = model.merge_and_unload()
    model.eval()

    samples = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start:start + batch_size]
        rendered = _apply_chat_template(tokenizer, [b["prompt"] for b in batch])
        inputs = tokenizer(rendered, return_tensors="pt", padding=True,
                           truncation=True, max_length=512).to(model.device)
        with torch.inference_mode():
            out_ids = model.generate(
                **inputs, max_new_tokens=max_new_tokens, do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )
        new_tokens = out_ids[:, inputs["input_ids"].shape[1]:]
        completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        for b, completion in zip(batch, completions):
            samples.append({
                "id": b["id"], "type": b["type"], "label": b["label"],
                "prompt": b["prompt"], "completion": completion,
                "is_refusal": _is_refusal(completion),
            })
        logger.info(f"Evaluated {min(start + batch_size, len(prompts))}/{len(prompts)}")

    safe = [s for s in samples if s["label"] == "safe"]
    unsafe = [s for s in samples if s["label"] == "unsafe"]
    n_safe_ref = sum(s["is_refusal"] for s in safe)
    n_unsafe_ref = sum(s["is_refusal"] for s in unsafe)

    def _rate(num, den):
        return round(num / den, 4) if den else None

    aggregate = {
        "n_safe": len(safe),
        "n_safe_refused": n_safe_ref,
        "refusal_rate_safe": _rate(n_safe_ref, len(safe)),       # over-refusal: BAJO mejor
        "n_unsafe": len(unsafe),
        "n_unsafe_refused": n_unsafe_ref,
        "refusal_rate_unsafe": _rate(n_unsafe_ref, len(unsafe)),  # safety: ALTO mejor
        "n_total": len(samples),
    }
    return {"aggregate": aggregate, "samples": samples}


def main() -> None:
    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    args = parse_args()

    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    cache_dir = Path(args.cache_dir) if args.cache_dir else (output_path.parent.parent / "xstest_cache")
    prompts = load_prompts(args.prompts_path, cache_dir=cache_dir)
    n_safe = sum(p["label"] == "safe" for p in prompts)
    logger.info(f"Loaded {len(prompts)} XSTest prompts ({n_safe} safe / {len(prompts) - n_safe} unsafe)")
    if len(prompts) < 100:
        logger.warning("Muy pocos prompts de XSTest — ¿falló la descarga? Resultados no representativos.")

    results = run_eval(args.base_model, args.adapter_path, prompts,
                       args.batch_size, args.max_new_tokens)

    output = {
        "metadata": {
            "base_model": args.base_model,
            "adapter_path": args.adapter_path,
            "step": args.step,
            "epoch": args.epoch,
        },
        "results": results,
    }
    with open(output_path, "w") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {output_path}")

    agg = results["aggregate"]
    logger.info(
        f"XSTest | over-refusal(safe)={agg['refusal_rate_safe']}  "
        f"refusal(unsafe)={agg['refusal_rate_unsafe']}"
    )


if __name__ == "__main__":
    main()
