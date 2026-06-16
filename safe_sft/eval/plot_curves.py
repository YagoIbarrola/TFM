"""
Genera las gráficas comparativas a partir de los *_curve.csv de results/.

- security_curve.csv → asr_standard vs paso (degradación de safety)
- task_curve.csv     → over-refusal (XSTest), perplexity, GSM8K vs paso

Uso:
    python eval/plot_curves.py --results_dir results --out_dir results/figures
"""
import argparse
import os
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

# Etiquetas legibles por experimento (los que no estén usan su propio id)
LABELS = {
    "baseline": "Baseline (sin SFT)",
    "exp_a": "Alpaca 0%",
    "exp_b": "Alpaca 15% BeaverTails (raw)",
    "exp_d": "Alpaca 15% HH-RLHF",
    "exp_e": "Alpaca 15% (respuesta real)",
    "exp_alpaca_p5": "Alpaca 5% (real)",
    "exp_alpaca_canned_single": "Alpaca 15% canned-single",
    "exp_alpaca_canned_pool": "Alpaca 15% canned-pool",
    "exp_alpaca_canned_single_p5": "Alpaca 5% canned-single",
    "exp_alpaca_canned_pool_p5": "Alpaca 5% canned-pool",
    "exp_c": "Math 0%",
    "exp_math_p5": "Math 5% (real)",
    "exp_math_p15": "Math 15% (real)",
    "exp_math_canned_single": "Math 15% canned-single",
    "exp_math_canned_pool": "Math 15% canned-pool",
    "exp_math_canned_single_p5": "Math 5% canned-single",
    "exp_math_canned_pool_p5": "Math 5% canned-pool",
}

# step sentinela de checkpoint-final (duplica la última fila real)
FINAL_STEP = 999999


def label(exp: str) -> str:
    return LABELS.get(exp, exp)


def load_curve(results_dir: Path, exp: str, fname: str) -> pd.DataFrame | None:
    path = results_dir / exp / fname
    if not path.is_file():
        return None
    df = pd.read_csv(path)
    df = df[df["step"] != FINAL_STEP].copy()      # quita el duplicado 'final'
    df = df.sort_values("step")
    return df


def line_plot(results_dir, exps, fname, col, title, ylabel, out, ylim=None,
              baseline_val=None, baseline_label="baseline"):
    """Una línea por experimento: col vs step."""
    plt.figure(figsize=(9, 5.5))
    plotted = 0
    for exp in exps:
        df = load_curve(results_dir, exp, fname)
        if df is None or col not in df.columns:
            continue
        sub = df.dropna(subset=[col])
        if sub.empty:
            continue
        plt.plot(sub["step"], sub[col], marker="o", ms=3, lw=1.6, label=label(exp))
        plotted += 1
    if plotted == 0:
        plt.close()
        return False
    if baseline_val is not None:
        plt.axhline(baseline_val, ls="--", c="gray", lw=1,
                    label=f"{baseline_label} ({baseline_val:.2f})")
    plt.xlabel("paso de entrenamiento")
    plt.ylabel(ylabel)
    plt.title(title)
    if ylim:
        plt.ylim(*ylim)
    plt.grid(alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(out, dpi=130)
    plt.close()
    print(f"  ✓ {out}")
    return True


def baseline_asr(results_dir: Path) -> float | None:
    """ASR_standard del baseline (step 0 en cualquier security_curve)."""
    for csv in results_dir.glob("*/security_curve.csv"):
        df = pd.read_csv(csv)
        row = df[df["step"] == 0]
        if not row.empty and "asr_standard" in df.columns:
            return float(row.iloc[0]["asr_standard"])
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results_dir", default="results")
    ap.add_argument("--out_dir", default="results/figures")
    args = ap.parse_args()

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Experimentos disponibles (con security_curve)
    available = sorted(p.parent.name for p in results_dir.glob("*/security_curve.csv"))
    print("Experimentos con datos:", ", ".join(available))

    base_asr = baseline_asr(results_dir)

    # --- 1) Headline: ablación canned en Alpaca (ASR std) ---
    line_plot(
        results_dir,
        ["exp_a", "exp_e", "exp_alpaca_canned_pool", "exp_alpaca_canned_single"],
        "security_curve.csv", "asr_standard",
        "Alpaca: diversidad de respuesta vs degradación de safety",
        "ASR standard (HarmBench) ↓", out_dir / "01_canned_alpaca_asr.png",
        ylim=(0, 1), baseline_val=base_asr, baseline_label="baseline",
    )

    # --- 2) Over-refusal (XSTest) en los canned de Alpaca ---
    plt.figure(figsize=(9, 5.5))
    any_xs = False
    for exp in ["exp_alpaca_canned_single", "exp_alpaca_canned_pool",
                "exp_math_canned_single", "exp_math_canned_pool"]:
        df = load_curve(results_dir, exp, "task_curve.csv")
        if df is None or "xstest_refusal_safe" not in df.columns:
            continue
        s = df.dropna(subset=["xstest_refusal_safe"])
        if s.empty:
            continue
        any_xs = True
        line, = plt.plot(s["step"], s["xstest_refusal_safe"], marker="o", ms=3, lw=1.6,
                         label=f"{label(exp)} — safe (over-refusal)")
        if "xstest_refusal_unsafe" in df.columns:
            su = df.dropna(subset=["xstest_refusal_unsafe"])
            plt.plot(su["step"], su["xstest_refusal_unsafe"], ls="--", lw=1.2,
                     c=line.get_color(), label=f"{label(exp)} — unsafe")
    if any_xs:
        plt.xlabel("paso de entrenamiento")
        plt.ylabel("tasa de rechazo (XSTest)")
        plt.title("Over-refusal: rechazo en prompts seguros (sólido) vs inseguros (discontinuo)")
        plt.ylim(0, 1.02)
        plt.grid(alpha=0.3)
        plt.legend(fontsize=8)
        plt.tight_layout()
        plt.savefig(out_dir / "02_canned_xstest.png", dpi=130)
        print(f"  ✓ {out_dir / '02_canned_xstest.png'}")
    plt.close()

    # --- 3) Overview: ASR std de todos los disponibles ---
    line_plot(
        results_dir, available, "security_curve.csv", "asr_standard",
        "Todos los experimentos: ASR standard vs paso",
        "ASR standard (HarmBench) ↓", out_dir / "03_all_asr.png",
        ylim=(0, 1), baseline_val=base_asr,
    )

    # --- 4) Perplexity (task) de los que tengan task_curve ---
    with_task = sorted(p.parent.name for p in results_dir.glob("*/task_curve.csv"))
    line_plot(
        results_dir, with_task, "task_curve.csv", "perplexity",
        "Perplexity en validación vs paso", "perplexity ↓",
        out_dir / "04_perplexity.png",
    )

    # --- 5) GSM8K (solo experimentos math con la métrica) ---
    math_exps = [e for e in with_task if "math" in e]
    line_plot(
        results_dir, math_exps, "task_curve.csv", "gsm8k_acc",
        "GSM8K accuracy vs paso (math)", "accuracy ↑",
        out_dir / "05_gsm8k.png", ylim=(0, 1),
    )

    print(f"\nFiguras en {out_dir}/")


if __name__ == "__main__":
    main()
