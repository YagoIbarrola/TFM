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
    "exp_alpaca_canned5_single": "Alpaca 5% canned-single",
    "exp_alpaca_canned5_pool": "Alpaca 5% canned-pool",
    "exp_dynamic_canned_single": "Dinámico deadband (canned-single)",
    "exp_dynamic_canned_pool": "Dinámico deadband (canned-pool)",
    "exp_dynamic_pid": "Dinámico PID (canned-single)",
    "exp_dynamic_bandit": "Dinámico bandit (canned-single)",
    "exp_dynamic_selfalign": "Dinámico deadband (self-align)",
    "exp_alpaca_selfalign": "Alpaca 15% self-align",
    "exp_alpaca_sa5": "Alpaca 5% self-align",
    "exp_dynamic_sa_pid": "Dinámico PID (self-align)",
    "exp_dynamic_sa_bandit": "Dinámico bandit (self-align)",
    "exp_math_sa5": "Math 5% self-align",
    "exp_math_sa15": "Math 15% self-align",
    "exp_dynamic_sa_math_deadband": "Math dinámico deadband (self-align)",
    "exp_dynamic_sa_math_pid": "Math dinámico PID (self-align)",
    "exp_dynamic_sa_math_bandit": "Math dinámico bandit (self-align)",
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


def _read_csv(path: Path) -> pd.DataFrame | None:
    """Lee un CSV; devuelve None si no existe, está vacío o no parsea."""
    if not path.is_file() or path.stat().st_size == 0:
        return None
    try:
        return pd.read_csv(path)
    except pd.errors.EmptyDataError:
        return None


def load_curve(results_dir: Path, exp: str, fname: str) -> pd.DataFrame | None:
    df = _read_csv(results_dir / exp / fname)
    if df is None:
        return None
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

    # --- 6/7) Régimen dinámico: paneles temporales ---
    dyn_data = {}
    for dyn_csv in sorted(results_dir.glob("*/dynamic_log.csv")):
        exp = dyn_csv.parent.name
        dlog = _read_csv(dyn_csv)
        if dlog is None or dlog.empty:
            continue
        sec = load_curve(results_dir, exp, "security_curve.csv")
        task = load_curve(results_dir, exp, "task_curve.csv")
        dyn_data[exp] = (dlog, sec, task)

        # 06: controlador (ratio vs ASR_bt)
        fig, ax1 = plt.subplots(figsize=(9, 5.5))
        ax1.plot(dlog["step"], dlog["ratio_used"], "o-", color="tab:blue",
                 drawstyle="steps-post", label="ratio safety")
        ax1.set_xlabel("paso de entrenamiento")
        ax1.set_ylabel("ratio de safety", color="tab:blue")
        ax1.tick_params(axis="y", labelcolor="tab:blue"); ax1.set_ylim(-0.02, 0.32)
        ax2 = ax1.twinx()
        ax2.plot(dlog["step"], dlog["asr_bt"], "s--", color="tab:red", label="ASR held-out BT")
        ax2.axhspan(0.15, 0.25, color="gray", alpha=0.15)
        ax2.set_ylabel("ASR_bt", color="tab:red")
        ax2.tick_params(axis="y", labelcolor="tab:red"); ax2.set_ylim(0, 1)
        plt.title(f"Controlador ({label(exp)}): ratio vs ASR_bt (banda muerta sombreada)")
        fig.tight_layout(); plt.savefig(out_dir / f"06_dynamic_{exp}.png", dpi=130); plt.close()
        print(f"  ✓ {out_dir / f'06_dynamic_{exp}.png'}")

        # 07: panel temporal completo (2 filas)
        fig, (axA, axB) = plt.subplots(2, 1, figsize=(10, 8.5), sharex=True)
        axA.plot(dlog["step"], dlog["ratio_used"], drawstyle="steps-post",
                 color="tab:blue", lw=2, label="ratio safety (controlador)")
        axA.set_ylabel("ratio safety", color="tab:blue")
        axA.tick_params(axis="y", labelcolor="tab:blue"); axA.set_ylim(-0.02, 0.32)
        axAr = axA.twinx()
        axAr.plot(dlog["step"], dlog["asr_bt"], "s-", color="tab:red", ms=4,
                  label="ASR held-out BT (sensor)")
        if sec is not None:
            axAr.plot(sec["step"], sec["asr_standard"], "^--", color="darkorange", ms=4,
                      label="ASR HarmBench std")
        axAr.axhspan(0.15, 0.25, color="gray", alpha=0.15)
        axAr.set_ylabel("ASR"); axAr.set_ylim(0, 1)
        ls = axA.get_lines() + axAr.get_lines()
        axA.legend(ls, [l.get_label() for l in ls], fontsize=8, loc="upper right")
        axA.set_title(f"Régimen dinámico ({label(exp)}) — controlador y safety")
        if task is not None:
            axB.plot(task["step"], task["xstest_refusal_safe"], "o-", color="tab:purple",
                     label="over-refusal XSTest (safe)")
            axB.set_ylabel("over-refusal (safe)", color="tab:purple")
            axB.tick_params(axis="y", labelcolor="tab:purple"); axB.set_ylim(0, 1)
            axBr = axB.twinx()
            axBr.plot(task["step"], task["perplexity"], "d--", color="tab:green", label="perplexity")
            axBr.set_ylabel("perplexity", color="tab:green")
            axBr.tick_params(axis="y", labelcolor="tab:green")
            ls = axB.get_lines() + axBr.get_lines()
            axB.legend(ls, [l.get_label() for l in ls], fontsize=8, loc="upper right")
        axB.set_xlabel("paso de entrenamiento")
        fig.tight_layout(); plt.savefig(out_dir / f"07_dynamic_panel_{exp}.png", dpi=130); plt.close()
        print(f"  ✓ {out_dir / f'07_dynamic_panel_{exp}.png'}")

    # 08: comparación entre runs dinámicos (single vs pool)
    if len(dyn_data) >= 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for exp, (dlog, sec, task) in dyn_data.items():
            ax1.plot(dlog["step"], dlog["asr_bt"], "o-", ms=4, label=label(exp))
            ax2.plot(dlog["step"], dlog["ratio_used"], drawstyle="steps-post", lw=2, label=label(exp))
        ax1.axhspan(0.15, 0.25, color="gray", alpha=0.15)
        ax1.set_ylabel("ASR held-out BT"); ax1.set_ylim(0, 1); ax1.legend(fontsize=8)
        ax1.set_title("Régimen dinámico: ASR_bt y ratio por paso")
        ax2.set_ylabel("ratio safety"); ax2.set_ylim(-0.02, 0.32)
        ax2.set_xlabel("paso de entrenamiento"); ax2.legend(fontsize=8)
        fig.tight_layout(); plt.savefig(out_dir / "08_dynamic_compare.png", dpi=130); plt.close()
        print(f"  ✓ {out_dir / '08_dynamic_compare.png'}")

    # --- 9/10/11) Dinámico vs estático (atributos comparables) ---
    single_asr = ["exp_a", "exp_e", "exp_alpaca_canned5_single",
                  "exp_alpaca_canned_single", "exp_dynamic_canned_single"]
    pool_asr = ["exp_a", "exp_e", "exp_alpaca_canned5_pool",
                "exp_alpaca_canned_pool", "exp_dynamic_canned_pool"]
    single_task = ["exp_alpaca_canned5_single", "exp_alpaca_canned_single",
                   "exp_dynamic_canned_single"]
    pool_task = ["exp_alpaca_canned5_pool", "exp_alpaca_canned_pool",
                 "exp_dynamic_canned_pool"]

    line_plot(results_dir, single_asr, "security_curve.csv", "asr_standard",
              "Dinámico vs estático (single): ASR HarmBench",
              "ASR standard ↓", out_dir / "09a_cmp_single_asr.png",
              ylim=(0, 1), baseline_val=base_asr)
    line_plot(results_dir, pool_asr, "security_curve.csv", "asr_standard",
              "Dinámico vs estático (pool): ASR HarmBench",
              "ASR standard ↓", out_dir / "09b_cmp_pool_asr.png",
              ylim=(0, 1), baseline_val=base_asr)

    line_plot(results_dir, single_task, "task_curve.csv", "xstest_refusal_safe",
              "Dinámico vs estático (single): over-refusal XSTest",
              "over-refusal (safe) ↓", out_dir / "10a_cmp_single_overrefusal.png", ylim=(0, 1))
    line_plot(results_dir, pool_task, "task_curve.csv", "xstest_refusal_safe",
              "Dinámico vs estático (pool): over-refusal XSTest",
              "over-refusal (safe) ↓", out_dir / "10b_cmp_pool_overrefusal.png", ylim=(0, 1))

    line_plot(results_dir, single_task + pool_task, "task_curve.csv", "perplexity",
              "Dinámico vs estático: perplexity", "perplexity ↓",
              out_dir / "11_cmp_perplexity.png")

    # --- 12) Trade-off final: ASR vs over-refusal (todos los Alpaca con ambas métricas) ---
    scatter_exps = sorted({p.parent.name for p in results_dir.glob("*/task_curve.csv")
                           if "math" not in p.parent.name})
    pts = []
    for exp in scatter_exps:
        sec = load_curve(results_dir, exp, "security_curve.csv")
        task = load_curve(results_dir, exp, "task_curve.csv")
        if sec is None or task is None or "xstest_refusal_safe" not in task.columns:
            continue
        s = sec.dropna(subset=["asr_standard"])
        t = task.dropna(subset=["xstest_refusal_safe"])
        if s.empty or t.empty:
            continue
        pts.append((exp, t.iloc[-1]["xstest_refusal_safe"], s.iloc[-1]["asr_standard"]))
    if pts:
        plt.figure(figsize=(8.5, 6.5))
        for exp, orr, asr in pts:
            is_dyn = "dynamic" in exp
            plt.scatter(orr, asr, s=140 if is_dyn else 90,
                        marker="*" if is_dyn else "o",
                        zorder=3, edgecolor="black", linewidth=0.6)
            plt.annotate(label(exp), (orr, asr), fontsize=8,
                         xytext=(6, 4), textcoords="offset points")
        if base_asr is not None:
            plt.axhline(base_asr, ls="--", c="gray", lw=1, label=f"ASR baseline ({base_asr:.2f})")
            plt.legend(fontsize=8)
        plt.xlabel("over-refusal en prompts seguros (XSTest) ↓")
        plt.ylabel("ASR standard (HarmBench) ↓")
        plt.title("Trade-off final: seguridad vs over-refusal\n(★ dinámico, ● estático; esquina inf-izq = ideal)")
        plt.xlim(0, 1); plt.ylim(0, 1); plt.grid(alpha=0.3)
        plt.tight_layout(); plt.savefig(out_dir / "12_tradeoff_final.png", dpi=130); plt.close()
        print(f"  ✓ {out_dir / '12_tradeoff_final.png'}")

    # --- 13) Headline: self-align vs canned vs referencias ---
    headline = ["exp_a", "exp_e", "exp_alpaca_canned_single", "exp_dynamic_canned_single",
                "exp_alpaca_selfalign", "exp_dynamic_selfalign"]
    line_plot(results_dir, headline, "security_curve.csv", "asr_standard",
              "Comparativa ASR HarmBench: self-align vs canned vs referencias",
              "ASR standard ↓", out_dir / "13a_headline_asr.png",
              ylim=(0, 1), baseline_val=base_asr)
    line_plot(results_dir, headline, "task_curve.csv", "xstest_refusal_safe",
              "Comparativa over-refusal: self-align vs canned",
              "over-refusal (safe) ↓", out_dir / "13b_headline_overrefusal.png", ylim=(0, 1))

    # --- 14) Self-align dosis-respuesta (Alpaca 0% / 5% / 15%) ---
    dose = ["exp_a", "exp_alpaca_sa5", "exp_alpaca_selfalign"]
    line_plot(results_dir, dose, "security_curve.csv", "asr_standard",
              "Self-align dosis-respuesta (Alpaca): ASR HarmBench",
              "ASR standard ↓", out_dir / "14a_selfalign_dose_asr.png",
              ylim=(0, 1), baseline_val=base_asr)
    line_plot(results_dir, dose, "task_curve.csv", "xstest_refusal_safe",
              "Self-align dosis-respuesta (Alpaca): over-refusal",
              "over-refusal (safe) ↓", out_dir / "14b_selfalign_dose_overrefusal.png", ylim=(0, 1))

    # --- 15) Controladores sobre self-align (deadband / PID / bandit) ---
    sa_ctrl = [e for e in ["exp_dynamic_selfalign", "exp_dynamic_sa_pid", "exp_dynamic_sa_bandit"]
               if _read_csv(results_dir / e / "dynamic_log.csv") is not None]
    if len(sa_ctrl) >= 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
        for e in sa_ctrl:
            dl = _read_csv(results_dir / e / "dynamic_log.csv")
            ax1.plot(dl["step"], dl["asr_bt"], "o-", ms=4, label=label(e))
            ax2.plot(dl["step"], dl["ratio_used"], drawstyle="steps-post", lw=2, label=label(e))
        ax1.axhspan(0.0, 0.0, color="gray", alpha=0)  # placeholder (target auto por run)
        ax1.set_ylabel("ASR held-out BT"); ax1.set_ylim(0, 1); ax1.legend(fontsize=8)
        ax1.set_title("Controladores sobre self-align: ASR_bt y ratio por paso")
        ax2.set_ylabel("ratio safety"); ax2.set_ylim(-0.02, 0.32)
        ax2.set_xlabel("paso de entrenamiento"); ax2.legend(fontsize=8)
        fig.tight_layout(); plt.savefig(out_dir / "15_controllers_selfalign.png", dpi=130); plt.close()
        print(f"  ✓ {out_dir / '15_controllers_selfalign.png'}")

    # --- 16/17) Rejilla por dominio (factorial self-align) ---
    COND_COLORS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    def factorial_grid(domain_name, conds, with_gsm8k, outfile):
        metrics = [
            ("security_curve.csv", "asr_standard", "ASR HarmBench ↓", (0, 1), base_asr),
            ("task_curve.csv", "bt_asr", "ASR BeaverTails held-out ↓", (0, 1), None),
            ("task_curve.csv", "xstest_refusal_safe", "over-refusal XSTest ↓", (0, 1), None),
            ("task_curve.csv", "perplexity", "perplexity ↓", None, None),
        ]
        if with_gsm8k:
            metrics.append(("task_curve.csv", "gsm8k_acc", "GSM8K accuracy ↑", (0, 1), None))
        metrics.append(("task_curve.csv", "ifeval_strict", "IFEval strict ↑", (0, 1), None))
        metrics.append(("task_curve.csv", "arc_acc", "ARC-Challenge ↑", (0, 1), None))

        # quita paneles de métricas sin ningún dato (p. ej. IFEval antes de re-evaluar)
        def _has_data(csv, col):
            for exp, _ in conds:
                df = load_curve(results_dir, exp, csv)
                if df is not None and col in df.columns and not df.dropna(subset=[col]).empty:
                    return True
            return False
        metrics = [m for m in metrics if _has_data(m[0], m[1])]
        if not metrics:
            return
        nrows = (len(metrics) + 1) // 2
        fig, axes = plt.subplots(nrows, 2, figsize=(13, 4.2 * nrows))
        axes = axes.flatten()
        legend_hl = None
        for ax, (csv, col, ylab, ylim, base) in zip(axes, metrics):
            drew = False
            for (exp, lbl), color in zip(conds, COND_COLORS):
                df = load_curve(results_dir, exp, csv)
                if df is None or col not in df.columns:
                    continue
                s = df.dropna(subset=[col])
                if s.empty:
                    continue
                ax.plot(s["step"], s[col], "o-", ms=3, lw=1.5, color=color, label=lbl)
                drew = True
            if base is not None:
                ax.axhline(base, ls="--", c="gray", lw=1, label=f"baseline ({base:.2f})")
            ax.set_title(ylab); ax.set_xlabel("paso"); ax.grid(alpha=0.3)
            if ylim:
                ax.set_ylim(*ylim)
            if drew and legend_hl is None:
                legend_hl = ax.get_legend_handles_labels()
        for ax in axes[len(metrics):]:
            ax.axis("off")
        if legend_hl:
            fig.legend(*legend_hl, loc="lower center", ncol=4, fontsize=9)
        fig.suptitle(f"Factorial self-align — {domain_name}", fontsize=13)
        fig.tight_layout(rect=[0, 0.06, 1, 0.97])
        fig.savefig(out_dir / outfile, dpi=130); plt.close(fig)
        print(f"  ✓ {out_dir / outfile}")

    factorial_grid("Alpaca", [
        ("exp_a", "0%"), ("exp_alpaca_sa5", "5%"), ("exp_alpaca_selfalign", "15%"),
        ("exp_dynamic_selfalign", "din. deadband"), ("exp_dynamic_sa_pid", "din. PID"),
        ("exp_dynamic_sa_bandit", "din. bandit"),
    ], False, "16_alpaca_selfalign_grid.png")

    factorial_grid("MetaMath", [
        ("exp_c", "0%"), ("exp_math_sa5", "5%"), ("exp_math_sa15", "15%"),
        ("exp_dynamic_sa_math_deadband", "din. deadband"), ("exp_dynamic_sa_math_pid", "din. PID"),
        ("exp_dynamic_sa_math_bandit", "din. bandit"),
    ], True, "17_math_selfalign_grid.png")

    # --- 18) CONTROL: estático (s42) vs estático (s43) vs dinámico-fijo 15% ---
    # s42 vs s43 = ruido de semilla; vs dinámico-fijo = gap de implementación.
    factorial_grid("Control — ruido de semilla vs gap de implementación (15%)", [
        ("exp_alpaca_selfalign", "estático 15% (s42)"),
        ("exp_sa15_seed43", "estático 15% (s43)"),
        ("exp_dynamic_sa_fixed15", "dinámico fijo 15%"),
    ], False, "18_control_static_vs_fixeddyn.png")

    print(f"\nFiguras en {out_dir}/")


if __name__ == "__main__":
    main()
