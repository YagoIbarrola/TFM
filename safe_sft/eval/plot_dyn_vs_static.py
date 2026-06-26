"""
Figuras: dinámicos (sensor HarmBench) vs estáticos self-align.
Dos figuras, una por juez:
  - fig_dyn_vs_static_kw.png   (HarmBench keyword)  estáticos + dinámicos hb_*
  - fig_dyn_vs_static_llm.png  (HarmBench LLM)      dinámicos hbllm_* (estáticos sin curva LLM)
Cada figura: ASR vs step | over-refusal vs step | Pareto final (over-refusal vs ASR).
Uso: python eval/plot_dyn_vs_static.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
RES = os.path.join(ROOT, "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)


def read(e, name):
    p = os.path.join(RES, e, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def series(e, sec_name):
    """Devuelve (steps, asr, steps_or, over_refusal) excluyendo la fila final 999999."""
    sec = read(e, sec_name)
    task = read(e, "task_curve.csv")
    tmap = {r["step"]: r for r in task}
    xs, asr, xs_o, orr = [], [], [], []
    for r in sec:
        s = r.get("step")
        if s in (None, "", "999999"):
            continue
        try:
            a = float(r["asr_standard"]); st = int(s)
        except (ValueError, KeyError):
            continue
        xs.append(st); asr.append(a)
        t = tmap.get(s, {})
        if t.get("xstest_refusal_safe") not in (None, ""):
            xs_o.append(st); orr.append(float(t["xstest_refusal_safe"]))
    return xs, asr, xs_o, orr


def final_point(e, sec_name):
    """Punto final = última fila disponible (incluida 999999, que usan los estáticos)."""
    sec = read(e, sec_name)
    task = read(e, "task_curve.csv")
    fa = fo = None
    for r in sec:
        if r.get("asr_standard") not in (None, ""):
            try: fa = float(r["asr_standard"])
            except ValueError: pass
    for r in task:
        if r.get("xstest_refusal_safe") not in (None, ""):
            try: fo = float(r["xstest_refusal_safe"])
            except ValueError: pass
    return fa, fo


def make_fig(title, statics, dynamics, sec_dyn, out, llm_note=False):
    fig, ax = plt.subplots(1, 3, figsize=(16, 4.8))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    cmap = plt.get_cmap("tab10")

    # --- panel 0: ASR vs step ; panel 1: over-refusal vs step ---
    items = [(e, "security_curve.csv", "static", lbl) for e, lbl in statics] + \
            [(e, sec_dyn, "dyn", lbl) for e, lbl in dynamics]
    for i, (e, sn, kind, lbl) in enumerate(items):
        xs, asr, xs_o, orr = series(e, sn)
        if not xs:
            continue
        ls = "--" if kind == "static" else "-"
        c = cmap(i % 10)
        ax[0].plot(xs, asr, ls, color=c, marker="o", ms=3, label=lbl, alpha=.85)
        fa, fo = final_point(e, sn)
        if xs_o:
            ax[1].plot(xs_o, orr, ls, color=c, marker="o", ms=3, label=lbl, alpha=.85)
        elif fo is not None:
            # estáticos: solo over-refusal final -> línea horizontal de referencia
            ax[1].axhline(fo, color=c, ls=":", lw=1.4, alpha=.85, label=f"{lbl} (final)")
        if fa is not None and fo is not None:
            ax[2].scatter(fa, fo, color=c, s=90,
                          marker=("s" if kind == "static" else "o"),
                          edgecolor="k", zorder=3)
            ax[2].annotate(lbl, (fa, fo), fontsize=7,
                           xytext=(4, 4), textcoords="offset points")

    ax[0].axhline(0.085, color="gray", ls=":", lw=1, label="baseline ASR 0.085")
    ax[0].set_xlabel("step"); ax[0].set_ylabel("HarmBench ASR"); ax[0].set_title("Safety (↓ mejor)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set_xlabel("step"); ax[1].set_ylabel("XSTest over-refusal (safe)")
    ax[1].set_title("Over-refusal (↓ mejor)"); ax[1].grid(alpha=.3)
    ax[1].legend(fontsize=7)
    ax[2].set_xlabel("HarmBench ASR (↓)"); ax[2].set_ylabel("over-refusal (↓)")
    ax[2].set_title("Pareto final (esquina inf-izq = ideal)"); ax[2].grid(alpha=.3)
    if llm_note:
        ax[2].text(.5, .02, "estáticos sin curva LLM (re-juzgar job 11)",
                   transform=ax[2].transAxes, ha="center", fontsize=8, color="firebrick")

    fig.tight_layout(rect=[0, 0, 1, .95])
    fig.savefig(os.path.join(FIG, out), dpi=130)
    print("escrito:", os.path.join(FIG, out))


STATICS = [("exp_a", "static 0% (alpaca puro)"),
           ("exp_alpaca_sa5", "static SA 5%"),
           ("exp_alpaca_selfalign", "static SA 15%")]

make_fig("Dinámicos vs estáticos — juez KEYWORD",
         STATICS,
         [("exp_dynamic_hb_deadband", "din deadband"),
          ("exp_dynamic_hb_pid", "din pid"),
          ("exp_dynamic_hb_bandit", "din bandit (base)")],
         "security_curve.csv", "fig_dyn_vs_static_kw.png")

make_fig("Dinámicos vs estáticos — juez LLM",
         [],  # estáticos no tienen curva LLM
         [("exp_dynamic_hbllm_deadband", "din deadband (LLM)"),
          ("exp_dynamic_hbllm_pid", "din pid (LLM)")],
         "security_curve_llm.csv", "fig_dyn_vs_static_llm.png", llm_note=True)
