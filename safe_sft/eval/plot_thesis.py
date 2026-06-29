"""
Figuras de la memoria (las que tienen datos sin ambigüedad).
Genera P01, P04, P05, P06, P08, P10, P12 en results/figures/.
Uso: python eval/plot_thesis.py
"""
import csv, os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Grupos con 3 semillas (42 base + 43 + 44) para los experimentos con OR-Bench.
SEED_GROUPS = [
    ("static SA5 + OR", ["exp_alpaca_saO5", "exp_alpaca_saO5_s43", "exp_alpaca_saO5_s44"], "C0", "s"),
    ("static SA15 + OR", ["exp_alpaca_saO15", "exp_alpaca_saO15_s43", "exp_alpaca_saO15_s44"], "C1", "D"),
    ("dyn deadband + OR", ["exp_dynamic_hbo_deadband", "exp_dynamic_hbo_deadband_s43", "exp_dynamic_hbo_deadband_s44"], "C2", "o"),
    ("dyn pid + OR", ["exp_dynamic_hbo_pid", "exp_dynamic_hbo_pid_s43", "exp_dynamic_hbo_pid_s44"], "C3", "^"),
]

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
BASE_ASR = 0.085

# Tamaño de fuente grande para los trípticos (legibilidad en la memoria).
FONTS = {"axes.titlesize": 16, "axes.labelsize": 15, "xtick.labelsize": 13,
         "ytick.labelsize": 13, "legend.fontsize": 12}


def read(e, name):
    p = os.path.join(RES, e, name)
    if not os.path.exists(p):
        return []
    with open(p) as f:
        return list(csv.DictReader(f))


def series(e, sec="security_curve.csv"):
    """steps/asr (HarmBench) y steps_or/over-refusal, excluyendo la fila 999999."""
    s = read(e, sec); t = {r["step"]: r for r in read(e, "task_curve.csv")}
    xs, asr, xo, orr = [], [], [], []
    for r in s:
        st = r.get("step")
        if st in (None, "", "999999"):
            continue
        try:
            xs.append(int(st)); asr.append(float(r["asr_standard"]))
        except (ValueError, KeyError):
            continue
        tr = t.get(st, {})
        if tr.get("xstest_refusal_safe") not in (None, ""):
            xo.append(int(st)); orr.append(float(tr["xstest_refusal_safe"]))
    return xs, asr, xo, orr


def final_point(e, sec="security_curve.csv"):
    fa = fo = fp = None
    for r in read(e, sec):
        if r.get("asr_standard") not in (None, ""):
            try: fa = float(r["asr_standard"])
            except ValueError: pass
    for r in read(e, "task_curve.csv"):
        if r.get("xstest_refusal_safe") not in (None, ""):
            try: fo = float(r["xstest_refusal_safe"])
            except ValueError: pass
        if r.get("perplexity") not in (None, ""):
            try: fp = float(r["perplexity"])
            except ValueError: pass
    return fa, fo, fp


def save(fig, name):
    p = os.path.join(FIG, name)
    fig.tight_layout(rect=[0, 0, 1, .95]); fig.savefig(p, dpi=130); plt.close(fig)
    print("escrito:", os.path.basename(p))


def asr_xstest_panels(title, items, out, pareto=True, orr_bars=False,
                      pareto_xlim=None, pareto_ylim=None):
    """items = [(exp, label, kind)] con kind in {'static','dyn'}.
    orr_bars: over-refusal como barras (valor final) en vez de trayectoria.
    pareto_xlim/ylim: límites fijos para no exagerar diferencias mínimas.
    Los puntos del Pareto van en LEYENDA (no anotados) para no solaparse."""
    n = 3 if pareto else 2
    with plt.rc_context(FONTS):
        fig, ax = plt.subplots(1, n, figsize=(5.9 * n, 5.2))
        fig.suptitle(title, fontsize=17, fontweight="bold")
        cmap = plt.get_cmap("tab10")
        blbl, bval, bcol = [], [], []
        for i, (e, lbl, kind) in enumerate(items):
            c = cmap(i % 10); ls = "--" if kind == "static" else "-"
            mk = "s" if kind == "static" else "o"
            xs, asr, xo, orr = series(e); fa, fo, _ = final_point(e)
            if xs:
                ax[0].plot(xs, asr, ls, color=c, marker="o", ms=4, label=lbl, alpha=.85)
            if orr_bars:
                blbl.append(lbl); bval.append(fo if fo is not None else 0); bcol.append(c)
            elif xo:
                ax[1].plot(xo, orr, ls, color=c, marker="o", ms=4, label=lbl, alpha=.85)
            elif fo is not None:
                ax[1].axhline(fo, color=c, ls=":", lw=1.6, alpha=.85, label=f"{lbl} (final)")
            if pareto and fa is not None and fo is not None:
                ax[2].scatter(fa, fo, color=c, s=150, marker=mk, edgecolor="k",
                              zorder=3, label=lbl)
        ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
        ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        if orr_bars:
            bars = ax[1].bar(range(len(blbl)), bval, color=bcol, edgecolor="k", width=.6)
            ax[1].set_xticks(range(len(blbl)))
            ax[1].set_xticklabels(blbl, fontsize=12, rotation=15, ha="right")
            ax[1].set(ylabel="XSTest over-refusal (final)", ylim=(0, 1),
                      title="Over-refusal (↓ mejor)")
            for b, v in zip(bars, bval):
                ax[1].text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}",
                           ha="center", fontsize=12)
            ax[1].grid(alpha=.3, axis="y")
        else:
            ax[1].set(xlabel="step", ylabel="XSTest over-refusal", title="Over-refusal (↓ mejor)")
            ax[1].legend(); ax[1].grid(alpha=.3)
        if pareto:
            ax[2].set(xlabel="HarmBench ASR (↓)", ylabel="over-refusal (↓)",
                      title="Pareto final (inf-izq = ideal)")
            if pareto_xlim: ax[2].set_xlim(*pareto_xlim)
            if pareto_ylim: ax[2].set_ylim(*pareto_ylim)
            ax[2].legend(); ax[2].grid(alpha=.3)
        save(fig, out)


# --- P01: Alpaca sin safety, HarmBench keyword a lo largo del train ---
def p01():
    xs, asr, _, _ = series("exp_a")
    fig, a = plt.subplots(figsize=(7, 4.6))
    a.plot(xs, asr, "-o", color="crimson", label="Alpaca 0% safety")
    a.axhline(BASE_ASR, color="gray", ls=":", label=f"baseline {BASE_ASR}")
    a.set(xlabel="step", ylabel="HarmBench ASR (juez keyword)",
          title="P01 — Degradación de safety: SFT Alpaca SIN safety")
    a.legend(); a.grid(alpha=.3)
    save(fig, "P01_alpaca_nosafety_hb.png")


def _hb_curves(title, items, out):
    """items = [(exp, label, color, linestyle, marker)] -> HarmBench ASR vs step."""
    fig, a = plt.subplots(figsize=(7.5, 4.6))
    for e, lbl, c, ls, mk in items:
        xs, asr, _, _ = series(e)
        if xs:
            a.plot(xs, asr, ls, color=c, marker=mk, ms=3, label=lbl, alpha=.85)
    a.axhline(BASE_ASR, color="gray", ls=":", label=f"baseline {BASE_ASR}")
    a.set(xlabel="step", ylabel="HarmBench ASR (juez keyword)", title=title)
    a.legend(fontsize=8); a.grid(alpha=.3)
    save(fig, out)


# --- P02: Alpaca sin safety vs estático BeaverTails 15% ---
def p02():
    _hb_curves("P02 — Alpaca sin safety vs estático BeaverTails 15%",
               [("exp_a", "Alpaca 0% (sin safety)", "crimson", "-", "o"),
                ("exp_e", "static BeaverTails 15%", "teal", "--", "s")],
               "P02_nosafety_vs_beavertails_hb.png")


# --- P03: sin safety + BeaverTails 15% + HH-RLHF 15% ---
def p03():
    _hb_curves("P03 — Sin safety vs BeaverTails 15% vs HH-RLHF 15%",
               [("exp_a", "Alpaca 0% (sin safety)", "crimson", "-", "o"),
                ("exp_e", "static BeaverTails 15%", "teal", "--", "s"),
                ("exp_d", "static HH-RLHF 15%", "darkorange", "--", "^")],
               "P03_nosafety_beavertails_hhrlhf_hb.png")


# --- P07: perplexity static seed42 / seed43 / dinámico fixed-15 ---
def p07():
    fig, a = plt.subplots(figsize=(7.5, 4.8))
    specs = [("exp_alpaca_selfalign", "static SA15 (seed42)", "crimson", "--"),
             ("exp_sa15_seed43", "static SA15 (seed43)", "firebrick", "--"),
             ("exp_dynamic_sa_fixed15", "dinámico fixed-15", "navy", "-")]
    for e, lbl, c, ls in specs:
        rows = read(e, "task_curve.csv")
        xy = [(int(r["step"]), float(r["perplexity"])) for r in rows
              if r.get("step") not in (None, "", "999999")
              and r.get("perplexity") not in (None, "")]
        if xy:
            xs, ys = zip(*sorted(xy))
            a.plot(xs, ys, ls, color=c, marker="o", ms=3, label=lbl, alpha=.85)
        else:  # solo final
            _, _, fp = final_point(e)
            if fp is not None:
                a.axhline(fp, color=c, ls=":", lw=1.4, alpha=.85, label=f"{lbl} (solo final)")
    a.set(xlabel="step", ylabel="perplexity (val Alpaca)",
          title="P07 — Perplexity: seeds estáticos vs dinámico fixed-15")
    a.legend(fontsize=8); a.grid(alpha=.3)
    save(fig, "P07_perplexity_seeds_vs_dyn.png")


# --- P04: canned single 5% vs 15% (over-refusal en barras; Pareto escala fija) ---
def p04():
    items = [("exp_a", "sin safety"),
             ("exp_alpaca_canned5_single", "single 5%"),
             ("exp_alpaca_canned_single", "single 15%")]
    cmap = plt.get_cmap("tab10")
    fonts = {"font.size": 14, "axes.titlesize": 16, "axes.labelsize": 15,
             "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 13}
    with plt.rc_context(fonts):
        fig, ax = plt.subplots(1, 3, figsize=(18, 5.6))
        fig.suptitle("P04 — Canned single: 0% vs 5% vs 15% (HarmBench / XSTest)",
                     fontsize=18, fontweight="bold")
        labels, orrs, cols = [], [], []
        for i, (e, lbl) in enumerate(items):
            c = cmap(i % 10)
            xs, asr, _, _ = series(e); fa, fo, _ = final_point(e)
            if xs:
                ax[0].plot(xs, asr, "--", color=c, marker="o", ms=4, label=lbl, alpha=.85)
            labels.append(lbl); orrs.append(fo if fo is not None else 0); cols.append(c)
            if fa is not None and fo is not None:
                ax[2].scatter(fa, fo, color=c, s=160, marker="s", edgecolor="k", zorder=3, label=lbl)
        ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
        ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)",
                  ylim=(0, 1.0))
        ax[0].legend(loc="upper right"); ax[0].grid(alpha=.3)
        # over-refusal en BARRAS (solo hay dato final)
        bars = ax[1].bar(range(len(labels)), orrs, color=cols, edgecolor="k", width=.55)
        ax[1].set_xticks(range(len(labels)))
        ax[1].set_xticklabels(labels, fontsize=12, rotation=12, ha="right")
        ax[1].set(ylabel="XSTest over-refusal (final)", ylim=(0, 1),
                  title="Over-refusal (↓ mejor)")
        for b, v in zip(bars, orrs):
            ax[1].text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}",
                       ha="center", fontsize=13)
        ax[1].grid(alpha=.3, axis="y")
        # Pareto a ESCALA FIJA amplia (no exagerar diferencias mínimas)
        ax[2].set(xlabel="HarmBench ASR (↓)", ylabel="over-refusal (↓)",
                  xlim=(0, 0.6), ylim=(0, 1), title="Pareto final (escala fija)")
        ax[2].legend(); ax[2].grid(alpha=.3)
        save(fig, "P04_canned_single_5_15.png")


# --- P05: canned single+pool, static vs dynamic (over-refusal en barras) ---
def p05():
    asr_xstest_panels(
        "P05 — Canned single/pool: estático vs dinámico (HarmBench / XSTest)",
        [("exp_alpaca_canned_single", "static single 15%", "static"),
         ("exp_alpaca_canned_pool", "static pool 15%", "static"),
         ("exp_dynamic_canned_single", "dyn single", "dyn"),
         ("exp_dynamic_canned_pool", "dyn pool", "dyn")],
        "P05_canned_static_vs_dynamic.png", orr_bars=True)


# --- P06: self-align static + dynamic ---
def p06():
    asr_xstest_panels(
        "P06 — Self-align: estáticos vs dinámicos (HarmBench / XSTest)",
        [("exp_alpaca_sa5", "static SA 5%", "static"),
         ("exp_alpaca_selfalign", "static SA 15%", "static"),
         ("exp_dynamic_selfalign", "dyn deadband", "dyn"),
         ("exp_dynamic_sa_pid", "dyn pid", "dyn"),
         ("exp_dynamic_sa_bandit", "dyn bandit", "dyn")],
        "P06_selfalign_static_vs_dynamic.png")


# --- P06a: estáticos self-align (sin OR) por dosis + canned single 5% ---
def p06a():
    asr_xstest_panels(
        "P06a — Estáticos self-align (sin OR): 0% / 5% / 15% vs canned single 5%",
        [("exp_a", "sin safety", "static"),
         ("exp_alpaca_sa5", "self-align 5%", "static"),
         ("exp_alpaca_selfalign", "self-align 15%", "static"),
         ("exp_alpaca_canned5_single", "canned single 5%", "static")],
        "P06a_selfalign_doses_vs_canned.png")


# --- P08: estático vs dinámico SIN reemplazo (barras; Pareto escala fija) ---
def p08():
    asr_xstest_panels(
        "P08 — Estático vs dinámico SIN reemplazo (HarmBench / XSTest)",
        [("exp_alpaca_selfalign", "static SA 15%", "static"),
         ("exp_dynamic_selfalign_wr", "dyn SA (sin reemplazo)", "dyn")],
        "P08_static_vs_dynamic_wr.png", orr_bars=True,
        pareto_xlim=(0, 0.6), pareto_ylim=(0, 1))


# --- P12: cambio a eval+sensor HarmBench, estático vs dinámico ---
def p12():
    asr_xstest_panels(
        "P12 — Sensor+eval HarmBench: estáticos vs dinámicos",
        [("exp_a", "static 0%", "static"),
         ("exp_alpaca_sa5", "static SA 5%", "static"),
         ("exp_alpaca_selfalign", "static SA 15%", "static"),
         ("exp_dynamic_hb_deadband", "dyn deadband", "dyn"),
         ("exp_dynamic_hb_pid", "dyn pid", "dyn"),
         ("exp_dynamic_hb_bandit", "dyn bandit", "dyn")],
        "P12_hb_static_vs_dynamic.png")


# --- P13: estáticos con OR-Bench (anti-over-refusal) vs sin OR ---
def p13():
    # (exp, label, color, ls=OR?) -> color por dosis, linestyle por OR
    specs = [("exp_alpaca_sa5", "SA 5% (sin OR)", "C0", "--"),
             ("exp_alpaca_saO5", "SA 5% + OR-Bench", "C0", "-"),
             ("exp_alpaca_selfalign", "SA 15% (sin OR)", "C3", "--"),
             ("exp_alpaca_saO15", "SA 15% + OR-Bench", "C3", "-")]
    with plt.rc_context(FONTS):
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
        fig.suptitle("P13 — Estáticos self-align: con OR-Bench (anti-over-refusal) vs sin OR",
                     fontsize=17, fontweight="bold")
        pts = {}
        for e, lbl, c, ls in specs:
            xs, asr, xo, orr = series(e); fa, fo, _ = final_point(e)
            if xs: ax[0].plot(xs, asr, ls, color=c, marker="o", ms=4, label=lbl, alpha=.85)
            if xo: ax[1].plot(xo, orr, ls, color=c, marker="o", ms=4, label=lbl, alpha=.85)
            if fa is not None and fo is not None:
                ax[2].scatter(fa, fo, color=c, s=150, marker=("o" if ls == "-" else "s"),
                              edgecolor="k", zorder=3, label=lbl)
                pts[e] = (fa, fo)
        # flechas sin-OR -> con-OR (muestran el desplazamiento)
        for a, b in [("exp_alpaca_sa5", "exp_alpaca_saO5"),
                     ("exp_alpaca_selfalign", "exp_alpaca_saO15")]:
            if a in pts and b in pts:
                ax[2].annotate("", xy=pts[b], xytext=pts[a],
                               arrowprops=dict(arrowstyle="->", color="gray", lw=1.6))
        ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
        ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].set(xlabel="step", ylabel="XSTest over-refusal", title="Over-refusal (↓ mejor)")
        ax[1].legend(); ax[1].grid(alpha=.3)
        ax[2].set(xlabel="HarmBench ASR (↓)", ylabel="over-refusal (↓)",
                  title="Pareto final (flecha = efecto OR-Bench)")
        ax[2].legend(); ax[2].grid(alpha=.3)
        save(fig, "P13_orbench_vs_plain_static.png")


# --- P14: dinámicos con OR-Bench vs sin OR (sensor HarmBench) ---
def p14():
    specs = [("exp_dynamic_hb_deadband", "deadband (sin OR)", "C0", "--"),
             ("exp_dynamic_hbo_deadband", "deadband + OR-Bench", "C0", "-"),
             ("exp_dynamic_hb_pid", "pid (sin OR)", "C3", "--"),
             ("exp_dynamic_hbo_pid", "pid + OR-Bench", "C3", "-")]
    with plt.rc_context(FONTS):
        fig, ax = plt.subplots(1, 3, figsize=(17, 5.4))
        fig.suptitle("P14 — Dinámicos (sensor HarmBench): con OR-Bench vs sin OR",
                     fontsize=17, fontweight="bold")
        pts = {}
        for e, lbl, c, ls in specs:
            xs, asr, xo, orr = series(e); fa, fo, _ = final_point(e)
            if xs: ax[0].plot(xs, asr, ls, color=c, marker="o", ms=4, label=lbl, alpha=.85)
            if xo: ax[1].plot(xo, orr, ls, color=c, marker="o", ms=4, label=lbl, alpha=.85)
            if fa is not None and fo is not None:
                ax[2].scatter(fa, fo, color=c, s=150, marker="o", edgecolor="k",
                              zorder=3, label=lbl); pts[e] = (fa, fo)
        for a, b in [("exp_dynamic_hb_deadband", "exp_dynamic_hbo_deadband"),
                     ("exp_dynamic_hb_pid", "exp_dynamic_hbo_pid")]:
            if a in pts and b in pts:
                ax[2].annotate("", xy=pts[b], xytext=pts[a],
                               arrowprops=dict(arrowstyle="->", color="gray", lw=1.6))
        ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
        ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)")
        ax[0].legend(); ax[0].grid(alpha=.3)
        ax[1].set(xlabel="step", ylabel="XSTest over-refusal", title="Over-refusal (↓ mejor)")
        ax[1].legend(); ax[1].grid(alpha=.3)
        ax[2].set(xlabel="HarmBench ASR (↓)", ylabel="over-refusal (↓)",
                  title="Pareto final (flecha = efecto OR-Bench)")
        ax[2].legend(); ax[2].grid(alpha=.3)
        save(fig, "P14_orbench_vs_plain_dynamic.png")


# --- P15: Pareto maestro (todo: estáticos/dinámicos, con/sin OR) ---
def p15():
    groups = [
        ("exp_a", "Alpaca 0% (sin safety)", "gray", "X"),
        ("exp_alpaca_sa5", "static SA 5%", "C0", "s"),
        ("exp_alpaca_selfalign", "static SA 15%", "C0", "D"),
        ("exp_alpaca_saO5", "static SA 5% + OR", "C2", "s"),
        ("exp_alpaca_saO15", "static SA 15% + OR", "C2", "D"),
        ("exp_dynamic_hb_deadband", "dyn deadband", "C3", "o"),
        ("exp_dynamic_hb_pid", "dyn pid", "C3", "^"),
        ("exp_dynamic_hbo_deadband", "dyn deadband + OR", "C4", "o"),
        ("exp_dynamic_hbo_pid", "dyn pid + OR", "C4", "^"),
    ]
    fig, a = plt.subplots(figsize=(8.5, 6.2))
    for e, lbl, c, mk in groups:
        fa, fo, _ = final_point(e)
        if fa is not None and fo is not None:
            a.scatter(fa, fo, color=c, s=150, marker=mk, edgecolor="k",
                      zorder=3, label=lbl)
    a.axvline(BASE_ASR, color="gray", ls=":", lw=1)
    a.text(BASE_ASR + .005, a.get_ylim()[1] * .02, "baseline ASR 0.085",
           rotation=90, va="bottom", fontsize=7, color="gray")
    a.set(xlabel="HarmBench ASR (↓ mejor)", ylabel="XSTest over-refusal (↓ mejor)",
          title="P15 — Pareto maestro (esquina inf-izq = ideal)")
    a.legend(fontsize=8, loc="upper right"); a.grid(alpha=.3)
    save(fig, "P15_pareto_maestro.png")


# --- P16: estáticos vs dinámicos, ambos con OR-Bench ---
def p16():
    asr_xstest_panels(
        "P16 — Con OR-Bench: estáticos vs dinámicos (HarmBench / XSTest)",
        [("exp_alpaca_saO5", "static SA 5% + OR", "static"),
         ("exp_alpaca_saO15", "static SA 15% + OR", "static"),
         ("exp_dynamic_hbo_deadband", "dyn deadband + OR", "dyn"),
         ("exp_dynamic_hbo_pid", "dyn pid + OR", "dyn")],
        "P16_orbench_static_vs_dynamic.png")


# --- P17: trayectorias en el plano ASR x over-refusal (con OR-Bench) ---
def p17():
    def traj(e):
        s = {r["step"]: r for r in read(e, "security_curve.csv")}
        t = {r["step"]: r for r in read(e, "task_curve.csv")}
        rows = []
        for st in s:
            if st in (None, "", "999999"):
                continue
            a = s[st].get("asr_standard"); o = t.get(st, {}).get("xstest_refusal_safe")
            if a not in (None, "") and o not in (None, ""):
                rows.append((int(st), float(a), float(o)))
        rows.sort()
        return rows

    specs = [("exp_alpaca_saO5", "static SA 5% + OR", "C0", "--"),
             ("exp_alpaca_saO15", "static SA 15% + OR", "C1", "--"),
             ("exp_dynamic_hbo_deadband", "dyn deadband + OR", "C2", "-"),
             ("exp_dynamic_hbo_pid", "dyn pid + OR", "C3", "-")]
    fig, a = plt.subplots(figsize=(8.8, 6.4))
    for e, lbl, c, ls in specs:
        r = traj(e)
        if not r:
            continue
        _, asr, orr = zip(*r)
        a.plot(asr, orr, ls, color=c, marker="o", ms=4, lw=1.3, alpha=.7, label=lbl)
        a.scatter([asr[0]], [orr[0]], color="white", edgecolor=c, s=90,
                  zorder=4, linewidths=1.5)               # inicio (hueco)
        a.scatter([asr[-1]], [orr[-1]], color=c, marker="*", s=320,
                  edgecolor="k", zorder=5)                 # final (estrella)
    a.axvline(BASE_ASR, color="gray", ls=":", lw=1)
    a.text(BASE_ASR + .003, a.get_ylim()[0], "baseline ASR 0.085", rotation=90,
           va="bottom", fontsize=7, color="gray")
    a.set(xlabel="HarmBench ASR (↓ mejor)", ylabel="XSTest over-refusal (↓ mejor)",
          title="P17 — Trayectorias durante el entrenamiento (con OR-Bench)\n"
                "○ inicio · ★ final · esquina inf-izq = ideal")
    a.legend(fontsize=8, loc="best"); a.grid(alpha=.3)
    save(fig, "P17_orbench_trayectorias_pareto.png")


# --- P18: Pareto final con 3 semillas (tenue) + media ± std (grande) ---
def p18():
    fig, a = plt.subplots(figsize=(8.8, 6.3))
    for lbl, exps, c, mk in SEED_GROUPS:
        xs, ys = [], []
        for e in exps:
            fa, fo, _ = final_point(e)
            if fa is not None and fo is not None:
                xs.append(fa); ys.append(fo)
        if not xs:
            continue
        a.scatter(xs, ys, color=c, marker=mk, s=55, alpha=.30, zorder=2)   # semillas
        mx, my, sx, sy = np.mean(xs), np.mean(ys), np.std(xs), np.std(ys)
        a.errorbar(mx, my, xerr=sx, yerr=sy, color=c, marker=mk, ms=14, mec="k",
                   capsize=4, lw=1.6, zorder=4, label=f"{lbl} (n={len(xs)})")
    a.axvline(BASE_ASR, color="gray", ls=":", lw=1)
    a.text(BASE_ASR + .002, a.get_ylim()[0], "baseline ASR 0.085", rotation=90,
           va="bottom", fontsize=7, color="gray")
    a.set(xlabel="HarmBench ASR (↓ mejor)", ylabel="XSTest over-refusal (↓ mejor)",
          title="P18 — Con OR-Bench: 3 semillas (tenue) + media ± std (grande)")
    a.legend(fontsize=8); a.grid(alpha=.3)
    save(fig, "P18_orbench_seeds_mean_pareto.png")


# --- P19: trayectorias media ± std sobre las 3 semillas ---
def p19():
    def agg(exps, kind):
        by = {}
        for e in exps:
            xs, asr, xo, orr = series(e)
            pairs = zip(xs, asr) if kind == "asr" else zip(xo, orr)
            for s, v in pairs:
                by.setdefault(s, []).append(v)
        steps = sorted(s for s, v in by.items() if len(v) >= 2)
        m = np.array([np.mean(by[s]) for s in steps])
        sd = np.array([np.std(by[s]) for s in steps])
        return steps, m, sd

    fig, ax = plt.subplots(1, 2, figsize=(13.5, 4.9))
    fig.suptitle("P19 — Con OR-Bench: media ± std de 3 semillas durante el entrenamiento",
                 fontsize=13, fontweight="bold")
    for lbl, exps, c, _ in SEED_GROUPS:
        for j, kind in enumerate(("asr", "orr")):
            st, m, sd = agg(exps, kind)
            if st:
                ax[j].plot(st, m, color=c, marker="o", ms=3, lw=1.4, label=lbl, alpha=.9)
                ax[j].fill_between(st, m - sd, m + sd, color=c, alpha=.13)
    ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
    ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    ax[1].set(xlabel="step", ylabel="XSTest over-refusal", title="Over-refusal (↓ mejor)")
    ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    save(fig, "P19_orbench_mean_trajectories.png")


# --- P20: frente de Pareto del dinámico barriendo el target ASR (×1, ×2) ---
def p20():
    def ms(exps):
        xs, ys = [], []
        for e in exps:
            fa, fo, _ = final_point(e)
            if fa is not None and fo is not None:
                xs.append(fa); ys.append(fo)
        if not xs:
            return None
        return np.mean(xs), np.mean(ys), np.std(xs), np.std(ys), xs, ys

    fronts = [
        ("deadband", "C2", "o", [
            ("×1 (~0.085)", ["exp_dynamic_hbo_deadband", "exp_dynamic_hbo_deadband_s43", "exp_dynamic_hbo_deadband_s44"]),
            ("×2 (~0.17)", ["exp_dynamic_hbo_deadband_m2", "exp_dynamic_hbo_deadband_m2_s43", "exp_dynamic_hbo_deadband_m2_s44"])]),
    ]
    # estáticos con OR: frente de dosis (5% -> 15%)
    statics = ("static + OR", "C0", "s", [
        ("5%", ["exp_alpaca_saO5", "exp_alpaca_saO5_s43", "exp_alpaca_saO5_s44"]),
        ("15%", ["exp_alpaca_saO15", "exp_alpaca_saO15_s43", "exp_alpaca_saO15_s44"])])

    fig, a = plt.subplots(figsize=(8.8, 6.3))
    for ctrl, c, mk, levels in fronts + [statics]:
        mxs, mys = [], []
        sweep = "dose" if ctrl == "static + OR" else "target"
        for tlabel, exps in levels:
            r = ms(exps)
            if r is None:
                continue
            mx, my, sx, sy, xs, ys = r
            a.scatter(xs, ys, color=c, marker=mk, s=45, alpha=.25, zorder=2)
            a.errorbar(mx, my, xerr=sx, yerr=sy, color=c, marker=mk, ms=13, mec="k",
                       capsize=4, lw=1.5, zorder=4)
            a.annotate(tlabel, (mx, my), fontsize=7, xytext=(6, 6),
                       textcoords="offset points", color=c)
            mxs.append(mx); mys.append(my)
        if len(mxs) >= 2:
            lbl = f"{ctrl} ({sweep} sweep)" if ctrl == "static + OR" else f"dyn {ctrl} (target sweep)"
            a.plot(mxs, mys, color=c, lw=1.8, alpha=.8, label=lbl)
    # Baseline: SFT sin safety (Alpaca 0%) — punto de partida del alignment-tax
    ba, bo, _ = final_point("exp_a")
    if ba is not None and bo is not None:
        a.scatter([ba], [bo], color="k", marker="*", s=240, zorder=6,
                  label="baseline (SFT sin safety)")
        a.annotate("Alpaca 0%", (ba, bo), fontsize=8, xytext=(-8, 8),
                   textcoords="offset points", ha="right", color="k")
    a.axvline(BASE_ASR, color="gray", ls=":", lw=1)
    a.text(BASE_ASR + .002, a.get_ylim()[0], "baseline ASR 0.085", rotation=90,
           va="bottom", fontsize=7, color="gray")
    a.set(xlabel="HarmBench ASR (↓ mejor)", ylabel="XSTest over-refusal (↓ mejor)",
          title="P20 — Pareto con OR-Bench: dinámico vs estático + baseline SFT sin safety (media±std n=3)")
    a.legend(fontsize=9); a.grid(alpha=.3)
    save(fig, "P20_dynamic_pareto_front.png")


if __name__ == "__main__":
    import shutil
    for f in (p01, p02, p03, p04, p05, p06, p06a, p07, p08, p12, p13, p14, p15, p16, p17, p18, p19, p20):
        f()
    # Figuras que ya existían (generadas por plot_curves.py) se alían con nombre Pxx.
    aliases = {
        "11_cmp_perplexity.png": "P06b_perplexity_static_vs_dyn.png",      # P06b
        "15_controllers_selfalign.png": "P10a_controllers_selfalign.png",  # P10
        "19_pid_variants.png": "P10b_pid_variants.png",                    # P10
    }
    for src, dst in aliases.items():
        sp = os.path.join(FIG, src)
        if os.path.exists(sp):
            shutil.copy(sp, os.path.join(FIG, dst)); print(f"alias:  {dst} (= {src})")
        else:
            print(f"AVISO: falta {src} (regenera con plot_curves.py)")
    # limpia figuras antiguas con nombres ya no usados
    for old in ("P10_pid_sensor_bt_vs_hb.png", "P02_static_beavertails_hb.png",
                "P03_hhrlhf_hb.png"):
        op = os.path.join(FIG, old)
        if os.path.exists(op):
            os.remove(op); print(f"eliminada: {old}")
