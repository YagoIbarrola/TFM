"""
Figuras de la memoria (las que tienen datos sin ambigüedad).
Genera P01, P04, P05, P06, P08, P10, P12 en results/figures/.
Uso: python eval/plot_thesis.py
"""
import csv, os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RES = os.path.join(os.path.dirname(__file__), "..", "..", "results")
FIG = os.path.join(RES, "figures")
os.makedirs(FIG, exist_ok=True)
BASE_ASR = 0.085


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
    fig, ax = plt.subplots(1, n, figsize=(5.3 * n, 4.7))
    fig.suptitle(title, fontsize=13, fontweight="bold")
    cmap = plt.get_cmap("tab10")
    blbl, bval, bcol = [], [], []
    for i, (e, lbl, kind) in enumerate(items):
        c = cmap(i % 10); ls = "--" if kind == "static" else "-"
        mk = "s" if kind == "static" else "o"
        xs, asr, xo, orr = series(e); fa, fo, _ = final_point(e)
        if xs:
            ax[0].plot(xs, asr, ls, color=c, marker="o", ms=3, label=lbl, alpha=.85)
        if orr_bars:
            blbl.append(lbl); bval.append(fo if fo is not None else 0); bcol.append(c)
        elif xo:
            ax[1].plot(xo, orr, ls, color=c, marker="o", ms=3, label=lbl, alpha=.85)
        elif fo is not None:
            ax[1].axhline(fo, color=c, ls=":", lw=1.4, alpha=.85, label=f"{lbl} (final)")
        if pareto and fa is not None and fo is not None:
            ax[2].scatter(fa, fo, color=c, s=110, marker=mk, edgecolor="k",
                          zorder=3, label=lbl)
    ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
    ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)")
    ax[0].legend(fontsize=7); ax[0].grid(alpha=.3)
    if orr_bars:
        bars = ax[1].bar(range(len(blbl)), bval, color=bcol, edgecolor="k", width=.6)
        ax[1].set_xticks(range(len(blbl)))
        ax[1].set_xticklabels(blbl, fontsize=7, rotation=15, ha="right")
        ax[1].set(ylabel="XSTest over-refusal (final)", ylim=(0, 1),
                  title="Over-refusal (↓ mejor)")
        for b, v in zip(bars, bval):
            ax[1].text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}",
                       ha="center", fontsize=8)
        ax[1].grid(alpha=.3, axis="y")
    else:
        ax[1].set(xlabel="step", ylabel="XSTest over-refusal", title="Over-refusal (↓ mejor)")
        ax[1].legend(fontsize=7); ax[1].grid(alpha=.3)
    if pareto:
        ax[2].set(xlabel="HarmBench ASR (↓)", ylabel="over-refusal (↓)",
                  title="Pareto final (inf-izq = ideal)")
        if pareto_xlim: ax[2].set_xlim(*pareto_xlim)
        if pareto_ylim: ax[2].set_ylim(*pareto_ylim)
        ax[2].legend(fontsize=7); ax[2].grid(alpha=.3)
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
    items = [("exp_alpaca_canned5_single", "canned single 5%"),
             ("exp_alpaca_canned_single", "canned single 15%")]
    cmap = plt.get_cmap("tab10")
    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.6))
    fig.suptitle("P04 — Canned single: 5% vs 15% (HarmBench / XSTest)",
                 fontsize=13, fontweight="bold")
    labels, orrs, cols = [], [], []
    for i, (e, lbl) in enumerate(items):
        c = cmap(i % 10)
        xs, asr, _, _ = series(e); fa, fo, _ = final_point(e)
        if xs:
            ax[0].plot(xs, asr, "--", color=c, marker="o", ms=3, label=lbl, alpha=.85)
        labels.append(lbl); orrs.append(fo if fo is not None else 0); cols.append(c)
        if fa is not None and fo is not None:
            ax[2].scatter(fa, fo, color=c, s=110, marker="s", edgecolor="k", zorder=3, label=lbl)
    ax[0].axhline(BASE_ASR, color="gray", ls=":", lw=1, label=f"baseline {BASE_ASR}")
    ax[0].set(xlabel="step", ylabel="HarmBench ASR", title="Safety (↓ mejor)")
    ax[0].legend(fontsize=8); ax[0].grid(alpha=.3)
    # over-refusal en BARRAS (solo hay dato final)
    bars = ax[1].bar(range(len(labels)), orrs, color=cols, edgecolor="k", width=.55)
    ax[1].set_xticks(range(len(labels))); ax[1].set_xticklabels(labels, fontsize=8)
    ax[1].set(ylabel="XSTest over-refusal (final)", ylim=(0, 1),
              title="Over-refusal (↓ mejor)")
    for b, v in zip(bars, orrs):
        ax[1].text(b.get_x() + b.get_width() / 2, v + .02, f"{v:.3f}", ha="center", fontsize=9)
    ax[1].grid(alpha=.3, axis="y")
    # Pareto a ESCALA FIJA amplia (no exagerar diferencias mínimas)
    ax[2].set(xlabel="HarmBench ASR (↓)", ylabel="over-refusal (↓)",
              xlim=(0, 0.6), ylim=(0, 1), title="Pareto final (escala fija)")
    ax[2].legend(fontsize=8); ax[2].grid(alpha=.3)
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


if __name__ == "__main__":
    import shutil
    for f in (p01, p02, p03, p04, p05, p06, p07, p08, p12):
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
