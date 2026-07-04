"""Figures from the result tables. matplotlib is imported lazily (Agg backend) so the rest
of the report works even where it is not installed. Each function returns the path written,
or None if there is nothing to plot."""
from __future__ import annotations


def _plt():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    return plt


def fig_anti_alignment_map(geometry_table, path):
    """Heatmap of <y, d_refuse> over models x layers (diverging, centered at 0 — the sign is
    the result)."""
    if geometry_table.empty:
        return None
    import numpy as np

    plt = _plt()
    pivot = geometry_table.pivot(index="model_id", columns="layer", values="projection")
    fig, ax = plt.subplots(figsize=(max(4, 0.6 * pivot.shape[1] + 2),
                                    max(2, 0.5 * pivot.shape[0] + 1)))
    vmax = float(np.nanmax(np.abs(pivot.values))) or 1.0
    im = ax.imshow(pivot.values, cmap="RdBu", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(pivot.shape[1]))
    ax.set_xticklabels(pivot.columns)
    ax.set_yticks(range(pivot.shape[0]))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("layer")
    ax.set_title(r"Anti-alignment map  $\langle \hat{y}, \hat{d}_{refuse}\rangle$")
    fig.colorbar(im, ax=ax, label="projection")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_refusal_bars(refusal_table, benchmark, path):
    """Grouped bars of refusal rate per (model, defense) for one benchmark, with CIs."""
    sub = refusal_table[refusal_table["benchmark"] == benchmark]
    if sub.empty:
        return None
    import numpy as np

    plt = _plt()
    pivot = sub.pivot(index="model_id", columns="defense", values="refusal_rate")
    models, defenses = list(pivot.index), list(pivot.columns)
    x = np.arange(len(models))
    w = 0.8 / max(1, len(defenses))
    fig, ax = plt.subplots(figsize=(max(5, 1.5 * len(models) + 2), 4))
    for i, defn in enumerate(defenses):
        ax.bar(x + i * w, pivot[defn].values, w, label=defn)
    ax.set_xticks(x + w * (len(defenses) - 1) / 2)
    ax.set_xticklabels(models, rotation=20, ha="right")
    ax.set_ylabel("refusal rate")
    ax.set_ylim(0, 1)
    ax.set_title(f"Refusal rate by defense — {benchmark}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_threshold_sweep(sweep, path, *, tau=None):
    """Detector threshold sensitivity (C4, Item 4): TPR (harmful), FPR (benign), FPR (XSTest) vs τ,
    with the chosen τ marked. `sweep` is the list of {tau,tpr,fpr_benign,fpr_over} dicts."""
    if not sweep:
        return None
    plt = _plt()
    taus = [p["tau"] for p in sweep]
    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(taus, [p.get("tpr") for p in sweep], marker="o", label="TPR (harmful)")
    ax.plot(taus, [p.get("fpr_benign") for p in sweep], marker="s", label="FPR (benign)")
    ax.plot(taus, [p.get("fpr_over") for p in sweep], marker="^", label="FPR (XSTest)")
    if tau is not None:
        ax.axvline(tau, color="k", ls="--", lw=1, label="chosen τ")
    ax.set_xlabel("threshold τ")
    ax.set_ylabel("firing rate")
    ax.set_ylim(0, 1)
    ax.set_title("Detector threshold sensitivity (C4)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_asr_vs_budget(attack_table, path):
    """ASR vs query budget, one line per (attack, defense) — the C5 headline curve. The wrapper's
    through-defense curve sitting below the undefended one is the defense claim (Item 6)."""
    if attack_table.empty:
        return None
    budget_cols = [c for c in attack_table.columns if str(c).startswith("asr@")]
    if not budget_cols:
        return None
    budgets = sorted(int(str(c).split("@")[1]) for c in budget_cols)

    plt = _plt()
    fig, ax = plt.subplots(figsize=(5, 4))
    plotted = False
    for _, r in attack_table.iterrows():
        ys = [r.get(f"asr@{b}") for b in budgets]
        if all(y is None or (isinstance(y, float) and y != y) for y in ys):  # all NaN
            continue
        ax.plot(budgets, ys, marker="o", label=f"{r['attack']} / {r['defense']}")
        plotted = True
    if not plotted:
        plt.close(fig)
        return None
    ax.set_xlabel("query budget")
    ax.set_ylabel("attack success rate")
    ax.set_ylim(0, 1)
    ax.set_title("Adversarial robustness — ASR vs query budget (C5)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path


def fig_alpha_tradeoff(alpha_table, path):
    """Refusal rate vs steering strength alpha, one line per benchmark (the safety/utility
    trade-off curve)."""
    if alpha_table.empty:
        return None
    import re

    plt = _plt()

    def alpha_of(point):
        m = re.search(r"alpha=([0-9.]+)", point)
        return float(m.group(1)) if m else float("nan")

    t = alpha_table.copy()
    t["alpha"] = t["point"].apply(alpha_of)
    t = t.sort_values("alpha")
    fig, ax = plt.subplots(figsize=(5, 4))
    for bench, g in t.groupby("benchmark"):
        ax.plot(g["alpha"], g["refusal_rate"], marker="o", label=bench)
    ax.set_xlabel(r"steering strength $\alpha$")
    ax.set_ylabel("refusal rate")
    ax.set_ylim(0, 1)
    ax.set_title("Steering strength trade-off")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return path
