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
