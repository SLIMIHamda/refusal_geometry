"""Generate all publication-quality diagrams (PNG) for the technical reference PDF."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Polygon
from matplotlib.lines import Line2D
from pathlib import Path

FIGDIR = Path(__file__).resolve().parent / "figs"
FIGDIR.mkdir(parents=True, exist_ok=True)

C = dict(navy="#14294B", blue="#2E6E9E", lblue="#5B9BD5", red="#B3322C", green="#2E8B57",
         orange="#D98324", purple="#6C5CE7", teal="#0E7C7B", gray="#5B6770",
         lgray="#EEF1F4", mgray="#C9D2DA", ink="#1C1C1C", white="#FFFFFF", gold="#C9A227")

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9})


def canvas(w_in, h_in, xmax=100.0):
    fig, ax = plt.subplots(figsize=(w_in, h_in))
    ymax = xmax * h_in / w_in
    ax.set_xlim(0, xmax); ax.set_ylim(0, ymax)
    ax.axis("off")
    return fig, ax, ymax


def box(ax, x, y, w, h, text, fc, tc="white", fs=9, bold=True, ec=None, lw=1.2,
        rounding=1.6, align="center", ls="-"):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={rounding}",
                       fc=fc, ec=ec or fc, lw=lw, ls=ls, zorder=3)
    ax.add_patch(p)
    ha = {"center": "center", "left": "left"}[align]
    tx = x + w / 2 if align == "center" else x + 1.6
    ax.text(tx, y + h / 2, text, ha=ha, va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", zorder=4)
    return (x, y, w, h)


def label(ax, x, y, text, fs=8.5, color=C["ink"], bold=False, ha="center", style="normal"):
    ax.text(x, y, text, ha=ha, va="center", color=color, fontsize=fs,
            fontweight="bold" if bold else "normal", fontstyle=style, zorder=5)


def arrow(ax, p1, p2, color=None, lw=1.7, style="-|>", rad=0.0, ls="-"):
    color = color or C["gray"]
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle=style, mutation_scale=12, color=color,
                                 lw=lw, ls=ls, connectionstyle=f"arc3,rad={rad}", zorder=2))


def cx(b): return b[0] + b[2] / 2
def top(b): return (b[0] + b[2] / 2, b[1] + b[3])
def bot(b): return (b[0] + b[2] / 2, b[1])
def left(b): return (b[0], b[1] + b[3] / 2)
def right(b): return (b[0] + b[2], b[1] + b[3] / 2)


def save(fig, name):
    path = FIGDIR / f"{name}.png"
    fig.savefig(path, dpi=200, bbox_inches="tight", pad_inches=0.06, facecolor="white")
    plt.close(fig)
    print("wrote", path.name)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# FIG 1 — Repository architecture (packages, files, dependencies, exec order)
# ─────────────────────────────────────────────────────────────────────────────
def fig_repo_architecture():
    fig, ax, ymax = canvas(10.6, 8.6)
    label(ax, 50, ymax - 2.2, "asw/  —  reproducible activation-steering harness",
          fs=12.5, bold=True, color=C["navy"])

    # Entry layer
    cli = box(ax, 30, ymax - 10, 40, 5, "harness/cli.py\nargparse entrypoint  ·  python -m asw.harness.cli",
              C["navy"], fs=8)
    nb = box(ax, 74, ymax - 10, 24, 5, "notebooks/\nasw_experiments.ipynb", C["gold"], tc=C["ink"], fs=8)
    cfgb = box(ax, 2, ymax - 10, 24, 5, "configs/\nbase + models/*.yaml", C["teal"], fs=8)
    arrow(ax, bot(nb), (cx(cli) + 14, ymax - 5.05), color=C["gold"], rad=-0.15)
    arrow(ax, right(cfgb), left(cli), color=C["teal"])

    # Orchestration layer
    orch = box(ax, 22, ymax - 17, 26, 4.6, "harness/evaluate.py\nevaluate_benchmark()", C["blue"], fs=8.5)
    gen = box(ax, 52, ymax - 17, 26, 4.6, "harness/generate.py\nHFGenerator · run_generation", C["blue"], fs=8.5)
    arrow(ax, (cx(cli) - 6, ymax - 10), top(orch))
    arrow(ax, (cx(cli) + 6, ymax - 10), top(gen))
    arrow(ax, right(orch), left(gen))

    # Scientific modules
    def sci(x, t, col): return box(ax, x, ymax - 26, 21.5, 5.4, t, col, fs=8)
    geo = sci(2, "geometry/\nextract · projection · trace\n(C1–C3)", C["red"])
    wrp = sci(26, "wrapper/\ncondition · steer · wrapper\n(C4)", C["green"])
    base = sci(50, "baselines/\ndefenses · alphasteer\n(Axis D)", C["orange"])
    atk = sci(74, "attacks/\ngcg · pair · multiturn\n(C5)", C["purple"])
    for s in (geo, wrp, base, atk):
        arrow(ax, (cx(s), ymax - 17.0), top(s), color=C["mgray"], style="-")
    arrow(ax, bot(orch), (cx(base), ymax - 20.6), color=C["mgray"], rad=0.1)

    # Engine + data + scoring row
    eng = box(ax, 2, ymax - 34, 30, 5, "models/\nloader.py · hooks.py\n(ActivationCapture · Steerer)", C["navy"], fs=8)
    dat = box(ax, 35, ymax - 34, 28, 5, "data/\nbenchmarks.py · download.py", C["gray"], fs=8.5)
    sco = box(ax, 66, ymax - 34, 32, 5, "scorers/ · eval/\nrefusal · judge · metrics", C["gray"], fs=8.5)
    for s in (geo, wrp, base, atk):
        arrow(ax, bot(s), (cx(eng) if s is geo else cx(s), ymax - 29.0), color=C["mgray"], style="-")
    arrow(ax, bot(geo), top(eng), color=C["red"], rad=0.0)
    arrow(ax, bot(wrp), (cx(eng) + 8, ymax - 29), color=C["green"], rad=0.1)
    arrow(ax, bot(gen), top(eng) if False else (cx(eng), ymax - 29), color=C["blue"], rad=-0.2)
    arrow(ax, bot(gen), (cx(sco), ymax - 29), color=C["blue"], rad=0.2)

    # Reproducibility spine
    spine = box(ax, 14, ymax - 42.5, 72, 5, "Reproducibility spine   ·   config.py  ·  repro.py  ·  db.py (runs.sqlite)  ·  runlog.py",
                C["ink"], fs=9)
    for s in (eng, dat, sco):
        arrow(ax, bot(s), (cx(s), ymax - 37.5), color=C["mgray"], style="-")

    # Report layer
    rep = box(ax, 24, ymax - 50, 54, 4.6, "report/  ·  load → tables → figures → build → REPORT.md", C["teal"], fs=7.8)
    arrow(ax, bot(spine), top(rep), color=C["teal"])

    # exec-order chips
    for i, (xx, yy) in enumerate([(28.0, ymax - 7.0), (20.0, ymax - 14.7), (0.9, ymax - 23.3),
                                  (0.9, ymax - 31.5), (12.0, ymax - 40.2), (21.5, ymax - 47.7)], 1):
        ax.add_patch(plt.Circle((xx, yy), 1.25, color=C["red"], zorder=6))
        ax.text(xx, yy, str(i), ha="center", va="center", color="white", fontsize=8, fontweight="bold", zorder=7)
    label(ax, 99.5, 2.2, "red ○ = execution order", fs=7.5, color=C["red"], ha="right", style="italic")
    return save(fig, "fig1_repo_architecture")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 2 — Scientific pipeline (publication figure)
# ─────────────────────────────────────────────────────────────────────────────
def fig_scientific_pipeline():
    fig, ax, ymax = canvas(11.2, 4.4)
    label(ax, 50, ymax - 1.7, "From the geometry of refusal to a geometry-aware defense",
          fs=12, bold=True, color=C["navy"])

    y = ymax - 11.5
    zoo = box(ax, 1.5, y, 14, 7.2, "Model zoo\naligned  ⇄\nuncensored", C["navy"], fs=8)
    c2 = box(ax, 18.0, y, 15, 7.2, "C2 · Extract\nd_refuse\nbehav. contrast", C["blue"], fs=7.8)
    c1 = box(ax, 36.0, y, 15, 7.2, "C1 · Anti-align\nmap  ⟨ŷ,d̂⟩\nuncens. ⟹ < 0", C["red"], fs=7.8)
    c4 = box(ax, 54.0, y, 15, 7.2, "C4 · Conditional\ngeometry-aware\nwrapper", C["green"], fs=7.8)
    ev = box(ax, 72.0, y, 12.5, 7.2, "Evaluation\nharmful · benign\n+ baselines", C["orange"], tc=C["ink"], fs=7.6)
    c5 = box(ax, 87.0, y, 11.5, 7.2, "C5 · Attacks\nGCG · PAIR\nmulti-turn", C["purple"], fs=7.6)
    for a, b in [(zoo, c2), (c2, c1), (c1, c4), (c4, ev), (ev, c5)]:
        arrow(ax, right(a), left(b), color=C["gray"], lw=2.0)

    c3 = box(ax, 36.0, y - 8.6, 15, 5, "C3 · Causal trace\nnoise-and-restore", C["teal"], fs=7.5)
    arrow(ax, bot(c1), top(c3), color=C["teal"], ls="--")

    rep = box(ax, 54.0, y - 8.2, 44.5, 4.6,
              "dual-scorer labels → CIs → runs.sqlite → REPORT.md", C["ink"], fs=8)
    arrow(ax, bot(ev), (cx(ev), y - 3.6), color=C["mgray"])
    arrow(ax, bot(c4), (cx(c4), y - 3.6), color=C["mgray"])

    label(ax, 50, y - 12.5, "Falsifiable prediction:  raw-addition rescues anti-aligned geometry, while "
          "projection-amplification helps aligned models yet fails on anti-aligned ones.",
          fs=7.7, color=C["red"], style="italic")
    return save(fig, "fig2_scientific_pipeline")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 3 — Layered dependency stack
# ─────────────────────────────────────────────────────────────────────────────
def fig_layers():
    fig, ax, ymax = canvas(8.6, 5.0)
    label(ax, 50, ymax - 1.6, "Layered design — science depends downward on a tested spine",
          fs=11, bold=True, color=C["navy"])
    rows = [
        ("Presentation   ·   CLI subcommands  ·  Kaggle notebook", C["gold"], C["ink"]),
        ("Science   ·   geometry · wrapper · baselines · attacks", C["red"], "white"),
        ("Orchestration   ·   evaluate · generate · report", C["blue"], "white"),
        ("Engine + Data + Scoring   ·   models · data · scorers · eval", C["green"], "white"),
        ("Reproducibility spine   ·   config · repro · db · runlog", C["ink"], "white"),
    ]
    h = 5.5
    for i, (t, fc, tc) in enumerate(rows):
        yy = ymax - 9 - i * (h + 1.4)
        box(ax, 8, yy, 84, h, t, fc, tc=tc, fs=9.2)
        if i:
            arrow(ax, (50, yy + h + 1.4), (50, yy + h), color=C["mgray"], style="-|>")
    label(ax, 50, 1.1, "imports point downward only — no upward dependency (keeps the spine GPU-free & unit-tested)",
          fs=7.6, color=C["gray"], style="italic")
    return save(fig, "fig3_layers")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 4 — Configuration hierarchy & hashing
# ─────────────────────────────────────────────────────────────────────────────
def fig_config():
    fig, ax, ymax = canvas(9.4, 4.2)
    label(ax, 50, ymax - 1.5, "Configuration resolution → deterministic provenance",
          fs=11, bold=True, color=C["navy"])
    y = ymax - 8
    b1 = box(ax, 2, y, 17, 6, "base.yaml\nseeds · decoding\nsplits · paths", C["teal"], fs=8)
    b2 = box(ax, 22, y, 17, 6, "models/<m>.yaml\nid · dtype · layers\n_base_: ../base.yaml", C["blue"], fs=8)
    res = box(ax, 43, y, 17, 6, "deep-merge →\nresolved config\n(dict)", C["green"], fs=8.5)
    hsh = box(ax, 64, y, 15, 6, "canonical JSON\n→ sha256[:12]\nconfig_hash", C["navy"], fs=8.5)
    row = box(ax, 82, y, 16, 6, "runs.sqlite\nrun row\n(config_hash)", C["ink"], fs=8.5)
    arrow(ax, right(b1), left(b2), color=C["gray"])
    arrow(ax, right(b2), left(res), color=C["gray"])
    arrow(ax, right(res), left(hsh), color=C["gray"])
    arrow(ax, right(hsh), left(row), color=C["gray"])
    label(ax, 50, 1.4, 'every paper number = "SELECT … WHERE config_hash = …"  — no orphan results',
          fs=8, color=C["red"], style="italic")
    return save(fig, "fig4_config")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 5 — Run lifecycle (sequence)
# ─────────────────────────────────────────────────────────────────────────────
def fig_runlifecycle():
    fig, ax, ymax = canvas(9.0, 5.6)
    label(ax, 50, ymax - 1.5, "Run lifecycle — run_context() wraps every experiment unit",
          fs=11, bold=True, color=C["navy"])
    actors = [("CLI\nhandler", 12, C["navy"]), ("run_context", 35, C["blue"]),
              ("db.py\nruns.sqlite", 60, C["ink"]), ("Parquet\nresults/", 85, C["gray"])]
    for t, x, col in actors:
        box(ax, x - 8, ymax - 8, 16, 3.4, t, col, fs=8.5)
        ax.add_line(Line2D([x, x], [3, ymax - 8.2], color=C["mgray"], lw=1.1, ls=(0, (4, 3)), zorder=1))
    steps = [
        (ymax - 12, 35, 60, "upsert  status='running' + env + config_hash", C["blue"]),
        (ymax - 17, 12, 35, "yield handle  → do work (load · hook · generate)", C["green"]),
        (ymax - 22, 85, 85, "write_prompt_rows()  per-temperature checkpoint", C["gray"]),
        (ymax - 27, 35, 60, "upsert  status='completed' + metrics_json + wall_clock", C["blue"]),
        (ymax - 32, 35, 60, "on exception → status='failed' + traceback (re-raised)", C["red"]),
    ]
    for yy, x1, x2, t, col in steps:
        if x1 == x2:
            arrow(ax, (x1, yy + 1.2), (x2 + 16, yy + 1.2), color=col, rad=-1.4, lw=1.5)
            label(ax, x2 + 1, yy + 3.0, t, fs=7.6, ha="left", color=C["ink"])
        else:
            arrow(ax, (x1, yy), (x2, yy), color=col, lw=1.6)
            label(ax, (x1 + x2) / 2, yy + 1.4, t, fs=7.6, color=C["ink"])
    label(ax, 50, 1.2, "deterministic run_id = hash(experiment | model | config_hash | seed) ⟹ resume skips finished units",
          fs=7.6, color=C["gray"], style="italic")
    return save(fig, "fig5_runlifecycle")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 6 — Residual-stream intervention (hooks)
# ─────────────────────────────────────────────────────────────────────────────
def fig_hooks():
    fig, ax, ymax = canvas(8.8, 5.8)
    label(ax, 50, ymax - 1.5, "Forward-hook intervention on the residual stream (no weight edits)",
          fs=10.5, bold=True, color=C["navy"])
    n = 6
    x0, bw, bh, gap = 16, 16, 4.2, 1.6
    ys = [ymax - 9 - i * (bh + gap) for i in range(n)]
    names = ["layer 17 …", "layer 16  ◀ steer", "layer 15  ◀ steer",
             "layer 14  ◀ steer", "layer 13  ◀ steer / capture", "layer 12 …"]
    cols = [C["mgray"], C["green"], C["green"], C["green"], C["green"], C["mgray"]]
    boxes = []
    for y, nm, c in zip(ys, names, cols):
        boxes.append(box(ax, x0, y, bw, bh, nm, c, tc="white" if c != C["mgray"] else C["ink"], fs=8))
    for i in range(n - 1):
        arrow(ax, top(boxes[i + 1]), bot(boxes[i]), color=C["gray"], lw=1.4)
    # capture branch
    cap = box(ax, x0 + bw + 10, ys[4], 22, bh, "ActivationCapture\nh[:, -1, :] → cpu fp32", C["blue"], fs=7.6)
    arrow(ax, right(boxes[4]), left(cap), color=C["blue"])
    # steer branch
    st = box(ax, x0 + bw + 10, ys[2], 22, bh + bh + gap, "Steerer / WrapperSteer\nh ← h + α·v  (·mask)", C["red"], fs=7.8)
    arrow(ax, right(boxes[2]), left(st), color=C["red"])
    arrow(ax, right(boxes[3]), (x0 + bw + 10, ys[3] + bh / 2), color=C["red"])
    label(ax, 50, ys[5] - 2.5, "vectors moved to each hidden state's device/dtype inside the hook "
          "(safe under device_map='auto' sharding)", fs=7.4, color=C["gray"], style="italic")
    label(ax, x0 + bw / 2, ys[0] + bh + 2.2, "residual stream ↑", fs=8, color=C["navy"], bold=True)
    return save(fig, "fig6_hooks")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 7 — Behavioural-contrast extraction (C2)
# ─────────────────────────────────────────────────────────────────────────────
def fig_extraction():
    fig, ax, ymax = canvas(9.6, 4.6)
    label(ax, 50, ymax - 1.5, "C2 — behavioural-contrast extraction of d_refuse",
          fs=11, bold=True, color=C["navy"])
    pr = box(ax, 2, ymax - 9.5, 17, 5.5, "AdvBench prompts\n[0:200] (extract split)", C["navy"], fs=8)
    nat = box(ax, 24, ymax - 7.0, 22, 4.4, "native condition\n(generation point)", C["blue"], fs=8)
    ref = box(ax, 24, ymax - 12.4, 22, 4.4, 'refusal condition\n("I cannot help …" prefix)', C["red"], fs=8)
    arrow(ax, right(pr), left(nat), color=C["blue"], rad=0.18)
    arrow(ax, right(pr), left(ref), color=C["red"], rad=-0.18)
    cap = box(ax, 50, ymax - 9.7, 20, 4.6, "terminal-token\nactivations  [N, d]\nper band layer", C["teal"], fs=7.8)
    arrow(ax, right(nat), (50, ymax - 6.2), color=C["blue"])
    arrow(ax, right(ref), (50, ymax - 12.8), color=C["red"])
    dim = box(ax, 74, ymax - 9.7, 24, 4.6, "diff-in-means → normalize\nd = (μ_ref − μ_nat)/‖·‖", C["green"], fs=7.8)
    arrow(ax, right(cap), left(dim), color=C["gray"])
    label(ax, 50, ymax - 16.5, "pairing on the SAME prompts cancels topic/length confounds that a naïve "
          "harmful-vs-harmless DIM leaves in", fs=7.6, color=C["red"], style="italic")
    label(ax, 86, ymax - 13.2, "→ cache/drefuse/<model>.npz", fs=7.6, color=C["gray"], style="italic")
    return save(fig, "fig7_extraction")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 8 — Wrapper decision logic (C4)
# ─────────────────────────────────────────────────────────────────────────────
def fig_wrapper():
    fig, ax, ymax = canvas(8.8, 6.2)
    label(ax, 50, ymax - 1.5, "C4 — geometry-aware conditional steering (per layer, per row)",
          fs=10.5, bold=True, color=C["navy"])
    inp = box(ax, 3, ymax - 9, 17, 4.6, "prompt batch\n(activations)", C["navy"], fs=8)
    # condition diamond
    dy = ymax - 9
    cxp, cyp = 33, dy + 2.3
    dia = Polygon([(cxp, cyp + 4.6), (cxp + 9, cyp), (cxp, cyp - 4.6), (cxp - 9, cyp)],
                  closed=True, fc=C["orange"], ec=C["orange"], zorder=3)
    ax.add_patch(dia)
    label(ax, cxp, cyp, "condition\nharmful?", fs=7.8, color=C["ink"], bold=True)
    arrow(ax, right(inp), (cxp - 9, cyp), color=C["gray"])
    benign = box(ax, 24, ymax - 19, 18, 4.4, "benign row →\npass through", C["mgray"], tc=C["ink"], fs=8)
    arrow(ax, (cxp, cyp - 4.6), top(benign), color=C["gray"]); label(ax, 30, ymax - 13.6, "no", fs=7.5, color=C["gray"])

    # geometry sign branch
    gate = box(ax, 50, dy, 18, 4.6, "geometry sign\n⟨y, d̂⟩ per layer", C["red"], fs=7.8)
    arrow(ax, (cxp + 9, cyp), left(gate), color=C["green"]); label(ax, 45, dy + 5.4, "yes", fs=7.5, color=C["green"])
    raw = box(ax, 74, dy + 3.0, 24, 4.4, "anti-aligned →  raw_add\nh + α·d̂", C["green"], fs=7.6)
    proj = box(ax, 74, dy - 3.4, 24, 4.4, "aligned/neutral →  project\nh + α·⟨h,d̂⟩·d̂", C["blue"], fs=7.6)
    arrow(ax, right(gate), left(raw), color=C["green"], rad=0.12)
    arrow(ax, right(gate), left(proj), color=C["blue"], rad=-0.12)
    out = box(ax, 60, ymax - 20, 26, 4.4, "WrapperSteer hook → steered generation\n(Generator protocol)", C["navy"], fs=7.6)
    arrow(ax, bot(raw), (cx(out) + 6, ymax - 15.6), color=C["mgray"], rad=0.1)
    arrow(ax, bot(proj), (cx(out) - 2, ymax - 15.6), color=C["mgray"])
    arrow(ax, bot(benign), (left(out)[0], left(out)[1]), color=C["mgray"], rad=-0.2)
    label(ax, 50, 1.2, "project-amplify scales the model's OWN refusal component ⟹ strengthens aligned, "
          "fails on anti-aligned (the C1 falsification)", fs=7.3, color=C["red"], style="italic")
    return save(fig, "fig8_wrapper")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 9 — Dual-scorer evaluation (sequence)
# ─────────────────────────────────────────────────────────────────────────────
def fig_eval():
    fig, ax, ymax = canvas(9.6, 4.7)
    label(ax, 50, ymax - 1.5, "Evaluation — generate (resumable) → dual-score → CIs → manifest",
          fs=10.5, bold=True, color=C["navy"])
    y = ymax - 9
    bench = box(ax, 1.5, y, 13, 6, "Benchmark\nExamples", C["navy"], fs=8)
    gen = box(ax, 17, y, 16, 6, "run_generation\nresume by prompt_id\n(T = 0.0, 0.7)", C["blue"], fs=7.8)
    r1 = box(ax, 36, y + 3.4, 19, 3.6, "RubricJudge (#1)\nstring rubric", C["green"], fs=7.8)
    r2 = box(ax, 36, y - 0.6, 19, 3.6, "HFClassifierJudge (#2)\nRoBERTa", C["teal"], fs=7.6)
    agg = box(ax, 58, y, 18, 6, "metrics\nClopper–Pearson CI\n+ κ agreement", C["orange"], tc=C["ink"], fs=7.6)
    out = box(ax, 79, y, 19, 6, "runs.sqlite (row)\n+ <run_id>.parquet", C["ink"], fs=7.8)
    arrow(ax, right(bench), left(gen), color=C["gray"])
    arrow(ax, right(gen), left(r1), color=C["green"], rad=0.12)
    arrow(ax, right(gen), left(r2), color=C["teal"], rad=-0.12)
    arrow(ax, right(r1), (58, y + 4.0), color=C["gray"], rad=0.05)
    arrow(ax, right(r2), (58, y + 2.0), color=C["gray"], rad=-0.05)
    arrow(ax, right(agg), left(out), color=C["gray"])
    label(ax, 50, ymax - 16.0, "labels ∈ {refusal, comply, unclear};  'unclear' excluded from the denominator;  k=0 keeps a meaningful exact upper bound",
          fs=7.4, color=C["gray"], style="italic")
    return save(fig, "fig9_eval")


# ─────────────────────────────────────────────────────────────────────────────
# FIG 10 — Orchestration & artifact flow
# ─────────────────────────────────────────────────────────────────────────────
def fig_orchestration():
    fig, ax, ymax = canvas(10.4, 5.6)
    label(ax, 50, ymax - 1.6, "Experiment orchestration — commands, caches, and generated artifacts",
          fs=11, bold=True, color=C["navy"])
    cmds = [("extract", C["blue"]), ("geometry-map", C["red"]), ("fit-condition", C["green"]),
            ("eval --defense", C["orange"]), ("ablate", C["purple"]), ("report", C["teal"])]
    x = 2
    cb = []
    for t, c in cmds:
        cb.append(box(ax, x, ymax - 8, 15.3, 4.2, "asw " + t, c, tc=C["ink"] if c == C["orange"] else "white", fs=7.6))
        x += 16.3
    for i in range(len(cb) - 1):
        arrow(ax, right(cb[i]), left(cb[i + 1]), color=C["mgray"], lw=1.2)

    caches = box(ax, 6, ymax - 17, 40, 5, "cache/\ndrefuse/*.npz · geometry/*.json · condition/*.npz", C["gray"], fs=7.8)
    db = box(ax, 50, ymax - 17, 22, 5, "results/runs.sqlite\n(run manifest)", C["ink"], fs=8)
    pq = box(ax, 75, ymax - 17, 23, 5, "results/<exp>/<run_id>.parquet\n(per-prompt rows)", C["navy"], fs=7.6)
    arrow(ax, bot(cb[0]), (20, ymax - 12.0), color=C["blue"], rad=0.0)
    arrow(ax, bot(cb[1]), (28, ymax - 12.0), color=C["red"])
    arrow(ax, bot(cb[2]), (38, ymax - 12.0), color=C["green"])
    arrow(ax, bot(cb[3]), top(db), color=C["orange"], rad=-0.05)
    arrow(ax, bot(cb[4]), (cx(db) + 6, ymax - 12.0), color=C["purple"])
    arrow(ax, bot(cb[3]), top(pq), color=C["orange"], rad=0.15)

    rep = box(ax, 28, ymax - 26, 44, 5, "report/  →  REPORT.md  ·  tables/*.csv  ·  figures/*.png", C["teal"], fs=8)
    arrow(ax, bot(db), (cx(rep) - 4, ymax - 21.0), color=C["teal"])
    arrow(ax, bot(caches), (cx(rep) - 12, ymax - 21.0), color=C["mgray"], ls="--")
    arrow(ax, bot(cb[5]), (cx(rep) + 14, ymax - 21.0), color=C["teal"], rad=0.1)
    label(ax, 50, 1.2, "caches make every stage idempotent; runs.sqlite + parquet make every figure regenerable",
          fs=7.6, color=C["gray"], style="italic")
    return save(fig, "fig10_orchestration")


if __name__ == "__main__":
    fig_repo_architecture()
    fig_scientific_pipeline()
    fig_layers()
    fig_config()
    fig_runlifecycle()
    fig_hooks()
    fig_extraction()
    fig_wrapper()
    fig_eval()
    fig_orchestration()
    print("\nall figures written to", FIGDIR)
