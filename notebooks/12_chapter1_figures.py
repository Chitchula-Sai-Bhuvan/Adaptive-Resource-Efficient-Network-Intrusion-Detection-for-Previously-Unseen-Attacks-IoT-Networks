"""Generate the figures used in Chapter 1 (Introduction) of the thesis.

All quantitative panels are read directly from results/tables/*.csv so the
figures stay consistent with the experiments. Output: vector PDF (for LaTeX)
and PNG previews in results/figures/chapter1/.

Run from the project root:  .venv/bin/python notebooks/12_chapter1_figures.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
TABLES = ROOT / "results" / "tables"
OUT = ROOT / "results" / "figures" / "chapter1"
OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------- style
NAVY, TEAL, CORAL, GOLD, GREY, LIGHT = (
    "#1f3b63", "#2a9d8f", "#d1495b", "#e9a23b", "#8d99ae", "#edf2f7")
plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["cmr10", "Computer Modern Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "axes.formatter.use_mathtext": True,
    "axes.unicode_minus": False,
    "font.size": 10,
    "axes.titlesize": 10.5,
    "axes.labelsize": 10,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "legend.frameon": False,
    "savefig.bbox": "tight",
    "savefig.pad_inches": 0.03,
})


def save(fig, name):
    fig.savefig(OUT / f"{name}.pdf")
    fig.savefig(OUT / f"{name}.png", dpi=220)
    plt.close(fig)
    print("wrote", name)


def panel_label(ax, s):
    ax.set_title(s, loc="left", fontweight="bold")


PRETTY = {"backdoor": "Backdoor", "ddos": "DDoS", "dos": "DoS",
          "injection": "Injection", "mitm": "MITM", "normal": "Normal",
          "password": "Password", "ransomware": "Ransomware",
          "scanning": "Scanning", "xss": "XSS"}

# ===================================================== Fig 1.1 closed world
ba = pd.read_csv(TABLES / "phase6_before_after.csv")
pred = pd.read_csv(TABLES / "mitm_prediction_distribution.csv")

fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.6),
                           gridspec_kw={"width_ratios": [1.25, 1]})
ba = ba.sort_values("F1 Before Adaptation", ascending=False)
names = [PRETTY[c] for c in ba["Class"]]
vals = ba["F1 Before Adaptation"].values
cols = [CORAL if c == "mitm" else NAVY for c in ba["Class"]]
a.bar(names, vals, color=cols, width=0.7)
a.set_ylim(0, 1.12)
a.set_ylabel("F1 score")
a.tick_params(axis="x", rotation=40)
for lbl in a.get_xticklabels():
    if lbl.get_text() == "MITM":
        lbl.set_color(CORAL)
        lbl.set_fontweight("bold")
mitm_x = names.index("MITM")
a.annotate("unseen\nclass\nF1 = 0.000", xy=(mitm_x, 0.02),
           xytext=(mitm_x, 0.38), color=CORAL, ha="center", fontsize=9,
           arrowprops=dict(arrowstyle="->", color=CORAL, lw=1.2))
a.axhline(vals[vals > 0].mean(), color=TEAL, lw=1, ls=":")
a.text(len(names) - 0.6, vals[vals > 0].mean() + 0.025,
       f"known-class mean {vals[vals > 0].mean():.3f}", color=TEAL,
       ha="right", fontsize=9)
panel_label(a, "(a) Per-class F1 of the baseline classifier")

pred = pred[pred["count"] > 0].sort_values("count")
total = pred["count"].sum()
pc = [CORAL if p == "normal" else GREY for p in pred["predicted_as"]]
b.barh([PRETTY[p] for p in pred["predicted_as"]], pred["count"], color=pc,
       height=0.65)
for y, (p, c) in enumerate(zip(pred["predicted_as"], pred["count"])):
    b.text(c + 8, y, f"{c}  ({100 * c / total:.1f}%)", va="center",
           fontsize=9, color=CORAL if p == "normal" else "black")
b.set_xlim(0, pred["count"].max() * 1.45)
b.set_xlabel(f"Number of unseen MITM test flows (total {total})")
b.grid(axis="y", visible=False)
panel_label(b, "(b) Labels assigned to unseen MITM flows")
fig.tight_layout(w_pad=2.5)
save(fig, "fig1_1_closed_world_failure")

# ================================================= Fig 1.2 update dilemma
rc = pd.read_csv(TABLES / "phase7_replay_comparison.csv")
short = {"Strategy A - Full Replay": "Full\nretraining",
         "Strategy B - Small Replay Buffer": "Replay\n(100/class)",
         "Strategy C - New Attack Only": "New data\nonly"}
rc["S"] = rc["Strategy"].map(short)
order = ["New data\nonly", "Full\nretraining", "Replay\n(100/class)"]
rc = rc.set_index("S").loc[order]
scol = [GOLD, GREY, TEAL]

fig, axs = plt.subplots(1, 4, figsize=(11, 3.1))
metrics = [
    ("MITM F1", "(a) New-attack F1", "higher is better", (0, 1.15)),
    ("Known Accuracy", "(b) Known-class accuracy", "higher is better", (0, 1.15)),
    ("Old Replay Samples", "(c) Stored historical flows", "lower is better", None),
    ("Training Time", "(d) Update time (s)", "lower is better", None),
]
for ax, (col, title, note, ylim) in zip(axs, metrics):
    v = rc[col].values
    # a zero-height bar cannot be drawn on a log axis, so it is omitted and
    # labelled explicitly instead of being faked as 1
    plot_v = [x if x > 0 else 0.0 for x in v]
    ax.bar(range(3), [x if x > 0 else float("nan") for x in plot_v]
           if col == "Old Replay Samples" else plot_v, color=scol, width=0.68)
    ax.set_xticks(range(3))
    ax.set_xticklabels(order, fontsize=8.5)
    if col == "Old Replay Samples":
        ax.set_yscale("log")
        ax.set_ylim(0.6, 6e5)
        labels = [f"{int(x):,}" if x > 0 else "none" for x in v]
    elif col == "Training Time":
        ax.set_ylim(0, max(v) * 1.3)
        labels = [f"{x:.2f}" for x in v]
    else:
        ax.set_ylim(*ylim)
        labels = [f"{x:.3f}" for x in v]
    for i, (pv, t) in enumerate(zip(plot_v, labels)):
        if col == "Old Replay Samples":
            y = pv * 1.6 if pv > 0 else 1.0
        else:
            y = pv + max(plot_v) * 0.03
        ax.text(i, y, t, ha="center", va="bottom", fontsize=8.5)
    ax.set_title(title, loc="left", fontweight="bold", fontsize=9.5)
    ax.text(0.98, -0.30, note, transform=ax.transAxes, ha="right",
            fontsize=8, color="dimgray", style="italic")
fig.tight_layout(w_pad=1.6)
save(fig, "fig1_2_update_dilemma")

# ============================================ Fig 1.3 fixed buffer vs volume
st = pd.read_csv(TABLES / "phase9_static_analysis.csv")
dy = pd.read_csv(TABLES / "phase9_dynamic_replay_comparison.csv")
dy = dy[dy["Strategy"] == "Dynamic Replay"].set_index("Unseen_Attack")
st = st.sort_values("New_To_Replay_Ratio").set_index("Unseen_Attack")

fig, ax = plt.subplots(figsize=(6.4, 3.7))
x = st["New_To_Replay_Ratio"].values
ax.axvspan(0, 1, color=LIGHT, zorder=0)
ax.text(0.5, 0.44, "buffer\nlarger than\nnew attack", ha="center",
        fontsize=8.5, color="dimgray")
ax.plot(x, st["Maximum_Forgetting"], "-o", color=CORAL, lw=2, ms=7,
        label="Fixed buffer (100 samples/class)")
ax.plot(x, dy.loc[st.index, "Maximum_Forgetting"], "-s", color=TEAL, lw=2,
        ms=7, label="Dynamic Replay (proposed)")
offsets = {"mitm": (8, -14), "ransomware": (-14, 10), "xss": (6, 8),
           "backdoor": (10, 4)}
for k in st.index:
    ax.annotate(f"{PRETTY[k]}\n(M = {st.loc[k, 'Adaptation_Samples']:,})",
                (st.loc[k, "New_To_Replay_Ratio"], st.loc[k, "Maximum_Forgetting"]),
                textcoords="offset points", xytext=offsets[k], fontsize=8,
                ha="left" if offsets[k][0] > 0 else "right")
ax.set_xlim(0, 5.0)
ax.set_ylim(0, 0.48)
ax.set_xlabel("New-attack samples per stored historical sample")
ax.set_ylabel("Maximum forgetting (F1 drop)")
ax.legend(loc="lower right", fontsize=8.5)
fig.tight_layout()
save(fig, "fig1_3_buffer_vs_volume")


# ============================================== Fig 1.4 framework overview
def box(ax, xy, w, h, text, fc, ec=NAVY, tc="black", fs=9, bold=False):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h,
                                boxstyle="round,pad=0.02,rounding_size=0.06",
                                fc=fc, ec=ec, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fs, color=tc, fontweight="bold" if bold else "normal",
            linespacing=1.25)


def arrow(ax, p, q, text=None, color=NAVY, rad=0.0, tpos=0.5, toff=(0, 0.1),
          ls="-"):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=12,
                                 color=color, lw=1.3, ls=ls,
                                 connectionstyle=f"arc3,rad={rad}"))
    if text:
        mx = p[0] + (q[0] - p[0]) * tpos + toff[0]
        my = p[1] + (q[1] - p[1]) * tpos + toff[1]
        ax.text(mx, my, text, ha="center", va="bottom", fontsize=8,
                color=color, style="italic")


fig, ax = plt.subplots(figsize=(11, 4.6))
ax.set_xlim(0, 11)
ax.set_ylim(0, 4.6)
ax.axis("off")

# stage backgrounds
ax.add_patch(FancyBboxPatch((1.75, 2.45), 5.35, 2.0,
             boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f4f8fb",
             ec=GREY, lw=0.8, ls="--"))
ax.text(1.9, 4.27, "Stage 1: Detection at the edge gateway", fontsize=9,
        color=NAVY, fontweight="bold")
ax.add_patch(FancyBboxPatch((1.75, 0.38), 9.1, 1.79,
             boxstyle="round,pad=0.02,rounding_size=0.1", fc="#f3faf8",
             ec=GREY, lw=0.8, ls="--"))
ax.text(1.9, 1.98, "Stage 2: Memory-bounded continual adaptation", fontsize=9,
        color=TEAL, fontweight="bold")

box(ax, (0.1, 3.0), 1.35, 0.95, "IoT network\ntraffic\n(flow features)", LIGHT)
box(ax, (2.0, 3.35), 2.2, 0.75, "Random Forest classifier\n$f_\\theta$ ($K$ known classes)",
    "white")
box(ax, (2.0, 2.6), 2.2, 0.6, "Isolation Forest\nnovelty score $s(\\mathbf{x})$", "white")
box(ax, (5.0, 3.35), 1.9, 0.75, "Known-class\nalert", "white", ec=NAVY)
box(ax, (5.0, 2.6), 1.9, 0.6, "Flagged novel\ntraffic", "#fdecee", ec=CORAL)
box(ax, (8.0, 3.0), 2.75, 1.0, "Analyst confirmation\nand labelling\n($M$ samples, $\\mathcal{D}_{new}$)",
    "#fff6e6", ec=GOLD)

box(ax, (8.0, 0.95), 2.75, 0.85,
    "Dynamic quota allocator\n$k=\\max(k_{min},\\min(k_{max},\\lfloor M/K\\rfloor))$",
    "white", ec=TEAL, fs=8.6)
box(ax, (4.9, 0.95), 2.4, 0.85,
    "Replay buffer $\\mathcal{B}$\n($k$ exemplars per class)\n$\\cup\\;\\mathcal{D}_{new}$",
    "white", ec=TEAL, fs=8.6)
box(ax, (2.0, 0.95), 2.2, 0.85, "Retrain on\n$\\mathcal{D}_{adapt}=\\mathcal{B}\\cup\\mathcal{D}_{new}$",
    "white", ec=TEAL, fs=8.6)
box(ax, (0.1, 0.95), 1.35, 0.85, "Updated model\n$f_{\\theta'}$\n($K{+}1$ classes)",
    TEAL, ec=TEAL, tc="white", bold=True, fs=8.6)

arrow(ax, (1.45, 3.6), (2.0, 3.72))
arrow(ax, (1.45, 3.35), (2.0, 2.9))
arrow(ax, (4.2, 3.72), (5.0, 3.72), "known", toff=(0, 0.05))
arrow(ax, (4.2, 2.9), (5.0, 2.9), "$s(\\mathbf{x})>\\tau$", color=CORAL,
      toff=(0, 0.04))
arrow(ax, (6.9, 2.9), (8.0, 3.3), color=CORAL)
arrow(ax, (9.375, 3.0), (9.375, 1.8), "$M$, $K$", color=GOLD, toff=(0.3, -0.1))
arrow(ax, (8.0, 1.375), (7.3, 1.375), "$k$", color=TEAL, toff=(0, 0.04))
arrow(ax, (4.9, 1.375), (4.2, 1.375), color=TEAL)
arrow(ax, (2.0, 1.375), (1.45, 1.375), color=TEAL)
arrow(ax, (0.775, 1.8), (0.775, 3.0), "redeploy", color=TEAL, ls="--",
      toff=(-0.42, -0.1))
ax.text(6.1, 0.55, "historical training data is never stored in full: "
        "only $K\\cdot k$ exemplars are kept on the device",
        ha="center", fontsize=8.3, color="dimgray", style="italic")
save(fig, "fig1_4_framework_overview")

# ============================================ Fig 1.5 thesis organisation
fig, ax = plt.subplots(figsize=(11, 1.9))
ax.set_xlim(0, 11)
ax.set_ylim(0, 1.9)
ax.axis("off")
chapters = [
    ("Ch. 1", "Introduction", "problem, objectives,\ncontributions"),
    ("Ch. 2", "Literature Review", "ML-based NIDS,\nanomaly detection, CL"),
    ("Ch. 3", "System Model", "threat model,\nformal problem"),
    ("Ch. 4", "Proposed Framework", "two-stage pipeline,\nDynamic Replay"),
    ("Ch. 5", "Results & Discussion", "TON$\\_$IoT experiments,\nanalysis"),
    ("Ch. 6", "Conclusion", "findings, limitations,\nfuture work"),
]
w, gap = 1.55, 0.27
for i, (c, t, d) in enumerate(chapters):
    x = 0.1 + i * (w + gap)
    fc = TEAL if i in (3, 4) else NAVY
    ax.add_patch(FancyBboxPatch((x, 0.95), w, 0.75,
                 boxstyle="round,pad=0.02,rounding_size=0.06", fc=fc, ec=fc))
    ax.text(x + w / 2, 1.43, c, ha="center", color="white", fontsize=8.5)
    ax.text(x + w / 2, 1.15, t, ha="center", color="white", fontsize=9,
            fontweight="bold")
    ax.text(x + w / 2, 0.55, d, ha="center", va="center", fontsize=8,
            color="dimgray", linespacing=1.2)
    if i < len(chapters) - 1:
        arrow(ax, (x + w + 0.02, 1.32), (x + w + gap - 0.02, 1.32))
save(fig, "fig1_5_thesis_organisation")
