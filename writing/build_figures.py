"""Generates the seven figures used in Chapter 4, as 300dpi PNGs in
writing/figures/, from the same numbers reported in the chapter's tables.
Plain matplotlib, no external style dependencies, print-safe categorical
colors reused from the project's validated palette (blue/orange/aqua).
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker

OUT = Path(__file__).parent / "figures"
OUT.mkdir(exist_ok=True)

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
GRID = "#dddddd"
INK = "#222222"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.edgecolor": INK,
    "axes.labelcolor": INK,
    "text.color": INK,
    "xtick.color": INK,
    "ytick.color": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
    "figure.facecolor": "white",
    "savefig.facecolor": "white",
})

ARCHS = ["ResNet-50", "EfficientNetB4", "VGG-16"]


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def grouped_bar(ax, labels, series, colors, ylabel, ylim=None, fmt="{:.1f}"):
    n_series = len(series)
    x = range(len(labels))
    width = 0.8 / n_series
    for i, (name, values) in enumerate(series):
        offs = [xi + (i - (n_series - 1) / 2) * width for xi in x]
        bars = ax.bar(offs, values, width=width * 0.92, label=name, color=colors[i],
                       edgecolor="white", linewidth=0.5)
        for b, v in zip(bars, values):
            ax.text(b.get_x() + b.get_width() / 2, v + (ylim[1] * 0.012 if ylim else 0.5),
                    fmt.format(v), ha="center", va="bottom", fontsize=8)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels)
    ax.set_ylabel(ylabel)
    if ylim:
        ax.set_ylim(*ylim)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    if n_series > 1:
        ax.legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=n_series)


# Figure 4.1 -- Seven-class accuracy: validation vs ISIC2018
fig, ax = plt.subplots(figsize=(6.3, 3.6))
grouped_bar(
    ax, ARCHS,
    [("HAM10000 validation", [65.8, 63.0, 64.0]), ("ISIC2018 test", [65.7, 61.4, 66.3])],
    [BLUE, ORANGE], "Accuracy (%)", ylim=(0, 80),
)
save(fig, "fig_4_1_seven_class_accuracy.png")

# Figure 4.2 -- Binary task: accuracy and kappa, two panels
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.4))
grouped_bar(axes[0], ARCHS, [("Accuracy", [81.7, 74.3, 81.5])], [BLUE], "Accuracy (%)", ylim=(0, 100))
grouped_bar(axes[1], ARCHS, [("Kappa", [0.502, 0.410, 0.492])], [ORANGE], "Cohen's kappa", ylim=(0, 0.65), fmt="{:.3f}")
save(fig, "fig_4_2_binary_accuracy_kappa.png")

# Figure 4.3 -- DDI kappa: fine-tuned vs joint
fig, ax = plt.subplots(figsize=(6.3, 3.6))
grouped_bar(
    ax, ARCHS,
    [("Fine-tuned", [0.167, 0.083, 0.071]), ("Joint", [0.356, 0.211, 0.304])],
    [ORANGE, BLUE], "Cohen's kappa, DDI held-out split", ylim=(0, 0.45), fmt="{:.3f}",
)
save(fig, "fig_4_3_ddi_kappa_strategy_comparison.png")

# Figure 4.4 -- Skin-tone stratified accuracy, joint model
fig, ax = plt.subplots(figsize=(6.9, 3.8))
grouped_bar(
    ax, ARCHS,
    [
        ("FST I to II", [68.8, 56.3, 71.9]),
        ("FST III to IV", [72.2, 75.0, 63.9]),
        ("FST V to VI", [83.9, 64.5, 77.4]),
    ],
    [BLUE, ORANGE, AQUA], "Accuracy (%)", ylim=(0, 100),
)
save(fig, "fig_4_4_skin_tone_stratified_accuracy.png")

# Figure 4.5 -- Grad-CAM IoU, n=30 vs n=150, two panels (seven-class / binary)
fig, axes = plt.subplots(1, 2, figsize=(6.9, 3.6), sharey=True)
grouped_bar(axes[0], ARCHS, [("n=30", [0.253, 0.239, 0.196]), ("n=150", [0.294, 0.259, 0.196])],
            [ORANGE, BLUE], "Mean IoU", ylim=(0, 0.36), fmt="{:.3f}")
axes[0].set_title("Seven-class task", fontsize=10)
grouped_bar(axes[1], ARCHS, [("n=30", [0.215, 0.167, 0.154]), ("n=150", [0.157, 0.141, 0.115])],
            [ORANGE, BLUE], "", ylim=(0, 0.36), fmt="{:.3f}")
axes[1].set_title("Binary task", fontsize=10)
save(fig, "fig_4_5_gradcam_sample_size_correction.png")

# Figure 4.6 -- Grad-CAM vs SHAP IoU across sample sizes, one panel per architecture
sample_data = {
    "ResNet-50": {"n": [15, 150, 500], "gc": [0.293, 0.294, 0.305], "shap": [0.259, 0.215, 0.208]},
    "EfficientNetB4": {"n": [15, 150, 400], "gc": [0.185, 0.259, 0.255], "shap": [0.273, 0.255, 0.241]},
    "VGG-16": {"n": [15, 150, 500], "gc": [0.200, 0.195, 0.202], "shap": [0.218, 0.255, 0.255]},
}
fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.4), sharey=True)
for ax, (arch, d) in zip(axes, sample_data.items()):
    xpos = range(len(d["n"]))
    ax.plot(xpos, d["gc"], "-o", color=BLUE, label="Grad-CAM", linewidth=2, markersize=5)
    ax.plot(xpos, d["shap"], "-o", color=ORANGE, label="SHAP", linewidth=2, markersize=5)
    ax.set_xticks(list(xpos))
    ax.set_xticklabels([f"n={n}" for n in d["n"]])
    ax.set_title(arch, fontsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
axes[0].set_ylabel("Mean IoU")
axes[0].set_ylim(0.10, 0.34)
axes[1].legend(frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=2)
save(fig, "fig_4_6_gradcam_shap_across_sample_sizes.png")

# Figure 4.7 -- Paired 95% CI diverging chart
rows = [
    ("ResNet-50 (n=500)", 0.0967, 0.0781, 0.1153, BLUE),
    ("VGG-16 (n=500)", -0.0527, -0.0673, -0.0381, ORANGE),
    ("EfficientNetB4 (n=400)", 0.0142, -0.0032, 0.0316, "#999999"),
    ("EfficientNetB4 (n=200)", 0.0067, -0.0181, 0.0315, "#999999"),
]
fig, ax = plt.subplots(figsize=(6.9, 3.4))
ypos = list(range(len(rows)))[::-1]
for y, (label, mean, lo, hi, color) in zip(ypos, rows):
    ax.plot([lo, hi], [y, y], color=color, linewidth=3, solid_capstyle="round")
    ax.plot(mean, y, "o", color=color, markersize=7, markeredgecolor="white", markeredgewidth=1)
ax.axvline(0, color=INK, linewidth=1, linestyle="--", alpha=0.6)
ax.set_yticks(ypos)
ax.set_yticklabels([r[0] for r in rows])
ax.set_xlabel("Grad-CAM minus SHAP mean IoU (95% confidence interval)")
ax.set_xlim(-0.10, 0.14)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)
ax.spines["left"].set_visible(False)
ax.tick_params(left=False)
save(fig, "fig_4_7_paired_confidence_intervals.png")

print("Done. Figures in", OUT)
