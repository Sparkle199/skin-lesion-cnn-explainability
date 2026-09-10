"""Builds the training/validation accuracy and loss curve figures for
Chapter 4, section 4.2, from the real per-epoch histories in
results/epoch_metrics.json (all six headline models, both tasks)."""

import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(r"C:\Users\SPARKLE\Project Implementation")
METRICS = json.loads((ROOT / "results" / "epoch_metrics.json").read_text())
OUT_DIR = ROOT / "writing" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#222222"
GRID = "#dddddd"

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 9,
    "axes.edgecolor": INK,
    "axes.grid": True,
    "grid.color": GRID,
    "grid.linewidth": 0.6,
    "axes.axisbelow": True,
})

ARCHS = [("resnet50", "ResNet-50"), ("efficientnetb4", "EfficientNetB4"), ("vgg16", "VGG-16")]


def build_figure(task_key, task_label, out_name):
    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.2), sharex=True)

    for col, (arch_key, arch_label) in enumerate(ARCHS):
        rows = METRICS[f"{arch_key}_{task_key}"]
        epochs = [r["epoch"] for r in rows]
        acc = [r["accuracy"] for r in rows]
        val_acc = [r["val_accuracy"] for r in rows]
        loss = [r["loss"] for r in rows]
        val_loss = [r["val_loss"] for r in rows]
        finetune_start = next(r["epoch"] for r in rows if r["phase"] == "finetune")
        transition = finetune_start - 0.5

        ax_acc = axes[0, col]
        ax_acc.plot(epochs, acc, "-o", color=BLUE, label="Train", linewidth=1.8, markersize=3)
        ax_acc.plot(epochs, val_acc, "-o", color=ORANGE, label="Validation", linewidth=1.8, markersize=3)
        ax_acc.axvline(transition, color="#999999", linestyle="--", linewidth=1)
        ax_acc.set_title(arch_label, fontsize=11, fontweight="bold")
        ax_acc.spines["top"].set_visible(False)
        ax_acc.spines["right"].set_visible(False)
        if col == 0:
            ax_acc.set_ylabel("Accuracy")

        ax_loss = axes[1, col]
        ax_loss.plot(epochs, loss, "-o", color=BLUE, linewidth=1.8, markersize=3)
        ax_loss.plot(epochs, val_loss, "-o", color=ORANGE, linewidth=1.8, markersize=3)
        ax_loss.axvline(transition, color="#999999", linestyle="--", linewidth=1)
        ax_loss.spines["top"].set_visible(False)
        ax_loss.spines["right"].set_visible(False)
        ax_loss.set_xlabel("Epoch")
        if col == 0:
            ax_loss.set_ylabel("Loss")

    handles = [
        plt.Line2D([0], [0], color=BLUE, marker="o", markersize=4, label="Train"),
        plt.Line2D([0], [0], color=ORANGE, marker="o", markersize=4, label="Validation"),
        plt.Line2D([0], [0], color="#999999", linestyle="--", label="Frozen to fine-tune transition"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"{task_label}: training and validation curves, all three architectures", fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.04, 1, 0.96])

    out = OUT_DIR / out_name
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


build_figure("seven_class", "Seven-class task", "fig_4_3_training_curves_seven_class.png")
build_figure("binary", "Binary task (joint training)", "fig_4_5_training_curves_binary.png")
