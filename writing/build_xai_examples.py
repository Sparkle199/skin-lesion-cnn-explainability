"""Builds three figures from real project outputs:
1. fig_3_x_gradcam_shap_method_example.png -- Chapter 3, section 3.9: what
   Grad-CAM and SHAP actually produce, original + both overlays, one image.
2. fig_4_x_confusion_matrices.png -- Chapter 4, section 4.2.1: row-normalised
   seven-class confusion matrix heatmaps, all three architectures.
3. fig_4_x_architecture_dependent_example.png -- Chapter 4, section 4.5: a
   real example of ResNet-50 favouring Grad-CAM and VGG-16 favouring SHAP,
   using the same per-image IoU values already reported in the text.
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

ROOT = Path(r"C:\Users\SPARKLE\Project Implementation")
HAM_IMG_DIR = ROOT / "data" / "raw" / "dataverse_files" / "HAM10000_images_combined_600x450"
OVERLAY_DIR = ROOT / "results" / "xai_overlays"
OUT_DIR = ROOT / "writing" / "figures"

BLUE = "#2a78d6"
ORANGE = "#eb6834"
INK = "#222222"

plt.rcParams.update({"font.family": "sans-serif", "text.color": INK})


def fig_method_example():
    image_id = "ISIC_0031203"
    original = mpimg.imread(HAM_IMG_DIR / f"{image_id}.jpg")
    gradcam = mpimg.imread(OVERLAY_DIR / "resnet50" / f"{image_id}_gradcam.png")
    shap = mpimg.imread(OVERLAY_DIR / "resnet50" / f"{image_id}_shap.png")

    fig, axes = plt.subplots(1, 3, figsize=(9.5, 3.6))
    for ax, img, title in zip(axes, [original, gradcam, shap], ["Original image", "Grad-CAM", "SHAP"]):
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(title, fontsize=12, fontweight="bold")
    fig.suptitle(f"ResNet-50, seven-class task, {image_id} (true class: melanoma, correctly predicted)",
                 fontsize=9.5, color="#555555", y=0.03)
    fig.tight_layout(rect=[0, 0.05, 1, 1])
    out = OUT_DIR / "fig_3_3_gradcam_shap_method_example.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_confusion_matrices():
    classes = ["akiec", "bcc", "bkl", "df", "mel", "nv", "vasc"]
    matrices = {
        "ResNet-50": np.array([
            [21, 10, 6, 5, 3, 0, 0],
            [6, 39, 6, 12, 8, 1, 3],
            [6, 8, 114, 6, 24, 6, 0],
            [1, 0, 1, 19, 0, 0, 0],
            [9, 4, 23, 3, 104, 14, 2],
            [4, 13, 65, 55, 195, 663, 6],
            [0, 0, 0, 0, 3, 2, 20],
        ]),
        "EfficientNetB4": np.array([
            [20, 6, 9, 3, 7, 0, 0],
            [5, 32, 18, 10, 6, 0, 4],
            [7, 6, 100, 4, 35, 10, 2],
            [2, 0, 4, 12, 2, 1, 0],
            [6, 4, 25, 4, 108, 10, 2],
            [7, 17, 73, 38, 207, 645, 14],
            [0, 0, 1, 0, 1, 2, 21],
        ]),
        "VGG-16": np.array([
            [32, 5, 5, 1, 2, 0, 0],
            [11, 43, 8, 10, 2, 1, 0],
            [9, 4, 108, 5, 33, 5, 0],
            [2, 0, 2, 15, 2, 0, 0],
            [15, 6, 32, 0, 88, 17, 1],
            [13, 24, 90, 34, 189, 647, 4],
            [0, 0, 0, 0, 5, 0, 20],
        ]),
    }

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.3))
    for ax, (name, mat) in zip(axes, matrices.items()):
        row_sums = mat.sum(axis=1, keepdims=True)
        norm = mat / row_sums * 100
        im = ax.imshow(norm, cmap="Blues", vmin=0, vmax=100)
        ax.set_xticks(range(7))
        ax.set_yticks(range(7))
        ax.set_xticklabels(classes, fontsize=8, rotation=45, ha="right")
        ax.set_yticklabels(classes, fontsize=8)
        ax.set_title(name, fontsize=11, fontweight="bold")
        for i in range(7):
            for j in range(7):
                val = norm[i, j]
                if val >= 1:
                    ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                            fontsize=6.5, color="white" if val > 50 else INK)
        if name == "ResNet-50":
            ax.set_ylabel("True class", fontsize=9)
        ax.set_xlabel("Predicted class", fontsize=9)

    fig.subplots_adjust(right=0.9, wspace=0.5)
    cbar_ax = fig.add_axes([0.92, 0.15, 0.015, 0.7])
    fig.colorbar(im, cax=cbar_ax, label="Row-normalised % (recall)")

    out = OUT_DIR / "fig_4_8_confusion_matrices.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def fig_architecture_dependent():
    resnet_id = "ISIC_0031203"
    vgg_id = "ISIC_0032026"
    resnet_gc = mpimg.imread(OVERLAY_DIR / "resnet50" / f"{resnet_id}_gradcam.png")
    resnet_shap = mpimg.imread(OVERLAY_DIR / "resnet50" / f"{resnet_id}_shap.png")
    vgg_gc = mpimg.imread(OVERLAY_DIR / "vgg16" / f"{vgg_id}_gradcam.png")
    vgg_shap = mpimg.imread(OVERLAY_DIR / "vgg16" / f"{vgg_id}_shap.png")

    fig, axes = plt.subplots(2, 2, figsize=(7.2, 7.6))

    panels = [
        (axes[0, 0], resnet_gc, "Grad-CAM", "IoU = 0.512", BLUE),
        (axes[0, 1], resnet_shap, "SHAP", "IoU = 0.291", "#999999"),
        (axes[1, 0], vgg_gc, "Grad-CAM", "IoU = 0.190", "#999999"),
        (axes[1, 1], vgg_shap, "SHAP", "IoU = 0.564", ORANGE),
    ]
    for ax, img, title, iou, color in panels:
        ax.imshow(img)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color(color)
            spine.set_linewidth(3)
        ax.set_title(f"{title}\n{iou}", fontsize=10.5, fontweight="bold", color=color)

    axes[0, 0].set_ylabel(f"ResNet-50\n{resnet_id}\n(correctly predicted mel)", fontsize=9)
    axes[1, 0].set_ylabel(f"VGG-16\n{vgg_id}\n(correctly predicted nv)", fontsize=9)

    fig.suptitle("Same faithfulness method, opposite winner by architecture", fontsize=11, fontweight="bold", y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = OUT_DIR / "fig_4_9_architecture_dependent_example.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


fig_method_example()
fig_confusion_matrices()
fig_architecture_dependent()
