"""Builds the benign-vs-malignant example figure for Chapter 3, section 3.4.2,
from two histopathology-confirmed HAM10000 images."""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

PROJECT_ROOT = Path(r"C:\Users\SPARKLE\Project Implementation")
IMG_DIR = PROJECT_ROOT / "data" / "raw" / "dataverse_files" / "HAM10000_images_combined_600x450"
OUT = PROJECT_ROOT / "writing" / "figures" / "fig_3_2_benign_malignant_example.png"

BENIGN_ID = "ISIC_0025177"   # nv (melanocytic nevus), histopathology-confirmed
MALIGNANT_ID = "ISIC_0025964"  # mel (melanoma), histopathology-confirmed

INK = "#222222"
GOOD = "#0ca30c"
CRITICAL = "#b3241f"

plt.rcParams.update({"font.family": "sans-serif", "text.color": INK})

fig, axes = plt.subplots(1, 2, figsize=(8.4, 4.6))

for ax, (image_id, label, dx, color) in zip(
    axes,
    [
        (BENIGN_ID, "Benign", "melanocytic nevus (nv)", GOOD),
        (MALIGNANT_ID, "Malignant", "melanoma (mel)", CRITICAL),
    ],
):
    img = mpimg.imread(IMG_DIR / f"{image_id}.jpg")
    ax.imshow(img)
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(color)
        spine.set_linewidth(3)
    ax.set_title(f"{label}\n{dx}", fontsize=12, fontweight="bold", color=color, pad=10)
    ax.set_xlabel(f"HAM10000 {image_id}\n(histopathology-confirmed)", fontsize=8.5, color="#555555")

fig.tight_layout()
fig.savefig(OUT, dpi=300, bbox_inches="tight")
print("wrote", OUT)
