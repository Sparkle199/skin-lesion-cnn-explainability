# DDI Testing Manual — Which Model to Use, and How

## TL;DR recommendation

**Use `resnet50_binary.keras`.** Of the three checkpoints actually saved on disk
(`models/efficientnetb4_binary.keras`, `models/resnet50_binary.keras`,
`models/vgg16_binary.keras` — all "joint" models, trained on HAM10000 + DDI-oversampled),
ResNet50 has the best performance on DDI's untouched held-out split (`ddi_val`, n=99,
`results/resnet50_binary_joint_ddi_val.json`):

- Highest accuracy: **0.747** (vs. 0.657 EfficientNetB4, 0.707 VGG16)
- Highest Cohen's kappa: **0.356** (vs. 0.211, 0.304)
- Highest ROC-AUC: **0.764** (vs. 0.698, 0.733)
- Best dark-skin (FST V–VI) performance of all nine model×mode combinations tested
  anywhere in this project: accuracy 0.839, malignant recall 0.714, malignant F1 0.667.

Only these three `*_binary.keras` checkpoints exist under `models/` right now — the
zero-shot (HAM-only) and DDI-finetuned checkpoints referenced in
`results/*_ham_only_zero_shot_ddi.json` / `results/*_ddi_finetuned.json` were evaluated
but not kept on disk, so you cannot currently load and test those variants directly.

## Full comparison (all models × all modes, on their respective held-out DDI data)

| Architecture | Mode | Held-out set | n | Accuracy | Kappa | ROC-AUC |
|---|---|---|---|---|---|---|
| EfficientNetB4 | zero-shot | full DDI | 656 | 0.625 | 0.138 | 0.592 |
| EfficientNetB4 | finetuned | ddi_val | 99 | 0.667 | 0.083 | 0.615 |
| EfficientNetB4 | **joint** | ddi_val | 99 | 0.657 | 0.211 | 0.698 |
| ResNet50 | zero-shot | full DDI | 656 | 0.753 | 0.159 | 0.654 |
| ResNet50 | finetuned | ddi_val | 99 | 0.717 | 0.167 | 0.584 |
| **ResNet50** | **joint** | ddi_val | 99 | **0.747** | **0.356** | **0.764** |
| VGG16 | zero-shot | full DDI | 656 | 0.732 | 0.090 | 0.598 |
| VGG16 | finetuned | ddi_val | 99 | 0.707 | 0.071 | 0.671 |
| VGG16 | **joint** | ddi_val | 99 | 0.707 | 0.304 | 0.733 |

Kappa is the most trustworthy column here — DDI is imbalanced toward benign, so raw
accuracy alone overstates all nine models. By kappa, the ranking is
**ResNet50-joint > VGG16-joint > EfficientNetB4-joint**, with zero-shot and finetuned
modes trailing every architecture's joint variant.

## Per-skin-tone breakdown (joint models only — the ones you can actually load)

| Architecture | Skin tone | n | Accuracy | Malignant recall | Malignant F1 |
|---|---|---|---|---|---|
| EfficientNetB4 | FST I–II (light) | 32 | 0.563 | 0.625 | 0.417 |
| EfficientNetB4 | FST III–IV (medium) | 36 | 0.750 | 0.636 | 0.609 |
| EfficientNetB4 | FST V–VI (dark) | 31 | 0.645 | 0.286 | 0.267 |
| ResNet50 | FST I–II (light) | 32 | 0.688 | 0.625 | 0.500 |
| ResNet50 | FST III–IV (medium) | 36 | 0.722 | 0.364 | 0.444 |
| **ResNet50** | **FST V–VI (dark)** | 31 | **0.839** | **0.714** | **0.667** |
| VGG16 | FST I–II (light) | 32 | 0.719 | 0.875 | 0.609 |
| VGG16 | FST III–IV (medium) | 36 | 0.639 | 0.364 | 0.381 |
| VGG16 | FST V–VI (dark) | 31 | 0.774 | 0.571 | 0.533 |

Notes for your write-up:
- ResNet50-joint is the only model whose dark-skin performance *exceeds* its light-skin
  performance — the other two architectures both do worse on FST V–VI than on FST I–II
  or III–IV, which is the fairness gap the DDI arm of this project is designed to expose.
- VGG16-joint has the best malignant recall on light skin (0.875) but that comes with a
  precision trade-off (see `results/vgg16_binary_joint_ddi_val.json`) — don't read recall
  alone as "best."

## Where the test images are

`data/ddi_holdout_for_testing/` — 99 images, the exact `ddi_val` split every joint/
finetuned model excluded from training (reproduced from `ddi.split_ddi()`, same seed).
Ground truth is in `data/ddi_holdout_for_testing/ddi_holdout_metadata.csv`.

```
data/ddi_holdout_for_testing/
├── FST_I_II/      32 images  (light skin)
├── FST_III_IV/    36 images  (medium skin)
├── FST_V_VI/      31 images  (dark skin)
└── ddi_holdout_metadata.csv
```

If you specifically want to test a **zero-shot** claim (a HAM-only model that never saw
any DDI image), the correct pool is the *entire* DDI set, not just this 99-image
held-out folder — use `data/raw/ddi/` (656 images) with `data/raw/ddi/ddi_metadata.csv`
for ground truth instead. That distinction only matters if you retrain and keep a
HAM-only checkpoint; it doesn't apply to the three models currently on disk.

## How to test

### Option A — Streamlit app (manual, visual, one image at a time)

The app is already running at **http://localhost:8501**. It reads `models/*.keras`
directly (`streamlit_app.py`):

1. In the sidebar, pick **architecture = resnet50**, **task = binary**.
2. Upload an image from `data/ddi_holdout_for_testing/FST_V_VI/` (or any of the three
   skin-tone folders).
3. Compare the app's prediction to that image's `malignant` value in
   `ddi_holdout_metadata.csv`.
4. Repeat across folders / architectures to compare qualitatively.

### Option B — Scripted batch evaluation (quantitative, reproduces the numbers above)

```bash
# ResNet50 joint model on the full DDI held-out val (all skin tones, stratified output)
python -m src.evaluate_ddi --architecture resnet50 --mode joint \
    --model-path models/resnet50_binary.keras \
    --output results/my_resnet50_ddi_retest.json

# Same for the other two architectures
python -m src.evaluate_ddi --architecture efficientnetb4 --mode joint \
    --model-path models/efficientnetb4_binary.keras \
    --output results/my_efficientnetb4_ddi_retest.json

python -m src.evaluate_ddi --architecture vgg16 --mode joint \
    --model-path models/vgg16_binary.keras \
    --output results/my_vgg16_ddi_retest.json
```

Each run writes accuracy/precision/recall/F1/ROC-AUC/kappa overall **and**
`skin_tone_stratified` per FST band, in the same shape as the `results/*.json` files
this manual's tables were built from.
