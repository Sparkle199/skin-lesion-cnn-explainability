# Stage 6 — Training

**Status:** Implemented, not yet run on Runpod.

## Purpose
Fine-tune all three architectures on both tasks — 6 training runs total — using a
consistent two-phase transfer-learning procedure.

## Inputs
- `train_df`/`val_df` (seven-class) or `train_binary`/`val_binary` (binary), from
  Stages 2–4.
- `model`, `base` from Stage 5.
- `class_weight`, and (binary task only) oversampled `train_ds` + `steps_per_epoch`,
  from Stage 4.

## Process
`src/train.py :: train()`:
1. **Phase 1 — frozen-head warmup** (default 5 epochs, `epochs_frozen`): backbone
   frozen, only the new head trains. Optimizer: Adam, learning rate `1e-3`.
2. `src/models/build.py :: unfreeze_top_layers(base, unfreeze_layers)` (default
   `unfreeze_layers=30`) unfreezes the top N backbone layers (BatchNorm layers stay
   frozen — see Stage 5).
3. **Phase 2 — fine-tuning** (default 10 epochs, `epochs_finetune`): re-compiled at a
   much lower learning rate, Adam `1e-5`, to avoid destroying the pretrained weights
   now that more of the backbone is trainable.
4. Both phases use `class_weight` (Stage 4) and, for the binary task, the oversampled
   dataset + `steps_per_epoch` (Stage 4); the seven-class task uses the plain augmented
   dataset with `steps_per_epoch=None` (Keras infers it from dataset size).

## CLI usage
```
python -m src.train --architecture resnet50 --task seven_class
python -m src.train --architecture efficientnetb4 --task binary
```
Run once per (architecture, task) pair — 6 invocations total:

| Architecture | seven_class | binary |
|---|---|---|
| resnet50 | ☐ | ☐ |
| efficientnetb4 | ☐ | ☐ |
| vgg16 | ☐ | ☐ |

(Tick these off in this file as runs complete, so it's obvious at a glance what's left.)

Configurable via CLI flags: `--epochs-frozen`, `--epochs-finetune`, `--unfreeze-layers`,
`--batch-size`, `--ddi-fraction` (binary task only, default 0.3 — see Stage 4).

## Outputs
- `models/{architecture}_{task}.keras` — one saved model per run (e.g.
  `models/resnet50_seven_class.keras`). **Not committed to git** (`.gitignore`
  excludes `*.keras`) — see [`10-streamlit-app.md`](10-streamlit-app.md) for how the
  Streamlit app is expected to access these.

## Code references
- `src/train.py :: prepare_data()`, `train()`, `main()`.
- `src/models/build.py :: build_model()`, `unfreeze_top_layers()` (Stage 5).

## Notes / open items
- Intended to run on the Runpod RTX 4090 pod (Stage 0) — not locally; requires
  TensorFlow and the extracted datasets (Stage 1) to be present.
- No experiment-tracking/logging beyond Keras's default stdout output currently — if
  comparing runs later becomes hard to do from memory, consider writing per-run
  training history (loss/accuracy curves) to `results/` alongside the Stage 7 metrics
  JSON, since Chapter 4 will likely want training curves, not just final metrics.
