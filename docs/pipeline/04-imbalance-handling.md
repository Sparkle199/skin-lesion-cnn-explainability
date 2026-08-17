# Stage 4 — Imbalance Handling

**Status:** Implemented.

## Purpose
Address two distinct imbalance problems:
1. **Class imbalance** within HAM10000 (~67% of images are class `nv`; confirmed
   counts: nv 6,705 / mel 1,113 / bkl 1,099 / bcc 514 / akiec 327 / vasc 142 / df 115).
2. **Source imbalance** in the binary task's combined corpus (DDI's 656 images vs.
   HAM10000's 10,015 — DDI is only ~6% of the combined corpus by volume).

## Inputs
- `train_df` / `train_binary` (Stages 2–3), for class-weight computation.
- The combined binary corpus, for source-imbalance correction.

## Process

### Class weighting (both tasks)
`src/data/augmentation.py :: compute_class_weights()` computes per-class weights from
the training label distribution, passed into `model.fit(class_weight=...)` at training
time (Stage 6).

### Augmentation (both tasks)
`src/data/pipeline.py :: make_dataset(..., training=True)` applies rotation, flipping,
zoom, and brightness augmentation to training batches. Validation datasets
(`training=False`) receive no augmentation — this matters because the faithfulness
check (Stage 7) and skin-tone-stratified evaluation both run on the un-augmented
validation distribution.

### Source-imbalance correction (binary task only)
Class weighting alone does not fix DDI being diluted to near-invisibility in a plain
shuffle of the combined corpus. Instead:
- `src/data/pipeline.py :: make_oversampled_binary_dataset()` draws each training batch
  from HAM10000 and DDI at a fixed **`ddi_fraction`** (default 0.3) — i.e. roughly 30%
  of every batch is DDI-sourced, well above DDI's natural ~6% share, so the model
  actually sees enough DDI examples per epoch for the added skin-tone diversity to have
  an effect.
- `src/data/oversampling.py :: compute_steps_per_epoch()` computes how many steps make
  up one epoch given this oversampling ratio.
- **Validation always uses the true, un-oversampled distribution**
  (`make_dataset()`, not `make_oversampled_binary_dataset()`) — oversampling is a
  training-only correction; evaluating on an artificially rebalanced validation set
  would misrepresent real-world performance.

## Outputs
- `class_weight` dict (both tasks), passed to `model.fit()`.
- `train_ds` (binary task: oversampled at `ddi_fraction`; seven-class task: plain
  augmented shuffle) and `steps_per_epoch` (binary task only).

## Code references
- `src/data/augmentation.py :: compute_class_weights()`.
- `src/data/oversampling.py :: compute_steps_per_epoch()`.
- `src/data/pipeline.py :: make_dataset()`, `make_oversampled_binary_dataset()`.
- `src/train.py :: train()` (wires all of the above together; see `--ddi-fraction` CLI
  argument).

## Notes / open items
- `ddi_fraction=0.3` is a chosen default, not derived from a formal search — worth a
  short ablation (e.g. 0.2/0.3/0.4) if time allows, to justify the value in Chapter 3
  rather than stating it as arbitrary.
