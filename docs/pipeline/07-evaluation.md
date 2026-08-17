# Stage 7 — Evaluation

**Status:** Implemented, not yet run (depends on Stage 6 producing saved models).

## Purpose
Compute the full metric suite for each trained model, on the right evaluation set(s)
for its task, plus the quantitative Grad-CAM faithfulness check.

## Inputs
- `models/{architecture}_{task}.keras` (Stage 6).
- Validation split for the model's task (Stage 2–3).
- For the seven-class task only: the ISIC2018 held-out test set (Stage 1).
- HAM10000 ground-truth segmentation masks (Stage 1), for the faithfulness check.

## Process
`src/evaluate_run.py :: evaluate()`:
1. Loads the saved model (`tf.keras.models.load_model`).
2. `predict_dataset()` runs inference over the task's validation set, returning
   `(y_true, y_pred, y_proba)` as integer class indices/probabilities.
3. `src/evaluate.py :: evaluate_predictions()` computes the core metric suite:
   accuracy, per-class precision/recall/F1, ROC-AUC (binary or one-vs-rest
   multiclass), Cohen's Kappa, and the confusion matrix.
4. **Seven-class task only:** also predicts on the ISIC2018 held-out test set and
   computes the same metric suite separately (`results["isic2018_test"]`) — an
   independent generalisation check beyond the internal HAM10000 split.
5. **Binary task only:** `src/evaluate.py :: stratified_binary_metrics()` breaks
   accuracy/precision/recall/F1 down by `skin_tone_group`. **Only rows with a
   non-null skin-tone group are included** — HAM10000 rows carry no skin-tone label,
   so this stratified breakdown reports DDI's contribution only, not the whole binary
   validation set. (The overall, non-stratified binary metrics from step 3 already
   cover the full HAM10000+DDI validation set — the stratified breakdown is an
   additional, narrower view.)
6. **Faithfulness check** (`run_faithfulness_check()`): samples up to
   `faithfulness_samples` (default 30) HAM10000-derived validation rows only (DDI rows
   are excluded — no ground-truth mask exists for them), generates a Grad-CAM heatmap
   per sample (`src/xai/gradcam.py :: make_gradcam_heatmap()`), and scores mean IoU/Dice
   against the ground-truth lesion mask (`src/xai/faithfulness.py :: mean_overlap()`).
   Returns `None` (and is omitted from output) if there are no HAM10000 rows to sample
   — not expected in practice here, since both tasks' validation sets include HAM10000,
   but the code guards for it.

## CLI usage
```
python -m src.evaluate_run --architecture resnet50 --task seven_class \
    --model-path models/resnet50_seven_class.keras
```
Run once per trained model — 6 invocations total, mirroring Stage 6's run matrix.

## Outputs
- `results/{architecture}_{task}.json` — one file per run, containing:
  `validation` (core metrics), `isic2018_test` (seven-class only),
  `skin_tone_stratified` (binary only, DDI rows only), `faithfulness` (mean IoU/Dice,
  HAM10000 rows only).

## Code references
- `src/evaluate_run.py :: predict_dataset()`, `run_faithfulness_check()`, `evaluate()`.
- `src/evaluate.py :: evaluate_predictions()`, `stratified_binary_metrics()`.
- `src/xai/gradcam.py :: make_gradcam_heatmap()`.
- `src/xai/faithfulness.py :: compute_overlap()`, `mean_overlap()`.
- `src/json_utils.py :: numpy_json_default()` (JSON serialisation of numpy types).

## Tests
- `tests/test_evaluate.py`, `tests/test_faithfulness.py`.

## Notes / open items
- Confirm `config.BACKBONE_LAYER_NAME` actually resolves the backbone submodel from a
  *reloaded* model the first time this runs for real — flagged as unverified in Stage 5.
- `faithfulness_samples=30` is a fixed sample size, not the full validation set — fine
  for a representative score, but note this in Chapter 4 as a sampled estimate, not an
  exhaustive one.
