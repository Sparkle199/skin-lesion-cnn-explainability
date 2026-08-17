# Stage 8 — XAI Generation (SHAP + saved overlays)

**Status:** Partially implemented — `src/xai/shap_explain.py` exists and is tested;
no script yet runs it across all 6 models or saves overlay images for demo/report use.

## Purpose
Generate SHAP attribution maps for each trained model (Grad-CAM is already produced
inline during Stage 7's faithfulness check) and persist visual overlays for both
Grad-CAM and SHAP, for use in the dissertation write-up and the Streamlit app
(Stage 10).

## Inputs
- `models/{architecture}_{task}.keras` (Stage 6).
- Sample images to explain — HAM10000 validation rows (for faithfulness-comparable
  Grad-CAM) and DDI validation rows (qualitative-only, no ground-truth mask).

## Process
`src/xai/shap_explain.py`:
1. `build_explainer()` wraps a trained model as a **black-box predict function** with
   a SHAP `Image` masker (`blur(128,128)`, Partition algorithm) — deliberately not
   `DeepExplainer`/`GradientExplainer`, so the same code works uniformly across all
   three architectures and both task head types (sigmoid/softmax) without depending on
   internal layer access or a specific Keras backend version. Each architecture's own
   preprocessing is already baked into the model by `src.models.build.build_model()`,
   so `model.predict` accepts raw pixel-scale images directly.
2. `explain_images()` runs the explainer over a batch of images (`max_evals=500`,
   `batch_size=50` by default), returning one attribution map per image per class.

**Not yet built:** a script that (a) loads each of the 6 saved models, (b) runs
`shap_explain` over a fixed sample of images per model, (c) uses
`src/xai/gradcam.py :: overlay_heatmap()` to render Grad-CAM overlays on the same
images for side-by-side comparison, and (d) saves the results to disk (e.g.
`results/xai_overlays/{architecture}_{task}/{image_id}_gradcam.png` and
`..._shap.png`) rather than only existing transiently in memory.

## DDI's qualitative check
DDI carries no ground-truth segmentation mask, so its "faithfulness" assessment is
visual inspection only — a human (student/supervisor) looks at whether the
Grad-CAM/SHAP attention zones plausibly align with the lesion, rather than a scored
IoU/Dice number. This is why saving overlay images (not just raw arrays) matters for
DDI specifically — the qualitative check *is* the saved image.

## Outputs (once the missing script is written)
- Saved Grad-CAM + SHAP overlay images per model, covering both a HAM10000 sample
  (faithfulness-scorable, Stage 7) and a DDI sample (qualitative-only).

## Code references
- `src/xai/shap_explain.py :: build_explainer()`, `explain_images()`.
- `src/xai/gradcam.py :: make_gradcam_heatmap()`, `overlay_heatmap()`.

## Tests
- `tests/test_shap_explain.py`.

## Notes / open items
- **This is the main remaining gap in Stage 8.** SHAP's `Image` masker with
  `max_evals=500` is comparatively expensive per image — budget for this when
  deciding sample size (don't attempt this over the full validation set; a fixed
  small sample per model, similar in spirit to Stage 7's `faithfulness_samples=30`,
  is the right scale).
- Once overlays are saved to disk, Stage 10 (Streamlit) should read them rather than
  recomputing SHAP live in the app — SHAP is too slow for interactive use.
