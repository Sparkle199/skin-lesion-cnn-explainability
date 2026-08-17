# Stage 5 — Model Construction

**Status:** Implemented.

## Purpose
Build one transfer-learning model per (architecture, task) pair, sharing a single
builder function across all three architectures and both tasks rather than
duplicating model-definition code six times.

## Inputs
- Architecture name (`resnet50` / `efficientnetb4` / `vgg16`).
- `num_classes` (7 for the seven-class task, 2 for the binary task).
- Per-architecture input size and preprocessing function (`src/config.py`,
  `src/models/build.py`).

## Process
`src/models/build.py :: build_model()`:
1. Loads the architecture's ImageNet-pretrained backbone (`include_top=False,
   weights="imagenet", pooling="avg"`), **frozen** (`base.trainable = False`).
2. Wraps it: input → architecture-specific `preprocess_input` → backbone → `Dropout(0.3)`
   → a task-appropriate head:
   - `num_classes == 2` (binary task): single sigmoid unit, `binary_crossentropy` loss.
   - `num_classes == 7` (seven-class task): softmax over 7 units, `categorical_crossentropy`
     loss.
3. Returns `(model, base)` — `base` is kept as a separate reference specifically so
   Stage 6's fine-tuning phase can target it directly (`unfreeze_top_layers(base, ...)`)
   without re-discovering it by layer name from inside the assembled `model`.

`src/models/build.py :: unfreeze_top_layers()` (used in Stage 6, defined here since
it operates on the model object this stage produces):
- Unfreezes only the top `num_layers` layers of the backbone, keeping earlier layers
  frozen — the standard progressive-unfreezing transfer-learning approach.
- **BatchNormalization layers within that unfrozen range stay frozen regardless** —
  updating BN running statistics on a comparatively small fine-tuning set is a common
  source of instability in transfer learning, so this is a deliberate exception, not an
  oversight.

`src/models/build.py :: preprocess_for()` exposes each architecture's own ImageNet
preprocessing function separately, so the XAI modules (Stage 7–8) can feed
correctly-preprocessed images directly to a model's backbone without duplicating the
architecture→preprocessing mapping.

## Outputs
- `model` (compiled in Stage 6, not here — this stage only constructs the graph).
- `base` — reference to the backbone submodel, needed by Stage 6.

## Code references
- `src/models/build.py :: build_model()`, `unfreeze_top_layers()`, `preprocess_for()`.
- `src/config.py :: IMAGE_SIZE` (per-architecture input resolution: EfficientNetB4 at
  380×380, ResNet-50/VGG-16 at 224×224).

## Notes / open items
- `BACKBONE_LAYER_NAME` in `src/config.py` (used by `src/evaluate_run.py` to re-fetch
  the backbone submodel from a *reloaded* saved model) is explicitly flagged there as
  **not yet verified against a real saved-and-reloaded model** — TensorFlow wasn't
  available in the environment where it was written. Confirm this holds (i.e. that
  `model.get_layer(config.BACKBONE_LAYER_NAME[architecture])` actually returns the
  backbone after `tf.keras.models.load_model()`) the first time Stage 7 is run for
  real, before trusting evaluation output.
