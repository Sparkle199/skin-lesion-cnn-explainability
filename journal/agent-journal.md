# Agent Journal

Evidence log for Appendix A.4 (Reflective account). Entries are appended after each
significant task, per `skills/reflection/SKILL.md`.

---

## 2026-07-31 — Project scaffolding (governance framework applied)

**What happened:** Read `AI_Coding_Governance_MSc (1).pdf` and scaffolded the project per
Appendix B: created `skills/` with the five required roles (planner, developer, test,
review, reflection) and three optional roles the student selected as justified
(risk-assessor, compliance, data). Wrote `spec/specification.md` from `project_desc.txt`
in the required WHAT/WHY/CONSTRAINTS/RISKS/ACCEPTANCE-CRITERIA form.

**Where uncertain / stuck:** None — the governance document's SKILL.md templates were
used verbatim for the five required and three optional roles, as instructed.

**Assumptions made:** None beyond what `project_desc.txt` and the governance PDF stated.

**What was learned / should change next time:** N/A (initial scaffolding).

---

## 2026-07-31 — DDI dataset added to proposal and specification

**What happened:** Student added the Diverse Dermatology Images (DDI) dataset to the
project after the original proposal was written, specifically to include more
darker-skinned patients than HAM10000 contains. Asked the student to clarify how DDI
would be used (held-out evaluation set vs. combined training data vs. both); student
confirmed combined training data. Edited `project_desc.txt` in eight places (Objectives
2 and 3, Analysis, Design, Development, Evaluation, Social Issues, Legal Issues) to
reflect this, and updated `spec/specification.md` to match.

**Where uncertain / stuck:**
- DDI's exact licence terms were not independently verified. The proposal and
  specification both carry an explicit note flagging this as unconfirmed and requiring
  the student to check DDI's official release documentation before submission.
- DDI's diagnostic taxonomy does not map 1:1 onto HAM10000's seven classes. This was
  surfaced as an open design decision (map to nearest HAM10000 class, or fall back to a
  malignant/benign designation) rather than resolved unilaterally, since it is a
  modelling choice with downstream effects on every later phase.

**Assumptions made:**
- Assumed "add DDI as combined training data" (the student's stated choice) implies DDI
  images are merged into the same training pipeline as HAM10000, not used as a separate
  parallel dataset — this shaped the wording added to the Development and Design
  sections.
- Assumed the student wants the resulting skin-tone bias check to be a stratified
  evaluation metric (accuracy broken down by Fitzpatrick skin type) rather than a
  qualitative discussion only, since this is the natural way to make "the addition of
  DDI actually reduces bias" verifiable rather than asserted. This was added to the
  Evaluation section and specification and should be confirmed with the supervisor.

**What was learned / should change next time:** Dataset changes made after the original
proposal was written can ripple into legal, social, design, and evaluation sections
simultaneously — worth checking all of these systematically (rather than only the
section where the dataset is first mentioned) whenever a dataset is added or changed.

---

## 2026-07-31 — DDI zip inspected; taxonomy-mismatch resolved as a second binary task

**What happened:** Extracted and inspected `ddidiversedermatologyimages.zip` directly
(656 PNGs + `ddi_metadata.csv`) rather than taking the earlier assumed DDI description on
trust. Computed actual distributions: `skin_tone` is 3 grouped bands (12/34/56 ≈
Fitzpatrick I-II/III-IV/V-VI), fairly even at 208/241/207; `malignant` is a clean
pre-existing boolean (171 malignant / 485 benign); `disease` has 78 distinct diagnoses,
dominated by melanocytic-nevi (119), seborrheic-keratosis (58), verruca-vulgaris (50),
basal-cell-carcinoma (41). Reported this to the student, flagging that a large share of
the 78 diagnoses (verruca-vulgaris, epidermal-cyst, acrochordon, neurofibroma, lipoma,
molluscum-contagiosum, etc.) have no reasonable HAM10000 seven-class counterpart, making
the "map DDI onto HAM10000's 7 classes" approach from the previous entry unworkable for
a meaningful fraction of DDI. Offered three resolutions; student chose to use DDI's
native malignant/benign flag as a second, added classification task rather than forcing
it into the seven-class taxonomy. Updated `project_desc.txt` (Objectives 2/3, Analysis,
Design, Development, Evaluation) and rewrote `spec/specification.md` to describe two
tasks: primary seven-class (HAM10000 only) and secondary binary malignant/benign
(HAM10000 relabelled + full DDI).

**Where uncertain / stuck:**
- The student's chosen option was phrased "reframe... as (or add)", i.e. it allowed
  either replacing the seven-class objective or adding a second task alongside it. Chose
  to add rather than replace, reasoning that the dissertation title and Aim are
  specifically about *multi-class* detection, so removing the seven-class task would
  contradict the project's own title. Flagged this interpretation explicitly to the
  student rather than silently deciding it.
- The malignant/benign split used to relabel HAM10000 (malignant = akiec, bcc, mel;
  benign = bkl, df, nv, vasc) follows common practice in HAM10000 literature but was not
  itself stated in the proposal — worth the student confirming this grouping with their
  supervisor, since akiec (actinic keratosis / early carcinoma-in-situ) is a borderline
  case some papers treat differently.

**Assumptions made:** None beyond the two items above — the rest of the dataset facts
(counts, distributions) were read directly from the extracted files, not assumed.

**What was learned / should change next time:** Take a stated dataset description on
trust only long enough to plan the next step — actually opening the file uncovered a
concrete blocker (the 40%-unmappable diagnoses) that a description alone did not reveal.
Verify data files early, before the proposal text commits to a specific handling
approach.

---

## 2026-07-31 — HAM10000 archive inspected; lesion-leakage risk and external test set found

**What happened:** Extracted and inspected `archive (1).zip`, the full HAM10000 release
obtained from the Harvard Dataverse. Confirmed it contains: 10,015 training images
(`HAM10000_images_combined_600x450/`), matching lesion segmentation masks
(`HAM10000_segmentations_lesion_tschandl/`, not used in this project), the metadata CSV
(`HAM10000_metadata`), and — not mentioned in the original proposal — the official
ISIC2018 Task 3 held-out test set (`ISIC2018_Task3_Test_Images/`, 1,511 images, with its
own ground truth CSV). Computed the `dx` distribution directly from the metadata (nv
6,705 / mel 1,113 / bkl 1,099 / bcc 514 / akiec 327 / vasc 142 / df 115 — confirms the
proposal's "almost 67%" imbalance claim is accurate) and found only 7,470 unique
`lesion_id`s across the 10,015 images, meaning roughly a quarter of images are repeat
photographs of the same lesion. Flagged the resulting data-leakage risk (image-level
splits could put the same lesion in both train and test) to the student as a correctness
fix rather than a discussion point, and asked whether to adopt the official ISIC2018 test
set as an independent generalisation check. Student confirmed yes. Updated
`project_desc.txt` (Analysis, Design, Development, Evaluation) and `spec/specification.md`
to require a lesion-level split for the primary task and to add ISIC2018 as its
independent test set.

**Where uncertain / stuck:** None — added the lesion-level split as a standard
correctness fix without asking, since forcing lesion-disjoint splits is uncontroversial
best practice for this dataset, not a genuine design trade-off.

**Assumptions made:** None — dataset facts were read directly from the extracted files.

**What was learned / should change next time:** Real dataset archives frequently include
more than the headline files a proposal names (here: segmentation masks and an official
held-out test set neither previous version of the proposal mentioned). Worth inspecting
the full archive contents, not just the file the proposal names, whenever a raw dataset
file is added to the project.

---

## 2026-07-31 — Development phase started: data pipeline, model builder, evaluation module

**What happened:** Student chose to move to implementation (Developer role) rather than
task-list planning or the Appendix A declaration. Built, against `spec/specification.md`:
- `src/config.py` — paths, seven-class list, malignant/benign grouping, DDI skin-tone
  band labels, per-architecture input sizes.
- `src/data/ham10000.py` — metadata loading, lesion-level stratified train/val split,
  ISIC2018 held-out test loader.
- `src/data/ddi.py` — DDI metadata loading (skin-tone band + binary label decoding) and
  a new `split_ddi` function.
- `src/data/binary_corpus.py` — HAM10000-to-binary relabelling and the combined
  HAM10000+DDI corpus builder for the secondary task.
- `src/data/augmentation.py`, `src/data/pipeline.py` — augmentation layer, class-weight
  helper, and the tf.data loading pipeline.
- `src/models/build.py` — shared transfer-learning builder (frozen backbone + head) and
  progressive-unfreeze function for all three architectures and both tasks.
- `src/evaluate.py` — the full metric suite plus skin-tone-stratified binary metrics.
- `src/train.py` — two-phase (frozen, then fine-tune) training entrypoint per
  architecture/task pair.

**Where uncertain / stuck:** The spec described a combined HAM10000+DDI corpus for the
binary task but did not say whether DDI itself needed its own train/validation split.
Recognised that without one, all 656 DDI images would land in training and none would be
available for the skin-tone-stratified evaluation the spec explicitly requires — so added
`ddi.split_ddi`, stratified jointly on skin-tone group and malignancy, as a necessary
consequence of an already-agreed requirement rather than a new open question.

**Assumptions made:** None beyond the split-ddi addition above, which follows directly
from an existing requirement rather than introducing a new one.

**How output was verified (not just generated):**
- All new modules were syntax-checked with `python -m py_compile`.
- `src/data/ham10000.py`, `src/data/ddi.py`, `src/data/binary_corpus.py` (pure
  pandas/sklearn, no TensorFlow dependency) were runtime-tested against synthetic data
  shaped like the real, directly-inspected schemas: confirmed zero lesion overlap between
  splits, correct malignant/benign relabelling, correct DDI skin-tone/label decoding, and
  a correctly-combined binary corpus.
- `src/evaluate.py` was runtime-tested against synthetic predictions for both the
  seven-class and binary cases, and for the skin-tone-stratified breakdown.
- `src/data/pipeline.py`, `src/models/build.py`, `src/train.py` depend on TensorFlow,
  which is not installed in this local environment (training is specified to run on
  Kaggle GPU infrastructure per the proposal) — these were syntax-checked only, not
  runtime-tested. This is a genuine gap, not a demonstrated success, and should be closed
  by running them on Kaggle (or with TensorFlow installed) before relying on them.

**What was learned / should change next time:** Separating pure-Python/pandas logic
(splitting, relabelling) from TensorFlow-dependent code (model building, tf.data
pipelines) made it possible to genuinely runtime-test most of the pipeline locally
despite not having TensorFlow or the real image files available in this environment —
worth keeping that separation deliberately as development continues.

---

## 2026-07-31 — Grad-CAM and SHAP wrappers added

**What happened:** Built `src/xai/gradcam.py` and `src/xai/shap_explain.py`, per spec's
requirement to apply both methods to each trained model on both tasks. Grad-CAM computes
gradients of the predicted class score against the backbone's last conv/activation layer
(mapped per architecture in a new `config.LAST_CONV_LAYER`), re-applying the model's own
final Dense layer to the backbone's pooled output rather than needing the backbone's
internals exposed through the outer wrapped model's graph. Also added
`models.build.preprocess_for` so the XAI wrappers reuse the exact same per-architecture
preprocessing as training, instead of duplicating that mapping. SHAP uses
`shap.Explainer` with an Image masker (Partition algorithm) treating the model as a
black-box predict function, so it works uniformly across all three architectures without
depending on internal layer access.

**Where uncertain / stuck:** None on the design itself. Deliberately did not implement a
quantitative faithfulness metric (e.g. overlap between Grad-CAM's high-activation region
and HAM10000's lesion segmentation masks, which are present in the archive and would give
an objective ground truth for "discriminative visual features") -- the proposal
explicitly states those masks are "not used within the current scope of this project"
(added in an earlier edit), so adding this now would silently expand scope again. Flagging
it here as a worthwhile option for the student/supervisor to decide on for the Evaluation
phase, rather than deciding it myself.

**Assumptions made:** None beyond the last-conv-layer name per architecture
(`conv5_block3_out` / `top_activation` / `block5_conv3`), which are the standard,
well-documented final feature layers for these three `keras.applications` models, not a
project-specific judgement call.

**How output was verified (not just generated):**
- `src/xai/gradcam.py` requires TensorFlow and a trained model; consistent with the
  earlier TF-dependent modules, this was syntax-checked only, not runtime-tested.
- `src/xai/shap_explain.py` was first caught failing its own wiring test: it imported
  `tensorflow` purely for a type hint on `model`, contradicting its own stated design
  ("treats the model as a black-box predict function"). Fixed by replacing the
  `tf.keras.Model` type hint with a small `Protocol` requiring only `.predict()`,
  removing the unnecessary import entirely. After also installing `shap` and
  `opencv-python-headless` locally (the latter required by `shap.maskers.Image`, and
  already listed in requirements.txt for the real run), it was runtime-tested against a
  stub predict function standing in for a real trained model (necessary since no trained
  model exists yet) -- confirmed `build_explainer` returns a working `PartitionExplainer`
  and `explain_images` runs end-to-end, returning attributions shaped
  (n_images, H, W, 3, n_classes) as expected. This checks the wiring, not whether real
  attributions would be meaningful, which needs an actual trained model.

**What was learned / should change next time:** The lesion segmentation masks bundled
with HAM10000 are a natural, objective ground truth for Grad-CAM/SHAP faithfulness
(measurable overlap, not just visual inspection) -- worth revisiting whether to bring
them into scope specifically for the Evaluation phase's faithfulness assessment, even
though they were excluded from the project more broadly.

---

## 2026-07-31 — Segmentation masks brought into scope for quantitative faithfulness

**What happened:** Student confirmed bringing HAM10000's lesion segmentation masks into
scope for the Evaluation phase's faithfulness check (the option flagged in the previous
entry). Verified the mask filename convention and pixel format directly by extracting one
real mask from the archive (`<image_id>_segmentation.png`, single-channel, binary
{0, 255}, matching each image's dimensions) rather than assuming it. Updated
`project_desc.txt` (Analysis, Evaluation) and `spec/specification.md` (WHAT,
CONSTRAINTS, SUCCESS CRITERIA, task decomposition) to describe a quantitative IoU/Dice
faithfulness score for HAM10000-derived predictions, explicit that this does not extend
to DDI (no equivalent mask exists there, so DDI's faithfulness stays visual-only). Added
`src.data.ham10000.load_segmentation_mask` and a new `src/xai/faithfulness.py`
(`compute_overlap`, `mean_overlap`).

**Where uncertain / stuck:** None -- the mask format was confirmed directly rather than
assumed, and IoU/Dice are standard, unambiguous overlap metrics once the mask format was
known.

**Assumptions made:** None beyond the metric choice itself (IoU and Dice together, both
standard for this kind of overlap check, rather than picking only one).

**How output was verified (not just generated):**
- `src/xai/faithfulness.py` first hit the same avoidable-coupling issue as the SHAP
  wrapper: it imported TensorFlow at module level for a resize step only needed when
  shapes mismatch. Fixed by moving the import inside that conditional branch, so the
  module's actual math has no hard TensorFlow dependency.
- `load_segmentation_mask` was runtime-tested against a real mask extracted from the
  archive (not synthetic data), confirming correct shape and boolean output.
- `compute_overlap` was runtime-tested against three hand-checked cases -- perfect
  match (IoU=Dice=1.0), complete disjoint (IoU=Dice=0.0), and a small worked-by-hand
  partial-overlap case (IoU=1/3, Dice=0.5) -- and `mean_overlap` was checked against the
  average of two of those cases. This verifies the metric math itself is correct, not
  yet whether real Grad-CAM/SHAP outputs are faithful, which needs a trained model.

**What was learned / should change next time:** The "import a heavy framework only for
a type hint or an edge-case branch" mistake recurred a second time (first in
shap_explain.py, now here) -- worth checking new modules for this specifically before
calling them done, not just after a test happens to catch it.

---

## 2026-07-31 — Ad-hoc verification formalised into a committed test suite (Test role)

**What happened:** Everything verified so far had lived in one-off scripts in the
session scratchpad, which is not part of the project and does not survive the session --
this contradicts the governance framework's "verification is the bottleneck" principle,
since none of that verification was actually reproducible or checked into the project.
Converted it into a real `tests/` suite (pytest, added to requirements.txt as a dev
dependency): `conftest.py` (synthetic HAM10000/DDI/mask fixtures shaped like the real,
directly-inspected schemas), `test_ham10000.py`, `test_ddi.py`, `test_binary_corpus.py`,
`test_evaluate.py`, `test_faithfulness.py`, `test_shap_explain.py`. Also added two tests
that hadn't existed even as scratch scripts: `load_segmentation_mask`'s actual file-read
path (via a synthetic mask written to a temp dir, not just the one real mask extracted
earlier) and its missing-file error case. All 23 tests pass.

**Where uncertain / stuck:** None -- this was reformatting already-designed
verification logic into permanent, assertion-based tests rather than new design work.

**Assumptions made:** Used pytest's `monkeypatch` fixture to redirect `src.config`
path constants to temp directories/files per test, rather than mutating global config
state directly -- standard pytest practice, not a project-specific judgement call.

**What remains untested, and why (per the Test role's own procedure -- this list should
not be allowed to go stale as more of the codebase is written):**
- `src/data/pipeline.py`, `src/models/build.py`, `src/train.py`, `src/xai/gradcam.py` --
  all require TensorFlow, which is not installed in this local environment. Syntax-
  checked only. This is the single biggest verification gap in the project so far and
  needs a real Kaggle (or local TensorFlow) run before any of it can be trusted.
- Real image data was never used for anything beyond the one segmentation mask and one
  DDI/HAM10000 metadata CSV extracted directly for inspection -- no test decodes an
  actual `.jpg`/`.png` lesion image, since doing so at scale would mean extracting the
  3.2GB and 237MB archives locally, which was deliberately avoided (see the two archive-
  inspection entries above; extraction happens on Kaggle, not here).
- The SHAP and Grad-CAM wrappers are only tested for wiring/shape correctness against a
  stub or hand-checked case, never for whether real attributions on a real trained model
  are actually faithful -- that is the whole point of the Evaluation phase and cannot be
  demonstrated before a model exists.
- No end-to-end test exists that runs `src.train.train(...)` and then `src.evaluate`
  against its output -- there is no orchestration script tying training to evaluation
  yet (next planned piece of Development work).

**What was learned / should change next time:** Verification done in a scratchpad is not
verification the project can rely on later -- it should be written as committed tests
from the start of a task, not moved into the repo as an afterthought once a few tasks
have already accumulated untracked scratch scripts.

---

## 2026-08-06 — Evaluation orchestration script (src/evaluate_run.py)

**What happened:** Built `src/evaluate_run.py`, the piece flagged as missing in the
previous entry: loads a trained model, predicts on its task's validation set (plus,
for the seven-class task, the ISIC2018 held-out test set), calls `src.evaluate`'s
metric functions, and runs the Grad-CAM faithfulness check from `src.xai.faithfulness`
against a sample of HAM10000-derived predictions. Writes results to JSON.

While designing it, found and fixed a real defect: `src.data.binary_corpus.
build_binary_corpus` explicitly selected a four-column subset that dropped `image_id`,
which the faithfulness check needs to look up each HAM10000 row's segmentation mask --
without it, no HAM10000 row surviving into the binary task's combined corpus could ever
be matched back to its ground-truth mask. Fixed by adding `image_id` to the kept
columns (None for DDI rows, which have no equivalent identifier or mask). Added a test
(`test_build_binary_corpus_keeps_ham10000_image_id_for_mask_lookup`) specifically
asserting this, on top of updating the existing column-list assertion that the fix
changed. All 4 binary_corpus tests still pass.

Also extracted `_json_default` (numpy-type JSON serialisation) out of evaluate_run.py
into a new `src/json_utils.py`: evaluate_run.py necessarily imports TensorFlow at the
top (unlike the two earlier false-economy cases, TF is core to what this module does,
not an incidental edge-case dependency), which would have made even this pure-numpy
helper untestable locally. Splitting it out kept it testable and makes it reusable
anywhere else `src.evaluate`'s numpy-array-containing results get written to disk.

**Where uncertain / stuck:** `config.BACKBONE_LAYER_NAME` (added this task) assumes a
reloaded saved model's backbone submodel is retrievable via `model.get_layer(name)`
using keras.applications' default model names ("resnet50", "efficientnetb4", "vgg16").
This is standard Keras behaviour but has not been verified against an actual
saved-and-reloaded model, since that needs TensorFlow. Flagged in the config.py comment
itself, not just here, since it's exactly the kind of assumption that should be checked
first when this finally runs on Kaggle.

**Assumptions made:** That `val_df`'s row order is preserved through `make_dataset`
(no shuffling when `training=False`) and therefore lines up positionally with
`predict_dataset`'s output for reattaching `y_true`/`y_pred` to the binary task's
DataFrame (needed for `stratified_binary_metrics`). This follows directly from how
`src/data/pipeline.py` was written (shuffle only under `training=True`), not a new
assumption introduced here.

**How output was verified (not just generated):**
- `src/evaluate_run.py` requires TensorFlow, a trained model, and real image files --
  none available locally. Syntax-checked only, consistent with the other TF-dependent
  modules (`pipeline.py`, `models/build.py`, `train.py`, `xai/gradcam.py`).
- `src/json_utils.py` has no TensorFlow dependency and was properly committed as tests
  (`tests/test_json_utils.py`): ndarray-to-list, numpy scalar types, and the
  unsupported-type error case. All pass.
- Also fixed a robustness gap caught by re-reading the script rather than a test:
  `main()` wrote to `results/<arch>_<task>.json` without ensuring the `results/`
  directory exists first. Added `Path(...).parent.mkdir(parents=True, exist_ok=True)`.

**What was learned / should change next time:** The binary_corpus bug here is the same
shape as a class of mistake worth watching for generally: a function that explicitly
selects "the columns I think are needed" is fragile against a *later* piece of code
needing one more column than the original author anticipated. Worth being more
conservative about narrowing DataFrames to a fixed column list until it's clear no
downstream consumer will need anything else.

---

## 2026-08-06 — Trade-off Analysis phase (src/trade_off.py)

**What happened:** Student chose to move to the final spec phase (Trade-off Analysis,
Objective 5 in the proposal) rather than run the pipeline for real on Kaggle first.
Built `src/trade_off.py`: loads all available `src.evaluate_run` JSON outputs, builds a
per-architecture comparison table (seven-class accuracy/kappa, ISIC2018 test accuracy,
faithfulness IoU/Dice, binary accuracy, skin-tone accuracy spread), and produces a
plain-text summary ranking architectures by accuracy and by faithfulness separately.

**Where uncertain / stuck:** None on the design itself, but made a deliberate framing
choice worth recording: the summary reports rankings on each dimension and explicitly
flags disagreement between them, rather than computing a single combined "best model"
score. Reasoning: the proposal's own Objective 5 calls this a "critical analysis" of a
trade-off, and collapsing accuracy and interpretability into one arbitrary weighted
score would substitute the script's judgement for the student's -- which the governance
framework's "delegate tasks, not judgement" principle says shouldn't happen. Flagged
this choice in the module's own docstring, not just here, since it's a framing decision
a reader of the code should see directly.

**Assumptions made:** Used the seven-class task's accuracy and faithfulness as the
primary comparison axes (rather than the binary task's), since the proposal's Objective
5 and Trade-off Analysis section frame the trade-off specifically around "the multi-class
classification problem." The binary task's accuracy and skin-tone spread are still
included in the comparison table for reference, just not used to drive the ranking.

**How output was verified (not just generated):** Unlike every module since
`src/models/build.py`, this one has no TensorFlow dependency at all -- it only reads
JSON -- so it was fully runtime-tested, not just syntax-checked: 7 tests in
`tests/test_trade_off.py` against synthetic result files shaped exactly like
`evaluate_run.evaluate`'s real output (missing-file handling, None-filling for
not-yet-run architectures, correct faithfulness/skin-tone-spread extraction, ranking
with missing values skipped, and both trade-off and single-leader summary cases). All
34 project tests pass together.

**What was learned / should change next time:** Not every orchestration-layer module
needs TensorFlow just because it sits downstream of TF-dependent ones -- separating
"reads other steps' output and reports on it" (this module) from "produces that output"
(evaluate_run.py) kept this piece fully testable where the modules feeding it aren't.
Worth looking for this separation deliberately in future phases too.
