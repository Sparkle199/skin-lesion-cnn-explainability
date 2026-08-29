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

---

## 2026-08-17 — Documentation set added: README, pipeline diagrams, decision log entry

**What happened:** Student asked for a step-by-step methodology from data processing
through to a new, previously-unplanned Streamlit demo stage, then asked for this to be
made durable as documentation. Added `README.md` (system overview, one exhaustive
cross-stage mermaid diagram, a per-stage status table) and
`docs/pipeline/00-infrastructure.md` through `10-streamlit-app.md` (one file per
pipeline stage, each documenting purpose/inputs/process/outputs/code references/status
against the real `src/` modules, not just the proposal text). Later added one
stage-specific mermaid diagram per doc, zooming into that stage's internal
branching/logic rather than repeating the README's cross-stage overview.

Stage 10 (Streamlit) and part of Stage 8 (saving SHAP/Grad-CAM overlays to disk, rather
than only producing them in memory) are documented as **not yet implemented** -- no code
exists for either. This is new scope beyond the original proposal (Streamlit was never
in `project_desc.txt`), added specifically to make the interpretability comparison
(Objective 5) tangible for the dissertation and any demo.

**Where uncertain / stuck:** A `spec/decisions.md` (Chapter-3-ready decision log,
ADR-style) was drafted for the Runpod GPU choice but the write was rejected/interrupted
by the student before being saved -- it does not exist in the repo. The GPU decision's
rationale ended up captured instead in `docs/pipeline/00-infrastructure.md`, so nothing
was lost, but a first attempt at README referenced the non-existent `spec/decisions.md`
file and had to be corrected before committing. Worth checking a referenced file
actually exists before citing it, not just before this specific fix.

**Assumptions made:** That every pipeline stage warranted its own diagram ("where
necessary" was read as "all 11 stages," since each has genuine internal branching or
sequencing worth visualising) rather than only a subset -- flagged here since the
student's phrasing left this open to a narrower reading.

**A factual error and its correction (worth recording since it affects a cost figure
now in the docs):** Initially recommended and documented the RTX 4090 at $0.99/hr --
this was actually the RTX 5090's price, mistakenly attributed to the 4090 while reading
down the pasted Runpod pricing list. Student caught the inconsistency by asking "RTX
4090 or RTX 5090?". Corrected to the real RTX 4090 rate ($0.74/hr) in
`docs/pipeline/00-infrastructure.md` (diagram + cost estimate, ~$27 not ~$36 for the
36-GPU-hr estimate). **Lesson: when transcribing a list of similar priced options,
re-verify the specific figure against the source list before it propagates into
multiple documents, rather than trusting the first write-down.**

**What was learned / should change next time:** Documentation that cites specific
numbers (prices) or specific file paths (spec/decisions.md) needs the same
verify-before-trusting discipline already applied to dataset facts earlier in this
project -- both mistakes above were caught by the student, not by self-review, and both
were avoidable with a direct re-check against the source (the pasted pricing list; the
actual repo contents) before writing.

---

## 2026-08-17 — Runpod pod provisioned; SSH access established; dataset upload started

**What happened:** Student set up a Runpod pod (RTX 4090) via the web UI (outside this
session) and asked to connect to it from here. Generated a new local SSH keypair
(`~/.ssh/id_ed25519`, no passphrase) and provided the public key for the student to
register with Runpod. First connection attempt failed (`Permission denied
(publickey,password)`) because the student had not yet actually saved the key on
Runpod. Rather than wait on the account-settings UI (which may require a pod restart to
take effect for an already-running pod), used Runpod's browser-based Web Terminal as a
faster path: gave the student a one-line command to append the public key directly to
the pod's `~/.ssh/authorized_keys` via the already-open web terminal, with no restart
needed. Retried and connected successfully; confirmed via `nvidia-smi` that the pod
does have the ordered RTX 4090 (24564 MiB VRAM), matching `docs/pipeline/00-infrastructure.md`.

Then began Stage 1 (data acquisition) for real: confirmed the pod has a large
persistent `/workspace` volume (658T available, network-backed), created
`/workspace/uploads/` and the `/workspace/data/raw/{dataverse_files,ddi}` destination
directories per `src/config.py`'s expected layout, and started uploading both local
archives (`archive (1).zip`, 3.2GB; `ddidiversedermatologyimages.zip`, 237MB) via `scp`
over the same SSH connection, running in the background given the transfer size.

**Where uncertain / stuck:**
- Upload bandwidth/duration from the student's local connection is unknown -- 3.2GB at
  home-upload speeds could take anywhere from minutes to over an hour. Not yet resolved
  as of this entry; extraction into the final `data/raw/...` layout
  (`docs/pipeline/01-data-acquisition.md`) has not happened yet and depends on this
  transfer finishing first.
- Only checked `nvidia-smi` and `/workspace` disk space on the pod so far -- have not
  yet verified the pod's installed Python/CUDA/TensorFlow versions are compatible with
  `requirements.txt` (`tensorflow>=2.15`). This should be checked before the first real
  `python -m src.train` invocation (Stage 6), not assumed.

**Assumptions made:**
- Assumed `/workspace` is the correct persistent-storage location for the datasets
  (rather than the pod's ephemeral container root) based on its large network-backed
  size (658T) relative to the tiny root filesystem implied by the earlier empty
  `/workspace` listing -- this matches Runpod's documented convention (container disk is
  ephemeral, `/workspace` is the persistent volume) but was not independently confirmed
  by reading Runpod's own docs in this session.
- Assumed `root` as the SSH login user, based on how the pod's home directory (`~`)
  resolved during key installation -- confirmed correct once the connection succeeded,
  not assumed indefinitely.

**How output was verified (not just generated):** The SSH connection was verified with
a real remote command (`hostname`, `nvidia-smi`), not just a successful exit code from
`ssh-keygen`/`scp`. The upload's progress was checked mid-transfer via a real `ls -la`
on the pod (`archive (1).zip` partially present, growing), not assumed to be proceeding
correctly from the backgrounded job's exit status alone.

**What was learned / should change next time:** Registering an SSH key through a cloud
provider's account-settings UI is not always immediately live for an already-running
pod -- checking for a faster path (here, the provider's own web terminal) avoided
asking the student to restart a pod that had probably already been paying/running for a
while. Worth checking for this kind of already-open side-channel access before assuming
a UI-driven fix requires a restart.

---

## 2026-08-17 — Stage 1 (data acquisition) completed on the Runpod pod

**What happened:** Continuing directly from the previous entry: the `scp` upload of
both archives finished (`archive (1).zip` 3.2GB, `ddidiversedermatologyimages.zip`
237MB, both to `/workspace/uploads/`). Before extracting, ran `unzip -l` on both
archives to check their internal structure rather than assuming it matched
`docs/pipeline/01-data-acquisition.md`'s description exactly: confirmed the HAM10000
archive already nests everything under a top-level `dataverse_files/` folder (so
extracting with `-d /workspace/data/raw/` lands it correctly), while the DDI archive is
flat -- 656 PNGs and `ddi_metadata.csv` directly at the zip root, no `ddi/` subfolder
(so it needed `-d /workspace/data/raw/ddi/` specifically, not the same target as
HAM10000). Extracted both (21,544 + 657 files), then verified every path and count
against `src/config.py`'s expectations directly rather than trusting a clean exit
code: HAM10000 images 10,015/10,015, segmentation masks 10,015/10,015, ISIC2018 test
images 1,511/1,511, DDI images 656/656, both `HAM10000_metadata` and
`ddi_metadata.csv` present -- all exact matches to the counts established when these
archives were first inspected locally on 2026-07-31. Total 3.3GB on disk. Set
`PROJECT_DATA_DIR=/workspace/data/raw` in the pod's `~/.bashrc` so `src/config.py`'s
`DATA_ROOT` resolves correctly for any future SSH session on this pod, without needing
to be re-exported manually each time.

**Where uncertain / stuck:** None -- this was mechanical execution and verification of
an already-designed data layout (`src/config.py`, `docs/pipeline/01-data-acquisition.md`),
not new design work. The one thing not yet done: the raw zip archives are still sitting
in `/workspace/uploads/` (3.4GB, now redundant with the extracted copy) -- left in
place rather than deleted, since `/workspace` has 658T available and deleting a
student's uploaded files without being asked is the kind of action this project's own
governance framework treats as requiring explicit confirmation, not silent cleanup.

**Assumptions made:** None beyond what's already recorded in the previous entry
(`/workspace` as the persistent-storage location, `root` as the SSH user) -- both held
up through this stage without issue.

**How output was verified (not just generated):** Every extracted path was checked
with a real `ls`/count on the pod against the exact numbers `src/config.py` and
`spec/specification.md` already state (not re-derived here, since they were already
confirmed once by direct archive inspection on 2026-07-31) -- this is a match-check
against previously-verified facts, not a first-time verification.

**What was learned / should change next time:** The two archives had genuinely
different internal layouts (nested vs. flat) despite both being handled by the same
two-line "just extract them" plan in `docs/pipeline/01-data-acquisition.md` -- worth
running `unzip -l` (or equivalent) before extracting any archive whose internal
structure hasn't been directly confirmed, even when a doc already describes the
expected destination layout, since the doc describes the destination, not necessarily
the archive's own internal path prefix.

---

## 2026-08-17 — Environment set up on the pod; the biggest verification gap finally closed

**What happened:** Student asked to confirm the existing scripts were actually useful,
not just plausible-looking. Since GitHub clone failed (repo is private, returns 401
unauthenticated), copied `src/`, `tests/`, `requirements.txt` to the pod directly via
`scp` instead of setting up GitHub credentials remotely. Installed `tensorflow[and-cuda]`
plus the rest of `requirements.txt` into a proper venv (a first attempt using the
system `pip` failed with Debian's `externally-managed-environment` guard -- caught only
because the failure was checked directly rather than trusted from a piped command's
exit code, see below). Confirmed TensorFlow 2.21.0 detects and computes on the RTX 4090
for real (`tf.config.list_physical_devices`, a live GPU add). Ran the existing 34-test
suite for real (all pass, unchanged from local runs -- these were already pure-Python/
pandas tests with no TF dependency).

Then wrote and ran a new test, `tests/test_integration_smoke.py`
(`test_train_evaluate_explain_roundtrip_on_real_data_subset`), specifically to close the
gap called out repeatedly since 2026-07-31: every TF-dependent module
(`models/build.py`, `data/pipeline.py`, `train.py`, `evaluate_run.py`, `xai/gradcam.py`)
had only ever been syntax-checked, never executed. This test builds a real ResNet-50 via
`build_model`, trains it for one frozen epoch and one fine-tune epoch on a small real
HAM10000 subset (32 train / 16 val images) via the real `make_dataset` pipeline, saves
and reloads it, predicts via `evaluate_run.predict_dataset`, scores via
`evaluate.evaluate_predictions`, and runs `xai.gradcam.make_gradcam_heatmap` +
`xai.faithfulness.compute_overlap` against a real ground-truth segmentation mask.
**Passed end-to-end in 136.81s**, including confirming the specific previously-flagged
uncertain assumption in `config.BACKBONE_LAYER_NAME` -- `reloaded.get_layer("resnet50")`
does correctly retrieve the backbone submodel after a real save/reload round-trip.

**Where uncertain / stuck:**
- The smoke test only exercises the seven-class task with `resnet50`. The binary task's
  distinct code paths -- `make_oversampled_binary_dataset`, `binary_corpus.
  build_binary_corpus` feeding into a real DDI+HAM10000 batch, `compute_steps_per_epoch`
  -- remain runtime-*un*tested against real data, as do `efficientnetb4` and `vgg16`
  specifically (only `resnet50` was smoke-tested; the other two share the same
  `build_model`/`unfreeze_top_layers` code path but at different input resolutions and
  different `BACKBONE_LAYER_NAME`/`LAST_CONV_LAYER` values, which haven't been
  individually confirmed).
- SHAP's `explain_images` was not exercised against a real trained model in this smoke
  test (only against a stub predict function, back on 2026-07-31) -- it's comparatively
  slow (`max_evals=500` per image) and was left for Stage 8's dedicated script rather
  than folded into this smoke test.

**Assumptions made:** That a 16-row validation subset was an acceptable smoke-test
size despite it being small enough to land only one class present in `y_true`/`y_pred`
(triggering `UndefinedMetricWarning` for ROC-AUC and Cohen's Kappa) -- acceptable here
since the goal was confirming the *code path* runs without error, not producing
meaningful metrics from 32 training images. Real runs (10,015 images) won't hit this
degenerate case.

**How output was verified (not just generated):** This entire entry *is* the
verification step the project was missing -- real GPU execution, real data, a real
saved-and-reloaded model, checked assertion-by-assertion (shapes, value ranges, the
specific backbone-retrieval assumption) rather than inferred from reading the code.
Additionally caught and fixed a real process bug while getting here: an initial
`pip install ... | tail -30` in a background command reported exit code 0 even though
`pip` itself had failed (Debian's PEP 668 guard) -- because piping through `tail`
silently replaces the reported exit code with `tail`'s, not the failing command's. Only
caught because the *next* command (checking for TensorFlow) failed honestly, prompting
a re-check rather than trusting the false-positive success notification.

**What was learned / should change next time:** Two lessons, both about not trusting a
success signal at face value: (1) never pipe a command whose exit code matters through
`tail`/`head`/`grep` without either checking `PIPESTATUS`/`pipefail` or writing the exit
code out explicitly (`cmd; echo EXIT_CODE:$?`), which is what every command after the
false-positive did; (2) a background task's "completed, exit code 0" notification
describes the wrapper, not necessarily the real work inside it -- worth reading the
actual captured output before reporting success to the student, which is now the
default habit for the rest of this session.

---

## 2026-08-17 — Smoke test extended to efficientnetb4 and vgg16

**What happened:** Student asked to run the new integration smoke test against the
other two architectures, not just resnet50. Parametrized
`test_train_evaluate_explain_roundtrip_on_real_data_subset` over
`["resnet50", "efficientnetb4", "vgg16"]` rather than writing separate test functions,
and re-ran it on the pod filtered to the two new architectures (`-k
'efficientnetb4 or vgg16'`, resnet50 already confirmed in the previous entry). Both
passed for real (216.72s total, genuine exit code 0 checked explicitly, not inferred
from the background wrapper).

This closes the remaining architecture-specific gap noted in the previous entry:
`config.BACKBONE_LAYER_NAME` and `config.LAST_CONV_LAYER` are now confirmed correct
(save/reload backbone retrieval, Grad-CAM target layer) for **all three** architectures,
not just resnet50 -- these were architecture-specific values that could plausibly have
been wrong for one of the three without the resnet50-only run catching it.

**Where uncertain / stuck:** None new -- same remaining gaps as the previous entry
still stand: the binary task's DDI-oversampling code path
(`make_oversampled_binary_dataset`, `binary_corpus.build_binary_corpus` feeding a real
combined batch) and SHAP against a real trained model are still not runtime-tested.

**Assumptions made:** None beyond the previous entry's.

**How output was verified:** Real pytest run on the pod, real log tail read directly
(not the background-task wrapper's summary), explicit `EXIT_CODE:$?` line confirmed 0.

**What was learned / should change next time:** Parametrizing the existing test over
`architecture` was less work and more valuable than writing three near-duplicate test
functions would have been, and it means any future fourth architecture only needs
adding to one list. Worth defaulting to parametrization over copy-paste whenever a test
is "the same steps, different constant" like this one was.

---

## 2026-08-17 — Binary task's own machinery verified end-to-end

**What happened:** Student asked to verify the binary task's specific code paths,
which the seven-class smoke test never touched. Added
`test_binary_task_roundtrip_on_real_data_subset` to
`tests/test_integration_smoke.py`: builds the real combined HAM10000+DDI corpus via
`prepare_data("binary")`, trains a sigmoid-head resnet50 through
`make_oversampled_binary_dataset` (real DDI-fraction-weighted batching, not a
synthetic stand-in) and `compute_steps_per_epoch`, saves/reloads, predicts, computes
`stratified_binary_metrics` against real DDI skin-tone-group labels, and runs
Grad-CAM + faithfulness on a HAM10000-derived row pulled back out of the combined
corpus. **Passed end-to-end in 131.94s**, real exit code confirmed.

While writing it, caught one thing worth recording precisely because it *would* have
been a real bug if missed rather than caught while writing the test: `build_binary_corpus`
concatenates HAM10000 rows before DDI rows, so a naive `.head(n)` on the combined
corpus would silently return HAM10000-only rows for small `n`, and
`make_oversampled_binary_dataset` raises `ValueError` if either source is empty. Wrote
a small `_subset` helper that pulls a slice from each `source` group explicitly instead
of `.head()`-ing the combined frame -- this was a test-construction issue, not a bug in
`src/data/binary_corpus.py` itself, but it's the same shape of mistake
(`build_binary_corpus`'s own docstring already warns about a *related* past bug: an
earlier version dropped `image_id` for the same "assumed structure without checking"
reason).

**Where uncertain / stuck:** None -- this closes the binary-task gap explicitly named
as remaining in both of the previous two entries. What's left unverified project-wide:
SHAP (`xai/shap_explain.py`) has still never been run against a real trained model,
only a stub predict function (2026-07-31). Everything else flagged since 2026-07-31 as
"only syntax-checked" now has real, passing, GPU-executed evidence.

**Assumptions made:** Used small, fixed subset sizes (24/8 train, 12/4 val,
HAM10000/DDI respectively) -- large enough that `make_oversampled_binary_dataset` and
`compute_steps_per_epoch` both operate on non-degenerate inputs (`steps_per_epoch`
comes out to 5, not clamped to the `max(1, ...)` floor), small enough to run in
~2 minutes. Not chosen to be statistically meaningful, only to exercise the code path
correctly -- consistent with the smoke-test framing established in the first entry.

**How output was verified:** Real pytest run on the pod, log read directly, explicit
`EXIT_CODE:0` confirmed -- same discipline as the previous two entries, not the
piped-command mistake from earlier in the session.

**What was learned / should change next time:** When subsetting a DataFrame built by
concatenating multiple sources, `.head(n)` is only safe if the caller knows (or checks)
the concatenation order -- worth defaulting to an explicit per-group slice whenever a
test needs guaranteed representation from more than one source, rather than assuming
row order.

---

## 2026-08-17 — Real 6-run training schedule launched; found and fixed a genuine data gap

**What happened:** Student asked to move to the real training schedule. Wrote
`scripts/run_all_training.sh` (all 6 architecture/task combinations, sequential --
single GPU, train then immediately evaluate each pair, then trade-off analysis at the
end) and launched it detached on the pod (`nohup ... & disown`, redirected stdin/stdout/
stderr) so it survives an SSH disconnect during a multi-hour run.

While monitoring progress (filtering the noisy per-step progress-bar log lines with
grep rather than reading raw tail output), found that `resnet50/seven_class`'s
*training* succeeded but its *evaluation* step failed
(`EVAL_EXIT_CODE[resnet50/seven_class]=1`), while `resnet50/binary`'s evaluation right
after it succeeded -- pointing at something specific to the seven-class-only code path
(the ISIC2018 held-out test set evaluation, which only runs for that task). The actual
error: `tensorflow.python.framework.errors_impl.NotFoundError` on
`ISIC_0035068.jpg`. Diffed every image_id in `ISIC2018_Task3_Test_GroundTruth.csv`
against the actual files in `ISIC2018_Task3_Test_Images/` (`comm -23`/`comm -13`) rather
than assuming this was an extraction mistake: confirmed exactly one row
(`ISIC_0035068`) has no corresponding file, and zero extra files exist -- a genuine gap
in the official dataset release itself (both this project's own earlier archive
inspection on 2026-07-31 and the extraction on 2026-08-17 already confirmed the file
counts, 1,511 images matching 1,511 CSV rows on the surface, but a 1:1 identity check
is different from a count check and this is exactly why the count check alone missed
it).

Fixed `src.data.ham10000.load_isic2018_test()` to check each row's image file exists
and drop any that don't, with a printed count/list of what was dropped rather than a
silent fix -- this turns a training-time crash into an explicit, logged data-loading
decision. Added `test_load_isic2018_test_drops_rows_with_no_matching_image_file`
(synthetic CSV + one real, one deliberately-missing file) to `tests/test_ham10000.py`;
all 6 tests in that file pass locally. Pushed the fix to the pod while
`efficientnetb4/seven_class` was still mid-training (a separate already-running Python
process, unaffected -- the fix only needed to land before that combination's *own*
evaluation step, which happens as a fresh process later in the same shell script) and
manually re-ran the failed `resnet50/seven_class` evaluation to backfill the missing
result rather than waiting for a full second pass.

**Where uncertain / stuck:** Whether `ISIC_0035068` being missing reflects a licensing/
consent withdrawal from the official release (a plausible, common reason a specific
image gets pulled from a redistributed medical dataset after initial publication) or
some other cause is not established -- not investigated further since it doesn't change
the correct handling (drop the row, log it, move on), but worth a one-line mention in
Chapter 4 that one ISIC2018 test image was unavailable and excluded, for completeness.

**Assumptions made:** That checking `Path.exists()` per-row (1,511 rows) is cheap
enough to do unconditionally on every call rather than caching or pre-computing --
correct for this dataset's scale, would need revisiting only for a much larger test set.

**How output was verified:** The `comm` diff against real extracted files (not assumed
from the earlier count-only check), the local test run (6/6 pass), and the retry
evaluation launched on the pod after pushing the fix (result pending at the time of
writing this entry -- see whether a following entry confirms it passed).

**What was learned / should change next time:** A file-count match (1,511 images,
1,511 CSV rows) is not the same guarantee as an identity match (the *same* 1,511 IDs on
both sides) -- worth doing an explicit `comm`/set-difference check, not just a `wc -l`
comparison, whenever two independently-sourced file lists are expected to correspond
1:1, especially for a dataset (like this one) assembled from an official release that
wasn't produced by this project and so can't be assumed internally consistent.

---

## 2026-08-17 — Retry evaluations run concurrently with training; one genuine OOM

**What happened:** Manually retried the two seven-class evaluations that failed on the
pre-fix ISIC2018 bug, backfilling results without waiting for a full second pass of the
whole schedule. `resnet50/seven_class` retry, run while `efficientnetb4/binary` was
mid-training on the same GPU, succeeded (`results/resnet50_seven_class.json` written) --
TensorFlow logged benign "ran out of memory trying to allocate ... this is not a
failure" backoff warnings but completed correctly. `efficientnetb4/seven_class` retry,
run while `vgg16/seven_class` had just started training, failed for real:
`tensorflow.python.framework.errors_impl.ResourceExhaustedError: Out of memory while
trying to allocate 16.00MiB` during `model.predict`. Unlike the resnet50 case, this was
a hard crash, not a logged-but-handled warning -- EfficientNetB4 at its native 380x380
input needs meaningfully more VRAM than resnet50/vgg16 at 224x224, and stacking it
against a concurrently-training job pushed past the RTX 4090's 24GB.

**Where uncertain / stuck:** Not yet retried again -- deliberately waiting for the main
`run_all_training.sh` script to reach `ALL_DONE` (GPU fully idle) before retrying
`efficientnetb4/seven_class` a second time, rather than guessing at a "probably safe"
concurrent moment.

**Assumptions made:** That evaluation-only workloads (`model.predict`, no gradient
computation) are meaningfully lighter than training and therefore usually safe to run
concurrently with a training job -- true for resnet50 in this instance, false for
efficientnetb4. Revising this assumption going forward: only run manual/retry
evaluations concurrently with training for the two smaller (224x224) architectures, not
efficientnetb4, and prefer waiting for GPU idle time when in doubt.

**How output was verified:** Read the actual retry log tail directly (not inferred from
the background wrapper's exit code) for both retries -- the resnet50 success and the
efficientnetb4 failure were both confirmed from real log content, consistent with the
verification discipline established earlier in this session.

**What was learned / should change next time:** "It's just inference, should be safe to
run alongside training" is architecture-dependent, not a general rule -- EfficientNetB4's
larger input resolution makes its memory footprint meaningfully different from the other
two architectures even at inference time. Worth treating GPU-idle as the default
assumption for anything involving efficientnetb4, and treating concurrent runs for the
other two as an explicit, considered exception rather than a default habit.

---

## 2026-08-17 — First full 6-run schedule complete; results dashboard built

**What happened:** All 6 (architecture, task) combinations finished training and
evaluation (`TRAIN_EXIT_CODE`/`EVAL_EXIT_CODE` = 0 for all 6, confirmed by reading the
real log, not the background-wrapper status), including the `efficientnetb4/seven_class`
retry once the GPU was fully idle (`nvidia-smi`: 0% utilization, 1MiB used, no python
processes) -- it passed cleanly this time, no OOM. Regenerated
`results/trade_off_summary.json` after backfilling the last missing result, since the
first version (auto-generated by `run_all_training.sh` at the very end) predated that
file and was silently incomplete.

**Headline result:** resnet50 leads on both seven-class accuracy (0.658) and Grad-CAM
faithfulness (mean IoU 0.253) among the three architectures -- `src/trade_off.py`'s own
summary states this explicitly, including its standard caveat that this should be
reconfirmed once results are examined more closely, not treated as final from one run.
Full comparison table: resnet50/efficientnetb4/vgg16 seven-class accuracy 0.658/0.630/
0.640; ISIC2018 held-out accuracy 0.657/0.614/0.663 (vgg16 actually edges resnet50 on
the *external* test set, despite resnet50 leading on the internal validation split --
worth a closer look, not just quoting the internal-split ranking, in Chapter 4); binary
accuracy 0.817/0.743/0.815; binary skin-tone accuracy spread (DDI groups, smaller =
more consistent) 0.151/0.188/0.135 -- vgg16 is most consistent across skin tones,
efficientnetb4 least.

Student then asked to see confusion matrices, epochs, and loss/validation curves.
`src/train.py`'s own `main()` discards the Keras `History` object (only saves the
model) -- this data was never written to disk in structured form during the run, only
printed to stdout as progress-bar text. Reconstructed it by grepping each
`logs/train_{arch}_{task}.log` for the one line per epoch that includes `val_accuracy`
(the final progress-bar update before the next `Epoch N/M` line), parsed into
`results/epoch_metrics.json` (15 epochs -- 5 frozen + 10 fine-tune -- per run, all 6
runs present). Confusion matrices were already present in each `results/{arch}_{task}.json`
(`evaluate_predictions`'s own output) -- extracted into `results/confusion_summary.json`
alongside accuracy/kappa/ROC-AUC/ISIC-accuracy/faithfulness for convenience.

Built an interactive dashboard artifact (published, see this session's Artifact output
for the URL) from this real data: 6 small-multiple training-curve panels (train solid /
val dashed, per-architecture categorical colour, accuracy/loss toggle, shared y-axis per
metric for direct comparability, dotted marker at the frozen-to-fine-tune boundary), 6
confusion-matrix heatmaps (sequential blue ramp, hover tooltips), and the full summary
table. Followed the project's `dataviz` skill procedure: form before colour, the
documented reference palette's first three categorical slots (pre-validated all-pairs in
both light and dark mode per `palette.md`, since three architectures is exactly the
context that guidance covers), 2px lines with dashed/solid for train/val, hover
tooltips with textContent-safe rendering, and full light/dark theme support via the
skill's CSS custom-property pattern.

**Where uncertain / stuck:**
- The palette validator (`validate_palette.js`) requires Node, unavailable in this
  environment -- relied on `palette.md`'s own documented validation result for this
  exact three-slot combination rather than re-running it. This is the palette's stated
  reference case, not an extrapolation, so treated as sufficient, but flagging that no
  fresh validator run backs this specific artifact.
- The vgg16-leads-on-ISIC2018-despite-resnet50-leading-internally discrepancy is noted
  but not investigated further here -- worth a real look (e.g. whether resnet50 is
  mildly overfit to the internal validation split's specific images) before writing
  Chapter 4's conclusion.

**Assumptions made:** That epoch boundaries can be reliably recovered from
`Epoch N/M` marker lines even amid interleaved progress-bar carriage-return spam in the
same log file -- verified by checking parsed counts (15 per run, matching
`epochs_frozen=5 + epochs_finetune=10`) before trusting the reconstructed curves, not
assumed from the parsing logic alone.

**How output was verified:** Every result quoted above was read directly from
`results/*.json` (real evaluation output) or `results/epoch_metrics.json` (parsed from
real training logs, count-checked), not summarised from memory of earlier log-tailing.
The dashboard's data-generation script (`gen_dashboard.py`) reads the same JSON files
directly rather than hand-transcribing numbers into HTML, so the rendered chart and the
underlying `results/` files cannot drift apart.

**What was learned / should change next time:** `src/train.py` should be extended to
save the Keras training history (e.g. `results/history_{arch}_{task}.json`) directly,
rather than relying on grep-parsing stdout logs after the fact -- this worked here but
is fragile (depends on Keras's exact progress-bar text format) and wastes the first
6-run schedule's worth of "this should really just be saved structured data" realisation.
Worth fixing in `src/train.py` before the next full training pass (e.g. hyperparameter
tuning or a second seed), not treating this grep-based reconstruction as the permanent
approach.

---

## 2026-08-17 — DDI methodology revised: sequential pretrain-then-finetune, plus a
## zero-shot generalisation stress test (resnet50 pilot)

**What happened:** Student proposed a significant methodology change for the binary
task, informed by standard transfer-learning practice: (1) treat HAM10000-only as the
real baseline to get right first (not just move past it once *an* accuracy number
exists), (2) use DDI as a **zero-shot generalisation/bias stress test** -- run a
HAM10000-only model on DDI without any DDI training at all, since that alone is
diagnostic -- and (3) if DDI is incorporated into training, do it as a **fine-tuning
stage after HAM10000 pretraining**, on the matched binary label space, rather than
mixing HAM10000 and DDI together from the first training batch (the original "binary"
task's design, via `make_oversampled_binary_dataset`, still kept as a first comparison
arm rather than discarded -- confirmed with the student before implementing, along with
scoping the pilot to resnet50 only rather than all three architectures, to sanity-check
results before committing more GPU hours).

Implemented:
- `src/train.py`: new `"binary_ham_only"` task (HAM10000 relabelled to binary, DDI
  never touched) alongside the existing `"seven_class"` and `"binary"` (joint-mixed)
  tasks. Column-set matched to the joint binary corpus's HAM10000 half so both share
  the same downstream `make_dataset()`/`evaluate_run.py` code paths without special-
  casing. Also fixed a real bug caught while extending this file: the frozen-phase
  training `History` object was previously discarded entirely (its `fit()` return
  value was never captured) -- `train()` now concatenates both phases' histories with a
  `"phase"` marker per epoch, and `main()` saves the full combined history to
  `results/history_{architecture}_{task}.json` (was previously never saved at all,
  see the immediately preceding entry for how that gap was first discovered).
- `src/finetune_ddi.py` (new): loads an already-trained `binary_ham_only` model and
  continues training on DDI's own train split only (`ddi.split_ddi`'s train half) at a
  deliberately low learning rate (`1e-6`, well below either of `src.train`'s two
  phases), reasoning that this model has already converged on a related-but-distinct
  distribution and DDI is small (~557 training images) -- a higher rate risks
  catastrophic forgetting rather than gentle adaptation.
- `src/evaluate_ddi.py` (new): two modes. `zero_shot` evaluates a `binary_ham_only`
  model on DDI's **full** 656 images (none of it was used in training this model, so
  none needs holding back) plus the model's own HAM10000 validation split, in one
  result file for direct in-distribution vs. out-of-distribution comparison.
  `finetuned` evaluates the fine-tuned model on DDI's held-out validation split (the
  half `finetune_ddi.py` excluded from its own training) plus a HAM10000 retention
  check. Neither computes Grad-CAM faithfulness -- DDI has no ground-truth mask, per
  the existing constraint.
- `tests/test_train.py` (new file -- no prior tests existed for `src/train.py`,
  previously only exercised indirectly via `tests/test_integration_smoke.py`): 5 tests
  covering `prepare_data` for all three tasks, including one specifically asserting
  `binary_ham_only` never calls `ddi.load_metadata()` at all -- a real leakage risk if
  it did, since the whole point of the zero-shot stress test is that DDI is genuinely
  untouched until fine-tuning.

**Pilot results (resnet50, real training + evaluation on the pod, not projected):**

| Approach | DDI accuracy | DDI kappa | DDI ROC-AUC | DDI malignant recall | HAM10000 accuracy |
|---|---|---|---|---|---|
| Joint/mixed (original `binary` task) | -- (DDI mixed into training, never held out) | -- | -- | -- | 0.817 |
| Zero-shot (`binary_ham_only`, DDI unseen) | 0.753 | 0.159 | 0.654 | 0.158 | 0.815 |
| Sequential fine-tune (5 epochs on DDI) | 0.717 | 0.167 | 0.584 | 0.269 | 0.754 (retention) |

The zero-shot result is the headline finding: a HAM10000-only binary model that looks
strong internally (accuracy 0.815, malignant recall 0.799, ROC-AUC 0.900) collapses to
malignant recall 0.158 and ROC-AUC 0.654 on DDI -- a stark, *measured* demonstration of
the generalisation/bias gap the Social Issues section of the proposal already
anticipated qualitatively, now backed by a number. Skin-tone breakdown (zero-shot):
malignant recall FST_V_VI (darkest) 0.104, FST_I_II 0.163, FST_III_IV 0.189 -- worst on
the darkest-skin group, consistent with the anticipated bias direction, though each
group is only ~207-241 DDI images so this should be described as suggestive, not
conclusive, without a significance check.

Fine-tuning genuinely improved malignant recall (0.158 -> 0.269) but at a real cost:
HAM10000 accuracy dropped 6 points (0.815 -> 0.754) and DDI's own ROC-AUC *worsened*
(0.654 -> 0.584) -- 5 epochs at this learning rate seems to have shifted the decision
boundary toward catching more malignant cases without improving overall discrimination,
alongside a genuine forgetting trade-off. The darkest-skin-tone group remained worst
after fine-tuning too, but the fine-tuned DDI validation split is only ~30-36 images per
skin-tone group -- too small to treat this as more than a suggestive pattern.

**Where uncertain / stuck:**
- Only 5 fine-tuning epochs at `1e-6` were tried -- whether more epochs, a different
  learning rate, or partial re-freezing would trade off the HAM10000-retention loss
  against DDI improvement differently is unexplored. This pilot answers "does the
  sequential approach work at all," not "what's the best fine-tuning configuration."
- The joint/mixed approach's own DDI-specific performance was never measured directly
  (it was trained *on* DDI, mixed with HAM10000, so there's no clean "zero-shot on
  DDI" number for it to compare against these two new approaches on equal footing --
  only its overall binary accuracy/skin-tone-spread, already in `results/resnet50_binary.json`).
  A fully equal three-way comparison would need the joint model evaluated on DDI's
  *validation* split specifically (which it never trained on, since `ddi.split_ddi`
  keeps a val split even for the joint corpus) -- not done in this pilot, worth adding
  if this comparison is written up formally.
- Per the student's explicit scoping decision, this is a resnet50-only pilot --
  efficientnetb4 and vgg16 have not been run through this methodology at all yet.

**Assumptions made:** That DDI's `split_ddi` (stratified by skin-tone-group +
malignancy, seed=`config.RANDOM_SEED`) called identically in both `finetune_ddi.py` and
`evaluate_ddi.py`'s `finetuned` mode produces the *same* split both times -- true by
construction (same function, same default seed, same input DataFrame), verified by
reading both call sites rather than assumed, since a seed mismatch here would silently
leak fine-tuning data into the "held-out" evaluation.

**How output was verified:** All new `prepare_data` logic was tested locally first
(pure pandas, no TensorFlow) using the existing `tests/test_ddi.py` CSV-fixture
pattern -- an earlier draft of these tests wrongly stubbed `ddi.load_metadata()`
directly, which skipped its real decoding logic and caused two tests to fail on a
genuine `KeyError: 'skin_tone_group'`; fixed by switching to the established
write-CSV-and-monkeypatch-the-path pattern instead. All 5 new tests plus the full
44-test suite were then run for real on the pod (not locally -- the student explicitly
corrected this mid-session: local execution was for the Streamlit demo specifically,
not this training/evaluation work) before any training was started. The
`binary_ham_only` training run, zero-shot evaluation, fine-tuning run, and fine-tuned
evaluation were all executed for real on the pod and their result files read directly,
not projected.

**What was learned / should change next time:** The student's proposed methodology is a
materially stronger research design than the original joint-mixed-only approach --
worth remembering for future dataset-combination decisions on this project (and
elsewhere): sequencing "get a solid single-source baseline, measure zero-shot transfer,
*then* decide how to adapt" surfaces information (the stark zero-shot collapse) that
joint training from scratch would have hidden entirely, since a jointly-trained model
never produces a "how does the HAM10000-only version generalise" number at all. Also
re-confirmed the local-vs-pod execution boundary explicitly established earlier in this
session: local `.venv` is for the Streamlit demo only; all training, evaluation, and
test execution belongs on the pod.

---

## 2026-08-18 — Pod restarted on a new instance; DDI pilot extended to all three architectures

**What happened:** Session resumed after a multi-day pause (student had stopped the
Runpod pod to avoid idle billing, per the cost flag raised before pausing). New pod
instance (new public IP/port, new hostname) but the **same `RUNPOD_VOLUME_ID`** as
before -- `/workspace` (code, venv, all 8 previously-trained models, all results, the
full extracted HAM10000+DDI data) survived completely intact on the persistent volume.
Only `/etc/environment`'s `PROJECT_DATA_DIR` needed re-adding, since that file lives on
the ephemeral pod instance, not the persistent volume -- confirmed this distinction
explicitly this time rather than re-discovering it by trial and error as happened with
the first pod.

Re-connected using the same local SSH keypair (still valid, DDI methodology code
already present on the volume from before) and extended the resnet50-only DDI pilot to
`efficientnetb4` and `vgg16` via a new `scripts/run_ddi_pilot.sh` (train
`binary_ham_only` -> zero-shot DDI eval -> fine-tune on DDI -> fine-tuned eval,
sequential per architecture, matching the individual commands already run by hand for
resnet50). All 8 steps (4 per architecture) succeeded for real on the pod.

**Results, extending the table from the previous entry:**

| Architecture | HAM-only malignant recall | DDI zero-shot recall | DDI zero-shot ROC-AUC | Fine-tuned DDI recall | HAM retention accuracy |
|---|---|---|---|---|---|
| resnet50 | 0.799 | 0.158 | 0.654 | 0.269 (+0.111) | 0.754 (-0.061) |
| efficientnetb4 | 0.871 | 0.480 | 0.592 | 0.269 (-0.211) | 0.775 (+0.031) |
| vgg16 | 0.799 | 0.123 | 0.598 | 0.154 (+0.031) | 0.800 (-0.030) |

Two findings beyond what the resnet50-only pilot could show:
1. **The zero-shot generalisation collapse is now a cross-architecture finding, not a
   resnet50-specific result** -- every architecture's malignant recall drops sharply
   from its own HAM10000 validation performance to DDI. efficientnetb4's zero-shot
   skin-tone breakdown is the cleanest confirmation yet of the anticipated bias
   direction: recall declines *monotonically* from lightest to darkest skin (FST I-II
   0.714 -> III-IV 0.446 -> V-VI 0.292), rather than just being lowest for the darkest
   group among otherwise-noisy numbers (as resnet50's and vgg16's zero-shot results
   were).
2. **Fine-tuning is not a uniformly safe default** -- resnet50 and vgg16 both improved
   DDI malignant recall after fine-tuning; efficientnetb4's *dropped* (0.480 -> 0.269),
   even though its HAM10000 retention accuracy improved slightly (unlike the other two,
   which both lost some retention accuracy). efficientnetb4 was already the best
   zero-shot generaliser of the three, and the sequential fine-tuning approach partially
   undid that advantage rather than building on it. This is a genuine negative result,
   reported as such rather than smoothed into "fine-tuning helps."

**Where uncertain / stuck:**
- The fine-tuned skin-tone-stratified breakdown remains noisy across all three
  architectures (~30-36 DDI validation images per skin-tone group after the held-out
  split) -- efficientnetb4's fine-tuned pattern even reverses direction (FST_V_VI
  recall 0.429, higher than FST_I_II's 0.250), which reads as sampling noise from a
  small split rather than a real effect, and should not be reported as a finding on its
  own without a larger validation set or repeated runs.
- Why efficientnetb4 responds differently to fine-tuning than the other two
  architectures (a real regression rather than an improvement) is not investigated
  further here -- a plausible hypothesis is that efficientnetb4's zero-shot behaviour
  was already closer to a local optimum for DDI's distribution specifically, and 5
  epochs at `1e-6` shifted it away from that rather than toward it, but this is
  speculation, not something the pilot's data confirms on its own.
- Same single-seed, single-configuration caveat as the resnet50-only pilot: this
  answers "does the pattern hold across architectures," not "what is the best
  fine-tuning configuration per architecture."

**Assumptions made:** None beyond what the resnet50-only pilot already assumed
(`ddi.split_ddi`'s seed consistency between `finetune_ddi.py` and `evaluate_ddi.py`).

**How output was verified:** All 8 steps' real exit codes were checked from the actual
log content (`grep`-extracted `*_EXIT[...]` lines), not inferred from the background
wrapper's status -- consistent with the verification discipline established earlier in
this session. Result JSONs were pulled from the pod and read directly for every number
quoted above, not projected from the resnet50-only pattern.

**What was learned / should change next time:** Extending a single-architecture pilot
to the full set can genuinely change the conclusion, not just add confirming data
points -- efficientnetb4's fine-tuning regression would have been missed entirely if
the resnet50 pilot's finding ("fine-tuning helps, at a retention cost") had been
generalised to all three architectures without actually running them. Worth treating
any single-architecture pilot result as provisional specifically on "does this apply to
the other architectures too," not just on hyperparameter choices, before writing it up
as a general conclusion.

---

## 2026-08-18 — Why efficientnetb4 behaves differently: a calibration-shift explanation

**What happened:** Student asked to investigate why efficientnetb4's DDI malignant
recall *dropped* after fine-tuning (0.480 -> 0.269) while resnet50's and vgg16's both
*improved* -- the anomaly flagged in the previous entry. Investigated using only the
result JSONs already pulled from the pod (no new training or pod access needed): for
each architecture, computed each model's overall "predicted malignant" rate on DDI
(from the existing confusion matrices: `(FP + TP) / total`) in both the zero-shot and
fine-tuned states, since recall alone conflates *how discriminative* a model is with
*how trigger-happy* it is.

**Finding:** every architecture's predict-malignant rate moved toward DDI's true
malignant base rate (171/656 = 26.1%) after fine-tuning -- but the three started on
opposite sides of it:

| Architecture | P(predict malignant) zero-shot | P(predict malignant) fine-tuned | Direction |
|---|---|---|---|
| resnet50 | 6.9% | 16.2% | up, toward 26% |
| vgg16 | 7.2% | 11.1% | up, toward 26% |
| efficientnetb4 | 36.4% | 21.2% | down, toward 26% |

resnet50 and vgg16 were both *under*-predicting malignant zero-shot (calling it only
~7% of the time against a true 26% rate) -- fine-tuning corrected this upward, which
mechanically raises recall. efficientnetb4 was doing the opposite: *over*-predicting
malignant zero-shot at 36.4%, nearly 1.4x the true rate. This is also consistent with a
detail already in the previous entry's table that wasn't fully explained there:
efficientnetb4 had the *highest* zero-shot recall (0.480) of the three architectures
but the *lowest* zero-shot kappa (0.410, vs. resnet50's 0.505 and vgg16's 0.532) --
exactly the signature of recall inflated by a low decision threshold rather than by
genuinely better discrimination. Fine-tuning corrected efficientnetb4's rate back down
toward the true base rate too, which necessarily lowers its recall even though the
model is arguably becoming *better* calibrated, not worse.

**Revised interpretation (corrects the previous entry's framing, not just adds to
it):** "efficientnetb4 responds badly to DDI fine-tuning" is technically true but
misleading on its own. The more accurate statement: fine-tuning nudges every
architecture's decision threshold toward DDI's true class balance regardless of which
direction that correction runs, and efficientnetb4 happened to be over-shooting rather
than under-shooting beforehand. This reframes the "architecture-dependent trade-off"
finding from the previous entry -- it's not that fine-tuning is unpredictable
per-architecture in some deep way, it's that each architecture's *starting*
calibration on DDI (itself likely shaped by domain shift interacting differently with
each architecture's frozen BatchNorm statistics, still calibrated to HAM10000, not
DDI -- unfreeze_top_layers keeps BatchNorm frozen even within the unfrozen layer range,
so DDI fine-tuning cannot adapt those statistics at all) determines which direction the
correction goes, and therefore whether recall goes up or down.

**Where uncertain / stuck:** Why efficientnetb4's zero-shot calibration on DDI is so
much more skewed toward over-predicting malignant than resnet50's or vgg16's is not
established here -- a plausible contributing factor (frozen BatchNorm statistics
calibrated to HAM10000's distribution meeting DDI's different imaging modality/skin-tone
distribution at inference time, with efficientnetb4's higher native input resolution
[380x380 vs. 224x224] potentially making this domain-shift interaction stronger) is
noted as a hypothesis, not confirmed -- would need either inspecting per-layer
activation statistics on DDI vs. HAM10000 inputs, or an ablation unfreezing BatchNorm
during fine-tuning, neither of which was done here.

**Assumptions made:** That `(FP + TP) / total` from each architecture's existing 2x2
confusion matrix is a valid proxy for "how often does this model predict malignant" --
true by definition of a binary confusion matrix's structure (column sum for the
predicted-malignant column), not an estimated or fitted quantity.

**How output was verified:** Computed directly from the confusion matrices already in
`results/*_binary_ham_only_zero_shot_ddi.json` and `results/*_binary_ddi_finetuned.json`
(committed in the previous entry) -- no new model runs, no new pod access, arithmetic
re-derived from numbers already verified as real in the prior two entries.

**What was learned / should change next time:** Recall (or accuracy) alone can make a
miscalibrated-but-lucky model look like the best generaliser -- checking the predicted-
positive *rate* against the true base rate, not just recall/precision individually,
would have surfaced efficientnetb4's over-triggering immediately in the original pilot
entry rather than needing a follow-up investigation. Worth including "predicted-positive
rate vs. true base rate" as a standard diagnostic alongside recall/precision/kappa for
any future binary-classification comparison on this project, not just this one.

---

## 2026-08-18 (continued) — Joint/mixed model evaluated on the same DDI held-out split:
## it wins, clearly, on every metric

**What happened:** Student asked what would produce a better result. Rather than
guessing at fine-tuning hyperparameters, closed the comparison gap flagged as missing
in both prior pilot entries: the joint/mixed model (`src.train`, task=`binary`) had
never been scored on DDI's *held-out* validation split specifically -- only on its
overall HAM10000+DDI-mixed validation figure, which isn't directly comparable to the
zero-shot/fine-tuned modes' DDI-only numbers. Added `evaluate_joint_on_ddi_val()` (new
`--mode joint` in `src/evaluate_ddi.py`): filters `prepare_data("binary")`'s combined
validation split down to its `source == "ddi"` rows, which are guaranteed to be
*exactly* the same `ddi_val` rows the other two modes use (same `ddi.split_ddi` call,
same seed, threaded through `build_binary_corpus`) -- not a coincidentally-similar
independent split. Ran for all three architectures on the pod; all three succeeded.

**Result -- the joint/mixed approach wins outright, not narrowly:**

| Architecture | Approach | Accuracy | Kappa | ROC-AUC | Malignant recall | Malignant precision |
|---|---|---|---|---|---|---|
| resnet50 | joint | 0.747 | **0.356** | **0.764** | 0.538 | 0.519 |
| resnet50 | zero-shot | 0.753 | 0.159 | 0.654 | 0.158 | 0.600 |
| resnet50 | fine-tuned | 0.717 | 0.167 | 0.584 | 0.269 | 0.438 |
| efficientnetb4 | joint | 0.657 | **0.211** | **0.698** | 0.538 | 0.389 |
| efficientnetb4 | zero-shot | 0.625 | 0.138 | 0.592 | 0.480 | 0.343 |
| efficientnetb4 | fine-tuned | 0.667 | 0.083 | 0.615 | 0.269 | 0.333 |
| vgg16 | joint | 0.707 | **0.304** | **0.733** | 0.577 | 0.455 |
| vgg16 | zero-shot | 0.732 | 0.090 | 0.598 | 0.123 | 0.447 |
| vgg16 | fine-tuned | 0.707 | 0.071 | 0.671 | 0.154 | 0.364 |

The joint model leads on kappa and ROC-AUC by a wide margin for all three
architectures (roughly 2-2.5x the kappa of either alternative in every case), and
achieves a genuinely balanced malignant recall/precision (both in the 0.39-0.58 range)
rather than the lopsided calibration failures the zero-shot and fine-tuned modes both
show. This makes mechanistic sense: the joint model saw DDI images throughout training
(oversampled to a 30% per-batch share via `make_oversampled_binary_dataset`), so it
never needed a post-hoc calibration correction the way a HAM10000-only model does --
it learned DDI's distribution directly, from the start, alongside HAM10000's.

**This materially revises the project's practical recommendation, not just an academic
footnote.** The zero-shot and sequential-fine-tune experiments remain valuable and
correct as *diagnostic* tools -- they are what revealed and explained the
generalisation gap and its calibration mechanics (previous two entries) -- but as a
recommendation for which trained model to actually use/report as the project's best
binary-task result, the answer is now clearly the original joint-mixed models
(`models/{architecture}_binary.keras`, already trained, no new training needed), not
the sequential-fine-tuned ones. The earlier framing across recent entries and
`spec/specification.md` ("fine-tuning is a genuine trade-off, architecture-dependent")
was accurate as far as it went, but incomplete -- it never established fine-tuning was
*better than the pre-existing joint approach at all*, only that it changed the
zero-shot model's calibration in different directions per architecture.

**Where uncertain / stuck:** Why joint training outperforms sequential fine-tuning so
clearly is not fully pinned down here -- plausible contributing factors (joint
training's BatchNorm statistics are shaped by DDI throughout, unlike fine-tuning's
frozen BN; joint training sees many more effective DDI exposures over 15 epochs at
30%-per-batch oversampling vs. fine-tuning's single pass over ddi_train for 5 epochs at
a very low learning rate) are stated as hypotheses, not confirmed by a controlled
ablation.

**Assumptions made:** That filtering the joint model's own validation split by
`source == "ddi"` is equivalent to `ddi.split_ddi`'s independent output -- verified by
tracing the actual code path (`prepare_data("binary")` -> `build_binary_corpus(val_df,
ddi_val)` where `ddi_val` comes from the identical `ddi.split_ddi(ddi_df)` call with
the same default seed) rather than assumed from the two numbers happening to look
similar.

**How output was verified:** Real evaluation run on the pod (all three `JOINT_EVAL_EXIT`
values confirmed `0` from actual log content), result JSONs pulled and read directly,
not estimated from the joint model's previously-known combined-validation numbers.

**What was learned / should change next time:** A "we tried an alternative, here's how
it compares to a baseline we assumed but never actually measured" gap can sit
unnoticed across multiple pilot entries even while doing otherwise-careful analysis
(the calibration-shift investigation, immediately before this entry, was real and
correct, but was answering "why does fine-tuning behave differently per architecture,"
not "is fine-tuning even the right thing to be doing" -- the two are different
questions, and only measuring the second one revealed the joint model was already
better all along). Worth explicitly asking "what haven't I actually measured yet,
independent of what I've explained so far" before treating a mechanistic explanation as
confirmation that the thing being explained was the right thing to focus on.

---

## 2026-08-28 (continued) -- Tier 4: SHAP run for real for the first time; larger
## Grad-CAM faithfulness sample reveals the binary model's n=30 estimate was optimistic

**What happened:** Branched `tier4-shap-and-larger-faithfulness-sample` from `main`.
Closed the two gaps flagged repeatedly since 2026-07-31: SHAP (`src.xai.shap_explain`)
had only ever been wiring-tested against a stub predict function, never a real trained
model; and the project's only Grad-CAM faithfulness numbers came from a 30-image
sample, small enough to be a noisy point estimate.

**SHAP against a real model, for the first time.** Wrote `src/run_shap_explain.py`:
for a sample of HAM10000 validation images, builds a real SHAP explainer around
resnet50's seven-class model, runs it, and computes IoU/Dice faithfulness for both
Grad-CAM and SHAP against the same ground-truth masks (via the same
`compute_overlap` function, so the two are directly comparable). Deliberately scoped
to the seven-class task only in this first run -- the binary task's single-sigmoid-
output head (`Dense(1, ...)`) doesn't obviously match `build_explainer`'s
`output_names=["benign","malignant"]` (2 names for a 1-column `predict()` output), an
untested combination flagged as an open item rather than risked here.

Tested cautiously: ran 3 images first (succeeded, ~77s, correctly-shaped SHAP output
`(3, 224, 224, 3, 7)`), then scaled to 15 images (succeeded, ~a few minutes). Saved
overlay images (Grad-CAM and SHAP, per sample) to `results/xai_overlays/resnet50/` and
a faithfulness summary to `results/shap_faithfulness_resnet50.json`. On this 15-image
sample: Grad-CAM mean_iou=0.293/mean_dice=0.423, SHAP mean_iou=0.259/mean_dice=0.393 --
Grad-CAM slightly more faithful than SHAP on average, but close enough, and n=15 small
enough, that this should be read as "the two methods are broadly comparable in
faithfulness here," not "Grad-CAM is proven better." (Per-sample accuracy on this
random 15-image subset was 8/15=53%, well below resnet50's overall 65.8% seven-class
accuracy -- small-sample noise, not a new finding, consistent with every other
small-sample check this session.)

**Larger Grad-CAM faithfulness sample (30 -> 150) for the two recommended-best
resnet50 models.** Re-ran `src.evaluate_run --faithfulness-samples 150` for
`resnet50_seven_class` and `resnet50_binary` (joint), writing to new
`*_faithfulness150.json` files rather than overwriting the committed n=30 results.

| Model | n=30 mean IoU | n=150 mean IoU | n=30 mean Dice | n=150 mean Dice |
|---|---|---|---|---|
| seven_class | 0.253 | 0.294 | 0.372 | 0.422 |
| binary (joint) | 0.215 | 0.157 | 0.304 | 0.226 |

The seven-class number moved modestly (0.253->0.294) and lands close to this same
session's independent 15-image SHAP-run Grad-CAM average (0.293) -- good cross-check
agreement from two separately-sampled runs. **The binary model's number moved
substantially and in the opposite direction** (0.215->0.157, a 27% relative drop) --
the original n=30 estimate, quoted in `results/confusion_summary.json` and used in
earlier trade-off framing, was an optimistic estimate that a larger sample corrects
downward. This should be treated as the more reliable number going forward for the
binary model's Grad-CAM faithfulness specifically.

**Where uncertain / stuck:**
- SHAP was run for resnet50 only, and only on the seven-class task -- efficientnetb4,
  vgg16, and the binary task (once the output_names/single-sigmoid-output mismatch is
  resolved or confirmed harmless) remain untested.
- 150 is still not the full validation set (1490 seven-class images) -- chosen as a
  5x increase over the original 30 at reasonable cost, not as a claim of exhaustive
  coverage. The binary model's corrected number in particular would benefit from an
  even larger sample or the full validation set if this number needs to go into a
  final dissertation table.
- Did not investigate *why* the binary model's faithfulness estimate was more volatile
  across sample sizes than the seven-class model's -- plausibly because the binary
  task's HAM10000-sourced validation rows are a smaller pool to sample from than the
  seven-class task's, making a 30-image draw proportionally less representative, but
  not confirmed.

**Assumptions made:** That normalising SHAP's per-class attribution map the same way
Grad-CAM's heatmap is normalised (`abs(values).sum(axis=-1)` then divide by max) is a
fair basis for comparison via `compute_overlap`'s fixed 0.5 threshold -- a reasonable,
documented choice, but not the only valid way to threshold a SHAP attribution map; a
different normalisation could shift the SHAP faithfulness numbers above.

**How output was verified:** Both experiments' real exit codes and console output were
checked directly (SHAP's printed per-image IoU values, the `EXIT1=0`/`EXIT2=0` lines
for the faithfulness150 runs) before pulling and trusting any result file. The
seven-class n=150 vs. SHAP-run n=15 cross-check (0.294 vs. 0.293 Grad-CAM mean IoU from
two independently sampled, differently-sized runs) was noticed and used as informal
corroboration, not assumed to agree in advance.

**What was learned / should change next time:** The binary model's faithfulness
number is now the second metric this session (after the seven-class/DDI kappa
rankings) where a small sample (n=30) gave a materially different answer than a larger
one -- reinforcing that this project's `faithfulness_samples` default of 30
(`src/evaluate_run.py`) is too small to trust at face value for any final reported
number, not just for the DDI-related metrics already flagged. Worth revisiting the
default, or explicitly re-running the full six-model faithfulness check at a larger
sample size, before treating any of the six models' original `results/*.json`
faithfulness figures as final.

---

## 2026-08-28 (continued) -- Tier 4 completed: n=150 faithfulness re-run for all 6
## base models finds a systematic pattern, not architecture-specific noise

**What happened:** Extended the n=30->150 Grad-CAM faithfulness re-run (previous
entry, resnet50 only) to the remaining 4 base (architecture, task) combinations
(efficientnetb4/vgg16 x seven_class/binary), completing the full 6-model set on the
same `tier4-shap-and-larger-faithfulness-sample` branch. Re-used `src.evaluate_run
--faithfulness-samples 150`, writing to `*_faithfulness150.json` rather than
overwriting the committed n=30 results. All 4 runs completed with real exit code 0.

**Full 6-model comparison:**

| Model | n=30 IoU | n=150 IoU | Δ% | n=30 Dice | n=150 Dice | Δ% |
|---|---|---|---|---|---|---|
| resnet50_seven_class | 0.253 | 0.294 | +16.5% | 0.372 | 0.422 | +13.2% |
| resnet50_binary | 0.215 | 0.157 | -27.1% | 0.304 | 0.226 | -25.5% |
| efficientnetb4_seven_class | 0.239 | 0.259 | +8.3% | 0.360 | 0.379 | +5.3% |
| efficientnetb4_binary | 0.167 | 0.141 | -15.5% | 0.259 | 0.222 | -14.3% |
| vgg16_seven_class | 0.196 | 0.196 | +0.2% | 0.302 | 0.302 | +0.2% |
| vgg16_binary | 0.154 | 0.115 | -25.2% | 0.236 | 0.181 | -23.1% |

**This is now a clear, systematic pattern, not architecture-specific noise from the
previous entry's resnet50-only result.** Every seven-class model's n=30 estimate was
reasonably close to its n=150 value (vgg16 literally unchanged at 0.196; resnet50 and
efficientnetb4 moved modestly upward). Every binary model's n=30 estimate was a
substantial overestimate (-15% to -27%), consistently across all three architectures.
This points to something structural about how the binary task's faithfulness sample is
drawn (likely: the binary task's HAM10000-sourced validation pool available for the
faithfulness sample differs in size/composition from the seven-class task's, or the
binary decision surface's Grad-CAM behaviour is inherently more sample-sensitive) --
not investigated further here, but no longer explainable as one architecture's fluke.

**Good news buried in this correction:** `src.trade_off`'s "resnet50 leads on both
accuracy and faithfulness" summary is driven by the seven-class faithfulness ranking
specifically (`build_comparison_table` only reads `seven["faithfulness"]`, never the
binary task's). That ranking -- resnet50 (0.294) > efficientnetb4 (0.259) > vgg16
(0.196) -- is **identical in order** before and after this correction, despite the
absolute IoU values shifting. The headline finding this project has repeated since the
first full training run is, if anything, more robustly supported now than it was on
the original n=30 sample, not undermined by it.

**Where uncertain / stuck:** The systematic binary-vs-seven-class difference in
n=30-estimate reliability is observed but not explained -- would need inspecting the
actual pool of HAM10000-sourced rows each task's faithfulness check samples from (via
`run_faithfulness_check`'s `ham_rows` argument in `src/evaluate_run.py`) to confirm the
"smaller/differently-composed pool" hypothesis, not done here.

**Assumptions made:** None beyond the previous entry's.

**How output was verified:** All 4 real exit codes checked from log content
before pulling results; the full 6-model table computed directly from the pulled JSON
files (Python, not hand-calculated), including the percentage deltas.

**What was learned / should change next time:** A pattern that looked like it might be
one model's peculiarity (resnet50_binary's -27% correction, previous entry) turned out
to be a property shared by an entire task category across all three architectures once
checked completely -- worth remembering as a general instinct: before writing up a
correction as isolated to one model, check whether the same correction shows up
elsewhere with the same structure (same task, same direction), since that changes the
explanation from "this one run was unlucky" to "this category of measurement has a
systematic issue," which is a materially different and more useful finding for the
dissertation to report.

---

## 2026-08-28 (continued) -- SHAP extended to efficientnetb4 and vgg16: which XAI
## method is "more faithful" flips by architecture, but the sample is too small to
## trust that flip on its own

**What happened:** Ran `src.run_shap_explain` (unchanged from the resnet50 run) for
efficientnetb4 and vgg16, same n=15 sample size, seven-class task, same random seed
(so the 15 sampled HAM10000 images are identical across all three architectures --
a genuine same-sample comparison, not just same-count). Both completed with real exit
code 0. Continued on `tier4-shap-and-larger-faithfulness-sample` rather than a new
tier, matching the established pattern of extending an existing tier's experiment to
cover the remaining architectures.

**Result -- the "which XAI method is more faithful" answer flips by architecture:**

| Architecture | Grad-CAM mean IoU | SHAP mean IoU | Winner (this sample) |
|---|---|---|---|
| resnet50 | 0.293 | 0.259 | Grad-CAM |
| efficientnetb4 | 0.185 | 0.273 | SHAP |
| vgg16 | 0.200 | 0.218 | SHAP (narrowly) |

This is a genuinely interesting result for Objective 5's accuracy/interpretability
trade-off analysis -- it suggests XAI method faithfulness may itself be architecture-
dependent, not a fixed property of Grad-CAM vs. SHAP in general.

**Important caveat, checked before treating this as a real finding rather than noise:**
efficientnetb4's Grad-CAM figure on this 15-image sample (0.185) diverges substantially
from its more reliable n=150 estimate from earlier this session (0.259, same
Tier 4 branch) -- the same small-sample unreliability pattern already established
twice this session (the binary-task faithfulness correction; the DDI held-out
comparisons). resnet50's and vgg16's n=15 Grad-CAM figures (0.293, 0.200) do agree
reasonably well with their own n=150 estimates (0.294, 0.196) -- so this isn't a
uniform "all n=15 numbers are unreliable" problem, but efficientnetb4 specifically
drew an unrepresentative 15-image sample for Grad-CAM. Since the SHAP-vs-Grad-CAM
comparison uses the *same* 15 images for both methods within each architecture, the
comparison itself is internally fair (both methods see the same efficientnetb4
sample's difficulty), but the *absolute* Grad-CAM number for efficientnetb4 in this
table should not be read against the other two architectures' Grad-CAM numbers without
accounting for this.

**Honest conclusion:** the architecture-dependent flip is suggestive and worth noting
in the dissertation's interpretability discussion, but should be reported as "this
15-image sample suggests XAI method faithfulness may vary by architecture, not yet
confirmed at a reliable sample size" rather than as a settled finding -- consistent
with how every other small-sample result this session has needed the same caveat.

**Where uncertain / stuck:** A larger, per-architecture SHAP sample (matching the n=150
scale now used for Grad-CAM) would be needed to confirm or overturn this flip -- not
attempted here given SHAP's substantially higher per-image cost (Partition explainer
with `max_evals=500`) compared to Grad-CAM's single backward pass; scaling SHAP to
n=150 x 3 architectures would be a meaningfully larger and slower undertaking than
anything run so far this session.

**Assumptions made:** None beyond the previous SHAP entry's (SHAP attribution
normalisation for `compute_overlap` comparability).

**How output was verified:** Both runs' real exit codes checked from log content;
all 15-per-architecture per-image IoU values read directly from console output before
pulling and aggregating the result files.

**What was learned / should change next time:** Extending an experiment to more
architectures can reveal that an apparently clean per-architecture number (Grad-CAM's
n=15 IoU here) doesn't actually agree with that same architecture's own larger-sample
estimate from earlier in the session -- worth cross-checking every new small-sample
number against any existing larger-sample estimate for the same model before
interpreting a cross-architecture comparison built on it, not just trusting that
"same script, same sample size" implies comparable reliability across architectures.

---

## 2026-08-29 -- SHAP scaled to n=150 for all three architectures: the efficientnetb4
## flip from the n=15 run does not survive a larger sample; vgg16's does

**What happened:** Connected to a fresh Runpod GPU pod (RTX 4090) with this project's
`/workspace/project` already present from a prior session, and used it to answer the
open item from the previous entry -- scale SHAP from n=15 up to n=150 (matching the
sample size already used for Grad-CAM's `*_faithfulness150` results) for all three
architectures, since a small-sample flip in "which XAI method is more faithful" is not
trustworthy on its own. Wrote `scripts/run_shap_n150.sh`, a thin wrapper around the
existing `src.run_shap_explain --n-samples 150` that backs up any pre-existing result
file before running and only promotes the new output to a `_n150.json` name if the run
actually exits 0.

**Operational hiccup, fixed before trusting any result:** the first attempt (all three
architectures, launched via `nohup ... & disown` so the job would survive SSH
disconnects) failed immediately for every architecture with
`FileNotFoundError: data/raw/dataverse_files/HAM10000_metadata` -- the pod's
`/workspace/project/data` symlink to `/workspace/data` was simply missing (not broken,
absent entirely), apparently lost between this pod session and whatever created the
project checkout. Recreated it (`ln -s /workspace/data /workspace/project/data`) and
confirmed the actual data files were reachable through it. This also exposed a bug in
the first version of `run_shap_n150.sh`: the rename step ran unconditionally, so on a
crashed run it renamed the *pre-existing* n=15 result file to the `_n150.json` name --
silently mislabelling n=15 data as n=150. Caught this before committing anything by
noticing the "new" `_n150.json` files were byte-identical in size to the n=15 backups
and shared their original timestamps. Fixed the script to check the run's real exit
code before renaming, deleted the mislabelled files, and re-ran cleanly. All three
architectures completed with real exit code 0 on the second attempt (~21 minutes
wall-clock total, sequential, single GPU: resnet50 ~5 min, efficientnetb4 ~10 min,
vgg16 ~6 min).

**Result -- n=150 mean IoU, compared to the n=15 figures from the previous entry:**

| Architecture | n=15 Grad-CAM / SHAP (winner) | n=150 Grad-CAM / SHAP (winner) | n=150 Grad-CAM / SHAP Dice |
|---|---|---|---|
| resnet50 | 0.293 / 0.259 (Grad-CAM) | 0.294 / 0.215 (Grad-CAM) | 0.422 / 0.337 |
| efficientnetb4 | 0.185 / 0.273 (SHAP) | 0.259 / 0.255 (**tied**) | 0.379 / 0.385 |
| vgg16 | 0.200 / 0.218 (SHAP, narrow) | 0.195 / 0.255 (SHAP, clearer) | 0.301 / 0.384 |

To judge whether each gap is real rather than sampling noise, computed the paired
per-image (Grad-CAM IoU - SHAP IoU) difference, its standard deviation, and the
resulting 95% CI on the mean gap at n=150:

| Architecture | Mean gap (GC-SHAP) | 95% CI at n=150 | Crosses zero? |
|---|---|---|---|
| resnet50 | +0.079 | [0.046, 0.112] | No -- Grad-CAM genuinely more faithful |
| efficientnetb4 | +0.004 | [-0.024, 0.032] | Yes -- genuine tie, not unresolved noise |
| vgg16 | -0.059 | [-0.088, -0.030] | No -- SHAP genuinely more faithful |

**Honest conclusion, superseding the previous entry's "flips by architecture" framing:**
efficientnetb4's n=15 result (SHAP winning by 0.088 IoU) does not survive a 10x larger
sample -- at n=150 the two methods are statistically indistinguishable for this
architecture, consistent with the caveat already flagged in the previous entry (this
architecture's n=15 Grad-CAM figure was itself known to be an unrepresentative draw).
vgg16's flip, by contrast, is real: SHAP is more faithful than Grad-CAM here, and the
gap is larger and more confidently non-zero at n=150 than it appeared at n=15.
resnet50 continues to clearly favour Grad-CAM. The corrected finding for the
dissertation's interpretability discussion is therefore **not** "faithfulness ranking
flips by architecture" as a general phenomenon, but "vgg16 specifically favours SHAP;
resnet50 favours Grad-CAM; efficientnetb4 shows no reliable difference between the
two" -- a real but more limited and more defensible claim.

**Where uncertain / stuck:**
- Did not investigate *why* vgg16 in particular favours SHAP while the other two
  favour (or tie on) Grad-CAM -- plausibly related to VGG16's lack of skip
  connections changing how localised Grad-CAM's last-conv-layer activations are
  relative to SHAP's model-agnostic perturbation approach, but not confirmed.
- The n=150 run also wrote up to 150 new Grad-CAM/SHAP overlay image pairs per
  architecture to `results/xai_overlays/<arch>/` on the pod (rather than the 15
  committed previously). These were **not** pulled into the repo or committed --
  bringing in ~450-900 additional PNGs for a summary-statistics result would bloat
  the repo for little benefit over the 15 overlay pairs already committed as visual
  examples. Only the three `shap_faithfulness_<arch>_n150.json` summary files were
  kept. The overlay images remain on the pod if needed later, but that pod is
  ephemeral and they should be treated as not durably available.
- Asked whether to go further to n=500: computed that the 95% CI would only narrow by
  ~45% (SE scales as 1/sqrt(n)), and would not change any of the three conclusions
  above (resnet50 and vgg16 already exclude zero; efficientnetb4 would very likely
  still include zero). Not run, given the marginal benefit for roughly 3x the compute
  cost.

**Assumptions made:** None beyond the previous SHAP entries' (same normalisation,
same `compute_overlap` comparison). Assumed the pod's missing `data` symlink was an
environment artefact of this particular pod session rather than a project bug -- the
symlink target (`/workspace/data`) existed with the expected files, only the link
itself was absent, consistent with a fresh pod re-attaching a persistent volume
without recreating a symlink that lives outside it.

**How output was verified:** Real exit codes for all three architectures checked from
the log (`SHAP_N150_EXIT[<arch>]=0`) before pulling any result file. The mislabelling
bug in the first attempt was caught by comparing file sizes/timestamps against the
known n=15 backups before trusting the numbers, not after. Mean IoU/Dice and the
paired-difference confidence intervals were computed directly from the 150-row result
files (`.venv/bin/python3` on the pod), not eyeballed from console output.

**What was learned / should change next time:** A background job launched with
`nohup ... & disown` protects against SSH disconnects but not against environment
drift between pod sessions (the missing symlink) -- worth a quick sanity check (e.g.
`ls data/` or a 1-image dry run) before launching a long unattended job on a pod,
rather than discovering a missing dependency only after all three architectures have
already failed. Separately: a script step that renames/moves a result file should
always gate on the producing command's actual exit code, never on "does a file exist
at the expected path" alone -- the latter can silently succeed against a stale file
left over from a previous, unrelated run.

---

## 2026-08-29 (continued) -- SHAP pushed to n=500 for resnet50/vgg16;
## efficientnetb4 hits a container memory ceiling at n=500, resolved at n=400/n=200

**What happened:** Having quantified (previous entry) that n=500 would only narrow
the 95% CI by ~45% over n=150 without changing any conclusion, the student asked to
run it anyway for tighter reported numbers. Ran `scripts/run_shap_n500.sh resnet50
efficientnetb4 vgg16` (same pattern as the n=150 script: exit-code-gated rename, see
previous entry). resnet50 and vgg16 both completed cleanly (exit 0, ~14 min and ~12
min respectively). **efficientnetb4 was killed with exit code 137 (SIGKILL) right
after its SHAP explainer finished all 500 images**, with no Python traceback -- the
process was killed externally, not by an application-level error.

**Root cause, diagnosed rather than assumed:** `free -h` on the pod reported 124GB
total host RAM, which was misleading -- this pod runs in a container with its own
cgroup memory limit, read directly from `/sys/fs/cgroup/memory.max` as ~61,000,000,000
bytes (~57 GiB), far below the host figure. `src/run_shap_explain.py` loads the
entire N-image batch into one array and calls SHAP's `explain_images()` on the whole
batch at once (nothing is streamed or released per-image), so peak memory scales with
N x image_size^2. efficientnetb4 uses 380x380 input images versus resnet50/vgg16's
224x224 (roughly 2.9x the pixels), which is exactly why the other two architectures
sailed through n=500 (peaking well under the ceiling) while efficientnetb4's memory
climbed past 47GB by n=400-equivalent progress and past the ~57GiB ceiling by n=500,
getting killed at the exact point where the explainer finishes and assembles the full
values array -- the same stage, not a random point in the run.

**Resolution:** retried efficientnetb4 directly (bypassing `run_shap_n500.sh`, which
hardcodes `--n-samples 500`) at n=400 -- completed successfully (exit 0, ~19 min,
peaked around 47GB, i.e. comfortably under the ceiling this time). The student then
asked for an additional, more conservative n=200 run for extra safety margin even
though n=400 had already succeeded cleanly; also completed (exit 0, ~11 min). Both
kept as separate result files (`shap_faithfulness_efficientnetb4_n400.json` and
`_n200.json`) rather than treating one as replacing the other.

**Operational mistake, caught immediately:** attempted to relaunch efficientnetb4 at
n=400 by passing `--n-samples-override=400` as an extra positional argument to
`run_shap_n500.sh`, which does not exist as a script option -- the script's `for arch
in "$@"` loop simply treated it as a second (invalid) architecture name and re-ran
efficientnetb4 at the hardcoded 500 again, heading toward the same OOM. Caught this
by checking the actually-running process's command line before it got far, killed it
(`kill -9`), and re-launched correctly by invoking `src.run_shap_explain` directly
with `--n-samples 400` instead of going through the 500-only wrapper script.

**Result -- final n=500/n=400 comparison (headline numbers), with n=200 as an
efficientnetb4 cross-check:**

| Architecture | n | Grad-CAM IoU | SHAP IoU | Grad-CAM Dice | SHAP Dice | Gap (GC-SHAP) | 95% CI | Winner |
|---|---|---|---|---|---|---|---|---|
| resnet50 | 500 | 0.305 | 0.208 | 0.435 | 0.325 | +0.097 | ±0.019 | Grad-CAM |
| vgg16 | 500 | 0.202 | 0.255 | 0.309 | 0.387 | -0.053 | ±0.015 | SHAP |
| efficientnetb4 | 400 | 0.255 | 0.241 | 0.374 | 0.368 | +0.014 | ±0.017 | tied (CI crosses 0) |
| efficientnetb4 | 200 | 0.251 | 0.245 | 0.368 | 0.373 | +0.007 | ±0.025 | tied (CI crosses 0) |

**Honest conclusion:** every number here is consistent with, and tighter than, the
n=150 entry's conclusion -- nothing flipped. resnet50's Grad-CAM-favouring gap grew
slightly (+0.079 at n=150 -> +0.097 at n=500) and remains clearly non-zero. vgg16's
SHAP-favouring gap is essentially unchanged (-0.059 at n=150 -> -0.053 at n=500) and
remains clearly non-zero -- this is now corroborated across three independent sample
sizes (15, 150, 500) with the same direction and similar magnitude each time, about as
solid as this project's evidence gets. efficientnetb4's near-zero gap is now
corroborated across *four* sample sizes (150: +0.004, 200: +0.007, 400: +0.014, and
the earlier n=15 which was itself noisy) -- consistently near zero and always crossing
zero at 95% confidence, which is a genuine, repeatedly-confirmed tie rather than an
unresolved question that a larger sample might still resolve one way. The dissertation
finding stands as: **vgg16 favours SHAP, resnet50 favours Grad-CAM, efficientnetb4
shows no reliable difference between the two methods** -- now on the project's largest
XAI-faithfulness sample sizes to date.

**Where uncertain / stuck:**
- The ~57GiB cgroup limit is specific to this pod/session and not a property of the
  codebase -- a different pod (or the same pod after a restart) could have a
  different limit, so `--n-samples 500` should not be assumed safe for efficientnetb4
  on any arbitrary future pod without re-checking `/sys/fs/cgroup/memory.max` first.
- Did not modify `run_shap_explain.py` to batch/stream images instead of holding the
  whole array in memory, which would remove the ceiling entirely -- treated as
  out of scope for this session since the immediate goal (a large, reliable
  efficientnetb4 sample) was achievable by simply lowering n.
- As before, the new overlay images (up to 500 per architecture) were not pulled from
  the pod or committed -- only the four new `shap_faithfulness_*.json` summary files.

**Assumptions made:** That n=400's successful peak (~47GB) generalises as "safe" for
efficientnetb4 on this pod -- reasonable given it completed cleanly with several GB of
headroom below the ~57GiB ceiling, but not stress-tested at, say, n=450.

**How output was verified:** Real exit codes checked for every run (`SHAP_N500_EXIT`,
`SHAP_N400_EXIT`, `SHAP_N200_EXIT`) before trusting any result file, exactly as
established in the previous entry. The OOM diagnosis itself was verified by reading
`/sys/fs/cgroup/memory.max` directly rather than assumed from the `exit 137` code
alone (137 is consistent with SIGKILL but not proof of OOM specifically without the
corroborating cgroup limit and the RSS-climbing-to-the-limit pattern observed via
repeated `ps aux` checks during the run).

**What was learned / should change next time:** `free -h` inside a container can
silently report the *host's* memory rather than the container's actual cgroup-enforced
limit -- when diagnosing an unexplained SIGKILL (exit 137) on a containerised pod,
check `/sys/fs/cgroup/memory.max` directly rather than trusting `free`. Separately,
before reusing an existing wrapper script for a different parameter value, check
whether the script actually exposes that parameter (`run_shap_n500.sh` had no
`--n-samples` override) rather than assuming an extra CLI argument will be honoured --
passing an unrecognised flag as a positional argument silently got treated as a second
architecture name instead of erroring, which could have wasted another ~15-20 minutes
of GPU time if not caught by checking the live process command line immediately after
launch.

---

## 2026-08-29 (continued) -- Formal paired 95% CI on the final Grad-CAM-vs-SHAP
## faithfulness gap, computed directly from the committed n=500/n=400/n=200 result
## files (no new SHAP or Grad-CAM run)

**What happened:** The previous entry's summary table already quoted a gap and a
95% CI per architecture, but folded into the results table rather than shown with
the same explicit per-architecture methodology (mean gap, SD of paired differences,
SE, CI bounds) that the n=150 entry used earlier. Re-ran that exact method -- paired
per-image (Grad-CAM IoU - SHAP IoU) difference, SD of that difference, SE =
SD/sqrt(n), 95% CI = mean +/- 1.96 x SE -- as its own documented, independently
verifiable step, reading directly from the four already-committed result files
(`shap_faithfulness_resnet50_n500.json`, `_vgg16_n500.json`,
`_efficientnetb4_n400.json`, `_efficientnetb4_n200.json`). No SHAP or Grad-CAM run
was re-executed; this is purely a recomputation over existing per-image data.

**Result:**

| Architecture (n) | Mean gap (GC-SHAP) | SD of paired diff | SE | 95% CI | Excludes zero? |
|---|---|---|---|---|---|
| resnet50 (n=500) | +0.0967 | 0.2124 | 0.0095 | [0.0781, 0.1153] | Yes -- Grad-CAM more faithful |
| vgg16 (n=500) | -0.0527 | 0.1664 | 0.0074 | [-0.0673, -0.0381] | Yes -- SHAP more faithful |
| efficientnetb4 (n=400) | +0.0142 | 0.1776 | 0.0089 | [-0.0032, 0.0316] | No -- tied |
| efficientnetb4 (n=200) | +0.0067 | 0.1788 | 0.0126 | [-0.0181, 0.0315] | No -- tied |

These figures match the previous entry's rounded summary-table gap/CI values
(resnet50 +0.097 +/-0.019, vgg16 -0.053 +/-0.015, efficientnetb4 n400 +0.014 +/-0.017,
n200 +0.007 +/-0.025) to the precision that entry reported -- this pass adds the full
SD/SE breakdown and the explicit lower/upper bounds rather than just the +/- margin,
and independently reconfirms none of the four conclusions changed.

**Honest conclusion:** unchanged from the previous entry, now with the full interval
math shown rather than summarised. resnet50 and vgg16 both have 95% CIs that clearly
exclude zero in opposite directions (Grad-CAM and SHAP respectively); efficientnetb4's
CI includes zero at both n=400 and n=200, consistent with every other sample size
tried for this architecture (n=15, 150, 200, 400) -- a repeatedly-confirmed tie, not
an artifact of any one run.

**Where uncertain / stuck:** None -- this was a recomputation over already-verified,
already-committed data, not a new experiment.

**Assumptions made:** None beyond the paired-difference CI method already established
and used without objection in the n=150 entry (95% CI via the normal approximation,
mean +/- 1.96 x SE, appropriate here given each n is >= 150).

**How output was verified:** Computed directly from the four committed JSON result
files via `.venv/bin/python3` on the pod (not eyeballed, not re-derived from the
earlier summary table). Cross-checked against the previous entry's rounded
gap/CI figures before writing this entry -- they agreed, which is expected since
both draw on the same underlying per-image data, but confirming that agreement
before writing "unchanged" here rather than assuming it.

**What was learned / should change next time:** When a result is first reported
folded into a larger summary table (as the gap/CI figures were in the previous
entry), it's worth a dedicated recomputation pass with the full intermediate
values (SD, SE, exact bounds) shown on their own before treating the number as
final for the dissertation -- catches transcription or rounding errors between the
raw computation and the table, and gives a self-contained, independently checkable
record rather than one only verifiable by re-deriving it from a denser table.
