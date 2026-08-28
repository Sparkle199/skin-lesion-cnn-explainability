# Specification

Source: `project_desc.txt` (project proposal, including the DDI addition). This
specification does not introduce anything beyond what the proposal states; it restates
it in the WHAT/WHY/CONSTRAINTS/RISKS/ACCEPTANCE-CRITERIA form required by the governance
framework (Appendix B.1).

## Goal (one sentence)
Train and compare three CNN architectures on two tasks — the primary seven-class
classification of HAM10000 lesions, and a secondary binary malignant/benign
classification on a combined HAM10000+DDI corpus — and evaluate the trade-off between
classification performance and interpretability (Grad-CAM, SHAP) for each.

## WHAT
- Implement three CNN classifiers: ResNet-50, EfficientNetB4, VGG-16.
- Use transfer learning from ImageNet-pretrained weights, with progressive/layer-by-layer
  unfreezing for fine-tuning.
- **Primary task (seven-class):** train and evaluate each architecture on HAM10000
  (10,015 images / 7,470 unique lesions, 7 diagnostic classes), splitting train/validation
  at the lesion level to avoid leakage from lesions photographed more than once, and
  evaluating additionally on the official ISIC2018 Task 3 held-out test set (1,511
  images, lesion-disjoint from HAM10000's training set).
- **Secondary task (binary malignant/benign):** HAM10000 relabelled to malignant/benign
  from its existing seven-class diagnosis (malignant = {akiec, bcc, mel}, benign = {bkl,
  df, nv, vasc}) combined with DDI (656 images, each already carrying a native
  malignant/benign label, used in full rather than mapped into the seven-class taxonomy
  — DDI's 78 specific diagnoses do not map cleanly onto HAM10000's 7 classes). Revised
  2026-08-17 (student-proposed, standard transfer-learning practice) into three
  comparison approaches for how DDI is incorporated, rather than one:
  1. **Joint/mixed** (original design): DDI mixed into training from the first batch
     via oversampled batching (see Imbalance handling below). **Confirmed the
     best-performing of the three approaches on DDI's held-out validation split, by a
     wide margin on every metric, across all three architectures (2026-08-18 — see
     journal).** This is now the project's recommended binary-task model; the two
     approaches below function as diagnostic tools that explain *why*, not as
     alternatives that outperform it.
  2. **Zero-shot generalisation stress test:** train a HAM10000-only binary baseline
     (task `binary_ham_only`), then evaluate it on DDI's full 656 images *without any
     DDI training at all* — this alone is diagnostic of how well the model generalises
     across imaging modality and skin tone, and is not something the joint-mixed
     approach can measure directly (its model never exists without DDI in its training
     data). Reveals a severe generalisation/calibration gap the joint model does not
     have to contend with, since it never has to correct a HAM10000-only calibration
     after the fact.
  3. **Sequential fine-tuning:** take the same HAM10000-only baseline and fine-tune it
     further on DDI's own train split only, at a low learning rate, rather than mixing
     both sources from scratch — evaluated on DDI's held-out validation split plus a
     HAM10000 retention check. **Does not outperform the joint/mixed approach on any
     metric for any architecture** (2026-08-18 finding) — useful for understanding the
     mechanics of DDI adaptation (a calibration-threshold correction, not a
     discrimination improvement — see journal, 2026-08-18 calibration-shift entry), not
     as a recommended training strategy in its own right.
  Piloted for resnet50 (2026-08-17), extended to all three architectures (2026-08-18):
  the zero-shot generalisation collapse holds across all three (malignant recall
  0.799–0.871 on HAM10000 down to 0.123–0.480 on DDI), and sequential fine-tuning is
  **not** uniformly beneficial even relative to zero-shot — it improved DDI recall for
  resnet50/vgg16 but *reduced* it for efficientnetb4 (explained as a calibration-
  threshold correction running in different directions per architecture, not
  architecture-specific fine-tuning fragility). Most importantly, **neither zero-shot
  nor fine-tuned ever beats the original joint/mixed model** on kappa or ROC-AUC, for
  any architecture, when evaluated on the identical DDI held-out split (joint kappa
  0.211–0.356 vs. 0.071–0.167 for the alternatives). Full comparison table in journal,
  2026-08-18 entries.
- Apply data augmentation (rotation, flipping, zoom, brightness) and class weighting to
  address class imbalance in HAM10000 (~67% of images in one class, confirmed: nv 6,705 /
  mel 1,113 / bkl 1,099 / bcc 514 / akiec 327 / vasc 142 / df 115) and the further
  source/size imbalance in the binary task's combined corpus (DDI's 656 images vs.
  HAM10000's 10,015).
- Apply Grad-CAM and SHAP to each trained model (both tasks) as post-hoc explainability
  methods.
- Evaluate and compare all three architectures on: accuracy, precision, recall, per-class
  F1, ROC-AUC, Cohen's Kappa, confusion matrices (primary task), and — because of DDI's
  skin-tone-group labels — binary-task performance stratified by skin tone (Fitzpatrick
  I-II / III-IV / V-VI, fairly evenly split at 208/241/207 images in DDI).
- Assess XAI faithfulness quantitatively for HAM10000-derived predictions, by computing
  IoU and Dice overlap between each thresholded Grad-CAM/SHAP attention map and the
  ground-truth lesion segmentation mask bundled with HAM10000 (10,015 masks, one per
  image, confirmed binary 0/255 single-channel PNGs matching each image's dimensions).
  DDI carries no equivalent mask, so its contribution to faithfulness remains a visual
  check only. Conduct a trade-off analysis between accuracy and interpretability across
  the three architectures.

## WHY
This is an experimental/artefact project (Artefact type: Experiment) testing whether CNN
architecture choice affects the trade-off between multi-class classification accuracy
and explainability, using skin cancer diagnosis as the applied test case. DDI was added
after the original proposal specifically to reduce the skin-tone homogeneity of HAM10000
(sourced predominantly from fair-skinned patients in Europe/Australia) and to make the
resulting bias measurable, via the binary task, rather than only acknowledged. The
project remains explicitly non-clinical — no diagnostic tool is being produced.

The 2026-08-17 revision to three DDI-incorporation approaches exists because the
original joint-mixed design could never answer "how biased is a model that has never
seen DDI at all?" — a jointly-trained model has no HAM10000-only counterpart to compare
against. The zero-shot result — malignant recall collapsing from each architecture's own
HAM10000 validation performance (0.799–0.871) to 0.123–0.480 on DDI, confirmed across
all three architectures (2026-08-18) — makes the Social Issues section's generalisation
concern a measured finding rather than a qualitative one. The fine-tuning result is more
nuanced than "partial recovery at a retention cost": it improved DDI recall for
resnet50/vgg16 but reduced it for efficientnetb4 (the best zero-shot generaliser of the
three) — explained by a calibration-threshold shift toward DDI's true class balance
running in different directions per architecture, not a discrimination improvement (see
journal, 2026-08-18). **Neither diagnostic approach beats the original joint/mixed
model, however** — evaluated on the identical DDI held-out split, the joint model leads
on kappa and ROC-AUC by a wide margin for all three architectures (2026-08-18). The
practical conclusion is therefore not "fine-tuning is a trade-off worth making," but
that training on HAM10000+DDI jointly from the start remains the best approach found so
far, while the zero-shot/fine-tuned experiments remain valuable for explaining *why* —
directly relevant to the "residual demographic bias" risk already logged below, which
joint training measurably reduces (via the joint model's far higher DDI-held-out kappa)
without eliminating it (DDI is still much smaller than HAM10000, so this is expected to
remain a partial mitigation, not a solved problem).

## CONSTRAINTS
- Datasets: HAM10000 and DDI only, each used strictly under its own licence.
  - HAM10000: CC BY-NC 4.0 (non-commercial, academic use, with attribution). Confirmed
    contents of the obtained release: 10,015 training images, matching lesion
    segmentation masks (`<image_id>_segmentation.png`, binary 0/255 single-channel,
    used as ground truth for the quantitative faithfulness check), `HAM10000_metadata`
    (lesion_id, image_id, dx, dx_type, age, sex, localization, dataset), plus the
    official ISIC2018 Task 3 held-out test set (1,511 images) and its ground truth CSV.
  - DDI: attribution to Stanford University required; **exact licence terms not yet
    independently verified — confirm from DDI's official release documentation before
    submission or before any redistribution of derived artefacts.**
- No new data collection, no patient contact, no participant recruitment.
- UK GDPR / Data Protection Act 2018 compliance; both datasets are already anonymised —
  no re-identification attempts.
- Light Touch Ethical Review required (proposal states a submission date of 10 July
  2026 — flagged previously as inconsistent with the project's actual working window;
  confirm the real target date with the supervisor/Project Coordinator).
- Frameworks/tools: Python, TensorFlow/Keras (Apache License 2.0), ImageNet pretrained
  weights, SHAP (MIT License), Grad-CAM libraries — all to be cited.
- Compute: Kaggle GPU infrastructure.
- No commercial use or exploitation of models, code, or findings at any point.

## RISKS
- **Class imbalance**: ~67% of HAM10000 falls into one class (nv); the binary task's
  combined corpus adds a second axis of imbalance (HAM10000 vs. DDI source/size).
- **Lesion-level data leakage**: HAM10000's 10,015 images cover only 7,470 unique
  lesions; an image-level (rather than lesion-level) split would let the same lesion
  appear in both train and validation/test, inflating reported performance. Mitigated by
  a lesion-level split for the primary task.
- **Taxonomy mismatch**: DDI's 78 diagnostic labels do not map 1:1 onto HAM10000's seven
  classes (e.g. verruca-vulgaris, epidermal-cyst, acrochordon, neurofibroma, lipoma have
  no seven-class counterpart). Resolved by keeping DDI out of the seven-class task
  entirely and using it only for the binary malignant/benign task, where its native label
  is used directly.
- **Residual demographic bias**: DDI reduces but does not eliminate HAM10000's skin-tone
  skew in the binary task, since DDI is much smaller; the seven-class task is unaffected
  by DDI and retains HAM10000's original skin-tone skew entirely. This must be reported
  explicitly, not presented as solved. Confirmed directly, not just anticipated, by the
  2026-08-17 resnet50 zero-shot pilot: HAM10000-only malignant recall 0.799 collapses to
  0.158 on DDI, worst on the darkest skin-tone group (FST_V_VI, recall 0.104).
- **Catastrophic forgetting / architecture-dependent fine-tuning outcome**: fine-tuning
  a HAM10000-pretrained model on DDI alone (sequential approach) risks trading HAM10000
  performance for DDI adaptation rather than improving both — and the two are not
  linked the same way across architectures. Confirmed across all three architectures
  (2026-08-18 pilot): resnet50 (DDI recall 0.158→0.269, HAM accuracy 0.815→0.754) and
  vgg16 (DDI recall 0.123→0.154, HAM accuracy 0.830→0.800) both traded HAM accuracy for
  DDI recall as expected; efficientnetb4 instead *lost* DDI recall after fine-tuning
  (0.480→0.269) while its HAM accuracy slightly improved (0.744→0.775) — the opposite
  trade-off. Must be reported per-architecture, not as one general "fine-tuning helps
  DDI at a HAM10000 cost" statement. **Superseded as a practical concern** by the
  2026-08-18 joint-model comparison: since the joint/mixed approach outperforms
  sequential fine-tuning outright on DDI's held-out split for every architecture, this
  risk is now primarily relevant as an explanation of *why* sequential fine-tuning
  underperforms, not as a trade-off the project needs to navigate when choosing which
  model to report as its best result.
- **Misclassification risk**: false negatives in malignant-vs-benign classification are
  the most consequential error type, even in a non-clinical experimental setting.
- **Licensing/attribution risk**: incorrect or missing attribution for HAM10000, DDI, or
  third-party frameworks would breach the stated legal/professional constraints.
- **Over-claiming risk**: presenting experimental findings as clinically validated, or
  presenting DDI's inclusion as having solved the bias problem, would breach the
  professional-conduct constraint in the proposal.

## SUCCESS / ACCEPTANCE CRITERIA
- All three models (ResNet-50, EfficientNetB4, VGG-16) trained and fine-tuned twice each:
  once on the lesion-split HAM10000 corpus (primary, seven-class), once on the combined
  HAM10000+DDI corpus (secondary, binary malignant/benign).
- Primary task metrics reported on both the internal lesion-level HAM10000 split and the
  independent ISIC2018 Task 3 held-out test set.
- Full metric suite (accuracy, precision, recall, per-class F1, ROC-AUC, Cohen's Kappa)
  and confusion matrices produced for the primary task; the same core metrics plus
  skin-tone-group-stratified results produced for the secondary task.
- Grad-CAM and SHAP outputs generated for each trained model (both tasks), with a
  quantitative IoU/Dice faithfulness score against HAM10000's lesion segmentation masks
  (HAM10000-derived predictions only) plus a visual faithfulness check for DDI-derived
  predictions, which have no equivalent ground-truth mask.
- A written trade-off analysis comparing accuracy against interpretability across the
  three architectures, concluding with a recommended model for this multi-class problem.
- All legal/ethical/professional constraints above observed and documented (licence
  attribution for both datasets, GDPR compliance, Light Touch Ethical Review, no
  over-claiming about bias mitigation or clinical validity).

## Task decomposition (ordered, per project lifecycle)
1. **Analysis** — literature review (CNN multi-class classification, transfer learning,
   class-imbalance methods, Grad-CAM/SHAP/LIME); review and justification of HAM10000
   (including its ISIC2018 held-out test set) and DDI.
2. **Design** — finalise architecture choices; design the lesion-level split for the
   primary task; design the HAM10000-to-binary relabelling and combined corpus for the
   secondary task; design the augmentation/class-weighting pipeline for both; confirm XAI
   method selection (Grad-CAM, SHAP).
3. **Development** — implement, transfer-learn, and fine-tune all three models on both
   tasks; apply Grad-CAM/SHAP post-training.
4. **Evaluation** — compute all metrics, confusion matrices, external-test-set results
   (primary task), and skin-tone-stratified results (secondary task); assess XAI
   faithfulness quantitatively (IoU/Dice vs. HAM10000 segmentation masks) and visually
   (DDI, which has no ground-truth mask).
5. **Trade-off Analysis** — compare accuracy vs. interpretability; select optimal model.

Dependencies: 2 depends on 1; 3 depends on 2; 4 depends on 3; 5 depends on 4.

## Roles in use (see `skills/`)
Required: planner, developer, test, review, reflection.
Optional (justified): risk-assessor (misclassification/ethical risk, lesion-leakage
risk), compliance (HAM10000 + DDI licensing, GDPR), data (interpreting training logs,
metrics, and skin-tone-stratified results).
