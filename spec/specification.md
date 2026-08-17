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
  1. **Joint/mixed** (original design, kept as a comparison arm, not superseded): DDI
     mixed into training from the first batch via oversampled batching (see Imbalance
     handling below).
  2. **Zero-shot generalisation stress test:** train a HAM10000-only binary baseline
     (task `binary_ham_only`), then evaluate it on DDI's full 656 images *without any
     DDI training at all* — this alone is diagnostic of how well the model generalises
     across imaging modality and skin tone, and is not something the joint-mixed
     approach can measure directly (its model never exists without DDI in its training
     data).
  3. **Sequential fine-tuning:** take the same HAM10000-only baseline and fine-tune it
     further on DDI's own train split only, at a low learning rate, rather than mixing
     both sources from scratch — evaluated on DDI's held-out validation split plus a
     HAM10000 retention check (has adapting to DDI cost baseline performance?).
  Piloted for resnet50 only as of 2026-08-17 (see journal); extending to
  efficientnetb4/vgg16 is not yet done.
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
against. The resnet50 pilot's zero-shot result (malignant recall 0.799 on HAM10000's own
validation split, collapsing to 0.158 on DDI) makes the Social Issues section's
generalisation concern a measured finding rather than a qualitative one, and the
subsequent fine-tuning result (partial recovery to 0.269, at a real cost to HAM10000
retention) demonstrates that incorporating a small, diverse dataset after the fact is a
genuine trade-off, not a free fix — directly relevant to the "residual demographic bias"
risk already logged below.

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
- **Catastrophic forgetting**: fine-tuning a HAM10000-pretrained model on DDI alone
  (sequential approach) risks trading HAM10000 performance for DDI adaptation rather than
  improving both. Confirmed as a real, not just theoretical, trade-off by the same pilot:
  5 epochs of DDI fine-tuning improved DDI malignant recall (0.158→0.269) but cost 6
  points of HAM10000 accuracy (0.815→0.754) and did not improve DDI's own ROC-AUC
  (0.654→0.584). Must be reported as a trade-off, not a straightforward improvement.
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
