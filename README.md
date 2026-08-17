# Deep Learning for Multi-Class Skin Cancer Detection: A Performance and Interpretability Analysis

MSc dissertation project. Trains and compares three CNN architectures (ResNet-50,
EfficientNetB4, VGG-16) on two dermoscopic-image classification tasks, then evaluates
the trade-off between classification performance and post-hoc explainability
(Grad-CAM, SHAP). See `spec/specification.md` for the full WHAT/WHY/CONSTRAINTS form and
`project_desc.txt` for the original proposal.

> **This is a living document.** It is updated as the pipeline is built out, not
> written once and left. If a stage's status below is stale relative to `src/`, trust
> `src/` and fix this file.

## System overview

Two classification tasks share one data foundation (HAM10000) and diverge downstream:

- **Primary task — seven-class:** HAM10000 only (10,015 images / 7,470 unique lesions),
  split at the lesion level, evaluated on an internal validation split *and* the
  independent ISIC2018 Task 3 held-out test set (1,511 images).
- **Secondary task — binary malignant/benign:** HAM10000 (relabelled to malignant/benign)
  combined with the full DDI dataset (656 images), included specifically to bring in
  skin-tone diversity absent from HAM10000. Evaluated on the internal split, additionally
  stratified by Fitzpatrick skin-tone group.

Every architecture is trained once per task (3 architectures × 2 tasks = 6 trained
models), evaluated with the same metric suite, explained with Grad-CAM and SHAP, and
finally compared in a trade-off analysis to select the best accuracy/interpretability
balance. A Streamlit app surfaces the trained models and results interactively as the
project's demonstrable artefact.

Infrastructure: Runpod on-demand GPU (RTX 4090, 24GB), chosen for compute-bound CNN
transfer learning within a fixed deadline. This deviates from the proposal's original
"Kaggle GPU infrastructure" statement — see `docs/pipeline/00-infrastructure.md` and
flag the switch to your supervisor before submission.

## Pipeline diagram

```mermaid
flowchart TD
    subgraph S0["Stage 0 — Infrastructure"]
        A0["Runpod GPU pod<br/>RTX 4090, 24GB"]
    end

    subgraph S1["Stage 1 — Data Acquisition"]
        A1a["archive (1).zip"] --> A1b["data/raw/dataverse_files/<br/>HAM10000 images + metadata +<br/>segmentation masks + ISIC2018 test set"]
        A1c["ddidiversedermatologyimages.zip"] --> A1d["data/raw/ddi/<br/>656 images + ddi_metadata.csv"]
    end

    subgraph S2["Stage 2 — Metadata Loading and Splitting"]
        B1["ham10000.py :: load_metadata()"] --> B2["ham10000.py :: lesion_level_split()<br/>train_df / val_df, lesion-level"]
        B3["ddi.py :: load_metadata()"] --> B4["ddi.py :: split_ddi()<br/>ddi_train / ddi_val"]
    end

    subgraph S3["Stage 3 — Task Corpus Construction"]
        C1["Seven-class task:<br/>train_df / val_df as-is"]
        C2["Binary task:<br/>binary_corpus.py :: build_binary_corpus()<br/>HAM10000 relabelled + DDI merged"]
    end

    subgraph S4["Stage 4 — Imbalance Handling"]
        D1["augmentation.py :: compute_class_weights()"]
        D2["oversampling.py :: compute_steps_per_epoch()"]
        D3["pipeline.py :: make_oversampled_binary_dataset()<br/>binary task only, ddi_fraction = 0.3"]
        D4["pipeline.py :: make_dataset()<br/>augmentation: rotate / flip / zoom / brightness"]
    end

    subgraph S5["Stage 5 — Model Construction"]
        E1["models/build.py :: build_model()<br/>ResNet-50 / EfficientNetB4 / VGG-16<br/>ImageNet-pretrained, frozen backbone"]
    end

    subgraph S6["Stage 6 — Training, 6 runs"]
        F1["train.py<br/>Phase 1: frozen-head warmup"]
        F2["models/build.py :: unfreeze_top_layers()"]
        F3["train.py<br/>Phase 2: fine-tune top N layers"]
        F4["models/&lbrace;architecture&rbrace;_&lbrace;task&rbrace;.keras"]
    end

    subgraph S7["Stage 7 — Evaluation"]
        G1["evaluate_run.py :: predict_dataset()"]
        G2["evaluate.py :: evaluate_predictions()<br/>accuracy / precision / recall / F1 / ROC-AUC / kappa"]
        G3["evaluate.py :: stratified_binary_metrics()<br/>binary task: by Fitzpatrick group"]
        G4["ISIC2018 held-out test<br/>seven-class task only"]
        G5["xai/gradcam.py :: make_gradcam_heatmap()"]
        G6["xai/faithfulness.py :: mean_overlap()<br/>IoU / Dice vs. HAM10000 masks"]
        G7["results/&lbrace;architecture&rbrace;_&lbrace;task&rbrace;.json"]
    end

    subgraph S8["Stage 8 — XAI Generation"]
        H1["xai/shap_explain.py<br/>SHAP attribution maps"]
        H2["DDI visual/qualitative check<br/>no ground-truth mask available"]
        H3["Saved overlay images<br/>Grad-CAM + SHAP, per model"]
    end

    subgraph S9["Stage 9 — Trade-off Analysis"]
        I1["trade_off.py<br/>aggregates all 6 results/*.json"]
        I2["Accuracy vs. interpretability comparison<br/>+ recommended model"]
    end

    subgraph S10["Stage 10 — Streamlit App"]
        J1["Page 1: Single-image demo<br/>upload to predict to Grad-CAM live"]
        J2["Page 2: Model comparison<br/>reads results/*.json"]
        J3["Page 3: Trade-off view<br/>reads trade_off.py output"]
    end

    A0 --> A1a
    A0 --> A1c
    A1b --> B1
    A1d --> B3
    B2 --> C1
    B2 --> C2
    B4 --> C2
    C1 --> D4
    C2 --> D1
    C2 --> D2
    C2 --> D3
    D1 --> E1
    D3 --> E1
    D4 --> E1
    E1 --> F1 --> F2 --> F3 --> F4
    F4 --> G1
    G1 --> G2
    G1 --> G3
    G4 --> G1
    G1 --> G5
    G5 --> G6
    G2 --> G7
    G3 --> G7
    G6 --> G7
    F4 --> H1
    H1 --> H3
    G5 --> H3
    H2 --> H3
    G7 --> I1
    I1 --> I2
    F4 --> J1
    G7 --> J2
    I2 --> J3
    H3 --> J1
```

## Pipeline stages

Each stage has its own doc under `docs/pipeline/` — purpose, inputs, process, outputs,
code references, and current status. Update the relevant stage doc (and the status
table below) whenever that stage's implementation changes.

| # | Stage | Status | Doc |
|---|---|---|---|
| 0 | Infrastructure | Decided (Runpod, RTX 4090) | [`docs/pipeline/00-infrastructure.md`](docs/pipeline/00-infrastructure.md) |
| 1 | Data acquisition | Manual step, not yet automated | [`docs/pipeline/01-data-acquisition.md`](docs/pipeline/01-data-acquisition.md) |
| 2 | Metadata loading & splitting | Implemented | [`docs/pipeline/02-metadata-loading-splitting.md`](docs/pipeline/02-metadata-loading-splitting.md) |
| 3 | Task corpus construction | Implemented | [`docs/pipeline/03-task-corpus-construction.md`](docs/pipeline/03-task-corpus-construction.md) |
| 4 | Imbalance handling | Implemented | [`docs/pipeline/04-imbalance-handling.md`](docs/pipeline/04-imbalance-handling.md) |
| 5 | Model construction | Implemented | [`docs/pipeline/05-model-construction.md`](docs/pipeline/05-model-construction.md) |
| 6 | Training | Implemented, not yet run on Runpod | [`docs/pipeline/06-training.md`](docs/pipeline/06-training.md) |
| 7 | Evaluation | Implemented, not yet run | [`docs/pipeline/07-evaluation.md`](docs/pipeline/07-evaluation.md) |
| 8 | XAI generation | Partially implemented (SHAP module exists; no saved-overlay step yet) | [`docs/pipeline/08-xai-generation.md`](docs/pipeline/08-xai-generation.md) |
| 9 | Trade-off analysis | Implemented, not yet run (depends on Stage 7 outputs) | [`docs/pipeline/09-trade-off-analysis.md`](docs/pipeline/09-trade-off-analysis.md) |
| 10 | Streamlit app | Planned — not yet built | [`docs/pipeline/10-streamlit-app.md`](docs/pipeline/10-streamlit-app.md) |

## Repository layout

```
src/
  config.py           # paths, class lists, label mappings, per-architecture settings
  data/                # Stages 1-4: loading, splitting, corpus construction, imbalance handling
  models/build.py      # Stage 5: model construction
  train.py             # Stage 6: training entrypoint
  evaluate.py           evaluate_run.py   # Stage 7: metrics + faithfulness
  xai/                 # Stages 7-8: Grad-CAM, SHAP, faithfulness scoring
  trade_off.py         # Stage 9: cross-model comparison
tests/                 # unit tests, one file per src/ module (pytest)
spec/
  specification.md     # WHAT/WHY/CONSTRAINTS/RISKS/ACCEPTANCE-CRITERIA
journal/
  agent-journal.md      # reflective log, evidence for dissertation Appendix A.4
docs/pipeline/          # this file's per-stage detail docs
```

## Related documents
- `project_desc.txt` — original project proposal text.
- `spec/specification.md` — governance-form specification derived from the proposal.
- `journal/agent-journal.md` — reflective account of decisions and uncertainty as the
  project was built (Appendix A.4 evidence).

## Data & licensing
- **HAM10000** — CC BY-NC 4.0, non-commercial academic use, attribution required.
- **DDI** — attribution to Stanford University required; exact licence terms not yet
  independently verified (confirm before submission — see `spec/specification.md`
  CONSTRAINTS).
- Neither dataset's raw files are committed to this repository (see `.gitignore`) —
  extract locally per `docs/pipeline/01-data-acquisition.md`.
