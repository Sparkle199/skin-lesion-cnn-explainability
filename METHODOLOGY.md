# Methodology

This document scaffolds **every methodology used across this project's development** —
both the *research* methodology (how the ML experiments were designed, run, and
verified) and the *process* methodology (how the work itself was governed, planned,
and recorded). It is a companion to `README.md` (system/pipeline status),
`spec/specification.md` (the WHAT/WHY/CONSTRAINTS/RISKS form), and
`journal/agent-journal.md` (the full reflective log this document summarises).

> Diagrams are [Mermaid](https://mermaid.js.org/) and render natively on GitHub.

## Contents

1. [Governance-driven development process](#1-governance-driven-development-process)
2. [Specification-first workflow](#2-specification-first-workflow)
3. [Dataset & task design](#3-dataset--task-design)
4. [Technical ML pipeline](#4-technical-ml-pipeline)
5. [Model architecture & training methodology](#5-model-architecture--training-methodology)
6. [Evaluation methodology](#6-evaluation-methodology)
7. [DDI fairness methodology — three competing strategies](#7-ddi-fairness-methodology--three-competing-strategies)
8. [Explainability (XAI) methodology](#8-explainability-xai-methodology)
9. [Statistical rigor: the small-sample-first, verify-at-scale cycle](#9-statistical-rigor-the-small-sample-first-verify-at-scale-cycle)
10. [Infrastructure & operational verification methodology](#10-infrastructure--operational-verification-methodology)
11. [Version control: tiered exploration branches](#11-version-control-tiered-exploration-branches)
12. [Testing methodology](#12-testing-methodology)
13. [Reflective practice](#13-reflective-practice)

---

## 1. Governance-driven development process

All work follows an AI-coding governance framework (`AI_Coding_Governance_MSc`):
development is split into fixed **roles**, each with a narrow mandate, rather than one
undifferentiated "write code" process. Five roles are required; three more were
adopted and justified for this project specifically because it is data- and
risk-heavy (`skills/<role>/SKILL.md` is the source of truth for each).

```mermaid
flowchart LR
    subgraph Required
        P["planner<br/><i>goal &rarr; spec &rarr; task list</i>"]
        D["developer<br/><i>implements only what the spec asks</i>"]
        T["test<br/><i>verifies; does not vouch</i>"]
        R["review<br/><i>independent second look</i>"]
        F["reflection<br/><i>records what actually happened</i>"]
    end
    subgraph "Optional — justified for this project"
        RA["risk-assessor<br/><i>misclassification &amp; lesion-leakage risk</i>"]
        C["compliance<br/><i>HAM10000 / DDI licensing, GDPR</i>"]
        DA["data<br/><i>reads logs, metrics, errors honestly</i>"]
    end

    P -->|"approved spec"| D
    D -->|"delivered change"| T
    T -->|"pass/fail + gaps"| R
    R -->|"blocking / advisory notes"| F
    F -.->|"informs next"| P

    RA -.->|"escalates to student"| P
    C -.->|"flags gaps"| P
    DA -.->|"summarises behaviour"| R
```

**Why these three optional roles specifically:** `risk-assessor` because a false
malignant/benign call carries real (if non-clinical) consequence and because a
lesion-level data leak would silently inflate every downstream number;
`compliance` because the project combines two differently-licensed datasets
(HAM10000 CC BY-NC 4.0, DDI attribution-required) inside a GDPR-scoped academic
project; `data` because nearly every deliverable *is* a metric, log, or model file
that needs an honest read before it can be trusted (see §9).

Each role is deliberately narrow — `developer` never invents requirements,
`test` "verifies, does not vouch," `review` only recommends, never imposes,
`risk-assessor` escalates rather than silently absorbing risk. The effect is that no
single step both produces a result and certifies it.

## 2. Specification-first workflow

No implementation work starts before a specification exists. `spec/specification.md`
restates the original proposal (`project_desc.txt`) in the governance framework's
required form — **WHAT / WHY / CONSTRAINTS / RISKS / SUCCESS CRITERIA** — and is
**revised in place** whenever a real finding changes the plan, rather than left to
drift out of date.

```mermaid
flowchart TD
    A["project_desc.txt<br/>(original proposal)"] --> B["planner role:<br/>restate as WHAT/WHY/CONSTRAINTS/RISKS/ACCEPTANCE"]
    B --> C["spec/specification.md"]
    C --> D["Task decomposition<br/>Analysis &rarr; Design &rarr; Development &rarr; Evaluation &rarr; Trade-off"]
    D --> E["developer implements one task"]
    E --> F["test verifies against acceptance criteria"]
    F --> G["review checks quality/maintainability"]
    G --> H["reflection logs outcome in agent-journal.md"]
    H -->|"finding changes the plan<br/>(e.g. DDI taxonomy mismatch,<br/>joint beats zero-shot/fine-tuned)"| B
    H -->|"finding confirms the plan"| D
```

This loop fired for real at least twice on record: the DDI taxonomy mismatch
(2026-07-31, DDI's 78 diagnoses don't map onto HAM10000's 7 classes, resolved by
making binary malignant/benign a second task rather than forcing a bad mapping), and
the DDI methodology revision (2026-08-17/18, one joint-training design became three
compared strategies once "how biased is a model that's never seen DDI?" turned out to
be unanswerable from the joint model alone). Both revisions are recorded directly in
`spec/specification.md`'s WHAT section, not just in the journal.

## 3. Dataset & task design

```mermaid
flowchart TD
    HAM["HAM10000<br/>10,015 images / 7,470 unique lesions<br/>7 diagnostic classes<br/>CC BY-NC 4.0"]
    DDI["DDI<br/>656 images, 78 diagnoses<br/>native malignant/benign label<br/>skin_tone: FST I-II/III-IV/V-VI"]
    ISIC["ISIC2018 Task 3<br/>1,511 images<br/>independent held-out test set"]

    HAM -->|"lesion-level split<br/>(no leakage)"| T1["Primary task<br/>Seven-class classification<br/>HAM10000 only"]
    HAM -->|"relabel via<br/>malignant = &lbrace;akiec,bcc,mel&rbrace;<br/>benign = &lbrace;bkl,df,nv,vasc&rbrace;"| T2["Secondary task<br/>Binary malignant/benign<br/>HAM10000 + DDI"]
    DDI -->|"used natively —<br/>78 diagnoses don't map<br/>onto the 7-class taxonomy"| T2
    ISIC -->|"external, lesion-disjoint<br/>test only"| T1
```

Two design decisions carry the rest of the methodology:

- **Lesion-level, not image-level, splitting** for the primary task. HAM10000's
  10,015 images cover only 7,470 unique lesions (some photographed more than once);
  an image-level split would let the same lesion leak across train/validation,
  inflating every reported number. `ham10000.py :: lesion_level_split()` enforces
  this once, upstream of everything else.
- **DDI is never forced into the seven-class taxonomy.** Its 78 diagnoses (e.g.
  verruca-vulgaris, epidermal-cyst, acrochordon) have no honest seven-class
  counterpart. Rather than a lossy best-effort mapping, DDI is used only for the
  binary task, where its own native malignant/benign flag is authoritative.

## 4. Technical ML pipeline

The end-to-end pipeline, current as of the SHAP/faithfulness-scaling work (this
supersedes `README.md`'s original stage list, which pre-dates Stage 8's full build-out
and Stages 8b–9):

```mermaid
flowchart TD
    subgraph S0["0 — Infrastructure"]
        A0["Runpod on-demand GPU pod<br/>RTX 4090, 24GB VRAM"]
    end

    subgraph S1["1 — Data Acquisition"]
        A1a["HAM10000 archive"] --> A1b["images + metadata +<br/>segmentation masks + ISIC2018 test set"]
        A1c["DDI archive"] --> A1d["656 images + ddi_metadata.csv"]
    end

    subgraph S2["2 — Metadata Loading &amp; Splitting"]
        B1["ham10000.py :: load_metadata()"] --> B2["lesion_level_split()<br/>train_df / val_df"]
        B3["ddi.py :: load_metadata()"] --> B4["split_ddi()<br/>ddi_train / ddi_val"]
    end

    subgraph S3["3 — Task Corpus Construction"]
        C1["Seven-class:<br/>train_df / val_df as-is"]
        C2["Binary:<br/>build_binary_corpus()<br/>HAM relabelled + DDI merged"]
    end

    subgraph S4["4 — Imbalance Handling"]
        D1["compute_class_weights()"]
        D2["compute_steps_per_epoch()"]
        D3["make_oversampled_binary_dataset()<br/>ddi_fraction = 0.3"]
        D4["make_dataset()<br/>rotate / flip / zoom / brightness"]
    end

    subgraph S5["5 — Model Construction"]
        E1["build_model()<br/>ResNet-50 / EfficientNetB4 / VGG-16<br/>ImageNet-pretrained, frozen backbone"]
    end

    subgraph S6["6 — Training (6 base runs)"]
        F1["Phase 1: frozen-backbone warmup"] --> F2["unfreeze_top_layers()"] --> F3["Phase 2: fine-tune top N layers"] --> F4["models/&lbrace;arch&rbrace;_&lbrace;task&rbrace;.keras"]
    end

    subgraph S6B["6b — DDI Strategy Branch (binary task)"]
        DA["Zero-shot:<br/>evaluate HAM-only model on full DDI"]
        DB["Fine-tuned:<br/>finetune_ddi.py on DDI train split"]
        DC["Joint:<br/>trained inside Stage 6 via oversampling"]
    end

    subgraph S7["7 — Evaluation"]
        G1["predict_dataset()"] --> G2["accuracy / precision / recall / F1 / ROC-AUC / kappa"]
        G1 --> G3["stratified_binary_metrics()<br/>by Fitzpatrick skin-tone group"]
        G4["ISIC2018 held-out test<br/>(seven-class only)"] --> G1
    end

    subgraph S8["8 — XAI Generation"]
        H1["gradcam.py :: make_gradcam_heatmap()"]
        H2["shap_explain.py :: build_explainer() + explain_images()<br/>Partition explainer, real trained model"]
        H3["faithfulness.py :: compute_overlap()<br/>IoU / Dice vs. HAM10000 segmentation masks"]
    end

    subgraph S8B["8b — Sample-Size Scaling &amp; Statistical Verification"]
        I1["Grad-CAM faithfulness: n=30 &rarr; n=150<br/>(all 6 base models)"]
        I2["SHAP faithfulness: n=15 &rarr; n=150 &rarr; n=500/400/200<br/>(cgroup memory ceiling forces per-arch cap)"]
        I3["Paired 95% CI on Grad-CAM &minus; SHAP gap<br/>mean &plusmn; 1.96&times;SE, per architecture"]
    end

    subgraph S9["9 — Trade-off Analysis"]
        J1["trade_off.py<br/>aggregates all results/*.json"]
        J2["Accuracy vs. interpretability comparison<br/>+ recommended model per task"]
    end

    subgraph S10["10 — Streamlit Demo App"]
        K1["Single-image demo: upload &rarr; predict &rarr; Grad-CAM"]
        K2["Model comparison view"]
        K3["Trade-off view"]
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
    E1 --> F1
    F4 --> DA
    F4 --> DB
    F4 -.->|"binary task trained<br/>with DDI mixed in"| DC
    F4 --> G1
    G4 --> G1
    F4 --> H1 --> H3
    F4 --> H2 --> H3
    H3 --> I1 --> I2 --> I3
    G2 --> J1
    G3 --> J1
    H3 --> J1
    I3 --> J1
    J1 --> J2
    F4 --> K1
    J1 --> K2
    J2 --> K3
```

## 5. Model architecture & training methodology

Three ImageNet-pretrained backbones (ResNet-50, EfficientNetB4, VGG-16), each with a
new classification head, trained identically in **two phases** per (architecture,
task) run: a frozen-backbone warmup so the new head doesn't destroy pretrained
features with large early gradients, then unfreezing the top *N* layers for a
low-learning-rate fine-tune. EfficientNetB4 uses its canonical 380×380 input; the
other two use the standard 224×224 — this single difference in per-image memory
footprint is what later forced a per-architecture sample-size cap during SHAP scaling
(§9, §10).

**Imbalance handling** operates at two levels:
- **Class weighting** (`compute_class_weights`) for the seven-class task, where `nv`
  alone is ~67% of HAM10000.
- **Oversampled batching** (`make_oversampled_binary_dataset`, `ddi_fraction = 0.3`)
  for the binary task's joint strategy, so DDI's 656 images aren't drowned out by
  HAM10000's 10,015 in every batch despite the ~15:1 size gap.

## 6. Evaluation methodology

Every run reports the same core suite — accuracy, per-class precision/recall/F1,
ROC-AUC, Cohen's κ, confusion matrix — computed identically for both tasks so
architectures are always compared on like-for-like metrics. Two extensions apply
per task:

- **Seven-class**: also evaluated on the independent ISIC2018 Task 3 test set
  (lesion-disjoint from HAM10000's training set), so the internal validation number
  is never the only evidence of generalisation.
- **Binary**: also stratified by DDI's Fitzpatrick skin-tone bands (FST I–II /
  III–IV / V–VI), since the entire reason DDI was added was to make skin-tone bias
  measurable rather than assumed.

Cohen's κ, not raw accuracy, is treated as the decisive metric wherever class
imbalance could make accuracy misleading (see §7) — a deliberate choice recorded in
the specification, not an afterthought.

## 7. DDI fairness methodology — three competing strategies

The binary task's original design (train jointly on HAM10000+DDI from the start) was
revised into **three compared strategies** once it became clear the joint model alone
couldn't answer "how biased is a model that has never seen DDI?".

```mermaid
flowchart TD
    HAM["HAM10000-only binary baseline<br/>(binary_ham_only)"]

    HAM -->|"evaluate directly,<br/>zero DDI training"| ZS["Zero-shot<br/><i>generalisation stress test</i>"]
    HAM -->|"finetune_ddi.py<br/>low-LR, DDI train split only"| FT["Fine-tuned<br/><i>sequential adaptation</i>"]
    JOINT["Train from scratch on<br/>HAM10000 + DDI mixed<br/>(oversampled, ddi_fraction=0.3)"] --> J["Joint<br/><i>original design</i>"]

    ZS --> EVAL["Evaluate all three<br/>on the identical DDI held-out split"]
    FT --> EVAL
    J --> EVAL

    EVAL --> RESULT["<b>Joint wins on &kappa; for all 3 architectures</b><br/>&kappa; 0.211&ndash;0.356 (joint) vs.<br/>0.071&ndash;0.167 (zero-shot / fine-tuned)"]

    style RESULT fill:#e8f4ea,stroke:#0ca30c,color:#0b3d0b
```

Each strategy answers a different question, which is why all three were kept rather
than discarding two once a winner emerged:

| Strategy | Question it answers | Outcome |
|---|---|---|
| **Zero-shot** | How badly does a HAM10000-only model generalise to DDI with *no* adaptation? | Malignant recall collapses from 0.799–0.871 (HAM10000) to 0.123–0.480 (DDI), across all 3 architectures — confirms the anticipated bias gap as a measured finding, not an assumption. |
| **Fine-tuned** | Does adapting the existing model on DDI alone recover performance? | Not uniformly — helped resnet50/vgg16, *hurt* efficientnetb4. Diagnosed as a calibration-threshold shift running in different directions per architecture, not a real discrimination change. |
| **Joint** | Is training on both sources together, from the start, actually the best approach? | Yes, decisively, on κ and ROC-AUC, for every architecture. Recommended as the project's best binary-task model; the other two remain diagnostic tools, not competing candidates. |

## 8. Explainability (XAI) methodology

Two post-hoc explanation methods are applied to every trained model and scored the
same way, so they are directly comparable rather than qualitatively eyeballed:

```mermaid
flowchart LR
    M["Trained model<br/>+ input image"] --> GC["Grad-CAM<br/>last-conv-layer activations<br/>&rarr; heatmap"]
    M --> SH["SHAP<br/>Partition explainer<br/>&rarr; per-pixel attribution"]
    GC --> N1["Normalise to [0,1]<br/>threshold at 0.5"]
    SH --> N2["Normalise to [0,1]<br/>threshold at 0.5"]
    N1 --> OV["compute_overlap()<br/>vs. HAM10000 ground-truth<br/>lesion segmentation mask"]
    N2 --> OV
    OV --> SCORE["IoU + Dice<br/>faithfulness score"]
```

Both maps are normalised and thresholded identically before scoring, so an IoU
difference between them reflects the method, not an inconsistent scoring rule. DDI
carries no segmentation mask, so its faithfulness check remains visual/qualitative
only — stated explicitly rather than silently applying a quantitative score where
none is possible.

## 9. Statistical rigor: the small-sample-first, verify-at-scale cycle

The single most repeated methodological pattern in this project's later stages: **a
small, cheap sample is run first, then deliberately not trusted until checked against
a larger one.** This happened independently for Grad-CAM faithfulness (n=30→150,
all 6 base models) and again for SHAP faithfulness (n=15→150→500), and both times the
small sample turned out to be misleading in a way only the larger sample exposed.

```mermaid
flowchart TD
    A["Run at small n<br/>(cheap, fast)"] --> B["Compute point estimate"]
    B --> C{"Cross-check against any<br/>existing larger-sample estimate<br/>for the same model?"}
    C -->|"disagrees"| D["Flag as unreliable —<br/>do not report as a finding yet"]
    C -->|"agrees"| E["Provisionally trust,<br/>but still scale up"]
    D --> F["Re-run at larger n"]
    E --> F
    F --> G["Compute paired 95% CI<br/>mean &plusmn; 1.96&times;SE on the difference"]
    G --> H{"CI excludes zero?"}
    H -->|"yes"| I["Report as a real, directional finding"]
    H -->|"no"| J["Report as a genuine tie —<br/>not an unresolved question"]
    I --> K["Cross-check again at an even<br/>larger n if cost allows"]
    J --> K
    K -.->|"repeat"| G
```

**Where this fired for real, with the actual numbers:**

| Check | Small-n result | Large-n result | What changed |
|---|---|---|---|
| Grad-CAM, binary task, resnet50 | n=30: IoU 0.215 | n=150: IoU 0.157 | −27% — the n=30 estimate was optimistic and had to be corrected downward before any cross-model comparison could trust it. |
| SHAP vs Grad-CAM, efficientnetb4 | n=15: SHAP wins (0.273 vs 0.185) | n=150/200/400: statistically tied (95% CI crosses zero at every size) | The apparent "SHAP is more faithful here" flip was a small-sample artefact, not confirmed at any larger size. |
| SHAP vs Grad-CAM, vgg16 | n=15: SHAP wins narrowly (0.218 vs 0.200) | n=500: SHAP wins clearly (0.255 vs 0.202, CI [−0.067, −0.038]) | This flip *was* real — it held and strengthened rather than disappearing. |

The methodological lesson generalised explicitly in the project's own record: a
clean-looking small-sample number is not evidence of reliability on its own — it
must be checked against either a larger sample or an existing estimate for the same
model before it is allowed to drive a conclusion.

## 10. Infrastructure & operational verification methodology

Compute-heavy stages (training, SHAP) ran on an on-demand Runpod GPU pod
(RTX 4090, 24GB VRAM) rather than the proposal's original Kaggle infrastructure — a
deliberate, documented deviation (`docs/pipeline/00-infrastructure.md`). Long jobs
follow a consistent discipline rather than being launched and hoped for:

```mermaid
flowchart TD
    A["Launch job with nohup + disown<br/>(survives SSH disconnect)"] --> B["Poll log + process state<br/>at sensible intervals, not continuously"]
    B --> C{"Exit code?"}
    C -->|"0"| D["Trust and pull the result file"]
    C -->|"non-zero / SIGKILL"| E["Diagnose root cause —<br/>never assume, never retry blind"]
    E --> F["Check the actual evidence:<br/>cgroup memory.max, not free -h;<br/>symlink targets; live process cmdline"]
    F --> G["Fix the specific cause"]
    G --> A
    D --> H["Rename/promote result only<br/>after confirming the real exit code —<br/>never on file-exists alone"]
```

Two real incidents this session shaped that last rule specifically: a background
job's rename step once promoted a **stale, pre-existing** result file to a
larger-sample-size name after the actual run had crashed (caught by comparing file
size/timestamp against a known backup, then fixed to gate renaming on the real exit
code); and an unexplained `SIGKILL` (exit 137) on a large SHAP run was traced not to
a code bug but to the pod's container `cgroup` memory limit (~57GiB), invisible to
`free -h`, which only `/sys/fs/cgroup/memory.max` revealed directly.

## 11. Version control: tiered exploration branches

After the core finding that joint DDI training beats both alternatives landed on
`main`, four **independent, parallel exploration branches** were opened from that
point — each pursuing one follow-up robustness question rather than one branch trying
to do everything sequentially:

```mermaid
gitGraph
   commit id: "Initial scaffolding"
   commit id: "README + pipeline diagram"
   commit id: "Integration smoke tests (3 archs)"
   commit id: "First 6-run training schedule"
   commit id: "DDI pilot: all 3 architectures"
   commit id: "Joint DDI beats zero-shot/fine-tuned"
   branch tier1-threshold-calibration-and-bootstrap-ci
   checkout tier1-threshold-calibration-and-bootstrap-ci
   commit id: "Bootstrap CIs on major comparisons"
   commit id: "Calibrate zero-shot/fine-tuned + McNemar's"
   checkout main
   branch tier2-class-weighted-finetune-and-bn-unfrozen-ablation
   commit id: "Class-weighted + BN-unfrozen ablation"
   checkout main
   branch tier3-multiseed-binary-joint-variance
   commit id: "Multi-seed variance check"
   checkout main
   branch tier4-shap-and-larger-faithfulness-sample
   commit id: "SHAP vs a real model, first time"
   commit id: "n=150 faithfulness, all 6 models"
   commit id: "SHAP extended to all 3 architectures"
   commit id: "SHAP scaled to n=150"
   commit id: "SHAP to n=500 (n=400 effnet)"
   commit id: "Formal paired 95% CI documented"
```

Each tier branch is scoped to one open question raised by the `main`-line findings:

| Branch | Open question it addresses |
|---|---|
| `tier1-threshold-calibration-and-bootstrap-ci` | Are the reported comparisons robust to decision-threshold choice, and do bootstrap CIs / McNemar's test support them statistically? |
| `tier2-class-weighted-finetune-and-bn-unfrozen-ablation` | Does class-weighting or unfreezing batch-norm during DDI fine-tuning change the fine-tuned strategy's outcome? |
| `tier3-multiseed-binary-joint-variance` | How much does the joint model's advantage vary across random seeds — is it a stable effect or one lucky run? |
| `tier4-shap-and-larger-faithfulness-sample` | Does the XAI faithfulness comparison (§8, §9) hold at a real trained model and at a statistically defensible sample size? |

As of this document, all four remain independent branches rather than merged back
into `main` — each is a self-contained line of evidence, not yet reconciled into one
combined result set.

## 12. Testing methodology

`tests/` mirrors `src/` one-to-one (unit tests per module: data loading, splitting,
oversampling, evaluation, faithfulness, SHAP wiring, JSON serialisation) plus one
**integration smoke test**, deliberately run against the real GPU and real data
rather than mocks, and parametrized over all three architectures so an
architecture-specific regression can't hide behind a single-architecture pass.
Per the `test` role's own rule (§1), tests report plainly what passed, what failed,
and what remains untested — they do not certify correctness beyond what was actually
exercised.

## 13. Reflective practice

`journal/agent-journal.md` is appended after every significant task — never rewritten
retroactively — in a fixed structure: **what happened, where the agent was uncertain
or stuck, what assumptions were made, how the output was verified, and what should
change next time.** This document (`METHODOLOGY.md`) is a structural summary of that
journal, not a replacement for it — the journal is the primary evidence record (it
also underpins the dissertation's Appendix A.4, Reflective Account); this file exists
to make the methodology it documents legible at a glance, including in diagram form,
without reading 1,700+ lines of chronological entries.
