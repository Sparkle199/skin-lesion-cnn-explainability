# Stage 0 — Infrastructure

**Status:** Decided.

## Purpose
Provide the compute environment all later stages (training, evaluation, XAI
generation) run on.

## Decision
Runpod on-demand GPU pod, RTX 4090 (24GB VRAM), rather than the proposal's originally
stated Kaggle GPU infrastructure.

## Why
- Six full fine-tuning runs are required (3 architectures × 2 tasks), each with
  layer-by-layer unfreezing, plus Grad-CAM/SHAP post-processing — this exceeds Kaggle's
  free-tier session length (12h) and weekly GPU quota (30h/week) at reasonable epoch
  counts.
- The workload (standard-sized CNNs, ~10k images, transfer learning) is compute-bound,
  not memory-bound — 24GB is comfortably sufficient for all three architectures
  (including EfficientNetB4 at its native 380×380 input and VGG-16's memory-heavy FC
  layers) at batch sizes of 32–64. Larger/multi-GPU tiers (A100/H100/H200/B200) were
  rejected as unjustified cost for this workload's scale.
- The RTX 4090 was chosen over cheaper 24GB cards (RTX A5000 at $0.27/hr, RTX 3090 at
  $0.50/hr) because its Ada-architecture throughput (TF32/FP16 with automatic mixed
  precision) is high enough that total *cost per experiment* — not hourly rate — is
  competitive or lower, while finishing runs faster against a fixed dissertation
  deadline.

## Action needed
This is a deviation from the approved proposal document (which states "Kaggle GPU
infrastructure"). Confirm with the supervisor that a paid cloud GPU provider is an
acceptable substitution, and mention the switch explicitly in Chapter 3.

## Cost estimate
At $0.99/hr, even a generous 6 hours/run × 6 runs (36 GPU-hrs) ≈ $36, plus time for
Grad-CAM/SHAP generation and any re-runs.

## Fallback if VRAM pressure occurs
If EfficientNetB4 at full resolution + batch size 64 approaches the 24GB ceiling during
actual runs, reduce batch size (with gradient accumulation if needed to preserve
effective batch size) rather than silently upgrading GPU tier — document this in
Chapter 3/4 if it happens.

## Outputs
A running Runpod pod with TensorFlow/Keras installed (see root `requirements.txt`) and
the extracted datasets available at the path `PROJECT_DATA_DIR` points to (see
[`01-data-acquisition.md`](01-data-acquisition.md)).
