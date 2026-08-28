#!/bin/bash
# Tier 2 ablation: for each architecture, fine-tune the HAM10000-only binary baseline
# on DDI under three variants (class-weighted, BatchNorm-unfrozen, both combined) and
# evaluate each against the existing baseline (models/{arch}_binary_ddi_finetuned.keras,
# already trained -- not re-run here). Sequential, single GPU.
#
# Usage: ./scripts/run_tier2_ablation.sh
set -uo pipefail

cd /workspace/project
mkdir -p logs models results

ARCHITECTURES="resnet50 efficientnetb4 vgg16"
VARIANTS=(
  "--class-weight|_cw"
  "--unfreeze-batchnorm|_bn"
  "--class-weight --unfreeze-batchnorm|_cw_bn"
)

for arch in $ARCHITECTURES; do
  for variant in "${VARIANTS[@]}"; do
    flags="${variant%%|*}"
    suffix="${variant##*|}"

    echo "=========================================="
    echo "=== FINETUNE  $arch$suffix  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.finetune_ddi --architecture "$arch" \
      --base-model-path "models/${arch}_binary_ham_only.keras" \
      $flags --output-suffix "$suffix" \
      2>&1 | tee "logs/finetune_${arch}${suffix}.log"
    echo "FINETUNE_EXIT[$arch$suffix]=${PIPESTATUS[0]}"

    echo "=========================================="
    echo "=== EVALUATE  $arch$suffix  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.evaluate_ddi --architecture "$arch" --mode finetuned \
      --model-path "models/${arch}_binary_ddi_finetuned${suffix}.keras" \
      --output "results/${arch}_binary_ddi_finetuned${suffix}.json" \
      2>&1 | tee "logs/eval_${arch}${suffix}.log"
    echo "EVAL_EXIT[$arch$suffix]=${PIPESTATUS[0]}"
  done
done

echo "TIER2_ABLATION_ALL_DONE"
