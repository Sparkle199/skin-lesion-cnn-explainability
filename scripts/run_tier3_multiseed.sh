#!/bin/bash
# Tier 3: multi-seed variance check for the joint/mixed binary models (the project's
# recommended best approach). No global TF seed is set anywhere in this pipeline, so
# each src.train invocation gets different weight initialisation/dropout even though
# the data split and shuffle order stay identical (those are separately seeded) --
# re-running the same (architecture, task) captures training-stochasticity variance,
# complementary to Tier 1's test-set-sampling bootstrap CIs, not a replacement for them.
#
# Existing models/{arch}_binary.keras count as "seed 1" (already trained, not re-run
# here). This produces two more seeds (_seed2, _seed3) per architecture.
#
# Usage: ./scripts/run_tier3_multiseed.sh
set -uo pipefail

cd /workspace/project
mkdir -p logs models results

ARCHITECTURES="resnet50 efficientnetb4 vgg16"
SEEDS="_seed2 _seed3"

for arch in $ARCHITECTURES; do
  for suffix in $SEEDS; do
    echo "=========================================="
    echo "=== TRAIN  $arch$suffix (binary, joint)  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.train --architecture "$arch" --task binary --output-suffix "$suffix" \
      2>&1 | tee "logs/train_${arch}_binary${suffix}.log"
    echo "TRAIN_EXIT[$arch$suffix]=${PIPESTATUS[0]}"

    echo "=========================================="
    echo "=== EVALUATE (mixed HAM+DDI val)  $arch$suffix  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.evaluate_run --architecture "$arch" --task binary \
      --model-path "models/${arch}_binary${suffix}.keras" \
      --output "results/${arch}_binary${suffix}.json" \
      2>&1 | tee "logs/eval_${arch}_binary${suffix}.log"
    echo "EVAL_RUN_EXIT[$arch$suffix]=${PIPESTATUS[0]}"

    echo "=========================================="
    echo "=== EVALUATE (DDI held-out, joint mode)  $arch$suffix  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.evaluate_ddi --architecture "$arch" --mode joint \
      --model-path "models/${arch}_binary${suffix}.keras" \
      --output "results/${arch}_binary_joint_ddi_val${suffix}.json" \
      2>&1 | tee "logs/eval_ddi_${arch}_joint${suffix}.log"
    echo "EVAL_DDI_EXIT[$arch$suffix]=${PIPESTATUS[0]}"
  done
done

echo "TIER3_MULTISEED_ALL_DONE"
