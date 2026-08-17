#!/bin/bash
# Runs all 6 (architecture, task) training + evaluation combinations sequentially,
# then the trade-off analysis. Intended to run on the Runpod pod, from /workspace/project.
set -uo pipefail

cd /workspace/project
mkdir -p logs models results

ARCHITECTURES="resnet50 efficientnetb4 vgg16"
TASKS="seven_class binary"

for arch in $ARCHITECTURES; do
  for task in $TASKS; do
    echo "=========================================="
    echo "=== TRAIN  $arch / $task  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.train --architecture "$arch" --task "$task" \
      2>&1 | tee "logs/train_${arch}_${task}.log"
    train_status=${PIPESTATUS[0]}
    echo "TRAIN_EXIT_CODE[$arch/$task]=$train_status"

    if [ "$train_status" -ne 0 ]; then
      echo "!!! Training failed for $arch/$task, skipping evaluation for this pair !!!"
      continue
    fi

    echo "=========================================="
    echo "=== EVALUATE  $arch / $task  $(date) ==="
    echo "=========================================="
    .venv/bin/python3 -m src.evaluate_run --architecture "$arch" --task "$task" \
      --model-path "models/${arch}_${task}.keras" \
      2>&1 | tee "logs/eval_${arch}_${task}.log"
    eval_status=${PIPESTATUS[0]}
    echo "EVAL_EXIT_CODE[$arch/$task]=$eval_status"
  done
done

echo "=========================================="
echo "=== TRADE-OFF ANALYSIS  $(date) ==="
echo "=========================================="
.venv/bin/python3 -m src.trade_off --results-dir results/ 2>&1 | tee "logs/trade_off.log"

echo "ALL_DONE"
