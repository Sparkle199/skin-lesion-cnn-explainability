#!/bin/bash
# Extends the resnet50 DDI pilot (2026-08-17) to the given architectures: for each,
# trains a HAM10000-only binary baseline, evaluates it zero-shot on DDI, fine-tunes it
# on DDI, and evaluates the fine-tuned model. Sequential -- single GPU, and
# efficientnetb4 in particular has previously OOM'd when run concurrently with another
# job (see journal, 2026-08-17 entry), so nothing else should be running on the GPU
# while this executes.
#
# Usage: ./scripts/run_ddi_pilot.sh efficientnetb4 vgg16
set -uo pipefail

cd /workspace/project
mkdir -p logs models results

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <architecture> [architecture ...]"
  exit 1
fi

for arch in "$@"; do
  echo "=========================================="
  echo "=== TRAIN binary_ham_only  $arch  $(date) ==="
  echo "=========================================="
  .venv/bin/python3 -m src.train --architecture "$arch" --task binary_ham_only \
    2>&1 | tee "logs/train_${arch}_binary_ham_only.log"
  echo "TRAIN_HAM_ONLY_EXIT[$arch]=${PIPESTATUS[0]}"

  echo "=========================================="
  echo "=== EVALUATE zero_shot  $arch  $(date) ==="
  echo "=========================================="
  .venv/bin/python3 -m src.evaluate_ddi --architecture "$arch" --mode zero_shot \
    --model-path "models/${arch}_binary_ham_only.keras" \
    2>&1 | tee "logs/eval_${arch}_zero_shot_ddi.log"
  echo "EVAL_ZERO_SHOT_EXIT[$arch]=${PIPESTATUS[0]}"

  echo "=========================================="
  echo "=== FINETUNE on DDI  $arch  $(date) ==="
  echo "=========================================="
  .venv/bin/python3 -m src.finetune_ddi --architecture "$arch" \
    --base-model-path "models/${arch}_binary_ham_only.keras" \
    2>&1 | tee "logs/finetune_${arch}_ddi.log"
  echo "FINETUNE_EXIT[$arch]=${PIPESTATUS[0]}"

  echo "=========================================="
  echo "=== EVALUATE finetuned  $arch  $(date) ==="
  echo "=========================================="
  .venv/bin/python3 -m src.evaluate_ddi --architecture "$arch" --mode finetuned \
    --model-path "models/${arch}_binary_ddi_finetuned.keras" \
    2>&1 | tee "logs/eval_${arch}_finetuned_ddi.log"
  echo "EVAL_FINETUNED_EXIT[$arch]=${PIPESTATUS[0]}"
done

echo "DDI_PILOT_ALL_DONE"
