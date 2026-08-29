#!/bin/bash
# Scales SHAP faithfulness further, from the n=150 run (2026-08-29, all 3
# architectures) up to n=500, to tighten the confidence interval on the
# Grad-CAM-vs-SHAP faithfulness gap. Sequential -- single GPU. Backs up any
# pre-existing plain-named result before running, and only renames to the
# _n500 suffix if the run actually exits 0 (see journal 2026-08-29 for why
# an unconditional rename is unsafe).
#
# Usage: ./scripts/run_shap_n500.sh resnet50 efficientnetb4 vgg16
set -uo pipefail

cd /workspace/project
mkdir -p logs results

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <architecture> [architecture ...]"
  exit 1
fi

for arch in "$@"; do
  echo "=========================================="
  echo "=== SHAP n=500  $arch  $(date) ==="
  echo "=========================================="

  if [ -f "results/shap_faithfulness_${arch}.json" ]; then
    cp "results/shap_faithfulness_${arch}.json" "results/shap_faithfulness_${arch}_preN500_backup.json"
  fi

  .venv/bin/python3 -m src.run_shap_explain --architecture "$arch" --n-samples 500 \
    2>&1 | tee "logs/shap_${arch}_n500.log"
  exit_code="${PIPESTATUS[0]}"
  echo "SHAP_N500_EXIT[$arch]=${exit_code}"

  if [ "$exit_code" -eq 0 ] && [ -f "results/shap_faithfulness_${arch}.json" ]; then
    mv "results/shap_faithfulness_${arch}.json" "results/shap_faithfulness_${arch}_n500.json"
  else
    echo "SKIPPING rename for $arch -- run failed"
  fi
done
