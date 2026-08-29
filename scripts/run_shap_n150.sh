#!/bin/bash
# Scales SHAP faithfulness from the n=15 pilot (2026-08-28, all 3 architectures) up
# to n=150, matching the sample size already used for Grad-CAM's faithfulness150
# results. Sequential -- single GPU. Backs up each architecture's n=15 result before
# running, and writes the n=150 result under a _n150 suffix so neither is lost.
#
# Usage: ./scripts/run_shap_n150.sh resnet50 efficientnetb4 vgg16
set -uo pipefail

cd /workspace/project
mkdir -p logs results

if [ "$#" -eq 0 ]; then
  echo "Usage: $0 <architecture> [architecture ...]"
  exit 1
fi

for arch in "$@"; do
  echo "=========================================="
  echo "=== SHAP n=150  $arch  $(date) ==="
  echo "=========================================="

  if [ -f "results/shap_faithfulness_${arch}.json" ]; then
    cp "results/shap_faithfulness_${arch}.json" "results/shap_faithfulness_${arch}_n15.json"
  fi

  .venv/bin/python3 -m src.run_shap_explain --architecture "$arch" --n-samples 150 \
    2>&1 | tee "logs/shap_${arch}_n150.log"
  exit_code="${PIPESTATUS[0]}"
  echo "SHAP_N150_EXIT[$arch]=${exit_code}"

  if [ "$exit_code" -eq 0 ] && [ -f "results/shap_faithfulness_${arch}.json" ]; then
    mv "results/shap_faithfulness_${arch}.json" "results/shap_faithfulness_${arch}_n150.json"
  else
    echo "SKIPPING rename for $arch -- run failed, leaving results/shap_faithfulness_${arch}.json (n=15) untouched"
  fi
done
