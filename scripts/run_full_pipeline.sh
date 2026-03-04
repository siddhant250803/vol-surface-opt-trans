#!/bin/bash
# Run full pipeline + regenerate report visuals.
# Requires: data at paths in configs/data.yaml
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

echo "Running pipeline (--skip-cache)..."
python -m pipeline.run --config configs/base.yaml --skip-cache

echo "Generating OT report visuals..."
PYTHONPATH=. python scripts/generate_ot_report.py

echo "Generating Gaussian W1/W2 demo..."
python scripts/gaussian_w1_w2_visualization.py

echo "Done. Outputs: outputs/report/ot_findings/ outputs/report/factor/"
