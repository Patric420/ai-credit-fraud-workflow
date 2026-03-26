#!/usr/bin/env bash
set -euo pipefail

cd /mnt/c/workhub/ads/ai-credit-fraud-workflow
source .venv-wsl/bin/activate

python - <<'PY'
import cudf
import cuml
print("cudf", cudf.__version__)
print("cuml", cuml.__version__)
PY

python fraud_pipeline_cpu_gpu_benchmark.py --rows 500000 --output-dir artifacts
