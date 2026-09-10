#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

cd "${REPO_ROOT}"

echo "========================================"
echo "LLM Engineering — GPU Environment Setup"
echo "========================================"

echo
echo "Repository: ${REPO_ROOT}"

echo
echo "=== Required commands ==="

for command in uv git nvidia-smi; do
    if ! command -v "${command}" >/dev/null 2>&1; then
        echo "ERROR: required command '${command}' was not found."
        exit 1
    fi

    echo "Found: ${command}"
done

echo
echo "=== System ==="
uname -a

echo
echo "=== NVIDIA driver and GPU ==="
nvidia-smi

echo
echo "=== Python environment ==="

# Use the committed lockfile without modifying dependency resolution.
uv sync --frozen

echo
uv run python --version

echo
echo "=== PyTorch CUDA validation ==="

uv run python - <<'PY'
import sys

import torch

print(f"Python: {sys.version.split()[0]}")
print(f"PyTorch: {torch.__version__}")
print(f"PyTorch CUDA runtime: {torch.version.cuda}")
print(f"CUDA available: {torch.cuda.is_available()}")

if not torch.cuda.is_available():
    raise RuntimeError(
        "PyTorch cannot access a CUDA-capable GPU."
    )

device_count = torch.cuda.device_count()

if device_count <= 0:
    raise RuntimeError(
        "CUDA is available but no CUDA devices were detected."
    )

print(f"CUDA device count: {device_count}")

for index in range(device_count):
    properties = torch.cuda.get_device_properties(index)

    memory_gib = properties.total_memory / (1024**3)

    print(
        f"GPU {index}: "
        f"{properties.name} "
        f"({memory_gib:.2f} GiB)"
    )

# Allocate a tiny tensor so validation covers an actual CUDA operation,
# rather than only CUDA device discovery.
tensor = torch.ones(1, device="cuda")

if tensor.device.type != "cuda":
    raise RuntimeError(
        "CUDA tensor allocation did not use a CUDA device."
    )

print("CUDA tensor allocation: OK")
PY

echo
echo "=== Benchmark output directory ==="

mkdir -p benchmark_results

echo "Ready: ${REPO_ROOT}/benchmark_results"

echo
echo "========================================"
echo "GPU environment is ready."
echo "========================================"