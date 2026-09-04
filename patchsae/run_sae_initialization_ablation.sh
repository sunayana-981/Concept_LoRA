#!/usr/bin/env bash
# Controlled three-arm SAE initialization ablation.
# All arguments are forwarded to the Python runner; use --print_only first.

set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${PROJECT_ROOT}"

exec python3 tasks/run_sae_initialization_ablation.py "$@"
