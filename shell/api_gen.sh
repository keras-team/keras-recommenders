#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

echo "Generating api directory with public APIs..."
# Generate API Files
python3 "${base_dir}"/api_gen.py
# Format code because `api_gen.py` might order
# imports differently.
pre-commit run --files $(find "${base_dir}"/keras_rs/api -type f) || true
