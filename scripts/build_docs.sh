#!/usr/bin/env bash
# Build documentation script

set -e

echo "Installing documentation dependencies..."
uv pip install -e ".[docs]"

echo "Cleaning previous builds..."
cd docs
rm -rf _build

echo "Building HTML documentation..."
make html

echo "Documentation built successfully!"
echo "Open docs/_build/html/index.html in your browser"
