#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
KERNEL_NAME="${KERNEL_NAME:-unsupervised-local}"
MODE="${1:-lab}"

setup_env() {
  "$PYTHON_BIN" -m venv "$VENV_DIR"
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"

  python -m pip install --upgrade pip
  python -m pip install -r "$ROOT_DIR/requirements.txt" jupyterlab ipykernel nbconvert
  python -m ipykernel install --user --name "$KERNEL_NAME" --display-name "Python ($KERNEL_NAME)"
}

run_all() {
  local notebook
  for notebook in \
    "$ROOT_DIR/notebooks/01_data_download.ipynb" \
    "$ROOT_DIR/notebooks/02_preprocessing.ipynb" \
    "$ROOT_DIR/notebooks/03_eda.ipynb" \
    "$ROOT_DIR/notebooks/04_feature_extraction.ipynb" \
    "$ROOT_DIR/notebooks/05_dimensionality_reduction.ipynb" \
    "$ROOT_DIR/notebooks/06_clustering.ipynb"
  do
    echo "Executing $(basename "$notebook")"
    python -m jupyter nbconvert \
      --to notebook \
      --execute \
      --inplace \
      --ExecutePreprocessor.timeout=-1 \
      --ExecutePreprocessor.kernel_name="$KERNEL_NAME" \
      "$notebook"
  done
}

case "$MODE" in
  setup)
    setup_env
    echo "Environment ready in $VENV_DIR"
    ;;
  lab)
    setup_env
    exec python -m jupyter lab --notebook-dir "$ROOT_DIR"
    ;;
  run)
    setup_env
    run_all
    ;;
  *)
    echo "Usage: $0 [setup|lab|run]" >&2
    exit 1
    ;;
esac
