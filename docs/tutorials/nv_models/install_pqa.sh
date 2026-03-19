#!/usr/bin/env bash
set -euo pipefail

usage() {
    cat <<EOF
Usage: $(basename "$0") [OPTIONS]

Create a venv with uv and install PaperQA in editable mode.
Run this from anywhere inside the paper-qa repo.

Options:
  -p, --python  VER     Python version          (default: 3.11)
  -e, --extras  EXTRAS  pip extras, comma-sep   (default: pymupdf,nemotron)
  -v, --venv    NAME    Venv directory name      (default: .venv)
  --install-python      Install Python via deadsnakes PPA (Ubuntu/Debian)
  --jupyter             Register a Jupyter kernel after install
  -h, --help            Show this help message
EOF
    exit 0
}

PYVER="3.11"
EXTRAS="pymupdf,nemotron"
VENV=".venv"
INSTALL_PYTHON=false
JUPYTER=false

while [[ $# -gt 0 ]]; do
    case "$1" in
        -p|--python)         PYVER="$2";  shift 2 ;;
        -e|--extras)         EXTRAS="$2"; shift 2 ;;
        -v|--venv)           VENV="$2";   shift 2 ;;
        --install-python)    INSTALL_PYTHON=true; shift ;;
        --jupyter)           JUPYTER=true; shift ;;
        -h|--help)           usage ;;
        *)                   echo "Unknown option: $1" >&2; usage ;;
    esac
done

PYBIN="python${PYVER}"

# Find the repo root (where pyproject.toml lives)
REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)" \
    || { echo "Error: not inside a git repo" >&2; exit 1; }
cd "$REPO_ROOT"
echo "==> Repo root: $REPO_ROOT"

# ── 1. Install Python (optional) ────────────────────────────────────
if $INSTALL_PYTHON; then
    echo "==> Installing Python $PYVER via deadsnakes PPA..."
    sudo apt-get update -qq
    sudo apt-get install -y -qq software-properties-common
    sudo add-apt-repository -y ppa:deadsnakes/ppa
    sudo apt-get update -qq
    sudo apt-get install -y -qq "python${PYVER}" "python${PYVER}-venv" "python${PYVER}-dev"
    $PYBIN --version
fi

# ── 2. Install uv if needed ─────────────────────────────────────────
if ! command -v uv &>/dev/null; then
    echo "==> Installing uv..."
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.local/bin:$PATH"
fi

# ── 3. Create venv & install ────────────────────────────────────────
echo "==> Creating venv '$VENV' with Python $PYVER..."
uv venv --python "$PYVER" "$VENV"
source "$VENV/bin/activate"

echo "==> Installing paper-qa in editable mode with extras: [$EXTRAS]"
uv pip install -e ".[$EXTRAS]"

# ── 4. Sanity check ─────────────────────────────────────────────────
echo "==> Verifying installation..."
python --version
python -c "import paperqa; print('paperqa:', paperqa.__file__)"

# ── 5. Jupyter kernel (optional) ────────────────────────────────────
if $JUPYTER; then
    echo "==> Registering Jupyter kernel 'paperqa-dev'..."
    uv pip install ipykernel
    python -m ipykernel install --user --name=paperqa-dev --display-name="PaperQA Dev"
    jupyter kernelspec list
fi

echo ""
echo "Done! Activate the environment with:"
echo "  source $REPO_ROOT/$VENV/bin/activate"
