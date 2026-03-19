# Installing PaperQA (Editable Mode)

## Prerequisites

- **Python >= 3.11** (the script can install it for you with `--install-python`)
- **curl** (only needed if `uv` isn't installed yet)

## Quick start

From anywhere inside the `paper-qa` repo:

```bash
./docs/tutorials/nv_models/install_pqa.sh
```

This creates a `.venv` at the repo root and installs PaperQA with the default
extras (`pymupdf,nemotron`).

## Helper script — `install_pqa.sh`

```
./install_pqa.sh [OPTIONS]
```

| Flag | Description | Default |
|------|-------------|---------|
| `-p, --python VER` | Python version for the venv | `3.11` |
| `-e, --extras EXTRAS` | Comma-separated pip extras | `pymupdf,nemotron` |
| `--install-python` | Install Python via deadsnakes PPA (Ubuntu/Debian) | off |
| `--jupyter` | Register a Jupyter kernel after install | off |

The script automatically finds the repo root via `git rev-parse`, so you can
run it from any subdirectory.

## Examples

### Default install

```bash
./install_pqa.sh
```

### Custom extras

```bash
./install_pqa.sh --extras "pymupdf,nemotron,ldp"
```

### Install Python 3.11 + Jupyter kernel

```bash
./install_pqa.sh --install-python --jupyter
```

### Use Python 3.12

```bash
./install_pqa.sh --python 3.12
```

## Step-by-step (manual)

If you prefer to run each step yourself:

### 1. (Optional) Install Python 3.11

```bash
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev
python3.11 --version
```

### 2. Create venv and install

```bash
# Install uv if needed
curl -LsSf https://astral.sh/uv/install.sh | sh

# From the repo root
uv venv --python 3.11 .venv
source .venv/bin/activate

uv pip install -e ".[pymupdf,nemotron]"
```

### 3. Verify

```bash
python --version
python -c "import paperqa; print(paperqa.__file__)"
python -c "import paperqa_nemotron; print('OK')"
```

### 4. (Optional) Jupyter kernel

```bash
uv pip install ipykernel
python -m ipykernel install --user --name=paperqa-dev --display-name="PaperQA Dev"
jupyter kernelspec list
```
