### 1. Clone Your Fork (or the main repo)
git clone https://github.com/hw-ju/paper-qa.git
cd paper-qa

# Switch to your branch if needed
git checkout selfhost_NIMs
git branch


### 2. (Optional) Install Python 3.11 (if not already installed)
# paper-qa requires >=3.11
# Commands below were used on a Brev instance with only Python 3.10 pre-installed.

# Ubuntu/Debian: add deadsnakes PPA and install Python 3.11
sudo apt update
sudo apt install -y software-properties-common
sudo add-apt-repository -y ppa:deadsnakes/ppa
sudo apt update
sudo apt install -y python3.11 python3.11-venv python3.11-dev

# Verify
python3.11 --version


### 3. Install in Editable Mode (with uv)
# Install uv if needed:
# curl -LsSf https://astral.sh/uv/install.sh | sh

# Create a Python 3.11 virtual environment with uv and activate it:
cd paper-qa
uv venv --python 3.11 .venv
source .venv/bin/activate

# Verify active Python
python --version

# Install PaperQA in editable mode with required extras:
# - pymupdf: PDF fallback when nemotron-parse fails on a page
# - nemotron: paper-qa-nemotron (parse_pdf_to_pages, LiteLLM call to nemotron-parse NIM)
uv pip install -e ".[pymupdf,nemotron]"

# Notebook support
uv pip install ipykernel

# Sanity check
python -c "import paperqa; import paperqa_nemotron; print('OK')"


### 4. Verify Installation
### Check Python Environment
which python
python --version

# Check installed packages
uv pip list | grep paper-qa

# Test Import
python -c "import paperqa; print(paperqa.__file__)"
# Expected output: path to `.../paper-qa/src/paperqa/__init__.py` under your repo


### 5. Register Jupyter Kernel
# Register the environment as a Jupyter kernel
python -m ipykernel install --user --name=paperqa-dev --display-name="PaperQA Dev"

# List Available Kernels
jupyter kernelspec list