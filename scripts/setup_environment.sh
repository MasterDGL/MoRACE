#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEFAULT_PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
CONDA_BASE="${CONDA_BASE:-}"
CONDA_ENV_NAME="${CONDA_ENV_NAME:-morace}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
INSTALL_DEV="${INSTALL_DEV:-1}"
MODEL_REQUIREMENTS="${MODEL_REQUIREMENTS:-}"
PROJECT_ROOT="${PROJECT_ROOT:-${DEFAULT_PROJECT_ROOT}}"
DATA_ROOT="${DATA_ROOT:-${PROJECT_ROOT}/.runtime}"

if [[ -z "${CONDA_BASE}" ]] && command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null || true)"
fi

if [[ -n "${CONDA_BASE}" && -f "${CONDA_BASE}/etc/profile.d/conda.sh" ]]; then
  # shellcheck disable=SC1090
  source "${CONDA_BASE}/etc/profile.d/conda.sh"
elif command -v conda >/dev/null 2>&1; then
  eval "$(conda shell.bash hook)"
else
  echo "conda not found"
  exit 1
fi

mkdir -p "${DATA_ROOT}/pip_cache"
mkdir -p "${DATA_ROOT}/pkgs"
mkdir -p "${DATA_ROOT}/.cache/huggingface/hub"
mkdir -p "${DATA_ROOT}/.cache/huggingface/transformers"
mkdir -p "${PROJECT_ROOT}"

export CONDA_PKGS_DIRS="${DATA_ROOT}/pkgs"
export PIP_CACHE_DIR="${DATA_ROOT}/pip_cache"
export MORACE_DATASET_ROOT="${PROJECT_ROOT}/data/splits"
export LLM4SD_DATASET_ROOT="${MORACE_DATASET_ROOT}"
export HF_HOME="${DATA_ROOT}/.cache/huggingface"
export TRANSFORMERS_CACHE="${DATA_ROOT}/.cache/huggingface/transformers"
export HUGGINGFACE_HUB_CACHE="${DATA_ROOT}/.cache/huggingface/hub"
export TMPDIR="${DATA_ROOT}/pkgs/tmp"
mkdir -p "${TMPDIR}"

if ! conda env list | awk '{print $1}' | grep -Fxq "${CONDA_ENV_NAME}"; then
  conda create -y -n "${CONDA_ENV_NAME}" "python=${PYTHON_VERSION}"
fi

conda activate "${CONDA_ENV_NAME}"

python -m pip install --upgrade pip setuptools wheel
if [[ "${INSTALL_DEV}" == "1" ]]; then
  python -m pip install -r "${PROJECT_ROOT}/requirements-dev.txt"
else
  python -m pip install -r "${PROJECT_ROOT}/requirements.txt"
fi
python -m pip install -e "${PROJECT_ROOT}"
if [[ -n "${MODEL_REQUIREMENTS}" ]]; then
  python -m pip install -r "${PROJECT_ROOT}/${MODEL_REQUIREMENTS}"
fi

python - <<'PY'
import importlib

modules = [
    "pandas",
    "numpy",
    "scipy",
    "sklearn",
]
for name in modules:
    importlib.import_module(name)

try:
    import rdkit
    print("rdkit ok", getattr(rdkit, "__version__", "unknown"))
except Exception as exc:
    raise SystemExit(f"rdkit import failed: {exc}")

PY

if [[ -n "${MODEL_REQUIREMENTS}" ]]; then
  python "${PROJECT_ROOT}/scripts/verify_repository.py" --runtime
fi

echo "[Setup] MoRACE core environment is ready."
echo "[Setup] Conda environment: ${CONDA_ENV_NAME}"
echo "[Setup] Data root: ${DATA_ROOT}"
echo "[Setup] Project root: ${PROJECT_ROOT}"
if [[ -z "${MODEL_REQUIREMENTS}" ]]; then
  echo "[Setup] Model-generation dependencies were not installed."
fi
