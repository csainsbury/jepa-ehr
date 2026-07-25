#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "== Clinical-JEPA Vast bootstrap =="
echo "Root: $ROOT"
date -u '+UTC %Y-%m-%dT%H:%M:%SZ'

if command -v nvidia-smi >/dev/null 2>&1; then
  echo "== GPU =="
  nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader || true
else
  echo "WARN: nvidia-smi not found; are NVIDIA drivers exposed in the container?" >&2
fi

echo "== Disk =="
df -h . || true

echo "== Python =="
python3 --version || true

if ! command -v uv >/dev/null 2>&1; then
  echo "uv not found; attempting user install"
  curl -LsSf https://astral.sh/uv/install.sh | sh || true
  export PATH="$HOME/.local/bin:$PATH"
fi
command -v uv >/dev/null 2>&1 && uv --version || echo "WARN: uv unavailable"

echo "== Python dependencies =="
# The remote entry point previously installed NOTHING, and `pip install -e .` pulls core deps only
# (numpy, pyyaml). Every HDF5-backed path — target-block extraction and both rollout exporters —
# hard-requires h5py, so a run would train successfully and then fail at export. Install the extras.
PY_BIN="${PYTHON:-python3}"
if "$PY_BIN" -m pip install -e ".[data,torch]"; then
  echo "installed extras: data (h5py) + torch"
else
  echo "WARN: editable install with extras failed; retrying core + h5py only" >&2
  "$PY_BIN" -m pip install -e . && "$PY_BIN" -m pip install "h5py>=3.10" || \
    { echo "ERROR: could not install h5py — HDF5 export paths WILL fail" >&2; exit 1; }
fi
"$PY_BIN" - <<'PYCHK'
import importlib.util, sys
missing = [m for m in ("numpy", "yaml", "h5py") if importlib.util.find_spec(m) is None]
print("dependency check:", "OK" if not missing else f"MISSING {missing}")
if missing:
    sys.exit(1)
try:
    import torch
    print(f"torch {torch.__version__} | cuda {torch.cuda.is_available()}")
except ImportError:
    print("WARN: torch absent — training/export unavailable, scoring layers still fine")
PYCHK

mkdir -p run-workspace logs repos data outputs checkpoints

cat > .env.example <<'ENV'
# Optional paths for later implementation. Do not commit secrets.
ASCEND_FLAT_REPO=https://github.com/csainsbury/ascend-flat.git
CLINICAL_JEPA_WORKSPACE=run-workspace
CLINICAL_JEPA_DATA_ROOT=/workspace/data
CLINICAL_JEPA_OUTPUT_ROOT=/workspace/outputs
# B2 credentials should be supplied only as environment variables if needed.
# B2_APPLICATION_KEY_ID=
# B2_APPLICATION_KEY=
# B2_BUCKET=
ENV

echo "Bootstrap complete. Next: python3 scripts/create_protocol_workspace.py --root \"$ROOT/run-workspace\""
