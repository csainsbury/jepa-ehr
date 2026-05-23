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
