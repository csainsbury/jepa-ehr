#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: download_decrypt_data_bundle.sh --url URL --sha256 SHA --dest DIR [options]
       download_decrypt_data_bundle.sh --input-file FILE --sha256 SHA --dest DIR [options]

Options:
  --passfile PATH   Passphrase file (default /root/.config/clinical-jepa/data-transfer.pass)
  --work-dir DIR    Temporary work dir (default /workspace/data-transfer-work)
  --strip N         Tar strip components (default 1)
  --preflight       Run data_bundle_preflight.py --strict after unpack if available

The encrypted input must be produced by pack_encrypt_upload_data.sh using:
  openssl enc -aes-256-cbc -pbkdf2 -iter 600000 -md sha256 -salt
USAGE
}

URL=""
INPUT_FILE=""
SHA256=""
DEST=""
PASSFILE="/root/.config/clinical-jepa/data-transfer.pass"
WORK_DIR="/workspace/data-transfer-work"
STRIP=1
RUN_PREFLIGHT=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --url) URL="$2"; shift 2 ;;
    --input-file) INPUT_FILE="$2"; shift 2 ;;
    --sha256) SHA256="$2"; shift 2 ;;
    --dest) DEST="$2"; shift 2 ;;
    --passfile) PASSFILE="$2"; shift 2 ;;
    --work-dir) WORK_DIR="$2"; shift 2 ;;
    --strip) STRIP="$2"; shift 2 ;;
    --preflight) RUN_PREFLIGHT=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown arg: $1" >&2; usage >&2; exit 2 ;;
  esac
done

if [[ -z "$SHA256" || -z "$DEST" ]]; then
  usage >&2; exit 2
fi
if [[ -z "$URL" && -z "$INPUT_FILE" ]]; then
  echo "Need --url or --input-file" >&2; exit 2
fi
if [[ ! -f "$PASSFILE" ]]; then
  echo "Missing passphrase file: $PASSFILE" >&2; exit 3
fi

mkdir -p "$WORK_DIR" "$DEST"
ENC="$WORK_DIR/bundle.tar.zst.enc"
TAR="$WORK_DIR/bundle.tar.zst"

if [[ -n "$INPUT_FILE" ]]; then
  cp "$INPUT_FILE" "$ENC"
else
  curl -fL "$URL" -o "$ENC"
fi

echo "$SHA256  $ENC" | sha256sum -c -
openssl enc -d -aes-256-cbc -pbkdf2 -iter 600000 -md sha256 -in "$ENC" -out "$TAR" -pass "file:$PASSFILE"

# Tar safety: no absolute paths or parent traversal.
if tar -I zstd -tf "$TAR" | grep -E '(^/|(^|/)\.\.(/|$))' >/tmp/clinical_jepa_bad_tar.$$; then
  echo "Unsafe paths in tar:" >&2
  cat /tmp/clinical_jepa_bad_tar.$$ >&2
  rm -f /tmp/clinical_jepa_bad_tar.$$
  exit 4
fi
rm -f /tmp/clinical_jepa_bad_tar.$$

tar -I zstd -xf "$TAR" -C "$DEST" --strip-components="$STRIP"

if [[ "$RUN_PREFLIGHT" == "1" ]]; then
  if [[ -f /workspace/clinical-jepa-autonomous-run/scripts/data_bundle_preflight.py ]]; then
    python3 /workspace/clinical-jepa-autonomous-run/scripts/data_bundle_preflight.py --root "$DEST" --strict --report "$DEST/data-bundle-preflight.json"
  elif [[ -f ./scripts/data_bundle_preflight.py ]]; then
    python3 ./scripts/data_bundle_preflight.py --root "$DEST" --strict --report "$DEST/data-bundle-preflight.json"
  else
    echo "WARN: data_bundle_preflight.py not found; skipping" >&2
  fi
fi

echo "Decrypted and unpacked bundle to: $DEST"
