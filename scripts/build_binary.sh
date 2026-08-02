#!/usr/bin/env bash
set -euo pipefail

# Compiles the identity-api server into a standalone native executable with
# Nuitka, so the service can be distributed as a binary instead of readable
# Python source.
#
# Usage:
#   scripts/build_binary.sh [output_dir]     (default output_dir: dist/)
#
# Requires nuitka + ordered-set in the active Python environment:
#   pip install nuitka ordered-set
#
# Notes:
# - Excludes the optional `openvino` (facial-ir) backend: the installed wheel
#   ships a malformed dylib that breaks Nuitka's macOS linking step. The
#   default onnxruntime (CoreML/CPU) backend is unaffected.
# - --static-libpython=no is required on pyenv/Homebrew Python builds, which
#   don't ship a static libpython; harmless elsewhere.
# - Nuitka does not cross-compile: run this on the OS/architecture you intend
#   to ship (e.g. inside a matching Docker container for Linux targets).
# - assets/ is not embedded in the binary. Distribute it alongside the
#   executable and point IDENTITY_ANALYSIS_ASSETS at it, same as the Docker
#   image does.

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUTPUT_DIR="${1:-$REPO_ROOT/dist}"
mkdir -p "$OUTPUT_DIR"
OUTPUT_DIR="$(cd "$OUTPUT_DIR" && pwd)"
ENTRY_FILE="$OUTPUT_DIR/server_entry.py"

cd "$REPO_ROOT"

if [ -f .venv/bin/activate ] && [ -z "${VIRTUAL_ENV:-}" ]; then
  # shellcheck disable=SC1091
  source .venv/bin/activate
fi

if ! python -c "import nuitka" >/dev/null 2>&1; then
  echo "Nuitka is not installed in this environment. Install it with:" >&2
  echo "  pip install nuitka ordered-set" >&2
  exit 1
fi

cat > "$ENTRY_FILE" <<'EOF'
from identity_analysis.api import main

if __name__ == "__main__":
    main()
EOF

python -m nuitka \
  --standalone \
  --static-libpython=no \
  --follow-imports \
  --nofollow-import-to=openvino \
  --include-package=identity_analysis \
  --include-package=uvicorn \
  --include-package=fastapi \
  --include-package=starlette \
  --include-package=onnxruntime \
  --include-package=PIL \
  --include-package-data=onnxruntime \
  --include-package-data=PIL \
  --output-dir="$OUTPUT_DIR" \
  --assume-yes-for-downloads \
  "$ENTRY_FILE"

BINARY="$OUTPUT_DIR/server_entry.dist/server_entry.bin"
echo
echo "Binary built at: $BINARY"
echo "Run it with, e.g.:"
echo "  IDENTITY_ANALYSIS_ASSETS=\"$REPO_ROOT/assets\" \"$BINARY\""
