#!/usr/bin/env bash
# Runner for the W7-X twin.
#
# Ubuntu's MKL packaging ships dispatch libraries (libmkl_avx2.so) that fail to
# resolve against libmkl_core unless the whole set is preloaded; VMEC++ links MKL
# for LAPACK, so every entry point goes through here. A packaging that resolves
# them on its own, and any host that is not glibc Linux, needs no preload:
# set W7X_TWIN_MKL_DIR to the directory holding them, or to "none" to skip it.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${W7X_TWIN_PYTHON:-$ROOT/venv/bin/python}"
MKL="${W7X_TWIN_MKL_DIR:-/lib/x86_64-linux-gnu}"

if [ "$MKL" != "none" ] && [ -e "$MKL/libmkl_core.so" ]; then
  export LD_PRELOAD="$MKL/libmkl_def.so:$MKL/libmkl_avx2.so:$MKL/libmkl_core.so:$MKL/libmkl_intel_lp64.so:$MKL/libmkl_intel_thread.so:$MKL/libiomp5.so"
fi
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export KMP_WARNINGS=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

if [ ! -x "$PYTHON" ]; then
  echo "no interpreter at $PYTHON" >&2
  echo "create one with: python -m venv venv && venv/bin/pip install -e ." >&2
  echo "or name another with W7X_TWIN_PYTHON" >&2
  exit 1
fi

cd "$ROOT"
exec "$PYTHON" "$@"
