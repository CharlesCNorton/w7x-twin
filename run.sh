#!/usr/bin/env bash
# Runner for the W7-X twin.
#
# Ubuntu's MKL packaging ships dispatch libraries (libmkl_avx2.so) that fail to
# resolve against libmkl_core unless the whole set is preloaded; VMEC++ links MKL
# for LAPACK, so every entry point goes through here.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MKL=/lib/x86_64-linux-gnu
export LD_PRELOAD="$MKL/libmkl_def.so:$MKL/libmkl_avx2.so:$MKL/libmkl_core.so:$MKL/libmkl_intel_lp64.so:$MKL/libmkl_intel_thread.so:$MKL/libiomp5.so"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-12}"
export KMP_WARNINGS=0
export PYTHONPATH="$ROOT/src:${PYTHONPATH:-}"

cd "$ROOT"
exec "$ROOT/venv/bin/python" "$@"
