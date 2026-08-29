#!/usr/bin/env bash
# The MONKES drift-kinetic runs, one script with a mode per question.
#
#   monkes_table.sh radial    <monkes_root> <boozmn> <run_dir> [threads]
#   monkes_table.sh scan      <monkes_root> <boozmn> <surface> <run_dir>
#   monkes_table.sh cases     <monkes_root> <cases_dir> [threads]
#   monkes_table.sh converged <monkes_root> <run_dir>
#   monkes_table.sh targeted  <monkes_root> <run_dir>
#
# radial: the radial scan every consumer reads, twelve surfaces at eleven
# collisionalities reaching below what the energy convolution samples and six electric
# fields spanning what E_r/v covers over it. A Boozer file that leaves out the
# innermost surfaces forces extrapolation there, worth tens of per cent in the
# coefficient, so the file must carry every interior surface.
# scan: the collisionality scan on one flux surface at 21 x 43 x 100, affordable across
# the whole range, its lowest point checked against a converged run.
# cases: two collisionalities on twelve surfaces for each prepared case directory, the
# ratio comparison against the effective-ripple scaling on more than one equilibrium.
# converged: the transport-carrying surfaces at 41 x 75 x 180, where successive
# refinement moves D_11 by 0.7 % and D_31 by 0.8 %, and the reference configuration
# MONKES distributes, so the D_11 difference against the published values is measured
# on that configuration.
# targeted: ratio stability at converged resolution, three surfaces against the same
# three of the radial scan, one collisionality wide where one suffices.
#
# MONKES carries no OpenMP directives, so its thread count is set by the BLAS and not
# by OMP_NUM_THREADS. Preloading a threaded MKL over the reference library it links
# runs the same arithmetic to eleven significant figures 31 times faster.
set -euo pipefail

MODE="${1:?mode: radial | scan | cases | converged | targeted}"
shift

MKL="${MKL_LIBRARY:-/usr/lib/x86_64-linux-gnu/libmkl_rt.so}"
THREADS="${MKL_NUM_THREADS:-18}"

solve () {  # dir boozmn surface n_theta n_zeta n_xi nu-block er-block skip-if-done
  local dir="$1" boozmn="$2" surface="$3" n_theta="$4" n_zeta="$5" n_xi="$6"
  local nu="$7" er="$8" skip="$9"
  if [ "$skip" = "yes" ] && [ -s "$dir/monkes_Monoenergetic_Database.dat" ]; then
    echo "  skip $(basename "$dir")"
    return 0
  fi
  rm -rf "$dir"; mkdir -p "$dir"
  cp "$boozmn" "$dir/boozmn.nc"
  cp "$ROOT/bin/main_monkes.x" "$dir/"
  {
    echo "&parameters"
    echo "N_theta = $n_theta"
    echo "N_zeta = $n_zeta"
    echo "N_xi = $n_xi"
    echo "nu ="
    echo "$nu"
    echo "E_r = $er"
    echo "/"
  } > "$dir/monkes_input.parameters"
  printf '&surface\ns=%s\n/\n' "$surface" > "$dir/monkes_input.surface"

  local started=$SECONDS
  if ( cd "$dir" && LD_PRELOAD="$MKL" MKL_NUM_THREADS="$THREADS" \
       ./main_monkes.x > run.log 2>&1 ); then
    echo "  $(basename "$dir"): $(( $(wc -l < "$dir/monkes_Monoenergetic_Database.dat") - 1 ))" \
         "points in $((SECONDS - started)) s"
  else
    echo "  $(basename "$dir"): failed after $((SECONDS - started)) s, see $dir/run.log"
  fi
}

case "$MODE" in
radial)
  ROOT="${1:-$HOME/monkes}"
  BOOZMN="${2:-$HOME/w7x-twin/cache/monkes_cases/standard_beta1/boozmn.nc}"
  RUN="${3:-$HOME/w7x-twin/cache/monkes_radial}"
  THREADS="${4:-${MKL_NUM_THREADS:-14}}"
  SURFACES=(0.02 0.05 0.10 0.16 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
  mkdir -p "$RUN"
  for SURFACE in "${SURFACES[@]}"; do
    solve "$RUN/s$SURFACE" "$BOOZMN" "$SURFACE" 31 55 140 "1.000000e-06,
3.000000e-06,
1.000000e-05,
3.000000e-05,
1.000000e-04,
3.000000e-04,
1.000000e-03,
3.000000e-03,
1.000000e-02,
3.000000e-02,
1.000000e-01" \
      "0.000000e+00, 3.000000e-05, 1.000000e-04, 3.000000e-04, 1.000000e-03, 3.000000e-03" yes
  done
  echo "done: $RUN"
  ;;
scan)
  ROOT="${1:-$HOME/monkes}"
  BOOZMN="${2:-$ROOT/run_bz/boozmn.nc}"
  SURFACE="${3:-0.2}"
  RUN="${4:-$ROOT/run_scan}"
  solve "$RUN" "$BOOZMN" "$SURFACE" 21 43 100 "1.000000e-05,
3.000000e-05,
1.000000e-04,
3.000000e-04,
1.000000e-03,
3.000000e-03,
1.000000e-02,
3.000000e-02,
1.000000e-01,
3.000000e-01,
1.000000e+00" "0.000000e+00" no
  echo "done: $RUN/monkes_Monoenergetic_Database.dat"
  ;;
cases)
  ROOT="${1:-$HOME/monkes}"
  CASES="${2:-$HOME/w7x-twin/cache/monkes_cases}"
  THREADS="${3:-$THREADS}"
  SURFACES=(0.02 0.05 0.10 0.16 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)
  for CASE_DIR in "$CASES"/*/; do
    CASE="$(basename "$CASE_DIR")"
    if [ ! -s "$CASE_DIR/boozmn.nc" ]; then
      echo "skip $CASE (no boozmn.nc)"
      continue
    fi
    echo "case $CASE"
    for SURFACE in "${SURFACES[@]}"; do
      solve "$CASE_DIR/s$SURFACE" "$CASE_DIR/boozmn.nc" "$SURFACE" 31 55 140 \
        "1.000000e-06,
1.000000e-05" "0.000000e+00" yes
    done
  done
  echo "done: $CASES"
  ;;
converged)
  ROOT="${1:-$HOME/monkes}"
  RUN="${2:-$HOME/monkes/run_converged}"
  NU="1.000000e-06,
1.000000e-05,
1.000000e-04,
1.000000e-03,
1.000000e-02,
1.000000e-01"
  mkdir -p "$RUN"
  echo "converged resolution on this equilibrium"
  for SURFACE in 0.05 0.16 0.35 0.55 0.75 0.95; do
    solve "$RUN/s$SURFACE" "$ROOT/run_bz/boozmn.nc" "$SURFACE" 41 75 180 \
      "$NU" "0.000000e+00, 1.000000e-04, 1.000000e-03" yes
  done
  REFERENCE=""
  for candidate in "$ROOT/run_ref/boozmn.nc" "$ROOT/examples/boozmn.nc" \
                   "$ROOT/run_ref/boozmn_w7x_ref.nc"; do
    [ -f "$candidate" ] && REFERENCE="$candidate" && break
  done
  if [ -n "$REFERENCE" ]; then
    echo "reference configuration from $REFERENCE"
    solve "$RUN/reference_s0.2" "$REFERENCE" 0.2 41 75 180 \
      "$NU" "0.000000e+00, 1.000000e-04, 1.000000e-03" yes
  else
    echo "no reference Boozer file found under $ROOT; skipping that comparison"
  fi
  echo "done: $RUN"
  ;;
targeted)
  ROOT="${1:-$HOME/monkes}"
  RUN="${2:-$HOME/monkes/run_targeted}"
  THREADS="${MKL_NUM_THREADS:-20}"
  mkdir -p "$RUN"
  echo "the reference configuration MONKES distributes, s = 0.2"
  solve "$RUN/reference_s0.2" "$ROOT/run_ref/boozmn.nc" 0.2 41 75 180 \
    "1.000000e-05,
1.000000e-04" "0.000000e+00" yes
  echo "this equilibrium across the profile, for ratio stability"
  for SURFACE in 0.05 0.16 0.85; do
    solve "$RUN/s$SURFACE" "$ROOT/run_bz/boozmn.nc" "$SURFACE" 41 75 180 \
      "1.000000e-05" "0.000000e+00" yes
  done
  echo "done: $RUN"
  ;;
*)
  echo "unknown mode: $MODE (radial | scan | cases | converged | targeted)" >&2
  exit 2
  ;;
esac
