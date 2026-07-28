#!/usr/bin/env bash
# The radial drift-kinetic scan on a Boozer file covering every interior surface.
#
#   monkes_radial.sh <monkes_root> <boozmn> <run_dir> [threads]
#
# Twelve surfaces, eleven collisionalities reaching below what the energy convolution
# samples, and six electric fields spanning what E_r/v covers over it. A Boozer file
# that leaves out the innermost surfaces forces extrapolation there, which is worth
# tens of per cent in the coefficient, so the file must carry every interior surface.
set -euo pipefail

ROOT="${1:-$HOME/monkes}"
BOOZMN="${2:-$HOME/w7x-twin/cache/monkes_cases/standard_beta1/boozmn.nc}"
RUN="${3:-$HOME/w7x-twin/cache/monkes_radial}"
THREADS="${4:-${MKL_NUM_THREADS:-14}}"
MKL="${MKL_LIBRARY:-/usr/lib/x86_64-linux-gnu/libmkl_rt.so}"

SURFACES=(0.02 0.05 0.10 0.16 0.25 0.35 0.45 0.55 0.65 0.75 0.85 0.95)

mkdir -p "$RUN"
for SURFACE in "${SURFACES[@]}"; do
  DIR="$RUN/s$SURFACE"
  if [ -s "$DIR/monkes_Monoenergetic_Database.dat" ]; then
    echo "skip $SURFACE"
    continue
  fi
  rm -rf "$DIR"; mkdir -p "$DIR"
  cp "$BOOZMN" "$DIR/boozmn.nc"
  cp "$ROOT/bin/main_monkes.x" "$DIR/"

  cat > "$DIR/monkes_input.parameters" <<'PARAMS'
&parameters
N_theta = 31
N_zeta = 55
N_xi = 140
nu =
1.000000e-06,
3.000000e-06,
1.000000e-05,
3.000000e-05,
1.000000e-04,
3.000000e-04,
1.000000e-03,
3.000000e-03,
1.000000e-02,
3.000000e-02,
1.000000e-01
E_r = 0.000000e+00, 3.000000e-05, 1.000000e-04, 3.000000e-04, 1.000000e-03, 3.000000e-03
/
PARAMS

  printf '&surface\ns=%s\n/\n' "$SURFACE" > "$DIR/monkes_input.surface"

  echo "solving s = $SURFACE"
  START=$SECONDS
  if ( cd "$DIR" && LD_PRELOAD="$MKL" MKL_NUM_THREADS="$THREADS" \
       ./main_monkes.x > run.log 2>&1 ); then
    echo "  $(( $(wc -l < "$DIR/monkes_Monoenergetic_Database.dat") - 1 )) points" \
         "in $((SECONDS - START)) s"
  else
    echo "  failed after $((SECONDS - START)) s, see $DIR/run.log"
  fi
done
echo "done: $RUN"
