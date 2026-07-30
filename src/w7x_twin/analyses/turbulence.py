"""Gyrokinetic stability, the growth-rate grid, the saturation response, and the computed balance."""

from __future__ import annotations

import concurrent.futures
import json
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.analyses._common import args, write_record
from w7x_twin.mhd.equilibrium import REFERENCE, SCAN, Twin
from w7x_twin.plasma import kinetics, neoclassical, transport

#: The stella binary every run shells out to.
STELLA = Path.home() / "src/stella/build/gnu/COMPILATION/stella"
#: Poloidal extent of the flux tube, in field periods; W7-X tubes need several to close.
FIELD_PERIODS = 5.0
#: Simulation time of a linear run; autostop ends one sooner once its growth rate settles.
TEND = 400.0
#: The records the entries write and read between them.
GRID_RECORD = Path("results/turbulence/growth_rate_grid.json")
CONSTANT_RECORD = Path("results/turbulence/mixing_length_constant.json")


# -- gyrokinetic -------------------------------------------------------------------

# Linear flux-tube stability per configuration: spectrum and threshold from queued stella runs.
#
#     python -m w7x_twin gyrokinetic [configuration ...]

LINEAR_WORK = Path("cache/stella")
GYROKINETIC_OUT = Path("results/turbulence/gyrokinetic.json")

#: Flux surface the tube sits on, as normalised toroidal flux.
TORFLUX = 0.64

#: Binormal wavenumbers, normalised to the ion gyroradius.
KY_VALUES = (0.4, 0.7, 1.0, 1.4)
#: Normalised temperature gradients a/L_T, spanning the ion-temperature-gradient
#: threshold, which for W7-X sits near 2.
LINEAR_GRADIENTS = (1.0, 2.0, 3.0, 4.0)

INPUT_TEMPLATE = """\
&geometry_options
  geometry_option = 'vmec'
/
&geometry_vmec
  vmec_filename = '{wout}'
  torflux = {torflux}
  nfield_periods = {periods}
  zeta_center = 0.0
  alpha0 = 0.0
  surface_option = 0
  verbose = .false.
/
&gyrokinetic_terms
  include_nonlinear = .false.
  include_parallel_streaming = .true.
  include_mirror = .true.
/
&species_options
  nspec = 2
/
&species_parameters_1
  z = 1.0
  mass = 1.0
  dens = 1.0
  temp = 1.0
  tprim = {tprim}
  fprim = 1.0
  type = 'ion'
/
&species_parameters_2
  z = -1.0
  mass = 5.43867E-04
  dens = 1.0
  temp = 1.0
  tprim = {tprim}
  fprim = 1.0
  type = 'electron'
/
&kxky_grid_option
  grid_option = 'range'
/
&kxky_grid_range
  naky = 1
  nakx = 1
  aky_min = {ky}
  aky_max = {ky}
  akx_min = 0.0
  akx_max = 0.0
/
&z_grid
  nzed = {nzed}
  nperiod = 1
  zed_equal_arc = .true.
/
&z_boundary_condition
  boundary_option = 'zero'
/
&velocity_grids
  nvgrid = 24
  nmu = 12
  vpa_max = 3.0
  vperp_max = 3.0
/
&diagnostics
  nwrite = 20
  save_for_restart = .false.
/
&diagnostics_omega
  write_omega_vs_kxky = .true.
  write_omega_avg_vs_kxky = .true.
/
&initialise_distribution
  initialise_distribution_option = 'default'
  phiinit = 0.01
/
&time_trace_options
  tend = {tend}
  autostop = .true.
/
&time_step
  delt = {delt}
/
&numerical_algorithms
  explicit_algorithm = 'rk3'
/
&debug_flags
  print_extra_info_to_terminal = .false.
/
"""


def write_wout(twin: Twin, configuration: str, directory: Path) -> Path:
    """The configuration's equilibrium as a VMEC output stella can read."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"wout_{configuration}.nc"
    if not path.exists():
        output = twin.solve(twin.state(configuration), SCAN)
        output.wout.save(path)
    return path


def read_growth_rate(directory: Path, stem: str) -> tuple[float, float]:
    """Dominant-mode growth rate and frequency from the tail of stella's averaged columns."""
    path = directory / f"{stem}.omega"
    if not path.exists():
        return float("nan"), float("nan")
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 6 or parts[0].startswith("#"):
            continue
        try:
            rows.append([float(value) for value in parts])
        except ValueError:
            continue
    if not rows:
        return float("nan"), float("nan")
    table = np.array(rows)
    tail = table[max(0, len(table) - 8) :]
    return float(np.nanmean(tail[:, 6])), float(np.nanmean(tail[:, 5]))


def linear_case(
    wout: Path, configuration: str, ky: float, tprim: float, verbose: bool = True
) -> dict:
    """One linear run at a single binormal wavenumber and gradient."""
    stem = f"{configuration}_ky{ky:g}_t{tprim:g}".replace(".", "p")
    directory = LINEAR_WORK / stem
    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shutil.copy(wout, directory / wout.name)

    # A finer parallel grid and a smaller step are needed as the mode gets shorter.
    # The run is long enough for the dominant eigenmode to emerge; autostop ends it
    # sooner when the growth rate has settled.
    nzed = 96 if ky <= 1.0 else 128
    delt = 0.05 if ky <= 1.0 else 0.02
    (directory / f"{stem}.in").write_text(
        INPUT_TEMPLATE.format(
            wout=wout.name, torflux=TORFLUX, periods=FIELD_PERIODS, ky=ky,
            tprim=tprim, nzed=nzed, delt=delt, tend=TEND,
        )
    )

    started = time.monotonic()
    completed = subprocess.run(
        [str(STELLA), f"{stem}.in"],
        cwd=directory,
        capture_output=True,
        text=True,
        check=False,
    )
    gamma, omega = read_growth_rate(directory, stem)
    elapsed = time.monotonic() - started
    if verbose:
        status = "ok" if completed.returncode == 0 else f"exit {completed.returncode}"
        print(
            f"  {configuration:22s} ky {ky:4.1f}  a/L_T {tprim:4.1f}  "
            f"gamma {gamma:9.4f}  omega {omega:9.4f}  {elapsed:6.1f} s  {status}"
        )
    return {
        "configuration": configuration,
        "ky": ky,
        "tprim": tprim,
        "growth_rate": gamma,
        "frequency": omega,
        "seconds": elapsed,
        "returncode": completed.returncode,
        "stderr": completed.stderr[-400:] if completed.returncode else "",
    }


def gyro_bohm(temperature_ev: float, field_t: float, minor_radius_m: float) -> float:
    """Gyro-Bohm diffusivity rho_i^2 v_ti / a, in m^2/s, the unit stella works in."""
    mass = 1.67262192369e-27
    charge = 1.602176634e-19
    thermal_speed = np.sqrt(2.0 * temperature_ev * charge / mass)
    gyroradius = mass * thermal_speed / (charge * field_t)
    return gyroradius**2 * thermal_speed / minor_radius_m


def quasilinear(rows: list[dict], configuration: str, tprim: float) -> float:
    """Mixing-length sum of gamma / k_y^2 over the spectrum, in gyro-Bohm units."""
    total = 0.0
    for ky in KY_VALUES:
        gamma = next(
            (
                r["growth_rate"]
                for r in rows
                if r["configuration"] == configuration
                and r["ky"] == ky
                and r["tprim"] == tprim
            ),
            float("nan"),
        )
        if np.isfinite(gamma) and gamma > 0:
            total += gamma / ky**2
    return total


def run_gyrokinetic() -> int:
    if not STELLA.exists():
        raise SystemExit(f"no stella binary at {STELLA}")
    configurations = args() or ["standard", "high_mirror_ref167"]

    twin = _common.twin()
    print(f"{twin.geometry}")
    print(f"flux tube on torflux {TORFLUX} over {FIELD_PERIODS} field periods")

    rows = []
    for configuration in configurations:
        wout = write_wout(twin, configuration, LINEAR_WORK / "geometry")
        for tprim in LINEAR_GRADIENTS:
            for ky in KY_VALUES:
                rows.append(linear_case(wout, configuration, ky, tprim))

    write_record(GYROKINETIC_OUT, {"cases": rows}, geometry=twin.geometry)

    print()
    layout = _common.Table(
        ("configuration", "22s"), ("a/L_T", "6.1f"),
        *((f"ky={k:g}", "10.4f") for k in KY_VALUES),
    )
    layout.begin()
    for configuration in configurations:
        for tprim in LINEAR_GRADIENTS:
            values = [
                next(
                    (
                        r["growth_rate"]
                        for r in rows
                        if r["configuration"] == configuration
                        and r["ky"] == ky
                        and r["tprim"] == tprim
                    ),
                    float("nan"),
                )
                for ky in KY_VALUES
            ]
            layout.row(configuration, tprim, *values)

    # The gyro-Bohm unit at the reference ion temperature, so the mixing-length sum
    # can be quoted alongside the neoclassical diffusivity it is meant to sit beside.
    equilibrium = twin.solve(twin.state(configurations[0]), SCAN)
    unit = gyro_bohm(1800.0, abs(float(equilibrium.wout.b0)), float(equilibrium.wout.Aminor_p))
    print()
    print(f"gyro-Bohm unit at 1.8 keV: {unit:.4f} m^2/s")
    layout = _common.Table(
        ("configuration", "22s"), ("a/L_T", "6.1f"), ("sum gamma/ky^2", "15.4f"),
        ("x gyro-Bohm", "13.3f"),
    )
    layout.begin()
    for configuration in configurations:
        for tprim in LINEAR_GRADIENTS:
            total = quasilinear(rows, configuration, tprim)
            layout.row(configuration, tprim, total, total * unit)
    print()
    print(
        "The mixing-length sum carries an order-one constant that only a nonlinear\n"
        "calculation fixes, so these compare configurations and gradients against each\n"
        "other rather than giving an absolute diffusivity."
    )
    return 0


# -- growth-rate-grid --------------------------------------------------------------

# The growth-rate grid over surfaces, both gradients and wavenumber; finished runs are kept.
#
#     python -m w7x_twin growth-rate-grid [configuration ...]

GRID_WORK = Path("cache/stella_grid")

#: Flux surfaces, as normalised toroidal flux. The power balance reads the table at every
#: surface, so the ones it interpolates between are what set its radial form.
GRID_SURFACES = (0.09, 0.25, 0.49, 0.64, 0.81)
#: Normalised temperature gradients a/L_T. The lowest two sit below the ion-temperature
#: gradient threshold, which for W7-X is near two, so the threshold is inside the table.
GRID_GRADIENTS = (0.5, 1.0, 2.0, 3.0, 4.0, 6.0)
#: Normalised density gradients a/L_n. A gas-fuelled W7-X profile is nearly flat and a
#: post-pellet one is peaked, and the difference between them is what this axis carries.
#: The axis reaches 6 because both profiles exceed 3 where the flat-topped density falls
#: to its edge value, and an interpolation clamped at the table's end applies no further
#: stabilisation exactly where the edge diffusivity sets the temperature profile.
DENSITY_GRADIENTS = (0.0, 1.0, 2.0, 3.0, 4.5, 6.0)
#: Binormal wavenumbers, normalised to the ion gyroradius.
#: Binormal wavenumbers in units of the ion sound radius. The last row is electron
#: scale: at the hydrogen mass ratio ky = 15 is ky rho_e = 0.35, where the ETG growth
#: rate peaks, so the grid carries one row of the electron-scale drive beside the
#: ion-scale rows the transport channel integrates.
WAVENUMBERS = (0.4, 0.7, 1.0, 2.0, 4.0, 15.0)
#: Concurrent stella processes. Each run is single-threaded, so this is the machine's
#: parallelism and not the code's.
GRID_WORKERS = 4
#: Geometry consistency tests the interface may fail before a run is retried with them
#: tolerated. Its default is zero, which drops surfaces whose two computations of the same
#: quantity differ in the last digits rather than in substance.
TOLERATED_INCONSISTENCIES = 8


def write_fine_wout(twin: Twin, configuration: str, directory: Path) -> Path:
    """The equilibrium at doubled radial resolution, fine enough for stella's geometry checks."""
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"wout_{configuration}_fine.nc"
    if not path.exists():
        output = twin.solve(twin.state(configuration), REFERENCE)
        output.wout.save(path)
    return path


def grid_case(
    wout: Path, configuration: str, torflux: float, ky: float, tprim: float, fprim: float,
    ranks: int = 1,
) -> dict:
    """One linear run per (surface, ky, gradients); a rejected geometry is retried once, recorded.

    ``ranks`` above one runs stella under MPI, which returns the same growth rate 2.9 times
    faster on four ranks and lets a batch fill cores the worker count leaves idle."""
    stem = (
        f"{configuration}_s{torflux:g}_ky{ky:g}_t{tprim:g}_f{fprim:g}"
    ).replace(".", "p")
    directory = GRID_WORK / stem
    record = directory / "answer.json"
    if record.is_file():
        stored = json.loads(record.read_text())
        if np.isfinite(stored.get("growth_rate", float("nan"))):
            return stored

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shutil.copy(wout, directory / wout.name)

    nzed = 96 if ky <= 1.0 else 128
    # The explicit step obeys the same stability margin the ion rows established, so
    # the electron-scale row shrinks it as 1/ky; the mode also grows and settles faster
    # by the same factor, so its trace is cut proportionally.
    delt = 0.05 if ky <= 1.0 else 0.02 if ky <= 4.0 else 0.08 / ky
    tend = TEND if ky <= 4.0 else TEND * 4.0 / ky
    started = time.monotonic()
    for tolerated in (0, TOLERATED_INCONSISTENCIES):
        text = INPUT_TEMPLATE.format(
            wout=wout.name, torflux=torflux, periods=FIELD_PERIODS, ky=ky,
            tprim=tprim, nzed=nzed, delt=delt, tend=tend,
        ).replace("fprim = 1.0", f"fprim = {fprim}")
        text = text.replace(
            "  verbose = .false.",
            f"  verbose = .false.\n"
            f"  n_tolerated_test_arrays_inconsistencies = {tolerated}",
        )
        (directory / f"{stem}.in").write_text(text)
        command = [str(STELLA), f"{stem}.in"]
        if ranks > 1:
            command = ["mpirun", "-np", str(ranks), *command]
        completed = subprocess.run(
            command, cwd=directory, capture_output=True, text=True, check=False,
        )
        gamma, omega = read_growth_rate(directory, stem)
        if np.isfinite(gamma):
            break

    answer = {
        "configuration": configuration,
        "torflux": torflux,
        "ky": ky,
        "tprim": tprim,
        "fprim": fprim,
        "growth_rate": gamma,
        "frequency": omega,
        "seconds": time.monotonic() - started,
        "returncode": completed.returncode,
        "tolerated_inconsistencies": tolerated,
        "error": completed.stderr[-400:] if completed.returncode else "",
    }
    record.write_text(json.dumps(answer, indent=2))
    return answer


def run_growth_rate_grid() -> int:
    if not STELLA.exists():
        raise SystemExit(f"no stella binary at {STELLA}")
    configurations = args() or ["standard", "op2_22ka"]

    twin = _common.twin()
    print(f"{twin.geometry}")
    total = (
        len(configurations) * len(GRID_SURFACES) * len(GRID_GRADIENTS)
        * len(DENSITY_GRADIENTS) * len(WAVENUMBERS)
    )
    print(
        f"{len(configurations)} configurations x {len(GRID_SURFACES)} surfaces x "
        f"{len(GRID_GRADIENTS)} temperature gradients x {len(DENSITY_GRADIENTS)} density "
        f"gradients x {len(WAVENUMBERS)} wavenumbers = {total} runs, {GRID_WORKERS} at a time"
    )

    jobs = []
    for configuration in configurations:
        wout = write_fine_wout(twin, configuration, GRID_WORK / "geometry")
        for torflux in GRID_SURFACES:
            for tprim in GRID_GRADIENTS:
                for fprim in DENSITY_GRADIENTS:
                    for ky in WAVENUMBERS:
                        jobs.append((wout, configuration, torflux, ky, tprim, fprim))

    rows = []
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=GRID_WORKERS) as pool:
        futures = {pool.submit(grid_case, *job): job for job in jobs}
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            answer = future.result()
            rows.append(answer)
            if done % 10 == 0 or done == len(jobs):
                elapsed = time.monotonic() - started
                failed = sum(1 for r in rows if not np.isfinite(r["growth_rate"]))
                print(
                    f"  {done}/{len(jobs)} in {elapsed / 60:.1f} min, {failed} without a "
                    f"growth rate",
                    flush=True,
                )

    # A run over some configurations must not drop the others from the record, so the
    # cases it did not touch are carried over rather than overwritten.
    if GRID_RECORD.exists():
        stored = json.loads(GRID_RECORD.read_text())
        kept = [
            case for case in stored.get("cases", [])
            if case.get("configuration") not in configurations
        ]
        if kept:
            print(f"keeping {len(kept)} cases of other configurations from the record")
        rows.extend(kept)
    rows.sort(
        key=lambda r: (r["configuration"], r["torflux"], r["tprim"], r["fprim"], r["ky"])
    )
    GRID_RECORD.parent.mkdir(parents=True, exist_ok=True)
    GRID_RECORD.write_text(
        json.dumps(
            {
                "geometry": twin.geometry.as_dict(),
                "surfaces": list(GRID_SURFACES),
                "gradients": list(GRID_GRADIENTS),
                "density_gradients": list(DENSITY_GRADIENTS),
                "wavenumbers": list(WAVENUMBERS),
                "cases": rows,
            },
            indent=2,
        )
    )

    print()
    for configuration in configurations:
        here = [r for r in rows if r["configuration"] == configuration]
        failed = [r for r in here if not np.isfinite(r["growth_rate"])]
        print(
            f"{configuration}: {len(here) - len(failed)} of {len(here)} runs returned a "
            f"growth rate"
        )
        if failed:
            surfaces = sorted({r["torflux"] for r in failed})
            print(f"  the ones that did not are on surfaces {surfaces}")
        tolerated = [r for r in here if r.get("tolerated_inconsistencies")]
        if tolerated:
            surfaces = sorted({r["torflux"] for r in tolerated})
            print(
                f"  {len(tolerated)} needed the geometry interface's consistency tests "
                f"tolerated, on surfaces {surfaces}"
            )

    # What the density gradient is worth, which is the axis the balance needed and did not
    # have. At fixed temperature gradient a peaked density profile moves the spectrum, and
    # the size of that is what a post-pellet enhancement has to come from.
    print()
    layout = _common.Table(
        ("configuration", "16s"), ("s", "6.2f"), ("a/L_T", "6.1f"),
        *((f"a/L_n={f:g}", "12.4f") for f in DENSITY_GRADIENTS),
    )
    layout.begin()
    for configuration in configurations:
        for torflux in GRID_SURFACES:
            for tprim in GRID_GRADIENTS:
                sums = []
                for fprim in DENSITY_GRADIENTS:
                    total_sum = 0.0
                    for ky in WAVENUMBERS:
                        match = [
                            r["growth_rate"]
                            for r in rows
                            if r["configuration"] == configuration
                            and r["torflux"] == torflux and r["tprim"] == tprim
                            and r["fprim"] == fprim and r["ky"] == ky
                        ]
                        value = match[0] if match else float("nan")
                        if np.isfinite(value) and value > 0.0:
                            total_sum += value / ky**2
                    sums.append(total_sum)
                layout.row(configuration, torflux, tprim, *sums)
    print(f"\nwrote {GRID_RECORD}")
    return 0


# -- saturation --------------------------------------------------------------------

# The saturation response measured by nonlinear flux-tube runs across surfaces and gradients.
#
#     python -m w7x_twin saturation [configuration]

NONLINEAR_WORK = Path("cache/stella_nonlinear")

#: Surfaces the constant is measured on, spanning the profile the balance integrates over.
NONLINEAR_SURFACES = (0.25, 0.49, 0.81)
#: Temperature gradients, sampling through the threshold near two.
NONLINEAR_GRADIENTS = (1.5, 2.0, 2.5, 3.0, 4.5)
#: Density gradient the grid runs carry, matching the grid's middle value.
DENSITY_GRADIENT = 1.0
#: Pellet-level cases as (surface, temperature gradient, density gradient).
PELLET_CASES: tuple[tuple[float, float, float], ...] = (
    (0.25, 3.0, 4.0),
    (0.49, 3.0, 4.0),
)
#: Ion temperature the gyro-Bohm unit is formed at, in eV.
REFERENCE_TEMPERATURE_EV = 1800.0
#: Perpendicular box and resolution. Small enough to afford nine runs, large enough that
#: the box holds several correlation lengths of the saturated state.
NX, NY, Y0 = 24, 16, 15.0
NZED = 48
#: Long enough for the drift test over the tail half to decide saturation.
NONLINEAR_TEND = 250.0
#: Longer trace for runs still trending at a substantial flux.
EXTEND_TEND = 400.0
#: Grown perpendicular boxes beside the grid, doubling both mode ranges, as
#: (nx, ny, surface, gradient, tend), one at the strongest gradient of each surface.
BOX_GROWTH: tuple[tuple[int, int, float, float, float], ...] = (
    (48, 32, 0.25, 4.5, 140.0),
    (48, 32, 0.49, 4.5, 140.0),
    (48, 32, 0.81, 4.5, 140.0),
)
#: Fraction of the trace taken as saturated, and the scatter above which it is not.
SATURATED_FROM = 0.5
SATURATION_TOLERANCE = 0.30
NONLINEAR_WORKERS = 8

TEMPLATE = """\
&geometry_options
  geometry_option = 'vmec'
/
&geometry_vmec
  vmec_filename = '{wout}'
  torflux = {torflux}
  nfield_periods = 5.0
  zeta_center = 0.0
  alpha0 = 0.0
  surface_option = 0
  verbose = .false.
/
&gyrokinetic_terms
  include_nonlinear = .true.
  include_parallel_streaming = .true.
  include_mirror = .true.
/
&species_options
  nspec = 2
/
&species_parameters_1
  z = 1.0
  mass = 1.0
  dens = 1.0
  temp = 1.0
  tprim = {tprim}
  fprim = {fprim}
  type = 'ion'
/
&species_parameters_2
  z = -1.0
  mass = 5.43867E-04
  dens = 1.0
  temp = 1.0
  tprim = {tprim}
  fprim = {fprim}
  type = 'electron'
/
&kxky_grid_option
  grid_option = 'box'
/
&kxky_grid_box
  nx = {nx}
  ny = {ny}
  y0 = {y0}
/
&z_grid
  nzed = {nzed}
  nperiod = 1
  zed_equal_arc = .true.
/
&z_boundary_condition
  boundary_option = 'linked'
/
&velocity_grids
  nvgrid = 16
  nmu = 8
  vpa_max = 3.0
  vperp_max = 3.0
/
&diagnostics
  nwrite = 20
  save_for_restart = .false.
/
&initialise_distribution
  initialise_distribution_option = 'default'
  phiinit = 0.01
/
&time_trace_options
  tend = {tend}
  autostop = .false.
/
&time_step
  delt = 0.02
/
&numerical_algorithms
  explicit_algorithm = 'rk3'
/
&debug_flags
  print_extra_info_to_terminal = .false.
/
"""


def read_heat_flux(directory: Path, stem: str) -> np.ndarray:
    """Time and both species' heat flux from the trace: the last block is ion then electron, in gyro-Bohm units."""
    path = directory / f"{stem}.fluxes"
    if not path.exists():
        return np.empty((0, 3))
    rows = []
    for line in path.read_text().splitlines():
        if line.strip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            values = [float(v) for v in parts]
        except ValueError:
            continue
        rows.append([values[0], values[5], values[6]])
    return np.array(rows) if rows else np.empty((0, 3))


def saturation_statistics(trace: np.ndarray) -> dict:
    """Tail statistics of both channels; saturation is judged on the ion channel's trend and scatter."""
    if len(trace) < 8:
        return {
            "nonlinear_saturation_state": "no flux trace",
            "saturated_ion_heat_flux_gyrobohm": float("nan"),
            "saturated_ion_heat_flux_scatter": float("nan"),
        }
    tail = trace[int(SATURATED_FROM * len(trace)) :]
    mean = float(np.mean(tail[:, 1]))
    scatter = float(np.std(tail[:, 1]))
    half = len(tail) // 2
    drift = abs(float(np.mean(tail[half:, 1]) - np.mean(tail[:half, 1])))
    saturated = (
        abs(mean) > 0.0
        and scatter / abs(mean) < SATURATION_TOLERANCE + 0.5
        and drift < max(scatter, SATURATION_TOLERANCE * abs(mean))
    )
    out = {
        "nonlinear_saturation_state": "saturated" if saturated else "trending",
        "saturated_ion_heat_flux_gyrobohm": mean,
        "saturated_ion_heat_flux_scatter": scatter,
        "trend_over_the_tail": drift,
        "trace_points": int(len(trace)),
        "final_time": float(trace[-1, 0]),
    }
    if trace.shape[1] > 2:
        out["saturated_electron_heat_flux_gyrobohm"] = float(np.mean(tail[:, 2]))
        out["saturated_electron_heat_flux_scatter"] = float(np.std(tail[:, 2]))
    return out


def nonlinear_case(
    wout: Path, configuration: str, torflux: float, tprim: float,
    nx: int = NX, ny: int = NY, tend: float = NONLINEAR_TEND,
    fprim: float = DENSITY_GRADIENT, ranks: int = 1, nzed: int = NZED,
) -> dict:
    """One nonlinear run, and the saturated flux it settles to.

    ``ranks`` above one runs stella under MPI, which returns the same flux 2.9 times faster
    on eight ranks; a parallel resolution other than the default gets its own cache entry."""
    prefix = "nl" if (nx, ny) == (NX, NY) else "nlbig"
    stem = f"{prefix}_{configuration}_s{torflux:g}_t{tprim:g}".replace(".", "p")
    if fprim != DENSITY_GRADIENT:
        stem += f"_f{fprim:g}".replace(".", "p")
    if nzed != NZED:
        stem += f"_z{nzed:d}"
    directory = NONLINEAR_WORK / stem
    record = directory / "answer.json"
    if record.is_file():
        answer = json.loads(record.read_text())
        # A run still trending at a substantial flux is rerun with the longer trace;
        # a saturated or negligible one stands.
        extendable = (
            answer.get("nonlinear_saturation_state") == "trending"
            and answer.get("saturated_ion_heat_flux_gyrobohm", 0.0) > 1.0
            and tend > answer.get("tend", NONLINEAR_TEND)
        )
        if not extendable:
            return answer

    if directory.exists():
        shutil.rmtree(directory)
    directory.mkdir(parents=True)
    shutil.copy(wout, directory / wout.name)
    (directory / f"{stem}.in").write_text(
        TEMPLATE.format(
            wout=wout.name, torflux=torflux, tprim=tprim, fprim=fprim,
            nx=nx, ny=ny, y0=Y0, nzed=nzed, tend=tend,
        )
    )

    command = [str(STELLA), f"{stem}.in"]
    if ranks > 1:
        command = ["mpirun", "-np", str(ranks), *command]
    started = time.monotonic()
    completed = subprocess.run(
        command, cwd=directory, capture_output=True, text=True, check=False,
    )
    trace = read_heat_flux(directory, stem)
    answer = {
        "configuration": configuration,
        "torflux": torflux,
        "gradient": tprim,
        "density_gradient": fprim,
        "box": [nx, ny],
        "nzed": nzed,
        "tend": tend,
        "seconds": time.monotonic() - started,
        "returncode": completed.returncode,
        "error": completed.stderr[-400:] if completed.returncode else "",
    }
    answer.update(saturation_statistics(trace))
    record.write_text(json.dumps(answer, indent=2))
    return answer


def collect(configuration: str) -> list[dict]:
    """Snapshot of every run directory, the saturation test applied to whatever trace exists."""
    answers = []
    for directory in sorted(NONLINEAR_WORK.glob(f"nl_{configuration}_*")) + sorted(
        NONLINEAR_WORK.glob(f"nlbig_{configuration}_*")
    ):
        record = directory / "answer.json"
        if record.is_file():
            answer = json.loads(record.read_text())
            # Answers written before the electron channel was read carry only the ion
            # one; the trace is still on disk, so read the electron tail from it.
            if "saturated_electron_heat_flux_gyrobohm" not in answer:
                trace = read_heat_flux(directory, directory.name)
                if len(trace) >= 8 and trace.shape[1] > 2:
                    tail = trace[int(SATURATED_FROM * len(trace)) :]
                    answer["saturated_electron_heat_flux_gyrobohm"] = float(
                        np.mean(tail[:, 2])
                    )
                    answer["saturated_electron_heat_flux_scatter"] = float(
                        np.std(tail[:, 2])
                    )
            answers.append(answer)
            continue
        stem = directory.name
        trace = read_heat_flux(directory, stem)
        parts = stem.replace("p", ".").split("_")
        try:
            fprim = DENSITY_GRADIENT
            if parts[-1].startswith("f"):
                fprim = float(parts[-1][1:])
                parts = parts[:-1]
            torflux, gradient = float(parts[-2][1:]), float(parts[-1][1:])
        except ValueError:
            continue
        answer = {
            "configuration": configuration,
            "torflux": torflux,
            "gradient": gradient,
            "density_gradient": fprim,
            "returncode": None,
        }
        if stem.startswith("nlbig"):
            for nx, ny, grown_torflux, grown_tprim, tend in BOX_GROWTH:
                if (
                    abs(grown_torflux - torflux) < 1e-9
                    and abs(grown_tprim - gradient) < 1e-9
                ):
                    answer["box"] = [nx, ny]
                    answer["tend"] = tend
        answer.update(saturation_statistics(trace))
        if len(trace) >= 8:
            answer["in_flight"] = True
        elif answer["nonlinear_saturation_state"] == "no flux trace":
            answer["nonlinear_saturation_state"] = "no flux trace yet"
        answers.append(answer)
    return answers


def run_saturation() -> int:
    snapshot = "collect" in args()
    arguments = [a for a in args() if a != "collect"]
    configuration = arguments[0] if arguments else "standard"
    if not GRID_RECORD.exists():
        raise SystemExit(f"no growth-rate grid at {GRID_RECORD}; run growth-rate-grid first")
    if not snapshot and not STELLA.exists():
        raise SystemExit(f"no stella binary at {STELLA}")

    twin = _common.twin()
    table = transport.GrowthRateTable.read(GRID_RECORD, configuration)
    equilibrium = twin.solve(twin.state(configuration), SCAN)
    minor = float(equilibrium.wout.Aminor_p)
    field_t = abs(float(equilibrium.wout.b0))
    unit = gyro_bohm(REFERENCE_TEMPERATURE_EV, field_t, minor)
    print(f"{twin.geometry}")
    print(
        f"{configuration}: {len(NONLINEAR_SURFACES)} surfaces x {len(NONLINEAR_GRADIENTS)} gradients "
        f"+ {len(BOX_GROWTH)} grown-box case(s), "
        f"gyro-Bohm unit {unit:.4f} m2/s at {REFERENCE_TEMPERATURE_EV / 1e3:.1f} keV"
    )

    if snapshot:
        answers = collect(configuration)
        print(f"snapshot of {len(answers)} run directories")
    else:
        wout = write_wout(twin, configuration, NONLINEAR_WORK / "geometry")
        # Longest first, so the grown boxes hold slots from the start of the pool.
        jobs = [
            (wout, configuration, torflux, tprim, nx, ny, tend)
            for nx, ny, torflux, tprim, tend in BOX_GROWTH
        ] + [
            (wout, configuration, torflux, tprim, NX, NY, EXTEND_TEND, fprim)
            for torflux, tprim, fprim in PELLET_CASES
        ] + [
            (wout, configuration, torflux, tprim, NX, NY, EXTEND_TEND)
            for torflux in NONLINEAR_SURFACES
            for tprim in NONLINEAR_GRADIENTS
        ]
        answers = []
        with concurrent.futures.ThreadPoolExecutor(max_workers=NONLINEAR_WORKERS) as pool:
            futures = [pool.submit(nonlinear_case, *job) for job in jobs]
            for done, future in enumerate(
                concurrent.futures.as_completed(futures), start=1
            ):
                answers.append(future.result())
                print(f"  {done}/{len(jobs)} runs finished", flush=True)

    points = []
    layout = _common.Table(
        ("s", "6.2f"), ("a/L_T", "6.1f"), ("sum gamma/ky^2", "15.4f"),
        ("Q_i [gB]", "10.3f"), ("Q_e [gB]", "10.3f"), ("scatter", "9.3f"),
        ("constant", "10.3f"), ("chi [m2/s]", "11.2f"), ("state", ">12s"),
    )
    print()
    layout.begin()
    for answer in sorted(answers, key=lambda a: (a["torflux"], a["gradient"])):
        total = table.mixing_length_sum(
            answer["torflux"], answer["gradient"],
            answer.get("density_gradient", DENSITY_GRADIENT),
        )
        flux = answer.get("saturated_ion_heat_flux_gyrobohm", float("nan"))
        electron_flux = answer.get(
            "saturated_electron_heat_flux_gyrobohm", float("nan")
        )
        # The heat diffusivity a flux-tube run reports is Q / (n T a/L_T) in its own units.
        diffusivity = flux / max(answer["gradient"], 1e-9)
        constant = diffusivity / total if total > 0.0 else float("nan")
        scatter = answer.get("saturated_ion_heat_flux_scatter", float("nan"))
        point = {
            **{k: v for k, v in answer.items() if k != "error"},
            "mixing_length_sum": total,
            "gyrobohm_unit_m2_s": unit,
            "diffusivity_gyrobohm": diffusivity,
            "constant": constant,
            "diffusivity_m2_s": diffusivity * unit,
        }
        points.append(point)
        layout.row(
            answer["torflux"], answer["gradient"], total, flux, electron_flux,
            scatter, constant, diffusivity * unit,
            answer.get("nonlinear_saturation_state", "unknown"),
        )

    saturated = [
        p for p in points
        if p.get("nonlinear_saturation_state") == "saturated"
        and np.isfinite(p.get("constant", float("nan")))
    ]
    default = [
        p for p in saturated if tuple(p.get("box", (NX, NY))) == (NX, NY)
    ]
    print()
    if default:
        values = np.array([p["constant"] for p in default])
        print(
            f"{len(saturated)} of {len(points)} runs saturated; on the {NX} x {NY} box "
            f"the constant is {np.median(values):.3f} in the median and spans "
            f"{values.min():.3f} to {values.max():.3f}, a factor of "
            f"{values.max() / max(values.min(), 1e-9):.2f}"
        )
        near = [p for p in default if p["gradient"] <= 2.0]
        above = [p for p in default if p["gradient"] > 2.0]
        if near and above:
            print(
                f"  near threshold it is {np.median([p['constant'] for p in near]):.3f} "
                f"and above it {np.median([p['constant'] for p in above]):.3f}"
            )
        surfaces = sorted({p["torflux"] for p in default})
        if len(surfaces) > 1:
            per = [
                np.median([p["constant"] for p in default if p["torflux"] == s])
                for s in surfaces
            ]
            print(
                "  across surfaces "
                + ", ".join(f"s = {s:.2f}: {v:.3f}" for s, v in zip(surfaces, per))
            )
        # The grown box against the grid point it repeats, which is what says whether the
        # constant is a saturation measurement or a box artifact.
        for point in saturated:
            if tuple(point.get("box", (NX, NY))) == (NX, NY):
                continue
            partner = next(
                (
                    p for p in default
                    if p["torflux"] == point["torflux"]
                    and p["gradient"] == point["gradient"]
                ),
                None,
            )
            if partner is not None:
                print(
                    f"  at {point['box'][0]} x {point['box'][1]} the constant is "
                    f"{point['constant']:.3f} against {partner['constant']:.3f} at "
                    f"{NX} x {NY}, on s = {point['torflux']:g} at "
                    f"a/L_T = {point['gradient']:g}"
                )
    else:
        print("no run saturated, so no constant is measured")

    write_record(
        CONSTANT_RECORD, {"points": points}, geometry=twin.geometry,
        reads=(GRID_RECORD,),
    )
    return 0


# -- turbulence --------------------------------------------------------------------

# The balance with both channels computed; the confinement time stands against ISS04 as a result.
#
#     python -m w7x_twin turbulence [heating in MW ...]

BALANCE_OUT = Path("results/plasma/turbulent_transport.json")
DEFAULT_POWERS = (2.0, 5.0, 10.0)
#: The post-pellet comparison: the digitised figure-6 profile peaks by 2.68 at the power
#: the pellet discharge stepped to, and the enhancement is what the density gradient is
#: worth to the turbulent channel, which now reads a/L_n as well as a/L_T.
POST_PELLET_PEAKING = 2.68
POST_PELLET_POWER_W = 4.9e6


def run_turbulence() -> int:
    powers = [v * 1e6 for v in args(float)] or [p * 1e6 for p in DEFAULT_POWERS]
    if not GRID_RECORD.exists():
        raise SystemExit(f"no growth-rate grid at {GRID_RECORD}; run growth_rate_grid.py")
    if not CONSTANT_RECORD.exists():
        raise SystemExit(f"no constant at {CONSTANT_RECORD}; run mixing_length_constant.py")

    table = transport.GrowthRateTable.read(GRID_RECORD)
    response = transport.MixingLengthResponse.read(CONSTANT_RECORD)
    if response is None:
        raise SystemExit(f"{CONSTANT_RECORD} carries no measured points to respond from")

    twin = _common.twin()
    print(f"{twin.geometry}")
    print(
        f"growth rates on {len(table.surfaces)} surfaces x {len(table.gradients)} "
        f"gradients x {len(table.wavenumbers)} wavenumbers"
    )
    print(
        f"saturation response on {len(response.surfaces)} surfaces, each curve through "
        f"{max(len(cx) for cx, _ in response.curves)} nonlinear runs:"
    )
    for surface, (cx, cq) in zip(response.surfaces, response.curves, strict=True):
        pairs = ", ".join(f"{x:.2f} -> {q:.1f}" for x, q in zip(cx, cq, strict=True))
        print(f"  s = {surface:.2f}: {pairs}  [gyro-Bohm against the linear sum]")

    equilibrium = twin.solve(twin.state("standard"), SCAN)
    minor = float(equilibrium.wout.Aminor_p)
    field_t = abs(float(equilibrium.wout.b0))
    coefficients = neoclassical.load_radial_profile(verbose=False)
    ripple = neoclassical.load_ripple()
    field_capture: dict = {}
    split_channels = neoclassical.split_diffusivity_model(
        coefficients, ripple, minor, capture=field_capture
    )
    # The same shell-by-shell construction as the profile comparisons: the measured
    # response is a cliff, and the profile-level iteration falls off it.
    turbulent_pair = (
        transport.local_turbulence(table, response, field_t, minor, species="electron"),
        transport.local_turbulence(table, response, field_t, minor, species="ion"),
    )
    quench = transport.shear_quench_model(table, field_t, minor)

    layout = _common.Table(
        ("density", ">7s"), ("P [MW]", "6.1f"), ("W [MJ]", "8.3f"),
        ("tau [s]", "9.4f"), ("ISS04 [s]", "10.4f"), ("over ISS04", "11.3f"),
        ("T_e(0)", "9.0f"), ("chi_neo(0)", "11.4f"), ("chi_turb(0)", "12.4f"),
    )
    print()
    layout.begin()

    rows = []
    for power, profiles, label in (
        [(p, kinetics.HIGH_PERFORMANCE, "flat") for p in powers]
        + [(POST_PELLET_POWER_W,
            kinetics.HIGH_PERFORMANCE.with_peaking(POST_PELLET_PEAKING), "peaked")]
    ):
        solution = transport.solve_split(
            equilibrium, profiles,
            transport.Heating(power_w=power),
            split_channels,
            turbulent_local=turbulent_pair,
            shear_quench=quench, field_capture=field_capture,
        )
        over = solution.confinement_time_s / solution.iss04_time_s
        base_chi = solution.chi_m2_s - solution.chi_anomalous_m2_s
        rows.append(
            {
                "density": label,
                "heating_power_w": power,
                "stored_energy_j": float(solution.stored_energy_j),
                "confinement_time_s": float(solution.confinement_time_s),
                "iss04_time_s": float(solution.iss04_time_s),
                "over_iss04": float(over),
                "electron_temperature_axis_ev": float(solution.electron_temperature_ev[0]),
                "chi_neoclassical_axis_m2_s": float(base_chi[0]),
                "chi_turbulent_axis_m2_s": float(solution.chi_anomalous_m2_s[0]),
                "s": solution.s.tolist(),
                "chi_neoclassical_m2_s": base_chi.tolist(),
                "chi_turbulent_m2_s": solution.chi_anomalous_m2_s.tolist(),
                "electron_temperature_ev": solution.electron_temperature_ev.tolist(),
            }
        )
        layout.row(
            label, power / 1e6, solution.stored_energy_j / 1e6,
            solution.confinement_time_s, solution.iss04_time_s, over,
            solution.electron_temperature_ev[0], base_chi[0],
            solution.chi_anomalous_m2_s[0],
        )

    # What the pellet's density gradient is worth: the peaked and the flat profile at the
    # same power, both with every channel computed, against the 1.30 over 0.70-0.83 the
    # machine separates them by.
    peaked = next(r for r in rows if r["density"] == "peaked")
    flat_at = min(
        (r for r in rows if r["density"] == "flat"),
        key=lambda r: abs(r["heating_power_w"] - peaked["heating_power_w"]),
    )
    print()
    print(
        f"peaking the density by {POST_PELLET_PEAKING:.2f} at "
        f"{peaked['heating_power_w'] / 1e6:.1f} MW moves the computed confinement from "
        f"{flat_at['over_iss04']:.3f} to {peaked['over_iss04']:.3f} times ISS04, a factor "
        f"of {peaked['over_iss04'] / flat_at['over_iss04']:.2f} against the "
        f"1.30 / 0.70 = 1.86 the machine separates the two regimes by"
    )

    print()
    reference = rows[len(rows) // 2]
    print(
        f"at {reference['heating_power_w'] / 1e6:.0f} MW the computed balance gives "
        f"{reference['over_iss04']:.2f} times the ISS04 scaling, against the "
        f"{transport.PUBLISHED_ISS04_ENHANCEMENT:.1f} the machine is reported at"
    )

    # Where the two channels stand against each other, on the same solve.
    index = np.argmin(np.abs(np.array(reference["s"]) - 0.25))
    print(
        f"at s = 0.25 the drift-kinetic channel is "
        f"{reference['chi_neoclassical_m2_s'][index]:.4f} m2/s and the turbulent one "
        f"{reference['chi_turbulent_m2_s'][index]:.4f} m2/s"
    )

    write_record(
        BALANCE_OUT,
        {
            "response_surfaces": [
                {
                    "torflux": float(surface),
                    "mixing_length_sum": cx.tolist(),
                    "flux_gyrobohm": cq.tolist(),
                }
                for surface, (cx, cq) in zip(
                    response.surfaces, response.curves, strict=True
                )
            ],
            "grid": str(GRID_RECORD),
            "cases": rows,
        },
        geometry=twin.geometry,
        reads=(GRID_RECORD, CONSTANT_RECORD),
    )
    return 0
