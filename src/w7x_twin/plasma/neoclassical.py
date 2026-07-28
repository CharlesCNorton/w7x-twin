"""Effective ripple, monoenergetic drift-kinetic coefficients, and the convolved fluxes."""

from __future__ import annotations

import dataclasses
import numpy as np
import shutil
import subprocess
import sys
import vmecpp
from pathlib import Path


# -- from neoclassical ------------------------------------------------------------

@dataclasses.dataclass
class EffectiveRipple:
    """Effective ripple against normalised toroidal flux."""

    s: np.ndarray
    rho: np.ndarray
    eps_32: np.ndarray

    @property
    def eps_eff(self) -> np.ndarray:
        return self.eps_32 ** (2.0 / 3.0)

    def at(self, s: float | np.ndarray) -> np.ndarray:
        """Effective ripple interpolated to other flux surfaces."""
        return np.interp(np.clip(s, 0.0, 1.0), self.s, self.eps_eff)


def effective_ripple(
    output: vmecpp.VmecOutput,
    interpreter: str,
    work_dir: str | Path = "cache",
    num_surfaces: int = 10,
    tag: str = "eps",
    verbose: bool = True,
) -> EffectiveRipple:
    """Effective ripple profile via the DESC worker; ``interpreter`` is a python with DESC."""
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    wout = work_dir / f"wout_{tag}.nc"
    result = work_dir / f"eps_eff_{tag}.npz"

    output.wout.save(wout)
    completed = subprocess.run(
        [
            interpreter,
            "-m",
            "w7x_twin.plasma._effective_ripple_desc",
            str(wout),
            str(result),
            str(num_surfaces),
        ],
        capture_output=True,
        text=True,
        env={"PYTHONPATH": str(Path(__file__).resolve().parents[2])},
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            f"effective ripple failed:\n{completed.stdout}\n{completed.stderr}"
        )
    if verbose:
        print(completed.stdout.strip())

    stored = np.load(result)
    return EffectiveRipple(
        s=stored["s"], rho=stored["rho"], eps_32=stored["eps_32"]
    )


@dataclasses.dataclass
class MonoenergeticCoefficients:
    """Monoenergetic coefficients on one surface; ``collisionality`` is nu/v in 1/m,
    ``radial_field`` is E_r/v, and ``d31`` is the bootstrap coefficient."""

    s: float
    collisionality: np.ndarray
    radial_field: np.ndarray
    d11: np.ndarray
    d31: np.ndarray
    d33: np.ndarray
    d33_spitzer: np.ndarray

    def one_over_nu_plateau(self) -> float:
        """D_11 nu/v at the lowest computed collisionality, constant in the 1/nu regime."""
        index = int(np.argmin(self.collisionality))
        return float(self.d11[index] * self.collisionality[index])


def monoenergetic_coefficients(
    output: vmecpp.VmecOutput,
    monkes_executable: str | Path,
    surface: float = 0.2,
    collisionalities: tuple[float, ...] = (1e-5, 1e-4, 1e-3, 1e-2),
    radial_field: float = 0.0,
    num_theta: int = 30,
    num_zeta: int = 60,
    num_legendre: int = 160,
    work_dir: str | Path = "cache/monkes",
    verbose: bool = True,
) -> MonoenergeticCoefficients:
    """Solve the drift-kinetic equation on one flux surface with MONKES."""
    work_dir = Path(work_dir)
    if work_dir.exists():
        shutil.rmtree(work_dir)
    work_dir.mkdir(parents=True)

    # MONKES selects its input by filename; a VMEC output named this way is read
    # directly, with no Boozer transformation step in between.
    output.wout.save(work_dir / "VMEC.nc")
    shutil.copy(monkes_executable, work_dir / "main_monkes.x")

    nu_block = ",\n".join(f"{value:.6e}" for value in collisionalities)
    (work_dir / "monkes_input.parameters").write_text(
        f"&parameters\nN_theta = {num_theta}\nN_zeta = {num_zeta}\n"
        f"N_xi = {num_legendre}\nnu =\n{nu_block}\n"
        f"E_r = {radial_field:.6e}\n/\n"
    )
    (work_dir / "monkes_input.surface").write_text(f"&surface\ns={surface}\n/\n")

    completed = subprocess.run(
        ["./main_monkes.x"],
        cwd=work_dir,
        capture_output=True,
        text=True,
        check=False,
    )
    database = work_dir / "monkes_Monoenergetic_Database.dat"
    if completed.returncode != 0 or not database.exists():
        raise RuntimeError(
            f"MONKES failed:\n{completed.stdout[-2000:]}\n{completed.stderr[-2000:]}"
        )

    rows = np.loadtxt(database, skiprows=1)
    rows = np.atleast_2d(rows)
    if verbose:
        print(f"[monkes] {len(rows)} points on s = {surface}")
    return MonoenergeticCoefficients(
        s=surface,
        collisionality=rows[:, 0],
        radial_field=rows[:, 1],
        d11=rows[:, 5],
        d31=rows[:, 6],
        d33=rows[:, 8],
        d33_spitzer=rows[:, 9],
    )


def load_monoenergetic(path: str | Path, surface: float = 0.2) -> MonoenergeticCoefficients:
    """Read a MONKES database file that has already been produced."""
    rows = np.atleast_2d(np.loadtxt(path, skiprows=1))
    return MonoenergeticCoefficients(
        s=surface,
        collisionality=rows[:, 0],
        radial_field=rows[:, 1],
        d11=rows[:, 5],
        d31=rows[:, 6],
        d33=rows[:, 8],
        d33_spitzer=rows[:, 9],
    )


@dataclasses.dataclass
class MonoenergeticProfile:
    """Monoenergetic coefficients on several surfaces, interpolated between bracketing solutions."""

    surfaces: np.ndarray
    tables: tuple[MonoenergeticCoefficients, ...]

    def __len__(self) -> int:
        return len(self.tables)


def load_monoenergetic_profile(paths: dict[float, str | Path]) -> MonoenergeticProfile:
    """Read one MONKES database per flux surface, given ``{s: path}``."""
    ordered = sorted((float(s), Path(p)) for s, p in dict(paths).items())
    return MonoenergeticProfile(
        surfaces=np.array([s for s, _ in ordered]),
        tables=tuple(load_monoenergetic(path, s) for s, path in ordered),
    )


#: Filename MONKES writes its coefficients to.
MONKES_DATABASE = "monkes_Monoenergetic_Database.dat"

#: The radial drift-kinetic scan every consumer reads, and its fallback with the narrower E_r axis.
RADIAL_SCANS = (
    "cache/monkes_field",
    "cache/monkes_radial",
)
#: The effective-ripple profile of the same equilibrium, on the same twelve surfaces.
RIPPLE_TABLE = "cache/eps_eff_standard_beta1.npz"
#: Surface results are reported at, one of the twelve the radial scan carries.
REFERENCE_SURFACE = 0.16

#: Surface the single-surface table was solved on, measured from its nu * D33_Spitzer against the scan.
SINGLE_SURFACE = 0.20


def load_radial_profile(verbose: bool = True) -> "MonoenergeticProfile":
    """The radial scan, from the first directory that carries one."""
    for directory in RADIAL_SCANS:
        profile = discover_monoenergetic_profile(directory)
        if profile is not None:
            if verbose:
                print(f"drift-kinetic tables from {directory}, {len(profile)} surfaces")
            return profile
    raise FileNotFoundError(f"no drift-kinetic scan under {RADIAL_SCANS}")


def load_ripple(path: str | Path | None = None) -> "EffectiveRipple":
    """The effective-ripple profile the radial scaling and the surface weights use."""
    stored = np.load(Path(path or RIPPLE_TABLE))
    return EffectiveRipple(
        s=stored["s"], rho=stored["rho"], eps_32=stored["eps_32"]
    )


#: Ion temperature as a fraction of the electron temperature, matching the ratio the
#: power balance carries.
ION_TEMPERATURE_FRACTION = 0.55


def diffusivity_model(
    coefficients,
    ripple,
    minor_radius_m: float,
    radial_field_v_m: float | None = None,
    ion_fraction: float = ION_TEMPERATURE_FRACTION,
    reference_surface: float = SINGLE_SURFACE,
    carbon_fraction: float | None = None,
):
    """Return ``chi(s, electron_temperature_ev, density)``, both species energy-weighted;
    ``radial_field_v_m`` of None solves ambipolarity per surface, a number imposes it."""

    def chi(s, electron_temperature_ev, density):
        s = np.asarray(s, dtype=float)
        out = np.empty_like(s)
        radius = minor_radius_m * np.sqrt(s)
        dln_n = np.gradient(np.log(np.maximum(density, 1e-30)), radius)
        dln_te = np.gradient(np.log(np.maximum(electron_temperature_ev, 1e-30)), radius)
        parts = None
        if carbon_fraction:
            from w7x_twin.plasma import kinetics

            parts = kinetics.composition(
                density, electron_temperature_ev, carbon_fraction
            )
        for index, (surface, temperature, n) in enumerate(
            zip(s, electron_temperature_ev, density, strict=True)
        ):
            ion_temperature = ion_fraction * float(temperature)
            z_effective = 1.0 if parts is None else float(parts.z_effective[index])
            ion_density = n if parts is None else float(parts.ion_density_m3[index])
            field = radial_field_v_m
            if field is None:
                field = 0.0
                for table, weight in surface_tables(
                    coefficients, float(surface), which="d11", ripple=ripple,
                    reference_surface=reference_surface,
                ):
                    if weight == 0.0:
                        continue
                    answer = ambipolar_field(
                        table,
                        density_m3=float(n),
                        electron_temperature_ev=float(temperature),
                        ion_temperature_ev=ion_temperature,
                        density_gradient=float(dln_n[index]),
                        electron_temperature_gradient=float(dln_te[index]),
                        ion_temperature_gradient=float(dln_te[index]),
                        bracket=(-25.0e3, 25.0e3),
                        num_probe=41,
                    )
                    if np.isfinite(answer["field"]):
                        field += weight * answer["field"]
            total = 0.0
            for table, weight in surface_tables(
                coefficients, float(surface), which="d11", ripple=ripple,
                reference_surface=reference_surface,
            ):
                if weight == 0.0:
                    continue
                electron = heat_diffusivity(
                    table, density_m3=float(n), temperature_ev=float(temperature),
                    mass=ELECTRON_MASS, charge_number=-1.0, radial_field_v_m=field,
                    z_effective=z_effective,
                )
                ion = heat_diffusivity(
                    table, density_m3=ion_density, temperature_ev=ion_temperature,
                    mass=PROTON_MASS, charge_number=1.0, radial_field_v_m=field,
                    z_effective=z_effective,
                )
                total += (
                    weight
                    * (float(temperature) * electron + ion_temperature * ion)
                    / (float(temperature) + ion_temperature)
                )
            out[index] = total
        return out

    return chi


def split_diffusivity_model(
    coefficients,
    ripple,
    minor_radius_m: float,
    radial_field_v_m: float | None = None,
    reference_surface: float = SINGLE_SURFACE,
    carbon_fraction: float | None = None,
    capture: dict | None = None,
):
    """Return ``(chi_e, chi_i)(s, Te, Ti, n)`` with the ambipolar field solved per point;
    ``capture`` stores the solved field profile under ``radial_field_v_m``."""

    def channels(s, electron_temperature_ev, ion_temperature_ev, density):
        s = np.asarray(s, dtype=float)
        electron_out = np.empty_like(s)
        ion_out = np.empty_like(s)
        field_out = np.zeros_like(s)
        radius = minor_radius_m * np.sqrt(s)
        dln_n = np.gradient(np.log(np.maximum(density, 1e-30)), radius)
        dln_te = np.gradient(
            np.log(np.maximum(electron_temperature_ev, 1e-30)), radius
        )
        dln_ti = np.gradient(np.log(np.maximum(ion_temperature_ev, 1e-30)), radius)
        parts = None
        if carbon_fraction:
            from w7x_twin.plasma import kinetics

            parts = kinetics.composition(
                density, electron_temperature_ev, carbon_fraction
            )
        for index, surface in enumerate(s):
            electron_t = float(electron_temperature_ev[index])
            ion_t = float(ion_temperature_ev[index])
            n = float(density[index])
            z_effective = 1.0 if parts is None else float(parts.z_effective[index])
            ion_density = n if parts is None else float(parts.ion_density_m3[index])
            field = radial_field_v_m
            if field is None:
                field = 0.0
                for table, weight in surface_tables(
                    coefficients, float(surface), which="d11", ripple=ripple,
                    reference_surface=reference_surface,
                ):
                    if weight == 0.0:
                        continue
                    answer = ambipolar_field(
                        table,
                        density_m3=n,
                        electron_temperature_ev=electron_t,
                        ion_temperature_ev=ion_t,
                        density_gradient=float(dln_n[index]),
                        electron_temperature_gradient=float(dln_te[index]),
                        ion_temperature_gradient=float(dln_ti[index]),
                        bracket=(-25.0e3, 25.0e3),
                        num_probe=41,
                    )
                    if np.isfinite(answer["field"]):
                        field += weight * answer["field"]
            electron_total = 0.0
            ion_total = 0.0
            for table, weight in surface_tables(
                coefficients, float(surface), which="d11", ripple=ripple,
                reference_surface=reference_surface,
            ):
                if weight == 0.0:
                    continue
                electron_total += weight * heat_diffusivity(
                    table, density_m3=n, temperature_ev=electron_t,
                    mass=ELECTRON_MASS, charge_number=-1.0,
                    radial_field_v_m=field, z_effective=z_effective,
                )
                ion_total += weight * heat_diffusivity(
                    table, density_m3=ion_density, temperature_ev=ion_t,
                    mass=PROTON_MASS, charge_number=1.0,
                    radial_field_v_m=field, z_effective=z_effective,
                )
            electron_out[index] = electron_total
            ion_out[index] = ion_total
            field_out[index] = field
        if capture is not None:
            capture["radial_field_v_m"] = field_out.copy()
        return electron_out, ion_out

    return channels


def discover_monoenergetic_profile(
    directory: str | Path,
) -> MonoenergeticProfile | None:
    """Read a radial scan in either on-disk layout, or None if the directory holds none."""
    directory = Path(directory)
    if not directory.is_dir():
        return None

    paths: dict[float, Path] = {}
    for entry in sorted(directory.glob("s*")):
        database = entry if entry.is_file() else entry / MONKES_DATABASE
        if not database.is_file() or database.stat().st_size == 0:
            continue
        try:
            surface = float(entry.name[1:].removesuffix(".dat"))
        except ValueError:
            continue
        paths[surface] = database
    return load_monoenergetic_profile(paths) if paths else None


def surface_tables(
    coefficients: MonoenergeticCoefficients | MonoenergeticProfile,
    s: float,
    which: str = "d11",
    ripple: EffectiveRipple | None = None,
    reference_surface: float = SINGLE_SURFACE,
) -> list[tuple[MonoenergeticCoefficients, float]]:
    """Tables and interpolation weights evaluating a coefficient at ``s``; a single-surface
    d11 scales as effective ripple to the 3/2, d31 is held flat."""
    if isinstance(coefficients, MonoenergeticProfile):
        surfaces = coefficients.surfaces
        if len(surfaces) == 1:
            return [(coefficients.tables[0], 1.0)]
        position = float(np.clip(s, surfaces[0], surfaces[-1]))
        upper = int(np.clip(np.searchsorted(surfaces, position), 1, len(surfaces) - 1))
        lower = upper - 1
        span = surfaces[upper] - surfaces[lower]
        weight = 0.0 if span == 0 else (position - surfaces[lower]) / span
        return [
            (coefficients.tables[lower], 1.0 - weight),
            (coefficients.tables[upper], weight),
        ]

    if ripple is None or which != "d11":
        return [(coefficients, 1.0)]
    scale = (
        float(ripple.at(s)) / float(ripple.at(reference_surface))
    ) ** 1.5
    return [(dataclasses.replace(coefficients, d11=coefficients.d11 * scale), 1.0)]


# -- from monoenergetic coefficients to a thermal diffusivity ----------------------

ELEMENTARY_CHARGE = 1.602176634e-19
ELECTRON_MASS = 9.1093837015e-31
PROTON_MASS = 1.67262192369e-27
VACUUM_PERMITTIVITY = 8.8541878128e-12


def coulomb_logarithm(density_m3: float, temperature_ev: float) -> float:
    """Electron Coulomb logarithm, NRL formulary form for thermal electrons."""
    return float(
        31.3 - np.log(np.sqrt(max(density_m3, 1.0)) / max(temperature_ev, 1.0))
    )


def _chandrasekhar(x: np.ndarray) -> np.ndarray:
    """(erf(x) - x erf'(x)) / (2 x^2), the Chandrasekhar function."""
    from scipy.special import erf

    x = np.asarray(x, dtype=float)
    small = x < 1e-8
    safe = np.where(small, 1.0, x)
    value = (erf(safe) - safe * (2.0 / np.sqrt(np.pi)) * np.exp(-safe**2)) / (
        2.0 * safe**2
    )
    return np.where(small, 0.0, value)


def deflection_frequency(
    speed: np.ndarray,
    density_m3: float,
    temperature_ev: float,
    mass: float,
    charge_number: float = -1.0,
    z_effective: float = 1.0,
) -> np.ndarray:
    """Pitch-angle deflection frequency of one species against ions and electrons."""
    from scipy.special import erf

    charge = abs(charge_number) * ELEMENTARY_CHARGE
    prefactor = (
        4.0
        * np.pi
        * density_m3
        * charge**2
        * ELEMENTARY_CHARGE**2
        * coulomb_logarithm(density_m3, temperature_ev)
        / ((4.0 * np.pi * VACUUM_PERMITTIVITY) ** 2 * mass**2 * speed**3)
    )
    thermal_speed = np.sqrt(2.0 * temperature_ev * ELEMENTARY_CHARGE / mass)
    x = speed / thermal_speed
    like_species = erf(x) - _chandrasekhar(x)
    return prefactor * (z_effective + like_species)


def physical_d11(
    monkes_d11: np.ndarray, speed: np.ndarray, mass: float, charge_number: float = -1.0
) -> np.ndarray:
    """Monoenergetic radial diffusion D_11 [m^2/s] = (m^2 v^3 / e^2) D_11^MONKES."""
    charge = abs(charge_number) * ELEMENTARY_CHARGE
    return (mass**2 * speed**3 / charge**2) * monkes_d11


#: Parallel electron conductivity as a fraction of what a pitch-angle-scattering operator
#: alone gives at the same effective charge [L. Spitzer and R. Haerm, Phys. Rev. 89 (1953)
#: 977, table III]. Like-particle collisions conserve momentum and cannot damp a current
#: directly; what they do is redistribute it over the tail the parallel response is carried
#: by, and the effect vanishes as the ion charge grows.
SPITZER_CHARGES: tuple[float, ...] = (1.0, 2.0, 4.0, 16.0, 1.0e6)
SPITZER_FACTORS: tuple[float, ...] = (0.5816, 0.6833, 0.7849, 0.9225, 1.0)


def spitzer_correction(z_effective) -> np.ndarray:
    """Electron-electron momentum-restoration factor, interpolated in log charge."""
    z = np.clip(np.asarray(z_effective, dtype=float), SPITZER_CHARGES[0], SPITZER_CHARGES[-1])
    return np.interp(np.log(z), np.log(SPITZER_CHARGES), SPITZER_FACTORS)


def parallel_current_drive(
    coefficients: MonoenergeticCoefficients,
    density_m3: float,
    temperature_ev: float,
    density_gradient: float,
    temperature_gradient: float,
    field_b00_t: float,
    mass: float = ELECTRON_MASS,
    charge_number: float = -1.0,
    z_effective: float = 1.0,
    radial_field_v_m: float = 0.0,
    num_energy: int = 64,
) -> float:
    """Per-species <J.B>_a = p_a B00 (2/sqrt(pi)) int dK sqrt(K) e^-K K D_31 [A1 + (K - 3/2) A2], in T A/m^2."""
    thermal_speed = np.sqrt(2.0 * temperature_ev * ELEMENTARY_CHARGE / mass)
    energy, weight = maxwellian_nodes(num_energy)
    speed = thermal_speed * np.sqrt(energy)

    nu = deflection_frequency(
        speed, density_m3, temperature_ev, mass, charge_number, z_effective
    )
    d31 = _interpolate_coefficient(
        coefficients, nu / speed, radial_field_v_m / speed, "d31"
    )

    charge = charge_number * ELEMENTARY_CHARGE
    a1 = density_gradient - charge * radial_field_v_m / (
        temperature_ev * ELEMENTARY_CHARGE
    )
    a2 = temperature_gradient

    pressure = density_m3 * temperature_ev * ELEMENTARY_CHARGE
    integrand = energy * d31 * (a1 + (energy - 1.5) * a2)
    return float(pressure * field_b00_t * np.sum(weight * integrand))


def maxwellian_nodes(num_energy: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Laguerre (alpha = 1/2) nodes and weights for the sqrt(K) e^-K measure,
    the 2/sqrt(pi) normalisation absorbed into the weights."""
    from scipy.special import roots_genlaguerre

    nodes, weights = roots_genlaguerre(num_energy, 0.5)
    return nodes, weights * (2.0 / np.sqrt(np.pi))


#: Points at each end of the table the continuation exponent is fitted over. Two takes
#: the end-point difference; more takes a least-squares fit, which is less exposed to
#: the last point being the least converged. :func:`extrapolation_sensitivity` reports
#: what the choice costs.
EXTRAPOLATION_FIT_POINTS = 2


def _end_slope(xp: np.ndarray, fp: np.ndarray, at_low_end: bool) -> float:
    """Slope at one end of a tabulated curve, over EXTRAPOLATION_FIT_POINTS points."""
    count = int(np.clip(EXTRAPOLATION_FIT_POINTS, 2, len(xp)))
    x = xp[:count] if at_low_end else xp[-count:]
    f = fp[:count] if at_low_end else fp[-count:]
    if count == 2:
        return float((f[1] - f[0]) / (x[1] - x[0]))
    return float(np.polyfit(x, f, 1)[0])


def _power_law_continued(x: np.ndarray, xp: np.ndarray, fp: np.ndarray) -> np.ndarray:
    """Linear interpolation continued outside the range with the end-point slopes."""
    if len(xp) < 2:
        # One tabulated point carries no slope, so the coefficient is held at it.
        return np.full_like(np.asarray(x, dtype=float), fp[0])
    out = np.interp(x, xp, fp)
    low = _end_slope(xp, fp, True)
    high = _end_slope(xp, fp, False)
    out = np.where(x < xp[0], fp[0] + low * (x - xp[0]), out)
    return np.where(x > xp[-1], fp[-1] + high * (x - xp[-1]), out)


def _along_collisionality(
    nu_over_v: np.ndarray, nu_hat: np.ndarray, sample: np.ndarray, which: str
) -> np.ndarray:
    """One coefficient against collisionality, at a single radial electric field."""
    logs = np.log(np.maximum(np.asarray(nu_over_v, dtype=float), 1e-300))
    if which == "d11":
        # Positive across three decades of magnitude, so both axes are logarithmic. A
        # solver-noise nonpositive entry at the table's extreme corner would poison the
        # end-slope fit through its logarithm, so it is dropped from the slice instead.
        keep = np.asarray(sample) > 0.0
        if not keep.all():
            nu_hat = np.asarray(nu_hat)[keep]
            sample = np.asarray(sample)[keep]
        if len(np.atleast_1d(sample)) < 2:
            fallback = float(np.atleast_1d(sample)[0]) if len(np.atleast_1d(sample)) else 0.0
            return np.full_like(np.asarray(nu_over_v, dtype=float), fallback)
        return np.exp(
            np.minimum(_power_law_continued(logs, np.log(nu_hat), np.log(sample)), 700.0)
        )
    # D_31 passes through zero near nu/v = 1e-3 and tends to a finite limit at both
    # ends of the table, so it is interpolated linearly in value against log nu and
    # held at its end values outside. Interpolating its magnitude in the logarithm
    # would carry the sign change across without the zero it passes through.
    return np.interp(logs, np.log(nu_hat), sample)


def _interpolate_coefficient(
    coefficients: MonoenergeticCoefficients,
    nu_over_v: np.ndarray,
    field_over_v: np.ndarray,
    which: str,
) -> np.ndarray:
    """Interpolate a coefficient over the (nu/v, |E_r/v|) product grid, both axes live."""
    values = np.asarray(getattr(coefficients, which), dtype=float)
    fields = np.unique(coefficients.radial_field)

    slices = []
    for value in fields:
        mask = coefficients.radial_field == value
        nu_hat = coefficients.collisionality[mask]
        order = np.argsort(nu_hat)
        slices.append(
            _along_collisionality(nu_over_v, nu_hat[order], values[mask][order], which)
        )
    if len(fields) == 1:
        return slices[0]

    stack = np.stack(slices)
    target = np.abs(
        np.broadcast_to(np.asarray(field_over_v, dtype=float), stack.shape[1:])
    )
    index = np.clip(np.searchsorted(fields, target) - 1, 0, len(fields) - 2)
    lower, upper = fields[index], fields[index + 1]
    weight = np.clip((target - lower) / np.maximum(upper - lower, 1e-300), 0.0, 1.0)
    columns = np.arange(stack.shape[1])
    return (1.0 - weight) * stack[index, columns] + weight * stack[index + 1, columns]


def particle_flux(
    coefficients: MonoenergeticCoefficients,
    density_m3: float,
    temperature_ev: float,
    density_gradient: float,
    temperature_gradient: float,
    mass: float = ELECTRON_MASS,
    charge_number: float = -1.0,
    z_effective: float = 1.0,
    radial_field_v_m: float = 0.0,
    num_energy: int = 64,
) -> float:
    """Gamma_a / n_a = -(2/sqrt(pi)) int dK sqrt(K) e^-K D_11 [A1 + (K - 3/2) A2], in m/s."""
    thermal_speed = np.sqrt(2.0 * temperature_ev * ELEMENTARY_CHARGE / mass)
    energy, weight = maxwellian_nodes(num_energy)
    speed = thermal_speed * np.sqrt(energy)

    nu = deflection_frequency(
        speed, density_m3, temperature_ev, mass, charge_number, z_effective
    )
    interpolated = _interpolate_coefficient(
        coefficients, nu / speed, radial_field_v_m / speed, "d11"
    )
    d11 = physical_d11(interpolated, speed, mass, charge_number)

    charge = charge_number * ELEMENTARY_CHARGE
    a1 = density_gradient - charge * radial_field_v_m / (
        temperature_ev * ELEMENTARY_CHARGE
    )
    a2 = temperature_gradient
    return float(-np.sum(weight * d11 * (a1 + (energy - 1.5) * a2)))


@dataclasses.dataclass(frozen=True)
class Species:
    """One species entering the ambipolarity condition."""

    name: str
    mass: float
    charge_number: float
    density_m3: float
    temperature_ev: float
    density_gradient: float
    temperature_gradient: float


def hydrogenic_species(
    density_m3: float,
    electron_temperature_ev: float,
    ion_temperature_ev: float,
    density_gradient: float,
    electron_temperature_gradient: float,
    ion_temperature_gradient: float,
) -> tuple[Species, ...]:
    """Electrons and one hydrogenic ion, the pure-hydrogen pair."""
    return (
        Species(
            "electron", ELECTRON_MASS, -1.0, density_m3, electron_temperature_ev,
            density_gradient, electron_temperature_gradient,
        ),
        Species(
            "ion", PROTON_MASS, 1.0, density_m3, ion_temperature_ev,
            density_gradient, ion_temperature_gradient,
        ),
    )


def carbon_species(
    composition,
    electron_temperature_ev: float,
    ion_temperature_ev: float,
    density_gradient: float,
    electron_temperature_gradient: float,
    ion_temperature_gradient: float,
    impurity_density_gradient: float | None = None,
    carbon_mass: float = 12.0 * PROTON_MASS,
) -> tuple[Species, ...]:
    """Electrons, diluted main ions and carbon at its local mean charge; the impurity
    gradient defaults to the electron one."""
    charge = float(np.atleast_1d(composition.charge)[0])
    n_e = float(np.atleast_1d(composition.electron_density_m3)[0])
    n_ion = float(np.atleast_1d(composition.ion_density_m3)[0])
    n_impurity = float(np.atleast_1d(composition.impurity_density_m3)[0])
    impurity_gradient = (
        density_gradient
        if impurity_density_gradient is None
        else impurity_density_gradient
    )
    return (
        Species(
            "electron", ELECTRON_MASS, -1.0, n_e, electron_temperature_ev,
            density_gradient, electron_temperature_gradient,
        ),
        Species(
            "ion", PROTON_MASS, 1.0, n_ion, ion_temperature_ev,
            density_gradient, ion_temperature_gradient,
        ),
        Species(
            "carbon", carbon_mass, charge, n_impurity, ion_temperature_ev,
            impurity_gradient, ion_temperature_gradient,
        ),
    )


def ambipolar_field_species(
    coefficients: MonoenergeticCoefficients,
    species: tuple[Species, ...],
    z_effective: float = 1.0,
    bracket: tuple[float, float] = (-40.0e3, 40.0e3),
    num_probe: int = 81,
    num_energy: int = 64,
) -> dict:
    """Radial electric field at which sum_a Z_a Gamma_a vanishes over the given species."""
    fields = np.linspace(bracket[0], bracket[1], num_probe)
    residual = np.array(
        [
            sum(
                item.charge_number
                * item.density_m3
                * particle_flux(
                    coefficients, item.density_m3, item.temperature_ev,
                    item.density_gradient, item.temperature_gradient,
                    item.mass, item.charge_number, z_effective, field, num_energy,
                )
                for item in species
            )
            for field in fields
        ]
    )
    return _roots_of(fields, residual)


def ambipolar_field(
    coefficients: MonoenergeticCoefficients,
    density_m3: float,
    electron_temperature_ev: float,
    ion_temperature_ev: float,
    density_gradient: float,
    electron_temperature_gradient: float,
    ion_temperature_gradient: float,
    z_effective: float = 1.0,
    bracket: tuple[float, float] = (-40.0e3, 40.0e3),
    num_probe: int = 81,
    num_energy: int = 64,
) -> dict:
    """Ambipolar E_r where Gamma_i = Gamma_e; every root returned, the largest-magnitude one operating."""
    return ambipolar_field_species(
        coefficients,
        hydrogenic_species(
            density_m3, electron_temperature_ev, ion_temperature_ev, density_gradient,
            electron_temperature_gradient, ion_temperature_gradient,
        ),
        z_effective,
        bracket,
        num_probe,
        num_energy,
    )


def _roots_of(fields: np.ndarray, residual: np.ndarray) -> dict:
    """Sign changes of the ambipolarity residual, and the operating root among them."""

    roots: list[float] = []
    for index in range(len(fields) - 1):
        low, high = residual[index], residual[index + 1]
        if low == 0.0:
            roots.append(float(fields[index]))
        elif low * high < 0.0:
            # Linear interpolation on the bracket is enough: the probe spacing is
            # 1 kV/m and the residual is smooth between sign changes.
            span = high - low
            roots.append(
                float(fields[index] - low * (fields[index + 1] - fields[index]) / span)
            )

    chosen = float(roots[int(np.argmax(np.abs(roots)))]) if roots else float("nan")
    return {
        "roots": roots,
        "field": chosen,
        "probe_fields": fields,
        "residual": residual,
    }


def extrapolated_weight(
    coefficients: MonoenergeticCoefficients,
    density_m3: float,
    temperature_ev: float,
    mass: float = ELECTRON_MASS,
    charge_number: float = -1.0,
    z_effective: float = 1.0,
    radial_field_v_m: float = 0.0,
    num_energy: int = 64,
) -> dict[str, float]:
    """Share of the convolution drawn from the continued region outside the solved table."""
    thermal_speed = np.sqrt(2.0 * temperature_ev * ELEMENTARY_CHARGE / mass)
    energy, weight = maxwellian_nodes(num_energy)
    speed = thermal_speed * np.sqrt(energy)

    nu_over_v = (
        deflection_frequency(
            speed, density_m3, temperature_ev, mass, charge_number, z_effective
        )
        / speed
    )
    field_over_v = np.abs(radial_field_v_m / speed)

    interpolated = _interpolate_coefficient(
        coefficients, nu_over_v, field_over_v, "d11"
    )
    contribution = (
        weight
        * (energy - 1.5) ** 2
        * physical_d11(interpolated, speed, mass, charge_number)
    )
    total = float(np.sum(contribution))
    if total <= 0.0:
        return {"below_collisionality": 0.0, "above_collisionality": 0.0, "above_field": 0.0}

    nu_hat = coefficients.collisionality
    fields = coefficients.radial_field

    def share(mask: np.ndarray) -> float:
        return float(np.sum(contribution[mask]) / total)

    return {
        "below_collisionality": share(nu_over_v < nu_hat.min()),
        "above_collisionality": share(nu_over_v > nu_hat.max()),
        "above_field": share(field_over_v > fields.max()),
    }


def heat_diffusivity(
    coefficients: MonoenergeticCoefficients,
    density_m3: float,
    temperature_ev: float,
    mass: float = ELECTRON_MASS,
    charge_number: float = -1.0,
    z_effective: float = 1.0,
    radial_field_v_m: float = 0.0,
    num_energy: int = 64,
) -> float:
    """Heat diffusivity chi = (2/sqrt(pi)) int dK sqrt(K) e^-K (K - 3/2)^2 D_11(K), in m^2/s."""
    thermal_speed = np.sqrt(2.0 * temperature_ev * ELEMENTARY_CHARGE / mass)
    energy, weight = maxwellian_nodes(num_energy)
    speed = thermal_speed * np.sqrt(energy)

    nu = deflection_frequency(
        speed, density_m3, temperature_ev, mass, charge_number, z_effective
    )
    interpolated = _interpolate_coefficient(
        coefficients, nu / speed, radial_field_v_m / speed, "d11"
    )
    d11_physical = physical_d11(interpolated, speed, mass, charge_number)
    return float(np.sum(weight * (energy - 1.5) ** 2 * d11_physical))


def heat_flux(
    coefficients: MonoenergeticCoefficients,
    density_m3: float,
    temperature_ev: float,
    density_gradient: float,
    temperature_gradient: float,
    mass: float = ELECTRON_MASS,
    charge_number: float = -1.0,
    z_effective: float = 1.0,
    radial_field_v_m: float = 0.0,
    num_energy: int = 64,
) -> float:
    """Q_a / (n_a T_a) = -(2/sqrt(pi)) int dK sqrt(K) e^-K (K - 3/2) D_11 [A1 + (K - 3/2) A2], in m/s."""
    thermal_speed = np.sqrt(2.0 * temperature_ev * ELEMENTARY_CHARGE / mass)
    energy, weight = maxwellian_nodes(num_energy)
    speed = thermal_speed * np.sqrt(energy)

    nu = deflection_frequency(
        speed, density_m3, temperature_ev, mass, charge_number, z_effective
    )
    interpolated = _interpolate_coefficient(
        coefficients, nu / speed, radial_field_v_m / speed, "d11"
    )
    d11 = physical_d11(interpolated, speed, mass, charge_number)

    charge = charge_number * ELEMENTARY_CHARGE
    a1 = density_gradient - charge * radial_field_v_m / (
        temperature_ev * ELEMENTARY_CHARGE
    )
    a2 = temperature_gradient
    return float(
        -np.sum(weight * (energy - 1.5) * d11 * (a1 + (energy - 1.5) * a2))
    )
