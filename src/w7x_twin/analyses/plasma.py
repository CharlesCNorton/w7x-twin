"""Profiles, transport, current and heating, solved and held against each other."""

from __future__ import annotations

import dataclasses
import time
from pathlib import Path

import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.analyses._common import arg, args, write_record
from w7x_twin.hardware.walls import inside_contour
from w7x_twin.mhd import diagnostics
from w7x_twin.mhd.equilibrium import SCAN, Twin
from w7x_twin.plasma import current, edge, kinetics, neoclassical, transport
from w7x_twin.plasma.kinetics import log_gradient
from w7x_twin.records import programmes

#: The surface cache/monkes_er.dat was solved on, taken from the package so every
#: entry that falls back to the single-surface table normalises the same way.
REFERENCE_SURFACE = neoclassical.SINGLE_SURFACE

load_coefficients = _common.drift_kinetic_coefficients

#: Ion temperature as a fraction of the electron temperature, matching the transport
#: model's own ratio.
ION_FRACTION = 0.55
#: Radial surfaces the balances and depositions are sampled on.
SURFACES = 81
#: Beam injection energy of the W7-X neutral beams, in eV.
BEAM_ENERGY_EV = 55.0e3
#: Particles per second the fuelling supplies.
THROUGHPUT = 1.0e22


def build_neoclassical(coefficients, ripple, radial_field_v_m=0.0, minor_radius=0.49):
    """chi_neo(s, T_e, n), from the one model in w7x_twin.neoclassical."""
    return neoclassical.diffusivity_model(
        coefficients, ripple, minor_radius,
        radial_field_v_m=radial_field_v_m,
        ion_fraction=ION_FRACTION,
        reference_surface=REFERENCE_SURFACE,
    )


# -- deposition --------------------------------------------------------------------

# Heating deposition from the |B| resonance and the attenuated beam path.
#
#     python -m w7x_twin deposition [heating MW]

DEPOSITION_OUT = Path("results/plasma/deposition.json")

#: Launch of the ray-traced case: the outer midplane at the bean plane, aimed at the
#: magnetic axis, which is the central-deposition aiming. The port coordinates are an
#: assumption the record carries; the refraction along the path is not.
RAY_LAUNCH_RPZ = (6.65, 0.0, 0.15)
#: Beam radius at the waist, in metres, which floors the deposition width.
RAY_WAIST_M = 0.025
#: Field band about the resonance counted as the absorbing layer, in tesla.
RESONANCE_LAYER_T = 0.02


def flux_label_map(equilibrium, phi: float, points: int = 161):
    """s on an (R, Z) grid at one plane by innermost-containing-surface lookup, one outside the boundary."""
    surfaces = int(equilibrium.wout.ns)
    r_edge, z_edge = diagnostics.boundary_cut(equilibrium.wout, phi, 128)
    r_axis = np.linspace(float(r_edge.min()) - 0.05, float(r_edge.max()) + 0.05, points)
    z_axis = np.linspace(float(z_edge.min()) - 0.05, float(z_edge.max()) + 0.05, points)
    grid_r, grid_z = np.meshgrid(r_axis, z_axis, indexing="ij")
    flat_r, flat_z = grid_r.ravel(), grid_z.ravel()
    label = np.ones(flat_r.shape)
    for surface in range(surfaces - 1, 0, -1):
        r_c, z_c = diagnostics.flux_surface(equilibrium.wout, surface, phi, 128)
        inside = inside_contour(flat_r, flat_z, r_c, z_c)
        label[inside] = surface / (surfaces - 1)
    from scipy.interpolate import RegularGridInterpolator

    return RegularGridInterpolator(
        (r_axis, z_axis), label.reshape(points, points),
        bounds_error=False, fill_value=1.0,
    )


def ray_traced_deposition(
    twin, equilibrium, minor_radius_m: float, profiles, s: np.ndarray, power_w: float
):
    """The traced ray's deposition and the ray itself; the deposition is None off-crossing."""
    from w7x_twin.magnetics.field import VacuumField

    vacuum = VacuumField(twin.response, twin.state("standard").currents)
    label_of = flux_label_map(equilibrium, RAY_LAUNCH_RPZ[1])

    def field_cartesian(point):
        radius = float(np.hypot(point[0], point[1]))
        angle = float(np.arctan2(point[1], point[0]))
        br, bp, bz = vacuum(radius, angle, float(point[2]))
        br, bp, bz = float(br[0]), float(bp[0]), float(bz[0])
        return np.array(
            [
                br * np.cos(angle) - bp * np.sin(angle),
                br * np.sin(angle) + bp * np.cos(angle),
                bz,
            ]
        )

    def density_at(point):
        radius = float(np.hypot(point[0], point[1]))
        s_local = float(label_of((radius, float(point[2]))))
        return float(profiles.density(np.array([min(s_local, 1.0)]))[0])

    axis_r, axis_z = diagnostics._axis_position(equilibrium.wout)
    launch = np.array(
        [
            RAY_LAUNCH_RPZ[0] * np.cos(RAY_LAUNCH_RPZ[1]),
            RAY_LAUNCH_RPZ[0] * np.sin(RAY_LAUNCH_RPZ[1]),
            RAY_LAUNCH_RPZ[2],
        ]
    )
    aim = np.array(
        [axis_r * np.cos(RAY_LAUNCH_RPZ[1]), axis_r * np.sin(RAY_LAUNCH_RPZ[1]), axis_z]
    )
    ray = transport.trace_cyclotron_ray(field_cartesian, density_at, launch, aim - launch)
    if not ray.crossed:
        return None, ray

    in_layer = np.abs(ray.field_t - transport.resonant_field_t()) < RESONANCE_LAYER_T
    points_in_layer = ray.path_m[: len(ray.field_t)][in_layer]
    radius_in = np.hypot(points_in_layer[:, 0], points_in_layer[:, 1])
    s_layer = np.array(
        [
            float(label_of((float(r_value), float(z_value))))
            for r_value, z_value in zip(radius_in, points_in_layer[:, 2])
        ]
    )
    centre = float(np.clip(np.mean(s_layer), 0.0, 1.0))
    waist_s = 2.0 * np.sqrt(max(centre, 1e-3)) * RAY_WAIST_M / minor_radius_m
    sigma = float(max(np.hypot(np.std(s_layer), waist_s), 0.01))
    shape = np.exp(-0.5 * ((s - centre) / sigma) ** 2)
    density_max = float(np.max(profiles.density(s)))
    reachable = density_max < transport.cutoff_density_m3()
    path_length = float(np.sum(np.linalg.norm(np.diff(ray.path_m, axis=0), axis=1)))
    profile_w = (
        power_w * (1.0 if reachable else 0.0) * shape
        / max(float(np.trapezoid(shape, s)), 1e-30)
    )
    traced = transport.Deposition(
        s=s,
        profile_w=profile_w,
        absorbed_fraction=1.0 if reachable else 0.0,
        peak_s=centre,
        note=(
            f"crossing at s = {centre:.3f} after {path_length:.2f} m, "
            f"layer width {sigma:.3f} in s"
        ),
    )
    return traced, ray


def run_deposition() -> int:
    power = 1e6 * arg(1, float, 5.0)

    twin = _common.twin()
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    analysis = diagnostics.analyse(equilibrium)
    print(f"{twin.geometry}")
    print(
        f"resonant field for X{transport.ECRH_HARMONIC} at "
        f"{transport.ECRH_FREQUENCY_HZ / 1e9:.0f} GHz is "
        f"{transport.resonant_field_t():.4f} T, and the axis carries "
        f"{analysis.b_axis_t:.4f} T"
    )
    cutoff = transport.cutoff_density_m3()
    print(f"the X{transport.ECRH_HARMONIC} cut-off density is {cutoff:.3e} m^-3")

    s = np.linspace(0.0, 1.0, SURFACES)
    profiles = kinetics.HIGH_PERFORMANCE
    density = profiles.density(s)
    temperature = profiles.electron_temperature(s)

    # |B| across each surface. The field falls as 1/R, so a surface spans a range and the
    # resonance is a layer inside it rather than a value the surface average can carry.
    axis_r = diagnostics._axis_position(equilibrium.wout)[0]
    field_min = np.empty_like(s)
    field_max = np.empty_like(s)
    for index, value in enumerate(s):
        surface = min(
            int(round(value * (int(equilibrium.wout.ns) - 1))),
            int(equilibrium.wout.ns) - 1,
        )
        r, _ = diagnostics.flux_surface(equilibrium.wout, surface, 0.0)
        local = analysis.b_axis_t * axis_r / np.maximum(r, 1e-6)
        field_min[index] = float(np.min(local))
        field_max[index] = float(np.max(local))
    resonant = transport.resonant_field_t()
    reached = (field_min <= resonant) & (resonant <= field_max)
    print(
        f"|B| spans {field_min.min():.4f} to {field_max.max():.4f} T over the plasma, and "
        f"the resonance is inside {int(np.count_nonzero(reached))} of {len(s)} surfaces"
    )
    # The field the resonance layer is measured against is the closest approach on each
    # surface, which is where the beam meets it.
    field = np.where(
        reached, resonant, np.where(field_max < resonant, field_max, field_min)
    )

    rows = []
    table = _common.Table(
        ("scheme", ">26s"), ("peak s", "7.3f"), ("absorbed", "9.3f"), ("note", "s")
    )
    print()
    table.begin(extra=20)

    for label, deposition in (
        (
            f"electron cyclotron X{transport.ECRH_HARMONIC}",
            transport.cyclotron_deposition(s, field, density, temperature, power),
        ),
        (
            "neutral beam, slowing down",
            transport.beam_deposition(
                s, density, temperature, power,
                injection_energy_ev=BEAM_ENERGY_EV,
                minor_radius_m=analysis.minor_radius_m,
            ),
        ),
    ):
        rows.append(
            {
                "scheme": label,
                "peak_s": deposition.peak_s,
                "absorbed_fraction": deposition.absorbed_fraction,
                "note": deposition.note,
                "profile_w": deposition.profile_w.tolist(),
            }
        )
        table.row(label, deposition.peak_s, deposition.absorbed_fraction, deposition.note)

    # The same resonance reached along a traced ray: the launch is an aiming
    # assumption, the bending through the density it crosses is not, so where the
    # crossing lands against the layer model is what the refraction is worth.
    traced, ray = ray_traced_deposition(
        twin, equilibrium, analysis.minor_radius_m, profiles, s, power
    )
    ray_row = {
        "scheme": f"electron cyclotron X{transport.ECRH_HARMONIC}, ray-traced",
        "note": ray.note,
        "path_length_m": float(
            np.sum(np.linalg.norm(np.diff(ray.path_m, axis=0), axis=1))
        ),
    }
    if traced is not None:
        rows.append({**ray_row, "peak_s": traced.peak_s,
                     "absorbed_fraction": traced.absorbed_fraction,
                     "note": traced.note, "profile_w": traced.profile_w.tolist()})
        table.row(
            "electron cyclotron, ray", traced.peak_s, traced.absorbed_fraction,
            traced.note,
        )
    else:
        rows.append({**ray_row, "peak_s": float("nan"), "absorbed_fraction": 0.0})
        print(f"{'electron cyclotron, ray':>26s} {'—':>7s} {'0.000':>9s}  {ray.note}")

    ion, electron = transport.beam_power_split(BEAM_ENERGY_EV, temperature)
    print()
    print(
        f"a {BEAM_ENERGY_EV / 1e3:.0f} keV beam gives {100 * float(ion[0]):.1f} per cent of "
        f"its power to the ions on axis and {100 * float(ion[-1]):.1f} per cent at the edge, "
        f"the critical energy running "
        f"{transport.critical_energy_ev(temperature)[0] / 1e3:.1f} to "
        f"{transport.critical_energy_ev(temperature)[-1] / 1e3:.1f} keV"
    )

    write_record(
        DEPOSITION_OUT,
        {
            "heating_w": power,
            "resonant_field_t": transport.resonant_field_t(),
            "cutoff_density_m3": cutoff,
            "beam_energy_ev": BEAM_ENERGY_EV,
            "s": s.tolist(),
            "field_t": field.tolist(),
            "ion_power_fraction": ion.tolist(),
            "electron_power_fraction": electron.tolist(),
            "schemes": rows,
        },
        geometry=twin.geometry,
    )
    return 0


# -- density -----------------------------------------------------------------------

# Density from a particle balance: the peaking follows the source location, edge gas to core pellet.
#
#     python -m w7x_twin density

DENSITY_OUT = Path("results/plasma/solved_density.json")
DENSITY_SURFACES = 121
#: Where a source can sit, in normalised toroidal flux: the separatrix for gas, inside it for
#: a pellet.
CENTRES = (0.95, 0.80, 0.60, 0.40, 0.20, 0.05)


def run_density() -> int:
    twin = _common.twin()
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    analysis = diagnostics.analyse(equilibrium)
    minor = analysis.minor_radius_m
    print(f"{twin.geometry}")
    print(f"minor radius {minor:.4f} m, {THROUGHPUT:.1e} particles per second")

    reference = kinetics.HIGH_PERFORMANCE
    published = programmes.MACHINE_MEASUREMENTS["pellet_density_peaking"]
    low, high = published.band()
    print(
        f"the prescribed profile peaks at {reference.peaking():.3f}, against a published "
        f"{low:.1f} to {high:.1f}"
    )

    s = np.linspace(0.0, 1.0, DENSITY_SURFACES)
    edge_density = reference.density_edge_m3

    table = _common.Table(
        ("source s", "9.2f"), ("peaking", "8.3f"), ("n(0) [m^-3]", "13.3e"),
        ("inside published", ">17s"),
    )
    print()
    table.begin()

    rows = []
    for centre in CENTRES:
        solution = transport.peaking_for_source(
            centre, s, minor, edge_density, THROUGHPUT
        )
        inside = bool(low <= solution.peaking <= high)
        rows.append(
            {
                "source_centre_s": centre,
                "peaking": solution.peaking,
                "axis_density_m3": float(solution.density_m3[0]),
                "inside_published": inside,
            }
        )
        table.row(centre, solution.peaking, solution.density_m3[0], "yes" if inside else "no")

    # The pinch is the one free parameter of the closure, so it is solved from the published
    # peaking rather than guessed: at a fixed diffusivity the peaking rises monotonically with
    # the inward velocity, which makes it a bisection.
    def peaking_at(pinch: float, centre: float = 0.95) -> float:
        model = transport.ParticleModel(pinch_m_s=pinch)
        return transport.peaking_for_source(
            centre, s, minor, edge_density, THROUGHPUT, model=model
        ).peaking

    low_v, high_v = -3.0, 0.0
    for _ in range(60):
        middle = 0.5 * (low_v + high_v)
        if peaking_at(middle) > published.value:
            low_v = middle
        else:
            high_v = middle
    pinch = 0.5 * (low_v + high_v)
    calibrated = transport.ParticleModel(pinch_m_s=pinch)
    print()
    print(
        f"an edge source reaches the published peaking of {published.value:.2f} at an inward "
        f"pinch of {abs(pinch):.3f} m/s against a diffusivity of "
        f"{calibrated.diffusivity_m2_s:.2f} m2/s"
    )
    table = _common.Table(
        ("source s", "9.2f"), ("peaking", "8.3f"), ("inside published", ">17s")
    )
    print()
    table.begin()
    at_pinch = []
    for centre in CENTRES:
        solution = transport.peaking_for_source(
            centre, s, minor, edge_density, THROUGHPUT, model=calibrated
        )
        inside = bool(low <= solution.peaking <= high)
        at_pinch.append(
            {"source_centre_s": centre, "peaking": solution.peaking,
             "inside_published": inside}
        )
        table.row(centre, solution.peaking, "yes" if inside else "no")

    # The evolution the pellets drive: from the flat-topped reference, a deposition
    # inside the separatrix marched in time until the peaking passes the published
    # value, so the answer carries when the profile gets there and not only whether.
    pellet = transport.gaussian_source(s, 0.30, 0.15, THROUGHPUT, minor)
    evolution = transport.evolve_density(
        reference.density(s), pellet, s, minor, edge_density,
        model=calibrated, peaking_target=published.value,
    )
    reached = evolution["time_to_target_s"]
    print()
    if np.isfinite(reached):
        print(
            f"a pellet source at s = 0.30 takes the reference profile from a peaking of "
            f"{evolution['peaking'][0]:.2f} to {published.value:.1f} in {reached:.3f} s"
        )
    else:
        print(
            f"a pellet source at s = 0.30 saturates at a peaking of "
            f"{evolution['peaking'][-1]:.2f} within {evolution['times_s'][-1]:.2f} s, "
            f"short of the published {published.value:.1f}"
        )

    # Gas fuelling is not a free choice of centre: the neutrals ionise where the density and
    # temperature put them, so the source follows from the profile it is fuelling.
    density = reference.density(s)
    temperature = reference.electron_temperature(s)
    edge_flux = THROUGHPUT / (4.0 * np.pi**2 * 5.5 * minor)
    ionisation = edge.ionisation_source_profile(
        s, density, temperature, minor, edge_flux
    )
    gas = transport.solve_density(ionisation, s, minor, edge_density, calibrated)
    weighted = float(
        np.trapezoid(ionisation * s, s)
        / max(np.trapezoid(ionisation, s), 1e-30)
    )
    print()
    print(
        f"recycling neutrals ionise at a power-weighted s of {weighted:.3f} and give a "
        f"peaking of {gas.peaking:.3f}"
    )
    inside_gas = bool(low <= gas.peaking <= high)
    print(
        f"which is {'inside' if inside_gas else 'outside'} the published {low:.1f} to "
        f"{high:.1f}"
    )

    write_record(
        DENSITY_OUT,
        {
            "minor_radius_m": minor,
            "throughput_per_s": THROUGHPUT,
            "prescribed_peaking": reference.peaking(),
            "published_band": [low, high],
            "sources": rows,
            "calibrated_pinch_m_s": pinch,
            "sources_at_calibrated_pinch": at_pinch,
            "gas_fuelled": {
                "ionisation_centroid_s": weighted,
                "peaking": gas.peaking,
                "inside_published": inside_gas,
            },
            "pellet_evolution": {
                "source_centre_s": 0.30,
                "initial_peaking": float(evolution["peaking"][0]),
                "final_peaking": float(evolution["peaking"][-1]),
                "time_to_published_s": float(reached),
                "times_s": [float(v) for v in evolution["times_s"]],
                "peaking": [float(v) for v in evolution["peaking"]],
            },
        },
        geometry=twin.geometry,
    )
    return 0


# -- efield ------------------------------------------------------------------------

# Ambipolar radial electric field per surface, every ion and electron root reported.
#
#     python -m w7x_twin efield [temperature scale]

EFIELD_OUT = Path("results/plasma/ambipolar_field.json")


#: Surfaces the field is solved on.
EFIELD_SURFACES = (0.05, 0.15, 0.25, 0.40, 0.55, 0.70, 0.85)

#: The two operating points, each solved through its own equilibrium and power balance.
CASES = (
    ("high performance", kinetics.HIGH_PERFORMANCE, 5.0e6),
    (
        "low density",
        dataclasses.replace(
            kinetics.HIGH_PERFORMANCE,
            density_axis_m3=2.0e19,
            density_edge_m3=0.3e19,
        ),
        5.0e6,
    ),
)




def solve_case(
    twin: Twin, profiles, heating_w: float, coefficients, ripple, scale: float
) -> list[dict]:
    """The ambipolar field across the profile of one operating point."""
    equilibrium = twin.solve_profiles("standard", profiles)

    # The temperature profile the power balance returns, so the gradients driving the
    # field are the solved ones rather than the prescribed shape.
    solution = transport.solve(
        equilibrium,
        profiles,
        heating=transport.Heating(power_w=heating_w),
        model=transport.TransportModel(
            renormalisation=transport.PUBLISHED_ISS04_ENHANCEMENT
        ),
        neoclassical=None,
    )
    minor = float(equilibrium.wout.Aminor_p)
    radius = minor * np.sqrt(solution.s)

    dln_n = log_gradient(solution.density_m3, radius)
    dln_te = log_gradient(solution.electron_temperature_ev, radius)
    dln_ti = log_gradient(solution.ion_temperature_ev, radius)

    layout = _common.Table(
        ("s", "6.3f"), ("Te [eV]", "9.0f"), ("Ti [eV]", "9.0f"), ("a/L_n", "7.2f"),
        ("a/L_Te", "8.2f"), ("roots [kV/m]", ">26s"), ("chosen", "8.2f"),
    )
    layout.begin()
    rows = []
    for surface in EFIELD_SURFACES:
        index = int(np.argmin(np.abs(solution.s - surface)))
        electron_t = scale * float(solution.electron_temperature_ev[index])
        ion_t = scale * float(solution.ion_temperature_ev[index])
        density = float(solution.density_m3[index])

        chosen: list[float] = []
        for table, weight in neoclassical.surface_tables(
            coefficients, surface, which="d11", ripple=ripple,
            reference_surface=REFERENCE_SURFACE,
        ):
            if weight == 0.0:
                continue
            answer = neoclassical.ambipolar_field(
                table,
                density_m3=density,
                electron_temperature_ev=electron_t,
                ion_temperature_ev=ion_t,
                density_gradient=float(dln_n[index]),
                electron_temperature_gradient=float(dln_te[index]),
                ion_temperature_gradient=float(dln_ti[index]),
            )
            chosen.append(weight * answer["field"])
            roots = answer["roots"]
        field = float(np.nansum(chosen))
        branch = "none" if not roots else ("electron" if field > 0.0 else "ion")
        rows.append(
            {
                "s": float(solution.s[index]),
                "electron_temperature_ev": electron_t,
                "ion_temperature_ev": ion_t,
                "density_m3": density,
                "a_over_ln": float(-minor * dln_n[index]),
                "a_over_lte": float(-minor * dln_te[index]),
                "roots_v_m": [float(r) for r in roots],
                "field_v_m": field,
                "branch": branch,
            }
        )
        layout.row(
            rows[-1]["s"], electron_t, ion_t, rows[-1]["a_over_ln"],
            rows[-1]["a_over_lte"], ", ".join(f"{r / 1e3:+.2f}" for r in roots) or "none",
            field / 1e3,
        )
    return rows


def run_efield() -> int:
    scale = arg(1, float, 1.0)
    twin = _common.twin()
    coefficients = load_coefficients()
    ripple = neoclassical.load_ripple()
    print(f"{twin.geometry}")

    cases = []
    for label, profiles, heating_w in CASES:
        print(
            f"\n{label}: axis density {profiles.density_axis_m3 / 1e19:.1f} x 10^19, "
            f"{heating_w / 1e6:.0f} MW"
        )
        rows = solve_case(twin, profiles, heating_w, coefficients, ripple, scale)
        cases.append(
            {
                "label": label,
                "heating_w": heating_w,
                "density_axis_m3": profiles.density_axis_m3,
                "temperature_scale": scale,
                "surfaces": rows,
            }
        )
        positive = [r for r in rows if r["branch"] == "electron"]
        print(
            f"{len(positive)} of {len(rows)} surfaces on the electron root"
            if positive
            else "every surface on the ion root"
        )

    write_record(
        EFIELD_OUT,
        {
            "renormalisation": transport.PUBLISHED_ISS04_ENHANCEMENT,
            "cases": cases,
        },
        geometry=twin.geometry,
    )
    return 0


# -- transport ---------------------------------------------------------------------

# Transport split: computed drift-kinetic channel held fixed, the remainder scaled to ISS04.



#: Confinement relative to the ISS04 scaling. Unity is the international scaling itself;
#: W7-X discharges are reported above it, and the enhancement moves every quantity the
#: power balance returns, so both are carried.
RENORMALISATIONS = (1.0, 1.3)





def run_transport() -> int:
    twin = _common.twin()
    profiles = kinetics.HIGH_PERFORMANCE
    equilibrium = twin.solve_profiles("standard", profiles)

    coefficients = load_coefficients()
    ripple = neoclassical.load_ripple()
    if isinstance(coefficients, neoclassical.MonoenergeticProfile):
        print(f"drift-kinetic tables on {len(coefficients)} flux surfaces")
        reference = coefficients.tables[
            int(np.argmin(np.abs(coefficients.surfaces - REFERENCE_SURFACE)))
        ]
        # The 1/nu coefficient is the surface's own transport level. Comparing its
        # radial dependence with the effective ripple to the 3/2 power tests the
        # scaling a single-surface solution has to carry.
        print(
            f"{'s':>6s} {'eps_eff':>9s} {'D11 nu/v':>11s} {'solved':>8s} "
            f"{'eps^3/2':>8s}"
        )
        base = reference.one_over_nu_plateau()
        base_ripple = float(ripple.at(REFERENCE_SURFACE))
        for table in coefficients.tables:
            plateau = table.one_over_nu_plateau()
            eps = float(ripple.at(table.s))
            print(
                f"{table.s:6.2f} {100 * eps:8.3f}% {plateau:11.4e} "
                f"{plateau / base:8.2f} {(eps / base_ripple) ** 1.5:8.2f}"
            )
        print()
    else:
        print(f"drift-kinetic table on s = {REFERENCE_SURFACE} alone")
        reference = coefficients
    print(
        f"  {len(np.unique(reference.collisionality))} collisionalities x "
        f"{len(np.unique(reference.radial_field))} radial electric fields, "
        f"1/nu coefficient {reference.one_over_nu_plateau():.4e}"
    )
    print(
        f"effective ripple at s = {REFERENCE_SURFACE}: "
        f"{float(ripple.at(REFERENCE_SURFACE)):.5f}"
    )

    header = (
        f"{'ISS04 x':>8s} {'E_r [kV/m]':>10s} {'P [MW]':>7s} {'W [MJ]':>8s} "
        f"{'Te(0)':>8s} {'chi_tot(0)':>11s} {'chi_neo(0)':>11s} {'chi_an(0)':>10s} "
        f"{'neo frac':>9s}"
    )
    print()
    print(header)
    print("-" * len(header))
    rows = []
    minor = float(equilibrium.wout.Aminor_p)
    for renormalisation in RENORMALISATIONS:
        model = transport.TransportModel(renormalisation=renormalisation)
        for field_kv in (0.0, 5.0, 10.0, 20.0, None):
            chi_neo = build_neoclassical(
                coefficients, ripple,
                None if field_kv is None else 1e3 * field_kv,
                minor_radius=minor,
            )
            powers = (2e6, 5e6, 10e6, 20e6) if field_kv == 0.0 else (5e6,)
            for power in powers:
                solution = transport.solve(
                    equilibrium,
                    profiles,
                    heating=transport.Heating(power_w=power),
                    model=model,
                    neoclassical=chi_neo,
                )
                fraction = solution.neoclassical_fraction
                rows.append((renormalisation, field_kv, power, solution))
                label = "ambipolar" if field_kv is None else f"{field_kv:.0f}"
                print(
                    f"{renormalisation:8.1f} {label:>10s} {power / 1e6:7.1f} "
                    f"{solution.stored_energy_j / 1e6:8.3f} "
                    f"{solution.electron_temperature_ev[0]:8.0f} "
                    f"{solution.chi_m2_s[0]:11.4f} "
                    f"{solution.chi_neoclassical_m2_s[0]:11.4f} "
                    f"{solution.chi_anomalous_m2_s[0]:10.4f} "
                    f"{100 * fraction[0]:8.1f}%"
                )

    print()
    print("radial profile at 5 MW, no radial electric field, ISS04 x 1.0")
    solution = [r for n, e, p, r in rows if p == 5e6 and e == 0.0 and n == 1.0][0]  # noqa: E501
    fraction = solution.neoclassical_fraction
    print(
        f"{'s':>6s} {'Te [eV]':>9s} {'chi_neo':>9s} {'chi_anom':>9s} {'neo %':>7s} "
        f"{'off table':>10s}"
    )
    for i in range(0, len(solution.s), max(1, len(solution.s) // 8)):
        # How much of chi_neo the continuation outside the solved table supplied.
        outside = 0.0
        for table, weight in neoclassical.surface_tables(
            coefficients, float(solution.s[i]), which="d11", ripple=ripple,
            reference_surface=REFERENCE_SURFACE,
        ):
            if weight == 0.0:
                continue
            shares = neoclassical.extrapolated_weight(
                table,
                density_m3=float(solution.density_m3[i]),
                temperature_ev=float(solution.electron_temperature_ev[i]),
            )
            outside += weight * sum(shares.values())
        print(
            f"{solution.s[i]:6.3f} {solution.electron_temperature_ev[i]:9.0f} "
            f"{solution.chi_neoclassical_m2_s[i]:9.4f} "
            f"{solution.chi_anomalous_m2_s[i]:9.4f} {100 * fraction[i]:6.1f}% "
            f"{100 * outside:9.1f}%"
        )
    return 0


# -- bootstrap ---------------------------------------------------------------------

# Self-consistent bootstrap by the Redl formula and by D_31, their gap decomposed input
# by input and both diffused onto the measured six-second current.

BOOTSTRAP_OUT = Path("results/plasma/bootstrap_routes.json")
#: The discharge whose measured toroidal current the diffused profiles are checked against.
MEASURED_PROGRAMME = "20180920.009"
#: Time the source states that current at, in seconds.
MEASURED_TIME_S = 6.0
#: Carbon content the attribution runs at, so both routes see one effective charge.
CARBON_FRACTION = 0.02
#: The discharge whose measured confinement the gas-fuelled profile is solved at.
GAS_FUELLED_PROGRAMME = "20180920.017"


#: Radial electric field the drift-kinetic drive is evaluated at, in V/m.
RADIAL_FIELD_V_M = 0.0




enclosed_current = current.enclosed_current_a


def ambipolar_field(equilibrium, profiles, coefficients, ripple) -> np.ndarray:
    """The radial electric field the ambipolarity condition fixes, on the half grid."""
    s = current.half_grid(int(equilibrium.wout.ns))
    minor = float(equilibrium.wout.Aminor_p)
    radius = minor * np.sqrt(s)
    density = profiles.density(s)
    electron = profiles.electron_temperature(s)
    ion = profiles.ion_temperature(s)

    dln_n = log_gradient(density, radius)
    dln_te = log_gradient(electron, radius)
    dln_ti = log_gradient(ion, radius)
    out = np.zeros_like(s)
    for index in range(len(s)):
        for table, weight in neoclassical.surface_tables(
            coefficients, float(s[index]), which="d11", ripple=ripple
        ):
            if weight == 0.0:
                continue
            answer = neoclassical.ambipolar_field(
                table, density_m3=float(density[index]),
                electron_temperature_ev=float(electron[index]),
                ion_temperature_ev=float(ion[index]),
                density_gradient=float(dln_n[index]),
                electron_temperature_gradient=float(dln_te[index]),
                ion_temperature_gradient=float(dln_ti[index]),
                bracket=(-25.0e3, 25.0e3), num_probe=41,
            )
            field = float(answer["field"])
            out[index] = field if np.isfinite(field) else 0.0
            break
    return out


def diffuse(equilibrium, s_target, jdotb, balance, profiles, samples: int = 121) -> dict:
    """Diffuse the saturated bootstrap profile at the balance's own Spitzer resistivity."""
    minor = float(equilibrium.wout.Aminor_p)
    radius = np.linspace(0.0, minor, samples)
    s = np.clip((radius / minor) ** 2, 0.0, 1.0)
    temperature = np.interp(s, balance.s, balance.electron_temperature_ev)
    density = np.interp(s, balance.s, balance.density_m3)
    charge = np.asarray(profiles.z_effective_profile(s), dtype=float)
    resistivity = current.spitzer_resistivity(temperature, density, charge)

    bootstrap = np.interp(
        s, s_target, np.asarray(jdotb) / abs(float(equilibrium.wout.b0))
    )
    times = np.linspace(0.0, 4.0 * MEASURED_TIME_S, 400)
    evolved = current.evolve(radius, resistivity, bootstrap, times)
    return {
        "resistive_time_s": float(evolved.resistive_time_s),
        "current_at_measured_time_a": float(
            np.interp(MEASURED_TIME_S, times, evolved.enclosed_current_a)
        ),
        "saturated_current_a": float(evolved.enclosed_current_a[-1]),
    }


def run_bootstrap() -> int:
    twin = _common.twin()
    profiles = kinetics.HIGH_PERFORMANCE

    coefficients = load_coefficients()
    ripple = neoclassical.load_ripple()
    if isinstance(coefficients, neoclassical.MonoenergeticProfile):
        print(f"drift-kinetic tables on {len(coefficients)} flux surfaces")
    else:
        print(f"drift-kinetic table on s = {REFERENCE_SURFACE} alone")

    equilibrium = twin.solve(twin.state("standard"), SCAN)
    helicity, amplitudes = current.dominant_helicity(equilibrium)
    print(
        f"symmetry direction measured from the |B| spectrum: n = {helicity}, "
        + ", ".join(f"{k} {v:+.4f} B00" for k, v in amplitudes.items())
    )
    print()

    results = {}
    for label, keywords in (
        ("Redl analytic formula", {"target": "redl"}),
        (
            "drift-kinetic D_31",
            {
                "target": "drift_kinetic",
                "coefficients": coefficients,
                "ripple": ripple,
                "radial_field_v_m": RADIAL_FIELD_V_M,
            },
        ),
    ):
        print(label)
        solution = current.solve_self_consistent(
            twin, "standard", profiles, verbose=True, **keywords
        )
        results[label] = solution
        print(
            f"  I_tor = {solution.total_current_a:.0f} A after "
            f"{solution.iterations} iterations, mismatch "
            f"{100 * solution.mismatch:.2f} %\n"
        )

    # The direction is checkable against the machine. Its OP1.2a current-mimic tapers
    # reproduce, with the planar coils alone, the equilibrium a given bootstrap current
    # would produce, and their edge transform rises from 0.878 to 0.952 across 0 to
    # 43 kA. A bootstrap current must therefore raise the edge transform here too;
    # passing the toroidal flux to the Redl formula with the wrong sign reverses it.
    pressure_only = twin.solve_profiles("standard", profiles)
    reference_iota = float(np.asarray(pressure_only.wout.iotaf)[-1])

    header = (
        f"{'drive':24s} {'I_tor [kA]':>11s} {'mismatch':>9s} {'<beta> [%]':>11s} "
        f"{'iota_edge':>10s} {'against pressure alone':>24s}"
    )
    print(header)
    print("-" * len(header))
    for label, solution in results.items():
        iota_edge = float(np.asarray(solution.output.wout.iotaf)[-1])
        direction = "raises" if iota_edge > reference_iota else "lowers"
        print(
            f"{label:24s} {solution.total_current_a / 1e3:11.2f} "
            f"{100 * solution.mismatch:8.2f}% "
            f"{100 * float(solution.output.wout.betatotal):11.3f} "
            f"{iota_edge:10.5f} {direction + ' it, ' + format(iota_edge - reference_iota, '+.5f'):>24s}"
        )
    print(f"\npressure alone at the same beta gives iota_edge {reference_iota:.5f}")

    # Where the two routes differ, on the same equilibrium. The self-consistent solves
    # above each carry their own equilibrium, so the profiles are re-formed on one of them
    # and every difference below is an input difference and not a geometry difference.
    print()
    print("attribution of the difference, on one equilibrium")
    seeded = dataclasses.replace(profiles, carbon_fraction=CARBON_FRACTION)
    # The finite-beta, current-carrying equilibrium the drift-kinetic route converged to,
    # not the vacuum one: a bootstrap comparison on a pressureless equilibrium compares two
    # formulas on geometry neither of them would be evaluated on.
    equilibrium = results["drift-kinetic D_31"].output
    s_redl, redl = current.redl_jdotb(equilibrium, seeded)
    ambipolar = ambipolar_field(equilibrium, seeded, coefficients, ripple)
    print(
        f"  carbon at {100 * CARBON_FRACTION:.0f} per cent, so the effective charge runs "
        f"{seeded.z_effective_profile(s_redl).min():.2f} to "
        f"{seeded.z_effective_profile(s_redl).max():.2f}"
    )
    print(
        f"  the ambipolar field averages {np.mean(ambipolar):+.0f} V/m and reaches "
        f"{np.max(np.abs(ambipolar)):.0f}"
    )

    variants = {
        "monoenergetic D_31 as solved": {},
        "the profile's effective charge": {"z_effective": True},
        "the ambipolar radial field": {"radial_field_v_m": ambipolar},
        "electron momentum restored": {"momentum_correction": True},
        "all three together": {
            "z_effective": True,
            "radial_field_v_m": ambipolar,
            "momentum_correction": True,
        },
    }
    header = (
        f"{'drift-kinetic drive with'.ljust(32)} {'median gap':>11s} {'worst gap':>10s} "
        f"{'I_tor [kA]':>11s} {'gap closed':>11s}"
    )
    print(header)
    print("-" * len(header))
    attribution = []
    baseline_gap = None
    for label, keywords in variants.items():
        _, drive = current.drift_kinetic_jdotb(
            equilibrium, seeded, coefficients, ripple, **keywords
        )
        relative = np.abs(drive - redl) / np.maximum(np.abs(redl), 1e-30)
        usable = np.isfinite(relative) & (np.abs(redl) > 0.02 * np.max(np.abs(redl)))
        median_gap = float(np.median(relative[usable]))
        worst_gap = float(np.max(relative[usable]))
        total = enclosed_current(equilibrium, s_redl, drive)
        if baseline_gap is None:
            baseline_gap = median_gap
        closed = 1.0 - median_gap / baseline_gap
        attribution.append(
            {
                "drive": label, "median_gap": median_gap, "worst_gap": worst_gap,
                "total_current_a": total, "share_of_gap_closed": closed,
            }
        )
        print(
            f"{label:32s} {100 * median_gap:10.1f}% {100 * worst_gap:9.1f}% "
            f"{total / 1e3:11.2f} {100 * closed:10.1f}%"
        )
    redl_total = enclosed_current(equilibrium, s_redl, redl)
    print(f"  the Redl formula on the same equilibrium encloses {redl_total / 1e3:.2f} kA")

    # Both routes diffused through the plasma's own resistivity, against the one measured
    # toroidal current available. A saturated bootstrap current is not what a discharge
    # carries six seconds in: the resistive time of this plasma is of that order. The
    # profile is that discharge's own, solved at its published power and at the confinement
    # gas-fuelled operation is measured to run at, not the reference profile's.
    print()
    programme = programmes.get(MEASURED_PROGRAMME)
    measured = programme.measured["toroidal_current_a"]
    power = programme.measured["heating_power_ecrh_w"].value
    enhancement = programmes.get(GAS_FUELLED_PROGRAMME).measured[
        "confinement_over_iss04"
    ].value
    balance = transport.solve(
        equilibrium, seeded, heating=transport.Heating(power_w=power),
        model=transport.TransportModel(renormalisation=enhancement),
    )
    discharge_profiles = balance.as_kinetic_profiles()
    print(
        f"{MEASURED_PROGRAMME} at {power / 1e6:.1f} MW and {enhancement:.2f} times ISS04 "
        f"holds {balance.stored_energy_j / 1e6:.3f} MJ, reaching "
        f"{balance.electron_temperature_ev[0] / 1e3:.2f} keV on axis"
    )
    diffused = []
    for label, keywords in (
        ("Redl analytic formula", None),
        ("D_31 at no radial field", {"z_effective": True}),
        ("D_31 at the ambipolar field", {"z_effective": True, "radial_field_v_m": ambipolar}),
    ):
        if keywords is None:
            s_drive, drive = current.redl_jdotb(equilibrium, discharge_profiles)
        else:
            s_drive, drive = current.drift_kinetic_jdotb(
                equilibrium, discharge_profiles, coefficients, ripple, **keywords
            )
        saturated = enclosed_current(equilibrium, s_drive, drive)
        answer = diffuse(equilibrium, s_drive, drive, balance, seeded)
        reached = answer["current_at_measured_time_a"]
        low, high = measured.band()
        diffused.append(
            {
                "drive": label, "saturated_a": saturated, **answer,
                "within_published_accuracy": bool(low <= abs(reached) <= high),
            }
        )
        print(
            f"  {label:28s} saturates at {saturated / 1e3:6.2f} kA, and reaches "
            f"{reached / 1e3:6.2f} kA at {MEASURED_TIME_S:.0f} s against a measured "
            f"{measured.value / 1e3:.1f} kA, "
            f"{100 * (abs(reached) / measured.value - 1):+.0f} per cent"
        )
    print(
        f"  the resistive time is {diffused[0]['resistive_time_s']:.2f} s, so six seconds "
        f"is {MEASURED_TIME_S / diffused[0]['resistive_time_s']:.1f} of them and the "
        f"measurement is of a nearly saturated current"
    )

    # None of the inputs closes the gap, so what is left is where the monoenergetic
    # coefficient is being read. The convolution samples the whole energy range and part of
    # it falls outside the collisionalities the drift-kinetic equation was solved on, where
    # the coefficient is continued rather than computed.
    s_probe = current.half_grid(int(equilibrium.wout.ns))
    coverage = []
    for fraction in (0.2, 0.5, 0.8):
        index = int(np.argmin(np.abs(s_probe - fraction)))
        for table, weight in neoclassical.surface_tables(
            coefficients, float(s_probe[index]), which="d31", ripple=ripple
        ):
            if weight == 0.0:
                continue
            share = neoclassical.extrapolated_weight(
                table,
                density_m3=float(discharge_profiles.density(s_probe[index : index + 1])[0]),
                temperature_ev=float(
                    discharge_profiles.electron_temperature(s_probe[index : index + 1])[0]
                ),
                radial_field_v_m=float(ambipolar[index]),
            )
            coverage.append({"s": float(s_probe[index]), **share})
            break
    print()
    for row in coverage:
        print(
            f"  at s = {row['s']:.2f} the convolution draws "
            f"{100 * row['above_collisionality']:.0f} per cent of its weight from above "
            f"the solved collisionality range and "
            f"{100 * row['below_collisionality']:.0f} per cent from below it"
        )
    worst = max(
        (r["above_collisionality"] + r["below_collisionality"] for r in coverage),
        default=0.0,
    )
    print(
        f"  no input closes the gap, and the largest share of the coefficient taken from "
        f"outside the solved table is {100 * worst:.0f} per cent, which is where the "
        f"remaining difference is"
    )

    write_record(
        BOOTSTRAP_OUT,
        {
            "saturated": {
                label: {
                    "total_current_a": solution.total_current_a,
                    "mismatch": solution.mismatch,
                }
                for label, solution in results.items()
            },
            "attribution": attribution,
            "redl_total_current_a": enclosed_current(equilibrium, s_redl, redl),
            "diffused": diffused,
            "measured_current_a": measured.value,
            "measured_time_s": MEASURED_TIME_S,
            "measured_source": measured.source,
            "table_coverage": coverage,
        },
        geometry=twin.geometry,
    )

    wrong_way = [
        label
        for label, solution in results.items()
        if float(np.asarray(solution.output.wout.iotaf)[-1]) <= reference_iota
    ]
    if wrong_way:
        print(f"direction disagrees with the machine for: {', '.join(wrong_way)}")
        return 1
    return 0


# -- coupled -----------------------------------------------------------------------

# Coupled transport, bootstrap and equilibrium against the sequential solve.
#
#     python -m w7x_twin coupled [power in MW ...]

COUPLED_OUT = Path("results/plasma/coupled_solve.json")
RENORMALISATION = transport.PUBLISHED_ISS04_ENHANCEMENT





def run_coupled() -> int:
    powers = [v * 1e6 for v in args(float)] or [2.0e6, 5.0e6, 10.0e6]
    twin = _common.twin()
    profiles = kinetics.HIGH_PERFORMANCE
    coefficients = load_coefficients()
    ripple = neoclassical.load_ripple()
    print(f"{twin.geometry}")
    print(f"ISS04 x {RENORMALISATION}, radial electric field solved from ambipolarity")

    reference = twin.solve_profiles("standard", profiles)
    vacuum = twin.solve(twin.state("standard"), SCAN)
    minor = float(reference.wout.Aminor_p)
    chi_neo = build_neoclassical(coefficients, ripple, None, minor)
    model = transport.TransportModel(renormalisation=RENORMALISATION)

    rows = []
    for power in powers:
        print(f"\n{power / 1e6:.0f} MW, coupled")
        started = time.monotonic()
        solution = current.solve_coupled(
            twin, "standard", profiles,
            heating=transport.Heating(power_w=power), model=model,
            neoclassical=chi_neo, drive="redl", verbose=True,
        )
        elapsed = time.monotonic() - started

        # The same power solved in sequence, which is what the other tables do: the
        # transport on a pressure-only equilibrium and the bootstrap on the prescribed
        # profiles.
        sequential_transport = transport.solve(
            reference, profiles, heating=transport.Heating(power_w=power),
            model=model, neoclassical=chi_neo,
        )
        sequential_boot = current.solve_self_consistent(
            twin, "standard", profiles, verbose=False, target="redl"
        )

        # The axis shift with the equilibrium current carried, which the pressure-only
        # beta scan cannot produce: the current's own field moves the axis.
        coupled_analysis = diagnostics.analyse(solution.output, vacuum_reference=vacuum)
        boot_analysis = diagnostics.analyse(
            sequential_boot.output, vacuum_reference=vacuum
        )
        print(
            f"  shift with the self-consistent current "
            f"{1e3 * coupled_analysis.axis_shift_m:.2f} mm at "
            f"<beta> {100 * coupled_analysis.beta_total:.3f} %, and "
            f"{1e3 * boot_analysis.axis_shift_m:.2f} mm at "
            f"{100 * boot_analysis.beta_total:.3f} % from the bootstrap solve alone"
        )

        entry = {
            "power_w": power,
            "iterations": len(solution.history),
            "converged": solution.converged,
            "seconds": elapsed,
            "coupled": {
                "stored_energy_j": solution.history[-1].stored_energy_j,
                "current_a": solution.history[-1].total_current_a,
                "temperature_axis_ev": solution.history[-1].electron_temperature_axis_ev,
                "beta": solution.history[-1].beta,
                "iota_edge": solution.history[-1].iota_edge,
                "shift_mm": 1e3 * coupled_analysis.axis_shift_m,
                "beta_analysed": float(coupled_analysis.beta_total),
            },
            "sequential": {
                "stored_energy_j": float(sequential_transport.stored_energy_j),
                "current_a": float(sequential_boot.total_current_a),
                "temperature_axis_ev": float(
                    sequential_transport.electron_temperature_ev[0]
                ),
                "beta": float(sequential_boot.output.wout.betatotal),
                "iota_edge": float(np.asarray(sequential_boot.output.wout.iotaf)[-1]),
                "shift_mm": 1e3 * boot_analysis.axis_shift_m,
                "beta_analysed": float(boot_analysis.beta_total),
            },
            "residual": solution.residual,
            "history": [dataclasses.asdict(step) for step in solution.history],
        }
        rows.append(entry)

    print()
    header = (
        f"{'P [MW]':>7s} {'steps':>6s} {'W coupled':>11s} {'W sequential':>13s} "
        f"{'I coupled':>11s} {'I sequential':>13s} {'iota coupled':>13s} {'sequential':>11s}"
    )
    print(header)
    print("-" * len(header))
    for entry in rows:
        c, s = entry["coupled"], entry["sequential"]
        print(
            f"{entry['power_w'] / 1e6:7.1f} {entry['iterations']:6d} "
            f"{c['stored_energy_j'] / 1e6:10.4f}M {s['stored_energy_j'] / 1e6:12.4f}M "
            f"{c['current_a'] / 1e3:10.3f}k {s['current_a'] / 1e3:12.3f}k "
            f"{c['iota_edge']:13.5f} {s['iota_edge']:11.5f}"
        )

    write_record(
        COUPLED_OUT,
        {"renormalisation": RENORMALISATION, "cases": rows},
        geometry=twin.geometry,
    )
    return 0


# -- computed ----------------------------------------------------------------------

# The balance with each computed input substituted alone and all together.
#
#     python -m w7x_twin computed [heating MW]

COMPUTED_OUT = Path("results/plasma/computed.json")
#: Carbon at the separatrix, the fraction the discharge comparison runs at.
CARBON = 0.02
#: Second main ion species as a share of the fuel, and its mass in amu.
SECOND_ION, SECOND_MASS = 0.5, 2.0


def field_range(equilibrium, analysis, s):
    """|B| minimum and maximum on each surface, for the resonance layer."""
    axis_r = diagnostics._axis_position(equilibrium.wout)[0]
    low = np.empty_like(s)
    high = np.empty_like(s)
    surfaces = int(equilibrium.wout.ns)
    for index, value in enumerate(s):
        cut = min(int(round(value * (surfaces - 1))), surfaces - 1)
        r, _ = diagnostics.flux_surface(equilibrium.wout, cut, 0.0)
        local = analysis.b_axis_t * axis_r / np.maximum(r, 1e-6)
        low[index], high[index] = float(np.min(local)), float(np.max(local))
    return low, high


def run_computed() -> int:
    power = 1e6 * arg(1, float, 5.0)

    twin = _common.twin()
    base = kinetics.HIGH_PERFORMANCE
    equilibrium = twin.solve_profiles("standard", base)
    analysis = diagnostics.analyse(equilibrium)
    minor = analysis.minor_radius_m
    print(f"{twin.geometry}")
    print(f"{power / 1e6:.1f} MW, minor radius {minor:.4f} m, {analysis.b_axis_t:.4f} T on axis")

    coefficients = neoclassical.load_radial_profile(verbose=False)
    ripple = neoclassical.load_ripple()
    chi_neoclassical = neoclassical.diffusivity_model(coefficients, ripple, minor)

    # Item 6: the anomalous channel from the growth-rate grid and the measured constant.
    anomalous = transport.anomalous_channel(
        abs(float(equilibrium.wout.b0)), minor, verbose=True
    )
    print(
        "anomalous channel: computed"
        if anomalous is not None
        else "anomalous channel: not on disk, the scaling stands in"
    )

    # Item 9: the deposition the resonance and the beam path give.
    s = np.linspace(0.0, 1.0, SURFACES)
    low, high = field_range(equilibrium, analysis, s)
    resonant = transport.resonant_field_t()
    reached = (low <= resonant) & (resonant <= high)
    layer = np.where(reached, resonant, np.where(high < resonant, high, low))
    deposition = transport.cyclotron_deposition(
        s, layer, base.density(s), base.electron_temperature(s), power
    )
    print(
        f"electron-cyclotron absorption peaks at s = {deposition.peak_s:.3f}, "
        f"{deposition.absorbed_fraction:.2f} absorbed: {deposition.note}"
    )
    # The traced ray stands in for the layer weighting where it crosses, so the balance
    # solves on the deposition the beam's own path gives.

    traced, ray = ray_traced_deposition(twin, equilibrium, minor, base, s, power)
    if traced is not None:
        deposition = traced
        print(f"the traced ray moves the deposition to s = {traced.peak_s:.3f}: {traced.note}")
    else:
        print(f"the traced ray did not cross ({ray.note}), so the layer weighting stands")

    cases = {
        "prescribed": dict(profiles=base, heating=transport.Heating(power_w=power),
                           anomalous=None),
        "computed deposition": dict(
            profiles=base,
            heating=transport.Heating.from_deposition(power, deposition),
            anomalous=None),
        "carbon from the target": dict(
            profiles=dataclasses.replace(base, carbon_fraction=CARBON,
                                         carbon_from_target=True, minor_radius_m=minor),
            heating=transport.Heating(power_w=power), anomalous=None),
        "second ion species": dict(
            profiles=dataclasses.replace(base, second_ion_fraction=SECOND_ION,
                                         second_ion_mass_amu=SECOND_MASS),
            heating=transport.Heating(power_w=power), anomalous=None),
    }
    if anomalous is not None:
        cases["computed turbulence"] = dict(
            profiles=base, heating=transport.Heating(power_w=power), anomalous=anomalous)
        cases["all four"] = dict(
            profiles=dataclasses.replace(
                base, carbon_fraction=CARBON, carbon_from_target=True,
                minor_radius_m=minor, second_ion_fraction=SECOND_ION,
                second_ion_mass_amu=SECOND_MASS),
            heating=transport.Heating.from_deposition(power, deposition),
            anomalous=anomalous)

    table = _common.Table(
        ("inputs", ">24s"), ("W [MJ]", "8.3f"), ("tau_E [s]", "10.4f"),
        ("over ISS04", "11.3f"), ("T_e(0) [eV]", "12.0f"), ("radiated", "9.4f"),
        ("neo share", "10.4f"),
    )
    print()
    table.begin()

    rows = []
    for label, keywords in cases.items():
        model = transport.TransportModel(
            renormalisation=(1.0 if keywords["anomalous"] is not None
                             else transport.PUBLISHED_ISS04_ENHANCEMENT)
        )
        solution = transport.solve(
            equilibrium, keywords["profiles"], heating=keywords["heating"],
            model=model, neoclassical=chi_neoclassical, anomalous=keywords["anomalous"],
        )
        over = float(solution.confinement_time_s / solution.iss04_time_s)
        rows.append(
            {
                "inputs": label,
                "stored_energy_j": float(solution.stored_energy_j),
                "confinement_time_s": float(solution.confinement_time_s),
                "over_iss04": over,
                "central_electron_temperature_ev": float(solution.electron_temperature_ev[0]),
                "radiated_fraction": float(solution.radiated_fraction),
                "neoclassical_fraction_on_axis": float(np.asarray(solution.neoclassical_fraction).ravel()[0]),
            }
        )
        table.row(
            label, solution.stored_energy_j / 1e6, solution.confinement_time_s, over,
            solution.electron_temperature_ev[0], solution.radiated_fraction,
            float(np.asarray(solution.neoclassical_fraction).ravel()[0]),
        )

    reference = rows[0]
    print()
    for row in rows[1:]:
        print(
            f"{row['inputs']:>24s} moves the stored energy by "
            f"{100 * (row['stored_energy_j'] / reference['stored_energy_j'] - 1):+.1f} % and "
            f"the confinement to {row['over_iss04']:.3f} times ISS04"
        )

    write_record(
        COMPUTED_OUT,
        {
            "heating_w": power,
            "carbon_edge_fraction": CARBON,
            "second_ion_fraction": SECOND_ION,
            "second_ion_mass_amu": SECOND_MASS,
            "deposition_peak_s": deposition.peak_s,
            "deposition_absorbed_fraction": deposition.absorbed_fraction,
            "anomalous_channel_available": anomalous is not None,
            "cases": rows,
        },
        geometry=twin.geometry,
    )
    return 0


# -- transient ---------------------------------------------------------------------

# Density and temperature marched through the waveform with the two-point layer
# supplying the pedestal at every step.
#
#     python -m w7x_twin transient [identifier]

TRANSIENT_OUT = Path("results/plasma/transient_discharge.json")
TIME_STEP_S = 1.0e-3
DURATION_S = 6.0
RECORD_EVERY = 20
#: The heating phase of 20180919.033 is ordinary gas-fuelled operation, so the
#: diffusivity is anchored at the confinement that regime is measured at.
ENHANCEMENT = transport.MEASURED_ISS04_RANGE[0]
#: The inward pinch the published peaking calibrated.
PINCH_M_S = -0.25
#: Bootstrap reference from the self-consistent solve, and the mimic tapers that
#: turn a net current into an edge transform.
REFERENCE_ENERGY_J = 1.099e6
REFERENCE_CURRENT_A = -12.86e3
MIMIC_CURRENT_A = (0.0, 11.0e3, 22.0e3, 32.0e3, 43.0e3)
MIMIC_TRANSFORM = (0.87836, 0.89470, 0.91259, 0.93143, 0.95157)


layer_constants = _common.layer_constants


def run_transient() -> int:
    identifier = arg(1, default="20180919.033")
    programme = programmes.get(identifier)
    ecrh = programme.measured["heating_power_ecrh_w"].value
    beam = (
        programme.measured["heating_power_nbi_w"].value
        if "heating_power_nbi_w" in programme.measured
        else ecrh
    )
    switch = programme.phase_s[0] if programme.phase_s else 1.7
    waveform = current.Waveform.steps(
        ((switch, ecrh), (DURATION_S - switch, beam))
    )

    twin = _common.twin()
    profiles = kinetics.HIGH_PERFORMANCE
    equilibrium = twin.solve_profiles("standard", profiles)
    analysis = diagnostics.analyse(equilibrium)
    minor = analysis.minor_radius_m
    major = analysis.major_radius_m
    print(f"{twin.geometry}")
    print(
        f"{identifier}: {ecrh / 1e6:.1f} MW to {switch:.2f} s, then {beam / 1e6:.1f} MW, "
        f"marched at {1e3 * TIME_STEP_S:.0f} ms steps over {DURATION_S:.0f} s at "
        f"{ENHANCEMENT:.2f} times ISS04"
    )

    s = np.linspace(0.0, 1.0, SURFACES)
    radius = minor * np.sqrt(np.clip(s, 0.0, 1.0))

    # The density the march starts on is the particle closure's own steady state at the
    # discharge's axis density, with the source scaled to hold it there: the closure is
    # linear in the density above its zero-source floor, so two stationary solves fix
    # the scale. Starting on the reference shape instead hands the march a profile the
    # closure does not sustain, and the density walks away from the operating point.
    particle_model = transport.ParticleModel(pinch_m_s=PINCH_M_S)
    source = transport.gaussian_source(s, 0.95, 0.15, THROUGHPUT, minor)
    probe = transport.solve_density(
        source, s, minor, profiles.density_edge_m3, particle_model
    )
    quiet = transport.solve_density(
        np.zeros_like(s), s, minor, profiles.density_edge_m3, particle_model
    )
    scale = (profiles.density_axis_m3 - float(quiet.density_m3[0])) / max(
        float(probe.density_m3[0]) - float(quiet.density_m3[0]), 1e-30
    )
    source = source * scale
    steady = transport.solve_density(
        source, s, minor, profiles.density_edge_m3, particle_model
    )
    density = np.asarray(steady.density_m3, dtype=float)
    print(
        f"the particle closure holds the axis at {density[0]:.2e} m^-3 at a peaking of "
        f"{steady.peaking:.2f}"
    )

    # The diffusivity of each phase, from the stationary balance at that phase's power
    # on the same density the march carries. The march carries the dynamics between the
    # two closures rather than a closure of its own, which is what anchoring to a
    # confinement scaling means in time.
    model = transport.TransportModel(renormalisation=ENHANCEMENT)
    balances = {
        power: transport.solve(
            equilibrium, density, heating=transport.Heating(power_w=power), model=model
        )
        for power in sorted({ecrh, beam})
    }
    chi_of = {
        power: np.interp(
            np.linspace(0.0, 1.0, SURFACES), balance.s, balance.chi_m2_s
        )
        for power, balance in balances.items()
    }

    connection, incidence, area = layer_constants()
    print(
        f"the layer closes over {area:.2f} m2 at {np.degrees(np.arcsin(incidence)):.2f} "
        f"degrees and {connection:.0f} m of connection"
    )
    # The march starts cold, so the approach to the first flat-top is part of the answer.
    temperature = np.full(SURFACES, 150.0)
    net_current = 0.0
    inductive = current.resistive_time_s(
        profiles.electron_temperature_axis_ev * 0.5, minor
    )

    # Deposition per volume in the march's own cylindrical metric, so the energy the
    # march conserves is exactly the power the waveform supplies.
    shape = transport.Heating(power_w=1.0).profile(s)
    volume_weight = 4.0 * np.pi**2 * major * radius
    shape_norm = float(np.trapezoid(shape * volume_weight, radius))

    def stored_energy(density_now: np.ndarray, temperature_now: np.ndarray) -> float:
        energy_density = (
            1.5
            * transport.ELEMENTARY_CHARGE
            * (1.0 + model.ion_fraction)
            * density_now
            * temperature_now
        )
        return float(np.trapezoid(energy_density * volume_weight, radius))

    steps = int(round(DURATION_S / TIME_STEP_S))
    times, energies, pedestals, targets, currents, transforms, axis_t = (
        [], [], [], [], [], [], []
    )
    for step in range(1, steps + 1):
        moment = step * TIME_STEP_S
        power = waveform.at(moment)
        chi = chi_of[min(chi_of, key=lambda p: abs(p - power))]

        marched = transport.evolve_density(
            density, source, s, minor, profiles.density_edge_m3,
            model=particle_model, time_step_s=TIME_STEP_S, num_steps=1,
            record_every=1,
        )
        density = np.asarray(marched["density_m3"], dtype=float)

        # The layer closed at this instant: the crossing power over the wetted area at
        # the traced incidence drives the two-point model at the marching edge density,
        # and its upstream temperature is the pedestal the core stands on.
        parallel = max(power, 1e3) / area / incidence
        closed = edge.solve_two_point(float(density[-1]), parallel, connection)
        pedestal = float(closed.upstream_temperature_ev)

        temperature = transport.temperature_step(
            temperature, density, chi, power * shape / shape_norm, s, minor,
            pedestal, TIME_STEP_S, ion_fraction=model.ion_fraction,
        )
        energy = stored_energy(density, temperature)
        target = REFERENCE_CURRENT_A * energy / REFERENCE_ENERGY_J
        net_current = (net_current + TIME_STEP_S / inductive * target) / (
            1.0 + TIME_STEP_S / inductive
        )

        if step % RECORD_EVERY == 0 or step == steps:
            times.append(moment)
            energies.append(energy)
            pedestals.append(pedestal)
            targets.append(float(closed.target_temperature_ev))
            currents.append(net_current)
            transforms.append(
                float(np.interp(abs(net_current), MIMIC_CURRENT_A, MIMIC_TRANSFORM))
            )
            axis_t.append(float(temperature[0]))

    table = _common.Table(
        ("t [s]", "7.2f"), ("P [MW]", "7.2f"), ("W [MJ]", "8.3f"), ("T_e(0)", "8.0f"),
        ("T_sep", "7.1f"), ("T_t", "7.2f"), ("I [kA]", "8.2f"), ("iota edge", "10.5f"),
    )
    print()
    table.begin()
    for moment in (0.2, 0.8, 1.6, 1.8, 2.5, 4.0, 6.0):
        index = int(np.argmin(np.abs(np.asarray(times) - moment)))
        table.row(
            times[index], waveform.at(times[index]) / 1e6, energies[index] / 1e6,
            axis_t[index], pedestals[index], targets[index], currents[index] / 1e3,
            transforms[index],
        )

    # The march against the stationary solve it must relax to: the flat-top energy at
    # the first power, held to the balance at the same power. The march lives in the
    # cylindrical metric and the balance on the flux geometry, so the gap carries that
    # difference beside the pedestal the layer set.
    settle_index = int(np.argmin(np.abs(np.asarray(times) - (switch - 0.1))))
    flat_top = energies[settle_index]
    stationary = float(balances[ecrh].stored_energy_j)
    print()
    print(
        f"the flat-top holds {flat_top / 1e6:.3f} MJ against the stationary balance's "
        f"{stationary / 1e6:.3f} at the same power and closure, "
        f"{100 * (flat_top / stationary - 1):+.1f} per cent, with the pedestal at "
        f"{pedestals[settle_index]:.0f} eV where the stationary solve holds 100"
    )
    after = np.asarray(times) >= switch
    settled = np.asarray(energies)[after]
    final = settled[-1]
    reached = np.abs(settled - final) <= 0.05 * abs(final)
    energy_settle = float(np.asarray(times)[after][int(np.argmax(reached))] - switch)
    current_final = currents[-1]
    print(
        f"after the step the energy settles within five per cent in "
        f"{energy_settle:.2f} s, and the current reaches {current_final / 1e3:.2f} kA "
        f"of the {REFERENCE_CURRENT_A * final / REFERENCE_ENERGY_J / 1e3:.2f} the "
        f"energy implies, the inductive time being {inductive:.1f} s"
    )

    write_record(
        TRANSIENT_OUT,
        {
            "identifier": identifier,
            "enhancement": ENHANCEMENT,
            "time_step_s": TIME_STEP_S,
            "layer": {
                "connection_length_m": connection,
                "incidence_sine": incidence,
                "wetted_area_m2": area,
            },
            "flat_top_energy_j": flat_top,
            "stationary_energy_j": stationary,
            "energy_settling_s": energy_settle,
            "inductive_time_s": inductive,
            "times_s": times,
            "stored_energy_j": energies,
            "pedestal_ev": pedestals,
            "target_ev": targets,
            "bootstrap_current_a": currents,
            "edge_transform": transforms,
            "axis_temperature_ev": axis_t,
        },
        geometry=twin.geometry,
    )
    return 0
