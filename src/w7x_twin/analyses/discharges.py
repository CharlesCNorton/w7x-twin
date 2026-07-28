"""The model against identified W7-X discharges and the machine's error field."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.analyses._common import arg, args, write_record
from w7x_twin.hardware import coils as coil_geometry, machine, walls
from w7x_twin.magnetics import field, fieldlines
from w7x_twin.magnetics.field import VacuumField
from w7x_twin.mhd import diagnostics
from w7x_twin.mhd.equilibrium import SCAN, Scenario, Twin
from w7x_twin.plasma import current, edge, kinetics, neoclassical, transport
from w7x_twin.plasma.kinetics import log_gradient
from w7x_twin.records import programmes

#: Confinement against the ISS04 scaling. The overview reports 1.4 for the discharge
#: that carries the highest triple product, so that is what the model runs at where a
#: discharge states no enhancement of its own.
RENORMALISATION = transport.PUBLISHED_ISS04_ENHANCEMENT
#: The fan across the scrape-off layer every load attribution launches, and how far it
#: is followed.
LAYER = (0.985, 1.40)
FAN_LINES = 120
TURNS = 200


# -- errorfield --------------------------------------------------------------------

# The machine's measured n = 1 error field, applied to the model.
#
# The negative of the measured trim correction synthesises the intrinsic error field and its divertor imbalance.
#
#     python -m w7x_twin errorfield [configuration]

ERRORFIELD_OUT = Path("results/magnetics/error_field.json")
HEATING_W = 5.0e6

#: Multiples of the measured correction the scan runs at. Zero is the ideal machine,
#: minus one is the machine's own error field, plus one is that error doubled the other
#: way, which is what a correction applied with the wrong sign would give.
SCALES = (-2.0, -1.0, -0.5, 0.0, 0.5, 1.0)
#: Below this the periodic module spread is the tracer's round-off and not a field, so the
#: ratio a measured asymmetry stands above it is not formed.
PERIODIC_FLOOR_TOLERANCE = 1.0e-9


def unit_weight_tally(
    vacuum, z_axis: float, starts: np.ndarray, separatrix: float, chi_edge: float,
    vessel, elements,
) -> tuple[dict[tuple[int, str], float], float, int, list[float]]:
    """Layer-weighted strike weight per (module, unit) over module-equivalent fans,
    with the total weight, the struck-line count and the connection lengths."""
    period = 2.0 * np.pi / walls.NUM_FIELD_PERIODS_DEFAULT
    weight_by_unit: dict[tuple[int, str], float] = {}
    total = 0.0
    struck_lines = 0
    connections: list[float] = []
    for module_index in range(walls.NUM_FIELD_PERIODS_DEFAULT):
        for offset in (+1.0, -1.0):
            section, _ = fieldlines.trace(
                vacuum, starts, np.full(starts.shape, offset * z_axis), turns=TURNS,
                plane_phi=module_index * period, vessel=vessel, components=elements,
            )
            strikes = section.strikes
            struck = strikes.struck
            if not struck.any():
                continue
            struck_lines += int(struck.sum())
            connections.extend(strikes.connection_length_m[struck].tolist())
            decay = edge.power_decay_length(
                chi_edge,
                float(np.median(strikes.connection_length_m[struck])),
                20.0,
            )
            weights = np.where(
                struck,
                edge.layer_weights(strikes.start_r, separatrix, decay),
                0.0,
            )
            module, is_upper = walls.unit_of(strikes.phi, strikes.z)
            for line in np.flatnonzero(struck):
                key = (int(module[line]), "upper" if is_upper[line] else "lower")
                weight_by_unit[key] = weight_by_unit.get(key, 0.0) + float(weights[line])
                total += float(weights[line])
    return weight_by_unit, total, struck_lines, connections


def module_loads(
    torus, currents, equilibrium, chi_edge: float, crossing: float,
    vessel, elements,
) -> dict:
    """Power per divertor module from module-equivalent fans; ``equilibrium`` must be the vacuum solve."""
    vacuum = VacuumField(torus, currents)
    r_axis, z_axis = fieldlines.find_axis(vacuum)
    r_lcfs, _ = diagnostics.boundary_cut(equilibrium.wout, 0.0)
    separatrix = float(r_lcfs.max())
    starts = r_axis + np.linspace(*LAYER, FAN_LINES) * (separatrix - r_axis)

    weight_by_unit, total, struck_lines, connections = unit_weight_tally(
        vacuum, z_axis, starts, separatrix, chi_edge, vessel, elements
    )
    if total <= 0.0:
        return {"units": {}, "lines": struck_lines}
    units = {
        f"module {key[0]} {key[1]}": crossing * weight / total
        for key, weight in sorted(weight_by_unit.items())
    }
    # An n = 1 field breaks the machine from module to module. Upper against lower is a
    # different asymmetry, set by where the fan is launched relative to the midplane, and
    # it is present with no trim current at all. Both are reported, and the module one is
    # what the published figure is compared against.
    by_module: dict[int, float] = {}
    for key, weight in weight_by_unit.items():
        by_module[key[0]] = by_module.get(key[0], 0.0) + weight
    modules = np.array([by_module[k] for k in sorted(by_module)])
    values = np.array(list(units.values()))

    def ratio(array: np.ndarray) -> float:
        return float(array.max() / array.min()) if array.min() > 0 else float("inf")

    return {
        "units": units,
        "modules": {str(k): crossing * v / total for k, v in sorted(by_module.items())},
        "lines": struck_lines,
        "loaded_units": int(len(values)),
        "loaded_modules": int(len(modules)),
        "module_max_over_min": ratio(modules),
        "module_relative_spread": float(modules.std() / modules.mean()),
        "unit_max_over_min": ratio(values),
        "unit_relative_spread": float(values.std() / values.mean()),
        "median_connection_length_m": float(np.median(connections)),
    }


#: Radius of the midplane circle the error field's harmonics are reported on, in metres.
#: Outside the plasma and inside the vessel, so the harmonics are of a vacuum field.


def _midplane_harmonics(torus, twin, waveform: dict[str, float], state) -> dict[int, float]:
    """Toroidal harmonics of the error field's radial component on a circle, machine field differenced away."""
    applied = {k: -v for k, v in waveform.items()}
    perturbed = field.radial_harmonics(
        field.VacuumField(torus, twin.with_currents(state, **applied).currents),
        field.HARMONIC_POINTS, field.HARMONIC_RADIUS_M,
    )
    reference = field.radial_harmonics(
        field.VacuumField(torus, state.currents),
        field.HARMONIC_POINTS, field.HARMONIC_RADIUS_M,
    )
    return {
        n: float(abs(perturbed[n] - reference[n])) for n in range(1, len(perturbed))
    }


def run_errorfield() -> int:
    configuration = arg(1, default="standard")
    setting = programmes.trim_setting(configuration)
    twin = _common.twin(coils_file="coils.w7x_full")
    # An n = 1 waveform is not periodic in the field period, so the trace reads a
    # whole-torus table while the equilibrium keeps the per-period one.
    torus = twin.full_torus_response()
    vessel = _common.vessel()
    elements = _common.components()
    print(f"{twin.geometry}")
    print(
        f"{configuration}: the measured correction is {setting.amplitude_a:.0f} A at "
        f"{setting.phase_degrees:.0f} degrees, by {setting.method}"
    )
    print(f"  {setting.source}")

    waveform = machine.trim_waveform(setting.amplitude_a, setting.phase_degrees)
    print("  " + ", ".join(f"{k} {v:+7.1f} A" for k, v in waveform.items()))

    profiles = kinetics.HIGH_PERFORMANCE
    finite_beta = twin.solve_profiles(configuration, profiles)
    # The launch anchor is the vacuum boundary, since the traced field is the vacuum one.
    equilibrium = twin.solve(twin.state(configuration), SCAN)
    balance = transport.solve(
        finite_beta, profiles, heating=transport.Heating(power_w=HEATING_W),
        model=transport.TransportModel(renormalisation=RENORMALISATION),
    )
    crossing = HEATING_W - balance.radiated_power_w
    chi_edge = float(balance.chi_m2_s[-1])
    print(
        f"  {crossing / 1e6:.2f} MW crosses the separatrix, edge diffusivity "
        f"{chi_edge:.3f} m2/s"
    )

    print()
    header = (
        f"{'scale':>7s} {'A0 [A]':>8s} {'lines':>6s} {'module max/min':>15s} "
        f"{'module spread':>14s} {'unit max/min':>13s} {'L_c [m]':>9s}"
    )
    print(header)
    print("-" * len(header))
    rows = []
    for scale in SCALES:
        state = twin.state(configuration)
        if scale:
            applied = {k: scale * v for k, v in waveform.items()}
            state = twin.with_currents(state, **applied)
        loads = module_loads(
            torus, state.currents, equilibrium, chi_edge, crossing, vessel, elements
        )
        rows.append({"scale": scale, "amplitude_a": scale * setting.amplitude_a, **loads})
        print(
            f"{scale:7.2f} {scale * setting.amplitude_a:8.1f} {loads['lines']:6d} "
            f"{loads.get('module_max_over_min', float('nan')):15.3f} "
            f"{loads.get('module_relative_spread', float('nan')):14.4f} "
            f"{loads.get('unit_max_over_min', float('nan')):13.3f} "
            f"{loads.get('median_connection_length_m', float('nan')):9.1f}"
        )

    ideal = next(r for r in rows if r["scale"] == 0.0)
    synthesised = next(r for r in rows if r["scale"] == -1.0)
    # The ideal coil set is five-fold periodic, so its five modules have to take the same
    # load. Whatever spread it shows is the tracing and the finite fan, and it is the
    # floor any measured asymmetry has to stand above to be a result.
    floor = ideal["module_relative_spread"]
    print()
    # The floor is a relative standard deviation of five loads that a periodic field makes
    # identical, so it comes back at the tracer's round-off rather than at exactly zero. An
    # equality test against zero misses that and divides by 1e-15.
    if floor < PERIODIC_FLOOR_TOLERANCE:
        # The tracer reproduces the five-fold periodicity exactly, so every module carries
        # the same load with no trim current and the whole spread below is the field.
        print(
            f"the ideal set spreads the modules by {floor:.1e} on {ideal['lines']} traced "
            "lines, which is the tracer's round-off, so the spread below is the field "
            "rather than the sampling"
        )
        signal = float("inf")
    else:
        signal = synthesised["module_relative_spread"] / floor
        print(
            f"the ideal set is five-fold periodic, so its module spread of {floor:.4f} is "
            f"the floor set by {ideal['lines']} traced lines rather than by the field, "
            f"and the synthesised error field stands {signal:.1f} times above it"
        )
    measured_uncorrected = programmes.MACHINE_MEASUREMENTS[
        "divertor_temperature_asymmetry_uncorrected"
    ]
    measured_corrected = programmes.MACHINE_MEASUREMENTS[
        "divertor_temperature_asymmetry_corrected"
    ]
    print()
    print(
        f"the ideal coil set spreads the load across modules to "
        f"{ideal['module_max_over_min']:.3f}, against the "
        f"{measured_corrected.value:.1f} the machine reaches once corrected"
    )
    print(
        f"the machine's own error field, put in at the measured amplitude and phase, "
        f"gives {synthesised['module_max_over_min']:.3f}, against the measured "
        f"{measured_uncorrected.value:.1f} before correction"
    )
    low, high = measured_uncorrected.band()
    inside = bool(low <= synthesised["module_max_over_min"] <= high)
    print(
        "which is inside the published accuracy"
        if inside
        else f"which is outside the published band {low:.2f} to {high:.2f}"
    )
    if not inside:
        # Two things stand between the two numbers, and both are stated rather than
        # folded into the residual. The trim geometry is reconstructed, so the field this
        # waveform produces carries that accuracy. And the source states that the divertor
        # misalignment contribution to the measured imbalance could not be separated from
        # the field's, so the published figure is not the field alone.
        print(
            "the trim filaments are an engineering reconstruction, and the source states "
            "the divertor misalignment could not be separated from the field, so the "
            "measured figure is not the field's contribution alone"
        )

    # The same asymmetry as a relative standard deviation, which is the measure a later
    # campaign minimised the load spread in. Both numbers are that measure, but this one is
    # taken over five modules from a sixty-line fan and the published one over the
    # divertor's resolved elements, so the two bound an order rather than agreeing to a
    # digit. That campaign also corrected the 2/2 harmonic with the in-vessel control coils,
    # which is not driven here.
    spread_uncorrected = programmes.MACHINE_MEASUREMENTS[
        "divertor_load_spread_uncorrected"
    ]
    spread_corrected = programmes.MACHINE_MEASUREMENTS["divertor_load_spread_corrected"]
    print()
    print(
        f"as a relative standard deviation the synthesised field spreads the modules by "
        f"{synthesised.get('module_relative_spread', float('nan')):.4f} and the ten units "
        f"by {synthesised.get('unit_relative_spread', float('nan')):.4f}"
    )
    print(
        f"  the machine measures {spread_uncorrected.value:.2f} with neither harmonic "
        f"corrected and {spread_corrected.value:.3f} with both, over the divertor's own "
        f"elements rather than five modules"
    )
    print(f"  {spread_uncorrected.source}")
    for correction in programmes.symmetrisation_settings(field_sense="forward"):
        print(
            f"  {correction.mode} on the {correction.circuit} coils at "
            f"{correction.coil_current_a:+.0f} A took it {correction.spread_before:.3f} to "
            f"{correction.spread_after:.3f} in {correction.programme}"
        )

    # The error field itself, rather than the setting that cancels it. Driving the trim
    # circuits with the negative of a measured correction puts the machine's own field into
    # the model, and its harmonics on a midplane circle are what any other representation
    # of that field would have to reproduce. They are the quantity to carry, since they do
    # not depend on the trim geometry being right about anything but its own shape.
    spectrum = _midplane_harmonics(torus, twin, waveform, twin.state(configuration))
    print()
    print("the error field on the R = 6.2 m midplane circle, as radial field harmonics")
    table = _common.Table(("n", "4d"), ("B_r [mT]", "11.4f"), ("over B_0", "11.2e"))
    table.begin()
    for n, value in sorted(spectrum.items()):
        if n > 6:
            continue
        table.row(n, 1e3 * value, value / 2.5)

    write_record(
        ERRORFIELD_OUT,
        {
            "configuration": configuration,
            "error_field_harmonics_t": {str(k): v for k, v in spectrum.items()},
            "trim_setting": {
                "amplitude_a": setting.amplitude_a,
                "phase_degrees": setting.phase_degrees,
                "planar_current_a": setting.planar_current_a,
                "method": setting.method,
                "source": setting.source,
            },
            "trim_geometry": (
                "engineering reconstruction from published dimensions, not measured "
                "filaments; the synthesised amplitude carries that accuracy"
            ),
            "waveform_a": waveform,
            "heating_power_w": HEATING_W,
            "power_crossing_separatrix_w": float(crossing),
            "scan": rows,
            "periodic_module_spread_floor": floor,
            "signal_over_floor": float(signal),
            "measured": {
                "uncorrected_asymmetry": measured_uncorrected.value,
                "corrected_asymmetry": measured_corrected.value,
                "source": measured_uncorrected.source,
                "synthesised_within_published_accuracy": inside,
            },
        },
        geometry=twin.geometry,
    )
    return 0


# -- symmetrise --------------------------------------------------------------------

# The measured 1/1 and 2/2 corrections driven together, and the load spread they leave.
#
# Negated 1/1 and 2/2 corrections synthesise the fields they cancel; measured spread 0.75 -> 0.27 -> 0.067.
#
#     python -m w7x_twin symmetrise [configuration]

SYMMETRISE_OUT = Path("results/discharges/symmetrise.json")
COILS = "coils.w7x_full"


def module_spread(torus, currents, vessel, elements, chi_edge, separatrix, crossing) -> dict:
    """Relative standard deviation of the load over the five modules and the ten units."""
    vacuum = field.VacuumField(torus, currents)
    r_axis, z_axis = fieldlines.find_axis(vacuum)
    starts = r_axis + np.linspace(*LAYER, FAN_LINES) * (separatrix - r_axis)

    weight_by_unit, total, struck_lines, _ = unit_weight_tally(
        vacuum, z_axis, starts, separatrix, chi_edge, vessel, elements
    )
    if total <= 0.0:
        return {"lines": struck_lines, "module_spread": float("nan"),
                "unit_spread": float("nan")}
    by_module: dict[int, float] = {}
    for key, weight in weight_by_unit.items():
        by_module[key[0]] = by_module.get(key[0], 0.0) + weight
    modules = np.array([by_module[k] for k in sorted(by_module)])
    units = np.array(list(weight_by_unit.values()))
    return {
        "lines": struck_lines,
        "module_spread": float(modules.std() / modules.mean()),
        "unit_spread": float(units.std() / units.mean()),
        "module_max_over_min": float(modules.max() / modules.min()),
        "modules": {str(k): crossing * v / total for k, v in sorted(by_module.items())},
    }


def run_symmetrise() -> int:
    configuration = arg(1, default="standard")
    forward = programmes.symmetrisation_settings(field_sense="forward")
    one_one = next(s for s in forward if s.mode == "1/1")
    two_two = next(s for s in forward if s.mode == "2/2")
    print(
        f"{one_one.mode} on the {one_one.circuit} coils at {one_one.coil_current_a:+.0f} A, "
        f"{two_two.mode} on the {two_two.circuit} coils at {two_two.coil_current_a:+.0f} A"
    )
    print(f"  {one_one.source}")

    twin = _common.twin(coils_file=COILS)
    torus = twin.full_torus_response()
    print(f"{twin.geometry}")
    vessel = _common.vessel()
    elements = _common.components()
    state = twin.state(configuration)

    equilibrium = twin.solve(twin.state(configuration), SCAN)
    balance = transport.solve(
        equilibrium, kinetics.HIGH_PERFORMANCE,
        heating=transport.Heating(power_w=5.0e6),
        model=transport.TransportModel(
            renormalisation=transport.PUBLISHED_ISS04_ENHANCEMENT
        ),
    )
    chi_edge = float(balance.chi_m2_s[-1])
    crossing = 5.0e6 - float(balance.radiated_power_w)
    r_lcfs, _ = diagnostics.boundary_cut(equilibrium.wout, 0.0)
    separatrix = float(r_lcfs.max())

    # The negative of each measured correction, which is the field it cancels. The trim phase
    # is measured from the module 1 trim coil; the harmonic phases the symmetrisation source
    # quotes are of b11 and b22 themselves, so the waveform phase is taken from the trim
    # setting that shares the convention.
    trim_setting = programmes.trim_setting(configuration)
    trim = {
        key: -value
        for key, value in machine.trim_waveform(
            abs(one_one.coil_current_a), trim_setting.phase_degrees
        ).items()
    }
    control = {
        key: -value
        for key, value in machine.control_waveform(
            abs(two_two.coil_current_a), two_two.mode_phase_degrees, mode=2
        ).items()
    }

    cases = {
        "ideal coils": {},
        "1/1 error field": trim,
        "2/2 error field": control,
        "both": {**trim, **control},
    }
    header = (
        f"{'field present':>18s} {'lines':>6s} {'module spread':>14s} {'unit spread':>12s} "
        f"{'module max/min':>15s}"
    )
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for label, waveform in cases.items():
        currents = (
            twin.with_currents(state, **waveform).currents if waveform else state.currents
        )
        answer = module_spread(
            torus, currents, vessel, elements, chi_edge, separatrix, crossing
        )
        rows.append({"field": label, **answer})
        print(
            f"{label:>18s} {answer['lines']:6d} {answer['module_spread']:14.4f} "
            f"{answer['unit_spread']:12.4f} "
            f"{answer.get('module_max_over_min', float('nan')):15.3f}",
            flush=True,
        )

    published = {
        "neither corrected": programmes.MACHINE_MEASUREMENTS[
            "divertor_load_spread_uncorrected"].value,
        "1/1 corrected": one_one.spread_after,
        "both corrected": two_two.spread_after,
    }
    print()
    print(
        "the machine measures "
        + ", ".join(f"{value:.3f} {name}" for name, value in published.items())
        + ", over the divertor's own elements rather than five modules"
    )
    ideal = next(r for r in rows if r["field"] == "ideal coils")
    both = next(r for r in rows if r["field"] == "both")
    print(
        f"an ideal set spreads the modules by {ideal['module_spread']:.2e}, which is the "
        f"tracer's round-off, and the two harmonics together put "
        f"{both['module_spread']:.4f} on it"
    )

    write_record(
        SYMMETRISE_OUT,
        {
            "configuration": configuration,
            "settings": [
                {"mode": s.mode, "circuit": s.circuit,
                 "coil_current_a": s.coil_current_a,
                 "mode_phase_degrees": s.mode_phase_degrees,
                 "spread_before": s.spread_before, "spread_after": s.spread_after,
                 "programme": s.programme}
                for s in forward
            ],
            "trim_waveform_a": trim,
            "control_waveform_a": control,
            "published": published,
            "cases": rows,
        },
        geometry=twin.geometry,
    )
    return 0


# -- trim-radius -------------------------------------------------------------------

# The trim coil mounting radius, pinned against the correction the machine measured.
#
# The trim mounting radius is pinned where the measured current produces the published harmonic amplitude.
#
#     python -m w7x_twin trim-radius [configuration]

TRIM_RADIUS_OUT = Path("results/discharges/trim_radius.json")
#: Mounting radii scanned, in metres. The reconstruction stands at eight.
RADII = (7.0, 7.5, 8.0, 8.5, 9.0)
#: The mode the correction cancels, as (poloidal, toroidal) mode numbers of the boundary.
MODE = (1, 1)
#: Coil filaments per turn of the scan.
NUM_POINTS = 96


def trim_field(twin: Twin, radius: float, waveform: dict[str, float]):
    """Field of the trim set alone at one mounting radius, the type B offset held published."""
    offset = coil_geometry.TYPE_B_RADIUS_M - coil_geometry.OUTER_VESSEL_RADIUS_M
    groups = coil_geometry.trim_coils(
        num_points=NUM_POINTS, radius=radius, type_b_radius=radius + offset
    )
    positions, currents = [], []
    for group in groups:
        amplitude = waveform.get(group.key, 0.0)
        for filament in group.filaments:
            positions.append(np.asarray(filament, dtype=float))
            currents.append(amplitude * group.turns)
    return positions, currents


def biot_savart(positions, currents, points: np.ndarray) -> np.ndarray:
    """Field of a set of closed filaments at Cartesian points, in tesla."""
    mu0_over_4pi = 1.0e-7
    out = np.zeros_like(points)
    for filament, current in zip(positions, currents, strict=True):
        start = filament[:-1]
        segment = filament[1:] - filament[:-1]
        delta = points[:, None, :] - start[None, :, :]
        # Field of a straight segment, in the closed form that avoids sampling it.
        end = points[:, None, :] - filament[1:][None, :, :]
        norm_a = np.linalg.norm(delta, axis=-1)
        norm_b = np.linalg.norm(end, axis=-1)
        cross = np.cross(delta, segment[None, :, :])
        denominator = (
            norm_a * norm_b * (norm_a * norm_b + np.einsum("psk,psk->ps", delta, end))
        )
        weight = np.where(
            denominator > 0.0, (norm_a + norm_b) / np.maximum(denominator, 1e-30), 0.0
        )
        out += current * mu0_over_4pi * np.einsum("psk,ps->pk", cross, weight)
    return out


class FilamentField:
    """A vacuum-field-like callable backed by filaments rather than a tabulated grid."""

    def __init__(self, positions, currents) -> None:
        self.positions = positions
        self.currents = currents

    def __call__(self, r, phi, z):
        r = np.asarray(r, dtype=float)
        phi = np.asarray(phi, dtype=float)
        z = np.asarray(z, dtype=float)
        points = np.stack([r * np.cos(phi), r * np.sin(phi), z], axis=-1)
        cartesian = biot_savart(self.positions, self.currents, points)
        b_x, b_y, b_z = cartesian[..., 0], cartesian[..., 1], cartesian[..., 2]
        return (
            b_x * np.cos(phi) + b_y * np.sin(phi),
            -b_x * np.sin(phi) + b_y * np.cos(phi),
            b_z,
        )


def run_trim_radius() -> int:
    configuration = arg(1, default="standard")
    setting = programmes.trim_setting(configuration)
    published = programmes.MACHINE_MEASUREMENTS["intrinsic_error_field_b11"]

    twin = _common.twin()
    equilibrium = twin.solve(twin.state(configuration), SCAN)
    print(f"{twin.geometry}")
    print(
        f"{configuration}: the measured correction is {setting.amplitude_a:.0f} A at "
        f"{setting.phase_degrees:.0f} degrees, and the field it cancels is published at "
        f"{published.value:.2e}"
    )
    print(f"  {published.source}")

    waveform = machine.trim_waveform(setting.amplitude_a, setting.phase_degrees)

    header = (
        f"{'radius [m]':>11s} {'b11':>12s} {'b11 per amp':>13s} "
        f"{'against published':>18s} {'current for it [A]':>19s}"
    )
    print()
    print(header)
    print("-" * len(header))
    rows = []
    for radius in RADII:
        positions, currents = trim_field(twin, radius, waveform)
        spectrum = field.normal_field_spectrum(
            FilamentField(positions, currents), equilibrium,
            reference_t=abs(float(equilibrium.wout.b0)),
        )
        amplitude = abs(spectrum[MODE])
        per_amp = amplitude / abs(setting.amplitude_a)
        needed = published.value / per_amp if per_amp > 0.0 else float("nan")
        rows.append(
            {
                "radius_m": radius,
                "b11": amplitude,
                "b11_per_ampere": per_amp,
                "ratio_to_published": amplitude / published.value,
                "current_for_published_a": needed,
            }
        )
        print(
            f"{radius:11.2f} {amplitude:12.3e} {per_amp:13.3e} "
            f"{amplitude / published.value:17.3f}x {needed:19.1f}"
        )

    # The radius at which the measured current produces the published amplitude. The
    # coupling falls monotonically with radius, so the crossing is a single interpolation.
    ratios = np.array([r["ratio_to_published"] for r in rows])
    radii = np.array([r["radius_m"] for r in rows])
    order = np.argsort(ratios)
    pinned = (
        float(np.interp(1.0, ratios[order], radii[order]))
        if ratios.min() <= 1.0 <= ratios.max()
        else float("nan")
    )
    print()
    if np.isfinite(pinned):
        print(
            f"the measured {abs(setting.amplitude_a):.0f} A produces the published "
            f"{published.value:.2e} at a mounting radius of {pinned:.3f} m, against the "
            f"{coil_geometry.OUTER_VESSEL_RADIUS_M:.1f} m the reconstruction stands at"
        )
        # The published amplitude carries its own uncertainty, so the radius does too, and
        # that band is what replaces the free parameter.
        low_target, high_target = published.band()
        band = []
        for target in (low_target, high_target):
            scaled = np.array([r["b11"] / target for r in rows])
            order = np.argsort(scaled)
            band.append(float(np.interp(1.0, scaled[order], radii[order])))
        band = sorted(band)
        print(
            f"the published amplitude's own {100 * published.relative_uncertainty:.0f} per "
            f"cent puts that radius between {band[0]:.3f} and {band[1]:.3f} m, a span of "
            f"{100 * (band[1] - band[0]) / pinned:.1f} per cent rather than a free parameter"
        )
    else:
        band = [float("nan"), float("nan")]
        print(
            f"the scan spans {ratios.min():.2f} to {ratios.max():.2f} times the published "
            f"amplitude and does not cross it, so the radius is outside "
            f"{radii.min():.1f} to {radii.max():.1f} m"
        )

    write_record(
        TRIM_RADIUS_OUT,
        {
            "configuration": configuration,
            "mode": list(MODE),
            "measured_current_a": setting.amplitude_a,
            "measured_phase_degrees": setting.phase_degrees,
            "published_b11": published.value,
            "published_relative_uncertainty": published.relative_uncertainty,
            "published_source": published.source,
            "reconstruction_radius_m": coil_geometry.OUTER_VESSEL_RADIUS_M,
            "pinned_radius_m": pinned,
            "pinned_radius_band_m": band,
            "scan": rows,
        },
        geometry=twin.geometry,
    )
    return 0


# -- intrinsic ---------------------------------------------------------------------

# Each deviation is scaled to the measured 1/1 amplitude and held against the 2/2 it predicts.
#
#     python -m w7x_twin intrinsic [configuration]

INTRINSIC_OUT = Path("results/discharges/intrinsic_error_field.json")
#: Deviation amplitudes scanned, in metres for a displacement and radians for a tilt.
DISPLACEMENTS = (0.002, 0.005, 0.010, 0.020)
TILTS = (0.0005, 0.001, 0.002, 0.005)
#: The module whose coils are displaced. One module out of place is the simplest deviation
#: that breaks the five-fold symmetry and so the one with the fewest free choices.
MODULE = 0
#: The two harmonics the symmetrisation measurement publishes.
MODES = ((1, 1), (2, 2))


def module_filaments(coils, currents, module: int) -> list[tuple[np.ndarray, float]]:
    """Filaments of one main-coil module, each at circuit current times its winding number."""
    out = []
    period = 2.0 * np.pi / 5
    for circuit, group in enumerate(coils.filaments):
        current = float(currents[circuit]) * float(coils.file_currents[circuit])
        for filament in group:
            filament = np.asarray(filament, dtype=float)
            angle = np.mod(
                np.arctan2(filament[:, 1].mean(), filament[:, 0].mean()), 2.0 * np.pi
            )
            if int(angle // period) == module:
                out.append((filament, current))
    return out


def deviate(filament: np.ndarray, mode: str, amplitude: float) -> np.ndarray:
    """One coil displaced or tilted by a given amplitude."""
    centre = filament.mean(axis=0)
    phi = np.arctan2(centre[1], centre[0])
    radial = np.array([np.cos(phi), np.sin(phi), 0.0])
    toroidal = np.array([-np.sin(phi), np.cos(phi), 0.0])
    vertical = np.array([0.0, 0.0, 1.0])
    if mode == "radial shift":
        return filament + amplitude * radial
    if mode == "vertical shift":
        return filament + amplitude * vertical
    if mode == "toroidal shift":
        return filament + amplitude * toroidal
    if mode == "tilt about the radial axis":
        axis = radial
    elif mode == "tilt about the vertical axis":
        axis = vertical
    else:
        raise ValueError(f"unknown deviation {mode!r}")
    # Rodrigues rotation of the coil about its own centre.
    offset = filament - centre
    cos, sin = np.cos(amplitude), np.sin(amplitude)
    rotated = (
        offset * cos
        + np.cross(axis, offset) * sin
        + axis[None, :] * (offset @ axis)[:, None] * (1.0 - cos)
    )
    return rotated + centre


def spectrum_of(equilibrium, members, mode: str, amplitude: float):
    """The harmonics a deviation of those filaments puts on the boundary."""
    positions, values = [], []
    for filament, current in members:
        # The deviation's field is the moved coil less the coil where it should be, so the
        # machine field cancels and what is left is the perturbation alone.
        positions.append(deviate(filament, mode, amplitude))
        values.append(current)
        positions.append(filament)
        values.append(-current)
    return field.normal_field_spectrum(
        FilamentField(positions, values), equilibrium,
        reference_t=abs(float(equilibrium.wout.b0)),
    )


def run_intrinsic() -> int:
    configuration = arg(1, default="standard")
    twin = _common.twin()
    equilibrium = twin.solve(twin.state(configuration), SCAN)
    state = twin.state(configuration)
    members = module_filaments(twin.coils, np.asarray(state.currents), MODULE)

    forward = programmes.symmetrisation_settings(field_sense="forward")
    one_one = next(s for s in forward if s.mode == "1/1")
    two_two = next(s for s in forward if s.mode == "2/2")
    published = programmes.MACHINE_MEASUREMENTS["intrinsic_error_field_b11"]
    print(f"{twin.geometry}")
    print(
        f"{len(members)} filaments in module {MODULE + 1}, and the machine publishes "
        f"{published.value:.2e} for both b11 and b22, at "
        f"{one_one.mode_phase_degrees:.0f} and {two_two.mode_phase_degrees:.0f} degrees"
    )

    header = (
        f"{'deviation':30s} {'amplitude':>12s} {'b11':>11s} {'b22':>11s} "
        f"{'b22 / b11':>10s} {'phase 1/1':>10s} {'phase 2/2':>10s}"
    )
    print()
    print(header)
    print("-" * len(header))

    rows = []
    for mode, amplitudes in (
        ("radial shift", DISPLACEMENTS),
        ("vertical shift", DISPLACEMENTS),
        ("toroidal shift", DISPLACEMENTS),
        ("tilt about the radial axis", TILTS),
        ("tilt about the vertical axis", TILTS),
    ):
        for amplitude in amplitudes:
            spectrum = spectrum_of(equilibrium, members, mode, amplitude)
            b11, b22 = spectrum[MODES[0]], spectrum[MODES[1]]
            rows.append(
                {
                    "deviation": mode,
                    "amplitude": amplitude,
                    "b11": abs(b11),
                    "b22": abs(b22),
                    "ratio": abs(b22) / max(abs(b11), 1e-30),
                    "phase_11_degrees": float(np.degrees(np.angle(b11))),
                    "phase_22_degrees": float(np.degrees(np.angle(b22))),
                }
            )
            print(
                f"{mode:30s} {amplitude:12.4f} {abs(b11):11.3e} {abs(b22):11.3e} "
                f"{abs(b22) / max(abs(b11), 1e-30):10.3f} "
                f"{np.degrees(np.angle(b11)):9.1f}° {np.degrees(np.angle(b22)):9.1f}°"
            )

    # Each deviation scaled to reproduce the measured 1/1, then held against the 2/2 it
    # then predicts. The field of a small displacement is linear in it, so the scaling is a
    # ratio and not another scan.
    print()
    header = (
        f"{'deviation':30s} {'amplitude for b11':>18s} {'b22 predicted':>14s} "
        f"{'against published':>18s}"
    )
    print(header)
    print("-" * len(header))
    summary = []
    for mode in sorted({r["deviation"] for r in rows}, key=lambda m: m):
        here = [r for r in rows if r["deviation"] == mode]
        # Linear fit through the origin of b11 against amplitude, so one coefficient.
        amplitudes = np.array([r["amplitude"] for r in here])
        b11 = np.array([r["b11"] for r in here])
        slope = float(np.sum(amplitudes * b11) / np.sum(amplitudes**2))
        needed = published.value / slope if slope > 0.0 else float("nan")
        ratio = float(np.median([r["ratio"] for r in here]))
        predicted = ratio * published.value
        summary.append(
            {
                "deviation": mode,
                "amplitude_for_b11": needed,
                "b22_over_b11": ratio,
                "b22_predicted": predicted,
                "b22_against_published": predicted / published.value,
            }
        )
        print(
            f"{mode:30s} {needed:18.5f} {predicted:14.3e} "
            f"{predicted / published.value:17.3f}x"
        )

    within = [s for s in summary if 0.5 <= s["b22_against_published"] <= 2.0]
    best = min(summary, key=lambda s: abs(np.log(s["b22_against_published"])))
    print()
    print(
        f"every deviation reproduces the measured 1/1 by construction; "
        f"{len(within)} of {len(summary)} also predict a 2/2 within a factor of two of the "
        f"published one"
    )
    print(
        f"the closest is a {best['deviation']} of {best['amplitude_for_b11'] * 1e3:.2f} "
        f"mm or mrad, which predicts a 2/2 of {best['b22_predicted']:.2e} against a "
        f"published {published.value:.2e}, a factor of "
        f"{max(best['b22_against_published'], 1 / best['b22_against_published']):.2f}"
    )
    spread = max(s["b22_against_published"] for s in summary) / min(
        s["b22_against_published"] for s in summary
    )
    print(
        f"on the 1/1 alone the deviations are indistinguishable; on the 2/2 they differ by "
        f"a factor of {spread:.1f}, which is what separates them"
    )

    write_record(
        INTRINSIC_OUT,
        {
            "configuration": configuration,
            "module": MODULE + 1,
            "filaments": len(members),
            "published_b11": published.value,
            "published_source": published.source,
            "measured_phase_11_degrees": one_one.mode_phase_degrees,
            "measured_phase_22_degrees": two_two.mode_phase_degrees,
            "scan": rows,
            "summary": summary,
            "closest": best,
        },
        geometry=twin.geometry,
    )
    return 0


# -- history -----------------------------------------------------------------------

# A discharge advanced through its heating waveform.
#
# 20180919.033 stepped 2 MW ECRH to 3.4 MW NBI at 1.7 s; the current and transform lag the energy by seconds.

def wolf033_trajectory(twin: Twin, record_path) -> list[tuple[float, float]]:
    """Kinetic stored energy at each drawn time of Wolf 2019 figure 11, in (s, J).

    The laser coordinate maps to r_eff about the profile peaks with the r_eff = 0.3 m
    white-window edge fixing the scale; the ion channel is carried at each time's
    peak T_i over peak T_e, its own abscissa being the NBI plane."""
    record = json.loads(Path(record_path).read_text())
    figures = {f["quantity"]: f for f in record["figures"]}
    output = twin.solve(twin.state("standard"), SCAN)
    wout = output.wout
    ns = int(wout.ns)
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)
    dv_ds = np.abs(np.asarray(wout.vp)[1:]) * 4.0 * np.pi**2
    minor = float(wout.Aminor_p)
    centre, edge = 0.735, float(figures["electron density"]["core_window"][1])
    scale = 0.3 / (edge - centre)

    out: list[tuple[float, float]] = []
    for index, series in enumerate(figures["electron density"]["series"]):
        time = float(series["time_s"])

        def profile(figure, unit_scale, index=index):
            entry = figures[figure]["series"][index]
            x = np.asarray(entry["x"])
            y = np.asarray(entry["y"]) * unit_scale
            r_eff = np.abs(x - centre) * scale
            order = np.argsort(r_eff)
            s = np.clip((r_eff[order] / minor) ** 2, 0.0, 1.0)
            return np.interp(s_half, s, y[order])

        n = profile("electron density", 1.0e19)
        te = profile("electron temperature", 1.0e3)
        ratio = float(
            np.max(figures["ion temperature"]["series"][index]["y"])
            / max(np.max(figures["electron temperature"]["series"][index]["y"]), 0.1)
        )
        pressure = 1.602176634e-19 * n * (te + ratio * te)
        out.append((time, 1.5 * float(np.trapezoid(pressure * dv_ds, s_half))))
    return out


#
#     python -m w7x_twin history [identifier]

HISTORY_OUT = Path("results/discharges/discharge_history.json")
STEPS = 601
#: Seconds the trace runs, covering both phases of the discharge and their relaxation.
DURATION_S = 6.0


def run_history() -> int:
    identifier = arg(1, default="20180919.033")
    programme = programmes.get(identifier)
    print(f"{programme.identifier}  ({programme.campaign}, epoch {programme.epoch})")
    print(f"  {programme.description}")

    ecrh = programme.measured["heating_power_ecrh_w"].value
    beam = (
        programme.measured["heating_power_nbi_w"].value
        if "heating_power_nbi_w" in programme.measured
        else ecrh
    )
    switch = programme.phase_s[0] if programme.phase_s else 1.7

    twin = _common.twin()
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    analysis = diagnostics.analyse(equilibrium)
    profiles = kinetics.HIGH_PERFORMANCE
    print(f"{twin.geometry}")
    print(
        f"{ecrh / 1e6:.1f} MW to {switch:.2f} s, then {beam / 1e6:.1f} MW, over "
        f"{DURATION_S:.1f} s"
    )

    waveform = current.Waveform.steps(
        ((switch, ecrh), (DURATION_S - switch, beam))
    )

    # The confinement time the scaling gives at this density and field, which is what the
    # energy relaxes on. It depends on the power rather than on the energy, so the closure is
    # the scaling itself.
    def confinement_time(energy_j: float, power_w: float) -> float:
        return float(
            transport.iss04_confinement_time(
                minor_radius_m=analysis.minor_radius_m,
                major_radius_m=analysis.major_radius_m,
                heating_power_w=max(power_w, 1e3),
                line_density_m3=profiles.density_axis_m3,
                field_t=analysis.b_axis_t,
                iota_two_thirds=analysis.iota_edge,
            )
            * transport.MEASURED_ISS04_RANGE[0]
        )

    # The bootstrap current the stored energy implies, taken linear in beta from the
    # self-consistent solve at the reference scenario, which is what the coupled solve gives.
    reference_energy = 1.099e6
    reference_current = -12.86e3

    def bootstrap_current(energy_j: float) -> float:
        return reference_current * energy_j / reference_energy

    # Edge transform against net current, from the machine's own current-mimic tapers.
    mimic_current = np.array([0.0, 11.0e3, 22.0e3, 32.0e3, 43.0e3])
    mimic_transform = np.array([0.87836, 0.89470, 0.91259, 0.93143, 0.95157])

    def edge_transform(current_a: float) -> float:
        return float(np.interp(abs(current_a), mimic_current, mimic_transform))

    trace = current.advance(
        waveform, confinement_time, bootstrap_current, edge_transform,
        minor_radius_m=analysis.minor_radius_m,
        temperature_for_resistivity_ev=profiles.electron_temperature_axis_ev * 0.5,
        steps=STEPS,
    )
    inductive = current.resistive_time_s(
        profiles.electron_temperature_axis_ev * 0.5, analysis.minor_radius_m
    )
    print(
        f"energy confinement {trace.confinement_time_s[-1]:.3f} s against an inductive time "
        f"of {inductive:.2f} s"
    )

    header = (
        f"{'t [s]':>7s} {'P [MW]':>8s} {'W [MJ]':>8s} {'I_boot [kA]':>12s} "
        f"{'iota edge':>10s}"
    )
    print()
    print(header)
    print("-" * len(header))
    for moment in (0.5, 1.0, 1.6, 1.8, 2.5, 3.5, 5.0, 6.0):
        if moment > DURATION_S:
            continue
        point = trace.at(moment)
        print(
            f"{point['time_s']:7.2f} {point['power_w'] / 1e6:8.2f} "
            f"{point['stored_energy_j'] / 1e6:8.3f} "
            f"{point['bootstrap_current_a'] / 1e3:12.2f} "
            f"{point['edge_transform']:10.5f}"
        )

    # How far behind the current runs: the energy settles within a few tau_E of the step and
    # the current within a few inductive times.
    after = trace.time_s >= switch
    energy_settled = float(
        trace.time_s[after][
            int(np.argmax(np.abs(np.diff(trace.stored_energy_j[after], prepend=0)) < 1e3))
        ]
        - switch
    )
    final = trace.bootstrap_current_a[-1]
    reached = np.abs(trace.bootstrap_current_a[after] - final) <= 0.05 * abs(final)
    current_settled = (
        float(trace.time_s[after][int(np.argmax(reached))] - switch)
        if reached.any()
        else float("nan")
    )
    print()
    print(
        f"after the step the energy settles in {energy_settled:.2f} s and the current "
        f"within five per cent of its final value in {current_settled:.2f} s"
    )

    write_record(
        HISTORY_OUT,
        {
            "identifier": programme.identifier,
            "ecrh_w": ecrh,
            "beam_w": beam,
            "switch_s": switch,
            "inductive_time_s": inductive,
            "time_s": trace.time_s.tolist(),
            "power_w": trace.power_w.tolist(),
            "stored_energy_j": trace.stored_energy_j.tolist(),
            "bootstrap_current_a": trace.bootstrap_current_a.tolist(),
            "edge_transform": trace.edge_transform.tolist(),
            "energy_settling_s": energy_settled,
            "current_settling_s": current_settled,
        },
        geometry=twin.geometry,
    )
    return 0


# -- profiles ----------------------------------------------------------------------

# The solved profiles against the measured ones, inside the bands the figures draw.
#
# Solved profiles against the pellet paper's digitised pair, point by point against the drawn bands.
#
#     python -m w7x_twin profiles [discharge ...]

PROFILES_OUT = Path("results/discharges/profile_residuals.json")
#: Heating power the balance is solved at where the discharge states none.
DEFAULT_POWER_W = 5.0e6
#: The two discharges the pellet paper's profile figures carry.
DISCHARGES = ("20181016.037", "20180920.017")
#: Confinement each is measured at, keyed by the programme that publishes it.
ENHANCEMENT = {
    "20181016.037": ("20171207.006", "confinement_over_iss04"),
    "20180920.017": ("20180920.017", "confinement_over_iss04"),
}


def compare_curve(name: str, modelled, profile, radius) -> dict:
    """Point-by-point residual of a solved profile against a measured one."""
    reference = profile.at(radius)
    spread = profile.uncertainty_at(radius)
    residual = modelled - reference
    # Where the figure draws no band the residual is reported against the curve alone.
    inside = (
        np.abs(residual) <= np.maximum(spread, 1e-30)
        if np.any(spread > 0.0)
        else np.zeros_like(residual, dtype=bool)
    )
    relative = residual / np.maximum(np.abs(reference), 1e-30)
    return {
        "quantity": name,
        "points": int(len(radius)),
        "median_relative_residual": float(np.median(np.abs(relative))),
        "worst_relative_residual": float(np.max(np.abs(relative))),
        "inside_the_band": float(np.mean(inside)),
        "band_half_width_median": float(np.median(spread)) if np.any(spread) else 0.0,
        "measured_peak": float(np.max(reference)),
        "modelled_peak": float(np.max(modelled)),
        "source": profile.source,
        "figure": profile.figure,
        "axis_residual": profile.axis_residual,
    }


def run_profiles() -> int:
    wanted = args() or list(DISCHARGES)
    twin = _common.twin()
    print(f"{twin.geometry}")

    coefficients = neoclassical.load_radial_profile(verbose=False)
    ripple = neoclassical.load_ripple()
    reference = twin.solve(twin.state("standard"), SCAN)
    minor = float(reference.wout.Aminor_p)
    chi_neoclassical = neoclassical.diffusivity_model(coefficients, ripple, minor)
    field_capture: dict = {}
    split_channels = neoclassical.split_diffusivity_model(
        coefficients, ripple, minor, capture=field_capture
    )
    anomalous = transport.anomalous_channel(
        abs(float(reference.wout.b0)), minor, verbose=False, local=True
    )
    quench = transport.quench_channel(abs(float(reference.wout.b0)), minor)
    print(
        f"minor radius {minor:.4f} m, anomalous channel "
        + ("computed" if anomalous is not None else "scaled to the confinement scaling")
        + (", two-temperature balance" if anomalous is not None else "")
    )

    records = []
    for discharge in wanted:
        density_profile = programmes.find(discharge, "electron density", "post-pellet") if (
            discharge == "20181016.037"
        ) else None
        electron = programmes.find(discharge, "temperature", "electron")
        ion = programmes.find(discharge, "temperature", "ion")
        programme, key = ENHANCEMENT[discharge]
        enhancement = programmes.get(programme).measured[key].value
        power = (
            programmes.get(discharge).measured["heating_power_ecrh_w"].value
            if "heating_power_ecrh_w" in programmes.get(discharge).measured
            else DEFAULT_POWER_W
        )
        print()
        print(
            f"{discharge}: {power / 1e6:.1f} MW at {enhancement:.2f} times ISS04, "
            f"profiles from figure {electron.figure}"
        )
        print(f"  {electron.source}")

        # The measured density where there is one, so the temperature comparison is not
        # answering a question about the density as well.
        base = kinetics.HIGH_PERFORMANCE
        if density_profile is not None:
            outboard = density_profile.x >= 0.0
            s_of = np.clip(density_profile.x[outboard] ** 2, 0.0, 1.0)
            values = 1.0e19 * density_profile.y[outboard]
            order = np.argsort(s_of)
            base = dataclasses.replace(
                base,
                density_axis_m3=float(np.max(values)),
                density_points=tuple(
                    (float(a), float(b))
                    for a, b in zip(s_of[order], values[order], strict=True)
                ),
            )
            print(
                f"  the measured density peaks at {np.max(values):.3e} m^-3 and peaks by "
                f"{density_profile.peaking():.2f}, carried as the drawn curve"
            )

        equilibrium = twin.solve_profiles("standard", base)
        # The deposition the traced ray gives, so the solved shapes carry the beam's
        # own path; where the ray does not cross, the resonance layer stands.
        from w7x_twin.analyses.plasma import ray_traced_deposition

        traced, ray = ray_traced_deposition(
            twin, equilibrium, minor, base, np.linspace(0.0, 1.0, 81), power
        )
        if traced is not None:
            heating_input = transport.Heating.from_deposition(power, traced)
            print(f"  deposition from the traced ray: {traced.note}")
        else:
            heating_input = transport.Heating(power_w=power)
            print(f"  the traced ray did not cross ({ray.note})")
        if anomalous is not None:
            # Both channels computed and the two temperatures coupled through the
            # collisional exchange, so the ion-to-electron ratio is an outcome: at the
            # post-pellet density the exchange equilibrates the pair the way the drawn
            # profiles show, which a held ratio cannot. The turbulent channel goes in
            # as the pointwise gradient model so each shell settles on the measured
            # response rather than oscillating across it.
            solution = transport.solve_split(
                equilibrium, base, heating_input, split_channels,
                turbulent_local=anomalous,
                shear_quench=quench, field_capture=field_capture,
            )
            print(
                f"  exchange carries {solution.exchange_power_w / 1e6:.2f} MW to the "
                f"ions; T_i/T_e on axis "
                f"{solution.ion_temperature_ev[0] / solution.electron_temperature_ev[0]:.2f}"
            )
        else:
            solution = transport.solve(
                equilibrium, base, heating=heating_input,
                model=transport.TransportModel(renormalisation=enhancement),
                neoclassical=chi_neoclassical, anomalous=anomalous,
            )

        radius = np.linspace(0.0, min(0.45, float(electron.x.max())), 41)
        s = np.clip((radius / minor) ** 2, 0.0, 1.0)
        checks = [
            compare_curve(
                "electron temperature",
                1e-3 * np.interp(s, solution.s, solution.electron_temperature_ev),
                electron, radius,
            ),
            compare_curve(
                "ion temperature",
                1e-3 * np.interp(s, solution.s, solution.ion_temperature_ev),
                ion, radius,
            ),
        ]
        if density_profile is not None:
            rho = np.linspace(0.0, 0.95, 41)
            checks.append(
                compare_curve(
                    "electron density",
                    1e-19 * np.interp(rho**2, solution.s, solution.density_m3),
                    density_profile, rho,
                )
            )

        for check in checks:
            print(
                f"  {check['quantity']:22s} median residual "
                f"{100 * check['median_relative_residual']:6.1f} %, "
                f"{100 * check['inside_the_band']:5.1f} % of points inside the drawn band, "
                f"peak {check['modelled_peak']:.3f} against {check['measured_peak']:.3f}"
            )
        records.append(
            {
                "discharge": discharge,
                "heating_power_w": power,
                "confinement_over_iss04": enhancement,
                "stored_energy_j": float(solution.stored_energy_j),
                "confinement_over_iss04_modelled": float(
                    solution.confinement_time_s / solution.iss04_time_s
                ),
                "checks": checks,
            }
        )

    print()
    for record in records:
        print(
            f"{record['discharge']}: the balance holds "
            f"{record['stored_energy_j'] / 1e6:.3f} MJ at "
            f"{record['confinement_over_iss04_modelled']:.3f} times ISS04 against a "
            f"measured {record['confinement_over_iss04']:.2f}"
        )
    inside = [
        check["inside_the_band"]
        for record in records
        for check in record["checks"]
        if check["band_half_width_median"] > 0.0
    ]
    if inside:
        print(
            f"across every profile carrying a band, "
            f"{100 * float(np.mean(inside)):.0f} per cent of the compared points fall "
            f"inside it"
        )

    write_record(PROFILES_OUT, {"discharges": records}, geometry=twin.geometry)
    return 0


# -- discharge ---------------------------------------------------------------------

# The model against identified W7-X discharges.
#
# Identified discharges reproduced from published heating power and configuration alone.
#
#     python -m w7x_twin discharge [identifier ...]

DISCHARGE_OUT = Path("results/discharges/reproduce_discharge.json")
#: The line count sets the arc resolution of the deposition profile the width and flux
#: rows are read from, not only the strike statistics.
DISCHARGE_LINES = 240
#: Toroidal launch planes across one field period. The strike line is a band inclined
#: across each target, and one plane samples a single comb of it; the published width
#: and flux are read off cameras that see the whole band, so the fan samples the
#: layer's own toroidal extent rather than one cut of it.
LAUNCH_PLANES = 5
#: Percentile of the connection-length distribution taken as the strike-line value. The
#: power is carried by the long field lines, not by the median of a fan.
STRIKE_LINE_PERCENTILE = 90


def compare(name: str, modelled: float, measured: programmes.Measured) -> dict:
    """One residual, against the band the published accuracy supports."""
    low, high = measured.band()
    inside = bool(low <= modelled <= high)
    relative = (
        (modelled - measured.value) / measured.value if measured.value else float("nan")
    )
    print(
        f"  {'ok  ' if inside else '??  '}{name:34s} model {modelled:12.4g} "
        f"measured {measured.value:12.4g} {measured.unit:9s} "
        f"{100 * relative:+8.1f} %"
    )
    return {
        "quantity": name,
        "modelled": float(modelled),
        "measured": measured.value,
        "unit": measured.unit,
        "relative_residual": float(relative),
        "within_published_accuracy": inside,
        "source": measured.source,
    }


def power_balance(
    twin: Twin, power_w: float, density_axis_m3: float, renormalisation: float,
    neoclassical=None, profiles: kinetics.KineticProfiles | None = None,
):
    """The power balance at this power, density and confinement enhancement."""
    profiles = dataclasses.replace(
        profiles or kinetics.HIGH_PERFORMANCE, density_axis_m3=density_axis_m3
    )
    equilibrium = twin.solve_profiles("standard", profiles)
    return transport.solve(
        equilibrium, profiles, heating=transport.Heating(power_w=power_w),
        model=transport.TransportModel(renormalisation=renormalisation),
        neoclassical=neoclassical,
    )


def stored_energy(
    twin: Twin, power_w: float, density_axis_m3: float,
    renormalisation: float = RENORMALISATION,
) -> tuple[float, float]:
    """Stored energy and confinement time the power balance returns at this power."""
    solution = power_balance(twin, power_w, density_axis_m3, renormalisation)
    return float(solution.stored_energy_j), float(solution.confinement_time_s)


def invert_density(
    twin: Twin,
    power_w: float,
    measured_energy_j: float,
    bracket: tuple[float, float] = (2.0e18, 4.0e20),
    iterations: int = 24,
    renormalisation: float = RENORMALISATION,
) -> float:
    """Axis density at which the model reproduces a measured stored energy, by bisection."""
    low, high = bracket
    for _ in range(iterations):
        middle = float(np.sqrt(low * high))
        energy, _ = stored_energy(twin, power_w, middle, renormalisation)
        if energy < measured_energy_j:
            low = middle
        else:
            high = middle
    return float(np.sqrt(low * high))


def digitised_kinetics(identifier: str, core_radius_m: float):
    """Digitised profiles as the share's inputs, with the measured core Ti/Te, or None if incomplete."""
    from w7x_twin.records import programmes as measured_profiles

    curves = measured_profiles.load()

    def pick(quantity: str, label: str):
        for curve in curves:
            if (
                curve.discharge == identifier
                and quantity in curve.quantity
                and label in curve.label
            ):
                return curve
        return None

    density = pick("density", "post-pellet")
    electron = pick("temperature", "electron")
    ion = pick("temperature", "ion")
    if density is None or electron is None or ion is None:
        return None
    span = float(np.max(electron.x))

    class Digitised:
        @staticmethod
        def density(s):
            rho = np.sqrt(np.clip(np.asarray(s, dtype=float), 0.0, 1.0))
            return density.at(rho) * 1e19

        @staticmethod
        def electron_temperature(s):
            rho = np.sqrt(np.clip(np.asarray(s, dtype=float), 0.0, 1.0))
            return electron.at(rho * span) * 1e3

    ratio = float(
        ion.at(np.array([core_radius_m]))[0]
        / max(float(electron.at(np.array([core_radius_m]))[0]), 1e-9)
    )
    return Digitised(), ratio


def digitised_temperatures(identifier: str):
    """Both drawn temperature profiles of a discharge as callables of s, in eV."""
    from w7x_twin.records import programmes as measured_profiles

    try:
        electron = measured_profiles.find(identifier, "temperature", "electron")
        ion = measured_profiles.find(identifier, "temperature", "ion")
    except KeyError:
        return None
    span = float(np.max(electron.x))

    def of(curve):
        def at(s):
            rho = np.sqrt(np.clip(np.asarray(s, dtype=float), 0.0, 1.0))
            return curve.at(rho * span) * 1e3

        return at

    return of(electron), of(ion)


def neoclassical_share(
    twin: Twin,
    profiles: kinetics.KineticProfiles,
    coefficients,
    ripple,
    power_w: float,
    radius_m: float | np.ndarray,
    minor_radius_m: float,
    surfaces: int = 80,
    ion_fraction: float = neoclassical.ION_TEMPERATURE_FRACTION,
    radial_field_v_m: float | None = None,
    z_effective: float = 1.0,
    species: str = "both",
) -> float | np.ndarray:
    """Drift-kinetic heat flux through a surface as a fraction of heating, both Onsager
    drives carried; ``species`` selects the channel and ``radius_m`` may be an array."""
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    s = np.linspace(1e-4, 1.0, surfaces)
    radius = minor_radius_m * np.sqrt(s)
    density = profiles.density(s)
    electron = profiles.electron_temperature(s)
    ion = ion_fraction * electron

    # Logarithmic gradients per metre, which is what the Onsager drives are.
    dln_n = log_gradient(density, radius)
    dln_te = log_gradient(electron, radius)
    dln_ti = log_gradient(ion, radius)

    volume = np.abs(np.asarray(equilibrium.wout.vp)) * 4.0 * np.pi**2
    grid = np.linspace(0.0, 1.0, len(volume))
    area = np.interp(s, grid, volume) * 2.0 * np.sqrt(s) / minor_radius_m

    wanted = np.atleast_1d(np.asarray(radius_m, dtype=float))
    shares = np.empty(wanted.shape)
    for position, value in enumerate(wanted):
        index = int(np.argmin(np.abs(radius - value)))
        total = 0.0
        for table, weight in neoclassical.surface_tables(
            coefficients, float(s[index]), which="d11", ripple=ripple
        ):
            if weight == 0.0:
                continue
            field = radial_field_v_m
            if field is None:
                answer = neoclassical.ambipolar_field(
                    table, density_m3=float(density[index]),
                    electron_temperature_ev=float(electron[index]),
                    ion_temperature_ev=float(ion[index]),
                    density_gradient=float(dln_n[index]),
                    electron_temperature_gradient=float(dln_te[index]),
                    ion_temperature_gradient=float(dln_ti[index]),
                    bracket=(-25.0e3, 25.0e3), num_probe=41,
                )
                field = float(answer["field"]) if np.isfinite(answer["field"]) else 0.0
            channels = {
                "electron": (
                    (neoclassical.ELECTRON_MASS, -1.0, electron[index], dln_te[index]),
                ),
                "ion": ((neoclassical.PROTON_MASS, 1.0, ion[index], dln_ti[index]),),
            }
            channels["both"] = channels["electron"] + channels["ion"]
            for mass, charge, temperature, gradient in channels[species]:
                reduced = neoclassical.heat_flux(
                    table,
                    density_m3=float(density[index]),
                    temperature_ev=float(temperature),
                    density_gradient=float(dln_n[index]),
                    temperature_gradient=float(gradient),
                    mass=mass, charge_number=charge, z_effective=z_effective,
                    radial_field_v_m=field,
                )
                # The reduced flux is Q / (n T); the power through the surface is that
                # times the density, the temperature in joules and the area.
                total += weight * reduced * density[index] * float(
                    temperature
                ) * kinetics.ELEMENTARY_CHARGE * area[index]
        shares[position] = total / max(power_w, 1e-30)
    return shares if np.ndim(radius_m) else float(shares[0])


#: The traced boundary layer, which two separate checks need and which costs a pair of
#: field-line traces to produce.
_LAYER: dict | None = None


def layer_geometry(twin: Twin) -> dict:
    """Trace the scrape-off layer once and keep what both consumers of it need."""
    global _LAYER
    if _LAYER is not None:
        return _LAYER
    vessel = _common.vessel()
    elements = _common.components()
    frame = walls.target_arc_frame(elements)
    vacuum = VacuumField(twin.response, twin.state("standard").currents)
    equilibrium = twin.solve(twin.state("standard"), SCAN)

    # Both toroidal directions at every plane, since a connection length is the distance
    # between the two surfaces a line ends on. Each plane's fan is anchored to its own
    # axis and its own boundary cut, and its launch offsets are remapped onto one
    # reference separatrix, which is the only thing the layer weights read.
    period = 2.0 * np.pi / walls.NUM_FIELD_PERIODS_DEFAULT
    separatrix = None
    per_plane = []
    lengths_parts = []
    closed_lines = 0
    for index in range(LAUNCH_PLANES):
        phi = index * period / LAUNCH_PLANES
        starts, axis_r, axis_z, outboard = fieldlines.fan_starts(
            vacuum, equilibrium.wout, LAYER, DISCHARGE_LINES, plane_phi=phi
        )
        if separatrix is None:
            separatrix = outboard
        connection = fieldlines.connection_lengths(
            vacuum, starts, np.full(starts.shape, axis_z), vessel, elements,
            turns=TURNS, plane_phi=phi,
        )
        closed_lines += int(connection.closed.sum())
        per_plane.append(
            dataclasses.replace(
                connection.forward, start_r=separatrix + (starts - outboard)
            )
        )
        lengths_parts.append(connection.length_m)
    strikes = fieldlines.Strikes(
        struck=np.concatenate([s.struck for s in per_plane]),
        r=np.concatenate([s.r for s in per_plane]),
        z=np.concatenate([s.z for s in per_plane]),
        phi=np.concatenate([s.phi for s in per_plane]),
        connection_length_m=np.concatenate(
            [s.connection_length_m for s in per_plane]
        ),
        start_r=np.concatenate([s.start_r for s in per_plane]),
        component=np.concatenate([s.component for s in per_plane]),
        component_names=per_plane[0].component_names,
    )
    lengths_m = np.concatenate(lengths_parts)
    wanted_elements = [i for i, e in enumerate(elements) if e.name in frame]
    mask = strikes.struck & np.isin(strikes.component, wanted_elements)
    print(
        f"  {LAUNCH_PLANES} launch planes x {DISCHARGE_LINES} lines: {closed_lines} of "
        f"{LAUNCH_PLANES * DISCHARGE_LINES} reach a surface both ways, so the rest carry a "
        f"lower bound on their length"
    )
    _LAYER = {
        "vessel": vessel, "elements": elements, "frame": frame, "vacuum": vacuum,
        "equilibrium": equilibrium, "separatrix": separatrix, "strikes": strikes,
        "lengths_m": lengths_m, "mask": mask,
        "wanted_elements": wanted_elements,
    }
    return _LAYER


def boundary_radiation_w(twin: Twin, solution, carbon_fraction: float) -> float:
    """Scrape-off-layer carbon radiation of this transport solution, in watts."""
    if carbon_fraction <= 0.0:
        return 0.0
    traced = layer_geometry(twin)
    strikes, mask = traced["strikes"], traced["mask"]
    if not mask.any():
        return 0.0
    separatrix = traced["separatrix"]
    lengths_m = traced["lengths_m"]
    length = float(np.percentile(lengths_m[mask], STRIKE_LINE_PERCENTILE))
    crossing = max(
        float(solution.heating_power_w) - float(solution.radiated_power_w), 1.0
    )
    # Per-line incidence against each target's own surface, the same construction the
    # width and flux rows run on. The horizontal target takes the field at several
    # times the fan's median angle, and pricing its tubes at the median understates
    # their parallel flux against the Lengyel loss by that factor squared.
    sines = []
    sine_by_line = np.full(strikes.start_r.shape, np.nan)
    for index in traced["wanted_elements"]:
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        sine = edge.surface_incidence_sine(
            traced["vacuum"],
            strikes.r[on_element],
            strikes.phi[on_element],
            strikes.z[on_element],
            walls.surface_frame(
                traced["elements"][index],
                strikes.r[on_element],
                strikes.z[on_element],
                strikes.phi[on_element],
            ),
        )
        sines.append(sine)
        sine_by_line[on_element] = sine
    incidence = float(np.median(np.concatenate(sines)))

    def area_of_width(width: float) -> float:
        weights = edge.layer_weights(strikes.start_r, separatrix, width)
        return edge.wetted_area(strikes, traced["elements"], traced["frame"], weights)[
            "area_m2"
        ]

    try:
        closed = edge.close_layer(
            float(solution.density_m3[-1]), crossing, length,
            float(solution.chi_m2_s[-1]), incidence, area_of_width,
        )
    except ValueError as failure:
        # One carbon fraction that will not close its layer is a point of a scan, not the
        # end of one, so it is reported and the scan carries on without it.
        print(f"        no boundary radiation at {carbon_fraction:.2f} carbon: {failure}")
        return 0.0
    # The radiator resolved along the targets: every arc bin is its own tube, so the
    # radiated power is a sum over the deposition rather than one tube standing for all.
    # Each tube carries the upstream density its own launch offset supplies, the
    # particle width being the heat width over root three at the study's chi = 3 D.
    weights = edge.layer_weights(strikes.start_r, separatrix, closed["width_m"])
    deposition = edge.target_profile(
        strikes, traced["elements"], traced["frame"], weights, crossing,
        incidence_by_line=sine_by_line, deskew=True,
        offset_by_line=strikes.start_r - separatrix,
    )
    return float(
        edge.target_radiator(
            deposition, float(solution.density_m3[-1]), incidence, carbon_fraction,
            density_width_m=closed["width_m"] / np.sqrt(3.0),
            dilution_by_element=edge.strip_dilution(
                strikes, traced["elements"], traced["frame"], weights
            ),
        )["radiated_w"]
    )


def _fraction_for_charge(
    z_effective: float,
    temperature_ev: float = 3.0e3,
    density_m3: float = 1.0e20,
    iterations: int = 60,
) -> float:
    """Carbon fraction whose composition gives this effective charge, by bisection."""
    from w7x_twin.plasma import kinetics

    def charge(fraction: float) -> float:
        parts = kinetics.composition(
            np.array([density_m3]), np.array([temperature_ev]), fraction
        )
        return float(parts.z_effective[0])

    low, high = 0.0, 0.15
    for _ in range(iterations):
        middle = 0.5 * (low + high)
        if charge(middle) < z_effective:
            low = middle
        else:
            high = middle
    return 0.5 * (low + high)


def run_discharge() -> int:
    wanted = args() or programmes.identifiers()
    twin = _common.twin()
    print(f"{twin.geometry}")
    print(f"where a discharge states no enhancement the model runs at {RENORMALISATION}")

    # The drift-kinetic transport, for the discharges whose source states what share of
    # the core power it carries.
    coefficients = neoclassical.load_radial_profile(verbose=False)
    ripple = neoclassical.load_ripple()
    reference = twin.solve(twin.state("standard"), SCAN)
    minor_radius = float(reference.wout.Aminor_p)
    chi_neoclassical = neoclassical.diffusivity_model(coefficients, ripple, minor_radius)
    chi_no_field = neoclassical.diffusivity_model(
        coefficients, ripple, minor_radius, radial_field_v_m=0.0
    )
    core_radius = programmes.MACHINE_MEASUREMENTS["core_neoclassical_radius_m"].value

    # The carbon fraction the machine's measured effective charge implies. Fully stripped
    # carbon at core temperatures gives Z_eff = 1 + 30 f exactly, but the fraction is
    # solved against the charge-state model rather than that limit so it holds wherever
    # the profile is cooler.
    measured_charge = programmes.MACHINE_MEASUREMENTS["effective_charge"].value
    carbon_fraction = _fraction_for_charge(measured_charge)
    print(
        f"an effective charge of {measured_charge:.2f} is a carbon fraction of "
        f"{carbon_fraction:.4f}"
    )

    records = []
    for identifier in wanted:
        programme = programmes.get(identifier)
        print()
        print(f"{programme.identifier}  ({programme.campaign}, epoch {programme.epoch})")
        print(f"  {programme.description}")
        checks: list[dict] = []

        # Each discharge carries its own confinement against the scaling, and they differ
        # by nearly a factor of two between gas-fuelled and post-pellet operation. The
        # model is run at the enhancement the discharge was measured at rather than at one
        # constant, so what is being checked is the balance and not the constant.
        enhancement = (
            programme.measured["confinement_over_iss04"].value
            if "confinement_over_iss04" in programme.measured
            else RENORMALISATION
        )
        if "confinement_over_iss04" in programme.measured:
            print(
                f"  the model runs at this discharge's measured {enhancement:.2f} "
                "times ISS04"
            )
        inverted: list[dict] = []
        for phase, power_key, energy_key, density_key in (
            ("heating phase", "heating_power_ecrh_w", "stored_energy_ecrh_j",
             "central_density_m3"),
            ("beam phase", "heating_power_nbi_w", "stored_energy_nbi_j",
             "central_density_late_m3"),
        ):
            if power_key not in programme.measured or energy_key not in programme.measured:
                continue
            power = programme.measured[power_key].value
            measured_energy = programme.measured[energy_key]

            # Forward: at the density the source states for this phase where it states one,
            # and at the reference profile's otherwise. A high-density beam phase held at
            # the reference density answers a question about a discharge that was not run.
            density_axis = (
                programme.measured[density_key].value
                if density_key in programme.measured
                else kinetics.HIGH_PERFORMANCE.density_axis_m3
            )
            if density_key in programme.measured:
                print(
                    f"      the {phase} runs at its published central density of "
                    f"{density_axis:.2e} m^-3"
                )
            energy, tau = stored_energy(twin, power, density_axis, enhancement)
            checks.append(
                compare(f"stored energy, {phase}", energy, measured_energy)
            )

            # Bracketed across the reported confinement range where none is published.
            if "confinement_over_iss04" not in programme.measured:
                low, high = transport.MEASURED_ISS04_RANGE
                band = [stored_energy(twin, power, kinetics.HIGH_PERFORMANCE.density_axis_m3, f)[0]
                        for f in (low, high)]
                print(
                    f"      no confinement published, so {low:.2f} to {high:.2f} times "
                    f"ISS04 gives {band[0] / 1e6:.2f} to {band[1] / 1e6:.2f} MJ against "
                    f"a measured {measured_energy.value / 1e6:.2f}"
                )
                inverted.append(
                    {
                        "phase": phase,
                        "quantity": "stored energy across the reported confinement range",
                        "heating_power_w": power,
                        "enhancement_range": [low, high],
                        "energy_range_j": band,
                        "measured_energy_j": measured_energy.value,
                        "measured_inside_band": bool(
                            band[0] <= measured_energy.value <= band[1]
                        ),
                    }
                )

            # Inverse: the density that reproduces the measured energy at the measured
            # power. The published density is a figure rather than a number, so this is
            # what the model says the discharge must have run at.
            density = invert_density(
                twin, power, measured_energy.value, renormalisation=enhancement
            )
            inverted.append(
                {
                    "phase": phase,
                    "heating_power_w": power,
                    "measured_energy_j": measured_energy.value,
                    "density_axis_m3": density,
                    "forward_energy_j": energy,
                    "forward_confinement_s": tau,
                }
            )
            print(
                f"      inverting: the measured {measured_energy.value / 1e6:.2f} MJ at "
                f"{power / 1e6:.1f} MW needs an axis density of "
                f"{density:.3e} m^-3, against the reference profile's "
                f"{kinetics.HIGH_PERFORMANCE.density_axis_m3:.3e}"
            )

        # The digitised profile evolution of Wolf 2019 figure 11 gives the discharge's
        # stored-energy trajectory from published data alone.
        wolf = Path(__file__).parents[1] / "records/wolf_fig11.json"
        if programme.identifier == "20180919.033" and wolf.is_file():
            trajectory = wolf033_trajectory(twin, wolf)
            shown = ", ".join(
                f"{t:.1f} s: {w / 1e6:.2f} MJ" for t, w in trajectory
            )
            print(f"      kinetic energy of the digitised profiles: {shown}")
            checks.append(
                compare(
                    "stored energy at the ECRH end, digitised profiles",
                    trajectory[0][1], programme.measured["stored_energy_ecrh_j"],
                )
            )
            checks.append(
                compare(
                    "stored energy at the beam-phase peak, digitised profiles",
                    max(w for _, w in trajectory[1:3]),
                    programme.measured["stored_energy_nbi_j"],
                )
            )

        # What fraction of the conducted power the drift-kinetic transport accounts for
        # inside the radius the source quotes. At a fixed profile the conducted flux is
        # proportional to the diffusivity, so this ratio is the power fraction.
        if "core_neoclassical_power_fraction" in programme.measured:
            power = (
                programme.measured["heating_power_ecrh_w"].value
                if "heating_power_ecrh_w" in programme.measured
                else 5.0e6
            )
            # At the discharge's own conditions rather than the reference profile's. The
            # published power balance was run at an effective charge of about 1.5 and at
            # the temperatures the post-pellet phase reached, and both move the share:
            # the impurity through the collision frequency, the temperature through the
            # T^(7/2) of the 1/nu regime.
            local = kinetics.HIGH_PERFORMANCE
            ion_fraction = neoclassical.ION_TEMPERATURE_FRACTION
            # The published fraction belongs to the phase its source quotes it for, and the
            # 1/nu coefficient goes as T^(7/2), so the temperature it is evaluated at has to
            # be that phase's. Where the source gives an ion temperature for the phase
            # rather than an equilibrated pair, that is what the ion channel is formed at.
            temperature_key = next(
                (
                    key
                    for key in ("central_temperature_ev", "pellet_phase_ion_temperature_ev")
                    if key in programme.measured
                ),
                None,
            )
            if temperature_key is not None:
                central = programme.measured[temperature_key].value
                local = dataclasses.replace(
                    local,
                    electron_temperature_axis_ev=central,
                    ion_temperature_axis_ev=central,
                )
                # The source states the two temperatures equilibrated at that value, and
                # the ion channel is the larger part of the neoclassical loss, so the ratio
                # it is evaluated at is a measured input rather than the model's default.
                ion_fraction = 1.0
                print(
                    f"      the share is formed at the {central / 1e3:.1f} keV the source "
                    f"gives for this phase, from {temperature_key}"
                )
            local = dataclasses.replace(local, carbon_fraction=carbon_fraction)
            density = (
                programme.measured["peak_density_m3"].value
                if "peak_density_m3" in programme.measured
                else kinetics.HIGH_PERFORMANCE.density_axis_m3
            )

            # The neoclassical flux depends on the density gradient as much as on the
            # temperature, and a post-pellet profile is peaked where the reference one is
            # flat. The peaking is what the source reports its profiles by, so the share is
            # solved across the range the source's own figure spans.
            peaking = programmes.MACHINE_MEASUREMENTS["pellet_density_peaking"]
            low_peak, high_peak = peaking.band()
            print(
                f"      at n(0) = {density:.2e} m^-3, T(0) = "
                f"{local.electron_temperature_axis_ev:.0f} eV and Z_eff = "
                f"{local.z_effective_profile(np.array([0.0]))[0]:.2f} on axis"
            )
            # A power balance analysis computes the neoclassical flux from the measured
            # profile. It does not solve a profile from a confinement scaling and then
            # evaluate the flux on that, which would put the drift-kinetic coefficient at
            # whatever temperature the scaling produced rather than at the measured one,
            # and the 1/nu coefficient goes as T^(7/2). So the profile here is prescribed
            # from what the source states and the flux is formed on its own gradient.
            # With and without the radial electric field. The field closes off the 1/nu
            # regime and is the largest single lever on the drift-kinetic flux, so the two
            # bracket what the coefficient can produce on this profile.
            shares = []
            for factor in np.linspace(low_peak, high_peak, 5):
                shaped = dataclasses.replace(
                    local.with_peaking(float(factor)), density_axis_m3=density
                )
                value = neoclassical_share(
                    twin, shaped, coefficients, ripple, power, core_radius,
                    minor_radius, ion_fraction=ion_fraction,
                    z_effective=measured_charge,
                )
                without = neoclassical_share(
                    twin, shaped, coefficients, ripple, power, core_radius,
                    minor_radius, ion_fraction=ion_fraction,
                    radial_field_v_m=0.0, z_effective=measured_charge,
                )
                shares.append((float(factor), value))
                print(
                    f"        peaking {factor:.2f}: {value:.4f} of the input power "
                    f"crosses {core_radius:.2f} m as drift-kinetic flux, "
                    f"{without:.4f} with no radial electric field"
                )
            # The source's own figure places the post-pellet phases at the top of the
            # peaking span it draws, and the published fraction belongs to that phase,
            # so the point value is formed there rather than at the span's centre.
            share = float(shares[-1][1])
            # The peaking is a range in the source, so the share it supports is one too.
            values = [v for _, v in shares]
            measured_share = programme.measured["core_neoclassical_power_fraction"]
            spans = bool(min(values) <= measured_share.value <= max(values))
            print(
                f"      across the source's own peaking range the share spans "
                f"{min(values):.4f} to {max(values):.4f}, which "
                + ("brackets" if spans else "does not reach")
                + f" the published {measured_share.value:.2f}"
            )
            checks.append(
                {
                    "quantity": "neoclassical share across the reported peaking range",
                    "modelled_low": float(min(values)),
                    "modelled_high": float(max(values)),
                    "measured": measured_share.value,
                    "unit": measured_share.unit,
                    "within_published_accuracy": spans,
                    "source": measured_share.source,
                }
            )
            checks.append(
                compare(
                    "neoclassical share of the core input power",
                    share,
                    programme.measured["core_neoclassical_power_fraction"],
                )
            )
            # Where the discharge's own profiles are digitised, the share is formed on
            # them directly: the density gradient enters as drawn rather than through a
            # peaking family, and the ion temperature is the measured one rather than a
            # fraction of the electron's.
            digitised = digitised_kinetics(programme.identifier, core_radius)
            if digitised is not None:
                measured_kinetics, ratio = digitised
                # The source states both shares over the region rather than at its
                # boundary: about half of the input power up to 30 cm, and 20 to 40 per
                # cent of the electron input power inside it. The share is therefore a
                # profile across the region, and the model's range is what those ranges
                # can be held against; the 1/nu coefficient's T^(7/2) makes it far
                # larger where the profile is hot than at the region's outer edge.
                region = np.linspace(0.10, core_radius, 5)
                on_measured = neoclassical_share(
                    twin, measured_kinetics, coefficients, ripple, power, region,
                    minor_radius, ion_fraction=ratio, z_effective=measured_charge,
                )
                print(
                    "      on the digitised profiles the share across the region is "
                    + ", ".join(
                        f"{r:.2f} m: {v:.3f}"
                        for r, v in zip(region, on_measured, strict=True)
                    )
                    + f", at the measured T_i/T_e of {ratio:.2f}"
                )
                low_band, high_band = measured_share.band()
                overlaps = bool(
                    float(np.min(on_measured)) <= high_band
                    and float(np.max(on_measured)) >= low_band
                )
                checks.append(
                    {
                        "quantity": "neoclassical share over the core region, digitised profiles",
                        "modelled_low": float(np.min(on_measured)),
                        "modelled_high": float(np.max(on_measured)),
                        "measured": measured_share.value,
                        "unit": measured_share.unit,
                        "within_published_accuracy": overlaps,
                        "source": measured_share.source,
                    }
                )
                print(
                    f"      {'ok  ' if overlaps else '??  '}spans "
                    f"{float(np.min(on_measured)):.3f} to {float(np.max(on_measured)):.3f} "
                    f"against the published {measured_share.value:.2f}"
                )
                if "core_electron_neoclassical_power_fraction" in programme.measured:
                    measured_electron = programme.measured[
                        "core_electron_neoclassical_power_fraction"
                    ]
                    electron_on_measured = neoclassical_share(
                        twin, measured_kinetics, coefficients, ripple, power,
                        region, minor_radius, ion_fraction=ratio,
                        z_effective=measured_charge, species="electron",
                    )
                    print(
                        "      the electron channel across the region is "
                        + ", ".join(
                            f"{r:.2f} m: {v:.3f}"
                            for r, v in zip(region, electron_on_measured, strict=True)
                        )
                    )
                    low_band, high_band = measured_electron.band()
                    overlaps = bool(
                        float(np.min(electron_on_measured)) <= high_band
                        and float(np.max(electron_on_measured)) >= low_band
                    )
                    checks.append(
                        {
                            "quantity": "electron share over the core region, digitised profiles",
                            "modelled_low": float(np.min(electron_on_measured)),
                            "modelled_high": float(np.max(electron_on_measured)),
                            "measured": measured_electron.value,
                            "unit": measured_electron.unit,
                            "within_published_accuracy": overlaps,
                            "source": measured_electron.source,
                        }
                    )
                    print(
                        f"      {'ok  ' if overlaps else '??  '}spans "
                        f"{float(np.min(electron_on_measured)):.3f} to "
                        f"{float(np.max(electron_on_measured)):.3f} against the "
                        f"published {measured_electron.value:.2f}"
                    )
            # The electron channel on its own, where the source separates it. It is the
            # smaller of the two and is driven by a different gradient, so agreeing on the
            # sum while disagreeing on the split would not be agreement.
            if "core_electron_neoclassical_power_fraction" in programme.measured:
                electron_shares = [
                    neoclassical_share(
                        twin,
                        dataclasses.replace(
                            local.with_peaking(float(factor)), density_axis_m3=density
                        ),
                        coefficients, ripple, power, core_radius, minor_radius,
                        ion_fraction=ion_fraction, z_effective=measured_charge,
                        species="electron",
                    )
                    for factor in np.linspace(low_peak, high_peak, 5)
                ]
                measured_electron = programme.measured[
                    "core_electron_neoclassical_power_fraction"
                ]
                print(
                    f"      the electron channel alone spans {min(electron_shares):.4f} to "
                    f"{max(electron_shares):.4f} against a published "
                    f"{measured_electron.value:.2f}"
                )
                checks.append(
                    compare(
                        "neoclassical share of the core electron power",
                        float(electron_shares[-1]),
                        measured_electron,
                    )
                )
            # What peaking the published fraction would need, which is the statement the
            # scan supports rather than a residual at one shape.
            target = programme.measured["core_neoclassical_power_fraction"].value
            factors = np.array([f for f, _ in shares])
            values = np.array([v for _, v in shares])
            if float(values.min()) <= target <= float(values.max()):
                order = np.argsort(values)
                needed = float(np.interp(target, values[order], factors[order]))
                print(
                    f"      the published {target:.2f} is reproduced at a peaking of "
                    f"{needed:.2f}, against the top of the span the post-pellet phase "
                    f"sits at"
                )

        # The radiated fraction is a check on the impurity model rather than on the power
        # balance: it is what the carbon in the plasma takes out before the power reaches a
        # flux surface, so the fraction the model returns fixes the carbon content the
        # discharge must have carried.
        if "radiated_fraction" in programme.measured:
            power = programme.measured["heating_power_ecrh_w"].value
            fractions = []
            for carbon in (0.0, 0.01, 0.02, 0.04, 0.08):
                profiles = dataclasses.replace(
                    kinetics.HIGH_PERFORMANCE, carbon_fraction=carbon
                )
                solved = transport.solve(
                    twin.solve(twin.state(programme.configuration), SCAN),
                    profiles,
                    heating=transport.Heating(power_w=power),
                    model=transport.TransportModel(renormalisation=RENORMALISATION),
                )
                # Bolometry measures the whole plasma, and the confined region is not where
                # carbon radiates: it is stripped there. The layer between the separatrix
                # and the target passes through the cooling-rate peak, so it is carried too.
                boundary = boundary_radiation_w(twin, solved, carbon)
                radiated = (float(solved.radiated_power_w) + boundary) / power
                fractions.append((carbon, float(radiated)))
                print(
                    f"        carbon {carbon:.2f}: {radiated:.3f} of the "
                    f"{power / 1e6:.1f} MW radiated, "
                    f"{float(solved.radiated_power_w) / 1e6:.3f} MW from the confined "
                    f"region and {boundary / 1e6:.3f} MW from the layer"
                )
            # Bolometry fixes the carbon the layer can carry, and the effective charge
            # fixes what the core carries: the check is the first inverted against the
            # second, two measurements about one impurity, rather than the radiated
            # fraction at a content chosen for the table.
            implied = programmes.Measured(
                value=carbon_fraction,
                unit="1",
                relative_uncertainty=programmes.MACHINE_MEASUREMENTS[
                    "effective_charge"
                ].relative_uncertainty,
                source="the carbon fraction the measured effective charge implies, "
                + programmes.MACHINE_MEASUREMENTS["effective_charge"].source,
            )
            target = programme.measured["radiated_fraction"].value
            values = [v for _, v in fractions]
            carbons = [c for c, _ in fractions]
            if values[0] <= target <= values[-1]:
                needed = float(np.interp(target, values, carbons))
                checks.append(
                    compare(
                        "carbon the measured radiated fraction needs", needed, implied
                    )
                )
                inverted.append(
                    {
                        "quantity": "radiated fraction",
                        "heating_power_w": power,
                        "measured_radiated_fraction": target,
                        "carbon_fraction": needed,
                        "scan": [
                            {"carbon_fraction": c, "radiated_fraction": v}
                            for c, v in fractions
                        ],
                    }
                )

        records.append(
            {
                "identifier": programme.identifier,
                "campaign": programme.campaign,
                "epoch": programme.epoch,
                "configuration": programme.configuration,
                "source": programme.source,
                "checks": checks,
                "inverted": inverted,
            }
        )

    # The machine-level quantities, which are not tied to one programme.
    print()
    print("machine quantities, against the same source")
    traced = layer_geometry(twin)
    vessel, elements, frame = traced["vessel"], traced["elements"], traced["frame"]
    vacuum, separatrix = traced["vacuum"], traced["separatrix"]
    strikes, lengths_m, mask = traced["strikes"], traced["lengths_m"], traced["mask"]
    wanted_elements = traced["wanted_elements"]
    equilibrium = traced["equilibrium"]

    machine_checks = []

    # The one Thomson-derived profile quantity the publications state as a number rather
    # than draw as a figure: the density peaking. The archive holds the profiles; what is
    # public is the range they span, so that is what the reference profile is held against.
    peaking = programmes.MACHINE_MEASUREMENTS["pellet_density_peaking"]
    machine_checks.append(
        compare(
            "density peaking of the reference profile",
            kinetics.HIGH_PERFORMANCE.peaking(),
            peaking,
        )
    )

    # The Shafranov shift, against what each source actually states. The overview's
    # 1 to 2 cm is the shift of the VMEC equilibrium overlaid on the x-ray tomogram,
    # computed at beta = 1 per cent with the standard theoretical pressure profile
    # p = p0 (1 - s), whose peaking is 2, on a tomogram whose grid resolves 3 cm; so
    # that comparison runs the publication's own construction. The machine's measured
    # displacement is the Minerva reconstruction, order 1 cm at the 346 kJ of
    # XP_20171108.040, and the model is held to it at that solve's own beta. The shift
    # at the machine's steeper drawn profiles is reported beside them: no published
    # number constrains it, the tomography resolution being coarser than the difference.
    shift_target = programmes.MACHINE_MEASUREMENTS["shafranov_shift_at_one_percent_beta_m"]
    reconstructed = programmes.MACHINE_MEASUREMENTS["reconstructed_axis_shift_m"]

    family = twin.solve(
        twin.state("standard", scenario=Scenario(peak_pressure_pa=5.0e4)), SCAN
    )
    family_beta = float(family.wout.betatotal)
    family = twin.solve(
        twin.state(
            "standard",
            scenario=Scenario(peak_pressure_pa=5.0e4 * 0.0105 / max(family_beta, 1e-9)),
        ),
        SCAN,
        restart_from=family,
    )
    family_beta = float(family.wout.betatotal)
    family_analysed = diagnostics.analyse(family, reference)
    family_shift = float(family_analysed.axis_shift_in_boundary_m)
    print()
    print(
        f"the standard profile family at beta {100 * family_beta:.2f} per cent, the "
        f"publication's own construction, shifts "
        f"{1e3 * family_shift:.1f} mm in the boundary frame"
    )
    machine_checks.append(
        compare(
            "Shafranov shift at the publication's construction",
            family_shift * 0.01 / family_beta,
            shift_target,
        )
    )

    ordinary = digitised_temperatures("20180920.017")
    s_knots = np.linspace(0.0, 1.0, 41)
    if ordinary is not None:
        electron_t, ion_t = ordinary
        knots_p = (
            kinetics.ELEMENTARY_CHARGE
            * kinetics.HIGH_PERFORMANCE.density(s_knots)
            * (electron_t(s_knots) + ion_t(s_knots))
        )
        print(
            "the drawn-profile solves run on the gas-fuelled 20180920.017's "
            "temperatures over the flat-topped density"
        )
    else:
        s_knots, knots_p = kinetics.HIGH_PERFORMANCE.pressure_spline()
    finite_beta = twin.solve(
        twin.state(
            "standard", scenario=Scenario.from_pressure_spline(s_knots, knots_p)
        ),
        SCAN,
    )
    beta = float(finite_beta.wout.betatotal)
    first_beta = beta

    # The reconstruction's own operating point: the drawn profiles rescaled to the
    # 0.38 per cent the reconstructed 346 kJ corresponds to.
    at_reconstruction = twin.solve(
        twin.state(
            "standard",
            scenario=Scenario.from_pressure_spline(
                s_knots, knots_p * 0.0038 / max(first_beta, 1e-9)
            ),
        ),
        SCAN,
        restart_from=finite_beta,
    )
    low_analysed = diagnostics.analyse(at_reconstruction, reference)
    print(
        f"at the reconstruction's beta of "
        f"{100 * float(at_reconstruction.wout.betatotal):.2f} per cent the axis moves "
        f"{1e3 * float(low_analysed.axis_shift_m):.1f} mm in the laboratory and "
        f"{1e3 * float(low_analysed.axis_shift_in_boundary_m):.1f} mm in the boundary "
        f"frame, against the reconstructed order 1 cm"
    )
    machine_checks.append(
        compare(
            "axis shift against the Minerva reconstruction",
            float(low_analysed.axis_shift_in_boundary_m),
            reconstructed,
        )
    )

    # The machine's own steeper profiles at one per cent, which is the model's statement
    # rather than a check: the published band belongs to the peaking-2 construction.
    finite_beta = twin.solve(
        twin.state(
            "standard",
            scenario=Scenario.from_pressure_spline(
                s_knots, knots_p * 0.0105 / max(first_beta, 1e-9)
            ),
        ),
        SCAN,
        restart_from=finite_beta,
    )
    beta = float(finite_beta.wout.betatotal)
    analysed = diagnostics.analyse(finite_beta, reference)
    shift = float(analysed.axis_shift_in_boundary_m)
    peaking_of_pressure = float(analysed.beta_axis / max(analysed.beta_total, 1e-30))
    print(
        f"at a volume-averaged beta of {100 * beta:.2f} per cent and a pressure peaking "
        f"of {peaking_of_pressure:.2f} the drawn profiles move the axis "
        f"{1e3 * float(analysed.axis_shift_m):.1f} mm in the laboratory and "
        f"{1e3 * shift:.1f} mm against the boundary's own centre"
    )

    if mask.any():
        # The published figure is the connection length at the strike line, which is the
        # long-connection part of the fan rather than its median.
        lengths = lengths_m[mask]
        strike_line = float(np.percentile(lengths, STRIKE_LINE_PERCENTILE))
        # An island divertor puts a line in one of two regimes, so the fan is split at the
        # order of magnitude between them and each part compared with its own figure.
        short = lengths[lengths < 100.0]
        long = lengths[lengths >= 100.0]
        if short.size:
            machine_checks.append(
                compare(
                    "connection length, outer side of the island",
                    float(np.median(short)),
                    programmes.MACHINE_MEASUREMENTS["outer_island_connection_length_m"],
                )
            )
        if long.size:
            machine_checks.append(
                compare(
                    "connection length, inside the island",
                    float(np.median(long)),
                    programmes.MACHINE_MEASUREMENTS["island_connection_length_m"],
                )
            )
        # The machine rows run at the carbon the measured effective charge implies, so
        # the layer the radiator stands in is the one the balance already paid for.
        balance = transport.solve(
            twin.solve(twin.state("standard"), SCAN),
            dataclasses.replace(
                kinetics.HIGH_PERFORMANCE, carbon_fraction=carbon_fraction
            ),
            heating=transport.Heating(power_w=5.0e6),
            model=transport.TransportModel(renormalisation=RENORMALISATION),
        )
        # The layer width and the target temperature close on each other: a colder target
        # widens the layer, and a wider layer lowers the flux that cools it. Both are
        # solved together, over the wetted area the strikes themselves define, at the
        # incidence the traced field makes with the target contours.
        chi_edge = float(balance.chi_m2_s[-1])
        upstream = float(balance.density_m3[-1])
        sines, measured = [], []
        sine_by_line = np.full(strikes.start_r.shape, np.nan)
        for index in wanted_elements:
            on_element = mask & (strikes.component == index)
            if not on_element.any():
                continue
            r = strikes.r[on_element]
            z = strikes.z[on_element]
            phi = strikes.phi[on_element]
            tangent_r, tangent_z = edge.contour_tangent(
                elements[index], r, z, phi
            )
            sines.append(
                edge.incidence_sine(
                    vacuum, r, phi, z, tangent_r, tangent_z
                )
            )
            measured.append(
                edge.surface_incidence_sine(
                    vacuum, r, phi, z,
                    walls.surface_frame(elements[index], r, z, phi),
                )
            )
            sine_by_line[on_element] = measured[-1]
        swept = float(np.median(np.concatenate(sines)))
        # The angle against each target's own surface, whose normal carries the toroidal
        # inclination its elements are built with. The design bound is what it is checked
        # against, not what it is set to.
        design = programmes.MACHINE_MEASUREMENTS["divertor_incidence_degrees"]
        incidence = float(np.median(np.concatenate(measured)))

        def area_of_width(width: float) -> float:
            weights = edge.layer_weights(strikes.start_r, separatrix, width)
            return edge.wetted_area(
                strikes, elements, frame, weights
            )["area_m2"]

        def length_of_width(width: float) -> float:
            weights = edge.layer_weights(strikes.start_r, separatrix, width)
            return edge.power_weighted_connection_length(lengths_m[mask], weights[mask])

        closed = edge.close_layer(
            upstream,
            5.0e6 - float(balance.radiated_power_w),
            strike_line,
            chi_edge,
            incidence,
            area_of_width,
            length_of_width=length_of_width,
        )
        # The layer width is fitted to infrared thermography and published in numbers, so
        # it is compared rather than only reported.
        print(
            f"      the layer closes at {closed['target_temperature_ev']:.2f} eV on the "
            f"target, over {closed['area_m2']:.3f} m2 wetted at "
            f"{np.degrees(np.arcsin(incidence)):.2f} degrees measured against the target "
            f"surfaces, inside the {design.value:.0f} degree design bound and against "
            f"{np.degrees(np.arcsin(swept)):.1f} off the swept contour, "
            f"{1e3 * closed['width_m']:.1f} mm wide"
        )
        # The published width and flux are read off infrared images of the target, so the
        # compared quantities are the same pair of one deposition profile: its integral
        # width and its peak, on the element that carries the peak.
        closed_weights = edge.layer_weights(
            strikes.start_r, separatrix, closed["width_m"]
        )
        profile = edge.target_profile(
            strikes, elements, frame, closed_weights,
            5.0e6 - float(balance.radiated_power_w),
            incidence_by_line=sine_by_line, deskew=True,
            offset_by_line=strikes.start_r - separatrix,
        )
        print(
            f"      the deposition profile peaks at "
            f"{profile['peak_heat_flux_w_m2'] / 1e6:.2f} MW/m2 over "
            f"{1e3 * profile['peak_integral_width_m']:.1f} mm on the "
            f"{profile['peak_element']}"
        )
        radiator = edge.target_radiator(
            profile, upstream, incidence, carbon_fraction,
            density_width_m=closed["width_m"] / np.sqrt(3.0),
            dilution_by_element=edge.strip_dilution(
                strikes, elements, frame, closed_weights
            ),
        )
        print(
            f"      the layer radiates {radiator['radiated_w'] / 1e6:.2f} MW along the "
            f"targets at the implied carbon of {carbon_fraction:.4f}, leaving a net "
            f"peak of {radiator['net_peak_heat_flux_w_m2'] / 1e6:.2f} MW/m2"
        )
        machine_checks.append(
            compare(
                "strike-line width",
                profile["peak_integral_width_m"],
                programmes.MACHINE_MEASUREMENTS["strike_line_width_m"],
            )
        )
        # The same solve at the perpendicular diffusivity a boundary code needs to
        # reproduce that width, rather than at the edge value of the core power balance.
        measured_chi = programmes.MACHINE_MEASUREMENTS[
            "sol_perpendicular_diffusivity_m2_s"
        ]
        at_measured_chi = edge.close_layer(
            upstream,
            5.0e6 - float(balance.radiated_power_w),
            strike_line,
            measured_chi.value,
            incidence,
            area_of_width,
            length_of_width=length_of_width,
        )
        print(
            f"      at the measured {measured_chi.value:.1f} m2/s rather than the core "
            f"balance's {chi_edge:.3f} the width is "
            f"{1e3 * at_measured_chi['width_m']:.1f} mm"
        )
        measured_weights = edge.layer_weights(
            strikes.start_r, separatrix, at_measured_chi["width_m"]
        )
        measured_profile = edge.target_profile(
            strikes, elements, frame, measured_weights,
            5.0e6 - float(balance.radiated_power_w),
            incidence_by_line=sine_by_line, deskew=True,
        )
        machine_checks.append(
            compare(
                "strike-line width at the measured diffusivity",
                measured_profile["peak_integral_width_m"],
                programmes.MACHINE_MEASUREMENTS["strike_line_width_m"],
            )
        )
        machine_checks.append(
            compare(
                "target heat flux",
                radiator["net_peak_heat_flux_w_m2"],
                programmes.MACHINE_MEASUREMENTS["divertor_measured_power_density_w_m2"],
            )
        )
        machine_checks.append(
            compare(
                "wetted area",
                closed["area_m2"],
                programmes.MACHINE_MEASUREMENTS["strike_line_area_m2"],
            )
        )

    write_record(
        DISCHARGE_OUT,
        {
            "renormalisation": RENORMALISATION,
            "programmes": records,
            "machine": machine_checks,
        },
        geometry=twin.geometry,
    )
    total = sum(len(record["checks"]) for record in records) + len(machine_checks)
    agreed = sum(
        1
        for record in records
        for check in record["checks"]
        if check["within_published_accuracy"]
    ) + sum(1 for check in machine_checks if check["within_published_accuracy"])
    print()
    print(f"{agreed} of {total} within the accuracy the sources support")
    return 0
