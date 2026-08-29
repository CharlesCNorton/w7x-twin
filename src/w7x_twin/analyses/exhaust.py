"""Divertor load, incidence, strike attribution, strike-line migration and recycling."""

from __future__ import annotations

import os

import dataclasses
import time
from pathlib import Path

import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.analyses._common import arg, layer_constants, write_record
from w7x_twin.hardware import machine, walls
from w7x_twin.magnetics import field, fieldlines, plasma_response
from w7x_twin.magnetics.field import VacuumField
from w7x_twin.mhd import diagnostics
from w7x_twin.mhd.equilibrium import SCAN, Twin
from w7x_twin.plasma import current as plasma_current
from w7x_twin.plasma import edge, kinetics, neoclassical, transport
from w7x_twin.records import programmes

RENORMALISATION = transport.PUBLISHED_ISS04_ENHANCEMENT
#: The fan across the scrape-off layer, and how far it is followed.
LAYER = (0.985, 1.40)
#: Toroidal launch planes across one field period: the strike band is inclined along
#: each target, and one plane samples a single comb of it.
LAUNCH_PLANES = 5
TURNS = 200
#: Percentile of the connection-length distribution taken as the strike-line value; the
#: power is carried by the long field lines, not by the median of a fan.
STRIKE_LINE_PERCENTILE = 90


# -- exhaust -----------------------------------------------------------------------

# Target heat load from measured inputs: crossing power over the wetted area, at the
# traced incidence, carried down the tube by the two-point model.
#
#     python -m w7x_twin exhaust [heating power in MW] [carbon fraction]

EXHAUST_OUT = Path("results/exhaust/heat_flux.json")
#: Surface results are reported at. This script reads the radial scan only, so it
#: never stands the single-surface table in and never needs that table's surface.
REFERENCE_SURFACE = neoclassical.REFERENCE_SURFACE

EXHAUST_LINES = 240


#: Upstream densities the approach to detachment is scanned over, in m^-3.
UPSTREAM_DENSITIES = (0.5e19, 1.0e19, 2.0e19, 4.0e19, 8.0e19)
#: Target temperature below which a divertor is conventionally called detached.
DETACHMENT_EV = 5.0


load_coefficients = _common.radial_coefficients


def fanned_strikes(vacuum, wout, lines: int, vessel, elements):
    """Strikes of one fan per launch plane, each anchored to its own axis and boundary
    cut and remapped onto the first plane's separatrix; returns them with that radius."""
    period = 2.0 * np.pi / walls.NUM_FIELD_PERIODS_DEFAULT
    separatrix = None
    per_plane = []
    for index in range(LAUNCH_PLANES):
        phi = index * period / LAUNCH_PLANES
        starts, _, axis_z, outboard = fieldlines.fan_starts(
            vacuum, wout, LAYER, lines, plane_phi=phi
        )
        if separatrix is None:
            separatrix = outboard
        section, _ = fieldlines.trace(
            vacuum, starts, np.full(starts.shape, axis_z), turns=TURNS,
            plane_phi=phi, vessel=vessel, components=elements,
        )
        per_plane.append(
            dataclasses.replace(
                section.strikes, start_r=separatrix + (starts - outboard)
            )
        )
    return fieldlines.Strikes.concatenate(per_plane), separatrix


def run_exhaust() -> int:
    power = arg(1, float, 5.0) * 1e6
    carbon = arg(2, float, 0.0)

    twin = _common.twin()
    vessel = _common.vessel()
    elements = _common.components()
    frame = walls.target_arc_frame(elements)
    print(f"{twin.geometry}")
    print(f"{power / 1e6:.0f} MW, carbon fraction {carbon:g}")

    profiles = dataclasses.replace(kinetics.HIGH_PERFORMANCE, carbon_fraction=carbon)
    equilibrium = twin.solve_profiles("standard", profiles)
    balance = transport.solve(
        equilibrium, profiles, heating=transport.Heating(power_w=power),
        model=transport.TransportModel(renormalisation=RENORMALISATION),
    )
    crossing = power - balance.radiated_power_w
    print(
        f"power crossing the separatrix {crossing / 1e6:.3f} MW of {power / 1e6:.1f} MW, "
        f"{100 * balance.radiated_fraction:.1f} % radiated"
    )

    # The traced scrape-off layer: where it lands, how far it runs, at what angle. The
    # fan is launched at several planes across one field period, each anchored to its
    # own axis and boundary cut and remapped onto one reference separatrix, so the
    # deposition samples the strike band's toroidal extent rather than one cut of it.
    # The anchoring boundary is the vacuum equilibrium's, since the traced field is the
    # vacuum one: the finite-beta boundary sits a Shafranov shift outboard of the field
    # the lines actually follow, and a fan anchored there misses the layer.
    vacuum = VacuumField(twin.response, twin.state("standard").currents)
    vacuum_equilibrium = twin.solve(twin.state("standard"), SCAN)
    strikes, separatrix = fanned_strikes(
        vacuum, vacuum_equilibrium.wout, EXHAUST_LINES, vessel, elements
    )
    wanted = [index for index, e in enumerate(elements) if e.name in frame]
    mask = strikes.struck & np.isin(strikes.component, wanted)
    if not mask.any():
        raise SystemExit("no line reached a divertor target")
    # The power is carried by the long field lines, not by the median of a fan that
    # includes lines terminating within a few metres, so the strike-line connection
    # length is the one the parallel conduction and the layer width are set by.
    connection = float(
        np.percentile(strikes.connection_length_m[mask], STRIKE_LINE_PERCENTILE)
    )
    print(
        f"connection length: median {np.median(strikes.connection_length_m[mask]):.1f} m, "
        f"at the strike line {connection:.1f} m"
    )

    # The angle a swept poloidal contour would give, which is what the poloidal tangent
    # alone measures. It is reported for comparison and is not what the load is formed at.
    sines = []
    for index in wanted:
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        tangent_r, tangent_z = edge.contour_tangent(
            elements[index], strikes.r[on_element], strikes.z[on_element],
            strikes.phi[on_element],
        )
        sines.append(
            edge.incidence_sine(
                vacuum, strikes.r[on_element], strikes.phi[on_element],
                strikes.z[on_element], tangent_r, tangent_z,
            )
        )
    swept = float(np.median(np.concatenate(sines)))
    # The angle against each component's own surface, whose normal is the cross product of
    # both surface derivatives. The targets are cut at the width of one element and their
    # contours move between cuts, so that normal carries the inclination the elements are
    # built with, and the incidence varies along the strike line with the field's own pitch.
    design = programmes.MACHINE_MEASUREMENTS["divertor_incidence_degrees"]
    measured = []
    per_target = {}
    sine_by_line = np.full(strikes.start_r.shape, np.nan)
    for index in wanted:
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        r = strikes.r[on_element]
        z = strikes.z[on_element]
        phi = strikes.phi[on_element]
        frame_of = walls.surface_frame(elements[index], r, z, phi)
        sine = edge.surface_incidence_sine(vacuum, r, phi, z, frame_of)
        per_target[elements[index].name] = float(
            np.degrees(np.arcsin(np.median(sine)))
        )
        measured.append(sine)
        sine_by_line[on_element] = sine
    spread = np.concatenate(measured)
    incidence = float(np.median(spread))
    print(
        f"incidence {np.degrees(np.arcsin(incidence)):.2f} degrees at the median, "
        f"{np.degrees(np.arcsin(spread.min())):.2f} to "
        f"{np.degrees(np.arcsin(spread.max())):.2f} across the strike line, measured "
        f"against each target's own surface"
    )
    print(
        "  per target: "
        + ", ".join(f"{name} {value:.2f} deg" for name, value in sorted(per_target.items()))
    )
    print(f"  design bound up to {design.value:.0f} degrees: {design.source}")
    print(f"  the swept contour alone would give {np.degrees(np.arcsin(swept)):.1f} degrees")

    # The radial width the power occupies, and with it the weight each traced line carries.
    # It sets the wetted area through the sound speed at the target, and the target
    # temperature follows from the flux that area implies, so the two are iterated.
    chi_edge = float(balance.chi_m2_s[-1])
    upstream_density = float(balance.density_m3[-1])

    def area_of_width(width: float) -> float:
        weights = edge.layer_weights(strikes.start_r, separatrix, width)
        return edge.wetted_area(strikes, elements, frame, weights)["area_m2"]

    def length_of_width(width: float) -> float:
        weights = edge.layer_weights(strikes.start_r, separatrix, width)
        return edge.power_weighted_connection_length(
            strikes.connection_length_m[mask], weights[mask]
        )

    closed = edge.close_layer(
        upstream_density, crossing, connection, chi_edge, incidence, area_of_width,
        length_of_width=length_of_width,
    )
    connection = float(closed["connection_length_m"])
    print(
        f"the power-weighted connection length closes at {connection:.1f} m, against "
        f"{np.percentile(strikes.connection_length_m[mask], STRIKE_LINE_PERCENTILE):.1f} m "
        f"at the {STRIKE_LINE_PERCENTILE}th percentile and "
        f"{np.median(strikes.connection_length_m[mask]):.1f} m at the median"
    )
    if not closed["converged"]:
        raise SystemExit(
            f"the layer width and the target temperature did not close: "
            f"{closed['target_temperature_ev']:.3f} eV after {closed['steps']} steps"
        )
    decay = closed["width_m"]
    weights = edge.layer_weights(strikes.start_r, separatrix, decay)
    geometry = edge.wetted_area(strikes, elements, frame, weights)

    target_flux = crossing / geometry["area_m2"]
    parallel_flux = target_flux / incidence
    print(
        f"perpendicular diffusivity at the edge {chi_edge:.3f} m2/s, power decay "
        f"length {1e3 * decay:.1f} mm"
    )
    print(
        f"{geometry['lines']} lines on the targets, wetted area "
        f"{geometry['area_m2']:.3f} m2, median connection length {connection:.1f} m"
    )

    # The deposition resolved along each target's arc, so the footprint width and the
    # peak flux come from one profile rather than a width from the footprint and a flux
    # from the average over every element.
    profile = edge.target_profile(
        strikes, elements, frame, weights, crossing, incidence_by_line=sine_by_line,
        deskew=True,
    )
    for name, row in sorted(profile["per_element"].items()):
        print(
            f"  {name:38s} peak {row['peak_heat_flux_w_m2'] / 1e6:6.2f} MW/m2 over "
            f"{1e3 * row['integral_width_m']:5.1f} mm"
        )
    print(
        f"profile peak {profile['peak_heat_flux_w_m2'] / 1e6:.2f} MW/m2 at "
        f"{1e3 * profile['peak_integral_width_m']:.1f} mm on the "
        f"{profile['peak_element']}"
    )

    # The same traced fan at imposed upstream decays bracketing the closure's own, so
    # the width and peak the cameras report are read directly against the layer width
    # and the pair the measurement wants can be located on one curve.
    width_scan = []
    for upstream in (0.005, 0.010, 0.015, decay, 0.030, 0.045, 0.060):
        scan_weights = edge.layer_weights(strikes.start_r, separatrix, upstream)
        scanned = edge.target_profile(
            strikes, elements, frame, scan_weights, crossing,
            incidence_by_line=sine_by_line, deskew=True,
        )
        width_scan.append(
            {
                "upstream_width_m": float(upstream),
                "peak_heat_flux_w_m2": scanned["peak_heat_flux_w_m2"],
                "peak_integral_width_m": scanned["peak_integral_width_m"],
                "peak_element": scanned["peak_element"],
            }
        )
    print("  upstream decay against the deposition it produces:")
    for row in width_scan:
        print(
            f"    {1e3 * row['upstream_width_m']:5.1f} mm -> "
            f"{row['peak_heat_flux_w_m2'] / 1e6:6.2f} MW/m2 over "
            f"{1e3 * row['peak_integral_width_m']:5.1f} mm"
        )
    # The exhaust chain carried as intervals: the traced mapping is fixed, so the
    # stated input uncertainties are sampled through the closure and the deskewed
    # profile, which are cheap per sample. The inputs sampled are the ones the sources
    # state: the heating power to five per cent, the perpendicular diffusivity to its
    # published fifty, the effective charge to its twenty through the carbon.
    rng = np.random.default_rng(20260727)
    chi_measured = programmes.MACHINE_MEASUREMENTS["sol_perpendicular_diffusivity_m2_s"]
    charge_spread = programmes.MACHINE_MEASUREMENTS["effective_charge"].relative_uncertainty
    samples = {"width_m": [], "wetted_m2": [], "peak_w_m2": [], "net_peak_w_m2": []}
    for _ in range(64):
        power_k = power * (1.0 + 0.05 * rng.standard_normal())
        chi_k = max(
            chi_edge * (1.0 + chi_measured.relative_uncertainty * rng.standard_normal()),
            0.05,
        )
        carbon_k = max(carbon * (1.0 + charge_spread * rng.standard_normal()), 0.0)
        crossing_k = max(power_k - float(balance.radiated_power_w), 1.0e5)
        try:
            closed_k = edge.close_layer(
                upstream_density, crossing_k, connection, chi_k, incidence,
                area_of_width, length_of_width=length_of_width,
            )
        except ValueError:
            continue
        weights_k = edge.layer_weights(strikes.start_r, separatrix, closed_k["width_m"])
        profile_k = edge.target_profile(
            strikes, elements, frame, weights_k, crossing_k,
            incidence_by_line=sine_by_line, deskew=True,
            offset_by_line=strikes.start_r - separatrix,
        )
        radiator_k = edge.target_radiator(
            profile_k, upstream_density, incidence, carbon_k,
            density_width_m=closed_k["width_m"] / np.sqrt(3.0),
            dilution_by_element=edge.strip_dilution(
                strikes, elements, frame, weights_k
            ),
        )
        samples["width_m"].append(profile_k["peak_integral_width_m"])
        samples["wetted_m2"].append(
            edge.wetted_area(strikes, elements, frame, weights_k)["area_m2"]
        )
        samples["peak_w_m2"].append(profile_k["peak_heat_flux_w_m2"])
        samples["net_peak_w_m2"].append(radiator_k["net_peak_heat_flux_w_m2"])
    intervals = {
        key: {
            "median": float(np.median(values)),
            "percentile_5": float(np.percentile(values, 5)),
            "percentile_95": float(np.percentile(values, 95)),
            "samples": len(values),
        }
        for key, values in samples.items()
        if values
    }
    print("  the chain sampled through the stated input uncertainties:")
    presentation = {
        "width_m": (1e3, "mm"),
        "wetted_m2": (1.0, "m2"),
        "peak_w_m2": (1e-6, "MW/m2"),
        "net_peak_w_m2": (1e-6, "MW/m2"),
    }
    for key, row in intervals.items():
        scale, unit = presentation[key]
        print(
            f"    {key:14s} {scale * row['median']:8.2f} {unit}, 5th to 95th "
            f"{scale * row['percentile_5']:.2f} to {scale * row['percentile_95']:.2f} "
            f"over {row['samples']} samples"
        )

    print(
        f"parallel flux at {np.degrees(np.arcsin(incidence)):.2f} degrees, "
        f"sin alpha {incidence:.4f}"
    )
    print(
        f"target heat flux {target_flux / 1e6:.3f} MW/m2, parallel "
        f"{parallel_flux / 1e6:.1f} MW/m2"
    )
    design_flux = programmes.MACHINE_MEASUREMENTS["divertor_local_power_density_w_m2"]
    low, high = design_flux.band()
    print(
        f"  against a design local power density of {design_flux.value / 1e6:.0f} MW/m2, "
        + (
            "inside the accuracy that carries"
            if low <= target_flux <= high
            else f"outside the {low / 1e6:.0f} to {high / 1e6:.0f} MW/m2 that implies"
        )
    )

    print()
    layout = _common.Table(
        ("n_u [m^-3]", "11.2e"), ("T_u [eV]", "9.1f"), ("T_t [eV]", "9.2f"),
        ("n_t [m^-3]", "11.2e"), ("q_target", ">10s"), ("regime", ">18s"),
    )
    layout.begin()
    rows = []
    for upstream in UPSTREAM_DENSITIES:
        solution = edge.solve_two_point(
            upstream, parallel_flux, connection
        )
        regime = (
            "detached" if solution.target_temperature_ev < DETACHMENT_EV
            else ("conduction limited" if solution.conduction_limited else "sheath limited")
        )
        rows.append(
            {
                "upstream_density_m3": upstream,
                "upstream_temperature_ev": solution.upstream_temperature_ev,
                "target_temperature_ev": solution.target_temperature_ev,
                "target_density_m3": solution.target_density_m3,
                "target_heat_flux_w_m2": target_flux,
                "regime": regime,
            }
        )
        layout.row(
            upstream, solution.upstream_temperature_ev,
            solution.target_temperature_ev, solution.target_density_m3,
            f"{target_flux / 1e6:9.3f}M", regime,
        )

    # The design angle is an upper bound, not a single value, and the strike line does not
    # meet every tile at it. The solve is repeated across the range so what the angle costs
    # is bounded rather than assumed. It leaves the target flux untouched and enters the
    # two-point model only through the parallel flux.
    print()
    print("the same solve against the incidence angle, at the reference upstream density")
    layout = _common.Table(
        ("alpha [deg]", "12.2f"), ("sin alpha", "10.4f"), ("q_par", ">10s"),
        ("width [mm]", "9.1f"), ("T_u [eV]", "9.1f"), ("T_t [eV]", "9.2f"),
        ("regime", ">18s"),
    )
    layout.begin()
    incidence_rows = []
    for degrees in (1.0, 2.0, 3.0, 5.0, 10.0, np.degrees(np.arcsin(swept))):
        sine = float(np.sin(np.radians(degrees)))
        # The layer width and the target temperature close on each other at every angle,
        # since a colder target widens the layer and a wider layer lowers the flux.
        angle_closed = edge.close_layer(
            upstream_density, crossing, connection, chi_edge, sine, area_of_width
        )
        width = angle_closed["width_m"]
        area = angle_closed["area_m2"]
        solution = angle_closed["solution"]
        regime = (
            "detached" if solution.target_temperature_ev < DETACHMENT_EV
            else ("conduction limited" if solution.conduction_limited else "sheath limited")
        )
        incidence_rows.append(
            {
                "incidence_degrees": degrees,
                "wetted_area_m2": float(area),
                "power_decay_length_m": float(width),
                "parallel_heat_flux_w_m2": float(crossing / area / sine),
                "upstream_temperature_ev": solution.upstream_temperature_ev,
                "target_temperature_ev": solution.target_temperature_ev,
                "regime": regime,
            }
        )
        layout.row(
            degrees, sine, f"{crossing / area / sine / 1e6:9.1f}M", 1e3 * width,
            solution.upstream_temperature_ev, solution.target_temperature_ev, regime,
        )

    # Detachment: the volumetric power and momentum loss that brings the target below the
    # threshold, and the recycled neutral flux and pressure at each upstream density.
    print()
    print("the volumetric loss detachment needs, and what recycles from the target")
    layout = _common.Table(
        ("n_u [m^-3]", "11.2e"), ("T_t attached", "13.2f"), ("f_pow", "8.3f"),
        ("T_t at f", "9.2f"), ("Gamma_t", "11.2e"), ("n_0 bound", "11.2e"),
        ("p_0 bound", "10.3f"),
    )
    layout.begin()
    detachment_rows = []
    for upstream in UPSTREAM_DENSITIES:
        attached = edge.solve_two_point(
            upstream, parallel_flux, connection
        )
        needed = edge.loss_for_detachment(
            upstream, parallel_flux, connection, DETACHMENT_EV
        )
        detached = (
            edge.solve_two_point_extended(
                upstream, parallel_flux, connection, power_loss=needed
            )
            if np.isfinite(needed)
            else attached
        )
        recycling = edge.recycling_balance(detached)
        detachment_rows.append(
            {
                "upstream_density_m3": upstream,
                "attached_target_ev": attached.target_temperature_ev,
                "power_loss_for_detachment": needed,
                "detached_target_ev": detached.target_temperature_ev,
                "target_flux_m2_s": recycling.target_flux_m2_s,
                "equilibrium_density_bound_m3": recycling.equilibrium_density_m3,
                "equilibrium_pressure_bound_pa": recycling.equilibrium_pressure_pa,
            }
        )
        layout.row(
            upstream, attached.target_temperature_ev, needed,
            detached.target_temperature_ev, recycling.target_flux_m2_s,
            recycling.equilibrium_density_m3, recycling.equilibrium_pressure_pa,
        )

    write_record(
        EXHAUST_OUT,
        {
            "heating_power_w": power,
            "carbon_fraction": carbon,
            "radiated_power_w": float(balance.radiated_power_w),
            "power_crossing_separatrix_w": float(crossing),
            "wetted": geometry,
            "power_decay_length_m": float(decay),
            "edge_diffusivity_m2_s": chi_edge,
            "connection_length_m": connection,
            "incidence_sine": incidence,
            "incidence_degrees": float(np.degrees(np.arcsin(incidence))),
            "design_bound_degrees": design.value,
            "swept_contour_incidence_sine": swept,
            "target_incidence_degrees": per_target,
            "incidence_range_degrees": [
                float(np.degrees(np.arcsin(spread.min()))),
                float(np.degrees(np.arcsin(spread.max()))),
            ],
            "target_heat_flux_w_m2": float(target_flux),
            "parallel_heat_flux_w_m2": float(parallel_flux),
            "target_profile": profile,
            "width_scan": width_scan,
            "intervals": intervals,
            "detachment_threshold_ev": DETACHMENT_EV,
            "upstream_scan": rows,
            "incidence_scan": incidence_rows,
            "detachment_scan": detachment_rows,
        },
        geometry=twin.geometry,
    )
    return 0


# -- incidence ---------------------------------------------------------------------

# Field incidence per target element against the surface's own toroidal inclination.
#
#     python -m w7x_twin incidence [heating MW]

INCIDENCE_OUT = Path("results/exhaust/target_incidence.json")
INCIDENCE_LINES = 120
#: Toroidal extent of one target element. Published for the W7-X high heat flux divertor
#: and reproduced by the cut spacing of the target component files.
ELEMENT_DEGREES = 0.5


def summarise(sine: np.ndarray) -> dict:
    """Incidence angles in degrees from an array of sines."""
    degrees = np.degrees(np.arcsin(np.clip(sine, 0.0, 1.0)))
    return {
        "count": int(degrees.size),
        "median": float(np.median(degrees)),
        "mean": float(np.mean(degrees)),
        "min": float(degrees.min()),
        "max": float(degrees.max()),
        "p5": float(np.percentile(degrees, 5)),
        "p95": float(np.percentile(degrees, 95)),
    }


def run_incidence() -> int:
    power = 1e6 * arg(1, float, 5.0)

    twin = _common.twin()
    vessel = _common.vessel()
    elements = _common.components()
    frame = walls.target_arc_frame(elements)

    profiles = dataclasses.replace(kinetics.HIGH_PERFORMANCE, carbon_fraction=0.0)
    finite_beta = twin.solve_profiles("standard", profiles)
    balance = transport.solve(
        finite_beta, profiles, heating=transport.Heating(power_w=power),
        model=transport.TransportModel(renormalisation=RENORMALISATION),
    )
    crossing = power - balance.radiated_power_w

    # The launch anchor is the vacuum boundary, since the traced field is the vacuum
    # one, and the fan is launched at several planes across one field period so the
    # per-element statistics see the band and not one comb of it.
    vacuum = VacuumField(twin.response, twin.state("standard").currents)
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    strikes, separatrix = fanned_strikes(
        vacuum, equilibrium.wout, INCIDENCE_LINES, vessel, elements
    )
    wanted = [index for index, e in enumerate(elements) if e.name in frame]
    mask = strikes.struck & np.isin(strikes.component, wanted)
    if not mask.any():
        raise SystemExit("no line reached a divertor target")
    print(f"{twin.geometry}")
    print(
        f"{int(mask.sum())} of {LAUNCH_PLANES * INCIDENCE_LINES} lines on a divertor target, "
        f"{crossing / 1e6:.3f} MW crossing the separatrix"
    )

    design = programmes.MACHINE_MEASUREMENTS["divertor_incidence_degrees"]
    swept_all, surface_all, per_element = [], [], []
    layout = _common.Table(
        ("target", "38s"), ("lines", "5d"), ("swept", ">8s"), ("surface", ">9s"),
        ("p5", ">7s"), ("p95", ">7s"), ("elements", "9d"),
    )
    print()
    layout.begin()

    for index in wanted:
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        r = strikes.r[on_element]
        z = strikes.z[on_element]
        phi = strikes.phi[on_element]

        # The swept-contour angle, which is what the poloidal tangent alone measures.
        tangent_r, tangent_z = edge.contour_tangent(
            elements[index], r, z, phi
        )
        swept = edge.incidence_sine(
            vacuum, r, phi, z, tangent_r, tangent_z
        )

        # The same strikes against the component's own surface.
        surface_frame = walls.surface_frame(elements[index], r, z, phi)
        surface = edge.surface_incidence_sine(
            vacuum, r, phi, z, surface_frame
        )

        swept_all.append(swept)
        surface_all.append(surface)
        struck_elements = np.unique(surface_frame["element"])
        summary = summarise(surface)
        layout.row(
            elements[index].name, int(on_element.sum()),
            f"{np.degrees(np.arcsin(np.median(swept))):7.2f}°",
            f"{summary['median']:8.2f}°", f"{summary['p5']:6.2f}°",
            f"{summary['p95']:6.2f}°", len(struck_elements),
        )

        for cut in struck_elements:
            here = surface_frame["element"] == cut
            degrees = np.degrees(np.arcsin(np.clip(surface[here], 0.0, 1.0)))
            # Where on the element each strike sits, so the angle is resolved across the
            # tile and not only from one element to the next.
            across = surface_frame["across"][here]
            along = surface_frame["along"][here]
            per_element.append(
                {
                    "target": elements[index].name,
                    "element": int(cut),
                    "phi_degrees": float(np.degrees(elements[index].phi[cut])),
                    "lines": int(here.sum()),
                    "incidence_degrees": float(np.median(degrees)),
                    "incidence_spread_degrees": float(degrees.max() - degrees.min()),
                    "toroidal_position": [float(v) for v in across],
                    "poloidal_arc_m": [float(v) for v in along],
                    "incidence_at_each_strike_degrees": [float(v) for v in degrees],
                }
            )

    swept_sine = np.concatenate(swept_all)
    surface_sine = np.concatenate(surface_all)
    swept_summary = summarise(swept_sine)
    surface_summary = summarise(surface_sine)

    print()
    print(
        f"swept contour   {swept_summary['median']:6.2f}° median, "
        f"{swept_summary['min']:.2f} to {swept_summary['max']:.2f}"
    )
    print(
        f"own surface     {surface_summary['median']:6.2f}° median, "
        f"{surface_summary['min']:.2f} to {surface_summary['max']:.2f}, "
        f"against a design bound of up to {design.value:.0f}°"
    )
    within = float(np.mean(np.degrees(np.arcsin(surface_sine)) <= design.value))
    print(f"{100 * within:.0f} % of strikes arrive inside that bound")

    # How much of the spread is variation across one element rather than between elements.
    # An element half a degree wide is not flat to the field, so the two are separable and
    # the answer says whether the target has to be resolved below the element to be right.
    resolved = [row for row in per_element if row["lines"] > 1]
    if resolved:
        within_element = float(np.median([row["incidence_spread_degrees"] for row in resolved]))
        between = float(
            np.std([row["incidence_degrees"] for row in per_element])
        )
        print(
            f"across one element the angle moves {within_element:.2f}° in the median, "
            f"against {between:.2f}° of scatter between elements, over "
            f"{len(resolved)} elements carrying more than one strike"
        )

    # The layer solved at the measured incidence rather than at the design angle.
    connection = float(np.percentile(strikes.connection_length_m[mask], STRIKE_LINE_PERCENTILE))
    chi_edge = float(balance.chi_m2_s[-1])
    upstream_density = float(balance.density_m3[-1])

    def area_of_width(width: float) -> float:
        weights = edge.layer_weights(strikes.start_r, separatrix, width)
        return edge.wetted_area(strikes, elements, frame, weights)["area_m2"]

    print()
    layout = _common.Table(
        ("incidence from", "16s"), ("angle", ">7s"), ("q_par", "10.1f"),
        ("width", ">10s"), ("area", "6.3f"), ("T_t", ">10s"), ("q_target", "8.2f"),
    )
    layout.begin()
    closed = {}
    for label, sine in (
        ("swept contour", float(np.median(swept_sine))),
        ("own surface", float(np.median(surface_sine))),
        ("design bound", float(np.sin(np.radians(design.value)))),
    ):
        solution = edge.close_layer(
            upstream_density, crossing, connection, chi_edge, sine, area_of_width
        )
        closed[label] = solution
        parallel = crossing / solution["area_m2"] / sine
        layout.row(
            label, f"{np.degrees(np.arcsin(sine)):6.2f}°", parallel / 1e6,
            f"{1e3 * solution['width_m']:6.1f} mm", solution["area_m2"],
            f"{solution['target_temperature_ev']:7.2f} eV", parallel * sine / 1e6,
        )

    write_record(
        INCIDENCE_OUT,
        {
            "heating_w": power,
            "power_crossing_separatrix_w": crossing,
            "connection_length_m": connection,
            "element_degrees": ELEMENT_DEGREES,
            "design_bound_degrees": design.value,
            "swept_contour": swept_summary,
            "own_surface": surface_summary,
            "fraction_within_design_bound": within,
            "per_element": per_element,
            "within_element_spread_degrees": (
                float(np.median([r["incidence_spread_degrees"] for r in resolved]))
                if resolved else float("nan")
            ),
            "between_element_scatter_degrees": float(
                np.std([r["incidence_degrees"] for r in per_element])
            ),
            "layer": {
                label: {
                    "width_m": value["width_m"],
                    "area_m2": value["area_m2"],
                    "target_temperature_ev": value["target_temperature_ev"],
                    "upstream_temperature_ev": value["solution"].upstream_temperature_ev,
                    "converged": value["converged"],
                }
                for label, value in closed.items()
            },
        },
        geometry=twin.geometry,
    )
    return 0


# -- strikes -----------------------------------------------------------------------

# Strikes resolved to the ten divertor units as a five-fold symmetry test.
#
#     python -m w7x_twin strikes [trim_current_a]

NUM_FIELD_PERIODS = 5
LINES_PER_MODULE = 12


def run_strikes() -> int:
    trim_current = arg(1, float, 0.0)

    if trim_current:
        twin = Twin(coils_file="coils.w7x_full", verbose=False)
        grid = field.full_torus_grid(twin.coils.grid)
        twin.response = field.build_response_table(
            twin.coils, grid=grid, cache_dir=twin.cache_dir, verbose=False
        )
        state = twin.with_currents(twin.state("standard"), trim_a1=trim_current)
        currents = state.currents
        label = f"standard, trim_a1 at {trim_current:.0f} A/turn"
    else:
        twin = _common.twin()
        currents = machine.get("standard").as_extcur()
        label = "standard, trim circuits unpowered"

    vacuum = VacuumField(twin.response, currents)
    vessel = _common.vessel()
    elements = _common.components()
    print(f"{label}: {len(elements)} plasma-facing components loaded")

    equilibrium = twin.solve(twin.state("standard"), SCAN)
    r_axis, z_axis = fieldlines.find_axis(vacuum)
    r_lcfs, _ = diagnostics.boundary_cut(equilibrium.wout, 0.0)
    half_width = r_lcfs.max() - r_axis

    # The same fan at the equivalent point of every module. Every module plane is an
    # image of the bean plane under the five-fold rotation, so the axis is solved once
    # and the fan repeated about it.
    fan = r_axis + np.linspace(*LAYER, LINES_PER_MODULE) * half_width
    period = 2.0 * np.pi / NUM_FIELD_PERIODS
    planes = [module * period for module in range(NUM_FIELD_PERIODS)]

    started = time.monotonic()
    tallies: dict[tuple[str, int, str], int] = {}
    element_lengths: dict[str, list[float]] = {}
    struck = considered = 0
    for phi in planes:
        section, _ = fieldlines.trace(
            vacuum,
            fan,
            np.full(fan.shape, z_axis),
            turns=TURNS,
            plane_phi=phi,
            vessel=vessel,
            components=elements,
        )
        strikes = section.strikes
        considered += len(fan)
        struck += int(np.count_nonzero(strikes.struck))
        for key, count in strikes.unit_tally(NUM_FIELD_PERIODS).items():
            tallies[key] = tallies.get(key, 0) + count
        for line in np.flatnonzero(strikes.struck):
            index = int(strikes.component[line])
            name = (
                walls.base_name(strikes.component_names[index])
                if index >= 0
                else "vessel wall"
            )
            element_lengths.setdefault(name, []).append(
                float(strikes.connection_length_m[line])
            )
    print(
        f"traced {considered} lines x {TURNS} turns in "
        f"{time.monotonic() - started:.0f} s; {struck} terminated"
    )

    print()
    layout = _common.Table(
        ("element", "34s"), ("lines", "6d"), ("median L_c", ">12s"), ("longest", ">11s"),
    )
    layout.begin()
    for name, lengths in sorted(
        element_lengths.items(), key=lambda item: -len(item[1])
    ):
        layout.row(
            name, len(lengths), f"{np.median(lengths):10.1f} m", f"{max(lengths):9.1f} m"
        )

    print()
    print("per divertor unit, module x upper or lower")
    elements_seen = sorted({name for name, _, _ in tallies})
    layout = _common.Table(
        ("element", "34s"),
        *((f"{m}{u}", "5d") for m in range(1, NUM_FIELD_PERIODS + 1) for u in "ul"),
    )
    layout.begin()
    for name in elements_seen:
        layout.row(
            name,
            *(
                tallies.get((name, module, unit), 0)
                for module in range(1, NUM_FIELD_PERIODS + 1)
                for unit in ("upper", "lower")
            ),
        )

    # The symmetry test: module-summed counts, which must agree when the field is
    # five-fold periodic and need not when a trim coil breaks it.
    per_module = [
        sum(count for (_, m, _), count in tallies.items() if m == module)
        for module in range(1, NUM_FIELD_PERIODS + 1)
    ]
    print()
    print(f"strikes per module: {per_module}")
    spread = max(per_module) - min(per_module)
    print(
        f"spread {spread} of {sum(per_module)}; "
        + ("periodicity intact" if spread == 0 else "periodicity broken")
    )
    return 0


# -- migration ---------------------------------------------------------------------

# Bootstrap current to edge transform to island position to strike line, end to end.
#
#     python -m w7x_twin migration [drive]     drive is redl or drift_kinetic

MIGRATION_OUT = Path("results/exhaust/strike_line_migration.json")
#: Interpreter carrying a CUDA-capable torch for the out-of-process Biot-Savart
#: worker, from the environment; unset leaves the summation on the CPU.
GPU_PYTHON = os.environ.get("W7X_TWIN_GPU_PYTHON", "")

#: Temperature scalings of the reference profiles, which set beta and with it the
#: bootstrap current. The first is the current-free reference.
TEMPERATURE_SCALES = (0.0, 0.5, 0.75, 1.0)

#: Volume sampling of the plasma current. The island region sits outside the plasma,
#: where the integrand is regular and this converges quickly.
NUM_THETA, NUM_ZETA, RADIAL_STRIDE = 80, 240, 4

MIGRATION_TURNS = 120
MIGRATION_LINES = 90
#: A narrow band just outside the last closed surface. The lines that carry the strike
#: line are the separatrix legs of the island chain; a wider fan reaches the stochastic
#: edge as well and spreads the strikes over the whole target, which buries the motion.
MIGRATION_LAYER = (1.00, 1.14)

load_drift_kinetic = _common.drift_kinetic_coefficients


def total_field(vacuum: VacuumField, parts) -> VacuumField:
    """Copy of the vacuum field with the plasma-current contribution added."""
    shape = (vacuum.num_phi, vacuum.num_z, vacuum.num_r)
    return vacuum.with_added_field(
        parts[0].reshape(shape), parts[1].reshape(shape), parts[2].reshape(shape)
    )


def resolution_floor(field, strikes, mask, steps_per_period=120) -> float:
    """Poloidal projection of one integration step at the strike: the strike-position quantum."""
    if not mask.any():
        return float("nan")
    r, phi, z = strikes.r[mask], strikes.phi[mask], strikes.z[mask]
    br, bp, bz = field(r, phi, z)
    dphi = 2.0 * np.pi / (steps_per_period * field.num_field_periods)
    return float(np.nanmedian(r * np.hypot(br, bz) / np.abs(bp) * dphi))


def strike_summary(strikes, names, elements, frame) -> dict:
    """Strike positions as arc length along the joined horizontal and vertical target contours."""
    if strikes.component is None:
        return {}
    wanted = {
        index: element
        for index, (name, element) in enumerate(zip(names, elements, strict=True))
        if element.name in frame
    }
    mask = strikes.struck & np.isin(strikes.component, list(wanted))
    if not mask.any():
        return {"lines": 0}

    positions: list[float] = []
    for index, element in wanted.items():
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        offset, reverse, span = frame[element.name]
        arc, _ = walls.arc_position(
            element, strikes.r[on_element], strikes.z[on_element],
            strikes.phi[on_element],
        )
        placed = offset + (span - arc if reverse else arc)
        positions.extend(placed.tolist())

    module, is_upper = walls.unit_of(strikes.phi[mask], strikes.z[mask])
    per_element = {
        element.name: int(np.count_nonzero(mask & (strikes.component == index)))
        for index, element in wanted.items()
    }
    return {
        "lines": int(mask.sum()),
        "median_arc_m": float(np.median(positions)),
        "mean_arc_m": float(np.mean(positions)),
        "inner_arc_m": float(np.percentile(positions, 10)),
        "outer_arc_m": float(np.percentile(positions, 90)),
        "spread_arc_m": float(np.std(positions)),
        "target_length_m": sum(span for _, _, span in frame.values()),
        "median_connection": float(np.median(strikes.connection_length_m[mask])),
        "per_module": [int(np.count_nonzero(module == m)) for m in range(1, 6)],
        "per_element": per_element,
        "upper_fraction": float(np.mean(is_upper)),
    }


def run_migration() -> int:
    drive = arg(1, default="redl")
    twin = _common.twin()
    vessel = _common.vessel()
    elements = _common.components()
    frame = walls.target_arc_frame(elements)
    print(f"drive: {drive}")
    print("  target arc: " + ", ".join(
        f"{name} {offset:.3f}-{offset + span:.3f} m" + (" reversed" if rev else "")
        for name, (offset, rev, span) in frame.items()))
    print(f"  {twin.geometry}")

    keywords = {"target": drive}
    if drive == "drift_kinetic":
        keywords["coefficients"] = load_drift_kinetic()
        keywords["ripple"] = neoclassical.load_ripple()

    vacuum = VacuumField(twin.response, twin.state("standard").currents)
    r_axis, z_axis = fieldlines.find_axis(vacuum)
    reference = twin.solve(twin.state("standard"), SCAN)

    rows = []
    for scale in TEMPERATURE_SCALES:
        started = time.monotonic()
        if scale == 0.0:
            equilibrium = reference
            current = 0.0
            field = vacuum
        else:
            profiles = kinetics.HIGH_PERFORMANCE.scaled(scale)
            solution = plasma_current.solve_self_consistent(
                twin, "standard", profiles, verbose=False, **keywords
            )
            equilibrium = solution.output
            current = solution.total_current_a
            distribution = plasma_response.current_distribution(
                equilibrium,
                num_theta=NUM_THETA,
                num_zeta=NUM_ZETA,
                radial_stride=RADIAL_STRIDE,
            )
            parts = plasma_response.field_on_grid(
                distribution, twin.coils.grid, interpreter=GPU_PYTHON or None, verbose=False
            )
            field = total_field(vacuum, parts)

        iota = np.asarray(equilibrium.wout.iotaf)
        # The fan is anchored to this equilibrium's own boundary, so every case samples
        # the same layer outside the plasma rather than the same absolute radii, which
        # the Shafranov shift would move the boundary out from under.
        r_lcfs, _ = diagnostics.boundary_cut(equilibrium.wout, 0.0)
        half_width = r_lcfs.max() - r_axis
        starts = r_axis + np.linspace(*MIGRATION_LAYER, MIGRATION_LINES) * half_width
        section, _ = fieldlines.trace(
            field,
            starts,
            np.full(starts.shape, z_axis),
            turns=MIGRATION_TURNS,
            plane_phi=0.0,
            vessel=vessel,
            components=elements,
        )
        summary = strike_summary(
            section.strikes, section.strikes.component_names, elements, frame
        )
        on_target = section.strikes.struck & np.isin(
            section.strikes.component,
            [i for i, e in enumerate(elements) if e.name in frame],
        )
        summary["resolution_floor_m"] = resolution_floor(field, section.strikes, on_target)
        rows.append(
            {
                "temperature_scale": scale,
                "beta": float(equilibrium.wout.betatotal),
                "bootstrap_current_a": float(current),
                "iota_edge": float(iota[-1]),
                "iota_axis": float(iota[0]),
                **summary,
            }
        )
        print(
            f"  T x {scale:.2f}: beta {100 * rows[-1]['beta']:.3f} %, "
            f"I_boot {current / 1e3:6.2f} kA, iota_edge {iota[-1]:.5f}, "
            f"{summary.get('lines', 0)} strikes on the horizontal target "
            f"in {time.monotonic() - started:.0f} s"
        )

    print()
    layout = _common.Table(
        ("<beta> [%]", "10.3f"), ("I_boot [kA]", "12.2f"), ("iota_edge", "10.5f"),
        ("mean [m]", "9.4f"), ("inner", "8.4f"), ("outer", "8.4f"),
        ("mean shift", ">12s"), ("inner shift", ">13s"), ("floor", ">9s"),
        ("lines", "6d"),
    )
    layout.begin()
    base = rows[0]
    for row in rows:
        mean = row.get("mean_arc_m")
        inner = row.get("inner_arc_m")
        shift = 1e3 * (mean - base["mean_arc_m"]) if mean is not None else float("nan")
        inner_shift = (
            1e3 * (inner - base["inner_arc_m"]) if inner is not None else float("nan")
        )
        layout.row(
            100 * row["beta"], row["bootstrap_current_a"] / 1e3, row["iota_edge"],
            mean if mean is not None else float("nan"),
            inner if inner is not None else float("nan"),
            row.get("outer_arc_m", float("nan")),
            f"{shift:9.1f} mm", f"{inner_shift:10.1f} mm",
            f"{1e3 * row.get('resolution_floor_m', float('nan')):6.1f} mm",
            row.get("lines", 0),
        )

    write_record(
        MIGRATION_OUT, {"drive": drive, "steps": rows}, geometry=twin.geometry
    )
    return 0


# -- recycling ---------------------------------------------------------------------

# Divertor neutral pressure solved from the recycling flux and the atomic rates.
#
#     python -m w7x_twin recycling [heating MW] [carbon fraction]

RECYCLING_OUT = Path("results/exhaust/recycling_pressure.json")
#: Upstream densities the ramp walks through, in m^-3.
UPSTREAM = (0.5e19, 1.0e19, 2.0e19, 4.0e19, 8.0e19)
#: Energy each ionisation removes from the flux tube: the potential plus the line
#: radiation that precedes it, in eV.
COST_PER_IONISATION_EV = 30.0


def run_recycling() -> int:
    power = 1e6 * arg(1, float, 5.0)
    crossing = power * 0.926

    # The layer the exhaust record carries, so this ramp and the heat-flux analysis
    # stand on one traced geometry rather than on constants that age apart.
    connection, incidence, area = layer_constants()
    parallel = crossing / area / incidence
    print(
        f"{crossing / 1e6:.3f} MW over {area:.3f} m2 at "
        f"{np.degrees(np.arcsin(incidence)):.2f} degrees is "
        f"{parallel / 1e6:.1f} MW/m2 parallel, over {connection:.0f} m"
    )

    layout = _common.Table(
        ("n_u [m^-3]", "11.2e"), ("T_t [eV]", "9.2f"), ("Gamma_t", "11.3e"),
        ("mfp [mm]", "9.2f"), ("n_0 [m^-3]", "11.3e"), ("p_0 [mPa]", "10.3f"),
        ("f_mom", "7.3f"),
    )
    print()
    layout.begin()

    rows = []
    for density in UPSTREAM:
        solution = edge.solve_two_point(
            density, parallel, connection
        )
        flux = (
            solution.target_density_m3
            * solution.sound_speed_m_s
        )
        layer = edge.recycling_layer(
            flux, solution.target_temperature_ev, solution.target_density_m3
        )
        rows.append(
            {
                "upstream_density_m3": density,
                "target_temperature_ev": solution.target_temperature_ev,
                "target_flux_m2_s": flux,
                "mean_free_path_m": layer.mean_free_path_m,
                "neutral_density_m3": layer.neutral_density_m3,
                "pressure_pa": layer.pressure_pa,
                "momentum_loss": layer.momentum_loss,
            }
        )
        layout.row(
            density, solution.target_temperature_ev, flux,
            1e3 * layer.mean_free_path_m, layer.neutral_density_m3,
            layer.pressure_mpa, layer.momentum_loss,
        )

    # Momentum loss alone raises the target temperature: it thins the target plasma, and a
    # thinner plasma needs a higher temperature to carry the same sheath flux. What detaches a
    # target is the power the neutrals take with them, and every ionisation costs the
    # potential plus the line radiation that precedes it.
    print()
    layout = _common.Table(
        ("n_u [m^-3]", "11.2e"), ("T_t [eV]", "9.2f"), ("f_pow", "7.3f"),
        ("f_mom", "7.3f"), ("T_t with both", "14.2f"),
    )
    layout.begin()
    for row in rows:
        cost = COST_PER_IONISATION_EV * edge.ELEMENTARY_CHARGE
        power_loss = min(row["target_flux_m2_s"] * cost / parallel, 0.95)
        detached = edge.solve_two_point_extended(
            row["upstream_density_m3"], parallel, connection,
            power_loss=power_loss,
            momentum_loss=min(row["momentum_loss"], 0.95),
        )
        row["power_loss"] = power_loss
        row["target_temperature_with_losses_ev"] = detached.target_temperature_ev
        layout.row(
            row["upstream_density_m3"], row["target_temperature_ev"], power_loss,
            row["momentum_loss"], detached.target_temperature_ev,
        )

    write_record(
        RECYCLING_OUT,
        {
            "heating_w": power,
            "crossing_power_w": crossing,
            "parallel_flux_w_m2": parallel,
            "connection_length_m": connection,
            "incidence_sine": incidence,
            "wetted_area_m2": area,
            "cases": rows,
        },
        # The layer this ramp stands on is traced geometry, so the record carries the
        # version it was traced under. Nothing here solves, so that version comes from
        # the inputs and not from a Twin.
        geometry=_common.current_geometry(),
        reads=(EXHAUST_OUT,),
    )
    return 0
