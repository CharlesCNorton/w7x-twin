"""Converged plasma-current field and the island it leaves in the edge; entry point ``response``."""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.hardware import machine
from w7x_twin.magnetics import field, fieldlines, plasma_response
from w7x_twin.mhd import diagnostics
from w7x_twin.plasma import current as plasma_current, kinetics, neoclassical


# -- response ----------------------------------------------------------------------

# Plasma-current field by axis-extrapolated volume integration and virtual casing sheet, cross-checked where both apply; the island is traced in the total field.
#
#     python -m w7x_twin response [configuration]

OUT = Path("results/magnetics/plasma_response.json")
GPU_PYTHON = "/home/unsymbolic/.venv/bin/python"
#: Volume samplings the axis value is extrapolated over.
SAMPLINGS = ((40, 120, 8), (60, 180, 4), (80, 240, 2), (120, 360, 1))
#: Boundary sheet resolutions, as (poloidal, toroidal) panels.
SHEET_SAMPLINGS = ((96, 960), (128, 1280), (256, 2560))
#: How far outside the boundary the boundary probes sit, as a fraction of the half-width.
BOUNDARY_CLEARANCE = 0.02
#: Panel widths of standoff the sheet needs to resolve a point, set by comparison against the volume integral.
SHEET_STANDOFF_PANELS = 8.0
#: Layer the island trace is launched across, as a fraction of the boundary half-width.
LAYER = (0.985, 1.40)
NUM_LINES = 60
TURNS = 400
#: Resonance the island chain sits on.
RESONANCE = (5, 5)


def panel_width(equilibrium, num_theta: int, num_zeta: int) -> float:
    """Largest side of one boundary panel, in metres."""
    wout = equilibrium.wout
    r, z = diagnostics.boundary_cut(wout, 0.0, num_theta)
    poloidal = float(np.sum(np.hypot(np.diff(r), np.diff(z)))) / num_theta
    toroidal = 2.0 * np.pi * float(wout.Rmajor_p) / num_zeta
    return max(poloidal, toroidal)


def probe_points(equilibrium, r_axis: float, z_axis: float) -> tuple[np.ndarray, list[str]]:
    """Probe points on the axis, just outside the boundary and in the island region."""
    r_lcfs, z_lcfs = diagnostics.boundary_cut(equilibrium.wout, 0.0, 128)
    outboard = int(np.argmax(r_lcfs))
    inboard = int(np.argmin(r_lcfs))
    half_width = float(r_lcfs.max()) - r_axis
    clearance = BOUNDARY_CLEARANCE * half_width
    rows = [
        (r_axis, z_axis, "magnetic axis"),
        (
            float(r_lcfs[outboard]) + clearance,
            float(z_lcfs[outboard]),
            "just outside the boundary, outboard",
        ),
        (
            float(r_lcfs[inboard]) - clearance,
            float(z_lcfs[inboard]),
            "just outside the boundary, inboard",
        ),
        (r_axis + 1.15 * half_width, z_axis, "island region"),
        (r_axis + 1.60 * half_width, z_axis, "outside the island"),
    ]
    points = np.array([[r * np.cos(0.0), r * np.sin(0.0), z] for r, z, _ in rows])
    return points, [label for _, _, label in rows]


def island_width(section, r_axis: float, z_axis: float, resonance) -> dict:
    """Radial extent of the island chain on the midplane of a Poincare section."""
    width, lines = fieldlines.midplane_island_span(section, r_axis, z_axis)
    if lines == 0:
        return {"width_m": float("nan"), "lines": 0}
    return {
        "width_m": width,
        "lines": lines,
        "resonance": f"{resonance[0]}/{resonance[1]}",
    }


def run_response() -> int:
    key = _common.arg(1, default="standard")
    twin = _common.twin()
    config = machine.get(key)
    vacuum = field.VacuumField(twin.response, config.as_extcur())
    r_axis, z_axis = fieldlines.find_axis(vacuum)
    print(f"{twin.geometry}")

    # Finite-beta equilibrium carrying its own bootstrap current.
    profiles = kinetics.HIGH_PERFORMANCE
    coefficients = neoclassical.discover_monoenergetic_profile(
        Path(neoclassical.RADIAL_SCANS[0])
    )
    ripple = neoclassical.load_ripple()
    solution = plasma_current.solve_self_consistent(
        twin, key, profiles, verbose=False,
        target="drift_kinetic" if coefficients is not None else "redl",
        coefficients=coefficients, ripple=ripple,
    )
    equilibrium = solution.output
    print(
        f"{key}: beta {100 * float(equilibrium.wout.betatotal):.3f} per cent carrying "
        f"{float(equilibrium.wout.ctor) / 1e3:.2f} kA"
    )

    points, labels = probe_points(equilibrium, r_axis, z_axis)

    # The volume integration, refined, and extrapolated to zero element size.
    print()
    print("volume integration of the plasma current")
    record = plasma_response.refine_until_converged(
        equilibrium, points, samplings=SAMPLINGS, tolerance=0.0,
        interpreter=GPU_PYTHON, verbose=True,
    )
    extrapolated, correction = plasma_response.extrapolated_field(record)

    # The same field from the boundary sheet, refined as the volume was.
    print()
    print("boundary sheet")
    casing = None
    previous = None
    sheet_changes = []
    for num_theta, num_zeta in SHEET_SAMPLINGS:
        started = time.monotonic()
        sheet = plasma_response.boundary_sheet(
            equilibrium, num_theta=num_theta, num_zeta=num_zeta, vacuum=vacuum
        )
        casing = plasma_response.virtual_casing_field(sheet, points)
        magnitude = np.linalg.norm(casing, axis=-1)
        change = (
            np.abs(magnitude - previous)
            if previous is not None
            else np.full(magnitude.shape, np.nan)
        )
        sheet_changes.append([float(v) for v in change])
        previous = magnitude
        print(
            f"  {num_theta}x{num_zeta}: {sheet.num_sources} panels carrying "
            f"{sheet.net_current_a() / 1e3:.3f} kA against the equilibrium's "
            f"{float(equilibrium.wout.ctor) / 1e3:.3f} kA"
            + (
                f", largest change {1e3 * float(np.nanmax(change)):.3f} mT"
                if np.isfinite(change).any() else ""
            )
            + f", {time.monotonic() - started:.1f} s"
        )

    layout = _common.Table(
        ("point", "36s"), ("volume [mT]", "12.4f"), ("extrapolated", "12.4f"),
        ("correction", "10.4f"), ("sheet [mT]", "11.4f"), ("difference", ">10s"),
        ("to boundary [mm]", ">16s"),
    )
    print()
    layout.begin()
    finest = panel_width(equilibrium, *SHEET_SAMPLINGS[-1])
    contour_r, contour_z = diagnostics.boundary_cut(equilibrium.wout, 0.0, 512)
    exterior = ~plasma_response.inside_boundary(
        equilibrium,
        np.hypot(points[:, 0], points[:, 1]),
        np.arctan2(points[:, 1], points[:, 0]),
        points[:, 2],
    )
    standoff = [
        float(
            np.min(
                np.hypot(
                    contour_r - np.hypot(point[0], point[1]), contour_z - point[2]
                )
            )
        )
        for point in points
    ]
    rows = []
    for index, label in enumerate(labels):
        volume = float(np.linalg.norm(record.field[-1][index]))
        limit = float(np.linalg.norm(extrapolated[index]))
        sheet_value = float(np.linalg.norm(casing[index]))
        difference = abs(sheet_value - limit) / max(limit, 1e-12)
        rows.append(
            {
                "point": label,
                "volume_t": volume,
                "extrapolated_t": limit,
                "extrapolation_correction_t": float(correction[index]),
                "sheet_t": sheet_value,
                "sheet_last_change_t": float(sheet_changes[-1][index]),
                "standoff_m": standoff[index],
                "outside_the_boundary": bool(exterior[index]),
                "resolved_by_the_sheet": bool(
                    exterior[index] and standoff[index] > SHEET_STANDOFF_PANELS * finest
                ),
                "relative_difference": difference,
            }
        )
        layout.row(
            label, 1e3 * volume, 1e3 * limit, 1e3 * float(correction[index]),
            1e3 * sheet_value, f"{100 * difference:9.2f}%",
            f"{1e3 * standoff[index]:11.1f}",
        )
    changes = record.changes()
    outside = [row for row in rows if row["outside_the_boundary"]]
    speaks_for = [row for row in outside if row["resolved_by_the_sheet"]]
    print()
    print(
        f"one boundary panel is {1e3 * finest:.1f} mm across at the finest sampling, and "
        f"the sheet needs {SHEET_STANDOFF_PANELS:.0f} of those between it and the point it "
        f"is asked about"
    )
    print(
        f"{len(speaks_for)} of the {len(outside)} exterior probes stands that far off, and "
        + (
            "there the two independent methods agree to "
            f"{100 * max(row['relative_difference'] for row in speaks_for):.1f} per cent"
            if speaks_for
            else "none of them does"
        )
    )
    print(
        "closer in the volume integral is the converged one: its last refinement moved the "
        f"island region by {1e3 * float(correction[3]):.4f} mT, so what the sheet adds is "
        "the one-sided limit on the boundary itself, which the volume integral has no value "
        "for at all"
    )
    print(
        f"\nthe last refinement moved the axis value by "
        f"{100 * abs(changes[-1]) / max(float(np.linalg.norm(record.field[-1][0])), 1e-12):.2f} "
        f"per cent of it, and the extrapolation carries "
        f"{100 * float(correction[0]) / max(float(np.linalg.norm(extrapolated[0])), 1e-12):.2f} "
        "per cent on top"
    )

    # The island in the total field; the resonance sits outside the boundary.
    print()
    # Plasma field on the trace grid from the converged volume integral.
    distribution = plasma_response.current_distribution(
        equilibrium, num_theta=SAMPLINGS[-1][0], num_zeta=SAMPLINGS[-1][1],
        radial_stride=SAMPLINGS[-1][2],
    )
    parts = plasma_response.field_on_grid(
        distribution, twin.coils.grid, interpreter=GPU_PYTHON, verbose=True
    )
    shape = (vacuum.num_phi, vacuum.num_z, vacuum.num_r)
    total = vacuum.with_added_field(
        parts[0].reshape(shape), parts[1].reshape(shape), parts[2].reshape(shape)
    )

    r_lcfs, _ = diagnostics.boundary_cut(equilibrium.wout, 0.0)
    half_width = float(r_lcfs.max()) - r_axis
    starts = r_axis + np.linspace(*LAYER, NUM_LINES) * half_width
    islands = {}
    for label, traced in (("vacuum field", vacuum), ("with the plasma field", total)):
        axis_r, axis_z = fieldlines.find_axis(traced)
        section, _ = fieldlines.trace(
            traced, starts, np.full(starts.shape, axis_z), turns=TURNS, plane_phi=0.0
        )
        answer = island_width(section, axis_r, axis_z, RESONANCE)
        islands[label] = {**answer, "axis_r_m": float(axis_r)}
        print(
            f"  {label:22s} axis at R = {axis_r:.4f} m, island spans "
            f"{1e3 * answer['width_m']:.1f} mm across {answer['lines']} lines"
        )
    shift = islands["with the plasma field"]["width_m"] - islands["vacuum field"]["width_m"]
    print(
        f"  the plasma's own field changes the island width by {1e3 * shift:+.1f} mm, so "
        f"an island statement no longer rests on the vacuum field alone"
    )

    _common.write_record(
        OUT,
        {
            "configuration": key,
            "beta": float(equilibrium.wout.betatotal),
            "toroidal_current_a": float(equilibrium.wout.ctor),
            "sheet_current_a": sheet.net_current_a(),
            "samplings": [list(s) for s in record.samplings],
            "sources": record.sources,
            "sheet_samplings": [list(s) for s in SHEET_SAMPLINGS],
            "sheet_changes_t": sheet_changes,
            "panel_width_m": finest,
            "points": rows,
            "islands": islands,
        },
        geometry=twin.geometry,
    )
    return 0
