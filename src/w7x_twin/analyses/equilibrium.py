"""Equilibria, their pressure response and stability, and the stepped-pressure island solves."""

from __future__ import annotations

import dataclasses
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from w7x_twin.analyses import _common
from w7x_twin.analyses._common import Table, arg, args, write_record
from w7x_twin.hardware import coils as coil_geometry, machine
from w7x_twin.magnetics import field, fieldlines
from w7x_twin.magnetics.field import VacuumField
from w7x_twin.mhd import diagnostics, stepped_pressure
from w7x_twin.mhd.equilibrium import REFERENCE, SCAN, Scenario, Twin
from w7x_twin.plasma import kinetics
from w7x_twin.records import ensemble

#: The SPEC executable and the directory the stepped-pressure solves run in. SPEC is
#: hand-built, so the environment names it; the default is where its own build
#: instructions leave it.
SPEC = Path(
    os.environ.get("W7X_TWIN_SPEC", Path.home() / "src/SPEC/build/build/bin/xspec")
)
SPEC_WORK = Path(os.environ.get("W7X_TWIN_SPEC_WORK", "cache/spec"))
#: The resonance the stepped-pressure island sits on; the OP1.2a mimic tapers cross
#: 5/6 inside the plasma.
RESONANCE = (5, 6)


# -- equilibrium -------------------------------------------------------------------

# Solve every configuration in the library and tabulate the machine diagnostics.

RESOLUTION = SCAN


SURVEY_OUT = Path("results/equilibrium/config_survey.json")


def run_equilibrium() -> int:
    twin = _common.twin()
    keys = args() or machine.all_keys()

    rows = []
    for key in keys:
        config = machine.get(key)
        state = twin.state(config)
        try:
            output = twin.solve(state, RESOLUTION)
        except RuntimeError as error:
            print(f"{key:22s} FAILED: {str(error).splitlines()[0][:90]}")
            continue
        d = diagnostics.analyse(output)
        rows.append((key, config, d))

    table = Table(
        ("configuration", "24s"), ("B0", "6.3f"), ("Bbean", "6.3f"), ("Btri", "6.3f"),
        ("R", "7.4f"), ("a", "6.4f"), ("V", "7.3f"), ("i_axis", "7.4f"),
        ("i_edge", "7.4f"), ("mirror%", "8.3f"), ("well%", "7.3f"), ("resonances", "s"),
    )
    table.begin()
    for key, _config, d in rows:
        table.row(
            key, d.b_axis_t, d.b_bean_t, d.b_triangle_t, d.major_radius_m,
            d.minor_radius_m, d.plasma_volume_m3, d.iota_axis, d.iota_edge,
            d.mirror_percent, 100 * d.magnetic_well_depth,
            ",".join(d.resonances_crossed) or "-",
        )

    # A subset run refreshes its own entries and leaves the rest of the record standing.
    payload = {}
    if SURVEY_OUT.exists() and set(keys) != set(machine.all_keys()):
        payload = json.loads(SURVEY_OUT.read_text()).get("configurations", {})
    payload.update(
        {
            key: {
                "label": config.label,
                "extcur": list(config.currents),
                "source": config.source,
                **{
                    k: (list(v) if isinstance(v, tuple) else v)
                    for k, v in d.as_dict().items()
                },
            }
            for key, config, d in rows
        }
    )
    write_record(
        SURVEY_OUT,
        {"configurations": payload},
        geometry=twin.geometry,
        note=f" ({len(rows)} configurations)",
    )
    return 0


# -- beta --------------------------------------------------------------------------

# Pressure scan at fixed currents, each step hot-restarted along one equilibrium branch.

PEAK_PRESSURES_PA = [0.0, 1e4, 2.5e4, 5e4, 7.5e4, 1e5, 1.5e5, 2e5, 2.5e5, 3e5]


def run_beta() -> int:
    configuration = arg(1, default="standard")
    twin = _common.twin()

    vacuum = twin.solve(twin.state(configuration), SCAN)
    previous = vacuum
    rows = []

    table = Table(
        ("p0 [kPa]", "9.1f"), ("<beta>%", "9.4f"), ("beta_ax%", "9.4f"),
        ("W_MHD[MJ]", "10.3f"), ("i_axis", "8.4f"), ("i_edge", "8.4f"),
        ("shift[mm]", "10.3f"), ("inner[mm]", "10.3f"), ("well%", "8.3f"),
        ("V[m^3]", "8.3f"), ("Mercier<0", ">10s"),
    )
    print(f"configuration: {configuration}")
    table.begin()

    for peak in PEAK_PRESSURES_PA:
        state = twin.state(
            configuration, scenario=Scenario(peak_pressure_pa=peak)
        )
        try:
            output = twin.solve(state, SCAN, restart_from=previous)
        except RuntimeError as error:
            print(f"{peak / 1e3:9.1f}  FAILED: {str(error).splitlines()[0][:70]}")
            break
        previous = output
        d = diagnostics.analyse(output, vacuum_reference=vacuum)
        rows.append({"peak_pressure_pa": peak, **d.as_dict()})
        table.row(
            peak / 1e3, 100 * d.beta_total, 100 * d.beta_axis,
            d.stored_energy_j / 1e6, d.iota_axis, d.iota_edge,
            1e3 * d.axis_shift_m, 1e3 * d.axis_shift_in_boundary_m,
            100 * d.magnetic_well_depth, d.plasma_volume_m3,
            f"{100 * d.mercier_unstable_fraction:9.1f}%",
        )

    # The shift at the measured profiles: the digitised post-pellet density and both
    # temperatures build the pressure, so the peaking entering the equilibrium is the
    # machine's own rather than the flat-topped synthetic, and the shift is compared
    # where the published one was measured.
    measured = measured_profile_case(twin, vacuum, previous)

    write_record(
        Path("results/equilibrium") / f"beta_scan_{configuration}.json",
        {
            "configuration": configuration,
            "steps": [
                {k: (list(v) if isinstance(v, tuple) else v) for k, v in r.items()}
                for r in rows
            ],
            "measured_profile": measured,
        },
        geometry=twin.geometry,
    )
    return 0


def measured_profile_case(twin: Twin, vacuum, restart) -> dict | None:
    """The equilibrium at the digitised profiles' own pressure, and its shift."""
    from w7x_twin.records import programmes as measured_profiles

    curves = measured_profiles.load()

    def pick(quantity: str, label: str):
        for curve in curves:
            if (
                curve.discharge == "20181016.037"
                and quantity in curve.quantity
                and label in curve.label
            ):
                return curve
        return None

    density = pick("density", "post-pellet")
    electron = pick("temperature", "electron")
    ion = pick("temperature", "ion")
    if density is None or electron is None or ion is None:
        print("\nno digitised post-pellet profile set; the measured case is skipped")
        return None

    s = np.linspace(0.0, 1.0, 41)
    rho = np.sqrt(s)
    r_eff = rho * float(np.max(electron.x))
    pressure = (
        1.602176634e-19
        * (density.at(rho) * 1e19)
        * 1e3
        * (electron.at(r_eff) + ion.at(r_eff))
    )

    print()
    print("the measured post-pellet profiles of 20181016.037 as the pressure")
    cases = []
    scale = 1.0
    for label in ("as measured", "scaled to one per cent"):
        state = twin.state(
            "standard", scenario=Scenario.from_pressure_spline(s, scale * pressure)
        )
        output = twin.solve(state, SCAN, restart_from=restart)
        d = diagnostics.analyse(output, vacuum_reference=vacuum)
        peaking = d.beta_axis / max(d.beta_total, 1e-30)
        cases.append(
            {
                "label": label,
                "pressure_scale": scale,
                "beta_percent": 100 * d.beta_total,
                "pressure_peaking": peaking,
                "shift_mm": 1e3 * d.axis_shift_m,
                "shift_in_boundary_mm": 1e3 * d.axis_shift_in_boundary_m,
            }
        )
        print(
            f"  {label:24s} <beta> {100 * d.beta_total:.3f} %, pressure peaking "
            f"{peaking:.2f}, shift {1e3 * d.axis_shift_m:.1f} mm in the laboratory and "
            f"{1e3 * d.axis_shift_in_boundary_m:.1f} mm in the boundary"
        )
        if d.beta_total <= 0.0:
            break
        scale = 0.0105 / d.beta_total
        restart = output
    return {"discharge": "20181016.037", "cases": cases}


# -- figure ------------------------------------------------------------------------

# Render the twin: coil set, flux surfaces, transform, field and pressure response.

FIGURE_DIR = Path("results")
#: Where the rendered figures live, which is where the README displays them from.
DOCS_DIR = Path("docs")
#: phi = 0 is the bean cross section, 36 degrees the triangular one at the half period.
PLANES_DEG = (0.0, 18.0, 36.0)
PLANE_NAMES = ("bean", "intermediate", "triangular")
CONFIGURATIONS = ("standard", "high_mirror_ref167", "op12a_22ka_mimic")


def plot_coils_3d(ax, coils: machine.CoilSet) -> None:
    colors = plt.cm.viridis(np.linspace(0, 0.85, coils.num_circuits))
    for circuit_index, group in enumerate(coils.filaments):
        for coil_index, xyz in enumerate(group):
            ax.plot(
                xyz[:, 0],
                xyz[:, 1],
                xyz[:, 2],
                color=colors[circuit_index],
                linewidth=0.5,
                alpha=0.75,
                label=coils.circuit_keys[circuit_index]
                if coil_index == 0
                else None,
            )
    ax.set_box_aspect((1, 1, 0.42))
    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.set_title("W7-X coil set: 50 non-planar + 20 planar", fontsize=10)
    ax.legend(fontsize=6, loc="upper right", ncol=2)


def plot_surface_3d(ax, wout, num_phi: int = 160, num_theta: int = 120) -> None:
    phi = np.linspace(0.0, 2.0 * np.pi, num_phi)
    xs, ys, zs = [], [], []
    for p in phi:
        r, z = diagnostics.boundary_cut(wout, p, num_theta)
        xs.append(r * np.cos(p))
        ys.append(r * np.sin(p))
        zs.append(z)
    ax.plot_surface(
        np.array(xs),
        np.array(ys),
        np.array(zs),
        rstride=2,
        cstride=4,
        color="#c94f2b",
        alpha=0.85,
        linewidth=0,
        antialiased=True,
    )


def run_figure() -> int:
    FIGURE_DIR.mkdir(exist_ok=True)
    twin = _common.twin()

    solutions = {key: twin.solve(twin.state(key), SCAN) for key in CONFIGURATIONS}
    reference = solutions["standard"]

    fig = plt.figure(figsize=(15.5, 11.0))
    grid = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 1.0], hspace=0.33, wspace=0.26)

    # -- coil set and last closed flux surface -----------------------------
    ax3d = fig.add_subplot(grid[0, :2], projection="3d")
    plot_coils_3d(ax3d, twin.coils)
    plot_surface_3d(ax3d, reference.wout)
    ax3d.set_title(
        "W7-X coil set and the last closed flux surface, standard configuration",
        fontsize=10,
    )

    # -- iota profiles -----------------------------------------------------
    ax = fig.add_subplot(grid[0, 2])
    for key, output in solutions.items():
        s, iota = diagnostics.iota_profile(output.wout)
        ax.plot(s, iota, label=key, linewidth=1.6)
    for n, m in diagnostics.DIVERTOR_RESONANCES:
        ax.axhline(n / m, color="0.6", linestyle=":", linewidth=1.0)
        ax.text(0.02, n / m + 0.004, f"{n}/{m}", fontsize=7, color="0.35")
    ax.set_xlabel("normalised toroidal flux  s")
    ax.set_ylabel(r"rotational transform  $\iota$")
    ax.set_title("Transform profile and island resonances", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # -- cross sections ----------------------------------------------------
    for column, (angle, name) in enumerate(zip(PLANES_DEG, PLANE_NAMES, strict=True)):
        ax = fig.add_subplot(grid[1, column])
        phi = np.deg2rad(angle)
        for r, z in diagnostics.cross_section(reference.wout, phi, num_surfaces=9):
            ax.plot(r, z, color="#2b6cb0", linewidth=0.8)
        r, z = diagnostics.boundary_cut(reference.wout, phi)
        ax.plot(r, z, color="#c94f2b", linewidth=1.8)
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        if column == 0:
            ax.set_ylabel("Z [m]")
        ax.set_title(rf"$\varphi$ = {angle:.0f}°  ({name})", fontsize=10)
        ax.grid(alpha=0.2)

    # -- |B| on axis -------------------------------------------------------
    ax = fig.add_subplot(grid[2, 0])
    for key, output in solutions.items():
        b = diagnostics.field_on_axis(output.wout)
        angle = np.linspace(0.0, 72.0, len(b), endpoint=False)
        ax.plot(angle, b, label=key, linewidth=1.5)
    ax.set_xlabel(r"toroidal angle $\varphi$ [deg]")
    ax.set_ylabel("|B| on axis [T]")
    ax.set_title("Mirror term over one field period", fontsize=10)
    ax.legend(fontsize=7)
    ax.grid(alpha=0.25)

    # -- pressure response -------------------------------------------------
    scan_path = FIGURE_DIR / "equilibrium/beta_scan_standard.json"
    if scan_path.exists():
        stored = json.loads(scan_path.read_text())
        scan = stored["steps"] if isinstance(stored, dict) else stored
        beta = [100 * row["beta_total"] for row in scan]

        ax = fig.add_subplot(grid[2, 1])
        ax.plot(beta, [1e3 * row["axis_shift_m"] for row in scan], "o-", color="#c94f2b")
        ax.set_xlabel(r"$\langle\beta\rangle$ [%]")
        ax.set_ylabel("Shafranov shift [mm]")
        ax.set_title("Axis shift against pressure", fontsize=10)
        ax.grid(alpha=0.25)

        ax = fig.add_subplot(grid[2, 2])
        ax.plot(
            beta,
            [100 * row["magnetic_well_depth"] for row in scan],
            "o-",
            color="#2b6cb0",
            label="magnetic well",
        )
        ax.set_xlabel(r"$\langle\beta\rangle$ [%]")
        ax.set_ylabel("well depth [%]")
        ax.set_title("Well deepening with pressure", fontsize=10)
        ax.grid(alpha=0.25)

    fig.suptitle(
        "Wendelstein 7-X digital twin: free-boundary VMEC++ from the IPP coil set",
        fontsize=13,
    )
    DOCS_DIR.mkdir(exist_ok=True)
    path = DOCS_DIR / "w7x_twin_overview.png"
    fig.savefig(path, dpi=135, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


# -- islands -----------------------------------------------------------------------

# Dense edge trace at the bean plane resolving the island chain the strike lines ride.

ISLANDS_DIR = DOCS_DIR
ISLAND_TURNS = 500


def draw(ax, poincare, colors, size: float) -> None:
    for index in range(int(poincare.line_index.max()) + 1):
        mask = poincare.line_index == index
        ax.plot(
            poincare.r[mask],
            poincare.z[mask],
            ".",
            color=colors[index],
            markersize=size,
            markeredgewidth=0,
            alpha=0.85,
        )


def run_islands() -> int:
    key = arg(1, default="standard")
    twin = _common.twin()
    config = machine.get(key)
    vacuum = VacuumField(twin.response, config.as_extcur())

    r_axis, z_axis = fieldlines.find_axis(vacuum)
    equilibrium = twin.solve(twin.state(key), SCAN)
    r_lcfs, z_lcfs = diagnostics.boundary_cut(equilibrium.wout, 0.0)
    half_width = r_lcfs.max() - r_axis

    # Two families: the confined core, and a dense fan through the edge where the
    # transform crosses the resonance and the islands live.
    core = r_axis + np.linspace(0.06, 0.96, 24) * half_width
    edge = r_axis + np.linspace(0.985, 1.40, 46) * half_width
    starts = np.concatenate([core, edge])

    started = time.monotonic()
    poincare, _ = fieldlines.trace(
        vacuum,
        starts,
        np.full(starts.shape, z_axis),
        turns=ISLAND_TURNS,
        plane_phi=0.0,
    )
    print(
        f"{key}: {starts.size} lines x {ISLAND_TURNS} turns in {time.monotonic() - started:.0f} s"
    )

    colors = plt.cm.turbo(np.linspace(0.06, 0.97, starts.size))

    fig, axes = plt.subplots(1, 2, figsize=(13.6, 7.4))

    draw(axes[0], poincare, colors, 0.55)
    axes[0].plot(r_lcfs, z_lcfs, color="#111111", linewidth=1.4, label="VMEC boundary")
    axes[0].plot([r_axis], [z_axis], "+", color="#111111", markersize=10)
    axes[0].set_title("Full section", fontsize=11)
    axes[0].legend(fontsize=8, loc="upper left")

    draw(axes[1], poincare, colors, 1.5)
    axes[1].plot(r_lcfs, z_lcfs, color="#111111", linewidth=1.4)
    axes[1].set_xlim(r_axis + 0.75 * half_width, r_axis + 1.75 * half_width)
    axes[1].set_ylim(-0.62, 0.62)
    axes[1].set_title("Edge, magnified: the island chain", fontsize=11)

    for ax in axes:
        ax.set_aspect("equal")
        ax.set_xlabel("R [m]")
        ax.grid(alpha=0.15)
    axes[0].set_ylabel("Z [m]")

    iota_edge = float(np.asarray(equilibrium.wout.iotaf)[-1])
    fig.suptitle(
        f"W7-X island divertor, {config.label}: vacuum field lines at the bean plane\n"
        f"VMEC stops at $\\iota$ = {iota_edge:.4f}; the island chain sits beyond it, "
        f"where the transform reaches the low-order rational",
        fontsize=12,
    )
    fig.tight_layout()
    ISLANDS_DIR.mkdir(exist_ok=True)
    path = ISLANDS_DIR / f"w7x_islands_{key}.png"
    fig.savefig(path, dpi=145, bbox_inches="tight")
    print(f"wrote {path}")
    return 0


# -- stability ---------------------------------------------------------------------

# Ballooning drive and tearing index beside VMEC's Mercier criterion.
#
#     python -m w7x_twin stability [beta per cent ...]

STABILITY_OUT = Path("results/equilibrium/stability_limits.json")
BETAS = (0.5, 1.05, 2.0, 3.0)
KNOTS = 41
BETA_TOLERANCE = 0.01
BETA_STEPS = 6


def run_stability() -> int:
    betas = tuple(a / 100.0 for a in args(float)) or tuple(b / 100.0 for b in BETAS)
    twin = _common.twin()
    print(f"{twin.geometry}")

    table = Table(
        ("beta [%]", "9.4f"), ("Mercier", "8.3f"), ("resistive", "10.3f"),
        ("H range", ">16s"), ("ballooning", "11.3f"), ("max alpha", "10.3f"),
        ("resonances", "11d"), ("tearing", "8d"), ("global unstable", "16d"),
    )
    print()
    table.begin()

    rows = []
    for target in betas:
        scale, beta, output = 1.0, float("nan"), None
        for _ in range(BETA_STEPS):
            output = twin.solve_profiles(
                "standard", kinetics.HIGH_PERFORMANCE,
                pressure_scale=scale, knots=KNOTS,
            )
            beta = float(output.wout.betatotal)
            if abs(beta - target) <= BETA_TOLERANCE * target:
                break
            scale *= target / max(beta, 1e-12)

        analysis = diagnostics.analyse(output)
        wout = output.wout
        surfaces = int(wout.ns)
        s = np.linspace(0.0, 1.0, surfaces)
        pressure = np.asarray(wout.presf)
        transform = np.asarray(wout.iotaf)

        ballooning = diagnostics.ballooning(
            s, pressure, transform, analysis.b_axis_t,
            analysis.minor_radius_m, analysis.major_radius_m,
        )
        # The resistive branch of the same interchange, on the solver's own Mercier
        # decomposition: shear does not stabilise it, so where H sits against (0, 1)
        # says whether the resistive criterion is the stricter of the two.
        resistive = diagnostics.resistive_interchange(output.mercier)
        h_interior = resistive.h[1:-1][np.isfinite(resistive.h[1:-1])]
        d_r_interior = resistive.d_r[1:-1][np.isfinite(resistive.d_r[1:-1])]

        # Enclosed toroidal current density, from the equilibrium's own profile.
        current = np.asarray(wout.jcuru) if hasattr(wout, "jcuru") else np.zeros(surfaces)
        resonances = diagnostics.resonances_in(transform)
        # Global interchange modes, which the local criteria cannot rule on.
        # Local magnetic well on the same convention as diagnostics.magnetic_well_depth:
        # positive where dV/ds falls outward, which is the stabilising case. VMEC stores dV/ds
        # with a zero at the axis, so the profile comes from specific_volume rather than the
        # raw array.
        well_s, specific = diagnostics.specific_volume(wout)
        local_well = -np.gradient(specific) / abs(float(specific[0]))
        well = np.interp(s, well_s, local_well)
        global_list = diagnostics.global_modes(
            s, pressure, transform, well, analysis.minor_radius_m,
            analysis.major_radius_m, analysis.b_axis_t,
        )
        tearing = []
        for m, n in resonances:
            result = diagnostics.tearing_index(
                s, current, transform, m, n, analysis.minor_radius_m
            )
            tearing.append(
                {
                    "m": m, "n": n, "s_resonant": result.s_resonant,
                    "delta_prime_per_m": result.delta_prime_per_m,
                    "unstable": result.unstable,
                }
            )
        unstable = [t for t in tearing if t["unstable"]]

        rows.append(
            {
                "beta": beta,
                "mercier_unstable_fraction": analysis.mercier_unstable_fraction,
                "resistive_interchange_unstable_fraction": (
                    analysis.resistive_interchange_unstable_fraction
                ),
                "resistive_d_r_minimum": (
                    float(np.min(d_r_interior)) if d_r_interior.size else float("nan")
                ),
                "geodesic_h_range": (
                    [float(h_interior.min()), float(h_interior.max())]
                    if h_interior.size
                    else [float("nan"), float("nan")]
                ),
                "ballooning_unstable_fraction": ballooning.unstable_fraction,
                "max_alpha": float(np.nanmax(ballooning.alpha)),
                "resonances": [f"{n}/{m}" for m, n in resonances],
                "global_modes": [
                    {"m": g.m, "n": g.n, "eigenvalue": g.eigenvalue,
                     "peak_s": g.peak_s, "unstable": g.unstable}
                    for g in global_list
                ],
                "tearing": tearing,
            }
        )
        table.row(
            100 * beta, analysis.mercier_unstable_fraction,
            analysis.resistive_interchange_unstable_fraction,
            f"{h_interior.min():7.3f} {h_interior.max():7.3f}"
            if h_interior.size
            else "-",
            ballooning.unstable_fraction, float(np.nanmax(ballooning.alpha)),
            len(resonances), len(unstable),
            sum(1 for g in global_list if g.unstable),
        )

    print()
    print(
        "the ballooning drive rises with beta and the shear does not, so the unstable "
        f"fraction runs {rows[0]['ballooning_unstable_fraction']:.3f} to "
        f"{rows[-1]['ballooning_unstable_fraction']:.3f} across the scan"
    )
    print(
        "the resistive interchange fraction runs "
        + ", ".join(
            f"{r['resistive_interchange_unstable_fraction']:.3f}" for r in rows
        )
        + " against Mercier's "
        + ", ".join(f"{r['mercier_unstable_fraction']:.3f}" for r in rows)
        + ", so the resistive branch opens nothing the ideal criterion had not"
    )

    crossing = tearing_configuration_case(twin)

    write_record(
        STABILITY_OUT,
        {"cases": rows, "tearing_configuration": crossing},
        geometry=twin.geometry,
    )
    return 0


#: Configuration whose transform crosses a low-order rational inside the plasma, so
#: the tearing index has a resonance to be evaluated at; the standard transform
#: crosses none.
TEARING_CONFIGURATION = "op12a_22ka_mimic"


def tearing_configuration_case(twin: Twin) -> dict:
    """Delta-prime and its resistive-layer growth rate on the crossing configuration."""
    from w7x_twin.plasma import current as plasma_current

    output = twin.solve_profiles(TEARING_CONFIGURATION, kinetics.HIGH_PERFORMANCE)
    analysis = diagnostics.analyse(output)
    wout = output.wout
    surfaces = int(wout.ns)
    s = np.linspace(0.0, 1.0, surfaces)
    transform = np.asarray(wout.iotaf)
    current_density = (
        np.asarray(wout.jcuru) if hasattr(wout, "jcuru") else np.zeros(surfaces)
    )
    resonances = diagnostics.resonances_in(transform)
    print()
    print(
        f"{TEARING_CONFIGURATION} at <beta> {100 * analysis.beta_total:.2f} per cent: "
        f"transform {transform[0]:.5f} to {transform[-1]:.5f} crosses "
        + (", ".join(f"{n}/{m}" for m, n in resonances) or "no rational")
    )

    profiles = kinetics.HIGH_PERFORMANCE
    density = profiles.density(s)
    temperature = profiles.electron_temperature(s)
    resistivity = plasma_current.spitzer_resistivity(temperature, density)
    radius = analysis.minor_radius_m * np.sqrt(np.clip(s, 1e-12, 1.0))
    d_iota_d_r = np.gradient(transform, radius)
    proton_mass = 1.67262192369e-27
    alfven = analysis.b_axis_t / np.sqrt(
        diagnostics.VACUUM_PERMEABILITY * density * proton_mass
    )

    layout = Table(
        ("resonance", ">9s"), ("s", "7.4f"), ("delta' [1/m]", "13.3f"),
        ("eta [ohm m]", "12.3e"), ("growth [1/s]", "13.3f"), ("layer time", ">11s"),
    )
    layout.begin()
    entries = []
    for m, n in resonances:
        result = diagnostics.tearing_index(
            s, current_density, transform, m, n, analysis.minor_radius_m
        )
        if not np.isfinite(result.s_resonant):
            continue
        eta = float(np.interp(result.s_resonant, s, resistivity))
        wavenumber_gradient = (
            m
            * abs(float(np.interp(result.s_resonant, s, d_iota_d_r)))
            / analysis.major_radius_m
        )
        speed = float(np.interp(result.s_resonant, s, alfven))
        growth = diagnostics.tearing_growth_rate(
            result.delta_prime_per_m, eta, wavenumber_gradient, speed
        )
        entries.append(
            {
                "m": m, "n": n, "s_resonant": result.s_resonant,
                "delta_prime_per_m": result.delta_prime_per_m,
                "unstable": result.unstable,
                "resistivity_ohm_m": eta,
                "parallel_wavenumber_gradient_per_m2": wavenumber_gradient,
                "alfven_speed_m_s": speed,
                "growth_rate_per_s": growth,
            }
        )
        layout.row(
            f"{n}/{m}", result.s_resonant, result.delta_prime_per_m, eta, growth,
            f"{1.0 / growth:9.3f} s" if growth > 0.0 else "stable",
        )
    return {
        "configuration": TEARING_CONFIGURATION,
        "beta": float(analysis.beta_total),
        "resonances": entries,
    }


# -- winding -----------------------------------------------------------------------

# Edge transform bounded over every admissible 108-turn winding-pack layout.
#
#     python -m w7x_twin winding [layers ...]

WINDING_OUT = Path("results/equilibrium/winding_pack.json")
#: Turns of the non-planar packs, published.
NON_PLANAR_TURNS = 108
#: Aspect ratio a coil casing admits. A pack far from square does not fit the casing
#: cross-section, which is what bounds the layouts to the ones tried.
MAX_ASPECT = 3.0
#: Layer the island trace is launched across, and the resonance it carries.
LAYER = (0.985, 1.30)
NUM_LINES = 40
WINDING_TURNS = 300


def layouts(turns: int, max_aspect: float = MAX_ASPECT) -> list[tuple[int, int]]:
    """Rectangular layouts of a turn count whose cross-section a casing admits."""
    out = []
    for layers in range(1, turns + 1):
        if turns % layers:
            continue
        per_layer = turns // layers
        aspect = max(layers, per_layer) / min(layers, per_layer)
        if aspect <= max_aspect:
            out.append((layers, per_layer))
    return out


def run_winding() -> int:
    wanted = args(int)
    admissible = layouts(NON_PLANAR_TURNS)
    if wanted:
        admissible = [pair for pair in admissible if pair[0] in wanted]

    twin = _common.twin()
    print(f"{twin.geometry}")
    print(
        f"{NON_PLANAR_TURNS} turns at a {1e3 * coil_geometry.TURN_PITCH_M:.1f} mm pitch "
        f"admit {len(admissible)} layouts inside an aspect ratio of {MAX_ASPECT:.0f}: "
        + ", ".join(f"{a} x {b}" for a, b in admissible)
    )
    assumed = (
        coil_geometry.NON_PLANAR_PACK.layers,
        coil_geometry.NON_PLANAR_PACK.turns_per_layer,
    )
    print(f"the package assumes {assumed[0]} x {assumed[1]}")

    table = Table(
        ("layout", ">12s"), ("width [mm]", "11.1f"), ("height [mm]", "12.1f"),
        ("iota edge", "11.5f"), ("island [mm]", "12.1f"),
    )
    print()
    table.begin()

    rows = []
    cases = [
        (f"{layers} x {per_layer}",
         coil_geometry.WindingPack(layers=layers, turns_per_layer=per_layer),
         f"coils.w7x_pack{layers}x{per_layer}")
        for layers, per_layer in admissible
    ]
    # The pack the package actually carries, whose pitches come from the released CAD
    # envelope rather than the 17.5 mm inference the hypothetical layouts share.
    cases.append(("CAD envelope", coil_geometry.NON_PLANAR_PACK, "coils.w7x_packcad"))
    for label, pack, filename in cases:
        layers, per_layer = pack.layers, pack.turns_per_layer
        path = twin.data_dir / filename
        if not path.exists():
            build_with_pack(twin, pack, path)
        equilibrium, iota_edge, island = solve_with(twin, path)
        width = (layers - 1) * pack.pitch + coil_geometry.CONDUCTOR_SIZE_M
        height = (per_layer - 1) * pack.pitch_across_turns + coil_geometry.CONDUCTOR_SIZE_M
        rows.append(
            {
                "label": label,
                "layers": layers,
                "turns_per_layer": per_layer,
                "width_m": width,
                "height_m": height,
                "iota_edge": iota_edge,
                "island_mm": island,
            }
        )
        table.row(label, 1e3 * width, 1e3 * height, iota_edge, island)

    iotas = np.array([r["iota_edge"] for r in rows])
    islands = np.array([r["island_mm"] for r in rows])
    print()
    print(
        f"across every admissible layout the edge transform spans {iotas.min():.5f} to "
        f"{iotas.max():.5f}, a range of "
        f"{100 * (iotas.max() - iotas.min()) / np.mean(iotas):.3f} per cent, and the island "
        f"spans {np.nanmin(islands):.1f} to {np.nanmax(islands):.1f} mm"
    )
    print(
        "so the layout is a bound of that size on the island the divertor is built around, "
        "not a free parameter"
    )

    write_record(
        WINDING_OUT,
        {
            "turns": NON_PLANAR_TURNS,
            "pitch_m": coil_geometry.TURN_PITCH_M,
            "conductor_size_m": coil_geometry.CONDUCTOR_SIZE_M,
            "max_aspect": MAX_ASPECT,
            "assumed_layout": list(assumed),
            "layouts": rows,
            "iota_edge_span": [float(iotas.min()), float(iotas.max())],
            "island_span_mm": [float(np.nanmin(islands)), float(np.nanmax(islands))],
        },
        geometry=twin.geometry,
    )
    return 0


def build_with_pack(twin: Twin, pack, path: Path) -> None:
    """Write a finite-build coils file using one winding-pack layout."""
    original = coil_geometry.NON_PLANAR_PACK
    coil_geometry.NON_PLANAR_PACK = pack
    try:
        coil_geometry.write_finite_build_coils_file(
            twin.data_dir / "coils.w7x", path
        )
    finally:
        coil_geometry.NON_PLANAR_PACK = original


def solve_with(reference: Twin, coils_path: Path) -> tuple[object, float, float]:
    """Edge transform and island width of the machine built with one coils file."""
    twin = Twin(
        data_dir=reference.data_dir, cache_dir=reference.cache_dir,
        coils_file=coils_path.name, verbose=False,
    )
    equilibrium = twin.solve(twin.state("standard"), SCAN)
    iota_edge = float(np.asarray(equilibrium.wout.iotaf)[-1])

    vacuum = field.VacuumField(twin.response, twin.state("standard").currents)
    starts, r_axis, z_axis, _ = fieldlines.fan_starts(
        vacuum, equilibrium.wout, LAYER, NUM_LINES
    )
    section, _ = fieldlines.trace(
        vacuum, starts, np.full(starts.shape, z_axis), turns=WINDING_TURNS, plane_phi=0.0
    )
    width, _ = fieldlines.midplane_island_span(section, r_axis, z_axis)
    return equilibrium, iota_edge, 1e3 * width


# -- ensemble ----------------------------------------------------------------------

# Machine quantities as Sobol-sampled tolerance intervals with their sampling error.
#
#     python -m w7x_twin ensemble [configuration] [count]

ENSEMBLE_OUT = Path("results/equilibrium/ensemble.json")


def run_ensemble() -> int:
    configuration = arg(1, default="standard")
    count = arg(2, int, 128)

    twin = _common.twin()
    tolerances = ensemble.Tolerances()
    print(f"{configuration}: {count} samples on a Sobol sequence")
    print(f"  {twin.geometry}")
    print(
        f"  per-circuit current {100 * tolerances.circuit_current:.2f} %, "
        f"common mode {100 * tolerances.common_mode:.2f} %, "
        f"temperature {100 * tolerances.temperature:.1f} %, "
        f"density {100 * tolerances.density:.1f} %"
    )
    print(
        f"  profile exponent {100 * tolerances.temperature_exponent:.1f} %, "
        f"toroidal flux {100 * tolerances.toroidal_flux:.2f} %, "
        f"heating power {100 * tolerances.heating_power:.1f} %, "
        f"a quoted tolerance spanning {tolerances.covers_sigma:g} sigma"
    )

    started = time.monotonic()
    result = ensemble.run(
        twin,
        configuration,
        count=count,
        tolerances=tolerances,
        resolution=SCAN,
        profiles=kinetics.HIGH_PERFORMANCE,
    )
    elapsed = time.monotonic() - started
    solved = len(result.samples["iota_edge"])
    print(
        f"  {solved} converged in {elapsed:.0f} s "
        f"({elapsed / max(solved, 1):.1f} s each), {result.failures} failed"
    )

    print()
    table = Table(
        ("quantity", "22s"), ("median", "12.5g"), ("+-", "9.2g"), ("5th", "12.5g"),
        ("95th", "12.5g"), ("spread", "11.4g"), ("relative", ">9s"),
    )
    table.begin()
    rows = []
    for label, unit, stats in result.rows():
        median = stats["median"]
        relative = stats["standard_deviation"] / abs(median) if median else float("nan")
        table.row(
            f"{label} [{unit}]" if unit else label, median, stats["median_error"],
            stats["percentile_5"], stats["percentile_95"],
            stats["standard_deviation"], f"{100 * relative:8.3f}%",
        )
        rows.append({"quantity": label, "unit": unit, **stats})

    write_record(
        ENSEMBLE_OUT,
        {
            "configuration": configuration,
            "samples": solved,
            "failures": result.failures,
            "seed": result.seed,
            "sampling": "scrambled Sobol",
            "dimensions": list(ensemble.DIMENSIONS),
            "tolerances": dataclasses.asdict(tolerances),
            "quantities": rows,
        },
        geometry=twin.geometry,
    )
    print(
        "The +- column is the bootstrap standard error of the median, which bounds the "
        "digits the interval supports."
    )
    return 0


# -- spec --------------------------------------------------------------------------

# Stepped-pressure scan over volume counts and interface placements about the resonance.
#
#     python -m w7x_twin spec [volumes ...]

SPEC_OUT = Path("results/equilibrium/spec.json")
CONFIGURATION = "op12a_22ka_mimic"
#: Volume counts the pressure jump is reduced over.
VOLUMES = (4, 6, 8, 12, 16)
#: The Newton converges to machine precision at these spectra and stalls near 1e-3
#: from Ntor 4 upward at any placement, and the resonant harmonic is one per period,
#: inside Ntor 2, so the scan runs where the force balance actually closes.
MPOL, NTOR = 6, 2
RADIAL = 8
RESTARTS = 4
#: Stop restarting when a step improves the residual by less than this factor.
IMPROVEMENT = 0.95
#: Force imbalance SPEC is asked to reach, and below which a solve is an equilibrium.
FORCE_TOLERANCE = 1.0e-10
#: Half-width in normalised flux of the volume the island is left inside.
BRACKET = 0.05


def solve(extension: str) -> tuple[int, float, float]:
    started = time.monotonic()
    completed = subprocess.run(
        [str(SPEC), extension], cwd=SPEC_WORK, capture_output=True, text=True, check=False
    )
    residual = stepped_pressure.force_residual(SPEC_WORK / f"{extension}.sp.h5")
    return completed.returncode, residual, time.monotonic() - started


def resonant_interfaces(count: int, resonant_s: float) -> tuple[float, ...]:
    """Interfaces with one on the resonant surface, closing the island by construction."""
    fixed = [resonant_s]
    remaining = count - 1 - len(fixed)
    if remaining <= 0:
        return (round(float(resonant_s), 6),)
    below = resonant_s
    above = 1.0 - resonant_s
    share = int(round(remaining * below / max(below + above, 1e-9)))
    share = min(max(share, 0), remaining)
    out = list(fixed)
    if share:
        out += list(np.linspace(resonant_s / (share + 1), resonant_s, share, endpoint=False))
    if remaining - share:
        out += list(np.linspace(resonant_s, 1.0, remaining - share + 2)[1:-1])
    return tuple(sorted(round(float(v), 6) for v in out if 0.0 < v < 1.0))


def noble_interfaces(
    count: int,
    s_grid: np.ndarray,
    iota_grid: np.ndarray,
    resonance: float,
    max_denominator: int = 24,
    clearance: float = 0.008,
    pinned: bool = False,
) -> tuple[float, ...]:
    """Interfaces on strongly-irrational transforms clear of every rational below
    ``max_denominator``; ``pinned`` puts one interface exactly on the resonance instead."""
    span = np.linspace(float(iota_grid[0]), float(iota_grid[-1]), 20001)
    inside = (span > iota_grid[0] + 1e-4) & (span < iota_grid[-1] - 1e-4)
    allowed = np.ones_like(span, dtype=bool) & inside
    for q in range(2, max_denominator + 1):
        # A quarter of the island-width scaling: wide enough that the measured
        # converging case clears it and the measured stalling cases do not.
        margin = 0.25 / q**2
        p_low = int(np.floor(span[0] * q)) - 1
        p_high = int(np.ceil(span[-1] * q)) + 1
        for p in range(p_low, p_high + 1):
            allowed &= np.abs(span - p / q) >= margin
    allowed_iotas = span[allowed]
    if allowed_iotas.size == 0:
        raise ValueError("no strongly-irrational transforms inside the profile span")

    if pinned:
        pair = [float(resonance)]
    else:
        below = allowed_iotas[allowed_iotas <= resonance - clearance]
        above = allowed_iotas[allowed_iotas >= resonance + clearance]
        if below.size == 0 or above.size == 0:
            raise ValueError("the bracketing pair has no strongly-irrational home")
        pair = [float(below[-1]), float(above[0])]

    remaining = count - 1 - len(pair)
    chosen = list(pair)
    # A sliver volume stalls the solve as surely as a near-rational interface, so the
    # fill keeps every pair at least half the even spacing apart.
    separation = 0.5 * float(span[-1] - span[0]) / count
    if remaining > 0:
        targets = np.linspace(span[0], span[-1], remaining + 2)[1:-1]
        for target in targets:
            candidates = allowed_iotas[
                np.all(
                    np.abs(allowed_iotas[:, None] - np.array(chosen)[None, :])
                    >= separation,
                    axis=1,
                )
            ]
            if candidates.size == 0:
                continue
            chosen.append(float(candidates[np.argmin(np.abs(candidates - target))]))
    order = np.argsort(iota_grid)
    placed = np.interp(sorted(chosen), iota_grid[order], s_grid[order])
    return tuple(round(float(v), 6) for v in placed if 0.02 < v < 0.99)


def bracketing_interfaces(count: int, resonant_s: float) -> tuple[float, ...]:
    """Interfaces with two bracketing the resonance, leaving the island a volume to form in."""
    inner = max(resonant_s - BRACKET, 1.0 / (2 * count))
    outer = min(resonant_s + BRACKET, 1.0 - 1.0 / (2 * count))
    fixed = [inner, outer]
    remaining = count - 1 - len(fixed)
    if remaining <= 0:
        return tuple(sorted(fixed))
    # Spread the rest over the two gaps in proportion to their width.
    below = max(inner, 1e-3)
    above = 1.0 - outer
    share = int(round(remaining * below / max(below + above, 1e-9)))
    share = min(max(share, 0), remaining)
    out = list(fixed)
    if share:
        out += list(np.linspace(inner / (share + 1), inner, share, endpoint=False))
    if remaining - share:
        out += list(
            np.linspace(outer, 1.0, remaining - share + 2)[1:-1]
        )
    return tuple(sorted(round(float(v), 6) for v in out if 0.0 < v < 1.0))


def run_spec() -> int:
    counts = tuple(args(int)) or VOLUMES
    if not SPEC.exists():
        raise SystemExit(f"no SPEC binary at {SPEC}")
    SPEC_WORK.mkdir(parents=True, exist_ok=True)

    twin = _common.twin()
    output = twin.solve_profiles(CONFIGURATION, kinetics.HIGH_PERFORMANCE, REFERENCE)
    transform = np.asarray(output.wout.iotaf)
    resonant = RESONANCE[0] / RESONANCE[1]
    grid = np.linspace(0.0, 1.0, len(transform))
    crossings = np.where(np.diff(np.sign(transform - resonant)))[0]
    if crossings.size == 0:
        raise SystemExit(
            f"the transform runs {transform.min():.5f} to {transform.max():.5f} and does "
            f"not cross {resonant:.5f}"
        )
    index = int(crossings[0])
    resonant_s = float(
        np.interp(resonant, transform[index:index + 2][::int(np.sign(
            transform[index + 1] - transform[index]) or 1)],
            grid[index:index + 2][::int(np.sign(
                transform[index + 1] - transform[index]) or 1)])
    )
    print(f"{twin.geometry}")
    print(
        f"{CONFIGURATION}: transform {transform[0]:.5f} to {transform[-1]:.5f}, crossing "
        f"{RESONANCE[0]:.0f}/{RESONANCE[1]:.0f} at s = {resonant_s:.4f}"
    )

    table = Table(
        ("vols", "5d"), ("interfaces", ">16s"), ("placed at", ">32s"),
        ("residual", "12.4e"), ("restarts", "6d"), ("island [mm]", "12.2f"),
        ("seconds", "8.1f"),
    )
    print()
    table.begin()

    rows = []
    for count, (placement, pinned) in [
        (count, pair)
        for count in counts
        for pair in (("on the resonance", True), ("bracketing it", False))
    ]:
        interfaces = noble_interfaces(
            count, grid, transform, resonant, pinned=pinned
        )
        base = f"spec_{CONFIGURATION}_v{count}_{placement.split()[0]}"
        stepped_pressure.write_input(
            output, SPEC_WORK, f"{base}_0", interfaces, mpol=MPOL, ntor=NTOR,
            radial_resolution=RADIAL, constraint=1, initialise=0,
        )
        code, residual, seconds = solve(f"{base}_0")
        best, best_extension, steps = residual, f"{base}_0", 1
        previous = f"{base}_0"
        for step in range(1, RESTARTS + 1):
            end = SPEC_WORK / f"{previous}.sp.end"
            if not end.is_file() or not np.isfinite(best):
                break
            extension = f"{base}_{step}"
            shutil.copy(end, SPEC_WORK / f"{extension}.sp")
            code, residual, elapsed = solve(extension)
            seconds += elapsed
            steps += 1
            if not np.isfinite(residual) or residual >= IMPROVEMENT * best:
                if np.isfinite(residual) and residual < best:
                    best, best_extension = residual, extension
                break
            best, best_extension, previous = residual, extension, extension

        island = float("nan")
        try:
            section = stepped_pressure.poincare_section(SPEC_WORK / f"{best_extension}.sp.h5")
            r, z = section["R"][:, :, 0], section["Z"][:, :, 0]
            spread = r.max(axis=1) - r.min(axis=1)
            centre = int(np.argmin(spread))
            axis_r, axis_z = float(r[centre].mean()), float(z[centre].mean())
            found = stepped_pressure.island_at_resonance(
                r, z, axis_r, axis_z, RESONANCE[0] / RESONANCE[1]
            )
            island = 1e3 * float(found["width_m"])
        except (OSError, KeyError, ValueError, IndexError):
            pass

        rows.append(
            {
                "volumes": count,
                "placement": placement,
                "interfaces": list(interfaces),
                "force_residual": best,
                "restarts": steps - 1,
                "island_mm": island,
                "seconds": seconds,
            }
        )
        shown = ", ".join(f"{v:.3f}" for v in interfaces[:4])
        if len(interfaces) > 4:
            shown += ", ..."
        table.row(count, placement, shown, best, steps - 1, island, seconds)

    finite = [r for r in rows if np.isfinite(r["force_residual"])]
    print()
    if finite:
        best = min(finite, key=lambda r: r["force_residual"])
        met = [r for r in finite if r["force_residual"] <= FORCE_TOLERANCE]
        print(
            f"the residual reaches {best['force_residual']:.4e} at {best['volumes']} volumes "
            f"against a {FORCE_TOLERANCE:.0e} tolerance, and {len(met)} of {len(rows)} cases "
            f"meet it; the rest span {min(r['force_residual'] for r in finite if r not in met):.4e} "
            f"to {max(r['force_residual'] for r in finite):.4e}"
            if len(met) < len(finite) else
            f"the residual reaches {best['force_residual']:.4e} at {best['volumes']} volumes, "
            f"and every case meets the {FORCE_TOLERANCE:.0e} tolerance"
        )
        for placement in ("on the resonance", "bracketing it"):
            here = [r for r in finite if r["placement"] == placement]
            if not here:
                continue
            best_here = min(here, key=lambda r: r["force_residual"])
            islands = [r["island_mm"] for r in here if np.isfinite(r["island_mm"])]
            print(
                f"with the interfaces {placement} the residual reaches "
                f"{best_here['force_residual']:.4e} at {best_here['volumes']} volumes"
                + (
                    f" and the island spans {min(islands):.2f} to {max(islands):.2f} mm"
                    if islands else " and no island forms"
                )
            )
        print(
            "an ideal interface forbids reconnection across itself, so one placed on the "
            "resonance closes the island by construction: the two placements are the same "
            "equilibrium asked two different questions"
        )

    write_record(
        SPEC_OUT,
        {
            "configuration": CONFIGURATION,
            "resonance": list(RESONANCE),
            "resonant_s": resonant_s,
            "mpol": MPOL, "ntor": NTOR, "radial_resolution": RADIAL,
            "cases": rows,
        },
        geometry=twin.geometry,
    )
    return 0


# -- stepped -----------------------------------------------------------------------

# A stepped-pressure equilibrium carrying its own 5/6 island, held against the vacuum trace.
#
#     python -m w7x_twin stepped [configuration] [temperature scale]

STEPPED_OUT = Path("results/equilibrium/island_equilibrium.json")

#: Transform values the interior interfaces are placed at, chosen away from low-order
#: rationals and bracketing the resonance.
INTERFACE_FLUX = (0.18, 0.45, 0.80)


def run_stepped() -> int:
    key = arg(1, default="op12a_22ka_mimic")
    scale = arg(2, float, 1.0)
    if not SPEC.exists():
        raise SystemExit(f"no SPEC executable at {SPEC}")

    twin = _common.twin()
    profiles = kinetics.HIGH_PERFORMANCE.scaled(scale)
    output = twin.solve_profiles(key, profiles, REFERENCE)
    iota = np.asarray(output.wout.iotaf)
    resonance = RESONANCE[0] / RESONANCE[1]
    print(f"{twin.geometry}")
    print(
        f"{key}: transform {iota[0]:.5f} to {iota[-1]:.5f}, "
        f"<beta> {100 * float(output.wout.betatotal):.3f} %"
    )
    if not (iota.min() <= resonance <= iota.max()):
        raise SystemExit(
            f"{RESONANCE[0]}/{RESONANCE[1]} = {resonance:.5f} is outside the transform "
            f"profile {iota.min():.5f} to {iota.max():.5f}; no interior island to resolve"
        )
    s_resonant = float(
        np.interp(resonance, iota, np.linspace(0.0, 1.0, len(iota)))
        if iota[-1] > iota[0]
        else np.interp(-resonance, -iota, np.linspace(0.0, 1.0, len(iota)))
    )
    print(
        f"{RESONANCE[0]}/{RESONANCE[1]} = {resonance:.5f} sits at s = {s_resonant:.4f}"
    )

    written = stepped_pressure.write_input(
        output, SPEC_WORK, f"w7x_{key}", INTERFACE_FLUX
    )
    print(
        f"{written.num_volumes} volumes, interfaces at transform "
        + ", ".join(f"{value:.4f}" for value in written.interface_transforms)
    )
    print(
        "  normalised flux " + ", ".join(f"{value:.4f}" for value in written.interface_flux)
    )
    print("  pressure [Pa] " + ", ".join(f"{value:.1f}" for value in written.pressures_pa))
    print(f"wrote {written.path}")

    if "--continuation" in sys.argv:
        print("\nsolving up a pressure ramp, each step from the one before")
        steps = stepped_pressure.continuation(written, SPEC)
        best = min(
            (step for step in steps if np.isfinite(step["force_residual"])),
            key=lambda step: step["force_residual"],
            default=None,
        )
        final = steps[-1] if steps else None
        print(
            f"the full-pressure step ended at "
            f"{final['force_residual']:.3e}" if final else "no step completed"
        )
        record = {
            "geometry": twin.geometry.as_dict(),
            "configuration": key,
            "continuation": steps,
            "best_residual": best["force_residual"] if best else float("nan"),
        }
        write_record(STEPPED_OUT.with_name("island_continuation.json"), record)
        return 0

    result = written.path.with_suffix(".sp.h5")
    if result.exists() and "--rerun" not in sys.argv:
        print(f"\n{result} is already there; pass --rerun to solve it again")

        class _Done:
            returncode = 0
            stdout = ""

        completed = _Done()
        elapsed = 0.0
    else:
        started = time.monotonic()
        completed = stepped_pressure.run(written, SPEC)
        elapsed = time.monotonic() - started
        tail = "\n".join(completed.stdout.strip().splitlines()[-12:])
        print(f"\nSPEC exited {completed.returncode} after {elapsed:.0f} s")
        print(tail)

    record = {
        "geometry": twin.geometry.as_dict(),
        "configuration": key,
        "temperature_scale": scale,
        "beta": float(output.wout.betatotal),
        "iota_axis": float(iota[0]),
        "iota_edge": float(iota[-1]),
        "resonance": f"{RESONANCE[0]}/{RESONANCE[1]}",
        "resonant_flux": s_resonant,
        "volumes": written.num_volumes,
        "interface_transforms": list(written.interface_transforms),
        "interface_flux": list(written.interface_flux),
        "volume_pressure_pa": list(written.pressures_pa),
        "returncode": completed.returncode,
        "seconds": elapsed,
    }

    if completed.returncode == 0 and result.exists():
        section = stepped_pressure.poincare_section(result)
        # The section at the first toroidal plane: (trajectory, return).
        r, z = section["R"][:, :, 0], section["Z"][:, :, 0]
        spread = r.max(axis=1) - r.min(axis=1)
        centre = int(np.argmin(spread))
        axis_r, axis_z = float(r[centre].mean()), float(z[centre].mean())
        print(
            f"\nPoincare section {r.shape}, axis trajectory {centre} at "
            f"R = {axis_r:.5f} m"
        )
        # A stalled Newton still writes a section, and xspec exits zero either way, so
        # the island below is only an equilibrium's island where the force balance
        # closed. The record carries the residual so the two cannot be confused.
        residual = stepped_pressure.force_residual(result)
        record["force_residual"] = residual
        record["converged"] = bool(np.isfinite(residual) and residual <= FORCE_TOLERANCE)
        print(
            f"force residual {residual:.4e} against a {FORCE_TOLERANCE:.0e} tolerance: "
            + ("in force balance" if record["converged"] else "NOT in force balance, so the "
               "island below is measured on a section the Newton left behind")
        )
        widest = stepped_pressure.island_width(r, z, axis_r, axis_z)
        at_resonance = stepped_pressure.island_at_resonance(
            r, z, axis_r, axis_z, resonance
        )
        record["poincare_shape"] = list(section["R"].shape)
        record["axis_r_m"] = axis_r
        record["stepped_widest_plateau"] = widest
        record["stepped_at_resonance"] = at_resonance
        print(
            f"stepped-pressure solution: widest winding plateau spans "
            f"{1e3 * widest['width_m']:.1f} mm over {widest['trajectories']} "
            f"trajectories at a winding of {widest['winding']:.5f}"
        )
        print(
            f"  at {RESONANCE[0]}/{RESONANCE[1]}: "
            f"{1e3 * at_resonance['width_m']:.1f} mm over "
            f"{at_resonance['trajectories']} trajectories"
        )

        # The same measurement on the traced vacuum field, so the two are comparable.
        vacuum = VacuumField(twin.response, twin.state(key).currents)
        traced_axis_r, traced_axis_z = fieldlines.find_axis(vacuum)
        r_lcfs, _ = diagnostics.boundary_cut(output.wout, 0.0)
        # A narrow, densely sampled band about the resonant radius: an island a few
        # millimetres wide is missed entirely by a fan spanning the whole minor radius.
        half = r_lcfs.max() - traced_axis_r
        fraction = float(np.sqrt(s_resonant))
        probe = traced_axis_r + np.linspace(fraction - 0.09, fraction + 0.09, 90) * half
        poincare, _ = fieldlines.trace(
            vacuum, probe, np.full(probe.shape, traced_axis_z), turns=600, plane_phi=0.0
        )
        counts = [
            int(np.count_nonzero(poincare.line_index == line))
            for line in range(int(poincare.line_index.max()) + 1)
        ]
        keep = min(counts)
        traced_r = np.array(
            [poincare.r[poincare.line_index == line][:keep] for line in range(len(counts))]
        )
        traced_z = np.array(
            [poincare.z[poincare.line_index == line][:keep] for line in range(len(counts))]
        )
        vacuum_found = stepped_pressure.island_at_resonance(
            traced_r, traced_z, traced_axis_r, traced_axis_z, resonance
        )
        record["traced"] = vacuum_found
        print(
            f"traced vacuum field: the {RESONANCE[0]}/{RESONANCE[1]} island spans "
            f"{1e3 * vacuum_found['width_m']:.1f} mm over "
            f"{vacuum_found['trajectories']} trajectories, whose winding "
            f"{vacuum_found['winding']:.5f} implies a transform locked to "
            f"{resonance:.5f}"
        )
    else:
        print("\nSPEC produced no output file; the record carries the input it was given")

    write_record(STEPPED_OUT, record)
    return 0 if completed.returncode == 0 else 1


# -- from koeberl -----------------------------------------------------------------

# The twin against Koeberl et al., MaxEnt 2023 (Zenodo 8095035), the published
# reconstruction with consistent uncertainties.
#
#     python -m w7x_twin koeberl

KOEBERL_WOUT = Path(
    "data/benchmarks/koeberl/Koeberl_MaxEnt_2023_data/data/eval_2000/wout_vmec_aux.nc"
)
KOEBERL_OUT = Path("results/benchmarks/koeberl.json")


def run_koeberl() -> int:
    import netCDF4

    if not KOEBERL_WOUT.exists():
        raise SystemExit(
            f"no reference at {KOEBERL_WOUT}; download Zenodo record 8095035 there"
        )
    reference = netCDF4.Dataset(str(KOEBERL_WOUT))

    def scalar(name: str) -> float:
        return float(np.ravel(np.asarray(reference.variables[name][:]))[-1])

    extcur = np.ravel(np.asarray(reference.variables["extcur"][:]))
    presf = np.asarray(reference.variables["presf"][:], dtype=float)
    iotaf = np.asarray(reference.variables["iotaf"][:], dtype=float)
    raxis = np.asarray(reference.variables["raxis_cc"][:], dtype=float)
    s_grid = np.linspace(0.0, 1.0, len(presf))

    twin = _common.twin()
    print(f"{twin.geometry}")
    print(
        "Koeberl reconstruction: extcur "
        + ", ".join(f"{value:.0f}" for value in extcur[:7])
        + f" A, beta {100 * scalar('betatotal'):.2f} %"
    )

    state = twin.state(
        "standard", scenario=Scenario.from_pressure_spline(s_grid[::4], presf[::4])
    )
    keys = ("npc1", "npc2", "npc3", "npc4", "npc5", "pca", "pcb")
    state = twin.with_currents(
        state, **{key: float(extcur[index]) for index, key in enumerate(keys)}
    )
    state.toroidal_flux_wb = float(
        np.sign(state.toroidal_flux_wb) * abs(scalar("phi"))
    )
    output = twin.solve(state, SCAN)
    wout = output.wout

    ours_axis = float(np.sum(np.asarray(wout.raxis_cc)))
    ref_axis = float(np.sum(raxis))
    our_iota = np.asarray(wout.iotaf)

    rows = []
    for name, ours, theirs in (
        ("axis R at phi = 0 [m]", ours_axis, ref_axis),
        ("iota on axis", float(our_iota[0]), float(iotaf[0])),
        ("iota at the edge", float(our_iota[-1]), float(iotaf[-1])),
        ("minor radius [m]", float(wout.Aminor_p), scalar("Aminor_p")),
        ("plasma volume [m3]", float(wout.volume_p), scalar("volume_p")),
        ("beta [%]", 100 * float(wout.betatotal), 100 * scalar("betatotal")),
    ):
        departure = 100.0 * (ours - theirs) / theirs if theirs else float("nan")
        rows.append(
            {"quantity": name, "computed": ours, "reconstructed": theirs,
             "departure_percent": departure}
        )
        print(f"  {name:24s} {ours:12.5f} against {theirs:12.5f} ({departure:+.2f} %)")

    write_record(
        KOEBERL_OUT,
        {
            "reference": "Koeberl et al., MaxEnt 2023, Zenodo 8095035",
            "extcur_a": [float(v) for v in extcur],
            "rows": rows,
        },
        geometry=twin.geometry,
    )
    return 0
