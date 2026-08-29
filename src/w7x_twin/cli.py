"""Command-line dispatcher mapping each command to its :mod:`w7x_twin.analyses` entry point."""

from __future__ import annotations

import importlib
import sys

#: Command name to (analysis module, entry point, description, required external code).
COMMANDS: dict[str, tuple[str, str, str, str]] = {
    "fetch": ("data", "run_fetch", "download the coil, vessel and component data", ""),
    "equilibrium": ("equilibrium", "run_equilibrium", "derived quantities of every configuration", ""),
    "beta": ("equilibrium", "run_beta", "the beta scan and the Shafranov shift with it", ""),
    "figure": ("equilibrium", "run_figure", "the overview figure", ""),
    "islands": ("equilibrium", "run_islands", "traced islands, Poincare sections and connection lengths", ""),
    "transport": ("plasma", "run_transport", "power balance split into drift-kinetic and anomalous", ""),
    "efield": ("plasma", "run_efield", "the ambipolar radial electric field", ""),
    "bootstrap": ("plasma", "run_bootstrap", "bootstrap current by both routes, self-consistent", ""),
    "coupled": ("plasma", "run_coupled", "transport, bootstrap and equilibrium converged together", ""),
    "density": ("plasma", "run_density", "density profile from a particle balance", ""),
    "computed": ("plasma", "run_computed", "the power balance on computed inputs", ""),
    "deposition": ("plasma", "run_deposition", "electron-cyclotron and beam absorption", ""),
    "winding": ("equilibrium", "run_winding", "the edge transform over every admissible winding-pack layout", ""),
    "stability": ("equilibrium", "run_stability", "Mercier, ballooning and tearing", ""),
    "response": ("magnetics", "run_response", "the plasma field converged, and the island it leaves", ""),
    "exhaust": ("exhaust", "run_exhaust", "heat load on the targets and the approach to detachment", ""),
    "incidence": ("exhaust", "run_incidence", "field incidence per target element", ""),
    "strikes": ("exhaust", "run_strikes", "strikes resolved to the ten divertor units", ""),
    "migration": ("exhaust", "run_migration", "bootstrap current to strike-line position", ""),
    "recycling": ("exhaust", "run_recycling", "divertor neutral pressure and the losses it drives", ""),
    "errorfield": ("discharges", "run_errorfield", "the measured n = 1 error field and its load imbalance", ""),
    "symmetrise": ("discharges", "run_symmetrise", "the measured 1/1 and 2/2 corrections and the load they leave", ""),
    "profiles": ("discharges", "run_profiles", "the solved profiles against the digitised measured ones", ""),
    "intrinsic": ("discharges", "run_intrinsic", "the intrinsic error field as a coil deviation", ""),
    "trim-radius": ("discharges", "run_trim_radius", "the trim coil mounting radius pinned against the measured correction", ""),
    "discharge": ("discharges", "run_discharge", "the model against identified W7-X programmes", ""),
    "history": ("discharges", "run_history", "a discharge advanced through its heating waveform", ""),
    "transient": ("plasma", "run_transient", "a discharge as one transient solution, the layer closing the edge", ""),
    "ensemble": ("equilibrium", "run_ensemble", "machine quantities as intervals", ""),
    "cad": ("data", "run_cad", "the released CAD against the reconstructed geometry", ""),
    "cut-contours": ("data", "run_cut_contours", "recut the component contours onto the released CAD surfaces", ""),
    "page-error": ("data", "run_page_error", "what the page's grid and tracer resolve against the model", ""),
    "validate": ("data", "run_validate", "the verification record, non-zero if anything disagrees", ""),
    "turbulence": ("turbulence", "run_turbulence", "power balance with both channels computed", "stella"),
    "gyrokinetic": ("turbulence", "run_gyrokinetic", "linear gyrokinetic growth rates", "stella"),
    "growth-rate-grid": ("turbulence", "run_growth_rate_grid", "the growth-rate grid over surfaces and both gradients", "stella"),
    "saturation": ("turbulence", "run_saturation", "the saturation response from nonlinear runs", "stella"),
    "spec": ("equilibrium", "run_spec", "stepped-pressure residual and the island the interfaces bracket", "SPEC"),
    "koeberl": ("equilibrium", "run_koeberl", "the twin against the published equilibrium reconstruction", ""),
    "stepped": ("equilibrium", "run_stepped", "stepped-pressure equilibrium and the island it carries", "SPEC"),
    "export-geometry": ("data", "run_export_geometry", "geometry export for the rendered page", ""),
    "export-field": ("data", "run_export_field", "per-circuit field export for the rendered page", ""),
}


def usage() -> str:
    width = max(len(name) for name in COMMANDS)
    lines = ["usage: python -m w7x_twin <command> [arguments]", "", "commands:"]
    for name, (_, _, description, needs) in COMMANDS.items():
        suffix = f"  (needs {needs})" if needs else ""
        lines.append(f"  {name:<{width}}  {description}{suffix}")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(usage())
        return 0
    command = argv[0]
    if command not in COMMANDS:
        print(f"unknown command {command!r}\n", file=sys.stderr)
        print(usage(), file=sys.stderr)
        return 2

    module_name, function, _, _ = COMMANDS[command]
    module = importlib.import_module(f"w7x_twin.analyses.{module_name}")
    # Analyses read sys.argv; the command name is dropped and the rest passed through.
    sys.argv = [f"w7x_twin {command}", *argv[1:]]
    return int(getattr(module, function)() or 0)
