"""Machine quantities as tolerance intervals: scrambled-Sobol sampling with a common-mode
current component, solved per sample, quantile error by bootstrap."""

from __future__ import annotations

import dataclasses

import numpy as np

from w7x_twin.mhd import diagnostics
from w7x_twin.mhd.equilibrium import SCAN, MachineState, Resolution, Scenario, Twin


@dataclasses.dataclass(frozen=True)
class Tolerances:
    """Input spreads as relative standard deviations, per-circuit and common-mode;
    ``covers_sigma`` maps a quoted bound onto a deviation."""

    circuit_current: float = 1.0e-3
    common_mode: float = 5.0e-4
    temperature: float = 0.05
    density: float = 0.03
    temperature_exponent: float = 0.08
    toroidal_flux: float = 1.0e-3
    #: Absorbed heating power, which the sources state to five per cent and which
    #: every power-balance quantity is conditioned on.
    heating_power: float = 0.05
    covers_sigma: float = 1.0

    def sigma(self, value: float) -> float:
        return value / max(self.covers_sigma, 1e-12)


#: Quantities reported with an interval. The last four run the pipeline past the
#: equilibrium: the power balance on the sample's own perturbed profiles at its own
#: perturbed heating power, and the Redl bootstrap on its own geometry.
QUANTITIES: tuple[tuple[str, str, str], ...] = (
    ("iota_axis", "transform on axis", ""),
    ("iota_edge", "transform at edge", ""),
    ("mirror_percent", "mirror term", "%"),
    ("magnetic_well_depth", "magnetic well", ""),
    ("plasma_volume_m3", "plasma volume", "m^3"),
    ("b_axis_t", "field on axis", "T"),
    ("aspect_ratio", "aspect ratio", ""),
    ("beta_total", "beta", ""),
    ("stored_energy_j", "stored energy", "J"),
    ("balance_energy_j", "stored energy, power balance", "J"),
    ("confinement_time_s", "confinement time", "s"),
    ("electron_temperature_axis_ev", "electron temperature on axis", "eV"),
    ("bootstrap_current_a", "bootstrap current, Redl", "A"),
)

#: Order of the sampled dimensions, which is what the Sobol sequence indexes.
DIMENSIONS = (
    "npc1", "npc2", "npc3", "npc4", "npc5", "pca", "pcb",
    "common_mode", "temperature", "density", "temperature_exponent", "toroidal_flux",
    "heating_power",
)

#: Heating power the balance quantities are conditioned on, perturbed per sample.
HEATING_W = 5.0e6
#: Confinement anchor of the per-sample balance.
ENHANCEMENT = 1.4


def normal_samples(count: int, seed: int) -> tuple[np.ndarray, int]:
    """A scrambled Sobol sequence mapped to standard normals, the count raised to a power of two."""
    from scipy.stats import norm, qmc

    exponent = max(4, int(np.ceil(np.log2(max(count, 2)))))
    engine = qmc.Sobol(d=len(DIMENSIONS), scramble=True, seed=seed)
    unit = engine.random_base2(exponent)
    # Keep the mapping away from the tails of the inverse normal at the unit endpoints.
    return norm.ppf(np.clip(unit, 1e-6, 1 - 1e-6)), 2**exponent


@dataclasses.dataclass
class EnsembleResult:
    """Samples of each quantity, and the interval and sampling error they imply."""

    samples: dict[str, np.ndarray]
    draws: np.ndarray
    failures: int
    seed: int

    def interval(self, key: str, lower: float = 5.0, upper: float = 95.0) -> dict:
        """Median, percentiles and spread, each with its bootstrap standard error."""
        values = self.samples[key]
        rng = np.random.default_rng(self.seed + 1)
        resampled = values[
            rng.integers(0, len(values), size=(512, len(values)))
        ]
        return {
            "median": float(np.median(values)),
            "median_error": float(np.std(np.median(resampled, axis=1))),
            "percentile_5": float(np.percentile(values, lower)),
            "percentile_5_error": float(
                np.std(np.percentile(resampled, lower, axis=1))
            ),
            "percentile_95": float(np.percentile(values, upper)),
            "percentile_95_error": float(
                np.std(np.percentile(resampled, upper, axis=1))
            ),
            "standard_deviation": float(np.std(values)),
            "standard_deviation_error": float(np.std(np.std(resampled, axis=1))),
        }

    def rows(self) -> list[tuple[str, str, dict]]:
        return [
            (label, unit, self.interval(key))
            for key, label, unit in QUANTITIES
            if key in self.samples and self.samples[key].size
        ]


def perturbed_state(
    twin: Twin,
    configuration: str,
    draw: np.ndarray,
    tolerances: Tolerances,
    scenario: Scenario | None = None,
    label: str = "",
) -> MachineState:
    """One actuator setting drawn from the tolerances, per-circuit plus common-mode."""
    state = twin.state(configuration, scenario=scenario)
    currents = np.asarray(state.currents, dtype=float)
    powered = currents != 0.0

    # Seven circuit dimensions are sampled; a coils file carrying more leaves the rest
    # unperturbed rather than reusing a draw.
    independent = np.zeros(len(currents))
    sampled = min(len(currents), 7)
    independent[:sampled] = (
        tolerances.sigma(tolerances.circuit_current) * draw[:sampled]
    )
    common = tolerances.sigma(tolerances.common_mode) * draw[7]
    factors = 1.0 + independent + common
    perturbed = np.where(powered, currents * factors, currents)

    flux = abs(state.toroidal_flux_wb) * (
        1.0 + tolerances.sigma(tolerances.toroidal_flux) * draw[11]
    )
    return dataclasses.replace(
        state,
        currents=perturbed,
        toroidal_flux_wb=twin.toroidal_flux_for(perturbed, flux),
        label=f"{state.label} {label}",
    )


def run(
    twin: Twin,
    configuration: str = "standard",
    count: int = 128,
    tolerances: Tolerances = Tolerances(),
    resolution: Resolution = SCAN,
    profiles=None,
    seed: int = 20260725,
    verbose: bool = True,
) -> EnsembleResult:
    """Solve a Sobol sample of the input space and collect the derived quantities."""
    draws, actual = normal_samples(count, seed)
    if verbose and actual != count:
        print(f"  Sobol balance raises {count} samples to {actual}")

    collected: dict[str, list[float]] = {key: [] for key, _, _ in QUANTITIES}
    failures = 0

    for index in range(actual):
        draw = draws[index]
        scenario = None
        if profiles is not None:
            scaled = dataclasses.replace(
                profiles,
                electron_temperature_axis_ev=profiles.electron_temperature_axis_ev
                * (1.0 + tolerances.sigma(tolerances.temperature) * draw[8]),
                ion_temperature_axis_ev=profiles.ion_temperature_axis_ev
                * (1.0 + tolerances.sigma(tolerances.temperature) * draw[8]),
                density_axis_m3=profiles.density_axis_m3
                * (1.0 + tolerances.sigma(tolerances.density) * draw[9]),
                temperature_exponent=profiles.temperature_exponent
                * (1.0 + tolerances.sigma(tolerances.temperature_exponent) * draw[10]),
            )
            knots_s, knots_p = scaled.pressure_spline()
            scenario = Scenario.from_pressure_spline(knots_s, knots_p)
        state = perturbed_state(
            twin, configuration, draw, tolerances, scenario, f"sample {index}"
        )
        try:
            output = twin.solve(state, resolution, cache=False)
        except RuntimeError:
            failures += 1
            continue
        summary = diagnostics.analyse(output)
        for key, _, _ in QUANTITIES:
            if hasattr(summary, key):
                collected[key].append(float(getattr(summary, key)))
        if profiles is not None:
            # The same sample carried down the pipeline: the balance on its own
            # perturbed profiles at its own perturbed heating power, and the Redl
            # bootstrap on its own geometry, so the intervals below are conditioned
            # on everything the equilibrium ones are.
            try:
                from w7x_twin.plasma import current as plasma_current
                from w7x_twin.plasma import transport as plasma_transport
                from w7x_twin.plasma.current import enclosed_current_a

                power = HEATING_W * (
                    1.0 + tolerances.sigma(tolerances.heating_power) * draw[12]
                )
                balance = plasma_transport.solve(
                    output, scaled,
                    heating=plasma_transport.Heating(power_w=power),
                    model=plasma_transport.TransportModel(
                        renormalisation=ENHANCEMENT
                    ),
                )
                s_drive, drive = plasma_current.redl_jdotb(output, scaled)
                collected["balance_energy_j"].append(float(balance.stored_energy_j))
                collected["confinement_time_s"].append(
                    float(balance.confinement_time_s)
                )
                collected["electron_temperature_axis_ev"].append(
                    float(balance.electron_temperature_ev[0])
                )
                collected["bootstrap_current_a"].append(
                    enclosed_current_a(output, s_drive, drive)
                )
            except (RuntimeError, ValueError):
                pass
        if verbose and (index + 1) % 16 == 0:
            print(f"  {index + 1} of {actual} solved")

    return EnsembleResult(
        samples={key: np.array(values) for key, values in collected.items()},
        draws=draws,
        failures=failures,
        seed=seed,
    )
