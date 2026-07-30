"""Power and particle balance, heating deposition, and the turbulent channel."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path

import numpy as np
import vmecpp

from w7x_twin.plasma.kinetics import ELEMENTARY_CHARGE, KineticProfiles

ELECTRON_MASS = 9.1093837015e-31
PROTON_MASS = 1.67262192369e-27
VACUUM_PERMITTIVITY = 8.8541878128e-12
SPEED_OF_LIGHT = 299792458.0


# -- from transport ---------------------------------------------------------------

PUBLISHED_ISS04_ENHANCEMENT = 1.4

#: Confinement relative to ISS04 across the range W7-X reports: 0.70 gas-fuelled, 1.40 at
#: the highest triple product.
MEASURED_ISS04_RANGE = (0.70, 1.40)


@dataclasses.dataclass
class Heating:
    """Deposited power and its radial distribution."""

    power_w: float = 5.0e6
    #: Gaussian deposition width in normalised toroidal flux; W7-X ECRH is central.
    deposition_width: float = 0.3
    #: Absorbed power per unit s from a computed deposition, which replaces the Gaussian
    #: when it is supplied. Sampled on ``deposition_s``.
    deposition_s: np.ndarray | None = None
    deposition_profile: np.ndarray | None = None

    @staticmethod
    def from_deposition(power_w: float, deposition) -> "Heating":
        """Heating whose radial form is a computed absorption profile."""
        return Heating(
            power_w=power_w,
            deposition_s=np.asarray(deposition.s, dtype=float),
            deposition_profile=np.asarray(deposition.profile_w, dtype=float),
        )

    def profile(self, s: np.ndarray) -> np.ndarray:
        if self.deposition_profile is not None and self.deposition_s is not None:
            return np.interp(
                np.asarray(s, dtype=float), self.deposition_s, self.deposition_profile
            )
        return self._gaussian(s)

    def _gaussian(self, s: np.ndarray) -> np.ndarray:
        """Normalised deposition density against ``s``, integrating to one over volume."""
        return np.exp(-((s / self.deposition_width) ** 2))


@dataclasses.dataclass
class TransportModel:
    """Heat diffusivity chi(r) = chi_0 (1 + edge_rise (r/a)^exponent), chi_0 solved to match ISS04."""

    edge_rise: float = 4.0
    exponent: float = 2.0
    #: ISS04 renormalisation. Unity is the international scaling itself; W7-X
    #: discharges are commonly reported at or slightly above it.
    renormalisation: float = 1.0
    #: Ion temperature as a fraction of the electron temperature. Electron-cyclotron
    #: heated W7-X plasmas run with the ions colder than the electrons.
    ion_fraction: float = 0.55

    def shape(self, r_normalised: np.ndarray) -> np.ndarray:
        return 1.0 + self.edge_rise * r_normalised**self.exponent


def iss04_confinement_time(
    minor_radius_m: float,
    major_radius_m: float,
    heating_power_w: float,
    line_density_m3: float,
    field_t: float,
    iota_two_thirds: float,
) -> float:
    """ISS04 stellarator confinement scaling with density in 1e19 m^-3 and power in MW, in seconds."""
    return (
        0.134
        * minor_radius_m**2.28
        * major_radius_m**0.64
        * (heating_power_w / 1.0e6) ** -0.61
        * (line_density_m3 / 1.0e19) ** 0.54
        * field_t**0.84
        * iota_two_thirds**0.41
    )


@dataclasses.dataclass
class TransportSolution:
    """Profiles obtained from the power balance, and the quantities behind them."""

    s: np.ndarray
    electron_temperature_ev: np.ndarray
    ion_temperature_ev: np.ndarray
    density_m3: np.ndarray
    pressure_pa: np.ndarray
    chi_m2_s: np.ndarray
    heat_flux_w: np.ndarray
    stored_energy_j: float
    confinement_time_s: float
    iss04_time_s: float
    heating_power_w: float
    #: Neoclassical part, computed from the drift-kinetic coefficients, and the
    #: remainder that the confinement scaling attributes to everything else.
    chi_neoclassical_m2_s: np.ndarray | None = None
    chi_anomalous_m2_s: np.ndarray | None = None
    #: Radiated power density and what it integrates to, when an impurity is carried.
    radiated_power_w_m3: np.ndarray | None = None
    radiated_power_w: float = 0.0
    carbon_fraction: float = 0.0
    #: The ion channel's own diffusivity and the exchange power feeding it, carried by
    #: the two-temperature solve alone.
    chi_ion_m2_s: np.ndarray | None = None
    exchange_power_w: float = 0.0

    @property
    def neoclassical_fraction(self) -> np.ndarray | None:
        if self.chi_neoclassical_m2_s is None:
            return None
        return self.chi_neoclassical_m2_s / np.maximum(self.chi_m2_s, 1e-30)

    @property
    def radiated_fraction(self) -> float:
        """Share of the heating power the plasma radiates rather than conducts out."""
        return self.radiated_power_w / max(self.heating_power_w, 1e-30)

    def as_kinetic_profiles(self) -> KineticProfiles:
        """Wrap the solved profiles so they can drive the equilibrium and bootstrap."""
        return SolvedProfiles(self)


class SolvedProfiles(KineticProfiles):
    """Kinetic profiles backed by a transport solution rather than by a formula."""

    def __init__(self, solution: TransportSolution) -> None:
        super().__init__(carbon_fraction=solution.carbon_fraction)
        self._solution = solution

    def density(self, s: np.ndarray) -> np.ndarray:
        return np.interp(np.clip(s, 0.0, 1.0), self._solution.s, self._solution.density_m3)

    def electron_temperature(self, s: np.ndarray) -> np.ndarray:
        return np.interp(
            np.clip(s, 0.0, 1.0), self._solution.s, self._solution.electron_temperature_ev
        )

    def ion_temperature(self, s: np.ndarray) -> np.ndarray:
        return np.interp(
            np.clip(s, 0.0, 1.0), self._solution.s, self._solution.ion_temperature_ev
        )


def _geometry(output: vmecpp.VmecOutput) -> tuple[np.ndarray, np.ndarray, float, float]:
    """(s, dV/dr, minor radius, major radius) on a full-grid mesh in s."""
    wout = output.wout
    ns = int(wout.ns)
    s_half = (np.arange(1, ns) - 0.5) / (ns - 1)
    dv_ds = np.abs(np.asarray(wout.vp)[1:])
    # vp rescaled to integrate to the reported plasma volume: dV/ds in m^3 per unit s.
    dv_ds = dv_ds * float(wout.volume_p) / float(np.trapezoid(dv_ds, s_half))

    minor = float(wout.Aminor_p)
    # Effective radius r = a sqrt(s), so dV/dr = (dV/ds)(ds/dr) = 2 sqrt(s) dV/ds / a.
    dv_dr = dv_ds * 2.0 * np.sqrt(s_half) / minor
    return s_half, dv_dr, minor, float(wout.Rmajor_p)


def solve(
    output: vmecpp.VmecOutput,
    density: np.ndarray | KineticProfiles,
    heating: Heating = Heating(),
    model: TransportModel = TransportModel(),
    edge_temperature_ev: float = 100.0,
    neoclassical=None,
    outer_iterations: int = 5,
    carbon_fraction: float | None = None,
    anomalous=None,
) -> TransportSolution:
    """Steady-state power balance on this equilibrium's flux geometry; with ``neoclassical``
    only the remainder is scaled to ISS04, with ``anomalous`` both channels are computed."""
    s, dv_dr, minor, major = _geometry(output)
    r_normalised = np.sqrt(s)

    profiles = density if isinstance(density, KineticProfiles) else None
    n = (
        profiles.density(s)
        if profiles is not None
        else np.interp(s, np.linspace(0, 1, len(density)), np.asarray(density))
    )

    # Power crossing each surface: deposition inside it less radiation inside it.
    radius = r_normalised * minor
    deposition = heating.profile(s) * dv_dr
    increments = 0.5 * (deposition[1:] + deposition[:-1]) * np.diff(radius)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    if cumulative[-1] <= 0:
        raise ValueError("heating deposition integrates to zero")
    deposited = heating.power_w * cumulative / cumulative[-1]

    fraction = (
        carbon_fraction
        if carbon_fraction is not None
        else getattr(profiles, "carbon_fraction", 0.0)
    )

    def enclosed_radiation(electron_temperature: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """(power density, power enclosed by each surface) from the impurity content."""
        if not fraction:
            return np.zeros_like(s), np.zeros_like(s)
        from w7x_twin.plasma import kinetics

        parts = kinetics.composition(n, electron_temperature, fraction)
        density = kinetics.radiated_power_density(parts, electron_temperature)["total"]
        weighted = density * dv_dr
        steps = 0.5 * (weighted[1:] + weighted[:-1]) * np.diff(radius)
        return density, np.concatenate([[0.0], np.cumsum(steps)])

    radiated_density = np.zeros_like(s)
    radiated_inside = np.zeros_like(s)
    heat_flux = deposited

    shape = model.shape(r_normalised)
    iota = np.asarray(output.wout.iotaf)
    iota_two_thirds = float(np.interp(2.0 / 3.0, np.linspace(0, 1, len(iota)), iota))
    volume_average_density = float(np.trapezoid(n * dv_dr, r_normalised * minor)) / float(
        np.trapezoid(dv_dr, r_normalised * minor)
    )
    target_time = model.renormalisation * iss04_confinement_time(
        minor, major, heating.power_w, volume_average_density,
        float(output.wout.b0), abs(iota_two_thirds),
    )
    target_energy = target_time * heating.power_w

    chi_neo = np.zeros_like(s)
    # Computed turbulent channel, independent of the scaled shape and bisection.
    chi_turbulent = np.zeros_like(s)

    def profile_for(chi0: float) -> tuple[np.ndarray, np.ndarray, float]:
        chi = chi_neo + chi_turbulent + chi0 * shape
        # Q = -S n chi dT/dr with T in energy units; the charge converts to eV.
        gradient = -np.maximum(heat_flux, 0.0) / np.maximum(
            dv_dr * n * chi * ELEMENTARY_CHARGE, 1e-30
        )
        # Integrate inward from the edge.
        temperature = np.empty_like(gradient)
        temperature[-1] = edge_temperature_ev
        for i in range(len(gradient) - 2, -1, -1):
            temperature[i] = temperature[i + 1] - 0.5 * (
                gradient[i] + gradient[i + 1]
            ) * (radius[i + 1] - radius[i])
        temperature = np.maximum(temperature, edge_temperature_ev)
        electron = temperature
        ion = model.ion_fraction * temperature
        pressure = ELEMENTARY_CHARGE * n * (electron + ion)
        energy = 1.5 * float(np.trapezoid(pressure * dv_dr, radius))
        return electron, ion, energy

    def solve_for_anomalous() -> float:
        # Stored energy falls monotonically with chi0, so a bisection closes on ISS04.
        low, high = 1e-6, 1e3
        for _ in range(90):
            mid = np.sqrt(low * high)
            _, _, energy = profile_for(mid)
            if energy > target_energy:
                low = mid
            else:
                high = mid
            if abs(energy - target_energy) < 1e-6 * target_energy:
                break
        return np.sqrt(low * high)

    # Computed channel: skip the bisection; a scaled first pass seeds the temperature.
    chi0 = solve_for_anomalous()
    electron, ion, energy = profile_for(chi0)

    if neoclassical is not None or fraction or anomalous is not None:
        for _ in range(outer_iterations):
            if neoclassical is not None:
                updated = np.asarray(neoclassical(s, electron, n), dtype=float)
                # Under-relax: the T^{7/2} neoclassical diffusivity oscillates undamped.
                chi_neo = 0.5 * chi_neo + 0.5 * updated
            if anomalous is not None:
                turbulent = np.asarray(anomalous(s, electron, n), dtype=float)
                chi_turbulent = 0.5 * chi_turbulent + 0.5 * turbulent
            if fraction:
                radiated_density, radiated_inside = enclosed_radiation(electron)
                heat_flux = deposited - radiated_inside
            chi0 = 0.0 if anomalous is not None else solve_for_anomalous()
            electron, ion, energy = profile_for(chi0)

    if fraction:
        from w7x_twin.plasma import kinetics

        parts = kinetics.composition(n, electron, fraction)
        pressure = ELEMENTARY_CHARGE * (
            n * electron + (parts.ion_density_m3 + parts.impurity_density_m3) * ion
        )
    else:
        pressure = ELEMENTARY_CHARGE * n * (electron + ion)
    return TransportSolution(
        s=s,
        electron_temperature_ev=electron,
        ion_temperature_ev=ion,
        density_m3=n,
        pressure_pa=pressure,
        chi_m2_s=chi_neo + chi_turbulent + chi0 * shape,
        heat_flux_w=heat_flux,
        stored_energy_j=energy,
        confinement_time_s=energy / heating.power_w,
        iss04_time_s=target_time / model.renormalisation,
        heating_power_w=heating.power_w,
        chi_neoclassical_m2_s=chi_neo if neoclassical is not None else None,
        chi_anomalous_m2_s=(
            chi_turbulent
            if anomalous is not None
            else (chi0 * shape if neoclassical is not None else None)
        ),
        radiated_power_w_m3=radiated_density if fraction else None,
        radiated_power_w=float(radiated_inside[-1]),
        carbon_fraction=float(fraction),
    )

#: NRL electron-ion energy equilibration coefficient, s^-1 at n in cm^-3 and T in eV.
EXCHANGE_COEFFICIENT = 3.2e-9


def exchange_power_density(
    density_m3: np.ndarray,
    electron_temperature_ev: np.ndarray,
    ion_temperature_ev: np.ndarray,
    z_effective: np.ndarray | float = 1.0,
    mean_ion_mass_amu: np.ndarray | float = 1.0,
) -> np.ndarray:
    """Collisional exchange from the NRL rate nu = 3.2e-9 Z^2 ln(Lambda) n_i / (mu T_e^1.5), in W/m^3."""
    from w7x_twin.plasma.neoclassical import coulomb_logarithm

    n = np.asarray(density_m3, dtype=float)
    electron = np.maximum(np.asarray(electron_temperature_ev, dtype=float), 1.0)
    ion = np.asarray(ion_temperature_ev, dtype=float)
    logarithm = np.array(
        [coulomb_logarithm(float(value), float(t)) for value, t in zip(n, electron)]
    )
    rate = (
        EXCHANGE_COEFFICIENT
        * np.asarray(z_effective, dtype=float) ** 2
        * logarithm
        * (1.0e-6 * n)
        / (np.asarray(mean_ion_mass_amu, dtype=float) * electron**1.5)
    )
    return 1.5 * n * rate * (electron - ion) * ELEMENTARY_CHARGE


def solve_split(
    output: vmecpp.VmecOutput,
    profiles: KineticProfiles,
    heating: Heating,
    channels,
    edge_temperature_ev: float = 100.0,
    chi_updates: int = 5,
    inner_iterations: int = 200,
    relaxation: float = 0.5,
    turbulent_local=None,
    shear_quench=None,
    field_capture: dict | None = None,
) -> TransportSolution:
    """Two-temperature balance solved shell by shell on the local gradients; ``turbulent_local``
    may be an (electron, ion) pair and ``shear_quench`` with ``field_capture`` damps it by the E_r shear."""
    s, dv_dr, minor, major = _geometry(output)
    radius = np.sqrt(s) * minor
    n = profiles.density(s)

    deposition = heating.profile(s) * dv_dr
    increments = 0.5 * (deposition[1:] + deposition[:-1]) * np.diff(radius)
    cumulative = np.concatenate([[0.0], np.cumsum(increments)])
    if cumulative[-1] <= 0:
        raise ValueError("heating deposition integrates to zero")
    deposited = heating.power_w * cumulative / cumulative[-1]

    fraction = getattr(profiles, "carbon_fraction", 0.0)
    z_profile = (
        np.asarray(profiles.z_effective_profile(s), dtype=float)
        if fraction
        else np.ones_like(s)
    )
    mass = np.asarray(profiles.mean_ion_mass_amu(s), dtype=float)

    def enclosed(density_w_m3: np.ndarray) -> np.ndarray:
        weighted = density_w_m3 * dv_dr
        steps = 0.5 * (weighted[1:] + weighted[:-1]) * np.diff(radius)
        return np.concatenate([[0.0], np.cumsum(steps)])

    def inward(flux_w: np.ndarray, chi: np.ndarray, edge_ev: float) -> np.ndarray:
        gradient = -np.maximum(flux_w, 0.0) / np.maximum(
            dv_dr * n * chi * ELEMENTARY_CHARGE, 1e-30
        )
        temperature = np.empty_like(gradient)
        temperature[-1] = edge_ev
        for index in range(len(gradient) - 2, -1, -1):
            temperature[index] = temperature[index + 1] - 0.5 * (
                gradient[index] + gradient[index + 1]
            ) * (radius[index + 1] - radius[index])
        return np.maximum(temperature, edge_ev)

    density_gradient = -minor * np.gradient(
        np.log(np.maximum(n, 1e-30)), np.maximum(radius, 1e-6)
    )

    if turbulent_local is None:
        electron_local = ion_local = None
    elif callable(turbulent_local):
        electron_local = ion_local = turbulent_local
    else:
        electron_local, ion_local = turbulent_local

    # One reference-temperature sweep per channel; the closure scales as the gyro-Bohm T^{3/2}.
    GRADIENT_GRID = np.concatenate(
        [np.arange(0.0, 8.05, 0.1), np.linspace(8.5, 80.0, 36)]
    )
    REFERENCE_EV = 1000.0
    electron_grid = ion_grid = None
    if turbulent_local is not None:
        mid_s_all = 0.5 * (s[:-1] + s[1:])
        mid_n_all = 0.5 * (n[:-1] + n[1:])
        mid_aln_all = 0.5 * (density_gradient[:-1] + density_gradient[1:])

        def swept(local_chi) -> np.ndarray:
            out = np.empty((len(mid_s_all), len(GRADIENT_GRID)))
            for k in range(len(mid_s_all)):
                for j, gradient in enumerate(GRADIENT_GRID):
                    out[k, j] = local_chi(
                        float(mid_s_all[k]), float(gradient),
                        float(mid_aln_all[k]), REFERENCE_EV, float(mid_n_all[k]),
                    )
            return out

        electron_grid = swept(electron_local)
        ion_grid = (
            electron_grid
            if ion_local is electron_local
            else swept(ion_local)
        )

    def marched(
        flux_w: np.ndarray,
        chi_base: np.ndarray,
        edge_ev: float,
        chi_grid: np.ndarray,
        quench: np.ndarray | None = None,
    ) -> np.ndarray:
        temperature = np.empty_like(s)
        temperature[-1] = edge_ev
        for k in range(len(s) - 2, -1, -1):
            dr = radius[k + 1] - radius[k]
            mid_n = 0.5 * (n[k] + n[k + 1])
            carry = 0.5 * (dv_dr[k] + dv_dr[k + 1]) * mid_n * ELEMENTARY_CHARGE
            base = 0.5 * (chi_base[k] + chi_base[k + 1])
            flux = max(0.5 * (flux_w[k] + flux_w[k + 1]), 0.0)
            outer_t = temperature[k + 1]
            row = chi_grid[k]
            damp = 1.0 if quench is None else 0.5 * (quench[k] + quench[k + 1])

            def imbalance(g: float) -> float:
                mid_t = outer_t + 0.5 * g * dr
                a_lt = minor * g / max(mid_t, 1.0)
                turbulent = min(
                    float(np.interp(a_lt, GRADIENT_GRID, row))
                    * (mid_t / REFERENCE_EV) ** 1.5
                    * damp,
                    1e4,
                )
                return carry * (base + turbulent) * g - flux

            low, high = 0.0, 5.0 * outer_t / minor
            for _ in range(12):
                if imbalance(high) >= 0.0:
                    break
                high *= 2.0
            for _ in range(48):
                middle = 0.5 * (low + high)
                if imbalance(middle) < 0.0:
                    low = middle
                else:
                    high = middle
            temperature[k] = outer_t + 0.5 * (low + high) * dr
        return np.maximum(temperature, edge_ev)

    electron = 1500.0 * (1.0 - s) + edge_temperature_ev
    ion = 0.7 * electron
    chi_electron = np.full_like(s, 1.0)
    chi_ion = np.full_like(s, 1.0)
    radiated_inside = np.zeros_like(s)
    radiated_density = np.zeros_like(s)
    exchange_inside = np.zeros_like(s)
    #: The exchange is stiff: its gain per electronvolt of separation dwarfs the fluxes
    #: it feeds, so the raw Picard map overshoots and the enclosed exchange is the
    #: variable to damp, capped by the power available to hand over.
    exchange_damping = 0.08

    quench_factors = None
    for update in range(chi_updates):
        pair = channels(s, electron, ion, n)
        chi_electron = np.nan_to_num(
            np.asarray(pair[0], dtype=float), nan=1.0, posinf=1e3
        )
        chi_ion = np.nan_to_num(np.asarray(pair[1], dtype=float), nan=1.0, posinf=1e3)
        chi_electron = np.maximum(chi_electron, 1e-3)
        chi_ion = np.maximum(chi_ion, 1e-3)
        if (
            shear_quench is not None
            and field_capture is not None
            and "radial_field_v_m" in field_capture
        ):
            quench_factors = np.asarray(
                shear_quench(
                    s, field_capture["radial_field_v_m"], electron, ion, n
                ),
                dtype=float,
            )
        for _ in range(inner_iterations):
            if fraction:
                from w7x_twin.plasma import kinetics

                parts = kinetics.composition(n, electron, fraction)
                radiated_density = kinetics.radiated_power_density(parts, electron)[
                    "total"
                ]
                radiated_inside = enclosed(radiated_density)
            available = np.maximum(deposited - radiated_inside, 0.0)
            target = np.minimum(
                enclosed(exchange_power_density(n, electron, ion, z_profile, mass)),
                0.98 * available,
            )
            exchange_inside = (
                (1.0 - exchange_damping) * exchange_inside
                + exchange_damping * target
            )
            if turbulent_local is not None:
                electron_new = marched(
                    deposited - exchange_inside - radiated_inside,
                    chi_electron, edge_temperature_ev, electron_grid,
                    quench_factors,
                )
                ion_new = marched(
                    exchange_inside, chi_ion, edge_temperature_ev, ion_grid,
                    quench_factors,
                )
            else:
                electron_new = inward(
                    deposited - exchange_inside - radiated_inside,
                    chi_electron, edge_temperature_ev,
                )
                ion_new = inward(exchange_inside, chi_ion, edge_temperature_ev)
            moved = max(
                float(np.max(np.abs(electron_new - electron) / np.maximum(electron, 1.0))),
                float(np.max(np.abs(ion_new - ion) / np.maximum(ion, 1.0))),
                float(
                    np.max(np.abs(target - exchange_inside))
                    / max(heating.power_w, 1.0)
                ),
            )
            electron = (1.0 - relaxation) * electron + relaxation * electron_new
            ion = (1.0 - relaxation) * ion + relaxation * ion_new
            if moved < 1e-5:
                break

    if fraction:
        from w7x_twin.plasma import kinetics

        parts = kinetics.composition(n, electron, fraction)
        pressure = ELEMENTARY_CHARGE * (
            n * electron + (parts.ion_density_m3 + parts.impurity_density_m3) * ion
        )
    else:
        pressure = ELEMENTARY_CHARGE * n * (electron + ion)
    energy = 1.5 * float(np.trapezoid(pressure * dv_dr, radius))

    iota = np.asarray(output.wout.iotaf)
    iota_two_thirds = float(np.interp(2.0 / 3.0, np.linspace(0, 1, len(iota)), iota))
    average = float(np.trapezoid(n * dv_dr, radius)) / float(
        np.trapezoid(dv_dr, radius)
    )
    iss04 = iss04_confinement_time(
        minor, major, heating.power_w, average, float(output.wout.b0),
        abs(iota_two_thirds),
    )

    turbulent_final = None
    if turbulent_local is not None:
        electron_gradient = -minor * np.gradient(
            np.log(np.maximum(electron, 1e-30)), np.maximum(radius, 1e-6)
        )
        ion_gradient = -minor * np.gradient(
            np.log(np.maximum(ion, 1e-30)), np.maximum(radius, 1e-6)
        )
        turbulent_final = np.array(
            [
                electron_local(float(sv), float(gv), float(dv), float(tv), float(nv))
                for sv, gv, dv, tv, nv in zip(
                    s, electron_gradient, density_gradient, electron, n, strict=True
                )
            ]
        )
        ion_turbulent = np.array(
            [
                ion_local(float(sv), float(gv), float(dv), float(tv), float(nv))
                for sv, gv, dv, tv, nv in zip(
                    s, ion_gradient, density_gradient, ion, n, strict=True
                )
            ]
        )
        if quench_factors is not None:
            turbulent_final = turbulent_final * quench_factors
            ion_turbulent = ion_turbulent * quench_factors
        chi_electron = chi_electron + turbulent_final
        chi_ion = chi_ion + ion_turbulent

    return TransportSolution(
        s=s,
        electron_temperature_ev=electron,
        ion_temperature_ev=ion,
        density_m3=n,
        pressure_pa=pressure,
        chi_m2_s=chi_electron,
        heat_flux_w=deposited - exchange_inside - radiated_inside,
        stored_energy_j=energy,
        confinement_time_s=energy / heating.power_w,
        iss04_time_s=iss04,
        heating_power_w=heating.power_w,
        chi_neoclassical_m2_s=None,
        chi_anomalous_m2_s=turbulent_final,
        chi_ion_m2_s=chi_ion,
        exchange_power_w=float(exchange_inside[-1]),
        radiated_power_w_m3=radiated_density if fraction else None,
        radiated_power_w=float(radiated_inside[-1]),
        carbon_fraction=float(fraction),
    )


# -- from particle_balance --------------------------------------------------------

@dataclasses.dataclass
class ParticleModel:
    """Diffusion and inward pinch for the particle channel."""

    #: Particle diffusivity at the edge, in m^2/s.
    diffusivity_m2_s: float = 0.2
    #: Radial form of the diffusivity, as the exponent of s.
    diffusivity_exponent: float = 1.0
    #: Inward pinch velocity at the edge, in m/s. Negative is inward.
    pinch_m_s: float = -0.5
    #: Floor the solved density is held above, in m^-3.
    floor_m3: float = 1.0e18

    def diffusivity(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        return self.diffusivity_m2_s * (0.1 + 0.9 * s**self.diffusivity_exponent)

    def pinch(self, s: np.ndarray) -> np.ndarray:
        s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
        return self.pinch_m_s * np.sqrt(s)


@dataclasses.dataclass
class ParticleSolution:
    """Solved density profile and the flux that produced it."""

    s: np.ndarray
    radius_m: np.ndarray
    density_m3: np.ndarray
    flux_m2_s: np.ndarray
    source_m3_s: np.ndarray
    #: n(0) / n(rho = 0.8), the ratio W7-X reports its profiles by.
    peaking: float
    #: Particles per second crossing the separatrix.
    throughput_s: float


def gaussian_source(
    s: np.ndarray, centre: float, width: float, total_per_second: float,
    minor_radius_m: float,
) -> np.ndarray:
    """Fuelling source centred on one surface, volume-normalised to ``total_per_second``."""
    s = np.clip(np.asarray(s, dtype=float), 0.0, 1.0)
    shape = np.exp(-0.5 * ((s - centre) / max(width, 1e-6)) ** 2)
    # Volume element per unit s, up to the 4 pi^2 R the normalisation divides out.
    weight = np.ones_like(s)
    integral = float(np.trapezoid(shape * weight, s))
    if integral <= 0.0:
        return np.zeros_like(s)
    return total_per_second * shape / integral


def solve_density(
    source_m3_s: np.ndarray,
    s: np.ndarray,
    minor_radius_m: float,
    edge_density_m3: float,
    model: ParticleModel | None = None,
) -> ParticleSolution:
    """Density profile carrying a given particle source, integrated inward from the edge."""
    model = model or ParticleModel()
    s = np.asarray(s, dtype=float)
    source = np.asarray(source_m3_s, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0))

    # Flux through each surface: the enclosed source rate per unit area.
    enclosed = np.zeros_like(radius)
    for index in range(1, len(radius)):
        enclosed[index] = float(
            np.trapezoid(
                source[: index + 1] * radius[: index + 1], radius[: index + 1]
            )
        )
    with np.errstate(divide="ignore", invalid="ignore"):
        flux = np.where(radius > 0.0, enclosed / np.maximum(radius, 1e-12), 0.0)

    diffusivity = model.diffusivity(s)
    pinch = model.pinch(s)

    # Gamma = -D dn/dr + v n, so dn/dr = (v n - Gamma) / D. Integrated inward from the edge.
    density = np.empty_like(radius)
    density[-1] = edge_density_m3
    for index in range(len(radius) - 1, 0, -1):
        step = radius[index] - radius[index - 1]
        gradient = (
            pinch[index] * density[index] - flux[index]
        ) / max(diffusivity[index], 1e-12)
        density[index - 1] = max(density[index] - gradient * step, model.floor_m3)

    peaking = float(density[0] / np.interp(0.8**2, s, density))
    return ParticleSolution(
        s=s,
        radius_m=radius,
        density_m3=density,
        flux_m2_s=flux,
        source_m3_s=source,
        peaking=peaking,
        throughput_s=float(np.trapezoid(source * radius, radius) * 4.0 * np.pi**2),
    )


def peaking_for_source(
    centre: float,
    s: np.ndarray,
    minor_radius_m: float,
    edge_density_m3: float,
    total_per_second: float,
    width: float = 0.15,
    model: ParticleModel | None = None,
) -> ParticleSolution:
    """The profile a source centred on one surface produces."""
    source = gaussian_source(s, centre, width, total_per_second, minor_radius_m)
    return solve_density(source, s, minor_radius_m, edge_density_m3, model)


def temperature_step(
    temperature_ev: np.ndarray,
    density_m3: np.ndarray,
    chi_m2_s: np.ndarray,
    heating_w_m3: np.ndarray,
    s: np.ndarray,
    minor_radius_m: float,
    edge_temperature_ev: float,
    time_step_s: float,
    ion_fraction: float = 0.55,
) -> np.ndarray:
    """One backward-Euler energy step: flux -e n chi dT/dr on faces, edge pinned, no axis flux, in eV."""
    from scipy.linalg import solve_banded

    s = np.asarray(s, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0))
    count = len(radius)
    temperature = np.asarray(temperature_ev, dtype=float).copy()
    density = np.asarray(density_m3, dtype=float)
    chi = np.asarray(chi_m2_s, dtype=float)
    heating = np.asarray(heating_w_m3, dtype=float)

    face_r = 0.5 * (radius[1:] + radius[:-1])
    face_c = 0.5 * (density[1:] * chi[1:] + density[:-1] * chi[:-1])
    spacing = np.diff(radius)
    capacity = 1.5 * ELEMENTARY_CHARGE * (1.0 + ion_fraction) * density

    lower = np.zeros(count)
    diagonal = np.ones(count)
    upper = np.zeros(count)
    right = np.empty(count)
    for index in range(1, count - 1):
        cell = 0.5 * (face_r[index] ** 2 - face_r[index - 1] ** 2)
        outward = face_r[index] * face_c[index] / spacing[index]
        inward = face_r[index - 1] * face_c[index - 1] / spacing[index - 1]
        a = ELEMENTARY_CHARGE * inward / cell
        c = ELEMENTARY_CHARGE * outward / cell
        lower[index] = -time_step_s * a
        diagonal[index] = capacity[index] + time_step_s * (a + c)
        upper[index] = -time_step_s * c
        right[index] = capacity[index] * temperature[index] + time_step_s * heating[index]
    cell = 0.5 * face_r[0] ** 2
    outward = ELEMENTARY_CHARGE * face_r[0] * face_c[0] / spacing[0] / cell
    diagonal[0] = capacity[0] + time_step_s * outward
    upper[0] = -time_step_s * outward
    right[0] = capacity[0] * temperature[0] + time_step_s * heating[0]
    diagonal[-1] = 1.0
    lower[-1] = 0.0
    right[-1] = edge_temperature_ev

    banded = np.zeros((3, count))
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diagonal
    banded[2, :-1] = lower[1:]
    return np.maximum(solve_banded((1, 1), banded, right), 1.0)


def evolve_density(
    initial_m3: np.ndarray,
    source_m3_s: np.ndarray,
    s: np.ndarray,
    minor_radius_m: float,
    edge_density_m3: float,
    model: ParticleModel | None = None,
    time_step_s: float = 1.0e-3,
    num_steps: int = 6000,
    peaking_target: float | None = None,
    record_every: int = 20,
) -> dict:
    """Density marched in time through a source by backward Euler, to a peaking or to saturation."""
    from scipy.linalg import solve_banded

    model = model or ParticleModel()
    s = np.asarray(s, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0))
    count = len(radius)
    density = np.asarray(initial_m3, dtype=float).copy()
    density[-1] = edge_density_m3
    source = np.asarray(source_m3_s, dtype=float)

    face_r = 0.5 * (radius[1:] + radius[:-1])
    face_d = 0.5 * (model.diffusivity(s)[1:] + model.diffusivity(s)[:-1])
    face_v = 0.5 * (model.pinch(s)[1:] + model.pinch(s)[:-1])
    spacing = np.diff(radius)

    # Backward Euler: (I - dt A) n_new = n_old + dt S, with A the flux divergence and
    # the last row pinned to the edge density.
    lower = np.zeros(count)
    diagonal = np.ones(count)
    upper = np.zeros(count)
    for index in range(1, count - 1):
        cell = 0.5 * (face_r[index] ** 2 - face_r[index - 1] ** 2)
        outward = face_r[index] * face_d[index] / spacing[index]
        inward = face_r[index - 1] * face_d[index - 1] / spacing[index - 1]
        advect_out = 0.5 * face_r[index] * face_v[index]
        advect_in = 0.5 * face_r[index - 1] * face_v[index - 1]
        a = (inward + advect_in) / cell
        c = (outward - advect_out) / cell
        b = -(inward - advect_in + outward + advect_out) / cell
        lower[index] = -time_step_s * a
        diagonal[index] = 1.0 - time_step_s * b
        upper[index] = -time_step_s * c
    # The axis cell: no flux through r = 0, so only the outer face enters.
    cell = 0.5 * face_r[0] ** 2
    outward = face_r[0] * face_d[0] / spacing[0]
    advect_out = 0.5 * face_r[0] * face_v[0]
    diagonal[0] = 1.0 + time_step_s * (outward + advect_out) / cell
    upper[0] = -time_step_s * (outward - advect_out) / cell

    banded = np.zeros((3, count))
    banded[0, 1:] = upper[:-1]
    banded[1, :] = diagonal
    banded[2, :-1] = lower[1:]

    def peaking_of(values: np.ndarray) -> float:
        return float(values[0] / np.interp(0.8**2, s, values))

    times = [0.0]
    peakings = [peaking_of(density)]
    reached = float("nan")
    for step in range(1, num_steps + 1):
        right = density + time_step_s * source
        right[-1] = edge_density_m3
        density = solve_banded((1, 1), banded, right)
        np.maximum(density, model.floor_m3, out=density)
        if step % record_every == 0 or step == num_steps:
            times.append(step * time_step_s)
            peakings.append(peaking_of(density))
        if peaking_target is not None and peaking_of(density) >= peaking_target:
            reached = step * time_step_s
            times.append(reached)
            peakings.append(peaking_of(density))
            break

    return {
        "s": s,
        "radius_m": radius,
        "density_m3": density,
        "times_s": np.asarray(times),
        "peaking": np.asarray(peakings),
        "time_to_target_s": reached,
    }


# -- from heating ------------------------------------------------------------------

# Heating deposition: 140 GHz X2 absorption on the |B| resonance, and beam ionisation with the critical-energy split.


#: W7-X electron-cyclotron system: 140 GHz, second harmonic, X-mode.
ECRH_FREQUENCY_HZ = 140.0e9
ECRH_HARMONIC = 2


def resonant_field_t(
    frequency_hz: float = ECRH_FREQUENCY_HZ, harmonic: int = ECRH_HARMONIC
) -> float:
    """Field strength at which a harmonic of the cyclotron frequency matches the beam."""
    return float(
        2.0 * np.pi * ELECTRON_MASS * frequency_hz / (harmonic * ELEMENTARY_CHARGE)
    )


def cutoff_density_m3(
    frequency_hz: float = ECRH_FREQUENCY_HZ, harmonic: int = ECRH_HARMONIC
) -> float:
    """X-mode cutoff density at the h-th harmonic, w_pe^2 = (1 - 1/h) w^2; 1.2e20 m^-3 for X2 at 140 GHz."""
    if harmonic < 1:
        return 0.0
    omega = 2.0 * np.pi * frequency_hz
    plasma_squared = (1.0 - 1.0 / harmonic) * omega**2
    if plasma_squared <= 0.0:
        return 0.0
    return float(
        plasma_squared * VACUUM_PERMITTIVITY * ELECTRON_MASS / ELEMENTARY_CHARGE**2
    )


@dataclasses.dataclass
class Deposition:
    """Absorbed power against normalised toroidal flux."""

    s: np.ndarray
    #: Power per unit s, normalised so its integral is the absorbed power.
    profile_w: np.ndarray
    absorbed_fraction: float
    #: Surface the absorption peaks on.
    peak_s: float
    note: str = ""

    def normalised(self) -> np.ndarray:
        total = float(np.trapezoid(self.profile_w, self.s))
        return self.profile_w / total if total > 0.0 else self.profile_w


def cyclotron_deposition(
    s: np.ndarray,
    field_on_surface_t: np.ndarray,
    density_m3: np.ndarray,
    temperature_ev: np.ndarray,
    power_w: float,
    frequency_hz: float = ECRH_FREQUENCY_HZ,
    harmonic: int = ECRH_HARMONIC,
    width_t: float = 0.02,
) -> Deposition:
    """Absorption on the |B| resonance layer, weighted by its density-temperature optical depth."""
    s = np.asarray(s, dtype=float)
    field = np.asarray(field_on_surface_t, dtype=float)
    density = np.asarray(density_m3, dtype=float)
    temperature = np.asarray(temperature_ev, dtype=float)

    resonant = resonant_field_t(frequency_hz, harmonic)
    # Distance from resonance in field, as a layer of finite width.
    layer = np.exp(-0.5 * ((field - resonant) / max(width_t, 1e-6)) ** 2)

    # Second-harmonic X-mode optical depth goes as n T at fixed geometry, which is what
    # decides how much of the beam is absorbed on the first pass.
    depth = density * temperature
    weight = layer * depth
    total = float(np.trapezoid(weight, s))
    if total <= 0.0:
        return Deposition(
            s=s, profile_w=np.zeros_like(s), absorbed_fraction=0.0, peak_s=float("nan"),
            note="no surface reaches the resonant field",
        )

    cutoff = cutoff_density_m3(frequency_hz, harmonic)
    reachable = float(np.max(density)) < cutoff
    absorbed = 1.0 if reachable else 0.0
    return Deposition(
        s=s,
        profile_w=power_w * absorbed * weight / total,
        absorbed_fraction=absorbed,
        peak_s=float(s[int(np.argmax(weight))]),
        note=(
            f"X{harmonic} reaches the resonance"
            if reachable
            else f"above the X{harmonic} cut-off of {cutoff:.3e} m^-3, so O-mode is needed"
        ),
    )


def appleton_hartree_n2(
    plasma_x: float, cyclotron_y: float, cos_angle: float, mode: int = -1
) -> float:
    """Appleton-Hartree cold-plasma index squared; ``mode`` -1 extraordinary, +1 ordinary,
    negative past a cutoff."""
    x = float(plasma_x)
    y = float(cyclotron_y)
    cos2 = min(max(cos_angle * cos_angle, 0.0), 1.0)
    sin2 = 1.0 - cos2
    one_minus = 1.0 - x
    half_transverse = 0.5 * y * y * sin2
    root = float(
        np.sqrt(
            half_transverse * half_transverse
            + one_minus * one_minus * y * y * cos2
        )
    )
    denominator = one_minus - half_transverse + mode * root
    if abs(denominator) < 1e-12:
        return float("-inf")
    return 1.0 - x * one_minus / denominator


@dataclasses.dataclass
class TracedRay:
    """One microwave ray marched to the resonance, a cutoff, or out of the domain."""

    path_m: np.ndarray
    field_t: np.ndarray
    index_squared: np.ndarray
    crossed: bool
    note: str


def trace_cyclotron_ray(
    field_cartesian,
    density_at,
    launch_m: np.ndarray,
    direction: np.ndarray,
    frequency_hz: float = ECRH_FREQUENCY_HZ,
    harmonic: int = ECRH_HARMONIC,
    mode: int = -1,
    step_m: float = 0.01,
    gradient_step_m: float = 0.003,
    max_steps: int = 2500,
) -> TracedRay:
    """March a ray by the scalar eikonal d(n t)/ds = grad n at the local magnetized index,
    ending at the resonance, a cutoff, or the domain edge."""
    omega = 2.0 * np.pi * frequency_hz
    resonant = resonant_field_t(frequency_hz, harmonic)

    def index_at(point: np.ndarray, tangent: np.ndarray) -> tuple[float, float]:
        b_vector = field_cartesian(point)
        magnitude = float(np.linalg.norm(b_vector))
        if not np.isfinite(magnitude) or magnitude <= 0.0:
            return float("nan"), float("nan")
        plasma_x = (
            float(density_at(point))
            * ELEMENTARY_CHARGE**2
            / (VACUUM_PERMITTIVITY * ELECTRON_MASS * omega**2)
        )
        cyclotron_y = ELEMENTARY_CHARGE * magnitude / (ELECTRON_MASS * omega)
        cos_angle = float(np.dot(tangent, b_vector)) / max(magnitude, 1e-30)
        return (
            appleton_hartree_n2(plasma_x, cyclotron_y, cos_angle, mode),
            magnitude,
        )

    point = np.asarray(launch_m, dtype=float).copy()
    tangent = np.asarray(direction, dtype=float)
    tangent = tangent / np.linalg.norm(tangent)

    path = [point.copy()]
    fields = []
    indices = []
    crossed = False
    note = "step limit reached"
    for _ in range(max_steps):
        n2, magnitude = index_at(point, tangent)
        fields.append(magnitude)
        indices.append(n2)
        if not np.isfinite(magnitude):
            note = "left the field's domain"
            break
        if magnitude >= resonant:
            crossed = True
            note = "crossed the resonance"
            break
        if not np.isfinite(n2) or n2 <= 0.0:
            note = "reached a cutoff"
            break
        n = float(np.sqrt(n2))

        gradient = np.zeros(3)
        for axis in range(3):
            offset = np.zeros(3)
            offset[axis] = gradient_step_m
            ahead, _ = index_at(point + offset, tangent)
            behind, _ = index_at(point - offset, tangent)
            if np.isfinite(ahead) and np.isfinite(behind) and ahead > 0 and behind > 0:
                gradient[axis] = (np.sqrt(ahead) - np.sqrt(behind)) / (
                    2.0 * gradient_step_m
                )
        flow = n * tangent + step_m * gradient
        norm = float(np.linalg.norm(flow))
        if norm < 1e-12:
            note = "the eikonal stalled"
            break
        tangent = flow / norm
        point = point + step_m * tangent
        path.append(point.copy())

    return TracedRay(
        path_m=np.asarray(path),
        field_t=np.asarray(fields),
        index_squared=np.asarray(indices),
        crossed=crossed,
        note=note,
    )


def critical_energy_ev(
    temperature_ev: np.ndarray, mass_ratio: float = 2.0, charge: float = 1.0
) -> np.ndarray:
    """Critical energy at which a fast ion heats ions and electrons equally, in eV."""
    t = np.maximum(np.asarray(temperature_ev, dtype=float), 1e-6)
    return 14.8 * t * mass_ratio * charge ** (2.0 / 3.0)


def beam_power_split(
    injection_energy_ev: float,
    temperature_ev: np.ndarray,
    mass_ratio: float = 2.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Analytic ion-electron split of beam power over the slowing-down from injection to rest."""
    critical = critical_energy_ev(temperature_ev, mass_ratio)
    x = np.maximum(injection_energy_ev / np.maximum(critical, 1e-9), 1e-9)
    # Stix: the ion fraction is (1/x) * integral_0^x dy / (1 + y^(3/2)).
    grid = np.linspace(0.0, 1.0, 257)
    ion = np.empty_like(np.atleast_1d(x), dtype=float)
    for index, value in enumerate(np.atleast_1d(x)):
        y = grid * value
        ion[index] = float(np.trapezoid(1.0 / (1.0 + y**1.5), y)) / value
    ion = np.clip(ion, 0.0, 1.0)
    return ion.reshape(np.shape(x)), 1.0 - ion.reshape(np.shape(x))


def beam_deposition(
    s: np.ndarray,
    density_m3: np.ndarray,
    temperature_ev: np.ndarray,
    power_w: float,
    injection_energy_ev: float = 55.0e3,
    minor_radius_m: float = 0.51,
    tangency_s: float = 0.1,
) -> Deposition:
    """Beam power absorbed along its path, attenuated by the line-integrated density."""
    s = np.asarray(s, dtype=float)
    density = np.asarray(density_m3, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0))

    # Stopping cross-section for a hydrogen beam at tens of keV, in m^2.
    cross_section = 3.0e-20 * (injection_energy_ev / 55.0e3) ** -0.5
    order = np.argsort(radius)
    outward = radius[order]
    along = density[order]
    increments = np.zeros_like(outward)
    increments[1:] = 0.5 * (along[1:] + along[:-1]) * np.diff(outward)
    suffix = np.cumsum(increments[::-1])[::-1]
    line_sorted = np.append(suffix[1:], 0.0)
    line = np.empty_like(line_sorted)
    line[order] = line_sorted

    surviving = np.exp(-cross_section * line)
    weight = density * surviving * np.exp(-0.5 * ((s - tangency_s) / 0.35) ** 2)
    total = float(np.trapezoid(weight, s))
    if total <= 0.0:
        return Deposition(
            s=s, profile_w=np.zeros_like(s), absorbed_fraction=0.0, peak_s=float("nan"),
            note="the beam is not absorbed",
        )
    # What the beam deposits inside the plasma rather than passing through.
    absorbed = float(1.0 - np.min(surviving))
    return Deposition(
        s=s,
        profile_w=power_w * absorbed * weight / total,
        absorbed_fraction=absorbed,
        peak_s=float(s[int(np.argmax(weight))]),
        note=f"stopping cross-section {cross_section:.2e} m^2",
    )


# -- fast-ion orbit losses ---------------------------------------------------------

# Collisionless guiding-centre following of beam ions from their deposition to the
# vessel: dX/dt = v_par b + b x (mu grad|B| + m v_par^2 kappa) / (qB) and
# dv_par/dt = -(mu/m) b . grad|B|, the equation set BEAMS3D integrates, on the
# trilinear field with its exact gradient.


@dataclasses.dataclass
class FastIonLosses:
    """Orbit-loss record of one followed beam population."""

    loss_fraction: float
    particles: int
    lost: int
    followed_s: float
    mean_loss_time_s: float
    times_s: np.ndarray
    lost_fraction_of_time: np.ndarray
    energy_drift: float
    note: str


def birth_positions(
    wout, s_samples: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(R, phi, Z) on the sampled flux surfaces at uniform poloidal and toroidal angle."""
    xm = np.asarray(wout.xm)
    xn = np.asarray(wout.xn)
    surfaces = np.clip(
        np.round(np.asarray(s_samples) * (int(wout.ns) - 1)).astype(int),
        0, int(wout.ns) - 1,
    )
    rmnc = np.asarray(wout.rmnc)[:, surfaces]
    zmns = np.asarray(wout.zmns)[:, surfaces]
    theta = rng.uniform(0.0, 2.0 * np.pi, len(surfaces))
    phi = rng.uniform(0.0, 2.0 * np.pi, len(surfaces))
    angle = xm[:, None] * theta[None, :] - xn[:, None] * phi[None, :]
    radius = np.sum(np.cos(angle) * rmnc, axis=0)
    height = np.sum(np.sin(angle) * zmns, axis=0)
    return radius, phi, height


def guiding_centre_rates(
    vacuum, radius, phi, height, v_par, mu_over_m, charge_over_m
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """(dR/dt, dphi/dt, dZ/dt, dv_par/dt) of the first-order guiding centre."""
    b_vec, grad = vacuum.with_gradient(radius, phi, height)
    magnitude = np.sqrt(np.sum(b_vec * b_vec, axis=0))
    unit = b_vec / np.maximum(magnitude, 1e-30)

    # Physical-component gradient of |B| and of the unit vector; the cylindrical basis
    # adds the b_phi^2 / R centrifugal and b_R b_phi / R coupling terms to (b.grad)b.
    grad_b = np.sum(b_vec[:, None, :] * grad, axis=0) / np.maximum(magnitude, 1e-30)
    unit_grad = (grad - unit[:, None, :] * grad_b[None, :, :]) / np.maximum(
        magnitude, 1e-30
    )
    kappa = np.sum(unit[None, :, :] * np.swapaxes(unit_grad, 0, 1), axis=1)
    safe_r = np.maximum(radius, 1e-9)
    kappa[0] -= unit[1] * unit[1] / safe_r
    kappa[1] += unit[0] * unit[1] / safe_r

    push = mu_over_m * grad_b + (v_par * v_par)[None, :] * kappa
    drift = (
        np.stack(
            [
                unit[1] * push[2] - unit[2] * push[1],
                unit[2] * push[0] - unit[0] * push[2],
                unit[0] * push[1] - unit[1] * push[0],
            ]
        )
        / (charge_over_m * np.maximum(magnitude, 1e-30))
    )
    velocity = v_par[None, :] * unit + drift
    return (
        velocity[0],
        velocity[1] / safe_r,
        velocity[2],
        -mu_over_m * np.sum(unit * grad_b, axis=0),
    )


def fast_ion_losses(
    vacuum,
    wout,
    vessel,
    deposition: Deposition,
    injection_energy_ev: float = 55.0e3,
    mass_amu: float = 1.0,
    particles: int = 512,
    follow_s: float = 2.0e-3,
    time_step_s: float = 1.0e-8,
    pitch: tuple[float, float] = (0.4, 0.8),
    co_injection: bool = False,
    wall_check_every: int = 2,
    seed: int = 20260728,
) -> FastIonLosses:
    """Follow a birth population sampled from the deposition until it strikes the
    vessel or the following time ends, and report the lost power fraction.

    Collisionless and at full injection energy, so what is counted is the prompt and
    ripple-trapped orbit loss; collisional scattering over the ~30 ms slowing time is
    outside the model. The pitch band is an aiming assumption the record carries."""
    rng = np.random.default_rng(seed)
    weights = np.clip(np.asarray(deposition.profile_w, dtype=float), 0.0, None)
    total = float(np.trapezoid(weights, deposition.s))
    if total <= 0.0:
        return FastIonLosses(
            loss_fraction=float("nan"), particles=0, lost=0, followed_s=follow_s,
            mean_loss_time_s=float("nan"), times_s=np.zeros(0),
            lost_fraction_of_time=np.zeros(0), energy_drift=float("nan"),
            note="the deposition carries no absorbed power",
        )
    density = weights / total
    s_samples = rng.choice(
        deposition.s, size=particles,
        p=density / np.sum(density),
    )
    radius, phi, height = birth_positions(wout, s_samples, rng)

    mass = mass_amu * PROTON_MASS
    speed = np.sqrt(2.0 * ELEMENTARY_CHARGE * injection_energy_ev / mass)
    sign = 1.0 if co_injection else -1.0
    pitch_samples = sign * rng.uniform(pitch[0], pitch[1], particles)
    v_par = speed * pitch_samples
    b_vec, _ = vacuum.with_gradient(radius, phi, height)
    field_t = np.sqrt(np.sum(b_vec * b_vec, axis=0))
    mu_over_m = 0.5 * speed * speed * (1.0 - pitch_samples**2) / field_t
    charge_over_m = ELEMENTARY_CHARGE / mass
    initial_energy = 0.5 * v_par**2 + mu_over_m * field_t

    steps = int(round(follow_s / time_step_s))
    grid_phi = np.linspace(0.0, 2.0 * np.pi / vessel.num_field_periods, 96, endpoint=False)
    wall = vessel.resample(grid_phi)
    slot_of = len(grid_phi) * vessel.num_field_periods / (2.0 * np.pi)

    alive = np.isfinite(field_t)
    lost = np.zeros(particles, dtype=bool)
    loss_time = np.full(particles, np.nan)
    record_every = max(1, steps // 200)
    times: list[float] = []
    lost_curve: list[float] = []

    for step in range(steps):
        if not alive.any():
            break
        dt = time_step_s

        def rates(r_now, p_now, z_now, v_now):
            return guiding_centre_rates(
                vacuum, r_now, p_now, z_now, v_now, mu_over_m, charge_over_m
            )

        k1 = rates(radius, phi, height, v_par)
        k2 = rates(
            radius + 0.5 * dt * k1[0], phi + 0.5 * dt * k1[1],
            height + 0.5 * dt * k1[2], v_par + 0.5 * dt * k1[3],
        )
        k3 = rates(
            radius + 0.5 * dt * k2[0], phi + 0.5 * dt * k2[1],
            height + 0.5 * dt * k2[2], v_par + 0.5 * dt * k2[3],
        )
        k4 = rates(
            radius + dt * k3[0], phi + dt * k3[1],
            height + dt * k3[2], v_par + dt * k3[3],
        )
        radius = radius + (dt / 6.0) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
        phi = phi + (dt / 6.0) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
        height = height + (dt / 6.0) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
        v_par = v_par + (dt / 6.0) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

        if step % wall_check_every == 0:
            left_grid = alive & ~(np.isfinite(radius) & np.isfinite(height))
            slots = (
                np.mod(phi, 2.0 * np.pi / vessel.num_field_periods) * slot_of
            ).astype(int) % len(grid_phi)
            outside = np.zeros(particles, dtype=bool)
            for slot in np.unique(slots[alive]):
                here = alive & (slots == slot)
                outside[here] = wall.outside(radius[here], height[here], int(slot))
            fresh = alive & (outside | left_grid)
            if fresh.any():
                lost |= fresh
                loss_time[fresh] = step * time_step_s
                alive &= ~fresh
                radius = np.where(alive, radius, np.nan)
        if step % record_every == 0:
            times.append(step * time_step_s)
            lost_curve.append(float(np.mean(lost)))

    survivors = alive & np.isfinite(radius) & np.isfinite(height)
    b_vec, _ = vacuum.with_gradient(
        np.where(survivors, radius, 6.0), phi, np.where(survivors, height, 0.0)
    )
    final_field = np.sqrt(np.sum(b_vec * b_vec, axis=0))
    final_energy = 0.5 * v_par**2 + mu_over_m * final_field
    drift = np.abs(final_energy[survivors] - initial_energy[survivors]) / np.maximum(
        initial_energy[survivors], 1e-30
    )
    return FastIonLosses(
        loss_fraction=float(np.mean(lost)),
        particles=particles,
        lost=int(np.count_nonzero(lost)),
        followed_s=follow_s,
        mean_loss_time_s=float(np.nanmean(loss_time)) if lost.any() else float("nan"),
        times_s=np.array(times),
        lost_fraction_of_time=np.array(lost_curve),
        energy_drift=float(np.median(drift)) if survivors.any() else float("nan"),
        note=(
            f"collisionless, full energy, pitch {pitch[0]:g} to {pitch[1]:g} "
            f"{'co' if co_injection else 'counter'}-injected, {follow_s * 1e3:g} ms"
        ),
    )


# -- from turbulence ---------------------------------------------------------------

# Turbulent diffusivity: the linear gamma / k_y^2 spectrum closed by the measured saturation response.



def gyro_bohm(temperature_ev: float, field_t: float, minor_radius_m: float) -> float:
    """Gyro-Bohm diffusivity rho_i^2 v_ti / a, in m^2/s."""
    speed = np.sqrt(ELEMENTARY_CHARGE * temperature_ev / PROTON_MASS)
    radius = PROTON_MASS * speed / (ELEMENTARY_CHARGE * field_t)
    return float(radius**2 * speed / minor_radius_m)


@dataclasses.dataclass(frozen=True)
class GrowthRateTable:
    """Growth rates indexed [surface, tprim, fprim, ky] in v_ti / a, NaN where unconverged."""

    surfaces: np.ndarray
    gradients: np.ndarray
    density_gradients: np.ndarray
    wavenumbers: np.ndarray
    rates: np.ndarray

    @classmethod
    def from_cases(cls, cases: list[dict], configuration: str | None = None) -> "GrowthRateTable":
        if configuration is not None:
            cases = [c for c in cases if c.get("configuration") == configuration]
        surfaces = np.array(sorted({float(c["torflux"]) for c in cases}))
        gradients = np.array(sorted({float(c["tprim"]) for c in cases}))
        density = np.array(sorted({float(c.get("fprim", 1.0)) for c in cases}))
        wavenumbers = np.array(sorted({float(c["ky"]) for c in cases}))
        rates = np.full(
            (len(surfaces), len(gradients), len(density), len(wavenumbers)), np.nan
        )
        for case in cases:
            i = int(np.argmin(np.abs(surfaces - float(case["torflux"]))))
            j = int(np.argmin(np.abs(gradients - float(case["tprim"]))))
            m = int(np.argmin(np.abs(density - float(case.get("fprim", 1.0)))))
            k = int(np.argmin(np.abs(wavenumbers - float(case["ky"]))))
            rates[i, j, m, k] = float(case["growth_rate"])
        return cls(surfaces, gradients, density, wavenumbers, rates)

    @classmethod
    def read(cls, path: str | Path, configuration: str | None = None) -> "GrowthRateTable":
        stored = json.loads(Path(path).read_text())
        return cls.from_cases(stored["cases"], configuration)

    def mixing_length_sum(
        self,
        surface: float,
        gradient: float,
        density_gradient: float | None = None,
        ky_max: float = 1.0,
    ) -> float:
        """Sum of gamma / k_y^2 over growing modes with k_y <= ``ky_max``, the ion scales
        the turbulent channel transports."""
        density = (
            float(np.mean(self.density_gradients))
            if density_gradient is None
            else float(density_gradient)
        )
        total = 0.0
        for k, ky in enumerate(self.wavenumbers):
            if ky > ky_max:
                continue
            block = self.rates[:, :, :, k]
            # One-dimensional interpolations taken in turn, since the grid has holes in it
            # where a run did not converge and a spline over it would spread them.
            per_surface = []
            for row in block:
                per_gradient = np.array(
                    [
                        np.interp(gradient, self.gradients, column)
                        if np.all(np.isfinite(column))
                        else np.nan
                        for column in row.T
                    ]
                )
                usable = np.isfinite(per_gradient)
                per_surface.append(
                    float(
                        np.interp(
                            density, self.density_gradients[usable], per_gradient[usable]
                        )
                    )
                    if usable.any()
                    else np.nan
                )
            per_surface = np.array(per_surface)
            usable = np.isfinite(per_surface)
            if not usable.any():
                continue
            value = float(np.interp(surface, self.surfaces[usable], per_surface[usable]))
            if value > 0.0:
                total += value / ky**2
        return total

    def peak_growth(
        self,
        surface: float,
        gradient: float,
        density_gradient: float | None = None,
        ky_max: float = 1.0,
    ) -> float:
        """Largest growth rate over the modes the channel transports, which is the rate a
        sheared flow must beat; ``ky_max`` matches :meth:`mixing_length_sum`."""
        density = (
            float(np.mean(self.density_gradients))
            if density_gradient is None
            else float(density_gradient)
        )
        peak = 0.0
        for k, ky in enumerate(self.wavenumbers):
            if ky > ky_max:
                continue
            block = self.rates[:, :, :, k]
            per_surface = []
            for row in block:
                per_gradient = np.array(
                    [
                        np.interp(gradient, self.gradients, column)
                        if np.all(np.isfinite(column))
                        else np.nan
                        for column in row.T
                    ]
                )
                usable = np.isfinite(per_gradient)
                per_surface.append(
                    float(
                        np.interp(
                            density, self.density_gradients[usable], per_gradient[usable]
                        )
                    )
                    if usable.any()
                    else np.nan
                )
            per_surface = np.array(per_surface)
            usable = np.isfinite(per_surface)
            if not usable.any():
                continue
            value = float(np.interp(surface, self.surfaces[usable], per_surface[usable]))
            peak = max(peak, value)
        return peak


class MixingLengthResponse:
    """Measured saturated flux in gyro-Bohm units against the linear sum, one curve per surface."""

    def __init__(self, surfaces, curves, electron_curves=None):
        self.surfaces = np.asarray(surfaces, dtype=float)
        self.curves = curves
        self.electron_curves = curves if electron_curves is None else electron_curves

    @classmethod
    def read(cls, path):
        """Build the response from the campaign record, or None if it carries too little."""
        import json
        from pathlib import Path

        try:
            stored = json.loads(Path(path).read_text())
        except (OSError, ValueError):
            return None
        # One point per surface and gradient, from the largest box that ran it: the
        # grown-box repeat of a grid point supersedes the default-box measurement.
        best: dict[tuple[float, float], tuple[float, float, float, float]] = {}
        for point in stored.get("points", []):
            x = point.get("mixing_length_sum")
            q = point.get("saturated_ion_heat_flux_gyrobohm")
            if x is None or q is None or not (np.isfinite(x) and np.isfinite(q)):
                continue
            # A point with no run time was snapshotted mid-flight and is not a measurement.
            if point.get("seconds") is None:
                continue
            electron = point.get("saturated_electron_heat_flux_gyrobohm", float("nan"))
            key = (
                round(float(point["torflux"]), 6),
                round(float(point["gradient"]), 6),
                round(float(point.get("density_gradient", 1.0)), 6),
            )
            area = float(np.prod(point.get("box", (0, 0))))
            if key not in best or area > best[key][0]:
                best[key] = (area, float(x), max(float(q), 0.0), float(electron))
        by_surface: dict[float, list[tuple[float, float, float]]] = {}
        for (surface, _, _), (_, x, q, electron) in best.items():
            by_surface.setdefault(surface, []).append((x, q, electron))
        surfaces = sorted(s for s, pts in by_surface.items() if len(pts) >= 2)
        if not surfaces:
            return None
        curves = []
        electron_curves = []
        for s in surfaces:
            pts = sorted(by_surface[s])
            x_values = np.array([p[0] for p in pts])
            ion = np.array([p[1] for p in pts])
            electron = np.array([p[2] for p in pts])
            curves.append((x_values, ion))
            # The electron curve falls back to the ion one where the record predates
            # the electron channel.
            electron_curves.append(
                (x_values, np.where(np.isfinite(electron), np.maximum(electron, 0.0), ion))
            )
        return cls(surfaces, tuple(curves), tuple(electron_curves))

    def __call__(self, s: float, x: float, species: str = "ion") -> float:
        chosen = self.curves if species == "ion" else self.electron_curves
        values = np.array(
            [float(np.interp(x, cx, cq)) for cx, cq in chosen]
        )
        return float(np.interp(s, self.surfaces, values))


def diffusivity_model(
    table: GrowthRateTable,
    constant,
    field_t: float,
    minor_radius_m: float,
    ion_fraction: float = 0.55,
    floor_m2_s: float = 0.0,
):
    """Turbulent ``chi(s, Te, n)`` at the profile's own local gradients; ``constant`` is the
    scalar closure or a :class:`MixingLengthResponse`."""

    def chi(s, electron_temperature_ev, density):
        s = np.asarray(s, dtype=float)
        temperature = np.asarray(electron_temperature_ev, dtype=float)
        number = np.asarray(density, dtype=float)
        radius = minor_radius_m * np.sqrt(s)
        # a / L_T and a / L_n from the profile, one-sided at the ends.
        gradient = -minor_radius_m * np.gradient(
            np.log(np.maximum(temperature, 1e-30)), radius
        )
        density_gradient = -minor_radius_m * np.gradient(
            np.log(np.maximum(number, 1e-30)), radius
        )
        out = np.empty_like(s)
        for index, (surface, grad, peaking, temp) in enumerate(
            zip(s, gradient, density_gradient, temperature, strict=True)
        ):
            total = table.mixing_length_sum(
                float(surface), float(max(grad, 0.0)), float(max(peaking, 0.0))
            )
            unit = gyro_bohm(ion_fraction * float(temp), field_t, minor_radius_m)
            if callable(constant):
                # The response is a flux; the driving gradient divides back out.
                out[index] = (
                    constant(float(surface), float(total))
                    / max(float(grad), 0.5)
                    * unit
                )
            else:
                out[index] = constant * total * unit
        return np.maximum(out, floor_m2_s)

    return chi


def local_turbulence(
    table: GrowthRateTable,
    constant,
    field_t: float,
    minor_radius_m: float,
    ion_fraction: float = 0.55,
    species: str = "ion",
):
    """Return the pointwise ``chi(s, a_lt, a_ln, temperature_ev, density)`` for the
    shell-by-shell solver; ``species`` selects the response channel."""

    def chi(s: float, a_lt: float, a_ln: float, temperature_ev: float, density: float) -> float:
        total = table.mixing_length_sum(float(s), float(a_lt), float(a_ln))
        unit = gyro_bohm(ion_fraction * float(temperature_ev), field_t, minor_radius_m)
        if isinstance(constant, MixingLengthResponse):
            # The response curve carries the saturated heat flux, and a flux is a
            # diffusivity times the gradient that drives it, so the local gradient
            # divides back out. At the calibration points this is exact.
            flux = constant(float(s), float(total), species=species)
            return min(flux / max(float(a_lt), 0.5) * unit, 1e4)
        if callable(constant):
            flux = constant(float(s), float(total))
            return min(flux / max(float(a_lt), 0.5) * unit, 1e4)
        return min(constant * total * unit, 1e4)

    return chi


#: Order-one weight of the flow shear against the fastest growing mode.
SHEAR_QUENCH_ALPHA = 1.0
PROTON_MASS_KG = 1.67262192e-27


def shear_quench_model(
    table: GrowthRateTable, field_t: float, minor_radius_m: float,
    alpha: float = SHEAR_QUENCH_ALPHA,
):
    """Waltz-rule quench factors 1 - alpha gamma_E / gamma_max, floored at zero,
    with the E x B shearing rate in the table's own growth-rate units."""

    def quench(s, radial_field_v_m, electron_ev, ion_ev, density):
        s = np.asarray(s, dtype=float)
        radius = minor_radius_m * np.sqrt(np.maximum(s, 1e-4))
        rotation = np.asarray(radial_field_v_m, dtype=float) / field_t
        shear = np.gradient(rotation, radius)
        thermal = np.sqrt(
            2.0 * ELEMENTARY_CHARGE * np.maximum(ion_ev, 10.0) / PROTON_MASS_KG
        )
        gamma_e = np.abs(shear) * minor_radius_m / np.maximum(thermal, 1.0)
        a_lt = -minor_radius_m * np.gradient(
            np.log(np.maximum(electron_ev, 1e-30)), radius
        )
        a_ln = -minor_radius_m * np.gradient(
            np.log(np.maximum(density, 1e-30)), radius
        )
        factors = np.ones_like(s)
        for index, surface in enumerate(s):
            fastest = table.peak_growth(
                float(surface), float(max(a_lt[index], 0.0)),
                float(max(a_ln[index], 0.0)),
            )
            if fastest <= 0.0:
                factors[index] = 0.0 if gamma_e[index] > 0.0 else 1.0
                continue
            factors[index] = float(
                np.clip(1.0 - alpha * gamma_e[index] / fastest, 0.0, 1.0)
            )
        return factors

    return quench


def anomalous_channel(
    field_t: float,
    minor_radius_m: float,
    table=None,
    constant: float | None = None,
    verbose: bool = False,
    local: bool = False,
):
    """Turbulent ``chi(s, Te, n)`` from the on-disk grid and response, or None without them;
    ``local`` returns the (electron, ion) pointwise pair instead."""
    import json
    from pathlib import Path

    if table is None:
        grid = Path("results/turbulence/growth_rate_grid.json")
        if not grid.is_file():
            if verbose:
                print(f"no growth-rate grid at {grid}")
            return None
        table = GrowthRateTable.read(grid)
    if constant is None:
        record = Path("results/turbulence/mixing_length_constant.json")
        if not record.is_file():
            if verbose:
                print(f"no mixing-length constant at {record}")
            return None
        constant = MixingLengthResponse.read(record)
        if constant is not None:
            if verbose:
                print(
                    f"saturation response on {len(constant.surfaces)} surfaces from "
                    f"{sum(len(cx) for cx, _ in constant.curves)} nonlinear runs"
                )
        else:
            stored = json.loads(record.read_text())
            values = [
                float(point["constant"])
                for point in stored.get("points", [])
                if point.get("nonlinear_saturation_state") == "saturated"
                and np.isfinite(point.get("constant", float("nan")))
            ]
            if not values:
                if verbose:
                    print(f"{record} carries no saturated point to take a constant from")
                return None
            constant = float(np.median(values))
            if verbose:
                print(
                    f"mixing-length constant {constant:.3f} from {len(values)} saturated "
                    f"nonlinear runs"
                )
    if local:
        return (
            local_turbulence(
                table, constant, field_t, minor_radius_m, species="electron"
            ),
            local_turbulence(table, constant, field_t, minor_radius_m, species="ion"),
        )
    return diffusivity_model(table, constant, field_t, minor_radius_m)


def quench_channel(field_t: float, minor_radius_m: float):
    """The shear quench built from the growth-rate grid on disk, or None without one."""
    from pathlib import Path

    grid = Path("results/turbulence/growth_rate_grid.json")
    if not grid.is_file():
        return None
    return shear_quench_model(GrowthRateTable.read(grid), field_t, minor_radius_m)
