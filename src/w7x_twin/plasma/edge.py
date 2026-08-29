"""The scrape-off layer: parallel transport to the target, and the neutrals it recycles."""

from __future__ import annotations

import dataclasses
import numpy as np


# -- from scrape_off_layer --------------------------------------------------------

ELEMENTARY_CHARGE = 1.602176634e-19
PROTON_MASS = 1.67262192369e-27
BOLTZMANN = 1.380649e-23

#: Spitzer-Haerm parallel electron heat conductivity coefficient, in W / (m eV^(7/2)).
KAPPA_0 = 2000.0
#: Sheath heat transmission coefficient for electrons and ions together.
SHEATH_TRANSMISSION = 7.0

#: Franck-Condon energy of the atoms a recycled molecule dissociates into, in eV. It sets
#: the speed at which neutrals leave the target, and with it the pressure their flux
#: implies.
FRANCK_CONDON_EV = 3.0
#: Molecular hydrogen, which is what a graphite surface returns.
MOLECULE_MASS = 2.0 * PROTON_MASS


@dataclasses.dataclass
class TwoPointSolution:
    """Upstream and target conditions of one flux tube."""

    upstream_density_m3: float
    upstream_temperature_ev: float
    target_temperature_ev: float
    target_density_m3: float
    parallel_heat_flux_w_m2: float
    connection_length_m: float
    #: True where conduction dominates, so the upstream temperature exceeds the target's
    #: by enough for the conduction term to set it.
    conduction_limited: bool

    @property
    def sound_speed_m_s(self) -> float:
        return float(
            np.sqrt(2.0 * self.target_temperature_ev * ELEMENTARY_CHARGE / PROTON_MASS)
        )


def sound_speed(
    temperature_ev: np.ndarray, ion_mass_amu: float = 1.0
) -> np.ndarray:
    """Isothermal ion sound speed at the sheath entrance for the fuel-averaged mass, in m/s."""
    return np.sqrt(
        2.0 * np.asarray(temperature_ev, dtype=float) * ELEMENTARY_CHARGE
        / (ion_mass_amu * PROTON_MASS)
    )


def solve_two_point(
    upstream_density_m3: float,
    parallel_heat_flux_w_m2: float,
    connection_length_m: float,
    sheath_transmission: float = SHEATH_TRANSMISSION,
    kappa_0: float = KAPPA_0,
    iterations: int = 80,
) -> TwoPointSolution:
    """Solve the two-point model for one flux tube by bisection on the target temperature."""
    if parallel_heat_flux_w_m2 <= 0.0 or connection_length_m <= 0.0:
        raise ValueError("the parallel heat flux and connection length must be positive")

    conduction = 3.5 * parallel_heat_flux_w_m2 * connection_length_m / kappa_0

    def upstream_from(target_ev: float) -> float:
        return float((target_ev**3.5 + conduction) ** (1.0 / 3.5))

    def residual(target_ev: float) -> float:
        # Root: sheath-carried flux equals the supplied flux.
        upstream = upstream_from(target_ev)
        target_density = (
            upstream_density_m3 * upstream / (2.0 * max(target_ev, 1e-12))
        )
        carried = (
            sheath_transmission
            * target_density
            * float(sound_speed(np.array([target_ev]))[0])
            * target_ev
            * ELEMENTARY_CHARGE
        )
        return carried - parallel_heat_flux_w_m2

    # The carried flux rises with T_t; bracket by widening from a low guess.
    low, high = 1.0e-3, 1.0e3
    if residual(low) > 0.0:
        target = low
    elif residual(high) < 0.0:
        target = high
    else:
        for _ in range(iterations):
            middle = np.sqrt(low * high)
            if residual(middle) > 0.0:
                high = middle
            else:
                low = middle
        target = float(np.sqrt(low * high))

    upstream = upstream_from(target)
    density = upstream_density_m3 * upstream / (2.0 * target)
    return TwoPointSolution(
        upstream_density_m3=upstream_density_m3,
        upstream_temperature_ev=upstream,
        target_temperature_ev=target,
        target_density_m3=density,
        parallel_heat_flux_w_m2=parallel_heat_flux_w_m2,
        connection_length_m=connection_length_m,
        conduction_limited=bool(upstream > 1.5 * target),
    )


def solve_two_point_extended(
    upstream_density_m3: float,
    parallel_heat_flux_w_m2: float,
    connection_length_m: float,
    power_loss: float = 0.0,
    momentum_loss: float = 0.0,
    sheath_transmission: float = SHEATH_TRANSMISSION,
    kappa_0: float = KAPPA_0,
    iterations: int = 80,
) -> TwoPointSolution:
    """Two-point model with volumetric losses: 2 n_t T_t = (1 - f_mom) n_u T_u and
    q_sheath = (1 - f_pow) q_par, both factors from zero (attached) to one."""
    if parallel_heat_flux_w_m2 <= 0.0 or connection_length_m <= 0.0:
        raise ValueError("the parallel heat flux and connection length must be positive")
    if not (0.0 <= power_loss < 1.0 and 0.0 <= momentum_loss < 1.0):
        raise ValueError("the loss factors run from zero to one")

    conduction = 3.5 * parallel_heat_flux_w_m2 * connection_length_m / kappa_0
    sheath_flux = (1.0 - power_loss) * parallel_heat_flux_w_m2

    def upstream_from(target_ev: float) -> float:
        return float((target_ev**3.5 + conduction) ** (1.0 / 3.5))

    def residual(target_ev: float) -> float:
        upstream = upstream_from(target_ev)
        target_density = (
            (1.0 - momentum_loss)
            * upstream_density_m3
            * upstream
            / (2.0 * max(target_ev, 1e-12))
        )
        carried = (
            sheath_transmission
            * target_density
            * float(sound_speed(np.array([target_ev]))[0])
            * target_ev
            * ELEMENTARY_CHARGE
        )
        return carried - sheath_flux

    low, high = 1.0e-4, 1.0e3
    if residual(low) > 0.0:
        target = low
    elif residual(high) < 0.0:
        target = high
    else:
        for _ in range(iterations):
            middle = np.sqrt(low * high)
            if residual(middle) > 0.0:
                high = middle
            else:
                low = middle
        target = float(np.sqrt(low * high))

    upstream = upstream_from(target)
    density = (
        (1.0 - momentum_loss) * upstream_density_m3 * upstream / (2.0 * target)
    )
    return TwoPointSolution(
        upstream_density_m3=upstream_density_m3,
        upstream_temperature_ev=upstream,
        target_temperature_ev=target,
        target_density_m3=density,
        parallel_heat_flux_w_m2=parallel_heat_flux_w_m2,
        connection_length_m=connection_length_m,
        conduction_limited=bool(upstream > 1.5 * target),
    )


def loss_for_detachment(
    upstream_density_m3: float,
    parallel_heat_flux_w_m2: float,
    connection_length_m: float,
    threshold_ev: float = 5.0,
    momentum_from_power: float = 0.0,
    probes: int = 200,
) -> float:
    """Volumetric power loss bringing the target below a temperature, or NaN;
    T_t depends only on (1 - f_pow)/(1 - f_mom), so momentum loss is held at zero."""
    for value in np.linspace(0.0, 0.995, probes):
        solution = solve_two_point_extended(
            upstream_density_m3,
            parallel_heat_flux_w_m2,
            connection_length_m,
            power_loss=float(value),
            momentum_loss=float(min(momentum_from_power * value, 0.995)),
        )
        if solution.target_temperature_ev < threshold_ev:
            return float(value)
    return float("nan")


@dataclasses.dataclass
class Recycling:
    """The particle flux leaving the target and the neutral pressure it sustains."""

    target_flux_m2_s: float
    recycled_flux_m2_s: float
    #: Density and pressure of an equilibrium gas whose one-sided wall flux equals the
    #: recycled flux. It is an upper bound on what stands in the divertor, not the
    #: divertor pressure: the neutrals are ionised rather than accumulated.
    equilibrium_density_m3: float
    equilibrium_pressure_pa: float


def recycling_balance(
    solution: TwoPointSolution,
    recycling_coefficient: float = 1.0,
    neutral_temperature_ev: float = FRANCK_CONDON_EV,
) -> Recycling:
    """Recycled neutral flux at the target and the equilibrium-gas pressure bound Gamma = n v_bar / 4."""
    flux = solution.target_density_m3 * solution.sound_speed_m_s
    recycled = recycling_coefficient * flux
    mean_speed = float(
        np.sqrt(
            8.0
            * neutral_temperature_ev
            * ELEMENTARY_CHARGE
            / (np.pi * MOLECULE_MASS)
        )
    )
    density = 4.0 * recycled / max(mean_speed, 1e-30)
    pressure = density * neutral_temperature_ev * ELEMENTARY_CHARGE
    return Recycling(
        target_flux_m2_s=float(flux),
        recycled_flux_m2_s=float(recycled),
        equilibrium_density_m3=float(density),
        equilibrium_pressure_pa=float(pressure),
    )


def power_decay_length(
    perpendicular_diffusivity_m2_s: float,
    connection_length_m: float,
    target_temperature_ev: float,
) -> float:
    """Scrape-off layer power width sqrt(chi_perp L / c_s), every term measured, in metres."""
    speed = float(sound_speed(np.array([max(target_temperature_ev, 1e-3)]))[0])
    return float(
        np.sqrt(perpendicular_diffusivity_m2_s * connection_length_m / max(speed, 1e-30))
    )


def integral_width(positions_m: np.ndarray, weights: np.ndarray, bins: int = 40) -> float:
    """Integral width of a heat-flux footprint: its integral divided by its peak."""
    positions = np.asarray(positions_m, dtype=float)
    weight = np.asarray(weights, dtype=float)
    if positions.size < 2 or weight.sum() <= 0:
        return 0.0
    edges = np.linspace(positions.min(), positions.max(), bins + 1)
    profile, _ = np.histogram(positions, bins=edges, weights=weight)
    spacing = float(edges[1] - edges[0])
    peak = float(profile.max())
    return float(profile.sum() * spacing / peak) if peak > 0 else 0.0


def wetted_area(strikes, elements, frame, weights: np.ndarray) -> dict:
    """Power-weighted wetted area per target: integral arc width times toroidal length, in m^2."""
    from ..hardware import walls as _components

    wanted = {
        index: element
        for index, element in enumerate(elements)
        if element.name in frame
    }
    mask = strikes.struck & np.isin(strikes.component, list(wanted))
    if not mask.any():
        return {"area_m2": 0.0, "lines": 0, "per_element": {}}

    total = 0.0
    spans: dict[str, dict] = {}
    for index, element in wanted.items():
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        arc, _ = _components.arc_position(
            element, strikes.r[on_element], strikes.z[on_element],
            strikes.phi[on_element],
        )
        width = integral_width(arc, weights[on_element])
        toroidal = (
            float(np.mean(element.r) * (element.phi.max() - element.phi.min()))
            * _components.NUM_FIELD_PERIODS_DEFAULT
            * 2.0
        )
        spans[element.name] = {
            "footprint_width_m": width,
            "toroidal_length_m": toroidal,
            "power_share": float(weights[on_element].sum() / weights[mask].sum()),
            "area_m2": width * toroidal,
        }
        total += width * toroidal
    return {"area_m2": total, "lines": int(mask.sum()), "per_element": spans}


def strip_dilution(
    strikes, elements, frame, weights: np.ndarray, slices: int = 12
) -> dict[str, float]:
    """Per-element smear factor of the diagonal strike strip: whole-element arc spread
    over the power-weighted per-toroidal-slice spread, one where the bins resolve the strip."""
    from ..hardware import walls as _components

    wanted = {
        index: element
        for index, element in enumerate(elements)
        if element.name in frame
    }
    mask = strikes.struck & np.isin(strikes.component, list(wanted))
    out: dict[str, float] = {}
    for index, element in wanted.items():
        on_element = mask & (strikes.component == index)
        if not on_element.any():
            continue
        arc, _ = _components.arc_position(
            element, strikes.r[on_element], strikes.z[on_element],
            strikes.phi[on_element],
        )
        share = weights[on_element]
        whole = integral_width(arc, share)
        period = 2.0 * np.pi / _components.NUM_FIELD_PERIODS_DEFAULT
        phase = np.mod(strikes.phi[on_element], period)
        edges = np.linspace(0.0, period, slices + 1)
        widths: list[float] = []
        slice_weights: list[float] = []
        for low, high in zip(edges[:-1], edges[1:]):
            inside = (phase >= low) & (phase < high)
            if inside.sum() < 4:
                continue
            widths.append(integral_width(arc[inside], share[inside]))
            slice_weights.append(float(share[inside].sum()))
        if not widths or sum(slice_weights) <= 0.0:
            out[element.name] = 1.0
            continue
        local = float(np.average(widths, weights=slice_weights))
        out[element.name] = max(whole / max(local, 1e-6), 1.0)
    return out


def target_profile(
    strikes, elements, frame, weights: np.ndarray, power_w: float,
    incidence_by_line: np.ndarray | None = None,
    deskew: bool = False,
    offset_by_line: np.ndarray | None = None,
) -> dict:
    """Heat flux binned along each target's arc from the traced strikes;
    ``deskew`` removes the strike band's fitted toroidal drift, conserving each bin's power."""
    from ..hardware import walls as _components

    wanted = {
        index: element
        for index, element in enumerate(elements)
        if element.name in frame
    }
    mask = strikes.struck & np.isin(strikes.component, list(wanted))
    out: dict = {
        "per_element": {},
        "peak_heat_flux_w_m2": 0.0,
        "peak_element": None,
        "peak_integral_width_m": 0.0,
    }
    total_weight = float(weights[mask].sum()) if mask.any() else 0.0
    if total_weight <= 0.0:
        return out
    period = 2.0 * np.pi / _components.NUM_FIELD_PERIODS_DEFAULT
    for index, element in wanted.items():
        on_element = mask & (strikes.component == index)
        count = int(on_element.sum())
        if count < 4:
            continue
        arc, _ = _components.arc_position(
            element, strikes.r[on_element], strikes.z[on_element],
            strikes.phi[on_element],
        )
        weight = weights[on_element]
        drift_m_per_rad = 0.0
        if deskew:
            # Band drift coordinate, lower units reflected as their contours are.
            phi_here = strikes.phi[on_element]
            z_here = strikes.z[on_element]
            toroidal = np.mod(np.where(z_here < 0.0, -phi_here, phi_here), period)
            mean_t = float(np.average(toroidal, weights=weight))
            mean_a = float(np.average(arc, weights=weight))
            spread = float(np.sum(weight * (toroidal - mean_t) ** 2))
            if spread > 1e-12:
                drift_m_per_rad = float(
                    np.sum(weight * (toroidal - mean_t) * (arc - mean_a)) / spread
                )
            arc = arc - drift_m_per_rad * (toroidal - mean_t)
        span = float(arc.max() - arc.min())
        if span < 1e-6:
            continue
        bins = int(np.clip(count // 4, 12, 48))
        edges = np.linspace(float(arc.min()), float(arc.max()), bins + 1)
        histogram, _ = np.histogram(arc, bins=edges, weights=weight)
        spacing = float(edges[1] - edges[0])
        toroidal = (
            float(np.mean(element.r) * (element.phi.max() - element.phi.min()))
            * _components.NUM_FIELD_PERIODS_DEFAULT
            * 2.0
        )
        flux = power_w * (histogram / total_weight) / (spacing * toroidal)
        peak = float(flux.max())
        integral = float(flux.sum() * spacing / peak) if peak > 0 else 0.0
        lengths = strikes.connection_length_m[on_element]
        weighted_length, _ = np.histogram(
            arc, bins=edges, weights=weight * np.nan_to_num(lengths)
        )
        with np.errstate(divide="ignore", invalid="ignore"):
            connection = np.where(histogram > 0.0, weighted_length / histogram, np.nan)
        row = {
            "arc_m": (0.5 * (edges[:-1] + edges[1:])).tolist(),
            "heat_flux_w_m2": flux.tolist(),
            "connection_m": connection.tolist(),
            "bin_area_m2": spacing * toroidal,
            "peak_heat_flux_w_m2": peak,
            "integral_width_m": integral,
            "band_drift_m_per_rad": drift_m_per_rad,
        }
        if incidence_by_line is not None:
            sines = np.asarray(incidence_by_line, dtype=float)[on_element]
            weighted_sine, _ = np.histogram(
                arc, bins=edges, weights=weight * np.nan_to_num(sines)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                row["incidence_sine"] = np.where(
                    histogram > 0.0, weighted_sine / histogram, np.nan
                ).tolist()
        if offset_by_line is not None:
            offsets = np.asarray(offset_by_line, dtype=float)[on_element]
            weighted_offset, _ = np.histogram(
                arc, bins=edges, weights=weight * np.nan_to_num(offsets)
            )
            with np.errstate(divide="ignore", invalid="ignore"):
                row["launch_offset_m"] = np.where(
                    histogram > 0.0, weighted_offset / histogram, np.nan
                ).tolist()
        out["per_element"][element.name] = row
        if peak > out["peak_heat_flux_w_m2"]:
            out["peak_heat_flux_w_m2"] = peak
            out["peak_element"] = element.name
            out["peak_integral_width_m"] = integral
    return out


def target_radiator(
    profile: dict,
    upstream_density_m3: float,
    incidence_sine_value: float,
    carbon_fraction: float,
    density_width_m: float | None = None,
    dilution_by_element: dict[str, float] | None = None,
) -> dict:
    """Lengyel radiator per arc bin as its own two-point tube, with launch-offset density
    decay and power-conserving wetted-strip pricing."""
    radiated_total = 0.0
    net_peak = 0.0
    net_peak_element = None
    per_element: dict[str, dict] = {}
    for name, row in profile.get("per_element", {}).items():
        flux = np.asarray(row["heat_flux_w_m2"], dtype=float)
        connection = np.asarray(row["connection_m"], dtype=float)
        fallback = float(np.nanmedian(connection)) if np.isfinite(connection).any() else 80.0
        bin_area = float(row["bin_area_m2"])
        dilution = float((dilution_by_element or {}).get(name, 1.0))
        dilution = dilution if np.isfinite(dilution) and dilution >= 1.0 else 1.0
        tube_flux = flux * dilution
        tube_area = bin_area / dilution
        sines = np.asarray(
            row.get("incidence_sine", [incidence_sine_value] * len(flux)), dtype=float
        )
        offsets = np.asarray(
            row.get("launch_offset_m", [0.0] * len(flux)), dtype=float
        )
        net = flux.copy()
        radiated_bins = np.zeros_like(flux)
        limited = 0
        radiating = 0
        tube_fractions: list[float] = []
        residences: list[float] = []
        for index in range(len(flux)):
            if flux[index] <= 0.0 or carbon_fraction <= 0.0:
                continue
            length = connection[index] if np.isfinite(connection[index]) else fallback
            local_sine = (
                sines[index]
                if np.isfinite(sines[index]) and sines[index] > 0.0
                else incidence_sine_value
            )
            local_density = upstream_density_m3
            if density_width_m is not None and np.isfinite(offsets[index]):
                local_density = upstream_density_m3 * float(
                    np.exp(-max(offsets[index], 0.0) / max(density_width_m, 1e-6))
                )
            parallel = tube_flux[index] / max(local_sine, 1e-6)
            solution = solve_two_point(
                max(local_density, 1e16), parallel, max(length, 1.0)
            )
            answer = boundary_radiation(
                solution, tube_area * local_sine, carbon_fraction
            )
            radiated = answer["power_w"]
            radiating += 1
            limited += int(answer["radiation_limited"])
            tube_fractions.append(answer["radiated_fraction_of_tube"])
            residences.append(answer["mean_ne_tau_m3_s"])
            radiated_bins[index] = radiated
            net[index] = max(flux[index] - radiated / bin_area, 0.0)
        radiated_total += float(radiated_bins.sum())
        peak = float(net.max()) if len(net) else 0.0
        per_element[name] = {
            "net_heat_flux_w_m2": net.tolist(),
            "radiated_w": float(radiated_bins.sum()),
            "net_peak_w_m2": peak,
            "bins_radiating": radiating,
            "radiation_limited_bins": limited,
            "tube_radiated_fraction_median": (
                float(np.median(tube_fractions)) if tube_fractions else 0.0
            ),
            "ne_tau_median_m3_s": (
                float(np.median(residences)) if residences else float("nan")
            ),
        }
        if peak > net_peak:
            net_peak = peak
            net_peak_element = name
    return {
        "per_element": per_element,
        "radiated_w": radiated_total,
        "net_peak_heat_flux_w_m2": net_peak,
        "net_peak_element": net_peak_element,
    }


def layer_weights(
    start_r: np.ndarray, separatrix_m: float, width_m: float
) -> np.ndarray:
    """Exponential power weight per traced line, offset from the innermost line so ratios never underflow."""
    offset = np.maximum(np.asarray(start_r, dtype=float) - float(separatrix_m), 0.0)
    return np.exp(-(offset - float(np.min(offset))) / max(float(width_m), 1e-12))


def power_weighted_connection_length(
    connection_length_m: np.ndarray, weights: np.ndarray
) -> float:
    """Connection length averaged over the power each line carries, in metres."""
    lengths = np.asarray(connection_length_m, dtype=float)
    share = np.asarray(weights, dtype=float)
    total = float(share.sum())
    return float(np.sum(share * lengths) / total) if total > 0.0 else float("nan")


def close_layer(
    upstream_density_m3: float,
    crossing_power_w: float,
    connection_length_m: float,
    perpendicular_diffusivity_m2_s: float,
    incidence_sine_value: float,
    area_of_width,
    initial_temperature_ev: float = 20.0,
    relaxation: float = 0.2,
    tolerance: float = 1.0e-6,
    iterations: int = 200,
    length_of_width=None,
) -> dict:
    """Relax the layer width and target temperature to their joint fixed point,
    with ``area_of_width`` and optionally ``length_of_width`` closing the loop."""
    given = {
        "upstream density": upstream_density_m3,
        "crossing power": crossing_power_w,
        "connection length": connection_length_m,
        "perpendicular diffusivity": perpendicular_diffusivity_m2_s,
        "incidence sine": incidence_sine_value,
    }
    bad = [name for name, value in given.items() if not np.isfinite(value) or value <= 0.0]
    if bad:
        raise ValueError(
            "the layer cannot be closed on "
            + ", ".join(f"{name} = {given[name]:g}" for name in bad)
        )

    temperature = float(initial_temperature_ev)
    length = float(connection_length_m)
    for step in range(iterations):
        width = power_decay_length(
            perpendicular_diffusivity_m2_s, length, temperature
        )
        if length_of_width is not None:
            length = float(length_of_width(width))
            width = power_decay_length(
                perpendicular_diffusivity_m2_s, length, temperature
            )
        area = float(area_of_width(width))
        if area <= 0.0:
            raise ValueError("the power-weighted footprint has no width")
        solution = solve_two_point(
            upstream_density_m3,
            crossing_power_w / area / incidence_sine_value,
            length,
        )
        updated = solution.target_temperature_ev
        if abs(updated - temperature) < tolerance * max(updated, 1.0):
            return {
                "converged": True,
                "steps": step + 1,
                "width_m": width,
                "area_m2": area,
                "connection_length_m": length,
                "target_temperature_ev": updated,
                "solution": solution,
            }
        temperature = (1.0 - relaxation) * temperature + relaxation * updated
    return {
        "converged": False,
        "steps": iterations,
        "width_m": width,
        "area_m2": area,
        "connection_length_m": length,
        "target_temperature_ev": temperature,
        "solution": solution,
    }


def boundary_radiation(
    solution: TwoPointSolution,
    parallel_area_m2: float,
    carbon_fraction: float,
    samples: int = 400,
    ion_mass_amu: float = 1.0,
    kappa_0: float = KAPPA_0,
) -> dict:
    """Layer carbon radiation from the Lengyel integral
    q(T)^2 = q_t^2 + 2 kappa0 f_C (n T)^2 int L_Z sqrt(T') dT', self-bounded by the tube's power, in W."""
    from w7x_twin.plasma import kinetics

    target = max(float(solution.target_temperature_ev), 1.0e-3)
    upstream = max(float(solution.upstream_temperature_ev), target * (1.0 + 1e-9))
    length = float(solution.connection_length_m)
    entering = float(solution.parallel_heat_flux_w_m2)
    if parallel_area_m2 <= 0.0 or length <= 0.0 or carbon_fraction <= 0.0:
        return {"power_w": 0.0, "radiated_fraction_of_tube": 0.0,
                "radiation_limited": False, "mean_ne_tau_m3_s": float("nan")}

    # Electron pressure along the tube, constant and anchored at the sheath entrance.
    pressure = 2.0 * float(solution.target_density_m3) * target
    # Residence over the mean Mach-1/2 flow, twice the sonic transit.
    residence = length / max(
        0.5 * float(sound_speed(np.array([upstream]), ion_mass_amu)[0]), 1e-12
    )
    ne_tau = float(solution.upstream_density_m3) * residence

    temperature = np.linspace(target, upstream, samples)
    emission = kinetics.cooling_rate(temperature, ne_tau)
    integral = float(
        np.trapezoid(np.sqrt(temperature) * emission, temperature)
    )
    loss = 2.0 * kappa_0 * carbon_fraction * pressure**2 * integral
    leaving = float(np.sqrt(max(entering**2 - loss, 0.0)))
    radiated_flux = entering - leaving
    return {
        "power_w": radiated_flux * parallel_area_m2,
        "radiated_fraction_of_tube": radiated_flux / max(entering, 1e-30),
        # Exhausted flux marks the radiation-limited, detaching case: a bound.
        "radiation_limited": bool(loss >= entering**2),
        "mean_ne_tau_m3_s": ne_tau,
    }


def incidence_sine(
    field, r: np.ndarray, phi: np.ndarray, z: np.ndarray, tangent_r: np.ndarray,
    tangent_z: np.ndarray,
) -> np.ndarray:
    """Sine of the field-to-surface angle for a toroidally swept poloidal contour."""
    b_r, b_phi, b_z = field(r, phi, z)
    magnitude = np.sqrt(b_r * b_r + b_phi * b_phi + b_z * b_z)
    length = np.hypot(tangent_r, tangent_z)
    normal_r = -tangent_z / np.maximum(length, 1e-30)
    normal_z = tangent_r / np.maximum(length, 1e-30)
    return np.abs(b_r * normal_r + b_z * normal_z) / np.maximum(magnitude, 1e-30)


def surface_incidence_sine(
    field, r: np.ndarray, phi: np.ndarray, z: np.ndarray, frame: dict,
) -> np.ndarray:
    """Sine of the field-to-surface angle, the normal being (-R Z_u, Z_u R_phi - R_u Z_phi, R R_u)."""
    b_r, b_phi, b_z = field(r, phi, z)
    magnitude = np.sqrt(b_r * b_r + b_phi * b_phi + b_z * b_z)

    tangent_r, tangent_z = frame["tangent_r"], frame["tangent_z"]
    dr_dphi, dz_dphi = frame["dr_dphi"], frame["dz_dphi"]
    normal_r = -r * tangent_z
    normal_phi = tangent_z * dr_dphi - tangent_r * dz_dphi
    normal_z = r * tangent_r
    length = np.sqrt(normal_r**2 + normal_phi**2 + normal_z**2)

    projection = b_r * normal_r + b_phi * normal_phi + b_z * normal_z
    return np.abs(projection) / np.maximum(magnitude * length, 1e-30)


def tilted_incidence_sine(
    field, r: np.ndarray, phi: np.ndarray, z: np.ndarray, tangent_r: np.ndarray,
    tangent_z: np.ndarray, tilt_rad: float,
) -> np.ndarray:
    """Incidence with the contour normal rotated by ``tilt_rad`` towards the toroidal direction."""
    b_r, b_phi, b_z = field(r, phi, z)
    magnitude = np.sqrt(b_r * b_r + b_phi * b_phi + b_z * b_z)
    length = np.hypot(tangent_r, tangent_z)
    normal_r = -tangent_z / np.maximum(length, 1e-30)
    normal_z = tangent_r / np.maximum(length, 1e-30)
    cos, sin = np.cos(tilt_rad), np.sin(tilt_rad)
    projection = cos * (b_r * normal_r + b_z * normal_z) + sin * b_phi
    return np.abs(projection) / np.maximum(magnitude, 1e-30)


def contour_tangent(component, r: np.ndarray, z: np.ndarray, phi: np.ndarray,
                    num_field_periods: int = 5) -> tuple[np.ndarray, np.ndarray]:
    """Tangent to a component's poloidal contour at the nearest point to each strike."""
    period = 2.0 * np.pi / num_field_periods
    angle = np.mod(np.asarray(phi, dtype=float), period)
    cut_index = np.argmin(
        np.abs(np.mod(component.phi, period)[None, :] - angle[:, None]), axis=1
    )
    tangent_r = np.empty(len(r))
    tangent_z = np.empty(len(r))
    for index in range(len(r)):
        cut = int(cut_index[index])
        contour_r = component.r[cut]
        contour_z = component.z[cut] if z[index] >= 0 else -component.z[cut]
        nearest = int(
            np.argmin(np.hypot(contour_r - r[index], contour_z - z[index]))
        )
        first = max(nearest - 1, 0)
        last = min(nearest + 1, len(contour_r) - 1)
        tangent_r[index] = contour_r[last] - contour_r[first]
        tangent_z[index] = contour_z[last] - contour_z[first]
    return tangent_r, tangent_z

# -- from neutrals ----------------------------------------------------------------

#: Ionisation potential of atomic hydrogen, in eV.
IONISATION_POTENTIAL_EV = 13.6
#: Franck-Condon energy of the atoms a recycled molecule dissociates into, in eV.


def ionisation_rate_m3_s(temperature_ev: np.ndarray) -> np.ndarray:
    """Electron-impact ionisation rate for atomic hydrogen, Voronov (1997) fit, in m^3/s."""
    t = np.maximum(np.asarray(temperature_ev, dtype=float), 1e-3)
    # Voronov: A dE^K X^K exp(-X) / (X + P) with X = dE / T, in cm^3/s.
    a, k, p, energy = 2.91e-8, 0.39, 0.0, IONISATION_POTENTIAL_EV
    x = energy / t
    rate = a * (x**k) * np.exp(-x) / (p + x)
    return np.where(t > 0.1, 1e-6 * rate, 0.0)


def charge_exchange_rate_m3_s(temperature_ev: np.ndarray) -> np.ndarray:
    """Proton-hydrogen charge-exchange rate, Riviere fit over a Maxwellian, in m^3/s."""
    t = np.maximum(np.asarray(temperature_ev, dtype=float), 1e-3)
    # 1e-14 * T^0.318 cm^3/s over the divertor range, from the folded Riviere section.
    return 1e-6 * 1.0e-8 * t**0.318


@dataclasses.dataclass
class NeutralLayer:
    """Recycling neutrals in front of one target."""

    #: Ion flux density arriving at the target, in m^-2 s^-1.
    target_flux_m2_s: float
    target_temperature_ev: float
    target_density_m3: float
    #: Distance a Franck-Condon atom travels before it is ionised, in metres.
    mean_free_path_m: float
    #: Neutral density the recycling flux sustains, in m^-3.
    neutral_density_m3: float
    #: Static neutral pressure at room temperature, in pascals.
    pressure_pa: float
    #: Fraction of the parallel momentum charge exchange removes over that path.
    momentum_loss: float

    @property
    def pressure_mpa(self) -> float:
        return 1e3 * self.pressure_pa


def franck_condon_speed_m_s(energy_ev: float = FRANCK_CONDON_EV) -> float:
    """Speed of an atom released at the Franck-Condon energy."""
    return float(np.sqrt(2.0 * energy_ev * ELEMENTARY_CHARGE / PROTON_MASS))


def recycling_layer(
    target_flux_m2_s: float,
    target_temperature_ev: float,
    target_density_m3: float,
    gas_temperature_k: float = 300.0,
    franck_condon_ev: float = FRANCK_CONDON_EV,
) -> NeutralLayer:
    """Neutral density and pressure the recycling flux sustains ahead of a target,
    atoms at the Franck-Condon energy ionised over v / (n_e <sigma v>)."""
    if target_flux_m2_s < 0.0 or target_density_m3 <= 0.0:
        raise ValueError("the target flux must be positive and the density non-zero")

    speed = franck_condon_speed_m_s(franck_condon_ev)
    ionisation = float(ionisation_rate_m3_s(np.array([target_temperature_ev]))[0])
    exchange = float(charge_exchange_rate_m3_s(np.array([target_temperature_ev]))[0])

    path = speed / (target_density_m3 * ionisation) if ionisation > 0.0 else float("inf")
    neutral_density = target_flux_m2_s / speed

    # Static pressure of that density at the gas temperature the walls hold it to.
    boltzmann = 1.380649e-23
    pressure = neutral_density * boltzmann * gas_temperature_k

    # Momentum loss share: charge exchange over total atomic rate along the path.
    total = ionisation + exchange
    momentum = float(exchange / total) if total > 0.0 else 0.0

    return NeutralLayer(
        target_flux_m2_s=float(target_flux_m2_s),
        target_temperature_ev=float(target_temperature_ev),
        target_density_m3=float(target_density_m3),
        mean_free_path_m=float(path),
        neutral_density_m3=float(neutral_density),
        pressure_pa=float(pressure),
        momentum_loss=momentum,
    )


def ionisation_source_profile(
    s: np.ndarray,
    density_m3: np.ndarray,
    temperature_ev: np.ndarray,
    minor_radius_m: float,
    edge_flux_m2_s: float,
    franck_condon_ev: float = FRANCK_CONDON_EV,
) -> np.ndarray:
    """Volumetric particle source from edge neutrals: derivative of the attenuated flux, in m^-3 s^-1."""
    s = np.asarray(s, dtype=float)
    density = np.asarray(density_m3, dtype=float)
    temperature = np.asarray(temperature_ev, dtype=float)
    radius = minor_radius_m * np.sqrt(np.clip(s, 0.0, 1.0))

    speed = franck_condon_speed_m_s(franck_condon_ev)
    rate = ionisation_rate_m3_s(temperature)
    # Inverse mean free path at each surface, integrated inward from the edge.
    attenuation = density * rate / speed

    # Optical depth inward from the separatrix: suffix integral of the inverse mean free path.
    order = np.argsort(radius)
    outward = radius[order]
    along = attenuation[order]
    increments = np.zeros_like(outward)
    increments[1:] = 0.5 * (along[1:] + along[:-1]) * np.diff(outward)
    suffix = np.cumsum(increments[::-1])[::-1]
    depth_sorted = np.append(suffix[1:], 0.0)
    depth = np.empty_like(depth_sorted)
    depth[order] = depth_sorted

    surviving = edge_flux_m2_s * np.exp(-depth)
    return attenuation * surviving
