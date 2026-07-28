"""Bootstrap current, its resistive diffusion, and the coupled solves that advance it."""

from __future__ import annotations

import dataclasses

import numpy as np
import vmecpp
from simsopt.mhd.bootstrap import compute_trapped_fraction, j_dot_B_Redl

from w7x_twin.mhd.equilibrium import Resolution, SCAN, Scenario, Twin
from w7x_twin.plasma import neoclassical, transport
from w7x_twin.plasma.kinetics import KineticProfiles


# -- from bootstrap ---------------------------------------------------------------

MU0 = 4.0e-7 * np.pi


@dataclasses.dataclass
class Geometry:
    """Flux-surface quantities the Redl formula needs, in simsopt's conventions."""

    s: np.ndarray
    iota: np.ndarray
    G: np.ndarray
    I: np.ndarray
    R: np.ndarray
    epsilon: np.ndarray
    f_t: np.ndarray
    psi_edge: float
    nfp: int


def half_grid(ns: int) -> np.ndarray:
    return (np.arange(1, ns) - 0.5) / (ns - 1)


def redl_geometry(
    output: vmecpp.VmecOutput, ntheta: int = 64, nphi: int = 65
) -> Geometry:
    """Evaluate the geometric inputs of the Redl formula on the VMEC half grid."""
    wout = output.wout
    ns = int(wout.ns)
    s = half_grid(ns)

    iota = np.asarray(wout.iotas)[1:]
    G = np.asarray(wout.bvco)[1:]
    I = np.asarray(wout.buco)[1:]

    bmnc = np.asarray(wout.bmnc)[:, 1:]
    gmnc = np.asarray(wout.gmnc)[:, 1:]
    xm = np.asarray(wout.xm_nyq)
    xn = np.asarray(wout.xn_nyq)

    theta = np.linspace(0.0, 2.0 * np.pi, ntheta, endpoint=False)
    phi = np.linspace(0.0, 2.0 * np.pi / wout.nfp, nphi, endpoint=False)
    phi2d, theta2d = np.meshgrid(phi, theta)
    angle = (
        xm[None, None, :] * theta2d[:, :, None] - xn[None, None, :] * phi2d[:, :, None]
    )
    cosangle = np.cos(angle)
    # (theta, phi, mode) x (mode, surface) -> (theta, phi, surface)
    mod_b = cosangle @ bmnc
    sqrt_g = cosangle @ gmnc

    _, _, epsilon, _, fsa_one_over_b, f_t = compute_trapped_fraction(mod_b, sqrt_g)

    return Geometry(
        s=s,
        iota=iota,
        G=G,
        I=I,
        # simsopt's effective major radius for shaped geometry.
        R=(G + iota * I) * fsa_one_over_b,
        epsilon=epsilon,
        f_t=f_t,
        # VMEC's signed toroidal flux; the wrong sign reverses the returned current.
        psi_edge=float(np.asarray(wout.phi)[-1]) / (2.0 * np.pi),
        nfp=int(wout.nfp),
    )


def dominant_helicity(
    output: vmecpp.VmecOutput, surface_fraction: float = 0.5
) -> tuple[int, dict[str, float]]:
    """Toroidal mode number of the strongest non-axisymmetric |B| component,
    with the helical (1, nfp), mirror (0, nfp) and axisymmetric (1, 0) amplitudes."""
    wout = output.wout
    xm = np.asarray(wout.xm_nyq)
    xn = np.asarray(wout.xn_nyq)
    nfp = int(wout.nfp)
    surface = int(np.clip(surface_fraction * (int(wout.ns) - 1), 1, int(wout.ns) - 1))
    amplitude = np.asarray(wout.bmnc)[:, surface]

    b00 = float(amplitude[(xm == 0) & (xn == 0)][0])

    def component(m: int, n: int) -> float:
        match = (xm == m) & (xn == n)
        return float(amplitude[match][0] / b00) if match.any() else 0.0

    amplitudes = {
        "helical_1_nfp": component(1, nfp),
        "mirror_0_nfp": component(0, nfp),
        "axisymmetric_1_0": component(1, 0),
    }

    non_axisymmetric = xn != 0
    strongest = np.argmax(np.abs(amplitude) * non_axisymmetric)
    return int(xn[strongest]), amplitudes


def redl_jdotb(
    output: vmecpp.VmecOutput,
    profiles: KineticProfiles,
    helicity_n: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Redl bootstrap <J.B> from the kinetic profiles on this equilibrium's geometry;
    ``helicity_n`` defaults to the measured dominant |B| symmetry direction."""
    if helicity_n is None:
        helicity_n, _ = dominant_helicity(output)
    geometry = redl_geometry(output)
    ne, te, ti = profiles.as_simsopt()
    # The effective charge enters as a profile, the carbon charge state following T_e.
    z_effective = (
        profiles.z_effective_as_simsopt()
        if getattr(profiles, "carbon_fraction", 0.0)
        else profiles.z_effective
    )
    result = j_dot_B_Redl(
        ne,
        te,
        ti,
        z_effective,
        helicity_n=helicity_n,
        s=geometry.s,
        G=geometry.G,
        R=geometry.R,
        iota=geometry.iota,
        epsilon=geometry.epsilon,
        f_t=geometry.f_t,
        psi_edge=geometry.psi_edge,
        nfp=geometry.nfp,
    )
    # simsopt returns the profile alongside a structure of intermediate quantities.
    jdotb = result[0] if isinstance(result, tuple) else result
    return geometry.s, np.asarray(jdotb, dtype=float)


def drift_kinetic_jdotb(
    output: vmecpp.VmecOutput,
    profiles: KineticProfiles,
    coefficients,
    ripple=None,
    reference_surface: float = 0.2,
    radial_field_v_m: float | np.ndarray = 0.0,
    z_effective: bool = False,
    momentum_correction: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    """Bootstrap <J.B> from the monoenergetic D_31 coefficient, Maxwellian-convolved per species;
    ``z_effective`` enters the collisionality and ``momentum_correction`` restores e-e conservation."""
    from w7x_twin.plasma import neoclassical

    wout = output.wout
    s = half_grid(int(wout.ns))
    minor = float(wout.Aminor_p)
    radius = minor * np.sqrt(s)

    density = profiles.density(s)
    electron_t = profiles.electron_temperature(s)
    ion_t = profiles.ion_temperature(s)

    def logarithmic_gradient(values: np.ndarray) -> np.ndarray:
        return np.gradient(np.log(np.maximum(values, 1e-30)), radius)

    dln_n = logarithmic_gradient(density)
    dln_te = logarithmic_gradient(electron_t)
    dln_ti = logarithmic_gradient(ion_t)

    b00 = float(np.abs(wout.b0))
    charge = (
        np.asarray(profiles.z_effective_profile(s), dtype=float)
        if z_effective
        else np.full_like(s, float(getattr(profiles, "z_effective", 1.0)))
    )
    restoration = (
        neoclassical.spitzer_correction(charge)
        if momentum_correction
        else np.ones_like(s)
    )
    # The ambipolar field is a profile, so the drive accepts one as readily as a constant.
    field = np.broadcast_to(np.asarray(radial_field_v_m, dtype=float), s.shape)

    jdotb = np.zeros_like(s)
    for i in range(len(s)):
        total = 0.0
        for table, weight in neoclassical.surface_tables(
            coefficients,
            float(s[i]),
            which="d31",
            ripple=ripple,
            reference_surface=reference_surface,
        ):
            if weight == 0.0:
                continue
            electron = neoclassical.parallel_current_drive(
                table, float(density[i]), float(electron_t[i]),
                float(dln_n[i]), float(dln_te[i]), b00,
                mass=neoclassical.ELECTRON_MASS, charge_number=-1.0,
                z_effective=float(charge[i]),
                radial_field_v_m=float(field[i]),
            )
            ion = neoclassical.parallel_current_drive(
                table, float(density[i]), float(ion_t[i]),
                float(dln_n[i]), float(dln_ti[i]), b00,
                mass=neoclassical.PROTON_MASS, charge_number=1.0,
                z_effective=float(charge[i]),
                radial_field_v_m=float(field[i]),
            )
            total += weight * (float(restoration[i]) * electron + ion)
        jdotb[i] = total
    return s, jdotb


def vmec_jdotb(output: vmecpp.VmecOutput) -> tuple[np.ndarray, np.ndarray]:
    """The equilibrium's own <J.B>, on the half grid, for comparison with the target."""
    wout = output.wout
    ns = int(wout.ns)
    jdotb = np.asarray(wout.jdotb)
    s_full = np.linspace(0.0, 1.0, ns)
    s = half_grid(ns)
    return s, np.interp(s, s_full, jdotb)


@dataclasses.dataclass
class BootstrapSolution:
    """Result of the self-consistent solve."""

    output: vmecpp.VmecOutput
    s: np.ndarray
    jdotb_target: np.ndarray
    jdotb_achieved: np.ndarray
    current_knots_s: np.ndarray
    current_knots_dids: np.ndarray
    total_current_a: float
    residual: float
    iterations: int

    @property
    def mismatch(self) -> float:
        """Residual as a fraction of the target's largest value."""
        scale = np.max(np.abs(self.jdotb_target))
        return float(self.residual / scale) if scale > 0 else float("nan")


def _scenario_for(
    profiles: KineticProfiles,
    knots_s: np.ndarray,
    knots_dids: np.ndarray,
) -> Scenario:
    """Scenario carrying a spline pressure profile and a spline current profile."""
    pressure_s, pressure_pa = profiles.pressure_spline()
    total = float(np.trapezoid(knots_dids, knots_s))
    return Scenario(
        pressure_profile=(1.0,),
        peak_pressure_pa=1.0,
        net_toroidal_current_a=total,
        current_profile=(0.0,),
        pressure_spline=(pressure_s, pressure_pa),
        current_spline=(knots_s, knots_dids),
    )


def solve_self_consistent(
    twin: Twin,
    configuration: str,
    profiles: KineticProfiles,
    resolution: Resolution = SCAN,
    num_knots: int = 6,
    outer_iterations: int = 4,
    tolerance: float = 2e-2,
    damping: float = 1.0,
    verbose: bool = True,
    target: str = "redl",
    coefficients=None,
    ripple=None,
    radial_field_v_m: float = 0.0,
    z_effective: bool = False,
    momentum_correction: bool = False,
) -> BootstrapSolution:
    """Newton-iterate the dI/ds knots until the equilibrium reproduces its own Redl bootstrap."""
    knots_s = np.linspace(0.0, 1.0, num_knots)
    knots = np.zeros(num_knots)

    if target not in ("redl", "drift_kinetic"):
        raise ValueError(f"unknown bootstrap target {target!r}")
    if target == "drift_kinetic" and coefficients is None:
        raise ValueError("drift-kinetic target needs monoenergetic coefficients")

    def bootstrap_target(out: vmecpp.VmecOutput):
        if target == "redl":
            return redl_jdotb(out, profiles)
        return drift_kinetic_jdotb(
            out, profiles, coefficients, ripple,
            radial_field_v_m=radial_field_v_m,
            z_effective=z_effective,
            momentum_correction=momentum_correction,
        )

    def solve(values: np.ndarray) -> vmecpp.VmecOutput:
        state = twin.state(
            configuration, scenario=_scenario_for(profiles, knots_s, values)
        )
        return twin.solve(state, resolution)

    output = solve(knots)
    residual = float("inf")
    iteration = 0

    for iteration in range(1, outer_iterations + 1):
        s, target_profile = bootstrap_target(output)
        _, achieved = vmec_jdotb(output)
        difference = achieved - target_profile
        residual = float(np.sqrt(np.mean(difference**2)))
        scale = float(np.max(np.abs(target_profile)))
        if verbose:
            print(
                f"  iteration {iteration}: residual {residual:.4g} "
                f"({100 * residual / scale:.2f} % of peak target), "
                f"I_tor {float(np.trapezoid(knots, knots_s)):.1f} A"
            )
        if residual / scale < tolerance:
            break

        # Finite-difference Jacobian of <J.B> with respect to the knot values.
        step = max(1.0e3, 0.05 * scale * 1.0e-3)
        jacobian = np.empty((len(s), num_knots))
        for k in range(num_knots):
            perturbed = knots.copy()
            perturbed[k] += step
            _, achieved_k = vmec_jdotb(solve(perturbed))
            jacobian[:, k] = (achieved_k - achieved) / step

        update, *_ = np.linalg.lstsq(jacobian, -difference, rcond=None)
        knots = knots + damping * update
        output = solve(knots)

    s, target_profile = bootstrap_target(output)
    _, achieved = vmec_jdotb(output)
    return BootstrapSolution(
        output=output,
        s=s,
        jdotb_target=target_profile,
        jdotb_achieved=achieved,
        current_knots_s=knots_s,
        current_knots_dids=knots,
        total_current_a=float(np.trapezoid(knots, knots_s)),
        residual=float(np.sqrt(np.mean((achieved - target_profile) ** 2))),
        iterations=iteration,
    )

# -- from current_diffusion -------------------------------------------------------


#: Spitzer parallel resistivity coefficient: eta = 1.03e-4 Z ln(Lambda) T^(-3/2) ohm metre
#: with the temperature in eV, from the NRL plasma formulary expression in ohm centimetres.
SPITZER_COEFFICIENT = 1.03e-4


def spitzer_resistivity(
    temperature_ev: np.ndarray, density_m3: np.ndarray, z_effective: np.ndarray = 1.0
) -> np.ndarray:
    """Parallel Spitzer resistivity in ohm metres."""
    temperature = np.maximum(np.asarray(temperature_ev, dtype=float), 1.0)
    logarithm = np.vectorize(neoclassical.coulomb_logarithm)(density_m3, temperature)
    return (
        SPITZER_COEFFICIENT
        * np.asarray(z_effective, dtype=float)
        * logarithm
        * temperature ** (-1.5)
    )


def conductivity_reduction(
    coefficients: neoclassical.MonoenergeticCoefficients,
    density_m3: float,
    temperature_ev: float,
    z_effective: float = 1.0,
    num_energy: int = 64,
) -> float:
    """Parallel conductivity over its Spitzer value: the trapped-particle reduction, below one."""
    thermal_speed = np.sqrt(
        2.0 * temperature_ev * neoclassical.ELEMENTARY_CHARGE / neoclassical.ELECTRON_MASS
    )
    energy, weight = neoclassical.maxwellian_nodes(num_energy)
    speed = thermal_speed * np.sqrt(energy)
    nu = neoclassical.deflection_frequency(
        speed, density_m3, temperature_ev, neoclassical.ELECTRON_MASS, -1.0, z_effective
    )
    collisionality = nu / speed
    field = np.zeros_like(collisionality)

    numerator = neoclassical._interpolate_coefficient(
        coefficients, collisionality, field, "d33"
    )
    denominator = neoclassical._interpolate_coefficient(
        coefficients, collisionality, field, "d33_spitzer"
    )
    # The parallel conductivity carries the same energy weight as the flow moment.
    measure = weight * energy
    top = float(np.sum(measure * numerator))
    bottom = float(np.sum(measure * denominator))
    return top / bottom if bottom != 0.0 else float("nan")


@dataclasses.dataclass
class Evolution:
    """Current density and enclosed current against radius and time."""

    radius_m: np.ndarray
    times_s: np.ndarray
    current_density_a_m2: np.ndarray  # (n_time, n_radius)
    enclosed_current_a: np.ndarray  # (n_time,)
    resistive_time_s: float

    def at(self, time_s: float) -> np.ndarray:
        """The current density profile at one time, interpolated between steps."""
        index = int(np.clip(np.searchsorted(self.times_s, time_s), 1, len(self.times_s) - 1))
        span = self.times_s[index] - self.times_s[index - 1]
        weight = 0.0 if span == 0 else (time_s - self.times_s[index - 1]) / span
        return (1 - weight) * self.current_density_a_m2[index - 1] + (
            weight * self.current_density_a_m2[index]
        )


#: Square of the first zero of J_0, the eigenvalue of the lowest diffusive mode of the
#: operator above on a disc with the edge value fixed.
FUNDAMENTAL_EIGENVALUE = 2.404825557695773**2


def resistive_time(radius_m: np.ndarray, resistivity: np.ndarray) -> float:
    """Lowest diffusive-mode decay time mu0 a^2 <1/eta> / j01^2, in seconds."""
    radius = np.asarray(radius_m, dtype=float)
    mean = float(
        np.trapezoid(radius / np.asarray(resistivity), radius)
        / np.trapezoid(radius, radius)
    )
    return float(MU0 * radius[-1] ** 2 * mean / FUNDAMENTAL_EIGENVALUE)


def evolve(
    radius_m: np.ndarray,
    resistivity: np.ndarray,
    bootstrap_density_a_m2: np.ndarray,
    times_s: np.ndarray,
    initial_a_m2: np.ndarray | None = None,
) -> Evolution:
    """Integrate current diffusion implicitly in time, second order in radius,
    with dj/dr = 0 on axis and zero loop voltage at the edge."""
    radius = np.asarray(radius_m, dtype=float)
    eta = np.asarray(resistivity, dtype=float)
    target = np.asarray(bootstrap_density_a_m2, dtype=float)
    times = np.asarray(times_s, dtype=float)
    count = len(radius)

    current = np.zeros(count) if initial_a_m2 is None else np.array(initial_a_m2, float)
    history = [current.copy()]

    # With u = eta (j - j_bs), the implicit step is one tridiagonal solve per time level.
    dr = np.diff(radius)
    up = np.zeros(count)
    down = np.zeros(count)
    for i in range(1, count - 1):
        spacing = 0.5 * (dr[i] + dr[i - 1])
        up[i] = 0.5 * (radius[i] + radius[i + 1]) / (radius[i] * spacing * dr[i])
        down[i] = 0.5 * (radius[i] + radius[i - 1]) / (radius[i] * spacing * dr[i - 1])

    def operator(values: np.ndarray) -> np.ndarray:
        out = np.zeros(count)
        weighted = eta * values
        for i in range(1, count - 1):
            out[i] = up[i] * (weighted[i + 1] - weighted[i]) - down[i] * (
                weighted[i] - weighted[i - 1]
            )
        return out

    source = operator(target)

    for step in range(1, len(times)):
        dt = float(times[step] - times[step - 1])
        matrix = np.zeros((count, count))
        rhs = MU0 * current / dt - source

        for i in range(1, count - 1):
            matrix[i, i - 1] = -down[i] * eta[i - 1]
            matrix[i, i] = MU0 / dt + (up[i] + down[i]) * eta[i]
            matrix[i, i + 1] = -up[i] * eta[i + 1]

        # Axis: dj/dr = 0.
        matrix[0, 0] = 1.0
        matrix[0, 1] = -1.0
        rhs[0] = 0.0
        # Edge: zero loop voltage, so eta (j - j_bs) vanishes there.
        matrix[-1, -1] = 1.0
        rhs[-1] = target[-1]

        current = np.linalg.solve(matrix, rhs)
        history.append(current.copy())

    profiles = np.array(history)
    enclosed = np.array(
        [float(np.trapezoid(2.0 * np.pi * radius * row, radius)) for row in profiles]
    )
    return Evolution(
        radius_m=radius,
        times_s=times,
        current_density_a_m2=profiles,
        enclosed_current_a=enclosed,
        resistive_time_s=resistive_time(radius, eta),
    )


# -- from coupled -----------------------------------------------------------------

@dataclasses.dataclass
class CoupledStep:
    """One outer iteration of the coupled solve."""

    iteration: int
    stored_energy_j: float
    total_current_a: float
    electron_temperature_axis_ev: float
    beta: float
    iota_edge: float
    bootstrap_mismatch: float


@dataclasses.dataclass
class CoupledSolution:
    """The converged state, and the history that reached it."""

    output: vmecpp.VmecOutput
    transport_solution: transport.TransportSolution
    profiles: KineticProfiles
    history: list[CoupledStep]
    converged: bool

    @property
    def residual(self) -> dict[str, float]:
        """Relative change in the coupled quantities over the last step."""
        if len(self.history) < 2:
            return {"stored_energy": float("nan"), "current": float("nan")}
        last, previous = self.history[-1], self.history[-2]
        return {
            "stored_energy": abs(last.stored_energy_j - previous.stored_energy_j)
            / max(abs(last.stored_energy_j), 1e-30),
            "current": abs(last.total_current_a - previous.total_current_a)
            / max(abs(last.total_current_a), 1e-30),
        }


def solve_coupled(
    twin: Twin,
    configuration: str,
    profiles: KineticProfiles,
    heating: transport.Heating = transport.Heating(),
    model: transport.TransportModel = transport.TransportModel(),
    neoclassical=None,
    resolution: Resolution = SCAN,
    drive: str = "redl",
    drive_keywords: dict | None = None,
    outer_iterations: int = 6,
    tolerance: float = 5.0e-3,
    verbose: bool = True,
) -> CoupledSolution:
    """Iterate the three until the stored energy and the current stop moving."""
    keywords = {"target": drive, **(drive_keywords or {})}
    history: list[CoupledStep] = []
    current_profiles = profiles
    solution = None
    boot = None

    for iteration in range(1, outer_iterations + 1):
        boot = solve_self_consistent(
            twin, configuration, current_profiles, resolution=resolution,
            verbose=False, **keywords,
        )
        solution = transport.solve(
            boot.output, current_profiles, heating=heating, model=model,
            neoclassical=neoclassical,
        )
        # Solved temperatures feed the next bootstrap and equilibrium; density stays prescribed.
        current_profiles = solution.as_kinetic_profiles()

        step = CoupledStep(
            iteration=iteration,
            stored_energy_j=float(solution.stored_energy_j),
            total_current_a=float(boot.total_current_a),
            electron_temperature_axis_ev=float(solution.electron_temperature_ev[0]),
            beta=float(boot.output.wout.betatotal),
            iota_edge=float(np.asarray(boot.output.wout.iotaf)[-1]),
            bootstrap_mismatch=float(boot.mismatch),
        )
        history.append(step)
        if verbose:
            print(
                f"  iteration {iteration}: W {step.stored_energy_j / 1e6:.4f} MJ, "
                f"I_tor {step.total_current_a / 1e3:+.3f} kA, "
                f"Te(0) {step.electron_temperature_axis_ev:.0f} eV, "
                f"iota_edge {step.iota_edge:.5f}"
            )
        if len(history) >= 2:
            residual = max(
                abs(history[-1].stored_energy_j - history[-2].stored_energy_j)
                / max(abs(history[-1].stored_energy_j), 1e-30),
                abs(history[-1].total_current_a - history[-2].total_current_a)
                / max(abs(history[-1].total_current_a), 1e-30),
            )
            if verbose:
                print(f"    residual {residual:.3e}")
            if residual < tolerance:
                break

    return CoupledSolution(
        output=boot.output,
        transport_solution=solution,
        profiles=current_profiles,
        history=history,
        converged=len(history) >= 2 and max(
            abs(history[-1].stored_energy_j - history[-2].stored_energy_j)
            / max(abs(history[-1].stored_energy_j), 1e-30),
            abs(history[-1].total_current_a - history[-2].total_current_a)
            / max(abs(history[-1].total_current_a), 1e-30),
        ) < tolerance,
    )

# -- from evolve ------------------------------------------------------------------



@dataclasses.dataclass
class Waveform:
    """Heating against time, as a piecewise-linear trace."""

    time_s: np.ndarray
    power_w: np.ndarray

    def at(self, time_s: float) -> float:
        return float(np.interp(time_s, self.time_s, self.power_w))

    @staticmethod
    def steps(segments: tuple[tuple[float, float], ...]) -> "Waveform":
        """A waveform from (duration, power) segments, held flat within each."""
        times: list[float] = []
        powers: list[float] = []
        clock = 0.0
        for duration, power in segments:
            times.extend([clock, clock + duration])
            powers.extend([power, power])
            clock += duration
        return Waveform(np.asarray(times), np.asarray(powers))


@dataclasses.dataclass
class Trace:
    """One solved discharge history."""

    time_s: np.ndarray
    power_w: np.ndarray
    stored_energy_j: np.ndarray
    confinement_time_s: np.ndarray
    bootstrap_current_a: np.ndarray
    edge_transform: np.ndarray

    def at(self, time_s: float) -> dict:
        index = int(np.argmin(np.abs(self.time_s - time_s)))
        return {
            "time_s": float(self.time_s[index]),
            "power_w": float(self.power_w[index]),
            "stored_energy_j": float(self.stored_energy_j[index]),
            "confinement_time_s": float(self.confinement_time_s[index]),
            "bootstrap_current_a": float(self.bootstrap_current_a[index]),
            "edge_transform": float(self.edge_transform[index]),
        }


def resistive_time_s(
    temperature_ev: float,
    minor_radius_m: float,
    z_effective: float = 1.0,
) -> float:
    """Inductive time of the column, mu_0 a^2 / eta at Spitzer resistivity, in seconds."""
    # Spitzer perpendicular resistivity, ohm m, with a Coulomb logarithm of 15.
    resistivity = 5.2e-5 * z_effective * 15.0 / max(temperature_ev, 1.0) ** 1.5
    return float(MU0 * minor_radius_m**2 / max(resistivity, 1e-12))


def advance(
    waveform: Waveform,
    confinement_time,
    bootstrap_current,
    edge_transform,
    minor_radius_m: float,
    temperature_for_resistivity_ev: float = 1000.0,
    steps: int = 400,
    initial_energy_j: float = 0.0,
    initial_current_a: float = 0.0,
) -> Trace:
    """Advance energy and current through a heating waveform via the three supplied
    callables ``confinement_time``, ``bootstrap_current`` and ``edge_transform``."""
    duration = float(waveform.time_s[-1])
    time = np.linspace(0.0, duration, steps)
    step = time[1] - time[0] if steps > 1 else duration

    inductive = resistive_time_s(temperature_for_resistivity_ev, minor_radius_m)

    energy = np.empty(steps)
    tau = np.empty(steps)
    current = np.empty(steps)
    transform = np.empty(steps)
    power = np.array([waveform.at(t) for t in time])

    energy[0] = initial_energy_j
    current[0] = initial_current_a
    tau[0] = float(confinement_time(energy[0], power[0]))
    transform[0] = float(edge_transform(current[0]))

    for index in range(1, steps):
        # Semi-implicit energy step: a step longer than tau_E cannot go negative.
        tau_now = max(float(confinement_time(energy[index - 1], power[index - 1])), 1e-6)
        energy[index] = (energy[index - 1] + step * power[index]) / (1.0 + step / tau_now)
        tau[index] = tau_now

        # Current relaxes toward what the pressure implies, on the inductive time.
        target = float(bootstrap_current(energy[index]))
        current[index] = (
            current[index - 1] + step / inductive * target
        ) / (1.0 + step / inductive)
        transform[index] = float(edge_transform(current[index]))

    return Trace(
        time_s=time,
        power_w=power,
        stored_energy_j=energy,
        confinement_time_s=tau,
        bootstrap_current_a=current,
        edge_transform=transform,
    )
